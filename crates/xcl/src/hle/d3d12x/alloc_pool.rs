//! A title's command allocator Reset does not take recordings the title still submits.
//!
//! On the PC a list's recording lives in its allocator: after `allocator.Reset()` a list that
//! recorded on it must not be submitted again ("The command allocator was reset after the
//! command list was recorded", debug layer id 541 – the runtime removes the device,
//! 0x887a0001). A retail title does exactly that, which its console lets it: one recording,
//! submitted at 61.6 s, at 63.2 s on a second queue and at 67.8 s again, with the allocator
//! it had been given reset at 67.2 s.
//!
//! So the title's allocator is a name for a PC allocator that changes: at the title's Reset,
//! when lists still hold recordings on the PC allocator in use (or the GPU has not finished
//! them), that one is put aside as it is and the title's allocator goes on with another – one
//! put aside earlier whose lists have all been reset since and whose submits are done, or a new
//! one. A list reset on the title's allocator records on the PC allocator in use. A list holds
//! one recording, so no more allocators are put aside than there are lists.
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::{Condvar, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

#[derive(Default)]
struct Backing {
    /// PC lists whose recording is on this allocator
    users: HashSet<usize>,
    /// queue → (fence, value after the last submit of one of them)
    marks: HashMap<usize, (ID3D12Fence, u64)>,
    /// true once this backing has been PC-reset while idle and no new recording/submit has touched it
    already_reset: bool,
    /// true while a thread is running `reset_pc_allocator` on this backing outside `STATE`
    resetting: bool,
}

impl Backing {
    unsafe fn free(&self) -> bool {
        !self.resetting && self.users.is_empty() && self.marks.values().all(|(f, v)| f.GetCompletedValue() >= *v)
    }
}

#[derive(Default)]
struct State {
    /// title allocator → the PC allocator in use (absent: itself)
    cur: HashMap<usize, usize>,
    /// title allocator → PC allocators put aside
    aside: HashMap<usize, Vec<usize>>,
    backings: HashMap<usize, Backing>,
    of_list: HashMap<usize, usize>,
    /// title allocator → the type of the lists on it
    types: HashMap<usize, i32>,
    fences: HashMap<usize, (ID3D12Fence, u64)>,
    made: u64,
    swaps: u64,
    /// PC list → its last submit (queue, fence, value after it, recording number then)
    last_submit: HashMap<usize, LastSubmit>,
    /// PC list → recordings so far (a Reset or creation starts the next one)
    recordings: HashMap<usize, u64>,
    resubmits: u64,
    /// Busy backings retained past `forget(alloc)` until their remaining users detach and GPU marks complete.
    orphaned: Vec<OrphanedBacking>,
}

struct OrphanedBacking {
    /// Native PC allocator handle detached from `s.backings` on `forget(alloc)` so future
    /// address reuse of `backing` never collides with this retired backing.
    backing: usize,
    /// True for XCL-created PC allocators (`backing != alloc`), which must be COM-released once idle.
    owned: bool,
    state: Backing,
}

struct LastSubmit {
    queue: usize,
    fence: ID3D12Fence,
    value: u64,
    recording: u64,
}

unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static RESET_DONE: Condvar = Condvar::new();

/// `XCL_ALLOC_SWAP=0` turns it off.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_ALLOC_SWAP").is_ok_and(|v| v.trim() == "0"))
}

fn detach_from_orphans(s: &mut State, list: usize) {
    for o in &mut s.orphaned {
        o.state.users.remove(&list);
    }
}

fn finish_reset_flag(s: &mut State, backing: usize, ok: bool) {
    if let Some(b) = s.backings.get_mut(&backing) {
        b.resetting = false;
        if ok {
            b.marks.clear();
            b.already_reset = true;
        }
    }
    for o in &mut s.orphaned {
        if o.backing == backing && o.state.resetting {
            o.state.resetting = false;
            if ok {
                o.state.marks.clear();
                o.state.already_reset = true;
            }
        }
    }
}

unsafe fn take_free_orphaned(s: &mut State) -> (Vec<usize>, Vec<usize>) {
    let mut made = Vec::new();
    let mut free_backings = Vec::new();
    let mut i = 0;
    while i < s.orphaned.len() {
        if s.orphaned[i].state.free() {
            let o = s.orphaned.swap_remove(i);
            free_backings.push(o.backing);
            if o.owned {
                made.push(o.backing);
            }
        } else {
            i += 1;
        }
    }
    (made, free_backings)
}

unsafe fn finish_reclaimed(made: Vec<usize>, free_backings: Vec<usize>) {
    for b in free_backings {
        super::console_list::allocator_reset(b as *mut c_void);
    }
    for b in made {
        drop(ID3D12CommandAllocator::from_raw(b as *mut c_void));
    }
}

fn move_list(s: &mut State, list: usize, backing: usize) {
    if let Some(old) = s.of_list.insert(list, backing) {
        if old != backing {
            if let Some(b) = s.backings.get_mut(&old) {
                b.users.remove(&list);
            }
        }
    }
    detach_from_orphans(s, list);
    let b = s.backings.entry(backing).or_default();
    b.users.insert(list);
    b.already_reset = false;
}

/// A split segment or stand-in PC list has been retired into `title_list`'s spare pool (or
/// superseded by a new recording on `wrapper`): it will not be submitted again until a future
/// `Reset` binds it to a new backing. Removing it from `users` lets its old backing become
/// reusable once its already-submitted GPU `marks` complete, while keeping `last_submit` so
/// `in_flight(list)` still prevents premature reuse.
pub(super) fn retire_recording(list: usize) {
    let (made, free_backings) = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        if let Some(backing) = s.of_list.remove(&list) {
            if let Some(b) = s.backings.get_mut(&backing) {
                b.users.remove(&list);
            }
        }
        detach_from_orphans(s, list);
        unsafe { take_free_orphaned(s) }
    };
    unsafe { finish_reclaimed(made, free_backings) };
}

/// The PC list's final Release ended its recording. Submitted work still holds the backing
/// through `marks`; removing the list must not remove those fences or the other users.
pub(super) fn forget_list(list: usize) {
    let (made, free_backings) = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        s.last_submit.remove(&list);
        s.recordings.remove(&list);
        if let Some(backing) = s.of_list.remove(&list) {
            if let Some(b) = s.backings.get_mut(&backing) {
                b.users.remove(&list);
            }
        }
        detach_from_orphans(s, list);
        unsafe { take_free_orphaned(s) }
    };
    unsafe { finish_reclaimed(made, free_backings) };
}

fn try_reserve_backing(s: &mut State, list: usize, alloc: usize, ty: i32) -> Option<(usize, Option<usize>)> {
    let backing = s.cur.get(&alloc).copied().unwrap_or(alloc);
    if s.backings.get(&backing).is_some_and(|b| b.resetting) {
        return None;
    }
    let prev = s.of_list.get(&list).copied();
    if ty >= 0 {
        s.types.insert(alloc, ty);
    }
    // Pin `backing` against concurrent `title_reset` while keeping `list` pinned in `prev`
    // (and `s.of_list[list] == prev`) until native `Reset(backing)` succeeds.
    s.backings.entry(backing).or_default().users.insert(list);
    Some((backing, prev))
}

fn commit_backing(s: &mut State, list: usize, backing: usize, prev: Option<usize>) {
    if let Some(old) = prev {
        if old != backing {
            if let Some(b) = s.backings.get_mut(&old) {
                b.users.remove(&list);
            }
        }
    }
    let mut orphaned_backing = false;
    for o in &mut s.orphaned {
        if o.backing == backing && o.state.users.contains(&list) {
            o.state.already_reset = false;
            orphaned_backing = true;
        } else {
            o.state.users.remove(&list);
        }
    }
    if orphaned_backing {
        // `backing` was moved to `s.orphaned` by a concurrent `forget(alloc)` mid-`reset_list`:
        // keep `list` pinned on its orphan and do not recreate a zombie `s.backings` entry.
        s.of_list.remove(&list);
    } else {
        s.of_list.insert(list, backing);
        let b = s.backings.entry(backing).or_default();
        b.users.insert(list);
        b.already_reset = false;
    }
    *s.recordings.entry(list).or_default() += 1;
}

fn rollback_backing(s: &mut State, list: usize, backing: usize, prev: Option<usize>) {
    // Only release the temporary reservation on `backing`; `prev` was never detached.
    if prev != Some(backing) {
        if let Some(b) = s.backings.get_mut(&backing) {
            b.users.remove(&list);
        }
        for o in &mut s.orphaned {
            if o.backing == backing {
                o.state.users.remove(&list);
            }
        }
    }
}

/// Reset `list` on the title's allocator `alloc`, reserving the target backing before native Reset
/// without detaching the previous recording until Reset succeeds so a concurrent `title_reset`
/// cannot reset either backing mid-call. No pool lock is held across the driver call.
pub fn reset_list(list: usize, alloc: usize, ty: i32, reset: impl FnOnce(usize) -> HRESULT) -> HRESULT {
    if !enabled() || alloc == 0 {
        return reset(alloc);
    }
    let (backing, prev) = {
        let mut g = STATE.lock().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let s = g.get_or_insert_with(State::default);
            if let Some(r) = try_reserve_backing(s, list, alloc, ty) {
                break r;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return crate::util::E_FAIL;
            }
            let (ng, _) = RESET_DONE.wait_timeout(g, deadline - now).unwrap();
            g = ng;
        }
    };
    let hr = reset(backing);
    let (made, free_backings) = {
        let mut g = STATE.lock().unwrap();
        let s = g.get_or_insert_with(State::default);
        if hr >= 0 {
            commit_backing(s, list, backing, prev);
        } else {
            rollback_backing(s, list, backing, prev);
        }
        unsafe { take_free_orphaned(s) }
    };
    unsafe { finish_reclaimed(made, free_backings) };
    hr
}

#[cfg(test)]
pub(super) use recording_backing as recording_backing_for_test;

pub(super) fn recording_backing(list: usize) -> Option<usize> {
    STATE.lock().unwrap().as_ref()?.of_list.get(&list).copied()
}

/// The PC allocator the title's allocator `alloc` stands for now.
pub fn current(alloc: usize) -> usize {
    if !enabled() {
        return alloc;
    }
    STATE.lock().unwrap().as_ref().and_then(|s| s.cur.get(&alloc).copied()).unwrap_or(alloc)
}

/// `list` (of type `ty`) was made recording on the PC allocator `raw` itself
/// (CreateCommandList).
pub fn created_on(list: usize, raw: usize, ty: i32) {
    if !enabled() || raw == 0 {
        return;
    }
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    s.types.entry(raw).or_insert(ty);
    move_list(s, list, raw);
    *s.recordings.entry(list).or_default() += 1;
}

/// Diagnostic (#1452): `lists` are about to be submitted to `queue`. A PC list whose previous
/// submit the GPU has not finished must not be submitted again on the PC (debug layer id 553,
/// "the command queue fence has not advanced past previous executions of the command list");
/// a console lets a title do it. Logged (the first 8, then at powers of two) with whether it is
/// the same recording and the same queue. Every submitted PC list is seen (`executed` marks them all).
pub unsafe fn note_resubmits(queue: *mut c_void, lists: &[*mut c_void]) -> usize {
    if !enabled() {
        return 0;
    }
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return 0 };
    let mut found = 0;
    for &l in lists {
        let l = l as usize;
        let Some(last) = s.last_submit.get(&l) else { continue };
        if last.fence.GetCompletedValue() >= last.value {
            continue;
        }
        let same_recording = s.recordings.get(&l).copied().unwrap_or(0) == last.recording;
        let same_queue = last.queue == queue as usize;
        found += 1;
        s.resubmits += 1;
        let n = s.resubmits;
        if n <= 8 || n.is_power_of_two() {
            crate::xlog_warn!(
                "D3D12.X: a command list is submitted while its previous submit is still on the GPU ({}, {}; #{n} so far) – the PC does not allow it (debug layer id 553)",
                if same_recording { "the same recording" } else { "a new recording" },
                if same_queue { "the same queue" } else { "another queue" },
            );
        }
    }
    found
}

/// Whether the GPU has not finished the last submit of the PC list `list` (#1452: it must not be
/// submitted again before that; `title_list` records the title's next recording elsewhere).
pub fn in_flight(list: usize) -> bool {
    let g = STATE.lock().unwrap();
    g.as_ref().and_then(|s| s.last_submit.get(&list)).is_some_and(|l| unsafe { l.fence.GetCompletedValue() } < l.value)
}

/// `lists` were just submitted to `queue`.
pub unsafe fn executed(queue: *mut c_void, lists: &[*mut c_void]) {
    if !enabled() || queue.is_null() {
        return;
    }
    // (every submit is marked, also of lists this module does not track – the PC lists behind
    // console lists – so note_resubmits sees them all)
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let key = queue as usize;
    if !s.fences.contains_key(&key) {
        let mut dev: Option<ID3D12Device> = None;
        let Some(f) = q.GetDevice(&mut dev).ok().and(dev).and_then(|d| d.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok()) else { return };
        s.fences.insert(key, (f, 0));
    }
    let (fence, value) = {
        let (f, v) = s.fences.get_mut(&key).unwrap();
        if q.Signal(&*f, *v + 1).is_err() {
            return;
        }
        *v += 1;
        (f.clone(), *v)
    };
    for &l in lists {
        let l_key = l as usize;
        if let Some(o) = s.orphaned.iter_mut().find(|o| o.state.users.contains(&l_key)) {
            o.state.marks.insert(key, (fence.clone(), value));
            o.state.already_reset = false;
        } else if let Some(&b) = s.of_list.get(&l_key) {
            let entry = s.backings.entry(b).or_default();
            entry.marks.insert(key, (fence.clone(), value));
            entry.already_reset = false;
        }
        let recording = s.recordings.get(&l_key).copied().unwrap_or(0);
        s.last_submit.insert(l_key, LastSubmit { queue: key, fence: fence.clone(), value, recording });
    }
}

/// Whether the GPU may still run a recording made on the title's allocator `alloc`.
pub(super) unsafe fn allocator_in_flight(alloc: usize) -> bool {
    let g = STATE.lock().unwrap();
    let Some(s) = g.as_ref() else { return false };
    let mine = s.cur.get(&alloc).into_iter().chain(s.aside.get(&alloc).into_iter().flatten()).chain([&alloc]);
    let busy = mine.filter_map(|b| s.backings.get(b)).any(|b| b.marks.values().any(|(f, v)| f.GetCompletedValue() < *v));
    busy
}

/// The title's allocator `alloc` is gone (its last Release): nothing of it stays here – a later
/// allocator may get its address, and must not go on with this one's PC allocators. The ones
/// made here go with it (those lists still record on are left alone: not freed, not tracked).
pub unsafe fn forget(alloc: usize) {
    let (made, free_backings): (Vec<usize>, Vec<usize>) = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else {
            drop(g);
            super::console_list::allocator_reset(alloc as *mut c_void);
            return;
        };
        s.types.remove(&alloc);
        let mut mine: Vec<usize> = s.aside.remove(&alloc).unwrap_or_default();
        mine.extend(s.cur.remove(&alloc));
        mine.push(alloc);
        mine.sort_unstable();
        mine.dedup();
        let mut made = Vec::new();
        let mut free_backings = Vec::new();
        for b in mine {
            match s.backings.remove(&b) {
                Some(busy) if !busy.free() => {
                    // Detach busy backing from `s.backings` so address reuse of `b` cannot
                    // collide, while preserving its users/marks until safe reclamation.
                    for &l in &busy.users {
                        if s.of_list.get(&l) == Some(&b) {
                            s.of_list.remove(&l);
                        }
                    }
                    s.orphaned.push(OrphanedBacking { backing: b, owned: b != alloc, state: busy });
                }
                _ => {
                    free_backings.push(b);
                    if b != alloc {
                        made.push(b);
                    }
                }
            }
        }
        let (o_made, o_free) = take_free_orphaned(s);
        made.extend(o_made);
        free_backings.extend(o_free);
        (made, free_backings)
    };
    // (outside the lock: UMA frees and Release callbacks must not re-enter STATE)
    finish_reclaimed(made, free_backings);
}

/// Pop idle put-aside backings of title allocator `a` whose lists and GPU work have finished
/// so the caller can reset them OUTSIDE `STATE` (preventing re-entrant deadlocks if resetting
/// an allocator drops a prelude `ID3D12CommandAllocator` whose `Release` calls `forget`).
unsafe fn take_free_aside(s: &mut State, a: usize) -> Vec<usize> {
    let Some(v) = s.aside.get_mut(&a) else {
        return Vec::new();
    };
    let mut ready = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let f = v[i];
        if s.backings.get(&f).is_some_and(|b| !b.already_reset && b.free()) {
            ready.push(v.remove(i));
            s.backings.entry(f).or_default().resetting = true;
        } else {
            i += 1;
        }
    }
    ready
}

unsafe fn restore_aside(s: &mut State, a: usize, f: usize, ok: bool) -> bool {
    finish_reset_flag(s, f, ok);
    RESET_DONE.notify_all();
    if s.types.contains_key(&a) {
        s.aside.entry(a).or_default().push(f);
        true
    } else {
        s.backings.remove(&f);
        false
    }
}

/// The title resets its allocator `alloc`. `None`: reset it as it is (nothing holds a
/// recording on it, or this is off). `Some(hr)`: done here – the PC allocator in use was put
/// aside with its recordings and another one took its place.
///
/// Releases `STATE` before every `reset_pc_allocator` call so COM callbacks triggered while
/// releasing per-allocator resources (e.g. prelude `ID3D12CommandAllocator::Release` -> `forget`)
/// can never self-deadlock on `STATE`.
pub unsafe fn title_reset(alloc: *mut c_void) -> Option<HRESULT> {
    if !enabled() {
        return None;
    }
    let a = alloc as usize;
    enum Action {
        CleanPrimary { aside: Vec<usize> },
        ResetCurrent { cur: usize, aside: Vec<usize> },
        SwapReuse { cur: usize, spare: usize, needs_reset: bool, aside: Vec<usize> },
        SwapNew { cur: usize, ty: i32, aside: Vec<usize> },
    }
    let action = {
        let mut g = STATE.lock().unwrap();
        let s = g.get_or_insert_with(State::default);
        let c = s.cur.get(&a).copied().unwrap_or(a);
        if s.backings.get(&c).is_none_or(|b| b.free()) {
            let aside = take_free_aside(s, a);
            s.backings.entry(c).or_default().resetting = true;
            if c == a {
                Action::CleanPrimary { aside }
            } else {
                Action::ResetCurrent { cur: c, aside }
            }
        } else {
            let ty = *s.types.get(&a)?;
            let reuse_idx = s.aside.get(&a).and_then(|v| v.iter().position(|f| s.backings.get(f).is_none_or(|b| b.free())));
            match reuse_idx {
                Some(idx) => {
                    let spare = s.aside.get_mut(&a).unwrap().remove(idx);
                    let needs_reset = s.backings.get(&spare).is_none_or(|b| !b.already_reset);
                    if needs_reset {
                        s.backings.entry(spare).or_default().resetting = true;
                    }
                    let aside = take_free_aside(s, a);
                    Action::SwapReuse { cur: c, spare, needs_reset, aside }
                }
                None => {
                    let aside = take_free_aside(s, a);
                    Action::SwapNew { cur: c, ty, aside }
                }
            }
        }
    };

    let finish_aside = |aside: Vec<usize>| {
        if aside.is_empty() {
            return;
        }
        let mut results = Vec::with_capacity(aside.len());
        for f in aside {
            let hr = super::real::reset_pc_allocator(f as *mut c_void);
            results.push((f, hr >= 0));
        }
        let mut orphans = Vec::new();
        let (o_made, o_free) = {
            let mut g = STATE.lock().unwrap();
            if let Some(s) = g.as_mut() {
                for (f, ok) in results {
                    if !restore_aside(s, a, f, ok) && f != a {
                        orphans.push(f);
                    }
                }
                take_free_orphaned(s)
            } else {
                (Vec::new(), Vec::new())
            }
        };
        finish_reclaimed(o_made, o_free);
        for f in orphans {
            drop(ID3D12CommandAllocator::from_raw(f as *mut c_void));
        }
    };

    match action {
        Action::CleanPrimary { aside } => {
            let hr = super::real::reset_pc_allocator(alloc);
            let (o_made, o_free) = {
                let mut g = STATE.lock().unwrap();
                if let Some(s) = g.as_mut() {
                    finish_reset_flag(s, a, hr >= 0);
                    take_free_orphaned(s)
                } else {
                    (Vec::new(), Vec::new())
                }
            };
            RESET_DONE.notify_all();
            finish_reclaimed(o_made, o_free);
            finish_aside(aside);
            Some(hr)
        }
        Action::ResetCurrent { cur, aside } => {
            let hr = super::real::reset_pc_allocator(cur as *mut c_void);
            let mut orphan = None;
            let (o_made, o_free) = {
                let mut g = STATE.lock().unwrap();
                if let Some(s) = g.as_mut() {
                    finish_reset_flag(s, cur, hr >= 0);
                    if !s.types.contains_key(&a) && cur != a && !s.orphaned.iter().any(|o| o.backing == cur) {
                        orphan = Some(cur);
                    }
                    take_free_orphaned(s)
                } else {
                    (Vec::new(), Vec::new())
                }
            };
            RESET_DONE.notify_all();
            finish_reclaimed(o_made, o_free);
            if let Some(f) = orphan {
                drop(ID3D12CommandAllocator::from_raw(f as *mut c_void));
            }
            finish_aside(aside);
            Some(hr)
        }
        Action::SwapReuse { cur, spare, needs_reset, aside } => {
            let hr = if needs_reset { super::real::reset_pc_allocator(spare as *mut c_void) } else { 0 };
            let mut orphan = None;
            let (o_made, o_free) = {
                let mut g = STATE.lock().unwrap();
                if let Some(s) = g.as_mut() {
                    if hr < 0 {
                        if !restore_aside(s, a, spare, false) && spare != a && !s.orphaned.iter().any(|o| o.backing == spare) {
                            orphan = Some(spare);
                        }
                    } else {
                        finish_reset_flag(s, spare, true);
                        if s.types.contains_key(&a) {
                            s.aside.entry(a).or_default().push(cur);
                            s.cur.insert(a, spare);
                            s.swaps += 1;
                            if s.swaps.is_power_of_two() {
                                let aside_n: usize = s.aside.values().map(|v| v.len()).sum();
                                crate::xlog_info!(
                                    "D3D12.X: {} allocator resets so far with recordings still held or on the GPU – another PC allocator took its place ({} made, {aside_n} put aside now)",
                                    s.swaps,
                                    s.made
                                );
                            }
                        } else if spare != a && !s.orphaned.iter().any(|o| o.backing == spare) {
                            s.backings.remove(&spare);
                            orphan = Some(spare);
                        }
                    }
                    take_free_orphaned(s)
                } else {
                    (Vec::new(), Vec::new())
                }
            };
            RESET_DONE.notify_all();
            finish_reclaimed(o_made, o_free);
            if let Some(f) = orphan {
                drop(ID3D12CommandAllocator::from_raw(f as *mut c_void));
            }
            finish_aside(aside);
            Some(hr)
        }
        Action::SwapNew { cur, ty, aside } => {
            let made = ID3D12CommandAllocator::from_raw_borrowed(&alloc).and_then(|this| {
                let mut dev: Option<ID3D12Device> = None;
                this.GetDevice(&mut dev).ok()?;
                dev?.CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE(ty)).ok()
            });
            let Some(made) = made else {
                finish_aside(aside);
                return None;
            };
            let raw = made.into_raw();
            if !super::real::is_adopted(raw) {
                super::real::adopt(raw, super::real::Kind::CommandAllocator);
            }
            let mut orphan = None;
            {
                let mut g = STATE.lock().unwrap();
                if let Some(s) = g.as_mut() {
                    if s.types.contains_key(&a) {
                        s.made += 1;
                        s.aside.entry(a).or_default().push(cur);
                        s.cur.insert(a, raw as usize);
                        s.swaps += 1;
                        if s.swaps.is_power_of_two() {
                            let aside_n: usize = s.aside.values().map(|v| v.len()).sum();
                            crate::xlog_info!(
                                "D3D12.X: {} allocator resets so far with recordings still held or on the GPU – another PC allocator took its place ({} made, {aside_n} put aside now)",
                                s.swaps,
                                s.made
                            );
                        }
                    } else {
                        orphan = Some(raw as usize);
                    }
                }
            }
            if let Some(f) = orphan {
                drop(ID3D12CommandAllocator::from_raw(f as *mut c_void));
            }
            finish_aside(aside);
            Some(0)
        }
    }
}

/// Reset idle put-aside backings of title allocator `a` whose lists and GPU work have finished
/// (used by lock-free unit tests via `title_reset_in`).
#[cfg(test)]
unsafe fn reclaim_free_aside(s: &mut State, a: usize, reset: &mut impl FnMut(*mut c_void) -> HRESULT) {
    let ready = take_free_aside(s, a);
    for f in ready {
        let ok = reset(f as *mut c_void) >= 0;
        let _ = restore_aside(s, a, f, ok);
    }
}

/// The allocator transition, with the PC Reset supplied so failure/retry needs no GPU in tests.
#[cfg(test)]
unsafe fn title_reset_in(s: &mut State, alloc: *mut c_void, mut reset: impl FnMut(*mut c_void) -> HRESULT) -> Option<HRESULT> {
    let a = alloc as usize;
    let c = s.cur.get(&a).copied().unwrap_or(a);
    if s.backings.get(&c).is_none_or(|b| b.free()) {
        #[cfg(test)]
        reclaim_free_aside(s, a, &mut reset);
        if c == a {
            if let Some(b) = s.backings.get_mut(&c) {
                b.marks.clear();
                b.already_reset = true;
            }
            return None;
        }
        let hr = reset(c as *mut c_void);
        if hr >= 0 {
            if let Some(b) = s.backings.get_mut(&c) {
                b.marks.clear();
                b.already_reset = true;
            }
        }
        return Some(hr);
    }
    let ty = *s.types.get(&a)?;
    // one put aside earlier that nothing uses any more, or a new one
    let reuse = s.aside.get(&a).and_then(|v| v.iter().copied().find(|f| s.backings.get(f).is_none_or(|b| b.free())));
    let next = match reuse {
        Some(f) => {
            let needs = s.backings.get(&f).is_none_or(|b| !b.already_reset);
            if needs {
                // Keep ownership until Reset succeeds so a failure can be retried.
                let hr = reset(f as *mut c_void);
                if hr < 0 {
                    return Some(hr);
                }
            }
            s.aside.get_mut(&a).unwrap().retain(|&x| x != f);
            if let Some(b) = s.backings.get_mut(&f) {
                b.marks.clear();
                b.already_reset = true;
            }
            f
        }
        None => {
            let this = ID3D12CommandAllocator::from_raw_borrowed(&alloc)?;
            let mut dev: Option<ID3D12Device> = None;
            this.GetDevice(&mut dev).ok()?;
            let made: ID3D12CommandAllocator = dev?.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE(ty)).ok()?;
            let raw = made.into_raw();
            // (made through the title's device it is adopted already)
            if !super::real::is_adopted(raw) {
                super::real::adopt(raw, super::real::Kind::CommandAllocator);
            }
            s.made += 1;
            raw as usize
        }
    };
    s.aside.entry(a).or_default().push(c);
    s.cur.insert(a, next);
    #[cfg(test)]
    reclaim_free_aside(s, a, &mut reset);
    s.swaps += 1;
    if s.swaps.is_power_of_two() {
        let aside: usize = s.aside.values().map(|v| v.len()).sum();
        crate::xlog_info!(
            "D3D12.X: {} allocator resets so far with recordings still held or on the GPU – another PC allocator took its place ({} made, {aside} put aside now)",
            s.swaps,
            s.made
        );
    }
    Some(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spare_whose_reset_fails_stays_available_for_retry() {
        let (alloc, spare, list) = (0x1000usize, 0x2000usize, 0x3000usize);
        let mut s = State::default();
        s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        move_list(&mut s, list, alloc);
        s.aside.insert(alloc, vec![spare]);
        s.backings.insert(spare, Backing::default());
        let mut calls = Vec::new();

        for _ in 0..2 {
            let hr = unsafe {
                title_reset_in(&mut s, alloc as *mut c_void, |raw| {
                    calls.push(raw as usize);
                    crate::util::E_FAIL
                })
            };
            assert_eq!(hr, Some(crate::util::E_FAIL));
            assert_eq!(s.aside.get(&alloc), Some(&vec![spare]), "the failed candidate is still owned and reusable");
            assert!(!s.cur.contains_key(&alloc), "the current backing did not change");
            assert_eq!(s.of_list.get(&list), Some(&alloc));
            assert!(s.backings[&alloc].users.contains(&list), "the old recording is still held");
            assert_eq!((s.made, s.swaps), (0, 0));
        }

        let hr = unsafe {
            title_reset_in(&mut s, alloc as *mut c_void, |raw| {
                calls.push(raw as usize);
                0
            })
        };
        assert_eq!(hr, Some(0));
        assert_eq!(calls, vec![spare; 3], "all attempts used the same spare without creating another");
        assert_eq!(s.cur.get(&alloc), Some(&spare));
        assert_eq!(s.aside.get(&alloc), Some(&vec![alloc]), "only the successful reset switches backings");
        assert_eq!(s.of_list.get(&list), Some(&alloc), "the earlier recording stays on its original backing");
        assert_eq!((s.made, s.swaps), (0, 1));
    }

    /// Final-release cleanup drops only the recording's hold: another live list and a pending
    /// GPU fence each still prevent reuse, then completion makes the backing reusable.
    #[test]
    fn released_lists_leave_only_live_recordings_and_gpu_work_holding_the_backing() {
        use std::cell::Cell;
        use std::mem::{offset_of, size_of};

        #[repr(C)]
        struct Fence {
            vtable: *const usize,
            value: Cell<u64>,
            releases: Cell<u32>,
        }
        unsafe extern "system" fn completed(this: *mut c_void) -> u64 {
            (*this.cast::<Fence>()).value.get()
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let f = &*this.cast::<Fence>();
            f.releases.set(f.releases.get() + 1);
            0
        }
        extern "system" fn unexpected_call() { std::process::abort(); }

        const SLOT: usize = size_of::<usize>();
        let mut vtable = [unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT];
        vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = completed as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = release as usize;
        let mut fence = Fence { vtable: vtable.as_ptr(), value: Cell::new(0), releases: Cell::new(0) };
        let mut ids = Box::new([0u8; 3]);
        let alloc = ids.as_mut_ptr() as usize;
        let (first, second) = (alloc + 1, alloc + 2);
        created_on(first, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        created_on(second, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        {
            let mut g = STATE.lock().unwrap();
            let b = g.as_mut().unwrap().backings.get_mut(&alloc).unwrap();
            // The map owns one synthetic fence reference, dropped by forget below.
            let f = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut fence).cast()) };
            b.marks.insert(1, (f, 7));
        }

        super::super::list_x::on_release(first);
        let first_removed = {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            !s.of_list.contains_key(&first) && s.of_list.get(&second) == Some(&alloc)
                && s.backings[&alloc].users == HashSet::from([second])
                && !unsafe { s.backings[&alloc].free() }
        };
        super::super::list_x::on_release(second);
        super::super::list_x::on_release(first); // duplicate cleanup is harmless
        let pending_still_held = {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            !s.of_list.contains_key(&second) && s.backings[&alloc].users.is_empty()
                && s.backings[&alloc].marks.len() == 1 && !unsafe { s.backings[&alloc].free() }
        };
        fence.value.set(7);
        let reusable = unsafe { STATE.lock().unwrap().as_ref().unwrap().backings[&alloc].free() };
        // Clean up the synthetic COM pointer before assertions, including on the old code.
        unsafe { forget(alloc) };
        assert!(first_removed, "another live recording must remain while the released list goes");
        assert!(pending_still_held, "final list Release must retain the submitted-work fence");
        assert!(reusable, "a released list cannot keep a completed backing busy");
        assert_eq!(fence.releases.get(), 1);
    }

    /// A list recorded once and submitted again after the title reset its allocator: the
    /// recording is still there (the device is not removed), the title's allocator goes on with
    /// another PC allocator, and the first one is used again once the list has moved on.
    #[test]
    fn a_recording_outlives_the_reset_of_the_titles_allocator() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let a = alloc.as_raw();
            if !super::super::real::is_adopted(a) {
                super::super::real::adopt(a, super::super::real::Kind::CommandAllocator);
            }
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let l = list.as_raw();
            created_on(l as usize, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            list.Close().unwrap();
            let lists = [Some(list.cast::<ID3D12CommandList>().unwrap())];
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done = |v: u64| {
                queue.Signal(&fence, v).unwrap();
                let start = std::time::Instant::now();
                while fence.GetCompletedValue() < v && start.elapsed().as_secs() < 10 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                assert!(fence.GetCompletedValue() >= v, "the queue finishes");
            };

            queue.ExecuteCommandLists(&lists);
            executed(queue.as_raw(), &[l]);
            done(1);

            // the title resets its allocator (through the hook: `real::alloc_reset`) …
            alloc.Reset().unwrap();
            let second = current(a as usize);
            assert_ne!(second, a as usize, "the list still holds its recording: another PC allocator takes the place");
            // … and submits the recording again
            queue.ExecuteCommandLists(&lists);
            executed(queue.as_raw(), &[l]);
            done(2);
            assert!(dev.GetDeviceRemovedReason().is_ok(), "the recording was still there");

            // the list is reset on the title's allocator: it records on the PC allocator in use
            let second_alloc = ID3D12CommandAllocator::from_raw_borrowed(&(second as *mut c_void)).unwrap().clone();
            assert_eq!(reset_list(l as usize, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |pc| {
                assert_eq!(pc, second);
                list.Reset(&second_alloc, None).map_or_else(|e| e.code().0, |_| 0)
            }), 0);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&lists);
            executed(queue.as_raw(), &[l]);
            done(3);

            // the next reset: the first PC allocator holds nothing now and is used again
            alloc.Reset().unwrap();
            assert_eq!(current(a as usize), a as usize, "the allocator put aside is free again");
            assert!(dev.GetDeviceRemovedReason().is_ok());

            // the title's allocator goes while another PC allocator stands for it: nothing of
            // it stays (the next allocator at its address is its own)
            assert_eq!(reset_list(l as usize, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |pc| {
                assert_eq!(pc, a as usize);
                list.Reset(&alloc, None).map_or_else(|e| e.code().0, |_| 0)
            }), 0);
            list.Close().unwrap();
            alloc.Reset().unwrap();
            assert_ne!(current(a as usize), a as usize);
            drop(lists);
            drop(list);
            drop(alloc);
            assert_eq!(current(a as usize), a as usize, "a released allocator leaves nothing behind");
            assert!(!STATE.lock().unwrap().as_ref().unwrap().types.contains_key(&(a as usize)));
        }
    }

    /// A list submitted again while its previous submit is still on the GPU is counted (and
    /// logged); once that submit is done, or after the list is forgotten, it is not.
    #[test]
    fn a_resubmit_while_the_previous_submit_runs_is_noticed() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            // a fence the GPU has not reached stands for a submit still running
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut ids = Box::new([0u8; 2]);
            let (alloc, list) = (ids.as_mut_ptr() as usize, ids.as_mut_ptr() as usize + 1);
            let queue = 0x10usize as *mut c_void;
            created_on(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            {
                let mut g = STATE.lock().unwrap();
                let s = g.as_mut().unwrap();
                let recording = s.recordings[&list];
                s.last_submit.insert(list, LastSubmit { queue: queue as usize, fence: fence.clone(), value: 1, recording });
            }
            let while_running = note_resubmits(queue, &[list as *mut c_void]);
            fence.Signal(1).unwrap();
            let after_done = note_resubmits(queue, &[list as *mut c_void]);
            {
                // running again, then the list goes: nothing is left to compare with
                let mut g = STATE.lock().unwrap();
                g.as_mut().unwrap().last_submit.get_mut(&list).unwrap().value = 2;
            }
            forget_list(list);
            let after_forget = note_resubmits(queue, &[list as *mut c_void]);
            forget(alloc);
            assert_eq!(while_running, 1, "a submit while the previous one runs must be noticed");
            assert_eq!(after_done, 0, "a finished previous submit is not a resubmit");
            assert_eq!(after_forget, 0, "a forgotten list has no previous submit");
        }
    }

    #[test]
    fn retired_segment_releases_backing_after_gpu_completes_while_preserving_in_flight() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut ids = Box::new([0u8; 3]);
            let alloc = ids.as_mut_ptr() as usize;
            let seg = alloc + 1;
            let queue = alloc + 2;
            created_on(seg, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            {
                let mut g = STATE.lock().unwrap();
                let s = g.as_mut().unwrap();
                s.backings.entry(alloc).or_default().marks.insert(queue, (fence.clone(), 1));
                let recording = s.recordings[&seg];
                s.last_submit.insert(seg, LastSubmit { queue, fence: fence.clone(), value: 1, recording });
            }

            // Retiring the segment into `title_list`'s spare pool detaches it from `users`
            // while keeping GPU fence `marks` and `in_flight(seg)` intact until completion.
            retire_recording(seg);
            assert!(in_flight(seg), "retired segment must stay in_flight until GPU finishes");
            {
                let g = STATE.lock().unwrap();
                let b = &g.as_ref().unwrap().backings[&alloc];
                assert!(b.users.is_empty(), "retired segment must no longer pin backing.users");
                assert!(!b.free(), "GPU mark must still hold the backing until signaled");
            }

            fence.Signal(1).unwrap();
            assert!(!in_flight(seg));
            {
                let g = STATE.lock().unwrap();
                assert!(g.as_ref().unwrap().backings[&alloc].free(), "backing must become free once GPU completes");
            }
            forget_list(seg);
            forget(alloc);
        }
    }

    #[test]
    fn clean_title_reset_reclaims_idle_aside_backings_once() {
        let (alloc, spare1, spare2) = (0x7E57_1000usize, 0x7E57_2000usize, 0x7E57_3000usize);
        let mut s = State::default();
        s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        s.aside.insert(alloc, vec![spare1, spare2]);
        s.backings.insert(spare1, Backing::default());
        s.backings.insert(spare2, Backing::default());
        let mut resets = Vec::new();

        unsafe {
            assert_eq!(title_reset_in(&mut s, alloc as *mut c_void, |r| { resets.push(r as usize); 0 }), None);
            assert_eq!(resets, vec![spare1, spare2], "clean reset must reclaim all idle aside backings");

            // A second clean reset must not redundantly reset already-reclaimed aside backings.
            resets.clear();
            assert_eq!(title_reset_in(&mut s, alloc as *mut c_void, |r| { resets.push(r as usize); 0 }), None);
            assert!(resets.is_empty(), "already-reset aside backings should not be reset again");
        }
    }

    #[test]
    fn stalled_primary_stand_in_reset_keeps_uma_blocks_until_native_backing_resets() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut ids = Box::new([0u8; 5]);
            let base = ids.as_mut_ptr() as usize;
            let (title_alloc, spare_backing, primary, stand_in, queue) = (base, base + 1, base + 2, base + 3, base + 4);

            // 1. Primary records on logical `title_alloc` (native backing == `title_alloc`)
            //    with UMA block `0x1111` and is submitted while GPU fence stays unsignaled.
            created_on(primary, title_alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            super::super::bind::reset(primary as *mut c_void, title_alloc, 0);
            super::super::bind::push_uma_for_test(primary as *mut c_void, 0x1111);
            {
                let mut g = STATE.lock().unwrap();
                let s = g.as_mut().unwrap();
                s.aside.insert(title_alloc, vec![spare_backing]);
                s.backings.insert(spare_backing, Backing::default());
                s.backings.entry(title_alloc).or_default().marks.insert(queue, (fence.clone(), 1));
                let recording = s.recordings[&primary];
                s.last_submit.insert(primary, LastSubmit { queue, fence: fence.clone(), value: 1, recording });
            }

            // 2. Title resets list while `primary` is still in flight: `title_list::reset` selects
            //    `stand_in`, detaches `primary`'s bind blocks, and retires `primary`'s recording.
            super::super::bind::reset(primary as *mut c_void, 0, 0);
            retire_recording(primary);
            assert!(in_flight(primary), "primary must remain in_flight until GPU finishes");

            // 3. Title resets logical `title_alloc`: `alloc_pool` swaps to `spare_backing` and
            //    ONLY resets `spare_backing` (NOT `title_alloc`, which still has GPU work in flight).
            let mut native_resets = Vec::new();
            {
                let mut g = STATE.lock().unwrap();
                let s = g.as_mut().unwrap();
                assert_eq!(
                    title_reset_in(s, title_alloc as *mut c_void, |r| { native_resets.push(r as usize); 0 }),
                    Some(0)
                );
            }
            assert_eq!(native_resets, vec![spare_backing], "swap reset must only reset the idle spare backing");
            assert_eq!(current(title_alloc), spare_backing);

            // `stand_in` recording on logical `title_alloc` must key its UMA blocks to `spare_backing`.
            super::super::bind::reset(stand_in as *mut c_void, title_alloc, 0);
            super::super::bind::push_uma_for_test(stand_in as *mut c_void, 0x2222);
            super::super::bind::reset(stand_in as *mut c_void, 0, 0);

            // Clearing `spare_backing` (what `reset_pc_allocator(spare_backing)` does) leaves
            // `primary`'s blocks under stalled native backing `title_alloc` untouched.
            assert_eq!(super::super::console_list::take_pending_for_test(spare_backing), vec![0x2222]);
            assert_eq!(super::super::console_list::take_pending_for_test(title_alloc), vec![0x1111]);

            // 4. Once the GPU signals the fence on `title_alloc`, the next title reset reclaims
            //    and resets native backing `title_alloc`, which is when its blocks may be freed.
            super::super::console_list::hold_until_reset(title_alloc, vec![0x1111]);
            fence.Signal(1).unwrap();
            native_resets.clear();
            {
                let mut g = STATE.lock().unwrap();
                let s = g.as_mut().unwrap();
                assert_eq!(
                    title_reset_in(s, title_alloc as *mut c_void, |r| { native_resets.push(r as usize); 0 }),
                    Some(0)
                );
            }
            assert!(native_resets.contains(&title_alloc), "completed aside backing must be reset on reclaim");
            assert_eq!(super::super::console_list::take_pending_for_test(title_alloc), vec![0x1111]);

            super::super::bind::forget(primary as *mut c_void);
            super::super::bind::forget(stand_in as *mut c_void);
            forget_list(primary);
            forget_list(stand_in);
            // Remove synthetic non-COM `spare_backing` before `forget` drops made PC allocators.
            {
                let mut g = STATE.lock().unwrap();
                if let Some(s) = g.as_mut() {
                    s.cur.remove(&title_alloc);
                    s.aside.remove(&title_alloc);
                    s.backings.remove(&spare_backing);
                }
            }
            forget(title_alloc);
        }
    }

    #[test]
    fn reset_list_blocks_concurrent_title_reset_and_rolls_back_on_failure() {
        let mut ids = Box::new([0u8; 3]);
        let base = ids.as_mut_ptr() as usize;
        let (alloc, spare, list) = (base, base + 1, base + 2);
        {
            let mut g = STATE.lock().unwrap();
            let s = g.get_or_insert_with(State::default);
            s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            s.aside.insert(alloc, vec![spare]);
            s.backings.insert(spare, Backing::default());
        }

        let mut mid_reset_calls = Vec::new();
        let hr = reset_list(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |backing| {
            assert_eq!(backing, alloc);
            // While `reset_list` is inside native Reset, `alloc` must already be pinned by `list`
            // so a concurrent `title_reset` swaps to `spare` instead of resetting `alloc`.
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            assert!(s.backings[&alloc].users.contains(&list));
            assert!(!unsafe { s.backings[&alloc].free() });
            let r = unsafe {
                title_reset_in(s, alloc as *mut c_void, |raw| {
                    mid_reset_calls.push(raw as usize);
                    0
                })
            };
            assert_eq!(r, Some(0));
            crate::util::E_FAIL
        });

        assert_eq!(hr, crate::util::E_FAIL);
        assert_eq!(mid_reset_calls, vec![spare], "concurrent title_reset must not reset the backing mid-reset_list");
        {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            assert!(!s.of_list.contains_key(&list), "failed reset_list must roll back of_list");
            assert!(!s.backings[&alloc].users.contains(&list), "failed reset_list must roll back backing.users");
            assert_eq!(s.recordings.get(&list).copied().unwrap_or(0), 0, "failed reset_list must not bump recordings");
            // Clean up synthetic non-COM spare before forget.
            s.cur.remove(&alloc);
            s.aside.remove(&alloc);
            s.backings.remove(&spare);
        }
        unsafe { forget(alloc) };
    }

    #[test]
    fn reset_list_waits_while_backing_is_resetting() {
        let mut ids = Box::new([0u8; 2]);
        let base = ids.as_mut_ptr() as usize;
        let (alloc, list) = (base, base + 1);
        {
            let mut g = STATE.lock().unwrap();
            let s = g.get_or_insert_with(State::default);
            s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            let b = s.backings.entry(alloc).or_default();
            b.resetting = true;
            assert!(!unsafe { b.free() }, "a resetting backing must never report free");
            assert!(try_reserve_backing(s, list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0).is_none());
        }

        let cleared = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cleared_bg = cleared.clone();
        let bg = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(5));
            {
                let mut g = STATE.lock().unwrap();
                if let Some(b) = g.as_mut().and_then(|s| s.backings.get_mut(&alloc)) {
                    b.resetting = false;
                }
                cleared_bg.store(true, std::sync::atomic::Ordering::Release);
            }
            RESET_DONE.notify_all();
        });

        let hr = reset_list(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |backing| {
            assert_eq!(backing, alloc);
            assert!(cleared.load(std::sync::atomic::Ordering::Acquire), "reset_list must not hand out backing before resetting clears");
            0
        });
        bg.join().unwrap();
        assert_eq!(hr, 0);
        forget_list(list);
        unsafe { forget(alloc) };
    }

    #[test]
    fn reset_list_preserves_previous_backing_until_native_success() {
        let mut ids = Box::new([0u8; 5]);
        let base = ids.as_mut_ptr() as usize;
        let (alloc, old, current, spare, list) = (base, base + 1, base + 2, base + 3, base + 4);
        {
            let mut g = STATE.lock().unwrap();
            let s = g.get_or_insert_with(State::default);
            s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            s.cur.insert(alloc, current);
            s.aside.insert(alloc, vec![old, spare]);
            s.backings.insert(spare, Backing { already_reset: true, ..Backing::default() });
            move_list(s, list, old);
        }

        let mut native_resets = Vec::new();
        let hr = reset_list(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |backing| {
            assert_eq!(backing, current);
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            assert!(s.backings[&old].users.contains(&list));
            assert!(s.backings[&current].users.contains(&list));
            assert_eq!(
                unsafe {
                    title_reset_in(s, alloc as *mut c_void, |raw| {
                        native_resets.push(raw as usize);
                        0
                    })
                },
                Some(0)
            );
            crate::util::E_FAIL
        });

        assert_eq!(hr, crate::util::E_FAIL);
        assert_eq!(recording_backing(list), Some(old));
        assert!(!native_resets.contains(&old), "the previous recording must stay pinned until native Reset succeeds");
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert!(s.backings[&old].users.contains(&list));
            assert!(!s.backings[&current].users.contains(&list));
        }

        // Now succeed: `list` moves cleanly from `old` to the current backing (`spare`).
        let hr_ok = reset_list(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |_| 0);
        assert_eq!(hr_ok, 0);
        assert_eq!(recording_backing(list), Some(spare));
        {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            assert!(!s.backings[&old].users.contains(&list));
            assert!(s.backings[&spare].users.contains(&list));
            // Clean up synthetic non-COM backings before forget.
            s.cur.remove(&alloc);
            s.aside.remove(&alloc);
            s.backings.remove(&old);
            s.backings.remove(&current);
            s.backings.remove(&spare);
            s.of_list.remove(&list);
        }
        forget_list(list);
        unsafe { forget(alloc) };
    }

    #[test]
    fn forget_does_not_release_a_busy_original_backing_twice_and_reclaims_when_idle() {
        let mut ids = Box::new([0u8; 4]);
        let base = ids.as_mut_ptr() as usize;
        let (alloc, next, list_a, list_b) = (base, base + 1, base + 2, base + 3);
        {
            let mut g = STATE.lock().unwrap();
            let s = g.get_or_insert_with(State::default);
            s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            s.aside.insert(alloc, vec![alloc]);
            s.cur.insert(alloc, next);
            move_list(s, list_a, alloc);
            move_list(s, list_b, next);
        }
        super::super::console_list::hold_until_reset(alloc, vec![0x1111]);

        unsafe { forget(alloc) };

        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert!(!s.backings.contains_key(&alloc), "forgotten busy backing must detach from active table for address-reuse safety");
            assert!(s.orphaned.iter().any(|o| o.backing == alloc && o.state.users.contains(&list_a)));
        }
        assert_eq!(
            super::super::console_list::take_pending_for_test(alloc),
            vec![0x1111],
            "a live old recording must keep its pending UMA blocks"
        );

        // Re-hold pending UMA and retire the last live recording: orphaned backing must reclaim cleanly.
        super::super::console_list::hold_until_reset(alloc, vec![0x2222]);
        retire_recording(list_a);
        assert!(
            super::super::console_list::take_pending_for_test(alloc).is_empty(),
            "retiring the last user of an orphaned backing must drain its pending UMA"
        );
        {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            assert!(!s.orphaned.iter().any(|o| o.backing == alloc));
            // Remove synthetic non-COM `next` orphan directly so Drop does not release a fake pointer.
            s.orphaned.retain(|o| o.backing != next);
        }
        forget_list(list_a);
        forget_list(list_b);
    }

    #[test]
    fn forget_racing_mid_reset_list_cleans_up_on_rollback_and_commit() {
        let mut ids = Box::new([0u8; 4]);
        let base = ids.as_mut_ptr() as usize;
        let (prev_alloc, alloc, list) = (base, base + 1, base + 2);
        created_on(list, prev_alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);

        // Rollback path: `forget(alloc)` runs mid-`reset_list`, then native Reset fails.
        let hr = reset_list(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |b| {
            assert_eq!(b, alloc);
            unsafe { forget(alloc) };
            crate::util::E_FAIL
        });
        assert_eq!(hr, crate::util::E_FAIL);
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert!(!s.backings.contains_key(&alloc), "rolled-back forgotten backing must not remain in backings");
            assert!(!s.orphaned.iter().any(|o| o.backing == alloc), "rolled-back forgotten backing must be reclaimed");
            assert_eq!(s.of_list.get(&list), Some(&prev_alloc));
            assert!(s.backings.get(&prev_alloc).is_some_and(|b| b.users.contains(&list)));
        }

        // Commit path: `forget(alloc)` runs mid-`reset_list`, then native Reset succeeds.
        let hr = reset_list(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |b| {
            assert_eq!(b, alloc);
            unsafe { forget(alloc) };
            0
        });
        assert_eq!(hr, 0);
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert!(!s.backings.contains_key(&alloc), "committed forgotten backing must not recreate a zombie backing");
            assert!(!s.of_list.contains_key(&list), "committed forgotten backing must not leave a stale of_list entry");
            assert!(s.backings.get(&prev_alloc).is_none_or(|b| !b.users.contains(&list)));
            assert!(
                s.orphaned.iter().any(|o| o.backing == alloc && o.state.users.contains(&list) && !o.state.already_reset),
                "committed list must stay pinned on the orphaned backing"
            );
        }

        // Retiring the list reclaims the orphaned backing cleanly.
        retire_recording(list);
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert!(!s.orphaned.iter().any(|o| o.backing == alloc));
        }
        forget_list(list);
        unsafe { forget(prev_alloc) };
    }
}
