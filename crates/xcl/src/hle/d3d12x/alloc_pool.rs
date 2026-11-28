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
use std::sync::Mutex;

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
}

impl Backing {
    unsafe fn free(&self) -> bool {
        self.users.is_empty() && self.marks.values().all(|(f, v)| f.GetCompletedValue() >= *v)
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
    /// Title allocators released while their submitted GPU fences are still running: held until
    /// every fence completes (`release_finished`, polled at `PresentX`) so VRAM `io` slabs and
    /// dispatch-chain scratch are never freed out from under active GPU work.
    pending_releases: Vec<PendingRelease>,
}

struct PendingRelease {
    marks: Vec<(ID3D12Fence, u64)>,
    made: Vec<usize>,
    cleanup: Box<dyn FnOnce() + Send>,
}

struct LastSubmit {
    queue: usize,
    fence: ID3D12Fence,
    value: u64,
    recording: u64,
}

unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// `XCL_ALLOC_SWAP=0` turns it off.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_ALLOC_SWAP").is_ok_and(|v| v.trim() == "0"))
}

fn move_list(s: &mut State, list: usize, backing: usize) {
    if let Some(old) = s.of_list.insert(list, backing) {
        if old != backing {
            if let Some(b) = s.backings.get_mut(&old) {
                b.users.remove(&list);
            }
        }
    }
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
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    if let Some(backing) = s.of_list.remove(&list) {
        if let Some(b) = s.backings.get_mut(&backing) {
            b.users.remove(&list);
        }
    }
}

/// The PC list's final Release ended its recording. Submitted work still holds the backing
/// through `marks`; removing the list must not remove those fences or the other users.
pub(super) fn forget_list(list: usize) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    s.last_submit.remove(&list);
    s.recordings.remove(&list);
    if let Some(backing) = s.of_list.remove(&list) {
        if let Some(b) = s.backings.get_mut(&backing) {
            b.users.remove(&list);
        }
    }
}

/// Reset `list` on the title's allocator `alloc`, committing its new recording association
/// only after native Reset succeeds. No pool lock is held across the driver call.
pub fn reset_list(list: usize, alloc: usize, ty: i32, reset: impl FnOnce(usize) -> HRESULT) -> HRESULT {
    let backing = current(alloc);
    let hr = reset(backing);
    if hr < 0 || !enabled() || alloc == 0 {
        return hr;
    }
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    if ty >= 0 {
        s.types.insert(alloc, ty);
    }
    move_list(s, list, backing);
    *s.recordings.entry(list).or_default() += 1;
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

/// How long a submit waits for the previous submits of its lists (`XCL_RESUBMIT_WAIT_MS`, 2 000;
/// 0: not at all, as before).
fn resubmit_wait() -> std::time::Duration {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    std::time::Duration::from_millis(*MS.get_or_init(|| std::env::var("XCL_RESUBMIT_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2000)))
}

/// `lists` are about to be submitted: the ones whose previous submit the GPU has not finished
/// (`note_resubmits`) wait for it first, `limit` at most for all of them; then they go as they
/// are. On the PC a list must not be executed again before its previous execution is done (debug
/// layer id 553); a console lets a title do it, and a retail title did it about once a second in
/// its world – 128 times before its device was removed with shader memory faults, and not once in
/// a run that ended well. Returns (lists waited for, of them not done in time).
pub unsafe fn wait_for_previous_submits(lists: &[*mut c_void]) -> (usize, usize) {
    wait_for_previous_submits_within(lists, resubmit_wait())
}

unsafe fn wait_for_previous_submits_within(lists: &[*mut c_void], limit: std::time::Duration) -> (usize, usize) {
    if !enabled() || limit.is_zero() {
        return (0, 0);
    }
    // (the fences are taken out: nothing is waited for with STATE held)
    let pending: Vec<(ID3D12Fence, u64)> = {
        let g = STATE.lock().unwrap();
        let Some(s) = g.as_ref() else { return (0, 0) };
        lists
            .iter()
            .filter_map(|&l| s.last_submit.get(&(l as usize)))
            .filter(|l| l.fence.GetCompletedValue() < l.value)
            .map(|l| (l.fence.clone(), l.value))
            .collect()
    };
    if pending.is_empty() {
        return (0, 0);
    }
    let start = std::time::Instant::now();
    let mut late = 0;
    for (fence, value) in &pending {
        // (XCL's own fence, read without the title's hook; a removed device reads as the largest
        // value: no wait then. Such a submit is milliseconds from done: a short poll)
        let mut polls = 0u32;
        while fence.GetCompletedValue() < *value {
            if start.elapsed() >= limit {
                late += 1;
                break;
            }
            if polls < 64 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            polls += 1;
        }
    }
    static WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = WAITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if late > 0 {
        crate::xlog_warn!(
            "D3D12.X: {late} of {} previous submits of lists submitted again were not done after {} ms – they go as they are (#{n} such submits)",
            pending.len(),
            limit.as_millis()
        );
    } else if n <= 8 || n.is_power_of_two() {
        crate::xlog_info!(
            "D3D12.X: a submit waited {:.1} ms for the previous submits of {} of its lists (#{n} such submits)",
            start.elapsed().as_secs_f64() * 1e3,
            pending.len()
        );
    }
    (pending.len(), late)
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
    super::flight::signaled("title", key, &fence, value);
    // (how many PC lists that value stands for: 0 is a signal alone)
    super::flight::event("title lists", key as u64, value, lists.len() as u64);
    super::culprit::submitted(key, value, &lists.iter().map(|&l| l as usize).collect::<Vec<_>>());
    for &l in lists {
        if let Some(&b) = s.of_list.get(&(l as usize)) {
            let entry = s.backings.entry(b).or_default();
            entry.marks.insert(key, (fence.clone(), value));
            entry.already_reset = false;
        }
        let recording = s.recordings.get(&(l as usize)).copied().unwrap_or(0);
        s.last_submit.insert(l as usize, LastSubmit { queue: key, fence: fence.clone(), value, recording });
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

/// Like `forget`, but when `alloc` (or any of its put-aside backings) still has unfinished GPU
/// fence marks, defers running `cleanup` (and dropping any XCL-created spare PC allocators) until
/// all those fences complete (`release_finished`). Returns `true` if deferred, `false` if all
/// GPU work was already done and the caller should run cleanup immediately.
pub unsafe fn release_when_done(alloc: usize, cleanup: Box<dyn FnOnce() + Send>) -> bool {
    let (deferred, immediate_made) = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else {
            return false;
        };
        s.types.remove(&alloc);
        let mut mine: Vec<usize> = s.aside.remove(&alloc).unwrap_or_default();
        mine.extend(s.cur.remove(&alloc));
        mine.push(alloc);
        let mut pending_marks: Vec<(ID3D12Fence, u64)> = Vec::new();
        let mut made_busy: Vec<usize> = Vec::new();
        let mut made_free: Vec<usize> = Vec::new();
        for b in mine {
            let mut busy = false;
            if let Some(gone) = s.backings.remove(&b) {
                for l in gone.users {
                    s.of_list.remove(&l);
                }
                for (_, (f, v)) in gone.marks {
                    if f.GetCompletedValue() < v {
                        pending_marks.push((f, v));
                        busy = true;
                    }
                }
            }
            if b != alloc {
                if busy {
                    made_busy.push(b);
                } else {
                    made_free.push(b);
                }
            }
        }
        if pending_marks.is_empty() {
            (false, made_free)
        } else {
            s.pending_releases.push(PendingRelease {
                marks: pending_marks,
                made: made_busy,
                cleanup,
            });
            (true, made_free)
        }
    };
    for b in immediate_made {
        drop(ID3D12CommandAllocator::from_raw(b as *mut c_void));
    }
    deferred
}

/// Release excess idle (`free()`, i.e. zero recording users and all GPU marks completed)
/// XCL-created put-aside PC command allocators and reset idle un-reset title-owned put-aside
/// allocators so driver command-ring VRAM and per-allocator scratch do not accumulate across
/// put-aside allocators. Keeps up to `keep_per_alloc` idle XCL spares per title allocator.
pub(super) unsafe fn trim_idle_aside(keep_per_alloc: usize) {
    let (victims, to_reset): (Vec<usize>, Vec<(usize, ID3D12CommandAllocator)>) = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        let mut out = Vec::new();
        let mut reset = Vec::new();
        for (&owner, v) in s.aside.iter_mut() {
            let mut kept_idle = 0usize;
            let mut i = 0;
            while i < v.len() {
                let f = v[i];
                if f == owner {
                    if s.backings.get(&f).is_some_and(|b| !b.already_reset && b.free()) {
                        let raw = f as *mut c_void;
                        if let Some(held) = ID3D12CommandAllocator::from_raw_borrowed(&raw).cloned() {
                            v.remove(i);
                            reset.push((owner, held));
                            continue;
                        }
                    }
                    i += 1;
                    continue;
                }
                let idle = s.backings.get(&f).is_some_and(|b| b.free());
                if idle {
                    if kept_idle < keep_per_alloc {
                        kept_idle += 1;
                        i += 1;
                    } else {
                        v.remove(i);
                        s.backings.remove(&f);
                        out.push(f);
                    }
                } else {
                    i += 1;
                }
            }
        }
        (out, reset)
    };
    for (owner, alloc_ref) in to_reset {
        let f = alloc_ref.as_raw() as usize;
        let hr = super::real::reset_pc_allocator(f as *mut c_void);
        {
            let mut g = STATE.lock().unwrap();
            if let Some(s) = g.as_mut() {
                let _ = restore_aside(s, owner, f, hr >= 0);
            }
        }
        drop(alloc_ref);
    }
    for f in victims {
        drop(ID3D12CommandAllocator::from_raw(f as *mut c_void));
    }
}

/// Run deferred `release_when_done` cleanups whose GPU fences have all completed (polled once
/// per `PresentX`, outside `STATE` so cleanups and COM drops cannot re-enter `STATE`).
pub unsafe fn release_finished() {
    let done: Vec<PendingRelease> = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        if s.pending_releases.is_empty() {
            return;
        }
        let mut ready = Vec::new();
        let mut i = 0;
        while i < s.pending_releases.len() {
            if s.pending_releases[i].marks.iter().all(|(f, v)| f.GetCompletedValue() >= *v) {
                ready.push(s.pending_releases.swap_remove(i));
            } else {
                i += 1;
            }
        }
        ready
    };
    for item in done {
        (item.cleanup)();
        for b in item.made {
            drop(ID3D12CommandAllocator::from_raw(b as *mut c_void));
        }
    }
}

/// The title's allocator `alloc` is gone (its last Release): nothing of it stays here – a later
/// allocator may get its address, and must not go on with this one's PC allocators. The ones
/// made here go with it (those lists still record on are left alone: not freed, not tracked).
pub unsafe fn forget(alloc: usize) {
    let made: Vec<usize> = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        s.types.remove(&alloc);
        let mut mine: Vec<usize> = s.aside.remove(&alloc).unwrap_or_default();
        mine.extend(s.cur.remove(&alloc));
        mine.push(alloc);
        let mut made = Vec::new();
        for b in mine {
            let free = s.backings.get(&b).is_none_or(|x| x.free());
            if let Some(gone) = s.backings.remove(&b) {
                for l in gone.users {
                    s.of_list.remove(&l);
                }
            }
            if b != alloc && free {
                made.push(b);
            }
        }
        made
    };
    // (outside the lock: their Release comes back here)
    for b in made {
        drop(ID3D12CommandAllocator::from_raw(b as *mut c_void));
    }
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
        } else {
            i += 1;
        }
    }
    ready
}

unsafe fn restore_aside(s: &mut State, a: usize, f: usize, ok: bool) -> bool {
    if ok {
        if let Some(b) = s.backings.get_mut(&f) {
            b.marks.clear();
            b.already_reset = true;
        }
    }
    if s.types.contains_key(&a) {
        s.aside.entry(a).or_default().push(f);
        true
    } else {
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
        ReturnToPrimary { old_cur: usize, needs_reset: bool, aside: Vec<usize> },
        ResetCurrent { cur: usize, aside: Vec<usize> },
        SwapReuse { cur: usize, owner: usize, spare: usize, needs_reset: bool, aside: Vec<usize> },
        SwapNew { cur: usize, ty: i32, aside: Vec<usize> },
    }
    let action = {
        let mut g = STATE.lock().unwrap();
        let s = g.get_or_insert_with(State::default);
        let c = s.cur.get(&a).copied().unwrap_or(a);
        if s.backings.get(&c).is_none_or(|b| b.free()) {
            if c != a {
                let owner_idx = s
                    .aside
                    .get(&a)
                    .and_then(|v| v.iter().position(|&f| f == a && s.backings.get(&f).is_none_or(|b| b.free())));
                if let Some(idx) = owner_idx {
                    s.aside.get_mut(&a).unwrap().remove(idx);
                    s.cur.remove(&a);
                    s.backings.remove(&c);
                    let needs_reset = s.backings.get(&a).is_none_or(|b| !b.already_reset);
                    let aside = take_free_aside(s, a);
                    Action::ReturnToPrimary { old_cur: c, needs_reset, aside }
                } else {
                    let aside = take_free_aside(s, a);
                    Action::ResetCurrent { cur: c, aside }
                }
            } else {
                let aside = take_free_aside(s, a);
                if let Some(b) = s.backings.get_mut(&c) {
                    b.marks.clear();
                    b.already_reset = true;
                }
                Action::CleanPrimary { aside }
            }
        } else {
            let ty = *s.types.get(&a)?;
            let reuse = s
                .aside
                .get(&a)
                .and_then(|v| v.iter().position(|f| s.backings.get(f).is_none_or(|b| b.free())))
                .map(|idx| (a, idx))
                // … also one another title allocator of the type put aside: a title that resets many
                // allocators while lists hold their recordings would otherwise make a PC allocator for each,
                // and keep it (only one made here: the title's own allocator goes with the title's Release)
                .or_else(|| {
                    s.aside
                        .iter()
                        .filter(|(o, _)| **o != a && s.types.get(o) == Some(&ty))
                        .find_map(|(o, v)| {
                            v.iter()
                                .position(|f| f != o && s.backings.get(f).is_none_or(|b| b.free()))
                                .map(|idx| (*o, idx))
                        })
                });
            match reuse {
                Some((owner, idx)) => {
                    let spare = s.aside.get_mut(&owner).unwrap().remove(idx);
                    let needs_reset = s.backings.get(&spare).is_none_or(|b| !b.already_reset);
                    let aside = take_free_aside(s, a);
                    Action::SwapReuse { cur: c, owner, spare, needs_reset, aside }
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
        {
            let mut g = STATE.lock().unwrap();
            if let Some(s) = g.as_mut() {
                for (f, ok) in results {
                    if !restore_aside(s, a, f, ok) && f != a {
                        orphans.push(f);
                    }
                }
            }
        }
        for f in orphans {
            drop(ID3D12CommandAllocator::from_raw(f as *mut c_void));
        }
    };

    match action {
        Action::CleanPrimary { aside } => {
            finish_aside(aside);
            None
        }
        Action::ReturnToPrimary { old_cur, needs_reset, aside } => {
            drop(ID3D12CommandAllocator::from_raw(old_cur as *mut c_void));
            let hr = if needs_reset {
                let hr = super::real::reset_pc_allocator(a as *mut c_void);
                if hr >= 0 {
                    let mut g = STATE.lock().unwrap();
                    if let Some(s) = g.as_mut() {
                        if let Some(b) = s.backings.get_mut(&a) {
                            b.marks.clear();
                            b.already_reset = true;
                        }
                    }
                }
                hr
            } else {
                0
            };
            finish_aside(aside);
            Some(hr)
        }
        Action::ResetCurrent { cur, aside } => {
            let hr = super::real::reset_pc_allocator(cur as *mut c_void);
            if hr >= 0 {
                let mut g = STATE.lock().unwrap();
                if let Some(s) = g.as_mut() {
                    if let Some(b) = s.backings.get_mut(&cur) {
                        b.marks.clear();
                        b.already_reset = true;
                    }
                }
            }
            finish_aside(aside);
            Some(hr)
        }
        Action::SwapReuse { cur, owner, spare, needs_reset, aside } => {
            let hr = if needs_reset { super::real::reset_pc_allocator(spare as *mut c_void) } else { 0 };
            let mut orphan = None;
            {
                let mut g = STATE.lock().unwrap();
                if let Some(s) = g.as_mut() {
                    if hr < 0 {
                        if !restore_aside(s, owner, spare, false) && spare != owner {
                            orphan = Some(spare);
                        }
                    } else {
                        if let Some(b) = s.backings.get_mut(&spare) {
                            b.marks.clear();
                            b.already_reset = true;
                        }
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
                        } else if spare != a {
                            orphan = Some(spare);
                        }
                    }
                }
            }
            if let Some(f) = orphan {
                drop(ID3D12CommandAllocator::from_raw(f as *mut c_void));
            }
            finish_aside(aside);
            Some(hr)
        }
        Action::SwapNew { cur, ty, aside } => {
            let this = ID3D12CommandAllocator::from_raw_borrowed(&alloc)?;
            let mut dev: Option<ID3D12Device> = None;
            this.GetDevice(&mut dev).ok()?;
            let made: ID3D12CommandAllocator = dev?.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE(ty)).ok()?;
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
    let reuse = s
        .aside
        .get(&a)
        .and_then(|v| v.iter().copied().find(|f| s.backings.get(f).is_none_or(|b| b.free())))
        .map(|f| (a, f));
    // … also one another title allocator of the type put aside: a title that resets many
    // allocators while lists hold their recordings would otherwise make a PC allocator for each,
    // and keep it (only one made here: the title's own allocator goes with the title's Release)
    let reuse = reuse.or_else(|| {
        s.aside.iter().filter(|(o, _)| **o != a && s.types.get(o) == Some(&ty)).find_map(|(o, v)| {
            v.iter().copied().find(|f| f != o && s.backings.get(f).is_none_or(|b| b.free())).map(|f| (*o, f))
        })
    });
    let next = match reuse {
        Some((owner, f)) => {
            let needs = s.backings.get(&f).is_none_or(|b| !b.already_reset);
            if needs {
                // Keep ownership until Reset succeeds so a failure can be retried.
                let hr = reset(f as *mut c_void);
                if hr < 0 {
                    return Some(hr);
                }
            }
            s.aside.get_mut(&owner).unwrap().retain(|&x| x != f);
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

    /// A title allocator that has no free one of its own takes one another of the type put aside,
    /// but never another title allocator itself, nor one still in use, nor one of another type.
    #[test]
    fn a_free_spare_another_title_allocator_put_aside_is_taken() {
        let (a, b, c, list) = (0x1000usize, 0x4000usize, 0x6000usize, 0x3000usize);
        let (spare_b, busy_c, other_type) = (0x5000usize, 0x7000usize, 0x8000usize);
        let mut s = State::default();
        s.types.insert(a, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        s.types.insert(b, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        s.types.insert(c, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        s.types.insert(0x9000, D3D12_COMMAND_LIST_TYPE_COMPUTE.0);
        move_list(&mut s, list, a);
        s.aside.insert(b, vec![b, spare_b]);
        s.aside.insert(c, vec![busy_c]);
        s.aside.insert(0x9000, vec![other_type]);
        move_list(&mut s, 0x7100, busy_c);
        let mut calls = Vec::new();
        let hr = unsafe {
            title_reset_in(&mut s, a as *mut c_void, |raw| {
                calls.push(raw as usize);
                0
            })
        };
        assert_eq!(hr, Some(0));
        assert_eq!(calls, vec![spare_b]);
        assert_eq!(s.cur.get(&a), Some(&spare_b));
        assert_eq!(s.aside.get(&b), Some(&vec![b]), "the title's own allocator stays with it");
        assert_eq!(s.aside.get(&a), Some(&vec![a]));
        assert_eq!((s.made, s.swaps), (0, 1));
        // (now a's: it goes with a, not with b)
        assert!(!s.aside.values().flatten().any(|&f| f == spare_b));
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

    /// A list whose previous submit still runs is waited for before it is submitted again: until
    /// that submit is done, or the limit; a list with nothing running is not waited for.
    #[test]
    fn a_resubmit_waits_for_the_previous_submit() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            // a fence the GPU has not reached stands for a submit still running
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut ids = Box::new([0u8; 2]);
            let (alloc, list) = (ids.as_mut_ptr() as usize, ids.as_mut_ptr() as usize + 1);
            created_on(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            {
                let mut g = STATE.lock().unwrap();
                let s = g.as_mut().unwrap();
                let recording = s.recordings[&list];
                s.last_submit.insert(list, LastSubmit { queue: 0x10, fence: fence.clone(), value: 1, recording });
            }
            let lists = [list as *mut c_void];
            // it never finishes: the limit ends the wait, and says so
            let t = std::time::Instant::now();
            assert_eq!(wait_for_previous_submits_within(&lists, std::time::Duration::from_millis(30)), (1, 1));
            assert!(t.elapsed() >= std::time::Duration::from_millis(30));
            // no limit: no wait
            assert_eq!(wait_for_previous_submits_within(&lists, std::time::Duration::ZERO), (0, 0));
            // it finishes meanwhile: the wait ends then, far before the limit
            let signal = fence.clone();
            let done = std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(20));
                signal.Signal(1).unwrap();
            });
            let t = std::time::Instant::now();
            assert_eq!(wait_for_previous_submits_within(&lists, std::time::Duration::from_secs(20)), (1, 0));
            assert!(t.elapsed() < std::time::Duration::from_secs(10), "the wait ends when the submit is done");
            done.join().unwrap();
            // nothing runs now: nothing to wait for
            assert_eq!(wait_for_previous_submits_within(&lists, std::time::Duration::from_secs(20)), (0, 0));
            forget_list(list);
            forget(alloc);
        }
    }

    #[test]
    fn trim_idle_aside_and_return_to_primary_reclaim_spares() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let ty = D3D12_COMMAND_LIST_TYPE_COPY;
            let owner: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let cur: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let spare1: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let spare2: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            for raw in [owner.as_raw(), cur.as_raw(), spare1.as_raw(), spare2.as_raw()] {
                if !super::super::real::is_adopted(raw) {
                    super::super::real::adopt(raw, super::super::real::Kind::CommandAllocator);
                }
            }
            let owner_raw = owner.as_raw();
            let owner_key = owner_raw as usize;
            let cur_key = cur.into_raw() as usize;
            let spare1_key = spare1.into_raw() as usize;
            let spare2_key = spare2.into_raw() as usize;
            {
                let mut g = STATE.lock().unwrap();
                let s = g.get_or_insert_with(State::default);
                s.types.insert(owner_key, ty.0);
                s.cur.insert(owner_key, cur_key);
                s.aside.insert(owner_key, vec![owner_key, spare1_key, spare2_key]);
                s.backings.insert(owner_key, Backing::default());
                s.backings.insert(cur_key, Backing::default());
                s.backings.insert(spare1_key, Backing::default());
                s.backings.insert(spare2_key, Backing::default());
            }

            // Trim to 1 idle spare: drops 1 XCL spare and resets the idle un-reset owner.
            trim_idle_aside(1);
            {
                let g = STATE.lock().unwrap();
                let s = g.as_ref().unwrap();
                let aside = &s.aside[&owner_key];
                assert_eq!(aside.len(), 2, "owner + 1 kept XCL spare");
                assert!(aside.contains(&owner_key));
                assert!(s.backings[&owner_key].already_reset, "idle owner must be reset in place");
            }

            // Trim to 0 idle spares: drops the last XCL spare, leaving only the owner in aside.
            trim_idle_aside(0);
            {
                let g = STATE.lock().unwrap();
                let s = g.as_ref().unwrap();
                assert_eq!(s.aside[&owner_key], vec![owner_key]);
            }

            // Clean title_reset when both cur and owner are idle returns to primary and drops cur.
            let hr = title_reset(owner_raw);
            assert_eq!(hr, Some(0));
            {
                let g = STATE.lock().unwrap();
                let s = g.as_ref().unwrap();
                assert!(!s.cur.contains_key(&owner_key), "cur must return to primary owner");
                assert!(s.aside[&owner_key].is_empty(), "owner must be removed from aside");
                assert!(!s.backings.contains_key(&cur_key), "spare cur backing must be dropped");
            }

            forget(owner_key);
            drop(owner);
        }
    }
}
