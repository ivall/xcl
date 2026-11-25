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
    /// whether this PC backing has held recordings or GPU submits since its last Reset
    dirty: bool,
}

impl Backing {
    unsafe fn free(&self) -> bool {
        self.users.is_empty() && self.marks.values().all(|(f, v)| f.GetCompletedValue() >= *v)
    }
}

struct OrphanedRetirement {
    marks: Vec<(ID3D12Fence, u64)>,
    uma: Vec<usize>,
}

const MAX_IDLE_SPARES_PER_ALLOC: usize = 2;

#[derive(Default)]
struct ReclaimBatch {
    uma: Vec<usize>,
    drop_spares: Vec<usize>,
    count: usize,
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
    /// owned spare PC allocators detached from released title allocators while still busy
    orphaned_spares: Vec<usize>,
    /// retired UMA blocks from busy title allocators awaiting GPU fence completion
    orphaned_retirements: Vec<OrphanedRetirement>,
    made: u64,
    swaps: u64,
    /// PC list → its last submit (queue, fence, value after it, recording number then)
    last_submit: HashMap<usize, LastSubmit>,
    /// PC list → recordings so far (a Reset or creation starts the next one)
    recordings: HashMap<usize, u64>,
    resubmits: u64,
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
    let entry = s.backings.entry(backing).or_default();
    entry.users.insert(list);
    entry.dirty = true;
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

/// Reset `list` on the title's allocator `alloc`, running `on_ok()` after native Reset succeeds
/// while `list` still holds its previous backing in `users` (preventing concurrent `reclaim_idle`
/// from draining the old backing before UMA blocks are retired), then moving `list` onto `backing`.
/// No pool lock is held across `reset` or `on_ok`.
pub fn reset_list_with_retire(
    list: usize,
    alloc: usize,
    ty: i32,
    reset: impl FnOnce(usize) -> HRESULT,
    on_ok: impl FnOnce(),
) -> HRESULT {
    let backing = current(alloc);
    let hr = reset(backing);
    if hr < 0 {
        return hr;
    }
    on_ok();
    if !enabled() || alloc == 0 {
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

/// Reset `list` on the title's allocator `alloc`, committing its new recording association
/// only after native Reset succeeds. No pool lock is held across the driver call.
pub fn reset_list(list: usize, alloc: usize, ty: i32, reset: impl FnOnce(usize) -> HRESULT) -> HRESULT {
    reset_list_with_retire(list, alloc, ty, reset, || {})
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
        if let Some(&b) = s.of_list.get(&(l as usize)) {
            let entry = s.backings.entry(b).or_default();
            entry.marks.insert(key, (fence.clone(), value));
            entry.dirty = true;
        }
        let recording = s.recordings.get(&(l as usize)).copied().unwrap_or(0);
        s.last_submit.insert(l as usize, LastSubmit { queue: key, fence: fence.clone(), value, recording });
    }
}

/// The title's allocator `alloc` is gone (its last Release): nothing of it stays here – a later
/// allocator may get its address, and must not go on with this one's PC allocators. Any busy
/// owned spares or retired UMA blocks stay tracked by GPU fences and are reclaimed once idle.
pub unsafe fn forget(alloc: usize) {
    let (uma_to_free, made): (Vec<usize>, Vec<usize>) = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        s.types.remove(&alloc);
        let mut mine: Vec<usize> = s.aside.remove(&alloc).unwrap_or_default();
        mine.extend(s.cur.remove(&alloc));
        mine.push(alloc);
        let mut uma_to_free = Vec::new();
        let mut made = Vec::new();
        let mut any_busy = false;
        let mut busy_marks = Vec::new();
        for b in mine {
            let free = s.backings.get(&b).is_none_or(|x| x.free());
            if free {
                if let Some(gone) = s.backings.remove(&b) {
                    for l in gone.users {
                        s.of_list.remove(&l);
                    }
                }
                if b != alloc {
                    made.push(b);
                    uma_to_free.extend(super::console_list::take_pending(b));
                }
            } else {
                any_busy = true;
                if let Some(entry) = s.backings.get(&b) {
                    busy_marks.extend(entry.marks.values().cloned());
                }
                if b != alloc {
                    if !s.orphaned_spares.contains(&b) {
                        s.orphaned_spares.push(b);
                    }
                } else if let Some(gone) = s.backings.remove(&b) {
                    for l in gone.users {
                        s.of_list.remove(&l);
                    }
                }
            }
        }
        let title_uma = super::console_list::take_pending(alloc);
        if !title_uma.is_empty() {
            if !any_busy {
                uma_to_free.extend(title_uma);
            } else {
                let marks = if !busy_marks.is_empty() {
                    busy_marks
                } else {
                    s.fences.values().cloned().collect()
                };
                s.orphaned_retirements.push(OrphanedRetirement { marks, uma: title_uma });
            }
        }
        (uma_to_free, made)
    };
    // (outside the lock: their Release and UMA free come back safely)
    if !uma_to_free.is_empty() {
        super::console_list::free_uma_blocks(&uma_to_free);
    }
    for b in made {
        drop(ID3D12CommandAllocator::from_raw(b as *mut c_void));
    }
    maybe_reclaim_idle();
}

/// Fence-retire UMA blocks whose owning title allocator is already gone (`allocator == 0`).
pub(super) fn retire_orphan_uma(blocks: Vec<usize>) {
    if blocks.is_empty() {
        return;
    }
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    s.orphaned_retirements.push(OrphanedRetirement {
        marks: s.fences.values().cloned().collect(),
        uma: blocks,
    });
}

/// The title resets its allocator `alloc`. `None`: reset it as it is (nothing holds a
/// recording on it, or this is off). `Some(hr)`: done here – the PC allocator in use was put
/// aside with its recordings and another one took its place.
pub unsafe fn title_reset(alloc: *mut c_void) -> Option<HRESULT> {
    if !enabled() {
        return None;
    }
    let hr = {
        let mut g = STATE.lock().unwrap();
        let s = g.get_or_insert_with(State::default);
        title_reset_in(s, alloc, |raw| super::real::reset_pc_allocator(raw))
    };
    maybe_reclaim_idle();
    hr
}

/// The allocator transition, with the PC Reset supplied so failure/retry needs no GPU in tests.
unsafe fn title_reset_in(s: &mut State, alloc: *mut c_void, mut reset: impl FnMut(*mut c_void) -> HRESULT) -> Option<HRESULT> {
    let a = alloc as usize;
    let c = s.cur.get(&a).copied().unwrap_or(a);
    if s.backings.get(&c).is_none_or(|b| b.free()) {
        if let Some(b) = s.backings.get_mut(&c) {
            b.marks.clear();
            b.dirty = false;
        }
        if c == a {
            return None;
        }
        return Some(reset(c as *mut c_void));
    }
    let ty = *s.types.get(&a)?;
    // one put aside earlier that nothing uses any more, or a new one
    let reuse = s.aside.get(&a).and_then(|v| v.iter().copied().find(|f| s.backings.get(f).is_none_or(|b| b.free())));
    let next = match reuse {
        Some(f) => {
            // Keep ownership until Reset succeeds so a failure can be retried.
            let hr = reset(f as *mut c_void);
            if hr < 0 {
                return Some(hr);
            }
            s.aside.get_mut(&a).unwrap().retain(|&x| x != f);
            if let Some(b) = s.backings.get_mut(&f) {
                b.marks.clear();
                b.dirty = false;
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

unsafe fn reclaim_idle_in<R, TU>(
    s: &mut State,
    mut reset: R,
    mut take_uma: TU,
) -> ReclaimBatch
where
    R: FnMut(*mut c_void) -> HRESULT,
    TU: FnMut(usize) -> Vec<usize>,
{
    let mut batch = ReclaimBatch::default();

    let aside_keys: Vec<usize> = s.aside.keys().copied().collect();
    for title_alloc in aside_keys {
        let cur_free = s.cur.get(&title_alloc).is_none_or(|c| s.backings.get(c).is_none_or(|b| b.free()));
        let Some(mut spares) = s.aside.remove(&title_alloc) else {
            continue;
        };
        let all_backings_free = cur_free && spares.iter().all(|f| s.backings.get(f).is_none_or(|b| b.free()));
        for &f in &spares {
            if !s.backings.get(&f).is_some_and(|b| b.dirty && b.free()) {
                continue;
            }
            if f == title_alloc {
                // Title owns `title_alloc`'s COM lifetime (`Release` runs before `forget`), and
                // `PENDING[title_alloc]` stays keyed by the title allocator on this tree: only flush
                // when every backing of `title_alloc` is idle, without dereferencing `title_alloc` off-thread.
                if all_backings_free {
                    if let Some(b) = s.backings.get_mut(&f) {
                        b.marks.clear();
                        b.dirty = false;
                    }
                    let u = take_uma(f);
                    if !u.is_empty() {
                        batch.uma.extend(u);
                        batch.count += 1;
                    }
                }
            } else if reset(f as *mut c_void) >= 0 {
                if let Some(b) = s.backings.get_mut(&f) {
                    b.marks.clear();
                    b.dirty = false;
                }
                batch.uma.extend(take_uma(f));
                batch.count += 1;
            }
        }
        let mut clean_free_owned = spares
            .iter()
            .filter(|&&f| f != title_alloc && s.backings.get(&f).is_none_or(|b| !b.dirty && b.free()))
            .count();
        while clean_free_owned > MAX_IDLE_SPARES_PER_ALLOC {
            let Some(pos) = spares
                .iter()
                .position(|&f| f != title_alloc && s.backings.get(&f).is_none_or(|b| !b.dirty && b.free()))
            else {
                break;
            };
            let f = spares.remove(pos);
            if let Some(gone) = s.backings.remove(&f) {
                for l in gone.users {
                    s.of_list.remove(&l);
                }
            }
            batch.uma.extend(take_uma(f));
            batch.drop_spares.push(f);
            batch.count += 1;
            clean_free_owned -= 1;
        }
        if !spares.is_empty() {
            s.aside.insert(title_alloc, spares);
        }
    }

    let orphaned = std::mem::take(&mut s.orphaned_spares);
    for b in orphaned {
        if s.backings.get(&b).is_none_or(|x| x.free()) {
            if let Some(gone) = s.backings.remove(&b) {
                for l in gone.users {
                    s.of_list.remove(&l);
                }
            }
            batch.uma.extend(take_uma(b));
            batch.drop_spares.push(b);
            batch.count += 1;
        } else {
            s.orphaned_spares.push(b);
        }
    }

    let retirements = std::mem::take(&mut s.orphaned_retirements);
    for r in retirements {
        if r.marks.iter().all(|(f, v)| f.GetCompletedValue() >= *v) {
            batch.uma.extend(r.uma);
            batch.count += 1;
        } else {
            s.orphaned_retirements.push(r);
        }
    }

    batch
}

/// Proactively reset idle put-aside PC allocators, trim excess spares, and flush completed
/// orphaned retirements so command allocator memory and held UMA blocks return promptly.
pub(super) fn reclaim_idle() -> usize {
    if !enabled() {
        return 0;
    }
    let batch = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else {
            return 0;
        };
        unsafe {
            reclaim_idle_in(
                s,
                |raw| {
                    if super::real::is_adopted(raw) {
                        super::real::reset_pc_allocator(raw)
                    } else {
                        crate::util::E_FAIL
                    }
                },
                super::console_list::take_pending,
            )
        }
    };
    let count = batch.count;
    if !batch.uma.is_empty() {
        unsafe { super::console_list::free_uma_blocks(&batch.uma) };
    }
    for b in batch.drop_spares {
        unsafe { drop(ID3D12CommandAllocator::from_raw(b as *mut c_void)) };
    }
    count
}

const RECLAIM_PERIOD: std::time::Duration = std::time::Duration::from_millis(500);

fn maybe_reclaim_idle() {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let Ok(mut last) = LAST.try_lock() else {
        return;
    };
    let now = std::time::Instant::now();
    if last.is_some_and(|t| now.saturating_duration_since(t) < RECLAIM_PERIOD) {
        return;
    }
    *last = Some(now);
    drop(last);
    let _ = reclaim_idle();
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
    fn reclaim_idle_resets_dirty_free_aside_backings_and_trims_excess_spares() {
        let mut ids = Box::new([0u8; 8]);
        let base = ids.as_mut_ptr() as usize;
        let (title, spare_busy, spare_dirty_free, spare_clean_1, spare_clean_2, list) =
            (base, base + 1, base + 2, base + 3, base + 4, base + 5);

        let mut s = State::default();
        s.types.insert(title, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        // `spare_busy` holds a live recording so it cannot be reset or trimmed yet.
        move_list(&mut s, list, spare_busy);
        s.backings.insert(title, Backing { dirty: true, ..Default::default() });
        s.backings.insert(spare_dirty_free, Backing { dirty: true, ..Default::default() });
        // Two additional clean free spares push clean free owned spares to 3 (> MAX_IDLE_SPARES_PER_ALLOC = 2).
        s.backings.insert(spare_clean_1, Backing { dirty: false, ..Default::default() });
        s.backings.insert(spare_clean_2, Backing { dirty: false, ..Default::default() });
        s.aside.insert(title, vec![title, spare_busy, spare_dirty_free, spare_clean_1, spare_clean_2]);

        let mut resets = Vec::new();
        let mut uma_taken = Vec::new();
        let batch = unsafe {
            reclaim_idle_in(
                &mut s,
                |raw| {
                    resets.push(raw as usize);
                    0
                },
                |b| {
                    uma_taken.push(b);
                    vec![b + 0x1000]
                },
            )
        };

        assert_eq!(
            resets,
            vec![spare_dirty_free],
            "only dirty free owned spares are natively reset; title-owned allocator is never dereferenced on background reclaim"
        );
        assert!(
            s.backings[&title].dirty,
            "title backing keeps pending UMA while spare_busy is still active"
        );
        assert!(s.backings[&spare_busy].dirty, "busy spare stays dirty and aside");
        assert!(!s.backings.contains_key(&spare_dirty_free), "oldest excess clean owned spare is removed");
        assert_eq!(batch.drop_spares, vec![spare_dirty_free], "excess owned spare above limit is queued for drop");
        assert!(!batch.drop_spares.contains(&title), "title's own allocator is never dropped by trim");
        assert_eq!(s.aside.get(&title), Some(&vec![title, spare_busy, spare_clean_1, spare_clean_2]));
        assert_eq!(uma_taken, vec![spare_dirty_free, spare_dirty_free]);
        assert_eq!(batch.uma, vec![spare_dirty_free + 0x1000, spare_dirty_free + 0x1000]);
        assert_eq!(batch.count, 2);

        // Once `spare_busy` finishes, all backings of `title` are free -> `title` and `spare_busy` flush cleanly.
        s.of_list.remove(&list);
        s.backings.get_mut(&spare_busy).unwrap().users.remove(&list);
        uma_taken.clear();
        let batch2 = unsafe {
            reclaim_idle_in(
                &mut s,
                |raw| {
                    resets.push(raw as usize);
                    0
                },
                |b| {
                    uma_taken.push(b);
                    vec![b + 0x2000]
                },
            )
        };
        assert!(!s.backings[&title].dirty, "title backing flushes once all backings are idle");
        assert!(!s.backings.contains_key(&spare_busy), "newly free excess spare is trimmed");
        assert_eq!(uma_taken, vec![title, spare_busy, spare_busy]);
        assert_eq!(batch2.drop_spares, vec![spare_busy]);
        assert_eq!(batch2.count, 3);
        drop(ids);
    }

    #[test]
    fn forget_retires_busy_owned_spares_and_title_alloc_pending_after_gpu_fence() {
        use std::cell::Cell;
        use std::mem::{offset_of, size_of};

        #[repr(C)]
        struct FakeFence {
            vtable: *const usize,
            value: Cell<u64>,
            refs: Cell<u32>,
        }
        unsafe extern "system" fn fence_add_ref(this: *mut c_void) -> u32 {
            let f = &*this.cast::<FakeFence>();
            let next = f.refs.get() + 1;
            f.refs.set(next);
            next
        }
        unsafe extern "system" fn fence_completed(this: *mut c_void) -> u64 {
            (*this.cast::<FakeFence>()).value.get()
        }
        unsafe extern "system" fn fence_release(this: *mut c_void) -> u32 {
            let f = &*this.cast::<FakeFence>();
            let next = f.refs.get().saturating_sub(1);
            f.refs.set(next);
            next
        }
        extern "system" fn unexpected_call() {
            std::process::abort();
        }
        #[repr(C)]
        struct FakeAlloc {
            vtable: *const usize,
            releases: Cell<u32>,
        }
        unsafe extern "system" fn alloc_fake_release(this: *mut c_void) -> u32 {
            let a = &*this.cast::<FakeAlloc>();
            a.releases.set(a.releases.get() + 1);
            0
        }

        const SLOT: usize = size_of::<usize>();
        let mut fence_vtable = [unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT];
        fence_vtable[1] = fence_add_ref as usize;
        fence_vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = fence_completed as usize;
        fence_vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = fence_release as usize;
        let alloc_vtable = [0usize, 0usize, alloc_fake_release as usize];

        let mut fence_title = Box::new(FakeFence { vtable: fence_vtable.as_ptr(), value: Cell::new(0), refs: Cell::new(1) });
        let mut fence_spare = Box::new(FakeFence { vtable: fence_vtable.as_ptr(), value: Cell::new(0), refs: Cell::new(1) });
        let mut fake_spare = Box::new(FakeAlloc { vtable: alloc_vtable.as_ptr(), releases: Cell::new(0) });
        let mut ids = Box::new([0u8; 4]);
        let title = ids.as_mut_ptr() as usize;
        let list = title + 1;
        let spare_busy = std::ptr::from_mut(&mut *fake_spare) as usize;

        {
            let mut g = STATE.lock().unwrap();
            let st = g.get_or_insert_with(State::default);
            st.cur.insert(title, title);
            st.aside.insert(title, vec![spare_busy]);
            let mut b_title = Backing { dirty: true, ..Default::default() };
            let ft = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut *fence_title).cast()) };
            b_title.marks.insert(1, (ft, 5));
            st.backings.insert(title, b_title);

            move_list(st, list, spare_busy);
            let fs = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut *fence_spare).cast()) };
            st.backings.get_mut(&spare_busy).unwrap().marks.insert(2, (fs, 9));
        }

        super::super::console_list::hold_until_reset(title, vec![0x7E57_1000]);
        super::super::console_list::hold_until_reset(spare_busy, vec![0x7E57_2000]);

        unsafe { forget(title) };
        assert_eq!(fake_spare.releases.get(), 0, "busy owned spare must not be COM-released early");
        assert_eq!(super::super::console_list::pending_len_for_test(title), 0, "busy title UMA captured into orphaned_retirements");
        assert_eq!(super::super::console_list::pending_len_for_test(spare_busy), 1, "busy spare UMA remains under spare_busy until free");

        // While GPU fences are unsatisfied, reclaim_idle_in holds both intact.
        let batch_busy = {
            let mut g = STATE.lock().unwrap();
            let st = g.as_mut().unwrap();
            assert!(!st.backings.contains_key(&title), "title removed from backings to prevent ABA aliasing");
            assert!(st.orphaned_spares.contains(&spare_busy));
            assert_eq!(st.orphaned_retirements.len(), 1);
            unsafe { reclaim_idle_in(st, |_| 0, super::super::console_list::take_pending) }
        };
        assert_eq!(batch_busy.count, 0);

        // Complete fences and finish list -> both retire cleanly.
        fence_title.value.set(5);
        forget_list(list);
        fence_spare.value.set(9);
        let batch_done = {
            let mut g = STATE.lock().unwrap();
            let st = g.as_mut().unwrap();
            unsafe { reclaim_idle_in(st, |_| 0, super::super::console_list::take_pending) }
        };
        assert_eq!(batch_done.count, 2);
        assert_eq!(batch_done.uma, vec![0x7E57_2000, 0x7E57_1000]);
        assert_eq!(batch_done.drop_spares, vec![spare_busy]);
        assert_eq!(fence_title.refs.get(), 0);
        assert_eq!(fence_spare.refs.get(), 0);
        for b in batch_done.drop_spares {
            unsafe { drop(ID3D12CommandAllocator::from_raw(b as *mut c_void)) };
        }
        assert_eq!(fake_spare.releases.get(), 1);
        drop(ids);
    }

    #[test]
    fn forget_busy_title_alloc_without_marks_and_reset_retire_ordering() {
        use std::cell::Cell;
        use std::mem::{offset_of, size_of};

        #[repr(C)]
        struct FakeFence {
            vtable: *const usize,
            value: Cell<u64>,
            refs: Cell<u32>,
        }
        unsafe extern "system" fn fence_add_ref(this: *mut c_void) -> u32 {
            let f = &*this.cast::<FakeFence>();
            let next = f.refs.get() + 1;
            f.refs.set(next);
            next
        }
        unsafe extern "system" fn fence_completed(this: *mut c_void) -> u64 {
            (*this.cast::<FakeFence>()).value.get()
        }
        unsafe extern "system" fn fence_release(this: *mut c_void) -> u32 {
            let f = &*this.cast::<FakeFence>();
            let next = f.refs.get().saturating_sub(1);
            f.refs.set(next);
            next
        }
        extern "system" fn unexpected_call() {
            std::process::abort();
        }
        const SLOT: usize = size_of::<usize>();
        let mut fence_vtable = [unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT];
        fence_vtable[1] = fence_add_ref as usize;
        fence_vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = fence_completed as usize;
        fence_vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = fence_release as usize;

        let mut fake_fence = Box::new(FakeFence { vtable: fence_vtable.as_ptr(), value: Cell::new(0), refs: Cell::new(1) });
        let mut ids = Box::new([0u8; 8]);
        let base = ids.as_mut_ptr() as usize;
        let (title_a, list_a, queue_key, title_b, spare_b, list_b) = (base, base + 1, base + 2, base + 3, base + 4, base + 5);

        // Part 1: busy title_a via live `list_a` ONLY (marks empty) captures PENDING[title_a] into orphaned_retirements.
        {
            let mut g = STATE.lock().unwrap();
            let st = g.get_or_insert_with(State::default);
            let f = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut *fake_fence).cast()) };
            st.fences.insert(queue_key, (f, 5));
            move_list(st, list_a, title_a);
        }
        super::super::console_list::hold_until_reset(title_a, vec![0x7E57_3000]);

        unsafe { forget(title_a) };
        assert_eq!(fake_fence.refs.get(), 2, "orphaned retirement cloned the queue fence ref");
        assert_eq!(super::super::console_list::pending_len_for_test(title_a), 0, "busy title UMA decoupled from dead address");
        let busy = {
            let mut g = STATE.lock().unwrap();
            let st = g.as_mut().unwrap();
            st.fences.remove(&queue_key);
            unsafe { reclaim_idle_in(st, |_| 0, super::super::console_list::take_pending) }
        };
        assert_eq!(busy.count, 0, "unsignaled queue fence must hold orphaned retirement");
        assert_eq!(fake_fence.refs.get(), 1);
        fake_fence.value.set(5);
        let done = {
            let mut g = STATE.lock().unwrap();
            let st = g.as_mut().unwrap();
            unsafe { reclaim_idle_in(st, |_| 0, super::super::console_list::take_pending) }
        };
        assert_eq!(done.count, 1);
        assert_eq!(done.uma, vec![0x7E57_3000]);
        assert_eq!(fake_fence.refs.get(), 0);

        // Part 2: reset_list_with_retire runs `on_ok` while `list_b` still holds `title_b` in users.
        {
            let mut g = STATE.lock().unwrap();
            let st = g.get_or_insert_with(State::default);
            st.types.insert(title_b, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            st.aside.insert(title_b, vec![title_b]);
            st.cur.insert(title_b, spare_b);
            move_list(st, list_b, title_b);
        }
        let mut mid_reclaim_count = usize::MAX;
        let hr = reset_list_with_retire(
            list_b,
            title_b,
            D3D12_COMMAND_LIST_TYPE_DIRECT.0,
            |_| 0,
            || {
                // `list_b` still holds `title_b` while `on_ok` retires `0x7E57_4000` onto `title_b`.
                super::super::console_list::hold_until_reset(title_b, vec![0x7E57_4000]);
                let mut g = STATE.lock().unwrap();
                let st = g.as_mut().unwrap();
                let mid = unsafe { reclaim_idle_in(st, |_| 0, super::super::console_list::take_pending) };
                mid_reclaim_count = mid.count;
            },
        );
        assert_eq!((hr, mid_reclaim_count), (0, 0), "concurrent reclaim must not fire while on_ok retires UMA onto old backing");

        // While `list_b` still records on `spare_b`, `PENDING[title_b]` stays deferred until every backing is idle.
        let while_recording = {
            let mut g = STATE.lock().unwrap();
            let st = g.as_mut().unwrap();
            unsafe { reclaim_idle_in(st, |_| 0, super::super::console_list::take_pending) }
        };
        assert_eq!(while_recording.count, 0, "busy cur spare must defer title_b pending UMA flush");

        forget_list(list_b);
        let after = {
            let mut g = STATE.lock().unwrap();
            let st = g.as_mut().unwrap();
            let after = unsafe { reclaim_idle_in(st, |_| 0, super::super::console_list::take_pending) };
            st.backings.clear();
            st.of_list.clear();
            st.cur.clear();
            st.aside.clear();
            st.types.clear();
            after
        };
        assert_eq!(after.uma, vec![0x7E57_4000]);
        assert_eq!(after.count, 1);
        drop(ids);
    }
}
