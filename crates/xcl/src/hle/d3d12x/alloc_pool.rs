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
    s.backings.entry(backing).or_default().users.insert(list);
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

/// `list` was retired to a spare pool after its wrapper reset and will not be submitted again
/// with its previous recording: drop its backing pin (`of_list` / `users`) while preserving
/// `last_submit` and `recordings` so `in_flight(list)` still blocks spare reuse until the GPU
/// finishes any submitted work (and `b.marks` still protects the backing allocator).
pub(super) fn release_recording(list: usize) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
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

/// `list` (of type `ty`; negative: unknown) was made recording on `pc_alloc` for `title_alloc`
/// (`CreateCommandList`, a mid-list split segment, or a reset stand-in).
pub fn created_on_backing(list: usize, title_alloc: usize, pc_alloc: usize, ty: i32) {
    if !enabled() || title_alloc == 0 || pc_alloc == 0 {
        return;
    }
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    if ty >= 0 {
        s.types.entry(title_alloc).or_insert(ty);
    }
    move_list(s, list, pc_alloc);
    *s.recordings.entry(list).or_default() += 1;
}

/// `list` (of type `ty`) was made recording on the PC allocator `raw` itself
/// (CreateCommandList).
#[cfg(test)]
fn created_on(list: usize, raw: usize, ty: i32) {
    created_on_backing(list, raw, raw, ty);
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
            s.backings.entry(b).or_default().marks.insert(key, (fence.clone(), value));
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

/// The title resets its allocator `alloc`. `None`: reset it as it is (nothing holds a
/// recording on it, or this is off). `Some(hr)`: done here – the PC allocator in use was put
/// aside with its recordings and another one took its place.
pub unsafe fn title_reset(alloc: *mut c_void) -> Option<HRESULT> {
    if !enabled() {
        return None;
    }
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    title_reset_in(s, alloc, |raw| super::real::reset_pc_allocator(raw))
}

/// The allocator transition, with the PC Reset supplied so failure/retry needs no GPU in tests.
unsafe fn title_reset_in(s: &mut State, alloc: *mut c_void, mut reset: impl FnMut(*mut c_void) -> HRESULT) -> Option<HRESULT> {
    let a = alloc as usize;
    let c = s.cur.get(&a).copied().unwrap_or(a);
    if s.backings.get(&c).is_none_or(|b| b.free()) {
        if let Some(b) = s.backings.get_mut(&c) {
            b.marks.clear();
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

    /// Retiring a split/stand-in segment into `title_list`'s spare pool must release its
    /// allocator backing pin while keeping `in_flight` active until the GPU finishes its submit.
    #[test]
    fn release_recording_unpins_backing_while_preserving_in_flight() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut ids = Box::new([0u8; 3]);
            let base = ids.as_mut_ptr() as usize;
            let (title_alloc, second, seg) = (base, base + 1, base + 2);
            let queue = 0x20usize as *mut c_void;
            created_on_backing(seg, title_alloc, second, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            {
                let mut g = STATE.lock().unwrap();
                let s = g.as_mut().unwrap();
                let recording = s.recordings[&seg];
                s.last_submit.insert(seg, LastSubmit { queue: queue as usize, fence: fence.clone(), value: 1, recording });
            }
            assert_eq!(recording_backing_for_test(seg), Some(second));
            assert!(in_flight(seg));

            release_recording(seg);
            assert_eq!(recording_backing_for_test(seg), None, "retired spare must release its backing pin");
            assert!(in_flight(seg), "retired spare must remain in_flight until the GPU finishes");
            assert_eq!(note_resubmits(queue, &[seg as *mut c_void]), 1);

            fence.Signal(1).unwrap();
            assert!(!in_flight(seg));
            forget_list(seg);
            forget(title_alloc);
            {
                let mut g = STATE.lock().unwrap();
                if let Some(s) = g.as_mut() {
                    s.backings.remove(&second);
                }
            }
        }
    }

    /// Creating a new command list or resetting an adopted PC list after `alloc` swapped to a
    /// substitute backing must record on `current(alloc)` so a later `alloc.Reset()` keeps the
    /// live recording intact instead of resetting the active backing underneath it.
    #[test]
    fn cl_reset_and_new_list_after_swap_use_active_backing() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let a = alloc.as_raw();
            if !super::super::real::is_adopted(a) {
                super::super::real::adopt(a, super::super::real::Kind::CommandAllocator);
            }
            let list1: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            created_on(list1.as_raw() as usize, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            list1.Close().unwrap();

            // First allocator reset puts `a` aside because `list1` still holds a recording on it.
            alloc.Reset().unwrap();
            let second = current(a as usize);
            assert_ne!(second, a as usize);

            // Resetting an adopted PC command list via `ID3D12GraphicsCommandList::Reset` with the
            // title's `alloc` must redirect to `second = current(a)`, not record onto busy `a`.
            let list2: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let l2 = list2.as_raw();
            if !super::super::real::is_adopted(l2) {
                super::super::real::adopt(l2, super::super::real::Kind::CommandList);
            }
            list2.Close().unwrap();
            list2.Reset(&alloc, None).expect("cl_reset must resolve title allocator to active backing");
            assert_eq!(recording_backing_for_test(l2 as usize), Some(second), "cl_reset must track list2 on the active substitute backing");
            list2.Close().unwrap();

            // A second allocator reset must keep `second` aside because `list2` is live on it.
            alloc.Reset().unwrap();
            let third = current(a as usize);
            assert_ne!(third, second, "active substitute backing holding list2 must not be reset");
            assert!(dev.GetDeviceRemovedReason().is_ok());

            drop(list2);
            drop(list1);
            drop(alloc);
            assert_eq!(current(a as usize), a as usize);
        }
    }

    #[test]
    fn created_list_after_swap_tracks_active_backing() {
        let (title_alloc, second, l1, l2) = (0x7E57_A000usize, 0x7E57_B000usize, 0x7E57_C000usize, 0x7E57_D000usize);
        {
            let mut g = STATE.lock().unwrap();
            let s = g.get_or_insert_with(State::default);
            s.types.insert(title_alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            s.cur.insert(title_alloc, second);
            s.aside.insert(title_alloc, vec![title_alloc]);
            move_list(s, l1, title_alloc);
        }
        created_on_backing(l2, title_alloc, second, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        assert_eq!(recording_backing_for_test(l2), Some(second));
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert!(!s.types.contains_key(&second));
            assert!(s.backings[&second].users.contains(&l2));
            assert!(!s.backings[&second].users.contains(&l1));
            assert_eq!(s.recordings.get(&l2).copied(), Some(1));
        }
        forget_list(l2);
        assert_eq!(recording_backing_for_test(l2), None);
        {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            assert!(s.backings[&second].users.is_empty());
            s.aside.remove(&title_alloc);
            s.cur.remove(&title_alloc);
            s.types.remove(&title_alloc);
            s.backings.remove(&second);
            s.backings.remove(&title_alloc);
            s.of_list.remove(&l1);
        }
    }

    /// Concurrent allocator swaps (`title_reset_in`) racing with command-list resets (`reset_list`),
    /// newly created segments (`created_on_backing`), and spare retirements (`release_recording`)
    /// must keep `of_list` and `backings[b].users` consistent and always pin each list to the
    /// exact backing its driver Reset received.
    #[test]
    fn concurrent_allocator_reset_and_list_reset_keep_backing_invariants() {
        use std::sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc, Barrier,
        };

        // Unique synthetic pointers outside any real heap/test ID range.
        let mut ids = Box::new([0u8; 32]);
        let base = ids.as_mut_ptr() as usize;
        let title_alloc = base;
        let spares: Vec<usize> = (1..=6).map(|i| base + i).collect();
        let all_backings: Vec<usize> = std::iter::once(title_alloc).chain(spares.iter().copied()).collect();
        let pin_list = base + 8;
        let worker_lists: Vec<usize> = (0..4).map(|i| base + 10 + i).collect();

        {
            let mut g = STATE.lock().unwrap();
            let s = g.get_or_insert_with(State::default);
            s.types.insert(title_alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            s.aside.insert(title_alloc, spares.clone());
            for &b in &spares {
                s.backings.entry(b).or_default();
            }
            move_list(s, pin_list, title_alloc);
        }

        // 1. Deterministic interleaving: thread 1 reads `current(title_alloc)` = `title_alloc` inside
        //    `reset_list`, pauses before returning from the driver callback while thread 2 swaps
        //    `title_alloc` to a spare backing, then thread 1 commits. `worker_lists[0]` must stay
        //    pinned to `title_alloc` (the backing its Reset actually used), not the swapped spare.
        let in_reset = Arc::new(Barrier::new(2));
        let swapped = Arc::new(Barrier::new(2));
        let l0 = worker_lists[0];
        let (in_reset_t, swapped_t) = (in_reset.clone(), swapped.clone());
        let observed = Arc::new(AtomicUsize::new(0));
        let observed_t = observed.clone();
        let t1 = std::thread::spawn(move || {
            let hr = reset_list(l0, title_alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |pc_alloc| {
                observed_t.store(pc_alloc, Ordering::SeqCst);
                in_reset_t.wait();
                swapped_t.wait();
                0
            });
            assert_eq!(hr, 0);
        });
        in_reset.wait();
        let swap_hr = {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            unsafe { title_reset_in(s, title_alloc as *mut c_void, |_| 0) }
        };
        assert_eq!(swap_hr, Some(0));
        let after_swap = current(title_alloc);
        assert_ne!(after_swap, title_alloc);
        swapped.wait();
        t1.join().unwrap();
        assert_eq!(observed.load(Ordering::SeqCst), title_alloc);
        assert_eq!(recording_backing_for_test(l0), Some(title_alloc), "reset_list must commit the exact backing passed to Reset");
        // Unpin `pin_list`; `l0` still pins `title_alloc`, so `title_alloc` is not free yet.
        release_recording(pin_list);
        assert!(!unsafe { STATE.lock().unwrap().as_ref().unwrap().backings[&title_alloc].free() });

        // 2. Multi-threaded stress: allocator resets race with concurrent list resets, creations,
        //    and retirements across 4 worker threads.
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for (idx, &list_id) in worker_lists.iter().enumerate() {
            let stop_w = stop.clone();
            handles.push(std::thread::spawn(move || {
                for step in 0..200usize {
                    if stop_w.load(Ordering::Relaxed) {
                        break;
                    }
                    let chosen = AtomicUsize::new(0);
                    let hr = reset_list(list_id, title_alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |pc| {
                        chosen.store(pc, Ordering::Relaxed);
                        if (step + idx) % 11 == 0 { crate::util::E_FAIL } else { 0 }
                    });
                    if hr >= 0 {
                        assert_eq!(recording_backing_for_test(list_id), Some(chosen.load(Ordering::Relaxed)));
                    }
                    if (step + idx) % 5 == 0 {
                        release_recording(list_id);
                    } else if (step + idx) % 7 == 0 {
                        let b = current(title_alloc);
                        created_on_backing(list_id, title_alloc, b, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
                    }
                }
            }));
        }
        let resetter = std::thread::spawn(move || {
            for _ in 0..200usize {
                let mut g = STATE.lock().unwrap();
                let s = g.as_mut().unwrap();
                // Keep at least one backing free in `aside` when all spares are briefly busy
                // so `title_reset_in` never tries to call COM `GetDevice` on the synthetic pointer.
                let cur_b = s.cur.get(&title_alloc).copied().unwrap_or(title_alloc);
                let cur_free = s.backings.get(&cur_b).is_none_or(|b| unsafe { b.free() });
                let has_free_aside = s.aside.get(&title_alloc).is_some_and(|v| v.iter().any(|f| s.backings.get(f).is_none_or(|b| unsafe { b.free() })));
                let busy_now: HashSet<usize> = s.backings.iter().filter(|(_, b)| !b.users.is_empty()).map(|(&k, _)| k).collect();
                if cur_free || has_free_aside {
                    let hr = unsafe {
                        title_reset_in(s, title_alloc as *mut c_void, |raw| {
                            // Verify no active list user is pinned on a backing being reset.
                            assert!(!busy_now.contains(&(raw as usize)), "resetting busy backing {raw:?}");
                            0
                        })
                    };
                    assert!(matches!(hr, None | Some(0)));
                }
                drop(g);
                std::thread::yield_now();
            }
        });
        for h in handles {
            h.join().unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        resetter.join().unwrap();

        // Check `of_list` <-> `backings[b].users` bijection and clean up all synthetic IDs.
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            for &l in &worker_lists {
                if let Some(&b) = s.of_list.get(&l) {
                    assert!(all_backings.contains(&b), "list mapped to unknown backing {b:#x}");
                    assert!(s.backings[&b].users.contains(&l), "missing reverse user link for list {l:#x}");
                }
            }
            for &b in &all_backings {
                if let Some(bk) = s.backings.get(&b) {
                    for &u in &bk.users {
                        assert_eq!(s.of_list.get(&u).copied(), Some(b), "stale user link {u:#x} on backing {b:#x}");
                    }
                }
            }
        }
        forget_list(pin_list);
        for &l in &worker_lists {
            forget_list(l);
        }
        {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            s.aside.remove(&title_alloc);
            s.cur.remove(&title_alloc);
            s.types.remove(&title_alloc);
            for &b in &all_backings {
                s.backings.remove(&b);
            }
        }
    }
}
