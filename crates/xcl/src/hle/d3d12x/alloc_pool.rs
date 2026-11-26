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
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

#[derive(Default)]
struct Backing {
    /// PC lists whose recording is on this allocator
    users: HashSet<usize>,
    /// queue → (fence, value after the last submit of one of them)
    marks: HashMap<usize, (ID3D12Fence, u64)>,
    /// Submits reserved before ExecuteCommandLists whose completion is not marked yet.
    submit_pins: usize,
    /// Probe reference transferred from the title's Release while this backing is busy.
    release_owner: Option<ID3D12CommandAllocator>,
    /// Its allocator namespace is gone; release its retained/raw owner once it becomes free.
    retire_when_free: bool,
}

impl Backing {
    unsafe fn free(&self) -> bool {
        self.users.is_empty() && self.submit_pins == 0 && self.marks.values().all(|(f, v)| f.GetCompletedValue() >= *v)
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
    /// PC list → object lifetime, so a late reservation cannot mark a reused address.
    list_lifetimes: HashMap<usize, u64>,
    next_list_lifetime: u64,
    resubmits: u64,
    /// (PC list, object lifetime) → reserved submits not yet associated with a queue fence.
    submit_pins: HashMap<(usize, u64), usize>,
}

struct LastSubmit {
    queue: usize,
    fence: ID3D12Fence,
    value: u64,
    recording: u64,
}

unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);

pub(super) fn release_with_probe(
    add_ref: impl FnOnce(),
    mut release: impl FnMut() -> u32,
    retain_probe: impl FnOnce() -> bool,
) -> (u32, bool) {
    add_ref();
    let after_caller = release();
    if after_caller == 1 && retain_probe() { (0, true) } else { (release(), false) }
}

/// Transfer the caller's temporary AddRef probe into pool ownership when releasing `raw` would
/// otherwise destroy an allocator that still owns recordings or unretired GPU work.
pub(super) unsafe fn retain_release_probe(raw: *mut c_void) -> bool {
    let mut g = STATE.lock().unwrap();
    let Some(backing) = g.as_mut().and_then(|s| s.backings.get_mut(&(raw as usize))) else { return false };
    if backing.free() || backing.release_owner.is_some() {
        return false;
    }
    backing.release_owner = Some(ID3D12CommandAllocator::from_raw(raw));
    true
}

/// Drop transferred Release probes only after leaving STATE: their hooked Release re-enters here.
pub(super) fn sweep_release_owners() {
    sweep_release_owners_with(super::real::release_retained_allocator, drop);
}

fn sweep_release_owners_with(
    mut release_transferred: impl FnMut(ID3D12CommandAllocator),
    mut release_made: impl FnMut(ID3D12CommandAllocator),
) {
    loop {
        let owner = {
            let mut g = STATE.lock().unwrap();
            let Some(s) = g.as_mut() else { return };
            if let Some(raw) = s.backings.iter().find_map(|(&raw, b)| (b.retire_when_free && unsafe { b.free() }).then_some(raw)) {
                let mut gone = s.backings.remove(&raw).unwrap();
                match gone.release_owner.take() {
                    Some(owner) => Some((owner, true)),
                    None => Some((unsafe { ID3D12CommandAllocator::from_raw(raw as *mut c_void) }, false)),
                }
            } else {
                return;
            }
        };
        if let Some((owner, transferred)) = owner {
            if transferred {
                release_transferred(owner);
            } else {
                release_made(owner);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct AllocatorSubmitStats {
    pub outer_execute_calls: u64,
    pub chunks: u64,
    pub multi_chunk_calls: u64,
    pub forced_wait_boundaries: u64,
    pub lists_marked: u64,
    /// Backing mark-map insertions, including overwrites when lists share a backing.
    pub backing_mark_writes: u64,
    pub signal_attempts: u64,
    pub signal_failures: u64,
    pub state_lock_wait_ns: u64,
    pub state_lock_hold_ns: u64,
}

struct SubmitCounters {
    outer_execute_calls: AtomicU64,
    chunks: AtomicU64,
    multi_chunk_calls: AtomicU64,
    forced_wait_boundaries: AtomicU64,
    lists_marked: AtomicU64,
    backing_mark_writes: AtomicU64,
    signal_attempts: AtomicU64,
    signal_failures: AtomicU64,
    state_lock_wait_ns: AtomicU64,
    state_lock_hold_ns: AtomicU64,
}

macro_rules! submit_counters {
    ($($name:ident),+ $(,)?) => {
        static SUBMIT: SubmitCounters = SubmitCounters { $($name: AtomicU64::new(0)),+ };

        pub fn submit_stats() -> AllocatorSubmitStats {
            AllocatorSubmitStats { $($name: SUBMIT.$name.load(Ordering::Relaxed)),+ }
        }
    };
}

submit_counters!(
    outer_execute_calls, chunks, multi_chunk_calls, forced_wait_boundaries,
    lists_marked, backing_mark_writes, signal_attempts, signal_failures,
    state_lock_wait_ns, state_lock_hold_ns,
);

fn add_duration(counter: &AtomicU64, elapsed: std::time::Duration) {
    counter.fetch_add(elapsed.as_nanos().min(u64::MAX as u128) as u64, Ordering::Relaxed);
}

struct SubmitStateGuard<'a> {
    guard: MutexGuard<'a, Option<State>>,
    held: std::time::Instant,
}

impl Deref for SubmitStateGuard<'_> {
    type Target = Option<State>;
    fn deref(&self) -> &Self::Target { &self.guard }
}

impl DerefMut for SubmitStateGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.guard }
}

impl Drop for SubmitStateGuard<'_> {
    fn drop(&mut self) {
        add_duration(&SUBMIT.state_lock_hold_ns, self.held.elapsed());
    }
}

fn lock_for_submit() -> SubmitStateGuard<'static> {
    let waiting = std::time::Instant::now();
    let guard = STATE.lock().unwrap();
    add_duration(&SUBMIT.state_lock_wait_ns, waiting.elapsed());
    SubmitStateGuard { guard, held: std::time::Instant::now() }
}

pub fn note_execute_call(chunks: usize) {
    let (multi_chunk, boundaries) = execute_topology(chunks);
    SUBMIT.outer_execute_calls.fetch_add(1, Ordering::Relaxed);
    SUBMIT.chunks.fetch_add(chunks as u64, Ordering::Relaxed);
    SUBMIT.multi_chunk_calls.fetch_add(multi_chunk, Ordering::Relaxed);
    SUBMIT.forced_wait_boundaries.fetch_add(boundaries, Ordering::Relaxed);
}

fn execute_topology(chunks: usize) -> (u64, u64) {
    ((chunks > 1) as u64, chunks.saturating_sub(1) as u64)
}

#[derive(Clone, Copy)]
struct SubmitEntry {
    list: usize,
    backing: Option<usize>,
    recording: u64,
    lifetime: u64,
}

/// A fail-closed reservation made before native ExecuteCommandLists. Dropping it before
/// `execution_possible` releases the reservation because no GPU work can refer to it yet.
#[must_use]
pub(super) struct BeginSubmit {
    entries: Vec<SubmitEntry>,
}

/// A reservation armed immediately before native execution. Once armed, execution may begin at
/// any point, so dropping it before `finish` deliberately leaves its allocator and lists pinned.
#[must_use]
pub(super) struct PendingSubmit {
    entries: Vec<SubmitEntry>,
}

fn release_submit_pins(s: &mut State, entries: &[SubmitEntry]) {
    for entry in entries {
        let key = (entry.list, entry.lifetime);
        if let Some(n) = s.submit_pins.get_mut(&key) {
            *n -= 1;
            if *n == 0 {
                s.submit_pins.remove(&key);
            }
        }
        if let Some(backing) = entry.backing.and_then(|b| s.backings.get_mut(&b)) {
            backing.submit_pins = backing.submit_pins.saturating_sub(1);
        }
    }
}

fn record_submit(s: &mut State, entries: &[SubmitEntry], queue: usize, fence: &ID3D12Fence, value: u64) {
    for entry in entries {
        if let Some(backing) = entry.backing {
            s.backings.entry(backing).or_default().marks.insert(queue, (fence.clone(), value));
        }
        if s.list_lifetimes.get(&entry.list) == Some(&entry.lifetime) {
            s.last_submit.insert(
                entry.list,
                LastSubmit { queue, fence: fence.clone(), value, recording: entry.recording },
            );
        }
    }
    release_submit_pins(s, entries);
}

/// Capture and pin the exact backing and recording that a native submit will execute. The caller
/// must keep this token until it knows ExecuteCommandLists cannot have run, or convert it with
/// `execution_possible` immediately before making the native call.
pub(super) fn begin_submit(lists: &[*mut c_void]) -> Option<BeginSubmit> {
    sweep_release_owners();
    if !enabled() {
        return Some(BeginSubmit { entries: Vec::new() });
    }
    let mut entries = Vec::new();
    entries.try_reserve(lists.len()).ok()?;
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    s.list_lifetimes.try_reserve(lists.len()).ok()?;
    s.submit_pins.try_reserve(lists.len()).ok()?;
    for &list in lists {
        let list = list as usize;
        if entries.iter().any(|entry: &SubmitEntry| entry.list == list) {
            continue;
        }
        let lifetime = match s.list_lifetimes.get(&list) {
            Some(&lifetime) => lifetime,
            None => {
                s.next_list_lifetime = s.next_list_lifetime.wrapping_add(1).max(1);
                s.list_lifetimes.insert(list, s.next_list_lifetime);
                s.next_list_lifetime
            }
        };
        entries.push(SubmitEntry {
            list,
            backing: s.of_list.get(&list).copied(),
            recording: s.recordings.get(&list).copied().unwrap_or(0),
            lifetime,
        });
    }
    for entry in &entries {
        *s.submit_pins.entry((entry.list, entry.lifetime)).or_default() += 1;
        if let Some(backing) = entry.backing {
            s.backings.get_mut(&backing).unwrap().submit_pins += 1;
        }
    }
    Some(BeginSubmit { entries })
}

impl BeginSubmit {
    /// Arm immediately before calling native ExecuteCommandLists. From this point onward, any
    /// unwind, drop, or completion-mark failure must retain the reservation permanently.
    pub(super) fn execution_possible(mut self) -> PendingSubmit {
        PendingSubmit { entries: std::mem::take(&mut self.entries) }
    }
}

impl Drop for BeginSubmit {
    fn drop(&mut self) {
        if !self.entries.is_empty() {
            let mut g = STATE.lock().unwrap();
            if let Some(s) = g.as_mut() {
                release_submit_pins(s, &self.entries);
            }
            drop(g);
            sweep_release_owners();
        }
    }
}

impl PendingSubmit {
    /// Associate the captured submit with the queue's existing retirement fence. `false` leaves
    /// all reservations permanently pinned: execution may have happened but cannot be retired.
    pub(super) unsafe fn finish(mut self, queue: *mut c_void) -> bool {
        if self.entries.is_empty() {
            return true;
        }
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return false };
        let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return false };
        let key = queue as usize;
        if !s.fences.contains_key(&key) {
            let mut dev: Option<ID3D12Device> = None;
            let Some(fence) = q.GetDevice(&mut dev).ok().and(dev).and_then(|d| d.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok()) else {
                return false;
            };
            s.fences.insert(key, (fence, 0));
        }
        let (fence, value) = {
            let (fence, value) = s.fences.get_mut(&key).unwrap();
            if q.Signal(&*fence, *value + 1).is_err() {
                return false;
            }
            *value += 1;
            (fence.clone(), *value)
        };
        record_submit(s, &self.entries, key, &fence, value);
        self.entries.clear();
        drop(g);
        sweep_release_owners();
        true
    }

    #[cfg(test)]
    fn finish_with_mark(mut self, queue: usize, fence: &ID3D12Fence, value: u64) {
        let mut g = STATE.lock().unwrap();
        record_submit(g.as_mut().unwrap(), &self.entries, queue, fence, value);
        self.entries.clear();
    }
}

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
    if let Some(lifetime) = s.list_lifetimes.remove(&list) {
        s.submit_pins.remove(&(list, lifetime));
    }
    if let Some(backing) = s.of_list.remove(&list) {
        if let Some(b) = s.backings.get_mut(&backing) {
            b.users.remove(&list);
        }
    }
    drop(g);
    sweep_release_owners();
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
    let mut g = lock_for_submit();
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
    let Some(s) = g.as_ref() else { return false };
    s.list_lifetimes
        .get(&list)
        .and_then(|&lifetime| s.submit_pins.get(&(list, lifetime)))
        .copied()
        .unwrap_or(0)
        != 0
        || s.last_submit.get(&list).is_some_and(|l| unsafe { l.fence.GetCompletedValue() } < l.value)
}

/// `lists` were just submitted to `queue`.
pub unsafe fn executed(queue: *mut c_void, lists: &[*mut c_void]) {
    if !enabled() || queue.is_null() {
        return;
    }
    // (every submit is marked, also of lists this module does not track – the PC lists behind
    // console lists – so note_resubmits sees them all)
    let mut g = lock_for_submit();
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
        SUBMIT.signal_attempts.fetch_add(1, Ordering::Relaxed);
        if q.Signal(&*f, *v + 1).is_err() {
            SUBMIT.signal_failures.fetch_add(1, Ordering::Relaxed);
            return;
        }
        *v += 1;
        (f.clone(), *v)
    };
    let mut backing_mark_writes = 0u64;
    for &l in lists {
        if let Some(&b) = s.of_list.get(&(l as usize)) {
            backing_mark_writes += 1;
            s.backings.entry(b).or_default().marks.insert(key, (fence.clone(), value));
        }
        let recording = s.recordings.get(&(l as usize)).copied().unwrap_or(0);
        s.last_submit.insert(l as usize, LastSubmit { queue: key, fence: fence.clone(), value, recording });
    }
    SUBMIT.lists_marked.fetch_add(lists.len() as u64, Ordering::Relaxed);
    SUBMIT.backing_mark_writes.fetch_add(backing_mark_writes, Ordering::Relaxed);
}

/// Whether the GPU may still run a recording made on the title's allocator `alloc`.
pub(super) unsafe fn allocator_in_flight(alloc: usize) -> bool {
    let g = STATE.lock().unwrap();
    let Some(s) = g.as_ref() else { return false };
    let mine = s.cur.get(&alloc).into_iter().chain(s.aside.get(&alloc).into_iter().flatten()).chain([&alloc]);
    let busy = mine.filter_map(|b| s.backings.get(b)).any(|b| {
        b.submit_pins != 0 || b.marks.values().any(|(f, v)| f.GetCompletedValue() < *v)
    });
    drop(g);
    sweep_release_owners();
    busy
}

/// The title's allocator `alloc` reached its logical final Release. Detach its namespace now;
/// backings still used by recordings/submits retain their native owner until a later sweep.
unsafe fn forget_in(s: &mut State, alloc: usize) -> (Vec<usize>, Vec<ID3D12CommandAllocator>) {
    s.types.remove(&alloc);
    let mut mine: Vec<usize> = s.aside.remove(&alloc).unwrap_or_default();
    mine.extend(s.cur.remove(&alloc));
    mine.push(alloc);
    mine.sort_unstable();
    mine.dedup();
    let mut made = Vec::new();
    let mut owners = Vec::new();
    for b in mine {
        let free = s.backings.get(&b).is_none_or(|x| x.free());
        if !free {
            if let Some(backing) = s.backings.get_mut(&b) {
                backing.retire_when_free = true;
            }
            continue;
        }
        let owner = s.backings.remove(&b).and_then(|mut gone| {
            for l in gone.users.drain() {
                s.of_list.remove(&l);
            }
            gone.release_owner.take()
        });
        if let Some(owner) = owner {
            owners.push(owner);
        } else if b != alloc {
            made.push(b);
        }
    }
    (made, owners)
}

pub unsafe fn forget(alloc: usize) {
    let (made, owners) = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        forget_in(s, alloc)
    };
    for owner in owners {
        super::real::release_retained_allocator(owner);
    }
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
    fn release_probe_transfers_only_the_logical_final_reference() {
        use std::cell::Cell;
        let refs = Cell::new(1u32);
        let retained = Cell::new(0u32);
        let (remaining, transferred) = release_with_probe(
            || refs.set(refs.get() + 1),
            || {
                refs.set(refs.get() - 1);
                refs.get()
            },
            || {
                retained.set(retained.get() + 1);
                true
            },
        );
        assert_eq!((remaining, transferred, refs.get(), retained.get()), (0, true, 1, 1));
        refs.set(refs.get() - 1); // the out-of-lock sweep releases the transferred probe
        assert_eq!(refs.get(), 0);

        refs.set(2);
        retained.set(0);
        let (remaining, transferred) = release_with_probe(
            || refs.set(refs.get() + 1),
            || {
                refs.set(refs.get() - 1);
                refs.get()
            },
            || {
                retained.set(retained.get() + 1);
                true
            },
        );
        assert_eq!((remaining, transferred, refs.get(), retained.get()), (1, false, 1, 0));
    }

    #[test]
    fn released_namespace_keeps_busy_made_backing_until_retirement() {
        let (alloc, made) = (0x7ffe_6100usize, 0x7ffe_6200usize);
        {
            let mut g = STATE.lock().unwrap();
            let s = g.get_or_insert_with(State::default);
            s.cur.insert(alloc, made);
            s.backings.insert(made, Backing { submit_pins: 1, ..Default::default() });
        }
        assert!(unsafe { allocator_in_flight(alloc) });
        unsafe { forget(alloc) };
        {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            assert!(s.backings[&made].retire_when_free);
            assert_eq!(s.backings[&made].submit_pins, 1);
            assert!(!s.cur.contains_key(&alloc));
            s.backings.remove(&made); // synthetic raw has no COM reference to sweep
        }
    }

    #[test]
    fn absent_free_made_backing_is_scheduled_for_release_once() {
        let (alloc, made) = (0x7ffe_6300usize, 0x7ffe_6400usize);
        let mut state = State::default();
        state.cur.insert(alloc, made);
        state.aside.insert(alloc, vec![made, alloc, made]);
        let (release, owners) = unsafe { forget_in(&mut state, alloc) };
        assert_eq!(release, vec![made]);
        assert!(owners.is_empty());
        assert!(!state.cur.contains_key(&alloc) && !state.aside.contains_key(&alloc));
    }

    #[test]
    fn transferred_owner_is_released_once_outside_state_after_logical_forget() {
        use std::cell::Cell;

        #[repr(C)]
        struct FakeAllocator {
            vtable: *const usize,
            refs: Cell<u32>,
            releases: Cell<u32>,
            release_outside_state: Cell<bool>,
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let object = &*this.cast::<FakeAllocator>();
            for _ in 0..10_000 {
                if STATE.try_lock().is_ok() {
                    object.release_outside_state.set(true);
                    break;
                }
                std::thread::yield_now();
            }
            object.releases.set(object.releases.get() + 1);
            object.refs.set(object.refs.get() - 1);
            object.refs.get()
        }
        extern "system" fn unexpected_call() { std::process::abort(); }

        let vtable = [unexpected_call as usize, unexpected_call as usize, release as usize];
        let mut object = FakeAllocator {
            vtable: vtable.as_ptr(), refs: Cell::new(1), releases: Cell::new(0), release_outside_state: Cell::new(false),
        };
        let raw = std::ptr::from_mut(&mut object).cast::<c_void>();
        let owner = unsafe { ID3D12CommandAllocator::from_raw(raw) };
        {
            let mut g = STATE.lock().unwrap();
            g.get_or_insert_with(State::default).backings.insert(
                raw as usize,
                Backing { submit_pins: 1, release_owner: Some(owner), ..Default::default() },
            );
        }
        unsafe { forget(raw as usize) };
        {
            let mut g = STATE.lock().unwrap();
            let backing = g.as_mut().unwrap().backings.get_mut(&(raw as usize)).unwrap();
            assert!(backing.retire_when_free);
            backing.submit_pins = 0;
        }
        sweep_release_owners_with(drop, drop);
        assert_eq!((object.refs.get(), object.releases.get()), (0, 1));
        assert!(object.release_outside_state.get());
        assert!(!STATE.lock().unwrap().as_ref().unwrap().backings.contains_key(&(raw as usize)));
    }

    #[test]
    fn execute_topology_counts_only_split_boundaries() {
        assert_eq!(execute_topology(1), (0, 0));
        assert_eq!(execute_topology(4), (1, 3));
    }

    #[test]
    fn submit_reservation_keeps_pre_reset_attribution_across_queues() {
        use std::cell::Cell;
        use std::mem::{offset_of, size_of};

        #[repr(C)]
        struct Fence {
            vtable: *const usize,
            value: Cell<u64>,
            refs: Cell<u32>,
        }
        unsafe extern "system" fn completed(this: *mut c_void) -> u64 {
            (*this.cast::<Fence>()).value.get()
        }
        unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
            let fence = &*this.cast::<Fence>();
            fence.refs.set(fence.refs.get() + 1);
            fence.refs.get()
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let fence = &*this.cast::<Fence>();
            fence.refs.set(fence.refs.get() - 1);
            fence.refs.get()
        }
        extern "system" fn unexpected_call() { std::process::abort(); }

        const SLOT: usize = size_of::<usize>();
        let mut vtable = [unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT];
        vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = completed as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.AddRef) / SLOT] = add_ref as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = release as usize;
        let mut fence_object = Fence { vtable: vtable.as_ptr(), value: Cell::new(0), refs: Cell::new(1) };
        let fence = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut fence_object).cast()) };

        let (alloc, next, list) = (0x7ffe_5100usize, 0x7ffe_5200usize, 0x7ffe_5300usize);
        created_on(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        let first = begin_submit(&[list as *mut c_void, list as *mut c_void]).unwrap().execution_possible();
        let cross_queue = begin_submit(&[list as *mut c_void]).unwrap().execution_possible();
        assert!(in_flight(list));
        let reset = std::thread::spawn(move || {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            s.aside.insert(alloc, vec![next]);
            s.backings.insert(next, Backing::default());
            unsafe {
                title_reset_in(s, alloc as *mut c_void, |candidate| {
                    assert_eq!(candidate as usize, next);
                    0
                })
            }
        })
        .join()
        .unwrap();
        assert_eq!(reset, Some(0), "the reserved backing is put aside instead of Reset");
        assert_eq!(reset_list(list, alloc, -1, |backing| {
            assert_eq!(backing, next);
            0
        }), 0);
        let second = begin_submit(&[list as *mut c_void]).unwrap().execution_possible();

        first.finish_with_mark(0x71, &fence, 3);
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!(s.backings[&alloc].marks.get(&0x71).unwrap().1, 3);
            assert!(!s.backings[&next].marks.contains_key(&0x71));
            assert_eq!(s.submit_pins.get(&(list, s.list_lifetimes[&list])), Some(&2));
            assert_eq!(s.last_submit[&list].recording, 1);
        }
        cross_queue.finish_with_mark(0x72, &fence, 4);
        second.finish_with_mark(0x73, &fence, 5);
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!(s.backings[&alloc].marks.get(&0x71).unwrap().1, 3);
            assert_eq!(s.backings[&alloc].marks.get(&0x72).unwrap().1, 4);
            assert_eq!(s.backings[&next].marks.get(&0x73).unwrap().1, 5);
            assert!(!s.submit_pins.contains_key(&(list, s.list_lifetimes[&list])));
            assert_eq!(s.last_submit[&list].recording, 2);
        }
        let released = begin_submit(&[list as *mut c_void]).unwrap().execution_possible();
        forget_list(list);
        created_on(list, next, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        let reused_address = begin_submit(&[list as *mut c_void]).unwrap();
        released.finish_with_mark(0x74, &fence, 6);
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!(s.backings[&next].marks.get(&0x74).unwrap().1, 6);
            assert!(!s.last_submit.contains_key(&list), "a late mark cannot resurrect a released list address");
            assert_eq!(s.submit_pins.get(&(list, s.list_lifetimes[&list])), Some(&1));
        }
        drop(reused_address);
        forget_list(list);
        unsafe { forget(alloc) };
        drop(fence);
        assert_eq!(fence_object.refs.get(), 0);
    }

    #[test]
    fn submit_reservation_drop_and_failure_are_fail_closed() {
        let (alloc, list) = (0x7ffe_5400usize, 0x7ffe_5500usize);
        created_on(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);

        let before_execute = begin_submit(&[list as *mut c_void, list as *mut c_void]).unwrap();
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!(s.submit_pins.get(&(list, s.list_lifetimes[&list])), Some(&1), "duplicates reserve once");
            assert_eq!(s.backings[&alloc].submit_pins, 1);
        }
        drop(before_execute);
        assert!(!in_flight(list), "a reservation abandoned before Execute is released");

        let armed_for_execute = begin_submit(&[list as *mut c_void]).unwrap().execution_possible();
        drop(armed_for_execute);
        assert!(in_flight(list), "an armed execution stays pinned without a completion mark");
        let failed_finish = begin_submit(&[list as *mut c_void]).unwrap().execution_possible();
        assert!(!unsafe { failed_finish.finish(std::ptr::null_mut()) });
        {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            assert_eq!(s.submit_pins.get(&(list, s.list_lifetimes[&list])), Some(&2));
            assert_eq!(s.backings[&alloc].submit_pins, 2);
        }

        // Final list cleanup removes its recording hold but cannot make the backing reusable
        // after execution became possible without a completion mark.
        forget_list(list);
        {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            assert!(!s.backings[&alloc].users.contains(&list));
            assert!(!s.submit_pins.keys().any(|&(p, _)| p == list), "a released address is not pinned against a later object");
            assert!(!unsafe { s.backings[&alloc].free() });
            s.backings.get_mut(&alloc).unwrap().submit_pins = 0;
        }
        unsafe { forget(alloc) };
    }

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
}
