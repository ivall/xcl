//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for where it is used, a `Dispatch` at its submit (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock, Weak};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::ID3D12PipelineState;

use super::build_queue::{BuildQueue, JobHandle, Priority};

const QUEUE_IDLE: u8 = 0;
const QUEUE_ACTIVE: u8 = 1;
const QUEUE_DONE: u8 = 2;

/// Process-lifetime ranked-preload outcomes; `started` means dequeued in the Preload band.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct RankedPreloadStats {
    pub submissions: u64,
    pub started: u64,
    pub promoted: u64,
}

struct Pending {
    build: Arc<Build>,
    /// the driver pipeline binds get till then (`shared`); 0: the stand-in is one itself
    driver: AtomicUsize,
    compute: bool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// a bind waited for it longer than `wait_budget`: binds go on without it until it is built
    late: std::sync::atomic::AtomicBool,
    /// its dispatches may be left out while it is built (`image_only`): binds do not wait for it
    spared: std::sync::atomic::AtomicBool,
}

/// A pipeline build independent of the title-visible stand-ins that requested it.
struct Build {
    compute_key: Option<Arc<super::real::compute_build_key::ComputeBuildKey>>,
    /// Immutable inputs for rebuilding this pipeline after its first job has been consumed.
    #[allow(dead_code)] // retained for the later opt-in eviction/rebuild stage
    recipe: Option<Recipe>,
    /// the build, until a worker takes it
    job: Mutex<Option<Job>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// Whether a title-visible stand-in owns this build, and its exact queued-count state.
    tracked: AtomicBool,
    queue_state: AtomicU8,
    /// the real pipeline state (one owned raw pointer shared by all `Pending`s), 0 until built
    /// or when building failed
    real: AtomicUsize,
    /// Last frame in which any owner resolved this build, for `XCL_ACTIVITY` retention ages.
    last_use_frame: AtomicU64,
    /// Frame in which `real` became retained; age never begins before the build completed.
    completed_frame: AtomicU64,
    /// This build already has one use record in the bounded eviction-candidate journal.
    /// Touched only while recipe retention is enabled.
    candidate_queued: AtomicBool,
    done: Mutex<bool>,
    cv: Condvar,
    /// every stand-in interested in this build was released before it finished
    orphan: std::sync::atomic::AtomicBool,
    /// Title-visible `Pending`s sharing this build. Changed while holding `done`.
    owners: AtomicUsize,
    /// One of the title's threads waits, or a recorded submit will wait, for this build. Polled
    /// by DXC and driver admission.
    wanted: AtomicBool,
    demand: Mutex<Demand>,
    queued: Mutex<Option<JobHandle>>,
}

const EVICTION_CANDIDATE_CAPACITY: usize = 1024;

/// Bounded lock-free MPMC ring. Bind threads append weak identities, so an unconsumed record does
/// not retain a released pipeline; a future policy can drain a fixed number on a background thread
/// without scanning the title-visible registry.
struct CandidateSlot<T> {
    sequence: AtomicUsize,
    value: std::cell::UnsafeCell<std::mem::MaybeUninit<Weak<T>>>,
}

unsafe impl<T: Send + Sync> Sync for CandidateSlot<T> {}

struct CandidateQueue<T, const N: usize> {
    enqueue: AtomicUsize,
    dequeue: AtomicUsize,
    slots: [CandidateSlot<T>; N],
}

unsafe impl<T: Send + Sync, const N: usize> Sync for CandidateQueue<T, N> {}

impl<T, const N: usize> CandidateQueue<T, N> {
    fn new() -> Self {
        // The sequence protocol needs distinct empty/full generations for each slot. At capacity
        // one the producer can observe its just-published sequence as the next empty generation
        // and overwrite an unread value. Signed sequence differences also require a half-range
        // bound across counter wrap.
        assert!(N > 1 && N.is_power_of_two() && N <= isize::MAX as usize);
        Self {
            enqueue: AtomicUsize::new(0),
            dequeue: AtomicUsize::new(0),
            slots: std::array::from_fn(|sequence| CandidateSlot {
                sequence: AtomicUsize::new(sequence),
                value: std::cell::UnsafeCell::new(std::mem::MaybeUninit::uninit()),
            }),
        }
    }

    fn push(&self, value: Weak<T>) -> Result<(), Weak<T>> {
        let mut position = self.enqueue.load(Ordering::Relaxed);
        loop {
            let slot = &self.slots[position & (N - 1)];
            let sequence = slot.sequence.load(Ordering::Acquire);
            let difference = sequence.wrapping_sub(position) as isize;
            if difference == 0 {
                match self.enqueue.compare_exchange_weak(
                    position, position.wrapping_add(1), Ordering::Relaxed, Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        unsafe { (*slot.value.get()).write(value) };
                        slot.sequence.store(position.wrapping_add(1), Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => position = actual,
                }
            } else if difference < 0 {
                return Err(value);
            } else {
                position = self.enqueue.load(Ordering::Relaxed);
            }
        }
    }

    fn pop(&self) -> Option<Weak<T>> {
        let mut position = self.dequeue.load(Ordering::Relaxed);
        loop {
            let slot = &self.slots[position & (N - 1)];
            let sequence = slot.sequence.load(Ordering::Acquire);
            let difference = sequence.wrapping_sub(position.wrapping_add(1)) as isize;
            if difference == 0 {
                match self.dequeue.compare_exchange_weak(
                    position, position.wrapping_add(1), Ordering::Relaxed, Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        let value = unsafe { (*slot.value.get()).assume_init_read() };
                        slot.sequence.store(position.wrapping_add(N), Ordering::Release);
                        return Some(value);
                    }
                    Err(actual) => position = actual,
                }
            } else if difference < 0 {
                return None;
            } else {
                position = self.dequeue.load(Ordering::Relaxed);
            }
        }
    }
}

impl<T, const N: usize> Drop for CandidateQueue<T, N> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

type EvictionCandidateQueue = CandidateQueue<Build, EVICTION_CANDIDATE_CAPACITY>;

fn eviction_candidates() -> &'static EvictionCandidateQueue {
    static CANDIDATES: OnceLock<EvictionCandidateQueue> = OnceLock::new();
    CANDIDATES.get_or_init(EvictionCandidateQueue::new)
}

/// Submit demand and the preload worker building this pipeline.
#[derive(Default)]
struct Demand {
    /// a call put off to the submit needs it (`need_by_submit`)
    by_submit: bool,
    /// thread id of the preload worker (lowest priority) building it, 0 = none
    preload_thread: u32,
}

thread_local! {
    /// what this worker is building
    static BUILDING: std::cell::RefCell<Option<Arc<Build>>> = const { std::cell::RefCell::new(None) };
    /// this is one of the workers reserved for low-priority preload builds
    static PRELOAD_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// this thread serves the normal (`BoundSoon` through `Foreground`) worker band
    static NORMAL_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// This thread is a worker and one of the title's threads waits for what it is building: the
/// build must not stand back for the title's sake (`real::big_compile_turn`).
pub(super) fn build_is_wanted() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|build| build.wanted.load(Ordering::Acquire)))
}

/// Whether one of the title's threads waits for what this thread is doing, to be asked later and
/// from another thread: a worker's build as `build_is_wanted`; a thread that is not XCL's is the
/// title's own.
pub fn wanted_here() -> crate::gpu::recomp::dxc::Wanted {
    if let Some(build) = BUILDING.with(|b| b.borrow().clone()) {
        return Box::new(move || build.wanted.load(Ordering::Acquire));
    }
    let own = crate::state::boot().is_some() && !std::thread::current().name().is_some_and(|n| n.starts_with("xcl-"));
    Box::new(move || own)
}

struct RunningBuild<'a> {
    build: &'a Build,
    preload: bool,
}

impl Drop for RunningBuild<'_> {
    fn drop(&mut self) {
        if self.preload {
            // (under the lock: `want` does not raise this thread after it)
            let mut demand = self.build.demand.lock().unwrap();
            demand.preload_thread = 0;
            if self.build.wanted.load(Ordering::Acquire) {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
            }
        }
        BUILDING.with(|b| *b.borrow_mut() = None);
    }
}

struct RestoreBuilding(Option<Arc<Build>>);

impl Drop for RestoreBuilding {
    fn drop(&mut self) {
        BUILDING.with(|b| *b.borrow_mut() = self.0.take());
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&str>() {
        message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else {
        "unknown panic"
    }
}

/// Run a pipeline job without letting one bad shader strand every waiter on its shared build.
fn run_job(job: Job) -> usize {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)) {
        Ok(real) => real,
        Err(payload) => {
            crate::xlog_error!("D3D12.X: panic while building pipeline: {}", panic_message(payload.as_ref()));
            0
        }
    }
}

/// A worker's pipeline build (`preload`: by one of the lowest-priority workers).
fn run_build(build: &Arc<Build>, job: Job) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_NORMAL};
    let preload = PRELOAD_WORKER.get();
    BUILDING.with(|b| *b.borrow_mut() = Some(build.clone()));
    if preload {
        let mut d = build.demand.lock().unwrap();
        d.preload_thread = unsafe { GetCurrentThreadId() };
        if build.wanted.load(Ordering::Acquire) {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL) };
        }
    }
    let real = {
        let _running = RunningBuild { build, preload };
        run_job(job)
    };
    complete(build, real);
}

impl Build {
    fn new(job: Job, sent: bool) -> Self {
        Self::new_keyed(job, sent, None, None)
    }

    fn new_keyed(
        job: Job,
        sent: bool,
        compute_key: Option<Arc<super::real::compute_build_key::ComputeBuildKey>>,
        recipe: Option<Recipe>,
    ) -> Self {
        Self {
            compute_key,
            recipe,
            job: Mutex::new(Some(job)),
            sent: std::sync::atomic::AtomicBool::new(sent),
            tracked: AtomicBool::new(false),
            queue_state: AtomicU8::new(QUEUE_IDLE),
            real: AtomicUsize::new(0),
            last_use_frame: AtomicU64::new(super::present::presents()),
            completed_frame: AtomicU64::new(0),
            candidate_queued: AtomicBool::new(false),
            done: Mutex::new(false),
            cv: Condvar::new(),
            orphan: Default::default(),
            owners: AtomicUsize::new(1),
            wanted: AtomicBool::new(false),
            demand: Default::default(),
            queued: Mutex::new(None),
        }
    }

    fn note_queued(&self) {
        if self.tracked.load(Ordering::Acquire) {
            // Publish the count before ACTIVE: completion may run as soon as the queue accepts
            // the job and must never subtract an entry which has not been added yet.
            QUEUED_BUILDS.fetch_add(1, Ordering::Relaxed);
            if !mark_queued(&self.queue_state) {
                QUEUED_BUILDS.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    fn finish_queued(&self) {
        if mark_queue_done(&self.queue_state) {
            QUEUED_BUILDS.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn promote(&self, priority: Priority) {
        if let Some(job) = self.queued.lock().unwrap().as_ref() {
            let _ = job.promote(priority);
        }
    }

    fn cancel_queued(&self) {
        if let Some(job) = self.queued.lock().unwrap().take() {
            let _ = job.cancel();
        }
    }

    /// One of the title's threads waits or a recorded submit will wait for it: the worker
    /// building it goes on now, at a normal priority.
    fn want(&self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_PRIORITY_NORMAL, THREAD_SET_LIMITED_INFORMATION};
        let newly_wanted = !self.wanted.swap(true, Ordering::AcqRel);
        let _ = start_normal_workers();
        self.promote(Priority::Foreground);
        if newly_wanted {
            let d = self.demand.lock().unwrap();
            if d.preload_thread == 0 {
                return;
            }
            unsafe {
                let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, d.preload_thread);
                if !h.is_null() {
                    SetThreadPriority(h, THREAD_PRIORITY_NORMAL);
                    CloseHandle(h);
                }
            }
        }
    }
}

fn mark_queued(state: &AtomicU8) -> bool {
    state.compare_exchange(QUEUE_IDLE, QUEUE_ACTIVE, Ordering::AcqRel, Ordering::Acquire).is_ok()
}

fn mark_queue_done(state: &AtomicU8) -> bool {
    state.swap(QUEUE_DONE, Ordering::AcqRel) == QUEUE_ACTIVE
}

impl Pending {
    fn note_use(&self) {
        if super::memory_report::enabled() {
            self.build.last_use_frame.fetch_max(super::present::presents(), Ordering::Relaxed);
        }
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
        }
    }
}

const PENDING_SHARDS: usize = 64;

/// Title-visible stand-in address → its build. Bind resolution is called by every recording
/// thread, so unrelated pipelines must not queue on one process-wide map lock.
struct PendingRegistry<V> {
    shards: [Mutex<Option<HashMap<usize, V>>>; PENDING_SHARDS],
}

impl<V> PendingRegistry<V> {
    const fn new() -> Self {
        Self { shards: [const { Mutex::new(None) }; PENDING_SHARDS] }
    }

    fn insert(&self, key: usize, value: V) -> Option<V> {
        self.shards[pending_shard_index(key)].lock().unwrap().get_or_insert_with(HashMap::new).insert(key, value)
    }

    fn remove(&self, key: usize) -> Option<V> {
        self.shards[pending_shard_index(key)].lock().unwrap().as_mut().and_then(|map| map.remove(&key))
    }
}

impl<V: Clone> PendingRegistry<V> {
    fn get(&self, key: usize) -> Option<V> {
        self.shards[pending_shard_index(key)].lock().unwrap().as_ref()?.get(&key).cloned()
    }
}

#[inline]
fn pending_shard_index(key: usize) -> usize {
    let mut mixed = (key >> 4).wrapping_mul(0x9E37_79B1);
    mixed ^= mixed >> 16;
    mixed & (PENDING_SHARDS - 1)
}

static PENDING: PendingRegistry<Arc<Pending>> = PendingRegistry::new();

/// Equal compute requests share only while their build is in flight. Weak values neither retain
/// completed builds nor turn this into another PSO cache.
static IN_FLIGHT: Mutex<Option<HashMap<Arc<super::real::compute_build_key::ComputeBuildKey>, Weak<Build>>>> = Mutex::new(None);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ComputeCoalescingStats {
    pub keyed_submissions: u64,
    pub coalescing_hits: u64,
    pub unique_keyed_builds: u64,
    pub current_in_flight: usize,
    pub peak_in_flight: usize,
}

struct CoalescingCounters {
    coalescing_hits: AtomicU64,
    unique_keyed_builds: AtomicU64,
    current_in_flight: AtomicUsize,
    peak_in_flight: AtomicUsize,
}

impl CoalescingCounters {
    const fn new() -> Self {
        Self {
            coalescing_hits: AtomicU64::new(0),
            unique_keyed_builds: AtomicU64::new(0),
            current_in_flight: AtomicUsize::new(0),
            peak_in_flight: AtomicUsize::new(0),
        }
    }

    fn hit(&self) {
        self.coalescing_hits.fetch_add(1, Ordering::Relaxed);
    }

    fn unique(&self) {
        self.unique_keyed_builds.fetch_add(1, Ordering::Relaxed);
        let current = self.current_in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak_in_flight.fetch_max(current, Ordering::Relaxed);
    }

    fn finished(&self, count: usize) {
        if count != 0 {
            self.current_in_flight.fetch_sub(count, Ordering::Relaxed);
        }
    }

    fn snapshot(&self) -> ComputeCoalescingStats {
        let coalescing_hits = self.coalescing_hits.load(Ordering::Relaxed);
        let unique_keyed_builds = self.unique_keyed_builds.load(Ordering::Relaxed);
        ComputeCoalescingStats {
            keyed_submissions: coalescing_hits.saturating_add(unique_keyed_builds),
            coalescing_hits,
            unique_keyed_builds,
            current_in_flight: self.current_in_flight.load(Ordering::Relaxed),
            peak_in_flight: self.peak_in_flight.load(Ordering::Relaxed),
        }
    }
}

static COALESCING: CoalescingCounters = CoalescingCounters::new();

pub fn compute_coalescing_stats() -> ComputeCoalescingStats {
    COALESCING.snapshot()
}

/// Unique completed native pipelines retained behind title-visible stand-ins, by frames since
/// their last use. Measurement only: command lists do not retain bound D3D12 objects, so these
/// ages are not evidence that a pipeline is safe to release.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetentionStats {
    pub retained_reals: usize,
    pub unused_60_frames: usize,
    pub unused_600_frames: usize,
    pub oldest_unused_frames: u64,
}

fn retention_stats_from(
    now: u64,
    builds: impl IntoIterator<Item = (usize, usize, u64, u64)>,
) -> RetentionStats {
    let mut seen = std::collections::HashSet::new();
    let mut stats = RetentionStats::default();
    for (identity, real, last_use, completed) in builds {
        if real == 0 || !seen.insert(identity) {
            continue;
        }
        let age = now.saturating_sub(last_use.max(completed));
        stats.retained_reals += 1;
        stats.unused_60_frames += (age >= 60) as usize;
        stats.unused_600_frames += (age >= 600) as usize;
        stats.oldest_unused_frames = stats.oldest_unused_frames.max(age);
    }
    stats
}

/// Snapshot for the `XCL_ACTIVITY` thread. The registry locks are taken only by that observer;
/// bind-time tracking is a present counter load plus one relaxed atomic maximum.
pub fn retention_stats() -> RetentionStats {
    if !super::memory_report::enabled() {
        return RetentionStats::default();
    }
    let mut builds = Vec::new();
    for shard in &PENDING.shards {
        let entries = shard.lock().unwrap();
        builds.extend(entries.iter().flat_map(|map| map.values()).map(|pending| {
            let build = &pending.build;
            (
                Arc::as_ptr(build) as usize,
                build.real.load(Ordering::Acquire),
                build.last_use_frame.load(Ordering::Relaxed),
                build.completed_frame.load(Ordering::Relaxed),
            )
        }));
    }
    retention_stats_from(super::present::presents(), builds)
}

/// One point-in-time, pressure-ready resident selected off the bind thread. A future policy must
/// still recheck age and call `try_evict_current`, which serializes every ownership transition.
#[allow(dead_code)]
struct EvictionCandidate {
    build: Arc<Build>,
    last_use_frame: u64,
    completed_frame: u64,
    clear_queued_on_drop: bool,
}

impl Drop for EvictionCandidate {
    fn drop(&mut self) {
        if self.clear_queued_on_drop {
            self.build.candidate_queued.store(false, Ordering::Release);
        }
    }
}

impl EvictionCandidate {
    /// Return a deferred candidate to the bounded weak journal without opening a duplicate-record
    /// window. If a crossing producer filled the freed slot, dropping this owner clears the flag
    /// so a later bind can retry instead of leaving the build permanently marked as queued.
    fn requeue_on<const N: usize>(mut self, candidates: &CandidateQueue<Build, N>) -> bool {
        if candidates.push(Arc::downgrade(&self.build)).is_err() {
            return false;
        }
        self.clear_queued_on_drop = false;
        true
    }
}

/// Recipe-enabled binds append at most one weak identity per build. This path is reached only after
/// the existing `XCL_PSO_REBUILD_RECIPES` gate selected generation-aware resolution; ordinary
/// binds pay no candidate check, queue operation, registry scan, or allocation.
fn note_eviction_candidate_on<const N: usize>(build: &Arc<Build>, frame: u64, candidates: &CandidateQueue<Build, N>) {
    build.last_use_frame.fetch_max(frame, Ordering::Relaxed);
    if build.candidate_queued.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return;
    }
    if candidates.push(Arc::downgrade(build)).is_err() {
        build.candidate_queued.store(false, Ordering::Release);
    }
}

fn note_eviction_candidate(build: &Arc<Build>) {
    note_eviction_candidate_on(build, super::present::presents(), eviction_candidates());
}

fn resident_candidate(build: &Arc<Build>) -> Option<EvictionCandidate> {
    let done = build.done.lock().unwrap();
    if !*done || build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
        return None;
    }
    let recipe = build.recipe.as_ref()?;
    let rebuild = recipe.0.rebuild.lock().unwrap();
    if rebuild.active || rebuild.waiters != 0 {
        return None;
    }
    let real = build.real.load(Ordering::Acquire);
    if real == 0 {
        return None;
    }
    let generation = recipe.0.generation.lock().unwrap();
    if generation.residency != Residency::Resident
        || generation.resident.as_ref().is_none_or(|resident| resident.raw != real)
    {
        return None;
    }
    let last_use_frame = build.last_use_frame.load(Ordering::Relaxed);
    let completed_frame = build.completed_frame.load(Ordering::Relaxed);
    drop(generation);
    drop(rebuild);
    drop(done);
    Some(EvictionCandidate {
        build: build.clone(), last_use_frame, completed_frame, clear_queued_on_drop: true,
    })
}

/// Drain at most `limit` journal records on a background policy turn. The weak allocation identity
/// prevents a released stand-in address from turning an old record into a new pipeline.
#[allow(dead_code)]
fn drain_eviction_candidates_on<const N: usize>(
    candidates: &CandidateQueue<Build, N>, limit: usize,
) -> Vec<EvictionCandidate> {
    let mut selected = Vec::new();
    for _ in 0..limit {
        let Some(build) = candidates.pop() else { break };
        let Some(build) = build.upgrade() else { continue };
        match resident_candidate(&build) {
            Some(candidate) => selected.push(candidate),
            None => {
                // A released/orphaned/active/nonresident build is not actionable. Allow a later
                // use after its state changes to journal it again.
                build.candidate_queued.store(false, Ordering::Release);
            }
        }
    }
    selected
}

#[allow(dead_code)]
fn drain_eviction_candidates(limit: usize) -> Vec<EvictionCandidate> {
    drain_eviction_candidates_on(eviction_candidates(), limit)
}

#[derive(Clone, Copy)]
struct EvictionSelectionLimits {
    max_scan: usize,
    max_select: usize,
    min_unused_frames: u64,
    min_resident_frames: u64,
}

/// One bounded, point-in-time policy candidate. Actual eviction must still call
/// `try_evict_current`, which revalidates every ownership transition.
#[allow(dead_code)]
struct SelectedEvictionCandidate {
    candidate: EvictionCandidate,
    unused_frames: u64,
    resident_frames: u64,
}

/// Frame counters practically never wrap. If they do, treating pre-wrap timestamps as recent is
/// conservative: a policy pauses eviction instead of assigning an enormous, false age.
fn conservative_frame_age(now: u64, since: u64) -> u64 {
    now.checked_sub(since).unwrap_or(0)
}

fn partition_eviction_candidates(
    candidates: Vec<EvictionCandidate>, now: u64, limits: EvictionSelectionLimits,
) -> (Vec<SelectedEvictionCandidate>, Vec<EvictionCandidate>) {
    if limits.max_select == 0 {
        return (Vec::new(), candidates);
    }
    let mut selected = Vec::new();
    let mut deferred = Vec::new();
    for candidate in candidates {
        // A bind after the journal record was drained still updates this atomic while the
        // candidate's dedup flag prevents another record. Use that latest point-in-time age,
        // not the older snapshot captured by `resident_candidate`.
        let last_use = candidate.build.last_use_frame.load(Ordering::Relaxed);
        let completed = candidate.build.completed_frame.load(Ordering::Relaxed);
        let unused_frames = conservative_frame_age(now, last_use);
        let resident_frames = conservative_frame_age(now, completed);
        if unused_frames >= limits.min_unused_frames && resident_frames >= limits.min_resident_frames {
            selected.push(SelectedEvictionCandidate { candidate, unused_frames, resident_frames });
        } else {
            deferred.push(candidate);
        }
    }
    selected.sort_unstable_by(|a, b| {
        b.unused_frames
            .cmp(&a.unused_frames)
            .then_with(|| b.resident_frames.cmp(&a.resident_frames))
            .then_with(|| {
                (Arc::as_ptr(&a.candidate.build) as usize).cmp(&(Arc::as_ptr(&b.candidate.build) as usize))
            })
    });
    if selected.len() > limits.max_select {
        deferred.extend(selected.drain(limits.max_select..).map(|selected| selected.candidate));
    }
    (selected, deferred)
}

/// Inspect and rank a bounded amount of journalled work on a future background policy turn. This
/// remains deliberately uncalled: pressure admission and the eviction/rehydration trigger are
/// separate stages.
#[allow(dead_code)]
fn select_eviction_candidates_on<const N: usize>(
    candidates: &CandidateQueue<Build, N>, now: u64, limits: EvictionSelectionLimits,
) -> Vec<SelectedEvictionCandidate> {
    if limits.max_scan == 0 || limits.max_select == 0 {
        return Vec::new();
    }
    let (selected, deferred) =
        partition_eviction_candidates(drain_eviction_candidates_on(candidates, limits.max_scan), now, limits);
    for candidate in deferred {
        let _ = candidate.requeue_on(candidates);
    }
    selected
}

#[allow(dead_code)]
fn select_eviction_candidates(limits: EvictionSelectionLimits) -> Vec<SelectedEvictionCandidate> {
    select_eviction_candidates_on(eviction_candidates(), super::present::presents(), limits)
}

fn forget_in_flight(build: &Arc<Build>) {
    let Some(key) = &build.compute_key else { return };
    let mut registry = IN_FLIGHT.lock().unwrap();
    let Some(builds) = registry.as_mut() else { return };
    if builds.get(key.as_ref()).is_some_and(|weak| Weak::ptr_eq(weak, &Arc::downgrade(build))) {
        builds.remove(key.as_ref());
        COALESCING.finished(1);
    }
}

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(build: &Arc<Build>, real: usize) {
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
        build.completed_frame.store(super::present::presents(), Ordering::Relaxed);
    }
    let (orphan, replaced_generation) = {
        let mut done = build.done.lock().unwrap();
        let orphan = build.orphan.load(Ordering::Acquire);
        let mut replaced_generation = None;
        if !orphan {
            if real != 0 {
                if let Some(recipe) = &build.recipe {
                    replaced_generation = recipe.publish(real).1;
                }
            } else if let Some(recipe) = &build.recipe {
                recipe.mark_failed();
            }
            build.real.store(real, Ordering::Release);
        }
        *done = true;
        build.finish_queued();
        build.cv.notify_all();
        (orphan, replaced_generation)
    };
    // outside `done`: a pipeline Release override can re-enter pipeline bookkeeping
    drop(replaced_generation);
    if orphan {
        release(real);
    }
    forget_in_flight(build);
}

/// Result shared by callers that requested the same overlapping recipe rebuild.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RebuildResult {
    NotReady,
    Failed,
    Abandoned,
    Published { generation: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RebuildRequest {
    Queued,
    Coalesced,
    NotReady,
    Abandoned,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RehydrateRequest {
    Queued,
    Coalesced,
    NotReady,
    Abandoned,
    Resident,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvictResult {
    Evicted { generation: u64 },
    NotReady,
    Active,
    Abandoned,
    Unavailable,
    NotResident,
}

/// Detach both owners of the current native generation. This remains deliberately uncalled until
/// a policy and a nonblocking rehydration path choose the correctness/performance tradeoff after
/// eviction. Exact recording and receipt pins independently keep the detached generation alive.
#[allow(dead_code)]
fn try_evict_current(build: &Build) -> EvictResult {
    let done = build.done.lock().unwrap();
    if !*done {
        return EvictResult::NotReady;
    }
    if build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
        return EvictResult::Abandoned;
    }
    let Some(recipe) = build.recipe.as_ref() else { return EvictResult::Unavailable };
    // Keep the rebuild turn reserved through the raw-owner swap. Every rebuild reservation takes
    // this lock, and publication takes `done`, so neither transition can cross this one.
    let rebuild = recipe.0.rebuild.lock().unwrap();
    if rebuild.active || rebuild.waiters != 0 {
        return EvictResult::Active;
    }
    let raw = build.real.load(Ordering::Acquire);
    if raw == 0 {
        return EvictResult::NotResident;
    }
    let Some(generation) = recipe.detach_current(raw) else { return EvictResult::NotResident };
    let generation_id = generation.id();
    let old_raw = build.real.swap(0, Ordering::AcqRel);
    debug_assert_eq!(old_raw, raw);
    drop(rebuild);
    drop(done);

    // COM Release can re-enter pipeline bookkeeping. Drop both owners outside every state lock;
    // an exact recording/receipt pin is another Arc to `generation` and survives independently.
    drop(generation);
    release(old_raw);
    EvictResult::Evicted { generation: generation_id }
}

/// Build and atomically publish one replacement generation. This remains deliberately uncalled
/// until submitted recording pins are retained to exact GPU retirement receipts: publication is
/// safe, but a policy must not make an old generation reclaimable while the GPU can still use it.
#[allow(dead_code)]
fn rebuild_once(build: &Build) -> Option<RebuildResult> {
    let recipe = build.recipe.as_ref()?;
    {
        let done = build.done.lock().unwrap();
        if !*done {
            return Some(RebuildResult::NotReady);
        }
        if build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
            return Some(RebuildResult::Abandoned);
        }
    }
    match recipe.begin_rebuild() {
        RebuildTurn::Follower(result) => return Some(result),
        RebuildTurn::Leader => {}
    }

    Some(rebuild_reserved(build, recipe))
}

/// Run an attempt which already owns the recipe's rebuild turn.
fn rebuild_reserved(build: &Build, recipe: &Recipe) -> RebuildResult {
    // A queued request can lose its last title-visible owner before a worker starts it. The queued
    // Arc keeps this state alive only so the attempt can retire; it must not revive the pipeline.
    {
        let done = build.done.lock().unwrap();
        if !*done || build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
            drop(done);
            recipe.finish_rebuild(RebuildResult::Abandoned);
            return RebuildResult::Abandoned;
        }
    }

    // Do expensive driver work without Build.done or the recipe state held. Binds keep using the
    // current immutable generation throughout the rebuild.
    let raw = run_job(recipe.initial_job());
    let mut replaced_generation = None;
    let mut replaced_raw = 0;
    let result = if raw == 0 {
        RebuildResult::Failed
    } else {
        let _done = build.done.lock().unwrap();
        if build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
            RebuildResult::Abandoned
        } else {
            let (generation, replaced) = recipe.publish(raw);
            replaced_generation = replaced;
            replaced_raw = build.real.swap(raw, Ordering::AcqRel);
            build.completed_frame.store(super::present::presents(), Ordering::Relaxed);
            RebuildResult::Published { generation: generation.id() }
        }
    };

    // Publish the result before COM Release, which can re-enter pipeline bookkeeping. Never drop
    // either old owner under Build.done or while this attempt is still blocking followers.
    recipe.finish_rebuild(result);
    drop(replaced_generation);
    if matches!(result, RebuildResult::Published { .. }) {
        release(replaced_raw);
    } else if raw != 0 {
        release(raw);
    }
    result
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);
static QUEUED_BUILDS: AtomicUsize = AtomicUsize::new(0);
/// binds waiting right now for a pipeline still being built (`title_held`)
static HELD: AtomicUsize = AtomicUsize::new(0);

/// The PC calls of the title's threads (`pc_call`): the ones going on, and how long the ones
/// that ended took since the last frame was shown.
struct PcCalls {
    /// (id, when it began)
    now: Vec<(u64, std::time::Instant)>,
    /// the frames shown when `spent` began to count (`None` before the first observation)
    frame: Option<u64>,
    /// when the latest frame advance was first observed; calls that started earlier are clamped here
    frame_since: Option<std::time::Instant>,
    /// the time of the PC calls that ended since then
    spent: std::time::Duration,
}

impl PcCalls {
    const fn new() -> Self {
        Self { now: Vec::new(), frame: None, frame_since: None, spent: std::time::Duration::ZERO }
    }

    /// `frame` frames have been shown: the calls before the last of them are not this wait's.
    fn at(&mut self, frame: u64, now: std::time::Instant) {
        match self.frame {
            None => self.frame = Some(frame),
            Some(prev) if frame > prev => {
                self.frame = Some(frame);
                self.frame_since = Some(now);
                self.spent = std::time::Duration::ZERO;
            }
            Some(_) => {}
        }
    }

    fn clamp(&self, since: std::time::Instant) -> std::time::Instant {
        self.frame_since.map_or(since, |f| since.max(f))
    }

    fn begin(&mut self, id: u64, frame: u64, now: std::time::Instant) {
        self.at(frame, now);
        self.now.push((id, now));
    }

    fn end(&mut self, id: u64, frame: u64, now: std::time::Instant) {
        self.at(frame, now);
        if let Some(i) = self.now.iter().position(|c| c.0 == id) {
            let (_, since) = self.now.swap_remove(i);
            self.spent += now.saturating_duration_since(self.clamp(since));
        }
    }

    /// Whether a call going on holds the title: the PC calls since the last frame, it included,
    /// have taken `PC_CALL_HOLDS_AFTER`.
    fn holds(&mut self, frame: u64, now: std::time::Instant) -> bool {
        self.at(frame, now);
        let spent = self.spent;
        self.now.iter().any(|(_, since)| spent + now.saturating_duration_since(self.clamp(*since)) >= PC_CALL_HOLDS_AFTER)
    }
}

static PC_CALLS: Mutex<PcCalls> = Mutex::new(PcCalls::new());
/// From this much time in PC calls since the last frame on, a PC call holds the title: the
/// console's calls return at once.
const PC_CALL_HOLDS_AFTER: std::time::Duration = std::time::Duration::from_millis(250);

/// Whether one of the title's threads is held where a console title never waits: a bind is
/// waiting for a pipeline still being built, or a PC call is going on after such calls have
/// kept the title's threads for long (`pc_call`). `kernelx::Sleep` does not count that time.
pub fn title_held() -> bool {
    HELD.load(Ordering::Acquire) != 0 || held_by_pc_call()
}

fn held_by_pc_call() -> bool {
    PC_CALLS.lock().unwrap().holds(super::present::presents(), std::time::Instant::now())
}

/// A call of the title into D3D12 that XCL or the PC's driver can keep for long where the
/// console's takes no time: a submission (XCL's uploads and tile mappings before the lists, the
/// driver's compiling in them), a texture's placement, a pipeline built in the call. While it
/// lasts the title is held (`title_held`), once the PC calls since the last frame shown, this
/// one included, have taken `PC_CALL_HOLDS_AFTER`: one long call, or many short ones in a row
/// (a retail title placed 16 384 textures in 29 s without a frame, 1 - 50 ms each). A title
/// that shows its frames is not held by its submissions.
pub struct PcCall(u64);

pub fn pc_call() -> PcCall {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    PC_CALLS.lock().unwrap().begin(id, super::present::presents(), std::time::Instant::now());
    PcCall(id)
}

impl Drop for PcCall {
    fn drop(&mut self) {
        PC_CALLS.lock().unwrap().end(self.0, super::present::presents(), std::time::Instant::now());
    }
}

/// Whether this thread is building a pipeline for `build` (a worker, not one of the title's).
pub(super) fn building_here() -> bool {
    BUILDING.with(|b| b.borrow().is_some())
}

/// A bind's wait for a pipeline still being built (`title_held`).
struct Held;

impl Held {
    fn new() -> Held {
        HELD.fetch_add(1, Ordering::AcqRel);
        Held
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        HELD.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

fn recipe_gate_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| value.trim() == "1")
}

/// Whether background builds retain their immutable inputs after the initial one-shot job.
/// This is preparation for a future eviction policy only; it does not rebuild or evict PSOs.
pub(super) fn rebuild_recipes_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let value = std::env::var("XCL_PSO_REBUILD_RECIPES").ok();
        let on = recipe_gate_value(value.as_deref());
        if on {
            crate::xlog_info!("D3D12.X: XCL_PSO_REBUILD_RECIPES=1 - retaining pipeline rebuild inputs only; pipelines are not evicted or rebuilt");
        }
        on
    })
}

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

struct Generation<T> {
    id: u64,
    raw: usize,
    _owner: T,
}

struct GenerationSlot<T> {
    next: u64,
    resident: Option<Arc<Generation<T>>>,
    residency: Residency,
}

impl<T> Default for GenerationSlot<T> {
    fn default() -> Self { Self { next: 0, resident: None, residency: Residency::Rebuilding } }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Residency {
    Resident,
    Evicted,
    Rebuilding,
    Failed,
}

impl<T> GenerationSlot<T> {
    fn publish(&mut self, raw: usize, owner: T) -> (Arc<Generation<T>>, Option<Arc<Generation<T>>>) {
        self.next = self.next.checked_add(1).expect("pipeline generation id exhausted");
        let generation = Arc::new(Generation { id: self.next, raw, _owner: owner });
        let replaced = self.resident.replace(generation.clone());
        self.residency = Residency::Resident;
        (generation, replaced)
    }

    fn pin(&self, raw: usize) -> Option<Arc<Generation<T>>> {
        self.resident.as_ref().filter(|generation| generation.raw == raw).cloned()
    }

    fn mark_rebuilding(&mut self) -> Residency {
        let previous = self.residency;
        self.residency = Residency::Rebuilding;
        previous
    }

    fn finish_rebuild(&mut self, result: RebuildResult, previous: Option<Residency>) {
        self.residency = if self.resident.is_some() {
            Residency::Resident
        } else {
            match result {
                RebuildResult::Failed => Residency::Failed,
                RebuildResult::Abandoned | RebuildResult::NotReady => match previous {
                    Some(Residency::Evicted) => Residency::Evicted,
                    Some(Residency::Failed) => Residency::Failed,
                    _ => Residency::Failed,
                },
                RebuildResult::Published { .. } => Residency::Failed,
            }
        };
    }

    fn detach(&mut self, raw: usize) -> Option<Arc<Generation<T>>> {
        if self.residency != Residency::Resident || self.resident.as_ref()?.raw != raw {
            return None;
        }
        self.residency = Residency::Evicted;
        self.resident.take()
    }
}

struct RecipeInner {
    build: Box<dyn Fn() -> usize + Send + Sync>,
    generation: Mutex<GenerationSlot<ID3D12PipelineState>>,
    rebuild: Mutex<RebuildState>,
    rebuild_done: Condvar,
}

#[derive(Default)]
struct RebuildState {
    active: bool,
    completed: u64,
    waiters: usize,
    last: Option<RebuildResult>,
    from: Option<Residency>,
}

#[derive(Debug)]
enum RebuildTurn {
    Leader,
    Follower(RebuildResult),
}

/// One immutable native pipeline generation. Its strong owner keeps the exact object alive even
/// after the title releases its stand-in or a later rebuild publishes the same raw address.
#[derive(Clone)]
pub(super) struct NativeGeneration(Arc<Generation<ID3D12PipelineState>>);

impl NativeGeneration {
    #[allow(dead_code)] // consumed by the later retirement diagnostics stage
    pub(super) fn id(&self) -> u64 { self.0.id }
    #[allow(dead_code)] // exposed to focused generation-capture tests
    pub(super) fn raw(&self) -> usize { self.0.raw }
    pub(super) fn same(&self, other: &Self) -> bool { Arc::ptr_eq(&self.0, &other.0) }
    // The owner allocation, rather than `raw`, distinguishes ABA reuse by a later generation.
    fn identity(&self) -> usize { Arc::as_ptr(&self.0) as usize }
}

struct GenerationRetirements<G, R> {
    pending: Vec<(R, Vec<G>)>,
    poisoned: Vec<G>,
    poisoned_identities: Option<HashSet<usize>>,
}

impl<G, R> GenerationRetirements<G, R> {
    const fn new() -> Self {
        Self { pending: Vec::new(), poisoned: Vec::new(), poisoned_identities: None }
    }

    fn sweep(&mut self, complete: impl Fn(&R) -> bool) -> Vec<(R, Vec<G>)> {
        let mut retired = Vec::new();
        let mut i = 0;
        while i < self.pending.len() {
            if complete(&self.pending[i].0) {
                retired.push(self.pending.swap_remove(i));
            } else {
                i += 1;
            }
        }
        retired
    }

    fn retire_pending(&mut self, receipt: R, pins: Vec<G>, complete: impl Fn(&R) -> bool) -> Vec<(R, Vec<G>)> {
        if !pins.is_empty() {
            self.pending.push((receipt, pins));
        }
        self.sweep(complete)
    }

    fn poison(&mut self, pins: Vec<G>, identity: impl Fn(&G) -> usize) {
        let identities = self.poisoned_identities.get_or_insert_with(HashSet::new);
        for pin in pins {
            if identities.insert(identity(&pin)) {
                self.poisoned.push(pin);
            }
        }
    }
}

fn dedup_generations<G>(pins: impl Iterator<Item = G>, identity: impl Fn(&G) -> usize) -> Vec<G> {
    let mut unique = Vec::new();
    let mut identities = HashSet::new();
    for pin in pins {
        if identities.insert(identity(&pin)) {
            unique.push(pin);
        }
    }
    unique
}

static RETIRING_GENERATIONS: Mutex<GenerationRetirements<NativeGeneration, super::alloc_pool::SubmissionReceipt>> =
    Mutex::new(GenerationRetirements::new());

/// Keep the exact native generations used by one executed title submission until its strong
/// queue receipt completes. Completed batches are swept whenever another pinned batch arrives.
pub(super) fn retire_native_generations<'a>(
    receipt: super::alloc_pool::SubmissionReceipt,
    pins: impl Iterator<Item = &'a NativeGeneration>,
) {
    if receipt.is_complete() {
        return;
    }
    let pins = dedup_generations(pins.cloned(), NativeGeneration::identity);
    if pins.is_empty() {
        return;
    }
    let mut retiring = RETIRING_GENERATIONS.lock().unwrap();
    let retired = retiring.retire_pending(receipt, pins, super::alloc_pool::SubmissionReceipt::is_complete);
    drop(retiring);
    drop(retired);
}

/// Native execution began but no exact retirement marker could be recorded. Retain every exact
/// generation permanently: dropping any of them could free a pipeline still referenced by GPU work.
pub(super) fn poison_native_generations<'a>(pins: impl Iterator<Item = &'a NativeGeneration>) {
    let pins = dedup_generations(pins.cloned(), NativeGeneration::identity);
    if pins.is_empty() {
        return;
    }
    let mut retiring = RETIRING_GENERATIONS.lock().unwrap();
    let retired = retiring.sweep(super::alloc_pool::SubmissionReceipt::is_complete);
    retiring.poison(pins, NativeGeneration::identity);
    drop(retiring);
    drop(retired);
}

/// Rebuildable immutable inputs for one pipeline. The first build and any later rebuild use the
/// same operation; retaining it does not by itself schedule or run more work. Its existing opt-in
/// allocation also owns the native generation, so ordinary one-shot builds allocate no new state.
#[derive(Clone)]
pub(super) struct Recipe(Arc<RecipeInner>);

impl Recipe {
    pub(super) fn new(build: impl Fn() -> usize + Send + Sync + 'static) -> Self {
        Self(Arc::new(RecipeInner {
            build: Box::new(build),
            generation: Mutex::new(GenerationSlot::default()),
            rebuild: Mutex::new(RebuildState::default()),
            rebuild_done: Condvar::new(),
        }))
    }

    fn run(&self) -> usize {
        (self.0.build)()
    }

    /// Publish before `Build.real` becomes visible. The returned generation owns one independent
    /// COM reference; the raw result remains the existing `Build.real` owner.
    fn publish(&self, raw: usize) -> (NativeGeneration, Option<NativeGeneration>) {
        let p = raw as *mut std::ffi::c_void;
        let owner = unsafe { ID3D12PipelineState::from_raw_borrowed(&p) }.expect("non-null pipeline result").clone();
        let (generation, replaced) = self.0.generation.lock().unwrap().publish(raw, owner);
        (NativeGeneration(generation), replaced.map(NativeGeneration))
    }

    fn pin_with_residency(&self, raw: usize) -> (Option<NativeGeneration>, Residency) {
        let generation = self.0.generation.lock().unwrap();
        (generation.pin(raw).map(NativeGeneration), generation.residency)
    }

    fn mark_failed(&self) {
        let mut generation = self.0.generation.lock().unwrap();
        if generation.resident.is_none() {
            generation.residency = Residency::Failed;
        }
    }

    fn detach_current(&self, raw: usize) -> Option<NativeGeneration> {
        self.0.generation.lock().unwrap().detach(raw).map(NativeGeneration)
    }

    fn initial_job(&self) -> Job {
        let recipe = self.clone();
        Box::new(move || recipe.run())
    }

    fn begin_rebuild(&self) -> RebuildTurn {
        let mut state = self.0.rebuild.lock().unwrap();
        loop {
            if !state.active && state.waiters == 0 {
                state.active = true;
                state.from = Some(self.0.generation.lock().unwrap().mark_rebuilding());
                return RebuildTurn::Leader;
            }
            if state.active {
                let completed = state.completed;
                state.waiters += 1;
                while state.completed == completed {
                    state = self.0.rebuild_done.wait(state).unwrap();
                }
                let result = state.last.expect("completed rebuild without a result");
                state.waiters -= 1;
                if state.waiters == 0 {
                    self.0.rebuild_done.notify_all();
                }
                return RebuildTurn::Follower(result);
            }
            // Followers of the completed attempt must consume `last` before another leader can
            // overwrite it. A request arriving during this drain starts (or joins) the next one.
            state = self.0.rebuild_done.wait(state).unwrap();
        }
    }

    /// Reserve one asynchronous attempt without waiting for an attempt already in flight.
    fn reserve_rebuild(&self) -> bool {
        let mut state = self.0.rebuild.lock().unwrap();
        if state.active || state.waiters != 0 {
            return false;
        }
        state.active = true;
        state.from = Some(self.0.generation.lock().unwrap().mark_rebuilding());
        true
    }

    /// Reserve a build only when no native generation is resident. Failed attempts are retryable;
    /// a live current generation is never replaced by this path.
    fn reserve_rehydrate(&self) -> Result<u64, RehydrateRequest> {
        let mut state = self.0.rebuild.lock().unwrap();
        if state.active || state.waiters != 0 {
            return Err(RehydrateRequest::Coalesced);
        }
        let mut generation = self.0.generation.lock().unwrap();
        if !matches!(generation.residency, Residency::Evicted | Residency::Failed) || generation.resident.is_some() {
            return Err(RehydrateRequest::Resident);
        }
        let turn = state.completed;
        state.active = true;
        state.from = Some(generation.mark_rebuilding());
        Ok(turn)
    }

    fn finish_rebuild(&self, result: RebuildResult) {
        let mut state = self.0.rebuild.lock().unwrap();
        self.0.generation.lock().unwrap().finish_rebuild(result, state.from.take());
        state.last = Some(result);
        state.active = false;
        state.completed = state.completed.checked_add(1).expect("pipeline rebuild serial exhausted");
        self.0.rebuild_done.notify_all();
    }

    /// A queued reservation was refused or discarded before it ran. Restore its retryable state
    /// and wake any synchronous follower which joined it.
    fn cancel_reserved_rebuild(&self, turn: u64) {
        let mut state = self.0.rebuild.lock().unwrap();
        if !state.active || state.completed != turn {
            return;
        }
        self.0.generation.lock().unwrap().finish_rebuild(RebuildResult::NotReady, state.from.take());
        state.last = Some(RebuildResult::NotReady);
        state.active = false;
        state.completed = state.completed.checked_add(1).expect("pipeline rebuild serial exhausted");
        self.0.rebuild_done.notify_all();
    }
}

/// Own one exact accepted rehydration turn until its queued closure completes. Queue refusal,
/// cancellation, shutdown, or an unwind drops the closure and restores a retryable residency;
/// its completed serial prevents an old guard from cancelling a newer turn.
struct RehydrateReservation {
    recipe: Recipe,
    turn: u64,
}

impl Drop for RehydrateReservation {
    fn drop(&mut self) {
        self.recipe.cancel_reserved_rebuild(self.turn);
    }
}

/// Drop an unstarted orphaned job outside all locks and wake any concurrent waiter.
fn drop_unstarted_job(build: &Arc<Build>) {
    let job = build.job.lock().unwrap().take();
    if let Some(job) = job {
        build.cancel_queued();
        {
            let mut done = build.done.lock().unwrap();
            *done = true;
            build.finish_queued();
            build.cv.notify_all();
        }
        drop(job);
        forget_in_flight(build);
    }
}

fn queue() -> &'static BuildQueue {
    static Q: OnceLock<BuildQueue> = OnceLock::new();
    Q.get_or_init(BuildQueue::new)
}

pub fn ranked_preload_stats() -> RankedPreloadStats {
    let snapshot = queue().ranked_preload_stats();
    RankedPreloadStats {
        submissions: snapshot.submissions,
        started: snapshot.started,
        promoted: snapshot.promoted,
    }
}

fn start_normal_workers() -> bool {
    static STARTED: OnceLock<bool> = OnceLock::new();
    *STARTED.get_or_init(|| {
        let q = queue();
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        let mut started = 0;
        for k in 0..n {
            let worker = q.worker_between(Priority::BoundSoon, Priority::Foreground);
            if std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || {
                NORMAL_WORKER.set(true);
                worker.run();
            }).is_ok() {
                started += 1;
            }
        }
        started != 0
    })
}

/// Normal workers waiting for a huge driver-compile slot, and bounded replacements serving their
/// priority band. Preload workers have their own band and do not need foreground replacements.
static PARKED_NORMAL: AtomicUsize = AtomicUsize::new(0);
static NORMAL_HELPERS: AtomicUsize = AtomicUsize::new(0);
static NORMAL_HELPERS_STARTED: AtomicUsize = AtomicUsize::new(0);

/// A normal pipeline worker is about to wait for a huge driver-compile slot. A bounded helper
/// serves its queue until workers return; calls from title and preload threads do nothing.
pub(super) fn parked_for_big_compile() -> ParkedWorker {
    if !NORMAL_WORKER.get() {
        return ParkedWorker(false);
    }
    let parked = PARKED_NORMAL.fetch_add(1, Ordering::AcqRel) + 1;
    let most = parked.min(crate::gpu::recomp::dxc::compile_threads());
    let mut helpers = NORMAL_HELPERS.load(Ordering::Acquire);
    while helpers < most {
        match NORMAL_HELPERS.compare_exchange(helpers, helpers + 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                spawn_normal_helper(helpers);
                break;
            }
            Err(now) => helpers = now,
        }
    }
    ParkedWorker(true)
}

pub(super) struct ParkedWorker(bool);

impl Drop for ParkedWorker {
    fn drop(&mut self) {
        if self.0 {
            PARKED_NORMAL.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

fn spawn_normal_helper(k: usize) {
    let worker = queue().worker_between(Priority::BoundSoon, Priority::Foreground);
    let n = NORMAL_HELPERS_STARTED.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: a pipeline worker waits for a huge compile slot – a helper serves its queue meanwhile ({n} helpers so far)");
    }
    let spawned = std::thread::Builder::new().name(format!("xcl-pso-h{k}")).stack_size(64 << 20).spawn(move || {
        NORMAL_WORKER.set(true);
        let mut retired = false;
        worker.run_while(|| {
            retired = retire_normal_helper();
            !retired
        });
        if !retired {
            NORMAL_HELPERS.fetch_sub(1, Ordering::AcqRel);
        }
    });
    if spawned.is_err() {
        NORMAL_HELPERS.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Reserve this helper's retirement once more helpers exist than parked normal workers.
fn retire_normal_helper() -> bool {
    let mut helpers = NORMAL_HELPERS.load(Ordering::Acquire);
    while helpers > PARKED_NORMAL.load(Ordering::Acquire) {
        match NORMAL_HELPERS.compare_exchange(helpers, helpers - 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(now) => helpers = now,
        }
    }
    false
}

fn start_preload_workers() -> bool {
    static STARTED: OnceLock<bool> = OnceLock::new();
    *STARTED.get_or_init(|| {
        let q = queue();
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        let mut started = 0;
        for k in 0..n {
            let worker = q.worker_between(Priority::Speculative, Priority::Preload);
            if std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
                PRELOAD_WORKER.set(true);
                worker.run();
            }).is_ok() {
                started += 1;
            }
        }
        started != 0
    })
}

fn run_reserved_rebuild(build: Arc<Build>, recipe: Recipe) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_NORMAL};
    let preload = PRELOAD_WORKER.get();
    BUILDING.with(|b| *b.borrow_mut() = Some(build.clone()));
    if preload {
        let mut demand = build.demand.lock().unwrap();
        demand.preload_thread = unsafe { GetCurrentThreadId() };
        if build.wanted.load(Ordering::Acquire) {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL) };
        }
    }
    let _running = RunningBuild { build: &build, preload };
    let _ = rebuild_reserved(&build, &recipe);
}

fn run_reserved_rehydrate(build: Arc<Build>, reservation: RehydrateReservation) {
    let recipe = reservation.recipe.clone();
    run_reserved_rebuild(build, recipe);
    // `rebuild_reserved` completed the turn. Drop is intentionally still armed so an unwind or a
    // future early return cannot strand Rebuilding; it observes inactive after normal completion.
    drop(reservation);
}

fn request_rebuild_on(build: Arc<Build>, q: &BuildQueue) -> RebuildRequest {
    let recipe = {
        let done = build.done.lock().unwrap();
        if !*done {
            return RebuildRequest::NotReady;
        }
        if build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
            return RebuildRequest::Abandoned;
        }
        let Some(recipe) = build.recipe.clone() else { return RebuildRequest::Unavailable };
        if !recipe.reserve_rebuild() {
            return RebuildRequest::Coalesced;
        }
        recipe
    };
    let failed_recipe = recipe.clone();
    if q.submit(Priority::Speculative, Box::new(move || run_reserved_rebuild(build, recipe))).is_none() {
        failed_recipe.finish_rebuild(RebuildResult::Failed);
        return RebuildRequest::Unavailable;
    }
    RebuildRequest::Queued
}

fn request_rehydrate_on(build: Arc<Build>, q: &BuildQueue) -> RehydrateRequest {
    let reservation = {
        let done = build.done.lock().unwrap();
        if !*done {
            return RehydrateRequest::NotReady;
        }
        if build.owners.load(Ordering::Acquire) == 0 || build.orphan.load(Ordering::Acquire) {
            return RehydrateRequest::Abandoned;
        }
        if build.real.load(Ordering::Acquire) != 0 {
            return RehydrateRequest::Resident;
        }
        let Some(recipe) = build.recipe.clone() else { return RehydrateRequest::Unavailable };
        let turn = match recipe.reserve_rehydrate() {
            Ok(turn) => turn,
            Err(result) => return result,
        };
        RehydrateReservation { recipe, turn }
    };
    if q.submit(Priority::Speculative, Box::new(move || run_reserved_rehydrate(build, reservation))).is_none() {
        // The refused work is dropped by `submit`, restoring Evicted/Failed before it returns.
        return RehydrateRequest::Unavailable;
    }
    RehydrateRequest::Queued
}

/// Queue one low-priority replacement build. This is intentionally uncalled until an eviction
/// policy has selected a pipeline; requesting never waits for driver work on the title thread.
#[allow(dead_code)]
fn request_rebuild(pso: usize) -> RebuildRequest {
    let Some(pending) = entry(pso) else { return RebuildRequest::Unavailable };
    if !start_preload_workers() {
        return RebuildRequest::Unavailable;
    }
    request_rebuild_on(pending.build.clone(), queue())
}

/// Queue low-priority reconstruction only for an absent recipe-backed generation. This is an
/// uncalled foundation: an eviction policy must decide when absence is acceptable before any bind
/// path may trigger it.
#[allow(dead_code)]
fn request_rehydrate(pso: usize) -> RehydrateRequest {
    let Some(pending) = entry(pso) else { return RehydrateRequest::Unavailable };
    if !start_preload_workers() {
        return RehydrateRequest::Unavailable;
    }
    request_rehydrate_on(pending.build.clone(), queue())
}

fn submit_job(q: &BuildQueue, build: &Arc<Build>, priority: Priority, preload: bool, preload_rank: Option<usize>) -> Option<JobHandle> {
    let queued_build = build.clone();
    q.submit_ranked(priority, preload_rank, Box::new(move || {
        let on_preload_worker = PRELOAD_WORKER.get();
        let built = build_queued(&queued_build);
        if built && preload && on_preload_worker {
            static DONE: AtomicUsize = AtomicUsize::new(0);
            let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
            }
        }
        queued_build.queued.lock().unwrap().take();
    }))
}

#[cfg(test)]
fn enqueue_on(q: &BuildQueue, build: &Arc<Build>, priority: Priority, preload: bool) {
    *build.queued.lock().unwrap() = submit_job(q, build, priority, preload, None);
}

fn schedule(build: &Arc<Build>, priority: Priority, preload: bool, preload_rank: Option<usize>) {
    if preload {
        if !start_preload_workers() {
            return;
        }
    } else if !start_normal_workers() {
        fail_unrunnable_build(build);
        return;
    }
    let mut queued = build.queued.lock().unwrap();
    let mut refused = false;
    if !build.sent.swap(true, Ordering::AcqRel) {
        let priority = if build.wanted.load(Ordering::Acquire) || build.demand.lock().unwrap().by_submit {
            Priority::Foreground
        } else {
            priority
        };
        *queued = submit_job(queue(), build, priority, preload, preload_rank);
        if queued.is_some() {
            build.note_queued();
        } else {
            refused = true;
        }
    }
    drop(queued);
    if refused {
        fail_unrunnable_build(build);
    }
}

/// No normal worker can run this job, or its queue refused it. Fail instead of leaving title
/// waiters asleep on work which cannot make progress.
fn fail_unrunnable_build(build: &Arc<Build>) {
    build.cancel_queued();
    let job = build.job.lock().unwrap().take();
    if let Some(job) = job {
        drop(job);
        complete(build, 0);
    }
}

fn build_queued(build: &Arc<Build>) -> bool {
    // (released by the title meanwhile)
    if build.orphan.load(Ordering::Acquire) {
        drop_unstarted_job(build);
        return false;
    }
    // (None: an orphaned build was dropped before this queued work ran)
    let job = build.job.lock().unwrap().take();
    if let Some(job) = job {
        run_build(build, job);
        true
    } else {
        false
    }
}

/// The caller also drains its batch. Multiple workers can therefore build
/// different chains without ever waiting for a free worker to start a piece.
pub(super) fn build_pieces(jobs: Vec<Job>) -> Vec<usize> {
    let count = jobs.len();
    let jobs = Arc::new(Mutex::new(jobs.into_iter().enumerate().collect::<Vec<_>>()));
    let (send, recv) = mpsc::channel();
    let drain = move || loop {
        let job = jobs.lock().unwrap().pop();
        let Some((i, job)) = job else { return };
        let _ = send.send((i, run_job(job)));
    };
    // the helpers build for this worker's pipeline: the title waiting for it waits for them too
    let parent = BUILDING.with(|b| b.borrow().clone());
    let _ = start_normal_workers();
    for _ in 1..count.min(crate::gpu::recomp::dxc::compile_threads()) {
        let (drain, parent) = (drain.clone(), parent.clone());
        let _ = queue().submit(Priority::BoundSoon, Box::new(move || {
            let before = BUILDING.with(|b| b.replace(parent));
            let _restore = RestoreBuilding(before);
            drain();
        }));
    }
    drain();
    let mut results = vec![0; count];
    for (i, result) in recv.iter().take(count) { results[i] = result; }
    results
}

/// Run one demanded auxiliary pipeline build on the foreground worker queue and wait for it.
/// Unlike `build_pieces`, the caller never drains or executes the job itself.
pub(super) fn build_foreground(job: Job) -> usize {
    let build = Arc::new(Build::new(job, false));
    build.want();
    schedule(&build, Priority::Foreground, false, None);
    let _held = Held::new();
    let _stall = crate::stall::span(crate::stall::StallKind::PipelineWait);
    let mut done = build.done.lock().unwrap();
    while !*done {
        done = build.cv.wait(done).unwrap();
    }
    build.real.load(Ordering::Acquire)
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_keyed(standin, compute, job, None);
}

pub(super) fn submit_keyed(
    standin: usize,
    compute: bool,
    job: Job,
    compute_key: Option<super::real::compute_build_key::ComputeBuildKey>,
) {
    submit_as(standin, compute, job, None, true, 0, compute_key);
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built when first used – the first bind of a graphics one
/// sends it to the workers, a compute one is built at its first bind. A retail game makes
/// ~51 000 pipelines in its first 30 s and uses few; each one built stays in the driver
/// (~400 KB), and the process ran out of commit.
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// `submit`, sent to the workers only on its first `resolve` (`lazy`) – or now, at a low
/// priority, when an earlier run used a pipeline of these shaders (`key`, 0 = none; `used`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_lazy_keyed(standin, compute, job, key, None);
}

pub(super) fn submit_lazy_keyed(
    standin: usize,
    compute: bool,
    job: Job,
    key: u64,
    compute_key: Option<super::real::compute_build_key::ComputeBuildKey>,
) {
    submit_as(standin, compute, job, None, false, key, compute_key);
}

pub(super) fn submit_recipe_keyed(
    standin: usize,
    compute: bool,
    recipe: Recipe,
    send: bool,
    key: u64,
    compute_key: Option<super::real::compute_build_key::ComputeBuildKey>,
) {
    let job = recipe.initial_job();
    submit_as(standin, compute, job, Some(recipe), send, key, compute_key);
}

fn submit_as(
    standin: usize,
    compute: bool,
    job: Job,
    recipe: Option<Recipe>,
    send: bool,
    key: u64,
    compute_key: Option<super::real::compute_build_key::ComputeBuildKey>,
) {
    debug_assert!(compute || compute_key.is_none());
    let build = if let Some(compute_key) = compute_key {
        let mut registry = IN_FLIGHT.lock().unwrap();
        let builds = registry.get_or_insert_with(HashMap::new);
        let before = builds.len();
        builds.retain(|_, build| build.strong_count() != 0);
        COALESCING.finished(before - builds.len());
        if let Some(build) = builds.get(&compute_key).and_then(Weak::upgrade) {
            let done = build.done.lock().unwrap();
            if !*done && !build.orphan.load(Ordering::Acquire) {
                build.owners.fetch_add(1, Ordering::Relaxed);
                drop(done);
                COALESCING.hit();
                build
            } else {
                if builds.remove(&compute_key).is_some() {
                    COALESCING.finished(1);
                }
                let compute_key = Arc::new(compute_key);
                let build = Arc::new(Build::new_keyed(job, false, Some(compute_key.clone()), recipe));
                builds.insert(compute_key, Arc::downgrade(&build));
                COALESCING.unique();
                build
            }
        } else {
            let compute_key = Arc::new(compute_key);
            let build = Arc::new(Build::new_keyed(job, false, Some(compute_key.clone()), recipe));
            builds.insert(compute_key, Arc::downgrade(&build));
            COALESCING.unique();
            build
        }
    } else {
        Arc::new(Build::new_keyed(job, false, None, recipe))
    };
    let p = Arc::new(Pending {
        build: build.clone(),
        driver: AtomicUsize::new(0),
        compute,
        key,
        used: Default::default(),
        late: Default::default(),
        spared: Default::default(),
    });
    build.tracked.store(true, Ordering::Release);
    PENDING.insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        schedule(&build, Priority::BoundSoon, false, None);
    } else if let Some(rank) = used::rank(key).filter(|_| preload_enabled()) {
        schedule(&build, Priority::Preload, true, Some(rank));
    }
}

/// The keys of the pipelines earlier runs used (`used-pipelines.txt` beside the DXIL cache):
/// what `xcl shaders --precompile --used-only` builds.
pub fn earlier_used_keys() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt")).unwrap_or_default())
}

/// Ask the used-pipeline writer to finish without ever joining it unboundedly.
pub fn shutdown_used() {
    used::shutdown(std::time::Duration::from_millis(500));
}

/// Best effort at process termination: only the one writer touches the append handle, and a
/// loader-lock-blocked writer cannot hold termination longer than `timeout`.
pub fn flush_used_bounded(timeout: std::time::Duration) {
    used::shutdown(timeout);
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use: until then a graphics one draws nothing, and a
/// compute one is waited for on the title's thread. A retail title's first frames of its world
/// use hundreds of pipelines for the first time at once. The pipelines of an earlier run are
/// built while the title loads instead (`preload`): measured on a build that skips a dispatch
/// instead of waiting for its pipeline, ~2 170 dispatches were skipped in those frames without
/// this and 94 – 150 with it.
mod used {
    use std::collections::{HashMap, HashSet};
    use std::sync::{OnceLock, RwLock};
    use std::time::Duration;

    use super::super::used_journal::Journal;

    struct Used {
        earlier: RwLock<HashMap<u64, usize>>,
        now: Journal,
    }

    static U: OnceLock<Used> = OnceLock::new();

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        parse_ranked(text).into_keys().collect()
    }

    /// First occurrence order of the valid, nonzero keys in a `used-pipelines.txt`.
    pub fn parse_ranked(text: &str) -> HashMap<u64, usize> {
        let mut ranked = HashMap::new();
        for key in text.lines().filter_map(|line| u64::from_str_radix(line.trim(), 16).ok()).filter(|&key| key != 0) {
            let rank = ranked.len();
            ranked.entry(key).or_insert(rank);
        }
        ranked
    }

    fn get() -> &'static Used {
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashMap::new()), now: Journal::memory(HashSet::new()) };
            }
            let path = crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt");
            let earlier = parse_ranked(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            let now = Journal::file(path, earlier.keys().copied().collect());
            Used { now, earlier: RwLock::new(earlier) }
        })
    }

    pub fn shutdown(timeout: Duration) {
        if let Some(used) = U.get() {
            let _ = used.now.shutdown_bounded(timeout);
        }
    }

    /// First-use rank when an earlier run used a pipeline of these shaders.
    pub fn rank(key: u64) -> Option<usize> {
        (key != 0).then(|| get().earlier.read().unwrap().get(&key).copied()).flatten()
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        get().now.note(key);
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.contains(key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        let mut earlier = get().earlier.write().unwrap();
        let rank = earlier.len();
        earlier.entry(key).or_insert(rank);
    }
}

/// Build pipelines from an earlier run in the preload worker band (`XCL_PSO_PRELOAD=0`: off).
fn preload_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0"))
}

#[cfg(test)]
mod used_tests {
    use super::*;

    #[repr(C)]
    struct RebuildFakePso {
        vtable: *const usize,
        refs: AtomicUsize,
    }

    unsafe extern "system" fn rebuild_fake_add_ref(this: *mut std::ffi::c_void) -> u32 {
        (*(this as *mut RebuildFakePso)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
    }

    unsafe extern "system" fn rebuild_fake_release(this: *mut std::ffi::c_void) -> u32 {
        (*(this as *mut RebuildFakePso)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
    }

    #[test]
    fn rebuild_recipe_gate_requires_exact_one() {
        assert!(recipe_gate_value(Some("1")));
        assert!(recipe_gate_value(Some(" 1\t")));
        for value in [None, Some(""), Some("0"), Some("true"), Some("2")] {
            assert!(!recipe_gate_value(value));
        }
    }

    #[test]
    fn eviction_candidate_ring_is_bounded_reusable_and_does_not_retain_builds() {
        assert!(std::panic::catch_unwind(CandidateQueue::<usize, 0>::new).is_err());
        assert!(std::panic::catch_unwind(CandidateQueue::<usize, 1>::new).is_err());
        let queue = CandidateQueue::<usize, 2>::new();
        let first = Arc::new(1usize);
        let second = Arc::new(2usize);
        let third = Arc::new(3usize);
        assert!(queue.push(Arc::downgrade(&first)).is_ok());
        assert!(queue.push(Arc::downgrade(&second)).is_ok());
        assert!(queue.push(Arc::downgrade(&third)).is_err(), "a full candidate ring accepted an unbounded record");
        assert_eq!(*queue.pop().unwrap().upgrade().unwrap(), 1);
        assert!(queue.push(Arc::downgrade(&third)).is_ok(), "a drained ring slot was not reusable");
        assert_eq!(*queue.pop().unwrap().upgrade().unwrap(), 2);
        assert_eq!(*queue.pop().unwrap().upgrade().unwrap(), 3);
        assert!(queue.pop().is_none());

        let builds = CandidateQueue::<Build, 2>::new();
        let build = Arc::new(Build::new(Box::new(|| 0), false));
        let weak = Arc::downgrade(&build);
        note_eviction_candidate_on(&build, 7, &builds);
        drop(build);
        assert!(weak.upgrade().is_none(), "an unconsumed candidate retained a released build");
        assert!(builds.pop().unwrap().upgrade().is_none(), "a released build record changed identity");
    }

    #[test]
    fn eviction_candidate_ring_mpmc_delivers_every_weak_once() {
        const COUNT: usize = 512;
        let queue = CandidateQueue::<usize, 64>::new();
        let values: Vec<_> = (0..COUNT).map(Arc::new).collect();
        let seen: Vec<_> = (0..COUNT).map(|_| AtomicUsize::new(0)).collect();
        let consumed = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for producer in 0..4 {
                let queue = &queue;
                let values = &values;
                scope.spawn(move || {
                    for index in (producer..COUNT).step_by(4) {
                        let mut value = Arc::downgrade(&values[index]);
                        loop {
                            match queue.push(value) {
                                Ok(()) => break,
                                Err(returned) => {
                                    value = returned;
                                    std::thread::yield_now();
                                }
                            }
                        }
                    }
                });
            }
            for _ in 0..4 {
                let queue = &queue;
                let seen = &seen;
                let consumed = &consumed;
                scope.spawn(move || loop {
                    if consumed.load(Ordering::Acquire) == COUNT {
                        break;
                    }
                    let Some(value) = queue.pop() else {
                        std::thread::yield_now();
                        continue;
                    };
                    let index = *value.upgrade().expect("payload died while retained by the test");
                    seen[index].fetch_add(1, Ordering::Relaxed);
                    consumed.fetch_add(1, Ordering::Release);
                });
            }
        });
        assert_eq!(consumed.load(Ordering::Relaxed), COUNT);
        assert!(seen.iter().all(|count| count.load(Ordering::Relaxed) == 1), "a payload was lost or delivered twice");
        assert!(values.iter().all(|value| Arc::weak_count(value) == 0), "a queued Weak payload was not dropped exactly once");
        assert!(queue.pop().is_none());
    }

    #[test]
    fn eviction_candidates_deduplicate_bound_drain_and_filter_ineligible_builds() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut second = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let second_raw = &mut second as *mut RebuildFakePso as usize;
        let first_recipe = Recipe::new(move || first_raw);
        let second_recipe = Recipe::new(move || second_raw);
        let first_build = Arc::new(Build::new_keyed(
            first_recipe.initial_job(), false, None, Some(first_recipe.clone()),
        ));
        let second_build = Arc::new(Build::new_keyed(
            second_recipe.initial_job(), false, None, Some(second_recipe.clone()),
        ));
        let first_job = first_build.job.lock().unwrap().take().unwrap();
        let second_job = second_build.job.lock().unwrap().take().unwrap();
        complete(&first_build, first_job());
        complete(&second_build, second_job());
        let queue = CandidateQueue::<Build, 4>::new();

        // Ordinary/default resolution bookkeeping does not touch the opt-in candidate journal.
        let pending = Pending {
            build: first_build.clone(), driver: AtomicUsize::new(0), compute: false, key: 0,
            used: AtomicBool::new(false), late: AtomicBool::new(false), spared: AtomicBool::new(false),
        };
        pending.note_use();
        assert!(!first_build.candidate_queued.load(Ordering::Acquire));

        note_eviction_candidate_on(&first_build, 10, &queue);
        note_eviction_candidate_on(&first_build, 30, &queue);
        note_eviction_candidate_on(&second_build, 20, &queue);
        let limits = EvictionSelectionLimits {
            max_scan: 2, max_select: 1, min_unused_frames: 0, min_resident_frames: 0,
        };
        let selected = select_eviction_candidates_on(&queue, 100, limits);
        assert_eq!(selected.len(), 1);
        assert!(Arc::ptr_eq(&selected[0].candidate.build, &second_build));
        assert_eq!(selected[0].unused_frames, 80);
        assert_eq!(drain_eviction_candidates_on(&queue, 0).len(), 0, "a zero budget inspected work");
        drop(selected);
        let selected = select_eviction_candidates_on(&queue, 100, limits);
        assert_eq!(selected.len(), 1);
        assert!(Arc::ptr_eq(&selected[0].candidate.build, &first_build));
        drop(selected);

        let recent_limits = EvictionSelectionLimits {
            max_scan: 2, max_select: 1, min_unused_frames: 10, min_resident_frames: 0,
        };
        note_eviction_candidate_on(&first_build, 95, &queue);
        for _ in 0..3 {
            assert!(select_eviction_candidates_on(&queue, 100, recent_limits).is_empty());
            assert!(first_build.candidate_queued.load(Ordering::Acquire));
        }
        let selected = select_eviction_candidates_on(&queue, 105, recent_limits);
        assert_eq!(selected.len(), 1, "a deferred candidate required another bind to reappear");
        assert!(Arc::ptr_eq(&selected[0].candidate.build, &first_build));
        assert!(queue.pop().is_none(), "repeated deferral duplicated a candidate record");
        drop(selected);

        assert!(matches!(first_recipe.begin_rebuild(), RebuildTurn::Leader));
        note_eviction_candidate_on(&first_build, 40, &queue);
        assert!(select_eviction_candidates_on(&queue, 100, limits).is_empty(), "an active rebuild became evictable");
        first_recipe.finish_rebuild(RebuildResult::Failed);
        first_build.owners.store(0, Ordering::Release);
        note_eviction_candidate_on(&first_build, 50, &queue);
        assert!(select_eviction_candidates_on(&queue, 100, limits).is_empty(), "a released build became evictable");

        drop(pending);
        let first_final = first_build.real.swap(0, Ordering::AcqRel);
        let second_final = second_build.real.swap(0, Ordering::AcqRel);
        drop(first_build);
        drop(second_build);
        drop(first_recipe);
        drop(second_recipe);
        release(first_final);
        release(second_final);
        assert_eq!((first.refs.load(Ordering::Relaxed), second.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn eviction_candidate_policy_uses_latest_age_and_orders_oldest_first() {
        let make = |last_use, completed| {
            let build = Arc::new(Build::new(Box::new(|| 0), false));
            build.last_use_frame.store(last_use, Ordering::Relaxed);
            build.completed_frame.store(completed, Ordering::Relaxed);
            build.candidate_queued.store(true, Ordering::Release);
            EvictionCandidate {
                build, last_use_frame: last_use, completed_frame: completed,
                clear_queued_on_drop: true,
            }
        };
        let oldest = make(10, 5);
        let next = make(20, 1);
        let recent = make(95, 0);
        let young_resident = make(1, 95);
        let wrapped = make(u64::MAX, 0);
        let oldest_build = oldest.build.clone();
        let next_build = next.build.clone();
        let limits = EvictionSelectionLimits {
            max_scan: 4, max_select: 2, min_unused_frames: 10, min_resident_frames: 20,
        };

        let (selected, _) =
            partition_eviction_candidates(vec![recent, next, wrapped, young_resident, oldest], 100, limits);
        assert_eq!(selected.len(), 2);
        assert!(Arc::ptr_eq(&selected[0].candidate.build, &oldest_build));
        assert_eq!((selected[0].unused_frames, selected[0].resident_frames), (90, 95));
        assert!(Arc::ptr_eq(&selected[1].candidate.build, &next_build));

        // A use after background drain is still visible through the Build atomic, even though the
        // candidate's dedup flag prevents a second journal record while this owner exists.
        let used_after_drain = make(30, 1);
        used_after_drain.build.last_use_frame.store(99, Ordering::Relaxed);
        assert!(partition_eviction_candidates(vec![used_after_drain], 100, limits).0.is_empty());
        assert_eq!(conservative_frame_age(3, u64::MAX), 0, "frame wrap assigned a false old age");
    }

    #[test]
    fn eviction_candidate_requeue_full_clears_the_dedup_flag() {
        let queue = CandidateQueue::<Build, 2>::new();
        let retained = [
            Arc::new(Build::new(Box::new(|| 0), false)),
            Arc::new(Build::new(Box::new(|| 0), false)),
        ];
        assert!(queue.push(Arc::downgrade(&retained[0])).is_ok());
        assert!(queue.push(Arc::downgrade(&retained[1])).is_ok());

        let build = Arc::new(Build::new(Box::new(|| 0), false));
        build.candidate_queued.store(true, Ordering::Release);
        let candidate = EvictionCandidate {
            build: build.clone(), last_use_frame: 0, completed_frame: 0,
            clear_queued_on_drop: true,
        };
        assert!(!candidate.requeue_on(&queue));
        assert!(!build.candidate_queued.load(Ordering::Acquire));
    }

    #[test]
    fn build_retains_a_reusable_recipe_after_its_initial_job_is_consumed() {
        let calls = Arc::new(AtomicUsize::new(0));
        let recipe_calls = calls.clone();
        let recipe = Recipe::new(move || {
            recipe_calls.fetch_add(1, Ordering::Relaxed) + 0x5100
        });
        let build = Build::new_keyed(recipe.initial_job(), false, None, Some(recipe));

        let initial = build.job.lock().unwrap().take().unwrap();
        assert_eq!(initial(), 0x5100);
        assert!(build.job.lock().unwrap().is_none());
        assert_eq!(build.recipe.as_ref().unwrap().run(), 0x5101);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn native_generation_slot_pins_exact_aba_owner_and_ids_publications() {
        struct Owner(Arc<AtomicUsize>);
        impl Drop for Owner {
            fn drop(&mut self) { self.0.fetch_add(1, Ordering::Relaxed); }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let mut slot = GenerationSlot::default();
        let (first, replaced) = slot.publish(0x7E57_ABA0, Owner(drops.clone()));
        assert!(replaced.is_none());
        let pinned = slot.pin(0x7E57_ABA0).unwrap();
        assert_eq!(first.id, 1);
        let (second, replaced) = slot.publish(0x7E57_ABA0, Owner(drops.clone()));
        assert_eq!(second.id, 2);
        assert!(!Arc::ptr_eq(&pinned, &second), "a reused raw address aliased its older generation");
        let unique = dedup_generations(
            [first.clone(), first.clone(), second.clone(), second.clone()].into_iter(),
            |generation| Arc::as_ptr(generation) as usize,
        );
        assert_eq!(unique.len(), 2, "a reused raw address collapsed distinct generations");
        drop(unique);
        drop(first);
        drop(replaced);
        assert_eq!(drops.load(Ordering::Relaxed), 0, "the old generation died while a pin held it");
        drop(pinned);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        drop(second);
        drop(slot);
        assert_eq!(drops.load(Ordering::Relaxed), 2);
    }

    #[derive(Clone)]
    struct FakeReceipt {
        fence: Arc<AtomicU64>,
        value: u64,
    }

    fn fake_receipt_complete(receipt: &FakeReceipt) -> bool {
        receipt.fence.load(Ordering::Relaxed) >= receipt.value
    }

    #[test]
    fn generation_retirement_deduplicates_and_uses_the_exact_receipt_owner() {
        let generation = Arc::new(());
        let pins = dedup_generations(
            [generation.clone(), generation.clone()].into_iter(),
            |generation| Arc::as_ptr(generation) as usize,
        );
        assert_eq!(pins.len(), 1);

        let old_fence = Arc::new(AtomicU64::new(0));
        let replacement_fence = Arc::new(AtomicU64::new(99));
        let mut retiring = GenerationRetirements::new();
        drop(retiring.retire_pending(
            FakeReceipt { fence: old_fence.clone(), value: 7 },
            pins,
            fake_receipt_complete,
        ));
        drop(retiring.sweep(fake_receipt_complete));
        assert_eq!(retiring.pending.len(), 1, "a replacement queue generation retired an old receipt");
        assert_eq!(Arc::strong_count(&generation), 2, "the in-flight batch did not retain its generation");

        old_fence.store(7, Ordering::Relaxed);
        drop(retiring.sweep(fake_receipt_complete));
        assert!(retiring.pending.is_empty());
        assert_eq!(Arc::strong_count(&generation), 1);
        assert_eq!(replacement_fence.load(Ordering::Relaxed), 99);

        drop(retiring.retire_pending(
            FakeReceipt { fence: old_fence, value: 7 },
            vec![generation.clone()],
            fake_receipt_complete,
        ));
        assert!(retiring.pending.is_empty(), "an already-complete receipt retained a final batch");
    }

    #[test]
    fn untracked_generation_pins_are_never_swept() {
        let generation = Arc::new(());
        let weak = Arc::downgrade(&generation);
        let pins = dedup_generations(
            [generation.clone(), generation.clone()].into_iter(),
            |generation| Arc::as_ptr(generation) as usize,
        );
        let mut retiring = GenerationRetirements::<Arc<()>, FakeReceipt>::new();
        retiring.poison(pins, |generation| Arc::as_ptr(generation) as usize);
        retiring.poison(vec![generation.clone()], |generation| Arc::as_ptr(generation) as usize);
        assert_eq!(retiring.poisoned.len(), 1, "a later poison duplicated an existing generation");
        drop(generation);

        drop(retiring.sweep(fake_receipt_complete));
        assert!(weak.upgrade().is_some(), "sweeping tracked receipts released an untracked generation");
        assert_eq!(retiring.poisoned.len(), 1);
    }

    #[test]
    fn newly_completed_batch_is_returned_for_drop_after_unlock() {
        struct DropPin(Arc<AtomicUsize>);
        impl Drop for DropPin {
            fn drop(&mut self) { self.0.fetch_add(1, Ordering::Relaxed); }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let fence = Arc::new(AtomicU64::new(1));
        let mut retiring = GenerationRetirements::new();
        let retired = retiring.retire_pending(
            FakeReceipt { fence, value: 1 },
            vec![DropPin(drops.clone())],
            fake_receipt_complete,
        );

        assert!(retiring.pending.is_empty());
        assert_eq!(drops.load(Ordering::Relaxed), 0, "the manager dropped a pin while it was borrowed mutably");
        drop(retired);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn concurrent_generation_publications_are_monotonic() {
        let slot = Arc::new(Mutex::new(GenerationSlot::default()));
        std::thread::scope(|scope| {
            for thread in 0..4usize {
                let slot = slot.clone();
                scope.spawn(move || {
                    for k in 0..100usize {
                        let (_, replaced) = slot.lock().unwrap().publish(thread * 100 + k + 1, ());
                        drop(replaced);
                    }
                });
            }
        });
        let slot = slot.lock().unwrap();
        assert_eq!(slot.next, 400);
        assert_eq!(slot.resident.as_ref().unwrap().id, 400);
    }

    #[test]
    fn overlapping_recipe_rebuild_requests_share_one_result() {
        let recipe = Recipe::new(|| 0);
        assert!(matches!(recipe.begin_rebuild(), RebuildTurn::Leader));
        let follower_recipe = recipe.clone();
        let follower = std::thread::spawn(move || follower_recipe.begin_rebuild());
        let started = std::time::Instant::now();
        loop {
            if recipe.0.rebuild.lock().unwrap().waiters == 1 {
                break;
            }
            assert!(started.elapsed() < std::time::Duration::from_secs(2), "follower did not join the active rebuild");
            std::thread::yield_now();
        }
        let result = RebuildResult::Published { generation: 7 };
        recipe.finish_rebuild(result);
        assert!(matches!(follower.join().unwrap(), RebuildTurn::Follower(r) if r == result));
        assert_eq!(recipe.0.rebuild.lock().unwrap().completed, 1);
    }

    #[test]
    fn next_leader_waits_for_prior_followers_to_consume_their_result() {
        let recipe = Recipe::new(|| 0);
        let first = RebuildResult::Published { generation: 11 };
        {
            // Deterministically model a completed attempt whose notified follower has not yet
            // reacquired the mutex. `last` must remain immutable until that waiter drains.
            let mut state = recipe.0.rebuild.lock().unwrap();
            state.completed = 1;
            state.last = Some(first);
            state.waiters = 1;
        }
        let next_recipe = recipe.clone();
        let (leader_tx, leader_rx) = mpsc::channel();
        let next = std::thread::spawn(move || leader_tx.send(next_recipe.begin_rebuild()).unwrap());
        assert!(leader_rx.recv_timeout(std::time::Duration::from_millis(50)).is_err(), "a new leader crossed an undrained prior follower");

        let consumed = {
            let mut state = recipe.0.rebuild.lock().unwrap();
            let result = state.last.unwrap();
            state.waiters = 0;
            recipe.0.rebuild_done.notify_all();
            result
        };
        assert_eq!(consumed, first);
        assert!(matches!(leader_rx.recv().unwrap(), RebuildTurn::Leader));
        recipe.finish_rebuild(RebuildResult::Published { generation: 12 });
        next.join().unwrap();
    }

    #[test]
    fn failed_rebuild_keeps_current_generation_and_success_replaces_it() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut second = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let second_raw = &mut second as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let recipe = Recipe::new(move || build_next.load(Ordering::Acquire));
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        assert_eq!((build.real.load(Ordering::Acquire), first.refs.load(Ordering::Relaxed)), (first_raw, 2));

        next.store(0, Ordering::Release);
        assert_eq!(rebuild_once(&build), Some(RebuildResult::Failed));
        assert_eq!(build.real.load(Ordering::Acquire), first_raw);
        assert_eq!(recipe.pin_with_residency(first_raw).0.unwrap().id(), 1);
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Resident);

        next.store(second_raw, Ordering::Release);
        assert_eq!(rebuild_once(&build), Some(RebuildResult::Published { generation: 2 }));
        assert_eq!(build.real.load(Ordering::Acquire), second_raw);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0, "both superseded owners release outside publication locks");
        assert_eq!(second.refs.load(Ordering::Relaxed), 2);

        let final_raw = build.real.swap(0, Ordering::AcqRel);
        drop(build);
        drop(recipe);
        assert_eq!(second.refs.load(Ordering::Relaxed), 1);
        release(final_raw);
        assert_eq!(second.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn eviction_detaches_both_native_owners_but_an_exact_pin_survives() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        let pin = recipe.pin_with_residency(raw).0.expect("resident generation");

        assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });
        assert_eq!(build.real.load(Ordering::Acquire), 0);
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Evicted);
        assert!(recipe.pin_with_residency(raw).0.is_none(), "evicted generation remained current");
        assert_eq!(native.refs.load(Ordering::Relaxed), 1, "the exact pin did not retain the native object");
        assert_eq!(try_evict_current(&build), EvictResult::NotResident, "repeated eviction detached another owner");

        drop(pin);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn eviction_between_resolution_and_pinning_falls_back_from_the_stale_raw() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let standin = 0x7E57_C154usize;
        let recipe = Recipe::new(move || raw);
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());

        // Deterministically interleave eviction in the real resolver's gap: it observed `raw`,
        // but has not yet taken Build.done and converted that observation into an exact pin.
        let resolved = resolve_pending(&pending, standin);
        let bound_unbuilt = pending.bound_unbuilt(standin, resolved);
        assert_eq!((resolved, bound_unbuilt), (raw, false));
        assert_eq!(try_evict_current(&pending.build), EvictResult::Evicted { generation: 1 });
        assert_eq!(native.refs.load(Ordering::Relaxed), 0, "eviction did not release both unpinned owners");

        let (chosen, pin, stale_fallback) = pin_resolved(standin, &pending, resolved, bound_unbuilt);
        assert_eq!(chosen, standin, "pin validation returned the freed pre-eviction raw pointer");
        assert!(pin.is_none());
        assert!(stale_fallback, "ExecuteIndirect must treat an eviction fallback as unbuilt");
        assert_eq!(released(standin), 0);
    }

    #[test]
    fn steady_evicted_compute_resolution_remains_fail_closed_as_unbuilt() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let standin = 0x7E57_C155usize;
        let recipe = Recipe::new(move || raw);
        submit_recipe_keyed(standin, true, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());
        assert_eq!(try_evict_current(&pending.build), EvictResult::Evicted { generation: 1 });

        let ((chosen, bound_unbuilt), pin) = resolve_wait_bound_and_pin(standin);
        assert_eq!((chosen, bound_unbuilt), (standin, true));
        assert!(pin.is_none());
        assert_eq!(released(standin), 0);
        drop(pending);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn eviction_refuses_an_active_rebuild_and_restores_resident_state_on_failure() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let recipe = Recipe::new(move || raw);
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());

        assert!(matches!(recipe.begin_rebuild(), RebuildTurn::Leader));
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Rebuilding);
        assert_eq!(try_evict_current(&build), EvictResult::Active);
        assert_eq!(build.real.load(Ordering::Acquire), raw);
        recipe.finish_rebuild(RebuildResult::Failed);
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Resident);

        let final_raw = build.real.swap(0, Ordering::AcqRel);
        drop(build);
        drop(recipe);
        release(final_raw);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn eviction_refuses_a_build_after_its_last_title_owner_releases() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut native = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut native as *mut RebuildFakePso as usize;
        let standin = 0x7E57_C153usize;
        let recipe = Recipe::new(move || raw);
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());

        let final_raw = released(standin);
        assert_eq!(try_evict_current(&pending.build), EvictResult::Abandoned);
        release(final_raw);
        drop(pending);
        assert_eq!(native.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn failed_initial_recipe_has_no_evictable_residency() {
        let standin = 0x7E57_C156usize;
        let recipe = Recipe::new(|| 0);
        submit_recipe_keyed(standin, true, recipe.clone(), false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());

        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Failed);
        assert_eq!(try_evict_current(&pending.build), EvictResult::NotResident);
        let ((chosen, bound_unbuilt), pin) = resolve_wait_bound_and_pin(standin);
        assert_eq!((chosen, bound_unbuilt), (standin, false), "a real build failure changed to transient-unbuilt");
        assert!(pin.is_none());
        assert_eq!(released(standin), 0);
    }

    #[test]
    fn release_during_rebuild_abandons_and_releases_the_new_raw() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let block = Arc::new((Mutex::new((false, false)), Condvar::new()));
        let build_block = block.clone();
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let recipe = Recipe::new(move || {
            let mut state = build_block.0.lock().unwrap();
            if state.0 {
                state.1 = true;
                build_block.1.notify_all();
                while state.0 {
                    state = build_block.1.wait(state).unwrap();
                }
            }
            build_next.load(Ordering::Acquire)
        });
        let standin = 0x7E57_C150usize;
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());

        next.store(replacement_raw, Ordering::Release);
        block.0.lock().unwrap().0 = true;
        let rebuilding = pending.build.clone();
        let thread = std::thread::spawn(move || rebuild_once(&rebuilding));
        {
            let mut state = block.0.lock().unwrap();
            while !state.1 {
                state = block.1.wait(state).unwrap();
            }
        }
        let old_raw = released(standin);
        release(old_raw);
        {
            let mut state = block.0.lock().unwrap();
            state.0 = false;
            block.1.notify_all();
        }
        assert_eq!(thread.join().unwrap(), Some(RebuildResult::Abandoned));
        assert_eq!(pending.build.real.load(Ordering::Acquire), 0);
        assert_eq!(replacement.refs.load(Ordering::Relaxed), 0, "an abandoned new raw leaked");
        drop(pending);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn async_rebuild_requests_coalesce_without_running_on_the_requester() {
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            0
        });
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe)));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        let q = BuildQueue::new();

        assert_eq!(request_rebuild_on(build.clone(), &q), RebuildRequest::Queued);
        assert_eq!(request_rebuild_on(build.clone(), &q), RebuildRequest::Coalesced);
        assert_eq!(calls.load(Ordering::Relaxed), 1, "requesting performed driver work");
        assert!(q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(build.recipe.as_ref().unwrap().0.rebuild.lock().unwrap().completed, 1);
    }

    #[test]
    fn async_rebuild_released_before_start_does_not_run_or_resurrect() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            first_raw
        });
        let standin = 0x7E57_C151usize;
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());
        let q = BuildQueue::new();
        assert_eq!(request_rebuild_on(pending.build.clone(), &q), RebuildRequest::Queued);

        let current = released(standin);
        release(current);
        assert!(q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(calls.load(Ordering::Relaxed), 1, "orphaned queued request ran its recipe");
        assert_eq!(pending.build.real.load(Ordering::Acquire), 0, "queued work resurrected a released build");
        drop(pending);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn async_rebuild_failure_keeps_current_generation() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let recipe = Recipe::new(move || build_next.load(Ordering::Acquire));
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        next.store(0, Ordering::Release);
        let q = BuildQueue::new();

        assert_eq!(request_rebuild_on(build.clone(), &q), RebuildRequest::Queued);
        assert!(q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(build.real.load(Ordering::Acquire), first_raw);
        assert_eq!(recipe.pin_with_residency(first_raw).0.unwrap().id(), 1);

        let final_raw = build.real.swap(0, Ordering::AcqRel);
        drop(build);
        drop(recipe);
        release(final_raw);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn async_rebuild_released_while_worker_runs_abandons_replacement() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let block = Arc::new((Mutex::new((false, false)), Condvar::new()));
        let build_block = block.clone();
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let recipe = Recipe::new(move || {
            if build_next.load(Ordering::Acquire) == replacement_raw {
                let mut state = build_block.0.lock().unwrap();
                state.0 = true;
                build_block.1.notify_all();
                while !state.1 { state = build_block.1.wait(state).unwrap(); }
            }
            build_next.load(Ordering::Acquire)
        });
        let standin = 0x7E57_C152usize;
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());
        next.store(replacement_raw, Ordering::Release);
        let q = Arc::new(BuildQueue::new());
        assert_eq!(request_rebuild_on(pending.build.clone(), &q), RebuildRequest::Queued);
        let worker_q = q.clone();
        let worker = std::thread::spawn(move || worker_q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        {
            let mut state = block.0.lock().unwrap();
            while !state.0 { state = block.1.wait(state).unwrap(); }
        }
        let old = released(standin);
        release(old);
        block.0.lock().unwrap().1 = true;
        block.1.notify_all();
        assert!(worker.join().unwrap());
        assert_eq!(pending.build.real.load(Ordering::Acquire), 0);
        assert_eq!(replacement.refs.load(Ordering::Relaxed), 0);
        drop(pending);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn async_rehydrate_coalesces_stays_unbuilt_until_worker_publication() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let standin = 0x7E57_C157usize;
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            build_next.load(Ordering::Acquire)
        });
        submit_recipe_keyed(standin, true, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());
        let q = BuildQueue::new();
        assert_eq!(request_rehydrate_on(pending.build.clone(), &q), RehydrateRequest::Resident);
        assert_eq!(try_evict_current(&pending.build), EvictResult::Evicted { generation: 1 });
        next.store(replacement_raw, Ordering::Release);

        assert_eq!(request_rehydrate_on(pending.build.clone(), &q), RehydrateRequest::Queued);
        assert_eq!(request_rehydrate_on(pending.build.clone(), &q), RehydrateRequest::Coalesced);
        assert_eq!(calls.load(Ordering::Relaxed), 1, "requester performed driver work");
        let ((chosen, unbuilt), pin) = resolve_wait_bound_and_pin(standin);
        assert_eq!((chosen, unbuilt), (standin, true));
        assert!(pin.is_none());

        assert!(q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(pending.build.real.load(Ordering::Acquire), replacement_raw);
        assert_eq!(pending.build.recipe.as_ref().unwrap().0.generation.lock().unwrap().residency, Residency::Resident);
        let ((chosen, unbuilt), pin) = resolve_wait_bound_and_pin(standin);
        assert_eq!((chosen, unbuilt), (replacement_raw, false));
        drop(pin);

        let final_raw = released(standin);
        release(final_raw);
        drop(pending);
        assert_eq!((first.refs.load(Ordering::Relaxed), replacement.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn refused_cancelled_and_failed_rehydrate_remain_retryable() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let next = Arc::new(AtomicUsize::new(first_raw));
        let build_next = next.clone();
        let recipe = Recipe::new(move || build_next.load(Ordering::Acquire));
        let build = Arc::new(Build::new_keyed(recipe.initial_job(), false, None, Some(recipe.clone())));
        let initial = build.job.lock().unwrap().take().unwrap();
        complete(&build, initial());
        assert_eq!(try_evict_current(&build), EvictResult::Evicted { generation: 1 });

        let refused = BuildQueue::new();
        refused.shutdown();
        assert_eq!(request_rehydrate_on(build.clone(), &refused), RehydrateRequest::Unavailable);
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Evicted);
        let cancelled = BuildQueue::new();
        assert_eq!(request_rehydrate_on(build.clone(), &cancelled), RehydrateRequest::Queued);
        cancelled.shutdown();
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Evicted);

        next.store(0, Ordering::Release);
        let failed = BuildQueue::new();
        assert_eq!(request_rehydrate_on(build.clone(), &failed), RehydrateRequest::Queued);
        assert!(failed.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Failed);
        next.store(replacement_raw, Ordering::Release);
        let retry = BuildQueue::new();
        assert_eq!(request_rehydrate_on(build.clone(), &retry), RehydrateRequest::Queued);
        assert!(retry.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(build.real.load(Ordering::Acquire), replacement_raw);

        let final_raw = build.real.swap(0, Ordering::AcqRel);
        drop(build);
        drop(recipe);
        release(final_raw);
        assert_eq!((first.refs.load(Ordering::Relaxed), replacement.refs.load(Ordering::Relaxed)), (0, 0));
    }

    #[test]
    fn completed_rehydrate_guard_cannot_cancel_a_newer_turn() {
        let recipe = Recipe::new(|| 0);
        recipe.mark_failed();
        let first_turn = recipe.reserve_rehydrate().unwrap();
        let old = RehydrateReservation { recipe: recipe.clone(), turn: first_turn };
        recipe.finish_rebuild(RebuildResult::Failed);

        let second_turn = recipe.reserve_rehydrate().unwrap();
        let new = RehydrateReservation { recipe: recipe.clone(), turn: second_turn };
        assert_eq!((first_turn, second_turn), (0, 1));
        drop(old);
        {
            let state = recipe.0.rebuild.lock().unwrap();
            assert!(state.active, "old guard cancelled the newer reservation");
            assert_eq!(state.completed, second_turn);
        }
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Rebuilding);

        drop(new);
        let state = recipe.0.rebuild.lock().unwrap();
        assert!(!state.active);
        assert_eq!(state.completed, second_turn + 1);
        drop(state);
        assert_eq!(recipe.0.generation.lock().unwrap().residency, Residency::Failed);
    }

    #[test]
    fn rehydrate_released_before_worker_does_not_run_or_resurrect() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        let standin = 0x7E57_C158usize;
        let recipe = Recipe::new(move || {
            build_calls.fetch_add(1, Ordering::Relaxed);
            first_raw
        });
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());
        assert_eq!(try_evict_current(&pending.build), EvictResult::Evicted { generation: 1 });
        let q = BuildQueue::new();
        assert_eq!(request_rehydrate_on(pending.build.clone(), &q), RehydrateRequest::Queued);

        assert_eq!(released(standin), 0);
        assert!(q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        assert_eq!(calls.load(Ordering::Relaxed), 1, "released rehydration ran its recipe");
        assert_eq!(pending.build.real.load(Ordering::Acquire), 0);
        drop(pending);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn rehydrate_released_during_worker_abandons_new_raw() {
        let vtable = [0usize, rebuild_fake_add_ref as usize, rebuild_fake_release as usize];
        let mut first = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let mut replacement = RebuildFakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let first_raw = &mut first as *mut RebuildFakePso as usize;
        let replacement_raw = &mut replacement as *mut RebuildFakePso as usize;
        let calls = Arc::new(AtomicUsize::new(0));
        let build_calls = calls.clone();
        let block = Arc::new((Mutex::new((false, false)), Condvar::new()));
        let build_block = block.clone();
        let recipe = Recipe::new(move || {
            if build_calls.fetch_add(1, Ordering::Relaxed) != 0 {
                let mut state = build_block.0.lock().unwrap();
                state.0 = true;
                build_block.1.notify_all();
                while !state.1 { state = build_block.1.wait(state).unwrap(); }
                replacement_raw
            } else {
                first_raw
            }
        });
        let standin = 0x7E57_C159usize;
        submit_recipe_keyed(standin, false, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let initial = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, initial());
        assert_eq!(try_evict_current(&pending.build), EvictResult::Evicted { generation: 1 });
        let q = Arc::new(BuildQueue::new());
        assert_eq!(request_rehydrate_on(pending.build.clone(), &q), RehydrateRequest::Queued);
        let worker_q = q.clone();
        let worker = std::thread::spawn(move || worker_q.try_run_one_between(Priority::Speculative, Priority::Speculative));
        {
            let mut state = block.0.lock().unwrap();
            while !state.0 { state = block.1.wait(state).unwrap(); }
        }
        assert_eq!(released(standin), 0);
        block.0.lock().unwrap().1 = true;
        block.1.notify_all();
        assert!(worker.join().unwrap());
        assert_eq!(pending.build.real.load(Ordering::Acquire), 0);
        assert_eq!(replacement.refs.load(Ordering::Relaxed), 0);
        drop(pending);
        assert_eq!(first.refs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn capture_preserves_lazy_for_bind_wait_and_identity_choices() {
        let standin = 0x7E57_C130usize;
        let real = 0x7E57_C131usize;
        let (go_tx, go_rx) = mpsc::channel();
        let (started_tx, started_rx) = mpsc::channel();
        submit_lazy(standin, true, Box::new(move || {
            started_tx.send(()).unwrap();
            go_rx.recv().unwrap();
            real
        }), 0);

        let (lazy, lazy_pin) = resolve_and_pin_mode(standin, ResolveMode::Lazy);
        assert_eq!(lazy, standin);
        assert!(lazy_pin.is_none());
        let (identity, identity_pin) = resolve_and_pin_if_enabled_identity(standin);
        assert_eq!(identity, standin);
        assert!(identity_pin.is_none());
        started_rx.recv().unwrap();
        let (result_tx, result_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(|| result_tx.send(resolve_and_pin_mode(standin, ResolveMode::ForBind).0).unwrap());
            assert!(result_rx.recv_timeout(std::time::Duration::from_millis(10)).is_err(), "for-bind compute resolution must still wait");
            go_tx.send(()).unwrap();
        });
        assert_eq!(result_rx.recv().unwrap(), real);
        assert_eq!(resolve_and_pin_mode(standin, ResolveMode::Wait).0, real);
        assert_eq!(released(standin), real);
    }

    #[test]
    fn capture_never_resolves_against_a_reused_pending_address() {
        let standin = 0x7E57_C138usize;
        let old_real = 0x7E57_C139usize;
        submit_lazy(standin, false, Box::new(move || old_real), 0);
        let old = entry(standin).unwrap();
        let job = old.build.job.lock().unwrap().take().unwrap();
        complete(&old.build, job());
        let removed = PENDING.remove(standin).unwrap();

        submit_lazy(standin, false, Box::new(|| 0x7E57_C13Ausize), 0);
        let (chosen, pin) = resolve_and_pin_pending(standin, &old, ResolveMode::Lazy);
        assert_eq!(chosen, old_real, "selection must use the retained Pending, not its reused map slot");
        assert!(pin.is_none());

        assert_eq!(released(standin), 0);
        drop(removed);
        drop(old);
    }

    #[test]
    fn bound_wait_capture_pins_the_exact_published_generation() {
        #[repr(C)]
        struct FakePso { vtable: *const usize, refs: AtomicUsize }
        unsafe extern "system" fn add_ref(this: *mut std::ffi::c_void) -> u32 {
            (*(this as *mut FakePso)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
        }
        unsafe extern "system" fn release(this: *mut std::ffi::c_void) -> u32 {
            (*(this as *mut FakePso)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
        }

        let vtable = [0usize, add_ref as usize, release as usize];
        let mut fake = FakePso { vtable: vtable.as_ptr(), refs: AtomicUsize::new(1) };
        let raw = &mut fake as *mut FakePso as usize;
        let standin = 0x7E57_C140usize;
        let recipe = Recipe::new(move || raw);
        submit_recipe_keyed(standin, true, recipe, false, 0, None);
        let pending = entry(standin).unwrap();
        let job = pending.build.job.lock().unwrap().take().unwrap();
        complete(&pending.build, job());

        let ((chosen, bound_unbuilt), pin) = resolve_wait_bound_and_pin(standin);
        let pin = pin.expect("published recipe generation must be captured");
        assert_eq!((chosen, bound_unbuilt, pin.raw(), pin.id()), (raw, false, raw, 1));
        assert_eq!(fake.refs.load(Ordering::Relaxed), 2, "generation pin shares the recipe's exact COM owner");

        drop(pin);
        assert_eq!(released(standin), raw);
        drop(pending);
        assert_eq!(unsafe { release(raw as *mut std::ffi::c_void) }, 0);
    }

    #[test]
    fn a_panicking_pipeline_job_completes_failed_and_wakes_its_waiter() {
        let build = Arc::new(Build::new(Box::new(|| panic!("simulated pipeline panic")), false));
        let waiting = build.clone();
        let (tx, rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let (done, _) = waiting
                .cv
                .wait_timeout_while(waiting.done.lock().unwrap(), std::time::Duration::from_secs(5), |done| !*done)
                .unwrap();
            tx.send((*done, waiting.real.load(Ordering::Acquire))).unwrap();
        });

        let job = build.job.lock().unwrap().take().unwrap();
        run_build(&build, job);
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(5)), Ok((true, 0)));
        waiter.join().unwrap();
        assert!(BUILDING.with(|building| building.borrow().is_none()), "panicking build left worker-local demand installed");

        let next = Arc::new(Build::new(Box::new(|| 0x5151), false));
        let job = next.job.lock().unwrap().take().unwrap();
        run_build(&next, job);
        assert_eq!(next.real.load(Ordering::Acquire), 0x5151, "the worker path did not survive a prior panic");

        let refused = Arc::new(Build::new(Box::new(|| 0x6161), false));
        fail_unrunnable_build(&refused);
        assert!(*refused.done.lock().unwrap(), "worker-start/submission failure did not complete the build");
        assert!(refused.job.lock().unwrap().is_none());
        assert_eq!(refused.real.load(Ordering::Acquire), 0);

        assert_eq!(
            build_pieces(vec![Box::new(|| 0x11), Box::new(|| panic!("simulated piece panic")), Box::new(|| 0x33)]),
            [0x11, 0, 0x33],
            "one failed split piece stranded or discarded its sibling results"
        );
    }

    #[test]
    fn pending_registry_routes_exact_lifetimes_across_independent_shards() {
        let registry = Arc::new(PendingRegistry::new());
        let keys: Vec<_> = (0..256).map(|i| 0x7E00_0000usize + i * 0x10).collect();
        let used: std::collections::HashSet<_> = keys.iter().map(|&key| pending_shard_index(key)).collect();
        assert!(used.len() >= PENDING_SHARDS / 2, "sequential aligned handles reached only {} shards", used.len());

        std::thread::scope(|scope| {
            for chunk in keys.chunks(32) {
                let registry = registry.clone();
                scope.spawn(move || {
                    for &key in chunk {
                        assert_eq!(registry.insert(key, key ^ 0x55AA), None);
                        assert_eq!(registry.get(key), Some(key ^ 0x55AA));
                        assert_eq!(registry.remove(key), Some(key ^ 0x55AA));
                        assert_eq!(registry.get(key), None);
                    }
                });
            }
        });

        let reused = 0x7E57_F000usize;
        assert_eq!(registry.insert(reused, 1), None);
        assert_eq!(registry.insert(reused, 2), Some(1), "replacement did not return the exact old address entry");
        assert_eq!(registry.remove(reused), Some(2));
        assert_eq!(registry.remove(reused), None);
    }

    #[test]
    fn a_busy_pending_shard_does_not_hold_an_unrelated_lookup() {
        let registry = Arc::new(PendingRegistry::new());
        let first = 0x7E10_0000usize;
        let second = (first + 0x10..).step_by(0x10).find(|&key| pending_shard_index(key) != pending_shard_index(first)).unwrap();
        registry.insert(second, 7usize);

        let held = registry.shards[pending_shard_index(first)].lock().unwrap();
        let (tx, rx) = mpsc::channel();
        let other = registry.clone();
        let thread = std::thread::spawn(move || tx.send(other.get(second)).unwrap());
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(2)), Ok(Some(7)));
        drop(held);
        thread.join().unwrap();
    }

    #[test]
    fn queued_count_state_is_once_only_and_completion_wins_races() {
        let state = AtomicU8::new(QUEUE_IDLE);
        assert!(mark_queued(&state));
        assert!(!mark_queued(&state));
        assert!(mark_queue_done(&state));
        assert!(!mark_queue_done(&state));
        assert!(!mark_queued(&state));

        let completed_first = AtomicU8::new(QUEUE_IDLE);
        assert!(!mark_queue_done(&completed_first));
        assert!(!mark_queued(&completed_first));

        let build = Build::new(Box::new(|| 0), false);
        build.note_queued();
        assert_eq!(build.queue_state.load(Ordering::Relaxed), QUEUE_IDLE, "auxiliary builds are not reported as title pipelines");
        build.tracked.store(true, Ordering::Release);
        build.note_queued();
        assert_eq!(build.queue_state.load(Ordering::Relaxed), QUEUE_ACTIVE);
        build.finish_queued();
        assert_eq!(build.queue_state.load(Ordering::Relaxed), QUEUE_DONE);
    }

    #[test]
    fn demanded_auxiliary_build_runs_on_a_worker() {
        let caller = std::thread::current().id();
        assert_eq!(build_foreground(Box::new(move || {
            assert_ne!(std::thread::current().id(), caller);
            0x7E57_C0DE
        })), 0x7E57_C0DE);
    }

    #[test]
    fn coalescing_counter_math_tracks_lifetime_current_and_peak() {
        let counters = CoalescingCounters::new();
        counters.unique();
        counters.hit();
        counters.unique();
        counters.finished(1);
        assert_eq!(
            counters.snapshot(),
            ComputeCoalescingStats {
                keyed_submissions: 3,
                coalescing_hits: 1,
                unique_keyed_builds: 2,
                current_in_flight: 1,
                peak_in_flight: 2,
            }
        );
        counters.finished(1);
        assert_eq!(counters.snapshot().current_in_flight, 0);
        assert_eq!(counters.snapshot().peak_in_flight, 2);
    }

    #[test]
    fn retention_age_counts_unique_reals_from_completion_or_last_use() {
        let stats = retention_stats_from(1_000, [
            (1, 0x101, 200, 300),
            (1, 0x101, 200, 300), // another stand-in sharing build 1
            (2, 0x202, 100, 995),
            (3, 0, 0, 0),        // failed or incomplete
            (4, 0x404, 0, 0),
        ]);
        assert_eq!(stats, RetentionStats {
            retained_reals: 3,
            unused_60_frames: 2,
            unused_600_frames: 2,
            oldest_unused_frames: 1_000,
        });
    }

    fn compute_key(tag: u8) -> super::super::real::compute_build_key::ComputeBuildKey {
        super::super::real::compute_build_key::ComputeBuildKey::capture(
            1,
            &windows::Win32::Graphics::Direct3D12::D3D12_COMPUTE_PIPELINE_STATE_DESC::default(),
            [tag].into(),
            false,
        )
    }

    fn submit_compute(standin: usize, tag: u8, job: Job) {
        submit_lazy_keyed(standin, true, job, 0, Some(compute_key(tag)));
    }

    #[test]
    fn coalesced_compute_owners_retain_the_winning_recipe() {
        let standins = [0x7E57_C091usize, 0x7E57_C092];
        let winner = Recipe::new(|| 0xC091);
        let duplicate = Recipe::new(|| panic!("duplicate recipe ran"));
        submit_recipe_keyed(standins[0], true, winner, false, 0, Some(compute_key(0x91)));
        submit_recipe_keyed(standins[1], true, duplicate, false, 0, Some(compute_key(0x91)));

        let (a, b) = (entry(standins[0]).unwrap(), entry(standins[1]).unwrap());
        assert!(Arc::ptr_eq(&a.build, &b.build));
        assert_eq!(a.build.recipe.as_ref().unwrap().run(), 0xC091);
        assert_eq!(released(standins[0]), 0);
        assert_eq!(released(standins[1]), 0);
    }

    #[test]
    fn concurrent_equal_compute_submissions_run_one_job() {
        let standins = [0x7E57_C101usize, 0x7E57_C102];
        let gate = Arc::new(std::sync::Barrier::new(3));
        let runs = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = standins.iter().map(|&standin| {
            let (gate, runs) = (gate.clone(), runs.clone());
            std::thread::spawn(move || {
                gate.wait();
                submit_compute(standin, 0xC1, Box::new(move || {
                    runs.fetch_add(1, Ordering::Relaxed);
                    0xC101
                }));
            })
        }).collect();
        gate.wait();
        for thread in threads { thread.join().unwrap() }

        let (a, b) = (entry(standins[0]).unwrap(), entry(standins[1]).unwrap());
        assert!(Arc::ptr_eq(&a.build, &b.build));
        assert_eq!(resolve_wait_for(standins[0], None), 0xC101);
        assert_eq!(resolve_wait_for(standins[1], None), 0xC101);
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!(released(standins[0]), 0, "the shared COM owner stays with the other stand-in");
        assert_eq!(released(standins[1]), 0xC101, "the last stand-in releases the one owned reference");
    }

    #[test]
    fn bound_state_follows_the_choice_from_a_shared_build() {
        let standins = [0x7E57_C111usize, 0x7E57_C112];
        let drivers = [0x7E57_D111usize, 0x7E57_D112];
        let (started_tx, started_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel::<()>();
        submit_compute(standins[0], 0xE6, Box::new(move || {
            started_tx.send(()).unwrap();
            finish_rx.recv().unwrap();
            0xC111
        }));
        submit_compute(standins[1], 0xE6, Box::new(|| panic!("duplicate shared build ran")));
        for (&standin, &driver) in standins.iter().zip(&drivers) {
            shared(standin, driver);
            image_only(standin);
        }

        let before = standins.map(|standin| resolve_wait_bound_for(standin, None));
        assert_eq!(before, [(drivers[0], true), (drivers[1], true)]);
        started_rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker did not start the build");
        finish_tx.send(()).unwrap();
        let start = std::time::Instant::now();
        while is_pending(standins[0]) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(standins[0]), "shared build did not finish");
        for ((&standin, &driver), &(bound, _)) in standins.iter().zip(&drivers).zip(&before) {
            let pending = entry(standin).unwrap();
            assert!(pending.bound_unbuilt(standin, bound), "the earlier driver stand-in stays unbuilt");
            assert_eq!(resolve_wait_bound_for(standin, None), (0xC111, false));
            assert_ne!(driver, 0xC111);
        }
        assert_eq!(released(standins[0]), 0);
        assert_eq!(released(standins[1]), 0xC111);
    }

    #[test]
    fn a_failed_builds_permanent_stand_in_is_not_unbuilt() {
        let standin = 0x7E57_C121usize;
        let driver = 0x7E57_D121usize;
        submit_lazy(standin, true, Box::new(|| 0), 0);
        shared(standin, driver);
        assert_eq!(resolve_wait_bound_for(standin, None), (driver, false));
        assert_eq!(resolve_wait_bound_for(0x7E57_C122, None), (0x7E57_C122, false));
        assert_eq!(released(standin), 0);
    }

    #[test]
    fn no_budget_wait_runs_a_shared_build_on_a_wanted_worker() {
        let standins = [0x7E57_C701usize, 0x7E57_C702];
        let (started_tx, started_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel::<()>();
        submit_compute(standins[0], 0xC7, Box::new(move || {
            started_tx.send((std::thread::current().name().map(str::to_owned), build_is_wanted(), wanted_here()())).unwrap();
            let _ = finish_rx.recv();
            0xC701
        }));
        submit_compute(standins[1], 0xC7, Box::new(|| panic!("duplicate shared build ran")));
        let (a, b) = (entry(standins[0]).unwrap(), entry(standins[1]).unwrap());
        assert!(Arc::ptr_eq(&a.build, &b.build), "equal compute requests share one Build");

        let (result_tx, result_rx) = mpsc::channel();
        let waiters: Vec<_> = standins.iter().map(|&standin| {
            let result_tx = result_tx.clone();
            std::thread::spawn(move || result_tx.send((standin, resolve_wait_for(standin, None))).unwrap())
        }).collect();
        drop(result_tx);
        let (thread, wanted, wanted_probe) = started_rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker did not start the build");
        assert!(thread.as_deref().is_some_and(|name| name.starts_with("xcl-pso-")), "build ran on caller {thread:?}");
        assert!(wanted && wanted_probe, "the worker did not observe foreground demand");
        assert!(matches!(result_rx.try_recv(), Err(mpsc::TryRecvError::Empty)), "a caller returned before the build finished");

        finish_tx.send(()).unwrap();
        let mut results: Vec<_> = result_rx.iter().collect();
        results.sort_unstable();
        for waiter in waiters { waiter.join().unwrap(); }
        assert_eq!(results, [(standins[0], 0xC701), (standins[1], 0xC701)]);
        assert_eq!(released(standins[0]), 0);
        assert_eq!(released(standins[1]), 0xC701);
    }

    #[test]
    fn unequal_compute_keys_keep_separate_builds() {
        let standins = [0x7E57_C201usize, 0x7E57_C202];
        submit_compute(standins[0], 0xC2, Box::new(|| 0));
        submit_compute(standins[1], 0xC3, Box::new(|| 0));
        assert!(!Arc::ptr_eq(&entry(standins[0]).unwrap().build, &entry(standins[1]).unwrap().build));
        assert_eq!((released(standins[0]), released(standins[1])), (0, 0));
    }

    #[test]
    fn failed_coalesced_build_wakes_every_waiter() {
        let standins = [0x7E57_C301usize, 0x7E57_C302];
        let runs = Arc::new(AtomicUsize::new(0));
        for &standin in &standins {
            let runs = runs.clone();
            submit_compute(standin, 0xC4, Box::new(move || {
                runs.fetch_add(1, Ordering::Relaxed);
                0
            }));
        }
        let threads: Vec<_> = standins.iter().map(|&standin| {
            std::thread::spawn(move || resolve_wait_for(standin, None))
        }).collect();
        for (thread, standin) in threads.into_iter().zip(standins) {
            assert_eq!(thread.join().unwrap(), standin);
        }
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!((released(standins[0]), released(standins[1])), (0, 0));
    }

    #[test]
    fn coalesced_preload_is_promoted_once_for_both_pending_owners() {
        let standins = [0x7E57_C401usize, 0x7E57_C402];
        submit_compute(standins[0], 0xC5, Box::new(|| 0xC401));
        submit_compute(standins[1], 0xC5, Box::new(|| panic!("duplicate job ran")));
        let build = entry(standins[0]).unwrap().build.clone();
        assert!(Arc::ptr_eq(&build, &entry(standins[1]).unwrap().build));
        let q = BuildQueue::new();
        enqueue_on(&q, &build, Priority::Preload, true);
        entry(standins[1]).unwrap().build.promote(Priority::Foreground);
        assert!(!q.try_run_one_between(Priority::Speculative, Priority::Preload));
        assert!(q.try_run_one_between(Priority::BoundSoon, Priority::Foreground));
        assert_eq!(build.real.load(Ordering::Acquire), 0xC401);
        assert_eq!(released(standins[0]), 0);
        assert_eq!(released(standins[1]), 0xC401);
    }

    #[test]
    fn submit_demand_is_shared_by_coalesced_stand_ins() {
        let standins = [0x7E57_C411usize, 0x7E57_C412];
        let (wanted_tx, wanted_rx) = mpsc::channel();
        let (go_tx, go_rx) = mpsc::channel();
        submit_compute(standins[0], 0xD7, Box::new(move || {
            wanted_tx.send(build_is_wanted()).unwrap();
            go_rx.recv().unwrap();
            0xC411
        }));
        submit_compute(standins[1], 0xD7, Box::new(|| panic!("duplicate job ran")));
        let build = entry(standins[0]).unwrap().build.clone();
        assert!(Arc::ptr_eq(&build, &entry(standins[1]).unwrap().build));

        need_by_submit(standins[1]);
        assert!(build.wanted.load(Ordering::Acquire));
        assert!(wanted_rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker builds it"));
        go_tx.send(()).unwrap();
        assert_eq!(resolve_wait_for(standins[0], None), 0xC411);
        assert_eq!(resolve_wait_for(standins[1], None), 0xC411);
        assert_eq!(released(standins[0]), 0);
        assert_eq!(released(standins[1]), 0xC411);
    }

    #[test]
    fn releasing_one_waiter_does_not_orphan_shared_build() {
        let standins = [0x7E57_C501usize, 0x7E57_C502];
        let runs = Arc::new(AtomicUsize::new(0));
        let count = runs.clone();
        submit_compute(standins[0], 0xC6, Box::new(move || {
            count.fetch_add(1, Ordering::Relaxed);
            0xC501
        }));
        submit_compute(standins[1], 0xC6, Box::new(|| panic!("duplicate job ran")));
        let build = entry(standins[1]).unwrap().build.clone();
        assert_eq!(released(standins[0]), 0);
        assert!(!build.orphan.load(Ordering::Acquire));
        assert!(build.job.lock().unwrap().is_some());
        assert_eq!(resolve_wait_for(standins[1], None), 0xC501);
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!(released(standins[1]), 0xC501);
    }

    #[test]
    fn build_completion_reaches_all_owners() {
        let build = Arc::new(Build::new(Box::new(|| 0), false));
        let waiter_build = build.clone();
        assert!(Arc::ptr_eq(&build, &waiter_build));
        let waiter = std::thread::spawn(move || {
            let mut done = waiter_build.done.lock().unwrap();
            while !*done {
                done = waiter_build.cv.wait(done).unwrap();
            }
            waiter_build.real.load(Ordering::Acquire)
        });

        complete(&build, 0xB017_D001);
        assert_eq!(waiter.join().unwrap(), 0xB017_D001);
        assert_eq!(build.real.swap(0, Ordering::AcqRel), 0xB017_D001);
    }

    #[test]
    fn wanted_preload_moves_to_foreground_and_builds_once() {
        let q = BuildQueue::new();
        let runs = Arc::new(AtomicUsize::new(0));
        let count = runs.clone();
        let build = Arc::new(Build::new(Box::new(move || {
            count.fetch_add(1, Ordering::Relaxed);
            0xB017_D002
        }), true));
        enqueue_on(&q, &build, Priority::Preload, true);

        build.want();
        assert!(!q.try_run_one_between(Priority::Speculative, Priority::Preload));
        assert!(q.try_run_one_between(Priority::BoundSoon, Priority::Foreground));
        assert!(!q.try_run_one());
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!(build.real.swap(0, Ordering::AcqRel), 0xB017_D002);
    }

    #[test]
    fn caller_can_steal_a_queued_build_without_running_it_twice() {
        let q = BuildQueue::new();
        let runs = Arc::new(AtomicUsize::new(0));
        let count = runs.clone();
        let build = Arc::new(Build::new(Box::new(move || {
            count.fetch_add(1, Ordering::Relaxed);
            0xB017_D003
        }), true));
        enqueue_on(&q, &build, Priority::BoundSoon, false);

        let job = build.job.lock().unwrap().take().unwrap();
        build.cancel_queued();
        complete(&build, job());
        assert!(!q.try_run_one());
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!(build.real.swap(0, Ordering::AcqRel), 0xB017_D003);
    }

    /// A PC call that has kept one of the title's threads for long holds the title, a short one
    /// does not, and the hold ends with the call.
    #[test]
    fn a_long_pc_call_holds_the_title() {
        let mut calls = PcCalls::new();
        let began = std::time::Instant::now();
        calls.begin(1, 0, began);
        calls.begin(2, 0, began + 2 * PC_CALL_HOLDS_AFTER);
        assert!(!calls.holds(0, began), "a call that has just begun holds the title");
        let later = began + 2 * PC_CALL_HOLDS_AFTER;
        assert!(calls.holds(0, later), "a PC call has lasted {:?}, but the title is not held", 2 * PC_CALL_HOLDS_AFTER);
        calls.end(1, 0, later);
        calls.end(2, 0, later);
        assert!(!calls.holds(0, later), "the calls are over, the title is still held");
    }

    /// Short PC calls add up between two frames: none of them is long, but after many in a row
    /// the one going on holds the title. The title's own time between them is not held, and a
    /// frame shown ends it.
    #[test]
    fn short_pc_calls_between_two_frames_hold_the_title() {
        let mut calls = PcCalls::new();
        let step = PC_CALL_HOLDS_AFTER / 10;
        let mut now = std::time::Instant::now();
        for id in 0..20 {
            calls.begin(id, 7, now);
            assert!(id >= 9 || !calls.holds(7, now + step), "{id} short calls hold the title already");
            now += step;
            calls.end(id, 7, now);
        }
        assert!(!calls.holds(7, now), "no PC call is going on, the title is held");
        calls.begin(20, 7, now);
        assert!(calls.holds(7, now), "twenty short PC calls since the last frame and one going on, but the title is not held");
        assert!(!calls.holds(8, now), "a frame was shown, the calls before it still hold the title");
        calls.end(20, 8, now);
    }

    /// While a bind waits for a pipeline still being built, the title is held (`title_held`):
    /// `kernelx::Sleep` does not count that time.
    #[test]
    fn a_bind_waiting_for_a_build_holds_the_title() {
        let standin = 0x7E57_B002usize;
        let (started, building) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        submit_lazy(standin, true, Box::new(move || {
            let _ = started.send(());
            let _ = finish.recv();
            0x5252
        }), 0);
        let bind = std::thread::spawn(move || resolve_wait_for(standin, None));
        building.recv_timeout(std::time::Duration::from_secs(10)).expect("the pipeline is being built");
        // (built in the bind, or by a worker with the bind about to wait for it)
        let t = std::time::Instant::now();
        while !title_held() && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(title_held(), "a bind waits for a build, but the title is not held");
        go.send(()).unwrap();
        assert_eq!(bind.join().unwrap(), 0x5252);
        let _ = released(standin);
    }

    /// A stand-in that is no driver object (`shared`) never reaches the driver: binds get the
    /// driver pipeline shared with other stand-ins until the real one is built, and after a build
    /// that failed.
    #[test]
    fn a_bind_of_a_shared_stand_in_gets_its_driver_pipeline() {
        let (fails, builds, driver) = (0x7E57_B011usize, 0x7E57_B012usize, 0x7E57_B013usize);
        submit_lazy(fails, false, Box::new(|| 0), 0);
        submit_lazy(builds, false, Box::new(|| 0x1234), 0);
        for s in [fails, builds] {
            shared(s, driver);
            assert_eq!(entry(s).unwrap().stand(s), driver);
        }
        assert_eq!(resolve_wait_for(fails, None), driver, "its build failed");
        assert_eq!(resolve(fails), driver);
        assert_eq!(resolve_wait_for(builds, None), 0x1234);
        assert_eq!(resolve_for_bind(builds), 0x1234);
        // (0x1234 is no pipeline: taken, not released)
        assert_eq!((released(fails), released(builds)), (0, 0x1234));
    }

    /// A bind does not wait for a compute pipeline that writes only images: it gets the stand-in
    /// at once, with no budget, the build goes to the workers
    /// unwanted, and once built the bind gets the real one. Without `image_only` the same bind
    /// waits until the pipeline is built.
    #[test]
    fn a_bind_does_not_wait_for_a_pipeline_that_writes_only_images() {
        let (spared, waited) = (0x7E57_B0A1usize, 0x7E57_B0A2usize);
        let (go, finish) = mpsc::channel::<()>();
        let (tx, rx) = mpsc::channel();
        submit_lazy(spared, true, Box::new(move || {
            let _ = tx.send(wanted_here()());
            let _ = finish.recv();
            0x5A5A
        }), 0);
        image_only(spared);
        // (on a thread: a bind that waits would wait for `go`, below)
        let (bound_tx, bound) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = bound_tx.send((resolve_wait_for(spared, None), resolve_wait_for(spared, None)));
        });
        let binds = bound.recv_timeout(std::time::Duration::from_secs(5));
        if binds.is_err() {
            let _ = go.send(());
        }
        assert_eq!(binds.expect("the binds waited for the build"), (spared, spared), "still being built: the stand-in");
        assert!(is_pending(spared));
        assert!(!rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker builds it"), "nobody waits: the build is not wanted");
        go.send(()).unwrap();
        let start = std::time::Instant::now();
        while is_pending(spared) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(spared, None), 0x5A5A, "built by a worker");
        // the same build without the mark: the bind waits for it
        submit_lazy(waited, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(100));
            0x5B5B
        }), 0);
        assert_eq!(resolve_wait_for(waited, None), 0x5B5B);
        let _ = released(spared);
        let _ = released(waited);
    }

    #[test]
    fn a_pipeline_a_submit_needs_is_built_before_the_queue() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let backlog = 8 * crate::gpu::recomp::dxc::compile_threads();
        let standins: Vec<usize> = (0..=backlog).map(|i| 0x7E57_E000usize + i).collect();
        for &standin in &standins {
            let order = order.clone();
            submit(standin, true, Box::new(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                order.lock().unwrap().push(standin);
                0x6161
            }));
        }
        let last = *standins.last().unwrap();
        need_by_submit(last);
        let start = std::time::Instant::now();
        while standins.iter().any(|&s| is_pending(s)) && start.elapsed() < std::time::Duration::from_secs(30) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let order = order.lock().unwrap();
        assert_eq!(order.len(), standins.len());
        let at = order.iter().position(|&s| s == last).unwrap();
        assert!(at < order.len() / 2, "made last and needed by a submit, it was built {at} of {}", order.len());
        for &standin in &standins {
            let _ = released(standin);
        }
    }

    #[test]
    fn a_build_a_submit_needs_is_wanted_from_record_time() {
        let standin = 0x7E57_B006usize;
        let (started_tx, started) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            started_tx.send(()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if build_is_wanted() { 0x5757 } else { 0 }
        }));
        started.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker did not start the build");
        need_by_submit(standin);
        assert_eq!(resolve_wait_for(standin, None), 0x5757, "record-time demand did not reach the running build");
        assert_eq!(released(standin), 0x5757);
    }

    #[test]
    fn binding_a_completed_compute_pipeline_neither_relooks_it_up_nor_requests_work() {
        let standin = 0x7E57_B00Ausize;
        submit(standin, true, Box::new(|| 0x5455));
        let p = entry(standin).unwrap();
        drop(p.build.cv.wait_while(p.build.done.lock().unwrap(), |done| !*done).unwrap());

        assert_eq!(resolve_for_bind(standin), 0x5455);
        assert_eq!(resolve_wait_for(standin, None), 0x5455);
        assert!(!p.build.wanted.load(Ordering::Acquire));
        assert_eq!(released(standin), 0x5455);
    }

    /// A bind waits for a pipeline still being built no longer than its budget and goes on with
    /// the stand-in; later binds do not wait for it again; once built it is the real one.
    #[test]
    fn a_bind_waits_for_a_slow_pipeline_no_longer_than_its_budget() {
        let standin = 0x7E57_B001usize;
        submit(standin, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(600));
            0x5151
        }));
        let budget = Some(std::time::Duration::from_millis(30));
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin, "not built within the budget: the stand-in");
        assert!(t.elapsed() < std::time::Duration::from_millis(400), "waited {:?}", t.elapsed());
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin);
        assert!(t.elapsed() < std::time::Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(standin, budget), 0x5151, "built by a worker");
        let _ = released(standin);
    }

    /// DXC admission probes demand while holding its own queue lock: submit demand is published
    /// through the atomic wanted signal, so the probe cannot take this build's demand lock.
    #[test]
    fn submit_demand_wanted_probe_is_lock_free() {
        let build = Arc::new(Build::new(Box::new(|| 0), false));
        let before = BUILDING.with(|building| building.replace(Some(build.clone())));
        let _restore = RestoreBuilding(before);
        let wanted = wanted_here();
        build.demand.lock().unwrap().by_submit = true;
        build.want();
        assert!(wanted(), "submit demand reaches DXC and driver admission");

        let demand = build.demand.lock().unwrap();
        let (tx, rx) = mpsc::channel();
        let thread = std::thread::spawn(move || tx.send(wanted()).unwrap());
        assert!(rx.recv_timeout(std::time::Duration::from_secs(2)).expect("wanted probe took the demand lock"));
        drop(demand);
        thread.join().unwrap();
    }

    /// A compile a worker's build starts is wanted once a bind waits for the build – asked from
    /// another thread, as the DXC workers do; a thread that builds nothing (no title runs here)
    /// wants nothing.
    #[test]
    fn a_waiting_bind_makes_the_builds_compiles_wanted() {
        let standin = 0x7E57_B005usize;
        let (tx, rx) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            tx.send(wanted_here()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            0x5454
        }));
        let wanted = rx.recv().unwrap();
        assert!(!wanted(), "nobody waits yet");
        assert_eq!(resolve_wait_for(standin, None), 0x5454);
        assert!(wanted());
        assert!(!wanted_here()());
        let _ = released(standin);
    }

    /// A worker's build hears that a bind waits for it (and only its own build does).
    #[test]
    fn a_waiting_bind_makes_the_build_wanted() {
        let standin = 0x7E57_B004usize;
        let (started_tx, started) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            let unwanted_first = !build_is_wanted();
            started_tx.send(()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if unwanted_first && build_is_wanted() { 0x5353 } else { 0 }
        }));
        started.recv().unwrap();
        assert!(!build_is_wanted(), "this thread builds nothing");
        assert_eq!(resolve_wait_for(standin, None), 0x5353, "the bind's wait reached the build");
        let _ = released(standin);
    }

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    #[test]
    fn used_file_ranks_follow_unique_valid_first_occurrences() {
        let ranks = used::parse_ranked("10\nnot a key\n10\n0\n 20 \ninvalid\n30\n20\n");
        assert_eq!(ranks.len(), 3);
        assert_eq!(ranks.get(&0x10), Some(&0));
        assert_eq!(ranks.get(&0x20), Some(&1));
        assert_eq!(ranks.get(&0x30), Some(&2));
    }

    /// A lazy pipeline of shaders an earlier run used is built without a bind; another one is
    /// built only at its first use, and that use is kept for the next run.
    #[test]
    fn a_pipeline_an_earlier_run_used_is_built_ahead_of_its_first_use() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| 0x4321), known_key);
        submit_lazy(unknown, false, Box::new(|| 0x4322), unknown_key);
        let start = std::time::Instant::now();
        while is_pending(known) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(known), 0x4321);
        assert!(is_pending(unknown), "nobody used it and no earlier run did");
        assert!(!used::noted(unknown_key));
        assert_eq!(resolve_wait(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }

    static WORKER_SATURATION_LOCK: Mutex<()> = Mutex::new(());

    /// Normal workers parked at the huge-driver-compile gate leave their priority band live, and
    /// the bounded replacement workers retire after the gate opens.
    #[test]
    fn parked_normal_workers_leave_the_queue_to_bounded_helpers() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::time::{Duration, Instant};

        struct OpenGate(Arc<(Mutex<bool>, Condvar)>);
        impl Drop for OpenGate {
            fn drop(&mut self) {
                *self.0 .0.lock().unwrap() = true;
                self.0 .1.notify_all();
            }
        }

        let outside = parked_for_big_compile();
        assert!(!outside.0, "a title/test thread must not start a replacement");
        drop(outside);

        start_normal_workers();
        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let open = OpenGate(gate.clone());
        for _ in 0..n {
            let (tx, gate) = (started_tx.clone(), gate.clone());
            queue().submit(Priority::Foreground, Box::new(move || {
                let _parked = parked_for_big_compile();
                tx.send(()).unwrap();
                let mut open = gate.0.lock().unwrap();
                while !*open {
                    open = gate.1.wait(open).unwrap();
                }
            })).unwrap();
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("every blocker parked");
        }
        assert!(NORMAL_HELPERS.load(Ordering::Acquire) <= n, "replacement count is bounded by the normal pool");
        let (ran_tx, ran_rx) = mpsc::channel();
        queue().submit(Priority::Foreground, Box::new(move || ran_tx.send(()).unwrap())).unwrap();
        ran_rx.recv_timeout(Duration::from_secs(10)).expect("a replacement served the queue");
        drop(open);

        let deadline = Instant::now() + Duration::from_secs(10);
        while (PARKED_NORMAL.load(Ordering::Acquire) != 0 || NORMAL_HELPERS.load(Ordering::Acquire) != 0) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(PARKED_NORMAL.load(Ordering::Acquire), 0);
        assert_eq!(NORMAL_HELPERS.load(Ordering::Acquire), 0, "replacement workers retired");
    }

    /// A queued normal worker job whose stand-in was released before a worker starts it is
    /// skipped as orphaned instead of entering the compiler.
    #[test]
    fn a_released_queued_pipeline_is_not_built_by_normal_workers() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let mut blockers = Vec::with_capacity(n);
        for k in 0..n {
            let standin = 0x7E57_9720usize + k;
            let started_tx = started_tx.clone();
            let gate = gate.clone();
            submit(standin, false, Box::new(move || {
                let _ = started_tx.send(());
                let mut open = gate.0.lock().unwrap();
                while !*open {
                    open = gate.1.wait(open).unwrap();
                }
                0x9720 + k
            }));
            blockers.push(standin);
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("normal workers saturated");
        }

        let target = 0x7E57_97FFusize;
        let ran = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        let ran_job = ran.clone();
        let drop_flag = DropFlag(dropped.clone());
        submit(target, false, Box::new(move || {
            let _keep = &drop_flag;
            ran_job.store(true, Ordering::Release);
            0x9721
        }));
        assert_eq!(released(target), 0);

        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();

        let deadline = Instant::now() + Duration::from_secs(10);
        while !dropped.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(dropped.load(Ordering::Acquire), "queued orphan job was never drained");
        assert!(!ran.load(Ordering::Acquire), "released job reached the compiler");

        for standin in blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }
    }

    #[test]
    fn pc_call_crossing_frame_boundary_does_not_charge_prior_frame_time() {
        let mut calls = PcCalls::new();
        let t0 = std::time::Instant::now();
        let ms = std::time::Duration::from_millis;
        calls.begin(1, 7, t0);
        // Frame 8 shown after 290 ms; call 1 ends 10 ms later -> only 10 ms counts on frame 8.
        let t_frame8 = t0 + PC_CALL_HOLDS_AFTER + ms(40);
        assert!(!calls.holds(8, t_frame8), "call from frame 7 must not immediately hold frame 8");
        calls.end(1, 8, t_frame8 + ms(10));
        let t_next = t_frame8 + ms(11);
        calls.begin(2, 8, t_next);
        assert!(!calls.holds(8, t_next), "pre-frame duration leaked into frame 8's budget");
        // A call genuinely stuck >= PC_CALL_HOLDS_AFTER after frame 8 still holds the title.
        assert!(!calls.holds(8, t_next + PC_CALL_HOLDS_AFTER - ms(20)));
        assert!(calls.holds(8, t_next + PC_CALL_HOLDS_AFTER));
        calls.end(2, 8, t_next + PC_CALL_HOLDS_AFTER);

        // Also when frame 9 is first noticed inside end() itself, pre-frame time is clamped to 0.
        let t_f9 = t_next + PC_CALL_HOLDS_AFTER + ms(10);
        calls.begin(3, 8, t_f9);
        let t_f9_end = t_f9 + PC_CALL_HOLDS_AFTER + ms(50);
        calls.end(3, 9, t_f9_end);
        calls.begin(4, 9, t_f9_end + ms(1));
        assert!(!calls.holds(9, t_f9_end + ms(1)), "end() on new frame must not charge prior-frame duration");
        calls.end(4, 9, t_f9_end + ms(1) + PC_CALL_HOLDS_AFTER);

        // A stale lower frame number must never rewind self.frame or wipe frame 9's spent budget.
        let t_stale = t_f9_end + PC_CALL_HOLDS_AFTER + ms(10);
        calls.begin(5, 8, t_stale);
        assert!(calls.holds(9, t_stale), "stale lower frame rewound frame 9 or cleared its spent budget");
        calls.end(5, 9, t_stale);
    }

    #[test]
    fn released_queued_pipeline_drops_job_immediately_before_workers_drain() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        struct OpenGate(Arc<(Mutex<bool>, Condvar)>);
        impl Drop for OpenGate {
            fn drop(&mut self) {
                *self.0 .0.lock().unwrap() = true;
                self.0 .1.notify_all();
            }
        }

        // Saturate normal workers and verify released() drops the queued job immediately.
        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let open_normal = OpenGate(gate.clone());
        let mut blockers = Vec::with_capacity(n);
        for k in 0..n {
            let standin = 0x7E57_A100usize + k;
            let (tx, g) = (started_tx.clone(), gate.clone());
            submit(standin, false, Box::new(move || {
                let _ = tx.send(());
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                0xA100 + k
            }));
            blockers.push(standin);
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("normal workers saturated");
        }
        let target_n = 0x7E57_A1FFusize;
        let (ran_n, dropped_n) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (ran_job, flag_n) = (ran_n.clone(), DropFlag(dropped_n.clone()));
        submit(target_n, false, Box::new(move || {
            let _keep = &flag_n;
            ran_job.store(true, Ordering::Release);
            0xA1FF
        }));
        assert_eq!(released(target_n), 0);
        assert!(dropped_n.load(Ordering::Acquire), "normal queued job was not dropped immediately on released()");
        drop(open_normal);
        assert!(!ran_n.load(Ordering::Acquire));
        for standin in blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }

        // Saturate preload workers and verify released() drops the preloaded job immediately.
        let m = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        let (p_tx, p_rx) = mpsc::channel();
        let p_gate = Arc::new((Mutex::new(false), Condvar::new()));
        let open_pre = OpenGate(p_gate.clone());
        let mut p_blockers = Vec::with_capacity(m);
        for k in 0..m {
            let standin = 0x7E57_A200usize + k;
            let key = 0x5EED_A200u64 + k as u64;
            used::earlier_run_used(key);
            let (tx, g) = (p_tx.clone(), p_gate.clone());
            submit_lazy(standin, false, Box::new(move || {
                let _ = tx.send(());
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                0xA200 + k
            }), key);
            p_blockers.push(standin);
        }
        for _ in 0..m {
            p_rx.recv_timeout(Duration::from_secs(10)).expect("preload workers saturated");
        }
        let target_p = 0x7E57_A2FFusize;
        let key_p = 0x5EED_A2FFu64;
        used::earlier_run_used(key_p);
        let (ran_p, dropped_p) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (ran_job, flag_p) = (ran_p.clone(), DropFlag(dropped_p.clone()));
        submit_lazy(target_p, false, Box::new(move || {
            let _keep = &flag_p;
            ran_job.store(true, Ordering::Release);
            0xA2FF
        }), key_p);
        assert_eq!(released(target_p), 0);
        assert!(dropped_p.load(Ordering::Acquire), "preload queued job was not dropped immediately on released()");
        drop(open_pre);
        assert!(!ran_p.load(Ordering::Acquire));
        for standin in p_blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.get(pso)
}

impl Pending {
    /// What a bind of the stand-in `pso` gives the driver while the pipeline is not built.
    fn stand(&self, pso: usize) -> usize {
        match self.driver.load(Ordering::Acquire) {
            0 => pso,
            d => d,
        }
    }

    /// `bound` was this pending pipeline's stand-in before its build finished successfully.
    /// A failed build keeps its stand-in permanently, so that is a finished pipeline choice.
    fn bound_unbuilt(&self, pso: usize, bound: usize) -> bool {
        let done = self.build.done.lock().unwrap();
        bound == self.stand(pso) && !(*done && self.build.real.load(Ordering::Acquire) == 0)
    }
}

/// Stand-in `standin` is no driver object (`standin::shell`): binds get `driver` in its place.
/// Told before the title has the stand-in.
pub fn shared(standin: usize, driver: usize) {
    if let Some(p) = entry(standin) {
        p.driver.store(driver, Ordering::Release);
    }
}

/// The pipeline to use for `pso` now: the real one once built, else the stand-in's driver
/// pipeline (`pso` itself when it is one).
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    resolve_pending(&p, pso)
}

fn resolve_pending(p: &Pending, pso: usize) -> usize {
    p.note_use();
    match p.build.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers
            let stand = p.stand(pso);
            schedule(&p.build, Priority::BoundSoon, false, None);
            p.build.promote(Priority::Foreground);
            stand
        }
        r => r,
    }
}

/// `resolve`, waiting for a pipeline still being built on a foreground worker.
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_for(pso, wait_budget())
}

/// The pipeline to bind and whether that exact choice was a stand-in for an unfinished build.
/// The answer stays true if a successful build ends after this bind chose its stand-in.
pub fn resolve_wait_bound(pso: usize) -> (usize, bool) {
    resolve_wait_bound_for(pso, wait_budget())
}

/// `resolve_wait`, waiting at most `budget` (None: until the pipeline is built).
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    resolve_wait_pending(&p, pso, budget)
}

fn resolve_wait_bound_for(pso: usize, budget: Option<std::time::Duration>) -> (usize, bool) {
    let Some(p) = entry(pso) else { return (pso, false) };
    p.note_use();
    let bound = resolve_wait_pending(&p, pso, budget);
    (bound, p.bound_unbuilt(pso, bound))
}

fn resolve_wait_pending(p: &Arc<Pending>, pso: usize, budget: Option<std::time::Duration>) -> usize {
    if let real @ 1.. = p.build.real.load(Ordering::Acquire) {
        return real;
    }
    let bound = if p.spared.load(Ordering::Acquire) {
        resolve_spared(p, pso)
    } else {
        // (the title's thread is held from here until the pipeline is built or the budget is over)
        let waiting = !*p.build.done.lock().unwrap();
        let _held = waiting.then(Held::new);
        let _stall = waiting.then(|| crate::stall::span(crate::stall::StallKind::PipelineWait)).flatten();
        if let Some(budget) = budget {
            resolve_within(p, pso, budget)
        } else {
            // Mark it wanted before an unsent build is queued: the worker and its DXC/driver
            // admission see demand from their first instruction. An already queued build is
            // promoted by `want`.
            p.build.want();
            schedule(&p.build, Priority::Foreground, false, None);
            let mut done = p.build.done.lock().unwrap();
            while !*done {
                done = p.build.cv.wait(done).unwrap();
            }
            match p.build.real.load(Ordering::Acquire) {
                0 => p.stand(pso),
                r => r,
            }
        }
    };
    bound
}

/// How long a bind waits for a pipeline still being built (`XCL_PSO_WAIT_MS`; unset or 0: until
/// it is built). A retail game's foreground worker can spend over 30 s in the driver on the
/// biggest translated programs; the waiting title thread remains held for that time.
fn wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0).map(std::time::Duration::from_millis))
}

/// Whether binds have an explicit time bound. A dispatch is then recorded immediately instead of
/// deferred to the submit, where it would only wait for the same bound.
pub(super) fn waits_bounded() -> bool {
    wait_budget().is_some()
}

/// `resolve_wait` for at most `budget`: the build goes to the workers (not this thread), and a
/// pipeline not built by then is `late` – it and later binds of it get the stand-in (their
/// dispatches are skipped) until a worker has built it.
fn resolve_within(p: &Arc<Pending>, pso: usize, budget: std::time::Duration) -> usize {
    let built = |p: &Pending| match p.build.real.load(Ordering::Acquire) {
        0 => p.stand(pso),
        r => r,
    };
    if *p.build.done.lock().unwrap() {
        return built(p);
    }
    schedule(&p.build, Priority::BoundSoon, false, None);
    if p.late.load(Ordering::Acquire) {
        return p.stand(pso);
    }
    p.build.want();
    let (done, _) = p.build.cv.wait_timeout_while(p.build.done.lock().unwrap(), budget, |d| !*d).unwrap();
    if *done {
        return built(p);
    }
    drop(done);
    p.late.store(true, Ordering::Release);
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: a pipeline still being built after {} ms: its binds go on without it until it is built ({} so far)", budget.as_millis(), n + 1);
    }
    p.stand(pso)
}

/// `XCL_PSO_IMAGE_SKIP=1`: a bind does not wait for a big compute program that writes nothing
/// but images (`gpu::effects::big_and_image_only`); its dispatches are left out until a worker
/// has built it. Off by default: what such a program draws is missing for that long.
///
/// A retail title's world start binds one ray-tracing resolve program per material (73 – 118
/// KiB of console code, about 1 MB of DXIL each): an RDNA2 driver takes 22 – 172 s for each,
/// one bind at a time, and the title stands still for all of them. `XCL_PSO_WAIT_MS` goes on without every late
/// program, also the culling passes whose counts an `ExecuteIndirect` reads next – a GPU hang.
pub fn image_skip() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("XCL_PSO_IMAGE_SKIP").is_ok_and(|v| v.trim() == "1"))
}

/// The compute pipeline behind stand-in `standin` writes nothing but images: binds go on
/// without it while it is built (`resolve_spared`). Told before the title has the stand-in.
pub fn image_only(standin: usize) {
    if let Some(p) = entry(standin) {
        p.spared.store(true, Ordering::Release);
    }
}

/// `resolve_wait` for a pipeline binds do not wait for (`image_only`): the real one once built,
/// else the stand-in, with the build sent to the workers. Nobody waits, so the build is not
/// `wanted`: a big one takes its turn among the driver's compiles and waits for free commit.
fn resolve_spared(p: &Arc<Pending>, pso: usize) -> usize {
    match p.build.real.load(Ordering::Acquire) {
        0 => {
            schedule(&p.build, Priority::BoundSoon, false, None);
            if !*p.build.done.lock().unwrap() && !p.late.swap(true, Ordering::AcqRel) {
                static N: AtomicUsize = AtomicUsize::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                if n < 16 || n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: a compute program that writes only images is still being built: its dispatches are left out until it is ({} so far)", n + 1);
                }
            }
            p.stand(pso)
        }
        r => r,
    }
}

/// For binding: compute pipelines are waited for, graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => {
            p.note_use();
            resolve_wait_pending(&p, pso, wait_budget())
        }
        Some(p) => resolve_pending(&p, pso),
        None => pso,
    }
}

/// Resolve and strongly pin the exact recipe-backed native generation as one operation. The
/// generation is chosen while holding `Build.done`, the same lock that publishes it and that
/// `released` uses before giving up the original raw owner. Ordinary one-shot builds keep their
/// existing resolution path and return no generation pin.
pub(super) fn resolve_and_pin_for_bind(pso: usize) -> (usize, Option<NativeGeneration>) {
    resolve_and_pin_mode(pso, ResolveMode::ForBind)
}

#[derive(Clone, Copy)]
enum ResolveMode { Lazy, Wait, ForBind }

fn resolve_and_pin_mode(pso: usize, mode: ResolveMode) -> (usize, Option<NativeGeneration>) {
    let Some(p) = entry(pso) else { return (pso, None) };
    resolve_and_pin_pending(pso, &p, mode)
}

fn resolve_and_pin_pending(pso: usize, p: &Arc<Pending>, mode: ResolveMode) -> (usize, Option<NativeGeneration>) {
    // Resolve directly against this retained Pending: a concurrent release/address reuse cannot
    // make selection consult a new registry entry while validation still uses the old recipe.
    let resolved = match mode {
        ResolveMode::Lazy => resolve_pending(p, pso),
        ResolveMode::Wait => {
            p.note_use();
            resolve_wait_pending(p, pso, wait_budget())
        }
        ResolveMode::ForBind if p.compute => {
            p.note_use();
            resolve_wait_pending(p, pso, wait_budget())
        }
        ResolveMode::ForBind => resolve_pending(p, pso),
    };
    let bound_unbuilt = p.bound_unbuilt(pso, resolved);
    let (resolved, generation, _) = pin_resolved(pso, p, resolved, bound_unbuilt);
    (resolved, generation)
}

fn pin_resolved(pso: usize, p: &Pending, resolved: usize, bound_unbuilt: bool) -> (usize, Option<NativeGeneration>, bool) {
    let Some(recipe) = p.build.recipe.as_ref() else { return (resolved, None, false) };
    note_eviction_candidate(&p.build);
    let _done = p.build.done.lock().unwrap();
    let current = p.build.real.load(Ordering::Acquire);
    if current == resolved {
        let (generation, residency) = recipe.pin_with_residency(resolved);
        let unavailable = generation.is_none() && matches!(residency, Residency::Evicted | Residency::Rebuilding);
        return (resolved, generation, unavailable);
    }
    // Preserve a resolver's deliberate stand-in choice. If an eventual rebuild replaced a
    // native generation between resolution and this lock, select and pin the current publication.
    // Eviction can instead leave no publication: fail closed to the retained stand-in, never the
    // ownerless raw pointer which resolution observed before it took this lock.
    if bound_unbuilt {
        return (resolved, None, false);
    }
    let (generation, residency) = recipe.pin_with_residency(current);
    match generation {
        Some(generation) => (current, Some(generation), false),
        None => {
            let stand = p.stand(pso);
            let unavailable = resolved != stand || matches!(residency, Residency::Evicted | Residency::Rebuilding);
            (stand, None, unavailable)
        }
    }
}

/// Preserve the ordinary resolver exactly unless recipe rebuilding was explicitly enabled.
pub(super) fn resolve_and_pin_if_enabled(pso: usize) -> (usize, Option<NativeGeneration>) {
    if rebuild_recipes_enabled() { resolve_and_pin_for_bind(pso) } else { (resolve_for_bind(pso), None) }
}

pub(super) fn resolve_and_pin_if_enabled_lazy(pso: usize) -> (usize, Option<NativeGeneration>) {
    if rebuild_recipes_enabled() { resolve_and_pin_mode(pso, ResolveMode::Lazy) } else { (resolve(pso), None) }
}

pub(super) fn resolve_and_pin_if_enabled_wait(pso: usize) -> (usize, Option<NativeGeneration>) {
    if rebuild_recipes_enabled() { resolve_and_pin_mode(pso, ResolveMode::Wait) } else { (resolve_wait(pso), None) }
}

pub(super) fn resolve_and_pin_if_enabled_identity(pso: usize) -> (usize, Option<NativeGeneration>) {
    (pso, None)
}

pub(super) fn resolve_wait_bound_and_pin_if_enabled(pso: usize) -> ((usize, bool), Option<NativeGeneration>) {
    if !rebuild_recipes_enabled() {
        return (resolve_wait_bound(pso), None);
    }
    resolve_wait_bound_and_pin(pso)
}

fn resolve_wait_bound_and_pin(pso: usize) -> ((usize, bool), Option<NativeGeneration>) {
    let Some(p) = entry(pso) else { return ((pso, false), None) };
    p.note_use();
    let bound = resolve_wait_pending(&p, pso, wait_budget());
    let choice = (bound, p.bound_unbuilt(pso, bound));
    let (resolved, generation, stale_fallback) = pin_resolved(pso, &p, choice.0, choice.1);
    ((resolved, choice.1 || stale_fallback), generation)
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release when this is its last owner
/// (0: another `Pending` owns the build, none, or still being built – the worker's result is
/// then released when its last owner is gone).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.remove(pso) else { return 0 };
    let mut orphaned = false;
    {
        let done = p.build.done.lock().unwrap();
        let owners = p.build.owners.fetch_sub(1, Ordering::Relaxed) - 1;
        if *done {
            return if owners == 0 { p.build.real.swap(0, Ordering::AcqRel) } else { 0 };
        }
        if owners == 0 {
            // the worker stores its result into a Build nobody will read: mark it orphaned
            p.build.orphan.store(true, Ordering::Release);
            p.build.finish_queued();
            orphaned = true;
        }
    }
    if orphaned {
        forget_in_flight(&p.build);
        drop_unstarted_job(&p.build);
    }
    0
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.build.done.lock().unwrap())
}

/// A call with `pso` was put off to the submit, which will wait for it: its shared build is
/// wanted from record time and goes before the workers' queue.
pub fn need_by_submit(pso: usize) {
    let Some(p) = entry(pso) else { return };
    p.build.want();
    let first = !std::mem::replace(&mut p.build.demand.lock().unwrap().by_submit, true);
    if first {
        schedule(&p.build, Priority::BoundSoon, false, None);
    }
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), QUEUED_BUILDS.load(Ordering::Relaxed))
}
