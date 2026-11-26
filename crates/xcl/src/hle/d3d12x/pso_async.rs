//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for where it is used, a `Dispatch` at its submit (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock, Weak};

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
        Self::new_keyed(job, sent, None)
    }

    fn new_keyed(job: Job, sent: bool, compute_key: Option<Arc<super::real::compute_build_key::ComputeBuildKey>>) -> Self {
        Self {
            compute_key,
            job: Mutex::new(Some(job)),
            sent: std::sync::atomic::AtomicBool::new(sent),
            tracked: AtomicBool::new(false),
            queue_state: AtomicU8::new(QUEUE_IDLE),
            real: AtomicUsize::new(0),
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
    }
    let orphan = {
        let mut done = build.done.lock().unwrap();
        let orphan = build.orphan.load(Ordering::Acquire);
        if !orphan {
            build.real.store(real, Ordering::Release);
        }
        *done = true;
        build.finish_queued();
        build.cv.notify_all();
        orphan
    };
    // outside `done`: the Release override can re-enter pipeline bookkeeping
    if orphan {
        release(real);
    }
    forget_in_flight(build);
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

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

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
    submit_as(standin, compute, job, true, 0, compute_key);
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
    submit_as(standin, compute, job, false, key, compute_key);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64, compute_key: Option<super::real::compute_build_key::ComputeBuildKey>) {
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
                let build = Arc::new(Build::new_keyed(job, false, Some(compute_key.clone())));
                builds.insert(compute_key, Arc::downgrade(&build));
                COALESCING.unique();
                build
            }
        } else {
            let compute_key = Arc::new(compute_key);
            let build = Arc::new(Build::new_keyed(job, false, Some(compute_key.clone())));
            builds.insert(compute_key, Arc::downgrade(&build));
            COALESCING.unique();
            build
        }
    } else {
        Arc::new(Build::new(job, false))
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
