//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for where it is used, a `Dispatch` at its submit (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

/// Sentinel for `Pending::waited_frame`: no frame has spent graphics wait budget on this PSO yet.
const UNWAITED: u64 = u64::MAX;

/// Lifecycle states for tracking whether a `Pending` pipeline currently counts toward
/// `WANTED_ACTIVE_COUNT`:
/// - `WANTED_NONE` (0): not wanted yet and not finished
/// - `WANTED_ACTIVE` (1): actively wanted and still queued/building (contributes +1)
/// - `WANTED_DONE` (2): completed, dropped unstarted, or permanently gated (contributes 0)
const WANTED_NONE: u8 = 0;
const WANTED_ACTIVE: u8 = 1;
const WANTED_DONE: u8 = 2;

/// Number of `wanted` pipelines currently queued or building on workers. While non-zero,
/// speculative `xcl-pso-pre-*` preload pauses so background builds never steal RAM/CPU from
/// pipelines the current frame actually needs.
static WANTED_ACTIVE_COUNT: AtomicUsize = AtomicUsize::new(0);

fn wanted_active() -> usize {
    WANTED_ACTIVE_COUNT.load(Ordering::Acquire)
}

struct Pending {
    /// the build, until a worker – or a bind that cannot wait for the queue – takes it
    job: Mutex<Option<Job>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    /// the driver pipeline binds get till then (`shared`); 0: the stand-in is one itself
    driver: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
    /// a bind waited for it longer than `wait_budget`: binds go on without it until it is built
    late: std::sync::atomic::AtomicBool,
    /// its dispatches may be left out while it is built (`image_only`): binds do not wait for it
    spared: std::sync::atomic::AtomicBool,
    /// shader issues `image_bvh*_intersect_ray`: held back without blocking until BVH builds succeed
    bvh_traversal: std::sync::atomic::AtomicBool,
    /// heavyweight RT/compute shader (`>= 512 KiB` DXIL / BVH traversal): lower worker queue
    /// priority than normal/MS PSOs so world-entry draws finish ahead of slow RT compiles
    deprioritized: std::sync::atomic::AtomicBool,
    /// first frame that waited on this graphics PSO within `XCL_PSO_FRAME_BUDGET_MS` (`UNWAITED`
    /// if never waited yet): later frames skip waiting while workers finish compiling it
    waited_frame: AtomicU64,
    /// tracks whether this entry currently contributes to `WANTED_ACTIVE_COUNT`
    wanted_state: AtomicU8,
    demand: Mutex<Demand>,
    /// what its first use tells besides `used::note` (`on_first_use`)
    on_use: Mutex<Option<OnUse>>,
}

impl Drop for Pending {
    fn drop(&mut self) {
        self.clear_wanted_active();
    }
}

/// Whether one of the title's threads waits for the pipeline, and the preload worker building it.
#[derive(Default)]
struct Demand {
    wanted: bool,
    /// a call put off to the submit needs it (`need_by_submit`)
    by_submit: bool,
    /// thread id of the preload worker (lowest priority) building it, 0 = none
    preload_thread: u32,
}

thread_local! {
    /// what this worker is building
    static BUILDING: std::cell::RefCell<Option<Arc<Pending>>> = const { std::cell::RefCell::new(None) };
}

/// This thread is a worker and one of the title's threads waits for what it is building: the
/// build must not stand back for the title's sake (`real::big_compile_turn`).
pub(super) fn build_is_wanted() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.demand.lock().unwrap().wanted))
}

/// Whether one of the title's threads waits for what this thread is doing, to be asked later and
/// from another thread: a worker's build as `build_is_wanted`; a thread that is not XCL's is the
/// title's own.
pub fn wanted_here() -> crate::gpu::recomp::dxc::Wanted {
    if let Some(p) = BUILDING.with(|b| b.borrow().clone()) {
        return Box::new(move || p.demand.lock().unwrap().wanted);
    }
    let own = crate::state::boot().is_some() && !std::thread::current().name().is_some_and(|n| n.starts_with("xcl-"));
    Box::new(move || own)
}

/// A worker's build of `p` (`preload`: by one of the lowest-priority workers).
fn build(p: &Arc<Pending>, job: Job, preload: bool) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL};
    BUILDING.with(|b| *b.borrow_mut() = Some(p.clone()));
    if preload {
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = unsafe { GetCurrentThreadId() };
        if d.wanted {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL) };
        }
    }
    // (a panic in a build ends it as a failed one: whoever waits for it is not left waiting)
    complete(p, std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).unwrap_or(0));
    if preload {
        // (under the lock: `want` does not raise this thread after it)
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = 0;
        if d.wanted {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
        }
    }
    BUILDING.with(|b| *b.borrow_mut() = None);
}

impl Pending {
    fn mark_wanted_active(&self) {
        if self
            .wanted_state
            .compare_exchange(WANTED_NONE, WANTED_ACTIVE, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            WANTED_ACTIVE_COUNT.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn clear_wanted_active(&self) {
        if self.wanted_state.swap(WANTED_DONE, Ordering::AcqRel) == WANTED_ACTIVE {
            WANTED_ACTIVE_COUNT.fetch_sub(1, Ordering::AcqRel);
        }
    }

    /// One of the title's threads is about to wait for it: the worker building it goes on now,
    /// at a normal priority.
    fn want(&self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_PRIORITY_NORMAL, THREAD_SET_LIMITED_INFORMATION};
        let became_wanted = {
            let mut d = self.demand.lock().unwrap();
            let first = !std::mem::replace(&mut d.wanted, true);
            if first && d.preload_thread != 0 {
                unsafe {
                    let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, d.preload_thread);
                    if !h.is_null() {
                        SetThreadPriority(h, THREAD_PRIORITY_NORMAL);
                        CloseHandle(h);
                    }
                }
            }
            first
        };
        if became_wanted {
            self.mark_wanted_active();
            crate::gpu::recomp::dxc::notify_wanted();
            notify_queue_wanted();
            super::real::notify_big_compile_wanted();
        }
    }

    fn note_use(&self) {
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
            let on_use = self.on_use.lock().unwrap().take();
            if let Some(f) = on_use {
                f();
            }
        }
    }
}

/// stand-in handed to the title → its real pipeline
static PENDING: std::sync::RwLock<Option<HashMap<usize, Arc<Pending>>>> = std::sync::RwLock::new(None);

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
    p.clear_wanted_active();
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
    }
    let orphan = {
        let mut done = p.done.lock().unwrap();
        let orphan = p.orphan.load(Ordering::Acquire);
        if !orphan {
            p.real.store(real, Ordering::Release);
        }
        *done = true;
        p.cv.notify_all();
        orphan
    };
    // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
    if orphan {
        release(real);
    }
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);
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

/// pipelines a submit will wait for: the workers take these before their queue (`need_by_submit`)
static NEEDED: Mutex<std::collections::VecDeque<Arc<Pending>>> = Mutex::new(std::collections::VecDeque::new());

fn next_needed() -> Option<Arc<Pending>> {
    NEEDED.lock().unwrap().pop_front()
}

/// Drop an unstarted orphaned job outside all locks and wake any concurrent waiter.
fn drop_unstarted_job(p: &Pending) {
    p.clear_wanted_active();
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        {
            let mut done = p.done.lock().unwrap();
            *done = true;
            p.cv.notify_all();
        }
        drop(job);
    }
}

type Work = Box<dyn FnOnce() + Send>;

struct PsoWork {
    work: Work,
    pending: Option<Arc<Pending>>,
}

/// Priority work queue for `xcl-pso-*` workers: pipelines waited on by a title thread (`wanted`)
/// jump ahead of queued background builds.
struct PsoQueue {
    items: Mutex<std::collections::VecDeque<PsoWork>>,
    cv: Condvar,
}

impl PsoQueue {
    fn new() -> Self {
        Self { items: Mutex::new(std::collections::VecDeque::new()), cv: Condvar::new() }
    }

    fn push(&self, work: Work, pending: Option<Arc<Pending>>) {
        self.items.lock().unwrap().push_back(PsoWork { work, pending });
        self.cv.notify_one();
    }

    fn notify_wanted(&self) {
        let _g = self.items.lock().unwrap();
        drop(_g);
        self.cv.notify_all();
    }

    fn pop(&self) -> Work {
        let mut g = self.items.lock().unwrap();
        loop {
            if let Some(pos) = g.iter().position(|item| {
                item.pending.as_ref().is_some_and(|p| {
                    p.compute
                        && !p.spared.load(Ordering::Acquire)
                        && !p.bvh_traversal.load(Ordering::Acquire)
                        && p.demand.lock().unwrap().wanted
                })
            }) {
                return g.remove(pos).unwrap().work;
            }
            if let Some(pos) = g.iter().position(|item| item.pending.as_ref().is_some_and(|p| p.demand.lock().unwrap().wanted)) {
                return g.remove(pos).unwrap().work;
            }
            if let Some(pos) = g.iter().position(|item| !item.pending.as_ref().is_some_and(|p| p.deprioritized.load(Ordering::Acquire))) {
                return g.remove(pos).unwrap().work;
            }
            if let Some(item) = g.pop_front() {
                return item.work;
            }
            g = self.cv.wait(g).unwrap();
        }
    }
}

static WORK_QUEUE: OnceLock<Arc<PsoQueue>> = OnceLock::new();

fn notify_queue_wanted() {
    if let Some(q) = WORK_QUEUE.get() {
        q.notify_wanted();
    }
}

/// Default `xcl-pso-*` worker count for a host with `logical_cpus` hardware threads and
/// `total_ram_gib` GiB of physical RAM.
///
/// Smaller hosts (`< 16` threads or `< 24 GiB` RAM) stay capped at `2..=6` workers to avoid
/// driver LLVM commit pressure; high-core / high-RAM hosts (`>= 16` threads and `>= 24 GiB` RAM)
/// scale cleanly up to `10` workers (`logical_cpus / 2`, clamped to `6..=10`), relying on
/// `driver_compile_identified`'s commit-headroom gate to keep peak commit bounded.
fn pso_worker_threads_for(logical_cpus: usize, total_ram_gib: u64) -> usize {
    if logical_cpus >= 16 && total_ram_gib >= 24 {
        (logical_cpus / 2).clamp(6, 10)
    } else {
        logical_cpus.saturating_sub(1).clamp(2, 6)
    }
}

fn total_physical_ram_gib() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..unsafe { std::mem::zeroed() } };
    (unsafe { GlobalMemoryStatusEx(&mut m) } != 0 && m.ullTotalPhys > 0).then_some(m.ullTotalPhys >> 30)
}

/// Number of `xcl-pso-*` worker threads: honors explicit `XCL_PSO_THREADS` (or `XCL_DXC_THREADS`),
/// otherwise scales via `pso_worker_threads_for(logical_cpus, total_ram_gib)`.
fn pso_worker_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("XCL_PSO_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .or_else(|| {
                std::env::var("XCL_DXC_THREADS")
                    .ok()
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .filter(|&n| n > 0)
            })
            .map(|n| n.clamp(1, 64))
            .unwrap_or_else(|| {
                let cpus = std::thread::available_parallelism().map(usize::from).unwrap_or(4);
                let ram_gib = total_physical_ram_gib().unwrap_or(0);
                pso_worker_threads_for(cpus, ram_gib)
            })
    })
}

/// Number of low-priority `xcl-pso-pre-*` preload threads (`1..=2` by default).
fn preload_worker_threads() -> usize {
    (pso_worker_threads() / 3).clamp(1, 2)
}

fn queue() -> &'static Arc<PsoQueue> {
    WORK_QUEUE.get_or_init(|| {
        let q = Arc::new(PsoQueue::new());
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = pso_worker_threads();
        for k in 0..n {
            let q = q.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(16 << 20).spawn(move || loop {
                while let Some(p) = next_needed() {
                    build_queued(&p);
                }
                let work = q.pop();
                while let Some(p) = next_needed() {
                    build_queued(&p);
                }
                work();
            });
        }
        q
    })
}

fn enqueue(p: Arc<Pending>) {
    let p_q = p.clone();
    queue().push(
        Box::new(move || build_queued(&p)),
        Some(p_q),
    );
}

/// Physical RAM & commit headroom thresholds (`min(ullAvailPhys, ullAvailPageFile)`) for
/// throttling background PSO compilation well before Windows starts paging out the title
/// (`XCL_PSO_MEM_THROTTLE=0`: off).
const PRELOAD_PAUSE_COMMIT: u64 = 6 << 30;
const WORKER_CAP_2_COMMIT: u64 = 5 << 30;
const WORKER_CAP_1_COMMIT: u64 = 3 << 30;

fn mem_throttle_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !cfg!(test) && !std::env::var("XCL_PSO_MEM_THROTTLE").is_ok_and(|v| v.trim() == "0"))
}

/// The system's free commit: what a driver compile can still allocate. (Free physical memory is
/// not it: a retail title's world entry dropped it to 1.3 GB for a few seconds with 12 GB of
/// commit free – the pages go to the page file – and the throttle on it held every pipeline
/// worker idle while the title waited 14 s for a compute pipeline.)
pub(super) fn avail_commit_bytes() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut m) } != 0).then_some(m.ullAvailPageFile)
}

/// Maximum concurrent top-level PSO worker builds allowed for the current available RAM/commit:
/// `< 3 GiB` -> 1 worker (never 0, no deadlock), `< 5 GiB` -> at most 2, otherwise `max_workers`.
fn worker_cap_for_commit(avail_commit: Option<u64>, max_workers: usize) -> usize {
    let max_workers = max_workers.max(1);
    match avail_commit {
        Some(avail) if avail < WORKER_CAP_1_COMMIT => 1,
        Some(avail) if avail < WORKER_CAP_2_COMMIT => max_workers.min(2),
        _ => max_workers,
    }
}

/// Pause speculative `xcl-pso-pre-*` preload builds while available RAM/commit is below 6 GiB.
fn should_pause_preload(avail_commit: Option<u64>) -> bool {
    avail_commit.is_some_and(|avail| avail < PRELOAD_PAUSE_COMMIT)
}

/// Pause speculative preload whenever ANY `wanted` pipeline is actively queued/in-flight OR
/// available RAM/commit headroom is below `PRELOAD_PAUSE_COMMIT`.
fn should_pause_preload_with_demand(avail_commit: Option<u64>, wanted: usize) -> bool {
    wanted > 0 || should_pause_preload(avail_commit)
}

struct ActiveWorkerGate {
    active: Mutex<usize>,
    cv: Condvar,
}

static ACTIVE_WORKERS: ActiveWorkerGate = ActiveWorkerGate {
    active: Mutex::new(0),
    cv: Condvar::new(),
};

struct ActiveWorkerGuard;

impl Drop for ActiveWorkerGuard {
    fn drop(&mut self) {
        let mut n = ACTIVE_WORKERS.active.lock().unwrap();
        *n = n.saturating_sub(1);
        ACTIVE_WORKERS.cv.notify_all();
    }
}

fn acquire_worker_slot(max_workers: usize) -> ActiveWorkerGuard {
    let mut n = ACTIVE_WORKERS.active.lock().unwrap();
    loop {
        let avail = mem_throttle_enabled().then(avail_commit_bytes).flatten();
        let cap = worker_cap_for_commit(avail, max_workers);
        if *n < cap {
            if cap < max_workers {
                static LOGGED: std::sync::Once = std::sync::Once::new();
                LOGGED.call_once(|| {
                    crate::xlog_info!(
                        "D3D12.X: throttling PSO workers to {cap} under low RAM/commit headroom ({} MiB free)",
                        avail.unwrap_or(0) >> 20
                    );
                });
            }
            *n += 1;
            return ActiveWorkerGuard;
        }
        n = ACTIVE_WORKERS.cv.wait_timeout(n, std::time::Duration::from_millis(50)).unwrap().0;
    }
}

fn build_queued(p: &Arc<Pending>) {
    // (released by the title meanwhile)
    if p.orphan.load(Ordering::Acquire) {
        drop_unstarted_job(p);
        return;
    }
    let _slot = acquire_worker_slot(pso_worker_threads());
    if p.orphan.load(Ordering::Acquire) {
        drop_unstarted_job(p);
        return;
    }
    // (None: a bind took it and builds it itself)
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        build(p, job, false);
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
        let _ = send.send((i, job()));
    };
    // the helpers build for this worker's pipeline: the title waiting for it waits for them too
    let parent = BUILDING.with(|b| b.borrow().clone());
    for _ in 1..count.min(pso_worker_threads()) {
        let (drain, parent_run, parent_q) = (drain.clone(), parent.clone(), parent.clone());
        queue().push(
            Box::new(move || {
                let before = BUILDING.with(|b| b.replace(parent_run));
                drain();
                BUILDING.with(|b| *b.borrow_mut() = before);
            }),
            parent_q,
        );
    }
    drain();
    let mut results = vec![0; count];
    for (i, result) in recv.iter().take(count) { results[i] = result; }
    results
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// Initial scheduling classification set atomically before a `Pending` entry is visible to
/// preload or normal workers (avoids any race where `xcl-pso-pre-*` pops a heavy/BVH pipeline
/// before its flags are applied).
#[derive(Clone, Copy, Default)]
pub struct SubmitFlags {
    pub spared: bool,
    pub bvh_traversal: bool,
    pub deprioritized: bool,
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true, 0, SubmitFlags::default());
}

/// Like [`submit`], with initial scheduling flags applied before the entry is queued.
pub fn submit_with_flags(standin: usize, compute: bool, job: Job, flags: SubmitFlags) {
    submit_as(standin, compute, job, true, 0, flags);
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
    submit_as(standin, compute, job, false, key, SubmitFlags::default());
}

/// Like [`submit_lazy`], applying `flags` atomically before insertion and excluding heavyweight /
/// BVH / spared pipelines from speculative `xcl-pso-pre-*` preloading so `THREAD_PRIORITY_LOWEST`
/// workers never stall `BIG_COMPILES`/`GIANT_COMPILES` on unreferenced >1 MiB shaders.
pub fn submit_lazy_with_flags(standin: usize, compute: bool, job: Job, key: u64, flags: SubmitFlags) {
    submit_as(standin, compute, job, false, key, flags);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64, flags: SubmitFlags) {
    let p = Arc::new(Pending {
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(send),
        real: AtomicUsize::new(0),
        driver: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        key,
        used: Default::default(),
        orphan: Default::default(),
        late: Default::default(),
        spared: std::sync::atomic::AtomicBool::new(flags.spared),
        bvh_traversal: std::sync::atomic::AtomicBool::new(flags.bvh_traversal),
        deprioritized: std::sync::atomic::AtomicBool::new(flags.deprioritized),
        waited_frame: AtomicU64::new(UNWAITED),
        wanted_state: AtomicU8::new(WANTED_NONE),
        demand: Default::default(),
        on_use: Default::default(),
    });
    PENDING.write().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        enqueue(p);
    } else if used::known(key) && !flags.deprioritized && !flags.bvh_traversal && !flags.spared {
        if let Some(q) = preload() {
            q.list.lock().unwrap().push_back(p);
            q.cv.notify_one();
        }
    }
}

/// Run on the thread of a lazy pipeline's first use, once (it is to hand work to another).
pub type OnUse = Box<dyn FnOnce() + Send>;

/// What the first use of lazy pipeline `standin` tells besides `used::note`: a graphics one's
/// description for the next run's preparation (`gfx_records::note`). Told before the title has
/// the stand-in.
pub(super) fn on_first_use(standin: usize, f: OnUse) {
    if let Some(p) = entry(standin) {
        *p.on_use.lock().unwrap() = Some(f);
    }
}

/// The shaders of a pipeline's set, told when the set's key is made (`real::OwnedDesc`).
pub(crate) fn shaders_of(key: u64, shaders: Vec<u64>) {
    used::parts(key, shaders);
}

/// The shaders of the pipelines earlier runs used, each by its own key (`used-shaders.txt`
/// beside the DXIL cache): what a title's shader preparation translates. Empty before a run
/// with this list has used a pipeline.
pub fn earlier_used_shaders() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(crate::gpu::recomp::dxc::cache_dir().join("used-shaders.txt")).unwrap_or_default())
}

/// The keys of the pipelines earlier runs used (`used-pipelines.txt` beside the DXIL cache):
/// what `xcl shaders --precompile --used-only` builds.
pub fn earlier_used_keys() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt")).unwrap_or_default())
}

/// Flush pending `used-pipelines.txt` keys on the background journal writer.
pub fn flush_used() {
    used::flush();
}

/// Flush and join the `used-pipelines.txt` journal worker on normal shutdown.
pub fn shutdown_used() {
    used::shutdown();
}

/// Best-effort bounded flush of pending `used-pipelines.txt` keys on process termination.
pub fn flush_used_bounded(timeout: std::time::Duration) {
    used::flush_bounded(timeout);
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
    use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
    use std::{
        collections::HashSet,
        fmt::Write as _,
        io::Write,
        path::PathBuf,
        thread::JoinHandle,
        time::{Duration, Instant},
    };

    /// Coalesce first-use bursts into a single disk write off the bind lock.
    const COALESCE_WINDOW: Duration = Duration::from_millis(25);
    /// Flush immediately once a burst reaches this many unique keys.
    const BATCH_IMMEDIATE: usize = 32;

    enum Sink {
        LazyFile(PathBuf, Option<std::fs::File>),
        #[cfg(test)]
        Writer(Box<dyn Write + Send>),
    }

    impl Sink {
        fn write_batch(&mut self, bytes: &[u8]) {
            match self {
                Self::LazyFile(path, file) => {
                    if file.is_none() {
                        if let Some(dir) = path.parent() {
                            let _ = std::fs::create_dir_all(dir);
                        }
                        *file = std::fs::OpenOptions::new().create(true).append(true).open(&*path).ok();
                    }
                    if file.as_mut().is_some_and(|f| f.write_all(bytes).is_err() || f.flush().is_err()) {
                        *file = None;
                    }
                }
                #[cfg(test)]
                Self::Writer(w) => {
                    if w.write_all(bytes).is_ok() {
                        let _ = w.flush();
                    }
                }
            }
        }
    }

    fn format_keys(batch: &[u64], buf: &mut String) {
        buf.clear();
        buf.reserve(batch.len() * 17);
        for &key in batch {
            let _ = writeln!(buf, "{key:016x}");
        }
    }

    fn append_to_path(path: &PathBuf, bytes: &[u8]) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = f.write_all(bytes);
            let _ = f.flush();
        }
    }

    #[derive(Default)]
    struct JournalState {
        seen: HashSet<u64>,
        pending: Vec<u64>,
        fallback_path: Option<PathBuf>,
        enqueued_seq: u64,
        completed_seq: u64,
        flush_now: bool,
        closed: bool,
        writer_stopped: bool,
    }

    struct JournalShared {
        state: Mutex<JournalState>,
        wake_cv: Condvar,
        flushed_cv: Condvar,
    }

    struct StopGuard<'a>(&'a JournalShared);
    impl Drop for StopGuard<'_> {
        fn drop(&mut self) {
            self.0.state.lock().unwrap_or_else(|e| e.into_inner()).writer_stopped = true;
            self.0.flushed_cv.notify_all();
        }
    }

    pub(super) struct Journal {
        shared: Arc<JournalShared>,
        writer: Mutex<Option<JoinHandle<()>>>,
    }

    impl Journal {
        fn new_shared(state: JournalState) -> Arc<JournalShared> {
            Arc::new(JournalShared { state: Mutex::new(state), wake_cv: Condvar::new(), flushed_cv: Condvar::new() })
        }

        fn in_memory(seen: HashSet<u64>) -> Self {
            Self {
                shared: Self::new_shared(JournalState { seen, writer_stopped: true, ..Default::default() }),
                writer: Mutex::new(None),
            }
        }

        fn with_sink(seen: HashSet<u64>, sink: Sink, coalesce: Duration) -> Self {
            let fallback_path = match &sink {
                Sink::LazyFile(p, _) => Some(p.clone()),
                #[cfg(test)]
                Sink::Writer(_) => None,
            };
            let shared = Self::new_shared(JournalState { seen, fallback_path, ..Default::default() });
            let bg = Arc::clone(&shared);
            let writer = std::thread::Builder::new()
                .name("xcl-pso-used".into())
                .spawn(move || writer_loop(&bg, sink, coalesce))
                .ok();
            if writer.is_none() {
                shared.state.lock().unwrap().writer_stopped = true;
                shared.flushed_cv.notify_all();
            }
            Self { shared, writer: Mutex::new(writer) }
        }

        #[cfg(test)]
        pub(super) fn with_writer(w: Box<dyn Write + Send>, c: Duration) -> Self {
            Self::with_sink(HashSet::new(), Sink::Writer(w), c)
        }

        #[cfg(test)]
        pub(super) fn with_file(p: PathBuf, c: Duration) -> Self {
            Self::with_sink(HashSet::new(), Sink::LazyFile(p, None), c)
        }

        #[cfg(test)]
        pub(super) fn without_writer(p: PathBuf) -> Self {
            Self {
                shared: Self::new_shared(JournalState { fallback_path: Some(p), writer_stopped: true, ..Default::default() }),
                writer: Mutex::new(None),
            }
        }

        pub(super) fn note(&self, key: u64) {
            if key == 0 {
                return;
            }
            let mut g = self.shared.state.lock().unwrap();
            if !g.seen.insert(key) || g.closed || (g.writer_stopped && g.fallback_path.is_none()) {
                return;
            }
            g.pending.push(key);
            g.enqueued_seq += 1;
            if matches!(g.pending.len(), 1 | BATCH_IMMEDIATE) {
                self.shared.wake_cv.notify_one();
            }
        }

        #[cfg(test)]
        pub(super) fn noted(&self, key: u64) -> bool {
            self.shared.state.lock().unwrap().seen.contains(&key)
        }

        pub(super) fn flush(&self) {
            let (fallback, batch) = {
                let mut g = self.shared.state.lock().unwrap();
                let target = g.enqueued_seq;
                if g.completed_seq >= target || (g.writer_stopped && g.pending.is_empty()) {
                    return;
                }
                if !g.writer_stopped {
                    g.flush_now = true;
                    self.shared.wake_cv.notify_one();
                    while g.completed_seq < target && !g.writer_stopped {
                        g = self.shared.flushed_cv.wait(g).unwrap();
                    }
                }
                if g.pending.is_empty() {
                    return;
                }
                let batch = std::mem::take(&mut g.pending);
                g.completed_seq = g.enqueued_seq;
                (g.fallback_path.clone(), batch)
            };
            if let Some(path) = fallback {
                let mut buf = String::new();
                format_keys(&batch, &mut buf);
                append_to_path(&path, buf.as_bytes());
            }
        }

        pub(super) fn flush_bounded(&self, timeout: Duration) {
            let (fallback, batch) = {
                let mut g = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
                let target = g.enqueued_seq;
                if g.completed_seq >= target || (g.writer_stopped && g.pending.is_empty()) {
                    return;
                }
                g.flush_now = true;
                self.shared.wake_cv.notify_one();
                let deadline = Instant::now() + timeout;
                while g.completed_seq < target && !g.writer_stopped {
                    let Some(rem) = deadline.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) else { break };
                    let (guard, res) = self.shared.flushed_cv.wait_timeout(g, rem).unwrap_or_else(|e| e.into_inner());
                    g = guard;
                    if res.timed_out() {
                        break;
                    }
                }
                if g.pending.is_empty() {
                    return;
                }
                // Worker timed out (e.g. blocked on the Windows loader lock during ExitProcess) or
                // failed to spawn: take the remaining keys and append them directly so none are lost.
                let batch = std::mem::take(&mut g.pending);
                g.completed_seq = g.enqueued_seq;
                (g.fallback_path.clone(), batch)
            };
            if let Some(path) = fallback {
                let mut buf = String::new();
                format_keys(&batch, &mut buf);
                append_to_path(&path, buf.as_bytes());
            }
        }

        pub(super) fn shutdown(&self) {
            {
                let mut g = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
                g.closed = true;
                g.flush_now = true;
                self.shared.wake_cv.notify_one();
            }
            if let Some(h) = self.writer.lock().unwrap_or_else(|e| e.into_inner()).take() {
                if h.thread().id() != std::thread::current().id() {
                    let _ = h.join();
                }
            } else {
                let mut g = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
                while !g.writer_stopped {
                    g = self.shared.flushed_cv.wait(g).unwrap_or_else(|e| e.into_inner());
                }
            }
            let (fallback, batch) = {
                let mut g = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
                if g.pending.is_empty() {
                    return;
                }
                let batch = std::mem::take(&mut g.pending);
                g.completed_seq = g.enqueued_seq;
                (g.fallback_path.clone(), batch)
            };
            if let Some(path) = fallback {
                let mut buf = String::new();
                format_keys(&batch, &mut buf);
                append_to_path(&path, buf.as_bytes());
            }
        }
    }

    impl Drop for Journal {
        fn drop(&mut self) {
            self.shutdown();
        }
    }

    fn writer_loop(shared: &JournalShared, mut sink: Sink, coalesce: Duration) {
        let _stop = StopGuard(shared);
        let mut buf = String::new();
        loop {
            let (batch, batch_seq) = {
                let mut g = shared.state.lock().unwrap();
                while g.pending.is_empty() && !g.closed {
                    if std::mem::take(&mut g.flush_now) {
                        shared.flushed_cv.notify_all();
                    }
                    g = shared.wake_cv.wait(g).unwrap();
                }
                if g.pending.is_empty() && g.closed {
                    break;
                }
                if !coalesce.is_zero() {
                    let deadline = Instant::now() + coalesce;
                    while g.pending.len() < BATCH_IMMEDIATE && !g.flush_now && !g.closed {
                        let Some(rem) = deadline.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) else { break };
                        let (guard, res) = shared.wake_cv.wait_timeout(g, rem).unwrap();
                        g = guard;
                        if res.timed_out() {
                            break;
                        }
                    }
                }
                g.flush_now = false;
                (std::mem::take(&mut g.pending), g.enqueued_seq)
            };
            format_keys(&batch, &mut buf);
            sink.write_batch(buf.as_bytes());
            {
                let mut g = shared.state.lock().unwrap();
                g.completed_seq = g.completed_seq.max(batch_seq);
            }
            shared.flushed_cv.notify_all();
        }
    }

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Journal,
        /// the shaders of each set made in this run, by the set's key; the shaders of the sets
        /// used so far and `used-shaders.txt`, where they are kept for the next run
        parts: Mutex<std::collections::HashMap<u64, Vec<u64>>>,
        shaders: Journal,
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0).collect()
    }

    static U: OnceLock<Used> = OnceLock::new();

    extern "C" {
        fn atexit(f: extern "C" fn()) -> std::ffi::c_int;
    }

    extern "C" fn on_process_exit() {
        shutdown();
    }

    fn init_from_path(path: PathBuf, coalesce: Duration) -> Used {
        let earlier = parse(&std::fs::read_to_string(&path).unwrap_or_default());
        if !earlier.is_empty() {
            crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
        }
        let shaders_path = path.with_file_name("used-shaders.txt");
        let earlier_shaders = parse(&std::fs::read_to_string(&shaders_path).unwrap_or_default());
        let now = Journal::with_sink(earlier.clone(), Sink::LazyFile(path, None), coalesce);
        let shaders = Journal::with_sink(earlier_shaders, Sink::LazyFile(shaders_path, None), coalesce);
        unsafe {
            atexit(on_process_exit);
        }
        Used { earlier: RwLock::new(earlier), now, parts: Default::default(), shaders }
    }

    fn get() -> &'static Used {
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                #[cfg(test)]
                if let Some(p) = std::env::var_os("XCL_TEST_USED_STATIC_PATH") {
                    return init_from_path(PathBuf::from(p), Duration::from_secs(60));
                }
                return Used {
                    earlier: RwLock::new(HashSet::new()),
                    now: Journal::in_memory(HashSet::new()),
                    parts: Default::default(),
                    shaders: Journal::in_memory(HashSet::new()),
                };
            }
            init_from_path(crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt"), COALESCE_WINDOW)
        })
    }

    pub fn flush() {
        if let Some(u) = U.get() {
            u.now.flush();
            u.shaders.flush();
        }
    }

    pub fn shutdown() {
        if let Some(u) = U.get() {
            u.now.shutdown();
            u.shaders.shutdown();
        }
    }

    pub fn flush_bounded(timeout: Duration) {
        if let Some(u) = U.get() {
            u.now.flush_bounded(timeout);
            u.shaders.flush_bounded(timeout);
        }
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        key != 0 && get().earlier.read().unwrap().contains(&key)
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        get().now.note(key);
        // its shaders, each once (also those of a set an earlier run used: the list of
        // shaders is younger than the list of sets)
        let parts = get().parts.lock().unwrap().get(&key).cloned().unwrap_or_default();
        for part in parts {
            get().shaders.note(part);
        }
    }

    /// The shaders of the set `key`, each by its own key (`real::shader_key`).
    pub fn parts(key: u64, shaders: Vec<u64>) {
        if key != 0 {
            get().parts.lock().unwrap().entry(key).or_insert(shaders);
        }
    }

    /// (tests) This run used the shader `key`.
    #[cfg(test)]
    pub fn shader_noted(key: u64) -> bool {
        get().shaders.noted(key)
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.noted(key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap().insert(key);
    }
}

/// The workers that build the pipelines of an earlier run ahead of their first use
/// (`XCL_PSO_PRELOAD=0`: off). Apart from the ones a bind sends to (`queue`) and at the lowest
/// priority: a pipeline that is bound is not held up by the ones made ahead (whoever takes
/// `job` first builds it), and the title's own threads are not either.
struct Preload {
    list: Mutex<std::collections::VecDeque<Arc<Pending>>>,
    cv: Condvar,
}

fn preload() -> Option<&'static Preload> {
    static Q: OnceLock<Option<Preload>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let q = Q
        .get_or_init(|| {
            if std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0") {
                return None;
            }
            Some(Preload { list: Mutex::new(Default::default()), cv: Condvar::new() })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let n = preload_worker_threads();
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(16 << 20).spawn(move || {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
                static DONE: AtomicUsize = AtomicUsize::new(0);
                loop {
                    let p = {
                        let mut g = q.list.lock().unwrap();
                        loop {
                            if let Some(p) = g.pop_front() {
                                break p;
                            }
                            g = q.cv.wait(g).unwrap();
                        }
                    };
                    // (released by the title meanwhile)
                    if p.orphan.load(Ordering::Acquire) {
                        drop_unstarted_job(&p);
                        continue;
                    }
                    // Never speculatively compile deprioritized/heavy/BVH/spared pipelines at
                    // THREAD_PRIORITY_LOWEST; leave `p.job` untouched until actually bound.
                    if p.deprioritized.load(Ordering::Acquire)
                        || p.bvh_traversal.load(Ordering::Acquire)
                        || p.spared.load(Ordering::Acquire)
                    {
                        continue;
                    }
                    // Pause speculative preload whenever ANY `wanted` pipeline is actively
                    // queued/in-flight or free RAM/commit headroom is below 6 GiB; leave `p.job`
                    // untouched so binds can take it immediately.
                    while !p.orphan.load(Ordering::Acquire)
                        && !p.demand.lock().unwrap().wanted
                        && p.job.lock().unwrap().is_some()
                    {
                        let avail = mem_throttle_enabled().then(avail_commit_bytes).flatten();
                        let low_mem = should_pause_preload(avail);
                        if !should_pause_preload_with_demand(avail, wanted_active()) {
                            break;
                        }
                        if low_mem {
                            static PAUSED_LOG: std::sync::Once = std::sync::Once::new();
                            PAUSED_LOG.call_once(|| {
                                crate::xlog_info!(
                                    "D3D12.X: pausing speculative PSO preload under low RAM/commit headroom (< 6 GiB, {} MiB free)",
                                    avail.unwrap_or(0) >> 20
                                );
                            });
                        }
                        std::thread::sleep(std::time::Duration::from_millis(25));
                    }
                    if p.orphan.load(Ordering::Acquire) {
                        drop_unstarted_job(&p);
                        continue;
                    }
                    // (None: a bind took it and builds it itself)
                    let job = p.job.lock().unwrap().take();
                    if let Some(job) = job {
                        build(&p, job, true);
                        let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_power_of_two() {
                            crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                        }
                    }
                }
            });
        }
    });
    Some(q)
}

#[cfg(test)]
mod used_tests {
    use super::*;

    /// The shaders of a set are kept when a pipeline of the set is used, each once, and not
    /// before: a set that is only made keeps nothing.
    #[test]
    fn the_shaders_of_a_used_set_are_kept_for_the_next_run() {
        let (set, other) = (0x5e70_0000_0000_0001u64, 0x5e70_0000_0000_0003u64);
        let (vs, ps, cs) = (0x5e70_0000_0000_0011u64, 0x5e70_0000_0000_0013u64, 0x5e70_0000_0000_0015u64);
        shaders_of(set, vec![vs, ps]);
        shaders_of(other, vec![cs]);
        assert!(!used::shader_noted(vs) && !used::shader_noted(ps), "made, not used yet");
        used::note(set);
        assert!(used::shader_noted(vs) && used::shader_noted(ps), "the shaders of the used set");
        assert!(!used::shader_noted(cs), "the other set was not used");
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
        let backlog = 8 * pso_worker_threads();
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
    fn used_note_does_not_block_on_slow_writer() {
        use std::{io::Write, time::Duration};
        struct Blocking(Option<mpsc::Sender<()>>, mpsc::Receiver<()>, Arc<Mutex<Vec<u8>>>);
        impl Write for Blocking {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if let Some(tx) = self.0.take() {
                    let _ = tx.send(());
                    let _ = self.1.recv();
                }
                self.2.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let ((entered_tx, entered_rx), (release_tx, release_rx)) = (mpsc::channel(), mpsc::channel());
        let out = Arc::new(Mutex::new(Vec::new()));
        let j = used::Journal::with_writer(Box::new(Blocking(Some(entered_tx), release_rx, out.clone())), Duration::ZERO);
        j.note(0x1111);
        entered_rx.recv_timeout(Duration::from_secs(5)).expect("writer entered write_all");
        j.note(0x2222);
        j.note(0x3333);
        assert!(j.noted(0x1111) && j.noted(0x2222) && j.noted(0x3333), "note/noted must not block on writer I/O");
        release_tx.send(()).unwrap();
        j.flush();
        assert_eq!(used::parse(std::str::from_utf8(&out.lock().unwrap()).unwrap()), [0x1111, 0x2222, 0x3333].into());
    }

    #[test]
    fn used_burst_coalesces_writes_and_deduplicates_keys() {
        use std::{io::Write, time::Duration};
        struct Counting(Arc<Mutex<(usize, Vec<u8>)>>);
        impl Write for Counting {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                let mut g = self.0.lock().unwrap();
                g.0 += 1;
                g.1.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let rec = Arc::new(Mutex::new((0usize, Vec::new())));
        let j = used::Journal::with_writer(Box::new(Counting(rec.clone())), Duration::from_millis(500));
        for k in (0u64..=40).chain(0u64..=40) {
            j.note(k);
        }
        j.flush();
        let (writes, bytes) = {
            let g = rec.lock().unwrap();
            (g.0, g.1.clone())
        };
        let text = String::from_utf8(bytes).unwrap();
        assert!(writes >= 1 && writes < 40, "expected coalesced writes (< 40), got {writes}");
        assert_eq!(text.lines().count(), 40);
        assert_eq!(used::parse(&text), (1u64..=40).collect());
    }

    #[test]
    fn used_flush_drop_lazy_file_and_io_error_are_graceful() {
        use std::{io::Write, time::Duration};
        let dir = std::env::temp_dir().join(format!("xcl-used-journal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("used-pipelines.txt");
        let j = used::Journal::with_file(path.clone(), Duration::from_secs(5));
        assert!(!dir.exists() && !path.exists(), "dir/file must open lazily on first batch write");
        j.note(0xAA01);
        j.note(0xAA02);
        j.flush();
        assert_eq!(used::parse(&std::fs::read_to_string(&path).unwrap()), [0xAA01, 0xAA02].into());
        j.note(0xAA03);
        drop(j);
        assert_eq!(used::parse(&std::fs::read_to_string(&path).unwrap()), [0xAA01, 0xAA02, 0xAA03].into());
        let _ = std::fs::remove_dir_all(&dir);

        struct Failing(Arc<AtomicUsize>);
        impl Write for Failing {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Err(std::io::Error::other("disk full"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::other("disk full"))
            }
        }
        let attempts = Arc::new(AtomicUsize::new(0));
        let j = used::Journal::with_writer(Box::new(Failing(attempts.clone())), Duration::ZERO);
        j.note(0xBAD1);
        j.note(0xBAD2);
        j.flush();
        j.note(0xBAD3);
        j.flush();
        assert!(attempts.load(Ordering::Acquire) >= 1 && j.noted(0xBAD1) && j.noted(0xBAD2) && j.noted(0xBAD3));
    }

    #[test]
    fn used_static_once_lock_flushes_on_process_exit() {
        use std::time::{Duration, Instant};
        if let Some(path) = std::env::var_os("XCL_TEST_USED_STATIC_PATH") {
            let mode = std::env::var("XCL_TEST_USED_EXIT_MODE").unwrap_or_default();
            match mode.as_str() {
                "normal" => {
                    used::note(0xE001);
                    used::note(0xE002);
                    return;
                }
                "flush" => {
                    used::note(0xE003);
                    flush_used();
                    let text = std::fs::read_to_string(&path).expect("flush_used must flush pending keys");
                    assert_eq!(used::parse(&text), [0xE003].into());
                    return;
                }
                "explicit" => {
                    used::note(0xE004);
                    shutdown_used();
                    let text = std::fs::read_to_string(&path).expect("shutdown_used must flush before exit");
                    assert_eq!(used::parse(&text), [0xE004].into());
                    shutdown_used();
                    flush_used_bounded(Duration::from_millis(50));
                    return;
                }
                "bounded" => {
                    used::note(0xE005);
                    flush_used_bounded(Duration::from_secs(5));
                    let text = std::fs::read_to_string(&path).expect("flush_used_bounded must flush before termination");
                    assert_eq!(used::parse(&text), [0xE005].into());
                    return;
                }
                other => panic!("unexpected mode {other}"),
            }
        }
        let exe = std::env::current_exe().unwrap();
        let dir = std::env::temp_dir().join(format!("xcl-used-exit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for (mode, expected) in [
            ("normal", &[0xE001u64, 0xE002][..]),
            ("flush", &[0xE003u64][..]),
            ("explicit", &[0xE004u64][..]),
            ("bounded", &[0xE005u64][..]),
        ] {
            let path = dir.join(mode).join("used-pipelines.txt");
            let start = Instant::now();
            let status = std::process::Command::new(&exe)
                .args(["--exact", "hle::d3d12x::pso_async::used_tests::used_static_once_lock_flushes_on_process_exit", "--nocapture"])
                .env("XCL_TEST_USED_STATIC_PATH", &path)
                .env("XCL_TEST_USED_EXIT_MODE", mode)
                .status()
                .unwrap();
            assert!(status.success(), "child ({mode}) failed: {status}");
            assert!(start.elapsed() < Duration::from_secs(10), "child ({mode}) waited for coalesce timer");
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("child ({mode}) lost keys on exit: {e}"));
            assert_eq!(used::parse(&text), expected.iter().copied().collect(), "child ({mode}) keys mismatch");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn used_flush_bounded_timeout_and_spawn_failure_fallback_are_sound() {
        use std::{io::Write, time::Duration};
        // 1. Slow writer finishing batch 1 after `flush_bounded` already drained batch 2 must not
        // regress `completed_seq` below `enqueued_seq` (which would hang a later `flush()`).
        struct BlockingOnce(Option<mpsc::Sender<()>>, mpsc::Receiver<()>, Option<mpsc::Sender<()>>, Arc<Mutex<Vec<u8>>>);
        impl Write for BlockingOnce {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if let Some(tx) = self.0.take() {
                    let _ = tx.send(());
                    let _ = self.1.recv_timeout(Duration::from_secs(5));
                }
                self.3.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                if let Some(tx) = self.2.take() {
                    let _ = tx.send(());
                }
                Ok(())
            }
        }
        let ((entered_tx, entered_rx), (release_tx, release_rx), (wrote_tx, wrote_rx)) = (mpsc::channel(), mpsc::channel(), mpsc::channel());
        let out = Arc::new(Mutex::new(Vec::new()));
        let j = Arc::new(used::Journal::with_writer(Box::new(BlockingOnce(Some(entered_tx), release_rx, Some(wrote_tx), out.clone())), Duration::ZERO));
        j.note(0xC001);
        entered_rx.recv_timeout(Duration::from_secs(5)).expect("writer entered batch 1");
        j.note(0xC002);
        j.flush_bounded(Duration::from_millis(20));
        release_tx.send(()).unwrap();
        wrote_rx.recv_timeout(Duration::from_secs(5)).expect("writer finished batch 1");
        std::thread::sleep(Duration::from_millis(25));
        let (done_tx, done_rx) = mpsc::channel();
        let j2 = Arc::clone(&j);
        let flusher = std::thread::spawn(move || {
            j2.flush();
            let _ = done_tx.send(());
        });
        done_rx.recv_timeout(Duration::from_secs(5)).expect("flush() must not hang when completed_seq stays monotonic");
        flusher.join().unwrap();
        drop(j);
        assert!(used::parse(std::str::from_utf8(&out.lock().unwrap()).unwrap()).contains(&0xC001));

        // 2. When writer spawn fails (`writer_stopped == true` with `fallback_path`), `note()`
        // still queues keys so `flush_bounded()` and `shutdown()` persist them directly.
        let dir = std::env::temp_dir().join(format!("xcl-used-nospawn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let p_bounded = dir.join("bounded").join("used-pipelines.txt");
        let j_b = used::Journal::without_writer(p_bounded.clone());
        j_b.note(0xF001);
        j_b.note(0xF002);
        j_b.flush_bounded(Duration::from_millis(20));
        assert_eq!(used::parse(&std::fs::read_to_string(&p_bounded).unwrap()), [0xF001, 0xF002].into());

        let p_shutdown = dir.join("shutdown").join("used-pipelines.txt");
        let j_s = used::Journal::without_writer(p_shutdown.clone());
        j_s.note(0xF003);
        j_s.shutdown();
        assert_eq!(used::parse(&std::fs::read_to_string(&p_shutdown).unwrap()), [0xF003].into());
        let _ = std::fs::remove_dir_all(&dir);
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

    /// What a pipeline's first use tells (a graphics one's description: `gfx_records`) is told
    /// once, at that use, not when it is made or built.
    #[test]
    fn a_first_use_is_told_once() {
        let (standin, key) = (0x7E57_7011usize, 0x5EED_0011u64);
        let told = Arc::new(AtomicUsize::new(0));
        submit_lazy(standin, true, Box::new(|| 0x4331), key);
        let t = told.clone();
        on_first_use(standin, Box::new(move || { t.fetch_add(1, Ordering::SeqCst); }));
        assert_eq!(told.load(Ordering::SeqCst), 0, "not at its making");
        assert_eq!(resolve_wait(standin), 0x4331);
        assert_eq!(told.load(Ordering::SeqCst), 1);
        assert_eq!(resolve(standin), 0x4331);
        assert_eq!(resolve_wait(standin), 0x4331);
        assert_eq!(told.load(Ordering::SeqCst), 1, "once");
        on_first_use(0x7E57_7012, Box::new(|| unreachable!("no such pipeline")));
        released(standin);
    }

    static WORKER_SATURATION_LOCK: Mutex<()> = Mutex::new(());

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

        let n = pso_worker_threads();
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

    /// With `XCL_PSO_WAIT_MS`, a submit's wait for a late pipeline that no worker can start (all
    /// are busy) neither ends with the stand-in nor waits for a free worker: after the budget it
    /// builds the pipeline itself.
    #[test]
    fn a_submit_builds_a_late_pipeline_no_worker_has_started() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::time::{Duration, Instant};
        let n = pso_worker_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let mut blockers = Vec::with_capacity(n);
        for k in 0..n {
            let standin = 0x7E57_A500usize + k;
            let (tx, g) = (started_tx.clone(), gate.clone());
            submit(standin, false, Box::new(move || {
                let _ = tx.send(());
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                0xA500 + k
            }));
            blockers.push(standin);
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("workers saturated");
        }
        let target = 0x7E57_A5FFusize;
        submit(target, true, Box::new(|| 0xA5FF));
        set_wait_budget_for_test(Some(Duration::from_millis(30)));
        assert_eq!(resolve_wait(target), target, "a bind as the title records: past the budget, late");
        let t = Instant::now();
        wait_for_submit(target);
        let took = t.elapsed();
        set_wait_budget_for_test(None);
        let built = resolve(target);
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        for standin in blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }
        let _ = released(target);
        assert_eq!(built, 0xA5FF, "the submit's wait ended without the pipeline");
        assert!(took < Duration::from_secs(5), "the submit waited for a free worker ({took:?})");
    }

    /// With `XCL_PSO_WAIT_MS`, a submit waits for a pipeline a worker is building at most
    /// `SUBMIT_WAIT_MOST`, then goes on (with the stand-in) and counts it.
    #[test]
    fn a_submit_waits_for_a_build_on_a_worker_no_longer_than_its_bound() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::time::{Duration, Instant};
        let standin = 0x7E57_A600usize;
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let g = gate.clone();
        submit(standin, true, Box::new(move || {
            let _ = started_tx.send(());
            let mut open = g.0.lock().unwrap();
            while !*open {
                open = g.1.wait(open).unwrap();
            }
            0xA600
        }));
        started_rx.recv_timeout(Duration::from_secs(10)).expect("a worker builds it");
        set_wait_budget_for_test(Some(Duration::from_millis(20)));
        set_submit_wait_for_test(Some(Duration::from_millis(300)));
        let gave_up = SUBMIT_GAVE_UP.load(Ordering::Relaxed);
        let t = Instant::now();
        wait_for_submit(standin);
        let took = t.elapsed();
        let pending = is_pending(standin);
        let counted = SUBMIT_GAVE_UP.load(Ordering::Relaxed) > gave_up;
        set_wait_budget_for_test(None);
        set_submit_wait_for_test(None);
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        let start = Instant::now();
        while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(2));
        }
        let _ = released(standin);
        assert!(pending, "the build ended before the bound");
        assert!(took >= Duration::from_millis(300) && took < Duration::from_secs(5), "the submit waited {took:?}");
        assert!(counted, "the submit that went on without its pipeline was not counted");
    }

    /// A build that panics on a worker ends as a failed one: binds and submits waiting for it go on.
    #[test]
    fn a_build_that_panics_on_a_worker_ends_as_failed() {
        use std::time::{Duration, Instant};
        let standin = 0x7E57_A700usize;
        submit(standin, true, Box::new(|| panic!("a build that panics (test)")));
        let start = Instant::now();
        while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(2));
        }
        let pending = is_pending(standin);
        let _ = released(standin);
        assert!(!pending, "the panicked build never ended");
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
        let n = pso_worker_threads();
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
        let m = preload_worker_threads();
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
    PENDING.read().unwrap().as_ref()?.get(&pso).cloned()
}

impl Pending {
    /// What a bind of the stand-in `pso` gives the driver while the pipeline is not built.
    fn stand(&self, pso: usize) -> usize {
        match self.driver.load(Ordering::Acquire) {
            0 => pso,
            d => d,
        }
    }
}

/// Register a permanently gated stand-in pipeline (e.g. RT compute held back under
/// `XCL_RT_GATE=hold`) with `done = true`, `job = None`, `sent = true`, and `real = 0`.
/// Never enqueues a worker job, never translates or calls `CreateComputePipelineState`, maps
/// `standin -> driver` cleanly via `stand()` for `SetPipelineState`/`resolve`/`resolve_wait`,
/// reports `is_pending(standin) == false`, and releases cleanly via `released()`.
pub fn submit_gated(standin: usize, driver: usize) {
    let p = Arc::new(Pending {
        job: Mutex::new(None),
        sent: std::sync::atomic::AtomicBool::new(true),
        real: AtomicUsize::new(0),
        driver: AtomicUsize::new(driver),
        done: Mutex::new(true),
        cv: Condvar::new(),
        compute: true,
        key: 0,
        used: Default::default(),
        orphan: Default::default(),
        late: std::sync::atomic::AtomicBool::new(true),
        spared: std::sync::atomic::AtomicBool::new(false),
        bvh_traversal: std::sync::atomic::AtomicBool::new(true),
        deprioritized: std::sync::atomic::AtomicBool::new(true),
        waited_frame: AtomicU64::new(UNWAITED),
        wanted_state: AtomicU8::new(WANTED_DONE),
        demand: Default::default(),
        on_use: Default::default(),
    });
    PENDING.write().unwrap().get_or_insert_with(HashMap::new).insert(standin, p);
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
    p.note_use();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers (unless RT-gated while BVH traversal is
            // held back, so gated RT shaders never enqueue or burn driver compile RAM)
            let stand = p.stand(pso);
            if p.bvh_traversal.load(Ordering::Acquire) && !super::real::bvh_traversal_allowed() {
                return stand;
            }
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue(p);
            }
            stand
        }
        r => r,
    }
}

/// `resolve`, waiting for a pipeline still being built:
/// - non-spared memory-writing producer compute pipelines always wait (never skipped by
///   `XCL_PSO_FRAME_BUDGET_MS`, because skipping a producer compute dispatch corrupts title memory);
/// - graphics pipelines wait within the per-frame graphics window (`XCL_PSO_FRAME_BUDGET_MS`,
///   default 12 s) and stamp `waited_frame` so a slow graphics PSO stalls at most one frame;
/// - `spared` (`image_only`) and `bvh_traversal` compute pipelines use `resolve_spared`.
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_for(pso, wait_budget())
}

/// A call put off to the submit is recorded now: wait for its compute pipeline to be built,
/// whether a bind already found it `late` or not. Its segment lies between the list's other work,
/// whose labels and fence signals the title reads as "done": recorded with the stand-in, the
/// dispatch is left out while they still fire. The submitting thread is held for that time
/// (`title_held`), as a bind's wait holds it. A build that fails still gives the stand-in, and a
/// program binds do not wait for (`image_only`) is not waited for.
///
/// Unset `XCL_PSO_WAIT_MS`: until the pipeline is built, as a bind waits. Set: the build is left
/// to the workers for that long first, as a bind leaves it (their turns for big compiles); one no
/// worker has started by then is built here. One a worker builds is waited for at most
/// `SUBMIT_WAIT_MOST`, and the call then goes on with the stand-in (logged and counted):
/// `title_held` stops only `kernelx::Sleep` and the GPU-wait watchdog, not every clock a title
/// reads.
pub fn wait_for_submit(pso: usize) {
    let Some(budget) = wait_budget() else {
        let _ = resolve_wait_for(pso, None);
        return;
    };
    let Some(p) = entry(pso) else { return };
    if !p.compute || p.spared.load(Ordering::Acquire) || *p.done.lock().unwrap() {
        return;
    }
    p.note_use();
    let _held = Held::new();
    let _prof = super::prof::span(super::prof::Kind::ComputePipelineWait);
    if !p.sent.swap(true, Ordering::AcqRel) {
        enqueue(p.clone());
    }
    p.want();
    let built = |p: &Pending, most: std::time::Duration| *p.cv.wait_timeout_while(p.done.lock().unwrap(), most, |d| !*d).unwrap().0;
    if built(&p, budget) {
        return;
    }
    // no worker has started it: built here, as without the budget
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        return complete(&p, job());
    }
    let most = submit_wait_most();
    if !built(&p, most) {
        let n = SUBMIT_GAVE_UP.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 16 || n.is_power_of_two() {
            crate::xlog_warn!("D3D12.X: deferred dispatch skipped after waiting {} s for its pipeline ({n} so far)", most.as_secs_f32());
        }
    }
}

/// put-off calls that went on with the stand-in after `SUBMIT_WAIT_MOST`
static SUBMIT_GAVE_UP: AtomicUsize = AtomicUsize::new(0);

/// How long a submit waits for a pipeline a worker builds, with `XCL_PSO_WAIT_MS` set: as long as
/// a retail title's watchdog lets its main thread stand still (30 s, #891).
const SUBMIT_WAIT_MOST: std::time::Duration = std::time::Duration::from_secs(30);

fn submit_wait_most() -> std::time::Duration {
    #[cfg(test)]
    if let Some(b) = SUBMIT_WAIT_FOR_TEST.with(|b| b.get()) {
        return b;
    }
    SUBMIT_WAIT_MOST
}

#[cfg(test)]
thread_local! {
    /// `SUBMIT_WAIT_MOST` for the submits of this test thread
    static SUBMIT_WAIT_FOR_TEST: std::cell::Cell<Option<std::time::Duration>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn set_submit_wait_for_test(most: Option<std::time::Duration>) {
    SUBMIT_WAIT_FOR_TEST.with(|b| b.set(most));
}

/// `resolve_wait`, waiting at most `budget` per pipeline (and for graphics pipelines, at most
/// the remaining per-frame graphics window).
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    if p.spared.load(Ordering::Acquire) || p.bvh_traversal.load(Ordering::Acquire) {
        return resolve_spared(&p, pso);
    }
    let frame = super::present::presents();
    // Memory-writing producer compute PSOs (`!spared && !bvh_traversal`) are never skipped by the
    // per-frame graphics wait window: skipping a producer dispatch leaves title memory/tables
    // uninitialized and causes Window-3 pointer corruption several frames later.
    let fcap = frame_budget_for_kind(frame_budget_policy(), p.compute, frame);
    resolve_wait_in(&p, pso, budget, fcap, frame, &FRAME_BUDGET_GFX)
}

/// Default per-frame graphics pipeline wait window (`12 s`).
const DEFAULT_FRAME_BUDGET_MS: u64 = 12_000;
/// Presented-frame threshold after which an opt-in `XCL_GFX_PSO_FRAME_BUDGET_MS` cap takes effect
/// for graphics-only bind waits, leaving initial boot/menu frames at `DEFAULT_FRAME_BUDGET_MS`.
const GAMEPLAY_FRAME_BUDGET_START_FRAME: u64 = 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameBudgetPolicy {
    /// Default policy when neither env var is set: keeps `DEFAULT_FRAME_BUDGET_MS` (`12 s`)
    /// across all frames so cold graphics pipelines are never prematurely skipped during loading.
    Default,
    /// Explicit override via `XCL_PSO_FRAME_BUDGET_MS`: applies uniformly across all frames.
    Explicit(Option<std::time::Duration>),
    /// Opt-in override via `XCL_GFX_PSO_FRAME_BUDGET_MS`: keeps `DEFAULT_FRAME_BUDGET_MS` while
    /// `frame < GAMEPLAY_FRAME_BUDGET_START_FRAME`, then caps graphics-only bind waits.
    OptInGameplay(Option<std::time::Duration>),
}

fn parse_budget_override(val: Option<&str>) -> Option<Option<std::time::Duration>> {
    let raw = val.map(str::trim).filter(|s| !s.is_empty())?;
    if raw.eq_ignore_ascii_case("off") || raw.eq_ignore_ascii_case("none") {
        return Some(None);
    }
    raw.parse::<u64>().ok().map(|ms| Some(std::time::Duration::from_millis(ms)))
}

/// Parses `XCL_PSO_FRAME_BUDGET_MS` (uniform override) and `XCL_GFX_PSO_FRAME_BUDGET_MS`
/// (opt-in post-boot graphics cap) into a `FrameBudgetPolicy`.
fn parse_frame_budget_policy_from(pso_env: Option<&str>, gfx_env: Option<&str>) -> FrameBudgetPolicy {
    if let Some(cap) = parse_budget_override(pso_env) {
        return FrameBudgetPolicy::Explicit(cap);
    }
    if let Some(cap) = parse_budget_override(gfx_env) {
        return FrameBudgetPolicy::OptInGameplay(cap);
    }
    FrameBudgetPolicy::Default
}

/// Effective graphics pipeline wait window for `frame` under `policy`.
fn frame_budget_for(policy: FrameBudgetPolicy, frame: u64) -> Option<std::time::Duration> {
    match policy {
        FrameBudgetPolicy::Default => Some(std::time::Duration::from_millis(DEFAULT_FRAME_BUDGET_MS)),
        FrameBudgetPolicy::Explicit(cap) => cap,
        FrameBudgetPolicy::OptInGameplay(cap) => {
            if frame < GAMEPLAY_FRAME_BUDGET_START_FRAME {
                Some(std::time::Duration::from_millis(DEFAULT_FRAME_BUDGET_MS))
            } else {
                cap
            }
        }
    }
}

/// Effective per-frame wait cap for a pipeline bind: producer compute pipelines are never capped.
fn frame_budget_for_kind(policy: FrameBudgetPolicy, compute: bool, frame: u64) -> Option<std::time::Duration> {
    if compute { None } else { frame_budget_for(policy, frame) }
}

/// Parses `XCL_PSO_FRAME_BUDGET_MS` for frame 0 (boot/default): unset/empty defaults to
/// `12_000 ms` (`12 s`), `0` skips waiting immediately, `off` / `none` disables the per-frame cap,
/// and a positive integer sets the per-frame graphics wait window in milliseconds.
#[cfg(test)]
fn parse_frame_budget(val: Option<&str>) -> Option<std::time::Duration> {
    frame_budget_for(parse_frame_budget_policy_from(val, None), 0)
}

fn frame_budget_policy() -> FrameBudgetPolicy {
    static B: OnceLock<FrameBudgetPolicy> = OnceLock::new();
    *B.get_or_init(|| {
        let pso_env = std::env::var("XCL_PSO_FRAME_BUDGET_MS").ok();
        let gfx_env = std::env::var("XCL_GFX_PSO_FRAME_BUDGET_MS").ok();
        parse_frame_budget_policy_from(pso_env.as_deref(), gfx_env.as_deref())
    })
}

static FRAME_BUDGET_GFX: Mutex<FrameWaitBudget> = Mutex::new(FrameWaitBudget::new());

/// Waits for `p` within both an optional per-PSO cap (`budget`) and an optional per-frame
/// graphics window (`frame_cap` tracked in `frame_state` for `frame`). Builds run on
/// `xcl-pso-*` workers (prioritized as `wanted`) so the caller always wakes when the window
/// expires; once a graphics PSO has spent budget on a frame (`waited_frame`), subsequent frames
/// skip waiting on it immediately until workers complete it. Non-spared memory-writing producer
/// compute pipelines pass `frame_cap = None` and always wait to completion.
fn resolve_wait_in(
    p: &Arc<Pending>,
    pso: usize,
    budget: Option<std::time::Duration>,
    frame_cap: Option<std::time::Duration>,
    frame: u64,
    frame_state: &Mutex<FrameWaitBudget>,
) -> usize {
    let built = |p: &Pending| match p.real.load(Ordering::Acquire) {
        0 => p.stand(pso),
        r => r,
    };
    let r = p.real.load(Ordering::Acquire);
    if r != 0 {
        return r;
    }
    if *p.done.lock().unwrap() {
        return built(p);
    }
    if budget.is_some() && p.late.load(Ordering::Acquire) {
        if !p.sent.swap(true, Ordering::AcqRel) {
            enqueue(p.clone());
        }
        return p.stand(pso);
    }
    // Recursive resolve on a worker thread: run inline to avoid worker-pool deadlock.
    if building_here() {
        let job = p.job.lock().unwrap().take();
        if let Some(job) = job {
            complete(p, job());
        }
        p.want();
        let mut done = p.done.lock().unwrap();
        while !*done {
            done = p.cv.wait(done).unwrap();
        }
        return built(p);
    }
    // Mark wanted BEFORE enqueueing so `PsoQueue::pop` jumps this pipeline ahead of background
    // and deprioritized builds and preload workers raise priority immediately.
    p.want();
    if !p.sent.swap(true, Ordering::AcqRel) {
        enqueue(p.clone());
    }
    let Some(fcap) = frame_cap else {
        let _held = Held::new();
        let _prof = super::prof::span(if p.compute { super::prof::Kind::ComputePipelineWait } else { super::prof::Kind::GraphicsPipelineWait });
        if let Some(b) = budget {
            return resolve_within(p, pso, b);
        }
        if !p.deprioritized.load(Ordering::Acquire) {
            let job = p.job.lock().unwrap().take();
            if let Some(job) = job {
                complete(p, job());
            }
        }
        let mut done = p.done.lock().unwrap();
        while !*done {
            done = p.cv.wait(done).unwrap();
        }
        return built(p);
    };
    // A slow graphics PSO that already got its wait turn on an earlier frame keeps compiling in
    // the background without re-burning `frame`'s graphics wait window every frame.
    let prev_waited = p.waited_frame.load(Ordering::Acquire);
    if prev_waited != UNWAITED && prev_waited != frame {
        return p.stand(pso);
    }
    let rem = frame_state.lock().unwrap().begin_wait(frame, fcap, std::time::Instant::now());
    if rem.is_zero() {
        return p.stand(pso);
    }
    let _ = p.waited_frame.compare_exchange(UNWAITED, frame, Ordering::AcqRel, Ordering::Acquire);
    let (wait_for, per_pso_binding) = match budget {
        Some(b) if b <= rem => (b, true),
        _ => (rem, false),
    };
    let _held = Held::new();
    let _prof = super::prof::span(if p.compute { super::prof::Kind::ComputePipelineWait } else { super::prof::Kind::GraphicsPipelineWait });
    let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), wait_for, |d| !*d).unwrap();
    let finished = *done;
    drop(done);
    let frame_timed_out = !finished && !per_pso_binding;
    frame_state.lock().unwrap().end_wait(frame, fcap, std::time::Instant::now(), frame_timed_out);
    if finished {
        return built(p);
    }
    if per_pso_binding {
        p.late.store(true, Ordering::Release);
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 16 || n.is_power_of_two() {
            crate::xlog_warn!(
                "D3D12.X: a pipeline still being built after {} ms: its binds go on without it until it is built ({} so far)",
                wait_for.as_millis(),
                n + 1
            );
        }
    } else {
        static N_GFX: AtomicUsize = AtomicUsize::new(0);
        let n = N_GFX.fetch_add(1, Ordering::Relaxed);
        if n < 16 || n.is_power_of_two() {
            crate::xlog_warn!(
                "D3D12.X: graphics pipeline wait window ({} ms) spent on frame {frame}: skipping remaining unready graphics work until next frame ({} so far)",
                fcap.as_millis(),
                n + 1
            );
        }
    }
    p.stand(pso)
}

/// How long a bind waits for a pipeline still being built (`XCL_PSO_WAIT_MS`; unset or 0: until
/// it is built). A retail game builds compute programs on its own threads: the driver takes up to
/// 30 s for the biggest translated ones, and the title's watchdog ends it when its main thread
/// has been held 30 s. It bounds binds as the title records; a call put off to the submit waits
/// there until its pipeline is built (`wait_for_submit`).
fn wait_budget() -> Option<std::time::Duration> {
    #[cfg(test)]
    if let Some(b) = BUDGET_FOR_TEST.with(|b| b.get()) {
        return Some(b);
    }
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0).map(std::time::Duration::from_millis))
}

#[cfg(test)]
thread_local! {
    /// `XCL_PSO_WAIT_MS` for the binds of this test thread
    static BUDGET_FOR_TEST: std::cell::Cell<Option<std::time::Duration>> = const { std::cell::Cell::new(None) };
}

/// Binds on this thread wait at most `budget` (None: as `XCL_PSO_WAIT_MS` says).
#[cfg(test)]
pub(super) fn set_wait_budget_for_test(budget: Option<std::time::Duration>) {
    BUDGET_FOR_TEST.with(|b| b.set(budget));
}

/// Binds wait at most `XCL_PSO_WAIT_MS`: a dispatch is then not put off to the submit
/// (`list_x::defer`), where it would only wait as long.
pub fn waits_bounded() -> bool {
    wait_budget().is_some()
}

/// `pso` is a stand-in marked `spared` (`image_only`) or `bvh_traversal`, whose dispatches do
/// not block or defer to submit segments.
pub fn is_spared(pso: usize) -> bool {
    entry(pso).is_some_and(|p| p.spared.load(Ordering::Acquire) || p.bvh_traversal.load(Ordering::Acquire))
}

/// `resolve_wait` for at most `budget`: the build goes to the workers (not this thread), and a
/// pipeline not built by then is `late` – it and later binds of it get the stand-in (their
/// dispatches are skipped) until a worker has built it.
fn resolve_within(p: &Arc<Pending>, pso: usize, budget: std::time::Duration) -> usize {
    let built = |p: &Pending| match p.real.load(Ordering::Acquire) {
        0 => p.stand(pso),
        r => r,
    };
    if *p.done.lock().unwrap() {
        return built(p);
    }
    if !p.sent.swap(true, Ordering::AcqRel) {
        enqueue(p.clone());
    }
    if p.late.load(Ordering::Acquire) {
        return p.stand(pso);
    }
    p.want();
    let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), budget, |d| !*d).unwrap();
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

/// Whether heavyweight compute shaders (`>= 512 KiB` DXIL / `>= 24 KiB` console code) compile
/// asynchronously off title threads (default on, `XCL_ASYNC_HEAVY_COMPUTE=0` to disable).
pub fn async_heavy_compute() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_ASYNC_HEAVY_COMPUTE").is_ok_and(|v| v.trim() == "0"))
}

/// The compute pipeline behind stand-in `standin` is a heavyweight shader (`>= 24 KiB` console
/// code): never compile it synchronously on title threads — queue it deprioritized behind
/// normal/MS PSOs so `xcl-pso-*` workers compile it under `BIG_COMPILES` gating. Does NOT mark
/// `spared`, so non-RT producer compute shaders that write memory are NEVER skipped (call
/// `image_only` or `bvh_traversal` separately only for pure image-writing or BVH RT shaders).
pub fn heavy_compute(standin: usize) {
    if let Some(p) = entry(standin) {
        p.deprioritized.store(true, Ordering::Release);
    }
}

/// The compute pipeline behind stand-in `standin` issues `image_bvh*_intersect_ray`: binds never
/// compile it synchronously on title threads; queue it deprioritized behind normal/MS PSOs.
pub fn bvh_traversal(standin: usize) {
    if let Some(p) = entry(standin) {
        p.deprioritized.store(true, Ordering::Release);
        p.bvh_traversal.store(true, Ordering::Release);
    }
}

/// Per-frame wait cap for non-critical spared compute PSOs (`XCL_PSO_FRAME_WAIT_MS`, default 24 ms,
/// `0` = never wait). Lets fast warm `.plib` hits finish on their first bind without stalling
/// `PresentX` past 24 ms/frame on cold compiles; never applies to memory-writing producer shaders.
fn frame_wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| {
        let ms = std::env::var("XCL_PSO_FRAME_WAIT_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(24);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    })
}

struct FrameWaitBudget {
    frame: u64,
    spent: std::time::Duration,
    active_waiters: usize,
    active_since: Option<std::time::Instant>,
}

impl FrameWaitBudget {
    const fn new() -> Self {
        Self { frame: 0, spent: std::time::Duration::ZERO, active_waiters: 0, active_since: None }
    }

    fn advance(&mut self, frame: u64, now: std::time::Instant) {
        if frame > self.frame {
            self.frame = frame;
            self.spent = std::time::Duration::ZERO;
            self.active_since = (self.active_waiters > 0).then_some(now);
        }
    }

    fn effective_spent(&self, now: std::time::Instant) -> std::time::Duration {
        let in_flight = self.active_since.map_or(std::time::Duration::ZERO, |s| now.saturating_duration_since(s));
        self.spent.saturating_add(in_flight)
    }

    #[cfg(test)]
    fn remaining_at(&mut self, frame: u64, cap: std::time::Duration, now: std::time::Instant) -> std::time::Duration {
        self.advance(frame, now);
        cap.saturating_sub(self.effective_spent(now))
    }

    /// Reserve a wait slot on `frame`: tracks wall-clock overlap across concurrent threads so
    /// staggered callers share one hard `cap` ceiling per frame and can never cascade past `cap`.
    fn begin_wait(&mut self, frame: u64, cap: std::time::Duration, now: std::time::Instant) -> std::time::Duration {
        self.advance(frame, now);
        let rem = cap.saturating_sub(self.effective_spent(now));
        if rem.is_zero() {
            return std::time::Duration::ZERO;
        }
        if self.active_waiters == 0 {
            self.active_since = Some(now);
        }
        self.active_waiters += 1;
        rem
    }

    fn end_wait(&mut self, frame: u64, cap: std::time::Duration, now: std::time::Instant, frame_timed_out: bool) {
        self.advance(frame, now);
        if self.active_waiters > 0 {
            self.active_waiters -= 1;
            if self.active_waiters == 0 {
                if let Some(since) = self.active_since.take() {
                    self.spent = self.spent.saturating_add(now.saturating_duration_since(since));
                }
            }
        }
        if frame_timed_out && frame == self.frame {
            self.spent = self.spent.max(cap);
        }
    }

    #[cfg(test)]
    fn remaining(&mut self, frame: u64, cap: std::time::Duration) -> std::time::Duration {
        self.remaining_at(frame, cap, std::time::Instant::now())
    }

    #[cfg(test)]
    fn record(&mut self, frame: u64, elapsed: std::time::Duration) {
        if frame > self.frame {
            self.frame = frame;
            self.spent = elapsed;
        } else if frame == self.frame {
            self.spent = self.spent.saturating_add(elapsed);
        }
    }
}

static FRAME_WAIT: Mutex<FrameWaitBudget> = Mutex::new(FrameWaitBudget::new());

/// `resolve_wait` for a pipeline binds do not wait for (`image_only` or `bvh_traversal`): the real
/// one once built, else the stand-in, with the build sent to the workers. For non-BVH `spared`
/// pipelines (e.g. pure image-only shaders), waits up to the remaining per-frame `frame_wait_budget`
/// (`XCL_PSO_FRAME_WAIT_MS`, default 24 ms) so fast warm `.plib` loads complete on first bind
/// without ever freezing `PresentX`. Nobody marks the build `wanted`, so cold big compiles stay
/// strictly behind critical work and commit headroom.
fn resolve_spared(p: &Arc<Pending>, pso: usize) -> usize {
    match p.real.load(Ordering::Acquire) {
        0 => {
            if p.bvh_traversal.load(Ordering::Acquire) && !super::real::bvh_traversal_allowed() {
                return p.stand(pso);
            }
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue(p.clone());
            }
            if !p.bvh_traversal.load(Ordering::Acquire) && !p.late.load(Ordering::Acquire) {
                if let Some(cap) = frame_wait_budget() {
                    let frame = super::present::presents();
                    let rem = FRAME_WAIT.lock().unwrap().begin_wait(frame, cap, std::time::Instant::now());
                    if !rem.is_zero() {
                        let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), rem, |d| !*d).unwrap();
                        let finished = *done;
                        drop(done);
                        FRAME_WAIT.lock().unwrap().end_wait(frame, cap, std::time::Instant::now(), !finished);
                        if finished {
                            return match p.real.load(Ordering::Acquire) {
                                0 => p.stand(pso),
                                r => r,
                            };
                        }
                    }
                }
            }
            if !*p.done.lock().unwrap() && !p.late.swap(true, Ordering::AcqRel) {
                static N: AtomicUsize = AtomicUsize::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                if n < 16 || n.is_power_of_two() {
                    let kind = if p.bvh_traversal.load(Ordering::Acquire) {
                        "a BVH ray-tracing compute program"
                    } else {
                        "a compute program that writes only images"
                    };
                    crate::xlog_info!("D3D12.X: {kind} is still being built: its dispatches are left out until it is ({} so far)", n + 1);
                }
            }
            p.stand(pso)
        }
        r => r,
    }
}

/// For binding: both graphics and compute pipelines wait within their own per-frame wait window
/// (`XCL_PSO_FRAME_BUDGET_MS`, default 12 s per kind per frame). Once a frame's window for that
/// kind is spent, remaining unready draws/dispatches get the stand-in and retry on the next frame.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(_) => resolve_wait(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.write().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    {
        let done = p.done.lock().unwrap();
        if *done {
            return p.real.swap(0, Ordering::AcqRel);
        }
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
    }
    drop_unstarted_job(&p);
    0
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// `pso` is a stand-in with no real pipeline yet (still building or build failed: draws nothing).
pub fn is_stand_in(pso: usize) -> bool {
    entry(pso).is_some_and(|p| p.real.load(Ordering::Acquire) == 0)
}

/// Whether `bound` (returned by `resolve*` for `standin`) is an unbuilt stand-in rather than the
/// real pipeline — remains `true` even if a worker finishes building `standin` between `resolve*`
/// and the caller's check (`#2598`), while returning `false` if the build failed (`real == 0`).
pub fn bound_unbuilt(standin: usize, bound: usize) -> bool {
    entry(standin).is_some_and(|p| {
        if !*p.done.lock().unwrap() {
            return true;
        }
        let real = p.real.load(Ordering::Acquire);
        real != 0 && bound != real
    })
}

/// One of the title's threads is about to wait for `pso`.
pub fn want(pso: usize) {
    if let Some(p) = entry(pso) {
        p.want();
    }
}

/// A call with `pso` was put off to the submit: its build goes before the workers' queue
/// and is immediately marked wanted so DXC / big-compile gates and preload threads prioritize it.
pub fn need_by_submit(pso: usize) {
    let Some(p) = entry(pso) else { return };
    p.want();
    if !std::mem::replace(&mut p.demand.lock().unwrap().by_submit, true) {
        NEEDED.lock().unwrap().push_back(p);
        queue().push(Box::new(|| ()), None);
    }
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.read().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}

#[cfg(test)]
mod pso_queue_tests {
    use super::*;

    fn dummy_pending(wanted: bool, deprioritized: bool) -> Arc<Pending> {
        Arc::new(Pending {
            job: Mutex::new(None),
            sent: std::sync::atomic::AtomicBool::new(true),
            real: AtomicUsize::new(0),
            driver: AtomicUsize::new(0),
            done: Mutex::new(false),
            cv: Condvar::new(),
            compute: false,
            key: 0,
            used: Default::default(),
            orphan: Default::default(),
            late: Default::default(),
            spared: Default::default(),
            bvh_traversal: Default::default(),
            deprioritized: std::sync::atomic::AtomicBool::new(deprioritized),
            waited_frame: AtomicU64::new(UNWAITED),
            wanted_state: AtomicU8::new(WANTED_NONE),
            demand: Mutex::new(Demand { wanted, by_submit: false, preload_thread: 0 }),
            on_use: Default::default(),
        })
    }

    fn dummy_compute_pending(wanted: bool, spared: bool) -> Arc<Pending> {
        Arc::new(Pending {
            job: Mutex::new(None),
            sent: std::sync::atomic::AtomicBool::new(true),
            real: AtomicUsize::new(0),
            driver: AtomicUsize::new(0),
            done: Mutex::new(false),
            cv: Condvar::new(),
            compute: true,
            key: 0,
            used: Default::default(),
            orphan: Default::default(),
            late: Default::default(),
            spared: std::sync::atomic::AtomicBool::new(spared),
            bvh_traversal: Default::default(),
            deprioritized: Default::default(),
            waited_frame: AtomicU64::new(UNWAITED),
            wanted_state: AtomicU8::new(WANTED_NONE),
            demand: Mutex::new(Demand { wanted, by_submit: false, preload_thread: 0 }),
            on_use: Default::default(),
        })
    }

    #[test]
    fn pso_queue_prioritizes_wanted_producer_compute_over_wanted_graphics() {
        let q = PsoQueue::new();
        let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        let push = |name: &'static str, p: Arc<Pending>| {
            let o = order.clone();
            q.push(Box::new(move || o.lock().unwrap().push(name)), Some(p));
        };
        push("gfx_wanted", dummy_pending(true, false));
        push("cs_spared_wanted", dummy_compute_pending(true, true));
        push("cs_producer_wanted", dummy_compute_pending(true, false));
        for _ in 0..3 {
            (q.pop())();
        }
        assert_eq!(*order.lock().unwrap(), ["cs_producer_wanted", "gfx_wanted", "cs_spared_wanted"]);
    }

    #[test]
    fn pso_queue_dequeues_wanted_then_normal_ahead_of_deprioritized() {
        let q = PsoQueue::new();
        let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        let push = |name: &'static str, p: Arc<Pending>| {
            let o = order.clone();
            q.push(Box::new(move || o.lock().unwrap().push(name)), Some(p));
        };
        let heavy1 = dummy_pending(false, true);
        let bg1 = dummy_pending(false, false);
        let promoted = dummy_pending(false, false);
        let wanted_now = dummy_pending(true, false);
        let heavy2 = dummy_pending(false, true);
        push("heavy1", heavy1);
        push("bg1", bg1);
        push("promoted", promoted.clone());
        push("wanted", wanted_now);
        push("heavy2", heavy2);
        promoted.demand.lock().unwrap().wanted = true;
        q.notify_wanted();
        for _ in 0..5 {
            (q.pop())();
        }
        assert_eq!(*order.lock().unwrap(), ["promoted", "wanted", "bg1", "heavy1", "heavy2"]);
    }

    #[test]
    fn heavy_compute_producer_runs_on_worker_and_is_never_skipped() {
        let standin = 0x7E57_B0C1usize;
        let (tx, rx) = mpsc::channel();
        submit_lazy(standin, true, Box::new(move || {
            let on_worker = std::thread::current().name().is_some_and(|n| n.starts_with("xcl-pso-"));
            let _ = tx.send(on_worker);
            0x5C5C
        }), 0);
        heavy_compute(standin);
        // Must wait for the worker and return the real PSO (never skip a memory-writing producer).
        assert_eq!(resolve_wait_for(standin, None), 0x5C5C);
        assert!(rx.recv_timeout(std::time::Duration::from_secs(10)).expect("ran on worker"), "heavy producer must compile on xcl-pso worker");
        let _ = released(standin);
    }

    #[test]
    fn commit_headroom_throttles_workers_and_pauses_preload() {
        assert_eq!(PRELOAD_PAUSE_COMMIT, 6 << 30);
        assert_eq!(WORKER_CAP_2_COMMIT, 5 << 30);
        assert_eq!(WORKER_CAP_1_COMMIT, 3 << 30);

        assert_eq!(worker_cap_for_commit(None, 6), 6);
        assert_eq!(worker_cap_for_commit(Some(7 << 30), 6), 6);
        assert_eq!(worker_cap_for_commit(Some(WORKER_CAP_2_COMMIT), 6), 6);
        assert_eq!(worker_cap_for_commit(Some(WORKER_CAP_2_COMMIT - 1), 6), 2);
        assert_eq!(worker_cap_for_commit(Some(WORKER_CAP_1_COMMIT), 6), 2);
        assert_eq!(worker_cap_for_commit(Some(WORKER_CAP_1_COMMIT - 1), 6), 1);
        assert_eq!(worker_cap_for_commit(Some(0), 6), 1, "must never drop to 0 workers");
        assert_eq!(worker_cap_for_commit(Some(WORKER_CAP_2_COMMIT - 1), 1), 1);

        assert!(!should_pause_preload(None));
        assert!(!should_pause_preload(Some(PRELOAD_PAUSE_COMMIT)));
        assert!(should_pause_preload(Some(PRELOAD_PAUSE_COMMIT - 1)));
        assert!(should_pause_preload(Some(0)));

        // Any active wanted pipeline pauses speculative preload even with ample free RAM.
        assert!(!should_pause_preload_with_demand(Some(16 << 30), 0));
        assert!(should_pause_preload_with_demand(Some(16 << 30), 1));
        assert!(should_pause_preload_with_demand(None, 2));
        assert!(should_pause_preload_with_demand(Some(PRELOAD_PAUSE_COMMIT - 1), 0));

        // Preload worker pool stays tightly bounded (1..=2).
        let m = preload_worker_threads();
        assert!((1..=2).contains(&m), "expected preload_worker_threads in 1..=2, got {m}");

        // Active wanted counter transitions cleanly without leaking on complete/drop.
        let p = dummy_pending(false, false);
        p.mark_wanted_active();
        p.mark_wanted_active(); // idempotent
        assert!(wanted_active() >= 1);
        p.clear_wanted_active();
        p.clear_wanted_active(); // idempotent
        p.mark_wanted_active(); // post-DONE must not re-increment
        assert_eq!(p.wanted_state.load(Ordering::Acquire), WANTED_DONE);
    }

    #[test]
    fn heavy_and_bvh_pipelines_are_not_speculatively_preloaded() {
        let heavy_standin = 0x7E57_B0D1usize;
        let heavy_key = 0x5EED_B0D1u64;
        used::earlier_run_used(heavy_key);
        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ran_job = ran.clone();
        submit_lazy_with_flags(
            heavy_standin,
            true,
            Box::new(move || {
                ran_job.store(true, Ordering::Release);
                0x5D5D
            }),
            heavy_key,
            SubmitFlags { spared: false, bvh_traversal: false, deprioritized: true },
        );
        std::thread::sleep(std::time::Duration::from_millis(80));
        assert!(!ran.load(Ordering::Acquire), "heavy deprioritized PSO must not be speculatively preloaded");
        assert!(is_pending(heavy_standin));
        // Once actually bound, resolve_wait compiles it on a worker and returns the real PSO.
        assert_eq!(resolve_wait_for(heavy_standin, None), 0x5D5D);
        assert!(ran.load(Ordering::Acquire));
        let _ = released(heavy_standin);
    }

    #[test]
    fn frame_wait_budget_caps_per_frame_and_resets_on_new_frame() {
        let cap = std::time::Duration::from_millis(24);
        let mut b = FrameWaitBudget::new();
        assert_eq!(b.remaining(1, cap), cap);
        b.record(1, std::time::Duration::from_millis(10));
        assert_eq!(b.remaining(1, cap), std::time::Duration::from_millis(14));
        b.record(1, std::time::Duration::from_millis(20));
        assert_eq!(b.remaining(1, cap), std::time::Duration::ZERO);
        // Advancing to frame 2 resets the budget cleanly.
        assert_eq!(b.remaining(2, cap), cap);
        // Stale lower frame number must not clear frame 2's spent budget.
        b.record(2, std::time::Duration::from_millis(12));
        b.record(1, std::time::Duration::from_millis(5));
        assert_eq!(b.remaining(2, cap), std::time::Duration::from_millis(12));
    }

    #[test]
    fn submit_gated_maps_driver_without_pending_or_enqueue() {
        let standin = 0x7E57_9A7Eu64 as usize;
        let driver = 0x7E57_D12Eu64 as usize;
        submit_gated(standin, driver);
        assert!(!is_pending(standin), "gated stand-in must not appear pending");
        assert_eq!(resolve(standin), driver);
        assert_eq!(resolve_wait(standin), driver);
        assert_eq!(resolve_for_bind(standin), driver);
        assert_eq!(released(standin), 0);
        assert_eq!(resolve(standin), standin);
    }

    #[test]
    fn frame_budget_env_parsing() {
        let ms = std::time::Duration::from_millis;
        assert_eq!(parse_frame_budget(None), Some(ms(12_000)));
        assert_eq!(parse_frame_budget(Some("")), Some(ms(12_000)));
        assert_eq!(parse_frame_budget(Some("0")), Some(ms(0)));
        assert_eq!(parse_frame_budget(Some("5000")), Some(ms(5000)));
        assert_eq!(parse_frame_budget(Some("off")), None);
        assert_eq!(parse_frame_budget(Some("none")), None);
        assert_eq!(parse_frame_budget(Some("bad")), Some(ms(12_000)));
    }

    #[test]
    fn frame_budget_waits_within_window_stamps_slow_gfx_once_and_never_skips_producer_compute() {
        use std::time::{Duration, Instant};

        let gfx_budget = Mutex::new(FrameWaitBudget::new());
        let cap = Some(Duration::from_millis(60));

        let (g_fast, g_slow1, g_slow2, c_prod) = (0x7E57_F001usize, 0x7E57_F002usize, 0x7E57_F003usize, 0x7E57_F004usize);
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let (go2_tx, go2_rx) = mpsc::channel::<()>();
        submit_lazy(g_fast, false, Box::new(|| {
            std::thread::sleep(Duration::from_millis(10));
            0xF101
        }), 0);
        submit_lazy(g_slow1, false, Box::new(move || {
            let _ = go_rx.recv();
            0xF102
        }), 0);
        submit_lazy(g_slow2, false, Box::new(move || {
            let _ = go2_rx.recv();
            0xF103
        }), 0);
        submit_lazy(c_prod, true, Box::new(|| {
            std::thread::sleep(Duration::from_millis(90));
            0xF104
        }), 0);
        heavy_compute(c_prod);

        // Frame 1: fast graphics PSO finishes within the 60 ms graphics window.
        let p_fast = entry(g_fast).unwrap();
        assert_eq!(resolve_wait_in(&p_fast, g_fast, None, cap, 1, &gfx_budget), 0xF101);

        // Frame 1: slow graphics PSO spends the rest of frame 1's graphics window, stamps
        // waited_frame = 1, and skips.
        let p_slow1 = entry(g_slow1).unwrap();
        assert_eq!(resolve_wait_in(&p_slow1, g_slow1, None, cap, 1, &gfx_budget), g_slow1);

        // Frame 1: second unready graphics PSO sees 0 ms remaining and skips immediately WITHOUT
        // stamping waited_frame (so it gets a fresh turn on Frame 2).
        let p_slow2 = entry(g_slow2).unwrap();
        let t0 = Instant::now();
        assert_eq!(resolve_wait_in(&p_slow2, g_slow2, None, cap, 1, &gfx_budget), g_slow2);
        assert!(t0.elapsed() < Duration::from_millis(25), "spent frame budget must skip immediately ({:?})", t0.elapsed());
        assert_eq!(p_slow2.waited_frame.load(Ordering::Acquire), UNWAITED);

        // Non-spared producer compute PSO (`!spared`, longer than 60 ms graphics window) MUST wait
        // to completion instead of being skipped!
        assert!(!is_spared(c_prod));
        assert_eq!(resolve_wait_for(c_prod, None), 0xF104);

        // Frame 2: g_slow1 already spent its wait turn on Frame 1 -> skips immediately without
        // burning Frame 2's graphics budget!
        let t1 = Instant::now();
        assert_eq!(resolve_wait_in(&p_slow1, g_slow1, None, Some(Duration::from_secs(5)), 2, &gfx_budget), g_slow1);
        assert!(t1.elapsed() < Duration::from_millis(25), "already-waited slow graphics PSO must skip immediately on next frame ({:?})", t1.elapsed());

        // Frame 2: g_slow2 never waited yet, so it gets Frame 2's full window and resolves once ready!
        go2_tx.send(()).unwrap();
        assert_eq!(resolve_wait_in(&p_slow2, g_slow2, None, Some(Duration::from_secs(5)), 2, &gfx_budget), 0xF103);

        // Once g_slow1's worker completes, it immediately resolves to its real pipeline!
        go_tx.send(()).unwrap();
        let wait_start = Instant::now();
        while is_pending(g_slow1) && wait_start.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(resolve_wait_in(&p_slow1, g_slow1, None, Some(Duration::from_secs(5)), 3, &gfx_budget), 0xF102);

        for s in [g_fast, g_slow1, g_slow2, c_prod] {
            let _ = released(s);
        }
    }

    #[test]
    fn wall_clock_frame_budget_prevents_staggered_multithread_cascade() {
        use std::time::{Duration, Instant};

        let cap = Duration::from_secs(12);
        let t0 = Instant::now();
        let mut b = FrameWaitBudget::new();

        // Thread 1 starts waiting at t=0: full 12 s.
        assert_eq!(b.begin_wait(1467, cap, t0), Duration::from_secs(12));
        // Thread 2 arrives staggered at t=10 s while thread 1 is still waiting: only 2 s left!
        assert_eq!(b.begin_wait(1467, cap, t0 + Duration::from_secs(10)), Duration::from_secs(2));
        // Thread 3 arrives at t=11.5 s: only 0.5 s left!
        assert_eq!(b.begin_wait(1467, cap, t0 + Duration::from_millis(11_500)), Duration::from_millis(500));
        // Thread 4 arrives at t=12.0 s: 0 ms left immediately (no cascade!).
        assert_eq!(b.begin_wait(1467, cap, t0 + Duration::from_secs(12)), Duration::ZERO);

        // All 3 active waiters finish at t=12 s -> budget spent on frame 1467.
        b.end_wait(1467, cap, t0 + Duration::from_secs(12), true);
        b.end_wait(1467, cap, t0 + Duration::from_secs(12), true);
        b.end_wait(1467, cap, t0 + Duration::from_secs(12), true);
        assert_eq!(b.remaining_at(1467, cap, t0 + Duration::from_secs(12)), Duration::ZERO);

        // Advancing to frame 1468 resets the wall-clock window cleanly.
        assert_eq!(b.begin_wait(1468, cap, t0 + Duration::from_secs(13)), Duration::from_secs(12));
        b.end_wait(1468, cap, t0 + Duration::from_secs(14), false);
        assert_eq!(b.remaining_at(1468, cap, t0 + Duration::from_secs(14)), Duration::from_secs(11));
    }

    #[test]
    fn bound_unbuilt_detects_standin_even_after_worker_finishes() {
        let (standin_ok, standin_fail, native) = (0x7E57_9801usize, 0x7E57_9802usize, 0x7E57_9803usize);
        let (tx, rx) = mpsc::channel::<()>();
        submit_lazy(standin_ok, false, Box::new(move || {
            let _ = rx.recv();
            0x9801
        }), 0);
        // While pending: bound stand-in is unbuilt.
        assert!(bound_unbuilt(standin_ok, standin_ok));
        // Finish the build after the caller already got `standin_ok`.
        let _ = resolve(standin_ok);
        tx.send(()).unwrap();
        let t0 = std::time::Instant::now();
        while is_pending(standin_ok) && t0.elapsed() < std::time::Duration::from_secs(5) {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(!is_pending(standin_ok));
        assert!(bound_unbuilt(standin_ok, standin_ok), "stale stand-in must still read unbuilt after worker completes");
        assert!(!bound_unbuilt(standin_ok, 0x9801), "resolved real pipeline must not read unbuilt");

        // Failed build (real == 0) is not treated as an in-flight unbuilt stand-in.
        submit(standin_fail, true, Box::new(|| 0));
        assert_eq!(resolve_wait(standin_fail), standin_fail);
        assert!(!bound_unbuilt(standin_fail, standin_fail));

        // Unknown / native PC pipeline is never unbuilt.
        assert!(!bound_unbuilt(native, native));

        let _ = released(standin_ok);
        let _ = released(standin_fail);
    }

    #[test]
    fn frame_cap_none_waits_for_graphics_pso_instead_of_skipping() {
        use std::time::{Duration, Instant};

        let gfx_budget = Mutex::new(FrameWaitBudget::new());

        // Graphics PSO with `frame_cap == None` and no per-PSO budget waits to completion.
        let g_wait = 0x7E57_F101usize;
        submit_lazy(g_wait, false, Box::new(|| {
            std::thread::sleep(Duration::from_millis(25));
            0xF201
        }), 0);
        let p_wait = entry(g_wait).unwrap();
        assert_eq!(resolve_wait_in(&p_wait, g_wait, None, None, 1, &gfx_budget), 0xF201);

        // Graphics PSO with `frame_cap == None` and a per-PSO budget respects `budget`
        // instead of skipping with 0 ms wait.
        let g_budget = 0x7E57_F102usize;
        let (go_tx, go_rx) = mpsc::channel::<()>();
        submit_lazy(g_budget, false, Box::new(move || {
            let _ = go_rx.recv();
            0xF202
        }), 0);
        let p_budget = entry(g_budget).unwrap();
        let t0 = Instant::now();
        assert_eq!(
            resolve_wait_in(&p_budget, g_budget, Some(Duration::from_millis(30)), None, 1, &gfx_budget),
            g_budget
        );
        assert!(
            t0.elapsed() >= Duration::from_millis(15),
            "must wait for budget instead of skipping immediately ({:?})",
            t0.elapsed()
        );
        assert!(p_budget.late.load(Ordering::Acquire));
        go_tx.send(()).unwrap();
        let wait_start = Instant::now();
        while is_pending(g_budget) && wait_start.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(
            resolve_wait_in(&p_budget, g_budget, Some(Duration::from_millis(30)), None, 1, &gfx_budget),
            0xF202
        );

        for s in [g_wait, g_budget] {
            let _ = released(s);
        }
    }

    #[test]
    fn frame_budget_keeps_default_12s_and_honors_opt_in_gfx_and_explicit_overrides() {
        let ms = std::time::Duration::from_millis;

        // Default (unset/empty/invalid): keeps 12 s across all frames (even >60) so cold
        // graphics PSOs during menus or world entry are never prematurely skipped.
        let default_policy = parse_frame_budget_policy_from(None, None);
        assert_eq!(default_policy, FrameBudgetPolicy::Default);
        assert_eq!(parse_frame_budget_policy_from(Some(""), Some("")), FrameBudgetPolicy::Default);
        assert_eq!(parse_frame_budget_policy_from(Some("bad"), Some("bad")), FrameBudgetPolicy::Default);
        for frame in [0, GAMEPLAY_FRAME_BUDGET_START_FRAME - 1, GAMEPLAY_FRAME_BUDGET_START_FRAME, 10_000] {
            assert_eq!(frame_budget_for_kind(default_policy, false, frame), Some(ms(DEFAULT_FRAME_BUDGET_MS)));
            assert_eq!(frame_budget_for_kind(default_policy, true, frame), None);
        }

        // Opt-in XCL_GFX_PSO_FRAME_BUDGET_MS=32: keeps 12 s before frame 60, caps graphics to
        // 32 ms from frame 60 onward, while leaving producer compute pipelines uncapped.
        let opt_in_32 = parse_frame_budget_policy_from(None, Some("32"));
        assert_eq!(opt_in_32, FrameBudgetPolicy::OptInGameplay(Some(ms(32))));
        assert_eq!(frame_budget_for_kind(opt_in_32, false, 0), Some(ms(DEFAULT_FRAME_BUDGET_MS)));
        assert_eq!(frame_budget_for_kind(opt_in_32, false, GAMEPLAY_FRAME_BUDGET_START_FRAME - 1), Some(ms(DEFAULT_FRAME_BUDGET_MS)));
        assert_eq!(frame_budget_for_kind(opt_in_32, false, GAMEPLAY_FRAME_BUDGET_START_FRAME), Some(ms(32)));
        assert_eq!(frame_budget_for_kind(opt_in_32, false, 10_000), Some(ms(32)));
        assert_eq!(frame_budget_for_kind(opt_in_32, true, GAMEPLAY_FRAME_BUDGET_START_FRAME), None);
        assert_eq!(frame_budget_for_kind(opt_in_32, true, 10_000), None);

        // Explicit XCL_PSO_FRAME_BUDGET_MS applies uniformly across all frames and overrides opt-in.
        let explicit_500 = parse_frame_budget_policy_from(Some("500"), Some("32"));
        assert_eq!(frame_budget_for_kind(explicit_500, false, 0), Some(ms(500)));
        assert_eq!(frame_budget_for_kind(explicit_500, false, 10_000), Some(ms(500)));
        assert_eq!(frame_budget_for_kind(explicit_500, true, 10_000), None);
        let explicit_zero = parse_frame_budget_policy_from(Some("0"), None);
        assert_eq!(frame_budget_for_kind(explicit_zero, false, 0), Some(ms(0)));
        assert_eq!(frame_budget_for_kind(explicit_zero, false, 10_000), Some(ms(0)));
        let explicit_off = parse_frame_budget_policy_from(Some("off"), Some("32"));
        assert_eq!(frame_budget_for_kind(explicit_off, false, 0), None);
        assert_eq!(frame_budget_for_kind(explicit_off, false, 10_000), None);
    }

    #[test]
    fn pso_worker_threads_scales_with_cpus_and_ram() {
        // Smaller hosts (< 16 threads or < 24 GiB RAM) stay capped at 2..=6.
        assert_eq!(pso_worker_threads_for(2, 16), 2);
        assert_eq!(pso_worker_threads_for(4, 16), 3);
        assert_eq!(pso_worker_threads_for(8, 16), 6);
        assert_eq!(pso_worker_threads_for(8, 32), 6);
        assert_eq!(pso_worker_threads_for(16, 16), 6);
        assert_eq!(pso_worker_threads_for(24, 16), 6);

        // High-core / high-RAM hosts (>= 16 threads and >= 24 GiB RAM) scale cleanly up to 10.
        assert_eq!(pso_worker_threads_for(16, 24), 8);
        assert_eq!(pso_worker_threads_for(16, 32), 8);
        assert_eq!(pso_worker_threads_for(20, 32), 10);
        assert_eq!(pso_worker_threads_for(24, 32), 10);
        assert_eq!(pso_worker_threads_for(32, 64), 10);
    }
}
