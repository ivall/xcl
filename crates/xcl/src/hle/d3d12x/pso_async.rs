//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

struct Pending {
    /// the build, until a worker – or a bind that cannot wait for the queue – takes it
    job: Mutex<Option<Job>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// Mesh / amplification stream pipeline: isolated onto `ms_queue()` so slow MS HLSL builds
    /// cannot wedge `gfx_queue()` VS+PS pipelines.
    mesh: bool,
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
    demand: Mutex<Demand>,
}

/// Whether one of the title's threads waits for the pipeline, and the preload worker building it.
#[derive(Default)]
struct Demand {
    wanted: bool,
    /// an unbounded bind (`want`) waits until the build finishes
    unbounded_wanted: bool,
    /// active bounded waits (`want_timed`); cleared when the wait exits so late builds return
    /// to `real::big_compile_turn` and `dxc::wait_for_memory` commit throttling
    timed_waiters: usize,
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
    complete(p, job());
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

fn set_preload_priority(tid: u32, priority: i32) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_SET_LIMITED_INFORMATION};
    unsafe {
        let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, tid);
        if !h.is_null() {
            SetThreadPriority(h, priority);
            CloseHandle(h);
        }
    }
}

/// Raises `Demand::wanted` only while a bounded bind wait is actively blocked on `Pending`.
struct TimedWantGuard<'a>(&'a Pending);

impl Drop for TimedWantGuard<'_> {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Threading::THREAD_PRIORITY_LOWEST;
        let restore_tid = {
            let mut d = self.0.demand.lock().unwrap();
            d.timed_waiters = d.timed_waiters.saturating_sub(1);
            if d.timed_waiters == 0 && !d.unbounded_wanted && std::mem::replace(&mut d.wanted, false) && d.preload_thread != 0 {
                Some(d.preload_thread)
            } else {
                None
            }
        };
        if let Some(tid) = restore_tid {
            set_preload_priority(tid, THREAD_PRIORITY_LOWEST);
        }
    }
}

impl Pending {
    /// One of the title's threads is about to wait unconditionally for it: the worker building
    /// it goes on now, at a normal priority.
    fn want(&self) {
        use windows_sys::Win32::System::Threading::THREAD_PRIORITY_NORMAL;
        let raise_tid = {
            let mut d = self.demand.lock().unwrap();
            d.unbounded_wanted = true;
            if !std::mem::replace(&mut d.wanted, true) && d.preload_thread != 0 {
                Some(d.preload_thread)
            } else {
                None
            }
        };
        if let Some(tid) = raise_tid {
            set_preload_priority(tid, THREAD_PRIORITY_NORMAL);
        }
    }

    /// One of the title's threads waits for a bounded budget: keep `wanted` raised only while
    /// the wait is active so a timed-out build returns to `big_compile_turn` / DXC memory gates.
    fn want_timed(&self) -> TimedWantGuard<'_> {
        use windows_sys::Win32::System::Threading::THREAD_PRIORITY_NORMAL;
        let raise_tid = {
            let mut d = self.demand.lock().unwrap();
            d.timed_waiters += 1;
            if !std::mem::replace(&mut d.wanted, true) && d.preload_thread != 0 {
                Some(d.preload_thread)
            } else {
                None
            }
        };
        if let Some(tid) = raise_tid {
            set_preload_priority(tid, THREAD_PRIORITY_NORMAL);
        }
        TimedWantGuard(self)
    }

    fn note_use(&self) {
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
        }
    }
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
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
    /// the frames shown when `spent` began to count
    frame: u64,
    /// the time of the PC calls that ended since then
    spent: std::time::Duration,
}

impl PcCalls {
    const fn new() -> Self {
        Self { now: Vec::new(), frame: 0, spent: std::time::Duration::ZERO }
    }

    /// `frame` frames have been shown: the calls before the last of them are not this wait's.
    fn at(&mut self, frame: u64) {
        if frame != self.frame {
            self.frame = frame;
            self.spent = std::time::Duration::ZERO;
        }
    }

    fn begin(&mut self, id: u64, now: std::time::Instant) {
        self.now.push((id, now));
    }

    fn end(&mut self, id: u64, frame: u64, now: std::time::Instant) {
        self.at(frame);
        if let Some(i) = self.now.iter().position(|c| c.0 == id) {
            let (_, since) = self.now.swap_remove(i);
            self.spent += now.saturating_duration_since(since);
        }
    }

    /// Whether a call going on holds the title: the PC calls since the last frame, it included,
    /// have taken `PC_CALL_HOLDS_AFTER`.
    fn holds(&mut self, frame: u64, now: std::time::Instant) -> bool {
        self.at(frame);
        let spent = self.spent;
        self.now.iter().any(|(_, since)| spent + now.saturating_duration_since(*since) >= PC_CALL_HOLDS_AFTER)
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
    PC_CALLS.lock().unwrap().begin(id, std::time::Instant::now());
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

type Work = Box<dyn FnOnce() + Send>;

/// Workers building demand-queued pipelines so one lane cannot starve another: classic graphics
/// pipelines run on `gfx_queue()`, mesh/amplification stream pipelines run on `ms_queue()`, and
/// compute shaders run on `queue()`.
fn spawn_queue(prefix: &'static str, threads: usize) -> Mutex<mpsc::Sender<Work>> {
    let (tx, rx) = mpsc::channel::<Work>();
    let rx = Arc::new(Mutex::new(rx));
    // translation recurses deeply on big programs; DXC itself runs on its own pool
    for k in 0..threads.max(1) {
        let rx = rx.clone();
        let _ = std::thread::Builder::new().name(format!("{prefix}-{k}")).stack_size(64 << 20).spawn(move || loop {
            let next = rx.lock().unwrap().recv();
            let Ok(work) = next else { return };
            work();
        });
    }
    Mutex::new(tx)
}

fn queue() -> &'static Mutex<mpsc::Sender<Work>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Work>>> = OnceLock::new();
    Q.get_or_init(|| spawn_queue("xcl-pso", crate::gpu::recomp::dxc::compile_threads()))
}

fn gfx_queue() -> &'static Mutex<mpsc::Sender<Work>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Work>>> = OnceLock::new();
    Q.get_or_init(|| spawn_queue("xcl-pso-gfx", crate::gpu::recomp::dxc::compile_threads()))
}

fn ms_queue() -> &'static Mutex<mpsc::Sender<Work>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Work>>> = OnceLock::new();
    Q.get_or_init(|| spawn_queue("xcl-pso-ms", crate::gpu::recomp::dxc::compile_threads()))
}

fn queue_for(p: &Pending) -> &'static Mutex<mpsc::Sender<Work>> {
    if p.compute {
        queue()
    } else if p.mesh {
        ms_queue()
    } else {
        gfx_queue()
    }
}

fn enqueue(p: Arc<Pending>) {
    let _ = queue_for(&p).lock().unwrap().send(Box::new(move || {
        // (released by the title meanwhile)
        if p.orphan.load(Ordering::Acquire) {
            let _ = p.job.lock().unwrap().take();
            return;
        }
        // (None: a bind took it and builds it itself)
        let job = p.job.lock().unwrap().take();
        if let Some(job) = job {
            build(&p, job, false);
        }
    }));
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
    for _ in 1..count.min(crate::gpu::recomp::dxc::compile_threads()) {
        let (drain, parent) = (drain.clone(), parent.clone());
        let _ = queue().lock().unwrap().send(Box::new(move || {
            let before = BUILDING.with(|b| b.replace(parent));
            drain();
            BUILDING.with(|b| *b.borrow_mut() = before);
        }));
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

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_lane(standin, compute, false, job);
}

/// `submit`, routing mesh pipelines (`mesh = true`) onto `ms_queue()`.
pub(super) fn submit_lane(standin: usize, compute: bool, mesh: bool, job: Job) {
    submit_as(standin, compute, !compute && mesh, job, true, 0);
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
    submit_lazy_lane(standin, compute, false, job, key);
}

/// `submit_lazy`, routing mesh pipelines (`mesh = true`) onto `ms_queue()`.
pub(super) fn submit_lazy_lane(standin: usize, compute: bool, mesh: bool, job: Job, key: u64) {
    submit_as(standin, compute, !compute && mesh, job, false, key);
}

fn submit_as(standin: usize, compute: bool, mesh: bool, job: Job, send: bool, key: u64) {
    let p = Arc::new(Pending {
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(send),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        mesh,
        key,
        used: Default::default(),
        orphan: Default::default(),
        late: Default::default(),
        spared: Default::default(),
        demand: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        enqueue(p);
    } else if used::known(key) {
        if let Some(q) = preload() {
            q.list.lock().unwrap().push_back(p);
            q.cv.notify_one();
        }
    }
}

/// The keys of the pipelines earlier runs used (`used-pipelines.txt` beside the DXIL cache):
/// what `xcl shaders --precompile --used-only` builds.
pub fn earlier_used_keys() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt")).unwrap_or_default())
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
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock, RwLock};

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0).collect()
    }

    fn get() -> &'static Used {
        static U: OnceLock<Used> = OnceLock::new();
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashSet::new()), now: Mutex::new((HashSet::new(), None)) };
            }
            let path = crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt");
            let earlier = parse(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
            Used { now: Mutex::new((earlier.clone(), file)), earlier: RwLock::new(earlier) }
        })
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
        let mut g = get().now.lock().unwrap();
        if g.0.insert(key) {
            if let Some(f) = g.1.as_mut() {
                let _ = writeln!(f, "{key:016x}");
            }
        }
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.lock().unwrap().0.contains(&key)
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
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
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

    /// A PC call that has kept one of the title's threads for long holds the title, a short one
    /// does not, and the hold ends with the call.
    #[test]
    fn a_long_pc_call_holds_the_title() {
        let mut calls = PcCalls::new();
        let began = std::time::Instant::now();
        calls.begin(1, began);
        calls.begin(2, began + 2 * PC_CALL_HOLDS_AFTER);
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
            calls.begin(id, now);
            assert!(id >= 9 || !calls.holds(7, now + step), "{id} short calls hold the title already");
            now += step;
            calls.end(id, 7, now);
        }
        assert!(!calls.holds(7, now), "no PC call is going on, the title is held");
        calls.begin(20, now);
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

    #[test]
    fn wait_caps_respect_env_and_title_boot_state() {
        use std::time::Duration;
        assert_eq!(resolve_caps(None, None, false), None);
        assert_eq!(
            resolve_caps(None, None, true),
            Some(WaitCaps { per_pso: Duration::from_millis(DEFAULT_TITLE_WAIT_MS), per_frame: Some(Duration::from_millis(DEFAULT_FRAME_WAIT_MS)) })
        );
        assert_eq!(resolve_caps(Some(0), None, true), None);
        assert_eq!(
            resolve_caps(Some(500), None, false),
            Some(WaitCaps { per_pso: Duration::from_millis(500), per_frame: Some(Duration::from_millis(DEFAULT_FRAME_WAIT_MS)) })
        );
        assert_eq!(
            resolve_caps(Some(3_000), None, true),
            Some(WaitCaps { per_pso: Duration::from_millis(3_000), per_frame: Some(Duration::from_millis(3_000)) })
        );
        assert_eq!(
            resolve_caps(Some(250), Some(0), true),
            Some(WaitCaps { per_pso: Duration::from_millis(250), per_frame: None })
        );
        assert_eq!(
            resolve_caps(Some(1_000), Some(500), true),
            Some(WaitCaps { per_pso: Duration::from_millis(500), per_frame: Some(Duration::from_millis(500)) })
        );
        assert_eq!(resolve_caps(Some(0), Some(500), true), Some(WaitCaps { per_pso: Duration::from_millis(500), per_frame: Some(Duration::from_millis(500)) }));
        assert_eq!(resolve_caps(None, Some(0), true), Some(WaitCaps { per_pso: Duration::from_millis(DEFAULT_TITLE_WAIT_MS), per_frame: None }));
        assert_eq!(resolve_caps(None, Some(0), false), None);
    }

    #[test]
    fn frame_budget_caps_cumulative_wait_and_resets_on_new_frame() {
        use std::time::Duration;
        let mut fb = FrameBudget::default();
        let caps = WaitCaps { per_pso: Duration::from_millis(250), per_frame: Some(Duration::from_millis(600)) };

        assert_eq!(fb.reserve(1, caps), (Duration::from_millis(250), false));
        fb.settle(1, Duration::from_millis(250), Duration::from_millis(250));
        assert_eq!(fb.reserve(1, caps), (Duration::from_millis(250), false));
        // Early wakeup returns unused reserved time to the frame
        fb.settle(1, Duration::from_millis(250), Duration::from_millis(150));
        // 200 ms left in frame 1: truncated below per_pso (250 ms)
        assert_eq!(fb.reserve(1, caps), (Duration::from_millis(200), true));
        fb.settle(1, Duration::from_millis(200), Duration::from_millis(200));
        // Exhausted for the rest of frame 1
        assert_eq!(fb.reserve(1, caps), (Duration::ZERO, true));

        // B1 regression: cross-frame settle from frame 1 must not wipe frame 2's spend
        let b1_caps = WaitCaps { per_pso: Duration::from_millis(100), per_frame: Some(Duration::from_millis(100)) };
        let mut cross = FrameBudget::new();
        assert_eq!(cross.reserve(1, b1_caps), (Duration::from_millis(100), false));
        assert_eq!(cross.reserve(2, b1_caps), (Duration::from_millis(100), false));
        cross.settle(2, Duration::from_millis(100), Duration::from_millis(40));
        // Late settle & reserve for stale frame 1 leave frame 2 with 60 ms remaining (not 100 ms)
        cross.settle(1, Duration::from_millis(100), Duration::from_millis(100));
        assert_eq!(cross.reserve(1, b1_caps), (Duration::ZERO, true));
        assert_eq!(cross.reserve(2, b1_caps), (Duration::from_millis(60), true));

        // Frame 3 resets the cumulative budget cleanly
        assert_eq!(fb.reserve(3, caps), (Duration::from_millis(250), false));

        // Uncapped frame leaves every wait at full per_pso without reserving
        let uncapped = WaitCaps { per_pso: Duration::from_millis(250), per_frame: None };
        fb.settle(3, Duration::from_millis(250), Duration::from_secs(10));
        assert_eq!(fb.reserve(3, uncapped), (Duration::from_millis(250), false));
    }

    #[test]
    fn frame_budget_stale_frame_sample_keeps_full_wait_when_per_frame_disabled() {
        use std::time::Duration;
        let mut fb = FrameBudget::new();
        let capped = WaitCaps { per_pso: Duration::from_millis(250), per_frame: Some(Duration::from_millis(500)) };
        let uncapped = WaitCaps { per_pso: Duration::from_millis(250), per_frame: None };

        // Caller B advances the shared budget to frame 2.
        assert_eq!(fb.reserve(1, capped), (Duration::from_millis(250), false));
        assert_eq!(fb.reserve(2, capped), (Duration::from_millis(250), false));

        // Caller A sampled frame 1 before B advanced: with per_frame == None it still gets full per_pso.
        assert_eq!(fb.reserve(1, uncapped), (Duration::from_millis(250), false));

        // With per_frame == Some(...), stale frame 1 is still rejected without disturbing frame 2.
        assert_eq!(fb.reserve(1, capped), (Duration::ZERO, true));
        assert_eq!(fb.reserve(2, capped), (Duration::from_millis(250), false));
    }

    #[test]
    fn a_frame_budget_caps_a_burst_of_slow_pipelines_and_resets_next_frame() {
        use std::time::{Duration, Instant};
        let fb = Arc::new(Mutex::new(FrameBudget::new()));
        let caps = WaitCaps { per_pso: Duration::from_millis(35), per_frame: Some(Duration::from_millis(75)) };

        let (tx1, rx1) = mpsc::channel();
        let (tx2, rx2) = mpsc::channel();
        let (tx3, rx3) = mpsc::channel();
        let (tx4, rx4) = mpsc::channel();
        let (p1, p2, p3, p4, ready) = (0x7E57_C001usize, 0x7E57_C002usize, 0x7E57_C003usize, 0x7E57_C004usize, 0x7E57_C005usize);
        submit(ready, true, Box::new(|| 0xA5));
        let start = Instant::now();
        while is_pending(ready) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(1));
        }
        submit_lazy(p1, true, Box::new(move || { let _ = rx1.recv_timeout(Duration::from_secs(30)); 0xA1 }), 0);
        submit_lazy(p2, true, Box::new(move || { let _ = rx2.recv_timeout(Duration::from_secs(30)); 0xA2 }), 0);
        submit_lazy(p3, true, Box::new(move || { let _ = rx3.recv_timeout(Duration::from_secs(30)); 0xA3 }), 0);
        submit_lazy(p4, true, Box::new(move || { let _ = rx4.recv_timeout(Duration::from_secs(30)); 0xA4 }), 0);

        let (e1, e2, e3, e4, er) = (entry(p1).unwrap(), entry(p2).unwrap(), entry(p3).unwrap(), entry(p4).unwrap(), entry(ready).unwrap());
        let t0 = Instant::now();
        assert_eq!(resolve_within_in_frame(&e1, p1, caps, 1, &fb), p1);
        assert!(e1.late.load(Ordering::Acquire), "full per-PSO timeout marks pipeline late");
        assert_eq!(resolve_within_in_frame(&e2, p2, caps, 1, &fb), p2);
        let _ = resolve_within_in_frame(&e3, p3, caps, 1, &fb);
        // A second thread in the same frame sees the already-exhausted shared frame budget.
        let (e4_t, fb_t) = (e4.clone(), fb.clone());
        let (res4, elapsed4) = std::thread::spawn(move || {
            let t4 = Instant::now();
            let r = resolve_within_in_frame(&e4_t, p4, caps, 1, &fb_t);
            (r, t4.elapsed())
        })
        .join()
        .unwrap();
        assert_eq!(res4, p4);
        assert!(elapsed4 < Duration::from_millis(40), "exhausted shared frame budget skips wait ({elapsed4:?})");
        assert!(t0.elapsed() < Duration::from_millis(350), "burst waited {:?}", t0.elapsed());
        assert!(!e4.late.load(Ordering::Acquire), "budget-skipped pipeline must not be marked late");
        assert_eq!(resolve_within_in_frame(&er, ready, caps, 1, &fb), 0xA5, "already-built pipeline still resolves when frame budget is exhausted");

        // Advancing to frame 2 resets the budget so p4 gets a fresh wait window.
        let t_next = Instant::now();
        assert_eq!(resolve_within_in_frame(&e4, p4, caps, 2, &fb), p4);
        assert!(t_next.elapsed() >= Duration::from_millis(20), "new frame resets wait budget ({:?})", t_next.elapsed());
        assert!(e4.late.load(Ordering::Acquire), "timed-out pipeline in fresh frame becomes late");

        for tx in [tx1, tx2, tx3, tx4] {
            let _ = tx.send(());
        }
        let start = Instant::now();
        for p in [p1, p2, p3, p4, ready] {
            while is_pending(p) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            assert!(!is_pending(p));
            let _ = released(p);
        }
    }

    #[test]
    fn timed_out_bind_releases_wanted_so_late_builds_respect_throttling() {
        use std::time::{Duration, Instant};
        let standin = 0x7E57_C010usize;
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            started_tx.send(wanted_here()).unwrap();
            let _ = done_rx.recv_timeout(Duration::from_secs(30));
            0xB1B1
        }));
        let wanted = started_rx.recv().unwrap();
        let p = entry(standin).unwrap();
        assert!(!wanted(), "not wanted before a bind waits");
        {
            let _guard = p.want_timed();
            assert!(wanted(), "wanted while a timed bind wait is active");
        }
        assert!(!wanted(), "dropping TimedWantGuard clears wanted");

        let fb = Mutex::new(FrameBudget::new());
        let caps = WaitCaps { per_pso: Duration::from_millis(25), per_frame: Some(Duration::from_millis(200)) };
        assert_eq!(resolve_within_in_frame(&p, standin, caps, 1, &fb), standin);
        assert!(!wanted(), "timed-out bind releases wanted so big_compile_turn / DXC memory gates still apply");
        assert!(p.late.load(Ordering::Acquire));

        let _ = done_tx.send(());
        let start = Instant::now();
        while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(released(standin), 0xB1B1);
    }

    #[test]
    fn concurrent_binds_in_same_frame_share_reserved_budget() {
        use std::sync::Barrier;
        use std::time::{Duration, Instant};

        let fb = Arc::new(Mutex::new(FrameBudget::new()));
        let caps = WaitCaps { per_pso: Duration::from_millis(200), per_frame: Some(Duration::from_millis(200)) };
        let (tx1, rx1) = mpsc::channel();
        let (tx2, rx2) = mpsc::channel();
        let (p1, p2) = (0x7E57_D001usize, 0x7E57_D002usize);
        submit(p1, true, Box::new(move || { let _ = rx1.recv_timeout(Duration::from_secs(30)); 0xD1 }));
        submit(p2, true, Box::new(move || { let _ = rx2.recv_timeout(Duration::from_secs(30)); 0xD2 }));
        let e1 = entry(p1).unwrap();
        let e2 = entry(p2).unwrap();

        let barrier = Arc::new(Barrier::new(2));
        let run = |e: Arc<Pending>, p: usize, fb: Arc<Mutex<FrameBudget>>, b: Arc<Barrier>| {
            std::thread::spawn(move || {
                b.wait();
                let t = Instant::now();
                let res = resolve_within_in_frame(&e, p, caps, 1, &fb);
                (res, t.elapsed())
            })
        };
        let t1 = run(e1.clone(), p1, fb.clone(), barrier.clone());
        let t2 = run(e2.clone(), p2, fb.clone(), barrier);
        let (r1, d1) = t1.join().unwrap();
        let (r2, d2) = t2.join().unwrap();
        assert_eq!((r1, r2), (p1, p2));
        let (short, long) = if d1 <= d2 { (d1, d2) } else { (d2, d1) };
        assert!(long >= Duration::from_millis(160), "reserving thread waits near full cap ({long:?})");
        assert!(short < Duration::from_millis(80), "concurrent thread sees zero remaining budget ({short:?})");
        assert!(d1 + d2 < Duration::from_millis(320), "combined wait must not double frame cap ({d1:?} + {d2:?})");

        let _ = tx1.send(());
        let _ = tx2.send(());
        let start = Instant::now();
        for p in [p1, p2] {
            while is_pending(p) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            assert!(!is_pending(p));
            let _ = released(p);
        }
    }

    /// A queued normal worker job whose stand-in was released before a worker starts it is
    /// skipped as orphaned instead of entering the compiler.
    #[test]
    fn a_released_queued_pipeline_is_not_built_by_normal_workers() {
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

    static WORKER_SATURATION_LOCK: Mutex<()> = Mutex::new(());

    fn assert_graphics_unblocked_when_lane_stalled(blocker_base: usize, gfx: usize, compute: bool, mesh: bool, blocker_msg: &str) {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let n = crate::gpu::recomp::dxc::compile_threads();
        let (rel_tx, rel_rx) = mpsc::channel::<()>();
        let rel_rx = Arc::new(Mutex::new(rel_rx));
        struct ReleaseOnDrop(mpsc::Sender<()>, usize);
        impl Drop for ReleaseOnDrop {
            fn drop(&mut self) {
                for _ in 0..self.1 {
                    let _ = self.0.send(());
                }
            }
        }
        let _release = ReleaseOnDrop(rel_tx, n);
        let (entered_tx, entered_rx) = mpsc::channel();
        for i in 0..n {
            let rel_rx = rel_rx.clone();
            let entered_tx = entered_tx.clone();
            submit_lane(blocker_base + i, compute, mesh, Box::new(move || {
                let _ = entered_tx.send(());
                let _ = rel_rx.lock().unwrap().recv_timeout(std::time::Duration::from_secs(10));
                0x6100 + i
            }));
        }
        for _ in 0..n {
            entered_rx.recv_timeout(std::time::Duration::from_secs(5)).expect(blocker_msg);
        }
        let (gfx_tx, gfx_rx) = mpsc::channel();
        submit(gfx, false, Box::new(move || {
            let _ = gfx_tx.send(std::thread::current().name().map(str::to_owned));
            0x6200
        }));
        let built_on = gfx_rx.recv_timeout(std::time::Duration::from_secs(5)).expect("graphics pipeline must not block behind stalled lane");
        assert!(built_on.as_deref().is_some_and(|s| s.starts_with("xcl-pso-gfx-")), "built on {built_on:?}");
        drop(_release);
        let start = std::time::Instant::now();
        while (0..n).any(|i| is_pending(blocker_base + i)) && start.elapsed() < std::time::Duration::from_secs(5) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        for i in 0..n {
            let _ = released(blocker_base + i);
        }
        let _ = released(gfx);
    }

    #[test]
    fn graphics_pipelines_are_not_blocked_by_stalled_compute_workers() {
        assert_graphics_unblocked_when_lane_stalled(0x7E57_9100, 0x7E57_9200, true, false, "compute workers entered");
    }

    #[test]
    fn graphics_pipelines_are_not_blocked_by_stalled_mesh_workers() {
        assert_graphics_unblocked_when_lane_stalled(0x7E57_9300, 0x7E57_9400, false, true, "mesh workers entered");
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// The pipeline to use for `pso` now: the real one once built, else `pso` itself.
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue(p);
            }
            pso
        }
        r => r,
    }
}

/// Default per-PSO bind wait when running a title (`XCL_PSO_WAIT_MS` unset).
const DEFAULT_TITLE_WAIT_MS: u64 = 250;
/// Default cumulative bind wait cap per `PresentX` frame (`XCL_PSO_FRAME_WAIT_MS` unset),
/// keeping `PresentX` advancing through bursts of cold pipeline builds.
const DEFAULT_FRAME_WAIT_MS: u64 = 2_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WaitCaps {
    /// Maximum wait for a single pipeline bind.
    per_pso: std::time::Duration,
    /// Optional process-wide cumulative wait cap within one `PresentX` frame.
    per_frame: Option<std::time::Duration>,
}

fn parse_env_ms(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok())
}

fn resolve_caps(pso_env: Option<u64>, frame_env: Option<u64>, title_running: bool) -> Option<WaitCaps> {
    let per_pso = match pso_env {
        Some(0) => None,
        Some(ms) => Some(std::time::Duration::from_millis(ms)),
        None => title_running.then(|| std::time::Duration::from_millis(DEFAULT_TITLE_WAIT_MS)),
    };
    match (per_pso, frame_env) {
        (None, Some(ms)) if ms > 0 => {
            let f = std::time::Duration::from_millis(ms);
            Some(WaitCaps { per_pso: f, per_frame: Some(f) })
        }
        (None, _) => None,
        (Some(p), Some(0)) => Some(WaitCaps { per_pso: p, per_frame: None }),
        (Some(p), Some(ms)) => {
            let f = std::time::Duration::from_millis(ms);
            Some(WaitCaps { per_pso: p.min(f), per_frame: Some(f) })
        }
        (Some(p), None) => {
            let f = std::time::Duration::from_millis(DEFAULT_FRAME_WAIT_MS).max(p);
            Some(WaitCaps { per_pso: p, per_frame: Some(f) })
        }
    }
}

/// How long a bind waits for a pipeline still being built:
/// - `XCL_PSO_WAIT_MS`: per-pipeline cap in ms; `0` waits until built unless
///   `XCL_PSO_FRAME_WAIT_MS` is explicitly set to a positive value (in which case the explicit
///   frame cap wins and bounds both per-pipeline and per-frame waits). When set while
///   `XCL_PSO_FRAME_WAIT_MS` is unset, the frame cap is `max(2,000 ms, XCL_PSO_WAIT_MS)`.
/// - `XCL_PSO_FRAME_WAIT_MS`: process-wide cumulative cap per `PresentX` frame in ms; `0`
///   disables the per-frame cap while keeping `XCL_PSO_WAIT_MS` (or the 250 ms title default when
///   `XCL_PSO_WAIT_MS` is unset while a title runs; unbounded when no title is booted).
/// - When both are unset: defaults to 250 ms per pipeline / 2,000 ms cumulative per frame while
///   a title is running so cold bursts skip dispatches instead of stalling the watchdog
///   (unbounded in unit tests without a booted title).
fn wait_caps() -> Option<WaitCaps> {
    static ENV: OnceLock<(Option<u64>, Option<u64>)> = OnceLock::new();
    let &(pso_env, frame_env) = ENV.get_or_init(|| (parse_env_ms("XCL_PSO_WAIT_MS"), parse_env_ms("XCL_PSO_FRAME_WAIT_MS")));
    resolve_caps(pso_env, frame_env, crate::state::boot().is_some())
}

#[derive(Default)]
struct FrameBudget {
    frame: u64,
    spent: std::time::Duration,
    reserved: std::time::Duration,
}

impl FrameBudget {
    const fn new() -> Self {
        Self { frame: 0, spent: std::time::Duration::ZERO, reserved: std::time::Duration::ZERO }
    }

    /// Reserve budget for one bind in `frame`, returning `(effective_budget, truncated_by_frame)`.
    /// When `per_frame` is `None`, returns full `per_pso` immediately before checking frame
    /// generation so stale frame samples never zero out waits when the per-frame cap is disabled.
    /// When `per_frame` is `Some(_)`, advancing `frame` clears the prior frame's accounting and
    /// stale frames never roll back or consume a newer frame's budget.
    fn reserve(&mut self, frame: u64, caps: WaitCaps) -> (std::time::Duration, bool) {
        let Some(limit) = caps.per_frame else {
            return (caps.per_pso, false);
        };
        if frame > self.frame {
            self.frame = frame;
            self.spent = std::time::Duration::ZERO;
            self.reserved = std::time::Duration::ZERO;
        } else if frame < self.frame {
            return (std::time::Duration::ZERO, true);
        }
        let left = limit.saturating_sub(self.spent.saturating_add(self.reserved));
        let (grant, truncated) = if left < caps.per_pso { (left, true) } else { (caps.per_pso, false) };
        self.reserved = self.reserved.saturating_add(grant);
        (grant, truncated)
    }

    /// Reconcile a reservation against the actual elapsed wait time, ignoring completions whose
    /// originating frame has already been superseded by a newer `PresentX`.
    fn settle(&mut self, frame: u64, reserved: std::time::Duration, waited: std::time::Duration) {
        if self.frame == frame {
            self.reserved = self.reserved.saturating_sub(reserved);
            self.spent = self.spent.saturating_add(waited);
        }
    }
}

struct FrameWaitGuard<'a> {
    fb: &'a Mutex<FrameBudget>,
    frame: u64,
    reserved: std::time::Duration,
    started: std::time::Instant,
}

impl Drop for FrameWaitGuard<'_> {
    fn drop(&mut self) {
        if !self.reserved.is_zero() {
            self.fb.lock().unwrap().settle(self.frame, self.reserved, self.started.elapsed());
        }
    }
}

static FRAME_BUDGET: Mutex<FrameBudget> = Mutex::new(FrameBudget::new());

/// `resolve`, waiting for a pipeline still being built – built here if unbounded and no worker
/// has started it yet, or waited on workers up to the per-PSO and process-wide per-frame budget.
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_with_caps(pso, wait_caps())
}

/// `resolve_wait`, waiting at most `budget` per pipeline (None: until the pipeline is built).
#[cfg(test)]
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>) -> usize {
    resolve_wait_with_caps(pso, budget.map(|per_pso| WaitCaps { per_pso, per_frame: None }))
}

fn resolve_wait_with_caps(pso: usize, caps: Option<WaitCaps>) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    if p.spared.load(Ordering::Acquire) {
        return resolve_spared(&p, pso);
    }
    if let Some(caps) = caps {
        let frame = super::present::frame_generation();
        return resolve_within_in_frame(&p, pso, caps, frame, &FRAME_BUDGET);
    }
    // (the title's thread is held from here until the pipeline is built)
    let _held = (!*p.done.lock().unwrap()).then(Held::new);
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        complete(&p, job());
    }
    p.want();
    let mut done = p.done.lock().unwrap();
    while !*done {
        done = p.cv.wait(done).unwrap();
    }
    match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    }
}

/// `resolve_wait` within `caps` for `frame`: the build goes to the workers (not this thread).
/// A pipeline that exhausts its full per-PSO budget is marked `late` – it and later binds of it
/// get the stand-in until a worker has built it. When the frame's cumulative wait cap runs out,
/// remaining pending binds in that `PresentX` generation continue immediately with the stand-in
/// without marking those pipelines `late`, keeping recording moving toward the next `PresentX`
/// where they receive a fresh wait window.
fn resolve_within_in_frame(p: &Arc<Pending>, pso: usize, caps: WaitCaps, frame: u64, fb: &Mutex<FrameBudget>) -> usize {
    let built = |p: &Pending| match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    };
    if *p.done.lock().unwrap() {
        return built(p);
    }
    if !p.sent.swap(true, Ordering::AcqRel) {
        enqueue(p.clone());
    }
    if p.late.load(Ordering::Acquire) {
        return pso;
    }
    let (budget, truncated_by_frame) = fb.lock().unwrap().reserve(frame, caps);
    if budget.is_zero() {
        static SKIPPED: AtomicUsize = AtomicUsize::new(0);
        let n = SKIPPED.fetch_add(1, Ordering::Relaxed);
        if n < 16 || n.is_power_of_two() {
            crate::xlog_warn!(
                "D3D12.X: per-frame pipeline wait budget ({} ms) exhausted in frame {frame}: continuing with stand-in ({} so far)",
                caps.per_frame.unwrap_or(caps.per_pso).as_millis(),
                n + 1
            );
        }
        return pso;
    }
    // (the title's thread is held while waiting for this pipeline; `wanted` drops when the wait
    // ends so timed-out huge builds return to `big_compile_turn` and DXC commit throttling)
    let _held = Held::new();
    let _want = p.want_timed();
    let budget_guard = FrameWaitGuard {
        fb,
        frame,
        reserved: if caps.per_frame.is_some() { budget } else { std::time::Duration::ZERO },
        started: std::time::Instant::now(),
    };
    let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), budget, |d| !*d).unwrap();
    drop(budget_guard);
    if *done {
        return built(p);
    }
    drop(done);
    if !truncated_by_frame {
        p.late.store(true, Ordering::Release);
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 16 || n.is_power_of_two() {
            crate::xlog_warn!("D3D12.X: a pipeline still being built after {} ms: its binds go on without it until it is built ({} so far)", budget.as_millis(), n + 1);
        }
    }
    pso
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
    match p.real.load(Ordering::Acquire) {
        0 => {
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue(p.clone());
            }
            if !*p.done.lock().unwrap() && !p.late.swap(true, Ordering::AcqRel) {
                static N: AtomicUsize = AtomicUsize::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                if n < 16 || n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: a compute program that writes only images is still being built: its dispatches are left out until it is ({} so far)", n + 1);
                }
            }
            pso
        }
        r => r,
    }
}

/// For binding: compute pipelines are waited for, graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_wait(pso),
        Some(_) => resolve(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    let done = p.done.lock().unwrap();
    if *done {
        p.real.swap(0, Ordering::AcqRel)
    } else {
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
        0
    }
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}
