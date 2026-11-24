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
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
    /// a bind waited for it longer than `wait_budget`: binds go on without it until it is built
    late: std::sync::atomic::AtomicBool,
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

/// Whether a bind is waiting for a pipeline still being built: one of the title's threads is
/// held where a console title never waits (`kernelx::Sleep` does not count that time).
pub fn title_held() -> bool {
    HELD.load(Ordering::Acquire) != 0
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

fn queue() -> &'static Mutex<mpsc::Sender<Arc<Pending>>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Arc<Pending>>>> = OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Arc<Pending>>();
        let rx = Arc::new(Mutex::new(rx));
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        for k in 0..n {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                let Ok(p) = next else { return };
                // (None: a bind took it and builds it itself)
                let job = p.job.lock().unwrap().take();
                if let Some(job) = job {
                    build(&p, job, false);
                }
            });
        }
        Mutex::new(tx)
    })
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true, 0);
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
    submit_as(standin, compute, job, false, key);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64) {
    let p = Arc::new(Pending {
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(send),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        key,
        used: Default::default(),
        orphan: Default::default(),
        late: Default::default(),
        demand: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        let _ = queue().lock().unwrap().send(p);
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
        assert_eq!(
            resolve_caps(Some(0), Some(500), true),
            Some(WaitCaps { per_pso: Duration::from_millis(500), per_frame: Some(Duration::from_millis(500)) })
        );
    }

    #[test]
    fn frame_budget_caps_cumulative_wait_and_resets_on_new_frame() {
        use std::time::Duration;
        let mut fb = FrameBudget::default();
        let caps = WaitCaps { per_pso: Duration::from_millis(250), per_frame: Some(Duration::from_millis(600)) };

        assert_eq!(fb.remaining(1, caps), (Duration::from_millis(250), false));
        fb.record(1, Duration::from_millis(250));
        assert_eq!(fb.remaining(1, caps), (Duration::from_millis(250), false));
        fb.record(1, Duration::from_millis(250));
        // 100 ms left in frame 1: truncated below per_pso (250 ms)
        assert_eq!(fb.remaining(1, caps), (Duration::from_millis(100), true));
        fb.record(1, Duration::from_millis(100));
        // Exhausted for the rest of frame 1
        assert_eq!(fb.remaining(1, caps), (Duration::ZERO, true));

        // Frame 2 resets the cumulative budget
        assert_eq!(fb.remaining(2, caps), (Duration::from_millis(250), false));

        // Uncapped frame leaves every wait at full per_pso
        let uncapped = WaitCaps { per_pso: Duration::from_millis(250), per_frame: None };
        fb.record(2, Duration::from_secs(10));
        assert_eq!(fb.remaining(2, uncapped), (Duration::from_millis(250), false));
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
        submit(p1, true, Box::new(move || { let _ = rx1.recv_timeout(Duration::from_secs(5)); 0xA1 }));
        submit(p2, true, Box::new(move || { let _ = rx2.recv_timeout(Duration::from_secs(5)); 0xA2 }));
        submit(p3, true, Box::new(move || { let _ = rx3.recv_timeout(Duration::from_secs(5)); 0xA3 }));
        submit(p4, true, Box::new(move || { let _ = rx4.recv_timeout(Duration::from_secs(5)); 0xA4 }));
        submit(ready, true, Box::new(|| 0xA5));
        let start = Instant::now();
        while is_pending(ready) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(1));
        }

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
        assert!(elapsed4 < Duration::from_millis(20), "exhausted shared frame budget skips wait ({elapsed4:?})");
        assert!(t0.elapsed() < Duration::from_millis(250), "burst waited {:?}", t0.elapsed());
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
            let _ = done_rx.recv_timeout(Duration::from_secs(5));
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
                let _ = queue().lock().unwrap().send(p);
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
///   frame cap wins and bounds both per-pipeline and per-frame waits).
/// - `XCL_PSO_FRAME_WAIT_MS`: process-wide cumulative cap per `PresentX` frame in ms; `0`
///   disables the per-frame cap while keeping `XCL_PSO_WAIT_MS`.
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
}

impl FrameBudget {
    const fn new() -> Self {
        Self { frame: 0, spent: std::time::Duration::ZERO }
    }

    /// Budget left for one bind in `frame`, returning `(effective_budget, truncated_by_frame)`.
    fn remaining(&mut self, frame: u64, caps: WaitCaps) -> (std::time::Duration, bool) {
        if self.frame != frame {
            self.frame = frame;
            self.spent = std::time::Duration::ZERO;
        }
        match caps.per_frame {
            Some(limit) => {
                let left = limit.saturating_sub(self.spent);
                if left < caps.per_pso {
                    (left, true)
                } else {
                    (caps.per_pso, false)
                }
            }
            None => (caps.per_pso, false),
        }
    }

    fn record(&mut self, frame: u64, waited: std::time::Duration) {
        if self.frame != frame {
            self.frame = frame;
            self.spent = std::time::Duration::ZERO;
        }
        self.spent = self.spent.saturating_add(waited);
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
/// remaining pending binds go on with the stand-in without marking those pipelines `late`, so
/// they get a fresh wait window on the next `PresentX` frame.
fn resolve_within_in_frame(p: &Arc<Pending>, pso: usize, caps: WaitCaps, frame: u64, fb: &Mutex<FrameBudget>) -> usize {
    let built = |p: &Pending| match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    };
    if *p.done.lock().unwrap() {
        return built(p);
    }
    if !p.sent.swap(true, Ordering::AcqRel) {
        let _ = queue().lock().unwrap().send(p.clone());
    }
    if p.late.load(Ordering::Acquire) {
        return pso;
    }
    let (budget, truncated_by_frame) = fb.lock().unwrap().remaining(frame, caps);
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
    let started = std::time::Instant::now();
    let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), budget, |d| !*d).unwrap();
    fb.lock().unwrap().record(frame, started.elapsed());
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
