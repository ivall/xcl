//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for where it is used, a `Dispatch` at its submit (a skipped dispatch would corrupt what it computes).
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
    demand: Mutex<Demand>,
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

impl Pending {
    /// One of the title's threads is about to wait for it: the worker building it goes on now,
    /// at a normal priority.
    fn want(&self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_PRIORITY_NORMAL, THREAD_SET_LIMITED_INFORMATION};
        let mut d = self.demand.lock().unwrap();
        if !std::mem::replace(&mut d.wanted, true) && d.preload_thread != 0 {
            unsafe {
                let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, d.preload_thread);
                if !h.is_null() {
                    SetThreadPriority(h, THREAD_PRIORITY_NORMAL);
                    CloseHandle(h);
                }
            }
        }
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

fn queue() -> &'static Mutex<mpsc::Sender<Work>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Work>>> = OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Work>();
        let rx = Arc::new(Mutex::new(rx));
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        for k in 0..n {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                let Ok(work) = next else { return };
                while let Some(p) = next_needed() {
                    build_queued(&p);
                }
                work();
            });
        }
        Mutex::new(tx)
    })
}

fn enqueue(p: Arc<Pending>) {
    let _ = queue().lock().unwrap().send(Box::new(move || build_queued(&p)));
}

fn build_queued(p: &Arc<Pending>) {
    // (released by the title meanwhile)
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
        driver: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
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
    used::parse(&used::read_used_bounded(&crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt"), used::MAX_USED_FILE_BYTES))
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
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, RwLock, TryLockError};
    use std::{collections::HashSet, fmt::Write as _, thread::JoinHandle, time::{Duration, Instant}};

    /// Coalesce first-use bursts into a single disk write off the bind lock.
    const COALESCE_WINDOW: Duration = Duration::from_millis(25);
    /// Flush immediately once a burst reaches this many unique keys.
    const BATCH_IMMEDIATE: usize = 64;
    /// Cap initial reads of `used-pipelines.txt` so an oversized/corrupt file cannot stall startup.
    pub(super) const MAX_USED_FILE_BYTES: u64 = 16 * 1024 * 1024;

    /// Read at most `max_bytes` from `path`; if truncated, drop any trailing partial line so a
    /// mid-line cut never parses a corrupt partial hex key.
    pub(super) fn read_used_bounded(path: &Path, max_bytes: u64) -> String {
        let Ok(file) = std::fs::File::open(path) else { return String::new() };
        let mut bytes = Vec::new();
        if file.take(max_bytes.saturating_add(1)).read_to_end(&mut bytes).is_err() {
            return String::new();
        }
        if bytes.len() as u64 > max_bytes {
            bytes.truncate(max_bytes as usize);
            if let Some(pos) = bytes.iter().rposition(|&b| b == b'\n') {
                bytes.truncate(pos + 1);
            } else {
                bytes.clear();
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn lock_until<'a, T>(m: &'a Mutex<T>, deadline: Instant) -> Option<MutexGuard<'a, T>> {
        loop {
            match m.try_lock() {
                Ok(g) => return Some(g),
                Err(TryLockError::Poisoned(e)) => return Some(e.into_inner()),
                Err(TryLockError::WouldBlock) => {
                    let rem = deadline.checked_duration_since(Instant::now())?;
                    if rem.is_zero() {
                        return None;
                    }
                    std::thread::sleep(rem.min(Duration::from_millis(1)));
                }
            }
        }
    }

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
                        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty() && !d.exists()) {
                            let _ = std::fs::create_dir_all(dir);
                        }
                        *file = std::fs::OpenOptions::new().create(true).append(true).open(&*path).ok();
                    }
                    if file.as_mut().is_some_and(|f| f.write_all(bytes).is_err() || f.flush().is_err()) { *file = None; }
                }
                #[cfg(test)]
                Self::Writer(w) => if w.write_all(bytes).is_ok() { let _ = w.flush(); },
            }
        }
    }

    fn format_keys(batch: &[u64], buf: &mut String) {
        buf.clear();
        buf.reserve(batch.len() * 17);
        for &key in batch { let _ = writeln!(buf, "{key:016x}"); }
    }

    fn append_to_path(path: &Path, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return true;
        }
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty() && !d.exists()) {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut f| {
                f.write_all(bytes)?;
                f.flush()
            })
            .is_ok()
    }

    #[derive(Default)]
    struct JournalState {
        seen: HashSet<u64>,
        pending: Vec<u64>,
        /// Keys currently owned by `writer_loop` while `sink.write_batch` is executing.
        in_flight: Vec<u64>,
        in_flight_seq: u64,
        in_flight_fallback_written: bool,
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
            Self { shared: Self::new_shared(JournalState { seen, writer_stopped: true, ..Default::default() }), writer: Mutex::new(None) }
        }
        fn with_sink_and_fallback(seen: HashSet<u64>, sink: Sink, fallback_path: Option<PathBuf>, coalesce: Duration) -> Self {
            let shared = Self::new_shared(JournalState { seen, fallback_path, ..Default::default() });
            let bg = Arc::clone(&shared);
            let writer = std::thread::Builder::new().name("xcl-pso-used".into()).spawn(move || writer_loop(&bg, sink, coalesce)).ok();
            if writer.is_none() {
                shared.state.lock().unwrap_or_else(|e| e.into_inner()).writer_stopped = true;
                shared.flushed_cv.notify_all();
            }
            Self { shared, writer: Mutex::new(writer) }
        }
        fn with_sink(seen: HashSet<u64>, sink: Sink, coalesce: Duration) -> Self {
            let fallback_path = match &sink {
                Sink::LazyFile(p, _) => Some(p.clone()),
                #[cfg(test)]
                Sink::Writer(_) => None,
            };
            Self::with_sink_and_fallback(seen, sink, fallback_path, coalesce)
        }
        #[cfg(test)]
        pub(super) fn with_writer(w: Box<dyn Write + Send>, c: Duration) -> Self { Self::with_sink(HashSet::new(), Sink::Writer(w), c) }
        #[cfg(test)]
        pub(super) fn with_writer_and_fallback(w: Box<dyn Write + Send>, fallback: PathBuf, c: Duration) -> Self {
            Self::with_sink_and_fallback(HashSet::new(), Sink::Writer(w), Some(fallback), c)
        }
        #[cfg(test)]
        pub(super) fn with_file(p: PathBuf, c: Duration) -> Self { Self::with_sink(HashSet::new(), Sink::LazyFile(p, None), c) }
        #[cfg(test)]
        pub(super) fn without_writer(p: PathBuf) -> Self {
            Self { shared: Self::new_shared(JournalState { fallback_path: Some(p), writer_stopped: true, ..Default::default() }), writer: Mutex::new(None) }
        }
        #[cfg(test)]
        pub(super) fn hold_state_lock_for_test(&self, entered: std::sync::mpsc::Sender<()>, release: std::sync::mpsc::Receiver<()>) -> JoinHandle<()> {
            let shared = Arc::clone(&self.shared);
            std::thread::spawn(move || {
                let _g = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                let _ = entered.send(());
                let _ = release.recv_timeout(Duration::from_secs(5));
            })
        }

        pub(super) fn note(&self, key: u64) {
            if key == 0 { return; }
            let mut g = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            if !g.seen.insert(key) || g.closed || (g.writer_stopped && g.fallback_path.is_none()) { return; }
            g.pending.push(key);
            g.enqueued_seq += 1;
            if matches!(g.pending.len(), 1 | BATCH_IMMEDIATE) { self.shared.wake_cv.notify_one(); }
        }

        #[cfg(test)]
        pub(super) fn noted(&self, key: u64) -> bool { self.shared.state.lock().unwrap_or_else(|e| e.into_inner()).seen.contains(&key) }

        #[allow(dead_code)]
        pub(super) fn flush(&self) {
            let mut g = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            let target = g.enqueued_seq;
            if g.completed_seq >= target || g.writer_stopped { return; }
            g.flush_now = true;
            self.shared.wake_cv.notify_one();
            while g.completed_seq < target && !g.writer_stopped {
                g = self.shared.flushed_cv.wait(g).unwrap_or_else(|e| e.into_inner());
            }
        }

        pub(super) fn flush_bounded(&self, timeout: Duration) {
            let deadline = Instant::now() + timeout;
            let (path, batch, taken_pending, took_in_flight, in_flight_seq, drained_seq) = {
                let Some(mut g) = lock_until(&self.shared.state, deadline) else { return };
                let target = g.enqueued_seq;
                let has_unwritten_in_flight = !g.in_flight.is_empty() && !g.in_flight_fallback_written;
                if g.completed_seq >= target || (g.writer_stopped && g.pending.is_empty() && !has_unwritten_in_flight) {
                    return;
                }
                if !g.writer_stopped {
                    g.flush_now = true;
                    self.shared.wake_cv.notify_one();
                    while g.completed_seq < target && !g.writer_stopped {
                        let Some(rem) = deadline.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) else { break };
                        let (guard, res) = self.shared.flushed_cv.wait_timeout(g, rem).unwrap_or_else(|e| e.into_inner());
                        g = guard;
                        if res.timed_out() { break; }
                    }
                    if g.completed_seq >= target { return; }
                }
                // Worker timed out (e.g. blocked on the Windows loader lock during ExitProcess) or
                // stopped/failed to spawn: only drain when a file fallback exists so a non-file
                // sink never loses pending keys or falsely advances `completed_seq`.
                let Some(path) = g.fallback_path.clone() else { return };
                let mut took_in_flight = false;
                let mut in_flight_seq = 0u64;
                let mut batch = Vec::new();
                if !g.in_flight.is_empty() && !g.in_flight_fallback_written {
                    batch.extend_from_slice(&g.in_flight);
                    g.in_flight_fallback_written = true;
                    in_flight_seq = g.in_flight_seq;
                    took_in_flight = true;
                }
                let taken_pending = std::mem::take(&mut g.pending);
                batch.extend_from_slice(&taken_pending);
                if batch.is_empty() { return; }
                let drained_seq = g.enqueued_seq;
                (path, batch, taken_pending, took_in_flight, in_flight_seq, drained_seq)
            };
            let mut buf = String::new();
            format_keys(&batch, &mut buf);
            let wrote = append_to_path(&path, buf.as_bytes());
            if let Some(mut g) = lock_until(&self.shared.state, Instant::now() + Duration::from_millis(25)) {
                if wrote {
                    g.completed_seq = g.completed_seq.max(drained_seq);
                    self.shared.flushed_cv.notify_all();
                } else {
                    if took_in_flight && g.in_flight_seq == in_flight_seq {
                        g.in_flight_fallback_written = false;
                    }
                    if !taken_pending.is_empty() && !g.closed {
                        let mut restored = taken_pending;
                        restored.extend_from_slice(&g.pending);
                        g.pending = restored;
                        if !g.writer_stopped {
                            self.shared.wake_cv.notify_one();
                        }
                    }
                }
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
                if h.thread().id() != std::thread::current().id() { let _ = h.join(); }
            } else {
                let mut g = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
                while !g.writer_stopped { g = self.shared.flushed_cv.wait(g).unwrap_or_else(|e| e.into_inner()); }
            }
            let (fallback, batch, drained_seq) = {
                let mut g = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
                let mut batch = Vec::new();
                if !g.in_flight.is_empty() && !g.in_flight_fallback_written {
                    batch.extend(std::mem::take(&mut g.in_flight));
                    g.in_flight_seq = 0;
                    g.in_flight_fallback_written = true;
                }
                batch.extend(std::mem::take(&mut g.pending));
                if batch.is_empty() { return; }
                (g.fallback_path.clone(), batch, g.enqueued_seq)
            };
            if let Some(path) = fallback {
                let mut buf = String::new();
                format_keys(&batch, &mut buf);
                if append_to_path(&path, buf.as_bytes()) {
                    let mut g = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
                    g.completed_seq = g.completed_seq.max(drained_seq);
                    self.shared.flushed_cv.notify_all();
                }
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
                let mut g = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                while g.pending.is_empty() && !g.closed {
                    if std::mem::take(&mut g.flush_now) { shared.flushed_cv.notify_all(); }
                    g = shared.wake_cv.wait(g).unwrap_or_else(|e| e.into_inner());
                }
                if g.pending.is_empty() && g.closed { break; }
                if !coalesce.is_zero() {
                    let deadline = Instant::now() + coalesce;
                    while g.pending.len() < BATCH_IMMEDIATE && !g.flush_now && !g.closed {
                        let Some(rem) = deadline.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) else { break };
                        let (guard, res) = shared.wake_cv.wait_timeout(g, rem).unwrap_or_else(|e| e.into_inner());
                        g = guard;
                        if res.timed_out() { break; }
                    }
                }
                if g.pending.is_empty() {
                    if std::mem::take(&mut g.flush_now) { shared.flushed_cv.notify_all(); }
                    continue;
                }
                g.flush_now = false;
                let batch = std::mem::take(&mut g.pending);
                let batch_seq = g.enqueued_seq;
                g.in_flight = batch.clone();
                g.in_flight_seq = batch_seq;
                g.in_flight_fallback_written = false;
                (batch, batch_seq)
            };
            format_keys(&batch, &mut buf);
            sink.write_batch(buf.as_bytes());
            {
                let mut g = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                g.in_flight.clear();
                g.in_flight_seq = 0;
                g.in_flight_fallback_written = false;
                g.completed_seq = g.completed_seq.max(batch_seq);
            }
            shared.flushed_cv.notify_all();
        }
    }

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Journal,
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
        let earlier = parse(&read_used_bounded(&path, MAX_USED_FILE_BYTES));
        if !earlier.is_empty() {
            crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
        }
        let now = Journal::with_sink(earlier.clone(), Sink::LazyFile(path, None), coalesce);
        unsafe { atexit(on_process_exit); }
        Used { earlier: RwLock::new(earlier), now }
    }

    fn get() -> &'static Used {
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                #[cfg(test)]
                if let Some(p) = std::env::var_os("XCL_TEST_USED_STATIC_PATH") {
                    return init_from_path(PathBuf::from(p), Duration::from_secs(60));
                }
                return Used { earlier: RwLock::new(HashSet::new()), now: Journal::in_memory(HashSet::new()) };
            }
            init_from_path(crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt"), COALESCE_WINDOW)
        })
    }

    pub fn shutdown() {
        if let Some(u) = U.get() {
            u.now.shutdown();
        }
    }

    pub fn flush_bounded(timeout: Duration) {
        if let Some(u) = U.get() {
            u.now.flush_bounded(timeout);
        }
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        key != 0 && get().earlier.read().unwrap_or_else(|e| e.into_inner()).contains(&key)
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
        key != 0 && get().now.noted(key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap_or_else(|e| e.into_inner()).insert(key);
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
                if let Some(tx) = self.0.take() { let _ = tx.send(()); let _ = self.1.recv(); }
                self.2.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
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
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let rec = Arc::new(Mutex::new((0usize, Vec::new())));
        let j = used::Journal::with_writer(Box::new(Counting(rec.clone())), Duration::from_millis(500));
        for k in (0u64..=40).chain(0u64..=40) { j.note(k); }
        j.flush();
        let (writes, bytes) = { let g = rec.lock().unwrap(); (g.0, g.1.clone()) };
        let text = String::from_utf8(bytes).unwrap();
        assert!((1..40).contains(&writes), "expected coalesced writes (< 40), got {writes}");
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
            fn flush(&mut self) -> std::io::Result<()> { Err(std::io::Error::other("disk full")) }
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
                "exit" => {
                    // On Windows, `std::process::exit` calls `ExitProcess` directly (bypassing CRT
                    // `atexit`), matching a title calling `ExitProcess`; `eh::install_crash_handler`
                    // routes `NtTerminateProcess` through `flush_used_bounded` before threads end.
                    crate::eh::install_crash_handler();
                    used::note(0xE003);
                    std::process::exit(0);
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
            ("exit", &[0xE003u64][..]),
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
                if let Some(tx) = self.2.take() { let _ = tx.send(()); }
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
        let flusher = std::thread::spawn(move || { j2.flush(); let _ = done_tx.send(()); });
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

    #[test]
    fn used_flush_bounded_respects_timeout_when_state_lock_is_held() {
        use std::time::{Duration, Instant};
        let dir = std::env::temp_dir().join(format!("xcl-used-lock-bound-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("used-pipelines.txt");
        let j = used::Journal::with_file(path.clone(), Duration::from_secs(5));
        j.note(0xAA11);

        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let holder = j.hold_state_lock_for_test(entered_tx, release_rx);
        entered_rx.recv_timeout(Duration::from_secs(5)).expect("holder acquired state lock");

        let start = Instant::now();
        j.flush_bounded(Duration::from_millis(30));
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(250), "flush_bounded blocked past timeout while state lock was held ({elapsed:?})");

        release_tx.send(()).unwrap();
        holder.join().unwrap();
        j.flush();
        assert_eq!(used::parse(&std::fs::read_to_string(&path).unwrap()), [0xAA11].into());
        drop(j);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn used_flush_bounded_persists_writer_in_flight_batch_and_preserves_non_fallback_pending() {
        use std::{io::Write, time::Duration};
        struct BlockingOnce {
            entered: Option<mpsc::Sender<()>>,
            release: mpsc::Receiver<()>,
            wrote: Option<mpsc::Sender<()>>,
            out: Arc<Mutex<Vec<u8>>>,
        }
        impl Write for BlockingOnce {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if let Some(tx) = self.entered.take() {
                    let _ = tx.send(());
                    let _ = self.release.recv_timeout(Duration::from_secs(5));
                }
                self.out.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                if let Some(tx) = self.wrote.take() { let _ = tx.send(()); }
                Ok(())
            }
        }

        let dir = std::env::temp_dir().join(format!("xcl-used-inflight-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // 1a. File-backed blocking sink: writer owns `0xB001` in-flight with `pending` empty.
        let p_single = dir.join("single").join("used-pipelines.txt");
        let ((e_tx, e_rx), (r_tx, r_rx), (w_tx, w_rx)) = (mpsc::channel(), mpsc::channel(), mpsc::channel());
        let out_a = Arc::new(Mutex::new(Vec::new()));
        let j_a = used::Journal::with_writer_and_fallback(
            Box::new(BlockingOnce { entered: Some(e_tx), release: r_rx, wrote: Some(w_tx), out: Arc::clone(&out_a) }),
            p_single.clone(),
            Duration::ZERO,
        );
        j_a.note(0xB001);
        e_rx.recv_timeout(Duration::from_secs(5)).expect("writer took 0xB001 in-flight");
        j_a.flush_bounded(Duration::from_millis(20));
        assert_eq!(
            used::parse(&std::fs::read_to_string(&p_single).expect("in-flight batch must persist on timeout")),
            [0xB001].into(),
        );
        r_tx.send(()).unwrap();
        w_rx.recv_timeout(Duration::from_secs(5)).expect("writer finished 0xB001");
        j_a.flush();
        drop(j_a);

        // 1b. File-backed blocking sink: writer owns `0xB001` in-flight AND `0xB002` is queued in `pending`.
        let p_both = dir.join("both").join("used-pipelines.txt");
        let ((e_tx, e_rx), (r_tx, r_rx), (w_tx, w_rx)) = (mpsc::channel(), mpsc::channel(), mpsc::channel());
        let out_b = Arc::new(Mutex::new(Vec::new()));
        let j_b = Arc::new(used::Journal::with_writer_and_fallback(
            Box::new(BlockingOnce { entered: Some(e_tx), release: r_rx, wrote: Some(w_tx), out: Arc::clone(&out_b) }),
            p_both.clone(),
            Duration::ZERO,
        ));
        j_b.note(0xB001);
        e_rx.recv_timeout(Duration::from_secs(5)).expect("writer took 0xB001 in-flight");
        j_b.note(0xB002);
        j_b.flush_bounded(Duration::from_millis(20));
        assert_eq!(
            used::parse(&std::fs::read_to_string(&p_both).expect("both in-flight and pending must persist")),
            [0xB001, 0xB002].into(),
        );
        r_tx.send(()).unwrap();
        w_rx.recv_timeout(Duration::from_secs(5)).expect("writer finished 0xB001");
        let (done_tx, done_rx) = mpsc::channel();
        let j_b2 = Arc::clone(&j_b);
        let flusher = std::thread::spawn(move || { j_b2.flush(); let _ = done_tx.send(()); });
        done_rx.recv_timeout(Duration::from_secs(5)).expect("subsequent flush() must return promptly");
        flusher.join().unwrap();
        drop(j_b);

        // 2. Non-fallback `Sink::Writer`: timed-out `flush_bounded` must leave both `in_flight` and
        // `pending` intact so the writer persists both keys once unblocked.
        let ((e_tx, e_rx), (r_tx, r_rx), (w_tx, w_rx)) = (mpsc::channel(), mpsc::channel(), mpsc::channel());
        let out_nf = Arc::new(Mutex::new(Vec::new()));
        let j_nf = used::Journal::with_writer(
            Box::new(BlockingOnce { entered: Some(e_tx), release: r_rx, wrote: Some(w_tx), out: Arc::clone(&out_nf) }),
            Duration::ZERO,
        );
        j_nf.note(0xC001);
        e_rx.recv_timeout(Duration::from_secs(5)).expect("writer took 0xC001 in-flight");
        j_nf.note(0xC002);
        j_nf.flush_bounded(Duration::from_millis(20));
        r_tx.send(()).unwrap();
        w_rx.recv_timeout(Duration::from_secs(5)).expect("writer finished 0xC001");
        j_nf.flush();
        assert_eq!(
            used::parse(std::str::from_utf8(&out_nf.lock().unwrap()).unwrap()),
            [0xC001, 0xC002].into(),
            "non-fallback flush_bounded must not drop pending 0xC002 or prematurely advance completed_seq",
        );
        drop(j_nf);

        // 3. Panicking writer with `fallback_path`: `shutdown()` persists the panicked in-flight batch.
        struct PanicOnce(Option<mpsc::Sender<()>>);
        impl Write for PanicOnce {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                if let Some(tx) = self.0.take() { let _ = tx.send(()); }
                panic!("simulated mid-batch writer panic");
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let p_panic = dir.join("panic").join("used-pipelines.txt");
        let (p_tx, p_rx) = mpsc::channel();
        let j_p = used::Journal::with_writer_and_fallback(Box::new(PanicOnce(Some(p_tx))), p_panic.clone(), Duration::ZERO);
        j_p.note(0xD001);
        p_rx.recv_timeout(Duration::from_secs(5)).expect("writer entered panicking write");
        j_p.shutdown();
        assert_eq!(
            used::parse(&std::fs::read_to_string(&p_panic).expect("shutdown must drain panicked in-flight batch")),
            [0xD001].into(),
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn used_bounded_read_caps_oversized_file_on_line_boundary() {
        let dir = std::env::temp_dir().join(format!("xcl-used-read-bound-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Cutting mid-way through line 2 ("5555666677778888\n") must drop the partial line so the
        // truncated hex prefix ("55556666") is never parsed as a bogus key.
        let p = dir.join("used-pipelines.txt");
        let content = "1111222233334444\n5555666677778888\n9999aaaabbbbcccc\n";
        std::fs::write(&p, content).unwrap();
        let capped = used::read_used_bounded(&p, 25);
        assert_eq!(capped, "1111222233334444\n");
        assert_eq!(used::parse(&capped), [0x1111_2222_3333_4444].into());

        // Cutting inside the very first line (no newline before cap) yields empty string.
        assert!(used::read_used_bounded(&p, 8).is_empty());

        // Exact-size file without trailing newline is not treated as truncated.
        let p_exact = dir.join("exact.txt");
        std::fs::write(&p_exact, "1111222233334444").unwrap();
        let full = used::read_used_bounded(&p_exact, 16);
        assert_eq!(used::parse(&full), [0x1111_2222_3333_4444].into());

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
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
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
            // a lazy one is wanted now: to the workers
            let stand = p.stand(pso);
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue(p);
            }
            stand
        }
        r => r,
    }
}

/// `resolve`, waiting for a pipeline still being built – built here if no worker has started
/// it yet (a queue of a title's hundreds of pipelines must not hold up the one it binds now).
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_for(pso, wait_budget())
}

/// `resolve_wait`, waiting at most `budget` (None: until the pipeline is built).
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    if p.spared.load(Ordering::Acquire) {
        return resolve_spared(&p, pso);
    }
    // (the title's thread is held from here until the pipeline is built or the budget is over)
    let _held = (!*p.done.lock().unwrap()).then(Held::new);
    if let Some(budget) = budget {
        return resolve_within(&p, pso, budget);
    }
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
        0 => p.stand(pso),
        r => r,
    }
}

/// How long a bind waits for a pipeline still being built (`XCL_PSO_WAIT_MS`; unset or 0: until
/// it is built). A retail game builds compute programs on its own threads: the driver takes up to
/// 30 s for the biggest translated ones, and the title's watchdog ends it when its main thread
/// has been held 30 s.
fn wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0).map(std::time::Duration::from_millis))
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
            p.stand(pso)
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

/// One of the title's threads is about to wait for `pso`.
pub fn want(pso: usize) {
    if let Some(p) = entry(pso) {
        p.want();
    }
}

/// A call with `pso` was put off to the submit: its build goes before the workers' queue.
pub fn need_by_submit(pso: usize) {
    let Some(p) = entry(pso) else { return };
    if !std::mem::replace(&mut p.demand.lock().unwrap().by_submit, true) {
        NEEDED.lock().unwrap().push_back(p);
        let _ = queue().lock().unwrap().send(Box::new(|| ()));
    }
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}
