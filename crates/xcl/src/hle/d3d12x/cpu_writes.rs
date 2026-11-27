//! Write32/64BitValueBottomOfPipeX into memory the GPU does not see: done by the CPU once the GPU
//! has finished the work before them.
//!
//! A console title marks its GPU progress with bottom-of-pipe writes into its own memory and
//! waits on them (a retail game: plain `VirtualAlloc` memory, outside XCL's unified memory).
//! The PC GPU cannot write there; dropped, the title waited for ever after its second frame.
//! Here each queue gets a fence of XCL's: after a submission whose lists carry such writes the
//! queue signals it, and a worker thread does the writes, in submission order, when the GPU is
//! there – what bottom of pipe means.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex};

use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Threading as th;

/// A bottom-of-pipe write into memory outside unified memory, applied by the CPU once the
/// queue's fence reaches the submission that recorded it.
#[derive(Clone)]
pub(super) enum Write {
    /// Immediate dwords to store starting at `va`.
    Immediate(usize, Vec<u32>),
    /// A 64-bit value resolved by the GPU into a persistently mapped READBACK buffer slot
    /// (`src`), copied atomically to `va` once the queue's fence completes. Holding `_buf` keeps
    /// the mapping alive until the worker finishes.
    ReadbackU64 { va: usize, src: usize, _buf: ID3D12Resource },
    /// A deferred `Write64BitValueBottomOfPipeX` into a registered sync cell: the store must
    /// wait for confirmed producer-fence completion, stale values never overwrite newer memory,
    /// and the sync-cell fence is published only after the store succeeds. The third field is
    /// the cell's registration generation at record time; a job from a retired era neither
    /// stores nor publishes.
    SyncU64(usize, u64, u64),
}

struct Job {
    fence: ID3D12Fence,
    value: u64,
    writes: Vec<Write>,
}

// SAFETY: the fence is a free-threaded COM object; the writes are plain data
unsafe impl Send for Job {}

/// queue → its fence, the last value signalled, the channel feeding its worker, and the
/// release-cancellation flag shared with that worker
struct QueueFence {
    fence: ID3D12Fence,
    value: u64,
    tx: mpsc::Sender<Job>,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

unsafe impl Send for QueueFence {}

static FENCES: Mutex<Option<HashMap<usize, QueueFence>>> = Mutex::new(None);

/// how long the worker waits for one submission before writing anyway (a removed device, a
/// queue that never gets there)
const WAIT_MS: u32 = 10_000;

/// Workers alive right now (test-only lifecycle evidence; compiled out of production).
#[cfg(test)]
pub(super) static LIVE_WORKERS: AtomicUsize = AtomicUsize::new(0);

/// Test-only seam: the queue key whose device queries all report removal. Compiled out of
/// production builds; production removal detection uses the real D3D12 device query only.
#[cfg(test)]
pub(super) static REMOVAL_INJECT_QUEUE: AtomicUsize = AtomicUsize::new(0);

/// Whether `queue`'s device is gone: a removed device reports its reason as an error, and a
/// failed query is never proof of health - the only healthy answer is a successful query
/// returning `Ok`. Never a completion signal: nothing here advances any fence or watermark.
unsafe fn device_gone(queue: usize, fence: &ID3D12Fence) -> bool {
    #[cfg(test)]
    if REMOVAL_INJECT_QUEUE.load(Ordering::SeqCst) == queue {
        return true;
    }
    let mut dev: Option<ID3D12Device> = None;
    if fence.GetDevice(&mut dev).is_err() {
        return true;
    }
    let Some(dev) = dev else { return true };
    let _ = queue;
    dev.GetDeviceRemovedReason().is_err()
}

/// The title released `queue`'s last reference (`list_x::on_queue_release`): cancel the
/// worker's strict waits (grace-bounded, see the worker loop) and drop the sender so no new
/// jobs are accepted. Already-buffered jobs keep their fence objects alive; a drained worker
/// then exits on the closed channel. Cancellation never publishes anything: a job whose
/// producer completion cannot be confirmed within the grace is dropped without storing.
pub(super) fn on_queue_released(queue: usize) {
    if let Some(m) = FENCES.lock().unwrap().as_mut() {
        if let Some(qf) = m.remove(&queue) {
            qf.cancel.store(true, Ordering::SeqCst);
        }
    }
}

/// Whether `queue` still has a live worker registration (test evidence for release paths).
#[cfg(test)]
pub(super) fn queue_registered(queue: usize) -> bool {
    FENCES.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(&queue))
}

/// The worker decision site performing a completion probe. Production passes the site through
/// unchanged; only test builds use it to authorize scripted probe results. Polling carries its
/// zero-based iteration so out-of-order or extra polling probes are distinguishable.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub(super) enum ProbeSite {
    StrictPickup,
    StrictPolling(u64),
    LegacyNonStrict,
}

/// D1.16 test seam: a per-queue ordered script of site-aware probe expectations. A probe is
/// authorized only when its exact site matches the next expectation. Unexpected probes are
/// recorded as script errors, do not consume expectations, and receive a conservative refusal
/// value solely so the worker can reach test-controlled cleanup. Compiled out of production;
/// with no script for a queue the real fence value is returned.
#[cfg(test)]
struct ProbeScript {
    expectations: Vec<(ProbeSite, u64)>,
    next: usize,
    errors: Vec<String>,
}

#[cfg(test)]
static PROBE_SCRIPTS: Mutex<Option<HashMap<usize, ProbeScript>>> = Mutex::new(None);

#[cfg(test)]
fn probe_scripts_lock() -> std::sync::MutexGuard<'static, Option<HashMap<usize, ProbeScript>>> {
    PROBE_SCRIPTS.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
pub(super) fn arm_probe_script(queue: usize, expectations: &[(ProbeSite, u64)]) {
    assert!(queue != 0, "probe scripts are scoped to a real queue key");
    assert!(!expectations.is_empty(), "a probe script needs at least one expectation");
    let script = ProbeScript { expectations: expectations.to_vec(), next: 0, errors: Vec::new() };
    probe_scripts_lock().get_or_insert_with(HashMap::new).insert(queue, script);
}

#[cfg(test)]
pub(super) fn clear_probe_script(queue: usize) {
    if let Some(m) = probe_scripts_lock().as_mut() {
        m.remove(&queue);
    }
}

#[cfg(test)]
fn refusal_value(site: ProbeSite) -> u64 {
    match site {
        ProbeSite::LegacyNonStrict => 0,
        ProbeSite::StrictPickup | ProbeSite::StrictPolling(_) => u64::MAX,
    }
}

/// One scripted probe, or None when no script exists for `queue`. This is the state machine
/// used by `probe_completed`; unit tests drive it directly for negative controls.
#[cfg(test)]
fn scripted_probe(queue: usize, site: ProbeSite) -> Option<u64> {
    if queue == 0 {
        return None;
    }
    let mut scripts = probe_scripts_lock();
    let script = scripts.as_mut().and_then(|m| m.get_mut(&queue))?;
    let index = script.next;
    let Some(&(expected_site, value)) = script.expectations.get(index) else {
        script.errors.push(format!("unexpected probe {site:?}: script exhausted after {} expectations", script.expectations.len()));
        return Some(refusal_value(site));
    };
    if expected_site != site {
        script.errors.push(format!("unexpected probe {site:?}: expectation {index} is {expected_site:?}"));
        return Some(refusal_value(site));
    }
    script.next += 1;
    Some(value)
}

/// None when the script is clean or absent; otherwise a diagnostic containing either the
/// first unexpected-probe error or the next unconsumed expectation.
#[cfg(test)]
pub(super) fn probe_script_problem(queue: usize) -> Option<String> {
    let scripts = probe_scripts_lock();
    let script = scripts.as_ref().and_then(|m| m.get(&queue))?;
    if let Some(error) = script.errors.first() {
        return Some(format!("{} probe script error(s); first: {error}", script.errors.len()));
    }
    if script.next < script.expectations.len() {
        let (site, _) = script.expectations[script.next];
        return Some(format!("unconsumed expectation {} of {}: {site:?}", script.next, script.expectations.len()));
    }
    None
}

/// The fence's completed value as the worker's decision input. `UINT64_MAX` is the
/// device-removal sentinel and must never be numerically compared against job values.
/// The test override is scoped to one queue so concurrent tests on other queues are unaffected.
unsafe fn probe_completed(queue: usize, fence: &ID3D12Fence, site: ProbeSite) -> u64 {
    #[cfg(test)]
    if let Some(value) = scripted_probe(queue, site) {
        return value;
    }
    let _ = queue;
    let _ = site;
    fence.GetCompletedValue()
}

/// A fence probe classified against the job's target value. `UINT64_MAX` is the
/// device-removal sentinel: the batch provably never completed, and the value must never
/// be numerically compared against job values - `MAX >= value` is always true and
/// `MAX < value` never is, so both comparisons lie for a removed device.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Completion {
    Complete,
    Incomplete,
    DeviceRemoved,
}

unsafe fn classify_completion(queue: usize, fence: &ID3D12Fence, value: u64, site: ProbeSite) -> Completion {
    let completed = probe_completed(queue, fence, site);
    if completed == u64::MAX {
        return Completion::DeviceRemoved;
    }
    if completed >= value {
        Completion::Complete
    } else {
        Completion::Incomplete
    }
}

/// Spawn one worker thread for `queue`, fed by the channel paired with the sender kept in its
/// `QueueFence`. Per-queue workers keep a queue blocked on its own producer fence from delaying
/// unrelated queues' writes, and give strict sync-cell jobs an independent completion path.
fn spawn_worker(queue: usize, rx: mpsc::Receiver<Job>, cancel: std::sync::Arc<std::sync::atomic::AtomicBool>) {
    let spawned = std::thread::Builder::new().name(format!("xcl-cpu-writes-q{queue:x}")).spawn(move || unsafe {
        #[cfg(test)]
        LIVE_WORKERS.fetch_add(1, Ordering::SeqCst);
        let event = th::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
        for job in rx {
            let strict = job.writes.iter().any(|w| matches!(w, Write::SyncU64(..)));
            // Set when the device is removed before the producer batch completed: the batch
            // provably never executed, so its sync-cell writes must never be stored.
            let mut removed_incomplete = false;
            // Set when a released queue's grace expired while the producer fence was still
            // incomplete: the job is abandoned - never stored, never published.
            let mut abandoned = false;
            // A strict job's pickup decision is made by exactly one classified probe: the
            // sentinel is never numerically compared. Legacy non-strict jobs keep their
            // historical raw comparison in the else arm.
            let wait_needed = if strict {
                match classify_completion(queue, &job.fence, job.value, ProbeSite::StrictPickup) {
                    Completion::Complete => false,
                    Completion::Incomplete => true,
                    Completion::DeviceRemoved => {
                        crate::xlog_warn!("D3D12.X: device removed before deferred sync-cell writes could wait for their queue - dropping {} writes", job.writes.len());
                        removed_incomplete = true;
                        false
                    }
                }
            } else {
                probe_completed(queue, &job.fence, ProbeSite::LegacyNonStrict) < job.value
            };
            if wait_needed {
                if strict {
                    // A job carrying sync-cell writes must never publish before its producer
                    // batch completes: on console the write did not happen. Poll until the
                    // fence completes or the device is removed; no timeout publishes early.
                    // Queue release does not cancel the wait outright - a slow-but-legitimate
                    // batch still lands - but bounds it: `WAIT_MS` of grace after cancellation,
                    // then the job is abandoned rather than waiting forever.
                    let mut deadline: Option<std::time::Instant> = None;
                    let mut polling_probe = 0u64;
                    loop {
                        let completion = classify_completion(queue, &job.fence, job.value, ProbeSite::StrictPolling(polling_probe));
                        polling_probe += 1;
                        match completion {
                            Completion::Complete => break,
                            Completion::DeviceRemoved => {
                                crate::xlog_warn!("D3D12.X: device removed while deferred writes waited for their queue (removal sentinel, wanted {}) - dropping {} writes", job.value, job.writes.len());
                                removed_incomplete = true;
                                break;
                            }
                            Completion::Incomplete => {}
                        }
                        if device_gone(queue, &job.fence) {
                            crate::xlog_warn!("D3D12.X: device removed while deferred writes waited for their queue (fence at {}, wanted {}) - dropping {} writes", job.fence.GetCompletedValue(), job.value, job.writes.len());
                            removed_incomplete = true;
                            break;
                        }
                        if cancel.load(Ordering::SeqCst) {
                            let d = deadline.get_or_insert_with(|| std::time::Instant::now() + std::time::Duration::from_millis(WAIT_MS as u64));
                            if std::time::Instant::now() >= *d {
                                crate::xlog_warn!("D3D12.X: deferred writes on a released queue never saw their producer batch complete (fence at {}, wanted {}) - dropping {} writes", job.fence.GetCompletedValue(), job.value, job.writes.len());
                                abandoned = true;
                                break;
                            }
                        }
                        std::thread::sleep(std::time::Duration::from_millis(20));
                    }
                } else if job.fence.SetEventOnCompletion(job.value, HANDLE(event as *mut c_void)).is_ok() {
                    th::WaitForSingleObject(event, WAIT_MS);
                    if job.fence.GetCompletedValue() < job.value {
                        static LATE: AtomicU64 = AtomicU64::new(0);
                        let k = LATE.fetch_add(1, Ordering::Relaxed) + 1;
                        if k.is_power_of_two() {
                            crate::xlog_warn!("D3D12.X: {k} writes after the GPU so far done although their queue had not got there after {WAIT_MS} ms (fence at {}, wanted {}; {} writes)", job.fence.GetCompletedValue(), job.value, job.writes.len());
                        }
                    }
                }
            }
            for w in &job.writes {
                match w {
                    Write::Immediate(va, words) => write_now(*va, words),
                    Write::ReadbackU64 { va, src, .. } => {
                        let value = (*(*src as *const AtomicU64)).load(Ordering::Acquire);
                        write_u64_now(*va, value);
                    }
                    Write::SyncU64(va, value, generation) => {
                        if removed_incomplete || abandoned {
                            // The producer batch never provably completed: on console this
                            // write never happened. Never store or publish it.
                            continue;
                        }
                        let (va, value, generation) = (*va, *value, *generation);
                        // Staleness decision, store, watermark update and fence publication
                        // linearize in one SYNC_GRAPH critical section inside list_x.
                        super::list_x::sync_cell_deferred_store(va, value, generation, || write_u64_now(va, value));
                    }
                }
            }
        }
        #[cfg(test)]
        LIVE_WORKERS.fetch_sub(1, Ordering::SeqCst);
    });
    if let Err(e) = spawned {
        crate::xlog_error!("D3D12.X: no thread for the writes after the GPU: {e}");
    }
}

/// The dwords at `va`, if the memory is still there and writable.
unsafe fn write_now(va: usize, words: &[u32]) {
    let bytes = 4 * words.len();
    if !crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, bytes)) {
        use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_READWRITE};
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT {
            return;
        }
        if m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) == 0 || (m.BaseAddress as usize + m.RegionSize) < va + bytes {
            return;
        }
    }
    for (i, &w) in words.iter().enumerate() {
        (*((va + 4 * i) as *const AtomicU32)).store(w, Ordering::Release);
    }
}

/// Whether `[va, va + 8)` is 8-byte aligned, committed, and writable without `PAGE_GUARD`.
unsafe fn u64_writable_now(va: usize) -> bool {
    if va == 0 || !va.is_multiple_of(8) {
        return false;
    }
    let Some(end) = va.checked_add(8) else { return false };
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, 8)) {
        return true;
    }
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_READWRITE};
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT {
        return false;
    }
    m.Protect & PAGE_GUARD == 0
        && m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0
        && (m.BaseAddress as usize).saturating_add(m.RegionSize) >= end
}

/// An 8-byte aligned 64-bit value at `va`, if the memory is still there and writable.
/// Returns whether the store actually happened: sync-cell publication depends on the answer.
unsafe fn write_u64_now(va: usize, value: u64) -> bool {
    if u64_writable_now(va) {
        (*(va as *const AtomicU64)).store(value, Ordering::Release);
        return true;
    }
    false
}

/// Advance an 8-byte aligned 64-bit value at `va` to at least `value` (`fetch_max`), if the memory
/// is still committed and writable. Best-effort check before the atomic write (same TOCTOU
/// trade-off as `write_u64_now`: a concurrent title decommit between check and store is unguarded).
/// Returns whether the store actually happened.
pub(super) unsafe fn max_u64_now(va: usize, value: u64) -> bool {
    if u64_writable_now(va) {
        (*(va as *const AtomicU64)).fetch_max(value, Ordering::AcqRel);
        return true;
    }
    false
}

/// Whether `[va, va + bytes)` is memory the CPU can write for the GPU later.
pub(super) fn writable(va: usize, bytes: usize) -> bool {
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, bytes)) {
        return true;
    }
    crate::hle::kernelx::lazy::ensure(va, bytes);
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_READWRITE};
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let found = unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0;
    found
        && m.State == MEM_COMMIT
        && m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0
        && m.BaseAddress as usize + m.RegionSize >= va + bytes
}

/// `writes` of lists just submitted to `queue`: done when the GPU has finished them.
pub(super) unsafe fn after(queue: *mut c_void, writes: Vec<Write>) {
    if writes.is_empty() || queue.is_null() {
        return;
    }
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    {
        let mut g = FENCES.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        let entry = match m.entry(queue as usize) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let mut dev: Option<ID3D12Device> = None;
                if q.GetDevice(&mut dev).is_err() {
                    return;
                }
                let Some(dev) = dev else { return };
                let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return };
                let (tx, rx) = mpsc::channel::<Job>();
                let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                spawn_worker(queue as usize, rx, cancel.clone());
                v.insert(QueueFence { fence, value: 0, tx, cancel })
            }
        };
        entry.value += 1;
        if q.Signal(&entry.fence, entry.value).is_err() {
            return;
        }
        let nwrites = writes.len();
        // The fence signal and the job send stay in submission order under concurrent
        // submission: both happen inside this lock.
        let _ = entry.tx.send(Job { fence: entry.fence.clone(), value: entry.value, writes });
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| crate::xlog_info!("D3D12.X: bottom-of-pipe writes outside unified memory are done by the CPU after the GPU ({nwrites} in the first submission)"));
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Dxgi::*;

    fn finish_script_test(queue: usize) -> String {
        let problem = probe_script_problem(queue).unwrap_or_else(|| "no probe-script problem was recorded".into());
        clear_probe_script(queue);
        assert!(probe_script_problem(queue).is_none(), "test cleanup failed");
        problem
    }

    /// A duplicate pickup probe is rejected immediately and cannot consume the polling value.
    #[test]
    fn d116_probe_script_rejects_duplicate_strict_pickup_without_consuming_polling() {
        let queue = usize::MAX - 0x11000;
        arm_probe_script(queue, &[(ProbeSite::StrictPickup, 0), (ProbeSite::StrictPolling(0), 42)]);
        assert_eq!(scripted_probe(queue, ProbeSite::StrictPickup), Some(0));
        assert_eq!(scripted_probe(queue, ProbeSite::StrictPickup), Some(u64::MAX));
        assert_eq!(scripted_probe(queue, ProbeSite::StrictPolling(0)), Some(42));
        let problem = finish_script_test(queue);
        assert!(problem.contains("unexpected probe StrictPickup"), "{problem}");
        assert!(problem.contains("expectation 1 is StrictPolling(0)"), "{problem}");
    }

    /// A polling probe with the wrong iteration is rejected and does not shift the script.
    #[test]
    fn d116_probe_script_rejects_out_of_order_polling_without_consuming_it() {
        let queue = usize::MAX - 0x12000;
        arm_probe_script(queue, &[(ProbeSite::StrictPickup, 0), (ProbeSite::StrictPolling(0), 42)]);
        assert_eq!(scripted_probe(queue, ProbeSite::StrictPickup), Some(0));
        assert_eq!(scripted_probe(queue, ProbeSite::StrictPolling(1)), Some(u64::MAX));
        assert_eq!(scripted_probe(queue, ProbeSite::StrictPolling(0)), Some(42));
        let problem = finish_script_test(queue);
        assert!(problem.contains("unexpected probe StrictPolling(1)"), "{problem}");
        assert!(problem.contains("expectation 1 is StrictPolling(0)"), "{problem}");
    }

    /// Once every expectation is consumed, another probe is an exhaustion error, not a repeat.
    #[test]
    fn d116_probe_script_rejects_probe_after_exhaustion() {
        let queue = usize::MAX - 0x13000;
        arm_probe_script(queue, &[(ProbeSite::StrictPickup, 0)]);
        assert_eq!(scripted_probe(queue, ProbeSite::StrictPickup), Some(0));
        assert_eq!(scripted_probe(queue, ProbeSite::StrictPolling(0)), Some(u64::MAX));
        let problem = finish_script_test(queue);
        assert!(problem.contains("script exhausted after 1 expectations"), "{problem}");
    }

    /// The harness reports an authorized expectation the worker never reached.
    #[test]
    fn d116_probe_script_detects_unconsumed_expectation() {
        let queue = usize::MAX - 0x14000;
        arm_probe_script(queue, &[(ProbeSite::StrictPickup, 0), (ProbeSite::StrictPolling(0), u64::MAX)]);
        assert_eq!(scripted_probe(queue, ProbeSite::StrictPickup), Some(0));
        let problem = finish_script_test(queue);
        assert!(problem.contains("unconsumed expectation 1 of 2: StrictPolling(0)"), "{problem}");
    }

    /// A strict-path script never doubles as authorization for a legacy probe.
    #[test]
    fn d116_probe_script_rejects_legacy_probe_not_in_script() {
        let queue = usize::MAX - 0x15000;
        arm_probe_script(queue, &[(ProbeSite::StrictPickup, 7)]);
        assert_eq!(scripted_probe(queue, ProbeSite::LegacyNonStrict), Some(0));
        assert_eq!(scripted_probe(queue, ProbeSite::StrictPickup), Some(7));
        let problem = finish_script_test(queue);
        assert!(problem.contains("unexpected probe LegacyNonStrict"), "{problem}");
        assert!(problem.contains("expectation 0 is StrictPickup"), "{problem}");
    }

    /// The value lands after the queue has done what came before it.
    #[test]
    fn writes_land_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let cell = Box::new([0u32; 2]);
            let va = cell.as_ptr() as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Write::Immediate(va, vec![7, 9])]);
            let t = std::time::Instant::now();
            while (*(va as *const AtomicU32)).load(Ordering::Acquire) != 7 {
                assert!(t.elapsed().as_secs() < 5, "never written");
                std::thread::yield_now();
            }
            assert_eq!(*cell, [7, 9]);
        }
    }

    /// A ReadbackU64 write copies the 64-bit value from the mapped readback buffer after the queue finishes.
    #[test]
    fn readback_u64_lands_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut rb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
            let rb = rb.unwrap();
            let mut ptr: *mut c_void = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut ptr)).unwrap();
            let expected = 0x1234_5678_9ABC_DEF0u64;
            (*(ptr as *const AtomicU64)).store(expected, Ordering::Release);

            let dst = Box::new(0u64);
            let va = dst.as_ref() as *const u64 as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Write::ReadbackU64 { va, src: ptr as usize, _buf: rb }]);
            let t = std::time::Instant::now();
            while (*(va as *const AtomicU64)).load(Ordering::Acquire) != expected {
                assert!(t.elapsed().as_secs() < 5, "readback value never written");
                std::thread::yield_now();
            }
            assert_eq!(*dst, expected);
        }
    }
}
