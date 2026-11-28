//! PSOs for a split compute entry. The first piece is the real PSO returned
//! to the caller. Other pieces and the lazily built whole form belong to it.
use super::*;
use crate::gpu::{dxil::kernel::DispatchChain, recomp::pso};
use std::sync::{Arc, LazyLock, OnceLock};

static PROGRAMS: AtomicU64 = AtomicU64::new(0);
static PIECES: AtomicU64 = AtomicU64::new(0);
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
/// slabs dispatched (`pso::dispatch_slabs`)
static SLABS: AtomicU64 = AtomicU64::new(0);
/// scratch alive now, the most there was, and the most a dispatch would have needed at once (bytes)
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static WANTED: AtomicU64 = AtomicU64::new(0);

/// (only when a count changed since the last line: called every 10 s in every run)
pub(crate) fn log_counts() {
    static LAST: Mutex<[u64; 7]> = Mutex::new([0; 7]);
    let now = [&PROGRAMS, &PIECES, &DISPATCHES, &SLABS, &LIVE, &PEAK, &WANTED].map(|c| c.load(Ordering::Relaxed));
    if std::mem::replace(&mut *LAST.lock().unwrap(), now) == now {
        return;
    }
    let mib = |a: &AtomicU64| a.load(Ordering::Relaxed) >> 20;
    crate::xlog_info!("compute split: {} programs split, {} pieces built, {} chained dispatches issued in {} slabs; scratch {} MiB (most {}, largest dispatch {} in one slab)",
        PROGRAMS.load(Ordering::Relaxed), PIECES.load(Ordering::Relaxed), DISPATCHES.load(Ordering::Relaxed), SLABS.load(Ordering::Relaxed),
        mib(&LIVE), mib(&PEAK), mib(&WANTED));
}

/// The scratch a dispatch uses at once, `XCL_SPLIT_SCRATCH_MB` (256): more groups than fit run in
/// slabs. A retail title's split programs need 190 KB for each group (median), 3 GB for 960x540
/// pixels in groups of 8x4.
fn scratch_budget() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_SPLIT_SCRATCH_MB").ok().and_then(|v| v.trim().parse().ok()).filter(|&m| m > 0).unwrap_or(256u64).min(4096) << 20)
}

struct PipelineBuilder {
    device: ID3D12Device,
    create: PsoFn,
    signature: ID3D12RootSignature,
    flags: D3D12_PIPELINE_STATE_FLAGS,
    node: u32,
    record: u64,
}

impl PipelineBuilder {
    unsafe fn pipeline(&self, dxil: &[u8], iid: &GUID) -> Option<ID3D12PipelineState> {
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { NodeMask: self.node, Flags: self.flags, ..Default::default() };
        let mut raw = std::ptr::null_mut();
        let (hr, in_driver, _) = compute_container(self.device.as_raw(), self.create, &desc, self.signature.as_raw(),
            dxil, self.record, iid, &mut raw, ComputeBuild::Chain);
        // (as `translated_compute_pso_once` logs a huge whole program the driver refuses)
        if hr < 0 && dxil.len() >= huge_pso_bytes() {
            crate::xlog_warn!(
                "D3D12.X: the driver refused a dispatch chain's compute PSO of record {:016x} ({} B DXIL, driver cache {}) with {hr:#x} after {} ms in the driver",
                self.record,
                dxil.len(),
                driver_cache::name(dxil, self.flags.0),
                in_driver.as_millis()
            );
        }
        (hr >= 0 && !raw.is_null()).then(|| ID3D12PipelineState::from_raw(raw))
    }
}

/// What a chain's whole form is made from: its console program, translated again when the whole
/// form is first wanted (from the translation caches: the same DXIL). Most chains never want it
/// (only an indirect dispatch runs the whole form), and their whole DXIL kept in memory was a
/// retail title's 148 MB for 198 split programs, against 22 KB of console code on average.
struct WholeSource {
    console: Vec<u8>,
    wave32: bool,
    // Preserve the first translation's window count and heap/switch form: a later worker
    // does not inherit `in_switch_form` from the thread that built the split pieces.
    windows: u32,
}

impl WholeSource {
    fn code(&self) -> Result<Vec<u8>, String> {
        pso::console_to_dxil_in(&self.console, pso::Stage::Compute, 0, self.wave32, self.windows).map(|c| c.dxil)
    }
}

struct Chain {
    builder: Arc<PipelineBuilder>,
    tail: Vec<ID3D12PipelineState>,
    words: u32,
    whole_source: WholeSource,
    whole: OnceLock<ID3D12PipelineState>,
    build_whole: Mutex<()>,
    /// a thread of its own builds the whole form (`whole_ready`)
    whole_sent: std::sync::atomic::AtomicBool,
}

/// A chain for the thread that builds its whole form (D3D12 objects are free-threaded).
struct ChainRef(Arc<Chain>);
unsafe impl Send for ChainRef {}

impl Chain {
    /// Build the whole form. `on_title_thread` is true only when a title thread is synchronously
    /// blocked waiting for this compile (registers a long PC call).
    unsafe fn whole(&self, on_title_thread: bool) -> Result<ID3D12PipelineState, String> {
        if let Some(p) = self.whole.get() { return Ok(p.clone()); }
        // (only when a title thread is held while this compile lasts, as in any other long PC call)
        let _call = on_title_thread.then(super::super::pso_async::pc_call);
        let _guard = self.build_whole.lock().unwrap();
        if let Some(p) = self.whole.get() { return Ok(p.clone()); }
        let code = self.whole_source.code().map_err(|e| format!("whole compute program: {e}"))?;
        let p = self.builder.pipeline(&code, &pc_iid(Kind::PipelineState)).ok_or("whole compute pipeline")?;
        let _ = self.whole.set(p.clone());
        Ok(p)
    }
}

type WholeJob = Box<dyn FnOnce() + Send + 'static>;

/// Single persistent worker (`xcl-pso-whole`, 64 MiB stack) for building split programs' whole
/// forms off the title thread. A single worker avoids spawning an unbounded number of 64 MiB-stack
/// OS threads behind `real::huge_compile_turn` (which already serializes late giant compiles), and
/// runs each job under `pso_async::with_late_build` so it gets half commit headroom. Each job is
/// wrapped in `catch_unwind` so a panic cannot kill the worker and strand queued chains.
fn whole_worker() -> Option<&'static std::sync::mpsc::Sender<WholeJob>> {
    static TX: OnceLock<Option<std::sync::mpsc::Sender<WholeJob>>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<WholeJob>();
        std::thread::Builder::new()
            .name("xcl-pso-whole".into())
            .stack_size(64 << 20)
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        super::super::pso_async::with_late_build(job);
                    }))
                    .is_err()
                    {
                        crate::xlog_warn!("D3D12.X: whole split-PSO worker caught a panic; continuing queued builds");
                    }
                }
            })
            .ok()
            .map(|_| tx)
    })
    .as_ref()
}

/// Clears `whole_sent` if a queued whole-form job panics or is dropped before completing normally.
struct WholeSentGuard {
    job: ChainRef,
    finished: bool,
}

impl WholeSentGuard {
    fn finish(mut self) {
        self.finished = true;
    }
}

impl Drop for WholeSentGuard {
    fn drop(&mut self) {
        if !self.finished && self.job.0.whole.get().is_none() {
            self.job.0.whole_sent.store(false, Ordering::Release);
        }
    }
}

fn enqueue_whole(job: WholeJob) -> bool {
    whole_worker().is_some_and(|tx| tx.send(job).is_ok())
}

static CHAINS: LazyLock<Mutex<HashMap<usize, Arc<Chain>>>> = LazyLock::new(Default::default);

pub(super) unsafe fn build(this: *mut c_void, create: PsoFn, desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC,
    iid: *const GUID, result: *mut *mut c_void, record: u64, console: &[u8], wave32: bool, windows: u32, samplers: &[pso::StaticSampler],
    plan: Arc<DispatchChain>) -> Option<usize> {
    if result.is_null() || iid.is_null() || plan.pieces.len() < 2 { return None; }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    let signature = pso::dispatch_chain_root_signature(dev, samplers).ok()?;
    let rs = signature.as_raw() as usize;
    let builder = Arc::new(PipelineBuilder { device: dev.clone(), create, signature,
        flags: desc.Flags, node: desc.NodeMask, record });
    let jobs = (0..plan.pieces.len()).map(|i| {
        let plan = plan.clone();
        let builder = builder.clone();
        let iid = if i == 0 { *iid } else { pc_iid(Kind::PipelineState) };
        Box::new(move || builder.pipeline(&plan.pieces[i], &iid).map_or(0, |p| p.into_raw() as usize)) as super::super::pso_async::Job
    }).collect();
    // Take ownership of every success, including when a different piece failed.
    let built: Vec<_> = super::super::pso_async::build_pieces(jobs).into_iter()
        .map(|p| (p != 0).then(|| ID3D12PipelineState::from_raw(p as *mut c_void))).collect();
    let mut pieces = built.into_iter().collect::<Option<Vec<_>>>()?.into_iter();
    let first = pieces.next()?;
    let chain = Arc::new(Chain { builder, tail: pieces.collect(), words: plan.dwords_per_group,
        whole_source: WholeSource { console: console.to_vec(), wave32, windows }, whole: OnceLock::new(), build_whole: Mutex::new(()), whole_sent: Default::default() });
    PROGRAMS.fetch_add(1, Ordering::Relaxed);
    PIECES.fetch_add(plan.pieces.len() as u64, Ordering::Relaxed);
    *result = first.into_raw();
    CHAINS.lock().unwrap().insert(*result as usize, chain);
    Some(rs)
}

pub(super) fn forget(pso: usize) {
    let gone = CHAINS.lock().unwrap().remove(&pso);
    drop(gone); // releasing a tail PSO also goes through forget_pso
}

pub(in crate::hle::d3d12x) unsafe fn whole(pso: usize) -> Result<Option<ID3D12PipelineState>, String> {
    let chain = CHAINS.lock().unwrap().get(&pso).cloned();
    chain.map(|c| c.whole(true)).transpose()
}

/// The whole form of a split program for an indirect dispatch, which runs no chain (its group
/// counts are the GPU's).
pub(in crate::hle::d3d12x) enum Whole {
    /// no split program: the pipeline as it is
    Plain,
    Ready(ID3D12PipelineState),
    /// still being built on the `xcl-pso-whole` worker: this dispatch goes without it
    Late,
}

/// `whole` where a compute pipeline's dispatches go on without it while it is built
/// (`pso_async::cs_goes_on`): the whole form, the largest program the driver compiles, is queued
/// onto the single `xcl-pso-whole` worker, once, and until it is there its dispatches are left out
/// as those of any late compute pipeline. Built in the call, a retail title's render thread stood
/// in the driver's compiler for it (1.1 MB of DXIL) and every other thread that dispatched it
/// stood behind `build_whole`: no frame for over two minutes after its first world frame.
pub(in crate::hle::d3d12x) unsafe fn whole_ready(pso: usize) -> Result<Whole, String> {
    let chain = CHAINS.lock().unwrap().get(&pso).cloned();
    let Some(chain) = chain else { return Ok(Whole::Plain) };
    if let Some(p) = chain.whole.get() {
        return Ok(Whole::Ready(p.clone()));
    }
    if !super::super::pso_async::cs_goes_on() {
        return chain.whole(true).map(Whole::Ready);
    }
    if !chain.whole_sent.swap(true, Ordering::AcqRel) {
        let guard = WholeSentGuard { job: ChainRef(chain.clone()), finished: false };
        let queued = enqueue_whole(Box::new(move || {
            if let Err(e) = unsafe { guard.job.0.whole(false) } {
                crate::xlog_warn!("D3D12.X: the whole form of a split compute program was not built ({e}): its indirect dispatches stay left out");
            }
            guard.finish();
        }));
        if !queued {
            chain.whole_sent.store(false, Ordering::Release);
            return chain.whole(true).map(Whole::Ready);
        }
    }
    Ok(Whole::Late)
}

struct Scratch {
    buffer: ID3D12Resource,
    bytes: u64,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        LIVE.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

struct Recording {
    scratch: Arc<Scratch>,
    first: ID3D12PipelineState,
    chain: Arc<Chain>,
}

static RECORDINGS: LazyLock<Mutex<HashMap<usize, Vec<Arc<Recording>>>>> = LazyLock::new(Default::default);
/// list → the allocator it records on and the scratch its chains share: they are dispatched one
/// after another, with a barrier between them (the recordings keep it alive)
static SHARED: LazyLock<Mutex<HashMap<usize, (usize, std::sync::Weak<Scratch>)>>> = LazyLock::new(Default::default);

/// `list` begins another recording, which the GPU may run beside the last one.
pub(in crate::hle::d3d12x) fn list_reset(list: usize) {
    SHARED.lock().unwrap().remove(&list);
}

/// `to` records right after `from` on the same allocator: its chains go on in `from`'s scratch.
pub(in crate::hle::d3d12x) fn list_follows(from: usize, to: usize) {
    let mut shared = SHARED.lock().unwrap();
    if let Some(e) = shared.get(&from).cloned() {
        shared.insert(to, e);
    }
}

/// The scratch of all recordings alive together, `XCL_SPLIT_SCRATCH_TOTAL_MB` (384). A recording
/// keeps its scratch until its allocator is reset, and a title that records ahead of the GPU has
/// many: with a bound on one dispatch alone a retail title's world took 1.2 GB and went on growing.
fn scratch_total() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_SPLIT_SCRATCH_TOTAL_MB").ok().and_then(|v| v.trim().parse().ok()).filter(|&m| m > 0).unwrap_or(384u64) << 20)
}

/// A recording gets at least this much (or one group's worth) however much the others hold.
const LEAST: u64 = 2 << 20;

/// How much scratch a dispatch of `needed` bytes (`group` for one group) runs with at once: the
/// list's own buffer (`had` bytes) when that is enough or no more may be made, else a new one of
/// the bytes given (second: make one).
fn scratch_bytes(needed: u64, group: u64, budget: u64, had: Option<u64>, live: u64, total: u64) -> (u64, bool) {
    let want = needed.min(budget).max(group);
    let room = total.saturating_sub(live);
    match had {
        Some(h) if h >= want || (room < want && h >= group) => (h, false),
        _ => (want.min(room.max(LEAST)).max(group), true),
    }
}

/// A scratch for a chain of `needed` bytes recorded on `list` with `allocator`, and whether a
/// chain before it on the list used it.
unsafe fn scratch_for(dev: &ID3D12Device, list: usize, allocator: usize, needed: u64, group: u64) -> Option<(Arc<Scratch>, bool)> {
    let had = SHARED.lock().unwrap().get(&list).filter(|s| s.0 == allocator).and_then(|s| s.1.upgrade());
    let (bytes, make) = scratch_bytes(needed, group, scratch_budget(), had.as_ref().map(|s| s.bytes), LIVE.load(Ordering::Relaxed), scratch_total());
    if !make {
        return had.map(|s| (s, true));
    }
    let s = Arc::new(Scratch { buffer: scratch_buffer(dev, bytes)?, bytes });
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
    SHARED.lock().unwrap().insert(list, (allocator, Arc::downgrade(&s)));
    Some((s, false))
}

pub(in crate::hle::d3d12x) fn allocator_reset(allocator: usize) {
    let recordings = RECORDINGS.lock().unwrap().remove(&allocator);
    drop(recordings);
}

unsafe fn scratch_buffer(dev: &ID3D12Device, bytes: u64) -> Option<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, ..Default::default()
    };
    let mut resource = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, CreationNodeMask: 1, VisibleNodeMask: 1, ..Default::default() },
        D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut resource).ok()?;
    resource
}

/// A piece or the whole form on `list`: on a list XCL adopted (a title's) past the hook of a
/// title's SetPipelineState, as `bind::bind_abi` binds a compute pipeline. Through the hook it
/// became the list's bound pipeline (the first one of a list split after it) and got a DRED
/// marker without a label.
unsafe fn bind(list: &ID3D12GraphicsCommandList, pso: &ID3D12PipelineState) {
    if is_adopted(list.as_raw()) {
        super::super::list_x::set_compute_pipeline(list.as_raw(), pso.as_raw());
    } else {
        list.SetPipelineState(pso);
    }
}

/// Called with the original six ABI views bound. The extended signature is
/// common to every piece and the whole fallback. Choose before recording any
/// part of the dispatch, including when scratch allocation fails.
pub(in crate::hle::d3d12x) unsafe fn dispatch(list: &ID3D12GraphicsCommandList, pso: usize, allocator: usize, grid: [u32; 3]) -> Result<(), String> {
    let [x, y, z] = grid;
    let chain = CHAINS.lock().unwrap().get(&pso).cloned();
    let Some(chain) = chain else { list.Dispatch(x, y, z); return Ok(()) };
    if grid.contains(&0) { return Ok(()); }
    // A title Reset can switch to a different PC allocator while this
    // recording is still submitted on the old one. Retain by backing.
    let allocator = if super::super::alloc_pool::enabled() {
        super::super::alloc_pool::recording_backing(list.as_raw() as usize).unwrap_or(0)
    } else { allocator };
    let group = u64::from(chain.words) * 4;
    let needed = grid.into_iter().fold(group, |s, n| s.saturating_mul(n as u64));
    WANTED.fetch_max(needed, Ordering::Relaxed);
    let scratch = (allocator != 0 && list.GetType() != D3D12_COMMAND_LIST_TYPE_BUNDLE)
        .then(|| scratch_for(&chain.builder.device, list.as_raw() as usize, allocator, needed, group)).flatten();
    let most = scratch.as_ref().map_or(1, |(s, _)| (s.bytes / group).max(1));
    let slabs = pso::dispatch_slabs(grid, most);
    let Some((scratch, used)) = scratch else {
        bind(list, &chain.whole(true)?);
        list.Dispatch(x, y, z);
        return Ok(());
    };
    let first = ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)).ok_or("first compute piece")?.clone();
    let held = Arc::new(Recording { scratch, first, chain });
    RECORDINGS.lock().unwrap().entry(allocator).or_default().push(held.clone());
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    SLABS.fetch_add(slabs.len() as u64, Ordering::Relaxed);
    pso::record_dispatch_chain(list, grid, most, &held.scratch.buffer, used, std::iter::once(&held.first).chain(&held.chain.tail), |p| bind(list, p));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{scratch_bytes, LEAST};

    #[test]
    fn the_whole_form_is_made_again_from_the_console_program() {
        use crate::gpu::recomp::pso::{console_to_dxil_in, mem_windows, Stage};
        use crate::gpu::xbox::tests::{fssm, sample, CODE};
        // (a compute program of one thread group of 1x1x1, as `pso::compile_time_bench` makes it)
        let mut f = fssm(&CODE, &[(6, 0)]);
        for offset in [0x13C, 0x140, 0x144] {
            f[offset..offset + 4].copy_from_slice(&1u32.to_le_bytes());
        }
        let blob = sample(5, &[f]);
        let windows = mem_windows();
        let first = match console_to_dxil_in(&blob, Stage::Compute, 0, false, windows) {
            Ok(c) => c.dxil,
            // (no DXC here)
            Err(e) if e.contains("dxcompiler") || e.contains("DXC") => {
                crate::gpu::recomp::runner::skip_test("no DXC");
                return;
            }
            Err(e) => panic!("the console program does not translate: {e}"),
        };
        let source = super::WholeSource { console: blob.clone(), wave32: false, windows };
        assert_eq!(source.code().unwrap(), first, "the DXIL the chain was split from");
        assert!(source.console.len() < first.len(), "a chain keeps less than the DXIL: {} against {}", source.console.len(), first.len());
    }

    #[test]
    fn the_whole_worker_preserves_each_programs_memory_windows() {
        use crate::gpu::{rdna2::encode, recomp::{dxc, pso, HEAP_WINDOWS}, xbox::tests::{fssm, sample}};
        use std::sync::mpsc;
        if dxc::location().is_err() {
            crate::gpu::recomp::runner::skip_test("no DXC");
            return;
        }
        // A memory write keeps the window access in both translated programs. This test
        // compiles only; it never dispatches the synthetic program to a GPU.
        let mut code = encode("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 7), ("DATA", 4), ("SADDR", 0x7D)], None).unwrap();
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let mut f = fssm(&code, &[(6, 0)]);
        for offset in [0x13C, 0x140, 0x144] {
            f[offset..offset + 4].copy_from_slice(&1u32.to_le_bytes());
        }
        let blob = sample(5, &[f]);
        let mut forms = Vec::new();
        for windows in [4, 4 | HEAP_WINDOWS] {
            let expected = pso::console_to_dxil_in(&blob, pso::Stage::Compute, 0, false, windows).unwrap().dxil;
            let source = super::WholeSource { console: blob.clone(), wave32: false, windows };
            let (tx, rx) = mpsc::channel();
            assert!(super::enqueue_whole(Box::new(move || { let _ = tx.send(source.code()); })));
            let actual = rx.recv_timeout(std::time::Duration::from_secs(30)).expect("whole worker completed").unwrap();
            assert_eq!(actual, expected, "the worker must keep the original window form {windows:#x}");
            forms.push(actual);
        }
        assert_ne!(forms[0], forms[1], "the fixture must distinguish root-UAV and descriptor-heap access");
    }

    const MB: u64 = 1 << 20;

    #[test]
    fn a_list_recorded_after_another_goes_on_in_its_scratch() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device};
        use windows::Win32::Graphics::Dxgi::*;
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev) }.is_err() {
            let warp = unsafe { CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>()) };
            if warp.and_then(|w| unsafe { D3D12CreateDevice(&w, D3D_FEATURE_LEVEL_11_0, &mut dev) }).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
        }
        let dev = dev.unwrap();
        let (a, b, c, alloc) = (0x7E57_5C01usize, 0x7E57_5C02usize, 0x7E57_5C03usize, 0x7E57_5CA0usize);
        let (first, used) = unsafe { super::scratch_for(&dev, a, alloc, MB, 4096) }.unwrap();
        assert!(!used);
        super::list_follows(a, b);
        let (second, used) = unsafe { super::scratch_for(&dev, b, alloc, MB, 4096) }.unwrap();
        assert!(used && std::sync::Arc::ptr_eq(&first, &second), "the list after it made a scratch of its own");
        let (third, _) = unsafe { super::scratch_for(&dev, c, alloc, MB, 4096) }.unwrap();
        assert!(!std::sync::Arc::ptr_eq(&first, &third));
        for list in [a, b, c] {
            super::list_reset(list);
        }
    }

    #[test]
    fn a_chains_scratch_is_bounded_by_what_all_recordings_hold() {
        let group = 187 << 10;
        // the first of a recording: as much as it needs, up to the budget of one dispatch
        assert_eq!(scratch_bytes(15 * MB, group, 64 * MB, None, 0, 384 * MB), (15 * MB, true));
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 0, 384 * MB), (64 * MB, true));
        // the others hold nearly all: what is left, and at least LEAST
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 364 * MB, 384 * MB), (20 * MB, true));
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 1200 * MB, 384 * MB), (LEAST, true));
        // the list's own buffer: kept when big enough, or when no more may be made
        assert_eq!(scratch_bytes(8 * MB, group, 64 * MB, Some(16 * MB), 100 * MB, 384 * MB), (16 * MB, false));
        assert_eq!(scratch_bytes(60 * MB, group, 64 * MB, Some(16 * MB), 380 * MB, 384 * MB), (16 * MB, false));
        // … but made again, larger, while there is room
        assert_eq!(scratch_bytes(60 * MB, group, 64 * MB, Some(16 * MB), 100 * MB, 384 * MB), (60 * MB, true));
        // one group always fits, whatever the bounds say
        assert_eq!(scratch_bytes(9 * MB, 9 * MB, 4 * MB, None, 400 * MB, 384 * MB), (9 * MB, true));
        assert_eq!(scratch_bytes(9 * MB, 9 * MB, 4 * MB, Some(MB), 400 * MB, 384 * MB), (9 * MB, true));
    }

    #[test]
    fn whole_worker_runs_all_jobs_on_one_late_thread() {
        use std::collections::HashSet;
        use std::sync::{mpsc, Arc, Mutex};
        use std::thread::ThreadId;

        assert!(!super::super::super::pso_async::build_is_late());
        let seen = Arc::new(Mutex::new(Vec::<(ThreadId, Option<String>, bool)>::new()));
        let (done_tx, done_rx) = mpsc::channel();
        const N: usize = 16;
        for _ in 0..N {
            let seen = seen.clone();
            let done_tx = done_tx.clone();
            assert!(super::enqueue_whole(Box::new(move || {
                let id = std::thread::current().id();
                let name = std::thread::current().name().map(str::to_string);
                let late = super::super::super::pso_async::build_is_late();
                seen.lock().unwrap().push((id, name, late));
                let _ = done_tx.send(());
            })));
        }
        for _ in 0..N {
            done_rx.recv_timeout(std::time::Duration::from_secs(5)).expect("whole worker job finished");
        }
        let rows = seen.lock().unwrap();
        assert_eq!(rows.len(), N);
        let threads: HashSet<ThreadId> = rows.iter().map(|(id, _, _)| *id).collect();
        assert_eq!(threads.len(), 1, "expected all whole-PSO jobs to run on a single worker thread");
        assert!(rows.iter().all(|(_, name, late)| name.as_deref() == Some("xcl-pso-whole") && *late));
    }

    #[test]
    fn with_late_build_resets_after_return_and_panic() {
        use super::super::super::pso_async::{build_is_late, with_late_build};

        assert!(!build_is_late());
        let inside = with_late_build(build_is_late);
        assert!(inside);
        assert!(!build_is_late());

        let panicked = std::panic::catch_unwind(|| {
            with_late_build(|| {
                assert!(build_is_late());
                panic!("unwind check");
            });
        });
        assert!(panicked.is_err());
        assert!(!build_is_late());
    }

    #[test]
    fn whole_worker_survives_a_panicking_job_and_runs_queued_work() {
        use std::sync::{mpsc, Arc};
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::Duration;

        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (done2_tx, done2_rx) = mpsc::channel();
        let (done3_tx, done3_rx) = mpsc::channel();

        // Simulate a pending flag cleared by a Drop guard if the first job panics.
        struct FlagGuard(Arc<AtomicBool>, bool);
        impl Drop for FlagGuard {
            fn drop(&mut self) {
                if !self.1 {
                    self.0.store(false, Ordering::Release);
                }
            }
        }
        let sent1 = Arc::new(AtomicBool::new(true));
        let guard1 = FlagGuard(sent1.clone(), false);

        assert!(super::enqueue_whole(Box::new(move || {
            let _g = guard1;
            let _ = entered_tx.send(std::thread::current().id());
            let _ = release_rx.recv();
            panic!("injected whole-PSO worker panic");
        })));
        let worker_tid = entered_rx.recv_timeout(Duration::from_secs(5)).expect("job 1 entered worker");

        // Queue job 2 while job 1 is still active on the worker, then let job 1 panic.
        assert!(super::enqueue_whole(Box::new(move || {
            let late = super::super::super::pso_async::build_is_late();
            let _ = done2_tx.send((std::thread::current().id(), late));
        })));
        let _ = release_tx.send(());

        let (tid2, late2) = done2_rx.recv_timeout(Duration::from_secs(5)).expect("job 2 ran after job 1 panicked");
        assert_eq!(tid2, worker_tid, "job 2 should run on the same surviving worker");
        assert!(late2, "late-build flag should still be active for job 2");
        assert!(!sent1.load(Ordering::Acquire), "panicking job's pending flag should be cleared on unwind");

        // A third job submitted after the panic is also accepted and executed.
        assert!(super::enqueue_whole(Box::new(move || {
            let _ = done3_tx.send(std::thread::current().id());
        })));
        let tid3 = done3_rx.recv_timeout(Duration::from_secs(5)).expect("job 3 ran after panic");
        assert_eq!(tid3, worker_tid);
    }
}
