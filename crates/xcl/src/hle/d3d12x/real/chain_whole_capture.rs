//! Opt-in first-demand capture sidecar for `dispatch_chain::Chain::whole` fallbacks (#2741).
//!
//! When `XCL_CAPTURE_CHAIN_WHOLE=<path>` is unset or whitespace-empty, this module performs
//! zero I/O, spawns no thread, queries no DXGI metadata, and acquires no locks.
//!
//! When enabled, the first demand for each distinct final chain-ABI whole PSO identity is
//! assigned a 1-based monotonic sequence number and serialized as a UTF-8 JSONL sidecar line
//! on a dedicated background writer thread (`xcl-chain-capture`), strictly outside
//! `Chain::build_whole` and storing replay metadata only (no shader bytes or local paths).

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;
use windows::Win32::Graphics::Dxgi::*;

use super::driver_cache;
use crate::gpu::recomp::{dxc, pso};

/// Environment variable enabling first-demand capture to a versioned JSONL sidecar path.
pub(super) const ENV_PATH: &str = "XCL_CAPTURE_CHAIN_WHOLE";
/// Optional environment variable capping the number of captured distinct demands (`0` = header only).
pub(super) const ENV_MAX: &str = "XCL_CAPTURE_CHAIN_WHOLE_MAX";

/// JSONL sidecar schema version.
pub(super) const SCHEMA_VERSION: u32 = 1;
/// Dispatch-chain root signature ABI version matching `pso::dispatch_chain_root_signature`:
/// 6 root UAVs (`u0..u5 space0`), chain scratch UAV (`u0 space2`), 8 root constants (`b0 space2`),
/// directly-indexed heaps, and ordered static samplers in `space1`.
pub(super) const CHAIN_RS_ABI_VERSION: u32 = 1;

const FLUSH_TIMEOUT: Duration = Duration::from_millis(250);

/// Why a split compute shader demanded its chain-ABI whole form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::hle::d3d12x) enum ChainWholeReason {
    Indirect,
    Bundle,
    MissingRecordingAllocator,
    ScratchUnavailable,
}

/// Deduplication identity for a chain-ABI whole PSO demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct ChainWholeIdentity {
    pub source_record_hash: u64,
    pub whole_dxil_hash: u64,
    pub whole_dxil_len: usize,
    pub effective_flags: i32,
    pub node_mask: u32,
    pub chain_rs_abi_version: u32,
    pub chain_rs_id: u64,
}

/// Ordered static sampler entry serialized in replay metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct StaticSamplerMeta {
    pub words: [u32; 4],
    pub comparison: bool,
}

/// Precomputed per-chain metadata stored on `dispatch_chain::Chain`.
#[derive(Debug, Clone)]
pub(super) struct ChainWholeMeta {
    pub identity: ChainWholeIdentity,
    pub static_samplers: Vec<StaticSamplerMeta>,
    pub disk_cache_name: String,
    pub mem_windows: u32,
    pub split_wave64: bool,
}

impl ChainWholeMeta {
    pub(super) fn new(
        source_record_hash: u64,
        whole_dxil: &[u8],
        samplers: &[pso::StaticSampler],
        effective_flags: i32,
        node_mask: u32,
        mem_windows: u32,
        split_wave64: bool,
    ) -> Self {
        let whole_dxil_hash = crate::gpu::shader::fnv64(whole_dxil);
        let chain_rs_id = chain_root_sig_id(samplers);
        Self {
            identity: ChainWholeIdentity {
                source_record_hash,
                whole_dxil_hash,
                whole_dxil_len: whole_dxil.len(),
                effective_flags,
                node_mask,
                chain_rs_abi_version: CHAIN_RS_ABI_VERSION,
                chain_rs_id,
            },
            static_samplers: samplers
                .iter()
                .map(|&(words, comparison)| StaticSamplerMeta { words, comparison })
                .collect(),
            disk_cache_name: driver_cache::name_for(whole_dxil, effective_flags, true),
            mem_windows,
            split_wave64,
        }
    }
}

/// Stable 64-bit identity for the dispatch-chain root signature derived from its ABI version and
/// ordered static samplers (`pso::StaticSampler`).
pub(super) fn chain_root_sig_id(samplers: &[pso::StaticSampler]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut add = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    add(b"xcl-chain-rs-v1\0");
    add(&CHAIN_RS_ABI_VERSION.to_le_bytes());
    add(&(samplers.len() as u64).to_le_bytes());
    for &(words, comparison) in samplers {
        for w in words {
            add(&w.to_le_bytes());
        }
        add(&[u8::from(comparison)]);
    }
    h
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CaptureConfig {
    pub path: PathBuf,
    pub sequence_cutoff: Option<u64>,
}

/// Parse `(path, max)` env values into an opt-in capture configuration.
/// Empty or whitespace-only `path` disables capture (`None`).
fn parse_config_values(path_raw: Option<&str>, max_raw: Option<&str>) -> (Option<CaptureConfig>, Option<String>) {
    let Some(path_str) = path_raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return (None, None);
    };
    let (sequence_cutoff, bad_max) = match max_raw.map(str::trim).filter(|s| !s.is_empty()) {
        None => (None, None),
        Some(raw) => match raw.parse::<u64>() {
            Ok(v) => (Some(v), None),
            Err(_) => (None, Some(raw.to_string())),
        },
    };
    (Some(CaptureConfig { path: PathBuf::from(path_str), sequence_cutoff }), bad_max)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct AdapterMeta {
    pub description: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub subsys_id: u32,
    pub revision: u32,
    pub is_software: bool,
    pub driver_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct AgilityMeta {
    pub status: String,
    pub version: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct DxcMeta {
    pub dxcompiler: dxc::Library,
    pub dxil: dxc::Library,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct TranslationOptionsMeta {
    pub direct_dxil: bool,
    pub direct_compact: Option<bool>,
    pub direct_compact_nest: Option<String>,
    pub per_lane: bool,
    pub safe_barrier: bool,
    pub heap_windows: bool,
    pub split_wave64_forced: bool,
    pub dispatch_piece_limit: usize,
    #[serde(default)]
    pub dxil_no_opt: Vec<String>,
    #[serde(default)]
    pub x64_all_masks: bool,
    #[serde(default = "default_true")]
    pub wave64_spin: bool,
    #[serde(default = "default_true")]
    pub recomp_structured: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct HeaderRecord {
    pub schema_version: u32,
    pub kind: String,
    pub xcl_version: String,
    pub translator_source_id: String,
    pub direct_dxil_source_id: String,
    pub direct_dxil_translator_version: u32,
    pub per_lane_source_id: String,
    pub chain_rs_abi_version: u32,
    pub sequence_cutoff: Option<u64>,
    pub captured_at_unix_ms: u64,
    pub adapter: Option<AdapterMeta>,
    pub agility_runtime: AgilityMeta,
    pub dxc: DxcMeta,
    pub translation_options: TranslationOptionsMeta,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct DemandRecord {
    pub schema_version: u32,
    pub kind: String,
    pub seq: u64,
    pub elapsed_ms: u64,
    pub timestamp_unix_ms: u64,
    pub present_count: u64,
    pub reason: ChainWholeReason,
    pub already_ready: bool,
    pub source_record_hash: String,
    pub whole_dxil_hash: String,
    pub whole_dxil_len: usize,
    pub effective_flags: i32,
    pub node_mask: u32,
    pub chain_rs_abi_version: u32,
    pub chain_rs_id: String,
    pub static_samplers: Vec<StaticSamplerMeta>,
    pub disk_cache_name: String,
    pub mem_windows: u32,
    pub split_wave64: bool,
}

fn format_umd_version(raw: i64) -> String {
    let v = raw as u64;
    format!(
        "{}.{}.{}.{}",
        (v >> 48) & 0xffff,
        (v >> 32) & 0xffff,
        (v >> 16) & 0xffff,
        v & 0xffff,
    )
}

fn unix_ms_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn probe_adapter_meta(device: Option<&ID3D12Device>) -> Option<AdapterMeta> {
    let dev = device?;
    unsafe {
        let luid = dev.GetAdapterLuid();
        let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(luid).ok()?;
        let desc = adapter.GetDesc1().ok()?;
        let driver_version = adapter
            .CheckInterfaceSupport(&IDXGIDevice::IID)
            .ok()
            .map(format_umd_version);
        Some(AdapterMeta {
            description: crate::gpu::adapter::name(&desc),
            vendor_id: desc.VendorId,
            device_id: desc.DeviceId,
            subsys_id: desc.SubSysId,
            revision: desc.Revision,
            is_software: (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0,
            driver_version,
        })
    }
}

fn current_agility_meta() -> AgilityMeta {
    match crate::gpu::agility::state() {
        crate::gpu::agility::State::Loaded { version, .. } => AgilityMeta {
            status: "loaded".into(),
            version: Some(*version),
        },
        crate::gpu::agility::State::NotFound => AgilityMeta {
            status: "not_found".into(),
            version: None,
        },
        crate::gpu::agility::State::Unavailable(_) => AgilityMeta {
            status: "unavailable".into(),
            version: None,
        },
    }
}

fn current_dxc_meta() -> DxcMeta {
    let tc = dxc::toolchain();
    DxcMeta {
        dxcompiler: tc.dxcompiler,
        dxil: tc.dxil,
    }
}

/// Canonical sorted list of direct-DXIL optimization passes disabled by `XCL_DXIL_NO_OPT`
/// (matching `gpu::dxil::module::opt::optimize`: `phi`, `cse`, `dce`, `merge`, or `all`).
pub(super) fn effective_dxil_no_opt(raw: Option<&str>) -> Vec<String> {
    const ALL_PASSES: [&str; 4] = ["cse", "dce", "merge", "phi"];
    let Some(val) = raw else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for pass in ALL_PASSES {
        if val.split(',').any(|x| {
            let t = x.trim();
            t == pass || t == "all"
        }) {
            out.push(pass.to_string());
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn translation_options_from(
    direct_dxil: bool,
    direct_compact: Option<bool>,
    direct_compact_nest: Option<String>,
    per_lane: bool,
    safe_barrier_raw: Option<&str>,
    heap_windows: bool,
    split_wave64_raw: Option<&str>,
    dispatch_piece_limit: usize,
    dxil_no_opt_raw: Option<&str>,
    x64_all_masks: bool,
    wave64_spin_raw: Option<&str>,
    recomp_no_structured: bool,
) -> TranslationOptionsMeta {
    TranslationOptionsMeta {
        direct_dxil,
        direct_compact,
        direct_compact_nest,
        per_lane,
        safe_barrier: safe_barrier_raw.is_none_or(|v| v.trim() != "0"),
        heap_windows,
        split_wave64_forced: split_wave64_raw == Some("1"),
        dispatch_piece_limit,
        dxil_no_opt: effective_dxil_no_opt(dxil_no_opt_raw),
        x64_all_masks,
        wave64_spin: wave64_spin_raw.is_none_or(|v| v.trim() != "0"),
        recomp_structured: !recomp_no_structured,
    }
}

fn current_translation_options() -> TranslationOptionsMeta {
    let safe_barrier = std::env::var("XCL_SAFE_BARRIER").ok();
    let split_wave64 = std::env::var("XCL_SPLIT_WAVE64").ok();
    let dxil_no_opt = std::env::var("XCL_DXIL_NO_OPT").ok();
    let wave64_spin = std::env::var("XCL_WAVE64_SPIN").ok();
    translation_options_from(
        crate::gpu::dxil::kernel::enabled(),
        crate::gpu::dxil::kernel::compact_mode(),
        crate::gpu::dxil::kernel::compact_nest_tag(),
        crate::gpu::recomp::lower_dxil::enabled(),
        safe_barrier.as_deref(),
        pso::heap_windows(),
        split_wave64.as_deref(),
        crate::gpu::dxil::kernel::dispatch_piece_limit(),
        dxil_no_opt.as_deref(),
        std::env::var_os("XCL_X64_ALL_MASKS").is_some(),
        wave64_spin.as_deref(),
        std::env::var_os("XCL_RECOMP_NO_STRUCTURED").is_some(),
    )
}

fn build_header(device: Option<&ID3D12Device>, sequence_cutoff: Option<u64>) -> HeaderRecord {
    HeaderRecord {
        schema_version: SCHEMA_VERSION,
        kind: "header".into(),
        xcl_version: env!("CARGO_PKG_VERSION").into(),
        translator_source_id: env!("XCL_TRANSLATOR_SOURCE_ID").into(),
        direct_dxil_source_id: env!("XCL_DIRECT_DXIL_SOURCE_ID").into(),
        direct_dxil_translator_version: crate::gpu::dxil::kernel::TRANSLATOR_VERSION,
        per_lane_source_id: env!("XCL_PER_LANE_SOURCE_ID").into(),
        chain_rs_abi_version: CHAIN_RS_ABI_VERSION,
        sequence_cutoff,
        captured_at_unix_ms: unix_ms_now(),
        adapter: probe_adapter_meta(device),
        agility_runtime: current_agility_meta(),
        dxc: current_dxc_meta(),
        translation_options: current_translation_options(),
    }
}

enum WriterMsg {
    Line(String),
    Flush(mpsc::Sender<bool>),
}

struct CaptureState {
    next_seq: u64,
    seen: HashSet<ChainWholeIdentity>,
}

/// Active first-demand capture session owning a dedicated background JSONL writer thread.
pub(super) struct CaptureSession {
    cutoff: Option<u64>,
    started: Instant,
    failed: Arc<AtomicBool>,
    state: Mutex<CaptureState>,
    tx: mpsc::Sender<WriterMsg>,
}

impl CaptureSession {
    pub(super) fn new(config: CaptureConfig, header: HeaderRecord) -> Self {
        let failed = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<WriterMsg>();
        let path = config.path;
        let failed_bg = failed.clone();
        if let Ok(header_line) = serde_json::to_string(&header) {
            let _ = tx.send(WriterMsg::Line(header_line));
        } else {
            failed.store(true, Ordering::Release);
        }
        if let Err(e) = std::thread::Builder::new()
            .name("xcl-chain-capture".into())
            .spawn(move || writer_loop(path, rx, failed_bg))
        {
            failed.store(true, Ordering::Release);
            crate::xlog_warn!("D3D12.X: chain-whole capture writer thread failed to start: {e}");
        }
        Self {
            cutoff: config.sequence_cutoff,
            started: Instant::now(),
            failed,
            state: Mutex::new(CaptureState { next_seq: 1, seen: HashSet::new() }),
            tx,
        }
    }

    /// Record a demand if it is the first for its final identity and within the sequence cutoff.
    /// Performs zero disk I/O and acquires only the leaf `state` mutex.
    pub(super) fn record(
        &self,
        meta: &ChainWholeMeta,
        reason: ChainWholeReason,
        already_ready: bool,
        present_count: u64,
    ) -> Option<u64> {
        if self.failed.load(Ordering::Acquire) {
            return None;
        }
        let mut g = self.state.lock().unwrap();
        if self.cutoff.is_some_and(|max| g.next_seq > max) {
            return None;
        }
        if !g.seen.insert(meta.identity) {
            return None;
        }
        let seq = g.next_seq;
        g.next_seq += 1;
        let rec = DemandRecord {
            schema_version: SCHEMA_VERSION,
            kind: "demand".into(),
            seq,
            elapsed_ms: self.started.elapsed().as_millis() as u64,
            timestamp_unix_ms: unix_ms_now(),
            present_count,
            reason,
            already_ready,
            source_record_hash: format!("{:016x}", meta.identity.source_record_hash),
            whole_dxil_hash: format!("{:016x}", meta.identity.whole_dxil_hash),
            whole_dxil_len: meta.identity.whole_dxil_len,
            effective_flags: meta.identity.effective_flags,
            node_mask: meta.identity.node_mask,
            chain_rs_abi_version: meta.identity.chain_rs_abi_version,
            chain_rs_id: format!("{:016x}", meta.identity.chain_rs_id),
            static_samplers: meta.static_samplers.clone(),
            disk_cache_name: meta.disk_cache_name.clone(),
            mem_windows: meta.mem_windows,
            split_wave64: meta.split_wave64,
        };
        let Ok(line) = serde_json::to_string(&rec) else {
            self.failed.store(true, Ordering::Release);
            return None;
        };
        // Enqueued under `g` (pure non-blocking in-memory push) so file order strictly matches `seq`.
        if self.tx.send(WriterMsg::Line(line)).is_err() {
            self.failed.store(true, Ordering::Release);
            return None;
        }
        Some(seq)
    }

    /// Wait up to `timeout` for all prior queued lines to flush to disk.
    pub(super) fn flush(&self, timeout: Duration) -> bool {
        let (ack_tx, ack_rx) = mpsc::channel();
        if self.tx.send(WriterMsg::Flush(ack_tx)).is_err() {
            return false;
        }
        ack_rx.recv_timeout(timeout).unwrap_or(false)
    }

    #[cfg(test)]
    pub(super) fn has_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }
}

fn open_writer(path: &Path) -> Result<BufWriter<File>, std::io::Error> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    Ok(BufWriter::new(file))
}

fn fail_writer(failed: &AtomicBool, warned: &mut bool, path: &Path, err: &std::io::Error) {
    failed.store(true, Ordering::Release);
    if !*warned {
        *warned = true;
        crate::xlog_warn!("D3D12.X: chain-whole capture disabled after I/O error on {}: {err}", path.display());
    }
}

fn writer_loop(path: PathBuf, rx: mpsc::Receiver<WriterMsg>, failed: Arc<AtomicBool>) {
    let mut warned = false;
    let mut writer = match open_writer(&path) {
        Ok(w) => Some(w),
        Err(e) => {
            fail_writer(&failed, &mut warned, &path, &e);
            None
        }
    };
    while let Ok(msg) = rx.recv() {
        match msg {
            WriterMsg::Line(line) => {
                let err = writer.as_mut().and_then(|w| {
                    w.write_all(line.as_bytes())
                        .and_then(|()| w.write_all(b"\n"))
                        .and_then(|()| w.flush())
                        .err()
                });
                if let Some(e) = err {
                    fail_writer(&failed, &mut warned, &path, &e);
                    writer = None;
                }
            }
            WriterMsg::Flush(ack) => {
                let ok = match writer.as_mut() {
                    Some(w) => match w.flush() {
                        Ok(()) => true,
                        Err(e) => {
                            fail_writer(&failed, &mut warned, &path, &e);
                            writer = None;
                            false
                        }
                    },
                    None => false,
                };
                let _ = ack.send(ok);
            }
        }
    }
}

fn env_config() -> Option<&'static CaptureConfig> {
    if cfg!(test) {
        return None;
    }
    static CFG: OnceLock<Option<CaptureConfig>> = OnceLock::new();
    CFG.get_or_init(|| {
        let path = std::env::var(ENV_PATH).ok();
        let max = std::env::var(ENV_MAX).ok();
        let (cfg, bad_max) = parse_config_values(path.as_deref(), max.as_deref());
        if let Some(raw) = bad_max {
            crate::xlog_warn!("D3D12.X: ignoring invalid {ENV_MAX}={raw:?}; capturing without a sequence cutoff");
        }
        cfg
    })
    .as_ref()
}

/// Whether opt-in `Chain::whole` capture is active (reads the cached `OnceLock` config without
/// touching the environment, disk, or locks on the disabled fast path).
#[inline]
pub(super) fn enabled() -> bool {
    env_config().is_some()
}

/// Lazily construct `ChainWholeMeta` only when capture is enabled so disabled split-chain builds
/// perform zero extra hashing, cache-key formatting, or sampler allocation.
#[inline]
pub(super) fn build_meta_if(enabled: bool, make: impl FnOnce() -> ChainWholeMeta) -> Option<ChainWholeMeta> {
    enabled.then(make)
}

static GLOBAL_SESSION: OnceLock<CaptureSession> = OnceLock::new();

fn global_session(device: Option<&ID3D12Device>) -> Option<&'static CaptureSession> {
    let cfg = env_config()?;
    Some(GLOBAL_SESSION.get_or_init(|| {
        let header = build_header(device, cfg.sequence_cutoff);
        CaptureSession::new(cfg.clone(), header)
    }))
}

/// Record a `Chain::whole` demand in the process-wide capture session when `XCL_CAPTURE_CHAIN_WHOLE`
/// is enabled. Must be called before `Chain::whole` waits on `build_whole` or creates a PSO.
pub(super) fn note_demand(
    device: &ID3D12Device,
    meta: &ChainWholeMeta,
    reason: ChainWholeReason,
    already_ready: bool,
) {
    let Some(session) = global_session(Some(device)) else {
        return;
    };
    let present_count = super::super::present::activity().0;
    let _ = session.record(meta, reason, already_ready, present_count);
}

/// Flush any pending sidecar records before process shutdown (`real::end_title`).
/// No-op when `XCL_CAPTURE_CHAIN_WHOLE` is unset or empty.
pub(super) fn flush_global() {
    let Some(cfg) = env_config() else {
        return;
    };
    let session = match GLOBAL_SESSION.get() {
        Some(s) => s,
        None => {
            let raw = super::device_raw();
            let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) };
            GLOBAL_SESSION.get_or_init(|| {
                let header = build_header(dev, cfg.sequence_cutoff);
                CaptureSession::new(cfg.clone(), header)
            })
        }
    };
    let _ = session.flush(FLUSH_TIMEOUT);
}

#[cfg(test)]
pub(super) fn build_header_for_test(device: Option<&ID3D12Device>, cutoff: Option<u64>) -> HeaderRecord {
    build_header(device, cutoff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn temp_path(tag: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let id = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "xcl_chain_whole_capture_test_{}_{}_{}.jsonl",
            std::process::id(),
            tag,
            id
        ))
    }

    fn test_header(cutoff: Option<u64>) -> HeaderRecord {
        HeaderRecord {
            schema_version: SCHEMA_VERSION,
            kind: "header".into(),
            xcl_version: "0.0.0-test".into(),
            translator_source_id: "0123456789abcdef".into(),
            direct_dxil_source_id: "fedcba9876543210".into(),
            direct_dxil_translator_version: 9,
            per_lane_source_id: "1111222233334444".into(),
            chain_rs_abi_version: CHAIN_RS_ABI_VERSION,
            sequence_cutoff: cutoff,
            captured_at_unix_ms: 1_700_000_000_000,
            adapter: Some(AdapterMeta {
                description: "Synthetic Adapter".into(),
                vendor_id: 0x10de,
                device_id: 0x2882,
                subsys_id: 0,
                revision: 0xa1,
                is_software: false,
                driver_version: Some(format_umd_version((32i64 << 48) | (16i64 << 16) | 2114)),
            }),
            agility_runtime: AgilityMeta { status: "not_found".into(), version: None },
            dxc: DxcMeta { dxcompiler: dxc::Library::default(), dxil: dxc::Library::default() },
            translation_options: TranslationOptionsMeta {
                direct_dxil: true,
                direct_compact: None,
                direct_compact_nest: None,
                per_lane: false,
                safe_barrier: true,
                heap_windows: false,
                split_wave64_forced: false,
                dispatch_piece_limit: 12_500,
                dxil_no_opt: Vec::new(),
                x64_all_masks: false,
                wave64_spin: true,
                recomp_structured: true,
            },
        }
    }

    fn read_lines(path: &Path) -> Vec<serde_json::Value> {
        let text = std::fs::read_to_string(path).expect("read sidecar");
        text.lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("valid jsonl line"))
            .collect()
    }

    #[test]
    fn off_by_default_creates_no_file_and_no_io() {
        assert_eq!(parse_config_values(None, None), (None, None));
        assert_eq!(parse_config_values(Some(""), Some("5")), (None, None));
        assert_eq!(parse_config_values(Some("   \t\n  "), Some("10")), (None, None));
        assert!(env_config().is_none(), "global capture must stay disabled during tests");
        let (cfg, bad) = parse_config_values(Some("  out.jsonl  "), Some("42"));
        assert_eq!(bad, None);
        assert_eq!(
            cfg,
            Some(CaptureConfig { path: PathBuf::from("out.jsonl"), sequence_cutoff: Some(42) })
        );
        let (cfg_bad, bad_raw) = parse_config_values(Some("out.jsonl"), Some("not-a-num"));
        assert_eq!(bad_raw.as_deref(), Some("not-a-num"));
        assert_eq!(
            cfg_bad,
            Some(CaptureConfig { path: PathBuf::from("out.jsonl"), sequence_cutoff: None })
        );
    }

    #[test]
    fn captures_all_four_reasons_in_monotonic_sequence_order() {
        let path = temp_path("all_four_reasons");
        let _ = std::fs::remove_file(&path);
        let session = CaptureSession::new(
            CaptureConfig { path: path.clone(), sequence_cutoff: None },
            test_header(None),
        );
        let reasons = [
            ChainWholeReason::Indirect,
            ChainWholeReason::Bundle,
            ChainWholeReason::MissingRecordingAllocator,
            ChainWholeReason::ScratchUnavailable,
        ];
        for (i, &reason) in reasons.iter().enumerate() {
            let dxil = vec![0xA0u8.wrapping_add(i as u8); 64 + i * 8];
            let meta = ChainWholeMeta::new(0x1000 + i as u64, &dxil, &[], 0, 0, 4, true);
            assert_eq!(session.record(&meta, reason, false, 10 + i as u64), Some(i as u64 + 1));
        }
        assert!(session.flush(Duration::from_secs(2)));
        drop(session);

        let rows = read_lines(&path);
        assert_eq!(rows.len(), 5, "expected 1 header + 4 demand lines");
        assert_eq!(rows[0]["kind"], "header");
        assert_eq!(rows[0]["schema_version"], SCHEMA_VERSION);
        assert_eq!(rows[0]["adapter"]["driver_version"], "32.0.16.2114");
        let expected_names = ["indirect", "bundle", "missing_recording_allocator", "scratch_unavailable"];
        for (idx, expected) in expected_names.iter().enumerate() {
            let rec: DemandRecord = serde_json::from_value(rows[idx + 1].clone()).expect("demand record");
            assert_eq!(rec.kind, "demand");
            assert_eq!(rec.seq, idx as u64 + 1);
            assert_eq!(rows[idx + 1]["reason"], *expected);
            assert_eq!(rec.present_count, 10 + idx as u64);
            assert!(!rec.already_ready);
            assert_eq!(rec.source_record_hash, format!("{:016x}", 0x1000 + idx as u64));
            assert_eq!(rec.whole_dxil_len, 64 + idx * 8);
            assert!(rec.disk_cache_name.ends_with(".chain.pso"));
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn deduplicates_first_demand_by_final_identity_and_keeps_distinct_identities_separate() {
        let path = temp_path("dedup_identity");
        let _ = std::fs::remove_file(&path);
        let session = CaptureSession::new(
            CaptureConfig { path: path.clone(), sequence_cutoff: None },
            test_header(None),
        );
        let s1: pso::StaticSampler = ([1, 2, 3, 4], false);
        let s2: pso::StaticSampler = ([5, 6, 7, 8], true);
        let base = ChainWholeMeta::new(0xDEAD_BEEF, b"whole-dxil-a", &[s1], 0, 0, 4, true);
        let same = ChainWholeMeta::new(0xDEAD_BEEF, b"whole-dxil-a", &[s1], 0, 0, 4, true);
        let diff_flags = ChainWholeMeta::new(0xDEAD_BEEF, b"whole-dxil-a", &[s1], 1, 0, 4, true);
        let diff_node = ChainWholeMeta::new(0xDEAD_BEEF, b"whole-dxil-a", &[s1], 0, 2, 4, true);
        let diff_samplers = ChainWholeMeta::new(0xDEAD_BEEF, b"whole-dxil-a", &[s1, s2], 0, 0, 4, true);
        let diff_dxil = ChainWholeMeta::new(0xDEAD_BEEF, b"whole-dxil-b", &[s1], 0, 0, 4, true);

        assert_eq!(session.record(&base, ChainWholeReason::Indirect, false, 1), Some(1));
        // Duplicate exact identity with a different reason must not create a second line.
        assert_eq!(session.record(&same, ChainWholeReason::ScratchUnavailable, true, 2), None);
        // Same source_record_hash but distinct final identities must each be recorded separately.
        assert_eq!(session.record(&diff_flags, ChainWholeReason::Indirect, false, 3), Some(2));
        assert_eq!(session.record(&diff_node, ChainWholeReason::Indirect, false, 4), Some(3));
        assert_eq!(session.record(&diff_samplers, ChainWholeReason::Bundle, false, 5), Some(4));
        assert_eq!(session.record(&diff_dxil, ChainWholeReason::MissingRecordingAllocator, false, 6), Some(5));
        assert!(session.flush(Duration::from_secs(2)));
        drop(session);

        let rows = read_lines(&path);
        assert_eq!(rows.len(), 6, "header + 5 distinct identities");
        assert_eq!(rows[1]["reason"], "indirect");
        assert_eq!(rows[2]["effective_flags"], 1);
        assert_eq!(rows[3]["node_mask"], 2);
        assert_eq!(rows[4]["static_samplers"].as_array().unwrap().len(), 2);
        assert_ne!(rows[1]["whole_dxil_hash"], rows[5]["whole_dxil_hash"]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn records_demand_even_when_whole_pso_is_already_ready() {
        let path = temp_path("already_ready");
        let _ = std::fs::remove_file(&path);
        let session = CaptureSession::new(
            CaptureConfig { path: path.clone(), sequence_cutoff: None },
            test_header(None),
        );
        let meta = ChainWholeMeta::new(0xCAFE_0001, b"ready-whole", &[], 0, 0, 4, true);
        assert_eq!(session.record(&meta, ChainWholeReason::Indirect, true, 77), Some(1));
        assert!(session.flush(Duration::from_secs(2)));
        drop(session);

        let rows = read_lines(&path);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["already_ready"], true);
        assert_eq!(rows[1]["present_count"], 77);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sequence_cutoff_caps_distinct_demands_including_zero() {
        // Cutoff = 2: duplicate does not consume sequence budget, 3rd distinct is dropped.
        let path2 = temp_path("cutoff_two");
        let _ = std::fs::remove_file(&path2);
        let s2 = CaptureSession::new(
            CaptureConfig { path: path2.clone(), sequence_cutoff: Some(2) },
            test_header(Some(2)),
        );
        let m1 = ChainWholeMeta::new(1, b"dxil-1", &[], 0, 0, 4, true);
        let m2 = ChainWholeMeta::new(2, b"dxil-2", &[], 0, 0, 4, true);
        let m3 = ChainWholeMeta::new(3, b"dxil-3", &[], 0, 0, 4, true);
        assert_eq!(s2.record(&m1, ChainWholeReason::Indirect, false, 1), Some(1));
        assert_eq!(s2.record(&m1, ChainWholeReason::Indirect, true, 2), None);
        assert_eq!(s2.record(&m2, ChainWholeReason::Bundle, false, 3), Some(2));
        assert_eq!(s2.record(&m3, ChainWholeReason::ScratchUnavailable, false, 4), None);
        assert!(s2.flush(Duration::from_secs(2)));
        drop(s2);
        let rows2 = read_lines(&path2);
        assert_eq!(rows2.len(), 3, "header + 2 demands");
        assert_eq!(rows2[0]["sequence_cutoff"], 2);
        assert_eq!(rows2[1]["seq"], 1);
        assert_eq!(rows2[2]["seq"], 2);
        let _ = std::fs::remove_file(&path2);

        // Cutoff = 0: header only, zero demands recorded.
        let path0 = temp_path("cutoff_zero");
        let _ = std::fs::remove_file(&path0);
        let s0 = CaptureSession::new(
            CaptureConfig { path: path0.clone(), sequence_cutoff: Some(0) },
            test_header(Some(0)),
        );
        assert_eq!(s0.record(&m1, ChainWholeReason::Indirect, false, 1), None);
        assert!(s0.flush(Duration::from_secs(2)));
        drop(s0);
        let rows0 = read_lines(&path0);
        assert_eq!(rows0.len(), 1, "header only when cutoff is 0");
        assert_eq!(rows0[0]["kind"], "header");
        assert_eq!(rows0[0]["sequence_cutoff"], 0);
        let _ = std::fs::remove_file(&path0);
    }

    #[test]
    fn chain_root_signature_identity_tracks_sampler_order_and_comparison() {
        let a: pso::StaticSampler = ([0x11, 0x22, 0x33, 0x44], false);
        let a_cmp: pso::StaticSampler = ([0x11, 0x22, 0x33, 0x44], true);
        let b: pso::StaticSampler = ([0x55, 0x66, 0x77, 0x88], false);
        let id_empty = chain_root_sig_id(&[]);
        let id_a = chain_root_sig_id(&[a]);
        let id_a_cmp = chain_root_sig_id(&[a_cmp]);
        let id_ab = chain_root_sig_id(&[a, b]);
        let id_ba = chain_root_sig_id(&[b, a]);
        assert_eq!(id_ab, chain_root_sig_id(&[a, b]));
        assert_ne!(id_empty, id_a);
        assert_ne!(id_a, id_a_cmp, "comparison flag must affect chain root signature identity");
        assert_ne!(id_ab, id_ba, "static sampler order must affect chain root signature identity");
    }

    #[test]
    fn unwritable_target_path_fails_gracefully_without_panicking() {
        let dir = temp_path("unwritable_dir");
        std::fs::create_dir_all(&dir).expect("create target dir");
        let session = CaptureSession::new(
            CaptureConfig { path: dir.clone(), sequence_cutoff: None },
            test_header(None),
        );
        let meta = ChainWholeMeta::new(0x9999, b"dxil", &[], 0, 0, 4, true);
        let _ = session.record(&meta, ChainWholeReason::Indirect, false, 1);
        assert!(!session.flush(Duration::from_secs(2)), "flushing a directory target must return false cleanly");
        assert!(session.has_failed());
        // Subsequent demands after failure degrade silently without panicking or stalling.
        assert_eq!(session.record(&meta, ChainWholeReason::Bundle, false, 2), None);
        drop(session);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disabled_capture_skips_metadata_construction_for_unset_and_empty_paths() {
        assert!(!enabled(), "global capture must be disabled by default in tests");
        let mut built = 0usize;
        for path_raw in [None, Some(""), Some("   \t\r\n  ")] {
            let (cfg, _) = parse_config_values(path_raw, Some("16"));
            let meta = build_meta_if(cfg.is_some(), || {
                built += 1;
                ChainWholeMeta::new(0x1234, b"whole", &[], 0, 0, 4, true)
            });
            assert!(meta.is_none(), "unset/empty config must leave capture_meta None");
        }
        let default_meta = build_meta_if(enabled(), || {
            built += 1;
            ChainWholeMeta::new(0x1234, b"whole", &[], 0, 0, 4, true)
        });
        assert!(default_meta.is_none());
        assert_eq!(built, 0, "ChainWholeMeta::new must never be invoked when capture is disabled");

        let (cfg_on, _) = parse_config_values(Some("capture.jsonl"), None);
        let meta_on = build_meta_if(cfg_on.is_some(), || {
            built += 1;
            ChainWholeMeta::new(0x1234, b"whole", &[], 0, 0, 4, true)
        });
        assert!(meta_on.is_some());
        assert_eq!(built, 1);
    }

    #[test]
    fn translation_options_record_dxil_no_opt_and_extra_compute_switches() {
        assert_eq!(effective_dxil_no_opt(None), Vec::<String>::new());
        assert_eq!(effective_dxil_no_opt(Some("")), Vec::<String>::new());
        assert_eq!(effective_dxil_no_opt(Some("dce")), vec!["dce"]);
        assert_eq!(effective_dxil_no_opt(Some(" phi , dce , unknown , phi ")), vec!["dce", "phi"]);
        assert_eq!(effective_dxil_no_opt(Some("all")), vec!["cse", "dce", "merge", "phi"]);

        let default_opts = translation_options_from(
            true, None, None, false, None, false, None, 12_500, None, false, None, false,
        );
        let dce_opts = translation_options_from(
            true, None, None, false, None, false, None, 12_500, Some("dce"), false, None, false,
        );
        let custom_opts = translation_options_from(
            true,
            Some(true),
            Some("all".into()),
            true,
            Some("0"),
            true,
            Some("1"),
            8000,
            Some("all"),
            true,
            Some("0"),
            true,
        );

        assert!(default_opts.dxil_no_opt.is_empty());
        assert!(!default_opts.x64_all_masks);
        assert!(default_opts.wave64_spin);
        assert!(default_opts.recomp_structured);

        assert_ne!(default_opts, dce_opts, "XCL_DXIL_NO_OPT=dce must change recorded translation options");
        assert_eq!(dce_opts.dxil_no_opt, vec!["dce"]);

        assert_eq!(custom_opts.dxil_no_opt, vec!["cse", "dce", "merge", "phi"]);
        assert!(custom_opts.x64_all_masks);
        assert!(!custom_opts.wave64_spin);
        assert!(!custom_opts.recomp_structured);
        assert!(!custom_opts.safe_barrier);
        assert!(custom_opts.split_wave64_forced);

        let json = serde_json::to_string(&dce_opts).expect("serialize");
        let parsed: TranslationOptionsMeta = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, dce_opts);

        // Backward-compatible defaults if an older 8-field JSON object is deserialized.
        let legacy = r#"{"direct_dxil":true,"direct_compact":null,"direct_compact_nest":null,"per_lane":false,"safe_barrier":true,"heap_windows":false,"split_wave64_forced":false,"dispatch_piece_limit":12500}"#;
        let from_legacy: TranslationOptionsMeta = serde_json::from_str(legacy).expect("legacy defaults");
        assert_eq!(from_legacy, default_opts);
    }
}
