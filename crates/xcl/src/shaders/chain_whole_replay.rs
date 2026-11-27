//! Selective offline validation and native whole-form `.chain.pso` prewarming for
//! `Chain::whole` capture sidecars (#2741 Steps 2–4).
//!
//! Reads the `schema_version: 1` UTF-8 JSONL capture format produced by
//! `hle::d3d12x::real::chain_whole_capture`, resolves demanded `source_record_hash` entries in
//! a single pass over the shader input, rebuilds each via the runtime compute translation path,
//! verifies full identity parity (ABI, translation options, final whole DXIL, effective flags,
//! node mask, ordered static samplers, and disk cache key), guards against ambiguous disk keys,
//! and creates/persists only the validated `.chain.pso` whole pipelines in first-demand order.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_12_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;

use super::{is_compute_container, pipeline_cap_kb, Input};
use crate::gpu::{recomp, shader};
use crate::hle::d3d12x::real::driver_cache;

/// Expected JSONL sidecar schema version (matches `chain_whole_capture::SCHEMA_VERSION`).
pub(super) const SCHEMA_VERSION: u32 = 1;
/// Expected dispatch-chain root signature ABI version (matches `chain_whole_capture::CHAIN_RS_ABI_VERSION`).
pub(super) const CHAIN_RS_ABI_VERSION: u32 = 1;

/// Ordered static sampler entry matching `chain_whole_capture::StaticSamplerMeta`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct StaticSamplerMeta {
    pub words: [u32; 4],
    pub comparison: bool,
}

impl StaticSamplerMeta {
    fn as_pso(self) -> recomp::pso::StaticSampler {
        (self.words, self.comparison)
    }
}

/// Stable 64-bit identity for the dispatch-chain root signature derived from its ABI version and
/// ordered static samplers (byte-for-byte identical to `chain_whole_capture::chain_root_sig_id`).
pub(super) fn chain_root_sig_id(samplers: &[recomp::pso::StaticSampler]) -> u64 {
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

fn default_true() -> bool {
    true
}

/// Mirrors #2786's 12-field `translation_options` header; serde accepts header `heap_windows` and
/// `split_wave64_forced` while `mem_windows` (including `HEAP_WINDOWS` / switch-form) and
/// `split_wave64` are validated per demand record.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct TranslationOptionsMeta {
    direct_dxil: bool,
    #[serde(default)]
    direct_compact: Option<bool>,
    #[serde(default)]
    direct_compact_nest: Option<String>,
    per_lane: bool,
    safe_barrier: bool,
    dispatch_piece_limit: usize,
    #[serde(default)]
    dxil_no_opt: Vec<String>,
    #[serde(default)]
    x64_all_masks: bool,
    #[serde(default = "default_true")]
    wave64_spin: bool,
    #[serde(default = "default_true")]
    recomp_structured: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct HeaderRecord {
    chain_rs_abi_version: u32,
    #[serde(default)]
    sequence_cutoff: Option<u64>,
    #[serde(default)]
    translation_options: Option<TranslationOptionsMeta>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct RawDemandRecord {
    seq: u64,
    source_record_hash: String,
    whole_dxil_hash: String,
    whole_dxil_len: usize,
    effective_flags: i32,
    node_mask: u32,
    chain_rs_abi_version: u32,
    chain_rs_id: String,
    #[serde(default)]
    static_samplers: Vec<StaticSamplerMeta>,
    disk_cache_name: String,
    mem_windows: u32,
    split_wave64: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DemandIdentity {
    source_record_hash: u64,
    whole_dxil_hash: u64,
    whole_dxil_len: usize,
    effective_flags: i32,
    node_mask: u32,
    chain_rs_abi_version: u32,
    chain_rs_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedDemand {
    seq: u64,
    identity: DemandIdentity,
    static_samplers: Vec<recomp::pso::StaticSampler>,
    disk_cache_name: String,
    mem_windows: u32,
    split_wave64: bool,
}

#[derive(Debug, Clone)]
struct ParsedCapture {
    header: Option<HeaderRecord>,
    effective_cutoff: Option<u64>,
    total_demand_rows: usize,
    dropped_by_cutoff: usize,
    duplicate_rows_skipped: usize,
    demands: Vec<ParsedDemand>,
}

fn parse_hex16(field: &str, value: &str, line: usize) -> Result<u64, String> {
    let s = value.trim();
    if s.len() != 16 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("line {line}: invalid 16-char hex {field}: {value:?}"));
    }
    u64::from_str_radix(s, 16).map_err(|e| format!("line {line}: invalid hex {field}: {e}"))
}

fn parse_capture(text: &str, cli_cutoff: Option<u64>) -> Result<ParsedCapture, String> {
    let mut header: Option<HeaderRecord> = None;
    let mut raw_demands: Vec<ParsedDemand> = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let ln = idx + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: serde_json::Value =
            serde_json::from_str(trimmed).map_err(|e| format!("line {ln}: invalid JSON: {e}"))?;
        let ver = val
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("line {ln}: missing integer schema_version"))?;
        if ver != u64::from(SCHEMA_VERSION) {
            return Err(format!("line {ln}: unsupported schema_version {ver} (expected {SCHEMA_VERSION})"));
        }
        let kind = val
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("line {ln}: missing string kind"))?;
        match kind {
            "header" => {
                if header.is_some() {
                    return Err(format!("line {ln}: duplicate header record"));
                }
                let h: HeaderRecord =
                    serde_json::from_value(val).map_err(|e| format!("line {ln}: invalid header: {e}"))?;
                header = Some(h);
            }
            "demand" => {
                let d: RawDemandRecord =
                    serde_json::from_value(val).map_err(|e| format!("line {ln}: invalid demand: {e}"))?;
                if d.seq == 0 {
                    return Err(format!("line {ln}: demand seq must be >= 1"));
                }
                let identity = DemandIdentity {
                    source_record_hash: parse_hex16("source_record_hash", &d.source_record_hash, ln)?,
                    whole_dxil_hash: parse_hex16("whole_dxil_hash", &d.whole_dxil_hash, ln)?,
                    whole_dxil_len: d.whole_dxil_len,
                    effective_flags: d.effective_flags,
                    node_mask: d.node_mask,
                    chain_rs_abi_version: d.chain_rs_abi_version,
                    chain_rs_id: parse_hex16("chain_rs_id", &d.chain_rs_id, ln)?,
                };
                raw_demands.push(ParsedDemand {
                    seq: d.seq,
                    identity,
                    static_samplers: d.static_samplers.into_iter().map(StaticSamplerMeta::as_pso).collect(),
                    disk_cache_name: d.disk_cache_name,
                    mem_windows: d.mem_windows,
                    split_wave64: d.split_wave64,
                });
            }
            other => return Err(format!("line {ln}: unknown record kind {other:?}")),
        }
    }

    let header_cutoff = header.as_ref().and_then(|h| h.sequence_cutoff);
    let effective_cutoff = match (cli_cutoff, header_cutoff) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };

    raw_demands.sort_by_key(|d| d.seq);
    let total_demand_rows = raw_demands.len();
    let mut dropped_by_cutoff = 0usize;
    let mut duplicate_rows_skipped = 0usize;
    let mut seen: HashSet<DemandIdentity> = HashSet::new();
    let mut demands = Vec::new();
    for d in raw_demands {
        if effective_cutoff.is_some_and(|max| d.seq > max) {
            dropped_by_cutoff += 1;
            continue;
        }
        if !seen.insert(d.identity) {
            duplicate_rows_skipped += 1;
            continue;
        }
        demands.push(d);
    }

    Ok(ParsedCapture {
        header,
        effective_cutoff,
        total_demand_rows,
        dropped_by_cutoff,
        duplicate_rows_skipped,
        demands,
    })
}

/// Canonical sorted subset of disabled direct-DXIL optimization passes matching
/// `gpu::dxil::module::opt::optimize` (`XCL_DXIL_NO_OPT=phi,cse,dce,merge` or `all`).
fn parse_dxil_no_opt(raw: Option<&str>) -> Vec<String> {
    const PASSES: [&str; 4] = ["cse", "dce", "merge", "phi"];
    let Some(off) = raw else {
        return Vec::new();
    };
    PASSES
        .iter()
        .copied()
        .filter(|&p| off.split(',').any(|x| x.trim() == p || x.trim() == "all"))
        .map(String::from)
        .collect()
}

/// Active translation environment parameters compared against the capture header when present.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ReplayEnv {
    direct_dxil: bool,
    direct_compact: Option<bool>,
    direct_compact_nest: Option<String>,
    per_lane: bool,
    safe_barrier: bool,
    dispatch_piece_limit: usize,
    dxil_no_opt: Vec<String>,
    x64_all_masks: bool,
    wave64_spin: bool,
    recomp_structured: bool,
    explicit_windows: Option<u32>,
    explicit_split: Option<bool>,
}

impl ReplayEnv {
    fn from_process(explicit_windows: Option<u32>, explicit_split: Option<bool>) -> Self {
        Self {
            direct_dxil: crate::gpu::dxil::kernel::enabled(),
            direct_compact: crate::gpu::dxil::kernel::compact_mode(),
            direct_compact_nest: crate::gpu::dxil::kernel::compact_nest_tag(),
            per_lane: crate::gpu::recomp::lower_dxil::enabled(),
            safe_barrier: std::env::var("XCL_SAFE_BARRIER").map_or(true, |v| v.trim() != "0"),
            dispatch_piece_limit: crate::gpu::dxil::kernel::dispatch_piece_limit(),
            dxil_no_opt: parse_dxil_no_opt(std::env::var("XCL_DXIL_NO_OPT").ok().as_deref()),
            x64_all_masks: std::env::var_os("XCL_X64_ALL_MASKS").is_some(),
            wave64_spin: std::env::var("XCL_WAVE64_SPIN").map_or(true, |v| v.trim() != "0"),
            recomp_structured: std::env::var_os("XCL_RECOMP_NO_STRUCTURED").is_none(),
            explicit_windows,
            explicit_split,
        }
    }

    fn header_mismatch(&self, h: &HeaderRecord) -> Option<String> {
        let opts = h.translation_options.as_ref()?;
        let mut diffs = Vec::new();
        if opts.direct_dxil != self.direct_dxil {
            diffs.push(format!("direct_dxil capture={} current={}", opts.direct_dxil, self.direct_dxil));
        }
        if opts.direct_compact != self.direct_compact {
            diffs.push(format!("direct_compact capture={:?} current={:?}", opts.direct_compact, self.direct_compact));
        }
        if opts.direct_compact_nest != self.direct_compact_nest {
            diffs.push(format!("direct_compact_nest capture={:?} current={:?}", opts.direct_compact_nest, self.direct_compact_nest));
        }
        if opts.per_lane != self.per_lane {
            diffs.push(format!("per_lane capture={} current={}", opts.per_lane, self.per_lane));
        }
        if opts.safe_barrier != self.safe_barrier {
            diffs.push(format!("safe_barrier capture={} current={}", opts.safe_barrier, self.safe_barrier));
        }
        if opts.dispatch_piece_limit != self.dispatch_piece_limit {
            diffs.push(format!(
                "dispatch_piece_limit capture={} current={}",
                opts.dispatch_piece_limit, self.dispatch_piece_limit
            ));
        }
        let mut cap_no_opt = opts.dxil_no_opt.clone();
        cap_no_opt.sort();
        cap_no_opt.dedup();
        if cap_no_opt != self.dxil_no_opt {
            diffs.push(format!("dxil_no_opt capture={:?} current={:?}", cap_no_opt, self.dxil_no_opt));
        }
        if opts.x64_all_masks != self.x64_all_masks {
            diffs.push(format!("x64_all_masks capture={} current={}", opts.x64_all_masks, self.x64_all_masks));
        }
        if opts.wave64_spin != self.wave64_spin {
            diffs.push(format!("wave64_spin capture={} current={}", opts.wave64_spin, self.wave64_spin));
        }
        if opts.recomp_structured != self.recomp_structured {
            diffs.push(format!("recomp_structured capture={} current={}", opts.recomp_structured, self.recomp_structured));
        }
        (!diffs.is_empty()).then(|| diffs.join(", "))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum IssueKind {
    MissingRecord,
    AbiMismatch,
    ConfigMismatch,
    UnsupportedDescriptor,
    TranslationFailed,
    FormChanged,
    WholeDxilMismatch,
    RootSignatureMismatch,
    CacheKeyMismatch,
    AmbiguousDiskKey,
}

impl IssueKind {
    fn label(self) -> &'static str {
        match self {
            Self::MissingRecord => "missing_record",
            Self::AbiMismatch => "abi_mismatch",
            Self::ConfigMismatch => "config_mismatch",
            Self::UnsupportedDescriptor => "unsupported_descriptor",
            Self::TranslationFailed => "translation_failed",
            Self::FormChanged => "form_changed",
            Self::WholeDxilMismatch => "whole_dxil_mismatch",
            Self::RootSignatureMismatch => "root_signature_mismatch",
            Self::CacheKeyMismatch => "cache_key_mismatch",
            Self::AmbiguousDiskKey => "ambiguous_disk_key",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlanIssue {
    seq: u64,
    source_record_hash: u64,
    kind: IssueKind,
    detail: String,
}

#[derive(Debug, Clone)]
struct PlannedItem {
    seq: u64,
    source_record_hash: u64,
    whole_dxil: Vec<u8>,
    samplers: Vec<recomp::pso::StaticSampler>,
    chain_rs_id: u64,
    effective_flags: i32,
    node_mask: u32,
    disk_cache_name: String,
}

struct ReplayPlan {
    selected: usize,
    dropped_by_cutoff: usize,
    duplicate_rows_skipped: usize,
    collapsed_shared_disk_keys: usize,
    ready: Vec<PlannedItem>,
    issues: Vec<PlanIssue>,
}

impl ReplayPlan {
    fn count_issue(&self, kind: IssueKind) -> usize {
        self.issues.iter().filter(|i| i.kind == kind).count()
    }
}

/// Resolve every demanded `source_record_hash` in a single pass over `input`, accepting supported
/// console compute containers (FSSM and FSPX/ACCC).
fn resolve_sources(input: &Input, wanted: &HashSet<u64>) -> HashMap<u64, (String, Vec<u8>)> {
    let mut found = HashMap::with_capacity(wanted.len());
    if wanted.is_empty() {
        return found;
    }
    for i in 0..input.len() {
        if found.len() == wanted.len() {
            break;
        }
        let (name, blob) = input.get(i);
        let Some(blob) = blob else { continue };
        let h = shader::fnv64(&blob);
        if !wanted.contains(&h) || found.contains_key(&h) {
            continue;
        }
        if is_compute_container(&blob) {
            found.insert(h, (name, blob));
        }
    }
    found
}

fn plan_replay<F>(
    capture: &ParsedCapture,
    sources: &HashMap<u64, (String, Vec<u8>)>,
    env: &ReplayEnv,
    mut translate: F,
) -> ReplayPlan
where
    F: FnMut(&[u8], bool, u32) -> Result<recomp::pso::Compiled, String>,
{
    let mut ready = Vec::new();
    let mut issues = Vec::new();

    let header_abi_bad = capture
        .header
        .as_ref()
        .filter(|h| h.chain_rs_abi_version != CHAIN_RS_ABI_VERSION)
        .map(|h| h.chain_rs_abi_version);
    let header_cfg_bad = capture.header.as_ref().and_then(|h| env.header_mismatch(h));

    for demand in &capture.demands {
        let seq = demand.seq;
        let rec_hash = demand.identity.source_record_hash;
        let mut push_issue = |kind: IssueKind, detail: String| {
            issues.push(PlanIssue { seq, source_record_hash: rec_hash, kind, detail });
        };

        if let Some(got) = header_abi_bad {
            push_issue(
                IssueKind::AbiMismatch,
                format!("capture header chain_rs_abi_version={got} (expected {CHAIN_RS_ABI_VERSION})"),
            );
            continue;
        }
        if demand.identity.chain_rs_abi_version != CHAIN_RS_ABI_VERSION {
            push_issue(
                IssueKind::AbiMismatch,
                format!(
                    "demand chain_rs_abi_version={} (expected {CHAIN_RS_ABI_VERSION})",
                    demand.identity.chain_rs_abi_version
                ),
            );
            continue;
        }
        if let Some(msg) = &header_cfg_bad {
            push_issue(IssueKind::ConfigMismatch, format!("header translation options differ: {msg}"));
            continue;
        }
        if env.explicit_windows.is_some_and(|w| w != demand.mem_windows) {
            push_issue(
                IssueKind::ConfigMismatch,
                format!("--windows {} does not match recorded mem_windows={}", env.explicit_windows.unwrap(), demand.mem_windows),
            );
            continue;
        }
        if env.explicit_split.is_some_and(|s| s != demand.split_wave64) {
            push_issue(
                IssueKind::ConfigMismatch,
                format!("--split {} does not match recorded split_wave64={}", env.explicit_split.unwrap(), demand.split_wave64),
            );
            continue;
        }
        // On current main, driver_cache::name_for omits NodeMask; reject non-zero node masks
        // until matching runtime/offline disk keys carry NodeMask (#2741 Step 2).
        if demand.identity.node_mask != 0 {
            push_issue(
                IssueKind::UnsupportedDescriptor,
                format!("non-zero node_mask={:#x} is not distinguished by current disk cache keys", demand.identity.node_mask),
            );
            continue;
        }
        let recorded_rs_id = chain_root_sig_id(&demand.static_samplers);
        if recorded_rs_id != demand.identity.chain_rs_id {
            push_issue(
                IssueKind::RootSignatureMismatch,
                format!(
                    "recorded static_samplers hash {recorded_rs_id:016x} != chain_rs_id {:016x}",
                    demand.identity.chain_rs_id
                ),
            );
            continue;
        }
        let Some((_name, blob)) = sources.get(&rec_hash) else {
            push_issue(
                IssueKind::MissingRecord,
                format!("source record {rec_hash:016x} not found in compute shader input"),
            );
            continue;
        };
        let compiled = match translate(blob, demand.split_wave64, demand.mem_windows) {
            Ok(c) => c,
            Err(e) => {
                push_issue(IssueKind::TranslationFailed, format!("retranslation failed: {e}"));
                continue;
            }
        };
        if !compiled.dispatch_chain.as_deref().is_some_and(|c| c.pieces.len() >= 2) {
            push_issue(
                IssueKind::FormChanged,
                "rebuilt shader no longer produces a split dispatch chain".into(),
            );
            continue;
        }
        let rebuilt_hash = shader::fnv64(&compiled.dxil);
        if rebuilt_hash != demand.identity.whole_dxil_hash || compiled.dxil.len() != demand.identity.whole_dxil_len {
            push_issue(
                IssueKind::WholeDxilMismatch,
                format!(
                    "rebuilt whole DXIL {rebuilt_hash:016x} ({} B) != recorded {:016x} ({} B)",
                    compiled.dxil.len(),
                    demand.identity.whole_dxil_hash,
                    demand.identity.whole_dxil_len
                ),
            );
            continue;
        }
        let rebuilt_rs_id = chain_root_sig_id(&compiled.samplers);
        if compiled.samplers != demand.static_samplers || rebuilt_rs_id != demand.identity.chain_rs_id {
            push_issue(
                IssueKind::RootSignatureMismatch,
                format!(
                    "rebuilt chain root signature {rebuilt_rs_id:016x} ({} samplers) != recorded {:016x} ({} samplers)",
                    compiled.samplers.len(),
                    demand.identity.chain_rs_id,
                    demand.static_samplers.len()
                ),
            );
            continue;
        }
        let expected_disk_name = driver_cache::name_for(&compiled.dxil, demand.identity.effective_flags, true);
        if expected_disk_name != demand.disk_cache_name {
            push_issue(
                IssueKind::CacheKeyMismatch,
                format!("computed disk key {expected_disk_name:?} != recorded {:?}", demand.disk_cache_name),
            );
            continue;
        }
        ready.push(PlannedItem {
            seq,
            source_record_hash: rec_hash,
            whole_dxil: compiled.dxil,
            samplers: compiled.samplers,
            chain_rs_id: rebuilt_rs_id,
            effective_flags: demand.identity.effective_flags,
            node_mask: demand.identity.node_mask,
            disk_cache_name: expected_disk_name,
        });
    }

    let (ready, collapsed_shared_disk_keys) = dedup_and_check_disk_keys(ready, &mut issues);

    ReplayPlan {
        selected: capture.demands.len(),
        dropped_by_cutoff: capture.dropped_by_cutoff,
        duplicate_rows_skipped: capture.duplicate_rows_skipped,
        collapsed_shared_disk_keys,
        ready,
        issues,
    }
}

/// Detect disk-key collisions across distinct validated whole pipelines and collapse identical
/// whole build targets sharing the same `disk_cache_name` so no two parallel workers ever race on
/// one `.chain.pso` file on Windows.
fn dedup_and_check_disk_keys(
    ready: Vec<PlannedItem>,
    issues: &mut Vec<PlanIssue>,
) -> (Vec<PlannedItem>, usize) {
    let mut first_by_key: HashMap<&str, usize> = HashMap::new();
    let mut colliding: HashSet<String> = HashSet::new();
    for (idx, item) in ready.iter().enumerate() {
        match first_by_key.get(item.disk_cache_name.as_str()) {
            None => {
                first_by_key.insert(item.disk_cache_name.as_str(), idx);
            }
            Some(&first_idx) => {
                let first = &ready[first_idx];
                let same_target = item.chain_rs_id == first.chain_rs_id
                    && item.node_mask == first.node_mask
                    && item.effective_flags == first.effective_flags
                    && item.whole_dxil == first.whole_dxil;
                if !same_target {
                    colliding.insert(item.disk_cache_name.clone());
                }
            }
        }
    }

    let mut kept = Vec::with_capacity(ready.len());
    let mut scheduled_keys: HashSet<String> = HashSet::new();
    let mut collapsed = 0usize;
    for item in ready {
        if colliding.contains(&item.disk_cache_name) {
            issues.push(PlanIssue {
                seq: item.seq,
                source_record_hash: item.source_record_hash,
                kind: IssueKind::AmbiguousDiskKey,
                detail: format!(
                    "multiple distinct whole pipelines (chain root signature, node mask, or DXIL bytes) share disk key {:?}",
                    item.disk_cache_name
                ),
            });
        } else if scheduled_keys.insert(item.disk_cache_name.clone()) {
            kept.push(item);
        } else {
            collapsed += 1;
        }
    }
    (kept, collapsed)
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct ExecutionCounts {
    created_with_supplied_blob: usize,
    created_fresh: usize,
    rejected_and_cold_retried: usize,
    persisted: usize,
    blob_unavailable: usize,
    write_failed: usize,
    cache_disabled: usize,
    capped: usize,
    refused: usize,
}

impl ExecutionCounts {
    fn is_complete(&self, planned_ready: usize, validation_issues: usize) -> bool {
        validation_issues == 0
            && self.capped == 0
            && self.refused == 0
            && self.blob_unavailable == 0
            && self.write_failed == 0
            && self.cache_disabled == 0
            && self.created_with_supplied_blob + self.persisted == planned_ready
    }
}

struct AtomicCounts {
    created_with_supplied_blob: AtomicUsize,
    created_fresh: AtomicUsize,
    rejected_and_cold_retried: AtomicUsize,
    persisted: AtomicUsize,
    blob_unavailable: AtomicUsize,
    write_failed: AtomicUsize,
    cache_disabled: AtomicUsize,
    capped: AtomicUsize,
    refused: AtomicUsize,
}

impl AtomicCounts {
    fn new() -> Self {
        Self {
            created_with_supplied_blob: AtomicUsize::new(0),
            created_fresh: AtomicUsize::new(0),
            rejected_and_cold_retried: AtomicUsize::new(0),
            persisted: AtomicUsize::new(0),
            blob_unavailable: AtomicUsize::new(0),
            write_failed: AtomicUsize::new(0),
            cache_disabled: AtomicUsize::new(0),
            capped: AtomicUsize::new(0),
            refused: AtomicUsize::new(0),
        }
    }

    fn snapshot(&self) -> ExecutionCounts {
        ExecutionCounts {
            created_with_supplied_blob: self.created_with_supplied_blob.load(Ordering::Relaxed),
            created_fresh: self.created_fresh.load(Ordering::Relaxed),
            rejected_and_cold_retried: self.rejected_and_cold_retried.load(Ordering::Relaxed),
            persisted: self.persisted.load(Ordering::Relaxed),
            blob_unavailable: self.blob_unavailable.load(Ordering::Relaxed),
            write_failed: self.write_failed.load(Ordering::Relaxed),
            cache_disabled: self.cache_disabled.load(Ordering::Relaxed),
            capped: self.capped.load(Ordering::Relaxed),
            refused: self.refused.load(Ordering::Relaxed),
        }
    }
}

struct ReplayBuilder {
    dev: ID3D12Device,
    signatures: Mutex<HashMap<Vec<u32>, ID3D12RootSignature>>,
    running: (Mutex<usize>, Condvar),
    most: usize,
    max_dxil: usize,
    counts: AtomicCounts,
}

// D3D12 devices and root signatures are free-threaded.
unsafe impl Sync for ReplayBuilder {}

impl ReplayBuilder {
    fn open() -> Result<Self, String> {
        unsafe {
            let factory: IDXGIFactory6 =
                CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| e.to_string())?;
            let sel = crate::gpu::adapter::Selector::from_env_or("auto");
            let dev = if sel == crate::gpu::adapter::Selector::Warp {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().map_err(|e| format!("WARP: {e}"))?;
                crate::gpu::agility::create_device(&warp, D3D_FEATURE_LEVEL_11_0).map_err(|e| e.to_string())?
            } else {
                let (adapter, _) = crate::gpu::adapter::candidates(&factory, &sel)
                    .into_iter()
                    .next()
                    .ok_or_else(|| crate::gpu::adapter::no_match(&factory, &sel))?;
                crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0).map_err(|e| e.to_string())?
            };
            let most = std::env::var("XCL_PIPELINE_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4usize)
                .max(1);
            let max_dxil = pipeline_cap_kb(std::env::var("XCL_PIPELINE_MAX_KB").ok().as_deref()).saturating_mul(1024);
            Ok(Self {
                dev,
                signatures: Mutex::new(HashMap::new()),
                running: (Mutex::new(0), Condvar::new()),
                most,
                max_dxil,
                counts: AtomicCounts::new(),
            })
        }
    }

    fn root_signature_for(&self, samplers: &[recomp::pso::StaticSampler]) -> Result<ID3D12RootSignature, String> {
        let key: Vec<u32> = samplers
            .iter()
            .flat_map(|(w, cmp)| w.iter().copied().chain([u32::from(*cmp)]))
            .collect();
        let mut g = self.signatures.lock().unwrap();
        if let Some(rs) = g.get(&key) {
            return Ok(rs.clone());
        }
        let rs = recomp::pso::dispatch_chain_root_signature(&self.dev, samplers)?;
        g.insert(key, rs.clone());
        Ok(rs)
    }

    fn build_one(&self, item: &PlannedItem) {
        let Some(path) = driver_cache::path_for(&item.whole_dxil, item.effective_flags, true) else {
            self.counts.cache_disabled.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let Ok(rs) = self.root_signature_for(&item.samplers) else {
            self.counts.refused.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(rs)),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: item.whole_dxil.as_ptr().cast(),
                BytecodeLength: item.whole_dxil.len(),
            },
            NodeMask: item.node_mask,
            Flags: D3D12_PIPELINE_STATE_FLAGS(item.effective_flags),
            ..Default::default()
        };
        let cached = driver_cache::load(&path);
        let over_cap = self.max_dxil != 0 && item.whole_dxil.len() > self.max_dxil;
        if cached.is_none() && over_cap {
            self.counts.capped.fetch_add(1, Ordering::Relaxed);
            drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
            return;
        }
        {
            let mut n = self.running.0.lock().unwrap();
            while *n >= self.most {
                n = self.running.1.wait(n).unwrap();
            }
            *n += 1;
        }
        let from_cache = cached.as_ref().and_then(|bytes| {
            desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE {
                pCachedBlob: bytes.as_ptr().cast(),
                CachedBlobSizeInBytes: bytes.len(),
            };
            unsafe { self.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) }
                .ok()
                .filter(|pso| unsafe { pso.GetCachedBlob() }.is_ok_and(|b| unsafe { b.GetBufferSize() } == bytes.len()))
        });
        if cached.is_some() && from_cache.is_none() {
            let _ = std::fs::remove_file(&path);
            if over_cap {
                *self.running.0.lock().unwrap() -= 1;
                self.running.1.notify_one();
                self.counts.capped.fetch_add(1, Ordering::Relaxed);
                drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
                return;
            }
            self.counts.rejected_and_cold_retried.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(_pso) = from_cache {
            self.counts.created_with_supplied_blob.fetch_add(1, Ordering::Relaxed);
        } else {
            desc.CachedPSO = D3D12_CACHED_PIPELINE_STATE::default();
            match unsafe { self.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) } {
                Ok(pso) => {
                    self.counts.created_fresh.fetch_add(1, Ordering::Relaxed);
                    match unsafe { pso.GetCachedBlob() } {
                        Ok(blob) => {
                            let ptr = unsafe { blob.GetBufferPointer() }.cast::<u8>();
                            let len = unsafe { blob.GetBufferSize() };
                            if ptr.is_null() || len == 0 {
                                self.counts.blob_unavailable.fetch_add(1, Ordering::Relaxed);
                            } else if driver_cache::write(&path, unsafe { std::slice::from_raw_parts(ptr, len) }) {
                                self.counts.persisted.fetch_add(1, Ordering::Relaxed);
                            } else {
                                self.counts.write_failed.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        Err(_) => {
                            self.counts.blob_unavailable.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                Err(_) => {
                    self.counts.refused.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        *self.running.0.lock().unwrap() -= 1;
        self.running.1.notify_one();
        drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
    }

    fn execute(&self, items: &[PlannedItem]) -> ExecutionCounts {
        let workers = items.len().min(self.most);
        if workers <= 1 {
            for item in items {
                self.build_one(item);
            }
        } else {
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..workers {
                    s.spawn(|| loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(i) else { break };
                        self.build_one(item);
                    });
                }
            });
        }
        self.counts.snapshot()
    }
}

pub(super) fn run(
    input: &Input,
    capture_path: &Path,
    cli_cutoff: Option<u64>,
    explicit_windows: Option<u32>,
    explicit_split: Option<bool>,
    t0: Instant,
) -> i32 {
    let text = match std::fs::read_to_string(capture_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("xcl shaders --replay-chain-whole: {}: {e}", capture_path.display());
            return 2;
        }
    };
    let capture = match parse_capture(&text, cli_cutoff) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("xcl shaders --replay-chain-whole: {}: {e}", capture_path.display());
            return 2;
        }
    };
    let wanted: HashSet<u64> = capture.demands.iter().map(|d| d.identity.source_record_hash).collect();
    let sources = resolve_sources(input, &wanted);
    let env = ReplayEnv::from_process(explicit_windows, explicit_split);
    let plan = plan_replay(&capture, &sources, &env, |blob, split_wave64, windows| {
        recomp::pso::console_to_dxil_in(blob, recomp::pso::Stage::Compute, 0, split_wave64, windows)
    });

    let counts = if plan.ready.is_empty() {
        ExecutionCounts::default()
    } else {
        let builder = match ReplayBuilder::open() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("xcl shaders --replay-chain-whole: no GPU device: {e}");
                return 2;
            }
        };
        builder.execute(&plan.ready)
    };

    let complete = counts.is_complete(plan.ready.len(), plan.issues.len());
    println!(
        "chain-whole replay ({:.2} s): total_rows {}, selected {}, dropped_by_cutoff {}, duplicates_skipped {}, collapsed_shared_keys {}, validated {}, issues {}",
        t0.elapsed().as_secs_f64(),
        capture.total_demand_rows,
        plan.selected,
        plan.dropped_by_cutoff,
        plan.duplicate_rows_skipped,
        plan.collapsed_shared_disk_keys,
        plan.ready.len(),
        plan.issues.len(),
    );
    if let Some(cutoff) = capture.effective_cutoff {
        println!("  effective_cutoff: {cutoff}");
    }
    println!(
        "  native: created_with_supplied_blob {}, created_fresh {}, rejected_and_cold_retried {}, persisted {}, blob_unavailable {}, write_failed {}, cache_disabled {}, capped {}, refused {}",
        counts.created_with_supplied_blob,
        counts.created_fresh,
        counts.rejected_and_cold_retried,
        counts.persisted,
        counts.blob_unavailable,
        counts.write_failed,
        counts.cache_disabled,
        counts.capped,
        counts.refused,
    );
    if !plan.issues.is_empty() {
        println!(
            "  validation_issues: missing {}, abi {}, config {}, unsupported {}, translation {}, form_changed {}, dxil {}, root_sig {}, cache_key {}, ambiguous_disk_key {}",
            plan.count_issue(IssueKind::MissingRecord),
            plan.count_issue(IssueKind::AbiMismatch),
            plan.count_issue(IssueKind::ConfigMismatch),
            plan.count_issue(IssueKind::UnsupportedDescriptor),
            plan.count_issue(IssueKind::TranslationFailed),
            plan.count_issue(IssueKind::FormChanged),
            plan.count_issue(IssueKind::WholeDxilMismatch),
            plan.count_issue(IssueKind::RootSignatureMismatch),
            plan.count_issue(IssueKind::CacheKeyMismatch),
            plan.count_issue(IssueKind::AmbiguousDiskKey),
        );
        for issue in &plan.issues {
            println!(
                "    seq={} record={:016x} [{}]: {}",
                issue.seq,
                issue.source_record_hash,
                issue.kind.label(),
                issue.detail,
            );
        }
    }
    println!("  complete: {complete}");
    if complete { 0 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn sample_env() -> ReplayEnv {
        ReplayEnv {
            direct_dxil: true,
            direct_compact: None,
            direct_compact_nest: None,
            per_lane: false,
            safe_barrier: true,
            dispatch_piece_limit: 12_500,
            dxil_no_opt: Vec::new(),
            x64_all_masks: false,
            wave64_spin: true,
            recomp_structured: true,
            explicit_windows: None,
            explicit_split: None,
        }
    }

    fn make_compiled(dxil: &[u8], samplers: &[recomp::pso::StaticSampler], pieces: usize) -> recomp::pso::Compiled {
        let dispatch_chain = (pieces >= 2).then(|| {
            Arc::new(crate::gpu::dxil::kernel::DispatchChain {
                pieces: (0..pieces).map(|i| vec![0xB0 + i as u8; 16 + i]).collect(),
                dwords_per_group: 4,
            })
        });
        recomp::pso::Compiled {
            dispatch_chain,
            dxil: dxil.to_vec(),
            samplers: samplers.to_vec(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn demand_json(
        seq: u64,
        record_hash: u64,
        whole_dxil: &[u8],
        flags: i32,
        node_mask: u32,
        abi_ver: u32,
        samplers: &[recomp::pso::StaticSampler],
        mem_windows: u32,
        split_wave64: bool,
    ) -> String {
        let rs_id = chain_root_sig_id(samplers);
        let disk_name = driver_cache::name_for(whole_dxil, flags, true);
        let samp_json: Vec<serde_json::Value> = samplers
            .iter()
            .map(|&(words, comparison)| serde_json::json!({ "words": words, "comparison": comparison }))
            .collect();
        serde_json::json!({
            "schema_version": SCHEMA_VERSION,
            "kind": "demand",
            "seq": seq,
            "elapsed_ms": seq * 10,
            "timestamp_unix_ms": 1_700_000_000_000u64 + seq,
            "present_count": seq,
            "reason": "indirect",
            "already_ready": false,
            "source_record_hash": format!("{record_hash:016x}"),
            "whole_dxil_hash": format!("{:016x}", shader::fnv64(whole_dxil)),
            "whole_dxil_len": whole_dxil.len(),
            "effective_flags": flags,
            "node_mask": node_mask,
            "chain_rs_abi_version": abi_ver,
            "chain_rs_id": format!("{rs_id:016x}"),
            "static_samplers": samp_json,
            "disk_cache_name": disk_name,
            "mem_windows": mem_windows,
            "split_wave64": split_wave64,
        })
        .to_string()
    }

    fn header_json(cutoff: Option<u64>, abi_ver: u32, direct_dxil: bool) -> String {
        serde_json::json!({
            "schema_version": SCHEMA_VERSION,
            "kind": "header",
            "xcl_version": "0.1.0",
            "translator_source_id": "699c1853e347d9bb",
            "direct_dxil_source_id": "f0d04875e42b84d5",
            "direct_dxil_translator_version": 9,
            "per_lane_source_id": "39cdba116441e869",
            "chain_rs_abi_version": abi_ver,
            "sequence_cutoff": cutoff,
            "captured_at_unix_ms": 1_700_000_000_000u64,
            "adapter": null,
            "agility_runtime": { "status": "not_found", "version": null },
            "dxc": {
                "dxcompiler": { "file_version": "1.9.2609.5", "major": 1, "minor": 9, "commit": "01b62ad4" },
                "dxil": { "file_version": "1.9.2609.5", "major": 1, "minor": 9, "commit": "" }
            },
            "translation_options": {
                "direct_dxil": direct_dxil,
                "direct_compact": null,
                "direct_compact_nest": null,
                "per_lane": false,
                "safe_barrier": true,
                "heap_windows": false,
                "split_wave64_forced": false,
                "dispatch_piece_limit": 12500,
                "dxil_no_opt": [],
                "x64_all_masks": false,
                "wave64_spin": true,
                "recomp_structured": true
            }
        })
        .to_string()
    }

    #[test]
    fn golden_chain_root_sig_id_matches_capture_schema_v1() {
        let a: recomp::pso::StaticSampler = ([0x11, 0x22, 0x33, 0x44], false);
        let a_cmp: recomp::pso::StaticSampler = ([0x11, 0x22, 0x33, 0x44], true);
        let b: recomp::pso::StaticSampler = ([0x55, 0x66, 0x77, 0x88], true);
        // Pinned byte-for-byte to Part 1 `chain_whole_capture::chain_root_sig_id` (schema_version=1).
        assert_eq!(chain_root_sig_id(&[]), 0x73fb_fe4e_b45e_d583);
        assert_eq!(chain_root_sig_id(&[a]), 0xa79e_2466_2b1c_0cd2);
        assert_eq!(chain_root_sig_id(&[a_cmp]), 0xa79e_2566_2b1c_0e85);
        assert_eq!(chain_root_sig_id(&[a, b]), 0x45b4_f285_8922_b776);
        assert_eq!(chain_root_sig_id(&[b, a]), 0xc55d_c87a_f47e_2de0);

        let fixture = format!(
            "{}\n{}\n",
            header_json(Some(64), 1, true),
            demand_json(1, 0x00C0_FFEE, b"whole-dxil-fixture", 0, 0, 1, &[a, b], 1, true),
        );
        let parsed = parse_capture(&fixture, None).expect("parse Part 1 schema v1 fixture");
        assert_eq!(parsed.demands.len(), 1);
        assert_eq!(parsed.demands[0].identity.chain_rs_id, 0x45b4_f285_8922_b776);
        assert_eq!(parsed.effective_cutoff, Some(64));
    }

    #[test]
    fn replay_plan_enforces_exact_cutoff_dedup_and_multi_identity_per_source_hash() {
        let blob_a = b"console-cs-a".to_vec();
        let hash_a = shader::fnv64(&blob_a);
        let hash_missing = 0xDEAD_0000_0000_9999u64;
        let dxil_w1 = b"whole-dxil-a-win1";
        let dxil_w4 = b"whole-dxil-a-win4";

        // Same source_record_hash with two distinct final identities (flags 0 vs 1, windows 1 vs 4),
        // an exact duplicate of seq=1 at seq=2, a missing record at seq=4, and seq=5 past cutoff=4.
        let sidecar = [
            header_json(Some(10), 1, true),
            demand_json(1, hash_a, dxil_w1, 0, 0, 1, &[], 1, true),
            demand_json(2, hash_a, dxil_w1, 0, 0, 1, &[], 1, true),
            demand_json(3, hash_a, dxil_w4, 1, 0, 1, &[], 4, true),
            demand_json(4, hash_missing, b"missing-dxil", 0, 0, 1, &[], 1, true),
            demand_json(5, hash_a, b"past-cutoff", 0, 0, 1, &[], 1, true),
        ]
        .join("\n");

        let parsed = parse_capture(&sidecar, Some(4)).expect("parse sidecar");
        assert_eq!(parsed.effective_cutoff, Some(4));
        assert_eq!(parsed.dropped_by_cutoff, 1);
        assert_eq!(parsed.duplicate_rows_skipped, 1);
        assert_eq!(parsed.demands.len(), 3);

        let mut sources = HashMap::new();
        sources.insert(hash_a, ("a.bin".into(), blob_a));
        let plan = plan_replay(&parsed, &sources, &sample_env(), |_blob, _split, windows| {
            let bytes: &[u8] = if windows == 1 { dxil_w1 } else { dxil_w4 };
            Ok(make_compiled(bytes, &[], 2))
        });

        assert_eq!(plan.ready.len(), 2, "both distinct final identities for hash_a must stay separate");
        assert_eq!(plan.ready[0].seq, 1);
        assert_eq!(plan.ready[0].effective_flags, 0);
        assert_eq!(plan.ready[1].seq, 3);
        assert_eq!(plan.ready[1].effective_flags, 1);
        assert_eq!(plan.issues.len(), 1);
        assert_eq!(plan.issues[0].kind, IssueKind::MissingRecord);
        assert_eq!(plan.issues[0].seq, 4);
        assert!(!ExecutionCounts::default().is_complete(plan.ready.len(), plan.issues.len()));

        // Cutoff = 0 selects zero demands.
        let parsed0 = parse_capture(&sidecar, Some(0)).expect("cutoff 0");
        assert!(parsed0.demands.is_empty());
        assert_eq!(parsed0.dropped_by_cutoff, 5);
    }

    #[test]
    fn replay_plan_rejects_abi_flags_node_samplers_dxil_and_disk_key_collisions() {
        let blob = b"console-cs-shared".to_vec();
        let hash = shader::fnv64(&blob);
        let mut sources = HashMap::new();
        sources.insert(hash, ("shared.bin".into(), blob));

        let s_a: recomp::pso::StaticSampler = ([1, 2, 3, 4], false);
        let s_b: recomp::pso::StaticSampler = ([5, 6, 7, 8], true);

        // 1. Bad ABI version
        let cap_abi = parse_capture(&demand_json(1, hash, b"dx", 0, 0, 99, &[], 1, true), None).unwrap();
        let p_abi = plan_replay(&cap_abi, &sources, &sample_env(), |_, _, _| Ok(make_compiled(b"dx", &[], 2)));
        assert_eq!(p_abi.count_issue(IssueKind::AbiMismatch), 1);

        // 2. Non-zero node mask rejected on current main disk keys
        let cap_node = parse_capture(&demand_json(1, hash, b"dx", 0, 2, 1, &[], 1, true), None).unwrap();
        let p_node = plan_replay(&cap_node, &sources, &sample_env(), |_, _, _| Ok(make_compiled(b"dx", &[], 2)));
        assert_eq!(p_node.count_issue(IssueKind::UnsupportedDescriptor), 1);

        // 3. Explicit CLI windows / split config mismatch
        let mut env_cfg = sample_env();
        env_cfg.explicit_windows = Some(4);
        let cap_cfg = parse_capture(&demand_json(1, hash, b"dx", 0, 0, 1, &[], 1, true), None).unwrap();
        let p_cfg = plan_replay(&cap_cfg, &sources, &env_cfg, |_, _, _| Ok(make_compiled(b"dx", &[], 2)));
        assert_eq!(p_cfg.count_issue(IssueKind::ConfigMismatch), 1);

        // 4. Header translation options mismatch (direct_dxil, dxil_no_opt, x64_all_masks)
        let cap_hcfg = parse_capture(
            &format!("{}\n{}", header_json(None, 1, false), demand_json(1, hash, b"dx", 0, 0, 1, &[], 1, true)),
            None,
        )
        .unwrap();
        let p_hcfg = plan_replay(&cap_hcfg, &sources, &sample_env(), |_, _, _| Ok(make_compiled(b"dx", &[], 2)));
        assert_eq!(p_hcfg.count_issue(IssueKind::ConfigMismatch), 1);

        let mut env_no_opt = sample_env();
        env_no_opt.dxil_no_opt = parse_dxil_no_opt(Some("dce"));
        assert_eq!(env_no_opt.dxil_no_opt, vec!["dce"]);
        assert_eq!(parse_dxil_no_opt(Some("all")), vec!["cse", "dce", "merge", "phi"]);
        let cap_ok_hdr = parse_capture(
            &format!("{}\n{}", header_json(None, 1, true), demand_json(1, hash, b"dx", 0, 0, 1, &[], 1, true)),
            None,
        )
        .unwrap();
        let p_no_opt = plan_replay(&cap_ok_hdr, &sources, &env_no_opt, |_, _, _| Ok(make_compiled(b"dx", &[], 2)));
        assert_eq!(p_no_opt.count_issue(IssueKind::ConfigMismatch), 1);

        let mut env_masks = sample_env();
        env_masks.x64_all_masks = true;
        let p_masks = plan_replay(&cap_ok_hdr, &sources, &env_masks, |_, _, _| Ok(make_compiled(b"dx", &[], 2)));
        assert_eq!(p_masks.count_issue(IssueKind::ConfigMismatch), 1);

        let mut env_spin = sample_env();
        env_spin.wave64_spin = false;
        let p_spin = plan_replay(&cap_ok_hdr, &sources, &env_spin, |_, _, _| Ok(make_compiled(b"dx", &[], 2)));
        assert_eq!(p_spin.count_issue(IssueKind::ConfigMismatch), 1);

        let mut env_struct = sample_env();
        env_struct.recomp_structured = false;
        let p_struct = plan_replay(&cap_ok_hdr, &sources, &env_struct, |_, _, _| Ok(make_compiled(b"dx", &[], 2)));
        assert_eq!(p_struct.count_issue(IssueKind::ConfigMismatch), 1);

        // 5. Rebuilt shader no longer splits (FormChanged)
        let cap_form = parse_capture(&demand_json(1, hash, b"dx", 0, 0, 1, &[], 1, true), None).unwrap();
        let p_form = plan_replay(&cap_form, &sources, &sample_env(), |_, _, _| Ok(make_compiled(b"dx", &[], 0)));
        assert_eq!(p_form.count_issue(IssueKind::FormChanged), 1);

        // 6. Final whole DXIL digest mismatch
        let cap_dxil = parse_capture(&demand_json(1, hash, b"dx-old", 0, 0, 1, &[], 1, true), None).unwrap();
        let p_dxil = plan_replay(&cap_dxil, &sources, &sample_env(), |_, _, _| Ok(make_compiled(b"dx-new", &[], 2)));
        assert_eq!(p_dxil.count_issue(IssueKind::WholeDxilMismatch), 1);

        // 7. Rebuilt static samplers mismatch
        let cap_rs = parse_capture(&demand_json(1, hash, b"dx", 0, 0, 1, &[s_a], 1, true), None).unwrap();
        let p_rs = plan_replay(&cap_rs, &sources, &sample_env(), |_, _, _| Ok(make_compiled(b"dx", &[s_b], 2)));
        assert_eq!(p_rs.count_issue(IssueKind::RootSignatureMismatch), 1);

        // 8. Effective flags / disk_cache_name tamper mismatch
        let bad_disk = demand_json(1, hash, b"dx", 0, 0, 1, &[], 1, true)
            .replace(".chain.pso", ".pso");
        let cap_key = parse_capture(&bad_disk, None).unwrap();
        let p_key = plan_replay(&cap_key, &sources, &sample_env(), |_, _, _| Ok(make_compiled(b"dx", &[], 2)));
        assert_eq!(p_key.count_issue(IssueKind::CacheKeyMismatch), 1);

        // 9. Ambiguous disk key collision: two distinct static-sampler sets sharing identical (dxil, flags)
        let collision_sidecar = format!(
            "{}\n{}",
            demand_json(1, hash, b"same-dxil", 0, 0, 1, &[s_a], 1, true),
            demand_json(2, hash, b"same-dxil", 0, 0, 1, &[s_b], 4, true),
        );
        let cap_col = parse_capture(&collision_sidecar, None).unwrap();
        let p_col = plan_replay(&cap_col, &sources, &sample_env(), |_, _, windows| {
            let samps = if windows == 1 { vec![s_a] } else { vec![s_b] };
            Ok(make_compiled(b"same-dxil", &samps, 2))
        });
        assert!(p_col.ready.is_empty(), "colliding disk keys must not be scheduled for build");
        assert_eq!(p_col.count_issue(IssueKind::AmbiguousDiskKey), 2);

        // 10. Distinct source records translating to identical (dxil, flags, node_mask, samplers)
        // collapse to one first-demand build target so workers never race on one `.chain.pso` path.
        let blob_b = b"console-cs-shared-other".to_vec();
        let hash_b = shader::fnv64(&blob_b);
        assert_ne!(hash, hash_b);
        sources.insert(hash_b, ("other.bin".into(), blob_b));
        let shared_sidecar = format!(
            "{}\n{}",
            demand_json(1, hash, b"identical-dxil", 0, 0, 1, &[s_a], 1, true),
            demand_json(2, hash_b, b"identical-dxil", 0, 0, 1, &[s_a], 1, true),
        );
        let cap_shared = parse_capture(&shared_sidecar, None).unwrap();
        let p_shared = plan_replay(&cap_shared, &sources, &sample_env(), |_, _, _| {
            Ok(make_compiled(b"identical-dxil", &[s_a], 2))
        });
        assert!(p_shared.issues.is_empty(), "identical targets sharing one disk key must not fail");
        assert_eq!(p_shared.ready.len(), 1, "shared identical disk key must schedule once");
        assert_eq!(p_shared.ready[0].seq, 1, "first-demand order must be preserved");
        assert_eq!(p_shared.collapsed_shared_disk_keys, 1);

        // 11. Differing whole_dxil bytes sharing a disk_cache_name are rejected as AmbiguousDiskKey.
        let mut dxil_col_issues = Vec::new();
        let (dxil_col_ready, dxil_col_collapsed) = dedup_and_check_disk_keys(
            vec![
                PlannedItem {
                    seq: 1,
                    source_record_hash: hash,
                    whole_dxil: b"dxil-bytes-a".to_vec(),
                    samplers: vec![s_a],
                    chain_rs_id: chain_root_sig_id(&[s_a]),
                    effective_flags: 0,
                    node_mask: 0,
                    disk_cache_name: "deadbeef-c-0.chain.pso".into(),
                },
                PlannedItem {
                    seq: 2,
                    source_record_hash: hash_b,
                    whole_dxil: b"dxil-bytes-b".to_vec(),
                    samplers: vec![s_a],
                    chain_rs_id: chain_root_sig_id(&[s_a]),
                    effective_flags: 0,
                    node_mask: 0,
                    disk_cache_name: "deadbeef-c-0.chain.pso".into(),
                },
            ],
            &mut dxil_col_issues,
        );
        assert!(dxil_col_ready.is_empty());
        assert_eq!(dxil_col_collapsed, 0);
        assert_eq!(dxil_col_issues.len(), 2);
        assert!(dxil_col_issues.iter().all(|i| i.kind == IssueKind::AmbiguousDiskKey));
    }

    #[test]
    fn replay_cli_arg_parsing_accepts_replay_flags_and_rejects_orphan_cutoff() {
        let args: Vec<String> = ["--replay-chain-whole", "cap.jsonl", "--replay-cutoff", "12", "--windows", "1", "--split", "1"]
            .into_iter()
            .map(String::from)
            .collect();
        let opts = super::super::parse_args(&args).expect("parse valid replay CLI args");
        assert_eq!(opts.replay_chain_whole, Some(PathBuf::from("cap.jsonl")));
        assert_eq!(opts.replay_cutoff, Some(12));
        assert!(opts.windows_explicit);
        assert_eq!(opts.windows, 1);
        assert_eq!(opts.split, Some(true));

        let orphan: Vec<String> = ["--replay-cutoff", "5"].into_iter().map(String::from).collect();
        let err = super::super::parse_args(&orphan).err().expect("orphan cutoff must error");
        assert!(err.contains("--replay-chain-whole"), "unexpected error: {err}");
    }

    #[test]
    fn unpersisted_capped_or_disabled_cache_marks_replay_incomplete() {
        let ok_warm = ExecutionCounts { created_with_supplied_blob: 2, ..Default::default() };
        assert!(ok_warm.is_complete(2, 0));

        let ok_fresh = ExecutionCounts { created_fresh: 2, persisted: 2, ..Default::default() };
        assert!(ok_fresh.is_complete(2, 0));

        let ok_mixed = ExecutionCounts {
            created_with_supplied_blob: 1,
            created_fresh: 1,
            rejected_and_cold_retried: 1,
            persisted: 1,
            ..Default::default()
        };
        assert!(ok_mixed.is_complete(2, 0));

        // Any validation issue, cap, refusal, unavailable blob, write failure, or disabled cache -> incomplete.
        assert!(!ok_fresh.is_complete(2, 1));
        assert!(!ExecutionCounts { created_fresh: 1, persisted: 1, capped: 1, ..Default::default() }.is_complete(2, 0));
        assert!(!ExecutionCounts { created_fresh: 1, persisted: 1, refused: 1, ..Default::default() }.is_complete(2, 0));
        assert!(!ExecutionCounts { created_fresh: 1, blob_unavailable: 1, ..Default::default() }.is_complete(1, 0));
        assert!(!ExecutionCounts { created_fresh: 1, write_failed: 1, ..Default::default() }.is_complete(1, 0));
        assert!(!ExecutionCounts { cache_disabled: 1, ..Default::default() }.is_complete(1, 0));
    }

    #[test]
    fn parser_rejects_malformed_json_bad_schema_and_duplicate_header() {
        assert!(parse_capture("{not json", None).is_err());
        assert!(parse_capture(r#"{"schema_version": 99, "kind": "header"}"#, None).is_err());
        let dup_hdr = format!("{}\n{}", header_json(None, 1, true), header_json(None, 1, true));
        assert!(parse_capture(&dup_hdr, None).is_err());
        assert!(parse_capture(r#"{"schema_version": 1, "kind": "unknown"}"#, None).is_err());
    }
}
