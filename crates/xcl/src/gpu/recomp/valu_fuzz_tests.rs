//! Cross-device and cross-backend fuzzing of single VALU instructions.
//!
//! Every VOP3 / VOP3P instruction in the RDNA2 table runs alone in a wave64
//! kernel, on 64 lanes × `XCL_FUZZ_ROUNDS` (default 4) inputs: special floats
//! (signed zeros, denormals, infinities, quiet and signalling NaNs), integer
//! edges, f16 edges in either half, and pseudo-random dwords, the same on
//! every run. Each instruction is recompiled by both backends (HLSL + DXC
//! and direct DXIL), and both must agree bit for bit on this adapter (any NaN
//! matching any NaN; a few ULPs for `rcp`, `rsq`, `sqrt`, `log`, `exp`, `sin`
//! and `cos`).
//!
//! With `XCL_FUZZ_DIR` set, the results are also written to
//! `<dir>/<adapter>.tsv`. Run once per adapter (`XCL_ADAPTER=…`, `warp`
//! included), then `valu_fuzz_compare` reports every instruction whose results
//! differ between those files: a translation that leans on behaviour DXIL
//! leaves to the device, so a title computes different values on different
//! GPUs.
//!
//! Both are ignored by default:
//! `cargo test -p xcl --lib valu_fuzz -- --ignored --nocapture --test-threads=1`.
//! `XCL_FUZZ_OPS` (comma-separated name parts) limits the instructions, and
//! `XCL_FUZZ_DXC_GIS=1` compiles the HLSL backend's kernels with DXC's
//! `-Gis` (IEEE strictness), to tell a fast-math difference from a lowering one.

use super::runner::{self, KernelState};
use super::Mode;
use crate::gpu::rdna2::{self, encode};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// VGPRs the sources read (pairs for 64-bit operands) and the result's.
const SRC_VGPR: [u64; 3] = [10, 12, 14];
const DST_VGPR: usize = 20;
/// SGPRs of a mask source (`v_cndmask_b32`'s VOP3 form, carry-ins), a
/// compare's result, and a carry-out.
const SRC_MASK: usize = 30;
const DST_MASK: usize = 20;
const CARRY_OUT: usize = 24;
const LANES: usize = 64;

/// xorshift64*: the same inputs on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
    fn pick<T: Copy>(&mut self, from: &[T]) -> T {
        from[self.next() as usize % from.len()]
    }
}

const F32_EDGES: &[u32] = &[
    0x0000_0000, 0x8000_0000, 0x3F80_0000, 0xBF80_0000, 0x3F00_0000, 0x4000_0000, 0x4049_0FDB,
    0x0000_0001, 0x8000_0001, 0x007F_FFFF, 0x807F_FFFF, 0x0080_0000, 0x8080_0000, 0x7F7F_FFFF,
    0xFF7F_FFFF, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0xFFC0_0000, 0x7F80_0001, 0x7FBF_FFFF,
    0x3F7F_FFFF, 0x3F80_0001, 0x4B80_0000, 0x4F00_0000, 0x4F80_0000, 0xCF00_0000,
];
const INT_EDGES: &[u32] = &[
    0, 1, 2, 31, 32, 33, 63, 64, 0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFF, 0x1_0000, 0x7FFF_FFFF,
    0x8000_0000, 0x8000_0001, 0xFFFF_FFFE, 0xFFFF_FFFF, 0xDEAD_BEEF, 0xAAAA_AAAA, 0x5555_5555,
];
const F16_EDGES: &[u16] = &[
    0x0000, 0x8000, 0x3C00, 0xBC00, 0x3800, 0x0001, 0x8001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x7C00,
    0xFC00, 0x7E00, 0xFE00, 0x7C01, 0x3BFF, 0x3C01,
];

fn input(rng: &mut Rng) -> u32 {
    match rng.next() % 6 {
        0 | 1 => rng.pick(F32_EDGES),
        2 => rng.pick(INT_EDGES),
        3 => u32::from(rng.pick(F16_EDGES)) | u32::from(rng.pick(F16_EDGES)) << 16,
        4 => u32::from(rng.pick(F16_EDGES)) | (rng.next() & 0xFFFF_0000),
        _ => rng.next(),
    }
}

/// Instructions left out: lane, EXEC and interpolation ones, and ones with
/// no per-lane result to compare.
fn skipped(name: &str) -> bool {
    const PARTS: &[&str] = &[
        "V_CMPX", "LANE", "INTERP", "V_NOP", "V_CLREXCP", "MBCNT", "V_MOVREL", "V_SWAP",
        "PIPEFLUSH", "TRIG_PREOP", "READFIRSTLANE", "V_MOV_B64",
    ];
    PARTS.iter().any(|p| name.contains(p))
}

/// Results that differ by a few ULPs between correct implementations.
fn approximate(name: &str) -> bool {
    ["RCP", "RSQ", "SQRT", "LOG", "EXP_", "SIN", "COS"].iter().any(|p| name.contains(p))
}

fn is_f32_nan(x: u32) -> bool {
    x & 0x7FFF_FFFF > 0x7F80_0000
}

fn is_f16_nan(x: u32) -> bool {
    x & 0x7FFF > 0x7C00
}

/// Whether `a` and `b` agree as one result dword of `name`.
fn agree(name: &str, a: u32, b: u32) -> bool {
    if a == b {
        return true;
    }
    if name.starts_with("V_PK_") && name.ends_with("_F16") || name == "V_CVT_PKRTZ_F16_F32" {
        return [0, 16].iter().all(|s| {
            let (x, y) = (a >> s & 0xFFFF, b >> s & 0xFFFF);
            x == y || (is_f16_nan(x) && is_f16_nan(y))
        });
    }
    // The result's type is the first in the name (`V_CVT_F16_F32`,
    // `V_DOT2_F32_F16`).
    const TYPES: &[&str] = &["F16", "F32", "F64", "I16", "U16", "I32", "U32", "I64", "U64", "B16", "B32", "B64"];
    let half = name.split('_').find(|t| TYPES.contains(t)) == Some("F16");
    let float = name.contains("_F") || name.contains("FREXP") || name.contains("LDEXP");
    if float && !half && is_f32_nan(a) && is_f32_nan(b) {
        return true;
    }
    if half && is_f16_nan(a) && is_f16_nan(b) && a >> 16 == b >> 16 {
        return true;
    }
    if approximate(name) {
        let (a, b, ulps, sign) =
            if half { (a & 0xFFFF, b & 0xFFFF, 1, 0x8000) } else { (a, b, 4, 0x8000_0000) };
        return a & sign == b & sign && a.abs_diff(b) <= ulps;
    }
    false
}

/// How an instruction's operands are placed, and where its result is read.
struct Case {
    name: &'static str,
    code: Vec<u8>,
    /// Source dwords per lane (`SRC_VGPR` pairs).
    srcs: Vec<(usize, u16)>,
    mask_src: bool,
    /// Result: VGPR dwords (1 or 2), or a lane mask in `DST_MASK`.
    dst_dwords: usize,
    mask_dst: bool,
    carry_out: bool,
}

/// The kernel for `inst`'s VOP3 / VOP3P form, if it has one this fuzzer can
/// feed (VGPR or mask sources, a VGPR or mask result).
fn case(inst: &'static rdna2::Inst) -> Option<Case> {
    let name = inst.name;
    if !name.starts_with("V_") || skipped(name) {
        return None;
    }
    let (enc, form) = inst.encs.iter().find_map(|e| match e.encoding {
        10 => Some((e, "ENC_VOP3")),
        37 => Some((e, "VOP3_SDST_ENC")),
        11 => Some((e, "ENC_VOP3P")),
        _ => None,
    })?;
    let mut fields: Vec<(&str, u64)> = Vec::new();
    let mut srcs = Vec::new();
    let (mut mask_src, mut dst_dwords, mut mask_dst, mut carry_out) = (false, 0, false, false);
    for op in enc.operands.iter().filter(|o| !o.implicit) {
        let Some(field) = op.field else { continue };
        let wide = op.size > 32;
        match field {
            "VDST" if op.ty == "OPR_VGPR" => {
                fields.push(("VDST", DST_VGPR as u64));
                dst_dwords = if wide { 2 } else { 1 };
            }
            "VDST" if op.ty.starts_with("OPR_S") => {
                fields.push(("VDST", DST_MASK as u64));
                mask_dst = true;
            }
            "SDST" => {
                fields.push(("SDST", CARRY_OUT as u64));
                carry_out = true;
            }
            "SRC0" | "SRC1" | "SRC2" => {
                let k = usize::from(field.as_bytes()[3] - b'0');
                if op.ty == "OPR_SREG" {
                    fields.push((field, SRC_MASK as u64));
                    mask_src = true;
                } else if op.ty.starts_with("OPR_SRC") || op.ty == "OPR_VGPR" {
                    fields.push((field, 256 + SRC_VGPR[k]));
                    srcs.push((SRC_VGPR[k] as usize, op.size));
                } else {
                    return None;
                }
            }
            _ => {}
        }
    }
    if dst_dwords == 0 && !mask_dst {
        return None;
    }
    if form == "ENC_VOP3P" {
        // Both halves of every source as usual (op_sel_hi set).
        fields.push(("OP_SEL_HI", 3));
        fields.push(("OP_SEL_HI_2", 1));
    }
    let mut code = encode(name, form, &fields, None).ok()?;
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).ok()?);
    Some(Case { name, code, srcs, mask_src, dst_dwords, mask_dst, carry_out })
}

/// The state of round `r` of `c`: every lane's sources, the mask source.
fn state(c: &Case, r: u64) -> KernelState {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ r.wrapping_mul(0x1_0001) ^ fnv(c.name.as_bytes()));
    let mut s = KernelState::default();
    for lane in 0..LANES {
        for &(reg, size) in &c.srcs {
            s.vgpr[lane][reg] = input(&mut rng);
            if size > 32 {
                s.vgpr[lane][reg + 1] = input(&mut rng);
            }
        }
    }
    if c.mask_src {
        s.sgpr[SRC_MASK] = rng.next();
        s.sgpr[SRC_MASK + 1] = rng.next();
    }
    s
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ u64::from(x)).wrapping_mul(0x100_0000_01b3))
}

/// `c`'s result dwords per lane, from a finished kernel.
fn results(c: &Case, r: &runner::KernelResult) -> Vec<u32> {
    let bit = |lo: u32, hi: u32, lane: usize| ((u64::from(hi) << 32 | u64::from(lo)) >> lane & 1) as u32;
    let mut out = Vec::new();
    for lane in 0..LANES {
        if c.mask_dst {
            out.push(bit(r.sgpr[DST_MASK], r.sgpr[DST_MASK + 1], lane));
        }
        for d in 0..c.dst_dwords {
            out.push(r.vgpr[lane][DST_VGPR + d]);
        }
        if c.carry_out {
            out.push(bit(r.sgpr[CARRY_OUT], r.sgpr[CARRY_OUT + 1], lane));
        }
    }
    out
}

/// The source dwords (and mask bit) behind result `i` of `c` (see [`results`]).
fn sources(c: &Case, i: usize) -> Vec<u32> {
    let per_lane = usize::from(c.mask_dst) + c.dst_dwords + usize::from(c.carry_out);
    let (round, lane) = (i / (LANES * per_lane), i % (LANES * per_lane) / per_lane);
    let s = state(c, round as u64);
    let mut out = Vec::new();
    for &(reg, size) in &c.srcs {
        out.push(s.vgpr[lane][reg]);
        if size > 32 {
            out.push(s.vgpr[lane][reg + 1]);
        }
    }
    if c.mask_src {
        let mask = u64::from(s.sgpr[SRC_MASK + 1]) << 32 | u64::from(s.sgpr[SRC_MASK]);
        out.push((mask >> lane & 1) as u32);
    }
    out
}

fn run_hlsl(code: &[u8], s: &KernelState) -> Result<runner::KernelResult, String> {
    let t = super::translate(code, &runner::options(Mode::Kernel))?;
    if !t.complete() {
        return Err(format!("unsupported: {:?}", t.unsupported));
    }
    let extra: &[&str] = if std::env::var("XCL_FUZZ_DXC_GIS").is_ok_and(|v| v == "1") { &["-Gis"] } else { &[] };
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", extra).map_err(|e| e.to_string())?;
    runner::run_with_statics(&dxil, s, &[], &[], &[], &t.static_samplers)
}

fn run_direct(code: &[u8], s: &KernelState) -> Result<runner::KernelResult, String> {
    let k = runner::direct_kernel_full(code, true)?;
    runner::run_with_statics(&k.dxil, s, &[], &[], &[], &k.static_samplers)
}

/// The first case where `a` and `b` disagree, as (case index, a, b).
fn first_mismatch(name: &str, a: &[u32], b: &[u32]) -> Option<(usize, u32, u32)> {
    a.iter().zip(b).enumerate().find(|(_, (x, y))| !agree(name, **x, **y)).map(|(i, (x, y))| (i, *x, *y))
}

fn hex(v: &[u32]) -> String {
    v.iter().map(|x| format!("{x:08x}")).collect::<Vec<_>>().join(",")
}

#[test]
#[ignore = "fuzzing: minutes on a GPU; run with --ignored"]
fn valu_fuzz_backends_agree() {
    let adapter = match (runner::adapter(), runner::wave64_programs_supported(), super::dxc::location()) {
        (Ok(a), Ok(true), Ok(_)) => a,
        (a, w, d) => {
            eprintln!("SKIPPED: GPU/DXC unavailable ({:?} / {:?} / {:?})", a.err(), w.ok(), d.err());
            return;
        }
    };
    let rounds: u64 = std::env::var("XCL_FUZZ_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let only: Vec<String> = std::env::var("XCL_FUZZ_OPS")
        .map(|v| v.split(',').map(|s| s.trim().to_ascii_uppercase()).collect())
        .unwrap_or_default();
    let mut cases: Vec<Case> = rdna2::all_instructions()
        .filter(|i| only.is_empty() || only.iter().any(|o| i.name.contains(o.as_str())))
        .filter_map(case)
        .collect();
    cases.sort_by_key(|c| c.name);
    cases.dedup_by_key(|c| c.name);

    let mut tsv = String::new();
    let mut report = String::new();
    let (mut compared, mut refused, mut differ) = (0, Vec::new(), 0);
    for c in &cases {
        let mut cols: BTreeMap<&str, Result<Vec<u32>, String>> = BTreeMap::new();
        for (backend, run) in [("hlsl", run_hlsl as fn(&[u8], &KernelState) -> _), ("direct", run_direct)] {
            let got = (0..rounds)
                .map(|r| run(&c.code, &state(c, r)).map(|k| results(c, &k)))
                .collect::<Result<Vec<_>, _>>()
                .map(|v| v.concat());
            if let Ok(v) = &got {
                let _ = writeln!(tsv, "{}\t{backend}\t{}", c.name, hex(v));
            }
            cols.insert(backend, got);
        }
        match (&cols["hlsl"], &cols["direct"]) {
            (Ok(h), Ok(d)) => {
                compared += 1;
                if first_mismatch(c.name, h, d).is_some() {
                    differ += 1;
                    let bad: Vec<usize> = (0..h.len()).filter(|&i| !agree(c.name, h[i], d[i])).collect();
                    let _ = writeln!(
                        report,
                        "{}: {} of {} differ between HLSL and direct DXIL",
                        c.name,
                        bad.len(),
                        h.len()
                    );
                    for &i in bad.iter().take(6) {
                        let _ = writeln!(
                            report,
                            "  sources {}: hlsl {:#010x}, direct {:#010x}",
                            hex(&sources(c, i)),
                            h[i],
                            d[i]
                        );
                    }
                }
            }
            (h, d) => {
                for (b, r) in [("hlsl", h), ("direct", d)] {
                    if let Err(e) = r {
                        refused.push(format!("{} ({b}): {}", c.name, e.lines().next().unwrap_or("")));
                    }
                }
            }
        }
    }
    if let Some(dir) = std::env::var_os("XCL_FUZZ_DIR") {
        let file: String =
            adapter.chars().map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' }).collect();
        let path = std::path::Path::new(&dir).join(format!("{file}.tsv"));
        std::fs::create_dir_all(&dir).expect("fuzz dir");
        std::fs::write(&path, format!("# {adapter}\n{tsv}")).expect("write fuzz results");
        eprintln!("results: {}", path.display());
    }
    eprintln!(
        "{adapter}: {} instructions, {compared} compared on both backends, {differ} differ, {} refused",
        cases.len(),
        refused.len()
    );
    eprint!("{report}");
    for r in &refused {
        eprintln!("refused: {r}");
    }
}

#[test]
#[ignore = "reads the files valu_fuzz_backends_agree wrote under XCL_FUZZ_DIR"]
fn valu_fuzz_compare() {
    let Some(dir) = std::env::var_os("XCL_FUZZ_DIR") else {
        eprintln!("SKIPPED: XCL_FUZZ_DIR not set");
        return;
    };
    // (name, backend) -> [(device, results)]
    let mut table: BTreeMap<(String, String), Vec<(String, Vec<u32>)>> = BTreeMap::new();
    let mut devices = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read XCL_FUZZ_DIR") {
        let path = entry.expect("entry").path();
        if path.extension().is_none_or(|e| e != "tsv") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("read results");
        let mut lines = text.lines();
        let device = lines.next().unwrap_or("").trim_start_matches("# ").to_string();
        devices.push(device.clone());
        for line in lines {
            let mut parts = line.split('\t');
            let (Some(name), Some(backend), Some(vals)) = (parts.next(), parts.next(), parts.next()) else {
                continue;
            };
            let vals = vals.split(',').filter_map(|v| u32::from_str_radix(v, 16).ok()).collect();
            table.entry((name.into(), backend.into())).or_default().push((device.clone(), vals));
        }
    }
    let mut md = format!("# XCL VALU cross-device fuzz\n\nDevices: {}.\n\n", devices.join(", "));
    let mut differ = 0;
    for ((name, backend), cols) in &table {
        if cols.len() < 2 {
            continue;
        }
        let base = &cols[0].1;
        let bad: Vec<usize> = (0..base.len())
            .filter(|&i| cols.iter().any(|(_, v)| v.get(i).is_none_or(|x| !agree(name, *x, base[i]))))
            .collect();
        if bad.is_empty() {
            continue;
        }
        differ += 1;
        let _ = writeln!(md, "## `{name}` ({backend}): {} of {} results differ\n", bad.len(), base.len());
        let _ = writeln!(md, "| result | {} |\n|---|{}", cols.iter().map(|c| c.0.as_str()).collect::<Vec<_>>().join(" | "), "---|".repeat(cols.len()));
        for &i in bad.iter().take(8) {
            let row: Vec<String> = cols.iter().map(|(_, v)| v.get(i).map_or("-".into(), |x| format!("`{x:#010x}`"))).collect();
            let _ = writeln!(md, "| {i} | {} |", row.join(" | "));
        }
        md.push('\n');
    }
    let _ = writeln!(md, "{differ} instruction/backend pairs differ between devices.");
    match std::env::var_os("XCL_FUZZ_OUT") {
        Some(p) => std::fs::write(p, &md).expect("write report"),
        None => println!("{md}"),
    }
    eprintln!("{} devices, {differ} instruction/backend pairs differ", devices.len());
}
