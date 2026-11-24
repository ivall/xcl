//! Split wave64 (docs/wave64-on-wave32.md): which masks need the exchange.
//!
//! In the split form every mask write (`v_cmp*`, carries) exchanges the two halves' ballots so
//! that both halves hold the whole 64-bit mask. A mask that is only ever read lane by lane –
//! `v_cndmask_b32`'s selector, the carry-in of `v_add_co_ci` / `v_sub(rev)_co_ci`, or even-aligned
//! 64-bit lane-wise SALU ops (`s_mov_b64`, `s_and_b64`, `s_or_b64`, `s_xor_b64`, `s_not_b64`,
//! `s_andn2_b64`, `s_orn2_b64`, `s_nand_b64`, `s_nor_b64`, `s_xnor_b64`, `s_wqm_b64`,
//! `s_cselect_b64`, `s_cmov_b64`) whose destination is not whole-live and whose `scc` (if written)
//! is dead – needs just the reading lane's own bit, which its own half has. EXEC is a mask like
//! the others: every VALU reads it lane by lane, so a `v_cmpx*` or an even-aligned
//! `s_*_saveexec_b64` / `s_*_wrexec_b64` whose new EXEC (and `scc`) is not read whole needs no
//! exchange either. This is a backward liveness analysis over the SGPRs and `scc`: a register is
//! *whole-live* after an instruction when some path may read its whole value (any non-lane-wise
//! SALU / SMEM / VALU scalar operand, a branch on VCCZ / EXECZ, `v_readfirstlane` and
//! `ds_(b)permute` for EXEC, a lane-wise op whose destination or written `scc` is live, …)
//! before it is written again. A mask write whose destination is not whole-live needs no
//! exchange.
//!
//! After a mask stays in its half the two halves hold different values in its registers, so a
//! whole read that is missed here – or a write taken for a full overwrite that is not one –
//! lets them diverge. The rules follow what the emitter does, not only the ISA tables:
//! - a write counts as an overwrite only when the emitter replaces the register whatever its
//!   old value: not `s_cmov*` (conditional), `s_addk` / `s_mulk` / `s_bitset*` (read the old
//!   value), `s_movreld` / `s_movrelsd` (write `s[dst + m0]`), nor SDWA compares that write
//!   their SDST (SD = 1) instead of VCC;
//! - `s_movrels*` read `s[src + m0]`: every register from `src` up;
//! - SDWA scalar sources (S0 / S1 bits) and SMEM register offsets are reads.
//!
//! Conservative where it cannot see: an indirect jump makes every SGPR whole-live, and so is
//! every SGPR at the end of a kernel-mode program (the runner reads them back). Every emitted loop may also be left where its iteration budget
//! (`LOOP_GUARD`) is checked: what is live after a loop, or at the end, is live before each of
//! its instructions.

use std::collections::BTreeMap;

use super::emit::ext_encoding;
use crate::gpu::rdna2::{self, Decoded, Variant, F_CONDITIONAL, F_INDIRECT, F_TERMINATOR};

/// SGPR set: bit n = s<n> (106/107 VCC, 124 M0, 126/127 EXEC).
pub type Regs = u128;

const VCC: Regs = 3 << 106;
const M0: Regs = 1 << 124;
const EXEC: Regs = 3 << 126;
const ALL: Regs = !0;

fn range(first: u64, n: u64) -> Regs {
    (first..(first + n).min(128)).fold(0, |m, r| m | 1u128 << r)
}

/// SGPRs a scalar source code names (`n` dwords from it): s0–s105, VCC, M0, EXEC; src_vccz /
/// src_execz are derived from the whole VCC / EXEC; NULL, constants and literals name none.
fn scalar_code(v: u64, n: u64) -> Regs {
    match v {
        251 => VCC,
        252 => EXEC,
        125 => 0,
        v if v <= 127 => range(v, n),
        _ => 0,
    }
}

/// Instructions whose mask operand is read lane by lane (the emitter's `lane_of`).
fn reads_mask_per_lane(name: &str) -> bool {
    matches!(name, "V_CNDMASK_B32" | "V_ADD_CO_CI_U32" | "V_SUB_CO_CI_U32" | "V_SUBREV_CO_CI_U32")
}

/// 64-bit SALU ops whose low and high dwords depend only on the corresponding dwords of their
/// inputs (no bit crosses the 32-lane half boundary): `(is_lane_wise, sets_scc)`.
/// When neither their destination nor (if `sets_scc`) `scc` is whole-live afterward, each half
/// can evaluate the op on its own 32-lane half word without exchanging the inputs.
fn lane_wise_salu64(name: &str) -> Option<bool> {
    match name {
        "S_MOV_B64" | "S_CMOV_B64" | "S_CSELECT_B64" => Some(false),
        "S_NOT_B64"
        | "S_WQM_B64"
        | "S_AND_B64"
        | "S_OR_B64"
        | "S_XOR_B64"
        | "S_ANDN2_B64"
        | "S_ORN2_B64"
        | "S_NAND_B64"
        | "S_NOR_B64"
        | "S_XNOR_B64" => Some(true),
        _ => None,
    }
}

fn is_cmp_suffix(s: &str) -> bool {
    matches!(
        s,
        "EQ_I32" | "LG_I32" | "GT_I32" | "GE_I32" | "LT_I32" | "LE_I32" | "EQ_U32" | "LG_U32" | "GT_U32" | "GE_U32" | "LT_U32" | "LE_U32"
    )
}

/// Whether `emit.rs` unconditionally overwrites `scc` for `d` (conservative: false whenever unsure).
fn writes_scc(d: &Decoded) -> bool {
    let n = d.inst.name;
    match d.enc.name {
        "ENC_SOPK" => n == "S_ADDK_I32" || n.strip_prefix("S_CMPK_").is_some_and(is_cmp_suffix),
        "ENC_SOPC" => {
            matches!(n, "S_CMP_EQ_U64" | "S_CMP_LG_U64" | "S_BITCMP0_B32" | "S_BITCMP1_B32" | "S_BITCMP0_B64" | "S_BITCMP1_B64")
                || n.strip_prefix("S_CMP_").is_some_and(is_cmp_suffix)
        }
        "ENC_SOP1" | "ENC_SOP2" => {
            if let Some(op) = n.strip_prefix("S_").and_then(|s| s.strip_suffix("_B32").or_else(|| s.strip_suffix("_B64"))) {
                if let Some(kind) = op.strip_suffix("_SAVEEXEC").or_else(|| op.strip_suffix("_WREXEC")) {
                    return matches!(kind, "AND" | "OR" | "XOR" | "ANDN2" | "ORN2" | "NAND" | "NOR" | "XNOR" | "ANDN1" | "ORN1");
                }
            }
            matches!(
                n,
                "S_NOT_B32"
                    | "S_BCNT0_I32_B32"
                    | "S_BCNT1_I32_B32"
                    | "S_ABS_I32"
                    | "S_AND_B32"
                    | "S_OR_B32"
                    | "S_XOR_B32"
                    | "S_ANDN2_B32"
                    | "S_ORN2_B32"
                    | "S_NAND_B32"
                    | "S_NOR_B32"
                    | "S_XNOR_B32"
                    | "S_ADD_U32"
                    | "S_SUB_U32"
                    | "S_ADDC_U32"
                    | "S_SUBB_U32"
                    | "S_ADD_I32"
                    | "S_SUB_I32"
                    | "S_LSHL_B32"
                    | "S_LSHR_B32"
                    | "S_ASHR_I32"
                    | "S_LSHL1_ADD_U32"
                    | "S_LSHL2_ADD_U32"
                    | "S_LSHL3_ADD_U32"
                    | "S_LSHL4_ADD_U32"
                    | "S_BFE_U32"
                    | "S_BFE_I32"
                    | "S_MIN_U32"
                    | "S_MAX_U32"
                    | "S_MIN_I32"
                    | "S_MAX_I32"
                    | "S_WQM_B32"
                    | "S_ABSDIFF_I32"
                    | "S_NOT_B64"
                    | "S_WQM_B64"
                    | "S_BCNT1_I32_B64"
                    | "S_BCNT0_I32_B64"
                    | "S_LSHL_B64"
                    | "S_LSHR_B64"
                    | "S_ASHR_I64"
                    | "S_BFE_U64"
                    | "S_BFE_I64"
                    | "S_AND_B64"
                    | "S_OR_B64"
                    | "S_XOR_B64"
                    | "S_ANDN2_B64"
                    | "S_ORN2_B64"
                    | "S_NAND_B64"
                    | "S_NOR_B64"
                    | "S_XNOR_B64"
            )
        }
        _ => false,
    }
}

/// Whether `d` may read `scc` (either implicitly or via scalar source code 253).
fn reads_scc(d: &Decoded) -> bool {
    let n = d.inst.name;
    if matches!(n, "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1" | "S_CSELECT_B32" | "S_CSELECT_B64" | "S_ADDC_U32" | "S_SUBB_U32")
        || n.starts_with("S_CMOV")
    {
        return true;
    }
    if d.variant == Variant::Sdwa {
        let Some(e) = ext_encoding(d) else { return true };
        let fv = |f: &str| rdna2::field_value(e, &d.words, f).unwrap_or(0);
        if (fv("S0") == 1 && fv("VSRC0") == 253) || (d.enc.field("VSRC1").is_some() && fv("S1") == 1 && fv("VSRC1") == 253) {
            return true;
        }
    }
    for o in d.ienc.operands {
        if o.output {
            continue;
        }
        if o.ty == "OPR_SSRC_SPECIAL_SCC" {
            return true;
        }
        if o.implicit {
            continue;
        }
        let Some(f) = o.field else { continue };
        if f == "LITERAL" || (f == "SRC0" && d.variant != Variant::Plain) {
            continue;
        }
        if !["SREG", "SSRC", "SDST", "OPR_SRC", "SMEM_OFFSET"].iter().any(|k| o.ty.contains(k)) {
            continue;
        }
        if rdna2::register_scale(f) == 1 && rdna2::field_value(d.enc, &d.words, f) == Some(253) {
            return true;
        }
    }
    false
}

/// Instructions that keep (part of) their destination's old value: the destination is read
/// and not overwritten.
fn keeps_destination(name: &str) -> bool {
    name.starts_with("S_CMOV") || matches!(name, "S_ADDK_I32" | "S_MULK_I32" | "S_BITSET0_B32" | "S_BITSET1_B32" | "S_BITSET0_B64" | "S_BITSET1_B64")
}

struct InstFlow {
    whole: Regs,
    kill: Regs,
    /// Even-aligned 64-bit lane-wise SALU op: `(dst, lane_srcs, sets_scc)`.
    lane_op: Option<(Regs, Regs, bool)>,
    /// Even-aligned `s_*_saveexec_b64` / `s_*_wrexec_b64`: `(dst, lane_src, is_wrexec)`.
    save_exec: Option<(Regs, Regs, bool)>,
    reads_scc: bool,
    writes_scc: bool,
}

/// `s_*_saveexec_b64` / `s_*_wrexec_b64` with even-aligned register operands:
/// `((dst, lane_src, is_wrexec), read whole whatever is live)`. The new EXEC (and `scc`) is
/// `op(src, EXEC)` lane by lane; `dst` gets the old EXEC, or the new one for `wrexec`.
fn save_exec_b64(d: &Decoded) -> Option<((Regs, Regs, bool), Regs)> {
    let name = d.inst.name;
    let wrexec = name.contains("_WREXEC");
    if d.variant != Variant::Plain || !name.ends_with("_B64") || !(wrexec || name.contains("_SAVEEXEC")) {
        return None;
    }
    let dst = match rdna2::field_value(d.enc, &d.words, "SDST")? {
        125 => 0,
        v if v < 126 && v & 1 == 0 => range(v, 2),
        _ => return None,
    };
    let (src, whole) = match rdna2::field_value(d.enc, &d.words, "SSRC0")? {
        251 => (0, VCC),
        252 => (0, EXEC),
        125 => (0, 0),
        v if v <= 126 && v & 1 == 0 => (range(v, 2), 0),
        v if v <= 127 => return None,
        _ => (0, 0),
    };
    Some(((dst, src, wrexec), whole))
}

fn flow(d: &Decoded) -> InstFlow {
    let (whole, kill) = uses(d);
    let r_scc = reads_scc(d);
    let w_scc = writes_scc(d);
    if let Some((se, forced)) = save_exec_b64(d) {
        return InstFlow { whole: forced, kill: se.0 | EXEC, lane_op: None, save_exec: Some(se), reads_scc: r_scc, writes_scc: w_scc };
    }
    let Some(sets_scc) = lane_wise_salu64(d.inst.name) else {
        return InstFlow { whole, kill, lane_op: None, save_exec: None, reads_scc: r_scc, writes_scc: w_scc };
    };
    if d.variant != Variant::Plain {
        return InstFlow { whole, kill, lane_op: None, save_exec: None, reads_scc: r_scc, writes_scc: w_scc };
    }
    // Verify even SGPR pair alignment on dst and all register sources so low/high half words can
    // never cross-alias; scalar special codes `vccz` (251) / `execz` (252) always read whole.
    let keeps = keeps_destination(d.inst.name);
    let (mut dst_regs, mut lane_srcs, mut force_whole) = (0 as Regs, 0 as Regs, 0 as Regs);
    for o in d.ienc.operands {
        let Some(f) = o.field else { continue };
        if o.implicit || f == "LITERAL" {
            continue;
        }
        if !["SREG", "SSRC", "SDST", "OPR_SRC"].iter().any(|k| o.ty.contains(k)) {
            continue;
        }
        let code = rdna2::field_value(d.enc, &d.words, f).unwrap_or(0) * rdna2::register_scale(f);
        let n = (o.size as u64).div_ceil(32).max(1);
        if o.output {
            if code == 125 {
                continue;
            }
            if code > 126 || code & 1 != 0 || n != 2 {
                return InstFlow { whole, kill, lane_op: None, save_exec: None, reads_scc: r_scc, writes_scc: w_scc };
            }
            let r = range(code, 2);
            dst_regs |= r;
            if keeps {
                lane_srcs |= r;
            }
        } else {
            match code {
                251 => force_whole |= VCC,
                252 => force_whole |= EXEC,
                125 => {}
                c if c <= 127 => {
                    if c > 126 || c & 1 != 0 || n != 2 {
                        return InstFlow { whole, kill, lane_op: None, save_exec: None, reads_scc: r_scc, writes_scc: w_scc };
                    }
                    lane_srcs |= range(c, 2);
                }
                _ => {}
            }
        }
    }
    InstFlow { whole: force_whole, kill, lane_op: Some((dst_regs, lane_srcs, sets_scc)), save_exec: None, reads_scc: r_scc, writes_scc: w_scc }
}

/// (read whole, overwritten) SGPRs of one instruction. Lane-by-lane mask reads are left out
/// of the first, writes that keep anything of the old value out of the second.
pub fn uses(d: &Decoded) -> (Regs, Regs) {
    let name = d.inst.name;
    if name.starts_with("S_MOVREL") {
        // `smovrel`: s[src + m0] is read, s[dst + m0] written (s_movrelsd_2: both) – any
        // register from the encoded one up to s123, VCC included; nothing for certain
        // overwritten
        let src = rdna2::field_value(d.enc, &d.words, "SSRC0").unwrap_or(0).min(124);
        return (range(src, 124 - src) | M0 | EXEC, 0);
    }
    let (mut whole, mut written) = (0 as Regs, 0 as Regs);
    let per_lane = reads_mask_per_lane(name);
    let keeps = keeps_destination(name);
    // SDWA: scalar sources and the compare's destination are in the extension dword
    let mut sdwa_dst = None;
    if d.variant == Variant::Sdwa {
        let Some(e) = ext_encoding(d) else { return (ALL, 0) };
        let fv = |n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
        if fv("S0") == 1 {
            whole |= scalar_code(fv("VSRC0"), 1);
        }
        if d.enc.field("VSRC1").is_some() && fv("S1") == 1 {
            whole |= scalar_code(fv("VSRC1"), 1);
        }
        if d.enc.name == "ENC_VOPC" && !name.starts_with("V_CMPX") {
            sdwa_dst = Some(if fv("SD") == 1 { scalar_code(fv("SDST"), 2) } else { VCC });
        }
    }
    for o in d.ienc.operands {
        let n = (o.size as u64).div_ceil(32).max(1);
        let regs = match (o.field, o.ty) {
            (_, "OPR_VCC") => VCC,
            (_, "OPR_EXEC" | "OPR_SDST_EXEC") => EXEC,
            (_, "OPR_SDST_M0") => M0,
            (None, _) | (Some("LITERAL"), _) => continue,
            (Some(f), ty) => {
                if o.implicit || !["SREG", "SSRC", "SDST", "OPR_SRC", "SMEM_OFFSET"].iter().any(|k| ty.contains(k)) {
                    continue;
                }
                // SDWA / DPP src0 is in the extension dword (above for SDWA; a VGPR for DPP)
                if f == "SRC0" && d.variant != Variant::Plain {
                    continue;
                }
                scalar_code(rdna2::field_value(d.enc, &d.words, f).unwrap_or(0) * rdna2::register_scale(f), n)
            }
        };
        if o.output {
            if keeps {
                whole |= regs;
            } else {
                written |= regs;
            }
        } else if !(per_lane && (o.ty == "OPR_VCC" || o.field == Some("SRC2"))) {
            whole |= regs;
        }
    }
    if let Some(dst) = sdwa_dst {
        written = (written & !VCC) | dst;
    }
    // EXEC reads the operand tables leave out: the first active lane, lanes of the other half
    if matches!(name, "V_READFIRSTLANE_B32" | "DS_BPERMUTE_B32" | "DS_PERMUTE_B32") {
        whole |= EXEC;
    }
    (whole, written)
}

/// For each instruction (index in `insts`): the SGPRs whose whole value may be read after it.
/// `end_reads_all`: the program's end reads every SGPR (kernel mode).
pub fn whole_live_out(insts: &[Decoded], end_reads_all: bool) -> Vec<Regs> {
    let pcs: BTreeMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
    let at_end = if end_reads_all { ALL } else { 0 };
    // successors; None = unknown (indirect): everything is live
    let succ: Vec<Option<Vec<usize>>> = insts
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let next = insts.get(i + 1).map(|n| n.pc).and_then(|pc| pcs.get(&pc).copied());
            // a far jump (`rdna2::resolve_far_jumps`) is an unconditional branch
            if let Some(t) = d.far_target {
                return Some(vec![*pcs.get(&t)?]);
            }
            if d.inst.flags & F_INDIRECT != 0 || d.inst.name == "S_SETPC_B64" || d.inst.name == "S_SWAPPC_B64" {
                return None;
            }
            let mut out = Vec::new();
            if let Some(t) = rdna2::branch_target(d) {
                out.push(*pcs.get(&t)?);
                if d.inst.flags & F_CONDITIONAL != 0 {
                    out.extend(next);
                }
            } else if d.inst.flags & F_TERMINATOR == 0 {
                out.extend(next);
            }
            Some(out)
        })
        .collect();
    // `LOOP_GUARD`: a loop may be left where its guard runs – at its top, before the head
    // instruction, and in the `pc` dispatcher form before every block of it – to the code
    // after it (structured) or the end of the program (dispatcher: no later block matches
    // the `pc` it leaves with). Taken as: before any instruction of a loop, whatever is live
    // after the loop or at the end may be read.
    let loops = super::loop_ranges(insts);
    let exits: Vec<Vec<Option<usize>>> = insts
        .iter()
        .map(|d| loops.iter().filter(|&&(a, b)| a <= d.pc && d.pc < b).map(|&(_, b)| pcs.range(b..).next().map(|(_, &j)| j)).collect())
        .collect();
    let flows: Vec<InstFlow> = insts.iter().map(flow).collect();
    let scc_at_end = end_reads_all;
    let mut live_in = vec![0 as Regs; insts.len()];
    let mut live_out = vec![0 as Regs; insts.len()];
    let mut scc_in = vec![false; insts.len()];
    let mut scc_out = vec![false; insts.len()];
    loop {
        let mut changed = false;
        for i in (0..insts.len()).rev() {
            let (out, sout) = match &succ[i] {
                None => (ALL, true),
                Some(s) if s.is_empty() => (at_end, scc_at_end),
                Some(s) => s.iter().fold((0, false), |(m, sc), &j| (m | live_in[j], sc || scc_in[j])),
            };
            let (left, sleft) = exits[i].iter().fold((0 as Regs, false), |(m, sc), e| {
                (m | at_end | e.map_or(0, |j| live_in[j]), sc || scc_at_end || e.is_some_and(|j| scc_in[j]))
            });
            let f = &flows[i];
            let mut gen = f.whole;
            if let Some((dst, srcs, sets_scc)) = f.lane_op {
                // Lane-wise 64-bit SALU op needs whole inputs iff its destination is whole-live
                // afterward, or it writes `scc` and `scc` is live afterward.
                if (out & dst) != 0 || (sets_scc && sout) {
                    gen |= srcs;
                }
            }
            if let Some((dst, src, wrexec)) = f.save_exec {
                // the new EXEC (or `scc`, derived from it) read whole needs both inputs whole; the
                // saved old EXEC read whole needs EXEC whole (the new one for `wrexec`)
                let (exec_w, dst_w) = ((out & EXEC) != 0 || sout, (out & dst) != 0);
                if exec_w || dst_w {
                    gen |= EXEC;
                }
                if exec_w || (wrexec && dst_w) {
                    gen |= src;
                }
            }
            let inn = gen | (out & !f.kill) | left;
            let sinn = f.reads_scc || (sout && !f.writes_scc) || sleft;
            if out != live_out[i] || inn != live_in[i] || sout != scc_out[i] || sinn != scc_in[i] {
                live_out[i] = out;
                live_in[i] = inn;
                scc_out[i] = sout;
                scc_in[i] = sinn;
                changed = true;
            }
        }
        if !changed {
            return live_out;
        }
    }
}

/// pc → SGPRs whole-live after that instruction, for the emitter.
pub fn by_pc(insts: &[Decoded], end_reads_all: bool) -> BTreeMap<u64, Regs> {
    insts.iter().map(|d| d.pc).zip(whole_live_out(insts, end_reads_all)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(bytes: &[u8]) -> Vec<Decoded> {
        let mut out = Vec::new();
        let mut off = 0usize;
        while off < bytes.len() {
            let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
            off += d.len;
            out.push(d);
        }
        out
    }

    fn mask_needed(live: &[Regs], i: usize, pair: u64) -> bool {
        live[i] & range(pair, 2) != 0
    }

    const V: u64 = 256;
    type I = (&'static str, &'static str, &'static [(&'static str, u64)]);
    const ENDPGM: I = ("S_ENDPGM", "ENC_SOPP", &[("SIMM16", 0)]);
    const CMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]);
    const CMP_LT: I = ("V_CMP_LT_U32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]);
    const CNDMASK: I = ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 3), ("VSRC1", 4)]);
    const MOV_S2_VCC: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 106)]);

    fn asm(lines: &[I]) -> Vec<Decoded> {
        let bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
        program(&bytes)
    }

    #[test]
    fn a_selector_only_mask_needs_no_exchange() {
        let p = asm(&[CMP, CNDMASK, ENDPGM]);
        assert!(!mask_needed(&whole_live_out(&p, false), 0, 106));
        // the kernel runner reads VCC at the end: still exchanged there
        assert!(mask_needed(&whole_live_out(&p, true), 0, 106));
    }

    #[test]
    fn a_carry_chain_needs_no_exchange() {
        let p = asm(&[
            ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 0), ("SDST", 106), ("SRC0", V), ("SRC1", V + 2)]),
            ("V_ADD_CO_CI_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V + 1), ("VSRC1", 3)]),
            ENDPGM,
        ]);
        let live = whole_live_out(&p, false);
        assert!(!mask_needed(&live, 0, 106));
        assert!(!mask_needed(&live, 1, 106));
    }

    #[test]
    fn scalar_reads_and_branches_keep_the_exchange() {
        let readers: [I; 3] = [
            ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 0)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 106)]),
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)]),
        ];
        for reader in readers {
            let p = asm(&[CMP, reader, ENDPGM]);
            assert!(mask_needed(&whole_live_out(&p, false), 0, 106), "{}", reader.0);
        }
        // s_and_saveexec_b64 keeps the exchange when the new EXEC is read whole
        let p = asm(&[CMP, ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]), CBRANCH_EXECZ, ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
        // s_mov_b64 keeps the exchange when its destination is read whole
        let p = asm(&[CMP, MOV_S2_VCC, ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 2)]), ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
    }

    #[test]
    fn a_whole_read_on_one_path_or_around_a_loop_keeps_the_exchange() {
        // v_cmp; s_cbranch_scc0 → skip; v_cndmask; s_branch → end; skip: s_bcnt1 vcc; end
        let p = asm(&[
            CMP,
            ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 2)]),
            CNDMASK,
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 1)]),
            BCNT_VCC,
            ENDPGM,
        ]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
        // loop: the mask written at the bottom is read whole at the top of the next iteration
        let p = asm(&[BCNT_VCC, CMP, ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)]), ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 1, 106));
    }

    #[test]
    fn overwritten_before_any_whole_read() {
        let p = asm(&[CMP, CNDMASK, CMP_LT, BCNT_VCC, ENDPGM]);
        let live = whole_live_out(&p, false);
        assert!(!mask_needed(&live, 0, 106));
        assert!(mask_needed(&live, 2, 106));
    }

    #[test]
    fn exec_is_exchanged_only_when_read_whole() {
        let cmpx: I = ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]);
        // v_cmpx; VALU; EXEC restored before s_cbranch_execz: the v_cmpx's EXEC is read lane by lane
        let p = asm(&[
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 126)]),
            cmpx,
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 3), ("VSRC1", 4)]),
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 10)]),
            CBRANCH_EXECZ,
            ENDPGM,
        ]);
        let live = whole_live_out(&p, false);
        assert!(!mask_needed(&live, 1, 126));
        assert!(mask_needed(&live, 0, 10), "the saved EXEC is restored and then read whole");
        // the kernel runner reads EXEC at the end
        let p = asm(&[cmpx, ENDPGM]);
        assert!(!mask_needed(&whole_live_out(&p, false), 0, 126));
        assert!(mask_needed(&whole_live_out(&p, true), 0, 126));
        for reader in [
            CBRANCH_EXECZ,
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)]),
            ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 2), ("ADDR", 0), ("DATA0", 1)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 252)]),
            ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 126)]),
        ] {
            let p = asm(&[cmpx, reader, ENDPGM]);
            assert!(mask_needed(&whole_live_out(&p, false), 0, 126), "{}", reader.0);
        }
    }

    #[test]
    fn saveexec_needs_whole_inputs_only_for_a_whole_read() {
        let saveexec: I = ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 20)]);
        let restore: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 10)]);
        let valu: I = ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 3), ("VSRC1", 4)]);
        let bcnt_s10: I = ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 10)]);
        // predicated region, EXEC restored before it is read whole
        let p = asm(&[CMP_S20, saveexec, valu, restore, CBRANCH_EXECZ, ENDPGM]);
        assert!(!mask_needed(&whole_live_out(&p, false), 0, 20));
        // the new EXEC read whole (branch), or its `scc`
        let p = asm(&[CMP_S20, saveexec, CBRANCH_EXECZ, valu, restore, ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 20));
        let p = asm(&[CMP_S20, saveexec, ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 0)]), restore, ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 20));
        // the saved value read whole: the old EXEC for saveexec, the new one for wrexec
        let p = asm(&[CMP_S20, saveexec, valu, bcnt_s10, ENDPGM]);
        assert!(!mask_needed(&whole_live_out(&p, false), 0, 20));
        let p = asm(&[CMP_S20, ("S_ANDN2_WREXEC_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 20)]), valu, bcnt_s10, ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 20));
        // an odd-aligned source pair is read whole
        let p = asm(&[CMP_S20, ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 21)]), valu, restore, ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 20));
    }

    const CMP_S20: I = ("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V), ("SRC1", V + 1)]);
    const BCNT_S20: I = ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 20)]);
    const BCNT_VCC: I = ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]);
    const CBRANCH_EXECZ: I = ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 0)]);

    // review findings: writes that are not full overwrites, reads the operand tables miss

    #[test]
    fn an_sdwa_compare_into_its_sdst_leaves_vcc() {
        let sdwa = |sd: u64| -> I {
            let f: &'static [(&'static str, u64)] = if sd == 1 {
                &[("SRC0", 0xF9), ("VSRC0", 0), ("VSRC1", 0), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("SD", 1), ("SDST", 22)]
            } else {
                &[("SRC0", 0xF9), ("VSRC0", 0), ("VSRC1", 0), ("SRC0_SEL", 6), ("SRC1_SEL", 6)]
            };
            ("V_CMP_EQ_U32", "VOPC_VOP_SDWA_SDST_ENC", f)
        };
        let p = asm(&[CMP, sdwa(1), BCNT_VCC, ENDPGM]);
        assert_eq!(uses(&p[1]).1, range(22, 2), "SD = 1 writes s[22:23], not VCC");
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
        // SD = 0: VCC is written, the first compare's mask is dead
        let p = asm(&[CMP, sdwa(0), BCNT_VCC, ENDPGM]);
        assert_eq!(uses(&p[1]).1, VCC);
        let live = whole_live_out(&p, false);
        assert!(!mask_needed(&live, 0, 106) && mask_needed(&live, 1, 106));
    }

    #[test]
    fn read_modify_write_destinations_are_reads() {
        let cases: [&[I]; 4] = [
            &[("S_BITSET1_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 191)])],
            &[("S_BITSET0_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 129)]), ("S_BITSET0_B32", "ENC_SOP1", &[("SDST", 21), ("SSRC0", 129)])],
            &[("S_ADDK_I32", "ENC_SOPK", &[("SDST", 20), ("SIMM16", 1)]), ("S_MULK_I32", "ENC_SOPK", &[("SDST", 21), ("SIMM16", 3)])],
            &[("S_CMOVK_I32", "ENC_SOPK", &[("SDST", 20), ("SIMM16", 1)]), ("S_CMOV_B32", "ENC_SOP1", &[("SDST", 21), ("SSRC0", 129)])],
        ];
        for rmw in cases {
            let mut l = vec![CMP_S20];
            l.extend_from_slice(rmw);
            l.extend([BCNT_S20, ENDPGM]);
            assert!(mask_needed(&whole_live_out(&asm(&l), false), 0, 20), "{}", rmw[0].0);
        }
    }

    #[test]
    fn scalar_relative_moves_reach_any_register_from_their_base() {
        // s_movrels_b64 s[0:1], s[100 + m0 : 101 + m0] – VCC with m0 = 6
        let p = asm(&[CMP, ("S_MOVRELS_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 100)]), ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 128)]), ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 0)]), ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
        // s_movreld_b64 s[20 + m0 : 21 + m0]: s[20:21] itself only when m0 = 0
        let p = asm(&[CMP_S20, ("S_MOVRELD_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 0)]), BCNT_S20, ENDPGM]);
        assert_eq!(uses(&p[1]).1, 0);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 20));
    }

    #[test]
    fn smem_register_offsets_and_sdwa_scalar_sources_are_reads() {
        let readers: [I; 3] = [
            ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 0), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 20)]),
            ("V_ADD_NC_U32", "VOP2_VOP_SDWA", &[("VDST", 11), ("SRC0", 0xF9), ("VSRC0", 1), ("VSRC1", 20), ("S1", 1), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("DST_SEL", 6)]),
            ("V_ADD_NC_U32", "VOP2_VOP_SDWA", &[("VDST", 11), ("SRC0", 0xF9), ("VSRC0", 20), ("S0", 1), ("VSRC1", 2), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("DST_SEL", 6)]),
        ];
        for r in readers {
            let p = asm(&[CMP_S20, r, ENDPGM]);
            assert!(uses(&p[1]).0 & range(20, 1) != 0, "{} {}", r.0, r.1);
            assert!(mask_needed(&whole_live_out(&p, false), 0, 20), "{} {}", r.0, r.1);
        }
        // a VGPR src1 is not
        let p = asm(&[CMP_S20, ("V_ADD_NC_U32", "VOP2_VOP_SDWA", &[("VDST", 11), ("SRC0", 0xF9), ("VSRC0", 1), ("VSRC1", 20), ("SRC0_SEL", 6), ("SRC1_SEL", 6), ("DST_SEL", 6)]), ENDPGM]);
        assert!(!mask_needed(&whole_live_out(&p, false), 0, 20));
    }

    #[test]
    fn a_selector_that_is_also_a_scalar_source_is_read_whole() {
        // v_cndmask_b32 v2, vcc_lo, v3, vcc: the selector per lane, src0 the whole VCC_LO
        let p = asm(&[CMP, ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", 106), ("SRC1", V + 3), ("SRC2", 106)]), ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
        let p = asm(&[CMP, ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 4), ("SRC1", V + 3), ("SRC2", 106)]), ENDPGM]);
        assert!(!mask_needed(&whole_live_out(&p, false), 0, 106));
    }

    #[test]
    fn a_far_jump_is_a_branch_not_an_unknown_jump() {
        // v_cmp vcc; s_getpc s[30:31]; s_add_u32 s30, s30, 20; s_addc_u32 s31, s31, 0;
        // s_setpc s[30:31] → +28 (past the s_mov that reads VCC whole); +28: v_cndmask; end
        let mut bytes = Vec::new();
        for (n, e, f, l) in [
            (CMP.0, CMP.1, CMP.2, None),
            ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 30)][..], None),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 30), ("SSRC0", 30), ("SSRC1", 255)][..], Some(20)),
            ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 31), ("SSRC0", 31), ("SSRC1", 128)][..], None),
            ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 30)][..], None),
            (BCNT_VCC.0, BCNT_VCC.1, BCNT_VCC.2, None),
            (CNDMASK.0, CNDMASK.1, CNDMASK.2, None),
            (ENDPGM.0, ENDPGM.1, ENDPGM.2, None),
        ] {
            bytes.extend(rdna2::encode(n, e, f, l).unwrap());
        }
        let mut p = program(&bytes);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106), "unresolved: an indirect jump");
        rdna2::resolve_far_jumps(&mut p);
        assert_eq!(p[4].far_target, Some(p[6].pc));
        assert!(!mask_needed(&whole_live_out(&p, false), 0, 106));
    }

    #[test]
    fn a_loop_may_be_left_at_its_top() {
        // loop: s_mov_b64 vcc, s[4:5]; v_cmp vcc; v_cndmask; s_branch loop – then s_bcnt1 vcc:
        // only the emitter's iteration budget leaves the loop, at its top, with the v_cmp's VCC
        let p = asm(&[
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 4)]),
            CMP,
            CNDMASK,
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFFC)]),
            BCNT_VCC,
            ENDPGM,
        ]);
        assert!(mask_needed(&whole_live_out(&p, false), 1, 106));
    }

    #[test]
    fn a_loop_left_before_its_head_overwrites() {
        // v_cmp vcc; v_cndmask; loop: s_mov_b64 vcc, s[4:5]; v_cndmask; s_branch loop – then
        // s_bcnt1 vcc: a budget already spent leaves the loop before its s_mov, with the
        // first v_cmp's VCC
        let p = asm(&[CMP, CNDMASK, ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 4)]), CNDMASK, ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFFD)]), BCNT_VCC, ENDPGM]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
    }

    #[test]
    fn a_dispatcher_loop_left_by_its_budget_ends_the_program() {
        // loop: v_cmp vcc; v_cndmask; s_mov_b64 vcc, s[4:5]; s_branch loop – then s_mov_b64
        // vcc, 0: structured, the loop's exit overwrites VCC; in the `pc` dispatcher form the
        // budget's break ends the program, whose end the kernel runner reads
        let p = asm(&[
            CMP,
            CNDMASK,
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 4)]),
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFFC)]),
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 128)]),
            ENDPGM,
        ]);
        assert!(mask_needed(&whole_live_out(&p, true), 0, 106));
        assert!(!mask_needed(&whole_live_out(&p, false), 0, 106));
    }

    #[test]
    fn s_movrelsd_2_reads_from_its_base_and_overwrites_nothing() {
        let mov = ("S_MOVRELSD_2_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 100)][..]);
        let p = asm(&[CMP_S20, mov, BCNT_S20, ENDPGM]);
        assert_eq!(uses(&p[1]).1, 0);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 20));
        // its source s[100 + (m0 & 0x3FF)] reaches VCC
        let p = asm(&[CMP, mov, ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 128)]), ENDPGM]);
        assert!(uses(&p[1]).0 & VCC == VCC);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
    }

    #[test]
    fn bitwise_mask_chain_for_selector_needs_no_exchange() {
        let cmp_s22: I = ("V_CMP_LT_U32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 2), ("SRC1", V + 3)]);
        let p = asm(&[
            CMP_S20,
            cmp_s22,
            ("S_AND_B64", "ENC_SOP2", &[("SDST", 24), ("SSRC0", 20), ("SSRC1", 22)]),
            ("S_NOT_B64", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 24)]),
            ("S_OR_B64", "ENC_SOP2", &[("SDST", 26), ("SSRC0", 24), ("SSRC1", 20)]),
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 26)]),
            CNDMASK,
            ENDPGM,
        ]);
        let live = whole_live_out(&p, false);
        assert!(!mask_needed(&live, 0, 20));
        assert!(!mask_needed(&live, 1, 22));
        // kernel mode reads all SGPRs and SCC at end: exchanges are kept
        let live_k = whole_live_out(&p, true);
        assert!(mask_needed(&live_k, 0, 20));
        assert!(mask_needed(&live_k, 1, 22));
    }

    #[test]
    fn bitwise_mask_with_live_scc_keeps_exchange() {
        let cmp_s22: I = ("V_CMP_LT_U32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 2), ("SRC1", V + 3)]);
        let and_vcc: I = ("S_AND_B64", "ENC_SOP2", &[("SDST", 106), ("SSRC0", 20), ("SSRC1", 22)]);
        let br_scc: I = ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 0)]);
        // SCC from s_and_b64 is live at s_cbranch_scc0 -> inputs must be exchanged
        let p = asm(&[CMP_S20, cmp_s22, and_vcc, CNDMASK, br_scc, ENDPGM]);
        let live = whole_live_out(&p, false);
        assert!(mask_needed(&live, 0, 20));
        assert!(mask_needed(&live, 1, 22));
        // Overwriting SCC before the branch makes s_and_b64's SCC dead -> no exchange needed
        let cmp_scc: I = ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)]);
        let p = asm(&[CMP_S20, cmp_s22, and_vcc, CNDMASK, cmp_scc, br_scc, ENDPGM]);
        let live = whole_live_out(&p, false);
        assert!(!mask_needed(&live, 0, 20));
        assert!(!mask_needed(&live, 1, 22));
    }

    #[test]
    fn bitwise_mask_feeding_whole_read_keeps_exchange() {
        let cmp_s22: I = ("V_CMP_LT_U32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 2), ("SRC1", V + 3)]);
        // Feeding s_and_saveexec_b64 (whole read of s[20:21])
        let p = asm(&[
            CMP_S20,
            cmp_s22,
            ("S_AND_B64", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 20), ("SSRC1", 22)]),
            ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 20)]),
            CBRANCH_EXECZ,
            ENDPGM,
        ]);
        let live = whole_live_out(&p, false);
        assert!(mask_needed(&live, 0, 20));
        assert!(mask_needed(&live, 1, 22));
        // Writing EXEC directly, then reading it whole
        let p = asm(&[CMP_S20, cmp_s22, ("S_AND_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 20), ("SSRC1", 22)]), CBRANCH_EXECZ, ENDPGM]);
        let live = whole_live_out(&p, false);
        assert!(mask_needed(&live, 0, 20));
        assert!(mask_needed(&live, 1, 22));
    }

    #[test]
    fn cselect_and_cmov_b64_propagate_lane_masks_and_keep_scc_uniform() {
        let cmp_s22: I = ("V_CMP_LT_U32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 2), ("SRC1", V + 3)]);
        // Uniform SCC from s_cmp_eq_u32 -> s_cselect_b64 needs no exchange on its mask inputs
        let p = asm(&[
            CMP_S20,
            cmp_s22,
            ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)]),
            ("S_CSELECT_B64", "ENC_SOP2", &[("SDST", 106), ("SSRC0", 20), ("SSRC1", 22)]),
            CNDMASK,
            ENDPGM,
        ]);
        let live = whole_live_out(&p, false);
        assert!(!mask_needed(&live, 0, 20));
        assert!(!mask_needed(&live, 1, 22));
        // s_cmov_b64 reads SCC from s_and_b64 -> s_and_b64's inputs must be exchanged so SCC is uniform
        let p = asm(&[
            CMP_S20,
            cmp_s22,
            ("S_AND_B64", "ENC_SOP2", &[("SDST", 24), ("SSRC0", 20), ("SSRC1", 22)]),
            ("S_CMOV_B64", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 20)]),
            CNDMASK,
            ENDPGM,
        ]);
        let live = whole_live_out(&p, false);
        assert!(mask_needed(&live, 0, 20));
        assert!(mask_needed(&live, 1, 22));
    }

    #[test]
    fn vccz_operand_and_odd_alignment_in_bitwise_ops_keep_exchange() {
        // SRC_VCCZ (251) tests all 64 bits of VCC -> VCC must stay exchanged
        let p = asm(&[
            CMP,
            ("S_AND_B64", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 22), ("SSRC1", 251)]),
            ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 3), ("SRC1", V + 4), ("SRC2", 20)]),
            ENDPGM,
        ]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 106));
        // Odd-aligned pair s[21:22] crosses even/odd half boundary -> stay conservative
        let p = asm(&[
            CMP_S20,
            ("S_AND_B64", "ENC_SOP2", &[("SDST", 24), ("SSRC0", 21), ("SSRC1", 22)]),
            ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 3), ("SRC1", V + 4), ("SRC2", 24)]),
            ENDPGM,
        ]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 20));
    }

    #[test]
    fn loop_exit_reading_scc_keeps_exchange_inside_loop() {
        // loop: v_cmp s[20:21]; s_and_b64 vcc, s[20:21], s[22:23]; v_cndmask; s_branch loop; s_cbranch_scc1
        let p = asm(&[
            CMP_S20,
            ("S_AND_B64", "ENC_SOP2", &[("SDST", 106), ("SSRC0", 20), ("SSRC1", 22)]),
            CNDMASK,
            ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0xFFFC)]),
            ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]),
            ENDPGM,
        ]);
        assert!(mask_needed(&whole_live_out(&p, false), 0, 20));
    }
}
