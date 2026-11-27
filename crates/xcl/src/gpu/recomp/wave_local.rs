// Copyright (c) 2026 VIctory, victory-fork, Vice-Tay, and XCL contributors.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Provenance: wave-local eligibility rules adapted from Vice-Tay
// (`crates/victory-shader/src/emit/lower.rs`, `wave_local` and no-cross-half
// wave32 checks; `crates/victory-shader/src/ir.rs`, `without_skips` scalar
// check) at commit `1f69d67` (https://github.com/TaynanGT/Vice-Tay,
// GPL-3.0-or-later; inheriting from https://github.com/Marvisak/VIctory
// `63bcca4` and https://github.com/fxceless/victory-fork `9a9091d`), with
// scalar-mask select handling after https://github.com/Kemerd/VIctory-NVIDIA-RT
// `0fae809e360a`.
// Adapted to XCL's decoded RDNA2 instruction stream on 2026-11-27.

//! Exact eligibility analysis for wave-local lowering of wave64 compute shaders on
//! 32-lane hardware (`NoCrossHalf` and `SingleWave`).

use super::lane_masks::{analyze_launch, LaunchExec, Summary};
use super::x64_masks;
use crate::gpu::rdna2::{self, Decoded};

const EXEC_LO: u8 = 126;
const VCC_LO: u8 = 106;
const M0: u8 = 124;
const NULL: u8 = 125;
const INLINE_ZERO: u64 = 128;
const INLINE_ONE: u64 = 129;
const INLINE_NEG_ONE: u64 = 193;
const SCC_CODE: u64 = 253;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveLocal { NoCrossHalf, SingleWave }

/// `XCL_WAVE_LOCAL=1`: run eligible wave64 compute shaders at wave32 without cross-half exchange.
pub fn enabled() -> bool {
    std::env::var("XCL_WAVE_LOCAL").is_ok_and(|v| v == "1")
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reject { pub pc: u64, pub reason: &'static str }
#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum SccMask { #[default] Plain, ExecMask, OtherMask }
#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct MaskState {
    // Pair starts proved to hold a lane mask or two zero words.
    must: u128,
    zero: u128,
    // Individual words that may still contain wave-dependent bits.
    may: u128,
    inv: u128,
    first_src: u128,
    scc: SccMask,
}

/// Classify whether a wave64 compute program runs wave-locally without cross-half exchange.
pub fn analyze(insts: &[Decoded], cs: &super::CsInterface) -> Result<WaveLocal, Reject> {
    let rej = |pc, reason| Err(Reject { pc, reason });
    if insts.is_empty() { return rej(0, "empty program"); }
    if cs.tg_size { return rej(0, "compute TG_SIZE depends on wave size"); }
    let (mut lds, mut barrier, mut wave_wide) = (false, false, None);
    for d in insts {
        let n = d.inst.name;
        if n == "S_GETREG_B32" { return rej(d.pc, "s_getreg may read wave id"); }
        if (d.inst.flags & rdna2::F_INDIRECT != 0 || matches!(n, "S_SETPC_B64" | "S_SWAPPC_B64")) && d.far_target.is_none() { return rej(d.pc, "indirect jump"); }
        if n.starts_with("S_MOVREL") { return rej(d.pc, "relative SGPR indexing"); }
        if d.ienc.operands.iter().any(|o| o.input && !o.ty.contains("VGPR") && o.field.is_some_and(|f| matches!(fv(d, f), Some(251 | 252)))) { return rej(d.pc, "reads vccz/execz"); }
        if d.enc.name == "ENC_DS" { match n { "DS_PERMUTE_B32" | "DS_BPERMUTE_B32" => return rej(d.pc, "ds_(b)permute crosses 32-lane halves"), "DS_SWIZZLE_B32" | "DS_NOP" => {} _ => lds = true } }
        barrier |= n == "S_BARRIER";
        if d.enc.name == "ENC_SMEM" && (fv(d, "GLC") == Some(1) || fv(d, "DLC") == Some(1)) { return rej(d.pc, "coherent scalar load"); }
        if matches!(n, "V_READLANE_B32" | "V_WRITELANE_B32") { return rej(d.pc, "v_readlane/v_writelane crosses halves"); }
        if matches!(n, "S_BITREPLICATE_B64_B32" | "S_QUADMASK_B64") { return rej(d.pc, "cross-half mask transform"); }
        if (n == "S_OR_SAVEEXEC_B64" && fv(d, "SSRC0") == Some(INLINE_NEG_ONE)) || (n == "S_NAND_SAVEEXEC_B64" && fv(d, "SSRC0") == Some(INLINE_ZERO)) { return rej(d.pc, "whole-wave section"); }
        if matches!(n, "S_BCNT0_I32_B64" | "S_FF0_I32_B64" | "S_FLBIT_I32_B64") { return rej(d.pc, "unsupported mask count op"); }
        if matches!(n, "V_MBCNT_LO_U32_B32" | "V_MBCNT_HI_U32_B32" | "V_READFIRSTLANE_B32" | "S_BCNT1_I32_B64" | "S_FF1_I32_B64") {
            wave_wide.get_or_insert(d.pc);
        }
    }
    let threads = cs.threads.iter().product::<u32>();
    if lds && threads > 32 { return rej(0, "LDS in group > 32 threads"); }
    if let Some(pc) = wave_wide.filter(|_| threads > 32) {
        return rej(pc, "cross-half wave op in group > 32 threads");
    }
    let launch = if threads % 64 == 0 { LaunchExec::All } else { LaunchExec::Unknown };
    let sum = analyze_launch(insts, true, launch).map_err(|reason| Reject { pc: 0, reason })?;
    let (partial, per_branch) = partial_blocks(&sum);
    if barrier { if let Some(d) = sum.order.iter().filter(|&&b| partial[b]).flat_map(|&b| sum.blocks[b].insts.clone()).map(|i| &insts[i]).find(|d| d.inst.name == "S_BARRIER") { return rej(d.pc, "s_barrier inside lane branch"); } }
    let masks_in = solve_masks(insts, &sum)?;
    let split_branches = if threads > 32 { per_branch.as_slice() } else { &[] };
    check_masks_and_scalars(insts, &sum, split_branches, &masks_in)?;
    Ok(if wave_wide.is_some() { WaveLocal::SingleWave } else { WaveLocal::NoCrossHalf })
}

fn fv(d: &Decoded, f: &str) -> Option<u64> { rdna2::field_value(d.enc, &d.words, f) }
fn bit(r: u8) -> u128 { if r == NULL || r > 127 { 0 } else { 1u128 << r } }
fn pair_at(r: u8) -> u128 { if r == NULL || r > 126 { 0 } else { 3u128 << r } }

/// Post-dominator partial-block analysis matching `lower::partial_blocks`, combined and per lane branch.
fn partial_blocks(sum: &Summary) -> (Vec<bool>, Vec<Vec<bool>>) {
    let (n, words) = (sum.blocks.len(), (sum.blocks.len() + 1).div_ceil(64));
    let (mut er, mut work) = (vec![false; n], (0..n).filter(|&b| sum.blocks[b].succ.is_empty()).collect::<Vec<_>>());
    while let Some(b) = work.pop() { if !std::mem::replace(&mut er[b], true) { work.extend(sum.blocks[b].preds.iter().copied()); } }
    let mut pdom = vec![vec![!0u64; words]; n + 1];
    pdom[n] = vec![0u64; words]; pdom[n][n / 64] |= 1 << (n % 64);
    while {
        let mut ch = false;
        for &b in sum.order.iter().rev() {
            let mut new = vec![if !er[b] || sum.blocks[b].succ.is_empty() { 0 } else { !0u64 }; words];
            if er[b] && !sum.blocks[b].succ.is_empty() { for &(s, _) in &sum.blocks[b].succ { for (w, x) in new.iter_mut().zip(&pdom[s]) { *w &= x; } } }
            new[b / 64] |= 1 << (b % 64);
            if new != pdom[b] { pdom[b] = new; ch = true; }
        }
        ch
    } {}
    let (mut partial, mut per_branch) = (vec![false; n], Vec::new());
    for &x in &sum.order {
        if !sum.blocks[x].lane_branch { continue; }
        let (mut seen, mut st, mut part_x) = (vec![false; n], sum.blocks[x].succ.iter().map(|s| s.0).collect::<Vec<_>>(), vec![false; n]);
        while let Some(b) = st.pop() {
            if std::mem::replace(&mut seen[b], true) { continue; }
            if sum.blocks[x].succ.iter().any(|&(s, _)| (pdom[s][b / 64] >> (b % 64)) & 1 == 0) { partial[b] = true; part_x[b] = true; }
            st.extend(sum.blocks[b].succ.iter().map(|s| s.0));
        }
        if part_x.iter().any(|&p| p) { per_branch.push(part_x); }
    }
    (partial, per_branch)
}

fn solve_cfg<T: Clone + PartialEq>(sum: &Summary, entry: T, mut join: impl FnMut(&mut T, &T), mut run: impl FnMut(usize, &mut T) -> Result<(), Reject>, timeout: &'static str) -> Result<Vec<T>, Reject> {
    let n = sum.blocks.len();
    let (mut in_s, mut out_s, mut init) = (vec![entry.clone(); n], vec![entry; n], vec![false; n]);
    for _ in 0..16 {
        let mut ch = false;
        for &b in &sum.order {
            if b != 0 {
                let mut ps = sum.blocks[b].preds.iter().copied().filter(|&p| init[p]);
                if let Some(p0) = ps.next() { let mut acc = out_s[p0].clone(); for p in ps { join(&mut acc, &out_s[p]); } in_s[b] = acc; }
            }
            let mut cur = in_s[b].clone();
            run(b, &mut cur)?;
            if !init[b] || cur != out_s[b] { out_s[b] = cur; init[b] = true; ch = true; }
        }
        if !ch { return Ok(in_s); }
    }
    Err(Reject { pc: 0, reason: timeout })
}

fn is_inverting_b64(n: &str) -> bool { matches!(n, "S_NOT_B64" | "S_NAND_B64" | "S_NOR_B64" | "S_XNOR_B64" | "S_ORN1_B64" | "S_ORN2_B64" | "S_NAND_SAVEEXEC_B64" | "S_NOR_SAVEEXEC_B64" | "S_XNOR_SAVEEXEC_B64" | "S_ORN1_SAVEEXEC_B64" | "S_ORN2_SAVEEXEC_B64" | "S_ANDN1_SAVEEXEC_B64" | "S_ANDN1_WREXEC_B64") }
fn is_saveexec(n: &str) -> bool { matches!(n, "S_AND_SAVEEXEC_B64" | "S_OR_SAVEEXEC_B64" | "S_XOR_SAVEEXEC_B64" | "S_ANDN2_SAVEEXEC_B64" | "S_ANDN2_WREXEC_B64") }
fn is_b64_mask_logic(n: &str) -> bool { matches!(n, "S_MOV_B64" | "S_CMOV_B64" | "S_AND_B64" | "S_OR_B64" | "S_XOR_B64" | "S_ANDN2_B64" | "S_CSELECT_B64" | "S_WQM_B64") || is_saveexec(n) }
fn is_permitted_mask_consumer(d: &Decoded) -> bool {
    let n = d.inst.name;
    is_b64_mask_logic(n) || matches!(n, "S_CBRANCH_EXECZ" | "S_CBRANCH_EXECNZ" | "S_BCNT1_I32_B64" | "S_FF1_I32_B64" | "V_MBCNT_LO_U32_B32" | "V_MBCNT_HI_U32_B32" | "V_CNDMASK_B32" | "V_ADD_CO_CI_U32" | "V_SUB_CO_CI_U32" | "V_SUBREV_CO_CI_U32")
}
fn scc_rw(d: &Decoded, out: bool) -> bool { d.ienc.operands.iter().any(|o| (if out { o.output } else { o.input }) && (o.ty == "OPR_SSRC_SPECIAL_SCC" || o.field.is_some_and(|f| fv(d, f) == Some(SCC_CODE) && (out || !o.ty.contains("VGPR"))))) }

fn mask_op_dest(d: &Decoded) -> Option<u8> {
    let n = d.inst.name;
    if n.starts_with("V_CMPX") { return Some(EXEC_LO); }
    let sdwa_vcc = |e: &str| d.enc.name == e && fv(d, "SD") != Some(1);
    if n.starts_with("V_CMP") { return if sdwa_vcc("VOPC_VOP_SDWA_SDST_ENC") { Some(VCC_LO) } else if d.enc.name.starts_with("ENC_VOP3") || d.enc.name.starts_with("VOP3_") { fv(d, "VDST").map(|r| r as u8) } else { Some(fv(d, "SDST").unwrap_or(VCC_LO as u64) as u8) }; }
    matches!(n, "V_ADD_CO_U32" | "V_SUB_CO_U32" | "V_SUBREV_CO_U32" | "V_ADD_CO_CI_U32" | "V_SUB_CO_CI_U32" | "V_SUBREV_CO_CI_U32" | "V_DIV_SCALE_F32" | "V_DIV_SCALE_F64" | "V_MAD_U64_U32" | "V_MAD_I64_I32")
        .then(|| if sdwa_vcc("VOP2_VOP_SDWA_SDST_ENC") { VCC_LO } else { fv(d, "SDST").unwrap_or(VCC_LO as u64) as u8 })
}

fn step_masks(d: &Decoded, st: &mut MaskState) -> Result<(u128, bool), Reject> {
    let n = d.inst.name;
    let (sr, sw) = rdna2::sgpr_uses(d);
    let dst_mask = mask_op_dest(d);
    let overwritten = x64_masks::uses(d).1;
    let mut wr = overwritten | sw.iter().fold(0u128, |m, &r| m | bit(r as u8)) | dst_mask.map_or(0, pair_at);
    if is_saveexec(n) || (is_inverting_b64(n) && n.contains("EXEC")) { wr |= pair_at(EXEC_LO); }
    if is_inverting_b64(n) && ((wr & pair_at(EXEC_LO)) != 0 || sr.iter().any(|&r| bit(r as u8) & st.may != 0)) {
        return Err(Reject { pc: d.pc, reason: "inverting 64-bit mask op" });
    }
    let full_exec = n.starts_with("V_CMPX") || is_saveexec(n)
        || (is_b64_mask_logic(n) && fv(d, "SDST") == Some(u64::from(EXEC_LO)));
    if wr & pair_at(EXEC_LO) != 0 && !full_exec {
        return Err(Reject { pc: d.pc, reason: "unsupported write to EXEC" });
    }
    let has_m = |m: u128, c: u64| c <= EXEC_LO as u64 && c != M0 as u64 && c != NULL as u64 && (m >> c) & 1 != 0;
    let is_inv = |inv: u128, c: u64| c == INLINE_NEG_ONE || (c <= EXEC_LO as u64 && (inv >> c) & 1 != 0);
    let (mut dst_is_mask, mut scc_set) = (false, None);
    if let Some(dst) = dst_mask {
        if dst != NULL && dst % 2 != 0 {
            return Err(Reject { pc: d.pc, reason: "unaligned mask pair" });
        }
        if dst != NULL { st.must |= bit(dst); st.may |= pair_at(dst); st.zero &= !bit(dst); st.inv &= !bit(dst); st.first_src &= !pair_at(dst); dst_is_mask = true; }
    } else if is_b64_mask_logic(n) {
        let (s0, s1, sd, inv_in) = (fv(d, "SSRC0").unwrap_or(0), fv(d, "SSRC1"), fv(d, "SDST").unwrap_or(NULL as u64) as u8, st.inv);
        let overlaps_mask = |code: u64| code <= u64::from(EXEC_LO) && st.may & pair_at(code as u8) != 0;
        if is_saveexec(n) || sd == EXEC_LO || overlaps_mask(s0)
            || s1.is_some_and(overlaps_mask) || (n == "S_CMOV_B64" && overlaps_mask(u64::from(sd)))
        {
            if sd != NULL && sd % 2 != 0 {
                return Err(Reject { pc: d.pc, reason: "unaligned mask pair" });
            }
            let ok = |c: u64| has_m(st.must, c) || matches!(c, INLINE_ZERO | INLINE_NEG_ONE) || (c <= EXEC_LO as u64 && (st.zero >> c) & 1 != 0);
            if !ok(s0) || s1.is_some_and(|c| !ok(c)) || (n == "S_CMOV_B64" && !ok(sd as u64)) {
                return Err(Reject { pc: d.pc, reason: if sd == EXEC_LO { "non-mask scalar written to EXEC" } else { "mask op with non-mask source" } });
            }
            let (exec_inv, sd_inv) = match n {
                "S_AND_B64" => (false, is_inv(inv_in, s0) && is_inv(inv_in, s1.unwrap_or(0))),
                "S_ANDN2_B64" | "S_MOV_B64" | "S_WQM_B64" => (false, is_inv(inv_in, s0)),
                "S_OR_B64" | "S_XOR_B64" | "S_CSELECT_B64" => (false, is_inv(inv_in, s0) || is_inv(inv_in, s1.unwrap_or(0))),
                "S_CMOV_B64" => (false, is_inv(inv_in, s0) || is_inv(inv_in, sd as u64)),
                "S_ANDN2_SAVEEXEC_B64" | "S_ANDN2_WREXEC_B64" => { let ni = is_inv(inv_in, s0); (ni, n.ends_with("_WREXEC_B64") && ni) }
                "S_OR_SAVEEXEC_B64" | "S_XOR_SAVEEXEC_B64" => (is_inv(inv_in, s0), false),
                _ => (false, false),
            };
            if (is_saveexec(n) && exec_inv) || (!is_saveexec(n) && sd == EXEC_LO && sd_inv) { return Err(Reject { pc: d.pc, reason: "inverting 64-bit mask op" }); }
            if sd != NULL && sd <= EXEC_LO { st.must |= bit(sd); st.may |= pair_at(sd); st.zero &= !bit(sd); if sd_inv { st.inv |= bit(sd); } else { st.inv &= !bit(sd); } }
            dst_is_mask = true;
            if matches!(n, "S_AND_B64" | "S_OR_B64" | "S_XOR_B64" | "S_ANDN2_B64" | "S_WQM_B64") || is_saveexec(n) {
                scc_set = Some(if is_saveexec(n) || sd == EXEC_LO { SccMask::ExecMask } else { SccMask::OtherMask });
            }
        }
    } else if n == "S_LSHL_B64" && fv(d, "SSRC0") == Some(INLINE_ONE) {
        let (s1, sd) = (fv(d, "SSRC1").unwrap_or(0), fv(d, "SDST").unwrap_or(NULL as u64) as u8);
        if s1 < 128 && (st.first_src >> s1) & 1 != 0 && sd < NULL { st.must |= bit(sd); st.may |= pair_at(sd); st.zero &= !bit(sd); st.inv &= !bit(sd); dst_is_mask = true; scc_set = Some(SccMask::ExecMask); }
    }
    if !dst_is_mask {
        let clr = (wr | (wr >> 1)) & !bit(EXEC_LO);
        st.must &= !clr;
        st.zero &= !clr;
        // Conditional writes can leave the old word intact.
        st.may &= !overwritten;
        st.inv &= !(overwritten & (overwritten >> 1));
        if n == "S_MOV_B64" && fv(d, "SSRC0") == Some(INLINE_ZERO) {
            if let Some(dst) = fv(d, "SDST").filter(|&r| r < u64::from(NULL) && r % 2 == 0) {
                st.zero |= bit(dst as u8);
            }
        }
    }
    st.must |= bit(EXEC_LO); st.may |= pair_at(EXEC_LO); st.inv &= !bit(EXEC_LO); st.first_src &= !wr;
    if n == "S_FF1_I32_B64" && has_m(st.must, fv(d, "SSRC0").unwrap_or(0)) {
        if let Some(sd) = fv(d, "SDST").filter(|&r| r < NULL as u64) { st.first_src |= bit(sd as u8); }
    }
    if let Some(k) = scc_set { st.scc = k; } else if scc_rw(d, true) { st.scc = SccMask::Plain; }
    Ok((if dst_is_mask { wr } else { 0 }, dst_is_mask))
}

fn solve_masks(insts: &[Decoded], sum: &Summary) -> Result<Vec<MaskState>, Reject> {
    let entry = MaskState { must: bit(EXEC_LO), may: pair_at(EXEC_LO), ..Default::default() };
    solve_cfg(sum, entry,
        |m, o| { m.must = (m.must & o.must) | bit(EXEC_LO); m.may = m.may | o.may | pair_at(EXEC_LO); m.zero &= o.zero; m.inv |= o.inv; m.first_src &= o.first_src; if m.scc != o.scc { m.scc = SccMask::OtherMask; } },
        |b, st| { for i in sum.blocks[b].insts.clone() { step_masks(&insts[i], st)?; } Ok(()) },
        "mask dataflow did not converge")
}

fn check_exec_in_branch(insts: &[Decoded], sum: &Summary, part: &[bool]) -> Result<(), Reject> {
    let n = sum.blocks.len();
    let (mut sub_in, mut sub_out, mut init) = (vec![0u128; n], vec![0u128; n], vec![false; n]);
    for _ in 0..16 {
        let mut ch = false;
        for &b in &sum.order {
            if !part[b] { continue; }
            let mut acc = None;
            for &p in &sum.blocks[b].preds {
                let v = if part[p] { if init[p] { sub_out[p] } else { continue } } else { bit(EXEC_LO) };
                acc = Some(acc.map_or(v, |a: u128| a & v));
            }
            let cur_in = acc.unwrap_or(0);
            let mut sub = cur_in;
            for i in sum.blocks[b].insts.clone() {
                let d = &insts[i];
                let (name, dst_mask) = (d.inst.name, mask_op_dest(d));
                let wr = x64_masks::uses(d).1 | rdna2::sgpr_uses(d).1.iter().fold(0u128, |m, &r| m | bit(r as u8))
                    | dst_mask.map_or(0, pair_at) | if is_saveexec(name) { pair_at(EXEC_LO) } else { 0 };
                let (s0, s1, sd) = (fv(d, "SSRC0").unwrap_or(0), fv(d, "SSRC1").unwrap_or(0), fv(d, "SDST").unwrap_or(NULL as u64) as u8);
                let is_sub = |s: u128, c: u64| c == INLINE_ZERO || (c <= u64::from(EXEC_LO) && c != u64::from(M0) && c != u64::from(NULL) && (s >> c) & 1 != 0);
                let res_sub = match name {
                    n if n.starts_with("V_CMPX") => true,
                    "S_AND_SAVEEXEC_B64" => true,
                    "S_ANDN2_SAVEEXEC_B64" | "S_ANDN2_WREXEC_B64" | "S_OR_SAVEEXEC_B64" | "S_XOR_SAVEEXEC_B64" | "S_MOV_B64" | "S_WQM_B64" | "S_ANDN2_B64" => is_sub(sub, s0),
                    "S_AND_B64" => is_sub(sub, s0) || is_sub(sub, s1),
                    "S_OR_B64" | "S_XOR_B64" | "S_CSELECT_B64" => is_sub(sub, s0) && is_sub(sub, s1),
                    "S_CMOV_B64" => is_sub(sub, s0) && is_sub(sub, u64::from(sd)),
                    _ => false,
                };
                if wr & pair_at(EXEC_LO) != 0 && !res_sub && !insts.get(i + 1).is_some_and(|next| next.inst.name == "S_ENDPGM") {
                    return Err(Reject { pc: d.pc, reason: "widening EXEC in a lane branch" });
                }
                sub &= !((wr | (wr >> 1)) & !bit(EXEC_LO));
                if let Some(dst) = dst_mask.filter(|&r| r != NULL && r != EXEC_LO) { sub |= bit(dst); }
                else if is_saveexec(name) && (!name.ends_with("_WREXEC_B64") || res_sub) && sd < EXEC_LO && sd != NULL { sub |= bit(sd); }
                else if is_b64_mask_logic(name) && sd < EXEC_LO && sd != NULL && res_sub { sub |= bit(sd); }
                sub |= bit(EXEC_LO);
            }
            if !init[b] || sub != sub_out[b] || cur_in != sub_in[b] {
                sub_in[b] = cur_in; sub_out[b] = sub; init[b] = true; ch = true;
            }
        }
        if !ch { return Ok(()); }
    }
    Err(Reject { pc: 0, reason: "exec dataflow did not converge" })
}

fn check_masks_and_scalars(insts: &[Decoded], sum: &Summary, per_branch: &[Vec<bool>], masks_in: &[MaskState]) -> Result<(), Reject> {
    let n = sum.blocks.len();
    let (mut live_in, mut wr_blk, mut b_use, mut b_def) = (vec![(0u128, false); n], vec![(0u128, false); n], vec![(0u128, false); n], vec![(0u128, false); n]);
    for &b in &sum.order {
        let (mut st, mut ops) = (masks_in[b], Vec::new());
        for i in sum.blocks[b].insts.clone() {
            let (d, pre_may_p, pre_inv_p, pre_scc, name) = (&insts[i], st.may, st.inv | (st.inv << 1), st.scc, insts[i].inst.name);
            let known_masks = st.must | st.zero;
            let _ = step_masks(d, &mut st)?;
            if (matches!(name, "S_BCNT1_I32_B64" | "S_FF1_I32_B64") && fv(d, "SSRC0").is_some_and(|c| c < 128 && bit(c as u8) & pre_inv_p != 0))
                || (matches!(name, "V_MBCNT_LO_U32_B32" | "V_MBCNT_HI_U32_B32") && fv(d, "SRC0").is_some_and(|c| c < 128 && bit(c as u8) & pre_inv_p != 0))
            {
                return Err(Reject { pc: d.pc, reason: "inverting 64-bit mask op" });
            }
            if matches!(name, "S_CBRANCH_VCCZ" | "S_CBRANCH_VCCNZ") && (pre_may_p & pair_at(VCC_LO)) != 0 { return Err(Reject { pc: d.pc, reason: "wave-wide vcc branch" }); }
            if scc_rw(d, false) && (pre_scc == SccMask::OtherMask || (pre_scc == SccMask::ExecMask && !(matches!(name, "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1") && sum.blocks[b].lane_branch))) { return Err(Reject { pc: d.pc, reason: "mask condition SCC read" }); }
            let (xm_r, xm_w) = x64_masks::uses(d);
            let lane_mask = matches!(name, "V_CNDMASK_B32" | "V_ADD_CO_CI_U32" | "V_SUB_CO_CI_U32" | "V_SUBREV_CO_CI_U32");
            let count_mask = matches!(name, "V_MBCNT_LO_U32_B32" | "V_MBCNT_HI_U32_B32");
            let count_reads_mask_data = count_mask && fv(d, "SRC1").is_some_and(|c| c < 128 && bit(c as u8) & pre_may_p != 0);
            let bad_scalar_count = matches!(name, "S_BCNT1_I32_B64" | "S_FF1_I32_B64")
                && fv(d, "SSRC0").is_some_and(|c| c < 128 && pre_may_p & pair_at(c as u8) != 0 && known_masks & bit(c as u8) == 0);
            let bad_mbcnt = count_mask && fv(d, "SRC0").is_some_and(|c| {
                c < 128 && pre_may_p & bit(c as u8) != 0
                    && !(known_masks & bit(c as u8) != 0 || (name == "V_MBCNT_HI_U32_B32" && c > 0 && known_masks & bit((c - 1) as u8) != 0))
            });
            if count_reads_mask_data || bad_scalar_count || bad_mbcnt || (xm_r & pre_may_p != 0 && (lane_mask || !is_permitted_mask_consumer(d))) {
                return Err(Reject { pc: d.pc, reason: "mask read as data" });
            }
            let lane_m = if lane_mask {
                let src = if d.enc.name.contains("VOP3") { fv(d, "SRC2") } else { Some(u64::from(VCC_LO)) };
                match src {
                    Some(INLINE_ZERO | INLINE_NEG_ONE) => 0,
                    Some(c) if c <= u64::from(EXEC_LO) && known_masks & bit(c as u8) != 0 => pair_at(c as u8),
                    _ => return Err(Reject { pc: d.pc, reason: "unproven lane mask" }),
                }
            } else { 0 };
            let sg_r = (xm_r | lane_m) & !pair_at(EXEC_LO);
            let sg_kill = xm_w & !pair_at(EXEC_LO);
            let sg_w = (xm_w | rdna2::sgpr_uses(d).1.iter().fold(0u128, |m, &r| m | bit(r as u8)) | mask_op_dest(d).map_or(0, pair_at)) & !pair_at(EXEC_LO);
            let (r_scc, w_scc) = (scc_rw(d, false), scc_rw(d, true));
            wr_blk[b].0 |= sg_w; wr_blk[b].1 |= w_scc;
            ops.push((sg_r, sg_kill, r_scc, w_scc));
        }
        let (mut us, mut ds, mut uc, mut dc) = (0u128, 0u128, false, false);
        for (sr, sw, rc, wc) in ops.into_iter().rev() { us = (us & !sw) | sr; ds = (ds | sw) & !sr; uc = rc || (!wc && uc); dc = !rc && (wc || dc); }
        b_use[b] = (us, uc); b_def[b] = (ds, dc);
    }
    while { let mut ch = false; for &b in sum.order.iter().rev() { let out = sum.blocks[b].succ.iter().fold((0u128, false), |a, &(s, _)| (a.0 | live_in[s].0, a.1 || live_in[s].1)); let nin = (b_use[b].0 | (out.0 & !b_def[b].0), b_use[b].1 || (out.1 && !b_def[b].1)); if nin != live_in[b] { live_in[b] = nin; ch = true; } } ch } {}
    for part in per_branch {
        let mut dirty_out = vec![(0u128, false); n];
        while { let mut ch = false; for &b in &sum.order { if part[b] { let din = sum.blocks[b].preds.iter().filter(|&&p| part[p]).fold((0u128, false), |a, &p| (a.0 | dirty_out[p].0, a.1 || dirty_out[p].1)); let dout = (din.0 | wr_blk[b].0, din.1 || wr_blk[b].1); if dout != dirty_out[b] { dirty_out[b] = dout; ch = true; } } } ch } {}
        for &b in &sum.order {
            if part[b] && sum.blocks[b].succ.iter().any(|&(s, _)| !part[s] && ((dirty_out[b].0 & live_in[s].0) != 0 || (dirty_out[b].1 && live_in[s].1))) {
                return Err(Reject { pc: sum.blocks[b].insts.clone().last().map_or(0, |i| insts[i].pc), reason: "scalar write in lane branch live at join" });
            }
        }
        check_exec_in_branch(insts, sum, part)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{pack, shader, xbox};

    type Op = (&'static str, &'static str, Vec<(&'static str, u64)>);

    fn op(name: &'static str, enc: &'static str, fields: Vec<(&'static str, u64)>) -> Op {
        (name, enc, fields)
    }

    fn asm(ops: &[Op]) -> Vec<Decoded> {
        let mut bytes = Vec::new();
        for (name, enc, fields) in ops {
            bytes.extend(rdna2::encode(name, enc, fields, None).unwrap());
        }
        let mut insts = Vec::new();
        let mut off = 0usize;
        while off + 4 <= bytes.len() {
            let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
            off += d.len;
            insts.push(d);
        }
        rdna2::resolve_far_jumps(&mut insts);
        insts
    }

    fn cs(threads: [u32; 3]) -> super::super::CsInterface {
        super::super::CsInterface { threads, user_sgprs: 2, tgid: [true, false, false], tg_size: false, tid_components: 1 }
    }

    #[test]
    fn rejects_restoring_exec_before_a_lane_branch_joins() {
        let insts = asm(&[
            op("V_CMP_GT_U32", "ENC_VOPC", vec![("SRC0", 160), ("VSRC1", 0)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 106)]),
            op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", 3)]),
            op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 2)]),
            op("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", 0), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(
            analyze(&insts, &cs([64, 1, 1])).unwrap_err().reason,
            "widening EXEC in a lane branch",
        );
    }

    #[test]
    fn allows_nested_saveexec_restore_inside_lane_branch() {
        let nested = |restore_reg| asm(&[
            op("V_CMP_GT_U32", "ENC_VOPC", vec![("SRC0", 160), ("VSRC1", 0)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 106)]),
            op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", 6)]),
            op("V_CMP_EQ_U32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", 0)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 0), ("SSRC0", 106)]),
            op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", 1)]),
            op("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 1), ("SRC0", 129), ("VSRC1", 1)]),
            op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", restore_reg)]),
            op("V_MOV_B32", "ENC_VOP1", vec![("VDST", 2), ("SRC0", 257)]),
            op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 2)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&nested(0), &cs([64, 1, 1])), Ok(WaveLocal::NoCrossHalf));
        assert_eq!(analyze(&nested(2), &cs([64, 1, 1])).unwrap_err().reason, "widening EXEC in a lane branch");
    }

    #[test]
    fn rejects_explicit_exec_word_writes() {
        for dst in [126, 127] {
            let insts = asm(&[
                op("S_MOV_B32", "ENC_SOP1", vec![("SDST", dst), ("SSRC0", 128)]),
                op("S_ENDPGM", "ENC_SOPP", vec![]),
            ]);
            assert_eq!(
                analyze(&insts, &cs([64, 1, 1])).unwrap_err().reason,
                "unsupported write to EXEC",
            );
        }
    }

    #[test]
    fn rejects_raw_bits_as_a_lane_mask() {
        let insts = asm(&[
            op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 106), ("SSRC0", 129)]),
            op("V_CNDMASK_B32", "ENC_VOP2", vec![("VDST", 2), ("SRC0", 128), ("VSRC1", 0)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(
            analyze(&insts, &cs([64, 1, 1])).unwrap_err().reason,
            "unproven lane mask",
        );
    }

    #[test]
    fn a_definitely_overwritten_mask_word_is_scalar_data() {
        let insts = asm(&[
            op("V_CMP_LT_U32", "ENC_VOPC", vec![("SRC0", 256), ("VSRC1", 1)]),
            op("S_MOV_B32", "ENC_SOP1", vec![("SDST", 106), ("SSRC0", 129)]),
            op("S_MOV_B32", "ENC_SOP1", vec![("SDST", 0), ("SSRC0", 106)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&insts, &cs([64, 1, 1])), Ok(WaveLocal::NoCrossHalf));
    }

    #[test]
    fn rejects_mask_data_in_sdwa_and_after_a_single_word_overwrite() {
        let cmp = || op("V_CMP_LT_U32", "ENC_VOPC", vec![("SRC0", 256), ("VSRC1", 1)]);
        let end = || op("S_ENDPGM", "ENC_SOPP", vec![]);
        let sdwa = asm(&[
            cmp(),
            op("V_MOV_B32", "VOP1_VOP_SDWA", vec![
                ("VDST", 2), ("SRC0", 0xF9), ("VSRC0", 106), ("S0", 1),
                ("SRC0_SEL", 6), ("DST_SEL", 6),
            ]),
            end(),
        ]);
        let remaining_word = asm(&[
            cmp(),
            op("S_MOV_B32", "ENC_SOP1", vec![("SDST", 106), ("SSRC0", 128)]),
            op("S_MOV_B32", "ENC_SOP1", vec![("SDST", 0), ("SSRC0", 107)]),
            end(),
        ]);
        let shifts_by_mask = asm(&[
            cmp(),
            op("S_LSHL_B64", "ENC_SOP2", vec![("SDST", 2), ("SSRC0", 129), ("SSRC1", 106)]),
            end(),
        ]);
        for insts in [sdwa, remaining_word, shifts_by_mask] {
            assert_eq!(analyze(&insts, &cs([64, 1, 1])).unwrap_err().reason, "mask read as data");
        }
    }

    #[test]
    fn rejects_reservation_split_across_two_host_waves() {
        let mut ops = alloc_idiom_seq(false);
        let (_, _, fields) = ops.iter_mut().find(|op| op.0 == "BUFFER_ATOMIC_ADD").unwrap();
        // The first lane uses its thread ID as the counter offset. The second half
        // would reserve from a different counter once it elects its own first lane.
        fields.push(("OFFEN", 1));
        let result = analyze(&asm(&ops), &cs([64, 1, 1]));
        assert_eq!(result.unwrap_err().reason, "cross-half wave op in group > 32 threads");
    }

    #[test]
    fn plain_wave64_compute_is_no_cross_half() {
        let insts = asm(&[
            op("V_CMP_LT_U32", "ENC_VOPC", vec![("SRC0", 256), ("VSRC1", 1)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 106)]),
            op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", 3)]),
            op("V_ADD_F32", "ENC_VOP2", vec![("VDST", 2), ("SRC0", 256), ("VSRC1", 1)]),
            op("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", 2), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128)]),
            op("S_OR_B64", "ENC_SOP2", vec![("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&insts, &cs([64, 1, 1])), Ok(WaveLocal::NoCrossHalf));
    }

    #[test]
    fn vop3_vcmp_and_wqm_accepted() {
        let ok = asm(&[
            op("V_CMP_EQ_F32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 256), ("SRC1", 257)]),
            op("V_CMP_LT_F32", "ENC_VOPC", vec![("SRC0", 256), ("VSRC1", 1)]),
            op("S_AND_B64", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 106), ("SSRC1", 0)]),
            op("S_XOR_B64", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 0), ("SSRC1", 193)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 0)]),
            op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", 2)]),
            op("S_WQM_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 126)]),
            op("S_OR_B64", "ENC_SOP2", vec![("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&ok, &cs([64, 1, 1])), Ok(WaveLocal::NoCrossHalf));

        let bitrep = asm(&[
            op("S_BITREPLICATE_B64_B32", "ENC_SOP1", vec![("SDST", 0), ("SSRC0", 4)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&bitrep, &cs([64, 1, 1])).unwrap_err().reason, "cross-half mask transform");

        let quadmask = asm(&[
            op("S_QUADMASK_B64", "ENC_SOP1", vec![("SDST", 0), ("SSRC0", 126)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&quadmask, &cs([64, 1, 1])).unwrap_err().reason, "cross-half mask transform");
    }

    #[test]
    fn intra_half_dpp_and_swizzle_allowed() {
        let insts = asm(&[
            op("V_ADD_F32", "VOP2_VOP_DPP16", vec![("VDST", 0), ("SRC0", 0xFA), ("VSRC0", 1), ("VSRC1", 2), ("DPP_CTRL", 0x101), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)]),
            op("DS_SWIZZLE_B32", "ENC_DS", vec![("VDST", 3), ("ADDR", 0), ("OFFSET0", 0x1F), ("OFFSET1", 0x4)]),
            op("V_PERMLANEX16_B32", "ENC_VOP3", vec![("VDST", 4), ("SRC0", 256 + 3), ("SRC1", 128), ("SRC2", 128)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&insts, &cs([64, 1, 1])), Ok(WaveLocal::NoCrossHalf));
    }

    #[test]
    fn lds_small_group_allowed_large_group_rejected() {
        let insts = asm(&[
            op("DS_WRITE_B32", "ENC_DS", vec![("ADDR", 0), ("DATA0", 1)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&insts, &cs([32, 1, 1])), Ok(WaveLocal::NoCrossHalf));
        assert_eq!(analyze(&insts, &cs([64, 1, 1])).unwrap_err().reason, "LDS in group > 32 threads");
    }

    #[test]
    fn rejects_tg_size_and_glc_smem_and_readlane_and_bpermute() {
        let base = asm(&[op("S_ENDPGM", "ENC_SOPP", vec![])]);
        let mut iface = cs([64, 1, 1]);
        iface.tg_size = true;
        assert_eq!(analyze(&base, &iface).unwrap_err().reason, "compute TG_SIZE depends on wave size");

        let glc = asm(&[
            op("S_LOAD_DWORD", "ENC_SMEM", vec![("SDATA", 4), ("SBASE", 0), ("SOFFSET", 125), ("GLC", 1)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&glc, &cs([64, 1, 1])).unwrap_err().reason, "coherent scalar load");

        let readlane = asm(&[
            op("V_READLANE_B32", "ENC_VOP3", vec![("VDST", 4), ("SRC0", 256), ("SRC1", 128)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&readlane, &cs([64, 1, 1])).unwrap_err().reason, "v_readlane/v_writelane crosses halves");

        let bperm = asm(&[
            op("DS_BPERMUTE_B32", "ENC_DS", vec![("VDST", 2), ("ADDR", 0), ("DATA0", 1)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&bperm, &cs([64, 1, 1])).unwrap_err().reason, "ds_(b)permute crosses 32-lane halves");
    }

    #[test]
    fn rejects_inverting_mask_op_and_wave_wide_branch() {
        let not_exec = asm(&[
            op("S_NOT_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 126)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&not_exec, &cs([64, 1, 1])).unwrap_err().reason, "inverting 64-bit mask op");

        let vcc_br = asm(&[
            op("V_CMP_EQ_U32", "ENC_VOPC", vec![("SRC0", 256), ("VSRC1", 1)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 106)]),
            op("S_CBRANCH_VCCZ", "ENC_SOPP", vec![("SIMM16", 1)]),
            op("V_ADD_F32", "ENC_VOP2", vec![("VDST", 2), ("SRC0", 256), ("VSRC1", 1)]),
            op("S_OR_B64", "ENC_SOP2", vec![("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&vcc_br, &cs([64, 1, 1])).unwrap_err().reason, "wave-wide vcc branch");

        for (bad, t) in [
            (asm(&[op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 193)]), op("S_ENDPGM", "ENC_SOPP", vec![])]), 64),
            (asm(&[op("S_XOR_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 0), ("SSRC0", 193)]), op("S_ENDPGM", "ENC_SOPP", vec![])]), 64),
            (asm(&[
                op("V_CMP_EQ_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 256), ("SRC1", 257)]),
                op("S_XOR_B64", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 0), ("SSRC1", 193)]),
                op("S_BCNT1_I32_B64", "ENC_SOP1", vec![("SDST", 4), ("SSRC0", 0)]),
                op("S_ENDPGM", "ENC_SOPP", vec![]),
            ]), 32),
        ] {
            assert_eq!(analyze(&bad, &cs([t, 1, 1])).unwrap_err().reason, "inverting 64-bit mask op");
        }

        let inv_and_saveexec = asm(&[
            op("V_CMP_EQ_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 256), ("SRC1", 257)]),
            op("S_XOR_B64", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 0), ("SSRC1", 193)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 0)]),
            op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", 1)]),
            op("V_ADD_F32", "ENC_VOP2", vec![("VDST", 2), ("SRC0", 256), ("VSRC1", 1)]),
            op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 2)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&inv_and_saveexec, &cs([64, 1, 1])), Ok(WaveLocal::NoCrossHalf));

        let reads_execz = asm(&[
            op("S_MOV_B32", "ENC_SOP1", vec![("SDST", 4), ("SSRC0", 252)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&reads_execz, &cs([64, 1, 1])).unwrap_err().reason, "reads vccz/execz");
    }

    #[test]
    fn rejects_live_scalar_write_inside_lane_branch() {
        let bad = asm(&[
            op("V_CMP_LT_U32", "ENC_VOPC", vec![("SRC0", 256), ("VSRC1", 1)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 106)]),
            op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", 1)]),
            op("S_MOV_B32", "ENC_SOP1", vec![("SDST", 4), ("SSRC0", 135)]),
            op("S_OR_B64", "ENC_SOP2", vec![("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)]),
            op("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 1), ("SRC0", 4), ("VSRC1", 0)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&bad, &cs([64, 1, 1])).unwrap_err().reason, "scalar write in lane branch live at join");

        let ok = asm(&[
            op("V_CMP_LT_U32", "ENC_VOPC", vec![("SRC0", 256), ("VSRC1", 1)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 106)]),
            op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", 2)]),
            op("S_MOV_B32", "ENC_SOP1", vec![("SDST", 4), ("SSRC0", 135)]),
            op("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 1), ("SRC0", 4), ("VSRC1", 0)]),
            op("S_OR_B64", "ENC_SOP2", vec![("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&ok, &cs([64, 1, 1])), Ok(WaveLocal::NoCrossHalf));

        let mask_bad = asm(&[
            op("V_CMP_EQ_U32", "ENC_VOP3", vec![("VDST", 4), ("SRC0", 128), ("SRC1", 256)]),
            op("V_CMP_LT_U32", "ENC_VOPC", vec![("SRC0", 256), ("VSRC1", 1)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 106)]),
            op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", 2)]),
            op("V_CMP_EQ_U32", "ENC_VOP3", vec![("VDST", 4), ("SRC0", 129), ("SRC1", 256)]),
            op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 2)]),
            op("V_CNDMASK_B32", "ENC_VOP3", vec![("VDST", 1), ("SRC0", 128), ("SRC1", 129), ("SRC2", 4)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&mask_bad, &cs([64, 1, 1])).unwrap_err().reason, "scalar write in lane branch live at join");

        let nested_bad = asm(&[
            op("V_CMP_LT_U32", "ENC_VOPC", vec![("SRC0", 256), ("VSRC1", 1)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 106)]),
            op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", 7)]),
            op("V_CMP_GT_U32", "ENC_VOPC", vec![("SRC0", 257), ("VSRC1", 0)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 6), ("SSRC0", 106)]),
            op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", 1)]),
            op("S_MOV_B32", "ENC_SOP1", vec![("SDST", 4), ("SSRC0", 135)]),
            op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 6)]),
            op("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 1), ("SRC0", 4), ("VSRC1", 0)]),
            op("S_MOV_B32", "ENC_SOP1", vec![("SDST", 4), ("SSRC0", 128)]),
            op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 2)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&nested_bad, &cs([64, 1, 1])).unwrap_err().reason, "scalar write in lane branch live at join");
    }

    #[test]
    fn rejects_barrier_inside_lane_branch() {
        let bad = asm(&[
            op("V_CMP_LT_U32", "ENC_VOPC", vec![("SRC0", 256), ("VSRC1", 1)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 106)]),
            op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", 1)]),
            op("S_BARRIER", "ENC_SOPP", vec![]),
            op("S_OR_B64", "ENC_SOP2", vec![("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&bad, &cs([64, 1, 1])).unwrap_err().reason, "s_barrier inside lane branch");

        let ok = asm(&[
            op("S_BARRIER", "ENC_SOPP", vec![]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&ok, &cs([64, 1, 1])), Ok(WaveLocal::NoCrossHalf));
    }

    fn alloc_idiom_seq(vop3_cmp: bool) -> Vec<Op> {
        let mut ops = vec![
            op("S_BCNT1_I32_B64", "ENC_SOP1", vec![("SDST", 4), ("SSRC0", 126)]),
            op("S_FF1_I32_B64", "ENC_SOP1", vec![("SDST", 5), ("SSRC0", 126)]),
            op("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 1), ("SRC0", 193), ("SRC1", 128)]),
            op("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 1), ("SRC0", 193), ("SRC1", 257)]),
        ];
        if vop3_cmp {
            ops.push(op("V_CMP_EQ_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 5), ("SRC1", 257)]));
            ops.push(op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 6), ("SSRC0", 0)]));
        } else {
            ops.push(op("V_CMP_EQ_U32", "ENC_VOPC", vec![("SRC0", 5), ("VSRC1", 1)]));
            ops.push(op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 6), ("SSRC0", 106)]));
        }
        ops.extend([
            op("V_MOV_B32", "ENC_VOP1", vec![("VDST", 2), ("SRC0", 4)]),
            op("BUFFER_ATOMIC_ADD", "ENC_MUBUF", vec![("VDATA", 2), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("GLC", 1)]),
            op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 6)]),
            op("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 8), ("SRC0", 258)]),
            op("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 3), ("SRC0", 126), ("SRC1", 128)]),
            op("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 3), ("SRC0", 127), ("SRC1", 259)]),
            op("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 4), ("SRC0", 8), ("VSRC1", 3)]),
            op("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", 9), ("VADDR", 4), ("SRSRC", 2), ("SOFFSET", 128)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        ops
    }

    #[test]
    fn single_wave_ops_accepted_in_one_host_wave() {
        assert_eq!(analyze(&asm(&alloc_idiom_seq(false)), &cs([32, 1, 1])), Ok(WaveLocal::SingleWave));
        assert_eq!(analyze(&asm(&alloc_idiom_seq(true)), &cs([32, 1, 1])), Ok(WaveLocal::SingleWave));
        let ff1_lshl = asm(&[
            op("S_FF1_I32_B64", "ENC_SOP1", vec![("SDST", 5), ("SSRC0", 126)]),
            op("S_LSHL_B64", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 129), ("SSRC1", 5)]),
            op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 6), ("SSRC0", 0)]),
            op("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 8), ("SRC0", 256)]),
            op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 6)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&ff1_lshl, &cs([32, 1, 1])), Ok(WaveLocal::SingleWave));
        assert_eq!(analyze(&ff1_lshl, &cs([64, 1, 1])).unwrap_err().reason, "cross-half wave op in group > 32 threads");
    }

    #[test]
    fn single_wave_rejects_above_32_threads_and_bad_masks() {
        for ops in [
            vec![op("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 1), ("SRC0", 193), ("SRC1", 128)]), op("S_ENDPGM", "ENC_SOPP", vec![])],
            vec![op("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 3), ("SRC0", 127), ("SRC1", 128)]), op("S_ENDPGM", "ENC_SOPP", vec![])],
            vec![op("S_BCNT1_I32_B64", "ENC_SOP1", vec![("SDST", 4), ("SSRC0", 126)]), op("S_ENDPGM", "ENC_SOPP", vec![])],
            vec![op("S_FF1_I32_B64", "ENC_SOP1", vec![("SDST", 5), ("SSRC0", 126)]), op("S_ENDPGM", "ENC_SOPP", vec![])],
            vec![op("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 8), ("SRC0", 256)]), op("S_ENDPGM", "ENC_SOPP", vec![])],
        ] {
            let insts = asm(&ops);
            assert_eq!(analyze(&insts, &cs([32, 1, 1])), Ok(WaveLocal::SingleWave));
            assert_eq!(analyze(&insts, &cs([64, 1, 1])).unwrap_err().reason, "cross-half wave op in group > 32 threads");
        }

        let inv_count = asm(&[
            op("V_CMP_LT_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 256), ("SRC1", 257)]),
            op("S_XOR_B64", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 0), ("SSRC1", 193)]),
            op("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 1), ("SRC0", 0), ("SRC1", 128)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        assert_eq!(analyze(&inv_count, &cs([32, 1, 1])).unwrap_err().reason, "inverting 64-bit mask op");

        for tainted in [
            asm(&[
                op("V_CMP_LT_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 256), ("SRC1", 257)]),
                op("S_MOV_B32", "ENC_SOP1", vec![("SDST", 0), ("SSRC0", 129)]),
                op("S_BCNT1_I32_B64", "ENC_SOP1", vec![("SDST", 4), ("SSRC0", 0)]),
                op("S_ENDPGM", "ENC_SOPP", vec![]),
            ]),
            asm(&[
                op("V_CMP_LT_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 256), ("SRC1", 257)]),
                op("S_MOV_B32", "ENC_SOP1", vec![("SDST", 0), ("SSRC0", 129)]),
                op("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 3), ("SRC0", 1), ("SRC1", 128)]),
                op("S_ENDPGM", "ENC_SOPP", vec![]),
            ]),
        ] {
            assert_eq!(analyze(&tainted, &cs([32, 1, 1])).unwrap_err().reason, "mask read as data");
        }
    }

    fn sample_hashes() -> Vec<u64> {
        std::env::var("XCL_SAMPLE_HASHES")
            .unwrap_or_default()
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .collect()
    }

    #[test]
    fn retail_wave_local_census() {
        let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
        if path.trim().is_empty() { return; }
        let data = std::fs::read(&path).unwrap();
        assert!(data.starts_with(pack::RPF9_MAGIC));
        let recs = pack::scan(&data, 0x40);
        let targets = sample_hashes();
        let (mut cs_total, mut cs_w32, mut cs_w64, mut ok_half, mut ok_single) = (0usize, 0, 0, 0, 0);
        let mut hist: std::collections::BTreeMap<&'static str, usize> = std::collections::BTreeMap::new();
        let mut target_rows: Vec<String> = Vec::new();
        for rec in &recs {
            let Some(blob) = pack::inflate(&data, rec) else { continue };
            let h = shader::fnv64(&blob);
            let Some(sh) = xbox::parse(&blob) else { continue };
            for f in &sh.fssm {
                let Some(super::super::Mode::Compute(cs)) = super::super::interface(&blob, &sh, f) else { continue };
                cs_total += 1;
                let Some(p) = f.programs.first() else { continue };
                let code = &f.data[p.offset..p.offset + p.len];
                if !super::super::fssm_wave64(f, code) { cs_w32 += 1; continue; }
                cs_w64 += 1;
                let (mut insts, mut off) = (Vec::new(), 0usize);
                while off + 4 <= code.len() {
                    let Ok(d) = rdna2::decode(&code[off..], off as u64) else { break };
                    off += d.len; insts.push(d);
                }
                rdna2::resolve_far_jumps(&mut insts);
                let classification = if off == code.len() {
                    analyze(&insts, &cs)
                } else {
                    Err(Reject { pc: off as u64, reason: "incomplete decode" })
                };
                let verdict = match classification {
                    Ok(WaveLocal::NoCrossHalf) => { ok_half += 1; "NoCrossHalf".to_string() }
                    Ok(WaveLocal::SingleWave) => { ok_single += 1; "SingleWave".to_string() }
                    Err(r) => { *hist.entry(r.reason).or_default() += 1; format!("Reject({} @{:#x})", r.reason, r.pc) }
                };
                if targets.contains(&h) { target_rows.push(format!("{h:016x} | threads={:?} | {verdict}", cs.threads)); }
            }
        }
        let ok_tot = ok_half + ok_single;
        eprintln!(
            "wave_local census: records={} cs_total={} wave32={} wave64={} | ok={} ({:.1}%: NoCrossHalf={}, SingleWave={})",
            recs.len(), cs_total, cs_w32, cs_w64, ok_tot,
            if cs_w64 == 0 { 0.0 } else { 100.0 * ok_tot as f64 / cs_w64 as f64 },
            ok_half, ok_single
        );
        let mut ranked: Vec<_> = hist.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        for (reason, cnt) in ranked {
            eprintln!("  reject {:>4} ({:5.1}%): {}", cnt, 100.0 * cnt as f64 / cs_w64.max(1) as f64, reason);
        }
        if !targets.is_empty() {
            target_rows.sort();
            eprintln!("target {} CS hashes matched={}/{}:", targets.len(), target_rows.len(), targets.len());
            for row in target_rows { eprintln!("  {row}"); }
        }
    }

    fn code_bytes(ops: &[Op]) -> (Vec<u8>, Vec<Decoded>) {
        let bytes: Vec<u8> = ops.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
        let (mut insts, mut off) = (Vec::new(), 0usize);
        while off + 4 <= bytes.len() {
            let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
            off += d.len;
            insts.push(d);
        }
        rdna2::resolve_far_jumps(&mut insts);
        (bytes, insts)
    }

    fn gpu_w32_ready() -> bool {
        use super::super::{dxc, runner};
        match (runner::adapter(), runner::wave32_supported(), dxc::location()) {
            (Ok(_), Ok(true), Ok(_)) => true,
            (a, w, d) => {
                runner::skip_test(if a.is_err() || w.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave32" });
                false
            }
        }
    }

    #[test]
    fn gpu_no_cross_half_matches_split_wave64() {
        if !gpu_w32_ready() { return; }
        use super::super::{dxc, lower_dxil, runner, translate, Mode, Options};
        use crate::gpu::dxil::kernel::{translate_compute_wave_local, KernelOptions};
        for threads in [[64, 1, 1], [8, 8, 1], [48, 1, 1], [128, 1, 1]] {
            let two_d = threads[1] > 1;
            let cs_i = super::super::CsInterface { threads, user_sgprs: 8, tgid: [true, false, false], tg_size: false, tid_components: if two_d { 2 } else { 1 } };
            let mut head: Vec<Op> = Vec::new();
            if two_d {
                head.push(op("V_LSHL_ADD_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 257), ("SRC1", 131), ("SRC2", 256)]));
            }
            head.extend([
                op("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 3), ("SRC0", 160), ("VSRC1", 0)]),
                op("DS_SWIZZLE_B32", "ENC_DS", vec![("VDST", 4), ("ADDR", 3), ("OFFSET0", 0x1F), ("OFFSET1", 0x04)]),
                op("V_PERMLANEX16_B32", "ENC_VOP3", vec![("VDST", 5), ("SRC0", 259), ("SRC1", 128), ("SRC2", 128)]),
                op("V_CMP_LT_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 133), ("SRC1", 256)]),
                op("S_XOR_B64", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 0), ("SSRC1", 193)]),
                op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 0)]),
            ]);
            let br_body = [op("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 4), ("SRC0", 164), ("VSRC1", 4)])];
            let br_skip = (code_bytes(&br_body).0.len() / 4) as u64;
            head.push(op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", br_skip)]));
            head.extend(br_body);
            head.extend([
                op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 2)]),
                op("S_WQM_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 126)]),
                op("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 5), ("SRC0", 144), ("VSRC1", 5)]),
                op("V_XOR_B32", "ENC_VOP2", vec![("VDST", 6), ("SRC0", 261), ("VSRC1", 4)]),
                op("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 7), ("SRC0", 130), ("VSRC1", 0)]),
                op("V_LSHL_ADD_U32", "ENC_VOP3", vec![("VDST", 7), ("SRC0", 8), ("SRC1", 137), ("SRC2", 263)]),
                op("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", 6), ("VADDR", 7), ("SRSRC", 1), ("SOFFSET", 128), ("OFFEN", 1)]),
                op("S_ENDPGM", "ENC_SOPP", vec![]),
            ]);
            let (code, insts) = code_bytes(&head);
            let split_t = translate(&code, &Options { mode: Mode::Compute(cs_i.clone()), wave64: true, split_wave64: true, wave_local: false, ..Default::default() }).unwrap();
            assert!(split_t.complete(), "{:?}", split_t.unsupported_why);
            assert!(split_t.wave_local.is_none());
            let split_dxil = dxc::compile(&split_t.hlsl, "main", "cs_6_6", &[]).unwrap();

            let wl_t = translate(&code, &Options { mode: Mode::Compute(cs_i.clone()), wave64: true, split_wave64: true, wave_local: true, ..Default::default() }).unwrap();
            assert!(wl_t.complete(), "{:?}", wl_t.unsupported_why);
            assert_eq!(wl_t.wave_local, Some(WaveLocal::NoCrossHalf));
            assert!(wl_t.hlsl.contains("[WaveSize(32)]"), "threads={threads:?}");
            assert!(!wl_t.hlsl.contains("xcl_x_meet") && !wl_t.hlsl.contains("xcl_x_ballot"), "threads={threads:?}");
            let wl_hlsl_dxil = dxc::compile(&wl_t.hlsl, "main", "cs_6_6", &[]).unwrap();
            let ko = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
            let wl_direct = translate_compute_wave_local(&code, ko, &cs_i).unwrap();
            assert_eq!(wl_direct.wave_local, Some(WaveLocal::NoCrossHalf));
            let wl_perlane = lower_dxil::compute_wave_local(&insts, &cs_i).unwrap();

            let mut io = vec![0u32; 8];
            io[4..8].copy_from_slice(&[0x100, 0, 0x1000, 0]);
            let mem = vec![0u32; 0x500 / 4];
            let expected = runner::dispatch(&split_dxil, [1, 1, 1], &io, &mem).unwrap();
            for (tag, dxil) in [("hlsl", &wl_hlsl_dxil), ("direct", &wl_direct.dxil), ("per-lane", &wl_perlane)] {
                let actual = runner::dispatch(dxil, [1, 1, 1], &io, &mem).unwrap();
                assert_eq!(actual, expected, "NoCrossHalf {tag} mismatch for threads={threads:?}");
            }
            if threads == [128, 1, 1] && std::env::var_os("XCL_SAMPLE_WAVE_LOCAL_TIMING").is_some() {
                let split_direct = crate::gpu::dxil::kernel::translate_compute(&code, ko, &cs_i).unwrap();
                let split_perlane = lower_dxil::compute_split(&insts, &cs_i).unwrap();
                let groups = [2048, 1, 1];
                let bytes = groups[0] * 512;
                io[6] = bytes;
                let mem = vec![0; ((0x100 + bytes) / 4) as usize];
                for (tag, split, local) in [
                    ("hlsl", split_dxil.as_slice(), wl_hlsl_dxil.as_slice()),
                    ("direct", split_direct.dxil.as_slice(), wl_direct.dxil.as_slice()),
                    ("per-lane", split_perlane.as_slice(), wl_perlane.as_slice()),
                ] {
                    let split_output = runner::dispatch(split, groups, &io, &mem).unwrap();
                    let local_output = runner::dispatch(local, groups, &io, &mem).unwrap();
                    assert_eq!(split_output, local_output, "profiled {tag} output");
                    let mut samples = [Vec::new(), Vec::new()];
                    for round in 0..4 {
                        for pass in 0..2 {
                            let form = (round + pass) % 2;
                            let times = runner::dispatch_timed([split, local][form], groups, &io, &mem, 12).unwrap();
                            samples[form].extend_from_slice(&times[2..]);
                        }
                    }
                    let [split_us, local_us] = samples.map(|mut times| {
                        times.sort_by(f64::total_cmp);
                        let mid = times.len() / 2;
                        500.0 * (times[mid - 1] + times[mid])
                    });
                    eprintln!("wave-local GPU {tag}: 40 samples/form, split {split_us:.3} us, local {local_us:.3} us");
                }
            }
        }
    }

    #[test]
    fn gpu_single_wave_matches_split_wave64() {
        if !gpu_w32_ready() { return; }
        use super::super::{dxc, lower_dxil, runner, translate, Mode, Options};
        use crate::gpu::dxil::kernel::{translate_compute_wave_local, KernelOptions};
        for (threads, use_cutoff) in [([32, 1, 1], true), ([16, 1, 1], false), ([24, 1, 1], false)] {
            let cs_i = super::super::CsInterface { threads, user_sgprs: 16, tgid: [true, false, false], tg_size: false, tid_components: 1 };
            let mut ops: Vec<Op> = if use_cutoff {
                vec![op("V_CMP_LE_U32", "ENC_VOPC", vec![("SRC0", 151), ("VSRC1", 0)])]
            } else {
                vec![
                    op("V_AND_B32", "ENC_VOP2", vec![("VDST", 9), ("SRC0", 129), ("VSRC1", 0)]),
                    op("V_CMP_EQ_U32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", 9)]),
                ]
            };
            ops.push(op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 2), ("SSRC0", 106)]));
            let body = [
                op("S_BCNT1_I32_B64", "ENC_SOP1", vec![("SDST", 4), ("SSRC0", 126)]),
                op("S_FF1_I32_B64", "ENC_SOP1", vec![("SDST", 5), ("SSRC0", 126)]),
                op("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 1), ("SRC0", 193), ("SRC1", 128)]),
                op("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 1), ("SRC0", 193), ("SRC1", 257)]),
                op("V_CMP_EQ_U32", "ENC_VOPC", vec![("SRC0", 5), ("VSRC1", 1)]),
                op("S_AND_SAVEEXEC_B64", "ENC_SOP1", vec![("SDST", 6), ("SSRC0", 106)]),
                op("V_MOV_B32", "ENC_VOP1", vec![("VDST", 2), ("SRC0", 4)]),
                op("V_MOV_B32", "ENC_VOP1", vec![("VDST", 8), ("SRC0", 128)]),
                op("BUFFER_ATOMIC_ADD", "ENC_MUBUF", vec![("VDATA", 2), ("VADDR", 8), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1), ("GLC", 1)]),
                op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 6)]),
                op("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 16), ("SRC0", 258)]),
                op("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 3), ("SRC0", 126), ("SRC1", 128)]),
                op("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 3), ("SRC0", 127), ("SRC1", 259)]),
                op("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 4), ("SRC0", 16), ("VSRC1", 3)]),
                op("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 4), ("SRC0", 130), ("VSRC1", 4)]),
                op("V_ADD_NC_U32", "ENC_VOP2", vec![("VDST", 9), ("SRC0", 144), ("VSRC1", 0)]),
                op("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", 9), ("VADDR", 4), ("SRSRC", 3), ("SOFFSET", 128), ("OFFEN", 1)]),
            ];
            let skip = (code_bytes(&body).0.len() / 4) as u64;
            ops.push(op("S_CBRANCH_EXECZ", "ENC_SOPP", vec![("SIMM16", skip)]));
            ops.extend(body);
            ops.extend([op("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 2)]), op("S_ENDPGM", "ENC_SOPP", vec![])]);
            let (code, insts) = code_bytes(&ops);
            assert_eq!(analyze(&insts, &cs_i), Ok(WaveLocal::SingleWave));

            let split_t = translate(&code, &Options { mode: Mode::Compute(cs_i.clone()), wave64: true, split_wave64: true, wave_local: false, ..Default::default() }).unwrap();
            assert!(split_t.complete(), "{:?}", split_t.unsupported_why);
            let split_dxil = dxc::compile(&split_t.hlsl, "main", "cs_6_6", &[]).unwrap();

            let wl_t = translate(&code, &Options { mode: Mode::Compute(cs_i.clone()), wave64: true, split_wave64: true, wave_local: true, ..Default::default() }).unwrap();
            assert!(wl_t.complete(), "{:?}", wl_t.unsupported_why);
            assert_eq!(wl_t.wave_local, Some(WaveLocal::SingleWave));
            assert!(wl_t.hlsl.contains("[WaveSize(32)]") && !wl_t.hlsl.contains("xcl_x_meet"));
            let wl_hlsl_dxil = dxc::compile(&wl_t.hlsl, "main", "cs_6_6", &[]).unwrap();
            let ko = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
            let wl_direct = translate_compute_wave_local(&code, ko, &cs_i).unwrap();
            assert_eq!(wl_direct.wave_local, Some(WaveLocal::SingleWave));
            let wl_perlane = lower_dxil::compute_wave_local(&insts, &cs_i).unwrap();

            let mut io = vec![0u32; 16];
            io[8..12].copy_from_slice(&[0x100, 0, 0x100, 0]);
            io[12..16].copy_from_slice(&[0x200, 0, 0x1000, 0]);
            let mem = vec![0u32; 0x600 / 4];
            let ref_out = runner::dispatch(&split_dxil, [1, 1, 1], &io, &mem).unwrap();
            let expected_cnt = ref_out[0x100 / 4] as usize;
            assert!(expected_cnt > 0);
            let mut expected_vals = ref_out[0x200 / 4..0x200 / 4 + expected_cnt].to_vec();
            expected_vals.sort_unstable();
            for (tag, dxil) in [("hlsl", &wl_hlsl_dxil), ("direct", &wl_direct.dxil), ("per-lane", &wl_perlane)] {
                let actual = runner::dispatch(dxil, [1, 1, 1], &io, &mem).unwrap();
                assert_eq!(actual[0x100 / 4] as usize, expected_cnt, "SingleWave {tag} count threads={threads:?}");
                let mut got = actual[0x200 / 4..0x200 / 4 + expected_cnt].to_vec();
                got.sort_unstable();
                assert_eq!(got, expected_vals, "SingleWave {tag} items threads={threads:?}");
            }
        }
    }

    #[test]
    fn gpu_rejected_shader_falls_back_to_split() {
        if !gpu_w32_ready() { return; }
        use super::super::{dxc, lower_dxil, runner, translate, Mode, Options};
        use crate::gpu::dxil::kernel::{translate_compute_wave_local, KernelOptions};
        let cs_i = cs([64, 1, 1]);
        let (code, insts) = code_bytes(&[
            op("V_READLANE_B32", "ENC_VOP3", vec![("VDST", 10), ("SRC0", 256), ("SRC1", 168)]),
            op("V_MOV_B32", "ENC_VOP1", vec![("VDST", 1), ("SRC0", 10)]),
            op("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 2), ("SRC0", 130), ("VSRC1", 0)]),
            op("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", 1), ("VADDR", 2), ("SRSRC", 1), ("SOFFSET", 128), ("OFFEN", 1)]),
            op("S_ENDPGM", "ENC_SOPP", vec![]),
        ]);
        let split_t = translate(&code, &Options { mode: Mode::Compute(cs_i.clone()), wave64: true, split_wave64: true, wave_local: false, ..Default::default() }).unwrap();
        let wl_t = translate(&code, &Options { mode: Mode::Compute(cs_i.clone()), wave64: true, split_wave64: true, wave_local: true, ..Default::default() }).unwrap();
        assert!(wl_t.wave_local.is_none());
        assert!(wl_t.hlsl.contains("xcl_x_rla_uniform"));
        assert_eq!(translate_compute_wave_local(&code, KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 }, &cs_i).unwrap().wave_local, None);
        assert_eq!(lower_dxil::compute_wave_local(&insts, &cs_i).unwrap_err().split(" @").next().unwrap(), "v_readlane/v_writelane crosses halves");
        let dx_ref = dxc::compile(&split_t.hlsl, "main", "cs_6_6", &[]).unwrap();
        let dx_wl = dxc::compile(&wl_t.hlsl, "main", "cs_6_6", &[]).unwrap();
        let mut io = vec![0u32; 8];
        io[4..8].copy_from_slice(&[0x100, 0, 0x1000, 0]);
        let mem = vec![0u32; 0x300 / 4];
        let out_ref = runner::dispatch(&dx_ref, [1, 1, 1], &io, &mem).unwrap();
        let out_wl = runner::dispatch(&dx_wl, [1, 1, 1], &io, &mem).unwrap();
        assert_eq!(out_wl, out_ref);
        assert!(out_wl[0x100 / 4..0x100 / 4 + 64].iter().all(|&x| x == 40));
    }

    #[test]
    fn retail_wave_local_bench() {
        let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
        if path.trim().is_empty() { return; }
        use super::super::{dxc, fssm_wave64, lower_dxil, translate, Mode, Options};
        use crate::gpu::dxil::kernel::{translate_compute, translate_compute_wave_local, KernelOptions};
        let data = std::fs::read(&path).unwrap();
        let targets = sample_hashes();
        let mut seen = 0usize;
        for rec in pack::scan(&data, 0x40) {
            let Some(blob) = pack::inflate(&data, &rec) else { continue };
            let h = shader::fnv64(&blob);
            if !targets.is_empty() && !targets.contains(&h) { continue; }
            let Some(sh) = xbox::parse(&blob) else { continue };
            for f in &sh.fssm {
                let Some(Mode::Compute(cs)) = super::super::interface(&blob, &sh, f) else { continue };
                let Some(p) = f.programs.first() else { continue };
                let code = &f.data[p.offset..p.offset + p.len];
                if !fssm_wave64(f, code) { continue; }
                let (mut insts, mut off) = (Vec::new(), 0usize);
                while off + 4 <= code.len() { let Ok(d) = rdna2::decode(&code[off..], off as u64) else { break }; off += d.len; insts.push(d); }
                if off != code.len() { continue; }
                rdna2::resolve_far_jumps(&mut insts);
                if targets.is_empty() && analyze(&insts, &cs).is_err() { continue; }
                let o_sp = Options { mode: Mode::Compute(cs.clone()), wave64: true, split_wave64: true, wave_local: false, ..Default::default() };
                let o_wl = Options { wave_local: true, ..o_sp.clone() };
                let (t_sp, t_wl) = (translate(code, &o_sp).unwrap(), translate(code, &o_wl).unwrap());
                let (t0, dx_sp) = (std::time::Instant::now(), dxc::compile(&t_sp.hlsl, "main", "cs_6_6", &[]).unwrap());
                let ms_sp = t0.elapsed().as_secs_f64() * 1000.0;
                let (t1, dx_wl) = (std::time::Instant::now(), dxc::compile(&t_wl.hlsl, "main", "cs_6_6", &[]).unwrap());
                let ms_wl = t1.elapsed().as_secs_f64() * 1000.0;
                let ko = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
                let k_sp = translate_compute(code, ko, &cs).map(|k| k.dxil.len());
                let k_wl = translate_compute_wave_local(code, ko, &cs).map(|k| k.dxil.len());
                let pl_sp = lower_dxil::compute_split(&insts, &cs).map(|b| b.len());
                let pl_wl = lower_dxil::compute_wave_local(&insts, &cs).map(|b| b.len());
                eprintln!(
                    "{h:016x} {:?}: HLSL {}B/{:.1}ms/{}B vs WL {}B/{:.1}ms/{}B (meet={}) | #1426 {:?} vs {:?} | per-lane {:?} vs {:?}",
                    t_wl.wave_local, t_sp.hlsl.len(), ms_sp, dx_sp.len(), t_wl.hlsl.len(), ms_wl, dx_wl.len(),
                    t_wl.hlsl.contains("xcl_x_meet"), k_sp, k_wl, pl_sp, pl_wl
                );
                if targets.is_empty() {
                    seen += 1;
                    if seen >= 4 { return; }
                }
            }
        }
    }
}
