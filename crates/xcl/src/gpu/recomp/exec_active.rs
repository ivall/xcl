//! Forward dataflow analysis proving where `EXEC` equals its entry value (all executing lanes
//! active).
//!
//! At entry of every non-kernel shader (`Vertex`, `Pixel`, `Compute`, `Mesh`, `Geometry`), the
//! emitter sets `exec = ballot(true)`, so every hardware lane executing the shader has its EXEC
//! bit set. In straight-line code and outside divergent `s_and_saveexec` / `v_cmpx` regions,
//! `EXEC` stays equal to that entry mask – or is restored to it by `s_mov_b64 exec, s_saved` /
//! `s_or_b64 exec, exec, s_saved` from a register that saved the entry mask.
//!
//! Where `all_active_pcs` includes `pc`, every executing lane is guaranteed active:
//! - VGPR writes emit `v = val;` instead of `v = select(lanebit(exec_lo, exec_hi), val, v);`,
//!   removing both the per-instruction `select`/`lanebit` IR and the false read dependency on
//!   the old VGPR value (which otherwise extends live ranges across DXC and the GPU driver).
//! - Lane-mask writes emit `ballot(cond)` instead of `ballot(and(lanebit(...), cond))`.
//!
//! Conservative where it cannot prove: `Mode::Kernel` starts with an arbitrary `EXEC` from `io`
//! (never all-active at entry), any non-fetch indirect jump makes the whole program conservative,
//! and every `LOOP_GUARD` exit out of a loop is treated as an incoming edge at the loop's exit.

use std::collections::{BTreeMap, BTreeSet};

use super::emit::ext_encoding;
use crate::gpu::rdna2::{self, Decoded, Variant, F_CONDITIONAL, F_INDIRECT, F_TERMINATOR};

type Regs = u128;

const VCC: Regs = 3 << 106;
const M0: Regs = 1 << 124;
const EXEC: Regs = 3 << 126;
const ALL: Regs = !0;

fn range(first: u64, n: u64) -> Regs {
    (first..(first + n).min(128)).fold(0, |m, r| m | 1u128 << r)
}

fn scalar_code(v: u64, n: u64) -> Regs {
    match v {
        125 => 0,
        v if v <= 127 => range(v, n),
        _ => 0,
    }
}

/// Superset of SGPRs (0..=127) that `d` may write. Over-approximating is always sound for a
/// must-analysis: at worst it drops a saved-EXEC register or marks EXEC unknown.
fn written_sgprs(d: &Decoded) -> Regs {
    let name = d.inst.name;
    if name.starts_with("S_MOVREL") {
        return ALL & !EXEC;
    }
    let mut written: Regs = 0;
    if d.variant == Variant::Sdwa {
        let Some(e) = ext_encoding(d) else { return ALL };
        if d.enc.name == "ENC_VOPC" && !name.starts_with("V_CMPX") {
            let fv = |n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
            written |= if fv("SD") == 1 { scalar_code(fv("SDST"), 2) } else { VCC };
        }
    }
    if name.starts_with("V_CMPX") || name.contains("SAVEEXEC") || name.contains("WREXEC") {
        written |= EXEC;
    }
    for o in d.ienc.operands {
        if !o.output {
            continue;
        }
        let n = (o.size as u64).div_ceil(32).max(1);
        let regs = match (o.field, o.ty) {
            (_, "OPR_VCC") => VCC,
            (_, "OPR_EXEC" | "OPR_SDST_EXEC") => EXEC,
            (_, "OPR_SDST_M0") => M0,
            (None, _) | (Some("LITERAL"), _) => continue,
            (Some(f), ty) => {
                if !["SREG", "SSRC", "SDST", "OPR_SRC"].iter().any(|k| ty.contains(k)) {
                    continue;
                }
                scalar_code(rdna2::field_value(d.enc, &d.words, f).unwrap_or(0) * rdna2::register_scale(f), n)
            }
        };
        written |= regs;
    }
    written
}

/// Clear any saved-entry bit whose register (or high half in wave64) overlaps `written`.
fn kill_saved(saved: Regs, written: Regs, wave64: bool) -> Regs {
    let mask = if wave64 { written | (written >> 1) } else { written };
    saved & !mask
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct State {
    /// `exec` is guaranteed to equal `entry_exec` (every executing lane active).
    exec_entry: bool,
    /// Bit `r` (0..=105, 108..=123) is 1 iff `s[r]` (wave32) or `s[r..r+2]` (wave64) is
    /// guaranteed to hold `entry_exec`.
    saved: Regs,
}

impl State {
    fn meet(self, other: Self) -> Self {
        Self { exec_entry: self.exec_entry && other.exec_entry, saved: self.saved & other.saved }
    }

    fn is_entry_src(self, code: u64) -> bool {
        if code == 126 {
            self.exec_entry
        } else if code <= 105 || (108..=123).contains(&code) {
            (self.saved & (1u128 << code)) != 0
        } else {
            false
        }
    }
}

fn is_normal_sgpr(code: u64, wave64: bool) -> bool {
    if wave64 {
        code <= 104 || (108..=122).contains(&code)
    } else {
        code <= 105 || (108..=123).contains(&code)
    }
}

fn step(st: State, d: &Decoded, wave64: bool) -> State {
    let name = d.inst.name;
    let written = written_sgprs(d);
    let mut saved = kill_saved(st.saved, written, wave64);
    let mut exec_entry = if written & EXEC != 0 { false } else { st.exec_entry };

    let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f).unwrap_or(0) * rdna2::register_scale(f);
    let save_bit = |r: u64| if is_normal_sgpr(r, wave64) { 1u128 << r } else { 0 };

    if wave64 {
        match name {
            n if n.ends_with("_SAVEEXEC_B64") => {
                if st.exec_entry {
                    saved |= save_bit(fv("SDST"));
                }
                exec_entry = false;
            }
            "S_MOV_B64" => {
                let (dst, src) = (fv("SDST"), fv("SSRC0"));
                if dst == 126 {
                    exec_entry = st.is_entry_src(src);
                } else if st.is_entry_src(src) {
                    saved |= save_bit(dst);
                }
            }
            "S_OR_B64" if fv("SDST") == 126 => {
                exec_entry = st.is_entry_src(fv("SSRC0")) || st.is_entry_src(fv("SSRC1"));
            }
            "S_WQM_B64" if fv("SDST") == 126 && fv("SSRC0") == 126 => {
                exec_entry = st.exec_entry;
            }
            _ => {}
        }
    } else {
        match name {
            n if n.ends_with("_SAVEEXEC_B32") => {
                if st.exec_entry {
                    saved |= save_bit(fv("SDST"));
                }
                exec_entry = false;
            }
            "S_MOV_B32" => {
                let (dst, src) = (fv("SDST"), fv("SSRC0"));
                if dst == 126 {
                    exec_entry = st.is_entry_src(src);
                } else if st.is_entry_src(src) {
                    saved |= save_bit(dst);
                }
            }
            "S_OR_B32" if fv("SDST") == 126 => {
                exec_entry = st.is_entry_src(fv("SSRC0")) || st.is_entry_src(fv("SSRC1"));
            }
            "S_WQM_B32" if fv("SDST") == 126 && fv("SSRC0") == 126 => {
                exec_entry = st.exec_entry;
            }
            _ => {}
        }
    }

    State { exec_entry, saved }
}

/// Set of instruction PCs where `EXEC` is guaranteed equal to `entry_exec` (all executing lanes
/// active). Empty when `insts` is empty or contains an unhandled indirect branch.
pub fn all_active_pcs(insts: &[Decoded], wave64: bool) -> BTreeSet<u64> {
    if insts.is_empty() {
        return BTreeSet::new();
    }
    let pcs: BTreeMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); insts.len()];
    for (i, d) in insts.iter().enumerate() {
        let f = d.inst.flags;
        if f & F_INDIRECT != 0 {
            // `s_swappc_b64 s[n:n+1], s[n:n+1]` is the ignored fetch-shader call (falls through)
            let is_fetch_call = d.inst.name == "S_SWAPPC_B64"
                && rdna2::field_value(d.enc, &d.words, "SDST") == rdna2::field_value(d.enc, &d.words, "SSRC0");
            if !is_fetch_call {
                return BTreeSet::new();
            }
            if i + 1 < insts.len() {
                succ[i].push(i + 1);
            }
            continue;
        }
        if let Some(t) = rdna2::branch_target(d) {
            if let Some(&j) = pcs.get(&t) {
                succ[i].push(j);
            }
            if f & F_CONDITIONAL != 0 && i + 1 < insts.len() {
                succ[i].push(i + 1);
            }
        } else if f & F_TERMINATOR == 0 && i + 1 < insts.len() {
            succ[i].push(i + 1);
        }
    }
    // `LOOP_GUARD` runs at the top of each loop `[a, b)` (before the instruction at `a`) and
    // breaks directly to `b`, so `in_state[head]` flows into `in_state[exit]`.
    let mut loop_guard_edges: Vec<(usize, usize)> = Vec::new();
    for (a, b) in super::loop_ranges(insts) {
        if let (Some(&head_i), Some((_, &exit_j))) = (pcs.get(&a), pcs.range(b..).next()) {
            loop_guard_edges.push((head_i, exit_j));
        }
    }

    let mut in_state: Vec<Option<State>> = vec![None; insts.len()];
    in_state[0] = Some(State { exec_entry: true, saved: 0 });

    loop {
        let mut changed = false;
        for i in 0..insts.len() {
            let Some(st) = in_state[i] else { continue };
            let out = step(st, &insts[i], wave64);
            for &j in &succ[i] {
                let next = match in_state[j] {
                    None => out,
                    Some(prev) => prev.meet(out),
                };
                if in_state[j] != Some(next) {
                    in_state[j] = Some(next);
                    changed = true;
                }
            }
        }
        for &(head_i, exit_j) in &loop_guard_edges {
            let Some(st) = in_state[head_i] else { continue };
            let next = match in_state[exit_j] {
                None => st,
                Some(prev) => prev.meet(st),
            };
            if in_state[exit_j] != Some(next) {
                in_state[exit_j] = Some(next);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    insts
        .iter()
        .enumerate()
        .filter_map(|(i, d)| in_state[i].filter(|s| s.exec_entry).map(|_| d.pc))
        .collect()
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

    const V: u64 = 256;
    type I = (&'static str, &'static str, &'static [(&'static str, u64)]);

    fn asm(lines: &[I]) -> Vec<Decoded> {
        let bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
        program(&bytes)
    }

    #[test]
    fn straight_line_without_exec_writes_is_all_active() {
        let p = asm(&[
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 0), ("SRC0", V + 1), ("VSRC1", 2)]),
            ("V_MUL_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V), ("VSRC1", 1)]),
            ("S_ENDPGM", "ENC_SOPP", &[("SIMM16", 0)]),
        ]);
        let active = all_active_pcs(&p, true);
        assert_eq!(active.len(), p.len());
    }

    #[test]
    fn saveexec_narrows_and_or_exec_restores_all_active() {
        let p = asm(&[
            // 0: all active
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 0), ("SRC0", V + 1), ("VSRC1", 2)]),
            // 1: s_and_saveexec_b64 s[4:5], vcc  -> saves entry exec in s[4:5], narrows exec
            ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)]),
            // 2: NOT all active (inside divergent region)
            ("V_MUL_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V), ("VSRC1", 1)]),
            // 3: s_or_b64 exec, exec, s[4:5] -> restores entry exec
            ("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 4)]),
            // 4: all active again!
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 3), ("VSRC1", 1)]),
            ("S_ENDPGM", "ENC_SOPP", &[("SIMM16", 0)]),
        ]);
        let active = all_active_pcs(&p, true);
        assert!(active.contains(&p[0].pc));
        assert!(active.contains(&p[1].pc));
        assert!(!active.contains(&p[2].pc), "inside saveexec region must NOT be all-active");
        assert!(!active.contains(&p[3].pc), "restore instruction itself still sees narrowed exec");
        assert!(active.contains(&p[4].pc), "after s_or_b64 exec, s_saved must be all-active again");
    }

    #[test]
    fn clobbering_high_half_of_saved_pair_prevents_false_restore() {
        let p = asm(&[
            // 0: save entry exec into s[4:5]
            ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)]),
            // 1: clobber s5 (high half of s[4:5])!
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 128)]),
            // 2: s_mov_b64 exec, s[4:5] -> s[4:5] is NO LONGER entry_exec
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 4)]),
            // 3: must NOT be considered all-active
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 0), ("SRC0", V + 1), ("VSRC1", 2)]),
            ("S_ENDPGM", "ENC_SOPP", &[("SIMM16", 0)]),
        ]);
        let active = all_active_pcs(&p, true);
        assert!(active.contains(&p[0].pc));
        assert!(!active.contains(&p[3].pc), "clobbered high half must invalidate saved pair");
    }

    #[test]
    fn wave32_saveexec_and_mov_restore() {
        let p = asm(&[
            ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)]),
            ("V_MUL_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V), ("VSRC1", 2)]),
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 4)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 1), ("VSRC1", 2)]),
            ("S_ENDPGM", "ENC_SOPP", &[("SIMM16", 0)]),
        ]);
        let active = all_active_pcs(&p, false);
        assert!(active.contains(&p[0].pc));
        assert!(!active.contains(&p[1].pc));
        assert!(!active.contains(&p[2].pc));
        assert!(active.contains(&p[3].pc));
    }
}
