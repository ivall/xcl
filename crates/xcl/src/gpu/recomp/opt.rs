//! Program-level static analysis and optimization metadata for the RDNA2 → HLSL recompiler.
//!
//! Computes PC-keyed liveness, dead-flag / dead-carry / dead-instruction facts, exact CFG
//! dominators, CFG-wide SGPR reaching definitions & constant values, memory-hazard reachability,
//! dominator-hoisted / CSE'd invariant `s_load_*` & `s_buffer_load_*` groups (with known-zero-hi
//! Window-0 fast path), universal non-`Kernel` known-zero-hi `s_load_*` / `s_buffer_load_*` sets,
//! non-consecutive intra-block `s_buffer_load_*` grouping, Wave32 & native-Wave64 lane-local
//! `bool` mask scalarization, intra-block constant propagation, and conditional
//! `globallycoherent` eligibility.
//!
//! Soundness invariants enforced throughout:
//! 1. **VGPR backward liveness under divergent `EXEC`**: every VGPR write is modeled as a
//!    read-modify-write (`kill_v = EMPTY`), so dual-mask `if/else` writes never lose liveness.
//! 2. **Dominator-hoisted `s_load_*` (`HoistedSmemGroup`) & `s_buffer_load_*` (`HoistedSbufGroup`)**:
//!    - Requires unique dominating reaching definitions (`(def_lo, def_hi)` for `s[b:b+1]`, or
//!      `(def_d0, def_d1, def_d2)` for `s[b..b+3]`) and zero writes to those base SGPRs on ANY
//!      CFG path (including `LOOP_GUARD` exits & loop cycles) from hoist point `H` to every use `U`.
//!    - Requires zero memory hazards (stores, atomics, `s_barrier`, cache inv/wb, `s_sendmsg`,
//!      calls/indirect branches, `unknown_succ`) on ANY CFG path from `H` to every use `U`,
//!      including loop-carried paths re-entering `U`.
//!    - Never hoists or CSEs `GLC`/`DLC` loads.
//!    - `known_hi == Some(0)` ONLY when the high base bits are statically proven `0` on all
//!      reaching paths (`s[b+1] == 0` for `s_load_*`, `(s[b+1] & 0xFFFF) == 0` for
//!      `s_buffer_load_*`), AND obeying the 4 GiB wrap guard: in `Mode::Kernel`, ONLY `s_load_*`
//!      with `min_off == 0 && max_end <= 4`; in non-`Kernel` modes, full 21-bit SMEM range
//!      (`[-1_048_576, 1_048_576]`; valid UMA allocations never straddle 4 GiB).
//!    - Sub-range subsumption on `s_load_*` only when `parent.offset >= 0` (preventing
//!      negative-offset underflow mismatch).
//! 3. **Intra-block `s_buffer_load_*` (`IntraBlockSbufGroup`)**: constant-offset (`soff == None`)
//!    only, `max_end - min_off <= 1_048_576` (and `max_span == 0` in `Mode::Kernel`), disjoint
//!    from `HoistedSbufGroup`, splitting epochs on any memory hazard, `GLC`/`DLC` SMEM op, or
//!    write to `sbase..sbase+3`; never uses `[loop]` or wrap-fallback switch duplication.
//! 4. **Lane-local `bool` masks (`lane_local_*`) — Convergence & Hazard Model**:
//!    - **Strictly intra-block (zero ballot motion)**: never moves, hoists, sinks, or deduplicates
//!      any `WaveActiveBallot` across basic-block boundaries, branches, merges, or loops; every
//!      scalarized def-use DAG is contained inside one basic block and must be dead at block exit
//!      (`!live_out_s[bi]`).
//!    - **Zero cross-lane observation (convergence-free replacement)**: disabled in `Mode::Kernel`
//!      and `split_wave64`; scalarizes `V_CMP_*` (non-`CMPX`), `V_ADD/SUB*_CO*_U32`, and dead-`scc`
//!      bitwise SALU chains (`S_AND/OR/XOR/NOT/ANDN2/ORN2/NAND/NOR/XNOR/MOV`) only when every
//!      consumer reads solely its own lane bit (`V_CNDMASK_B32` mask slot, `V_ADD/SUB*_CO_CI_U32`
//!      carry-in slot, or another valid lane-local bitwise op). Any whole-mask consumer
//!      (`S_CBRANCH_VCC*`, `S_BCNT*`, `S_FF*`, `EXEC` updates, scalar ALU data operands, memory
//!      operands, or unknown/indirect ops) rejects the entire chain. Because lane `i` only ever
//!      reads bit `i` of its own lane's ballot, `lanebit(WaveActiveBallot(and(act, cond)))` is
//!      identically `and(act, cond)` per lane with zero cross-lane convergence semantics to preserve.
//!    - **Divergent `EXEC` & widening guard (`needs_exec_mask`)**: if any `EXEC`-widening op
//!      (`is_exec_widening`, e.g., `s_mov`/`s_or`/`s_xor`/`s_not`/`s_nand`/`s_nor`/`s_xnor` to
//!      `exec`, `s_wqm`, `s_andn2_saveexec`) executes between a def and its last consumer in the
//!      block, `needs_exec_mask = true` captures `and(act, cond)` at the def site so lanes disabled
//!      at def remain `false` after `EXEC` is later restored.
//! 5. **Dead `vcc`/carry `sdst`**: for `wave64`, both `n` and `n+1` must be dead; for wave32,
//!    only `n` must be dead; `Null` (`125` or `> 127`) is always dead. Carry-in reads on
//!    `v_add_co_ci*`/`v_sub*_co_ci*` are always preserved.
//! 6. **`Mode::Kernel` epilogue**: all SGPRs, VGPRs, and `scc` are seeded live at `s_endpgm`.
//! 7. **Unknown / indirect ops**: conservatively mark all registers live/read and restrict
//!    hoisting to intra-block only.
//! 8. **Conditional `mem_coherent`**: false only when zero `GLC`/`DLC`/`SLC`, zero memory
//!    atomics, zero cache inv/wb ops, and `!(has_s_barrier && has_mem_store)`.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use super::emit::ext_encoding;
use super::{Mode, Options};
use crate::gpu::rdna2::{self, Decoded, Operand, Variant, F_CONDITIONAL, F_INDIRECT, F_TERMINATOR};

/// SGPR bitset: bit `n` = `s<n>` (0..=105 plain, 106/107 VCC, 108..=123 ttmp, 124 M0, 126/127 EXEC).
type SRegs = u128;

const S_VCC_LO: SRegs = 1u128 << 106;
const S_VCC: SRegs = 3u128 << 106;
const S_M0: SRegs = 1u128 << 124;
const S_EXEC: SRegs = 3u128 << 126;
const S_ALL: SRegs = !0u128;
const S_PLAIN_MASK: SRegs = (1u128 << 106) - 1;

#[inline]
fn sreg_range(first: u64, n: u64) -> SRegs {
    if first >= 128 || n == 0 {
        return 0;
    }
    let end = (first + n).min(128);
    let mut m = 0u128;
    for r in first..end {
        if r != 125 {
            m |= 1u128 << r;
        }
    }
    m
}

/// 256-bit VGPR set (`v0..=v255`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct VRegs([u64; 4]);

impl VRegs {
    const EMPTY: Self = Self([0; 4]);
    const ALL: Self = Self([!0u64; 4]);

    #[inline]
    fn set(&mut self, r: u64) {
        if r < 256 {
            self.0[(r >> 6) as usize] |= 1u64 << (r & 63);
        }
    }

    #[inline]
    fn set_range(&mut self, first: u64, n: u64) {
        let end = (first + n).min(256);
        for r in first..end {
            self.set(r);
        }
    }

    #[inline]
    fn union(self, other: Self) -> Self {
        Self([
            self.0[0] | other.0[0],
            self.0[1] | other.0[1],
            self.0[2] | other.0[2],
            self.0[3] | other.0[3],
        ])
    }

    #[inline]
    fn intersect(self, other: Self) -> Self {
        Self([
            self.0[0] & other.0[0],
            self.0[1] & other.0[1],
            self.0[2] & other.0[2],
            self.0[3] & other.0[3],
        ])
    }

    #[inline]
    fn is_empty(self) -> bool {
        (self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0
    }
}

/// One deduplicated load entry inside a `HoistedSmemGroup` or `IntraBlockSbufGroup`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HoistedLoad {
    /// Dword-aligned byte offset relative to `sbase` (signed 21-bit for `s_load_*`,
    /// unsigned 21-bit for `s_buffer_load_*`).
    pub offset: i64,
    /// Number of dwords loaded (`1`, `2`, `4`, `8`, or `16`).
    pub count: u8,
}

/// Dominator-hoisted / CSE'd group of invariant `s_load_dword*` ops sharing the exact same
/// 64-bit base pointer `s[sbase:sbase+1]` in a memory-hazard-free dominated region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HoistedSmemGroup {
    /// Unique group id (`0..N`) used for HLSL temp naming (`sm_g{id}_{uid}`).
    pub id: u32,
    /// Block `start_pc` of the dominator hoist block `H`.
    #[allow(dead_code)]
    pub hoist_block: u64,
    /// Base SGPR pair start index (`s[sbase:sbase+1]`, always `< 105`).
    pub sbase: u8,
    /// Statically proven `Some(0)` ONLY when `s[sbase+1] == 0` on all reaching paths AND
    /// wrap-safe (`Mode::Kernel`: `min_off == 0 && max_end <= 4`; non-`Kernel`: `[-1_048_576, 1_048_576]`).
    /// Note: ONLY `hi == 0` maps unconditionally to Window 0 regardless of runtime `xcl_uma_base`.
    pub known_hi: Option<u32>,
    /// Minimum byte offset across all `loads`.
    #[allow(dead_code)]
    pub min_off: i64,
    /// Maximum end byte offset (`offset + 4 * count`) across all `loads`.
    #[allow(dead_code)]
    pub max_end: i64,
    /// Deduplicated loads in this group (`1..=96`), sorted by `(offset, count)`.
    pub loads: Vec<HoistedLoad>,
}

/// Maps a dominated `s_load_dword*` PC to its pre-loaded temp inside a `HoistedSmemGroup`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SmemUse {
    /// `HoistedSmemGroup::id`.
    pub group_id: u32,
    /// Index into `HoistedSmemGroup::loads` (`0..loads.len()`).
    pub load_idx: u16,
    /// `HoistedSmemGroup::loads[load_idx].count` (`1`, `2`, `4`, `8`, or `16`).
    pub parent_count: u8,
    /// Sub-dword offset inside the parent load (`0` for exact match; `> 0` if subsumed).
    pub dword_offset: u8,
    /// Number of dwords to copy into `s[sdata..sdata+count]`.
    pub count: u8,
}

/// Dominator-hoisted / CSE'd cross-block group of invariant constant-offset (`SOFFSET == 125`)
/// `s_buffer_load_dword*` ops sharing the exact same V# descriptor dwords `s[sbase..sbase+3]`
/// (`d0, d1, d2`) in a memory-hazard-free dominated region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HoistedSbufGroup {
    /// Unique group id (`0..N`) used for HLSL temp naming (`sbh_g{id}_{uid}`).
    pub id: u32,
    /// Block `start_pc` of the dominator hoist block `H`.
    #[allow(dead_code)]
    pub hoist_block: u64,
    /// Base 4-SGPR descriptor start index (`s[sbase..sbase+3]` read; always `sbase + 4 <= 106`).
    pub sbase: u8,
    /// Statically proven `Some(0)` ONLY when non-`Kernel` AND `(const_d1 & 0xFFFF) == 0`.
    pub known_hi: Option<u32>,
    /// Minimum unsigned byte offset across all `loads`.
    #[allow(dead_code)]
    pub min_off: i64,
    /// Maximum end byte offset (`offset + 4 * count`) across all `loads`.
    #[allow(dead_code)]
    pub max_end: i64,
    /// Deduplicated loads in this group (`1..=96`), sorted by `(offset, count)`.
    pub loads: Vec<HoistedLoad>,
}

/// Non-consecutive intra-block group of constant-offset (`soff == None`) `s_buffer_load_dword*`
/// ops sharing the same V# descriptor `s[sbase..sbase+3]` within a hazard-free, base-unmodified
/// epoch of one basic block. Emitted at `first_pc` under ONE zero-loop switch (or 0 switches when
/// `known_hi == Some(0)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IntraBlockSbufGroup {
    /// Unique group id (`0..N`) used for HLSL temp naming (`sb_g{id}_{uid}`).
    pub id: u32,
    /// PC of the first `s_buffer_load_*` instruction in this group (where the group executes).
    pub first_pc: u64,
    /// Base 4-SGPR descriptor start index (`s[sbase..sbase+4]`, always `<= 102`).
    pub sbase: u8,
    /// Shared dynamic SGPR offset (guaranteed `None` in `opt_v3` to avoid switch-doubling fallbacks).
    pub soff: Option<u8>,
    /// Statically proven `Some(0)` ONLY when non-`Kernel` AND `(const_d1 & 0xFFFF) == 0`.
    pub known_hi: Option<u32>,
    /// Minimum unsigned byte offset across all `loads`.
    #[allow(dead_code)]
    pub min_off: i64,
    /// Maximum end byte offset (`offset + 4 * count`) across all `loads`.
    #[allow(dead_code)]
    pub max_end: i64,
    /// Deduplicated loads in this group (`1..=96`), sorted by `(offset, count)`.
    pub loads: Vec<HoistedLoad>,
}

/// Maps a member `s_buffer_load_dword*` PC to its pre-loaded temp inside a `HoistedSbufGroup`
/// or `IntraBlockSbufGroup`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SbufUse {
    /// `IntraBlockSbufGroup::id`.
    pub group_id: u32,
    /// Index into `IntraBlockSbufGroup::loads` (`0..loads.len()`).
    pub load_idx: u16,
    /// `IntraBlockSbufGroup::loads[load_idx].count` (`1`, `2`, `4`, `8`, or `16`).
    pub parent_count: u8,
    /// Sub-dword offset inside the parent load (`0` for exact match; `> 0` if subsumed).
    pub dword_offset: u8,
    /// Number of dwords to copy into `s[sdata..sdata+count]`.
    pub count: u8,
}

/// Kind of VALU/VOPC instruction defining a lane-local `bool` mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum LaneLocalDefKind {
    Cmp,
    CarryOut,
}

/// Metadata for a `V_CMP_*` or `V_ADD/SUB*_CO*_U32` whose mask output is strictly lane-local.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LaneLocalDef {
    /// Unique `bool ml_{id}` variable identifier (declared inline at this def PC).
    pub id: u32,
    /// `true` iff an `EXEC`-widening op intervenes before a consumer (`and(act, (cond))`);
    /// `false` => bare `(cond)` with zero `WaveActiveBallot` and zero `lanebit(exec)`.
    pub needs_exec_mask: bool,
    #[allow(dead_code)]
    pub kind: LaneLocalDefKind,
}

/// Operand of a lane-local bitwise SALU instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneLocalOperand {
    /// Another lane-local `bool ml_{id}` defined earlier in the same block.
    Local(u32),
    /// Current lane's `EXEC` bit (`act`).
    Exec,
    /// Constant `false` (`0`) or `true` (`-1` / all-1s).
    Const(bool),
    /// Plain SGPR or VCC mask register (`lanebit(s{r}, 0u)` in wave32, `lanebit(s{r}, s{r+1})` in wave64).
    Sgpr(u8),
}

/// Logical operation for a lane-local bitwise SALU instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneLocalBitOp {
    Mov,
    Not,
    And,
    Or,
    Xor,
    Andn2,
    Orn2,
    Nand,
    Nor,
    Xnor,
}

/// Metadata for a dead-`scc` bitwise SALU instruction operating on lane-local `bool` masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LaneLocalBitwise {
    /// Unique `bool ml_{dst_id}` variable identifier defined by this instruction.
    pub dst_id: u32,
    pub op: LaneLocalBitOp,
    pub src0: LaneLocalOperand,
    /// `None` for `Mov` / `Not`, `Some` for binary bitwise ops.
    pub src1: Option<LaneLocalOperand>,
}

/// Program-level analysis results consumed by `emit.rs` and `mod.rs`.
#[derive(Debug)]
pub(crate) struct ProgramOpt {
    /// Guardrail #8: true iff Compute `mem..mem3` MUST keep `globallycoherent`.
    /// False ONLY when ALL 4 hold: zero `GLC`/`DLC`/`SLC` ops, zero memory atomics,
    /// zero cache-invalidate/writeback ops, and `!(has_s_barrier && has_mem_store)`.
    pub mem_coherent: bool,
    /// Weapon 1A: all dominator-hoisted / CSE'd invariant `s_load_*` groups.
    hoisted_smem_groups: Vec<HoistedSmemGroup>,
    hoisted_smem_at_entry: HashMap<u64, Vec<usize>>,
    hoisted_smem_before: HashMap<u64, Vec<usize>>,
    hoisted_smem_after: HashMap<u64, Vec<usize>>,
    smem_uses: HashMap<u64, SmemUse>,
    /// Weapon 1A-sbuf: cross-block dominator-hoisted / CSE'd invariant `s_buffer_load_*` groups.
    hoisted_sbuf_groups: Vec<HoistedSbufGroup>,
    hoisted_sbuf_at_entry: HashMap<u64, Vec<usize>>,
    hoisted_sbuf_before: HashMap<u64, Vec<usize>>,
    hoisted_sbuf_after: HashMap<u64, Vec<usize>>,
    hoisted_sbuf_uses: HashMap<u64, SbufUse>,
    /// Universal non-`Kernel` `s_load_dword*` PCs with statically proven `s[sbase+1] == 0`.
    known_zero_hi_sload: HashSet<u64>,
    /// Universal non-`Kernel` `s_buffer_load_dword*` PCs with statically proven `(s[sbase+1] & 0xFFFF) == 0`.
    known_zero_hi_sbuf: HashSet<u64>,
    /// Weapon 1B: non-consecutive intra-block `s_buffer_load_*` groups keyed by `first_pc`.
    intra_block_sbuf_groups: HashMap<u64, IntraBlockSbufGroup>,
    sbuf_uses: HashMap<u64, SbufUse>,
    /// Weapon 3: Wave32 & native-Wave64 lane-local `bool` mask scalarization.
    lane_local_defs: HashMap<u64, LaneLocalDef>,
    lane_local_bitwise: HashMap<u64, LaneLocalBitwise>,
    lane_local_uses: HashMap<u64, u32>,
    dead_scc: HashSet<u64>,
    dead_carry_sdst: HashSet<u64>,
    dead_inst: HashSet<u64>,
    const_srcs: HashMap<u64, [Option<u32>; 3]>,
    const_sgprs: HashMap<(u64, u8), u32>,
}

impl Default for ProgramOpt {
    fn default() -> Self {
        Self {
            mem_coherent: true,
            hoisted_smem_groups: Vec::new(),
            hoisted_smem_at_entry: HashMap::new(),
            hoisted_smem_before: HashMap::new(),
            hoisted_smem_after: HashMap::new(),
            smem_uses: HashMap::new(),
            hoisted_sbuf_groups: Vec::new(),
            hoisted_sbuf_at_entry: HashMap::new(),
            hoisted_sbuf_before: HashMap::new(),
            hoisted_sbuf_after: HashMap::new(),
            hoisted_sbuf_uses: HashMap::new(),
            known_zero_hi_sload: HashSet::new(),
            known_zero_hi_sbuf: HashSet::new(),
            intra_block_sbuf_groups: HashMap::new(),
            sbuf_uses: HashMap::new(),
            lane_local_defs: HashMap::new(),
            lane_local_bitwise: HashMap::new(),
            lane_local_uses: HashMap::new(),
            dead_scc: HashSet::new(),
            dead_carry_sdst: HashSet::new(),
            dead_inst: HashSet::new(),
            const_srcs: HashMap::new(),
            const_sgprs: HashMap::new(),
        }
    }
}

impl ProgramOpt {
    /// True iff `scc` written by the instruction at `pc` is overwritten on all paths before any read.
    pub(crate) fn is_dead_scc(&self, pc: u64) -> bool {
        self.dead_scc.contains(&pc)
    }

    /// True iff carry/mask `sdst` of the instruction at `pc` is `Null` or provably dead before any read.
    pub(crate) fn is_dead_carry_sdst(&self, pc: u64) -> bool {
        self.dead_carry_sdst.contains(&pc)
    }

    /// True iff the instruction at `pc` is a pure register-only ALU/mov/cmp with no live outputs or side effects.
    pub(crate) fn is_dead_inst(&self, pc: u64) -> bool {
        self.dead_inst.contains(&pc)
    }

    /// Known 32-bit raw bit value of input operand `idx` (`0..3`, in `Em::ops` input order) at `pc`.
    pub(crate) fn const_src(&self, pc: u64, idx: usize) -> Option<u32> {
        if idx >= 3 {
            return None;
        }
        self.const_srcs.get(&pc).and_then(|arr| arr[idx])
    }

    /// Known 32-bit raw bit value of SGPR `sgpr` (read by the instruction at `pc`) from constant propagation.
    #[allow(dead_code)]
    pub(crate) fn const_sgpr(&self, pc: u64, sgpr: u64) -> Option<u32> {
        if sgpr >= 106 {
            return None;
        }
        self.const_sgprs.get(&(pc, sgpr as u8)).copied()
    }

    /// All dominator-hoisted / CSE'd invariant `s_load_*` groups (for declaring `sm_g{id}_{uid}` at `main()` scope).
    pub(crate) fn hoisted_smem_groups(&self) -> &[HoistedSmemGroup] {
        &self.hoisted_smem_groups
    }

    /// Indices into `hoisted_smem_groups()` to emit at the entry of block `start_pc` (before block instructions).
    pub(crate) fn hoisted_smem_at_block_entry(&self, start_pc: u64) -> &[usize] {
        self.hoisted_smem_at_entry.get(&start_pc).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Indices into `hoisted_smem_groups()` to emit immediately BEFORE instruction `pc` (even if `pc` is `dead_inst`).
    pub(crate) fn hoisted_smem_before_pc(&self, pc: u64) -> &[usize] {
        self.hoisted_smem_before.get(&pc).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Indices into `hoisted_smem_groups()` to emit immediately AFTER instruction `pc` (never on a terminator).
    pub(crate) fn hoisted_smem_after_pc(&self, pc: u64) -> &[usize] {
        self.hoisted_smem_after.get(&pc).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Dominated `s_load_*` instruction replaced by a copy from a hoisted `sm_g{group_id}_{load_idx}` temp.
    pub(crate) fn smem_use(&self, pc: u64) -> Option<SmemUse> {
        self.smem_uses.get(&pc).copied()
    }

    /// True iff `pc` is a non-`Kernel` `s_load_dword*` whose high base SGPR `s[sbase+1]` is
    /// statically proven `0` (and offset is within the safe UMA Window-0 fast-path range).
    pub(crate) fn is_known_zero_hi_sload(&self, pc: u64) -> bool {
        self.known_zero_hi_sload.contains(&pc)
    }

    /// True iff `pc` is a non-`Kernel` `s_buffer_load_dword*` whose descriptor word 1 has
    /// statically proven `(s[sbase+1] & 0xFFFF) == 0` (Window-0 fast path on UMA).
    pub(crate) fn is_known_zero_hi_sbuf(&self, pc: u64) -> bool {
        self.known_zero_hi_sbuf.contains(&pc)
    }

    /// All dominator-hoisted / CSE'd cross-block invariant `s_buffer_load_*` groups (for declaring
    /// `sbh_g{id}_{uid}` at `main()` scope).
    pub(crate) fn hoisted_sbuf_groups(&self) -> &[HoistedSbufGroup] {
        &self.hoisted_sbuf_groups
    }

    /// Indices into `hoisted_sbuf_groups()` to emit at the entry of block `start_pc`.
    pub(crate) fn hoisted_sbuf_at_block_entry(&self, start_pc: u64) -> &[usize] {
        self.hoisted_sbuf_at_entry.get(&start_pc).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Indices into `hoisted_sbuf_groups()` to emit immediately BEFORE instruction `pc`.
    pub(crate) fn hoisted_sbuf_before_pc(&self, pc: u64) -> &[usize] {
        self.hoisted_sbuf_before.get(&pc).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Indices into `hoisted_sbuf_groups()` to emit immediately AFTER instruction `pc` (never on a terminator).
    pub(crate) fn hoisted_sbuf_after_pc(&self, pc: u64) -> &[usize] {
        self.hoisted_sbuf_after.get(&pc).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Dominated `s_buffer_load_*` instruction replaced by a copy from a hoisted `sbh_g{group_id}_{load_idx}` temp.
    pub(crate) fn hoisted_sbuf_use(&self, pc: u64) -> Option<SbufUse> {
        self.hoisted_sbuf_uses.get(&pc).copied()
    }

    /// Non-consecutive intra-block `s_buffer_load_*` group starting at `first_pc`, if any.
    pub(crate) fn intra_block_sbuf_group(&self, first_pc: u64) -> Option<&IntraBlockSbufGroup> {
        self.intra_block_sbuf_groups.get(&first_pc)
    }

    /// Member `s_buffer_load_*` instruction replaced by a copy from `sb_g{group_id}_{load_idx}`.
    pub(crate) fn sbuf_use(&self, pc: u64) -> Option<SbufUse> {
        self.sbuf_uses.get(&pc).copied()
    }

    /// Lane-local `bool ml_{id}` definition for a `V_CMP_*` or `V_ADD/SUB*_CO*_U32` at `pc`.
    pub(crate) fn lane_local_def(&self, pc: u64) -> Option<LaneLocalDef> {
        self.lane_local_defs.get(&pc).copied()
    }

    /// Lane-local `bool ml_{dst_id}` bitwise SALU op at `pc`.
    pub(crate) fn lane_local_bitwise(&self, pc: u64) -> Option<LaneLocalBitwise> {
        self.lane_local_bitwise.get(&pc).copied()
    }

    /// Lane-local `bool ml_{id}` mask consumer (`V_CNDMASK_B32` or `V_ADD/SUB*_CO_CI_U32`) at `pc`.
    pub(crate) fn lane_local_use(&self, pc: u64) -> Option<u32> {
        self.lane_local_uses.get(&pc).copied()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CarryDst {
    Null,
    Sgpr(u64),
}

#[derive(Clone, Debug)]
struct InstInfo {
    s_read: SRegs,
    s_kill: SRegs,
    s_mod: SRegs,
    scc_read: bool,
    scc_write: bool,
    v_read: VRegs,
    v_mod: VRegs,
    carry_sdst: Option<CarryDst>,
    pure_eliminable: bool,
    exec_widen: bool,
    indirect_sgpr: bool,
    indirect_vgpr: bool,
}

#[inline]
fn fval(d: &Decoded, field: &str) -> u64 {
    rdna2::field_value(d.enc, &d.words, field).unwrap_or(0)
}

#[inline]
fn scaled_field(d: &Decoded, field: &str) -> u64 {
    fval(d, field) * rdna2::register_scale(field)
}

/// Same explicit-operand partition as `Em::ops(d)` in `emit.rs`.
fn explicit_ops(d: &Decoded) -> (Vec<&'static Operand>, Vec<&'static Operand>) {
    let mut outs = Vec::new();
    let mut ins = Vec::new();
    for o in d.ienc.operands {
        if o.implicit && !(o.field.is_none() && o.ty == "OPR_VCC") {
            continue;
        }
        if o.output {
            outs.push(o);
        } else if o.input {
            ins.push(o);
        }
    }
    (outs, ins)
}

fn scalar_code_regs(code: u64, n: u64, wave64: bool) -> (SRegs, bool) {
    match code {
        251 => (if wave64 { S_VCC } else { S_VCC_LO }, false),
        252 => (S_EXEC, false),
        253 => (0, true),
        125 => (0, false),
        v if v <= 127 => (sreg_range(v, n), false),
        _ => (0, false),
    }
}

fn inline_const_bits(code: u64, literal: Option<u32>, f16: bool) -> Option<u32> {
    Some(match code {
        125 => 0,
        128..=192 => (code - 128) as u32,
        193..=208 => (-((code - 192) as i32)) as u32,
        235..=238 => 0,
        239 => 0x7FFF_FFFF,
        240..=248 => {
            if f16 {
                match code {
                    240 => 0x3800,
                    241 => 0xB800,
                    242 => 0x3C00,
                    243 => 0xBC00,
                    244 => 0x4000,
                    245 => 0xC000,
                    246 => 0x4400,
                    247 => 0xC400,
                    248 => 0x3118,
                    _ => return None,
                }
            } else {
                match code {
                    240 => 0.5f32.to_bits(),
                    241 => (-0.5f32).to_bits(),
                    242 => 1.0f32.to_bits(),
                    243 => (-1.0f32).to_bits(),
                    244 => 2.0f32.to_bits(),
                    245 => (-2.0f32).to_bits(),
                    246 => 4.0f32.to_bits(),
                    247 => (-4.0f32).to_bits(),
                    248 => 0.159_154_94f32.to_bits(),
                    _ => return None,
                }
            }
        }
        255 => literal?,
        _ => return None,
    })
}

fn is_valid_cmp_suffix(s: &str) -> bool {
    matches!(
        s,
        "eq_i32"
            | "lg_i32"
            | "gt_i32"
            | "ge_i32"
            | "lt_i32"
            | "le_i32"
            | "eq_u32"
            | "lg_u32"
            | "gt_u32"
            | "ge_u32"
            | "lt_u32"
            | "le_u32"
    )
}

fn salu_writes_scc(enc: &str, name: &str) -> bool {
    if enc == "ENC_SOPC" {
        return true;
    }
    if enc == "ENC_SOPK" {
        return name == "S_ADDK_I32" || name.starts_with("S_CMPK_");
    }
    if name.contains("SAVEEXEC") || name.contains("WREXEC") {
        return true;
    }
    matches!(
        name,
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

fn is_supported_salu(enc: &str, name: &str) -> bool {
    match enc {
        "ENC_SOPK" => {
            matches!(name, "S_MOVK_I32" | "S_CMOVK_I32" | "S_ADDK_I32" | "S_MULK_I32")
                || name
                    .strip_prefix("S_CMPK_")
                    .is_some_and(|rest| is_valid_cmp_suffix(&rest.to_ascii_lowercase()))
        }
        "ENC_SOPC" => {
            matches!(
                name,
                "S_BITCMP0_B64" | "S_BITCMP1_B64" | "S_BITCMP0_B32" | "S_BITCMP1_B32" | "S_CMP_EQ_U64" | "S_CMP_LG_U64"
            ) || name
                .strip_prefix("S_CMP_")
                .is_some_and(|rest| is_valid_cmp_suffix(&rest.to_ascii_lowercase()))
        }
        "ENC_SOP1" | "ENC_SOP2" => matches!(
            name,
            "S_MOV_B32"
                | "S_CMOV_B32"
                | "S_NOT_B32"
                | "S_BREV_B32"
                | "S_BCNT0_I32_B32"
                | "S_BCNT1_I32_B32"
                | "S_FF0_I32_B32"
                | "S_FF1_I32_B32"
                | "S_FLBIT_I32_B32"
                | "S_FLBIT_I32"
                | "S_ABS_I32"
                | "S_SEXT_I32_I8"
                | "S_SEXT_I32_I16"
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
                | "S_MUL_I32"
                | "S_MUL_HI_U32"
                | "S_MUL_HI_I32"
                | "S_LSHL_B32"
                | "S_LSHR_B32"
                | "S_ASHR_I32"
                | "S_LSHL1_ADD_U32"
                | "S_LSHL2_ADD_U32"
                | "S_LSHL3_ADD_U32"
                | "S_LSHL4_ADD_U32"
                | "S_BFM_B32"
                | "S_BFE_U32"
                | "S_BFE_I32"
                | "S_MIN_U32"
                | "S_MAX_U32"
                | "S_MIN_I32"
                | "S_MAX_I32"
                | "S_CSELECT_B32"
                | "S_WQM_B32"
                | "S_BITSET0_B32"
                | "S_BITSET1_B32"
                | "S_ABSDIFF_I32"
                | "S_PACK_LL_B32_B16"
                | "S_PACK_LH_B32_B16"
                | "S_PACK_HH_B32_B16"
                | "S_GETPC_B64"
                | "S_MOV_B64"
                | "S_CMOV_B64"
                | "S_NOT_B64"
                | "S_CSELECT_B64"
                | "S_WQM_B64"
                | "S_BREV_B64"
                | "S_BCNT1_I32_B64"
                | "S_BCNT0_I32_B64"
                | "S_FF1_I32_B64"
                | "S_FF0_I32_B64"
                | "S_FLBIT_I32_B64"
                | "S_FLBIT_I32_I64"
                | "S_LSHL_B64"
                | "S_LSHR_B64"
                | "S_ASHR_I64"
                | "S_BFM_B64"
                | "S_BFE_U64"
                | "S_BFE_I64"
                | "S_BITSET0_B64"
                | "S_BITSET1_B64"
                | "S_AND_B64"
                | "S_OR_B64"
                | "S_XOR_B64"
                | "S_ANDN2_B64"
                | "S_ORN2_B64"
                | "S_NAND_B64"
                | "S_NOR_B64"
                | "S_XNOR_B64"
        ),
        _ => false,
    }
}

fn is_supported_pure_valu(d: &Decoded) -> bool {
    if d.variant != Variant::Plain {
        return false;
    }
    let name = d.inst.name;
    if name.starts_with("V_CMP_") {
        return true;
    }
    if name.starts_with("V_CMPX_")
        || name.starts_with("V_INTERP_")
        || name.starts_with("V_MOVREL")
        || name.starts_with("V_PERMLANE")
        || matches!(
            name,
            "V_NOP"
                | "V_READLANE_B32"
                | "V_READFIRSTLANE_B32"
                | "V_WRITELANE_B32"
                | "V_SWAPREL_B32"
                | "V_DIV_SCALE_F32"
        )
    {
        return false;
    }
    matches!(
        name,
        "V_MOV_B32"
            | "V_MOV_FED_B32"
            | "V_NOT_B32"
            | "V_BFREV_B32"
            | "V_FFBH_U32"
            | "V_FFBL_B32"
            | "V_FFBH_I32"
            | "V_CVT_F32_I32"
            | "V_CVT_F32_U32"
            | "V_CVT_U32_F32"
            | "V_CVT_I32_F32"
            | "V_CVT_RPI_I32_F32"
            | "V_CVT_FLR_I32_F32"
            | "V_CVT_F32_UBYTE0"
            | "V_CVT_F32_UBYTE1"
            | "V_CVT_F32_UBYTE2"
            | "V_CVT_F32_UBYTE3"
            | "V_CVT_F32_F16"
            | "V_CVT_F16_F32"
            | "V_CVT_PKRTZ_F16_F32"
            | "V_PACK_B32_F16"
            | "V_FMA_MIX_F32"
            | "V_FMA_MIXLO_F16"
            | "V_FMA_MIXHI_F16"
            | "V_RCP_F32"
            | "V_RCP_IFLAG_F32"
            | "V_RSQ_F32"
            | "V_SQRT_F32"
            | "V_EXP_F32"
            | "V_LOG_F32"
            | "V_SIN_F32"
            | "V_COS_F32"
            | "V_FRACT_F32"
            | "V_TRUNC_F32"
            | "V_FLOOR_F32"
            | "V_CEIL_F32"
            | "V_RNDNE_F32"
            | "V_FREXP_MANT_F32"
            | "V_FREXP_EXP_I32_F32"
            | "V_LDEXP_F32"
            | "V_ADD_F32"
            | "V_SUB_F32"
            | "V_SUBREV_F32"
            | "V_MUL_F32"
            | "V_MUL_LEGACY_F32"
            | "V_MIN_F32"
            | "V_MAX_F32"
            | "V_FMAC_F32"
            | "V_MAC_F32"
            | "V_FMAC_LEGACY_F32"
            | "V_FMA_F32"
            | "V_MAD_F32"
            | "V_FMA_LEGACY_F32"
            | "V_FMAMK_F32"
            | "V_FMAAK_F32"
            | "V_MADMK_F32"
            | "V_MADAK_F32"
            | "V_MIN3_F32"
            | "V_MAX3_F32"
            | "V_MED3_F32"
            | "V_CUBEID_F32"
            | "V_CUBESC_F32"
            | "V_CUBETC_F32"
            | "V_CUBEMA_F32"
            | "V_ADD_NC_U32"
            | "V_SUB_NC_U32"
            | "V_SUBREV_NC_U32"
            | "V_ADD_NC_I32"
            | "V_SUB_NC_I32"
            | "V_MUL_LO_U32"
            | "V_MUL_HI_U32"
            | "V_MUL_HI_I32"
            | "V_MUL_U32_U24"
            | "V_MUL_HI_U32_U24"
            | "V_MUL_I32_I24"
            | "V_MUL_HI_I32_I24"
            | "V_MAD_U32_U24"
            | "V_MAD_I32_I24"
            | "V_AND_B32"
            | "V_OR_B32"
            | "V_XOR_B32"
            | "V_XNOR_B32"
            | "V_LSHLREV_B32"
            | "V_LSHRREV_B32"
            | "V_ASHRREV_I32"
            | "V_LSHL_ADD_U32"
            | "V_ADD_LSHL_U32"
            | "V_LSHL_OR_B32"
            | "V_AND_OR_B32"
            | "V_OR3_B32"
            | "V_XOR3_B32"
            | "V_ADD3_U32"
            | "V_XAD_U32"
            | "V_BFE_U32"
            | "V_BFE_I32"
            | "V_BFI_B32"
            | "V_BFM_B32"
            | "V_BCNT_U32_B32"
            | "V_PERM_B32"
            | "V_ALIGNBIT_B32"
            | "V_ALIGNBYTE_B32"
            | "V_MIN_U32"
            | "V_MAX_U32"
            | "V_MIN_I32"
            | "V_MAX_I32"
            | "V_MIN3_U32"
            | "V_MAX3_U32"
            | "V_MED3_U32"
            | "V_MIN3_I32"
            | "V_MAX3_I32"
            | "V_MED3_I32"
            | "V_SAD_U32"
            | "V_SAD_U8"
            | "V_SAD_HI_U8"
            | "V_SAD_U16"
            | "V_MSAD_U8"
            | "V_LERP_U8"
            | "V_MAD_U32_U16"
            | "V_MAD_I32_I16"
            | "V_CVT_PK_U16_U32"
            | "V_CVT_PK_I16_I32"
            | "V_CVT_PKNORM_I16_F32"
            | "V_CVT_PKNORM_U16_F32"
            | "V_CVT_PKNORM_I16_F16"
            | "V_CVT_PKNORM_U16_F16"
            | "V_SAT_PK_U8_I16"
            | "V_DIV_FMAS_F32"
            | "V_DIV_FIXUP_F32"
            | "V_CVT_PK_U8_F32"
            | "V_MBCNT_LO_U32_B32"
            | "V_MBCNT_HI_U32_B32"
            | "V_CNDMASK_B32"
            | "V_ADD_CO_U32"
            | "V_SUB_CO_U32"
            | "V_SUBREV_CO_U32"
            | "V_ADD_CO_CI_U32"
            | "V_SUB_CO_CI_U32"
            | "V_SUBREV_CO_CI_U32"
            | "V_SWAP_B32"
            | "V_LSHLREV_B64"
            | "V_LSHRREV_B64"
            | "V_ASHRREV_I64"
            | "V_MAD_U64_U32"
            | "V_MAD_I64_I32"
            | "V_ADD_F64"
            | "V_MUL_F64"
            | "V_FMA_F64"
            | "V_MIN_F64"
            | "V_MAX_F64"
            | "V_RCP_F64"
            | "V_CVT_F64_F32"
            | "V_CVT_F64_I32"
            | "V_CVT_F64_U32"
            | "V_FLOOR_F64"
            | "V_CEIL_F64"
            | "V_FRACT_F64"
            | "V_RNDNE_F64"
            | "V_FREXP_MANT_F64"
            | "V_SQRT_F64"
            | "V_RSQ_F64"
            | "V_TRUNC_F64"
            | "V_LDEXP_F64"
            | "V_CVT_F32_F64"
            | "V_CVT_I32_F64"
            | "V_CVT_U32_F64"
            | "V_FREXP_EXP_I32_F64"
            | "V_ADD_F16"
            | "V_SUB_F16"
            | "V_SUBREV_F16"
            | "V_MUL_F16"
            | "V_MIN_F16"
            | "V_MAX_F16"
            | "V_FMA_F16"
            | "V_MED3_F16"
            | "V_MIN3_F16"
            | "V_MAX3_F16"
            | "V_MED3_U16"
            | "V_MIN3_U16"
            | "V_MAX3_U16"
            | "V_MED3_I16"
            | "V_MIN3_I16"
            | "V_MAX3_I16"
            | "V_FREXP_MANT_F16"
            | "V_FREXP_EXP_I16_F16"
            | "V_MAD_F16"
            | "V_FMAC_F16"
            | "V_FMAMK_F16"
            | "V_FMAAK_F16"
            | "V_LDEXP_F16"
            | "V_RCP_F16"
            | "V_RSQ_F16"
            | "V_SQRT_F16"
            | "V_LOG_F16"
            | "V_EXP_F16"
            | "V_SIN_F16"
            | "V_COS_F16"
            | "V_FRACT_F16"
            | "V_FLOOR_F16"
            | "V_CEIL_F16"
            | "V_TRUNC_F16"
            | "V_RNDNE_F16"
            | "V_CVT_F16_U16"
            | "V_CVT_F16_I16"
            | "V_CVT_U16_F16"
            | "V_CVT_I16_F16"
            | "V_CVT_NORM_U16_F16"
            | "V_CVT_NORM_I16_F16"
            | "V_ADD_NC_U16"
            | "V_ADD_U16"
            | "V_SUB_NC_U16"
            | "V_SUB_U16"
            | "V_SUBREV_NC_U16"
            | "V_SUBREV_U16"
            | "V_ADD_NC_I16"
            | "V_SUB_NC_I16"
            | "V_MUL_LO_U16"
            | "V_LSHLREV_B16"
            | "V_LSHRREV_B16"
            | "V_ASHRREV_I16"
            | "V_MAX_U16"
            | "V_MIN_U16"
            | "V_MAX_I16"
            | "V_MIN_I16"
            | "V_MAD_U16"
            | "V_MAD_I16"
            | "V_DOT2_F32_F16"
            | "V_DOT2C_F32_F16"
            | "V_DOT4C_I32_I8"
            | "V_DOT2_I32_I16"
            | "V_DOT2_U32_U16"
            | "V_DOT4_U32_U8"
            | "V_DOT4_I32_I8"
            | "V_DOT8_U32_U4"
            | "V_DOT8_I32_I4"
            | "V_PK_ADD_F16"
            | "V_PK_MUL_F16"
            | "V_PK_FMA_F16"
            | "V_PK_FMAC_F16"
            | "V_PK_MIN_F16"
            | "V_PK_MAX_F16"
            | "V_PK_ADD_U16"
            | "V_PK_SUB_U16"
            | "V_PK_ADD_I16"
            | "V_PK_SUB_I16"
            | "V_PK_MUL_LO_U16"
            | "V_PK_MAD_U16"
            | "V_PK_MAD_I16"
            | "V_PK_LSHLREV_B16"
            | "V_PK_LSHRREV_B16"
            | "V_PK_ASHRREV_I16"
            | "V_PK_MAX_U16"
            | "V_PK_MIN_U16"
            | "V_PK_MAX_I16"
            | "V_PK_MIN_I16"
    )
}

fn is_rmw_or_partial_vgpr_write(d: &Decoded) -> bool {
    let name = d.inst.name;
    if d.variant != Variant::Plain {
        // SDWA preserves unused bytes/words; DPP16 row/bank mask or !bound_ctrl preserves lanes.
        return true;
    }
    if matches!(
        name,
        "V_FMAC_F32"
            | "V_MAC_F32"
            | "V_FMAC_LEGACY_F32"
            | "V_FMAC_F16"
            | "V_PK_FMAC_F16"
            | "V_DOT2C_F32_F16"
            | "V_DOT4C_I32_I8"
            | "V_FMA_MIXLO_F16"
            | "V_FMA_MIXHI_F16"
            | "V_CVT_F16_F32"
            | "V_INTERP_P2_F16"
            | "V_PERMLANE16_B32"
            | "V_PERMLANEX16_B32"
            | "V_WRITELANE_B32"
            | "V_SWAP_B32"
    ) {
        return true;
    }
    // Plain 16-bit VALU (`valu16` in emit.rs) preserves the untouched 16-bit half of VDST.
    const WIDE16: &[&str] = &[
        "V_CVT_PKNORM_I16_F32",
        "V_CVT_PKNORM_U16_F32",
        "V_CVT_PKNORM_I16_F16",
        "V_CVT_PKNORM_U16_F16",
        "V_CVT_PK_U16_U32",
        "V_CVT_PK_I16_I32",
        "V_MAD_U32_U16",
        "V_MAD_I32_I16",
        "V_SAD_U16",
        "V_SAT_PK_U8_I16",
        "V_CVT_F32_F16",
        "V_CVT_PKRTZ_F16_F32",
        "V_PACK_B32_F16",
    ];
    if (name.contains("F16") || name.contains("_I16") || name.contains("_U16") || name.contains("_B16"))
        && !name.starts_with("V_PK_")
        && !name.starts_with("V_DOT")
        && !name.starts_with("V_CMP")
        && !name.starts_with("V_FMA_MIX")
        && !WIDE16.contains(&name)
        && (d.enc.name.starts_with("ENC_VOP") || d.enc.name.starts_with("VOP"))
    {
        return true;
    }
    // D16 sub-dword loads preserve the untouched half of the destination VGPR.
    if name.contains("_D16_") && (name.contains("_LOAD_") || name.contains("_READ_")) {
        return true;
    }
    false
}

/// Returns true if an `EXEC`-modifying instruction can WIDEN `EXEC` (turn on previously inactive lanes).
fn is_exec_widening(d: &Decoded, s_mod: SRegs) -> bool {
    if (s_mod & S_EXEC) == 0 {
        return false;
    }
    let name = d.inst.name;
    // V_CMPX_* computes `ballot(act & cond)` — always a subset of current EXEC.
    if name.starts_with("V_CMPX_") {
        return false;
    }
    // `and_saveexec` / `and_wrexec` compute `src & exec`; `andn1_saveexec/wrexec` compute `~src & exec`.
    // Note: `andn2_saveexec` computes `src & ~exec`, which WIDENS EXEC!
    if matches!(
        name,
        "S_AND_SAVEEXEC_B32"
            | "S_AND_SAVEEXEC_B64"
            | "S_AND_WREXEC_B32"
            | "S_AND_WREXEC_B64"
            | "S_ANDN1_SAVEEXEC_B32"
            | "S_ANDN1_SAVEEXEC_B64"
            | "S_ANDN1_WREXEC_B32"
            | "S_ANDN1_WREXEC_B64"
    ) {
        return false;
    }
    if matches!(name, "S_AND_B32" | "S_AND_B64") {
        let s0 = fval(d, "SSRC0");
        let s1 = fval(d, "SSRC1");
        // Strictly narrowing iff one operand is EXEC (126).
        return s0 != 126 && s1 != 126;
    }
    if matches!(name, "S_ANDN2_B32" | "S_ANDN2_B64") {
        // `s_andn2` computes `SSRC0 & ~SSRC1`: narrowing iff SSRC0 == EXEC (126).
        return fval(d, "SSRC0") != 126;
    }
    true
}

/// Extract carry/mask `sdst` for instructions that emit `write_mask(&sdst, ...)` in `emit.rs`.
fn carry_sdst_of(d: &Decoded, outs: &[&Operand]) -> Option<CarryDst> {
    let name = d.inst.name;
    let is_carry_alu = matches!(
        name,
        "V_ADD_CO_U32"
            | "V_SUB_CO_U32"
            | "V_SUBREV_CO_U32"
            | "V_ADD_CO_CI_U32"
            | "V_SUB_CO_CI_U32"
            | "V_SUBREV_CO_CI_U32"
            | "V_MAD_U64_U32"
            | "V_MAD_I64_I32"
            | "V_DIV_SCALE_F32"
    );
    if is_carry_alu {
        let Some(o) = outs.get(1) else {
            return Some(if matches!(name, "V_MAD_U64_U32" | "V_MAD_I64_I32") {
                CarryDst::Null
            } else {
                CarryDst::Sgpr(106)
            });
        };
        return Some(match o.field {
            Some(f) => {
                let v = scaled_field(d, f);
                if v == 125 || v > 127 {
                    CarryDst::Null
                } else {
                    CarryDst::Sgpr(v)
                }
            }
            None => match o.ty {
                "OPR_VCC" => CarryDst::Sgpr(106),
                "OPR_EXEC" => CarryDst::Sgpr(126),
                _ => CarryDst::Null,
            },
        });
    }
    if name.starts_with("V_CMP_") {
        if d.variant == Variant::Sdwa {
            if let Some(e) = ext_encoding(d) {
                if rdna2::field_value(e, &d.words, "SD") == Some(1) {
                    let sd = rdna2::field_value(e, &d.words, "SDST").unwrap_or(106);
                    return Some(if sd == 125 || sd > 127 {
                        CarryDst::Null
                    } else {
                        CarryDst::Sgpr(sd)
                    });
                }
            }
            return Some(CarryDst::Sgpr(106));
        }
        let Some(o) = outs.first() else {
            return Some(CarryDst::Sgpr(106));
        };
        return Some(match o.field {
            Some(f) => {
                let v = scaled_field(d, f);
                if v == 125 || v > 127 {
                    CarryDst::Null
                } else {
                    CarryDst::Sgpr(v)
                }
            }
            None => match o.ty {
                "OPR_VCC" => CarryDst::Sgpr(106),
                "OPR_EXEC" => CarryDst::Sgpr(126),
                _ => CarryDst::Null,
            },
        });
    }
    None
}

fn analyze_inst(d: &Decoded, opt: &Options) -> InstInfo {
    let name = d.inst.name;
    let enc = d.enc.name;
    let wave64 = opt.wave64;
    let mask_words: u64 = if wave64 { 2 } else { 1 };

    let mut s_read: SRegs = 0;
    let mut s_kill: SRegs = 0;
    let mut s_mod: SRegs = 0;
    let mut scc_read = false;
    let mut scc_write = false;
    let mut v_read = VRegs::EMPTY;
    let mut v_mod = VRegs::EMPTY;

    let (outs, _ins) = explicit_ops(d);
    let carry_sdst = carry_sdst_of(d, &outs);

    // 1. Indirect SGPR moves (`s_movrel*`): conservative all-SGPR read, zero kill.
    if name.starts_with("S_MOVREL") {
        let dst = fval(d, "SDST").min(123);
        let mod_mask = if name.starts_with("S_MOVRELS_") {
            sreg_range(dst, if name.ends_with("_B64") { 2 } else { 1 })
        } else {
            sreg_range(dst, 124u64.saturating_sub(dst))
        };
        return InstInfo {
            s_read: S_ALL,
            s_kill: 0,
            s_mod: mod_mask,
            scc_read: false,
            scc_write: false,
            v_read: VRegs::EMPTY,
            v_mod: VRegs::EMPTY,
            carry_sdst: None,
            pure_eliminable: false,
            exec_widen: false,
            indirect_sgpr: true,
            indirect_vgpr: false,
        };
    }

    // 2. Indirect VGPR moves (`v_movrel*` / `v_swaprel*`): conservative all-VGPR read/mod.
    if name.starts_with("V_MOVREL") || name == "V_SWAPREL_B32" {
        s_read |= S_M0 | S_EXEC;
        let src = fval(d, "SRC0");
        if src < 256 {
            let (sr, sc) = scalar_code_regs(src, 1, wave64);
            s_read |= sr;
            scc_read |= sc;
        }
        return InstInfo {
            s_read,
            s_kill: 0,
            s_mod: 0,
            scc_read,
            scc_write: false,
            v_read: VRegs::ALL,
            v_mod: VRegs::ALL,
            carry_sdst: None,
            pure_eliminable: false,
            exec_widen: false,
            indirect_sgpr: false,
            indirect_vgpr: true,
        };
    }

    // 3. SOPP special instructions.
    if enc == "ENC_SOPP" {
        match name {
            "S_CBRANCH_SCC0" | "S_CBRANCH_SCC1" => scc_read = true,
            "S_CBRANCH_VCCZ" | "S_CBRANCH_VCCNZ" => s_read |= if wave64 { S_VCC } else { S_VCC_LO },
            "S_CBRANCH_EXECZ" | "S_CBRANCH_EXECNZ" => s_read |= S_EXEC,
            "S_SENDMSG" if matches!(opt.mode, Mode::Mesh(_)) => s_read |= S_M0,
            "S_SENDMSG" if matches!(opt.mode, Mode::Geometry(_)) => s_read |= S_EXEC,
            _ => {}
        }
        return InstInfo {
            s_read,
            s_kill: 0,
            s_mod: 0,
            scc_read,
            scc_write: false,
            v_read: VRegs::EMPTY,
            v_mod: VRegs::EMPTY,
            carry_sdst: None,
            pure_eliminable: false,
            exec_widen: false,
            indirect_sgpr: false,
            indirect_vgpr: false,
        };
    }

    // 4. `s_setreg_b32` reads `SDST` (field named SDST in SOPK is actually a source!).
    if name == "S_SETREG_B32" {
        let (sr, sc) = scalar_code_regs(fval(d, "SDST"), 1, wave64);
        s_read |= sr;
        scc_read |= sc;
        return InstInfo {
            s_read,
            s_kill: 0,
            s_mod: 0,
            scc_read,
            scc_write: false,
            v_read: VRegs::EMPTY,
            v_mod: VRegs::EMPTY,
            carry_sdst: None,
            pure_eliminable: false,
            exec_widen: false,
            indirect_sgpr: false,
            indirect_vgpr: false,
        };
    }

    // 5. Indirect branches / calls (Guardrail #7): conservatively read everything.
    if (name == "S_SETPC_B64" && d.far_target.is_none() && !matches!(opt.mode, Mode::Geometry(_)))
        || name.starts_with("S_CALL")
        || (name == "S_SWAPPC_B64" && fval(d, "SDST") != fval(d, "SSRC0"))
    {
        return InstInfo {
            s_read: S_ALL,
            s_kill: 0,
            s_mod: 0,
            scc_read: true,
            scc_write: false,
            v_read: VRegs::ALL,
            v_mod: VRegs::EMPTY,
            carry_sdst: None,
            pure_eliminable: false,
            exec_widen: false,
            indirect_sgpr: true,
            indirect_vgpr: true,
        };
    }

    // 6. Standard operand-table traversal for SGPRs & VGPRs.
    let keeps_s_dst = name.starts_with("S_CMOV")
        || matches!(
            name,
            "S_ADDK_I32" | "S_MULK_I32" | "S_BITSET0_B32" | "S_BITSET1_B32" | "S_BITSET0_B64" | "S_BITSET1_B64"
        );
    let is_mask_write_op = carry_sdst.is_some() || name.starts_with("V_CMPX_");
    let is_per_lane_mask_read = matches!(
        name,
        "V_CNDMASK_B32" | "V_ADD_CO_CI_U32" | "V_SUB_CO_CI_U32" | "V_SUBREV_CO_CI_U32"
    );

    // SDWA extension dword handling (S0, S1, SD/SDST).
    let mut sdwa_vopc_dst: Option<SRegs> = None;
    if d.variant == Variant::Sdwa {
        if let Some(e) = ext_encoding(d) {
            let fv = |n: &str| rdna2::field_value(e, &d.words, n).unwrap_or(0);
            if fv("S0") == 1 {
                let (sr, sc) = scalar_code_regs(fv("VSRC0"), 1, wave64);
                s_read |= sr;
                scc_read |= sc;
            } else {
                v_read.set(fv("VSRC0"));
            }
            if d.enc.field("VSRC1").is_some() {
                if fv("S1") == 1 {
                    let (sr, sc) = scalar_code_regs(fv("VSRC1"), 1, wave64);
                    s_read |= sr;
                    scc_read |= sc;
                } else {
                    v_read.set(fv("VSRC1"));
                }
            }
            if enc == "ENC_VOPC" && !name.starts_with("V_CMPX") {
                sdwa_vopc_dst = Some(if fv("SD") == 1 {
                    scalar_code_regs(fv("SDST"), mask_words, wave64).0
                } else {
                    sreg_range(106, mask_words)
                });
            }
        } else {
            // Unknown SDWA encoding: conservative fallback.
            return InstInfo {
                s_read: S_ALL,
                s_kill: 0,
                s_mod: 0,
                scc_read: true,
                scc_write: false,
                v_read: VRegs::ALL,
                v_mod: VRegs::EMPTY,
                carry_sdst: None,
                pure_eliminable: false,
                exec_widen: false,
                indirect_sgpr: false,
                indirect_vgpr: false,
            };
        }
    } else if d.variant == Variant::Dpp16 || d.variant == Variant::Dpp8 {
        if let Some(e) = ext_encoding(d) {
            let v0 = rdna2::field_value(e, &d.words, "VSRC0").unwrap_or(0);
            v_read.set(v0);
        } else {
            v_read = VRegs::ALL;
        }
        s_read |= S_EXEC;
    }

    // Iterate spec operands for SGPRs.
    for o in d.ienc.operands {
        if o.ty == "OPR_SSRC_SPECIAL_SCC" {
            if o.input {
                scc_read = true;
            }
            continue;
        }
        let n_base = (o.size as u64).div_ceil(32).max(1);
        let (regs, reads_scc) = match (o.field, o.ty) {
            (_, "OPR_VCC") => {
                let n = if is_mask_write_op || is_per_lane_mask_read {
                    mask_words
                } else {
                    n_base
                };
                (sreg_range(106, n), false)
            }
            (_, "OPR_EXEC" | "OPR_SDST_EXEC") => (S_EXEC, false),
            (_, "OPR_SDST_M0") => (S_M0, false),
            (None, _) | (Some("LITERAL"), _) => continue,
            (Some(f), ty) => {
                if o.implicit
                    || !["SREG", "SSRC", "SDST", "OPR_SRC", "SMEM_OFFSET"]
                        .iter()
                        .any(|k| ty.contains(k))
                {
                    continue;
                }
                if f == "SRC0" && d.variant != Variant::Plain {
                    continue;
                }
                if f == "VSRC1" && d.variant == Variant::Sdwa {
                    continue;
                }
                let raw = scaled_field(d, f);
                let n = if o.output && is_mask_write_op && (f == "SDST" || (f == "VDST" && name.starts_with("V_CMP"))) {
                    mask_words
                } else if !o.output && is_per_lane_mask_read && f == "SRC2" {
                    mask_words
                } else {
                    n_base
                };
                scalar_code_regs(raw, n, wave64)
            }
        };
        if reads_scc && !o.output {
            scc_read = true;
        }
        if o.output {
            s_mod |= regs;
            if keeps_s_dst {
                s_read |= regs;
            } else {
                s_kill |= regs;
            }
        }
        if o.input {
            s_read |= regs;
        }
    }

    if let Some(dst_regs) = sdwa_vopc_dst {
        s_kill = (s_kill & !S_VCC) | dst_regs;
        s_mod = (s_mod & !S_VCC) | dst_regs;
    }

    // SOPK: `SDST` is read by `s_cmovk`, `s_addk`, `s_mulk`, `s_cmpk_*`.
    if enc == "ENC_SOPK" {
        let sd = fval(d, "SDST");
        let r = scalar_code_regs(sd, 1, wave64).0;
        if name.starts_with("S_CMPK_") {
            s_read |= r;
            s_kill &= !r;
            s_mod &= !r;
        } else if name != "S_MOVK_I32" {
            s_read |= r;
            s_kill &= !r;
            s_mod |= r;
        }
    }

    // Saveexec / wrexec always read and write EXEC and write SCC.
    if name.contains("SAVEEXEC") || name.contains("WREXEC") {
        s_read |= S_EXEC;
        s_kill |= S_EXEC;
        s_mod |= S_EXEC;
        scc_write = true;
    }

    // Explicit SCC reads/writes for SALU.
    if matches!(enc, "ENC_SOP1" | "ENC_SOP2" | "ENC_SOPC" | "ENC_SOPK") {
        if matches!(
            name,
            "S_CMOV_B32" | "S_CMOV_B64" | "S_CMOVK_I32" | "S_CSELECT_B32" | "S_CSELECT_B64" | "S_ADDC_U32" | "S_SUBB_U32"
        ) {
            scc_read = true;
        }
        if salu_writes_scc(enc, name) {
            scc_write = true;
        } else if !is_supported_salu(enc, name) && !name.contains("SAVEEXEC") && !name.contains("WREXEC") {
            // Unknown SALU op: conservatively assume it may read scc and all SGPRs.
            scc_read = true;
            s_read |= S_ALL;
            s_kill = 0;
        }
    }

    // Extra encoding-specific SGPR & VGPR uses (catching everything `rdna2::vgpr_uses` misses).
    let (vr, vw) = rdna2::vgpr_uses(d);
    for r in vr {
        v_read.set(r);
    }
    for w in vw {
        v_mod.set(w);
    }
    // Fix up SDWA VSRC1 if S1==1 (scalar, not VGPR — `rdna2::vgpr_uses` treats VSRC1 as VGPR).
    if d.variant == Variant::Sdwa {
        if let Some(e) = ext_encoding(d) {
            if d.enc.field("VSRC1").is_some() && rdna2::field_value(e, &d.words, "S1") == Some(1) {
                // Recompute v_read cleanly from SDWA fields only.
                v_read = VRegs::EMPTY;
                if rdna2::field_value(e, &d.words, "S0") != Some(1) {
                    v_read.set(rdna2::field_value(e, &d.words, "VSRC0").unwrap_or(0));
                }
            }
        }
    }

    match enc {
        "ENC_SOP1" | "ENC_SOP2" | "ENC_SOPC" | "ENC_SOPK" => {}
        "ENC_SMEM" => {
            if matches!(name, "S_MEMTIME" | "S_MEMREALTIME") {
                let sdata = fval(d, "SDATA");
                let r = sreg_range(sdata, 2);
                s_kill |= r;
                s_mod |= r;
            } else if let Some(count) = smem_load_count(name) {
                let sbase = scaled_field(d, "SBASE");
                let base_n = if name.starts_with("S_BUFFER_LOAD_") { 4 } else { 2 };
                s_read |= sreg_range(sbase, base_n);
                let soff = fval(d, "SOFFSET");
                if soff != 125 {
                    let (sr, sc) = scalar_code_regs(soff, 1, wave64);
                    s_read |= sr;
                    scc_read |= sc;
                }
                let sdata = fval(d, "SDATA");
                let dst = sreg_range(sdata, count as u64);
                s_kill |= dst;
                s_mod |= dst;
            } else {
                // Stores / atomics / cache ops: never kill.
                s_kill = 0;
            }
        }
        "ENC_MUBUF" | "ENC_MTBUF" => {
            let srsrc = scaled_field(d, "SRSRC");
            s_read |= sreg_range(srsrc, 4);
            let (sr, sc) = scalar_code_regs(fval(d, "SOFFSET"), 1, wave64);
            s_read |= sr;
            scc_read |= sc;
            s_read |= S_EXEC;
            let vdata = fval(d, "VDATA");
            if name.starts_with("BUFFER_ATOMIC_") {
                let x2 = name.ends_with("_X2");
                let cmpswap = name.contains("CMPSWAP");
                let words = match (x2, cmpswap) {
                    (false, false) => 1,
                    (false, true) => 2,
                    (true, false) => 2,
                    (true, true) => 4,
                };
                v_read.set_range(vdata, words);
                if fval(d, "GLC") == 1 {
                    v_mod.set_range(vdata, if x2 { 2 } else { 1 });
                } else {
                    v_mod = VRegs::EMPTY;
                }
            }
            if is_rmw_or_partial_vgpr_write(d) {
                v_read = v_read.union(v_mod);
            }
        }
        "ENC_MIMG" => {
            let srsrc = scaled_field(d, "SRSRC");
            let t_words = if fval(d, "R128") == 1 { 4 } else { 8 };
            s_read |= sreg_range(srsrc, t_words);
            let base = name.trim_start_matches("IMAGE_");
            let sampling = base.starts_with("SAMPLE") || base.starts_with("GATHER4") || base == "GET_LOD";
            if sampling {
                let ssamp = scaled_field(d, "SSAMP");
                s_read |= sreg_range(ssamp, 4);
            }
            s_read |= S_EXEC;
            // Address VGPRs: sequential or NSA.
            let nsa = fval(d, "NSA");
            if nsa > 0 {
                v_read.set(fval(d, "VADDR"));
                if let Some(nsa_enc) = rdna2::encoding(&format!("MIMG_NSA{nsa}")) {
                    for j in 0..(4 * nsa) {
                        let letter = (b'A' + j.min(25) as u8) as char;
                        if let Some(v) = rdna2::field_value(nsa_enc, &d.words, &format!("VADDR{letter}")) {
                            v_read.set(v);
                        }
                    }
                } else {
                    v_read = VRegs::ALL;
                }
            } else {
                // Sequential VADDR: conservatively cover up to 12 address VGPRs for BVH/grad samples.
                let vaddr = fval(d, "VADDR");
                let n_addr = if base.starts_with("BVH") {
                    12
                } else if sampling {
                    10
                } else {
                    4
                };
                v_read.set_range(vaddr, n_addr);
            }
            // Data VGPRs.
            let vdata = fval(d, "VDATA");
            let dmask = fval(d, "DMASK");
            let d16 = fval(d, "D16") == 1;
            let comps = (dmask & 0xF).count_ones() as u64;
            let n_data = if base.starts_with("GATHER4") || base.starts_with("BVH") {
                if d16 { 2 } else { 4 }
            } else if d16 {
                comps.div_ceil(2).max(1)
            } else {
                comps.max(1)
            };
            if base.starts_with("STORE") {
                v_read.set_range(vdata, n_data);
                v_mod = VRegs::EMPTY;
            } else if base.starts_with("ATOMIC_") {
                let words = if base.contains("CMPSWAP") { 2 } else { 1 };
                v_read.set_range(vdata, words);
                if fval(d, "GLC") == 1 {
                    v_mod.set_range(vdata, 1);
                } else {
                    v_mod = VRegs::EMPTY;
                }
            } else {
                v_mod.set_range(vdata, n_data);
            }
        }
        "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH" => {
            s_read |= S_EXEC;
            let saddr = fval(d, "SADDR");
            let has_saddr = saddr != 0x7D && saddr != 0x7F && saddr < 128;
            if has_saddr {
                let n = if enc == "ENC_FLAT_SCRATCH" { 1 } else { 2 };
                s_read |= sreg_range(saddr, n);
            } else if !name.ends_with("_ADDTID") {
                let addr = fval(d, "ADDR");
                let n = if enc == "ENC_FLAT_SCRATCH" { 1 } else { 2 };
                v_read.set_range(addr, n);
            }
            if name.contains("_ATOMIC_") {
                let data = fval(d, "DATA");
                let x2 = name.ends_with("_X2");
                let cmpswap = name.contains("CMPSWAP");
                let words = match (x2, cmpswap) {
                    (false, false) => 1,
                    (false, true) => 2,
                    (true, false) => 2,
                    (true, true) => 4,
                };
                v_read.set_range(data, words);
                if fval(d, "GLC") == 1 {
                    v_mod.set_range(fval(d, "VDST"), if x2 { 2 } else { 1 });
                } else {
                    v_mod = VRegs::EMPTY;
                }
            }
            if is_rmw_or_partial_vgpr_write(d) {
                v_read = v_read.union(v_mod);
            }
        }
        "ENC_DS" => {
            s_read |= S_EXEC;
            if matches!(name, "DS_WRITE_ADDTID_B32" | "DS_READ_ADDTID_B32") {
                s_read |= S_M0;
                if name == "DS_WRITE_ADDTID_B32" {
                    v_read.set(fval(d, "DATA0"));
                } else {
                    v_mod.set(fval(d, "VDST"));
                }
            } else if name == "DS_SWIZZLE_B32" {
                v_read.set(fval(d, "ADDR"));
                v_mod.set(fval(d, "VDST"));
            } else if matches!(name, "DS_BPERMUTE_B32" | "DS_PERMUTE_B32") {
                v_read.set(fval(d, "ADDR"));
                v_read.set(fval(d, "DATA0"));
                v_mod.set(fval(d, "VDST"));
            } else {
                v_read.set(fval(d, "ADDR"));
                if name.starts_with("DS_READ2") {
                    let n = if name.ends_with("_B64") { 4 } else { 2 };
                    v_mod.set_range(fval(d, "VDST"), n);
                } else if name.starts_with("DS_WRITE2") {
                    let n = if name.ends_with("_B64") { 2 } else { 1 };
                    v_read.set_range(fval(d, "DATA0"), n);
                    v_read.set_range(fval(d, "DATA1"), n);
                    v_mod = VRegs::EMPTY;
                } else if name.contains("WRXCHG2") {
                    v_read.set(fval(d, "DATA0"));
                    v_read.set(fval(d, "DATA1"));
                    if name.contains("_RTN_") {
                        v_mod.set_range(fval(d, "VDST"), 2);
                    }
                } else if name.contains("_RTN_") {
                    v_mod.set(fval(d, "VDST"));
                }
                if is_rmw_or_partial_vgpr_write(d) {
                    v_read = v_read.union(v_mod);
                }
            }
        }
        "ENC_EXP" => {
            s_read |= S_EXEC;
            // `emit.rs:1840` unconditionally calls `self.vreg(VSRC0..3)`.
            for f in ["VSRC0", "VSRC1", "VSRC2", "VSRC3"] {
                v_read.set(fval(d, f));
            }
            v_mod = VRegs::EMPTY;
        }
        "ENC_VINTRP" => {
            s_read |= S_EXEC;
            if matches!(name, "V_INTERP_P1_F32" | "V_INTERP_P1LL_F16" | "V_INTERP_P1LV_F16") {
                // `emit.rs:2009` emits only a comment for p1 (does not write VDST).
                v_mod = VRegs::EMPTY;
            } else if name == "V_INTERP_P2_F16" {
                v_read = v_read.union(v_mod);
            }
        }
        e if e.starts_with("ENC_VOP") || e.starts_with("VOP") => {
            s_read |= S_EXEC;
            if name == "V_SWAP_B32" {
                let s0 = fval(d, "SRC0");
                let vd = fval(d, "VDST");
                v_read.set(vd);
                v_mod.set(vd);
                if s0 >= 256 {
                    v_read.set(s0 - 256);
                    v_mod.set(s0 - 256);
                }
            } else if name.starts_with("V_INTERP_") {
                if matches!(name, "V_INTERP_P1_F32" | "V_INTERP_P1LL_F16" | "V_INTERP_P1LV_F16") {
                    v_mod = VRegs::EMPTY;
                } else if name == "V_INTERP_P2_F16" {
                    v_read = v_read.union(v_mod);
                }
            } else if is_rmw_or_partial_vgpr_write(d) {
                v_read = v_read.union(v_mod);
            }
        }
        _ => {
            // Unknown encoding: conservative over-approximation.
            s_read = S_ALL;
            s_kill = 0;
            scc_read = true;
            v_read = VRegs::ALL;
        }
    }

    let exec_widen = is_exec_widening(d, s_mod);

    let has_any_output = s_mod != 0 || !v_mod.is_empty() || scc_write;
    let pure_eliminable = has_any_output
        && (s_mod & (S_EXEC | S_M0)) == 0
        && match enc {
            "ENC_SOP1" | "ENC_SOP2" | "ENC_SOPC" | "ENC_SOPK" => {
                is_supported_salu(enc, name) && !name.contains("SAVEEXEC") && !name.contains("WREXEC")
            }
            e if e.starts_with("ENC_VOP") || e.starts_with("VOP") => is_supported_pure_valu(d),
            _ => false,
        };

    InstInfo {
        s_read,
        s_kill,
        s_mod,
        scc_read,
        scc_write,
        v_read,
        v_mod,
        carry_sdst,
        pure_eliminable,
        exec_widen,
        indirect_sgpr: false,
        indirect_vgpr: false,
    }
}

fn smem_load_count(name: &str) -> Option<usize> {
    let rest = name
        .strip_prefix("S_BUFFER_LOAD_")
        .or_else(|| name.strip_prefix("S_LOAD_"))?;
    Some(match rest {
        "DWORD" => 1,
        "DWORDX2" => 2,
        "DWORDX4" => 4,
        "DWORDX8" => 8,
        "DWORDX16" => 16,
        _ => return None,
    })
}

/// Guardrail #8: check if Compute `mem..mem3` requires `globallycoherent`.
fn compute_mem_coherent(insts: &[Decoded]) -> bool {
    let mut has_s_barrier = false;
    let mut has_mem_store = false;

    for d in insts {
        let name = d.inst.name;
        let enc = d.enc.name;

        // Condition 1: any GLC / DLC / SLC memory flag.
        if matches!(
            enc,
            "ENC_SMEM" | "ENC_MUBUF" | "ENC_MTBUF" | "ENC_MIMG" | "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH"
        ) {
            if rdna2::field_value(d.enc, &d.words, "GLC") == Some(1)
                || rdna2::field_value(d.enc, &d.words, "DLC") == Some(1)
                || rdna2::field_value(d.enc, &d.words, "SLC") == Some(1)
            {
                return true;
            }
        }

        // Condition 2: any memory atomic.
        if name.starts_with("S_ATOMIC")
            || name.starts_with("S_BUFFER_ATOMIC")
            || name.starts_with("BUFFER_ATOMIC")
            || name.starts_with("FLAT_ATOMIC")
            || name.starts_with("GLOBAL_ATOMIC")
        {
            return true;
        }

        // Condition 3: any cache invalidate / writeback op (excluding instruction cache `S_ICACHE_INV`).
        if name != "S_ICACHE_INV"
            && (name.contains("DCACHE_INV")
                || name.contains("DCACHE_WB")
                || name.contains("DCACHE_DISCARD")
                || name.contains("GL0_INV")
                || name.contains("GL1_INV")
                || name.contains("WBL2")
                || (enc == "ENC_SMEM" && name.ends_with("_INV"))
                || (enc == "ENC_MUBUF" && name.ends_with("_INV")))
        {
            return true;
        }

        // Condition 4: `s_barrier` combined with any `mem..mem3` store.
        if name == "S_BARRIER" {
            has_s_barrier = true;
        }
        match enc {
            "ENC_MUBUF" | "ENC_MTBUF" => {
                if name.starts_with("BUFFER_STORE") || name.starts_with("TBUFFER_STORE") {
                    has_mem_store = true;
                } else if !name.starts_with("BUFFER_LOAD")
                    && !name.starts_with("TBUFFER_LOAD")
                    && !name.ends_with("_INV")
                    && !name.ends_with("WBL2")
                {
                    has_mem_store = true;
                }
            }
            "ENC_FLAT" | "ENC_FLAT_GLBL" => {
                if name.contains("_STORE_") || !name.contains("_LOAD_") {
                    has_mem_store = true;
                }
            }
            "ENC_SMEM" => {
                if name.starts_with("S_STORE") || name.starts_with("S_BUFFER_STORE") {
                    has_mem_store = true;
                } else if smem_load_count(name).is_none()
                    && !matches!(name, "S_MEMTIME" | "S_MEMREALTIME")
                    && !name.contains("_INV")
                {
                    has_mem_store = true;
                }
            }
            _ => {}
        }
    }

    has_s_barrier && has_mem_store
}

/// Maximum byte span for a single hoisted SMEM or `s_buffer_load` group (full 21-bit SMEM range;
/// valid non-`Kernel` UMA allocations never straddle 4 GiB, so zero wrap-fallback duplication!).
const MAX_SMEM_GROUP_SPAN: i64 = 1_048_576;
/// Maximum number of deduplicated loads per hoisted SMEM or `s_buffer_load` group.
const MAX_SMEM_GROUP_LOADS: usize = 96;

/// Returns true iff `d` can mutate global/UMA memory, synchronize memory visibility across
/// lanes/workgroups, or transfer control to unknown code that may mutate memory.
fn is_memory_hazard(d: &Decoded, opt: &Options) -> bool {
    let name = d.inst.name;
    let enc = d.enc.name;
    match enc {
        "ENC_SMEM" => {
            smem_load_count(name).is_none()
                && !matches!(name, "S_MEMTIME" | "S_MEMREALTIME" | "S_ICACHE_INV")
        }
        "ENC_MUBUF" | "ENC_MTBUF" => {
            !name.starts_with("BUFFER_LOAD") && !name.starts_with("TBUFFER_LOAD")
        }
        "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH" => {
            !name.contains("_LOAD_") || name.contains("_STORE_") || name.contains("_ATOMIC_")
        }
        "ENC_MIMG" => {
            let base = name.trim_start_matches("IMAGE_");
            base.starts_with("STORE") || base.starts_with("ATOMIC_")
        }
        "ENC_SOPP" => matches!(name, "S_BARRIER" | "S_SENDMSG"),
        "ENC_SOP1" | "ENC_SOP2" | "ENC_SOPC" | "ENC_SOPK" => {
            name.starts_with("S_CALL")
                || (name == "S_SETPC_B64"
                    && d.far_target.is_none()
                    && !matches!(opt.mode, Mode::Geometry(_)))
                || (name == "S_SWAPPC_B64" && fval(d, "SDST") != fval(d, "SSRC0"))
        }
        "ENC_DS" | "ENC_EXP" | "ENC_VINTRP" => false,
        e if e.starts_with("ENC_VOP") || e.starts_with("VOP") => false,
        _ => true,
    }
}

fn is_terminator_inst(d: &Decoded) -> bool {
    d.far_target.is_some()
        || rdna2::branch_target(d).is_some()
        || d.inst.flags & F_TERMINATOR != 0
        || d.inst.flags & F_INDIRECT != 0
        || matches!(
            d.inst.name,
            "S_ENDPGM" | "S_ENDPGM_SAVED" | "S_ENDPGM_ORDERED_PS_DONE" | "S_SETPC_B64"
        )
}

/// Exact CFG dominator tree computed via iterative RPO + Cooper-Harvey-Kennedy algorithm.
struct DomTree {
    rpo_order: Vec<usize>,
    rpo_num: Vec<usize>,
    idom: Vec<Option<usize>>,
}

impl DomTree {
    fn build(num_blocks: usize, succs: &[Vec<usize>], preds: &[Vec<usize>]) -> Self {
        let mut visited = vec![false; num_blocks];
        let mut postorder = Vec::with_capacity(num_blocks);

        // Iterative DFS from entry block 0.
        if num_blocks > 0 {
            let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
            visited[0] = true;
            while let Some((b, si)) = stack.last_mut() {
                let b_val = *b;
                if *si < succs[b_val].len() {
                    let s = succs[b_val][*si];
                    *si += 1;
                    if !visited[s] {
                        visited[s] = true;
                        stack.push((s, 0));
                    }
                } else {
                    postorder.push(b_val);
                    stack.pop();
                }
            }
        }

        let mut rpo_order = postorder;
        rpo_order.reverse();
        let mut rpo_num = vec![usize::MAX; num_blocks];
        for (idx, &b) in rpo_order.iter().enumerate() {
            rpo_num[b] = idx;
        }

        let mut idom: Vec<Option<usize>> = vec![None; num_blocks];
        if num_blocks > 0 {
            idom[0] = Some(0);
        }

        let intersect = |mut f1: usize, mut f2: usize, idom: &[Option<usize>], rpo_num: &[usize]| -> usize {
            while f1 != f2 {
                while rpo_num[f1] > rpo_num[f2] {
                    f1 = idom[f1].unwrap();
                }
                while rpo_num[f2] > rpo_num[f1] {
                    f2 = idom[f2].unwrap();
                }
            }
            f1
        };

        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo_order.iter().skip(1) {
                let mut new_idom: Option<usize> = None;
                for &p in &preds[b] {
                    if idom[p].is_some() {
                        new_idom = Some(match new_idom {
                            None => p,
                            Some(cur) => intersect(p, cur, &idom, &rpo_num),
                        });
                    }
                }
                if new_idom.is_some() && idom[b] != new_idom {
                    idom[b] = new_idom;
                    changed = true;
                }
            }
        }

        Self {
            rpo_order,
            rpo_num,
            idom,
        }
    }

    #[inline]
    fn is_reachable(&self, b: usize) -> bool {
        b < self.rpo_num.len() && self.rpo_num[b] != usize::MAX
    }

    fn dominates(&self, a: usize, b: usize) -> bool {
        if !self.is_reachable(a) || !self.is_reachable(b) || self.rpo_num[a] > self.rpo_num[b] {
            return false;
        }
        let mut cur = b;
        while cur != a {
            if cur == 0 {
                return false;
            }
            cur = match self.idom[cur] {
                Some(p) => p,
                None => return false,
            };
        }
        true
    }

    fn lca(&self, mut f1: usize, mut f2: usize) -> Option<usize> {
        if !self.is_reachable(f1) || !self.is_reachable(f2) {
            return None;
        }
        while f1 != f2 {
            while self.rpo_num[f1] > self.rpo_num[f2] {
                f1 = self.idom[f1]?;
            }
            while self.rpo_num[f2] > self.rpo_num[f1] {
                f2 = self.idom[f2]?;
            }
        }
        Some(f1)
    }
}

/// Unique reaching-definition identity for a plain SGPR `s0..s105`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum DefVal {
    Bottom,
    Entry,
    Inst(usize),
    Top,
}

#[inline]
fn meet_def(a: DefVal, b: DefVal) -> DefVal {
    match (a, b) {
        (DefVal::Bottom, x) | (x, DefVal::Bottom) => x,
        (x, y) if x == y => x,
        _ => DefVal::Top,
    }
}

/// Lattice value for CFG-wide plain SGPR constant propagation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConstVal {
    Bottom,
    Known(u32),
    Top,
}

#[inline]
fn meet_const(a: ConstVal, b: ConstVal) -> ConstVal {
    match (a, b) {
        (ConstVal::Bottom, x) | (x, ConstVal::Bottom) => x,
        (ConstVal::Known(x), ConstVal::Known(y)) if x == y => ConstVal::Known(x),
        _ => ConstVal::Top,
    }
}

#[inline]
fn resolve_const_scalar(d: &Decoded, code: u64, state: &[ConstVal; 106]) -> Option<u32> {
    if code < 106 {
        match state[code as usize] {
            ConstVal::Known(v) => Some(v),
            _ => None,
        }
    } else {
        inline_const_bits(code, d.literal, false)
    }
}

/// Apply one instruction's effect to `(defs, consts)` for plain SGPRs `0..106`.
fn step_sgpr_def_const(
    inst_idx: usize,
    d: &Decoded,
    info: &InstInfo,
    defs: &mut [DefVal; 106],
    consts: &mut [ConstVal; 106],
) {
    let name = d.inst.name;
    let mut new_c: [(usize, ConstVal); 2] = [(usize::MAX, ConstVal::Top); 2];
    let mut new_c_len = 0usize;

    match name {
        "S_MOV_B32" => {
            let sd = fval(d, "SDST") as usize;
            if sd < 106 {
                if let Some(v) = resolve_const_scalar(d, fval(d, "SSRC0"), consts) {
                    new_c[0] = (sd, ConstVal::Known(v));
                    new_c_len = 1;
                }
            }
        }
        "S_MOVK_I32" => {
            let sd = fval(d, "SDST") as usize;
            if sd < 106 {
                let v = fval(d, "SIMM16") as u16 as i16 as i32 as u32;
                new_c[0] = (sd, ConstVal::Known(v));
                new_c_len = 1;
            }
        }
        "S_MOV_B64" => {
            let sd = fval(d, "SDST") as usize;
            let s0 = fval(d, "SSRC0");
            if sd + 1 < 106 {
                match s0 {
                    0..=104 => {
                        let lo = consts[s0 as usize];
                        let hi = consts[s0 as usize + 1];
                        if matches!(lo, ConstVal::Known(_)) {
                            new_c[new_c_len] = (sd, lo);
                            new_c_len += 1;
                        }
                        if matches!(hi, ConstVal::Known(_)) {
                            new_c[new_c_len] = (sd + 1, hi);
                            new_c_len += 1;
                        }
                    }
                    128..=208 => {
                        if let Some(v32) = inline_const_bits(s0, d.literal, false) {
                            let v64 = v32 as i32 as i64;
                            new_c[0] = (sd, ConstVal::Known(v64 as u32));
                            new_c[1] = (sd + 1, ConstVal::Known((v64 >> 32) as u32));
                            new_c_len = 2;
                        }
                    }
                    255 => {
                        if let Some(lit) = d.literal {
                            new_c[0] = (sd, ConstVal::Known(lit));
                            new_c[1] = (sd + 1, ConstVal::Known(0));
                            new_c_len = 2;
                        }
                    }
                    _ => {}
                }
            }
        }
        "S_XOR_B32" => {
            let sd = fval(d, "SDST") as usize;
            let s0 = fval(d, "SSRC0");
            let s1 = fval(d, "SSRC1");
            if sd < 106 {
                if s0 == s1 && s0 < 106 {
                    new_c[0] = (sd, ConstVal::Known(0));
                    new_c_len = 1;
                } else if let (Some(a), Some(b)) = (
                    resolve_const_scalar(d, s0, consts),
                    resolve_const_scalar(d, s1, consts),
                ) {
                    new_c[0] = (sd, ConstVal::Known(a ^ b));
                    new_c_len = 1;
                }
            }
        }
        "S_XOR_B64" => {
            let sd = fval(d, "SDST") as usize;
            let s0 = fval(d, "SSRC0");
            let s1 = fval(d, "SSRC1");
            if sd + 1 < 106 && s0 == s1 && s0 < 105 {
                new_c[0] = (sd, ConstVal::Known(0));
                new_c[1] = (sd + 1, ConstVal::Known(0));
                new_c_len = 2;
            }
        }
        "S_AND_B32" => {
            let sd = fval(d, "SDST") as usize;
            if sd < 106 {
                let a = resolve_const_scalar(d, fval(d, "SSRC0"), consts);
                let b = resolve_const_scalar(d, fval(d, "SSRC1"), consts);
                if a == Some(0) || b == Some(0) {
                    new_c[0] = (sd, ConstVal::Known(0));
                    new_c_len = 1;
                } else if let (Some(a), Some(b)) = (a, b) {
                    new_c[0] = (sd, ConstVal::Known(a & b));
                    new_c_len = 1;
                }
            }
        }
        "S_AND_B64" => {
            let sd = fval(d, "SDST") as usize;
            if sd + 1 < 106 {
                let s0 = fval(d, "SSRC0");
                let s1 = fval(d, "SSRC1");
                let hi_of = |code: u64| -> Option<u32> {
                    if code < 105 {
                        match consts[code as usize + 1] {
                            ConstVal::Known(v) => Some(v),
                            _ => None,
                        }
                    } else if matches!(code, 128..=208) {
                        inline_const_bits(code, d.literal, false).map(|v| ((v as i32 as i64) >> 32) as u32)
                    } else if code == 255 && d.literal.is_some() {
                        Some(0)
                    } else {
                        None
                    }
                };
                let (h0, h1) = (hi_of(s0), hi_of(s1));
                if h0 == Some(0) || h1 == Some(0) {
                    new_c[0] = (sd + 1, ConstVal::Known(0));
                    new_c_len = 1;
                } else if let (Some(a), Some(b)) = (h0, h1) {
                    new_c[0] = (sd + 1, ConstVal::Known(a & b));
                    new_c_len = 1;
                }
            }
        }
        "S_OR_B32" => {
            let sd = fval(d, "SDST") as usize;
            if sd < 106 {
                let a = resolve_const_scalar(d, fval(d, "SSRC0"), consts);
                let b = resolve_const_scalar(d, fval(d, "SSRC1"), consts);
                if a == Some(u32::MAX) || b == Some(u32::MAX) {
                    new_c[0] = (sd, ConstVal::Known(u32::MAX));
                    new_c_len = 1;
                } else if let (Some(a), Some(b)) = (a, b) {
                    new_c[0] = (sd, ConstVal::Known(a | b));
                    new_c_len = 1;
                }
            }
        }
        "S_SUB_U32" | "S_SUB_I32" => {
            let sd = fval(d, "SDST") as usize;
            let s0 = fval(d, "SSRC0");
            let s1 = fval(d, "SSRC1");
            if sd < 106 {
                if s0 == s1 && s0 < 106 {
                    new_c[0] = (sd, ConstVal::Known(0));
                    new_c_len = 1;
                } else if let (Some(a), Some(b)) = (
                    resolve_const_scalar(d, s0, consts),
                    resolve_const_scalar(d, s1, consts),
                ) {
                    new_c[0] = (sd, ConstVal::Known(a.wrapping_sub(b)));
                    new_c_len = 1;
                }
            }
        }
        "S_ADD_U32" | "S_ADD_I32" | "S_MUL_I32" | "S_LSHL_B32" | "S_LSHR_B32" | "S_ASHR_I32"
        | "S_PACK_LL_B32_B16" | "S_PACK_LH_B32_B16" | "S_PACK_HH_B32_B16" => {
            let sd = fval(d, "SDST") as usize;
            if sd < 106 {
                if let (Some(a), Some(b)) = (
                    resolve_const_scalar(d, fval(d, "SSRC0"), consts),
                    resolve_const_scalar(d, fval(d, "SSRC1"), consts),
                ) {
                    let v = match name {
                        "S_ADD_U32" | "S_ADD_I32" => a.wrapping_add(b),
                        "S_MUL_I32" => a.wrapping_mul(b),
                        "S_LSHL_B32" => a << (b & 31),
                        "S_LSHR_B32" => a >> (b & 31),
                        "S_ASHR_I32" => ((a as i32) >> (b & 31)) as u32,
                        "S_PACK_LL_B32_B16" => (a & 0xFFFF) | (b << 16),
                        "S_PACK_LH_B32_B16" => (a & 0xFFFF) | (b & 0xFFFF_0000),
                        "S_PACK_HH_B32_B16" => (a >> 16) | (b & 0xFFFF_0000),
                        _ => unreachable!(),
                    };
                    new_c[0] = (sd, ConstVal::Known(v));
                    new_c_len = 1;
                }
            }
        }
        "S_ADDK_I32" | "S_MULK_I32" => {
            let sd = fval(d, "SDST") as usize;
            if sd < 106 {
                if let ConstVal::Known(cur) = consts[sd] {
                    let k = fval(d, "SIMM16") as u16 as i16 as i32 as u32;
                    let v = if name == "S_ADDK_I32" {
                        cur.wrapping_add(k)
                    } else {
                        cur.wrapping_mul(k)
                    };
                    new_c[0] = (sd, ConstVal::Known(v));
                    new_c_len = 1;
                }
            }
        }
        "S_NOT_B32" => {
            let sd = fval(d, "SDST") as usize;
            if sd < 106 {
                if let Some(a) = resolve_const_scalar(d, fval(d, "SSRC0"), consts) {
                    new_c[0] = (sd, ConstVal::Known(!a));
                    new_c_len = 1;
                }
            }
        }
        _ => {}
    }

    if info.indirect_sgpr {
        *defs = [DefVal::Top; 106];
        *consts = [ConstVal::Top; 106];
        return;
    }

    let mut m = info.s_mod & S_PLAIN_MASK;
    while m != 0 {
        let r = m.trailing_zeros() as usize;
        m &= m - 1;
        defs[r] = DefVal::Inst(inst_idx);
        consts[r] = ConstVal::Top;
    }
    for &(r, cv) in &new_c[..new_c_len] {
        consts[r] = cv;
    }
}

/// Forward CFG-wide fixpoint for SGPR reaching definitions and constant values.
fn solve_cfg_sgpr_defs_and_consts(
    num_blocks: usize,
    dom: &DomTree,
    cfg_preds: &[Vec<usize>],
    block_ranges: &[(usize, usize)],
    eff_end: &[usize],
    insts: &[Decoded],
    infos: &[InstInfo],
    dead_inst: &HashSet<u64>,
    has_unknown_cf: bool,
) -> (Vec<[DefVal; 106]>, Vec<[ConstVal; 106]>) {
    let mut def_in = vec![[DefVal::Bottom; 106]; num_blocks];
    let mut def_out = vec![[DefVal::Bottom; 106]; num_blocks];
    let mut const_in = vec![[ConstVal::Bottom; 106]; num_blocks];
    let mut const_out = vec![[ConstVal::Bottom; 106]; num_blocks];

    if num_blocks == 0 {
        return (def_in, const_in);
    }

    if has_unknown_cf {
        // Conservative under indirect branches: every block entry is Top.
        for bi in 0..num_blocks {
            def_in[bi] = [DefVal::Top; 106];
            const_in[bi] = [ConstVal::Top; 106];
        }
        def_in[0] = [DefVal::Entry; 106];
        return (def_in, const_in);
    }

    def_in[0] = [DefVal::Entry; 106];
    const_in[0] = [ConstVal::Top; 106];

    loop {
        let mut changed = false;
        for &bi in &dom.rpo_order {
            let (in_d, in_c) = if bi == 0 {
                let mut d0 = [DefVal::Entry; 106];
                let mut c0 = [ConstVal::Top; 106];
                for &p in &cfg_preds[0] {
                    if dom.is_reachable(p) {
                        for r in 0..106 {
                            d0[r] = meet_def(d0[r], def_out[p][r]);
                            c0[r] = meet_const(c0[r], const_out[p][r]);
                        }
                    }
                }
                (d0, c0)
            } else {
                let mut d_acc = [DefVal::Bottom; 106];
                let mut c_acc = [ConstVal::Bottom; 106];
                for &p in &cfg_preds[bi] {
                    if dom.is_reachable(p) {
                        for r in 0..106 {
                            d_acc[r] = meet_def(d_acc[r], def_out[p][r]);
                            c_acc[r] = meet_const(c_acc[r], const_out[p][r]);
                        }
                    }
                }
                (d_acc, c_acc)
            };

            let mut cur_d = in_d;
            let mut cur_c = in_c;
            let s_idx = block_ranges[bi].0;
            let e_idx = eff_end[bi];
            for k in s_idx..e_idx {
                if dead_inst.contains(&insts[k].pc) {
                    continue;
                }
                step_sgpr_def_const(k, &insts[k], &infos[k], &mut cur_d, &mut cur_c);
            }

            if in_d != def_in[bi]
                || cur_d != def_out[bi]
                || in_c != const_in[bi]
                || cur_c != const_out[bi]
            {
                def_in[bi] = in_d;
                def_out[bi] = cur_d;
                const_in[bi] = in_c;
                const_out[bi] = cur_c;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    (def_in, const_in)
}

/// Canonical reaching-def key for grouping `s_load_*` base pointer halves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum DefKey {
    Entry,
    Inst(usize),
    /// Block-entry identity for a `Top` reaching def (allows safe intra-block grouping &
    /// `known_hi == Some(0)` without cross-block hoisting!).
    LocalEntry(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HoistLoc {
    BlockEntry(u64),
    BeforePc(u64),
    AfterPc(u64),
}

#[derive(Clone, Debug)]
struct SmemCand {
    bi: usize,
    local_idx: usize,
    pc: u64,
    off: i64,
    count: u8,
    end: i64,
}

/// Deduplicate & subsume `(offset, count)` pairs into sorted `HoistedLoad`s.
/// Sub-range subsumption (`parent` covers `child`) is applied only when `allow_subsume_neg || parent.offset >= 0`.
fn dedup_and_subsume_loads(
    pairs: &[(i64, u8)],
    allow_subsume_neg: bool,
) -> (Vec<HoistedLoad>, HashMap<(i64, u8), (u16, u8, u8)>) {
    let mut uniq: Vec<(i64, u8)> = pairs.to_vec();
    uniq.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
    uniq.dedup();

    let mut loads: Vec<HoistedLoad> = Vec::with_capacity(uniq.len());
    let mut map: HashMap<(i64, u8), (u16, u8, u8)> = HashMap::with_capacity(uniq.len());

    for &(off, count) in &uniq {
        let end = off + 4 * (count as i64);
        let mut found: Option<(u16, u8, u8)> = None;
        for (idx, parent) in loads.iter().enumerate() {
            let p_end = parent.offset + 4 * (parent.count as i64);
            if parent.offset == off && parent.count == count {
                found = Some((idx as u16, 0, parent.count));
                break;
            }
            if (allow_subsume_neg || parent.offset >= 0) && parent.offset <= off && end <= p_end {
                let dw_off = ((off - parent.offset) / 4) as u8;
                found = Some((idx as u16, dw_off, parent.count));
                break;
            }
        }
        if let Some(m) = found {
            map.insert((off, count), m);
        } else {
            let idx = loads.len() as u16;
            loads.push(HoistedLoad { offset: off, count });
            map.insert((off, count), (idx, 0, count));
        }
    }

    (loads, map)
}

struct HoistedSmemResult {
    groups: Vec<HoistedSmemGroup>,
    at_entry: HashMap<u64, Vec<usize>>,
    before_pc: HashMap<u64, Vec<usize>>,
    after_pc: HashMap<u64, Vec<usize>>,
    uses: HashMap<u64, SmemUse>,
}

/// Weapon 1A: Dominator-hoisted & CSE'd invariant `s_load_dword*` groups + known-zero-hi Window-0 fast path.
#[allow(clippy::too_many_arguments)]
fn analyze_hoisted_smem(
    insts: &[Decoded],
    infos: &[InstInfo],
    block_starts: &[u64],
    block_ranges: &[(usize, usize)],
    eff_end: &[usize],
    dom: &DomTree,
    cfg_preds: &[Vec<usize>],
    unknown_succ: &[bool],
    inst_hazard: &[bool],
    dead_inst: &HashSet<u64>,
    def_in: &[[DefVal; 106]],
    const_in: &[[ConstVal; 106]],
    opt: &Options,
    has_unknown_cf: bool,
) -> HoistedSmemResult {
    let num_blocks = block_starts.len();
    let kernel_mode = matches!(opt.mode, Mode::Kernel);

    // Per-instruction block lookup and per-block hazard / SGPR-mod summaries.
    let mut inst_block = vec![0usize; insts.len()];
    let mut inst_local_idx = vec![0usize; insts.len()];
    let mut block_has_hazard = vec![false; num_blocks];
    let mut block_mod_s = vec![0u128; num_blocks];
    let mut block_indirect_sgpr = vec![false; num_blocks];

    for bi in 0..num_blocks {
        let s_idx = block_ranges[bi].0;
        let e_idx = eff_end[bi];
        for (li, k) in (s_idx..e_idx).enumerate() {
            inst_block[k] = bi;
            inst_local_idx[k] = li;
            if dead_inst.contains(&insts[k].pc) {
                continue;
            }
            if inst_hazard[k] {
                block_has_hazard[bi] = true;
            }
            block_mod_s[bi] |= infos[k].s_mod;
            if infos[k].indirect_sgpr {
                block_indirect_sgpr[bi] = true;
            }
        }
    }

    // Collect candidate `s_load_dword*` grouped by `(sbase, key_lo, key_hi)` in deterministic order.
    let mut groups_by_key: BTreeMap<(u8, DefKey, DefKey), Vec<SmemCand>> = BTreeMap::new();
    // Track per-block `(const_s_before_inst)` on demand by re-stepping block `H` when validating.
    for &bi in &dom.rpo_order {
        let s_idx = block_ranges[bi].0;
        let e_idx = eff_end[bi];
        let mut cur_d = def_in[bi];
        let mut cur_c = const_in[bi];

        for (li, k) in (s_idx..e_idx).enumerate() {
            let d = &insts[k];
            if dead_inst.contains(&d.pc) {
                continue;
            }
            if d.enc.name == "ENC_SMEM" && d.inst.name.starts_with("S_LOAD_") {
                if let Some(count) = smem_load_count(d.inst.name) {
                    let is_glc_dlc = rdna2::field_value(d.enc, &d.words, "GLC") == Some(1)
                        || rdna2::field_value(d.enc, &d.words, "DLC") == Some(1);
                    let sbase = scaled_field(d, "SBASE");
                    let soff = rdna2::field_value(d.enc, &d.words, "SOFFSET").unwrap_or(125);
                    let raw_imm = fval(d, "OFFSET") as u32;
                    let off = ((raw_imm << 11) as i32 >> 11) as i64;

                    if !is_glc_dlc && sbase + 2 <= 106 && soff == 125 && (off & 3) == 0 {
                        let to_key = |dv: DefVal| -> DefKey {
                            if has_unknown_cf {
                                return DefKey::LocalEntry(bi);
                            }
                            match dv {
                                DefVal::Entry => DefKey::Entry,
                                DefVal::Inst(di) => DefKey::Inst(di),
                                DefVal::Top | DefVal::Bottom => DefKey::LocalEntry(bi),
                            }
                        };
                        let key_lo = to_key(cur_d[sbase as usize]);
                        let key_hi = to_key(cur_d[sbase as usize + 1]);
                        groups_by_key
                            .entry((sbase as u8, key_lo, key_hi))
                            .or_default()
                            .push(SmemCand {
                                bi,
                                local_idx: li,
                                pc: d.pc,
                                off,
                                count: count as u8,
                                end: off + 4 * (count as i64),
                            });
                    }
                }
            }
            step_sgpr_def_const(k, d, &infos[k], &mut cur_d, &mut cur_c);
        }
    }

    let def_block_of = |key: DefKey| -> usize {
        match key {
            DefKey::Entry => 0,
            DefKey::Inst(di) => inst_block[di],
            DefKey::LocalEntry(bi) => bi,
        }
    };

    // Validate a candidate cluster: returns `Some((hoist_bi, h_idx, HoistLoc))` if 100% sound.
    let validate_cluster = |sbase: u8,
                            key_lo: DefKey,
                            key_hi: DefKey,
                            cluster: &[SmemCand]|
     -> Option<(usize, usize, HoistLoc)> {
        if cluster.is_empty() {
            return None;
        }
        let mut min_off = i64::MAX;
        let mut max_end = i64::MIN;
        let mut h = cluster[0].bi;
        for u in cluster {
            min_off = min_off.min(u.off);
            max_end = max_end.max(u.end);
            h = dom.lca(h, u.bi)?;
        }
        if max_end - min_off > MAX_SMEM_GROUP_SPAN {
            return None;
        }
        if !dom.dominates(def_block_of(key_lo), h) || !dom.dominates(def_block_of(key_hi), h) {
            return None;
        }

        let hs_idx = block_ranges[h].0;
        let he_idx = eff_end[h];
        let h_len = he_idx - hs_idx;

        let min_use_in_h = cluster
            .iter()
            .filter(|u| u.bi == h)
            .map(|u| u.local_idx)
            .min();

        let (h_idx, hoist_loc) = if let Some(u_li) = min_use_in_h {
            // Def inside H must strictly precede the hoist point.
            if let DefKey::Inst(di) = key_lo {
                if inst_block[di] == h && inst_local_idx[di] >= u_li {
                    return None;
                }
            }
            if let DefKey::Inst(di) = key_hi {
                if inst_block[di] == h && inst_local_idx[di] >= u_li {
                    return None;
                }
            }
            (u_li, HoistLoc::BeforePc(insts[hs_idx + u_li].pc))
        } else if h_len == 0 {
            (0, HoistLoc::BlockEntry(block_starts[h]))
        } else {
            let last_k = he_idx - 1;
            if is_terminator_inst(&insts[last_k]) {
                let place_li = h_len - 1;
                if let DefKey::Inst(di) = key_lo {
                    if inst_block[di] == h && inst_local_idx[di] >= place_li {
                        return None;
                    }
                }
                if let DefKey::Inst(di) = key_hi {
                    if inst_block[di] == h && inst_local_idx[di] >= place_li {
                        return None;
                    }
                }
                (place_li, HoistLoc::BeforePc(insts[last_k].pc))
            } else {
                // Fallthrough block: emit immediately after the last instruction of H.
                (h_len, HoistLoc::AfterPc(insts[last_k].pc))
            }
        };

        let base_mask = (1u128 << sbase) | (1u128 << (sbase + 1));

        // Helper: check slice of instructions `lo..hi` inside block `bi` for hazard or base clobber.
        let slice_is_clean = |bi: usize, lo: usize, hi: usize| -> bool {
            let bs = block_ranges[bi].0;
            for li in lo..hi {
                let k = bs + li;
                if dead_inst.contains(&insts[k].pc) {
                    continue;
                }
                if inst_hazard[k]
                    || infos[k].indirect_sgpr
                    || (infos[k].s_mod & base_mask) != 0
                {
                    return false;
                }
            }
            true
        };

        let mut checked_h_suffix = false;
        let mut clean_interm_checked: HashSet<usize> = HashSet::new();

        for u in cluster {
            if u.bi == h {
                if u.local_idx < h_idx || !slice_is_clean(h, h_idx, u.local_idx) {
                    return None;
                }
            } else {
                if !checked_h_suffix {
                    if unknown_succ[h] || !slice_is_clean(h, h_idx, h_len) {
                        return None;
                    }
                    checked_h_suffix = true;
                }
                if !slice_is_clean(u.bi, 0, u.local_idx) {
                    return None;
                }
                // Backwards BFS from `cfg_preds[u.bi]` stopping at `h`.
                // Note: `u.bi` itself is NOT pre-marked visited so cycles back to `u.bi`
                // without passing through `h` DO visit `u.bi` and check `block_has_hazard[u.bi]`!
                if !clean_interm_checked.contains(&u.bi) {
                    let mut visited: HashSet<usize> = HashSet::new();
                    let mut queue: VecDeque<usize> = VecDeque::new();
                    for &p in &cfg_preds[u.bi] {
                        if dom.is_reachable(p) && visited.insert(p) {
                            queue.push_back(p);
                        }
                    }
                    while let Some(m) = queue.pop_front() {
                        if m == h {
                            continue;
                        }
                        if m == 0 {
                            // Reached entry without passing through H -> H does not dominate U!
                            return None;
                        }
                        if block_has_hazard[m]
                            || block_indirect_sgpr[m]
                            || (block_mod_s[m] & base_mask) != 0
                            || unknown_succ[m]
                        {
                            return None;
                        }
                        for &p in &cfg_preds[m] {
                            if dom.is_reachable(p) && visited.insert(p) {
                                queue.push_back(p);
                            }
                        }
                    }
                    clean_interm_checked.insert(u.bi);
                }
            }
        }

        Some((h, h_idx, hoist_loc))
    };

    // Compute `const_s[sbase + 1]` at exact point `(h, h_idx)`.
    let const_hi_at = |h: usize, h_idx: usize, sbase: u8| -> Option<u32> {
        let mut cur_d = def_in[h];
        let mut cur_c = const_in[h];
        let hs = block_ranges[h].0;
        for li in 0..h_idx {
            let k = hs + li;
            if dead_inst.contains(&insts[k].pc) {
                continue;
            }
            step_sgpr_def_const(k, &insts[k], &infos[k], &mut cur_d, &mut cur_c);
        }
        match cur_c[sbase as usize + 1] {
            ConstVal::Known(v) => Some(v),
            _ => None,
        }
    };

    let mut out = HoistedSmemResult {
        groups: Vec::new(),
        at_entry: HashMap::new(),
        before_pc: HashMap::new(),
        after_pc: HashMap::new(),
        uses: HashMap::new(),
    };

    for ((sbase, key_lo, key_hi), cands) in groups_by_key {
        // Fast path: try the entire set as one cluster first!
        let clusters: Vec<(Vec<SmemCand>, usize, usize, HoistLoc)> =
            if let Some((h, h_idx, loc)) = validate_cluster(sbase, key_lo, key_hi, &cands) {
                vec![(cands, h, h_idx, loc)]
            } else {
                // Partition greedily into maximal hazard-free clusters.
                let mut open: Vec<(Vec<SmemCand>, usize, usize, HoistLoc)> = Vec::new();
                for cand in cands {
                    let mut placed = false;
                    let start_try = open.len().saturating_sub(4);
                    for entry in open[start_try..].iter_mut().rev() {
                        entry.0.push(cand.clone());
                        if let Some((h, h_idx, loc)) = validate_cluster(sbase, key_lo, key_hi, &entry.0) {
                            entry.1 = h;
                            entry.2 = h_idx;
                            entry.3 = loc;
                            placed = true;
                            break;
                        }
                        entry.0.pop();
                    }
                    if !placed {
                        let single = vec![cand];
                        if let Some((h, h_idx, loc)) = validate_cluster(sbase, key_lo, key_hi, &single) {
                            open.push((single, h, h_idx, loc));
                        }
                    }
                }
                open
            };

        for (cluster, h, h_idx, hoist_loc) in clusters {
            let pairs: Vec<(i64, u8)> = cluster.iter().map(|u| (u.off, u.count)).collect();
            let (all_loads, pair_map) = dedup_and_subsume_loads(&pairs, false);
            let hi_const = const_hi_at(h, h_idx, sbase);

            // Chunk into groups of `<= MAX_SMEM_GROUP_LOADS`.
            let num_chunks = all_loads.len().div_ceil(MAX_SMEM_GROUP_LOADS);
            for chunk_i in 0..num_chunks {
                let lo_l = chunk_i * MAX_SMEM_GROUP_LOADS;
                let hi_l = (lo_l + MAX_SMEM_GROUP_LOADS).min(all_loads.len());
                let chunk_loads: Vec<HoistedLoad> = all_loads[lo_l..hi_l].to_vec();

                let mut chunk_uses: Vec<(&SmemCand, u16, u8, u8)> = Vec::new();
                for u in &cluster {
                    let (global_li, dw_off, parent_cnt) = pair_map[&(u.off, u.count)];
                    let gi = global_li as usize;
                    if gi >= lo_l && gi < hi_l {
                        chunk_uses.push((u, (gi - lo_l) as u16, dw_off, parent_cnt));
                    }
                }
                if chunk_uses.is_empty() {
                    continue;
                }

                let min_off = chunk_loads.iter().map(|l| l.offset).min().unwrap();
                let max_end = chunk_loads
                    .iter()
                    .map(|l| l.offset + 4 * (l.count as i64))
                    .max()
                    .unwrap();

                // CRITICAL: ONLY `hi == 0` unconditionally maps to Window 0 regardless of `xcl_uma_base`!
                // Guardrail (`#326`): in `Mode::Kernel`, ONLY allow `known_hi = Some(0)` when
                // `min_off == 0 && max_end <= 4` (cannot cross 4 GiB boundary!).
                // In non-`Kernel` modes, valid UMA allocations never straddle 4 GiB, so the full
                // 21-bit SMEM immediate range `[-1_048_576, 1_048_576]` is Window-0 safe.
                let known_hi = if hi_const == Some(0) {
                    if kernel_mode {
                        (min_off == 0 && max_end <= 4).then_some(0)
                    } else if min_off >= -1_048_576 && max_end <= 1_048_576 {
                        Some(0)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let worth_emitting = if opt.mem_windows > 1 {
                    chunk_uses.len() >= 2 || known_hi.is_some()
                } else {
                    chunk_uses.len() >= 2
                };
                if !worth_emitting {
                    continue;
                }

                let gid = out.groups.len() as u32;
                let gidx = out.groups.len();
                out.groups.push(HoistedSmemGroup {
                    id: gid,
                    hoist_block: block_starts[h],
                    sbase,
                    known_hi,
                    min_off,
                    max_end,
                    loads: chunk_loads,
                });
                match hoist_loc {
                    HoistLoc::BlockEntry(bpc) => out.at_entry.entry(bpc).or_default().push(gidx),
                    HoistLoc::BeforePc(pc) => out.before_pc.entry(pc).or_default().push(gidx),
                    HoistLoc::AfterPc(pc) => out.after_pc.entry(pc).or_default().push(gidx),
                }
                for (u, load_idx, dword_offset, parent_count) in chunk_uses {
                    out.uses.insert(
                        u.pc,
                        SmemUse {
                            group_id: gid,
                            load_idx,
                            parent_count,
                            dword_offset,
                            count: u.count,
                        },
                    );
                }
            }
        }
    }

    out
}

/// Universal non-`Kernel` Window-0 fast-path sets (`known_zero_hi_sload`, `known_zero_hi_sbuf`).
///
/// Covers EVERY non-`Kernel` `s_load_dword*` (including `soff != 125`, dynamic `s_lo`, and
/// post-store loads) where `s[sbase+1] == 0` (and `imm_off >= 0` when `soff != 125`, or
/// `-1_048_576 <= imm_off <= 1_048_576` when `soff == 125`), plus EVERY non-`Kernel`
/// `s_buffer_load_dword*` where `(s[sbase+1] & 0xFFFF) == 0`.
#[allow(clippy::too_many_arguments)]
fn collect_known_zero_hi_smem(
    insts: &[Decoded],
    infos: &[InstInfo],
    block_ranges: &[(usize, usize)],
    eff_end: &[usize],
    dom: &DomTree,
    dead_inst: &HashSet<u64>,
    def_in: &[[DefVal; 106]],
    const_in: &[[ConstVal; 106]],
    opt: &Options,
    has_unknown_cf: bool,
) -> (HashSet<u64>, HashSet<u64>) {
    let mut known_zero_hi_sload = HashSet::new();
    let mut known_zero_hi_sbuf = HashSet::new();
    if matches!(opt.mode, Mode::Kernel) {
        return (known_zero_hi_sload, known_zero_hi_sbuf);
    }

    for bi in 0..block_ranges.len() {
        if !has_unknown_cf && !dom.is_reachable(bi) {
            continue;
        }
        let s_idx = block_ranges[bi].0;
        let e_idx = eff_end[bi];
        let mut cur_d = def_in[bi];
        let mut cur_c = const_in[bi];

        for k in s_idx..e_idx {
            let d = &insts[k];
            if dead_inst.contains(&d.pc) {
                continue;
            }
            if d.enc.name == "ENC_SMEM" && smem_load_count(d.inst.name).is_some() {
                let sbase = scaled_field(d, "SBASE");
                if d.inst.name.starts_with("S_LOAD_") {
                    if sbase + 2 <= 106 && cur_c[sbase as usize + 1] == ConstVal::Known(0) {
                        let soff = rdna2::field_value(d.enc, &d.words, "SOFFSET").unwrap_or(125);
                        let raw_imm = fval(d, "OFFSET") as u32;
                        let imm_off = ((raw_imm << 11) as i32 >> 11) as i64;
                        let off_ok = if soff != 125 {
                            imm_off >= 0
                        } else {
                            (-1_048_576..=1_048_576).contains(&imm_off)
                        };
                        if off_ok {
                            known_zero_hi_sload.insert(d.pc);
                        }
                    }
                } else if d.inst.name.starts_with("S_BUFFER_LOAD_") && sbase + 4 <= 106 {
                    if matches!(cur_c[sbase as usize + 1], ConstVal::Known(w) if (w & 0xFFFF) == 0) {
                        known_zero_hi_sbuf.insert(d.pc);
                    }
                }
            }
            step_sgpr_def_const(k, d, &infos[k], &mut cur_d, &mut cur_c);
        }
    }

    (known_zero_hi_sload, known_zero_hi_sbuf)
}

struct HoistedSbufResult {
    groups: Vec<HoistedSbufGroup>,
    at_entry: HashMap<u64, Vec<usize>>,
    before_pc: HashMap<u64, Vec<usize>>,
    after_pc: HashMap<u64, Vec<usize>>,
    uses: HashMap<u64, SbufUse>,
}

/// Cross-block dominator-hoisted & CSE'd invariant constant-offset (`SOFFSET == 125`)
/// `s_buffer_load_dword*` groups sharing reaching definitions `(def_d0, def_d1, def_d2)`
/// (`sbase..sbase+3`; `s_buffer_load_*` never reads `d3`).
/// Single-block groups are left to `analyze_intra_block_sbuf` so their temps stay block-scoped.
#[allow(clippy::too_many_arguments)]
fn analyze_hoisted_sbuf(
    insts: &[Decoded],
    infos: &[InstInfo],
    block_starts: &[u64],
    block_ranges: &[(usize, usize)],
    eff_end: &[usize],
    dom: &DomTree,
    cfg_preds: &[Vec<usize>],
    unknown_succ: &[bool],
    inst_hazard: &[bool],
    dead_inst: &HashSet<u64>,
    def_in: &[[DefVal; 106]],
    const_in: &[[ConstVal; 106]],
    opt: &Options,
    has_unknown_cf: bool,
) -> HoistedSbufResult {
    let mut out = HoistedSbufResult {
        groups: Vec::new(),
        at_entry: HashMap::new(),
        before_pc: HashMap::new(),
        after_pc: HashMap::new(),
        uses: HashMap::new(),
    };
    if has_unknown_cf {
        return out;
    }

    let num_blocks = block_starts.len();
    let kernel_mode = matches!(opt.mode, Mode::Kernel);

    let mut inst_block = vec![0usize; insts.len()];
    let mut inst_local_idx = vec![0usize; insts.len()];
    let mut block_has_hazard = vec![false; num_blocks];
    let mut block_mod_s = vec![0u128; num_blocks];
    let mut block_indirect_sgpr = vec![false; num_blocks];

    for bi in 0..num_blocks {
        let s_idx = block_ranges[bi].0;
        let e_idx = eff_end[bi];
        for (li, k) in (s_idx..e_idx).enumerate() {
            inst_block[k] = bi;
            inst_local_idx[k] = li;
            if dead_inst.contains(&insts[k].pc) {
                continue;
            }
            if inst_hazard[k] {
                block_has_hazard[bi] = true;
            }
            block_mod_s[bi] |= infos[k].s_mod;
            if infos[k].indirect_sgpr {
                block_indirect_sgpr[bi] = true;
            }
        }
    }

    // Group candidate constant-offset `s_buffer_load_*` by `(sbase, key_d0, key_d1, key_d2)`.
    let mut groups_by_key: BTreeMap<(u8, DefKey, DefKey, DefKey), Vec<SmemCand>> = BTreeMap::new();
    for &bi in &dom.rpo_order {
        let s_idx = block_ranges[bi].0;
        let e_idx = eff_end[bi];
        let mut cur_d = def_in[bi];
        let mut cur_c = const_in[bi];

        for (li, k) in (s_idx..e_idx).enumerate() {
            let d = &insts[k];
            if dead_inst.contains(&d.pc) {
                continue;
            }
            if d.enc.name == "ENC_SMEM" && d.inst.name.starts_with("S_BUFFER_LOAD_") {
                if let Some(count) = smem_load_count(d.inst.name) {
                    let is_glc_dlc = rdna2::field_value(d.enc, &d.words, "GLC") == Some(1)
                        || rdna2::field_value(d.enc, &d.words, "DLC") == Some(1);
                    let sbase = scaled_field(d, "SBASE");
                    let soff = rdna2::field_value(d.enc, &d.words, "SOFFSET").unwrap_or(125);
                    let off = fval(d, "OFFSET") as u32 as i64;

                    if !is_glc_dlc && sbase + 4 <= 106 && soff == 125 && (off & 3) == 0 {
                        let to_key = |dv: DefVal| -> DefKey {
                            match dv {
                                DefVal::Entry => DefKey::Entry,
                                DefVal::Inst(di) => DefKey::Inst(di),
                                DefVal::Top | DefVal::Bottom => DefKey::LocalEntry(bi),
                            }
                        };
                        let key_d0 = to_key(cur_d[sbase as usize]);
                        let key_d1 = to_key(cur_d[sbase as usize + 1]);
                        let key_d2 = to_key(cur_d[sbase as usize + 2]);
                        groups_by_key
                            .entry((sbase as u8, key_d0, key_d1, key_d2))
                            .or_default()
                            .push(SmemCand {
                                bi,
                                local_idx: li,
                                pc: d.pc,
                                off,
                                count: count as u8,
                                end: off + 4 * (count as i64),
                            });
                    }
                }
            }
            step_sgpr_def_const(k, d, &infos[k], &mut cur_d, &mut cur_c);
        }
    }

    let def_block_of = |key: DefKey| -> usize {
        match key {
            DefKey::Entry => 0,
            DefKey::Inst(di) => inst_block[di],
            DefKey::LocalEntry(bi) => bi,
        }
    };

    let validate_cluster = |sbase: u8,
                            keys: [DefKey; 3],
                            cluster: &[SmemCand]|
     -> Option<(usize, usize, HoistLoc)> {
        if cluster.is_empty() {
            return None;
        }
        let mut min_off = i64::MAX;
        let mut max_end = i64::MIN;
        let mut h = cluster[0].bi;
        for u in cluster {
            min_off = min_off.min(u.off);
            max_end = max_end.max(u.end);
            h = dom.lca(h, u.bi)?;
        }
        if max_end - min_off > MAX_SMEM_GROUP_SPAN {
            return None;
        }
        for key in keys {
            if !dom.dominates(def_block_of(key), h) {
                return None;
            }
        }

        let hs_idx = block_ranges[h].0;
        let he_idx = eff_end[h];
        let h_len = he_idx - hs_idx;

        let min_use_in_h = cluster
            .iter()
            .filter(|u| u.bi == h)
            .map(|u| u.local_idx)
            .min();

        let (h_idx, hoist_loc) = if let Some(u_li) = min_use_in_h {
            for key in keys {
                if let DefKey::Inst(di) = key {
                    if inst_block[di] == h && inst_local_idx[di] >= u_li {
                        return None;
                    }
                }
            }
            (u_li, HoistLoc::BeforePc(insts[hs_idx + u_li].pc))
        } else if h_len == 0 {
            (0, HoistLoc::BlockEntry(block_starts[h]))
        } else {
            let last_k = he_idx - 1;
            if is_terminator_inst(&insts[last_k]) {
                let place_li = h_len - 1;
                for key in keys {
                    if let DefKey::Inst(di) = key {
                        if inst_block[di] == h && inst_local_idx[di] >= place_li {
                            return None;
                        }
                    }
                }
                (place_li, HoistLoc::BeforePc(insts[last_k].pc))
            } else {
                (h_len, HoistLoc::AfterPc(insts[last_k].pc))
            }
        };

        let base_mask = sreg_range(sbase as u64, 3);

        let slice_is_clean = |bi: usize, lo: usize, hi: usize| -> bool {
            let bs = block_ranges[bi].0;
            for li in lo..hi {
                let k = bs + li;
                if dead_inst.contains(&insts[k].pc) {
                    continue;
                }
                if inst_hazard[k]
                    || infos[k].indirect_sgpr
                    || (infos[k].s_mod & base_mask) != 0
                {
                    return false;
                }
            }
            true
        };

        let mut checked_h_suffix = false;
        let mut clean_interm_checked: HashSet<usize> = HashSet::new();

        for u in cluster {
            if u.bi == h {
                if u.local_idx < h_idx || !slice_is_clean(h, h_idx, u.local_idx) {
                    return None;
                }
            } else {
                if !checked_h_suffix {
                    if unknown_succ[h] || !slice_is_clean(h, h_idx, h_len) {
                        return None;
                    }
                    checked_h_suffix = true;
                }
                if !slice_is_clean(u.bi, 0, u.local_idx) {
                    return None;
                }
                if !clean_interm_checked.contains(&u.bi) {
                    let mut visited: HashSet<usize> = HashSet::new();
                    let mut queue: VecDeque<usize> = VecDeque::new();
                    for &p in &cfg_preds[u.bi] {
                        if dom.is_reachable(p) && visited.insert(p) {
                            queue.push_back(p);
                        }
                    }
                    while let Some(m) = queue.pop_front() {
                        if m == h {
                            continue;
                        }
                        if m == 0 {
                            return None;
                        }
                        if block_has_hazard[m]
                            || block_indirect_sgpr[m]
                            || (block_mod_s[m] & base_mask) != 0
                            || unknown_succ[m]
                        {
                            return None;
                        }
                        for &p in &cfg_preds[m] {
                            if dom.is_reachable(p) && visited.insert(p) {
                                queue.push_back(p);
                            }
                        }
                    }
                    clean_interm_checked.insert(u.bi);
                }
            }
        }

        Some((h, h_idx, hoist_loc))
    };

    let const_d1_at = |h: usize, h_idx: usize, sbase: u8| -> Option<u32> {
        let mut cur_d = def_in[h];
        let mut cur_c = const_in[h];
        let hs = block_ranges[h].0;
        for li in 0..h_idx {
            let k = hs + li;
            if dead_inst.contains(&insts[k].pc) {
                continue;
            }
            step_sgpr_def_const(k, &insts[k], &infos[k], &mut cur_d, &mut cur_c);
        }
        match cur_c[sbase as usize + 1] {
            ConstVal::Known(v) => Some(v),
            _ => None,
        }
    };

    for ((sbase, key_d0, key_d1, key_d2), cands) in groups_by_key {
        // Skip keys whose candidates all live in a single block (handled by `IntraBlockSbufGroup`
        // with block-local temps or `known_zero_hi_sbuf` inline).
        let first_bi = cands[0].bi;
        if cands.iter().all(|c| c.bi == first_bi) {
            continue;
        }
        let keys = [key_d0, key_d1, key_d2];

        let clusters: Vec<(Vec<SmemCand>, usize, usize, HoistLoc)> =
            if let Some((h, h_idx, loc)) = validate_cluster(sbase, keys, &cands) {
                vec![(cands, h, h_idx, loc)]
            } else {
                let mut open: Vec<(Vec<SmemCand>, usize, usize, HoistLoc)> = Vec::new();
                for cand in cands {
                    let mut placed = false;
                    let start_try = open.len().saturating_sub(4);
                    for entry in open[start_try..].iter_mut().rev() {
                        entry.0.push(cand.clone());
                        if let Some((h, h_idx, loc)) = validate_cluster(sbase, keys, &entry.0) {
                            entry.1 = h;
                            entry.2 = h_idx;
                            entry.3 = loc;
                            placed = true;
                            break;
                        }
                        entry.0.pop();
                    }
                    if !placed {
                        let single = vec![cand];
                        if let Some((h, h_idx, loc)) = validate_cluster(sbase, keys, &single) {
                            open.push((single, h, h_idx, loc));
                        }
                    }
                }
                open
            };

        for (cluster, h, h_idx, hoist_loc) in clusters {
            let pairs: Vec<(i64, u8)> = cluster.iter().map(|u| (u.off, u.count)).collect();
            let (all_loads, pair_map) = dedup_and_subsume_loads(&pairs, true);
            let d1_const = const_d1_at(h, h_idx, sbase);

            let num_chunks = all_loads.len().div_ceil(MAX_SMEM_GROUP_LOADS);
            for chunk_i in 0..num_chunks {
                let lo_l = chunk_i * MAX_SMEM_GROUP_LOADS;
                let hi_l = (lo_l + MAX_SMEM_GROUP_LOADS).min(all_loads.len());
                let chunk_loads: Vec<HoistedLoad> = all_loads[lo_l..hi_l].to_vec();

                let mut chunk_uses: Vec<(&SmemCand, u16, u8, u8)> = Vec::new();
                for u in &cluster {
                    let (global_li, dw_off, parent_cnt) = pair_map[&(u.off, u.count)];
                    let gi = global_li as usize;
                    if gi >= lo_l && gi < hi_l {
                        chunk_uses.push((u, (gi - lo_l) as u16, dw_off, parent_cnt));
                    }
                }
                if chunk_uses.len() < 2 {
                    continue;
                }
                let chunk_first_bi = chunk_uses[0].0.bi;
                let is_cross_block = chunk_uses.iter().any(|(u, ..)| u.bi != chunk_first_bi);
                if !is_cross_block {
                    continue;
                }

                let min_off = chunk_loads.iter().map(|l| l.offset).min().unwrap();
                let max_end = chunk_loads
                    .iter()
                    .map(|l| l.offset + 4 * (l.count as i64))
                    .max()
                    .unwrap();
                let max_last_dw_off = chunk_loads
                    .iter()
                    .map(|l| l.offset + 4 * (l.count as i64 - 1))
                    .max()
                    .unwrap();
                if max_end - min_off > MAX_SMEM_GROUP_SPAN {
                    continue;
                }
                if kernel_mode && max_last_dw_off != min_off {
                    continue;
                }

                let known_hi = if !kernel_mode && d1_const.is_some_and(|w| (w & 0xFFFF) == 0) {
                    Some(0)
                } else {
                    None
                };

                let gid = out.groups.len() as u32;
                let gidx = out.groups.len();
                out.groups.push(HoistedSbufGroup {
                    id: gid,
                    hoist_block: block_starts[h],
                    sbase,
                    known_hi,
                    min_off,
                    max_end,
                    loads: chunk_loads,
                });
                match hoist_loc {
                    HoistLoc::BlockEntry(bpc) => out.at_entry.entry(bpc).or_default().push(gidx),
                    HoistLoc::BeforePc(pc) => out.before_pc.entry(pc).or_default().push(gidx),
                    HoistLoc::AfterPc(pc) => out.after_pc.entry(pc).or_default().push(gidx),
                }
                for (u, load_idx, dword_offset, parent_count) in chunk_uses {
                    out.uses.insert(
                        u.pc,
                        SbufUse {
                            group_id: gid,
                            load_idx,
                            parent_count,
                            dword_offset,
                            count: u.count,
                        },
                    );
                }
            }
        }
    }

    out
}

#[derive(Clone, Debug)]
struct SbufCand {
    pc: u64,
    off: i64,
    count: u8,
}

/// Weapon 1B: Non-consecutive intra-block same-base constant-offset (`SOFFSET == 125`)
/// `s_buffer_load_*` grouping (zero `[loop]`, zero switch-doubling wrap fallbacks).
#[allow(clippy::too_many_arguments)]
fn analyze_intra_block_sbuf(
    insts: &[Decoded],
    infos: &[InstInfo],
    block_ranges: &[(usize, usize)],
    eff_end: &[usize],
    dom: &DomTree,
    inst_hazard: &[bool],
    dead_inst: &HashSet<u64>,
    def_in: &[[DefVal; 106]],
    const_in: &[[ConstVal; 106]],
    hoisted_sbuf_uses: &HashMap<u64, SbufUse>,
    opt: &Options,
) -> (HashMap<u64, IntraBlockSbufGroup>, HashMap<u64, SbufUse>) {
    let mut groups: HashMap<u64, IntraBlockSbufGroup> = HashMap::new();
    let mut uses: HashMap<u64, SbufUse> = HashMap::new();
    let mut next_id = 0u32;
    let kernel_mode = matches!(opt.mode, Mode::Kernel);

    let flush_one = |sbase: u8,
                     known_hi: Option<u32>,
                     cands: Vec<SbufCand>,
                     next_id: &mut u32,
                     groups: &mut HashMap<u64, IntraBlockSbufGroup>,
                     uses: &mut HashMap<u64, SbufUse>| {
        if cands.len() < 2 {
            return;
        }
        let first_pc = cands[0].pc;
        let pairs: Vec<(i64, u8)> = cands.iter().map(|c| (c.off, c.count)).collect();
        let (loads, pair_map) = dedup_and_subsume_loads(&pairs, true);
        let min_off = loads.iter().map(|l| l.offset).min().unwrap();
        let max_end = loads
            .iter()
            .map(|l| l.offset + 4 * (l.count as i64))
            .max()
            .unwrap();
        let max_last_dw_off = loads
            .iter()
            .map(|l| l.offset + 4 * (l.count as i64 - 1))
            .max()
            .unwrap();
        if max_end - min_off > MAX_SMEM_GROUP_SPAN {
            return;
        }
        // In `Mode::Kernel`, ONLY emit when `max_span == 0` so zero `else` wrap fallback is needed.
        if kernel_mode && max_last_dw_off != min_off {
            return;
        }
        let gid = *next_id;
        *next_id += 1;
        groups.insert(
            first_pc,
            IntraBlockSbufGroup {
                id: gid,
                first_pc,
                sbase,
                soff: None,
                known_hi,
                min_off,
                max_end,
                loads,
            },
        );
        for c in cands {
            let (load_idx, dword_offset, parent_count) = pair_map[&(c.off, c.count)];
            uses.insert(
                c.pc,
                SbufUse {
                    group_id: gid,
                    load_idx,
                    parent_count,
                    dword_offset,
                    count: c.count,
                },
            );
        }
    };

    for &bi in &dom.rpo_order {
        let s_idx = block_ranges[bi].0;
        let e_idx = eff_end[bi];
        let mut cur_d = def_in[bi];
        let mut cur_c = const_in[bi];
        // Map `sbase -> (cands, min_off, max_end, uniq_est, known_hi)`.
        let mut active: BTreeMap<u8, (Vec<SbufCand>, i64, i64, usize, Option<u32>)> = BTreeMap::new();

        let flush_all = |active: &mut BTreeMap<u8, (Vec<SbufCand>, i64, i64, usize, Option<u32>)>,
                         next_id: &mut u32,
                         groups: &mut HashMap<u64, IntraBlockSbufGroup>,
                         uses: &mut HashMap<u64, SbufUse>| {
            let entries: Vec<_> = std::mem::take(active).into_iter().collect();
            for (sb, (cands, _, _, _, known_hi)) in entries {
                flush_one(sb, known_hi, cands, next_id, groups, uses);
            }
        };

        for k in s_idx..e_idx {
            let d = &insts[k];
            if dead_inst.contains(&d.pc) {
                continue;
            }
            let info = &infos[k];

            let is_glc_dlc_smem = d.enc.name == "ENC_SMEM"
                && (rdna2::field_value(d.enc, &d.words, "GLC") == Some(1)
                    || rdna2::field_value(d.enc, &d.words, "DLC") == Some(1));
            if inst_hazard[k] || is_glc_dlc_smem {
                flush_all(&mut active, &mut next_id, &mut groups, &mut uses);
                step_sgpr_def_const(k, d, info, &mut cur_d, &mut cur_c);
                continue;
            }

            // Step 1: check if `d` is a valid constant-offset (`SOFFSET == 125`) `s_buffer_load_dword*`
            // not already claimed by `HoistedSbufGroup`, BEFORE breaking on `s_mod` so even if the
            // LAST op in the group overwrites `sbase`, it joins safely!
            if !hoisted_sbuf_uses.contains_key(&d.pc)
                && d.enc.name == "ENC_SMEM"
                && d.inst.name.starts_with("S_BUFFER_LOAD_")
            {
                if let Some(count) = smem_load_count(d.inst.name) {
                    let sbase = scaled_field(d, "SBASE");
                    let soff_raw = rdna2::field_value(d.enc, &d.words, "SOFFSET").unwrap_or(125);
                    let off = fval(d, "OFFSET") as u32 as i64;
                    if soff_raw == 125 && sbase + 4 <= 106 && (off & 3) == 0 {
                        let key = sbase as u8;
                        let end = off + 4 * (count as i64);
                        let cand = SbufCand {
                            pc: d.pc,
                            off,
                            count: count as u8,
                        };
                        let must_flush = active.get(&key).is_some_and(|(_, cur_min, cur_max, uniq_est, _)| {
                            cur_max.max(&end) - cur_min.min(&off) > MAX_SMEM_GROUP_SPAN
                                || *uniq_est >= MAX_SMEM_GROUP_LOADS
                        });
                        if must_flush {
                            if let Some((old_cands, _, _, _, old_hi)) = active.remove(&key) {
                                flush_one(key, old_hi, old_cands, &mut next_id, &mut groups, &mut uses);
                            }
                        }
                        let entry = active.entry(key).or_insert_with(|| {
                            let known_hi = (!kernel_mode
                                && matches!(cur_c[key as usize + 1], ConstVal::Known(w) if (w & 0xFFFF) == 0))
                            .then_some(0);
                            (Vec::new(), off, end, 0, known_hi)
                        });
                        let is_new_pair = !entry.0.iter().any(|c| c.off == off && c.count == cand.count);
                        if is_new_pair {
                            entry.3 += 1;
                        }
                        entry.1 = entry.1.min(off);
                        entry.2 = entry.2.max(end);
                        entry.0.push(cand);
                    }
                }
            }

            // Step 2: flush any active group whose `sbase..sbase+3` (`d0, d1, d2`) was modified by `d`.
            if info.indirect_sgpr {
                flush_all(&mut active, &mut next_id, &mut groups, &mut uses);
            } else if (info.s_mod & S_PLAIN_MASK) != 0 {
                let to_flush: Vec<u8> = active
                    .keys()
                    .copied()
                    .filter(|&sb| (info.s_mod & sreg_range(sb as u64, 3)) != 0)
                    .collect();
                for key in to_flush {
                    if let Some((cands, _, _, _, known_hi)) = active.remove(&key) {
                        flush_one(key, known_hi, cands, &mut next_id, &mut groups, &mut uses);
                    }
                }
            }

            step_sgpr_def_const(k, d, info, &mut cur_d, &mut cur_c);
        }

        flush_all(&mut active, &mut next_id, &mut groups, &mut uses);
    }

    (groups, uses)
}

#[inline]
fn is_valid_mask_reg(r: u64, wave64: bool) -> bool {
    if wave64 {
        r < 105 || r == 106
    } else {
        r < 106 || r == 106
    }
}

#[inline]
fn mask_reg_bits(r: u8, wave64: bool) -> SRegs {
    if wave64 {
        sreg_range(r as u64, 2)
    } else {
        1u128 << r
    }
}

fn classify_lane_local_def(d: &Decoded, wave64: bool) -> Option<(u8, LaneLocalDefKind)> {
    let name = d.inst.name;
    let kind = if name.starts_with("V_CMP_") {
        LaneLocalDefKind::Cmp
    } else if matches!(
        name,
        "V_ADD_CO_U32"
            | "V_SUB_CO_U32"
            | "V_SUBREV_CO_U32"
            | "V_ADD_CO_CI_U32"
            | "V_SUB_CO_CI_U32"
            | "V_SUBREV_CO_CI_U32"
    ) {
        LaneLocalDefKind::CarryOut
    } else {
        return None;
    };
    let (outs, _) = explicit_ops(d);
    match carry_sdst_of(d, &outs) {
        Some(CarryDst::Sgpr(r)) if is_valid_mask_reg(r, wave64) => Some((r as u8, kind)),
        _ => None,
    }
}

/// Returns `(mask_reg, non_mask_s_read)` for per-lane mask consumers (`V_CNDMASK_B32` / `V_ADD/SUB*_CO_CI_U32`).
fn classify_lane_local_reader(d: &Decoded, wave64: bool) -> Option<(u8, SRegs)> {
    if d.variant != Variant::Plain {
        return None;
    }
    if !matches!(
        d.inst.name,
        "V_CNDMASK_B32" | "V_ADD_CO_CI_U32" | "V_SUB_CO_CI_U32" | "V_SUBREV_CO_CI_U32"
    ) {
        return None;
    }
    let mask_raw = if d.enc.field("SRC2").is_some() {
        scaled_field(d, "SRC2")
    } else {
        106 // implicit OPR_VCC
    };
    if !is_valid_mask_reg(mask_raw, wave64) {
        return None;
    }

    // Collect any SGPR read in data operands `SRC0` / `SRC1` (never treat a data read as lane-local!).
    let mut non_mask_s: SRegs = 0;
    for f in ["SRC0", "SRC1"] {
        if d.enc.field(f).is_some() {
            let code = scaled_field(d, f);
            if code < 256 {
                non_mask_s |= scalar_code_regs(code, 1, wave64).0;
            }
        }
    }
    Some((mask_raw as u8, non_mask_s))
}

fn classify_bitwise_mask_op(
    d: &Decoded,
    wave64: bool,
    is_dead_scc: bool,
) -> Option<(LaneLocalBitOp, u8, u64, Option<u64>)> {
    let op = match (wave64, d.inst.name) {
        (false, "S_MOV_B32") | (true, "S_MOV_B64") => LaneLocalBitOp::Mov,
        (false, "S_NOT_B32") | (true, "S_NOT_B64") => LaneLocalBitOp::Not,
        (false, "S_AND_B32") | (true, "S_AND_B64") => LaneLocalBitOp::And,
        (false, "S_OR_B32") | (true, "S_OR_B64") => LaneLocalBitOp::Or,
        (false, "S_XOR_B32") | (true, "S_XOR_B64") => LaneLocalBitOp::Xor,
        (false, "S_ANDN2_B32") | (true, "S_ANDN2_B64") => LaneLocalBitOp::Andn2,
        (false, "S_ORN2_B32") | (true, "S_ORN2_B64") => LaneLocalBitOp::Orn2,
        (false, "S_NAND_B32") | (true, "S_NAND_B64") => LaneLocalBitOp::Nand,
        (false, "S_NOR_B32") | (true, "S_NOR_B64") => LaneLocalBitOp::Nor,
        (false, "S_XNOR_B32") | (true, "S_XNOR_B64") => LaneLocalBitOp::Xnor,
        _ => return None,
    };
    if op != LaneLocalBitOp::Mov && !is_dead_scc {
        return None;
    }
    let sdst = fval(d, "SDST");
    if !is_valid_mask_reg(sdst, wave64) {
        return None;
    }
    let s0 = fval(d, "SSRC0");
    let s1 = if matches!(op, LaneLocalBitOp::Mov | LaneLocalBitOp::Not) {
        None
    } else {
        Some(fval(d, "SSRC1"))
    };
    Some((op, sdst as u8, s0, s1))
}

#[derive(Clone, Copy)]
enum RawLlOp {
    Node(usize),
    Exec,
    Const(bool),
    Sgpr(u8),
}

enum LlNodeKind {
    Valu(LaneLocalDefKind),
    Bitwise {
        op: LaneLocalBitOp,
        src0: RawLlOp,
        src1: Option<RawLlOp>,
    },
}

struct LlNode {
    pc: u64,
    local_idx: usize,
    reg: u8,
    kind: LlNodeKind,
    invalid: bool,
    direct_uses: Vec<usize>,
    bitwise_consumers: Vec<usize>,
}

struct LaneLocalResult {
    defs: HashMap<u64, LaneLocalDef>,
    bitwise: HashMap<u64, LaneLocalBitwise>,
    uses: HashMap<u64, u32>,
}

/// Weapon 3: Wave32 & native-Wave64 lane-local `bool` mask scalarization.
///
/// **Convergence & hazard model**:
/// - Strictly **intra-block**: never moves, reorders, or deduplicates `WaveActiveBallot` across
///   branches, merges, or loops; every chain must die before block exit (`!live_out_s[bi]`).
/// - **Zero cross-lane observation**: every consumer must be a per-lane mask reader
///   (`V_CNDMASK_B32`, `V_ADD/SUB*_CO_CI_U32` carry-in, or a dead-`scc` bitwise mask op whose own
///   consumers are also lane-local). Any whole-mask read (`S_CBRANCH_VCC*`, `S_BCNT*`, `S_FF*`,
///   `EXEC` writes, scalar data operands, memory operands, indirect/unknown ops) aborts the chain.
/// - **Divergent `EXEC` restore safety**: if any `EXEC`-widening instruction executes between a
///   def and its last use in the block, `needs_exec_mask = true` forces `and(act, cond)` at the
///   def site so inactive lanes stay `0` even if `EXEC` is widened before the consumer.
/// - Disabled in `Mode::Kernel` (epilogue dumps SGPR masks) and `split_wave64` (`x64_masks.rs`).
#[allow(clippy::too_many_arguments)]
fn analyze_lane_local_masks(
    insts: &[Decoded],
    infos: &[InstInfo],
    block_ranges: &[(usize, usize)],
    eff_end: &[usize],
    live_out_s: &[SRegs],
    dead_scc: &HashSet<u64>,
    dead_carry_sdst: &HashSet<u64>,
    dead_inst: &HashSet<u64>,
    opt: &Options,
) -> LaneLocalResult {
    let mut out = LaneLocalResult {
        defs: HashMap::new(),
        bitwise: HashMap::new(),
        uses: HashMap::new(),
    };

    // Disabled in `Mode::Kernel` (register dump tests) and `split_wave64` (0% overlap with `x64_masks.rs`).
    if matches!(opt.mode, Mode::Kernel) || (opt.wave64 && opt.split_wave64) {
        return out;
    }

    let wave64 = opt.wave64;
    let mut next_ml_id = 0u32;
    let mask_range_bits: SRegs = (1u128 << 108) - 1; // s0..s105 + vcc(106..107)

    for bi in 0..block_ranges.len() {
        let s_idx = block_ranges[bi].0;
        let e_idx = eff_end[bi];
        let slice = &insts[s_idx..e_idx];
        let info_slice = &infos[s_idx..e_idx];

        let mut nodes: Vec<LlNode> = Vec::new();
        let mut active: [Option<usize>; 108] = [None; 108];
        let mut reader_uses: Vec<(u64, usize)> = Vec::new();

        let resolve_src = |code: u64, literal: Option<u32>, active: &[Option<usize>; 108]| -> Option<RawLlOp> {
            if code == 126 {
                return Some(RawLlOp::Exec);
            }
            if is_valid_mask_reg(code, wave64) {
                let r = code as usize;
                if let Some(ni) = active[r] {
                    if !wave64 || active[r + 1] == Some(ni) {
                        return Some(RawLlOp::Node(ni));
                    }
                    return None;
                }
                if wave64 && active[r + 1].is_some() {
                    return None;
                }
                return Some(RawLlOp::Sgpr(code as u8));
            }
            if code == 128 {
                return Some(RawLlOp::Const(false));
            }
            if code == 193 {
                return Some(RawLlOp::Const(true));
            }
            if code == 255 {
                if literal == Some(0) {
                    return Some(RawLlOp::Const(false));
                }
                if !wave64 && literal == Some(u32::MAX) {
                    return Some(RawLlOp::Const(true));
                }
            }
            None
        };

        for (li, (d, info)) in slice.iter().zip(info_slice.iter()).enumerate() {
            if dead_inst.contains(&d.pc) {
                continue;
            }

            let mut pending_bitwise: Option<(LaneLocalBitOp, u8, RawLlOp, Option<RawLlOp>)> = None;

            // Step A: classify reads.
            if info.indirect_sgpr {
                for slot in active.iter_mut() {
                    if let Some(ni) = slot.take() {
                        nodes[ni].invalid = true;
                    }
                }
                continue;
            }

            if let Some((mask_reg, non_mask_s)) = classify_lane_local_reader(d, wave64) {
                // Any active mask reg read as a 32-bit DATA operand is NOT lane-local!
                let mut m = non_mask_s & mask_range_bits;
                while m != 0 {
                    let r = m.trailing_zeros() as usize;
                    m &= m - 1;
                    if let Some(ni) = active[r] {
                        nodes[ni].invalid = true;
                    }
                }
                // Check aligned active node on `mask_reg`.
                let mr = mask_reg as usize;
                let matched = match active[mr] {
                    Some(ni) if !wave64 || active[mr + 1] == Some(ni) => Some(ni),
                    _ => None,
                };
                if let Some(ni) = matched {
                    reader_uses.push((d.pc, ni));
                    nodes[ni].direct_uses.push(li);
                    // Any OTHER active mask reg read by this instruction is non-lane-local.
                    let other_reads = (info.s_read & mask_range_bits) & !mask_reg_bits(mask_reg, wave64);
                    let mut om = other_reads;
                    while om != 0 {
                        let r = om.trailing_zeros() as usize;
                        om &= om - 1;
                        if let Some(oni) = active[r] {
                            nodes[oni].invalid = true;
                        }
                    }
                } else {
                    // Misaligned or not active -> invalidate any active mask reg read by `info.s_read`.
                    let mut om = info.s_read & mask_range_bits;
                    while om != 0 {
                        let r = om.trailing_zeros() as usize;
                        om &= om - 1;
                        if let Some(oni) = active[r] {
                            nodes[oni].invalid = true;
                        }
                    }
                }
            } else if let Some((op, sdst, s0, s1)) = classify_bitwise_mask_op(d, wave64, dead_scc.contains(&d.pc)) {
                let r0 = resolve_src(s0, d.literal, &active);
                let r1 = match s1 {
                    Some(c1) => resolve_src(c1, d.literal, &active).map(Some),
                    None => Some(None),
                };
                let has_local_input = matches!(r0, Some(RawLlOp::Node(_)))
                    || matches!(r1, Some(Some(RawLlOp::Node(_))));
                if has_local_input {
                    if let (Some(op0), Some(op1)) = (r0, r1) {
                        pending_bitwise = Some((op, sdst, op0, op1));
                    }
                }
                if pending_bitwise.is_none() {
                    let mut om = info.s_read & mask_range_bits;
                    while om != 0 {
                        let r = om.trailing_zeros() as usize;
                        om &= om - 1;
                        if let Some(oni) = active[r] {
                            nodes[oni].invalid = true;
                        }
                    }
                }
            } else {
                // Regular instruction: every read of an active mask reg is a whole-mask read -> invalidate!
                let mut om = info.s_read & mask_range_bits;
                while om != 0 {
                    let r = om.trailing_zeros() as usize;
                    om &= om - 1;
                    if let Some(oni) = active[r] {
                        nodes[oni].invalid = true;
                    }
                }
            }

            // Step B: process kills & partial/conditional writes on `active`.
            let mod_mask = info.s_mod & mask_range_bits;
            if mod_mask != 0 {
                let mut m = mod_mask;
                while m != 0 {
                    let r = m.trailing_zeros() as usize;
                    m &= m - 1;
                    if let Some(ni) = active[r] {
                        let nreg = nodes[ni].reg;
                        let nbits = mask_reg_bits(nreg, wave64);
                        if (info.s_kill & nbits) != nbits {
                            // Partial or conditional write preserves part of the SGPR -> invalidate!
                            nodes[ni].invalid = true;
                        }
                        active[nreg as usize] = None;
                        if wave64 {
                            active[nreg as usize + 1] = None;
                        }
                    }
                }
            }

            // Step C: create new candidate def node.
            if let Some((op, sdst, op0, op1)) = pending_bitwise {
                let new_ni = nodes.len();
                if let RawLlOp::Node(p) = op0 {
                    nodes[p].bitwise_consumers.push(new_ni);
                }
                if let Some(RawLlOp::Node(p)) = op1 {
                    if !matches!(op0, RawLlOp::Node(p0) if p0 == p) {
                        nodes[p].bitwise_consumers.push(new_ni);
                    }
                }
                nodes.push(LlNode {
                    pc: d.pc,
                    local_idx: li,
                    reg: sdst,
                    kind: LlNodeKind::Bitwise {
                        op,
                        src0: op0,
                        src1: op1,
                    },
                    invalid: false,
                    direct_uses: Vec::new(),
                    bitwise_consumers: Vec::new(),
                });
                active[sdst as usize] = Some(new_ni);
                if wave64 {
                    active[sdst as usize + 1] = Some(new_ni);
                }
            } else if !dead_carry_sdst.contains(&d.pc) {
                if let Some((sdst, kind)) = classify_lane_local_def(d, wave64) {
                    let new_ni = nodes.len();
                    nodes.push(LlNode {
                        pc: d.pc,
                        local_idx: li,
                        reg: sdst,
                        kind: LlNodeKind::Valu(kind),
                        invalid: false,
                        direct_uses: Vec::new(),
                        bitwise_consumers: Vec::new(),
                    });
                    active[sdst as usize] = Some(new_ni);
                    if wave64 {
                        active[sdst as usize + 1] = Some(new_ni);
                    }
                }
            }
        }

        // Step D: any def still active at block exit that is live-out MUST remain a real SGPR!
        for slot in &active {
            if let Some(ni) = *slot {
                let nbits = mask_reg_bits(nodes[ni].reg, wave64);
                if (live_out_s[bi] & nbits) != 0 {
                    nodes[ni].invalid = true;
                }
            }
        }

        // Step E: propagate invalidity cleanly through bitwise chains (including zero-consumer
        // bitwise ops that were not dead-eliminated, so their producers keep writing real SGPRs!).
        loop {
            let mut changed = false;
            let mut has_consumer = vec![false; nodes.len()];
            for ni in (0..nodes.len()).rev() {
                if nodes[ni].invalid {
                    continue;
                }
                has_consumer[ni] = !nodes[ni].direct_uses.is_empty()
                    || nodes[ni]
                        .bitwise_consumers
                        .iter()
                        .any(|&c| !nodes[c].invalid && has_consumer[c]);
            }
            for ni in 0..nodes.len() {
                if nodes[ni].invalid {
                    continue;
                }
                // If any bitwise consumer became invalid, it will run as a normal SGPR op and read our SGPR!
                if nodes[ni].bitwise_consumers.iter().any(|&c| nodes[c].invalid) {
                    nodes[ni].invalid = true;
                    changed = true;
                    continue;
                }
                // A bitwise node with no live consumer or zero remaining valid Local inputs must not be rewritten.
                if let LlNodeKind::Bitwise { src0, src1, .. } = nodes[ni].kind {
                    let s0_valid_local = matches!(src0, RawLlOp::Node(p) if !nodes[p].invalid);
                    let s1_valid_local = matches!(src1, Some(RawLlOp::Node(p)) if !nodes[p].invalid);
                    if !has_consumer[ni] || (!s0_valid_local && !s1_valid_local) {
                        nodes[ni].invalid = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Step F: compute max transitive consumer index (reverse topological order) & emit valid nodes.
        let mut max_use_idx: Vec<Option<usize>> = vec![None; nodes.len()];
        for ni in (0..nodes.len()).rev() {
            if nodes[ni].invalid {
                continue;
            }
            let mut mx: Option<usize> = nodes[ni].direct_uses.iter().copied().max();
            for &c in &nodes[ni].bitwise_consumers {
                if !nodes[c].invalid {
                    if let Some(cm) = max_use_idx[c] {
                        mx = Some(mx.map_or(cm, |cur| cur.max(cm)));
                    }
                }
            }
            max_use_idx[ni] = mx;
        }

        let mut global_id: Vec<Option<u32>> = vec![None; nodes.len()];
        for ni in 0..nodes.len() {
            if nodes[ni].invalid || max_use_idx[ni].is_none() {
                continue;
            }
            let id = next_ml_id;
            next_ml_id += 1;
            global_id[ni] = Some(id);
        }

        let map_op = |op: RawLlOp| -> LaneLocalOperand {
            match op {
                RawLlOp::Node(p) => match global_id[p] {
                    Some(pid) => LaneLocalOperand::Local(pid),
                    None => LaneLocalOperand::Sgpr(nodes[p].reg),
                },
                RawLlOp::Exec => LaneLocalOperand::Exec,
                RawLlOp::Const(b) => LaneLocalOperand::Const(b),
                RawLlOp::Sgpr(r) => LaneLocalOperand::Sgpr(r),
            }
        };

        for (ni, node) in nodes.iter().enumerate() {
            let Some(id) = global_id[ni] else {
                continue;
            };
            match node.kind {
                LlNodeKind::Valu(kind) => {
                    let end_idx = max_use_idx[ni].unwrap();
                    let needs_exec_mask = info_slice[node.local_idx + 1..end_idx]
                        .iter()
                        .any(|info| info.exec_widen);
                    out.defs.insert(
                        node.pc,
                        LaneLocalDef {
                            id,
                            needs_exec_mask,
                            kind,
                        },
                    );
                }
                LlNodeKind::Bitwise { op, src0, src1 } => {
                    out.bitwise.insert(
                        node.pc,
                        LaneLocalBitwise {
                            dst_id: id,
                            op,
                            src0: map_op(src0),
                            src1: src1.map(map_op),
                        },
                    );
                }
            }
        }

        for (use_pc, ni) in reader_uses {
            if let Some(gid) = global_id[ni] {
                out.uses.insert(use_pc, gid);
            }
        }
    }

    out
}

/// Intra-block forward constant propagation for `const_srcs` and `const_sgprs`, seeded with
/// CFG-wide proven SGPR constants at block entry.
fn propagate_block_constants(
    slice: &[Decoded],
    infos: &[InstInfo],
    init_s: [Option<u32>; 106],
    const_srcs: &mut HashMap<u64, [Option<u32>; 3]>,
    const_sgprs: &mut HashMap<(u64, u8), u32>,
) {
    let mut known_s: [Option<u32>; 106] = init_s;
    let mut known_v: [Option<u32>; 256] = [None; 256];

    let resolve_scalar = |d: &Decoded, code: u64, f16: bool, ks: &[Option<u32>; 106]| -> Option<u32> {
        if code < 106 {
            ks[code as usize]
        } else {
            inline_const_bits(code, d.literal, f16)
        }
    };

    let resolve_valu = |d: &Decoded, code: u64, f16: bool, ks: &[Option<u32>; 106], kv: &[Option<u32>; 256]| -> Option<u32> {
        if code >= 256 && code < 512 {
            kv[(code - 256) as usize]
        } else {
            resolve_scalar(d, code, f16, ks)
        }
    };

    for (d, info) in slice.iter().zip(infos.iter()) {
        // Record known SGPR values for every plain SGPR read by this instruction.
        let read_plain = info.s_read & S_PLAIN_MASK;
        if read_plain != 0 {
            let mut m = read_plain;
            while m != 0 {
                let r = m.trailing_zeros() as usize;
                m &= m - 1;
                if let Some(v) = known_s[r] {
                    const_sgprs.insert((d.pc, r as u8), v);
                }
            }
        }

        // Record known 32-bit operand values `[src0, src1, src2]` for plain VALU/VOPC instructions.
        if d.variant == Variant::Plain && (d.enc.name.starts_with("ENC_VOP") || d.enc.name.starts_with("VOP")) {
            let name = d.inst.name;
            let src_f16 = matches!(
                name,
                "V_CVT_F32_F16" | "V_PACK_B32_F16" | "V_CVT_PKNORM_I16_F16" | "V_CVT_PKNORM_U16_F16"
            ) || (name.starts_with("V_CMP") && name.ends_with("_F16"))
                || (name.starts_with("V_PK_") && name.ends_with("_F16"));
            let (_, ins) = explicit_ops(d);
            let mut vals: [Option<u32>; 3] = [None; 3];
            let mut any = false;
            for (idx, o) in ins.iter().take(3).enumerate() {
                if o.size > 32 {
                    continue;
                }
                let v = match o.field {
                    Some("LITERAL") => d.literal,
                    Some(f) => {
                        let raw = scaled_field(d, f);
                        let width = d.enc.field_width(f);
                        if o.ty.starts_with("OPR_VGPR") && width <= 8 {
                            if raw < 256 {
                                known_v[raw as usize]
                            } else {
                                None
                            }
                        } else if o.ty == "OPR_SIMM16" {
                            Some(raw as u32)
                        } else if o.ty == "OPR_SIMM32" {
                            d.literal
                        } else {
                            resolve_valu(d, raw, src_f16, &known_s, &known_v)
                        }
                    }
                    None => None,
                };
                if v.is_some() {
                    vals[idx] = v;
                    any = true;
                }
            }
            if any {
                const_srcs.insert(d.pc, vals);
            }
        }

        // Compute newly produced constant(s) BEFORE invalidating modified registers.
        let name = d.inst.name;
        let mut new_s: [(usize, u32); 2] = [(usize::MAX, 0); 2];
        let mut new_s_len = 0usize;
        let mut new_v: Option<(usize, u32)> = None;

        match name {
            "S_MOV_B32" => {
                let sd = fval(d, "SDST") as usize;
                if sd < 106 {
                    if let Some(v) = resolve_scalar(d, fval(d, "SSRC0"), false, &known_s) {
                        new_s[0] = (sd, v);
                        new_s_len = 1;
                    }
                }
            }
            "S_MOVK_I32" => {
                let sd = fval(d, "SDST") as usize;
                if sd < 106 {
                    let v = fval(d, "SIMM16") as u16 as i16 as i32 as u32;
                    new_s[0] = (sd, v);
                    new_s_len = 1;
                }
            }
            "S_MOV_B64" => {
                let sd = fval(d, "SDST") as usize;
                let s0 = fval(d, "SSRC0");
                if sd + 1 < 106 {
                    match s0 {
                        0..=104 => {
                            if let (Some(lo), Some(hi)) = (known_s[s0 as usize], known_s[s0 as usize + 1]) {
                                new_s[0] = (sd, lo);
                                new_s[1] = (sd + 1, hi);
                                new_s_len = 2;
                            }
                        }
                        128..=208 => {
                            if let Some(v32) = inline_const_bits(s0, d.literal, false) {
                                let v64 = v32 as i32 as i64;
                                new_s[0] = (sd, v64 as u32);
                                new_s[1] = (sd + 1, (v64 >> 32) as u32);
                                new_s_len = 2;
                            }
                        }
                        255 => {
                            if let Some(lit) = d.literal {
                                new_s[0] = (sd, lit);
                                new_s[1] = (sd + 1, 0);
                                new_s_len = 2;
                            }
                        }
                        _ => {}
                    }
                }
            }
            "S_OR_B32" | "S_AND_B32" | "S_XOR_B32" | "S_ADD_U32" | "S_SUB_U32" | "S_LSHL_B32" | "S_LSHR_B32"
            | "S_PACK_LL_B32_B16" | "S_PACK_LH_B32_B16" | "S_PACK_HH_B32_B16" => {
                let sd = fval(d, "SDST") as usize;
                if sd < 106 {
                    if let (Some(a), Some(b)) = (
                        resolve_scalar(d, fval(d, "SSRC0"), false, &known_s),
                        resolve_scalar(d, fval(d, "SSRC1"), false, &known_s),
                    ) {
                        let v = match name {
                            "S_OR_B32" => a | b,
                            "S_AND_B32" => a & b,
                            "S_XOR_B32" => a ^ b,
                            "S_ADD_U32" => a.wrapping_add(b),
                            "S_SUB_U32" => a.wrapping_sub(b),
                            "S_LSHL_B32" => a << (b & 31),
                            "S_LSHR_B32" => a >> (b & 31),
                            "S_PACK_LL_B32_B16" => (a & 0xFFFF) | (b << 16),
                            "S_PACK_LH_B32_B16" => (a & 0xFFFF) | (b & 0xFFFF_0000),
                            "S_PACK_HH_B32_B16" => (a >> 16) | (b & 0xFFFF_0000),
                            _ => unreachable!(),
                        };
                        new_s[0] = (sd, v);
                        new_s_len = 1;
                    }
                }
            }
            "S_ADDK_I32" => {
                let sd = fval(d, "SDST") as usize;
                if sd < 106 {
                    if let Some(cur) = known_s[sd] {
                        let k = fval(d, "SIMM16") as u16 as i16 as i32 as u32;
                        new_s[0] = (sd, cur.wrapping_add(k));
                        new_s_len = 1;
                    }
                }
            }
            "V_MOV_B32" | "V_MOV_FED_B32" if d.variant == Variant::Plain => {
                let vd = fval(d, "VDST") as usize;
                if vd < 256 {
                    if let Some(v) = resolve_valu(d, fval(d, "SRC0"), false, &known_s, &known_v) {
                        new_v = Some((vd, v));
                    }
                }
            }
            _ => {}
        }

        // Invalidate modified SGPRs / VGPRs (and clear on indirect writes or EXEC widening).
        if info.indirect_sgpr {
            known_s = [None; 106];
        } else {
            let mut m = info.s_mod & S_PLAIN_MASK;
            while m != 0 {
                let r = m.trailing_zeros() as usize;
                m &= m - 1;
                known_s[r] = None;
            }
        }
        if info.indirect_vgpr || info.exec_widen {
            known_v = [None; 256];
        } else if !info.v_mod.is_empty() {
            for word_idx in 0..4 {
                let mut w = info.v_mod.0[word_idx];
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    w &= w - 1;
                    known_v[word_idx * 64 + bit] = None;
                }
            }
        }

        for &(r, v) in &new_s[..new_s_len] {
            known_s[r] = Some(v);
        }
        if let Some((r, v)) = new_v {
            if !info.exec_widen {
                known_v[r] = Some(v);
            }
        }
    }
}

/// Analyze a decoded RDNA2 shader program and return optimization metadata.
pub(crate) fn analyze(insts: &[Decoded], leaders: &[u64], opt: &Options) -> ProgramOpt {
    if insts.is_empty() {
        return ProgramOpt {
            mem_coherent: false,
            ..ProgramOpt::default()
        };
    }

    let mem_coherent = compute_mem_coherent(insts);

    // Build sorted, deduplicated block leader PCs matching `mod.rs`.
    let mut block_starts: Vec<u64> = Vec::with_capacity(leaders.len().max(1));
    for &l in leaders {
        let idx = insts.partition_point(|d| d.pc < l);
        if idx < insts.len() && insts[idx].pc == l {
            block_starts.push(l);
        }
    }
    if block_starts.is_empty() || block_starts[0] != insts[0].pc {
        block_starts.push(insts[0].pc);
    }
    block_starts.sort_unstable();
    block_starts.dedup();

    let num_blocks = block_starts.len();
    let mut pc_to_block: HashMap<u64, usize> = HashMap::with_capacity(num_blocks);
    for (bi, &pc) in block_starts.iter().enumerate() {
        pc_to_block.insert(pc, bi);
    }

    let mut block_ranges: Vec<(usize, usize)> = Vec::with_capacity(num_blocks);
    for bi in 0..num_blocks {
        let s_idx = insts.partition_point(|d| d.pc < block_starts[bi]);
        let e_idx = if bi + 1 < num_blocks {
            insts.partition_point(|d| d.pc < block_starts[bi + 1])
        } else {
            insts.len()
        };
        block_ranges.push((s_idx, e_idx.max(s_idx)));
    }

    // Per-instruction semantic analysis.
    let infos: Vec<InstInfo> = insts.iter().map(|d| analyze_inst(d, opt)).collect();

    // Geometry ES -> GS jump target, if any.
    let gs_jump = match &opt.mode {
        Mode::Geometry(g) => Some(g.gs_start.filter(|&s| s > 0)),
        _ => None,
    };

    // Build CFG edges and block-level liveness transfer summaries.
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
    let mut is_end: Vec<bool> = vec![false; num_blocks];
    let mut unknown_succ: Vec<bool> = vec![false; num_blocks];
    let mut gen_s: Vec<SRegs> = vec![0; num_blocks];
    let mut kill_s: Vec<SRegs> = vec![0; num_blocks];
    let mut gen_v: Vec<VRegs> = vec![VRegs::EMPTY; num_blocks];
    let mut gen_scc: Vec<bool> = vec![false; num_blocks];
    let mut kill_scc: Vec<bool> = vec![false; num_blocks];
    let mut eff_end: Vec<usize> = Vec::with_capacity(num_blocks);

    for bi in 0..num_blocks {
        let (s_idx, e_idx) = block_ranges[bi];
        if s_idx == e_idx {
            eff_end.push(e_idx);
            if bi + 1 < num_blocks {
                succs[bi].push(bi + 1);
            } else {
                is_end[bi] = true;
            }
            continue;
        }

        // Stop at the first terminator/branch inside the block (matches `emit_block_insts`).
        let mut last_idx = e_idx - 1;
        for k in s_idx..e_idx {
            let d = &insts[k];
            let is_term = d.far_target.is_some()
                || rdna2::branch_target(d).is_some()
                || d.inst.flags & F_TERMINATOR != 0
                || d.inst.flags & F_INDIRECT != 0
                || matches!(
                    d.inst.name,
                    "S_ENDPGM" | "S_ENDPGM_SAVED" | "S_ENDPGM_ORDERED_PS_DONE" | "S_SETPC_B64"
                );
            if is_term {
                last_idx = k;
                break;
            }
        }
        eff_end.push(last_idx + 1);

        // Compute block-level gen/kill summary (VGPRs have ZERO kill!).
        let mut gs: SRegs = 0;
        let mut ks: SRegs = 0;
        let mut gv = VRegs::EMPTY;
        let mut gscc = false;
        let mut kscc = false;
        for info in &infos[s_idx..=last_idx] {
            gs |= info.s_read & !ks;
            ks |= info.s_kill;
            gv = gv.union(info.v_read);
            if info.scc_read && !kscc {
                gscc = true;
            }
            if info.scc_write {
                kscc = true;
            }
        }
        gen_s[bi] = gs;
        kill_s[bi] = ks;
        gen_v[bi] = gv;
        gen_scc[bi] = gscc;
        kill_scc[bi] = kscc;

        // Determine block successors.
        let last = &insts[last_idx];
        let lname = last.inst.name;
        if let Some(t) = last.far_target {
            if let Some(&tb) = pc_to_block.get(&t) {
                succs[bi].push(tb);
            } else {
                unknown_succ[bi] = true;
            }
        } else if lname == "S_SETPC_B64" {
            if let Some(gs_target) = gs_jump {
                if let Some(t) = gs_target {
                    if let Some(&tb) = pc_to_block.get(&t) {
                        succs[bi].push(tb);
                    } else {
                        unknown_succ[bi] = true;
                    }
                    if opt.wave64 && opt.split_wave64 {
                        is_end[bi] = true;
                    }
                } else {
                    is_end[bi] = true;
                }
            } else {
                unknown_succ[bi] = true;
            }
        } else if lname == "S_SWAPPC_B64" {
            if fval(last, "SDST") == fval(last, "SSRC0") {
                // Ignored fetch-shader call: falls through.
                if bi + 1 < num_blocks {
                    succs[bi].push(bi + 1);
                } else {
                    is_end[bi] = true;
                }
            } else {
                unknown_succ[bi] = true;
            }
        } else if last.inst.flags & F_INDIRECT != 0 || lname.starts_with("S_CALL") {
            unknown_succ[bi] = true;
        } else if let Some(t) = rdna2::branch_target(last) {
            if let Some(&tb) = pc_to_block.get(&t) {
                succs[bi].push(tb);
            } else {
                unknown_succ[bi] = true;
            }
            if last.inst.flags & F_CONDITIONAL != 0 {
                if bi + 1 < num_blocks {
                    succs[bi].push(bi + 1);
                } else {
                    is_end[bi] = true;
                }
            }
        } else if matches!(lname, "S_ENDPGM" | "S_ENDPGM_SAVED" | "S_ENDPGM_ORDERED_PS_DONE") {
            is_end[bi] = true;
        } else if last.inst.flags & F_TERMINATOR != 0 {
            unknown_succ[bi] = true;
        } else if bi + 1 < num_blocks {
            succs[bi].push(bi + 1);
        } else {
            is_end[bi] = true;
        }
    }

    for bi in 0..num_blocks {
        for &sb in &succs[bi] {
            preds[sb].push(bi);
        }
    }

    // `LOOP_GUARD` exits (matching `x64_masks.rs`): any block inside a loop `[a, b)` may
    // exit via `xcl_budget == 0` to the first block at `>= b` (or program end).
    let loops = super::loop_ranges(insts);
    let mut loop_guard_exits: Vec<Vec<Option<usize>>> = vec![Vec::new(); num_blocks];
    for &(a, b) in &loops {
        let first_bi = block_starts.partition_point(|&pc| pc < a);
        let after_bi = block_starts.partition_point(|&pc| pc < b);
        let exit_blk = (after_bi < num_blocks).then_some(after_bi);
        for exits in &mut loop_guard_exits[first_bi..after_bi] {
            if !exits.contains(&exit_blk) {
                exits.push(exit_blk);
            }
        }
    }

    // Guardrail #6: in `Mode::Kernel`, all SGPRs, VGPRs, and `scc` are live at `s_endpgm`.
    let kernel_mode = matches!(opt.mode, Mode::Kernel);
    let (end_s, end_v, end_scc) = if kernel_mode {
        (S_ALL, VRegs::ALL, true)
    } else {
        (S_EXEC, VRegs::EMPTY, false)
    };

    // Block-level backward liveness fixpoint.
    let mut live_in_s: Vec<SRegs> = vec![S_EXEC; num_blocks];
    let mut live_out_s: Vec<SRegs> = vec![S_EXEC; num_blocks];
    let mut live_in_v: Vec<VRegs> = vec![VRegs::EMPTY; num_blocks];
    let mut live_out_v: Vec<VRegs> = vec![VRegs::EMPTY; num_blocks];
    let mut live_in_scc: Vec<bool> = vec![false; num_blocks];
    let mut live_out_scc: Vec<bool> = vec![false; num_blocks];

    loop {
        let mut changed = false;
        for bi in (0..num_blocks).rev() {
            let (mut out_s, mut out_v, mut out_scc) = if unknown_succ[bi] {
                (S_ALL, VRegs::ALL, true)
            } else {
                let mut os = S_EXEC;
                let mut ov = VRegs::EMPTY;
                let mut oscc = false;
                if is_end[bi] {
                    os |= end_s;
                    ov = ov.union(end_v);
                    oscc |= end_scc;
                }
                for &sb in &succs[bi] {
                    os |= live_in_s[sb];
                    ov = ov.union(live_in_v[sb]);
                    oscc |= live_in_scc[sb];
                }
                (os, ov, oscc)
            };

            // Add `LOOP_GUARD` exits (leaving to after-loop or program end).
            for &exit in &loop_guard_exits[bi] {
                out_s |= end_s;
                out_v = out_v.union(end_v);
                out_scc |= end_scc;
                if let Some(eb) = exit {
                    out_s |= live_in_s[eb];
                    out_v = out_v.union(live_in_v[eb]);
                    out_scc |= live_in_scc[eb];
                }
            }

            let in_s = gen_s[bi] | (out_s & !kill_s[bi]) | S_EXEC;
            let in_v = gen_v[bi].union(out_v); // ZERO VGPR kill (Guardrail #1)!
            let in_scc = gen_scc[bi] || (out_scc && !kill_scc[bi]);

            if out_s != live_out_s[bi]
                || in_s != live_in_s[bi]
                || out_v != live_out_v[bi]
                || in_v != live_in_v[bi]
                || out_scc != live_out_scc[bi]
                || in_scc != live_in_scc[bi]
            {
                live_out_s[bi] = out_s;
                live_in_s[bi] = in_s;
                live_out_v[bi] = out_v;
                live_in_v[bi] = in_v;
                live_out_scc[bi] = out_scc;
                live_in_scc[bi] = in_scc;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Per-instruction backward sweep for `dead_scc`, `dead_carry_sdst`, `dead_inst`.
    // In `Mode::Kernel` (test harness) or synthetic graphics-only unit tests with zero observable
    // side-effecting ops (no exports, memory, LDS, or sendmsg), skip `dead_inst` so isolated
    // single-instruction tests preserve their emitted statements.
    let has_observable_side_effect = insts.iter().any(|d| {
        matches!(
            d.enc.name,
            "ENC_EXP"
                | "ENC_MUBUF"
                | "ENC_MTBUF"
                | "ENC_MIMG"
                | "ENC_FLAT"
                | "ENC_FLAT_GLBL"
                | "ENC_FLAT_SCRATCH"
                | "ENC_DS"
                | "ENC_SMEM"
        ) || d.inst.name == "S_SENDMSG"
    });
    let allow_dead_inst = !kernel_mode && has_observable_side_effect;

    let mut dead_scc = HashSet::new();
    let mut dead_carry_sdst = HashSet::new();
    let mut dead_inst = HashSet::new();

    for bi in 0..num_blocks {
        let s_idx = block_ranges[bi].0;
        let e_idx = eff_end[bi];
        let mut ls = live_out_s[bi];
        let mut lv = live_out_v[bi];
        let mut lscc = live_out_scc[bi];

        for k in (s_idx..e_idx).rev() {
            let d = &insts[k];
            let info = &infos[k];

            if info.scc_write && !lscc {
                dead_scc.insert(d.pc);
            }
            if let Some(cdst) = info.carry_sdst {
                let is_dead = match cdst {
                    CarryDst::Null => true,
                    CarryDst::Sgpr(code) => {
                        if code >= 126 {
                            false
                        } else {
                            let mask = if opt.wave64 {
                                sreg_range(code, 2)
                            } else {
                                1u128 << code
                            };
                            (ls & mask) == 0
                        }
                    }
                };
                if is_dead {
                    dead_carry_sdst.insert(d.pc);
                }
            }

            if allow_dead_inst
                && info.pure_eliminable
                && (info.s_mod & ls) == 0
                && info.v_mod.intersect(lv).is_empty()
                && (!info.scc_write || !lscc)
            {
                dead_inst.insert(d.pc);
                continue;
            }

            ls = (ls & !info.s_kill) | info.s_read | S_EXEC;
            lv = lv.union(info.v_read); // ZERO VGPR kill (Guardrail #1)!
            lscc = (lscc && !info.scc_write) || info.scc_read;
        }
    }

    // Build augmented CFG (including `LOOP_GUARD` exits) for exact dominators & hazard reachability.
    let mut cfg_succs = succs;
    let mut cfg_preds = preds;
    for bi in 0..num_blocks {
        for &exit in &loop_guard_exits[bi] {
            if let Some(eb) = exit {
                if !cfg_succs[bi].contains(&eb) {
                    cfg_succs[bi].push(eb);
                    cfg_preds[eb].push(bi);
                }
            }
        }
    }
    let dom = DomTree::build(num_blocks, &cfg_succs, &cfg_preds);
    let inst_hazard: Vec<bool> = insts.iter().map(|d| is_memory_hazard(d, opt)).collect();

    let has_unknown_cf = unknown_succ.iter().any(|&u| u);

    // CFG-wide SGPR reaching definitions & constant values.
    let (def_in, const_in) = solve_cfg_sgpr_defs_and_consts(
        num_blocks,
        &dom,
        &cfg_preds,
        &block_ranges,
        &eff_end,
        insts,
        &infos,
        &dead_inst,
        has_unknown_cf,
    );

    // Universal non-`Kernel` Window-0 fast-path sets (`s_load_*` and `s_buffer_load_*`).
    let (known_zero_hi_sload, known_zero_hi_sbuf) = collect_known_zero_hi_smem(
        insts,
        &infos,
        &block_ranges,
        &eff_end,
        &dom,
        &dead_inst,
        &def_in,
        &const_in,
        opt,
        has_unknown_cf,
    );

    // Weapon 1A: Dominator-hoisted / CSE'd invariant `s_load_*` groups + known-zero-hi Window-0 fast path.
    let hoisted_smem = analyze_hoisted_smem(
        insts,
        &infos,
        &block_starts,
        &block_ranges,
        &eff_end,
        &dom,
        &cfg_preds,
        &unknown_succ,
        &inst_hazard,
        &dead_inst,
        &def_in,
        &const_in,
        opt,
        has_unknown_cf,
    );

    // Weapon 1A-sbuf: Cross-block dominator-hoisted / CSE'd invariant `s_buffer_load_*` groups.
    let hoisted_sbuf = analyze_hoisted_sbuf(
        insts,
        &infos,
        &block_starts,
        &block_ranges,
        &eff_end,
        &dom,
        &cfg_preds,
        &unknown_succ,
        &inst_hazard,
        &dead_inst,
        &def_in,
        &const_in,
        opt,
        has_unknown_cf,
    );

    // Weapon 1B: Non-consecutive intra-block `s_buffer_load_*` grouping (zero `[loop]`, disjoint from `hoisted_sbuf`).
    let (intra_block_sbuf_groups, sbuf_uses) = analyze_intra_block_sbuf(
        insts,
        &infos,
        &block_ranges,
        &eff_end,
        &dom,
        &inst_hazard,
        &dead_inst,
        &def_in,
        &const_in,
        &hoisted_sbuf.uses,
        opt,
    );

    // Weapon 3: Wave32 & native-Wave64 lane-local `bool` mask scalarization.
    let lane_local = analyze_lane_local_masks(
        insts,
        &infos,
        &block_ranges,
        &eff_end,
        &live_out_s,
        &dead_scc,
        &dead_carry_sdst,
        &dead_inst,
        opt,
    );

    // Intra-block constant propagation (seeded with CFG-wide proven SGPR constants).
    let mut const_srcs: HashMap<u64, [Option<u32>; 3]> = HashMap::new();
    let mut const_sgprs: HashMap<(u64, u8), u32> = HashMap::new();

    for bi in 0..num_blocks {
        let s_idx = block_ranges[bi].0;
        let e_idx = eff_end[bi];
        let slice = &insts[s_idx..e_idx];
        let info_slice = &infos[s_idx..e_idx];

        let mut init_s: [Option<u32>; 106] = [None; 106];
        for r in 0..106 {
            if let ConstVal::Known(v) = const_in[bi][r] {
                init_s[r] = Some(v);
            }
        }
        propagate_block_constants(slice, info_slice, init_s, &mut const_srcs, &mut const_sgprs);
    }

    ProgramOpt {
        mem_coherent,
        hoisted_smem_groups: hoisted_smem.groups,
        hoisted_smem_at_entry: hoisted_smem.at_entry,
        hoisted_smem_before: hoisted_smem.before_pc,
        hoisted_smem_after: hoisted_smem.after_pc,
        smem_uses: hoisted_smem.uses,
        hoisted_sbuf_groups: hoisted_sbuf.groups,
        hoisted_sbuf_at_entry: hoisted_sbuf.at_entry,
        hoisted_sbuf_before: hoisted_sbuf.before_pc,
        hoisted_sbuf_after: hoisted_sbuf.after_pc,
        hoisted_sbuf_uses: hoisted_sbuf.uses,
        known_zero_hi_sload,
        known_zero_hi_sbuf,
        intra_block_sbuf_groups,
        sbuf_uses,
        lane_local_defs: lane_local.defs,
        lane_local_bitwise: lane_local.bitwise,
        lane_local_uses: lane_local.uses,
        dead_scc,
        dead_carry_sdst,
        dead_inst,
        const_srcs,
        const_sgprs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::recomp::CsInterface;

    const V: u64 = 256;

    struct InstSpec<'a> {
        name: &'a str,
        enc: &'a str,
        fields: &'a [(&'a str, u64)],
        lit: Option<u32>,
    }

    fn i<'a>(name: &'a str, enc: &'a str, fields: &'a [(&'a str, u64)]) -> InstSpec<'a> {
        InstSpec {
            name,
            enc,
            fields,
            lit: None,
        }
    }

    fn il<'a>(name: &'a str, enc: &'a str, fields: &'a [(&'a str, u64)], lit: u32) -> InstSpec<'a> {
        InstSpec {
            name,
            enc,
            fields,
            lit: Some(lit),
        }
    }

    fn assemble(specs: &[InstSpec]) -> (Vec<Decoded>, Vec<u64>) {
        let mut bytes = Vec::new();
        for s in specs {
            bytes.extend(rdna2::encode(s.name, s.enc, s.fields, s.lit).unwrap());
        }
        let mut insts = Vec::new();
        let mut off = 0usize;
        while off + 4 <= bytes.len() {
            let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
            off += d.len;
            insts.push(d);
        }
        rdna2::resolve_far_jumps(&mut insts);
        let starts: std::collections::BTreeSet<u64> = insts.iter().map(|d| d.pc).collect();
        let mut leaders = std::collections::BTreeSet::from([0u64]);
        for d in &insts {
            let next = d.pc + d.len as u64;
            if let Some(t) = rdna2::branch_target(d) {
                leaders.insert(t);
                leaders.insert(next);
            } else if d.inst.flags & F_TERMINATOR != 0 {
                leaders.insert(next);
            }
        }
        let leaders_vec: Vec<u64> = leaders.into_iter().filter(|l| starts.contains(l)).collect();
        (insts, leaders_vec)
    }

    fn cs_opts(wave64: bool) -> Options {
        Options {
            mode: Mode::Compute(CsInterface {
                threads: [64, 1, 1],
                user_sgprs: 4,
                tgid: [true, false, false],
                tg_size: false,
                tid_components: 1,
            }),
            wave64,
            ..Default::default()
        }
    }

    fn endpgm<'a>() -> InstSpec<'a> {
        i("S_ENDPGM", "ENC_SOPP", &[("SIMM16", 0)])
    }

    #[test]
    fn dead_scc_overwritten_vs_live_scc_branch_and_addc() {
        // I0: s_add_u32 s2, s0, s1        -> scc overwritten by I1 -> dead_scc = true
        // I1: s_add_u32 s3, s2, 1         -> scc read by I2 (s_addc_u32) -> dead_scc = false
        // I2: s_addc_u32 s4, s3, 0        -> scc overwritten by I3 -> dead_scc = true
        // I3: s_cmp_eq_u32 s4, 0          -> scc read by I4 (s_cbranch_scc0) -> dead_scc = false
        // I4: s_cbranch_scc0 +1
        // I5: buffer_store_dword ...
        // I6: s_endpgm
        let (insts, leaders) = assemble(&[
            i("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 0), ("SSRC1", 1)]),
            i("S_ADD_U32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 2), ("SSRC1", 129)]),
            i("S_ADDC_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 3), ("SSRC1", 128)]),
            i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 128)]),
            i("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 2)]),
            i(
                "BUFFER_STORE_DWORD",
                "ENC_MUBUF",
                &[("VDATA", 0), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128)],
            ),
            endpgm(),
        ]);
        let opt = analyze(&insts, &leaders, &cs_opts(false));
        assert!(opt.is_dead_scc(insts[0].pc));
        assert!(!opt.is_dead_scc(insts[1].pc), "s_addc_u32 reads scc");
        assert!(opt.is_dead_scc(insts[2].pc));
        assert!(!opt.is_dead_scc(insts[3].pc), "s_cbranch_scc0 reads scc");
    }

    #[test]
    fn dead_carry_sdst_wave32_wave64_null_and_carry_in() {
        // Wave64: s[10:11] written by v_add_co_u32; s11 (hi half) read later -> NOT dead in wave64, DEAD in wave32!
        let (insts, leaders) = assemble(&[
            i(
                "V_ADD_CO_U32",
                "VOP3_SDST_ENC",
                &[("VDST", 1), ("SDST", 10), ("SRC0", V), ("SRC1", V + 2)],
            ),
            i("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 11)]),
            i(
                "BUFFER_STORE_DWORD",
                "ENC_MUBUF",
                &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)],
            ),
            endpgm(),
        ]);
        let opt64 = analyze(&insts, &leaders, &cs_opts(true));
        assert!(!opt64.is_dead_carry_sdst(insts[0].pc), "wave64: hi half s11 is live");
        let opt32 = analyze(&insts, &leaders, &cs_opts(false));
        assert!(opt32.is_dead_carry_sdst(insts[0].pc), "wave32: only s10 written and s10 is dead");

        // Dst::Null (125) is ALWAYS dead even in Mode::Kernel!
        let (insts_null, leaders_null) = assemble(&[
            i(
                "V_ADD_CO_U32",
                "VOP3_SDST_ENC",
                &[("VDST", 1), ("SDST", 125), ("SRC0", V), ("SRC1", V + 2)],
            ),
            endpgm(),
        ]);
        let opt_k = analyze(&insts_null, &leaders_null, &Options { mode: Mode::Kernel, wave64: true, ..Default::default() });
        assert!(opt_k.is_dead_carry_sdst(insts_null[0].pc), "Dst::Null carry is always dead");

        // Carry-in chain: v_add_co_u32 -> v_add_co_ci_u32 (carry-out dead, carry-in MUST stay live!).
        let (insts_ci, leaders_ci) = assemble(&[
            i(
                "V_ADD_CO_U32",
                "VOP3_SDST_ENC",
                &[("VDST", 1), ("SDST", 106), ("SRC0", V), ("SRC1", V + 2)],
            ),
            i("V_ADD_CO_CI_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 1), ("VSRC1", 3)]),
            i(
                "BUFFER_STORE_DWORD",
                "ENC_MUBUF",
                &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)],
            ),
            endpgm(),
        ]);
        let opt_ci = analyze(&insts_ci, &leaders_ci, &cs_opts(true));
        assert!(!opt_ci.is_dead_carry_sdst(insts_ci[0].pc), "carry-in of v_add_co_ci reads vcc");
        assert!(opt_ci.is_dead_carry_sdst(insts_ci[1].pc), "final carry-out of v_add_co_ci is dead");
    }

    #[test]
    fn vgpr_rmw_dual_mask_if_else_soundness() {
        // Dual-mask pattern under divergent EXEC:
        //   v_mov_b32 v5, 1          // written under EXEC_A
        //   s_andn2_b32 exec_lo, s0, exec_lo  // switch to EXEC_B (EXEC-widening!)
        //   v_mov_b32 v5, 2          // written under EXEC_B — MUST NOT kill earlier write to v5!
        //   s_mov_b32 exec_lo, s0    // restore EXEC_A | EXEC_B
        //   buffer_store_dword v5    // reads v5 across BOTH lane masks
        let (insts, leaders) = assemble(&[
            i("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 129)]),
            i("S_ANDN2_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 0), ("SSRC1", 126)]),
            i("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 130)]),
            i("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)]),
            i(
                "BUFFER_STORE_DWORD",
                "ENC_MUBUF",
                &[("VDATA", 5), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)],
            ),
            endpgm(),
        ]);
        let opt = analyze(&insts, &leaders, &cs_opts(false));
        assert!(!opt.is_dead_inst(insts[0].pc), "VGPR write under EXEC_A must stay live");
        assert!(!opt.is_dead_inst(insts[2].pc), "VGPR write under EXEC_B must stay live");
    }

    #[test]
    fn kernel_mode_preserves_live_outs() {
        let (insts, leaders) = assemble(&[
            i("S_ADD_U32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 0), ("SSRC1", 1)]),
            i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", V), ("VSRC1", 1)]),
            endpgm(),
        ]);
        let opt = analyze(&insts, &leaders, &Options { mode: Mode::Kernel, wave64: true, ..Default::default() });
        assert!(!opt.is_dead_scc(insts[0].pc), "Kernel dumps scc at endpgm");
        assert!(!opt.is_dead_inst(insts[0].pc), "Kernel dumps s5 at endpgm");
        assert!(!opt.is_dead_inst(insts[1].pc), "Kernel dumps v7 at endpgm");
    }

    #[test]
    fn mem_coherent_four_condition_gate() {
        // Plain load + store (no GLC/DLC/SLC, no atomic, no inv, no barrier) -> mem_coherent = false.
        let (insts, leaders) = assemble(&[
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 4), ("SBASE", 0), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 0), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128)]),
            endpgm(),
        ]);
        assert!(!analyze(&insts, &leaders, &cs_opts(false)).mem_coherent);

        // Condition 1: GLC flag -> mem_coherent = true.
        let (insts, leaders) = assemble(&[
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 4), ("SBASE", 0), ("OFFSET", 0), ("SOFFSET", 125), ("GLC", 1)]),
            endpgm(),
        ]);
        assert!(analyze(&insts, &leaders, &cs_opts(false)).mem_coherent);

        // Condition 2: memory atomic -> mem_coherent = true.
        let (insts, leaders) = assemble(&[
            i("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("VDATA", 0), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            endpgm(),
        ]);
        assert!(analyze(&insts, &leaders, &cs_opts(false)).mem_coherent);

        // Condition 3: cache invalidate -> mem_coherent = true.
        let (insts, leaders) = assemble(&[
            i("BUFFER_GL0_INV", "ENC_MUBUF", &[]),
            endpgm(),
        ]);
        assert!(analyze(&insts, &leaders, &cs_opts(false)).mem_coherent);

        // Condition 4a: s_barrier WITHOUT mem store -> mem_coherent = false.
        let (insts, leaders) = assemble(&[
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 4), ("SBASE", 0), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("S_BARRIER", "ENC_SOPP", &[("SIMM16", 0)]),
            endpgm(),
        ]);
        assert!(!analyze(&insts, &leaders, &cs_opts(false)).mem_coherent);

        // Condition 4b: s_barrier WITH mem store -> mem_coherent = true.
        let (insts, leaders) = assemble(&[
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 0), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            i("S_BARRIER", "ENC_SOPP", &[("SIMM16", 0)]),
            endpgm(),
        ]);
        assert!(analyze(&insts, &leaders, &cs_opts(false)).mem_coherent);
    }

    #[test]
    fn hoisted_smem_lca_sibling_branches_and_known_zero_hi() {
        // Block 0 (pc=0): s_mov_b32 s5, 0; s_cmp_eq_u32 s0, 0; s_cbranch_scc0 else(pc=32)
        // Block 1 (pc=12): s_load_dwordx4 s[8:11], s[4:5], 0; s_load_dwordx4 s[12:15], s[4:5], 16; s_branch join(pc=48)
        // Block 2 (pc=32): s_load_dwordx4 s[8:11], s[4:5], 0; s_load_dword s12, s[4:5], 24
        // Block 3 (pc=48): s_endpgm
        let (insts, leaders) = assemble(&[
            i("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 128)]),
            i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)]),
            i("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 5)]),
            i("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 2), ("OFFSET", 16), ("SOFFSET", 125)]),
            i("S_BRANCH", "ENC_SOPP", &[("SIMM16", 4)]),
            i("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 2), ("OFFSET", 24), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        let mut opts = cs_opts(false);
        opts.mem_windows = 4;
        let opt = analyze(&insts, &leaders, &opts);

        assert_eq!(opt.hoisted_smem_groups().len(), 1);
        let g = &opt.hoisted_smem_groups()[0];
        assert_eq!(g.hoist_block, 0, "must hoist to LCA block 0");
        assert_eq!(g.sbase, 4);
        assert_eq!(g.known_hi, Some(0), "s_mov_b32 s5, 0 proves known_hi = Some(0)");
        assert_eq!(
            g.loads,
            vec![
                HoistedLoad { offset: 0, count: 4 },
                HoistedLoad { offset: 16, count: 4 },
            ],
            "exact duplicate (0,4) deduplicated and (24,1) subsumed into (16,4)"
        );
        assert_eq!(
            opt.hoisted_smem_before_pc(insts[2].pc),
            &[0usize],
            "hoisted immediately before block 0 branch terminator"
        );
        let u3 = opt.smem_use(insts[3].pc).unwrap();
        let u6 = opt.smem_use(insts[6].pc).unwrap();
        assert_eq!(u3, u6);
        assert_eq!(u3.load_idx, 0);
        assert_eq!(u3.dword_offset, 0);
        let u7 = opt.smem_use(insts[7].pc).unwrap();
        assert_eq!(u7.load_idx, 1);
        assert_eq!(u7.parent_count, 4);
        assert_eq!(u7.dword_offset, 2);
        assert_eq!(u7.count, 1);

        // Single-use `s_load` with `known_hi == Some(0)` under `mem_windows = 4` also gets zero-switch group!
        let (insts_single, leaders_single) = assemble(&[
            i("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 128)]),
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 12), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        let opt_s = analyze(&insts_single, &leaders_single, &opts);
        assert_eq!(opt_s.hoisted_smem_groups().len(), 1);
        assert_eq!(opt_s.hoisted_smem_groups()[0].known_hi, Some(0));
        assert!(opt_s.smem_use(insts_single[1].pc).is_some());
    }

    #[test]
    fn hoisted_smem_store_and_loop_carried_hazard_rejection() {
        let mut opts = cs_opts(false);
        opts.mem_windows = 4;

        // A. Store between two `s_load`s off unknown-hi `s[4:5]` -> MUST NOT merge across store!
        let (insts_st, leaders_st) = assemble(&[
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 0), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 9), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        let opt_st = analyze(&insts_st, &leaders_st, &opts);
        assert!(opt_st.hoisted_smem_groups().is_empty(), "must not CSE across store");

        // B. Loop-carried store: `s_load` in block 0, loop block 1 has `s_load` THEN `buffer_store_dword` THEN back-edge!
        let (insts_lp, leaders_lp) = assemble(&[
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            // block 1 (pc=8):
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 9), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 0), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFB)]), // back -5 dwords to pc=8
            endpgm(),
        ]);
        let opt_lp = analyze(&insts_lp, &leaders_lp, &opts);
        assert!(
            opt_lp.hoisted_smem_groups().is_empty(),
            "loop-carried store after s_load must prevent hoisting out of loop"
        );

        // C. `s_barrier` between two `s_load`s -> MUST NOT merge across barrier!
        let (insts_bar, leaders_bar) = assemble(&[
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("S_BARRIER", "ENC_SOPP", &[("SIMM16", 0)]),
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 9), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        assert!(
            analyze(&insts_bar, &leaders_bar, &opts).hoisted_smem_groups().is_empty(),
            "must not hoist/CSE across s_barrier"
        );

        // D. `GLC == 1` `s_load` -> NEVER hoisted even with `s_mov_b32 s5, 0`!
        let (insts_glc, leaders_glc) = assemble(&[
            i("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 128)]),
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125), ("GLC", 1)]),
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 9), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125), ("GLC", 1)]),
            endpgm(),
        ]);
        assert!(
            analyze(&insts_glc, &leaders_glc, &opts).hoisted_smem_groups().is_empty(),
            "GLC loads must never be hoisted or CSEd"
        );
    }

    #[test]
    fn hoisted_smem_kernel_mode_4gib_wrap_guard() {
        let k_opts = Options {
            mode: Mode::Kernel,
            wave64: false,
            mem_windows: 4,
            ..Default::default()
        };
        let mut c_opts = cs_opts(false);
        c_opts.mem_windows = 4;

        // Kernel mode: off=0, count=1 -> cannot wrap 4 GiB -> known_hi = Some(0).
        let (i0, l0) = assemble(&[
            i("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 128)]),
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        let opt0 = analyze(&i0, &l0, &k_opts);
        assert_eq!(opt0.hoisted_smem_groups().len(), 1);
        assert_eq!(opt0.hoisted_smem_groups()[0].known_hi, Some(0));

        // Kernel mode: off=4, count=1 -> COULD wrap 4 GiB on synthetic pointer -> known_hi MUST be None!
        let (i1, l1) = assemble(&[
            i("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 128)]),
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 4), ("SOFFSET", 125)]),
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 9), ("SBASE", 2), ("OFFSET", 8), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        let opt1_k = analyze(&i1, &l1, &k_opts);
        assert_eq!(opt1_k.hoisted_smem_groups().len(), 1);
        assert_eq!(opt1_k.hoisted_smem_groups()[0].known_hi, None, "Kernel mode off!=0 must keep 64-bit switch");
        let opt1_c = analyze(&i1, &l1, &c_opts);
        assert_eq!(opt1_c.hoisted_smem_groups()[0].known_hi, Some(0), "Compute mode allows UMA known_hi=Some(0)");

        // Kernel mode: off=0, count=2 (max_end=8 > 4) -> known_hi MUST be None!
        let (i2, l2) = assemble(&[
            i("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 128)]),
            i("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        assert!(
            analyze(&i2, &l2, &k_opts).hoisted_smem_groups().is_empty(),
            "Kernel mode dwordx2 at off=0 must not get known_hi=Some(0)"
        );

        // Non-zero constant `s_mov_b32 s5, 1` in Compute -> known_hi MUST be None (depends on runtime `xcl_uma_base`)!
        let (i3, l3) = assemble(&[
            i("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 129)]),
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 9), ("SBASE", 2), ("OFFSET", 4), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        let opt3 = analyze(&i3, &l3, &c_opts);
        assert_eq!(opt3.hoisted_smem_groups().len(), 1);
        assert_eq!(opt3.hoisted_smem_groups()[0].known_hi, None, "hi==1 requires runtime xcl_uma_base subtraction");
    }

    #[test]
    fn intra_block_sbuf_non_consecutive_and_epoch_breaks() {
        // Non-consecutive `s_buffer_load_*` off s[8:11] (`SBASE = 4`) across intervening ALU ops,
        // with duplicate + subsumption + final op overwriting s8 (safe!) followed by post-overwrite load!
        let (insts, leaders) = assemble(&[
            i("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 4), ("OFFSET", 16), ("SOFFSET", 125)]),
            i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V), ("VSRC1", 1)]),
            i("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 4), ("OFFSET", 16), ("SOFFSET", 125)]),
            i("S_ADD_U32", "ENC_SOP2", &[("SDST", 30), ("SSRC0", 16), ("SSRC1", 20)]),
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 24), ("SBASE", 4), ("OFFSET", 24), ("SOFFSET", 125)]),
            // Last op in epoch overwrites s8 (part of sbase 8..12) -> safely joins epoch and breaks AFTER!
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 4), ("OFFSET", 32), ("SOFFSET", 125)]),
            // Subsequent load sees NEW s8 -> must NOT join earlier epoch!
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 25), ("SBASE", 4), ("OFFSET", 36), ("SOFFSET", 125)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            endpgm(),
        ]);
        let opt = analyze(&insts, &leaders, &cs_opts(false));
        let grp = opt
            .intra_block_sbuf_group(insts[0].pc)
            .expect("expected intra-block sbuf group at first load");
        assert_eq!(grp.sbase, 8);
        assert_eq!(grp.soff, None);
        assert_eq!(
            grp.loads,
            vec![
                HoistedLoad { offset: 16, count: 4 },
                HoistedLoad { offset: 32, count: 1 },
            ]
        );
        assert_eq!(opt.sbuf_use(insts[0].pc).unwrap().load_idx, 0);
        assert_eq!(opt.sbuf_use(insts[2].pc).unwrap().load_idx, 0);
        let u4 = opt.sbuf_use(insts[4].pc).unwrap();
        assert_eq!(u4.load_idx, 0);
        assert_eq!(u4.dword_offset, 2);
        assert_eq!(u4.count, 1);
        assert_eq!(opt.sbuf_use(insts[5].pc).unwrap().load_idx, 1);
        assert!(opt.sbuf_use(insts[6].pc).is_none(), "post-sbase-overwrite load must not join earlier epoch");

        // Intervening ALU write to `s9` (inside `sbase = 8..12`) must break epoch!
        let (insts_w, leaders_w) = assemble(&[
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("S_MOV_B32", "ENC_SOP1", &[("SDST", 9), ("SSRC0", 128)]),
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 17), ("SBASE", 4), ("OFFSET", 4), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        let opt_w = analyze(&insts_w, &leaders_w, &cs_opts(false));
        assert!(opt_w.intra_block_sbuf_group(insts_w[0].pc).is_none(), "write to sbase+1 must break epoch");
    }

    #[test]
    fn lane_local_bool_masks_wave32_wave64_exec_widen_and_guards() {
        // 1. Wave32: `v_cmp_lt_f32 vcc, v0, v1 -> v_cndmask_b32 v2, v3, v4` (no EXEC widening).
        let (insts_simple, leaders_simple) = assemble(&[
            i("V_CMP_LT_F32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]),
            i("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 3), ("VSRC1", 4)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            endpgm(),
        ]);
        let opt32 = analyze(&insts_simple, &leaders_simple, &cs_opts(false));
        let def = opt32.lane_local_def(insts_simple[0].pc).expect("v_cmp should be lane-local");
        assert!(!def.needs_exec_mask);
        assert_eq!(def.kind, LaneLocalDefKind::Cmp);
        assert_eq!(opt32.lane_local_use(insts_simple[1].pc), Some(def.id));

        // Same shader in `Mode::Kernel` and `split_wave64 = true` -> DISABLED!
        let opt_k = analyze(
            &insts_simple,
            &leaders_simple,
            &Options { mode: Mode::Kernel, wave64: false, ..Default::default() },
        );
        assert!(opt_k.lane_local_def(insts_simple[0].pc).is_none());
        let mut split_opts = cs_opts(true);
        split_opts.split_wave64 = true;
        let opt_split = analyze(&insts_simple, &leaders_simple, &split_opts);
        assert!(opt_split.lane_local_def(insts_simple[0].pc).is_none());

        // 2. Wave32: `v_cmp -> s_and_b32 vcc, vcc, exec_lo -> v_cndmask` (dead scc!).
        let (insts_bw, leaders_bw) = assemble(&[
            i("V_CMP_LT_F32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]),
            i("S_AND_B32", "ENC_SOP2", &[("SDST", 106), ("SSRC0", 106), ("SSRC1", 126)]),
            i("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 3), ("VSRC1", 4)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            endpgm(),
        ]);
        let opt_bw = analyze(&insts_bw, &leaders_bw, &cs_opts(false));
        let d0 = opt_bw.lane_local_def(insts_bw[0].pc).expect("v_cmp lane-local");
        let bw = opt_bw.lane_local_bitwise(insts_bw[1].pc).expect("s_and_b32 lane-local");
        assert_eq!(bw.op, LaneLocalBitOp::And);
        assert_eq!(bw.src0, LaneLocalOperand::Local(d0.id));
        assert_eq!(bw.src1, Some(LaneLocalOperand::Exec));
        assert_eq!(opt_bw.lane_local_use(insts_bw[2].pc), Some(bw.dst_id));

        // 3. Wave32: `v_add_co_u32 -> v_add_co_ci_u32` (carry-out -> carry-in).
        let (insts_co, leaders_co) = assemble(&[
            i("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 1), ("SDST", 106), ("SRC0", V), ("SRC1", V + 2)]),
            i("V_ADD_CO_CI_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 1), ("VSRC1", 3)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            endpgm(),
        ]);
        let opt_co = analyze(&insts_co, &leaders_co, &cs_opts(false));
        let d_co = opt_co.lane_local_def(insts_co[0].pc).expect("v_add_co carry-out lane-local");
        assert_eq!(d_co.kind, LaneLocalDefKind::CarryOut);
        assert!(!d_co.needs_exec_mask);
        assert_eq!(opt_co.lane_local_use(insts_co[1].pc), Some(d_co.id));

        // 4. EXEC-widening between `v_cmp` and `v_cndmask` -> `needs_exec_mask == true`!
        let (insts_ew, leaders_ew) = assemble(&[
            i("V_CMP_LT_F32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]),
            i("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
            i("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 3), ("VSRC1", 4)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            endpgm(),
        ]);
        let opt_ew = analyze(&insts_ew, &leaders_ew, &cs_opts(false));
        let d_ew = opt_ew.lane_local_def(insts_ew[0].pc).expect("still lane-local with EXEC widen");
        assert!(d_ew.needs_exec_mask, "EXEC widening must set needs_exec_mask=true");
        assert_eq!(opt_ew.lane_local_use(insts_ew[2].pc), Some(d_ew.id));

        // 5. Whole-mask readers (`s_cbranch_vccz` and `s_bcnt1_i32_b32`) MUST reject scalarization!
        let (insts_br, leaders_br) = assemble(&[
            i("V_CMP_LT_F32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]),
            i("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 3), ("VSRC1", 4)]),
            i("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 2)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            endpgm(),
        ]);
        let opt_br = analyze(&insts_br, &leaders_br, &cs_opts(false));
        assert!(opt_br.lane_local_def(insts_br[0].pc).is_none(), "s_cbranch_vccz requires real vcc");
        assert!(opt_br.lane_local_use(insts_br[1].pc).is_none());

        let (insts_cnt, leaders_cnt) = assemble(&[
            i("V_CMP_LT_F32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]),
            i("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 106)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 8)]),
            endpgm(),
        ]);
        let opt_cnt = analyze(&insts_cnt, &leaders_cnt, &cs_opts(false));
        assert!(opt_cnt.lane_local_def(insts_cnt[0].pc).is_none(), "s_bcnt1 reads whole mask");

        // 6. Native Wave64 (`wave64: true, split_wave64: false`): `v_cmp -> s_and_b64 -> v_cndmask`!
        let (insts_w64, leaders_w64) = assemble(&[
            i("V_CMP_LT_F32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]),
            i("S_AND_B64", "ENC_SOP2", &[("SDST", 106), ("SSRC0", 106), ("SSRC1", 126)]),
            i("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 3), ("VSRC1", 4)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            endpgm(),
        ]);
        let opt64 = analyze(&insts_w64, &leaders_w64, &cs_opts(true));
        let d64 = opt64.lane_local_def(insts_w64[0].pc).expect("native wave64 v_cmp lane-local");
        let bw64 = opt64.lane_local_bitwise(insts_w64[1].pc).expect("native wave64 s_and_b64 lane-local");
        assert_eq!(bw64.src0, LaneLocalOperand::Local(d64.id));
        assert_eq!(bw64.src1, Some(LaneLocalOperand::Exec));
        assert_eq!(opt64.lane_local_use(insts_w64[2].pc), Some(bw64.dst_id));
    }

    #[test]
    fn intra_block_const_prop_for_mul_legacy_and_mubuf_stride() {
        let (insts, leaders) = assemble(&[
            il("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 255)], 16u32 << 16), // stride = 16 in srsrc+1
            i("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 244)]),                // 2.0f
            i("V_MUL_LEGACY_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 1), ("VSRC1", 2)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 3), ("VADDR", 0), ("SRSRC", 1), ("SOFFSET", 128)]),
            endpgm(),
        ]);
        let opt = analyze(&insts, &leaders, &cs_opts(false));
        assert_eq!(opt.const_src(insts[2].pc, 0), None);
        assert_eq!(opt.const_src(insts[2].pc, 1), Some(2.0f32.to_bits()));
        assert_eq!(opt.const_sgpr(insts[3].pc, 5), Some(16u32 << 16));
    }

    #[test]
    fn known_zero_hi_sload_and_sbuf_universal_and_offset_range() {
        let neg16 = ((-16i32) as u32 & 0x1F_FFFF) as u64;
        let (insts, leaders) = assemble(&[
            i("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 128)]), // s5 = 0 (s_load hi)
            il("S_MOV_B32", "ENC_SOP1", &[("SDST", 9), ("SSRC0", 255)], 0x0010_0000), // s9: stride=16, base_hi=0
            il("S_MOV_B32", "ENC_SOP1", &[("SDST", 13), ("SSRC0", 255)], 0x0010_0001), // s13: base_hi=1 (NOT zero!)
            // I3: s_load with soff != 125 and imm >= 0 -> known_zero_hi_sload = true
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 0)]),
            // I4: s_load with soff != 125 and imm < 0 -> known_zero_hi_sload = false
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 21), ("SBASE", 2), ("OFFSET", neg16), ("SOFFSET", 0)]),
            // I5: store hazard!
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 0), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            // I6: post-store s_load with soff == 125 and negative imm (-16) -> known_zero_hi_sload = true
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 22), ("SBASE", 2), ("OFFSET", neg16), ("SOFFSET", 125)]),
            // I7: post-store s_load with soff == 125 and offset > 65536 (131072 <= 1_048_576) -> known_zero_hi_sload = true + HoistedSmemGroup known_hi = Some(0)
            i("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 23), ("SBASE", 2), ("OFFSET", 131_072), ("SOFFSET", 125)]),
            // I8: post-store s_buffer_load with soff != 125 and (d1 & 0xFFFF) == 0 -> known_zero_hi_sbuf = true
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 24), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 0)]),
            // I9: s_buffer_load with (d1 & 0xFFFF) == 1 -> known_zero_hi_sbuf = false
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 25), ("SBASE", 6), ("OFFSET", 0), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        let mut opts = cs_opts(false);
        opts.mem_windows = 4;
        let opt = analyze(&insts, &leaders, &opts);

        assert!(opt.is_known_zero_hi_sload(insts[3].pc), "soff != 125 with imm >= 0");
        assert!(!opt.is_known_zero_hi_sload(insts[4].pc), "soff != 125 with imm < 0 must be false");
        assert!(opt.is_known_zero_hi_sload(insts[6].pc), "post-store soff == 125 with neg imm");
        assert!(opt.is_known_zero_hi_sload(insts[7].pc), "post-store soff == 125 with offset 131072");
        assert!(opt.is_known_zero_hi_sbuf(insts[8].pc), "post-store s_buffer_load with (d1 & 0xFFFF) == 0 and soff != 125");
        assert!(!opt.is_known_zero_hi_sbuf(insts[9].pc), "s_buffer_load with (d1 & 0xFFFF) == 1 must be false");

        // Raised non-Kernel HoistedSmemGroup known_hi range (> 65536, <= 1_048_576):
        let u7 = opt.smem_use(insts[7].pc).expect("offset 131072 hoisted under raised 1_048_576 range");
        assert_eq!(opt.hoisted_smem_groups()[u7.group_id as usize].known_hi, Some(0));

        // Mode::Kernel must NEVER populate universal known_zero_hi_sload / known_zero_hi_sbuf.
        let k_opts = Options {
            mode: Mode::Kernel,
            wave64: false,
            mem_windows: 4,
            ..Default::default()
        };
        let opt_k = analyze(&insts, &leaders, &k_opts);
        for d in &insts {
            assert!(!opt_k.is_known_zero_hi_sload(d.pc), "Kernel must disable known_zero_hi_sload");
            assert!(!opt_k.is_known_zero_hi_sbuf(d.pc), "Kernel must disable known_zero_hi_sbuf");
        }
    }

    #[test]
    fn hoisted_sbuf_lca_sibling_branches_dedup_known_zero_hi_and_hazards() {
        let mut opts = cs_opts(false);
        opts.mem_windows = 4;

        // 1. Sibling branches: LCA = block 0, `d1` has `(d1 & 0xFFFF) == 0`, `d3` (`s7`) modified in block 1 (ignored!),
        //    exact dedup `(0, 4)` + subsumption `(24, 1)` into `(16, 4)`, disjoint from `IntraBlockSbufGroup`.
        // Block 0 (pc=0):  I0 s_mov_b32 s5, 0x00040000 (8B); I1 s_cmp_eq_u32 (4B); I2 s_cbranch_scc0 +6 -> pc=40 (4B)
        // Block 1 (pc=16): I3 s_buffer_load_dwordx4 off=0 (8B); I4 s_mov_b32 s7, 1 (4B); I5 s_buffer_load_dwordx4 off=16 (8B); I6 s_branch +4 -> pc=56 (4B)
        // Block 2 (pc=40): I7 s_buffer_load_dwordx4 off=0 (8B); I8 s_buffer_load_dword off=24 (8B)
        // Block 3 (pc=56): I9 s_endpgm (4B)
        let (insts, leaders) = assemble(&[
            il("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 255)], 0x0004_0000),
            i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)]),
            i("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 6)]),
            i("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("S_MOV_B32", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 129)]),
            i("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 2), ("OFFSET", 16), ("SOFFSET", 125)]),
            i("S_BRANCH", "ENC_SOPP", &[("SIMM16", 4)]),
            i("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 2), ("OFFSET", 24), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        let opt = analyze(&insts, &leaders, &opts);

        assert_eq!(opt.hoisted_sbuf_groups().len(), 1);
        let g = &opt.hoisted_sbuf_groups()[0];
        assert_eq!(g.hoist_block, 0, "must hoist to LCA block 0");
        assert_eq!(g.sbase, 4);
        assert_eq!(g.known_hi, Some(0), "(0x00040000 & 0xFFFF) == 0 proves known_hi = Some(0)");
        assert_eq!(
            g.loads,
            vec![
                HoistedLoad { offset: 0, count: 4 },
                HoistedLoad { offset: 16, count: 4 },
            ],
            "exact duplicate (0,4) deduplicated and (24,1) subsumed into (16,4)"
        );
        assert_eq!(opt.hoisted_sbuf_before_pc(insts[2].pc), &[0usize]);

        let u3 = opt.hoisted_sbuf_use(insts[3].pc).expect("I3 hoisted");
        let u5 = opt.hoisted_sbuf_use(insts[5].pc).expect("I5 hoisted");
        let u7 = opt.hoisted_sbuf_use(insts[7].pc).expect("I7 hoisted");
        let u8 = opt.hoisted_sbuf_use(insts[8].pc).expect("I8 hoisted");
        assert_eq!(u3, u7);
        assert_eq!(u3.load_idx, 0);
        assert_eq!(u5.load_idx, 1);
        assert_eq!(u8.load_idx, 1);
        assert_eq!(u8.parent_count, 4);
        assert_eq!(u8.dword_offset, 2);
        assert_eq!(u8.count, 1);

        // Disjointness from IntraBlockSbufGroup: no hoisted PC may appear in `sbuf_use` or `intra_block_sbuf_group`.
        for idx in [3, 5, 7, 8] {
            assert!(opt.sbuf_use(insts[idx].pc).is_none(), "hoisted PC must be excluded from intra-block sbuf");
            assert!(opt.intra_block_sbuf_group(insts[idx].pc).is_none());
        }

        // 2. Hazard rejection: store before block 1's load prevents hoisting to LCA block 0.
        let (insts_st, leaders_st) = assemble(&[
            i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)]),
            i("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 5)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 0), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("S_BRANCH", "ENC_SOPP", &[("SIMM16", 2)]),
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        assert!(
            analyze(&insts_st, &leaders_st, &opts).hoisted_sbuf_groups().is_empty(),
            "store on path from LCA to use must block cross-block s_buffer_load hoisting"
        );

        // 3. Loop-carried store rejection: load in block 0 + load then store in loop block 1.
        let (insts_lp, leaders_lp) = assemble(&[
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 9), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 0), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)]),
            i("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFB)]),
            endpgm(),
        ]);
        assert!(
            analyze(&insts_lp, &leaders_lp, &opts).hoisted_sbuf_groups().is_empty(),
            "loop-carried store must block cross-block s_buffer_load hoisting"
        );

        // 4. GLC == 1 rejection across branches.
        let (insts_glc, leaders_glc) = assemble(&[
            i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)]),
            i("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 3)]),
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125), ("GLC", 1)]),
            i("S_BRANCH", "ENC_SOPP", &[("SIMM16", 2)]),
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125), ("GLC", 1)]),
            endpgm(),
        ]);
        assert!(
            analyze(&insts_glc, &leaders_glc, &opts).hoisted_sbuf_groups().is_empty(),
            "GLC s_buffer_load must never be hoisted"
        );
    }

    #[test]
    fn intra_block_sbuf_skips_soff_sets_known_hi_and_kernel_max_span_zero() {
        let mut opts = cs_opts(false);
        opts.mem_windows = 4;

        // 1. Dynamic `soff` (`SOFFSET != 125`) -> MUST NOT form an IntraBlockSbufGroup (no switch doubling!).
        let (insts_soff, leaders_soff) = assemble(&[
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 0)]),
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 17), ("SBASE", 4), ("OFFSET", 4), ("SOFFSET", 0)]),
            endpgm(),
        ]);
        let opt_soff = analyze(&insts_soff, &leaders_soff, &opts);
        assert!(opt_soff.intra_block_sbuf_group(insts_soff[0].pc).is_none(), "must skip soff.is_some()");
        assert!(opt_soff.sbuf_use(insts_soff[0].pc).is_none());
        assert!(opt_soff.sbuf_use(insts_soff[1].pc).is_none());

        // 2. Constant-offset intra-block group with `(d1 & 0xFFFF) == 0` -> `known_hi == Some(0)` in Compute,
        //    and rejected in Mode::Kernel because `max_span == 4 > 0`!
        let (insts_hi, leaders_hi) = assemble(&[
            il("S_MOV_B32", "ENC_SOP1", &[("SDST", 9), ("SSRC0", 255)], 0x0008_0000),
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 17), ("SBASE", 4), ("OFFSET", 4), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        let opt_c = analyze(&insts_hi, &leaders_hi, &opts);
        let grp = opt_c
            .intra_block_sbuf_group(insts_hi[1].pc)
            .expect("constant-offset intra-block sbuf group expected in Compute");
        assert_eq!(grp.soff, None);
        assert_eq!(grp.known_hi, Some(0));

        let k_opts = Options {
            mode: Mode::Kernel,
            wave64: false,
            mem_windows: 4,
            ..Default::default()
        };
        let opt_k = analyze(&insts_hi, &leaders_hi, &k_opts);
        assert!(
            opt_k.intra_block_sbuf_group(insts_hi[1].pc).is_none(),
            "Mode::Kernel must reject max_span > 0 intra-block sbuf groups"
        );

        // 3. Mode::Kernel with duplicate offset (`max_span == 0`) -> allowed with `known_hi == None`.
        let (insts_k0, leaders_k0) = assemble(&[
            il("S_MOV_B32", "ENC_SOP1", &[("SDST", 9), ("SSRC0", 255)], 0x0008_0000),
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 125)]),
            i("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 17), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 125)]),
            endpgm(),
        ]);
        let opt_k0 = analyze(&insts_k0, &leaders_k0, &k_opts);
        let grp_k0 = opt_k0
            .intra_block_sbuf_group(insts_k0[1].pc)
            .expect("Mode::Kernel allows max_span == 0 duplicate s_buffer_load_dword");
        assert_eq!(grp_k0.known_hi, None, "Kernel mode must never set known_hi = Some(0) on sbuf");
        assert_eq!(grp_k0.loads, vec![HoistedLoad { offset: 0, count: 1 }]);
    }
}
