//! The compact form: each distinct block of a program lowered once, in a loop over a table.

use super::cfg::{Cfg, Term};
use super::lower::{ext_encoding, LResult, Lw, VAR_BUDGET};
use super::lower_mem::VAR_TMP;
use super::module::{Block, Pred, Value};
use std::collections::{BTreeSet, HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Leg {
    Abs(u32),
    Rel(u32),
}

pub struct Plan {
    /// the reachable blocks in address order: the loop's block numbers
    pub order: Vec<usize>,
    /// the class of each
    pub class: Vec<u32>,
    /// a block of each class (a block number)
    pub rep: Vec<usize>,
    /// per class: sorted distinct `(taken, not taken)` legs
    pub(super) targets: Vec<Vec<(Leg, Leg)>>,
    /// per block: its index in its class's `targets`
    pub(super) choice: Vec<u32>,
}

/// Most 32-bit words of tables a program may need to take the form by itself.
pub const TABLE_WORDS: usize = 8192;
/// `TABLE_WORDS` on a GPU that runs only 32-lane waves.
pub const TABLE_WORDS_WAVE32: usize = 12288;

/// Most branch-target choices lowered in one branchless `select` leaf (#1448).
pub(super) const SELECT_LEAF_MAX: usize = 32;

impl Plan {
    /// Bits of a table entry: the class, and the index in the class's `targets`.
    fn widths(&self) -> (u32, u32) {
        let m = self.targets.iter().map(Vec::len).max().unwrap_or(1);
        (bits(self.rep.len()), usize::BITS - m.saturating_sub(1).leading_zeros())
    }

    /// 32-bit words of `xcl_blocks`.
    pub fn words(&self) -> usize {
        let (cb, pb) = self.widths();
        (self.order.len() * (cb + pb) as usize).div_ceil(32) + 1
    }

    /// Total `(taken, not taken)` entries across all class palettes.
    pub fn palette_len(&self) -> usize {
        self.targets.iter().map(Vec::len).sum()
    }

    pub(super) fn pack_blocks(&self) -> Vec<u32> {
        let (cb, pb) = self.widths();
        let (w, mut packed) = ((cb + pb) as usize, vec![0u32; self.words()]);
        for (b, (&c, &ch)) in self.class.iter().zip(&self.choice).enumerate() {
            let (e, at) = (c as u64 | ((ch as u64) << cb), b * w);
            packed[at / 32] |= (e << (at % 32)) as u32;
            packed[at / 32 + 1] |= (e << (at % 32) >> 32) as u32;
        }
        packed
    }

    #[cfg(test)]
    pub(super) fn step(&self, packed: &[u32], b: u32, taken: bool) -> u32 {
        let (cb, pb) = self.widths();
        let bit = b as usize * (cb + pb) as usize;
        let e = (packed[bit / 32] >> (bit % 32)) | packed[bit / 32 + 1].wrapping_shl(1).wrapping_shl(31 - (bit % 32) as u32);
        let c = (e & (u32::MAX >> (32 - cb))) as usize;
        let ch = if pb == 0 { 0 } else { ((e >> cb) & (u32::MAX >> (32 - pb))) as usize };
        let (lt, lf) = self.targets[c][ch];
        match if taken { lt } else { lf } {
            Leg::Abs(t) => t,
            Leg::Rel(d) => b.wrapping_add(d),
        }
    }
}

/// Whether a program takes the form by itself: large, mostly repeated blocks, small tables.
pub fn wanted(cfg: &Cfg, p: &Plan, tables: usize) -> bool {
    let insts = |n: usize| cfg.blocks[p.order[n]].insts.len();
    let all: usize = (0..p.order.len()).map(insts).sum();
    let once: usize = p.rep.iter().map(|&n| insts(n)).sum();
    large(cfg) && once * 10 < all * 3 && p.words() <= tables && p.palette_len() <= tables
}

/// Whether a program is large enough for `wanted` to be asked.
pub fn large(cfg: &Cfg) -> bool {
    cfg.rpo.iter().map(|&b| cfg.blocks[b].insts.len()).sum::<usize>() >= 20_000
}

pub const END: u32 = u32::MAX;

/// Sorts the reachable blocks into classes the lowering would emit the same for.
pub fn plan(cfg: &Cfg, lw: &Lw) -> Option<Plan> {
    let mut order = cfg.rpo.clone();
    order.sort_by_key(|&b| (cfg.blocks[b].pc, b));
    let mut number = vec![END; cfg.blocks.len()];
    for (n, &b) in order.iter().enumerate() {
        number[b] = n as u32;
    }
    let mut ids: HashMap<(Vec<u64>, &'static str), u32> = HashMap::new();
    let (mut class, mut rep, mut next) = (Vec::new(), Vec::new(), Vec::new());
    for (n, &b) in order.iter().enumerate() {
        let blk = &cfg.blocks[b];
        let mut key = vec![cfg.headers.contains(&b) as u64];
        for &i in &blk.insts {
            let d = &cfg.insts[i];
            key.push(d.words.len() as u64);
            key.extend(d.words.iter().map(|&w| w as u64));
            key.push(d.literal.map_or(u64::MAX, |l| l as u64));
            let sampler = lw.static_samplers.get(&d.pc);
            key.push(sampler.is_some() as u64);
            key.extend(sampler.into_iter().flatten().map(|&w| w as u64));
            // what `Lw::write_mask` asks of this address, for any register a mask can go to
            if let Some(&l) = lw.whole_live.get(&d.pc) {
                let sdwa = ext_encoding(d).and_then(|e| crate::gpu::rdna2::field_value(e, &d.words, "SDST"));
                let to = crate::gpu::rdna2::sgpr_uses(d).1.into_iter().chain([106]).chain(sdwa);
                key.extend(to.filter(|&r| r < 126).map(|r| (l >> r) as u64 & 3));
            }
            if d.inst.name == "S_GETPC_B64" {
                key.push(d.pc);
            }
        }
        let (end, to) = match blk.term {
            Term::End => ("end", (END, END)),
            Term::Jump(t) => ("jump", (number[t], number[t])),
            Term::Cond(i, t, f) => (cfg.insts[i].inst.name, (number[t], number[f])),
        };
        let fresh = ids.len() as u32;
        let c = *ids.entry((key, end)).or_insert(fresh);
        if c == fresh {
            rep.push(n);
        }
        class.push(c);
        next.push(to);
    }
    let (targets, choice) = compress_targets(&class, rep.len(), &next);
    let p = Plan { order, class, rep, targets, choice };
    let (cb, pb) = p.widths();
    (cb + pb <= 32).then_some(p)
}

/// Builds each class's sorted `(taken, not_taken)` leg palette, picking `Abs` vs `Rel` per leg
/// by frequency within the class (preferring `Abs` on ties so singletons emit pure constants).
fn compress_targets(class: &[u32], classes: usize, next: &[(u32, u32)]) -> (Vec<Vec<(Leg, Leg)>>, Vec<u32>) {
    let mut members = vec![Vec::new(); classes];
    for (b, &c) in class.iter().enumerate() {
        members[c as usize].push(b);
    }
    let (mut targets, mut choice) = (Vec::with_capacity(classes), vec![0u32; next.len()]);
    for bs in &members {
        if next[bs[0]].0 == END {
            targets.push(vec![(Leg::Abs(END), Leg::Abs(END))]);
            continue;
        }
        let pick = |leg: fn((u32, u32)) -> u32| -> Vec<Leg> {
            let (mut abs, mut rel) = (HashMap::new(), HashMap::new());
            for &b in bs {
                let t = leg(next[b]);
                *abs.entry(t).or_insert(0usize) += 1;
                *rel.entry(t.wrapping_sub(b as u32)).or_insert(0usize) += 1;
            }
            bs.iter()
                .map(|&b| {
                    let (t, d) = (leg(next[b]), leg(next[b]).wrapping_sub(b as u32));
                    if abs[&t] >= rel[&d] {
                        Leg::Abs(t)
                    } else {
                        Leg::Rel(d)
                    }
                })
                .collect()
        };
        let (lt, lf) = (pick(|p| p.0), pick(|p| p.1));
        let list: Vec<(Leg, Leg)> = lt.iter().copied().zip(lf.iter().copied()).collect::<BTreeSet<_>>().into_iter().collect();
        for ((&b, &t), &f) in bs.iter().zip(&lt).zip(&lf) {
            choice[b] = list.binary_search(&(t, f)).expect("a pair of the class") as u32;
        }
        targets.push(list);
    }
    (targets, choice)
}

fn bits(n: usize) -> u32 {
    (usize::BITS - n.saturating_sub(1).leading_zeros()).max(1)
}

/// A constant table of 32-bit words.
fn table(lw: &mut Lw, name: &str, words: &[u32]) -> Value {
    let vals: Vec<Value> = words.iter().map(|&w| lw.c(w)).collect();
    let i32 = lw.s.m.i32();
    let at = lw.s.m.array(words.len() as u64, i32);
    let init = lw.s.m.aggregate(at, &vals);
    lw.s.m.global(name, at, 0, Some(init), true, 4)
}

fn word(lw: &mut Lw, table: Value, i: Value) -> Value {
    let z = lw.c(0);
    let p = lw.s.m.gep(table, &[z, i]);
    lw.s.m.load(p, 4)
}

fn slot(lw: &mut Lw, var: u32) -> Value {
    let Lw { s, init, .. } = lw;
    init.slot(s, var)
}

/// Default class threshold and inner group size for `Nest::Auto`.
pub const AUTO_NEST_MIN_CLASSES: usize = 512;
pub const AUTO_NEST_GROUP_SIZE: usize = 32;

/// Dispatch nesting policy for the direct DXIL compact form (`XCL_DIRECT_COMPACT_NEST`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nest {
    /// Single flat `switch(class)` with one shared loop latch.
    Flat,
    /// Two-level nested switch (groups of [`AUTO_NEST_GROUP_SIZE`]) once class count reaches
    /// [`AUTO_NEST_MIN_CLASSES`], flat below that.
    Auto,
    /// Two-level nested switch with explicit power-of-two inner group size whenever `classes > p`.
    Fixed(usize),
}

impl Nest {
    /// Resolves the inner group size for a program with `classes` distinct block classes,
    /// returning `None` for flat dispatch.
    pub fn group_size(self, classes: usize) -> Option<usize> {
        match self {
            Nest::Flat => None,
            Nest::Auto => (classes >= AUTO_NEST_MIN_CLASSES).then_some(AUTO_NEST_GROUP_SIZE),
            Nest::Fixed(p) => (p >= 2 && p.is_power_of_two() && classes > p).then_some(p),
        }
    }

    /// Non-default cache tag folded into the direct-DXIL translated cache identity.
    pub fn cache_tag(self) -> Option<String> {
        match self {
            Nest::Auto => None,
            Nest::Flat => Some("xcl-direct-compact-nest-0".into()),
            Nest::Fixed(p) => Some(format!("xcl-direct-compact-nest-{p}")),
        }
    }
}

/// `XCL_DIRECT_COMPACT_NEST`: `auto` or unset nests `>= 512` classes into groups of 32;
/// `0` / `off` keeps a flat switch; a power of two in `2..=65536` fixes the inner group size.
pub fn nest_mode() -> Nest {
    parse_nest(std::env::var("XCL_DIRECT_COMPACT_NEST").ok().as_deref())
}

pub(crate) fn parse_nest(val: Option<&str>) -> Nest {
    let Some(raw) = val.map(str::trim) else { return Nest::Auto };
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return Nest::Auto;
    }
    if raw == "0" || raw.eq_ignore_ascii_case("off") {
        return Nest::Flat;
    }
    match raw.parse::<usize>() {
        Ok(p) if (2..=65536).contains(&p) && p.is_power_of_two() => Nest::Fixed(p),
        _ => Nest::Auto,
    }
}

/// Lowers the program as one loop from `head` to `exit` that switches to each block's class.
pub fn emit(lw: &mut Lw, cfg: &Cfg, p: &Plan, head: Block, exit: Block) -> LResult<()> {
    emit_with_nest(lw, cfg, p, head, exit, nest_mode())
}

/// Like [`emit`], with an explicit [`Nest`] policy.
pub fn emit_with_nest(lw: &mut Lw, cfg: &Cfg, p: &Plan, head: Block, exit: Block, nest: Nest) -> LResult<()> {
    let n = p.order.len();
    let (cb, pb) = p.widths();
    let width = cb + pb;
    let packed = p.pack_blocks();
    let blocks = table(lw, "xcl_blocks", &packed);

    let (at, to) = (lw.tmp_var(), lw.tmp_var());
    let (at, to, budget) = (slot(lw, at), slot(lw, to), slot(lw, VAR_BUDGET));
    lw.s.m.set_block(head);
    let b = lw.s.m.load(at, 4);
    let w = lw.c(width);
    let bit = lw.mul(b, w);
    let i = lw.lshrc(bit, 5);
    let sh = lw.andc(bit, 31);
    let lo = word(lw, blocks, i);
    let i1 = lw.addc(i, 1);
    let hi = word(lw, blocks, i1);
    let lo = lw.lshr(lo, sh);
    let hi = lw.shlc(hi, 1);
    let k31 = lw.c(31);
    let up = lw.sub(k31, sh);
    let hi = lw.shl(hi, up);
    let e = lw.or(lo, hi);
    let class = lw.andc(e, u32::MAX >> (32 - cb));
    let choice = if pb == 0 {
        lw.c(0)
    } else {
        let raw = lw.lshrc(e, cb);
        lw.andc(raw, u32::MAX >> (32 - pb))
    };
    let cases: Vec<Block> = p.rep.iter().map(|_| lw.block(Some(0))).collect();
    let latch = lw.block(Some(0));
    let group_size = nest.group_size(p.rep.len());
    let group_latches: Vec<Block> = if let Some(gs) = group_size {
        let groups = p.rep.len().div_ceil(gs);
        let shift = gs.trailing_zeros();
        let entries: Vec<Block> = (0..groups).map(|_| lw.block(Some(0))).collect();
        let latches: Vec<Block> = (0..groups).map(|_| lw.block(Some(0))).collect();
        let grp = lw.lshrc(class, shift);
        let outer_arms: Vec<(Value, Block)> = entries.iter().enumerate().skip(1).map(|(g, &blk)| (lw.c(g as u32), blk)).collect();
        lw.s.m.switch(grp, entries[0], &outer_arms);
        for (g, &entry_blk) in entries.iter().enumerate() {
            let lo = g * gs;
            let hi = (lo + gs).min(p.rep.len());
            lw.s.m.set_block(entry_blk);
            if hi - lo == 1 {
                lw.s.m.br(cases[lo]);
            } else {
                let inner_arms: Vec<(Value, Block)> = (lo + 1..hi).map(|k| (lw.c(k as u32), cases[k])).collect();
                lw.s.m.switch(class, cases[lo], &inner_arms);
            }
        }
        latches
    } else {
        let arms: Vec<(Value, Block)> = cases.iter().enumerate().skip(1).map(|(k, &c)| (lw.c(k as u32), c)).collect();
        lw.s.m.switch(class, cases[0], &arms);
        Vec::new()
    };
    for (k, (&c, &r)) in cases.iter().zip(&p.rep).enumerate() {
        let target_latch = group_size.map_or(latch, |gs| group_latches[k / gs]);
        lw.s.m.set_block(c);
        lw.init.mem = Some(c);
        lw.ssa.begin_log();
        let blk = &cfg.blocks[p.order[r]];
        if cfg.headers.contains(&p.order[r]) {
            // a loop head spends the loop budget
            let left = lw.s.m.load(budget, 4);
            let none = lw.cmpc(Pred::Eq, left, 0);
            let (out, body) = (lw.block(Some(1)), lw.block(Some(1)));
            lw.cond_br(none, out, body);
            lw.s.m.set_block(out);
            let end = lw.c(END);
            lw.s.m.store(to, end, 4);
            lw.s.m.br(target_latch);
            lw.s.m.set_block(body);
            let one = lw.c(1);
            let l = lw.sub(left, one);
            lw.s.m.store(budget, l, 4);
        }
        for &i in &blk.insts {
            lw.inst(&cfg.insts[i])?;
        }
        let cond = if let Term::Cond(i, ..) = blk.term { Some(lw.branch_cond(&cfg.insts[i])?) } else { None };
        for var in lw.ssa.end_log().into_iter().filter(|&v| v < VAR_TMP) {
            let (v, p) = (lw.rd(var), slot(lw, var));
            lw.s.m.store(p, v, 4);
        }
        let next = emit_next(lw, b, choice, cond, &p.targets[k]);
        lw.s.m.store(to, next, 4);
        lw.s.m.br(target_latch);
    }
    if group_size.is_some() {
        // Each group latch performs the loop-exit check rather than forwarding with an empty
        // unconditional branch, so driver CFG simplification preserves the two-level merge fan-in
        // during SSA promotion without adding extra runtime instructions.
        for &gl in &group_latches {
            lw.s.m.set_block(gl);
            let nxt = lw.s.m.load(to, 4);
            let stop = lw.cmpc(Pred::UGe, nxt, n as u32);
            lw.s.m.cond_br(stop, exit, latch);
        }
        // the epilogue's reads load in exit
        lw.init.mem = Some(exit);
        lw.s.m.set_block(latch);
        let next = lw.s.m.load(to, 4);
        lw.s.m.store(at, next, 4);
        lw.br(head);
        lw.seal(head);
    } else {
        // the epilogue's reads load here
        lw.init.mem = Some(latch);
        lw.s.m.set_block(latch);
        let next = lw.s.m.load(to, 4);
        let stop = lw.cmpc(Pred::UGe, next, n as u32);
        let cont = lw.block(Some(1));
        lw.cond_br(stop, exit, cont);
        lw.s.m.set_block(cont);
        lw.s.m.store(at, next, 4);
        lw.br(head);
        lw.seal(head);
    }
    Ok(())
}

fn select_legs(lw: &mut Lw, b: Value, choice: Value, lo: usize, legs: &[Leg]) -> Value {
    if legs.iter().all(|&l| l == legs[0]) {
        return match legs[0] {
            Leg::Abs(t) => lw.c(t),
            Leg::Rel(d) => lw.addc(b, d),
        };
    }
    if legs.iter().all(|l| matches!(l, Leg::Rel(_))) {
        let abs: Vec<Leg> = legs.iter().map(|&l| match l { Leg::Rel(d) => Leg::Abs(d), _ => l }).collect();
        let d = select_legs(lw, b, choice, lo, &abs);
        return lw.add(b, d);
    }
    let mid = legs.len() / 2;
    let (lhs, rhs) = (select_legs(lw, b, choice, lo, &legs[..mid]), select_legs(lw, b, choice, lo + mid, &legs[mid..]));
    let (p, k) = if legs.len() == 2 { (Pred::Eq, lo) } else { (Pred::ULt, lo + mid) };
    let cond = lw.cmpc(p, choice, k as u32);
    lw.sel(cond, lhs, rhs)
}

fn emit_tree(lw: &mut Lw, choice: Value, lo: usize, len: usize, cap: usize, leaf: &mut dyn FnMut(&mut Lw, usize, usize) -> Value) -> Value {
    if len <= cap {
        return leaf(lw, lo, lo + len);
    }
    let mid = len / 2;
    let lt = lw.cmpc(Pred::ULt, choice, (lo + mid) as u32);
    let (left_in, right_in) = (lw.s.m.new_block(), lw.s.m.new_block());
    lw.s.m.cond_br(lt, left_in, right_in);
    lw.s.m.set_block(left_in);
    let (left_val, left_out) = (emit_tree(lw, choice, lo, mid, cap, leaf), lw.s.m.current_block());
    lw.s.m.set_block(right_in);
    let (right_val, right_out) = (emit_tree(lw, choice, lo + mid, len - mid, cap, leaf), lw.s.m.current_block());
    let join = lw.s.m.new_block();
    lw.s.m.set_block(left_out);
    lw.s.m.br(join);
    lw.s.m.set_block(right_out);
    lw.s.m.br(join);
    lw.s.m.set_block(join);
    let i32 = lw.s.m.i32();
    let phi = lw.s.m.phi(i32);
    lw.s.m.add_incoming(phi, left_val, left_out);
    lw.s.m.add_incoming(phi, right_val, right_out);
    phi
}

fn emit_next(lw: &mut Lw, b: Value, choice: Value, cond: Option<Value>, choices: &[(Leg, Leg)]) -> Value {
    if choices[0].0 == Leg::Abs(END) {
        return lw.c(END);
    }
    let legs_t: Vec<Leg> = choices.iter().map(|&(t, _)| t).collect();
    let legs_f: Vec<Leg> = choices.iter().map(|&(_, f)| f).collect();
    match (cond, legs_f.iter().all(|&f| f == legs_f[0])) {
        (None, _) => emit_tree(lw, choice, 0, legs_t.len(), SELECT_LEAF_MAX, &mut |lw, lo, hi| select_legs(lw, b, choice, lo, &legs_t[lo..hi])),
        (Some(c), true) => {
            let taken = emit_tree(lw, choice, 0, legs_t.len(), SELECT_LEAF_MAX, &mut |lw, lo, hi| select_legs(lw, b, choice, lo, &legs_t[lo..hi]));
            let not_taken = select_legs(lw, b, choice, 0, &legs_f[..1]);
            lw.sel(c, taken, not_taken)
        }
        (Some(c), false) => emit_tree(lw, choice, 0, legs_t.len(), SELECT_LEAF_MAX / 2, &mut |lw, lo, hi| {
            let (t, f) = (select_legs(lw, b, choice, lo, &legs_t[lo..hi]), select_legs(lw, b, choice, lo, &legs_f[lo..hi]));
            lw.sel(c, t, f)
        }),
    }
}
