//! Cheap cleanup of the entry function before it is written: unreachable
//! blocks emptied, trivial phis replaced by their single value, common
//! subexpressions merged (pure instructions along the dominator tree, wave
//! intrinsics within a block), dead instructions removed.
//!
//! The translator emits each guest instruction's per-lane expansion as is
//! (repeated lane-bit / ballot / extract sequences, operands of discarded
//! results); this pass removes the redundancy without changing what any
//! remaining instruction computes.

use std::collections::HashMap;

use super::{BinOp, Block, CastOp, FnAttrs, Module, Op, Pred, Ty, Value};

impl Op {
    /// Calls `f` on every value operand.
    pub(super) fn for_each_operand(&self, mut f: impl FnMut(Value)) {
        match self {
            Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) | Op::Store(a, b, _) | Op::AtomicRmw(_, a, b) => {
                f(*a);
                f(*b)
            }
            Op::Cast(_, v, _) | Op::Extract(v, _) | Op::Load(v, _) | Op::Alloca(_, v, _) | Op::CondBr(v, ..) => f(*v),
            Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
                f(*a);
                f(*b);
                f(*c)
            }
            Op::Call(_, args) => args.iter().copied().for_each(f),
            Op::Phi(inc) => inc.iter().for_each(|p| f(p.0)),
            Op::Gep(p, idx) => {
                f(*p);
                idx.iter().copied().for_each(f)
            }
            Op::Switch(v, _, cases) => {
                f(*v);
                cases.iter().for_each(|c| f(c.0))
            }
            Op::Br(_) | Op::Ret | Op::Unreachable => {}
        }
    }

    /// Rewrites every value operand through `f`.
    pub(super) fn map_operands(&mut self, mut f: impl FnMut(Value) -> Value) {
        match self {
            Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) | Op::Store(a, b, _) | Op::AtomicRmw(_, a, b) => {
                *a = f(*a);
                *b = f(*b)
            }
            Op::Cast(_, v, _) | Op::Extract(v, _) | Op::Load(v, _) | Op::Alloca(_, v, _) | Op::CondBr(v, ..) => *v = f(*v),
            Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
                *a = f(*a);
                *b = f(*b);
                *c = f(*c)
            }
            Op::Call(_, args) => args.iter_mut().for_each(|a| *a = f(*a)),
            Op::Phi(inc) => inc.iter_mut().for_each(|p| p.0 = f(p.0)),
            Op::Gep(p, idx) => {
                *p = f(*p);
                idx.iter_mut().for_each(|v| *v = f(*v))
            }
            Op::Switch(v, _, cases) => {
                *v = f(*v);
                cases.iter_mut().for_each(|c| c.0 = f(c.0))
            }
            Op::Br(_) | Op::Ret | Op::Unreachable => {}
        }
    }
}

/// How an instruction may be merged with an identical one.
#[derive(PartialEq)]
enum Cse {
    No,
    /// pure: any dominating copy will do
    Dom,
    /// a wave intrinsic: same result only under the same active lanes (same block)
    Block,
}

impl Module {
    fn wave_call(&self, f: u32) -> bool {
        let n = &self.funcs[f as usize].name;
        n.starts_with("dx.op.wave") || n.starts_with("dx.op.quad")
    }

    fn is_const_global_ptr(&self, v: Value) -> bool {
        match v {
            Value::Global(g) => self.globals[g as usize].constant,
            Value::Inst(i) => match &self.insts[i as usize].op {
                Op::Gep(base, _) => self.is_const_global_ptr(*base),
                _ => false,
            },
            _ => false,
        }
    }

    fn cse_kind(&self, op: &Op) -> Cse {
        match op {
            Op::Bin(..) | Op::Cast(..) | Op::Cmp(..) | Op::Select(..) | Op::Extract(..) | Op::Gep(..) => Cse::Dom,
            Op::Load(p, _) if self.is_const_global_ptr(*p) => Cse::Dom,
            Op::Phi(_) if cfg!(test) && std::env::var_os("XCL_SAMPLE_PHI_CSE").is_some() => Cse::Block,
            Op::Call(f, _) if self.funcs[*f as usize].attrs == FnAttrs::ReadNone => Cse::Dom,
            Op::Call(f, _) if self.wave_call(*f) && !self.funcs[*f as usize].name.starts_with("dx.op.waveIsFirstLane") => Cse::Block,
            _ => Cse::No,
        }
    }

    /// Conservative bitmask of bits that *may* be 1 in `v` (0 bits are guaranteed zero),
    /// for integer widths `1..=32` up to `depth` instructions deep.
    fn known_nonzero_bits(&self, v: Value, w: u32, depth: u8) -> u64 {
        let mask = (1u64 << w) - 1;
        if let Some(c) = self.const_u64(v) {
            return c & mask;
        }
        if depth == 0 {
            return mask;
        }
        let Value::Inst(j) = v else { return mask };
        let d = depth - 1;
        let bound_mask = |upper: u64| if upper == 0 { 0 } else { (u64::MAX >> upper.leading_zeros()) & mask };
        match &self.insts[j as usize].op {
            Op::Bin(BinOp::And, x, y, 0) => self.known_nonzero_bits(*x, w, d) & self.known_nonzero_bits(*y, w, d),
            Op::Bin(BinOp::Or | BinOp::Xor, x, y, 0) => (self.known_nonzero_bits(*x, w, d) | self.known_nonzero_bits(*y, w, d)) & mask,
            Op::Bin(BinOp::LShr, x, sh, 0) => match self.const_u64(*sh) {
                Some(s) if s < w as u64 => self.known_nonzero_bits(*x, w, d) >> s,
                _ => mask,
            },
            Op::Bin(BinOp::Shl, x, sh, 0) => match self.const_u64(*sh) {
                Some(s) if s < w as u64 => (self.known_nonzero_bits(*x, w, d) << s) & mask,
                _ => mask,
            },
            Op::Bin(BinOp::AShr, x, sh, 0) => match self.const_u64(*sh) {
                Some(s) if s < w as u64 => {
                    let kx = self.known_nonzero_bits(*x, w, d);
                    if kx & (1u64 << (w - 1)) == 0 {
                        kx >> s
                    } else {
                        mask
                    }
                }
                _ => mask,
            },
            Op::Bin(BinOp::Add, x, y, 0) => {
                let (kx, ky) = (self.known_nonzero_bits(*x, w, d), self.known_nonzero_bits(*y, w, d));
                if kx & ky == 0 {
                    (kx | ky) & mask
                } else if kx + ky <= mask {
                    bound_mask(kx + ky)
                } else {
                    mask
                }
            }
            Op::Bin(BinOp::Mul, x, y, 0) => {
                let (kx, ky) = (self.known_nonzero_bits(*x, w, d), self.known_nonzero_bits(*y, w, d));
                if kx == 0 || ky == 0 {
                    return 0;
                }
                let tz = kx.trailing_zeros() + ky.trailing_zeros();
                if tz >= w {
                    return 0;
                }
                let low = mask & !((1u64 << tz) - 1);
                match kx.checked_mul(ky) {
                    Some(prod) if prod <= mask => bound_mask(prod) & low,
                    _ => low,
                }
            }
            Op::Bin(BinOp::UDiv, x, y, 0) => match self.const_u64(*y) {
                Some(div) if div > 0 => bound_mask(self.known_nonzero_bits(*x, w, d) / div),
                _ => mask,
            },
            Op::Bin(BinOp::URem, x, y, 0) => match self.const_u64(*y) {
                Some(div) if div > 0 => bound_mask(self.known_nonzero_bits(*x, w, d).min(div - 1)),
                _ => mask,
            },
            Op::Select(_, t, f) => (self.known_nonzero_bits(*t, w, d) | self.known_nonzero_bits(*f, w, d)) & mask,
            Op::Cast(CastOp::ZExt, inner, _) => match self.int_width(self.value_ty(*inner)) {
                Some(iw) if iw < w => self.known_nonzero_bits(*inner, iw, d) & mask,
                _ => mask,
            },
            Op::Cast(CastOp::Trunc, inner, _) => match self.int_width(self.value_ty(*inner)) {
                Some(iw) if iw <= 32 => self.known_nonzero_bits(*inner, iw, d) & mask,
                _ => mask,
            },
            Op::Call(f, args) => {
                let name = &self.funcs[*f as usize].name;
                if name == "dx.op.binary.i32" && args.len() == 3 && w == 32 {
                    match self.const_u64(args[0]) {
                        // UMax: result is either operand, so zero bits in both stay zero.
                        Some(39) => (self.known_nonzero_bits(args[1], 32, d) | self.known_nonzero_bits(args[2], 32, d)) & mask,
                        // UMin: result <= min(ka, kb), so bits above the MSB of min(ka, kb) are zero.
                        Some(40) => {
                            let upper = self.known_nonzero_bits(args[1], 32, d).min(self.known_nonzero_bits(args[2], 32, d));
                            bound_mask(upper)
                        }
                        _ => mask,
                    }
                } else if name == "dx.op.unaryBits.i32" && args.len() == 2 && w == 32 && self.const_u64(args[0]) == Some(31) {
                    // Countbits on i32 is in 0..=32.
                    0x3F
                } else {
                    mask
                }
            }
            _ => mask,
        }
    }

    /// Cheap algebraic & constant simplification on already-resolved operands.
    fn simplify_op(&mut self, op: &Op, ty: Ty) -> Option<Value> {
        // Keep 64-bit instructions intact so Int64Ops / Doubles shader flags stay consistent.
        if matches!(self.types[ty.0 as usize], super::TypeDef::Int(64) | super::TypeDef::Double) {
            return None;
        }
        match *op {
            Op::Select(c, t, f) => {
                if t == f {
                    return Some(t);
                }
                match self.const_u64(c) {
                    Some(1) => Some(t),
                    Some(0) => Some(f),
                    _ => None,
                }
            }
            Op::Bin(bop, a, b, 0) => {
                let w = self.int_width(ty)?;
                // avoid dropping 64-bit ops that keep Int64Ops shader flags consistent
                if w > 32 {
                    return None;
                }
                let mask = (1u64 << w) - 1;
                let (ca, cb) = (self.const_u64(a), self.const_u64(b));
                if let (Some(x), Some(y)) = (ca, cb) {
                    let r = match bop {
                        BinOp::Add => Some(x.wrapping_add(y)),
                        BinOp::Sub => Some(x.wrapping_sub(y)),
                        BinOp::Mul => Some(x.wrapping_mul(y)),
                        BinOp::And => Some(x & y),
                        BinOp::Or => Some(x | y),
                        BinOp::Xor => Some(x ^ y),
                        BinOp::Shl if y < w as u64 => Some(x << y),
                        BinOp::LShr if y < w as u64 => Some(x >> y),
                        BinOp::AShr if y < w as u64 => Some((((x << (64 - w)) as i64) >> (64 - w + y as u32)) as u64),
                        BinOp::UDiv if y != 0 => Some(x / y),
                        BinOp::URem if y != 0 => Some(x % y),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r & mask));
                    }
                }
                match (bop, cb) {
                    (BinOp::Add | BinOp::Sub | BinOp::Or | BinOp::Xor | BinOp::Shl | BinOp::LShr | BinOp::AShr, Some(0)) => return Some(a),
                    (BinOp::And, Some(m)) if m == mask => return Some(a),
                    (BinOp::Or, Some(m)) if m == mask => return Some(b),
                    (BinOp::And | BinOp::Mul, Some(0)) => return Some(b),
                    (BinOp::Mul, Some(1)) => return Some(a),
                    _ => {}
                }
                match (bop, ca) {
                    (BinOp::Add | BinOp::Or | BinOp::Xor, Some(0)) => return Some(b),
                    (BinOp::And, Some(m)) if m == mask => return Some(b),
                    (BinOp::Or, Some(m)) if m == mask => return Some(a),
                    (BinOp::And | BinOp::Mul | BinOp::Shl | BinOp::LShr | BinOp::AShr, Some(0)) => return Some(a),
                    (BinOp::Mul, Some(1)) => return Some(b),
                    _ => {}
                }
                if a == b {
                    match bop {
                        BinOp::And | BinOp::Or => return Some(a),
                        BinOp::Sub | BinOp::Xor => return Some(self.const_int(ty, 0)),
                        _ => {}
                    }
                }
                match bop {
                    BinOp::Xor => {
                        for (cand, other) in [(a, b), (b, a)] {
                            if let Value::Inst(j) = cand {
                                if let Op::Bin(BinOp::Xor, x, y, 0) = self.insts[j as usize].op {
                                    if other == y {
                                        return Some(x);
                                    }
                                    if other == x {
                                        return Some(y);
                                    }
                                }
                            }
                        }
                    }
                    BinOp::And => {
                        let ka = self.known_nonzero_bits(a, w, 4);
                        let kb = self.known_nonzero_bits(b, w, 4);
                        if ka & kb == 0 {
                            return Some(self.const_int(ty, 0));
                        }
                        if let Some(m) = cb {
                            if ka & !m == 0 {
                                return Some(a);
                            }
                        }
                        if let Some(m) = ca {
                            if kb & !m == 0 {
                                return Some(b);
                            }
                        }
                    }
                    BinOp::LShr => {
                        if let Some(s) = cb.filter(|&s| s < w as u64) {
                            if (self.known_nonzero_bits(a, w, 4) >> s) == 0 {
                                return Some(self.const_int(ty, 0));
                            }
                        }
                    }
                    BinOp::Shl => {
                        if let Some(s) = cb.filter(|&s| s < w as u64) {
                            if ((self.known_nonzero_bits(a, w, 4) << s) & mask) == 0 {
                                return Some(self.const_int(ty, 0));
                            }
                        }
                    }
                    BinOp::AShr => {
                        if let Some(s) = cb.filter(|&s| s < w as u64) {
                            let ka = self.known_nonzero_bits(a, w, 4);
                            if ka & (1u64 << (w - 1)) == 0 && (ka >> s) == 0 {
                                return Some(self.const_int(ty, 0));
                            }
                        }
                    }
                    BinOp::Or => {
                        if let Some((v, m)) = cb.map(|m| (a, m)).or_else(|| ca.map(|m| (b, m))) {
                            if let Value::Inst(j) = v {
                                if let Op::Bin(BinOp::Or, x, y, 0) = self.insts[j as usize].op {
                                    if let Some(c1) = self.const_u64(y).or_else(|| self.const_u64(x)) {
                                        if c1 | m == c1 {
                                            return Some(v);
                                        }
                                        if (c1 | m) == mask {
                                            return Some(self.const_int(ty, mask));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
                None
            }
            Op::Cmp(p, a, b) => {
                let at = self.value_ty(a);
                let w = self.int_width(at)?;
                if w > 32 {
                    return None;
                }
                if let (Some(x), Some(y)) = (self.const_u64(a), self.const_u64(b)) {
                    let sx = |v: u64| ((v << (64 - w)) as i64) >> (64 - w);
                    let r = match p {
                        Pred::Eq => Some(x == y),
                        Pred::Ne => Some(x != y),
                        Pred::UGt => Some(x > y),
                        Pred::UGe => Some(x >= y),
                        Pred::ULt => Some(x < y),
                        Pred::ULe => Some(x <= y),
                        Pred::SGt => Some(sx(x) > sx(y)),
                        Pred::SGe => Some(sx(x) >= sx(y)),
                        Pred::SLt => Some(sx(x) < sx(y)),
                        Pred::SLe => Some(sx(x) <= sx(y)),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r as u64));
                    }
                }
                if a == b {
                    let r = match p {
                        Pred::Eq | Pred::ULe | Pred::UGe | Pred::SLe | Pred::SGe => Some(true),
                        Pred::Ne | Pred::ULt | Pred::UGt | Pred::SLt | Pred::SGt => Some(false),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r as u64));
                    }
                }
                None
            }
            Op::Cast(CastOp::BitCast, Value::Inst(j), to) => {
                if matches!(self.types[self.insts[j as usize].ty.0 as usize], super::TypeDef::Int(64) | super::TypeDef::Double) {
                    return None;
                }
                if let Op::Cast(CastOp::BitCast, inner, _) = self.insts[j as usize].op {
                    if self.value_ty(inner) == to {
                        return Some(inner);
                    }
                }
                None
            }
            Op::Cast(cop @ (CastOp::ZExt | CastOp::SExt | CastOp::Trunc), v, to) => {
                let sw = self.int_width(self.value_ty(v))?;
                let dw = self.int_width(to)?;
                if sw > 32 || dw > 32 {
                    return None;
                }
                let x = self.const_u64(v)?;
                let r = match cop {
                    CastOp::ZExt | CastOp::Trunc => x & ((1u64 << dw) - 1),
                    CastOp::SExt => ((((x << (64 - sw)) as i64) >> (64 - sw)) as u64) & ((1u64 << dw) - 1),
                    _ => unreachable!(),
                };
                Some(self.const_int(to, r))
            }
            _ => None,
        }
    }

    /// Whether an unused instruction can go. 64-bit values stay: helpers already set
    /// the Int64Ops / Doubles / DoubleExtensions flags for them.
    fn removable(&self, op: &Op, ty: Ty, wide: &[Ty]) -> bool {
        if wide.contains(&ty) {
            return false;
        }
        let mut w = false;
        op.for_each_operand(|v| w |= wide.contains(&self.value_ty(v)));
        if w {
            return false;
        }
        match op {
            Op::Bin(..) | Op::Cast(..) | Op::Cmp(..) | Op::Select(..) | Op::Extract(..) | Op::Gep(..) | Op::Phi(_) | Op::Load(..) | Op::Alloca(..) => true,
            // only calls whose presence sets no shader flag (the flags are collected while
            // building and must match the code: no wave ops, f64 math, typed / raw loads)
            Op::Call(f, _) => {
                let d = &self.funcs[*f as usize];
                matches!(d.attrs, FnAttrs::ReadNone | FnAttrs::ReadOnly)
                    && !d.name.ends_with(".f64")
                    && ["dx.op.unary.", "dx.op.binary.", "dx.op.tertiary.", "dx.op.loadInput.", "dx.op.annotateHandle", "dx.op.createHandleFromBinding", "dx.op.cbufferLoadLegacy."]
                        .iter()
                        .any(|p| d.name.starts_with(p))
            }
            _ => false,
        }
    }

    /// Whether every non-phi operand of the reachable instructions is written before its
    /// use and has a value (what the writer checks).
    fn uses_well_formed(&self, order: &[u32]) -> bool {
        let mut at = vec![usize::MAX; self.insts.len()];
        let mut k = 0;
        for &b in order {
            for &i in &self.blocks[b as usize] {
                at[i as usize] = k;
                k += 1;
            }
        }
        let mut ok = true;
        for &b in order {
            for &i in &self.blocks[b as usize] {
                let op = &self.insts[i as usize].op;
                if matches!(op, Op::Phi(_)) {
                    continue;
                }
                op.for_each_operand(|v| {
                    if let Value::Inst(j) = v {
                        ok &= at[j as usize] < at[i as usize] && !matches!(self.types[self.insts[j as usize].ty.0 as usize], super::TypeDef::Void);
                    }
                });
            }
        }
        ok
    }

    /// Immediate dominators of the blocks in `order` (reverse postorder), by
    /// Cooper, Harvey and Kennedy's iteration ("A Simple, Fast Dominance Algorithm").
    pub(super) fn idoms(&self, order: &[u32]) -> Vec<Option<usize>> {
        let n = self.blocks.len();
        let mut rpo = vec![usize::MAX; n];
        for (k, &b) in order.iter().enumerate() {
            rpo[b as usize] = k;
        }
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &b in order {
            for s in self.successors(b as usize) {
                preds[s.0 as usize].push(b as usize);
            }
        }
        let mut idom: Vec<Option<usize>> = vec![None; n];
        idom[0] = Some(0);
        let mut changed = true;
        while changed {
            changed = false;
            for &b in order.iter().skip(1) {
                let b = b as usize;
                let mut new: Option<usize> = None;
                for &p in &preds[b] {
                    if idom[p].is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => p,
                        Some(mut x) => {
                            let mut y = p;
                            while x != y {
                                while rpo[x] > rpo[y] {
                                    x = idom[x].unwrap();
                                }
                                while rpo[y] > rpo[x] {
                                    y = idom[y].unwrap();
                                }
                            }
                            x
                        }
                    });
                }
                if new.is_some() && idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        idom
    }

    /// Cleans up the entry function (see the module docs of `opt`). Values
    /// keep their identities; replaced ones simply lose their uses.
    ///
    /// `XCL_DXIL_NO_OPT=phi,cse,dce,merge` (any subset, or `all`) skips passes, for bisecting.
    pub fn optimize(&mut self) {
        let off = std::env::var("XCL_DXIL_NO_OPT").unwrap_or_default();
        let skip = |p: &str| off.split(',').any(|x| x.trim() == p || x.trim() == "all");
        let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
        // malformed IR (a use before its definition or of a void result) is left for the
        // writer to report, not cleaned away
        if !self.uses_well_formed(&order) {
            return;
        }
        // unreachable blocks: emptied (the writer drops them)
        let mut reach = vec![false; self.blocks.len()];
        for &b in &order {
            reach[b as usize] = true;
        }
        for (b, r) in reach.iter().enumerate() {
            if !r {
                self.blocks[b].clear();
            }
        }
        for &b in &order {
            for k in 0..self.blocks[b as usize].len() {
                let i = self.blocks[b as usize][k] as usize;
                if let Op::Phi(inc) = &mut self.insts[i].op {
                    inc.retain(|p| reach[p.1 .0 as usize]);
                }
            }
        }

        let mut repl: Vec<Option<Value>> = vec![None; self.insts.len()];

        // trivial phis (all incoming values the phi itself or one value), to a fixed point
        while !skip("phi") {
            let mut changed = false;
            for &b in &order {
                for &i in &self.blocks[b as usize] {
                    let i = i as usize;
                    if repl[i].is_some() {
                        continue;
                    }
                    let Op::Phi(inc) = &self.insts[i].op else { break };
                    let me = Value::Inst(i as u32);
                    let mut same: Option<Value> = None;
                    let mut trivial = true;
                    for &(v, _) in inc {
                        let v = resolve(&mut repl, v);
                        if v == me || Some(v) == same {
                            continue;
                        }
                        if same.is_some() {
                            trivial = false;
                            break;
                        }
                        same = Some(v);
                    }
                    if let (true, Some(v)) = (trivial, same) {
                        repl[i] = Some(v);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // CSE along the dominator tree (preorder walk with a scoped table)
        let idom = self.idoms(&order);
        let mut kids: Vec<Vec<usize>> = vec![Vec::new(); self.blocks.len()];
        for &b in order.iter().skip(1) {
            if let Some(d) = idom[b as usize] {
                kids[d].push(b as usize);
            }
        }
        let mut table: HashMap<(Op, Ty), Value> = HashMap::new();
        // (block, undo log length) frames; undo log holds keys to remove / restore
        let mut undo: Vec<((Op, Ty), Option<Value>)> = Vec::new();
        enum Step {
            Enter(usize),
            Leave(usize),
        }
        let mut stack = if skip("cse") { Vec::new() } else { vec![Step::Enter(0)] };
        let mut cse_replaced = false;
        while let Some(step) = stack.pop() {
            let b = match step {
                Step::Leave(mark) => {
                    while undo.len() > mark {
                        let (k, old) = undo.pop().unwrap();
                        match old {
                            Some(v) => table.insert(k, v),
                            None => table.remove(&k),
                        };
                    }
                    continue;
                }
                Step::Enter(b) => b,
            };
            stack.push(Step::Leave(undo.len()));
            let mut local: HashMap<(Op, Ty), Value> = HashMap::new();
            for k in 0..self.blocks[b].len() {
                let i = self.blocks[b][k] as usize;
                let mut op = std::mem::replace(&mut self.insts[i].op, Op::Ret);
                op.map_operands(|v| resolve(&mut repl, v));
                if repl[i].is_none() {
                    if let Some(s) = self.simplify_op(&op, self.insts[i].ty) {
                        repl[i] = Some(s);
                        cse_replaced = true;
                    } else {
                        let kind = self.cse_kind(&op);
                        if kind != Cse::No {
                            let key = (op.clone(), self.insts[i].ty);
                            let t = if kind == Cse::Dom { &mut table } else { &mut local };
                            if let Some(&v) = t.get(&key) {
                                repl[i] = Some(v);
                                cse_replaced = true;
                            } else {
                                let old = t.insert(key.clone(), Value::Inst(i as u32));
                                if kind == Cse::Dom {
                                    undo.push((key, old));
                                }
                            }
                        } else if matches!(&op, Op::Call(f, _) if self.funcs[*f as usize].name == "dx.op.discard") {
                            // `dx.op.discard` demotes lanes mid-block, changing the active lane
                            // set for subsequent wave/quad intrinsics in the same basic block.
                            local.clear();
                        }
                    }
                }
                self.insts[i].op = op;
            }
            for &c in kids[b].iter().rev() {
                stack.push(Step::Enter(c));
            }
        }
        // second trivial-phi fixed point if CSE/simplification collapsed incoming values
        if cse_replaced && !skip("phi") {
            loop {
                let mut changed = false;
                for &b in &order {
                    for &i in &self.blocks[b as usize] {
                        let i = i as usize;
                        if repl[i].is_some() {
                            continue;
                        }
                        let Op::Phi(inc) = &self.insts[i].op else { break };
                        let me = Value::Inst(i as u32);
                        let mut same: Option<Value> = None;
                        let mut trivial = true;
                        for &(v, _) in inc {
                            let v = resolve(&mut repl, v);
                            if v == me || Some(v) == same {
                                continue;
                            }
                            if same.is_some() {
                                trivial = false;
                                break;
                            }
                            same = Some(v);
                        }
                        if let (true, Some(v)) = (trivial, same) {
                            repl[i] = Some(v);
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        // phis may name values defined later (back edges): resolve once more
        for &b in &order {
            for k in 0..self.blocks[b as usize].len() {
                let i = self.blocks[b as usize][k] as usize;
                self.insts[i].op.map_operands(|v| resolve(&mut repl, v));
            }
        }

        // dead instructions: live = side effects, terminators and their operands, transitively
        let (i64t, f64t) = self.existing_i64_f64();
        let wide: Vec<Ty> = i64t.into_iter().chain(f64t).collect();
        let mut live = vec![false; self.insts.len()];
        let mut work: Vec<u32> = Vec::new();
        for &b in &order {
            for &i in &self.blocks[b as usize] {
                if repl[i as usize].is_none() && (skip("dce") || !self.removable(&self.insts[i as usize].op, self.insts[i as usize].ty, &wide)) {
                    live[i as usize] = true;
                    work.push(i);
                }
            }
        }
        while let Some(i) = work.pop() {
            self.insts[i as usize].op.for_each_operand(|v| {
                if let Value::Inst(j) = v {
                    if !live[j as usize] {
                        live[j as usize] = true;
                        work.push(j);
                    }
                }
            });
        }
        for b in &mut self.blocks {
            b.retain(|&i| live[i as usize]);
        }

        if !skip("merge") && self.simplify_cfg() {
            let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
            let mut repl2: Vec<Option<Value>> = vec![None; self.insts.len()];
            let mut phi_changed = false;
            while !skip("phi") {
                let mut changed = false;
                for &b in &order {
                    for &i in &self.blocks[b as usize] {
                        let i = i as usize;
                        if repl2[i].is_some() {
                            continue;
                        }
                        let Op::Phi(inc) = &self.insts[i].op else { break };
                        let me = Value::Inst(i as u32);
                        let mut same: Option<Value> = None;
                        let mut trivial = true;
                        for &(v, _) in inc {
                            let v = resolve(&mut repl2, v);
                            if v == me || Some(v) == same {
                                continue;
                            }
                            if same.is_some() {
                                trivial = false;
                                break;
                            }
                            same = Some(v);
                        }
                        if let (true, Some(v)) = (trivial, same) {
                            repl2[i] = Some(v);
                            changed = true;
                            phi_changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            if phi_changed {
                for &b in &order {
                    for k in 0..self.blocks[b as usize].len() {
                        let i = self.blocks[b as usize][k] as usize;
                        self.insts[i].op.map_operands(|v| resolve(&mut repl2, v));
                    }
                }
            }
            if !skip("dce") {
                live.fill(false);
                work.clear();
                for &b in &order {
                    for &i in &self.blocks[b as usize] {
                        if repl2[i as usize].is_none() && !self.removable(&self.insts[i as usize].op, self.insts[i as usize].ty, &wide) {
                            live[i as usize] = true;
                            work.push(i);
                        }
                    }
                }
                while let Some(i) = work.pop() {
                    self.insts[i as usize].op.for_each_operand(|v| {
                        if let Value::Inst(j) = v {
                            if !live[j as usize] {
                                live[j as usize] = true;
                                work.push(j);
                            }
                        }
                    });
                }
                for b in &mut self.blocks {
                    b.retain(|&i| live[i as usize]);
                }
            }
            debug_assert!(self.uses_well_formed(&order));
        }
    }

    /// Simplifies the CFG to a fixed point:
    /// 1. Collapses degenerate `CondBr(_, t, t)` and `Switch` cases matching the default target.
    /// 2. Bypasses empty trampoline blocks (`[Br(t)]`) whenever `t`'s phis allow it.
    /// 3. Merges single-predecessor blocks `a -> b` along unconditional branches.
    fn simplify_cfg(&mut self) -> bool {
        let mut ever_changed = false;
        let mut phi_repl: Vec<Option<Value>> = Vec::new();
        loop {
            let mut changed = false;
            let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();

            // Step 0: simplify degenerate terminators (`CondBr(_, t, t)`, `Switch` cases -> default).
            for &b_u32 in &order {
                let b = b_u32 as usize;
                let Some(&term_i) = self.blocks[b].last() else { continue };
                let b_blk = Block(b_u32);
                match &mut self.insts[term_i as usize].op {
                    Op::CondBr(_, t, f) if *t == *f => {
                        let target = *t;
                        self.insts[term_i as usize].op = Op::Br(target);
                        self.drop_phi_edges(target.0 as usize, b_blk, 1);
                        changed = true;
                    }
                    Op::Switch(_, def, cases) => {
                        let def = *def;
                        let before = cases.len();
                        cases.retain(|&(_, target)| target != def);
                        let dropped = before - cases.len();
                        let became_empty = cases.is_empty();
                        if dropped > 0 {
                            self.drop_phi_edges(def.0 as usize, b_blk, dropped);
                            changed = true;
                        }
                        if became_empty {
                            self.insts[term_i as usize].op = Op::Br(def);
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }

            // Step 1: bypass empty trampoline blocks (`[Br(t)]`, `b != 0`, `t != b`).
            // `preds[x]` is maintained as a superset of live blocks whose terminator targets `x`.
            let mut preds: Vec<Vec<usize>> = vec![Vec::new(); self.blocks.len()];
            for &b_u32 in &order {
                let b = b_u32 as usize;
                for s in self.successors(b) {
                    let s = s.0 as usize;
                    if !preds[s].contains(&b) {
                        preds[s].push(b);
                    }
                }
            }
            for &b_u32 in &order {
                let b = b_u32 as usize;
                if b == 0 || self.blocks[b].len() != 1 {
                    continue;
                }
                let Op::Br(t_blk) = self.insts[self.blocks[b][0] as usize].op else { continue };
                let t = t_blk.0 as usize;
                if t == b || self.blocks[t].is_empty() {
                    continue;
                }
                let b_blk = Block(b_u32);
                let cand_preds = preds[b].clone();
                let target_has_phi = self.blocks[t]
                    .first()
                    .is_some_and(|&i| matches!(self.insts[i as usize].op, Op::Phi(_)));
                if target_has_phi {
                    // Bypassing a multi-edge trampoline into a phi block would duplicate phi
                    // operands for every phi in `t`; only bypass when `b` has a single incoming edge.
                    let total_edges: usize = cand_preds
                        .iter()
                        .filter(|&&p| p != b && !self.blocks[p].is_empty())
                        .map(|&p| self.count_term_edges(p, b_blk))
                        .sum();
                    if total_edges > 1 {
                        continue;
                    }
                }
                let mut all_redirected = true;
                let mut any_redirected = false;
                for p in cand_preds {
                    if p == b || self.blocks[p].is_empty() {
                        continue;
                    }
                    let edges = self.count_term_edges(p, b_blk);
                    if edges == 0 {
                        continue;
                    }
                    let p_blk = Block(p as u32);
                    if !self.phi_redirect_safe(t, b_blk, p_blk) {
                        all_redirected = false;
                        continue;
                    }
                    self.rewrite_term_target(p, b_blk, t_blk);
                    self.add_redirected_phi_edges(t, b_blk, p_blk, edges);
                    if !preds[t].contains(&p) {
                        preds[t].push(p);
                    }
                    any_redirected = true;
                    changed = true;
                }
                if any_redirected && all_redirected {
                    self.blocks[b].clear();
                    self.remove_all_phi_edges(t, b_blk);
                }
            }

            // Step 2: merge single-predecessor linear blocks (`a -> b`).
            let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
            let mut pred_edges = vec![0u32; self.blocks.len()];
            for &b_u32 in &order {
                for s in self.successors(b_u32 as usize) {
                    pred_edges[s.0 as usize] += 1;
                }
            }
            let mut any_phi_repl = false;
            for &a_u32 in &order {
                let a = a_u32 as usize;
                if self.blocks[a].is_empty() {
                    continue;
                }
                let a_blk = Block(a_u32);
                while let Some(&term_i) = self.blocks[a].last() {
                    let Op::Br(b_blk) = self.insts[term_i as usize].op else { break };
                    let b = b_blk.0 as usize;
                    if b == 0 || b == a || self.blocks[b].is_empty() || pred_edges[b] != 1 {
                        break;
                    }
                    // Verify every leading phi in `b` has an incoming value from `a` before mutating.
                    let mut phis_ok = true;
                    for &inst in &self.blocks[b] {
                        let Op::Phi(inc) = &self.insts[inst as usize].op else { break };
                        if !inc.iter().any(|&(_, bb)| bb == a_blk) {
                            phis_ok = false;
                            break;
                        }
                    }
                    if !phis_ok {
                        break;
                    }
                    // Pop `Br(b)` from `a` and move `b`'s non-phi instructions into `a`.
                    self.blocks[a].pop();
                    let b_insts = std::mem::take(&mut self.blocks[b]);
                    for inst in b_insts {
                        if let Op::Phi(inc) = &self.insts[inst as usize].op {
                            let v = inc.iter().find(|&&(_, bb)| bb == a_blk).unwrap().0;
                            if phi_repl.is_empty() {
                                phi_repl.resize(self.insts.len(), None);
                            }
                            phi_repl[inst as usize] = Some(v);
                            any_phi_repl = true;
                        } else {
                            self.def_block[inst as usize] = a_blk;
                            self.blocks[a].push(inst);
                        }
                    }
                    // Update successor phis from `b` to `a`.
                    let mut seen_s = Vec::new();
                    for s_blk in self.successors(a) {
                        if seen_s.contains(&s_blk) {
                            continue;
                        }
                        seen_s.push(s_blk);
                        let s = s_blk.0 as usize;
                        for k in 0..self.blocks[s].len() {
                            let i = self.blocks[s][k] as usize;
                            let Op::Phi(inc) = &mut self.insts[i].op else { break };
                            for (_, bb) in inc.iter_mut() {
                                if *bb == b_blk {
                                    *bb = a_blk;
                                }
                            }
                        }
                    }
                    pred_edges[b] = 0;
                    changed = true;
                }
            }
            if any_phi_repl {
                let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
                for &b_u32 in &order {
                    let b = b_u32 as usize;
                    for k in 0..self.blocks[b].len() {
                        let i = self.blocks[b][k] as usize;
                        self.insts[i].op.map_operands(|v| resolve(&mut phi_repl, v));
                    }
                }
                phi_repl.fill(None);
            }

            if !changed {
                break;
            }
            ever_changed = true;
        }
        ever_changed
    }

    fn count_term_edges(&self, b: usize, target: Block) -> usize {
        let Some(&term_i) = self.blocks[b].last() else { return 0 };
        match &self.insts[term_i as usize].op {
            Op::Br(t) => usize::from(*t == target),
            Op::CondBr(_, t, f) => usize::from(*t == target) + usize::from(*f == target),
            Op::Switch(_, d, cases) => usize::from(*d == target) + cases.iter().filter(|&&(_, c)| c == target).count(),
            _ => 0,
        }
    }

    fn rewrite_term_target(&mut self, b: usize, from: Block, to: Block) {
        let Some(&term_i) = self.blocks[b].last() else { return };
        match &mut self.insts[term_i as usize].op {
            Op::Br(t) => {
                if *t == from {
                    *t = to;
                }
            }
            Op::CondBr(_, t, f) => {
                if *t == from {
                    *t = to;
                }
                if *f == from {
                    *f = to;
                }
            }
            Op::Switch(_, d, cases) => {
                if *d == from {
                    *d = to;
                }
                for (_, c) in cases.iter_mut() {
                    if *c == from {
                        *c = to;
                    }
                }
            }
            _ => {}
        }
    }

    fn phi_redirect_safe(&self, target: usize, via: Block, pred: Block) -> bool {
        for &i in &self.blocks[target] {
            let Op::Phi(inc) = &self.insts[i as usize].op else { break };
            let Some(&(v_via, _)) = inc.iter().find(|&&(_, bb)| bb == via) else {
                return false;
            };
            if inc.iter().any(|&(v_p, bb)| bb == pred && v_p != v_via) {
                return false;
            }
        }
        true
    }

    fn add_redirected_phi_edges(&mut self, target: usize, via: Block, pred: Block, count: usize) {
        for k in 0..self.blocks[target].len() {
            let i = self.blocks[target][k] as usize;
            let Op::Phi(inc) = &mut self.insts[i].op else { break };
            let v_via = inc.iter().find(|&&(_, bb)| bb == via).unwrap().0;
            for _ in 0..count {
                inc.push((v_via, pred));
            }
        }
    }

    fn remove_all_phi_edges(&mut self, target: usize, from: Block) {
        for k in 0..self.blocks[target].len() {
            let i = self.blocks[target][k] as usize;
            let Op::Phi(inc) = &mut self.insts[i].op else { break };
            inc.retain(|&(_, bb)| bb != from);
        }
    }

    fn drop_phi_edges(&mut self, target: usize, from: Block, count: usize) {
        if count == 0 {
            return;
        }
        for k in 0..self.blocks[target].len() {
            let i = self.blocks[target][k] as usize;
            let Op::Phi(inc) = &mut self.insts[i].op else { break };
            let mut rem = count;
            inc.retain(|&(_, bb)| {
                if rem > 0 && bb == from {
                    rem -= 1;
                    false
                } else {
                    true
                }
            });
        }
    }
}

/// The value `v` stands for after replacements (path-compressing the chain).
fn resolve(repl: &mut [Option<Value>], v: Value) -> Value {
    let mut r = v;
    while let Value::Inst(i) = r {
        match repl[i as usize] {
            Some(n) => r = n,
            None => break,
        }
    }
    let mut c = v;
    while let Value::Inst(i) = c {
        match repl[i as usize] {
            Some(n) if n != r => {
                repl[i as usize] = Some(r);
                c = n;
            }
            _ => break,
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_calls(m: &Module, target_fn: u32) -> usize {
        m.blocks
            .iter()
            .flat_map(|b| b.iter())
            .filter(|&&i| matches!(m.insts[i as usize].op, Op::Call(f, _) if f == target_fn))
            .count()
    }

    #[test]
    fn block_cse_invalidates_wave_and_quad_calls_across_discard() {
        let build = |op_name: &str, with_discard: bool| -> (Module, u32) {
            let mut m = Module::new();
            m.define_entry("main");
            let b = m.entry_block();
            m.set_block(b);

            let void = m.void();
            let i1 = m.i1();
            let i32t = m.i32();

            let wave_ty = m.func_ty(i32t, &[i32t, i1]);
            let wave_fn = m.declare(op_name, wave_ty, FnAttrs::NoUnwind);
            let discard_ty = m.func_ty(void, &[i32t, i1]);
            let discard_fn = m.declare("dx.op.discard", discard_ty, FnAttrs::NoUnwind);
            let sink_ty = m.func_ty(void, &[i32t, i32t]);
            let sink_fn = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

            let c_op = m.i32c(116);
            let c_true = m.i1c(true);
            let c_discard_op = m.i32c(82);

            let first = m.call(wave_fn, &[c_op, c_true]);
            if with_discard {
                m.call(discard_fn, &[c_discard_op, c_true]);
            }
            let second = m.call(wave_fn, &[c_op, c_true]);
            m.call(sink_fn, &[first, second]);
            m.ret();
            m.optimize();
            (m, wave_fn)
        };

        // Without `dx.op.discard`, identical wave/quad calls in the same block still CSE to 1.
        let (m_no_discard, f_no_discard) = build("dx.op.waveActiveBallot", false);
        assert_eq!(count_calls(&m_no_discard, f_no_discard), 1);

        // Across `dx.op.discard`, active lanes change so both calls must be preserved.
        for op_name in ["dx.op.waveActiveBallot", "dx.op.quadReadLaneAt"] {
            let (m_with_discard, f_with_discard) = build(op_name, true);
            assert_eq!(
                count_calls(&m_with_discard, f_with_discard),
                2,
                "{op_name} must not be CSE-merged across dx.op.discard"
            );
        }
    }

    fn reachable_blocks(m: &Module) -> usize {
        m.block_layout().0.len()
    }

    #[test]
    fn simplify_cfg_collapses_trampoline_chain_and_merges_linear_blocks() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let t1 = m.new_block();
        let t2 = m.new_block();
        let mid = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let c7 = m.i32c(7);
        m.br(t1);

        m.set_block(t1);
        m.br(t2);

        m.set_block(t2);
        m.br(mid);

        m.set_block(mid);
        let p = m.phi(i32t);
        m.add_incoming(p, c7, t2);
        let c3 = m.i32c(3);
        let sum = m.bin(BinOp::Add, p, c3);
        m.br(exit);

        m.set_block(exit);
        m.call(sink, &[sum]);
        m.ret();

        m.optimize();
        assert_eq!(reachable_blocks(&m), 1, "entire linear trampoline chain should collapse into entry");
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
        assert_eq!(count_calls(&m, sink), 1);
    }

    #[test]
    fn simplify_cfg_preserves_phi_conflict_trampoline() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let tramp = m.new_block();
        let join = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let lane = m.call(src, &[]);
        let z = m.i32c(0);
        let cond = m.cmp(Pred::Eq, lane, z);
        let v1 = m.i32c(1);
        let v2 = m.i32c(2);
        m.cond_br(cond, tramp, join);

        m.set_block(tramp);
        m.br(join);

        m.set_block(join);
        let p = m.phi(i32t);
        m.add_incoming(p, v1, entry);
        m.add_incoming(p, v2, tramp);
        m.call(sink, &[p]);
        m.ret();

        m.optimize();
        // `tramp` cannot be bypassed because `entry` already feeds a different value into `join`'s phi.
        assert_eq!(reachable_blocks(&m), 3);
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
    }

    #[test]
    fn simplify_cfg_collapses_same_target_cond_br_and_switch() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let mid = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let lane = m.call(src, &[]);
        let z = m.i32c(0);
        let c1 = m.i32c(1);
        let cond = m.cmp(Pred::Eq, lane, z);
        let v = m.i32c(42);
        m.cond_br(cond, mid, mid);

        m.set_block(mid);
        let p = m.phi(i32t);
        m.add_incoming(p, v, entry);
        m.add_incoming(p, v, entry);
        m.switch(lane, exit, &[(z, exit), (c1, exit)]);

        m.set_block(exit);
        let p2 = m.phi(i32t);
        m.add_incoming(p2, p, mid);
        m.add_incoming(p2, p, mid);
        m.add_incoming(p2, p, mid);
        m.call(sink, &[p2]);
        m.ret();

        m.optimize();
        assert_eq!(reachable_blocks(&m), 1, "degenerate CondBr and Switch should collapse and merge");
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
        assert_eq!(count_calls(&m, sink), 1);
    }

    #[test]
    fn simplify_cfg_merges_loop_latch_and_rewrites_header_phi() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let header = m.new_block();
        let latch = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let z = m.i32c(0);
        let one = m.i32c(1);
        let limit = m.i32c(10);
        m.br(header);

        m.set_block(header);
        let iv = m.phi(i32t);
        m.add_incoming(iv, z, entry);
        m.br(latch);

        m.set_block(latch);
        let next = m.bin(BinOp::Add, iv, one);
        m.add_incoming(iv, next, latch);
        let cont = m.cmp(Pred::ULt, next, limit);
        m.cond_br(cont, header, exit);

        m.set_block(exit);
        m.call(sink, &[next]);
        m.ret();

        m.optimize();
        // `latch` merges into `header`, leaving `entry`, `header` (self-loop), `exit`.
        assert_eq!(reachable_blocks(&m), 3);
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
        // Header phi's backedge predecessor must now be `header` itself.
        let Value::Inst(iv_i) = iv else { unreachable!() };
        let Op::Phi(inc) = &m.insts[iv_i as usize].op else { panic!("expected phi") };
        assert!(inc.iter().any(|&(_, b)| b == header), "header phi backedge should point to header after latch merge");
        assert!(!inc.iter().any(|&(_, b)| b == latch), "merged latch must not remain in header phi");
    }

    #[test]
    fn simplify_cfg_does_not_inflate_multi_pred_phi_trampoline() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let left = m.new_block();
        let right = m.new_block();
        let other = m.new_block();
        let tramp = m.new_block();
        let join = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let lane = m.call(src, &[]);
        let c0 = m.i32c(0);
        let c1 = m.i32c(1);
        let v1 = m.i32c(11);
        let v2 = m.i32c(22);
        m.switch(lane, other, &[(c0, left), (c1, right)]);

        m.set_block(left);
        m.call(sink, &[c0]);
        m.br(tramp);

        m.set_block(right);
        m.call(sink, &[c1]);
        m.br(tramp);

        m.set_block(other);
        m.br(join);

        m.set_block(tramp);
        m.br(join);

        m.set_block(join);
        let p = m.phi(i32t);
        m.add_incoming(p, v1, tramp);
        m.add_incoming(p, v2, other);
        m.call(sink, &[p]);
        m.ret();

        m.optimize();
        // `other` (1 pred -> phi) IS bypassed into `entry -> join`, while `tramp` (2 preds -> phi) stays
        // so `join`'s phi stays at 2 entries instead of growing to 3.
        let Value::Inst(p_i) = p else { unreachable!() };
        let Op::Phi(inc) = &m.insts[p_i as usize].op else { panic!("expected phi") };
        assert_eq!(inc.len(), 2);
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
    }

    #[test]
    fn simplify_op_folds_bitmasks_double_xor_and_all_ones_or() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        m.set_block(entry);

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        let x = m.call(src, &[]);
        let c0 = m.i32c(0);
        let c2 = m.i32c(2);
        let c24 = m.i32c(24);
        let all = m.i32c(u32::MAX);
        let mff = m.i32c(0xFF);
        let mhi = m.i32c(0xFFFF_0000);
        let mlo = m.i32c(0x0000_FFFF);
        let align4 = m.i32c(!3u32);
        let low2 = m.i32c(3);
        let k = m.i32c(0xA5A5_5A5A);

        // 1. All-ones `or` absorption + `ashr 0, x -> 0`
        let or_all = m.bin(BinOp::Or, x, all);
        let ashr_z = m.bin(BinOp::AShr, c0, x);
        // 2. Double `xor` cancellation
        let x1 = m.bin(BinOp::Xor, x, k);
        let x2 = m.bin(BinOp::Xor, x1, k);
        // 3. Redundant and disjoint masks on `lshr`
        let shr = m.bin(BinOp::LShr, x, c24);
        let shr_keep = m.bin(BinOp::And, shr, mff);
        let shr_zero = m.bin(BinOp::And, shr, mhi);
        // 4. Redundant and disjoint masks on `shl`
        let shl = m.bin(BinOp::Shl, x, c2);
        let shl_keep = m.bin(BinOp::And, shl, align4);
        let shl_zero = m.bin(BinOp::And, shl, low2);
        // 5. Nested `and` / `or` masks
        let a1 = m.bin(BinOp::And, x, mff);
        let a_keep = m.bin(BinOp::And, a1, mlo);
        let a_zero = m.bin(BinOp::And, a1, mhi);
        let o1 = m.bin(BinOp::Or, x, mlo);
        let o_keep = m.bin(BinOp::Or, o1, mff);
        let o_all = m.bin(BinOp::Or, o1, mhi);
        // 6. Mask on `zext(i1)`
        let cond = m.cmp(Pred::Eq, x, c0);
        let z = m.cast(CastOp::ZExt, cond, i32t);
        let z_keep = m.bin(BinOp::And, z, mff);
        let z_zero = m.bin(BinOp::And, z, mhi);

        for v in [or_all, ashr_z, x2, shr_keep, shr_zero, shl_keep, shl_zero, a_keep, a_zero, o_keep, o_all, z_keep, z_zero] {
            m.call(sink, &[v]);
        }
        m.ret();
        m.optimize();

        let args: Vec<Value> = m.blocks[entry.0 as usize]
            .iter()
            .filter_map(|&i| match &m.insts[i as usize].op {
                Op::Call(f, a) if *f == sink => Some(a[0]),
                _ => None,
            })
            .collect();
        assert_eq!(args.len(), 13);
        assert_eq!(m.const_u64(args[0]), Some(u32::MAX as u64), "or(x, ~0) -> ~0");
        assert_eq!(m.const_u64(args[1]), Some(0), "ashr(0, x) -> 0");
        assert_eq!(args[2], x, "xor(xor(x, k), k) -> x");
        assert_eq!(args[3], shr, "and(lshr(x, 24), 0xFF) -> lshr(x, 24)");
        assert_eq!(m.const_u64(args[4]), Some(0), "and(lshr(x, 24), 0xFFFF0000) -> 0");
        assert_eq!(args[5], shl, "and(shl(x, 2), !3) -> shl(x, 2)");
        assert_eq!(m.const_u64(args[6]), Some(0), "and(shl(x, 2), 3) -> 0");
        assert_eq!(args[7], a1, "and(and(x, 0xFF), 0xFFFF) -> and(x, 0xFF)");
        assert_eq!(m.const_u64(args[8]), Some(0), "and(and(x, 0xFF), 0xFFFF0000) -> 0");
        assert_eq!(args[9], o1, "or(or(x, 0xFFFF), 0xFF) -> or(x, 0xFFFF)");
        assert_eq!(m.const_u64(args[10]), Some(u32::MAX as u64), "or(or(x, 0xFFFF), 0xFFFF0000) -> ~0");
        assert_eq!(args[11], z, "and(zext(i1), 0xFF) -> zext(i1)");
        assert_eq!(m.const_u64(args[12]), Some(0), "and(zext(i1), 0xFFFF0000) -> 0");
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
    }
}

#[cfg(test)]
mod known_bits_tests {
    use super::*;
    use crate::gpu::dxil::{
        container::Stage,
        ops::{Binary, BitCount},
        shader::Shader,
        validate,
    };
    use crate::gpu::recomp::runner;

    #[test]
    fn known_nonzero_bits_folds_compound_masks_and_shifts() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let i32t = m.i32();
        let void = m.void();
        let src_ty = m.func_ty(i32t, &[]);
        let sink_ty = m.func_ty(void, &[i32t]);
        let bin_ty = m.func_ty(i32t, &[i32t, i32t, i32t]);
        let un_ty = m.func_ty(i32t, &[i32t, i32t]);
        let src = m.declare("src", src_ty, FnAttrs::NoUnwind);
        let sink = m.declare("sink", sink_ty, FnAttrs::NoUnwind);
        let umin_fn = m.declare("dx.op.binary.i32", bin_ty, FnAttrs::ReadNone);
        let cnt_fn = m.declare("dx.op.unaryBits.i32", un_ty, FnAttrs::ReadNone);
        m.set_block(entry);

        let x = m.call(src, &[]);
        let y = m.call(src, &[]);
        let c0 = m.i32c(0);
        let c8 = m.i32c(8);
        let c16 = m.i32c(16);
        let c24 = m.i32c(24);
        let c31 = m.i32c(31);
        let c40 = m.i32c(40);
        let mff = m.i32c(0xFF);
        let mff00 = m.i32c(0xFF00);
        let mlo16 = m.i32c(0xFFFF);
        let mhi16 = m.i32c(0xFFFF_0000);
        let c256 = m.i32c(0x100);

        // 1. Multi-hop `or` + `shl` + `and` (packed u16): mask by 0xFFFF is redundant; lshr 16 -> 0.
        let b0 = m.bin(BinOp::And, x, mff);
        let b1 = m.bin(BinOp::And, y, mff);
        let b1_hi = m.bin(BinOp::Shl, b1, c8);
        let packed = m.bin(BinOp::Or, b0, b1_hi);
        let packed_keep = m.bin(BinOp::And, packed, mlo16);
        let packed_shr0 = m.bin(BinOp::LShr, packed, c16);
        let packed_ashr0 = m.bin(BinOp::AShr, packed, c16);

        // 2. Non-constant disjoint `and`: low-16 value & high-16 value -> 0.
        let hi_val = m.bin(BinOp::Shl, y, c16);
        let disjoint = m.bin(BinOp::And, packed, hi_val);

        // 3. `select` of bounded values: redundant mask kept, disjoint shift -> 0.
        let cond = m.cmp(Pred::Eq, x, c0);
        let sel = m.select(cond, b0, b1_hi);
        let sel_keep = m.bin(BinOp::And, sel, mlo16);
        let sel_shl0 = m.bin(BinOp::Shl, hi_val, c16);

        // 4. `add` of disjoint masks (`b0 + b1_hi`) and bounded `mul` (`b0 * b1` <= 65025 <= 0xFFFF).
        let add_disj = m.bin(BinOp::Add, b0, b1_hi);
        let add_keep = m.bin(BinOp::And, add_disj, mlo16);
        let mul_u16 = m.bin(BinOp::Mul, b0, b1);
        let mul_zero = m.bin(BinOp::And, mul_u16, mhi16);

        // 5. `dx.op.binary.i32(UMin)` and `dx.op.unaryBits.i32(Countbits)`.
        let clamped = m.call(umin_fn, &[c40, x, mff]);
        let clamped_keep = m.bin(BinOp::And, clamped, mff);
        // Subtle soundness case: umin(x & 0x100, y) can be 0xFF (255) OR 0x100 (256), so & 0xFF MUST NOT fold!
        let bit8 = m.bin(BinOp::And, x, c256);
        let clamped_256 = m.call(umin_fn, &[c40, bit8, y]);
        let clamped_256_mask = m.bin(BinOp::And, clamped_256, mff);
        let pop = m.call(cnt_fn, &[c31, x]);
        let pop_zero = m.bin(BinOp::And, pop, mff00);

        // 6. Negative-capable `ashr` MUST NOT fold to 0 when sign bit may be set.
        let sign_ashr = m.bin(BinOp::AShr, hi_val, c24);

        for v in [
            packed_keep,
            packed_shr0,
            packed_ashr0,
            disjoint,
            sel_keep,
            sel_shl0,
            add_keep,
            mul_zero,
            clamped_keep,
            clamped_256_mask,
            pop_zero,
            sign_ashr,
        ] {
            m.call(sink, &[v]);
        }
        m.ret();
        m.optimize();

        let args: Vec<Value> = m.blocks[entry.0 as usize]
            .iter()
            .filter_map(|&i| match &m.insts[i as usize].op {
                Op::Call(f, a) if *f == sink => Some(a[0]),
                _ => None,
            })
            .collect();
        assert_eq!(args.len(), 12);
        assert_eq!(args[0], packed, "and(or(x&0xFF, (y&0xFF)<<8), 0xFFFF) -> packed");
        assert_eq!(m.const_u64(args[1]), Some(0), "lshr(packed16, 16) -> 0");
        assert_eq!(m.const_u64(args[2]), Some(0), "ashr(packed16, 16) -> 0");
        assert_eq!(m.const_u64(args[3]), Some(0), "and(packed16, y<<16) -> 0");
        assert_eq!(args[4], sel, "and(select(c, x&0xFF, (y&0xFF)<<8), 0xFFFF) -> sel");
        assert_eq!(m.const_u64(args[5]), Some(0), "shl(y<<16, 16) -> 0");
        assert_eq!(args[6], add_disj, "and((x&0xFF)+((y&0xFF)<<8), 0xFFFF) -> add_disj");
        assert_eq!(m.const_u64(args[7]), Some(0), "and((x&0xFF)*(y&0xFF), 0xFFFF0000) -> 0");
        assert_eq!(args[8], clamped, "and(umin(x, 0xFF), 0xFF) -> clamped");
        assert_eq!(args[9], clamped_256_mask, "and(umin(x&0x100, y), 0xFF) must NOT fold away");
        assert_eq!(m.const_u64(args[10]), Some(0), "and(countbits(x), 0xFF00) -> 0");
        assert_eq!(args[11], sign_ashr, "ashr(y<<16, 24) has sign bit unknown and must stay");
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
    }

    #[test]
    fn known_nonzero_bits_gpu_parity() {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let res = s.raw_uav(1, 0);
        let mem = s.raw_uav(2, 0);
        let lane = s.flattened_thread_id_in_group();
        let c2 = s.m.i32c(2);
        let c8 = s.m.i32c(8);
        let c16 = s.m.i32c(16);
        let c256 = s.m.i32c(0x100);
        let mff = s.m.i32c(0xFF);
        let mlo16 = s.m.i32c(0xFFFF);
        let mhi16 = s.m.i32c(0xFFFF_0000);
        let off0 = s.m.bin(BinOp::Shl, lane, c2);
        let b64 = s.m.i32c(64);
        let idx1 = s.m.bin(BinOp::Add, lane, b64);
        let off1 = s.m.bin(BinOp::Shl, idx1, c2);
        let x = s.raw_load(mem, off0, 1)[0];
        let y = s.raw_load(mem, off1, 1)[0];

        let lo_x = s.m.bin(BinOp::And, x, mff);
        let lo_y = s.m.bin(BinOp::And, y, mff);
        let y_sh8 = s.m.bin(BinOp::Shl, lo_y, c8);
        let packed = s.m.bin(BinOp::Or, lo_x, y_sh8);
        let o0 = s.m.bin(BinOp::And, packed, mlo16);
        let o1 = s.m.bin(BinOp::LShr, packed, c16);
        let mul = s.m.bin(BinOp::Mul, lo_x, lo_y);
        let o2 = s.m.bin(BinOp::And, mul, mhi16);
        let bit8 = s.m.bin(BinOp::And, x, c256);
        let cl = s.binary(Binary::UMin, bit8, y);
        let o3 = s.m.bin(BinOp::And, cl, mff);
        let cnt = s.bit_count(BitCount::Countbits, x);
        let o4 = s.m.bin(BinOp::And, cnt, mlo16);

        for (k, v) in [o0, o1, o2, o3, o4].into_iter().enumerate() {
            let base = s.m.i32c(64 * k as u32);
            let slot = s.m.bin(BinOp::Add, base, lane);
            let off = s.m.bin(BinOp::Shl, slot, c2);
            s.raw_store(res, off, &[v]);
        }
        let dxil = s.finish([32, 1, 1], Some(32));
        validate::validate(&dxil).expect("known_nonzero_bits GPU shader validates");
        if !matches!(runner::wave32_supported(), Ok(true)) {
            runner::skip_test("no wave32");
            return;
        }
        let mut input = vec![0u32; 128];
        for l in 0..32usize {
            input[l] = (l as u32).wrapping_mul(0x0101_0107) ^ 0x8000_0100;
            input[64 + l] = if l % 2 == 0 { 0xFF } else { 0x105 + l as u32 };
        }
        let r = runner::run_with(&dxil, &runner::KernelState::default(), &input, &[], &[]).expect("GPU run");
        for l in 0..32usize {
            let (xv, yv) = (input[l], input[64 + l]);
            let want_packed = (xv & 0xFF) | ((yv & 0xFF) << 8);
            assert_eq!(r.raw[l], want_packed & 0xFFFF, "o0 lane {l}");
            assert_eq!(r.raw[64 + l], want_packed >> 16, "o1 lane {l}");
            assert_eq!(r.raw[128 + l], ((xv & 0xFF) * (yv & 0xFF)) & 0xFFFF_0000, "o2 lane {l}");
            assert_eq!(r.raw[192 + l], (xv & 0x100).min(yv) & 0xFF, "o3 lane {l}");
            assert_eq!(r.raw[256 + l], xv.count_ones() & 0xFFFF, "o4 lane {l}");
        }
    }
}
