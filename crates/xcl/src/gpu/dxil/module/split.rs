//! Partition the optimized compute entry at CFG edges. A dispatch cannot
//! jump back to an earlier dispatch, so a strongly connected component is
//! indivisible. Each crossing edge has its own compact state layout.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::{BinOp, Block, CastOp, Module, Op, TypeDef, Value};
use crate::gpu::dxil::shader::{Feature, RawUav, Shader};

#[derive(Debug)]
pub(crate) struct Edge {
    pub from: Block,
    pub to: Block,
    pub live: Vec<Value>,
}

pub(crate) struct Plan {
    pub blocks: Vec<Vec<Block>>,
    pub edges: Vec<Edge>,
}

#[derive(Clone)]
pub struct DispatchChain {
    pub pieces: Vec<Vec<u8>>,
    pub dwords_per_group: u32,
}

struct Scratch {
    uav: RawUav,
    base: Value,
    bytes_per_slot: u32,
}

impl Scratch {
    fn new(s: &mut Shader, stride: u32, threads: u32, lane: Value) -> Self {
        let uav = s.raw_uav(0, 2);
        let cb = s.cbv(0, 2, 16);
        let zero = s.m.i32c(0);
        let ty = s.m.i32();
        let [width, height, _, _] = s.cbuffer_load_legacy(cb, zero, ty);
        let [x, y, z] = [0, 1, 2].map(|axis| s.group_id(axis));
        let row = s.m.bin(BinOp::Mul, z, height);
        let row = s.m.bin(BinOp::Add, row, y);
        let first = s.m.bin(BinOp::Mul, row, width);
        let group = s.m.bin(BinOp::Add, first, x);
        // Give consecutive logical lanes consecutive words for a slot, including
        // the continuation tag in slot zero. A workgroup owns all its slots.
        let group_words = s.m.i32c(stride * threads);
        let first = s.m.bin(BinOp::Mul, group, group_words);
        let index = s.m.bin(BinOp::Add, first, lane);
        let four = s.m.i32c(4);
        Self { uav, base: s.m.bin(BinOp::Mul, index, four), bytes_per_slot: threads * 4 }
    }

    /// Every logical thread participates, including threads whose guest path
    /// skipped this piece. Exactly one thread owns each shared word.
    fn transfer_shared(&self, s: &mut Shader, lane: Value, lanes: u32, arrays: &[(Value, u32, u32)], restore: bool) {
        use crate::gpu::dxil::ops::barrier::GROUP_SYNC;
        if !restore { s.barrier(GROUP_SYNC); }
        for &(global, count, slot) in arrays {
            for page in 0..count.div_ceil(lanes) {
                let remaining = count - page * lanes;
                let mut join = None;
                if remaining < lanes {
                    let n = s.m.i32c(remaining);
                    let active = s.m.cmp(super::Pred::ULt, lane, n);
                    let body = s.m.new_block();
                    let next = s.m.new_block();
                    s.m.cond_br(active, body, next);
                    s.m.set_block(body);
                    join = Some(next);
                }
                let zero = s.m.i32c(0);
                let base = s.m.i32c(page * lanes);
                let index = s.m.bin(BinOp::Add, base, lane);
                let ptr = s.m.gep(global, &[zero, index]);
                let addr = self.at(&mut s.m, slot + page);
                if restore {
                    let v = s.raw_load(self.uav, addr, 1)[0];
                    s.m.store(ptr, v, 4);
                } else {
                    let v = s.m.load(ptr, 4);
                    s.raw_store(self.uav, addr, &[v]);
                }
                if let Some(next) = join {
                    s.m.br(next);
                    s.m.set_block(next);
                }
            }
        }
        if restore { s.barrier(GROUP_SYNC); }
    }

    fn at(&self, m: &mut Module, slot: u32) -> Value {
        let offset = m.i32c(slot * self.bytes_per_slot);
        m.bin(BinOp::Add, self.base, offset)
    }
}

impl Shader {
    /// Native wave32 uses the hardware lane in a single full wave. Split
    /// wave64 recreates the flat console thread ID on every dispatch, including
    /// when the hardware schedules the waves in a different order.
    pub(crate) fn split_compute(&self, threads: [u32; 3], split_tid: Option<Value>, limit: usize) -> Option<DispatchChain> {
        let lanes: u32 = threads.into_iter().try_fold(1u32, u32::checked_mul)?;
        let multi = split_tid.is_some() && lanes > 64;
        // The guest LDS (with the split form's exchange words at its top) is
        // saved and restored at each cut, and the plan cuts it only where the
        // whole group passes.
        let lds = self.m.globals.iter().position(|g| g.name == "xcl_lds");
        if self.m.globals.iter().any(|g| !g.constant && g.name != "xcl_lds"
            && !(split_tid.is_some() && matches!(g.name.as_str(), "xcl_xw" | "xcl_xs"))
            && !(multi && g.name == "xcl_xc"))
        {
            return None;
        }
        if self.m.blocks.iter().map(Vec::len).sum::<usize>() <= limit { return None; }
        let mut prefix = Vec::new();
        if let Some(tid) = split_tid {
            let end = self.m.blocks[0].iter().position(|&i| Value::Inst(i) == tid)?;
            prefix.extend_from_slice(&self.m.blocks[0][..=end]);
        }
        let mut consumed = HashSet::new();
        for &i in self.m.blocks.iter().flatten() {
            self.m.insts[i as usize].op.for_each_operand(|v| { consumed.insert(v); });
        }
        let root_is = |v, name| self.m.global_root(v).is_some_and(|g| self.m.globals[g].name == name);
        let spins: HashSet<_> = self.m.spin_regions.iter().flatten().copied().collect();
        let mut reads = HashSet::new();
        for (b, block) in self.m.blocks.iter().enumerate() {
            // In a single split console wave each inter-half exchange begins
            // with TGSM stores, one rendezvous, peer loads and a rendezvous.
            // A block ending anywhere inside that sequence cannot be a cut.
            let mut exchange = 0;
            for &i in block {
                if b == 0 && prefix.contains(&i) { continue; }
                match &self.m.insts[i as usize].op {
                    Op::Store(p, _, _) if lds.is_some() && root_is(*p, "xcl_lds") => {}
                    Op::Load(p, _) | Op::AtomicRmw(_, p, _) | Op::CmpXchg(p, _, _) if lds.is_some() && root_is(*p, "xcl_lds") => {
                        if !spins.contains(&Block(b as u32)) { reads.insert(Block(b as u32)); }
                    }
                    Op::Call(f, args) if lds.is_some() && self.m.funcs[*f as usize].name == "dx.op.barrier"
                        && self.m.const_u64(args[1]) == Some(crate::gpu::dxil::ops::barrier::GROUP_SYNC as u64) => {}
                    Op::AtomicRmw(super::RmwOp::Add, p, _) if multi && root_is(*p, "xcl_xc") => {}
                    Op::Load(p, _) | Op::Store(p, _, _) if multi && root_is(*p, "xcl_xs") => {}
                    Op::Call(f, args) if multi && spins.contains(&Block(b as u32))
                        && self.m.funcs[*f as usize].name == "dx.op.barrier"
                        && self.m.const_u64(args[1]) == Some(crate::gpu::dxil::ops::barrier::TGSM_FENCE as u64) => {}
                    Op::Alloca(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => return None,
                    Op::Store(..) if split_tid.is_some() && !multi && exchange <= 1 => exchange = 1,
                    Op::Load(p, _) if self.m.global_root(*p).is_some_and(|g| self.m.globals[g].constant) => {}
                    Op::Load(..) if split_tid.is_some() && !multi && matches!(exchange, 2 | 3) => exchange = 3,
                    Op::Load(..) | Op::Store(..) => return None,
                    Op::Call(f, args) if self.m.funcs[*f as usize].name == "dx.op.barrier"
                            && self.m.const_u64(args[1]) == Some(crate::gpu::dxil::ops::barrier::GROUP_SYNC as u64) => {
                        exchange = match exchange { 1 => 2, 3 => 0, _ => return None };
                    }
                    // Sub-dword stores use interlocked AND/OR. They return an
                    // old word that nobody observes; every write stays in place.
                    Op::Call(f, args) if self.m.funcs[*f as usize].name == "dx.op.atomicBinOp.i32"
                        && matches!(self.m.const_u64(args[2]), Some(1 | 2))
                        && !consumed.contains(&Value::Inst(i)) => {}
                    Op::Call(f, _) if ["dx.op.barrier", "dx.op.atomic"].iter().any(|p| self.m.funcs[*f as usize].name.starts_with(p)) => return None,
                    _ => {}
                }
            }
            if exchange != 0 { return None; }
        }
        // One console wave follows the guest's scalar control flow: its threads
        // diverge only in guarded per-lane accesses, which hold no barrier or
        // return, so any cut keeps its guest barriers paired. Several waves
        // can branch apart and return apart, so they are cut only where the
        // whole group passes. A loop that reads the LDS (a load or an atomic)
        // may poll it until another wave writes past a cut, so such a loop
        // keeps the program whole.
        let plan = self.m.dispatch_plan(limit, lds.filter(|_| multi).map(|_| &reads))?;
        let mut stride = 1;
        for edge in &plan.edges {
            let mut words = 1;
            for &v in &edge.live {
                words += self.m.transport_words(v)?;
            }
            stride = stride.max(words);
        }
        let mut shared = Vec::new();
        if multi {
            for (i, g) in self.m.globals.iter().enumerate().filter(|(_, g)| matches!(g.name.as_str(), "xcl_xc" | "xcl_xs")) {
                let TypeDef::Array(n, _) = self.m.types[g.ty.0 as usize] else { return None };
                shared.push((Value::Global(i as u32), u32::try_from(n).ok()?, stride));
                stride = stride.checked_add(u32::try_from(n).ok()?.div_ceil(lanes))?;
            }
        }
        if let Some(g) = lds {
            let TypeDef::Array(n, _) = self.m.types[self.m.globals[g].ty.0 as usize] else { return None };
            shared.push((Value::Global(g as u32), u32::try_from(n).ok()?, stride));
            stride = stride.checked_add(u32::try_from(n).ok()?.div_ceil(lanes))?;
        }
        let pieces = (0..plan.blocks.len()).map(|i| self.make_piece(&plan, i, stride, threads, &prefix, &shared)).collect::<Option<Vec<_>>>()?;
        Some(DispatchChain { pieces, dwords_per_group: stride.checked_mul(lanes)? })
    }

    fn make_piece(&self, plan: &Plan, stage: usize, stride: u32, threads: [u32; 3], prefix: &[u32], shared: &[(Value, u32, u32)]) -> Option<Vec<u8>> {
        let src = &self.m;
        let mut dst = self.dispatch_piece();
        let mut values = HashMap::new();
        for &id in prefix {
            let def = &src.insts[id as usize];
            let mut op = def.op.clone();
            op.map_operands(|v| *values.get(&v).unwrap_or(&v));
            values.insert(Value::Inst(id), dst.m.push(op, def.ty));
        }
        let lane = prefix.last().map(|&i| values[&Value::Inst(i)]).unwrap_or_else(|| dst.wave_lane_index());
        let lanes = threads.iter().product();
        let scratch = Scratch::new(&mut dst, stride, lanes, lane);
        if stage > 0 && !shared.is_empty() {
            // The split form's lane counter is in the LDS: every wave takes
            // its lanes before any restore overwrites it.
            if !prefix.is_empty() && shared.iter().any(|&(g, ..)| src.global_root(g).is_some_and(|g| src.globals[g].name == "xcl_lds")) {
                dst.barrier(crate::gpu::dxil::ops::barrier::GROUP_SYNC);
            }
            scratch.transfer_shared(&mut dst, lane, lanes, shared, true);
        }
        let dispatch_entry = dst.m.current_block();
        let local: HashMap<_, _> = plan.blocks[stage].iter().map(|&b| (b, dst.m.new_block())).collect();
        let mut hops = HashMap::new();
        let mut exits = HashMap::new();
        let mut cases = Vec::new();
        let mut paths = Vec::new();
        let mut incoming = std::collections::BTreeMap::<Value, Vec<(Vec<Value>, Block)>>::new();
        let mut statics = BTreeSet::new();
        let merge = dst.m.new_block();
        let done = dst.m.new_block();
        let selector = if stage != 0 { Some(dst.raw_load(scratch.uav, scratch.base, 1)[0]) } else { None };
        for (index, edge) in plan.edges.iter().enumerate() {
            let tag = dst.m.i32c(index as u32 + 1);
            if local.contains_key(&edge.from) {
                exits.insert((edge.from, edge.to), (dst.m.new_block(), tag));
            }
            if let Some(&target) = local.get(&edge.to) {
                let bridge = dst.m.new_block();
                dst.m.set_block(bridge);
                let mut slot = 1;
                for &v in &edge.live {
                    let n = src.transport_words(v)?;
                    if n == 0 {
                        statics.insert(v);
                        continue;
                    }
                    let words = (slot..slot + n).map(|word| {
                        let at = scratch.at(&mut dst.m, word);
                        dst.raw_load(scratch.uav, at, 1)[0]
                    }).collect();
                    incoming.entry(v).or_default().push((words, bridge));
                    slot += n;
                }
                dst.m.br(merge);
                let hop = dst.m.new_block();
                dst.m.set_block(hop);
                dst.m.br(target);
                cases.push((tag, bridge));
                paths.push((bridge, tag, hop));
                hops.insert((edge.from, edge.to), hop);
            }
        }
        dst.m.set_block(dispatch_entry);
        if let Some(sel) = selector {
            dst.m.switch(sel, done, &cases);
            dst.m.set_block(merge);
            let word = dst.m.i32();
            let mut unpack = Vec::new();
            for (original, inc) in incoming {
                let mut words = Vec::new();
                for k in 0..inc[0].0.len() {
                    let phi = dst.m.phi(word);
                    for &(pred, _, _) in &paths {
                        let v = inc.iter().find(|p| p.1 == pred).map(|p| p.0[k]).unwrap_or_else(|| dst.m.undef(word));
                        dst.m.add_incoming(phi, v, pred);
                    }
                    words.push(phi);
                }
                unpack.push((original, words));
            }
            for (original, words) in unpack {
                values.insert(original, src.unpack(original, &words, &mut dst.m));
            }
            for handle in statics {
                src.rebuild_static(handle, &mut dst.m, &mut values)?;
            }
            let routes: Vec<_> = paths.iter().map(|&(_, tag, hop)| (tag, hop)).collect();
            dst.m.switch(sel, done, &routes);
        } else {
            dst.m.br(local[&Block(0)]);
        }
        dst.m.set_block(done);
        let checkpoint = !shared.is_empty() && stage + 1 < plan.blocks.len();
        if checkpoint { scratch.transfer_shared(&mut dst, lane, lanes, shared, false); }
        dst.m.ret();

        // Assign all instruction identities before fixing operands. A loop's
        // phi can use an instruction in a block copied later.
        let mut copied = Vec::new();
        for &block in &plan.blocks[stage] {
            dst.m.set_block(local[&block]);
            for &i in &src.blocks[block.0 as usize] {
                if block == Block(0) && prefix.contains(&i) { continue; }
                let old = &src.insts[i as usize];
                if matches!(old.op, Op::Ret) && stage + 1 < plan.blocks.len() {
                    let zero = dst.m.i32c(0);
                    dst.raw_store(scratch.uav, scratch.base, &[zero]);
                    if checkpoint { dst.m.br(done); continue; }
                }
                let v = dst.m.push(old.op.clone(), old.ty);
                values.insert(Value::Inst(i), v);
                copied.push((block, inst(v)?));
            }
        }
        for (block, id) in copied {
            let op = &mut dst.m.insts[id].op;
            let mut complete = true;
            op.map_operands(|v| match v {
                Value::Inst(_) => values.get(&v).copied().unwrap_or_else(|| { complete = false; v }),
                _ => v,
            });
            if !complete { return None; }
            let target = |b: Block| local.get(&b).copied().or_else(|| exits.get(&(block, b)).map(|p| p.0));
            match op {
                Op::Phi(inc) => for (_, p) in inc {
                    *p = *local.get(p).or_else(|| hops.get(&(*p, block)))?;
                },
                Op::Br(b) => *b = target(*b)?,
                Op::CondBr(_, t, f) => { *t = target(*t)?; *f = target(*f)?; }
                Op::Switch(_, d, cases) => {
                    *d = target(*d)?;
                    for (_, t) in cases { *t = target(*t)?; }
                }
                _ => {}
            }
        }
        for edge in &plan.edges {
            let Some(&(b, tag)) = exits.get(&(edge.from, edge.to)) else { continue };
            dst.m.set_block(b);
            let mut slot = 1;
            for &v in &edge.live {
                let n = src.transport_words(v)?;
                if n != 0 {
                    let words = src.pack(v, *values.get(&v)?, &mut dst.m);
                    for (k, word) in words.into_iter().enumerate() {
                        let at = scratch.at(&mut dst.m, slot + k as u32);
                        dst.raw_store(scratch.uav, at, &[word]);
                    }
                    slot += n;
                }
            }
            dst.raw_store(scratch.uav, scratch.base, &[tag]);
            if checkpoint { dst.m.br(done); } else { dst.m.ret(); }
        }
        dst.infer_piece_features();
        // a refused piece leaves the whole program unsplit: say so
        let refused = |e: String| crate::xlog_warn!("compute split: piece {} of {} refused, the program stays whole: {}",
            stage + 1, plan.blocks.len(), e.lines().next().unwrap_or_default());
        let blob = dst.try_finish(threads, Some(32)).map_err(|e| refused(e.to_string())).ok()?;
        crate::gpu::dxil::validate::validate(&blob).map_err(refused).ok()?;
        Some(blob)
    }

    fn infer_piece_features(&mut self) {
        for f in [Feature::Doubles, Feature::Int64Ops, Feature::DoubleExtensions,
                  Feature::ResourceHeapIndexing, Feature::SamplerHeapIndexing] {
            self.features.remove(&f);
        }
        // typed UAV loads (`texture_load` without a mip): the whole program's flag stays only on
        // the pieces that have one, the validator wants flags to match usage
        let uav_load = |args: &[Value], m: &Module| args.get(2).is_some_and(|&mip| matches!(mip, Value::Const(c) if matches!(m.consts[c as usize], super::ConstDef::Undef(_))));
        if !self.m.blocks.iter().flatten().any(|&i| matches!(&self.m.insts[i as usize].op,
            Op::Call(f, args) if self.m.funcs[*f as usize].name.starts_with("dx.op.textureLoad") && uav_load(args, &self.m))) {
            self.features.remove(&Feature::TypedUavLoadAdditionalFormats);
        }
        for def in &self.m.insts {
            if let Op::Call(f, args) = &def.op {
                if self.m.funcs[*f as usize].name == "dx.op.createHandleFromHeap" {
                    self.features.insert(if self.m.const_u64(args[2]) == Some(0) {
                        Feature::ResourceHeapIndexing
                    } else { Feature::SamplerHeapIndexing });
                }
            }
        }
    }
}

fn inst(v: Value) -> Option<usize> {
    if let Value::Inst(i) = v { Some(i as usize) } else { None }
}

impl Module {
    fn global_root(&self, v: Value) -> Option<usize> {
        match v {
            Value::Global(g) => Some(g as usize),
            Value::Inst(i) => match &self.insts[i as usize].op {
                Op::Gep(p, _) => self.global_root(*p),
                _ => None,
            },
            _ => None,
        }
    }

    /// Only an index is transported. Resource handles and TGSM pointers are
    /// created inside the receiving dispatch, using their original properties.
    fn indexed(&self, v: Value) -> Option<Value> {
        let op = &self.insts[inst(v)?].op;
        if let Op::Gep(Value::Global(g), indices) = op {
            if matches!(self.globals[*g as usize].name.as_str(), "xcl_xs" | "xcl_xc" | "xcl_lds") && indices.len() == 2 && self.const_u64(indices[0]) == Some(0) {
                return Some(indices[1]);
            }
        }
        if let Op::Call(f, a) = op {
            if self.funcs[*f as usize].name == "dx.op.annotateHandle" && matches!(a[2], Value::Const(_)) {
                if let Op::Call(create, args) = &self.insts[inst(a[1])?].op {
                    if self.funcs[*create as usize].name == "dx.op.createHandleFromHeap" && args[2..].iter().all(|v| matches!(v, Value::Const(_))) {
                        return Some(args[1]);
                    }
                }
            }
        }
        None
    }

    fn transport_words(&self, v: Value) -> Option<u32> {
        let def = &self.insts[inst(v)?];
        match &self.types[def.ty.0 as usize] {
            TypeDef::Int(1..=32) | TypeDef::Float => Some(1),
            TypeDef::Int(64) => Some(2),
            TypeDef::Struct(..) if self.static_handle(v) => Some(0),
            _ if self.indexed(v).is_some() => Some(1),
            _ => None,
        }
    }

    fn static_handle(&self, v: Value) -> bool {
        let Value::Inst(i) = v else { return true };
        let Op::Call(f, args) = &self.insts[i as usize].op else { return false };
        match self.funcs[*f as usize].name.as_str() {
            "dx.op.createHandleFromBinding" => args.iter().all(|v| matches!(v, Value::Const(_))),
            "dx.op.annotateHandle" => args.iter().all(|&v| self.static_handle(v)),
            _ => false,
        }
    }

    fn rebuild_static(&self, v: Value, dst: &mut Module, map: &mut HashMap<Value, Value>) -> Option<Value> {
        let Value::Inst(id) = v else { return Some(v) };
        if let Some(&old) = map.get(&v) { return Some(old); }
        let Op::Call(f, args) = &self.insts[id as usize].op else { return None };
        let args: Vec<_> = args.iter().map(|&a| self.rebuild_static(a, dst, map)).collect::<Option<_>>()?;
        let new = dst.call(*f, &args);
        map.insert(v, new);
        Some(new)
    }

    fn pack(&self, original: Value, v: Value, dst: &mut Module) -> Vec<Value> {
        if let Some(index) = dst.indexed(v) { return vec![index]; }
        let ty = self.insts[inst(original).unwrap()].ty;
        let word = dst.i32();
        match self.types[ty.0 as usize] {
            TypeDef::Int(32) => vec![v],
            TypeDef::Int(64) => {
                let shift = dst.const_int(ty, 32);
                let high = dst.bin(BinOp::LShr, v, shift);
                vec![dst.cast(CastOp::Trunc, v, word), dst.cast(CastOp::Trunc, high, word)]
            }
            TypeDef::Float => vec![dst.bitcast(v, word)],
            _ => vec![dst.cast(CastOp::ZExt, v, word)],
        }
    }

    fn unpack(&self, original: Value, words: &[Value], dst: &mut Module) -> Value {
        if self.indexed(original).is_some() {
            match &self.insts[inst(original).unwrap()].op {
                Op::Gep(base, indices) => return dst.gep(*base, &[indices[0], words[0]]),
                Op::Call(f, annotate) => {
                    let Op::Call(create, args) = &self.insts[inst(annotate[1]).unwrap()].op else { unreachable!() };
                    let handle = dst.call(*create, &[args[0], words[0], args[2], args[3]]);
                    return dst.call(*f, &[annotate[0], handle, annotate[2]]);
                }
                _ => unreachable!(),
            }
        }
        let ty = self.insts[inst(original).unwrap()].ty;
        match self.types[ty.0 as usize] {
            TypeDef::Int(32) => words[0],
            TypeDef::Int(64) => {
                let low = dst.cast(CastOp::ZExt, words[0], ty);
                let high = dst.cast(CastOp::ZExt, words[1], ty);
                let shift = dst.const_int(ty, 32);
                let shifted = dst.bin(BinOp::Shl, high, shift);
                dst.bin(BinOp::Or, low, shifted)
            }
            TypeDef::Float => dst.bitcast(words[0], ty),
            _ => dst.cast(CastOp::Trunc, words[0], ty),
        }
    }
}

impl Module {
    /// Preserve symbol and type indices when copying instructions into a
    /// piece. The destination has its own blocks, instructions and metadata.
    pub(in crate::gpu::dxil) fn declarations(&self) -> Self {
        Self {
            types: self.types.clone(),
            type_ids: self.type_ids.clone(),
            consts: self.consts.clone(),
            const_ids: self.const_ids.clone(),
            funcs: self.funcs.clone(),
            func_ids: self.func_ids.clone(),
            globals: self.globals.clone(),
            ..Self::new()
        }
    }

    /// `uniform`: the program shares state across its thread group (guest
    /// LDS and barriers). A piece then starts only at a component that every
    /// thread enters exactly once, so the group meets at each cut and a guest
    /// barrier in a piece sees the same threads as in the whole program. A
    /// loop over a block of `uniform` keeps the whole program.
    pub(crate) fn dispatch_plan(&self, limit: usize, uniform: Option<&HashSet<Block>>) -> Option<Plan> {
        let (order, _) = self.block_layout();
        let count = |b: usize| self.blocks[b].len();
        if limit == 0 || order.iter().copied().map(count).sum::<usize>() <= limit {
            return None;
        }
        let mut predecessors = vec![Vec::new(); self.blocks.len()];
        for &b in &order {
            for target in self.successors(b) {
                predecessors[target.0 as usize].push(b);
            }
        }

        // In reverse postorder of the original graph, flooding the transpose
        // visits one component at a time in topological order (Kosaraju).
        let mut visited = vec![false; self.blocks.len()];
        let mut blocks = vec![Vec::new()];
        let mut stages = vec![usize::MAX; self.blocks.len()];
        let mut components = Vec::new();
        for &root in &order {
            if visited[root] {
                continue;
            }
            let mut component = Vec::new();
            let mut cost = 0;
            let mut stack = vec![root];
            visited[root] = true;
            while let Some(b) = stack.pop() {
                cost += count(b);
                component.push(b);
                for &p in &predecessors[b] {
                    if !visited[p] {
                        visited[p] = true;
                        stack.push(p);
                    }
                }
            }
            for &b in &component {
                stages[b] = components.len();
            }
            components.push((cost, component));
        }
        // A wait polling the peer's counter cannot complete if the peer's
        // signal was deferred to another dispatch. Pack these SCCs together.
        let mut through: Vec<usize> = (0..components.len()).collect();
        for region in &self.spin_regions {
            let mut span = region.iter().map(|b| stages[b.0 as usize]).filter(|&i| i != usize::MAX);
            if let Some(i) = span.next() {
                let (lo, hi) = span.fold((i, i), |(lo, hi), i| (lo.min(i), hi.max(i)));
                through[lo] = through[lo].max(hi);
            }
        }
        // Components are in topological order. A cut before component `i`
        // is crossed once by every thread when no earlier edge jumps past `i`
        // and no earlier block returns.
        let mut starts = vec![true; components.len()];
        if let Some(pinned) = uniform {
            let mut reach = 0;
            let mut returned = false;
            for (i, (_, component)) in components.iter().enumerate() {
                starts[i] = reach <= i && !returned;
                for &b in component {
                    let next = self.successors(b);
                    let looped = component.len() > 1 || next.contains(&Block(b as u32));
                    if looped && pinned.contains(&Block(b as u32)) { return None; }
                    returned |= next.is_empty();
                    reach = next.iter().map(|t| stages[t.0 as usize]).fold(reach, usize::max);
                }
            }
        }
        let mut i = 0;
        let mut used = 0;
        while i < components.len() {
            let first = i;
            let mut last = i;
            let mut cost = 0;
            while i <= last || (i < components.len() && !starts[i]) {
                last = last.max(through[i]);
                cost += components[i].0;
                i += 1;
            }
            if cost > limit { return None; }
            if used + cost > limit {
                blocks.push(Vec::new());
                used = 0;
            }
            used += cost;
            for (_, component) in &components[first..i] {
                for &b in component { stages[b] = blocks.len() - 1; }
            }
        }

        let mut crossing = BTreeSet::new();
        for &b in &order {
            let from = Block(b as u32);
            blocks[stages[b]].push(from);
            for to in self.successors(b) {
                if stages[b] != stages[to.0 as usize] {
                    debug_assert!(stages[b] < stages[to.0 as usize]);
                    crossing.insert((from, to));
                }
            }
        }
        let mut live_in = vec![HashSet::new(); self.blocks.len()];
        let mut phi_input = HashMap::<_, BTreeSet<_>>::new();
        let mut work = Vec::new();
        for &b in &order {
            for &inst in &self.blocks[b] {
                let op = &self.insts[inst as usize].op;
                // A phi reads each operand on its incoming edge, including
                // when the definition is its predecessor's last instruction.
                if let Op::Phi(inputs) = op {
                    for &(value, pred) in inputs {
                        if let Value::Inst(i) = value {
                            let edge = (pred, Block(b as u32));
                            if crossing.contains(&edge) {
                                phi_input.entry(edge).or_default().insert(value);
                            }
                            if self.def_block[i as usize] != pred {
                                work.push((pred.0 as usize, i));
                            }
                        }
                    }
                } else {
                    op.for_each_operand(|v| {
                        if let Value::Inst(i) = v {
                            if self.def_block[i as usize].0 as usize != b {
                                work.push((b, i));
                            }
                        }
                    });
                }
            }
        }
        while let Some((b, value)) = work.pop() {
            if live_in[b].insert(value) {
                for &p in &predecessors[b] {
                    if self.def_block[value as usize].0 as usize != p {
                        work.push((p, value));
                    }
                }
            }
        }
        let edges = crossing.into_iter().map(|(from, to)| {
            let mut live = phi_input.remove(&(from, to)).unwrap_or_default();
            live.extend(live_in[to.0 as usize].iter().copied().map(Value::Inst));
            Edge { from, to, live: live.into_iter().collect() }
        }).collect();
        Some(Plan { blocks, edges })
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::{
        dxil::{kernel::{Entry, KernelOptions}, lower::Lw, module::{BinOp, CastOp, Pred, Value}},
        recomp::{runner, CsInterface},
    };

    fn compare(threads: u32) {
        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        let geometry = [threads, 1, 1];
        let split = threads == 64;
        let cs = CsInterface { threads: geometry, ..Default::default() };
        let options = KernelOptions { wave64: split, split, mem_windows: 1, sampler_base: 0 };
        let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
        let (tid, _) = lw.setup_lanes_compute(geometry, false);
        let mem = lw.mems[0];
        let group = lw.s.group_id(0);
        let count = lw.c(threads);
        let base = lw.mul(group, count);
        let index = lw.add(base, tid);
        let offset = lw.shlc(index, 2);
        let original = lw.s.raw_load(mem, offset, 1)[0];
        // Two different dynamic dwords cross every cut as one i64. With
        // slot-major scratch the high word is a wave width past the low word.
        let i64_ty = lw.s.m.i64();
        let lo = lw.s.m.cast(CastOp::ZExt, original, i64_ty);
        let hi = lw.s.m.cast(CastOp::ZExt, tid, i64_ty);
        let shift = lw.s.m.const_int(i64_ty, 32);
        let hi = lw.s.m.bin(BinOp::Shl, hi, shift);
        let pair = lw.s.m.bin(BinOp::Or, lo, hi);
        let trips = lw.andc(tid, 3);
        let trips = lw.addc(trips, 1);
        let mut value = original;
        let i32 = lw.s.m.i32();
        let zero = lw.c(0);

        // Each loop both consumes the previous loop's result and reuses
        // the original per-lane input. All lanes eventually reconverge.
        for round in 1..=16 {
            let pred = lw.cur();
            let body = lw.s.m.new_block();
            let exit = lw.s.m.new_block();
            lw.s.m.br(body);
            lw.s.m.set_block(body);
            let counter = lw.s.m.phi(i32);
            let acc = lw.s.m.phi(i32);
            lw.s.m.add_incoming(counter, zero, pred);
            lw.s.m.add_incoming(acc, value, pred);
            let next = lw.addc(counter, 1);
            value = lw.add(acc, original);
            value = lw.addc(value, round);
            lw.s.m.add_incoming(counter, next, body);
            lw.s.m.add_incoming(acc, value, body);
            let again = lw.cmp(Pred::ULt, next, trips);
            lw.s.m.cond_br(again, body, exit);
            lw.s.m.set_block(exit);
            if split && round == 8 {
                let other_half = lw.exchange(lw.half);
                value = lw.add(value, other_half);
            }
        }
        let pair_lo = lw.s.m.cast(CastOp::Trunc, pair, i32);
        let high = lw.s.m.bin(BinOp::LShr, pair, shift);
        let pair_hi = lw.s.m.cast(CastOp::Trunc, high, i32);
        let both = lw.xor(pair_lo, pair_hi);
        value = lw.add(value, both);
        lw.s.raw_store(mem, offset, &[value]);
        lw.s.m.ret();
        lw.s.m.optimize();
        let chain = lw.s.split_compute(geometry, split.then_some(tid), 40);
        let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
        crate::gpu::dxil::validate::validate(&whole).expect("valid unsplit fixture");
        let chain = chain.expect("split the synthetic program");
        assert!(chain.pieces.len() >= 3);
        let initial: Vec<_> = (1..=threads * 3).collect();
        let expected: Vec<_> = initial.iter().enumerate().map(|(i, &v)| {
            let n = (i as u32 % 4) + 1;
            v + n * (16 * v + 136) + u32::from(split && i as u32 % threads < 32) + (v ^ (i as u32 % threads))
        }).collect();
        let reference = runner::dispatch(&whole, [3, 1, 1], &[], &initial).unwrap();
        assert_eq!(reference, expected);
        let result = runner::dispatch_split(&chain, [3, 1, 1], &initial).unwrap();
        assert_eq!(result, reference);
    }

    /// A typed UAV load in one piece only: the whole program declares its flag, the other pieces
    /// must not (the validator wants flags to match usage, and a refused piece leaves it whole).
    #[test]
    fn dispatch_chain_declares_typed_uav_loads_only_where_they_are() {
        use crate::gpu::dxil::ops::{CompType, TexKind};

        let geometry = [32, 1, 1];
        let options = KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 };
        let cs = CsInterface { threads: geometry, ..Default::default() };
        let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
        let (tid, _) = lw.setup_lanes_compute(geometry, false);
        let mem = lw.mems[0];
        let offset = lw.shlc(tid, 2);
        let first = lw.s.raw_load(mem, offset, 1)[0];
        let entry = lw.cur();
        let middle = lw.s.m.new_block();
        let last = lw.s.m.new_block();
        let width = lw.s.wave_lane_count();
        let taken = lw.cmp(Pred::ULt, tid, width);
        lw.s.m.cond_br(taken, middle, last);
        lw.s.m.set_block(middle);
        let slot = lw.s.m.i32c(9);
        let texture = lw.s.heap_texture(slot, TexKind::D2, true, CompType::F32, 4, false);
        lw.s.typed_uav_load_additional_formats();
        let texel = lw.s.texture_load(texture, None, &[tid, tid], &[]);
        let i32 = lw.s.m.i32();
        let bits = lw.s.m.cast(CastOp::BitCast, texel[1], i32);
        let loaded = lw.xor(first, bits);
        lw.s.m.br(last);
        lw.s.m.set_block(last);
        let value = lw.s.m.phi(i32);
        lw.s.m.add_incoming(value, loaded, middle);
        lw.s.m.add_incoming(value, first, entry);
        lw.s.raw_store(mem, offset, &[value]);
        lw.s.m.ret();
        lw.s.m.optimize();
        let limit = lw.s.m.blocks.iter().map(Vec::len).max().unwrap();
        let chain = lw.s.split_compute(geometry, None, limit).expect("every piece validates");
        assert!(chain.pieces.len() >= 2);
    }

    #[test]
    fn dispatch_chain_four_wave64s_preserves_partner_epochs() {
        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        let geometry = [16, 16, 1];
        let cs = CsInterface { threads: geometry, ..Default::default() };
        let options = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
        let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
        let (tid, _) = lw.setup_lanes_compute(geometry, false);
        let half = lw.lshrc(tid, 5);
        let group = lw.s.group_id(0);
        let group_base = lw.shlc(group, 8);
        let index = lw.add(group_base, tid);
        let addr = lw.shlc(index, 2);
        let mem = lw.mems[0];
        let initial = lw.s.raw_load(mem, addr, 1)[0];
        let acc = lw.tmp_var();
        lw.wr(acc, initial);
        let odd_wave = lw.andc(tid, 64);
        let odd_wave = lw.nz(odd_wave);
        // Only odd guest waves perform every exchange. A single common
        // exchange count for all guest waves would corrupt their results.
        for round in 1..=12 {
            let mut exchange = |lw: &mut Lw| {
                let payload = lw.addc(half, round);
                let peer = lw.exchange(payload);
                let old = lw.rd(acc);
                let sum = lw.add(old, peer);
                lw.wr(acc, sum);
                Ok(())
            };
            if round % 2 == 1 { lw.guarded(odd_wave, &mut exchange).unwrap(); }
            else { exchange(&mut lw).unwrap(); }
            let next = lw.block(Some(1));
            lw.br(next);
            lw.s.m.set_block(next);
        }
        let v = lw.rd(acc);
        lw.s.raw_store(mem, addr, &[v]);
        lw.s.m.ret();
        lw.s.m.optimize();
        let chain = lw.s.split_compute(geometry, Some(tid), 160).expect("split four guest waves");
        assert!(chain.pieces.len() >= 3);
        let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
        let input: Vec<u32> = (1..=512).collect();
        let expected: Vec<_> = input.iter().enumerate().map(|(i, v)| {
            let tid = i as u32 % 256;
            v + (1..=12).filter(|round| round % 2 == 0 || tid & 64 != 0)
                .map(|round| ((tid / 32) ^ 1) + round).sum::<u32>()
        }).collect();
        let groups = [2, 1, 1];
        let reference = runner::dispatch(&whole, groups, &[], &input).unwrap();
        assert_eq!(reference, expected);
        assert_eq!(runner::dispatch_split(&chain, groups, &input).unwrap(), reference);
    }

    /// Guest waves that returned still participate in the checkpoint rendezvous
    /// while other waves keep advancing their partner counters and slot banks.
    #[test]
    fn dispatch_chain_exited_waves_preserve_other_waves_3d() {
        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        println!("CHAIN_EXIT_GPU={}", runner::adapter().unwrap());
        let geometry = [16, 16, 1];
        let cs = CsInterface { threads: geometry, ..Default::default() };
        let options = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
        let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
        let (tid, _) = lw.setup_lanes_compute(geometry, false);
        let half = lw.lshrc(tid, 5);
        let wave = lw.lshrc(tid, 6);
        let [x, y, z] = [0, 1, 2].map(|axis| lw.s.group_id(axis));
        let z = lw.shlc(z, 1);
        let row = lw.add(z, y);
        let row = lw.shlc(row, 1);
        let group = lw.add(row, x);
        let base = lw.shlc(group, 8);
        let index = lw.add(base, tid);
        let addr = lw.shlc(index, 2);
        let mem = lw.mems[0];
        let initial = lw.s.raw_load(mem, addr, 1)[0];
        let acc = lw.tmp_var();
        lw.wr(acc, initial);
        let not_second_wave = lw.cmpc(Pred::Ne, wave, 1);
        for round in 1..=12 {
            if round == 1 || round == 7 {
                let exit = lw.block(Some(1));
                let live = lw.block(Some(1));
                let stops = lw.cmpc(Pred::Eq, wave, if round == 1 { 0 } else { 2 });
                lw.cond_br(stops, exit, live);
                lw.s.m.set_block(exit);
                let old = lw.rd(acc);
                let value = if round == 1 { lw.addc(old, 100_000) } else { old };
                lw.s.raw_store(mem, addr, &[value]);
                lw.s.m.ret();
                lw.s.m.set_block(live);
            }
            let mut exchange = |lw: &mut Lw| {
                let payload = lw.addc(half, round);
                let peer = lw.exchange(payload);
                let old = lw.rd(acc);
                let value = lw.add(old, peer);
                lw.wr(acc, value);
                Ok(())
            };
            if round % 2 == 1 { lw.guarded(not_second_wave, &mut exchange).unwrap(); }
            else { exchange(&mut lw).unwrap(); }
            let next = lw.block(Some(1));
            lw.br(next);
            lw.s.m.set_block(next);
        }
        let value = lw.rd(acc);
        lw.s.raw_store(mem, addr, &[value]);
        lw.s.m.ret();
        lw.s.m.optimize();
        let chain = lw.s.split_compute(geometry, Some(tid), 160).expect("split with exited guest waves");
        assert!(chain.pieces.len() >= 3);
        let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
        crate::gpu::dxil::validate::validate(&whole).expect("valid whole shader");
        let input: Vec<_> = (0..2064u32).map(|i| if i < 2048 { i + 17 } else { 0xdeadbeef }).collect();
        let expected: Vec<_> = input.iter().enumerate().map(|(i, &v)| {
            if i >= 2048 { return v; }
            let tid = i as u32 % 256;
            let peer = (tid / 32) ^ 1;
            match tid / 64 {
                0 => v + 100_000,
                1 => v + 6 * peer + 42,
                2 => v + 6 * peer + 21,
                _ => v + 12 * peer + 78,
            }
        }).collect();
        println!("CHAIN_EXIT_STAGE whole pieces={} scratch_dwords_per_group={}", chain.pieces.len(), chain.dwords_per_group);
        let reference = runner::dispatch(&whole, [2, 2, 2], &[], &input).unwrap();
        assert_eq!(reference, expected, "whole program CPU oracle");
        println!("CHAIN_EXIT_STAGE split");
        let result = runner::dispatch_split(&chain, [2, 2, 2], &input).unwrap();
        assert_eq!(result, expected, "split program CPU oracle");
        println!("CHAIN_EXIT checked_words=2048 guards=16 pieces={} scratch_dwords_per_group={}", chain.pieces.len(), chain.dwords_per_group);
    }

    #[test]
    fn dispatch_chain_wave32_live_loops() { compare(32); }

    /// A guest wave64 split across two hardware wave32s.
    #[test]
    fn dispatch_chain_wave64_live_loops_and_exchange() { compare(64); }
    /// Early exits and forward skips retain their own continuation tags; group
    /// indexing must keep every lane separate across all three dispatch axes.
    #[test]
    fn dispatch_chain_divergent_forward_edges_3d() {
        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        println!("CHAIN_GPU={}", runner::adapter().unwrap());
        for threads in [32, 64] {
            let geometry = [threads, 1, 1];
            let split = threads == 64;
            let cs = CsInterface { threads: geometry, ..Default::default() };
            let options = KernelOptions { wave64: split, split, mem_windows: 1, sampler_base: 0 };
            let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
            let (tid, _) = lw.setup_lanes_compute(geometry, false);
            let mem = lw.mems[0];
            let [x, y, z] = [0, 1, 2].map(|axis| lw.s.group_id(axis));
            let three = lw.c(3);
            let row = lw.mul(z, three);
            let row = lw.add(row, y);
            let row = lw.shlc(row, 1);
            let group = lw.add(row, x);
            let count = lw.c(threads);
            let base = lw.mul(group, count);
            let index = lw.add(base, tid);
            let offset = lw.shlc(index, 2);
            let original = lw.s.raw_load(mem, offset, 1)[0];
            let mode = lw.andc(tid, 3);
            let trips = lw.addc(mode, 1);
            let zero = lw.c(0);
            let one = lw.c(1);
            let i32 = lw.s.m.i32();
            let early = lw.s.m.new_block();
            let choose = lw.s.m.new_block();
            let first = lw.s.m.new_block();
            let middle = lw.s.m.new_block();
            let second = lw.s.m.new_block();
            let done = lw.s.m.new_block();
            let exits = lw.cmp(Pred::Eq, mode, zero);
            lw.s.m.cond_br(exits, early, choose);
            lw.s.m.set_block(early);
            let marker = lw.addc(index, 100_000);
            lw.s.raw_store(mem, offset, &[marker]);
            lw.s.m.ret();
            lw.s.m.set_block(choose);
            let skips = lw.cmp(Pred::Eq, mode, one);
            lw.s.m.cond_br(skips, middle, first);
            lw.s.m.set_block(first);
            let mut value = original;
            let mut middle_value = None;
            for round in 1..=16 {
                if round == 9 {
                    let previous = lw.cur();
                    lw.s.m.br(middle);
                    lw.s.m.set_block(middle);
                    let joined = lw.s.m.phi(i32);
                    lw.s.m.add_incoming(joined, original, choose);
                    lw.s.m.add_incoming(joined, value, previous);
                    value = joined;
                    middle_value = Some(joined);
                    let stops = lw.cmp(Pred::Eq, mode, three);
                    lw.s.m.cond_br(stops, done, second);
                    lw.s.m.set_block(second);
                }
                let pred = lw.cur();
                let body = lw.s.m.new_block();
                let exit = lw.s.m.new_block();
                lw.s.m.br(body);
                lw.s.m.set_block(body);
                let counter = lw.s.m.phi(i32);
                let acc = lw.s.m.phi(i32);
                lw.s.m.add_incoming(counter, zero, pred);
                lw.s.m.add_incoming(acc, value, pred);
                let next = lw.addc(counter, 1);
                value = lw.add(acc, original);
                value = lw.addc(value, round);
                lw.s.m.add_incoming(counter, next, body);
                lw.s.m.add_incoming(acc, value, body);
                let again = lw.cmp(Pred::ULt, next, trips);
                lw.s.m.cond_br(again, body, exit);
                lw.s.m.set_block(exit);
            }
            let last = lw.cur();
            lw.s.m.br(done);
            lw.s.m.set_block(done);
            let final_value = lw.s.m.phi(i32);
            lw.s.m.add_incoming(final_value, middle_value.unwrap(), middle);
            lw.s.m.add_incoming(final_value, value, last);
            lw.s.raw_store(mem, offset, &[final_value]);
            lw.s.m.ret();
            lw.s.m.optimize();
            let plan = lw.s.m.dispatch_plan(64, None).expect("partition branch fixture");
            let stage = |b| plan.blocks.iter().position(|blocks| blocks.contains(&b)).unwrap();
            assert!(plan.edges.iter().any(|edge| stage(edge.to) > stage(edge.from) + 1), "exercise a forward skip over an intermediate piece");
            let chain = lw.s.split_compute(geometry, split.then_some(tid), 64).expect("split branch fixture");
            assert!(chain.pieces.len() >= 3);
            let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
            crate::gpu::dxil::validate::validate(&whole).expect("valid unsplit fixture");
            let words = (threads * 12) as usize;
            let initial: Vec<_> = (0..words + 16).map(|i| if i < words { i as u32 + 17 } else { 0xdeadbeef }).collect();
            let expected: Vec<_> = initial.iter().enumerate().map(|(i, &v)| {
                if i >= words { return v; }
                match i as u32 % 4 {
                    0 => 100_000 + i as u32,
                    1 => v + 2 * (8 * v + 100),
                    2 => v + 3 * (16 * v + 136),
                    _ => v + 4 * (8 * v + 36),
                }
            }).collect();
            let reference = runner::dispatch(&whole, [2, 3, 2], &[], &initial).unwrap();
            assert_eq!(reference, expected, "unsplit {threads}");
            let result = runner::dispatch_split(&chain, [2, 3, 2], &initial).unwrap();
            assert_eq!(result, expected, "split {threads}");
            println!("CHAIN_BRANCH threads={threads} pieces={} scratch_words_per_group={} checked_words={words} guards=16", chain.pieces.len(), chain.dwords_per_group);
        }
    }

    /// A program with the guest LDS, as the translator declares it when the
    /// console code has a DS instruction.
    fn lds_lw(threads: u32, split: bool) -> Lw {
        let ds = crate::gpu::rdna2::encode("DS_WRITE_B32", "ENC_DS", &[("ADDR", 0), ("DATA0", 0)], None).unwrap();
        let ds = crate::gpu::rdna2::decode(&ds, 0).unwrap();
        let cs = CsInterface { threads: [threads, 1, 1], ..Default::default() };
        let options = KernelOptions { wave64: split, split, mem_windows: 1, sampler_base: 0 };
        let lw = Lw::new(std::slice::from_ref(&ds), &Default::default(), options, &Entry::Compute(cs));
        assert!(lw.lds.is_some());
        lw
    }

    /// One loop per round, run `(tid & 3) + 1` times by each lane: `v += trips * (x + round)`.
    fn lane_loops(lw: &mut Lw, rounds: std::ops::RangeInclusive<u32>, mut v: Value, x: Value, trips: Value) -> Value {
        let i32 = lw.s.m.i32();
        let zero = lw.c(0);
        for round in rounds {
            let pred = lw.cur();
            let body = lw.s.m.new_block();
            let exit = lw.s.m.new_block();
            lw.s.m.br(body);
            lw.s.m.set_block(body);
            let counter = lw.s.m.phi(i32);
            let acc = lw.s.m.phi(i32);
            lw.s.m.add_incoming(counter, zero, pred);
            lw.s.m.add_incoming(acc, v, pred);
            let next = lw.addc(counter, 1);
            v = lw.add(acc, x);
            v = lw.addc(v, round);
            lw.s.m.add_incoming(counter, next, body);
            lw.s.m.add_incoming(acc, v, body);
            let again = lw.cmp(Pred::ULt, next, trips);
            lw.s.m.cond_br(again, body, exit);
            lw.s.m.set_block(exit);
        }
        v
    }

    fn lds_read(lw: &mut Lw, index: Value) -> Value {
        let g = lw.lds.unwrap();
        let p = lw.s.gs_ptr(g, index);
        lw.s.m.load(p, 4)
    }

    fn lds_write(lw: &mut Lw, index: Value, v: Value) {
        let g = lw.lds.unwrap();
        let p = lw.s.gs_ptr(g, index);
        lw.s.m.store(p, v, 4);
    }

    fn guest_barrier(lw: &mut Lw) {
        lw.s.barrier(crate::gpu::dxil::ops::barrier::GROUP_SYNC);
    }

    /// Every thread enters each piece of `plan` once, at one block, and only
    /// the last piece returns: the group meets at every cut.
    fn uniform_cuts(plan: &super::Plan, m: &crate::gpu::dxil::module::Module) -> bool {
        let stage = |b| plan.blocks.iter().position(|blocks| blocks.contains(&b)).unwrap();
        let last = plan.blocks.len() - 1;
        let returns = plan.blocks[..last].iter().flatten().any(|&b| m.successors(b.0 as usize).is_empty());
        let skips = plan.edges.iter().any(|e| stage(e.to) != stage(e.from) + 1);
        let entries = (1..=last).any(|k| plan.edges.iter().filter(|e| stage(e.to) == k).map(|e| e.to).collect::<std::collections::BTreeSet<_>>().len() != 1);
        !returns && !skips && !entries
    }

    /// Negative control: without the group-uniform rule, some plan of `m` has
    /// a cut that not every thread crosses. With it, none does.
    fn cuts_only_where_the_group_meets(m: &crate::gpu::dxil::module::Module) {
        let limits = 40..=160;
        assert!(limits.clone().filter_map(|limit| m.dispatch_plan(limit, None)).any(|plan| !uniform_cuts(&plan, m)),
            "the fixture has cuts the group does not pass together");
        let plans: Vec<_> = limits.filter_map(|limit| m.dispatch_plan(limit, Some(&Default::default()))).collect();
        assert!(plans.iter().any(|plan| plan.blocks.len() >= 3));
        assert!(plans.iter().all(|plan| uniform_cuts(plan, m)), "a cut the group does not pass together");
    }

    /// The guest LDS crosses cuts: written and counted with an LDS atomic before
    /// a guest barrier, read by other threads (other halves and waves) after
    /// several cuts, written and read again across a second barrier. Lanes take
    /// different loop counts; with several waves, odd and even waves take
    /// different branches, where the group may not be cut.
    fn lds_chain(threads: u32, split: bool) {
        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        let geometry = [threads, 1, 1];
        let multi = threads > 64;
        let mut lw = lds_lw(threads, split);
        let (tid, _) = lw.setup_lanes_compute(geometry, false);
        let mem = lw.mems[0];
        let group = lw.s.group_id(0);
        let count = lw.c(threads);
        let base = lw.mul(group, count);
        let index = lw.add(base, tid);
        let offset = lw.shlc(index, 2);
        let x = lw.s.raw_load(mem, offset, 1)[0];
        // the halves' exchange keeps its state in the translator's SSA, which
        // knows only the entry block here
        let peer_half = split.then(|| {
            let half = lw.half;
            lw.exchange(half)
        });
        let mode = lw.andc(tid, 3);
        let trips = lw.addc(mode, 1);
        let three = lw.c(3);
        let a = lw.mul(x, three);
        let a = lw.addc(a, 1);
        lds_write(&mut lw, tid, a);
        let counter = lw.c(4000);
        let zero = lw.c(0);
        lds_write(&mut lw, counter, zero);
        guest_barrier(&mut lw);
        let g = lw.lds.unwrap();
        let p = lw.s.gs_ptr(g, counter);
        lw.s.m.atomic_rmw(crate::gpu::dxil::module::RmwOp::Add, p, x);
        guest_barrier(&mut lw);
        let mut v = lane_loops(&mut lw, 1..=8, x, x, trips);
        let other = lw.addc(tid, 33);
        let n = lw.c(threads);
        let other = lw.bin(BinOp::URem, other, n);
        let peer = lds_read(&mut lw, other);
        let sum = lds_read(&mut lw, counter);
        v = lw.add(v, peer);
        v = lw.add(v, sum);
        if multi {
            let odd = lw.andc(tid, 64);
            let odd = lw.nz(odd);
            let (xb, yb, join) = (lw.s.m.new_block(), lw.s.m.new_block(), lw.s.m.new_block());
            lw.s.m.cond_br(odd, xb, yb);
            lw.s.m.set_block(xb);
            let vx = lane_loops(&mut lw, 20..=21, v, x, trips);
            let xe = lw.cur();
            lw.s.m.br(join);
            lw.s.m.set_block(yb);
            let vy = lane_loops(&mut lw, 30..=31, v, x, trips);
            let ye = lw.cur();
            lw.s.m.br(join);
            lw.s.m.set_block(join);
            let i32 = lw.s.m.i32();
            v = lw.s.m.phi(i32);
            lw.s.m.add_incoming(v, vx, xe);
            lw.s.m.add_incoming(v, vy, ye);
        }
        if let Some(peer_half) = peer_half {
            v = lw.add(v, peer_half);
        }
        guest_barrier(&mut lw);
        lds_write(&mut lw, tid, v);
        guest_barrier(&mut lw);
        v = lane_loops(&mut lw, 9..=16, v, x, trips);
        let last = lw.c(threads - 1);
        let mirror = lw.sub(last, tid);
        let mirrored = lds_read(&mut lw, mirror);
        v = lw.add(v, mirrored);
        lw.s.raw_store(mem, offset, &[v]);
        lw.s.m.ret();
        lw.s.m.optimize();

        let limit = if multi { 80 } else { 56 };
        let chain = lw.s.split_compute(geometry, split.then_some(tid), limit).expect("split the LDS program");
        assert!(chain.pieces.len() >= 3, "{} pieces", chain.pieces.len());
        if multi {
            cuts_only_where_the_group_meets(&lw.s.m);
        }
        let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
        crate::gpu::dxil::validate::validate(&whole).expect("valid unsplit fixture");
        let groups = 3;
        let words = (threads * groups) as usize;
        let input: Vec<u32> = (0..words + 16).map(|i| if i < words { (i as u32).wrapping_mul(2654435761) >> 8 } else { 0xdeadbeef }).collect();
        let mut expected = input.clone();
        for g in 0..groups as usize {
            let n = threads as usize;
            let x = &input[g * n..(g + 1) * n];
            let loops = |t: usize, rounds: std::ops::RangeInclusive<u32>| rounds.map(|r| ((t as u32 & 3) + 1).wrapping_mul(x[t].wrapping_add(r))).fold(0u32, u32::wrapping_add);
            let sum = x.iter().fold(0u32, |a, &b| a.wrapping_add(b));
            let stored: Vec<u32> = (0..n).map(|t| {
                let mut v = x[t].wrapping_add(loops(t, 1..=8));
                v = v.wrapping_add(x[(t + 33) % n].wrapping_mul(3).wrapping_add(1)).wrapping_add(sum);
                if multi { v = v.wrapping_add(loops(t, if t & 64 != 0 { 20..=21 } else { 30..=31 })); }
                if split { v = v.wrapping_add((t as u32 >> 5 & 1) ^ 1); }
                v
            }).collect();
            for t in 0..n {
                expected[g * n + t] = stored[t].wrapping_add(loops(t, 9..=16)).wrapping_add(stored[n - 1 - t]);
            }
        }
        let reference = runner::dispatch(&whole, [groups, 1, 1], &[], &input).unwrap();
        assert_eq!(reference, expected, "whole program, {threads} threads");
        let result = runner::dispatch_split(&chain, [groups, 1, 1], &input).unwrap();
        assert_eq!(result, expected, "split program, {threads} threads");
        println!("CHAIN_LDS threads={threads} split={split} pieces={} scratch_dwords_per_group={} checked_words={words} guards=16", chain.pieces.len(), chain.dwords_per_group);
    }

    #[test]
    fn dispatch_chain_lds_wave32() { lds_chain(32, false); }

    #[test]
    fn dispatch_chain_lds_wave64_split() { lds_chain(64, true); }

    #[test]
    fn dispatch_chain_lds_two_wave64s() { lds_chain(128, true); }

    #[test]
    fn dispatch_chain_lds_three_wave64s() { lds_chain(192, true); }

    #[test]
    fn dispatch_chain_lds_four_wave64s() { lds_chain(256, true); }

    /// Guest wave 0 returns early and the other waves meet at guest barriers
    /// afterwards. A cut after the return would leave wave 0 waiting at the
    /// piece exit while the others wait at a guest barrier: the plan keeps
    /// everything from the return on in the last piece.
    #[test]
    fn dispatch_chain_lds_cuts_before_a_guest_wave_returns() {
        let threads = 256;
        let geometry = [threads, 1, 1];
        let mut lw = lds_lw(threads, true);
        let (tid, _) = lw.setup_lanes_compute(geometry, false);
        let mem = lw.mems[0];
        let group = lw.s.group_id(0);
        let count = lw.c(threads);
        let base = lw.mul(group, count);
        let index = lw.add(base, tid);
        let offset = lw.shlc(index, 2);
        let x = lw.s.raw_load(mem, offset, 1)[0];
        let mode = lw.andc(tid, 3);
        let trips = lw.addc(mode, 1);
        lds_write(&mut lw, tid, x);
        guest_barrier(&mut lw);
        let mut v = lane_loops(&mut lw, 1..=14, x, x, trips);
        let wave = lw.lshrc(tid, 6);
        let first = lw.cmpc(Pred::Eq, wave, 0);
        let (exit, live) = (lw.s.m.new_block(), lw.s.m.new_block());
        lw.s.m.cond_br(first, exit, live);
        lw.s.m.set_block(exit);
        lw.s.raw_store(mem, offset, &[v]);
        lw.s.m.ret();
        lw.s.m.set_block(live);
        v = lane_loops(&mut lw, 15..=16, v, x, trips);
        guest_barrier(&mut lw);
        lds_write(&mut lw, tid, v);
        guest_barrier(&mut lw);
        v = lane_loops(&mut lw, 17..=18, v, x, trips);
        let other = lw.c(255);
        let other = lw.sub(other, tid);
        let peer = lds_read(&mut lw, other);
        v = lw.add(v, peer);
        lw.s.raw_store(mem, offset, &[v]);
        lw.s.m.ret();
        lw.s.m.optimize();

        let limit = 64;
        let m = &lw.s.m;
        let ret = |b: &super::Block| m.successors(b.0 as usize).is_empty();
        let barrier = |b: &super::Block| m.blocks[b.0 as usize].iter().any(|&i| matches!(&m.insts[i as usize].op,
            super::Op::Call(f, _) if m.funcs[*f as usize].name == "dx.op.barrier"));
        // the guest barriers that wave 0 skips by returning
        let fork = (0..m.blocks.len()).find(|&b| { let s = m.successors(b); s.len() == 2 && s.iter().any(ret) }).unwrap();
        let mut after = std::collections::BTreeSet::new();
        let mut work = m.successors(fork);
        while let Some(b) = work.pop() {
            if after.insert(b) { work.extend(m.successors(b.0 as usize)); }
        }
        // one of them in a piece that wave 0 still has to pass
        let late_barrier = |plan: &super::Plan| plan.blocks[..plan.blocks.len() - 1].iter().flatten()
            .any(|b| after.contains(b) && barrier(b));
        assert!((40..=160).filter_map(|limit| m.dispatch_plan(limit, None)).any(|plan| late_barrier(&plan)),
            "the negative control cuts after the return");
        assert!((40..=160).filter_map(|limit| m.dispatch_plan(limit, Some(&Default::default()))).all(|plan| !late_barrier(&plan)));
        cuts_only_where_the_group_meets(m);
        let plan = m.dispatch_plan(limit, Some(&Default::default())).unwrap();
        assert!(plan.blocks.len() >= 3, "{} pieces", plan.blocks.len());

        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        let chain = lw.s.split_compute(geometry, Some(tid), limit).expect("split before the return");
        assert_eq!(chain.pieces.len(), plan.blocks.len());
        let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
        let input: Vec<u32> = (0..512 + 16).map(|i| if i < 512 { i * 7 + 3 } else { 0xdeadbeef }).collect();
        let mut expected = input.clone();
        for g in 0..2 {
            let x = &input[g * 256..(g + 1) * 256];
            let loops = |t: usize, rounds: std::ops::RangeInclusive<u32>| rounds.map(|r| ((t as u32 & 3) + 1) * (x[t] + r)).sum::<u32>();
            let stored: Vec<u32> = (0..256).map(|t| x[t] + loops(t, 1..=14) + loops(t, 15..=16)).collect();
            for t in 0..256 {
                expected[g * 256 + t] = if t < 64 { x[t] + loops(t, 1..=14) }
                    else { stored[t] + loops(t, 17..=18) + if 255 - t < 64 { x[255 - t] } else { stored[255 - t] } };
            }
        }
        let reference = runner::dispatch(&whole, [2, 1, 1], &[], &input).unwrap();
        assert_eq!(reference, expected, "whole program");
        assert_eq!(runner::dispatch_split(&chain, [2, 1, 1], &input).unwrap(), expected, "split program");
    }

    /// Another guest wave may poll the LDS in a loop (with an atomic or a
    /// plain load) until a write that a cut would defer: such a group of
    /// several waves stays whole. One wave cannot wait for another.
    #[test]
    fn dispatch_chain_lds_poll_loop_keeps_several_waves_whole() {
        for (threads, looped, atomic) in [(128, true, true), (128, true, false), (128, false, true), (64, true, true), (64, true, false)] {
            let geometry = [threads, 1, 1];
            let mut lw = lds_lw(threads, true);
            let (tid, _) = lw.setup_lanes_compute(geometry, false);
            let mem = lw.mems[0];
            let offset = lw.shlc(tid, 2);
            let x = lw.s.raw_load(mem, offset, 1)[0];
            let mode = lw.andc(tid, 3);
            let trips = lw.addc(mode, 1);
            let mut v = lane_loops(&mut lw, 1..=6, x, x, trips);
            let g = lw.lds.unwrap();
            let zero = lw.c(0);
            let p = lw.s.gs_ptr(g, zero);
            if looped {
                let pred = lw.cur();
                let body = lw.s.m.new_block();
                let exit = lw.s.m.new_block();
                lw.s.m.br(body);
                lw.s.m.set_block(body);
                let i32 = lw.s.m.i32();
                let old = lw.s.m.phi(i32);
                lw.s.m.add_incoming(old, zero, pred);
                let got = if atomic { lw.s.m.atomic_rmw(crate::gpu::dxil::module::RmwOp::Add, p, zero) } else { lw.s.m.load(p, 4) };
                lw.s.m.add_incoming(old, got, body);
                let wait = lw.cmp(Pred::ULt, got, x);
                lw.s.m.cond_br(wait, body, exit);
                lw.s.m.set_block(exit);
                v = lw.add(v, old);
            } else {
                let got = lw.s.m.atomic_rmw(crate::gpu::dxil::module::RmwOp::Add, p, x);
                v = lw.add(v, got);
            }
            v = lane_loops(&mut lw, 7..=12, v, x, trips);
            lw.s.raw_store(mem, offset, &[v]);
            lw.s.m.ret();
            lw.s.m.optimize();
            let chain = lw.s.split_compute(geometry, Some(tid), 48);
            // one guest wave cannot wait for another
            assert_eq!(chain.is_some(), !(looped && threads > 64), "{threads} threads, LDS polled in a loop: {looped}, atomic: {atomic}");
        }
    }
}
