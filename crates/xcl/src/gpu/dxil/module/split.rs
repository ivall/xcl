//! Partition the optimized compute entry at CFG edges. A dispatch cannot
//! jump back to an earlier dispatch, so a strongly connected component is
//! indivisible. Each crossing edge has its own compact state layout.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

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

/// The part of a dispatch a chain runs at once (`pso::record_dispatch_chain`): its groups in X
/// and Y, and the dispatch's group its first one is.
struct Slab {
    width: Value,
    height: Value,
    origin: [Value; 3],
}

impl Slab {
    fn new(s: &mut Shader) -> Self {
        let cb = s.cbv(0, 2, 32);
        let ty = s.m.i32();
        let [size, origin] = [0, 1].map(|row| {
            let row = s.m.i32c(row);
            s.cbuffer_load_legacy(cb, row, ty)
        });
        Self { width: size[0], height: size[1], origin: [origin[0], origin[1], origin[2]] }
    }

    /// The program's value for its copied instruction `v`: a group id is the dispatch's, not the slab's.
    fn group_id(&self, src: &Module, op: &Op, m: &mut Module, v: Value) -> Value {
        match op {
            Op::Call(f, args) if src.funcs[*f as usize].name.starts_with("dx.op.groupId") => match args.get(1).and_then(|&a| src.const_u64(a)) {
                Some(axis @ 0..=2) => m.bin(BinOp::Add, v, self.origin[axis as usize]),
                _ => v,
            },
            _ => v,
        }
    }
}

impl Scratch {
    fn new(s: &mut Shader, slab: &Slab, stride: u32, threads: u32, lane: Value) -> Self {
        let uav = s.raw_uav(0, 2);
        let (width, height) = (slab.width, slab.height);
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

    fn at(&self, m: &mut Module, slot: u32) -> Value {
        let offset = m.i32c(slot * self.bytes_per_slot);
        m.bin(BinOp::Add, self.base, offset)
    }
}

impl Shader {
    /// Start with full native wave32 groups. A fresh dispatch can assign a
    /// different physical wave to a group; wave-lane index identifies the
    /// continuation within a full, single-wave group.
    pub(crate) fn split_compute(&self, threads: [u32; 3], split_tid: Option<Value>, limit: usize) -> Option<DispatchChain> {
        if self.m.globals.iter().any(|g| !g.constant && !(split_tid.is_some() && matches!(g.name.as_str(), "xcl_xw" | "xcl_xs"))) {
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
        for (b, block) in self.m.blocks.iter().enumerate() {
            // In a single split console wave each inter-half exchange begins
            // with TGSM stores, one rendezvous, peer loads and a rendezvous.
            // A block ending anywhere inside that sequence cannot be a cut.
            let mut exchange = 0;
            for &i in block {
                if b == 0 && prefix.contains(&i) { continue; }
                match &self.m.insts[i as usize].op {
                    Op::Alloca(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => return None,
                    Op::Store(..) if split_tid.is_some() && exchange <= 1 => exchange = 1,
                    Op::Load(p, _) if self.m.global_root(*p).is_some_and(|g| self.m.globals[g].constant) => {}
                    Op::Load(..) if split_tid.is_some() && matches!(exchange, 2 | 3) => exchange = 3,
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
        let plan = self.m.dispatch_plan(limit)?;
        // Slot zero is the tag. Every transported SSA value owns its payload slots
        // until the chain finishes. An outgoing store can never overwrite a value
        // restored by the same piece, even when the restore is sunk past a branch.
        let mut stride = 1u32;
        let mut slots = BTreeMap::new();
        for value in plan.edges.iter().flat_map(|e| e.live.iter().copied()).collect::<BTreeSet<_>>() {
            let words = self.m.transport_words(value)?;
            if words != 0 {
                slots.insert(value, stride);
                stride = stride.checked_add(words)?;
            }
        }
        if cfg!(test) && std::env::var_os("XCL_SAMPLE_SLOT_REUSE").is_some() {
            (stride, slots) = self.m.reuse_piece_slots(&plan, &slots)?;
        }
        stride.checked_mul(threads[0])?.checked_mul(4)?;
        let pieces = (0..plan.blocks.len()).map(|i| self.make_piece(&plan, i, stride, &slots, threads, &prefix)).collect::<Option<Vec<_>>>()?;
        Some(DispatchChain { pieces, dwords_per_group: stride.checked_mul(threads[0])? })
    }

    fn make_piece(&self, plan: &Plan, stage: usize, stride: u32, slots: &BTreeMap<Value, u32>, threads: [u32; 3], prefix: &[u32]) -> Option<Vec<u8>> {
        let src = &self.m;
        let mut dst = self.dispatch_piece();
        let mut values = HashMap::new();
        let slab = Slab::new(&mut dst);
        for &id in prefix {
            let def = &src.insts[id as usize];
            let mut op = def.op.clone();
            op.map_operands(|v| *values.get(&v).unwrap_or(&v));
            let v = dst.m.push(op, def.ty);
            values.insert(Value::Inst(id), slab.group_id(src, &def.op, &mut dst.m, v));
        }
        let lane = prefix.last().map(|&i| values[&Value::Inst(i)]).unwrap_or_else(|| dst.wave_lane_index());
        let scratch = Scratch::new(&mut dst, &slab, stride, threads[0], lane);
        let local: HashMap<_, _> = plan.blocks[stage].iter().map(|&b| (b, dst.m.new_block())).collect();
        let mut hops = HashMap::new();
        let mut exits = HashMap::new();
        let mut cases = Vec::new();
        let mut deferred = BTreeMap::new();
        let mut incoming = BTreeSet::new();
        let done = dst.m.new_block();
        let selector = if stage != 0 { Some(dst.raw_load(scratch.uav, scratch.base, 1)[0]) } else { None };
        for (index, edge) in plan.edges.iter().enumerate() {
            let tag = dst.m.i32c(index as u32 + 1);
            if local.contains_key(&edge.from) {
                exits.insert((edge.from, edge.to), (dst.m.new_block(), tag));
            }
            if let Some(&target) = local.get(&edge.to) {
                incoming.extend(edge.live.iter().copied());
                let hop = dst.m.new_block();
                dst.m.set_block(hop);
                dst.m.br(target);
                cases.push((tag, hop));
                hops.insert((edge.from, edge.to), hop);
            }
        }
        dst.m.set_block(Block(0));
        if let Some(sel) = selector {
            for v in incoming {
                if values.contains_key(&v) { continue; }
                if slots.contains_key(&v) {
                    let id = dst.m.detached_phi(src.insts[inst(v)?].ty);
                    deferred.insert(id, v);
                    values.insert(v, id);
                } else {
                    src.rebuild_static(v, &mut dst.m, &mut values)?;
                }
            }
            dst.m.switch(sel, done, &cases);
        } else {
            dst.m.br(local[&Block(0)]);
        }
        dst.m.set_block(done);
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
                }
                let v = dst.m.push(old.op.clone(), old.ty);
                values.insert(Value::Inst(i), slab.group_id(src, &old.op, &mut dst.m, v));
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
            for &v in &edge.live {
                // The defining piece writes a distinct slot. Later pieces can bypass
                // it or read it without copying it at their own outgoing edges.
                if let Some(&slot) = slots.get(&v) {
                    if !local.contains_key(&src.def_block[inst(v)?]) { continue; }
                    let words = src.pack(v, *values.get(&v)?, &mut dst.m);
                    for (k, word) in words.into_iter().enumerate() {
                        let at = scratch.at(&mut dst.m, slot + k as u32);
                        dst.raw_store(scratch.uav, at, &[word]);
                    }
                }
            }
            dst.raw_store(scratch.uav, scratch.base, &[tag]);
            dst.m.ret();
        }
        src.place_restores(&mut dst, &scratch, slots, &deferred)?;
        dst.m.sink_cheap_piece_ops();
        dst.infer_piece_features();
        // a refused piece leaves the whole program unsplit: say so
        let refused = |e: String| crate::xlog_warn!("compute split: piece {} of {} refused, the program stays whole: {}",
            stage + 1, plan.blocks.len(), e.lines().next().unwrap_or_default());
        if !dst.m.terminated() {
            dst.m.ret();
        }
        dst.m.optimize();
        dst.prune_unused_bindings();
        let blob = dst.finish_optimized_canonical(threads, Some(32)).map_err(|e| refused(e.to_string())).ok()?;
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

fn common_dominator(mut a: usize, mut b: usize, rank: &[Option<u32>], dom: &[Option<usize>]) -> usize {
    while a != b {
        if rank[a] > rank[b] {
            a = dom[a].unwrap();
        } else {
            b = dom[b].unwrap();
        }
    }
    a
}

fn inst(v: Value) -> Option<usize> {
    if let Value::Inst(i) = v { Some(i as usize) } else { None }
}

impl Module {
    fn reuse_piece_slots(&self, plan: &Plan, slots: &BTreeMap<Value, u32>) -> Option<(u32, BTreeMap<Value, u32>)> {
        let mut stage = vec![0; self.blocks.len()];
        for (i, blocks) in plan.blocks.iter().enumerate() {
            for block in blocks { stage[block.0 as usize] = i; }
        }
        let mut last = BTreeMap::new();
        for edge in &plan.edges {
            for &v in &edge.live {
                let end = last.entry(v).or_insert(0);
                *end = (*end).max(stage[edge.to.0 as usize]);
            }
        }
        let mut ranges = Vec::with_capacity(slots.len());
        for &v in slots.keys() {
            let start = stage[self.def_block[inst(v)?].0 as usize];
            ranges.push((start, v, last[&v], self.transport_words(v)? as usize));
        }
        ranges.sort_unstable();
        let mut occupied_until = Vec::new();
        let mut layout = BTreeMap::new();
        for (start, v, end, size) in ranges {
            // A restore can be sunk to the end of its piece, so even values last
            // consumed in the writer's own piece must keep separate slots.
            let index = occupied_until.windows(size).position(|words| words.iter().all(|&e| e < start))
                .unwrap_or(occupied_until.len());
            occupied_until.resize(occupied_until.len().max(index + size), end);
            occupied_until[index..index + size].fill(end);
            layout.insert(v, u32::try_from(index + 1).ok()?);
        }
        Some((u32::try_from(occupied_until.len() + 1).ok()?, layout))
    }

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

    // Phi operands are consumed on incoming edges. A forwarding-only input has
    // no placeholder users, so it requires no load in this piece.
    fn place_restores(
        &self,
        dst: &mut Shader,
        scratch: &Scratch,
        slots: &BTreeMap<Value, u32>,
        deferred: &BTreeMap<Value, Value>,
    ) -> Option<()> {
        let (order, rank) = dst.m.block_layout();
        let dom = dst.m.idoms(&order.iter().map(|&b| b as u32).collect::<Vec<_>>());
        let mut uses = HashMap::new();
        for &b in &order {
            for &id in &dst.m.blocks[b] {
                let op = &dst.m.insts[id as usize].op;
                let mut record = |v: Value, p: usize| {
                    if deferred.contains_key(&v) {
                        uses.entry(v).and_modify(|d| *d = common_dominator(*d, p, &rank, &dom)).or_insert(p);
                    }
                };
                if let Op::Phi(edges) = op {
                    for &(v, p) in edges {
                        record(v, p.0 as usize);
                    }
                } else {
                    op.for_each_operand(|v| record(v, b));
                }
            }
        }
        let mut replacements = HashMap::new();
        for (&placeholder, &v) in deferred {
            let Some(&b) = uses.get(&placeholder) else { continue };
            dst.m.set_block(Block(b as u32));
            let terminator = dst.m.blocks[b].pop()?;
            let previous = dst.m.blocks[b].len();
            let slot = slots[&v];
            let words = (slot..slot + self.transport_words(v)?)
                .map(|k| {
                    let at = scratch.at(&mut dst.m, k);
                    dst.raw_load(scratch.uav, at, 1)[0]
                })
                .collect::<Vec<_>>();
            let value = self.unpack(v, &words, &mut dst.m);
            let added = dst.m.blocks[b].split_off(previous);
            // In block zero the scratch address is materialized before the switch.
            // At every other block entry only the leading phis must precede a load.
            let at = if b == 0 {
                previous
            } else {
                dst.m.blocks[b].iter().take_while(|&&i| matches!(dst.m.insts[i as usize].op, Op::Phi(_))).count()
            };
            dst.m.blocks[b].splice(at..at, added);
            dst.m.blocks[b].push(terminator);
            replacements.insert(placeholder, value);
        }
        dst.m.replace_uses(&replacements);
        Some(())
    }

    fn sink_cheap_piece_ops(&mut self) {
        // Only sink a cheap definition whose nonconstant inputs are already used in
        // the destination block. Moving it cannot introduce a new live-in there.
        let (order, _) = self.block_layout();
        let mut uses: HashMap<Value, BTreeSet<usize>> = HashMap::new();
        for &b in &order {
            for &id in &self.blocks[b] {
                let op = &self.insts[id as usize].op;
                if let Op::Phi(inc) = op {
                    for &(v, p) in inc {
                        uses.entry(v).or_default().insert(p.0 as usize);
                    }
                } else {
                    op.for_each_operand(|v| {
                        uses.entry(v).or_default().insert(b);
                    });
                }
            }
        }
        // Moving a producer and then a consumer based on stale use sets may violate
        // dominance. Limit this pass to candidates whose consumers stay in place.
        let eligible = |op: &Op| matches!(op, Op::Cast(CastOp::BitCast | CastOp::Trunc, ..) | Op::Select(..));
        let mut feeds_candidate = HashSet::new();
        for def in &self.insts {
            if eligible(&def.op) {
                def.op.for_each_operand(|v| {
                    feeds_candidate.insert(v);
                });
            }
        }
        for &b in &order {
            for id in self.blocks[b].clone() {
                let val = Value::Inst(id);
                if !eligible(&self.insts[id as usize].op) || feeds_candidate.contains(&val) {
                    continue;
                }
                let Some(sites) = uses.get(&val) else { continue };
                if sites.len() != 1 {
                    continue;
                }
                let t = *sites.first().unwrap();
                if t == b {
                    continue;
                }
                let mut shared = true;
                self.insts[id as usize].op.for_each_operand(|v| {
                    if let Value::Inst(_) = v {
                        shared &= uses.get(&v).is_some_and(|vs| vs.contains(&t))
                            && self.def_block[inst(v).unwrap()].0 as usize != t;
                    }
                });
                if !shared {
                    continue;
                }
                let at =
                    self.blocks[t].iter().take_while(|&&i| matches!(self.insts[i as usize].op, Op::Phi(_))).count();
                self.blocks[b].retain(|&i| i != id);
                self.blocks[t].insert(at, id);
                self.def_block[id as usize] = Block(t as u32);
            }
        }
    }

    /// Only an index is transported. Resource handles and TGSM pointers are
    /// created inside the receiving dispatch, using their original properties.
    fn indexed(&self, v: Value) -> Option<Value> {
        let op = &self.insts[inst(v)?].op;
        if let Op::Gep(Value::Global(g), indices) = op {
            if self.globals[*g as usize].name == "xcl_xs" && indices.len() == 2 && self.const_u64(indices[0]) == Some(0) {
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

    pub(crate) fn dispatch_plan(&self, limit: usize) -> Option<Plan> {
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
        let mut used = 0;
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
            // The target is soft for an indivisible component. A component
            // above it occupies a piece by itself; bound that exception so a
            // very large loop still selects the whole pipeline.
            if cost > limit.saturating_mul(4) {
                return None;
            }
            if used != 0 && used + cost > limit {
                blocks.push(Vec::new());
                used = 0;
            }
            used += cost;
            for b in component {
                stages[b] = blocks.len() - 1;
            }
        }

        if blocks.len() == 1 { return None; }

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
        dxil::{kernel::{Entry, KernelOptions}, lower::Lw, module::{BinOp, CastOp, Pred}},
        recomp::{runner, CsInterface},
    };

    #[test]
    fn oversized_loop_occupies_one_piece_up_to_the_hard_ceiling() {
        use super::{Block, Module};

        let mut m = Module::new();
        let entry = m.entry_block();
        let header = m.new_block();
        let body = m.new_block();
        let exit = m.new_block();
        let word = m.i32();
        let zero = m.i32c(0);
        let one = m.i32c(1);
        m.br(header);
        m.set_block(header);
        let count = m.phi(word);
        m.add_incoming(count, zero, entry);
        let more = m.cmp(Pred::ULt, count, one);
        m.cond_br(more, body, exit);
        m.set_block(body);
        let mut next = count;
        for _ in 0..12 { next = m.bin(BinOp::Add, next, one); }
        m.add_incoming(count, next, body);
        m.br(header);
        m.set_block(exit);
        m.ret();

        // Header (3) + body (13) is exactly four times a target of 4.
        let plan = m.dispatch_plan(4).expect("keep the loop in its own piece");
        assert_eq!(plan.blocks, [vec![entry], vec![header, body], vec![exit]]);
        assert!(m.dispatch_plan(3).is_none());

        // An oversized first block must not create an empty leading piece.
        // If the program fits wholly in this one component, splitting has no benefit.
        let mut m = Module::new();
        let one = m.i32c(1);
        for _ in 0..4 { m.bin(BinOp::Add, one, one); }
        let end = m.new_block();
        m.br(end);
        m.set_block(end);
        m.ret();
        assert_eq!(m.dispatch_plan(4).unwrap().blocks, [vec![Block(0)], vec![end]]);
        let mut m = Module::new();
        let one = m.i32c(1);
        for _ in 0..4 { m.bin(BinOp::Add, one, one); }
        m.ret();
        assert!(m.dispatch_plan(4).is_none());
    }

    fn compare(threads: u32, extra_adds: u32) {
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
        let mut large_loop = None;
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
            if round == 8 {
                for _ in 0..extra_adds { value = lw.add(value, original); }
                if extra_adds > 0 { large_loop = Some(body); }
            }
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
        if let Some(oversized) = large_loop {
            let n = lw.s.m.block_insts(oversized).len();
            assert!(n > 40 && n <= 160);
            let plan = lw.s.m.dispatch_plan(40).expect("partition around oversized loop");
            assert!(plan.blocks.iter().any(|piece| piece == &[oversized]));
        }
        let chain = lw.s.split_compute(geometry, split.then_some(tid), 40);
        let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
        crate::gpu::dxil::validate::validate(&whole).expect("valid unsplit fixture");
        let chain = chain.expect("split the synthetic program");
        assert!(chain.pieces.len() >= 3);
        let initial: Vec<_> = (1..=threads * 3).collect();
        let expected: Vec<_> = initial.iter().enumerate().map(|(i, &v)| {
            let n = (i as u32 % 4) + 1;
            v + n * ((16 + extra_adds) * v + 136) + u32::from(split && i as u32 % threads < 32) + (v ^ (i as u32 % threads))
        }).collect();
        let reference = runner::dispatch(&whole, [3, 1, 1], &[], &initial).unwrap();
        assert_eq!(reference, expected);
        let result = runner::dispatch_split(&chain, [3, 1, 1], &initial).unwrap();
        assert_eq!(result, reference);
    }

    fn check_persistent_payload(reuse_hole: bool) {
        use super::Value;

        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        fn mix(lw: &mut Lw, mut v: Value, seed: u32) -> Value {
            let multiplier = lw.c(33);
            for i in 1..=12 {
                let key = lw.c(seed + i);
                v = lw.xor(v, key);
                v = lw.mul(v, multiplier);
            }
            v
        }
        fn expected_mix(v: u32, seed: u32) -> u32 {
            (1..=12).fold(v, |x, i| (x ^ (seed + i)).wrapping_mul(33))
        }

        let geometry = [32, 1, 1];
        let options = KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 };
        let cs = CsInterface { threads: geometry, ..Default::default() };
        let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
        let (tid, _) = lw.setup_lanes_compute(geometry, false);
        let mem = lw.mems[0];
        let offset = lw.shlc(tid, 2);
        let first = lw.s.raw_load(mem, offset, 1)[0];
        let saved = mix(&mut lw, first, 10);
        let middle = lw.s.m.new_block();
        let last = lw.s.m.new_block();
        // every lane takes the middle block; the branch only keeps the optimiser from merging it
        let entry = lw.cur();
        let width = lw.s.wave_lane_count();
        let taken = lw.cmp(Pred::ULt, tid, width);
        lw.s.m.cond_br(taken, middle, last);
        lw.s.m.set_block(middle);
        let replacement = mix(&mut lw, if reuse_hole { first } else { tid }, 30);
        lw.s.raw_store(mem, offset, &[replacement]);
        lw.s.m.br(last);
        lw.s.m.set_block(last);
        let i32 = lw.s.m.i32();
        let arrived = lw.s.m.phi(i32);
        lw.s.m.add_incoming(arrived, replacement, middle);
        lw.s.m.add_incoming(arrived, first, entry);
        let input = if reuse_hole { lw.xor(saved, arrived) } else { saved };
        let output = mix(&mut lw, input, 50);
        lw.s.raw_store(mem, offset, &[output]);
        lw.s.m.ret();
        lw.s.m.optimize();

        let limit = lw.s.m.blocks.iter().map(Vec::len).max().unwrap();
        let plan = lw.s.m.dispatch_plan(limit).unwrap();
        assert_eq!(plan.blocks.len(), 3);
        let incoming = plan.edges.iter().find(|edge| edge.to == middle).unwrap();
        let outgoing = plan.edges.iter().find(|edge| edge.from == middle && edge.to == last).unwrap();
        assert!(incoming.live.contains(&saved) && outgoing.live.contains(&saved));
        if reuse_hole {
            // In compact per-edge order, the replacement occupies saved's old
            // slot. A delayed load of saved from that slot would get replacement.
            let slot = |edge: &super::Edge, v| {
                edge.live.iter().filter(|&&x| lw.s.m.transport_words(x) == Some(1)).position(|&x| x == v).unwrap()
            };
            assert_eq!(slot(incoming, saved), slot(outgoing, replacement));
        }

        let chain = lw.s.split_compute(geometry, None, limit).unwrap();
        let whole = lw.s.finish_optimized(geometry, Some(32)).unwrap();
        let initial: Vec<u32> = (0..32).map(|i| 1009 * i + 77).collect();
        let mut expected: Vec<_> = initial.iter().map(|&v| {
            let saved = expected_mix(v, 10);
            expected_mix(if reuse_hole { saved ^ expected_mix(v, 30) } else { saved }, 50)
        }).collect();
        // The runners read back at least 64 words. Only lanes 0..32 write.
        expected.resize(64, 0);
        assert_eq!(runner::dispatch(&whole, [1, 1, 1], &[], &initial).unwrap(), expected);
        assert_eq!(runner::dispatch_split(&chain, [1, 1, 1], &initial).unwrap(), expected);
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
    fn dispatch_chain_payload_skips_middle_piece() { check_persistent_payload(false); }

    #[test]
    fn dispatch_chain_payload_survives_reused_edge_slot() { check_persistent_payload(true); }

    #[test]
    fn dispatch_chain_wave32_live_loops() { compare(32, 0); }

    /// A guest wave64 split across two hardware wave32s.
    #[test]
    fn dispatch_chain_wave64_live_loops_and_exchange() { compare(64, 0); }
    #[test]
    fn dispatch_chain_oversized_loop_matches_whole() {
        compare(32, 48);
        compare(64, 48);
    }

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
            let plan = lw.s.m.dispatch_plan(64).expect("partition branch fixture");
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
            if std::env::var_os("XCL_SAMPLE_CHAIN_TIMINGS").is_some() {
                for depth in [2, 128] {
                    let groups = [2, 3, depth];
                    let words = (threads * 6 * depth) as usize;
                    let initial: Vec<_> = (0..words + 16).map(|i| if i < words { i as u32 + 17 } else { 0xdeadbeef }).collect();
                    let mut expected = initial.clone();
                    for _ in 0..9 {
                        for (i, v) in expected[..words].iter_mut().enumerate() {
                            let (scale, bias): (u32, u32) = match i % 4 {
                                0 => { *v = 100_000 + i as u32; continue; }
                                1 => (17, 200),
                                2 => (49, 408),
                                _ => (33, 144),
                            };
                            *v = v.wrapping_mul(scale).wrapping_add(bias);
                        }
                    }
                    let mut rows = [Vec::new(), Vec::new()];
                    for round in 0..4 {
                        for pass in 0..2 {
                            let variant = (round + pass) % 2;
                            let samples = if variant == 0 {
                                runner::dispatch_timed(&whole, groups, &[], &initial, 9).unwrap()
                            } else {
                                let (out, times) = runner::dispatch_split_samples(&chain, groups, &[], &initial, u64::MAX, 9, true).unwrap();
                                assert_eq!(out, expected, "timed chain {threads} at depth {depth}");
                                times
                            };
                            rows[variant].extend_from_slice(&samples[1..]);
                        }
                    }
                    for (variant, row) in rows.iter_mut().enumerate() {
                        row.sort_by(f64::total_cmp);
                        println!("CHAIN_TIME lanes={threads} groups={} variant={variant} n={} median_us={:.3}", 6 * depth,
                            row.len(), 500.0 * (row[row.len()/2 - 1] + row[row.len()/2]));
                    }
                }
            }
            // in slabs: single groups, a row (and a part of one), a plane, all
            for most in [1, 2, 3, 6, 7, 12] {
                let result = runner::dispatch_split_by(&chain, [2, 3, 2], &initial, most).unwrap();
                assert_eq!(result, expected, "split {threads}, {most} groups at once");
            }
            println!("CHAIN_BRANCH threads={threads} pieces={} scratch_words_per_group={} checked_words={words} guards=16", chain.pieces.len(), chain.dwords_per_group);
        }
    }

    #[test]
    fn canonical_pieces_match_despite_unused_constants_and_order() {
        use crate::gpu::dxil::container::Stage;
        use crate::gpu::dxil::module::{Block, CastOp, Pred};
        use crate::gpu::dxil::shader::Shader;
        use crate::gpu::shader::fnv64;
        use crate::hle::d3d12x::real::driver_cache;

        let build_variant = |salt: u32, reverse_consts: bool, extra_unused_binding: bool, k_a: u32, f_bits: u32| -> Vec<Vec<u8>> {
            let mut s = Shader::new(Stage::Compute, (6, 6));
            let i32 = s.m.i32();
            let f32 = s.m.f32();
            let out = s.raw_uav(0, 0);
            let unused_cb = extra_unused_binding.then(|| s.cbv(3, 0, 16 + salt));
            let tid = s.thread_id_in_group(0);
            let two = s.m.i32c(2);
            let off = s.m.bin(BinOp::Shl, tid, two);

            // Variant-specific constants, globals, and floats isolated to piece 0.
            let c_salt = s.m.i32c(0x1357_0000u32.wrapping_add(salt));
            let c_dead = s.m.i32c(0xDEAD_0000u32.wrapping_add(salt));
            let f_nan = s.m.f32c(0x7FC0_0001u32 + salt);
            let _unused_global = {
                let at = s.m.array(4, i32);
                let elems: Vec<_> = (0..4).map(|k| s.m.i32c(salt * 10 + k)).collect();
                let init = s.m.aggregate(at, &elems);
                s.m.global(&format!("dead_tbl_{salt}"), at, 0, Some(init), true, 4)
            };

            // Pre-intern piece-1 constants in opposite orders before piece 1 is built.
            let k_b = 0x3333_4444u32;
            let b_scale = 1.5f32.to_bits();
            if reverse_consts {
                let _ = (s.m.i32c(k_b), s.m.i32c(k_a), s.m.f32c(f_bits), s.m.f32c(b_scale));
            } else {
                let _ = (s.m.i32c(k_a), s.m.i32c(k_b), s.m.f32c(b_scale), s.m.f32c(f_bits));
            }

            // Dead computation folding away dead constants and optional unused CBV load.
            let nan_bits = s.m.cast(CastOp::BitCast, f_nan, i32);
            let dead_sum = s.m.bin(BinOp::Xor, c_dead, nan_bits);
            let _dead_only = if let Some(cb) = unused_cb {
                let zero = s.m.i32c(0);
                let loaded = s.cbuffer_load_legacy(cb, zero, i32)[0];
                s.m.bin(BinOp::Add, loaded, dead_sum)
            } else {
                dead_sum
            };

            // Piece 0 writes salt-dependent value to out[tid].
            let p0_val = s.m.bin(BinOp::Add, tid, c_salt);
            s.raw_store(out, off, &[p0_val]);

            let one = s.m.i32c(1);
            let three = s.m.i32c(3);
            let mode = s.m.bin(BinOp::And, tid, three);
            let trips = s.m.bin(BinOp::Add, mode, one);
            let zero = s.m.i32c(0);

            // Six sequential back-edge loops so split_compute splits cleanly;
            // piece 0 absorbs variant differences while downstream pieces match identically.
            let mut cur_val = p0_val;
            let mut pred = Block(0);
            for step in 0..6u32 {
                let body = s.m.new_block();
                let next = s.m.new_block();
                s.m.br(body);
                s.m.set_block(body);
                let cnt = s.m.phi(i32);
                let phi = s.m.phi(i32);
                s.m.add_incoming(cnt, zero, pred);
                s.m.add_incoming(phi, cur_val, pred);
                let c_step = s.m.i32c(if step < 2 { salt.wrapping_add(step + 1) } else { k_a.wrapping_add(step) });
                let mixed = s.m.bin(BinOp::Add, phi, c_step);
                let c_kb = s.m.i32c(k_b);
                let mixed2 = s.m.bin(BinOp::Xor, mixed, c_kb);
                let fp = s.m.cast(CastOp::UIToFP, mixed2, f32);
                let c_scale = s.m.f32c(b_scale);
                let scaled = s.m.fbin(BinOp::Mul, fp, c_scale, 0);
                let step_val = if step >= 4 {
                    // Applied directly inside downstream blocks (with 0 fast-math flags)
                    // so -0.0 vs +0.0 vs NaN bit patterns are tested inside piece 2.
                    let c_f = s.m.f32c(f_bits);
                    let shifted = s.m.fbin(BinOp::Add, scaled, c_f, 0);
                    s.m.cast(CastOp::BitCast, shifted, i32)
                } else {
                    s.m.cast(CastOp::FPToUI, scaled, i32)
                };
                let next_cnt = s.m.bin(BinOp::Add, cnt, one);
                s.m.add_incoming(cnt, next_cnt, body);
                s.m.add_incoming(phi, step_val, body);
                let again = s.m.cmp(Pred::ULt, next_cnt, trips);
                s.m.cond_br(again, body, next);
                s.m.set_block(next);
                cur_val = step_val;
                pred = next;
            }
            s.raw_store(out, off, &[cur_val]);
            s.m.ret();
            s.m.optimize();
            let chain = s.split_compute([32, 1, 1], None, 32).expect("should split");
            assert!(chain.pieces.len() >= 2, "expected >=2 pieces, got {}", chain.pieces.len());
            for (idx, p) in chain.pieces.iter().enumerate() {
                crate::gpu::dxil::validate::validate(p).unwrap_or_else(|e| panic!("piece {idx} invalid: {e}"));
            }
            chain.pieces
        };

        let neg_zero = (-0.0f32).to_bits();
        let pos_zero = 0.0f32.to_bits();
        let a = build_variant(1, false, false, 0x1111_2222, neg_zero);
        let a_repeat = build_variant(1, false, false, 0x1111_2222, neg_zero);
        let b = build_variant(99, true, true, 0x1111_2222, neg_zero);
        let c_changed_const = build_variant(99, true, true, 0x1111_2223, neg_zero);
        let d_pos_zero = build_variant(99, true, true, 0x1111_2222, pos_zero);
        let e_nan = build_variant(99, true, true, 0x1111_2222, 0x7FC0_0042);

        println!(
            "CANON_PIECE1 pieces=({},{}) len_a={} len_b={} same={} fnv={:016x}",
            a.len(),
            b.len(),
            a.last().unwrap().len(),
            b.last().unwrap().len(),
            a[1..] == b[1..],
            fnv64(a.last().unwrap())
        );
        assert_eq!(a, a_repeat, "serialization must be deterministic across repeated runs");
        assert_eq!(a[1..], b[1..], "downstream pieces must ignore piece-0 unused state, bindings, and constant order");
        assert_eq!(
            driver_cache::name_for(a.last().unwrap(), 0, true),
            driver_cache::name_for(b.last().unwrap(), 0, true),
            "canonical downstream piece must produce identical driver cache identity"
        );
        assert_ne!(a.last(), c_changed_const.last(), "changing a referenced constant must change piece output");
        assert_ne!(a.last(), d_pos_zero.last(), "-0.0 and +0.0 bit patterns must stay distinct");
        assert_ne!(d_pos_zero.last(), e_nan.last(), "distinct float bit patterns must stay distinct");
    }

    #[test]
    fn canonical_pieces_preserve_dynamic_indexing_and_match_gpu() {
        use crate::gpu::dxil::module::{CastOp, Pred};
        use crate::gpu::dxil::validate::{disassemble, validate};
        use crate::gpu::shader::fnv64;

        let build = |salt: u32, dead_first: bool, tail_entry: u32| -> (Vec<u8>, super::DispatchChain) {
            let threads = [32, 1, 1];
            let cs = CsInterface { threads, ..Default::default() };
            let options = KernelOptions { wave64: false, split: true, mem_windows: 1, sampler_base: 0 };
            let mut lw = Lw::new(&[], &Default::default(), options, &Entry::Compute(cs));
            let (tid, _) = lw.setup_lanes_compute(threads, false);
            let mem = lw.mems[0];
            let i32 = lw.s.m.i32();
            let zero = lw.c(0);
            let one = lw.c(1);
            let three = lw.c(3);
            let seven = lw.c(7);

            let at = lw.s.m.array(8, i32);
            let live_elems: Vec<_> = (0..8u32)
                .map(|k| lw.c(if k == 7 { tail_entry } else { 0x100 + k * 13 }))
                .collect();
            let dead_elems: Vec<_> = (0..8u32).map(|k| lw.c(0xDEAD_0000 + salt * 17 + k)).collect();
            let make_dead = |lw: &mut Lw| {
                let init = lw.s.m.aggregate(at, &dead_elems);
                lw.s.m.global(&format!("dead_lut_{salt}"), at, 0, Some(init), true, 4)
            };
            let make_live = |lw: &mut Lw| {
                let init = lw.s.m.aggregate(at, &live_elems);
                lw.s.m.global("live_lut", at, 0, Some(init), true, 4)
            };
            let live_lut = if dead_first {
                let _ = make_dead(&mut lw);
                make_live(&mut lw)
            } else {
                let l = make_live(&mut lw);
                let _ = make_dead(&mut lw);
                l
            };

            let two = lw.c(2);
            let off = lw.shl(tid, two);
            let seed = lw.addc(tid, salt);
            let mode = lw.and(tid, three);
            let trips = lw.add(mode, one);

            let mut cur = seed;
            let mut pred = lw.cur();
            for step in 0..8u32 {
                let body = lw.s.m.new_block();
                let next = lw.s.m.new_block();
                lw.s.m.br(body);
                lw.s.m.set_block(body);
                let cnt = lw.s.m.phi(i32);
                let acc = lw.s.m.phi(i32);
                lw.s.m.add_incoming(cnt, zero, pred);
                lw.s.m.add_incoming(acc, cur, pred);
                let next_val = if step == 7 {
                    // Downstream piece dynamically indexes `live_lut[tid & 7]`
                    // (potentially touching all 8 entries, including index 7).
                    let idx = lw.and(tid, seven);
                    let gep = lw.s.m.gep(live_lut, &[zero, idx]);
                    let entry = lw.s.m.load(gep, 4);
                    let mix = lw.add(acc, entry);
                    let nz = lw.s.m.f32c((-0.0f32).to_bits());
                    let nz_i = lw.s.m.cast(CastOp::BitCast, nz, i32);
                    lw.s.m.bin(BinOp::Xor, mix, nz_i)
                } else {
                    lw.addc(acc, if step == 0 { salt } else { step + 0x20 })
                };
                let next_cnt = lw.add(cnt, one);
                lw.s.m.add_incoming(cnt, next_cnt, body);
                lw.s.m.add_incoming(acc, next_val, body);
                let again = lw.cmp(Pred::ULt, next_cnt, trips);
                lw.s.m.cond_br(again, body, next);
                lw.s.m.set_block(next);
                cur = next_val;
                pred = next;
            }
            lw.s.raw_store(mem, off, &[cur]);
            lw.s.m.ret();
            lw.s.m.optimize();

            let chain = lw.s.split_compute(threads, None, 24).expect("should split");
            assert!(chain.pieces.len() >= 2, "expected >=2 pieces, got {}", chain.pieces.len());
            for (idx, p) in chain.pieces.iter().enumerate() {
                validate(p).unwrap_or_else(|e| panic!("piece {idx} invalid: {e}"));
            }
            let whole = lw.s.finish_optimized(threads, Some(32)).unwrap();
            validate(&whole).expect("valid whole shader");
            (whole, chain)
        };

        let (whole_a, chain_a) = build(5, false, 0x7777_0001);
        let (whole_b, chain_b) = build(42, true, 0x7777_0001);
        let (whole_c, chain_c) = build(42, true, 0x7777_0099);

        let last_a = chain_a.pieces.last().unwrap();
        let last_b = chain_b.pieces.last().unwrap();
        let last_c = chain_c.pieces.last().unwrap();
        assert_eq!(last_a, last_b, "dynamically indexed tail piece should ignore unused dead_lut and declaration order");
        assert_ne!(last_b, last_c, "changing a dynamically indexed LUT entry must change piece identity");
        assert_ne!(fnv64(last_b), fnv64(last_c));

        let disasm = disassemble(last_b).expect("disassemble canonical tail piece");
        assert!(disasm.contains("live_lut"), "missing live_lut in disassembly");
        assert!(!disasm.contains("dead_lut"), "dead_lut leaked into canonical tail piece");
        assert!(disasm.contains(&format!("{}", 0x7777_0001u32 as i32)), "dynamically indexed entry [7] dropped");

        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        let initial = vec![0u32; 32];
        for (tag, whole, chain) in [("a", &whole_a, &chain_a), ("b", &whole_b, &chain_b), ("c", &whole_c, &chain_c)] {
            let want = runner::dispatch(whole, [1, 1, 1], &[], &initial).unwrap();
            let got = runner::dispatch_split(chain, [1, 1, 1], &initial).unwrap();
            assert_eq!(got, want, "split output mismatch for variant {tag}");
        }
        let gpu_b = runner::dispatch_split(&chain_b, [1, 1, 1], &initial).unwrap();
        let gpu_c = runner::dispatch_split(&chain_c, [1, 1, 1], &initial).unwrap();
        assert_ne!(gpu_b[7], gpu_c[7], "changing dynamically indexed entry [7] must change lane 7 GPU output");
    }

    #[test]
    fn canonical_bitcode_drops_unreachable_phi_operands_and_distinguishes_undef() {
        use crate::gpu::dxil::container::Stage;
        use crate::gpu::dxil::module::Block;
        use crate::gpu::dxil::shader::Shader;
        use crate::gpu::dxil::validate::{disassemble, validate};

        let build = |with_dead_phi: bool, use_undef: bool| -> Vec<u8> {
            let mut s = Shader::new(Stage::Compute, (6, 6));
            let i32 = s.m.i32();
            let out = s.raw_uav(0, 0);
            let tid = s.thread_id_in_group(0);
            let two = s.m.i32c(2);
            let off = s.m.bin(BinOp::Shl, tid, two);
            let live = s.m.i32c(0x2222_3333);

            let entry = Block(0);
            let dead = with_dead_phi.then(|| s.m.new_block());
            let dead_val = with_dead_phi.then(|| s.m.i32c(0xBAD0_DEAD));
            let join = s.m.new_block();
            s.m.br(join);

            if let (Some(db), Some(dv)) = (dead, dead_val) {
                s.m.set_block(db);
                let g_dead = {
                    let at = s.m.array(2, i32);
                    let init = s.m.aggregate(at, &[dv, dv]);
                    s.m.global("unreachable_tbl", at, 0, Some(init), true, 4)
                };
                let zero = s.m.i32c(0);
                let gep = s.m.gep(g_dead, &[zero, zero]);
                let _ = s.m.load(gep, 4);
                s.m.br(join);
            }

            s.m.set_block(join);
            let merged = if with_dead_phi {
                let phi = s.m.phi(i32);
                s.m.add_incoming(phi, live, entry);
                s.m.add_incoming(phi, dead_val.unwrap(), dead.unwrap());
                phi
            } else {
                live
            };
            if use_undef {
                s.raw_store(out, off, &[merged]);
            } else {
                let zero = s.m.i32c(0);
                s.raw_store(out, off, &[merged, zero]);
            }
            s.m.ret();
            s.m.optimize();
            let blob = s.finish_optimized_canonical([32, 1, 1], Some(32)).expect("valid shader");
            validate(&blob).expect("DXC validation");
            blob
        };

        let clean_zero = build(false, false);
        let dead_zero = build(true, false);
        let clean_undef = build(false, true);
        let dis = disassemble(&dead_zero).unwrap();
        assert!(!dis.contains("unreachable_tbl"), "unreachable block global leaked");
        assert!(!dis.contains("-1158620499") && !dis.contains("3134250669"), "unreachable phi constant leaked");
        assert!(disassemble(&clean_undef).unwrap().contains("undef"), "expected undef operand in canonical bitcode");
        assert_eq!(clean_zero, dead_zero, "unreachable block & incoming phi constant must drop cleanly");
        assert_ne!(clean_zero, clean_undef, "undef must remain distinct from initialized store operands");
    }

    #[test]
    fn wave64_metadata_dynamic_bindings_and_split_static_samplers_stay_valid() {
        use crate::gpu::dxil::container::Stage;
        use crate::gpu::dxil::module::Block;
        use crate::gpu::dxil::ops::{CompType, TexKind};
        use crate::gpu::dxil::shader::Shader;
        use crate::gpu::dxil::validate::{disassemble, validate};
        use crate::gpu::recomp::pso::dispatch_chain_root_signature;
        use crate::gpu::recomp::runner;
        use std::mem::transmute_copy;
        use windows::Win32::Graphics::Direct3D::*;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::*;

        let part = |blob: &[u8], cc: &[u8; 4]| -> Vec<u8> {
            let n = u32::from_le_bytes(blob[28..32].try_into().unwrap()) as usize;
            (0..n)
                .find_map(|i| {
                    let off = u32::from_le_bytes(blob[32 + 4 * i..36 + 4 * i].try_into().unwrap()) as usize;
                    (&blob[off..off + 4] == cc).then(|| {
                        let len = u32::from_le_bytes(blob[off + 4..off + 8].try_into().unwrap()) as usize;
                        blob[off + 8..off + 8 + len].to_vec()
                    })
                })
                .expect("container part")
        };

        // 1) Wave64 vs Wave32 metadata (`PSV0` min/max wave + `dx.entryPoints`) under dead decls.
        let build_wave = |wave: u32, with_noise: bool| -> Vec<u8> {
            let mut s = Shader::new(Stage::Compute, (6, 6));
            let u = s.raw_uav(0, 0);
            if with_noise {
                let i32 = s.m.i32();
                let at = s.m.array(2, i32);
                let c1 = s.m.i32c(111);
                let c2 = s.m.i32c(222);
                let init = s.m.aggregate(at, &[c1, c2]);
                let _ = s.m.global("dead_wave_tbl", at, 0, Some(init), true, 4);
                let _ = s.m.i32c(0xDEAD_BEEF);
            }
            let tid = s.thread_id_in_group(0);
            let two = s.m.i32c(2);
            let off = s.m.bin(BinOp::Shl, tid, two);
            s.raw_store(u, off, &[tid]);
            s.m.ret();
            let blob = s.finish_optimized_canonical([wave, 1, 1], Some(wave)).expect("valid compute");
            validate(&blob).expect("DXC validation");
            blob
        };
        let (w64_a, w64_b, w32) = (build_wave(64, false), build_wave(64, true), build_wave(32, true));
        assert_eq!(w64_a, w64_b, "wave64 canonical bitcode must drop unused decls");
        assert_ne!(w64_a, w32, "wave64 vs wave32 must stay distinct");
        let psv64 = part(&w64_a, b"PSV0");
        assert_ne!(psv64, part(&w32, b"PSV0"), "PSV0 wave size must differ");
        assert!(disassemble(&w64_a).unwrap().contains("i32 64"), "dx.entryPoints wave64 flag missing");

        // 2) Non-constant `ResBind` fallback: `referenced_bindings()` returns None so nothing is pruned.
        let mut s_dyn = Shader::new(Stage::Compute, (6, 6));
        let u0 = s_dyn.raw_uav(0, 0);
        let _u1 = s_dyn.raw_uav(1, 0);
        let _unused_cb = s_dyn.cbv(4, 0, 16);
        let i32t = s_dyn.m.i32();
        let i8t = s_dyn.m.i8();
        let i1t = s_dyn.m.i1();
        let bind_ty = s_dyn.m.named_struct("dx.types.ResBind", &[i32t, i32t, i32t, i8t]);
        let dyn_resbind = s_dyn.m.undef(bind_ty);
        let zero = s_dyn.m.i32c(0);
        let int8 = s_dyn.m.int(8);
        let i8p = s_dyn.m.ptr(int8, 0);
        let ht = s_dyn.m.named_struct("dx.types.Handle", &[i8p]);
        let fty = s_dyn.m.func_ty(ht, &[i32t, bind_ty, i32t, i1t]);
        let cf = s_dyn.m.declare("dx.op.createHandleFromBinding", fty, super::super::FnAttrs::ReadNone);
        let op217 = s_dyn.m.i32c(217);
        let fls = s_dyn.m.i1c(false);
        let _ = s_dyn.m.call(cf, &[op217, dyn_resbind, zero, fls]);
        s_dyn.raw_store(u0, zero, &[zero]);
        s_dyn.m.ret();
        assert!(s_dyn.m.referenced_bindings().is_none(), "non-constant ResBind must disable pruning");
        let n_before = s_dyn.bindings.len();
        s_dyn.prune_unused_bindings();
        assert_eq!(s_dyn.bindings.len(), n_before, "non-constant ResBind must keep all bindings");

        // 3) Split compute piece static-sampler subset pruning (`s0` only in piece 0, `s1` only in piece 1)
        // + heap texture sampling (`ResourceHeapIndexing`) + D3D12 PSO creation with shared root signature.
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let uav = s.raw_uav(0, 0);
        let _unused_srv = s.srv_raw(9, 0);
        let s0 = s.sampler(0, 1, false);
        let s1 = s.sampler(1, 1, false);
        let tid = s.thread_id_in_group(0);
        let two = s.m.i32c(2);
        let off = s.m.bin(BinOp::Shl, tid, two);
        let fzero = s.m.f32c(0);
        let z = s.m.i32c(0);
        let one = s.m.i32c(1);
        let three = s.m.i32c(3);
        let tmask = s.m.bin(BinOp::And, tid, three);
        let trips = s.m.bin(BinOp::Add, tmask, one);

        // Piece 0 samples using s0 (`space=1, lower=0`).
        let tex0 = s.heap_texture(z, TexKind::D2, false, CompType::F32, 4, true);
        let sample0 = s.sample_level(tex0, s0, &[fzero, fzero], &[], fzero)[0];
        let mut acc = s.m.cast(CastOp::BitCast, sample0, i32t);

        let mut pred = Block(0);
        for step in 0..4u32 {
            let body = s.m.new_block();
            let next = s.m.new_block();
            s.m.br(body);
            s.m.set_block(body);
            let cnt = s.m.phi(i32t);
            let val = s.m.phi(i32t);
            s.m.add_incoming(cnt, z, pred);
            s.m.add_incoming(val, acc, pred);
            let updated = if step == 3 {
                // Tail piece samples using s1 (`space=1, lower=1`).
                let tex1 = s.heap_texture(tid, TexKind::D2, false, CompType::F32, 4, true);
                let sample1 = s.sample_level(tex1, s1, &[fzero, fzero], &[], fzero)[0];
                let s1_bits = s.m.cast(CastOp::BitCast, sample1, i32t);
                s.m.bin(BinOp::Add, val, s1_bits)
            } else {
                let inc = s.m.i32c(step + 7);
                s.m.bin(BinOp::Add, val, inc)
            };
            let next_cnt = s.m.bin(BinOp::Add, cnt, one);
            s.m.add_incoming(cnt, next_cnt, body);
            s.m.add_incoming(val, updated, body);
            let again = s.m.cmp(Pred::ULt, next_cnt, trips);
            s.m.cond_br(again, body, next);
            s.m.set_block(next);
            acc = updated;
            pred = next;
        }
        s.raw_store(uav, off, &[acc]);
        s.m.ret();
        s.m.optimize();

        let chain = s.split_compute([32, 1, 1], None, 16).expect("split sampler shader");
        assert!(chain.pieces.len() >= 2, "expected >=2 pieces");
        for (idx, p) in chain.pieces.iter().enumerate() {
            validate(p).unwrap_or_else(|e| panic!("piece {idx} invalid: {e}"));
        }
        let d_first = disassemble(&chain.pieces[0]).unwrap();
        let d_last = disassemble(chain.pieces.last().unwrap()).unwrap();
        assert!(!d_first.contains("i32 9, i32 9"), "unused t9 leaked into piece 0");
        assert!(!d_last.contains("i32 9, i32 9"), "unused t9 leaked into tail piece");
        // Piece 0 keeps s0 (lower=0, space=1) and drops s1 (lower=1, space=1).
        assert!(d_first.contains("ResBind { i32 0, i32 0, i32 1, i8 3 }"), "piece 0 missing s0");
        assert!(!d_first.contains("ResBind { i32 1, i32 1, i32 1, i8 3 }"), "piece 0 kept unused s1");
        // Tail piece keeps only s1 (lower=1, space=1) with dx.resources id=0 and drops s0.
        assert!(d_last.contains("ResBind { i32 1, i32 1, i32 1, i8 3 }"), "tail piece missing s1");
        assert!(!d_last.contains("ResBind { i32 0, i32 0, i32 1, i8 3 }"), "tail piece kept unused s0");

        if !runner::wave32_supported().unwrap_or(false) {
            runner::skip_test("no wave32 GPU");
            return;
        }
        let static_samplers = [([0u32; 4], false), ([0x0000_1000u32, 0, 0, 0], false)];
        unsafe {
            let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let sel = crate::gpu::adapter::Selector::from_env_or("auto");
            let (adapter, _) = crate::gpu::adapter::candidates(&factory, &sel).into_iter().next().unwrap();
            let dev = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0).unwrap();
            let rs = dispatch_chain_root_signature(&dev, &static_samplers).expect("chain root sig");
            for (idx, piece) in chain.pieces.iter().enumerate() {
                dev.CreateComputePipelineState::<ID3D12PipelineState>(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: transmute_copy(&rs),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: piece.as_ptr().cast(), BytecodeLength: piece.len() },
                    ..Default::default()
                })
                .unwrap_or_else(|e| panic!("D3D12 CreateComputePipelineState failed on pruned sampler piece {idx}: {e}"));
            }
        }
    }

}
