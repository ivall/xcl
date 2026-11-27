//! A conservative summary of directly indexed resource descriptors in emitted DXIL.
//! Register bindings and direct heap indices have different meanings. A finite set here
//! does not resolve root tables, fixed-function resources or command-list dependencies.

use std::collections::BTreeSet;
use std::sync::{Mutex, OnceLock};

use super::{Module, Op, Value};

const MAX_SLOTS: usize = 256;
const MAX_INDEX_NODES: usize = 64;

#[derive(Debug, PartialEq, Eq)]
struct Access {
    /// None means a shader may index anywhere in the resource heap.
    heap_slots: Option<BTreeSet<u32>>,
    /// Legacy or root-table bindings need a separate resolution step.
    register_bindings: bool,
}

impl Module {
    fn descriptor_access(&self) -> Access {
        let (order, _) = self.block_layout();
        let mut access = Access { heap_slots: Some(BTreeSet::new()), register_bindings: false };
        for i in order.into_iter().flat_map(|b| &self.blocks[b]) {
            let Op::Call(f, args) = &self.insts[*i as usize].op else { continue };
            match self.funcs[*f as usize].name.as_str() {
                "dx.op.createHandleFromHeap" => {
                    let Some(slots) = &mut access.heap_slots else { continue };
                    let mut remaining = MAX_INDEX_NODES;
                    let operands = args.get(1).zip(args.get(2));
                    match operands.and_then(|(&index, &sampler)| self.const_u64(sampler).map(|kind| (index, kind))) {
                        Some((_, 1)) => (),
                        Some((index, 0)) if self.collect_heap_slots(index, &mut remaining, slots) => (),
                        _ => access.heap_slots = None,
                    }
                }
                "dx.op.createHandle" | "dx.op.createHandleFromBinding" => access.register_bindings = true,
                name if name.starts_with("dx.op.createHandle") => {
                    access.heap_slots = None;
                    access.register_bindings = true;
                }
                _ => (),
            }
        }
        access
    }

    fn collect_heap_slots(&self, value: Value, remaining: &mut usize, slots: &mut BTreeSet<u32>) -> bool {
        // A cycle, unsupported expression or too large a proof stays unknown. The budget
        // also bounds work on expressions that refer to the same value repeatedly.
        let Some(next) = remaining.checked_sub(1) else { return false };
        *remaining = next;
        if let Some(slot) = self.const_u64(value).and_then(|n| u32::try_from(n).ok()) {
            slots.insert(slot);
            return slots.len() <= MAX_SLOTS;
        }
        let Value::Inst(i) = value else { return false };
        match &self.insts[i as usize].op {
            Op::Select(_, yes, no) => self.collect_heap_slots(*yes, remaining, slots) && self.collect_heap_slots(*no, remaining, slots),
            Op::Phi(edges) => !edges.is_empty() && edges.iter().all(|&(incoming, _)| self.collect_heap_slots(incoming, remaining, slots)),
            _ => false,
        }
    }
}

#[derive(Default)]
struct Counts {
    no_heap: u64,
    finite_heap: u64,
    unknown_heap: u64,
    register_bindings: u64,
}

static ENABLED: OnceLock<bool> = OnceLock::new();
static COUNTS: Mutex<Counts> = Mutex::new(Counts { no_heap: 0, finite_heap: 0, unknown_heap: 0, register_bindings: 0 });

pub(super) fn observe(module: &Module) {
    if !*ENABLED.get_or_init(|| std::env::var_os("XCL_DXIL_HEAP_ACCESS").is_some_and(|v| v == "1")) {
        return;
    }
    let access = module.descriptor_access();
    let mut counts = COUNTS.lock().unwrap();
    match access.heap_slots {
        Some(slots) if slots.is_empty() => counts.no_heap += 1,
        Some(_) => counts.finite_heap += 1,
        None => counts.unknown_heap += 1,
    }
    counts.register_bindings += u64::from(access.register_bindings);
    let total = counts.no_heap + counts.finite_heap + counts.unknown_heap;
    let report = total <= 4 || total.is_multiple_of(64);
    let (empty, finite, unknown, register) = (counts.no_heap, counts.finite_heap, counts.unknown_heap, counts.register_bindings);
    drop(counts);
    if report {
        crate::xlog_info!("DXIL heap access (emitted modules): {total} total, {empty} no resource heap, {finite} finite slots, {unknown} unknown slots, {register} with register bindings");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::dxil::{container::Stage, module::Pred, shader::Shader};

    #[test]
    fn collects_all_select_and_phi_slots_and_keeps_register_bindings_separate() {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let thread = s.thread_id(0);
        let a = s.m.i32c(20);
        let b = s.m.i32c(24);
        let c = s.m.i32c(40);
        let cond = s.m.cmp(Pred::ULt, thread, a);
        let selected = s.m.select(cond, a, b);
        let entry = s.m.current_block();
        let other = s.m.new_block();
        let join = s.m.new_block();
        s.m.cond_br(cond, other, join);
        s.m.set_block(other);
        s.m.br(join);
        s.m.set_block(join);
        let ty = s.m.i32();
        let value = s.m.phi(ty);
        s.m.add_incoming(value, selected, entry);
        s.m.add_incoming(value, c, other);
        s.handle_from_heap(value, false, true);
        s.handle_from_heap(a, false, false);
        s.handle_from_heap(thread, true, true);
        s.raw_uav(1, 0);
        s.m.ret();
        let dead = s.m.new_block();
        s.m.set_block(dead);
        s.handle_from_heap(thread, false, true);
        s.m.ret();
        assert_eq!(s.m.descriptor_access(), Access { heap_slots: Some(BTreeSet::from([20, 24, 40])), register_bindings: true });
    }

    #[test]
    fn one_dynamic_resource_index_makes_the_whole_heap_possible() {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let thread = s.thread_id(0);
        let fixed = s.m.i32c(32);
        s.handle_from_heap(fixed, false, false);
        s.handle_from_heap(thread, false, true);
        s.m.ret();
        assert_eq!(s.m.descriptor_access(), Access { heap_slots: None, register_bindings: false });
    }

    #[test]
    fn a_loop_carried_index_or_an_oversized_set_stays_unknown() {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let entry = s.m.current_block();
        let body = s.m.new_block();
        s.m.br(body);
        s.m.set_block(body);
        let ty = s.m.i32();
        let index = s.m.phi(ty);
        let zero = s.m.i32c(0);
        s.m.add_incoming(index, zero, entry);
        s.m.add_incoming(index, index, body);
        s.handle_from_heap(index, false, false);
        s.m.br(body);
        assert_eq!(s.m.descriptor_access().heap_slots, None);

        let mut s = Shader::new(Stage::Compute, (6, 6));
        for n in 0..=MAX_SLOTS {
            let index = s.m.i32c(n as u32);
            s.handle_from_heap(index, false, false);
        }
        s.m.ret();
        assert_eq!(s.m.descriptor_access().heap_slots, None);
    }
}
