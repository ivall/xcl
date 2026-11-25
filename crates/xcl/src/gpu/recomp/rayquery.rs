//! Opt-in replacement of verified whole traversal regions with inline RayQuery.

use super::{Mode, Options};
use crate::gpu::{rdna2, shader};

pub const MAGIC: u32 = 0x58445231;
pub const HOST_AS_ROOT: u32 = 6;
pub const SHADOW_RECORD: u64 = 0x9d51ce32ab9a9ab5;

#[derive(Clone, Copy, Debug)]
pub(super) struct Region {
    pub start: u64,
    pub end: u64,
}

pub(super) fn identify(code: &[u8], opt: &Options, insts: &[rdna2::Decoded]) -> Option<Region> {
    if !opt.host_rayquery || !opt.structured {
        return None;
    }
    let hash = shader::fnv64(code);
    let region = if matches!(opt.mode, Mode::Compute(_)) && hash == 0x65cd3172658edef2 && code.len() == 9724 {
        Region { start: 0x13d4, end: 0x1c6c }
    } else {
        #[cfg(test)]
        if matches!(opt.mode, Mode::Kernel) && hash == 0x4f7800379203c280 && code.len() == 2204 {
            return Some(Region { start: 0, end: 2200 });
        }
        return None;
    };
    // No entry into the middle of the replaced region, and a single continuation.
    if insts.iter().any(|d| rdna2::branch_target(d).is_some_and(|target| {
        let inside = d.pc >= region.start && d.pc < region.end;
        (!inside && target > region.start && target < region.end)
            || (inside && (target < region.start || target > region.end))
    })) {
        return None;
    }
    Some(region)
}

pub(super) const LOOKUP: &str = r#"
RaytracingAccelerationStructure xcl_q_scene : register(t0, space2);
// One native TLAS is bound per dispatch. Other guest addresses use the original traversal.
bool xcl_query_matches(uint lo, uint hi) {
    return io.Load(460u) == 0x58445231u && io.Load(452u) == lo && io.Load(456u) == (hi & 0xffffu);
}
"#;

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
mod tests {
    use super::super::{dxc, runner, translate};
    use crate::gpu::{bvh, shader, xbox};

    #[test]
    fn gpu_retail_shadow_region_matches_console_scene() {
        let Some(path) = std::env::var_os("XCL_SAMPLE_RAYQUERY") else {
            runner::skip_test("no XCL_SAMPLE_RAYQUERY");
            return;
        };
        if runner::adapter().is_err() || dxc::location().is_err() {
            runner::skip_test("no GPU or DXC");
            return;
        }
        let blob = std::fs::read(path).unwrap();
        assert_eq!(shader::fnv64(&blob), 0x9d51ce32ab9a9ab5);
        let sh = xbox::parse(&blob).unwrap();
        let f = sh.fssm.iter().find(|f| f.stages.iter().any(|s| s.hw == 6)).unwrap();
        let p = f.programs.first().unwrap();
        let full = &f.data[p.offset..p.offset + p.len];
        println!("retail program hash {:016x}, {} bytes", shader::fnv64(full), full.len());
        let mut insts = Vec::new();
        let mut at = 0;
        while at < full.len() {
            let d = crate::gpu::rdna2::decode(&full[at..], at as u64).unwrap();
            at += d.len; insts.push(d);
        }
        let cfg = super::super::cfg::Cfg::new(&insts).unwrap();
        for scalar in [false, true] {
            use std::collections::BTreeSet;
            let uses = |d: &crate::gpu::rdna2::Decoded| if scalar { crate::gpu::rdna2::sgpr_uses(d) } else { crate::gpu::rdna2::vgpr_uses(d) };
            let mut live = vec![BTreeSet::<u64>::new(); cfg.start.len()];
            loop {
                let mut changed = false;
                for b in (0..cfg.start.len()).rev() {
                    let mut next: BTreeSet<u64> = cfg.succ[b].iter().flat_map(|&s| live[s].iter().copied()).collect();
                    for i in cfg.range(b).rev() {
                        let (r,w) = uses(&insts[i]);
                        for reg in w { next.remove(&reg); }
                        next.extend(r);
                    }
                    changed |= live[b] != next; live[b] = next;
                }
                if !changed { break; }
            }
            let end = insts.iter().position(|d| d.pc == 0x1c6c).unwrap();
            let written: BTreeSet<u64> = insts.iter().filter(|d| d.pc >= 0x13d4 && d.pc < 0x1c6c).flat_map(|d| uses(d).1).collect();
            let out: Vec<_> = written.intersection(&live[cfg.block_of[end]]).copied().collect();
            println!("retail region live {} outputs: {out:?}", if scalar { "scalar" } else { "vector" });
            assert_eq!(out, if scalar { vec![] } else { vec![28, 31] });
        }
        let mode = super::super::interface(&blob, &sh, f).unwrap();
        let mut native = runner::options(mode);
        native.host_rayquery = true;
        native.mem_windows = 4;
        let full_translation = translate(full, &native).unwrap();
        assert!(full_translation.complete(), "{:?}", full_translation.unsupported);
        assert_eq!(full_translation.hlsl.matches("TraceRayInline").count(), 1);
        let dxil = dxc::compile(&full_translation.hlsl, "main", "cs_6_6", &[]).unwrap();
        println!("Full retail shader with RayQuery compiled: {} B DXIL", dxil.len());
        let mut altered = full.to_vec();
        altered[0] ^= 1;
        assert!(super::identify(&altered, &native, &insts).is_none());
        native.host_rayquery = false;
        assert!(super::identify(full, &native, &insts).is_none());
        let mut code = full[0x13d4..0x1c6c].to_vec();
        code.extend_from_slice(&0xbf810000u32.to_le_bytes());
        let opt = runner::options(super::super::Mode::Kernel);
        let t = translate(&code, &opt).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported);
        let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
        let triangle = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]];
        let bottom = bvh::build(&[triangle], 0);
        let top = bvh::build_top(&[bvh::Instance {
            transform: [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
            id: 0, mask: 0xff, contribution: 0, flags: 0,
            blas_address: 0x4000, blas_root: bottom.root, blas_bounds: bottom.bounds(),
        }]);
        let mut memory = vec![0u8; 0x8000];
        memory[0x1000..0x1000 + top.data.len()].copy_from_slice(&top.data);
        memory[0x4000..0x4000 + bottom.data.len()].copy_from_slice(&bottom.data);
        let memory: Vec<u32> = memory.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
        let mut state = runner::KernelState::default();
        state.sgpr[4] = 0x1000;
        for (lane, v) in state.vgpr.iter_mut().enumerate() {
            let origin = if lane % 2 == 0 { [0.25f32, 0.25, 1.] } else { [2., 2., 1.] };
            for (r, value) in [(6, origin[0]), (66, origin[1]), (62, origin[2]),
                (47, origin[0]), (48, origin[1]), (49, origin[2]),
                (11, 0.), (12, 0.), (52, -1.), (50, 0.), (51, 0.), (5, -1.), (8, 10.)] {
                v[r] = value.to_bits();
            }
            for r in [35, 36, 37] { v[r] = u32::MAX; }
        }
        let result = runner::run(&dxil, &state, &memory).unwrap();
        for lane in 0..4 {
            println!("lane {lane}: limit {}, distance {}, flag {}", f32::from_bits(result.vgpr[lane][8]),
                f32::from_bits(result.vgpr[lane][28]), f32::from_bits(result.vgpr[lane][31]));
        }
        for lane in (0..64).step_by(2) {
            assert_eq!(f32::from_bits(result.vgpr[lane][8]), 1.);
            assert_eq!(f32::from_bits(result.vgpr[lane][28]), 1.);
        }
    }
}
