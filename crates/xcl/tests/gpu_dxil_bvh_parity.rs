//! GPU differential parity tests between the direct RDNA2 → DXIL translator
//! (`xcl::gpu::dxil::kernel::translate`, validated with `dxil.dll`) and the HLSL + DXC path
//! (`xcl::gpu::recomp`) for `IMAGE_BVH_INTERSECT_RAY` and `IMAGE_BVH64_INTERSECT_RAY`:
//! Box32 (sorted/unsorted, NaN/inverted boxes, grazing rays, `t_max` culling), Box16,
//! rotated triangle pairs, `EXEC` inactive-lane masking, and 2-window UMA addressing.

use std::ops::Range;

use xcl::gpu::bvh;
use xcl::gpu::desc::bvh_words;
use xcl::gpu::dxil::kernel::{translate as direct_translate, KernelOptions};
use xcl::gpu::dxil::validate;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner::{self, KernelResult, KernelState};
use xcl::gpu::recomp::{dxc, translate as hlsl_translate, Options};
use xcl::gpu::uma::Uma;

fn half_bits(x: f32) -> u32 {
    let b = x.to_bits();
    let s = (b >> 16) & 0x8000;
    let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let m = b & 0x7F_FFFF;
    if e <= 0 {
        s
    } else if e >= 31 {
        s | 0x7C00
    } else {
        s | ((e as u32) << 10) | (m >> 13)
    }
}

fn gpu_modes() -> Vec<(bool, bool)> {
    match (runner::adapter(), runner::wave64_programs_supported(), dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => {
            if runner::wave64_supported() == Ok(true) {
                vec![(false, false), (true, false)]
            } else {
                vec![(false, false), (true, true)]
            }
        }
        (a, w, d) => {
            runner::skip_test(if a.is_err() || w.is_err() {
                "no GPU"
            } else if d.is_err() {
                "no DXC"
            } else {
                "no wave64"
            });
            Vec::new()
        }
    }
}

fn compile_both(label: &str, code: &[u8], wave64: bool, split: bool, mem_windows: u32) -> (Vec<u8>, Vec<u8>) {
    let hlsl_opt = Options { wave64, split_wave64: split, mem_windows, ..Default::default() };
    let t = hlsl_translate(code, &hlsl_opt).unwrap_or_else(|e| panic!("{label}: hlsl translate: {e}"));
    assert!(t.complete(), "{label}: hlsl incomplete: {:?}", t.unsupported_why);
    let hlsl_dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[])
        .unwrap_or_else(|e| panic!("{label}: dxc compile: {e}"));

    let direct_dxil = direct_translate(
        code,
        KernelOptions { wave64, split, mem_windows, sampler_base: 0 },
    )
    .unwrap_or_else(|e| panic!("{label} (w64={wave64} split={split}): direct translate: {e}"));
    validate::validate(&direct_dxil)
        .unwrap_or_else(|e| panic!("{label} (w64={wave64} split={split}): dxil validate: {e}"));
    (hlsl_dxil, direct_dxil)
}

fn assert_vgpr_parity(
    label: &str,
    r_direct: &KernelResult,
    r_hlsl: &KernelResult,
    lanes: usize,
    vgprs: Range<usize>,
) {
    for l in 0..lanes {
        for v in vgprs.clone() {
            assert_eq!(
                r_direct.vgpr[l][v], r_hlsl.vgpr[l][v],
                "{label} lane {l} v{v}: direct={:#x} hlsl={:#x}",
                r_direct.vgpr[l][v], r_hlsl.vgpr[l][v]
            );
        }
    }
}

fn mimg(name: &str, vdata: u64, vaddr: u64, srsrc: u64) -> Vec<u8> {
    encode(name, "ENC_MIMG", &[("VDATA", vdata), ("VADDR", vaddr), ("SRSRC", srsrc), ("DMASK", 15)], None)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn direct_dxil_bvh_box32_box16_and_exec_mask_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    let base = 0x1000usize;
    let mut mem = vec![0u32; 0x2000 / 4];
    let put = |mem: &mut Vec<u32>, at: usize, vals: &[u32]| {
        mem[(base + at) / 4..(base + at) / 4 + vals.len()].copy_from_slice(vals);
    };
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let (root, box16, tri_a, tri_b) = (5u32, (0x80u32 >> 3) | 4, 0xC0u32 >> 3, 0x100u32 >> 3);

    // Box32 at 0: child0 (t=5), child1 box16 (t=2), child2 NaN-disabled, child3 inverted
    put(&mut mem, 0, &[tri_a, box16, tri_b, 0x140u32 >> 3]);
    put(&mut mem, 16, &ff(&[-1.0, -1.0, 5.0, 1.0, 1.0, 6.0]));
    put(&mut mem, 40, &ff(&[-1.0, -1.0, 2.0, 1.0, 1.0, 3.0]));
    put(&mut mem, 64, &ff(&[f32::NAN, -1.0, 1.0, 1.0, 1.0, 2.0]));
    put(&mut mem, 88, &ff(&[2.0, -1.0, 1.0, -2.0, 1.0, 2.0]));
    // Box16 at 0x80: one valid child in z=6..7
    put(&mut mem, 0x80, &[tri_b, u32::MAX, u32::MAX, u32::MAX]);
    put(&mut mem, 0x90, &[
        half_bits(-1.0) | (half_bits(-1.0) << 16),
        half_bits(6.0) | (half_bits(1.0) << 16),
        half_bits(1.0) | (half_bits(7.0) << 16),
    ]);

    let mut code = Vec::new();
    // Keep lanes 0..23 active; lanes 24+ must preserve their sentinel VGPRs
    code.extend(encode("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 128 + 24), ("VSRC1", 90)], None).unwrap());
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 100, 0, 2));  // box32 sorted
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 104, 0, 3));  // box32 unsorted
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 108, 20, 2)); // box16
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[(base >> 8) as u32, 1 << 31, u32::MAX, 0]);
    st.sgpr[12..16].copy_from_slice(&[(base >> 8) as u32, 0, u32::MAX, 0]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[90] = l as u32;
        let (ox, tmax) = match l % 4 {
            0 => (0.0f32, 100.0f32),  // hits both boxes
            1 => (0.0f32, 3.5f32),    // tmax culls z=5..6 box and box16 z=6..7
            2 => (-1.0f32, 100.0f32), // grazing edge (o.x == mn.x, dx == 0, inv.x == +inf)
            _ => (5.0f32, 100.0f32),  // misses all boxes
        };
        let ray = [tmax, ox, 0.0, 0.0, 0.0, 0.0, 1.0, f32::INFINITY, f32::INFINITY, 1.0];
        r[0] = root;
        r[20] = box16;
        for (k, x) in ray.iter().enumerate() {
            r[1 + k] = x.to_bits();
            r[21 + k] = x.to_bits();
        }
        for reg in 100..112 {
            r[reg] = 0xDEAD_BEEF;
        }
    }

    let data: Vec<u8> = mem[base / 4..].iter().flat_map(|x| x.to_le_bytes()).collect();
    for (wave64, split) in modes {
        let (hlsl_dxil, direct_dxil) = compile_both("bvh_box32_box16", &code, wave64, split, 1);
        let r_hlsl = runner::run(&hlsl_dxil, &st, &mem).unwrap();
        let r_direct = runner::run(&direct_dxil, &st, &mem).unwrap();
        let lanes = if wave64 { 64 } else { 32 };
        assert_vgpr_parity("bvh_box32_box16", &r_direct, &r_hlsl, lanes, 100..112);

        for l in 0..lanes {
            if l >= 24 {
                assert_eq!(&r_direct.vgpr[l][100..112], &[0xDEAD_BEEF; 12], "inactive lane {l}");
                continue;
            }
            let (ox, tmax) = match l % 4 {
                0 => (0.0, 100.0),
                1 => (0.0, 3.5),
                2 => (-1.0, 100.0),
                _ => (5.0, 100.0),
            };
            let o = [ox, 0.0, 0.0];
            let d = [0.0, 0.0, 1.0];
            let inv = [f32::INFINITY, f32::INFINITY, 1.0];
            assert_eq!(r_direct.vgpr[l][100..104], bvh::intersect_node(&data, root, tmax, o, d, inv, true));
            assert_eq!(r_direct.vgpr[l][104..108], bvh::intersect_node(&data, root, tmax, o, d, inv, false));
            assert_eq!(r_direct.vgpr[l][108..112], bvh::intersect_node(&data, box16, tmax, o, d, inv, true));
        }
    }
}

#[test]
fn direct_dxil_bvh_triangle_pair_and_bvh64_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    let base = 0x1000usize;
    let mut mem = vec![0u32; 0x2000 / 4];
    let put = |mem: &mut Vec<u32>, at: usize, vals: &[u32]| {
        mem[(base + at) / 4..(base + at) / 4 + vals.len()].copy_from_slice(vals);
    };
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    // Rotated quad triangle pair at +0x40 in z=5
    let (p0, p1, p2, p3) = ([-1.0f32, -1.0, 5.0], [1.0, -1.0, 5.0], [-1.0, 1.0, 5.0], [1.0, 1.0, 5.0]);
    let pair = 0x40usize;
    put(&mut mem, pair, &ff(&[p1, p2, p0, p3].concat()));
    put(&mut mem, pair + 48, &[0, 11, 12, 0x0104]);
    let (t0, t1) = ((pair as u32) >> 3, ((pair as u32) >> 3) | 1);

    let mut code = Vec::new();
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 100, 0, 2));   // 32-bit node ptr (tri0 or tri1)
    code.extend(mimg("IMAGE_BVH64_INTERSECT_RAY", 104, 20, 2)); // 64-bit node ptr (same triangle)
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[(base >> 8) as u32, 1 << 31, u32::MAX, 0]);
    let mut state = 0x9E37_79B9u32;
    let mut next_f = |lo: f32, hi: f32| {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        lo + (state as f32 / u32::MAX as f32) * (hi - lo)
    };
    let mut rays = Vec::new();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let node = if l % 2 == 0 { t0 } else { t1 };
        let o = [next_f(-1.3, 1.3), next_f(-1.3, 1.3), 0.0];
        let d = [next_f(-0.08, 0.08), next_f(-0.08, 0.08), 1.0];
        let inv = d.map(|x| 1.0 / x);
        let ray = [100.0f32, o[0], o[1], o[2], d[0], d[1], d[2], inv[0], inv[1], inv[2]];
        r[0] = node;
        r[20] = node;
        r[21] = 0;
        for (k, x) in ray.iter().enumerate() {
            r[1 + k] = x.to_bits();
            r[22 + k] = x.to_bits();
        }
        rays.push((node, o, d, inv));
    }

    let data: Vec<u8> = mem[base / 4..].iter().flat_map(|x| x.to_le_bytes()).collect();
    for (wave64, split) in modes {
        let (hlsl_dxil, direct_dxil) = compile_both("bvh_tri_pair", &code, wave64, split, 1);
        let r_hlsl = runner::run(&hlsl_dxil, &st, &mem).unwrap();
        let r_direct = runner::run(&direct_dxil, &st, &mem).unwrap();
        let lanes = if wave64 { 64 } else { 32 };
        let mut hits = 0usize;
        for l in 0..lanes {
            let (node, o, d, inv) = rays[l];
            let cpu = bvh::intersect_node(&data, node, 100.0, o, d, inv, true).map(f32::from_bits);
            let dir32: [f32; 4] = [0, 1, 2, 3].map(|k| f32::from_bits(r_direct.vgpr[l][100 + k]));
            let hl32: [f32; 4] = [0, 1, 2, 3].map(|k| f32::from_bits(r_hlsl.vgpr[l][100 + k]));
            let dir64: [u32; 4] = r_direct.vgpr[l][104..108].try_into().unwrap();
            assert_eq!(dir64, r_direct.vgpr[l][100..104], "lane {l}: bvh64 vs bvh32");
            assert_eq!(dir32[0].is_infinite(), hl32[0].is_infinite(), "lane {l}: hit/miss direct vs hlsl");
            assert_eq!(dir32[0].is_infinite(), cpu[0].is_infinite(), "lane {l}: hit/miss direct vs cpu");
            if !cpu[0].is_infinite() {
                for k in 0..4 {
                    assert!((dir32[k] / dir32[1] - hl32[k] / hl32[1]).abs() < 1e-5, "lane {l}: direct {dir32:?} vs hlsl {hl32:?}");
                    assert!((dir32[k] / dir32[1] - cpu[k] / cpu[1]).abs() < 1e-5, "lane {l}: direct {dir32:?} vs cpu {cpu:?}");
                }
                hits += 1;
            }
        }
        assert!(hits >= 4, "expected triangle hits, got {hits}");
    }
}

#[test]
fn direct_dxil_bvh64_multi_window_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    let Some(arena) = Uma::with_windows(2, 1 << 20) else {
        println!("UMA 2-window unavailable – skipping");
        runner::skip_test("no GPU");
        return;
    };
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    assert_eq!(w1 - w0, 1 << 32);
    let write_words = |addr: u64, vals: &[u32]| unsafe {
        let p = addr as *mut u32;
        for (k, &v) in vals.iter().enumerate() {
            *p.add(k) = v;
        }
    };
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let base_off = 0x1000u64;
    let bvh_addr = w1 + base_off;
    let tri_rel = 0x80u32 >> 3;
    write_words(bvh_addr, &[tri_rel, u32::MAX, u32::MAX, u32::MAX]);
    write_words(bvh_addr + 16, &ff(&[-1.0, -1.0, 5.0, 1.0, 1.0, 6.0]));
    write_words(bvh_addr + 0x80, &ff(&[-1.0, -1.0, 5.0, 1.0, -1.0, 5.0, -1.0, 1.0, 5.0]));
    write_words(bvh_addr + 0x80 + 60, &[9]);
    // Poison window 0 at the same offset so a dropped upper window address fails loudly
    write_words(w0 + base_off, &[0xDEAD_BEEF, u32::MAX, u32::MAX, u32::MAX]);
    write_words(w0 + base_off + 16, &ff(&[-100.0, -100.0, 1.0, 100.0, 100.0, 2.0]));

    let mut code = Vec::new();
    code.extend(mimg("IMAGE_BVH64_INTERSECT_RAY", 100, 0, 2)); // 64-bit node ptr into window 1
    code.extend(mimg("IMAGE_BVH64_INTERSECT_RAY", 104, 20, 2)); // 64-bit tri ptr into window 1
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 108, 40, 3)); // 48-bit descriptor base into window 1
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&bvh_words(0, true));
    st.sgpr[12..16].copy_from_slice(&bvh_words(bvh_addr, true));
    let root64 = (bvh_addr >> 3) | 5;
    let tri64 = (bvh_addr + 0x80) >> 3;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let ox = if l % 2 == 0 { -1.0f32 } else { 0.0f32 };
        let ray = [100.0f32, ox, 0.0, 0.0, 0.0, 0.0, 1.0, f32::INFINITY, f32::INFINITY, 1.0];
        r[0] = root64 as u32;
        r[1] = (root64 >> 32) as u32;
        r[20] = tri64 as u32;
        r[21] = (tri64 >> 32) as u32;
        r[40] = 5;
        for (k, x) in ray.iter().enumerate() {
            r[2 + k] = x.to_bits();
            r[22 + k] = x.to_bits();
            r[41 + k] = x.to_bits();
        }
    }

    for (wave64, split) in modes {
        let (hlsl_dxil, direct_dxil) = compile_both("bvh64_multi_window", &code, wave64, split, 2);
        let r_hlsl = runner::run_in_arena(&hlsl_dxil, &st, &arena).unwrap();
        let r_direct = runner::run_in_arena(&direct_dxil, &st, &arena).unwrap();
        let lanes = if wave64 { 64 } else { 32 };
        assert_vgpr_parity("bvh64_multi_window", &r_direct, &r_hlsl, lanes, 100..112);
        for l in 0..lanes {
            assert_eq!(&r_direct.vgpr[l][100..104], &[tri_rel, u32::MAX, u32::MAX, u32::MAX]);
            assert_eq!(&r_direct.vgpr[l][108..112], &[tri_rel, u32::MAX, u32::MAX, u32::MAX]);
            let tri: [f32; 4] = [0, 1, 2, 3].map(|k| f32::from_bits(r_direct.vgpr[l][104 + k]));
            assert!(!tri[0].is_infinite() && ((tri[0] / tri[1]) - 5.0).abs() < 1e-4);
        }
    }
}
