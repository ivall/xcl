//! End-to-end GPU tests for 64-bit typed `IMAGE_ATOMIC_*` direct DXIL lowering (#1948, #1269).
//!
//! Exercises all 16 64-bit typed `MIMG` atomics (`ADD`, `SUB`, `AND`, `OR`, `XOR`, `UMIN`,
//! `UMAX`, `SMIN`, `SMAX`, `SWAP`, `INC`, `DEC`, `FMIN`, `FMAX`, `CMPSWAP`, `FCMPSWAP`) on
//! `DXGI_FORMAT_R32G32_UINT` `Texture2D` and `Texture2DArray` UAVs in both wave32 and wave64
//! (native or split) kernels, plus a 32-bit `IMAGE_ATOMIC_CMPSWAP` (`DMASK=0x3`) control.
//! Skips cleanly via `runner::skip_test` when the adapter lacks SM 6.6 64-bit typed/heap atomics.

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32_UINT};

use xcl::gpu::dxil::kernel::{self, KernelOptions};
use xcl::gpu::dxil::validate;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner::{self, KernelState, TestTexture, TexDim};
use xcl::hle::d3d12x::real;

fn mimg(name: &str, fields: &[(&str, u64)]) -> Vec<u8> {
    encode(name, "ENC_MIMG", fields, None).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn endpgm() -> Vec<u8> {
    encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap()
}

/// XCL image T#: word 0 = SRV heap index (`2 * i`, UAV at `2 * i + 1`), word 1 = `format << 20`,
/// word 3 = `ty << 28` (`9` = 2D, `13` = 2DArray).
fn tsharp(index: u32, format: u32, ty: u32) -> [u32; 8] {
    [index, format << 20, 0, ty << 28, 0, 0, 0, 0]
}

fn set_u64(v: &mut [u32], reg: usize, x: u64) {
    v[reg] = x as u32;
    v[reg + 1] = (x >> 32) as u32;
}

fn get_u64(v: &[u32], reg: usize) -> u64 {
    v[reg] as u64 | ((v[reg + 1] as u64) << 32)
}

fn tex_u64(bytes: &[u8], idx: usize) -> u64 {
    u64::from_le_bytes(bytes[idx * 8..idx * 8 + 8].try_into().unwrap())
}

fn u64_bytes(vals: &[u64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Returns the `(wave64, split)` kernel modes supported on this GPU, or `None` after calling
/// `runner::skip_test` when the adapter lacks GPU / wave32+wave64 / 64-bit typed heap atomics.
fn supported_modes() -> Option<Vec<(bool, bool)>> {
    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real {
        runner::skip_test("no GPU");
        return None;
    }
    let raw = real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }?.clone();
    unsafe {
        let mut o9 = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
        let ok9 = dev
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &mut o9 as *mut _ as _, std::mem::size_of_val(&o9) as u32)
            .is_ok()
            && o9.AtomicInt64OnTypedResourceSupported.as_bool();
        let mut o11 = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
        let ok11 = dev
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS11, &mut o11 as *mut _ as _, std::mem::size_of_val(&o11) as u32)
            .is_ok()
            && o11.AtomicInt64OnDescriptorHeapResourceSupported.as_bool();
        if !ok9 || !ok11 {
            runner::skip_test("no 64-bit typed heap atomics");
            return None;
        }
    }
    let mut modes = Vec::new();
    if runner::wave32_supported() == Ok(true) {
        modes.push((false, false));
    }
    if runner::wave64_programs_supported() == Ok(true) {
        modes.push((true, runner::split_wave64()));
    }
    if modes.is_empty() {
        runner::skip_test("no 32- or 64-lane waves");
        return None;
    }
    Some(modes)
}

fn translate_and_validate(code: &[u8], wave64: bool, split: bool) -> Vec<u8> {
    let dxil = kernel::translate(code, KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 })
        .unwrap_or_else(|e| panic!("translate (wave64={wave64}, split={split}): {e}"));
    validate::validate(&dxil).unwrap_or_else(|e| panic!("validate (wave64={wave64}, split={split}): {e}"));
    dxil
}

/// Contended 64-bit `ADD`, `SUB`, `AND`, `OR`, `XOR`, `UMIN`, `UMAX`, `SMIN`, `SMAX`, `SWAP`
/// on `Texture2D` (`DIM=1`) and `Texture2DArray` (`DIM=5`) UAVs, verifying 32-bit boundary
/// carries/borrows, signed vs unsigned ordering, `GLC=1` old-value history, `GLC=0` non-write,
/// and `EXEC=0` inactive-lane preservation.
#[test]
fn gpu_image_atomic64_integer_and_bitwise_ops_on_d2_and_d2array() {
    let Some(modes) = supported_modes() else { return };

    // Tex 0 (SRSRC=4, s[16:23]): 8x1 D2 R32G32_UINT UAV
    //   (0,0): ADD (GLC=1, v10:v11)
    //   (1,0): SUB (GLC=1, v12:v13)
    //   (2,0): AND (GLC=0, v14:v15)
    //   (3,0): OR  (GLC=0, v16:v17)
    //   (4,0): XOR (GLC=1, v18:v19)
    //   (5,0): SWAP (GLC=1, v20:v21)
    // Tex 1 (SRSRC=6, s[24:31]): 2x1x2 D2Array R32G32_UINT UAV
    //   (0,0,0): UMIN (GLC=1, v22:v23)
    //   (1,0,0): UMAX (GLC=0, v24:v25)
    //   (0,0,1): SMIN (GLC=1, v26:v27)
    //   (1,0,1): SMAX (GLC=1, v28:v29)
    let mut code = Vec::new();
    let d2_ops = [
        ("IMAGE_ATOMIC_ADD", 10, 0, 1),
        ("IMAGE_ATOMIC_SUB", 12, 2, 1),
        ("IMAGE_ATOMIC_AND", 14, 4, 0),
        ("IMAGE_ATOMIC_OR", 16, 6, 0),
        ("IMAGE_ATOMIC_XOR", 18, 8, 1),
        ("IMAGE_ATOMIC_SWAP", 20, 30, 1),
    ];
    for (op, vd, va, glc) in d2_ops {
        code.extend(mimg(op, &[("VDATA", vd), ("VADDR", va), ("SRSRC", 4), ("DMASK", 0x3), ("DIM", 1), ("UNORM", 1), ("GLC", glc)]));
    }
    let arr_ops = [
        ("IMAGE_ATOMIC_UMIN", 22, 32, 1),
        ("IMAGE_ATOMIC_UMAX", 24, 35, 0),
        ("IMAGE_ATOMIC_SMIN", 26, 38, 1),
        ("IMAGE_ATOMIC_SMAX", 28, 41, 1),
    ];
    for (op, vd, va, glc) in arr_ops {
        code.extend(mimg(op, &[("VDATA", vd), ("VADDR", va), ("SRSRC", 6), ("DMASK", 0x3), ("DIM", 5), ("UNORM", 1), ("GLC", glc)]));
    }
    code.extend(endpgm());

    const ADD_INIT: u64 = 0x0000_0000_FFFF_FFF0;
    const ADD_STEP: u64 = 0x0000_0001_0000_0005;
    const SUB_INIT: u64 = 0x0000_0080_0000_0010;
    const SUB_STEP: u64 = 0x0000_0001_0000_0003;
    const AND_INIT: u64 = 0xFFFF_FFFF_FFFF_FFFF;
    const OR_INIT: u64 = 0x0000_0000_0000_0000;
    const XOR_INIT: u64 = 0x0123_4567_89AB_CDEF;
    const SWAP_INIT: u64 = 0xCAFE_BABE_1234_5678;

    for (wave64, split) in modes {
        let lanes = if wave64 { 64usize } else { 32usize };
        let dxil = translate_and_validate(&code, wave64, split);

        // Disable lanes where `l % 7 == 0` in EXEC so inactive lanes are tested for both
        // no-texel-mutation and untouched VDATA registers.
        let mut exec_mask = 0u64;
        let mut active_lanes = Vec::new();
        for l in 0..lanes {
            if l % 7 != 0 {
                exec_mask |= 1u64 << l;
                active_lanes.push(l);
            }
        }
        let n_act = active_lanes.len() as u64;

        let mut st = KernelState::default();
        st.sgpr[126] = exec_mask as u32;
        st.sgpr[127] = if wave64 { (exec_mask >> 32) as u32 } else { 0 };
        st.sgpr[16..24].copy_from_slice(&tsharp(0, 62, 9));  // D2 at heap 0 (UAV at 1)
        st.sgpr[24..32].copy_from_slice(&tsharp(2, 62, 13)); // D2Array at heap 2 (UAV at 3)

        let mut want_and = AND_INIT;
        let mut want_or = OR_INIT;
        let mut want_xor = XOR_INIT;
        let mut want_umin = u64::MAX;
        let mut want_umax = 0u64;
        let mut want_smin = i64::MAX;
        let mut want_smax = i64::MIN;
        let mut swap_in = vec![0u64; lanes];

        for l in 0..lanes {
            let v = &mut st.vgpr[l];
            // Coordinates for D2 texels (x, 0)
            for (va, x) in [(0, 0), (2, 1), (4, 2), (6, 3), (8, 4), (30, 5)] {
                v[va] = x;
                v[va + 1] = 0;
            }
            // Coordinates for D2Array texels (x, y, slice)
            for (va, (x, y, z)) in [(32, (0, 0, 0)), (35, (1, 0, 0)), (38, (0, 0, 1)), (41, (1, 0, 1))] {
                v[va] = x;
                v[va + 1] = y;
                v[va + 2] = z;
            }

            let is_act = (exec_mask >> l) & 1 != 0;
            let sentinel = 0xDEAD_0000_0000_0000u64 | (l as u64 * 0x101);
            let and_v = if is_act { !(1u64 << (l % 64)) } else { 0 };
            let or_v = if is_act { 1u64 << (l % 64) } else { u64::MAX };
            let xor_v = 0x1000_0001_0000_0001u64.wrapping_mul(l as u64 + 1);
            let swp_v = 0xA000_0000_0000_0000u64 | (l as u64 + 1);
            swap_in[l] = swp_v;

            // Signed/unsigned values include both negative i64 (high bit set) and large positive i64,
            // plus poison values on inactive lanes that would skew min/max if inactive lanes ran.
            let u_val = if is_act {
                match l % 4 {
                    0 => 0x0000_0002_0000_0000u64 + l as u64,
                    1 => 0x8000_0000_0000_0000u64 + l as u64, // negative as i64, huge as u64
                    2 => 0xFFFF_FFFF_FFFF_FF00u64 + l as u64, // -256+l as i64, huge as u64
                    _ => 0x0000_0000_0000_0100u64 + l as u64,
                }
            } else {
                0 // inactive poison for umin
            };
            let s_val: i64 = if is_act {
                match l % 4 {
                    0 => -1_000_000_000_000i64 + l as i64,
                    1 => 5_000_000_000_000i64 + l as i64,
                    2 => -50i64 - l as i64,
                    _ => 100i64 + l as i64,
                }
            } else {
                i64::MIN // inactive poison for smin
            };

            set_u64(v, 10, if is_act { ADD_STEP } else { sentinel });
            set_u64(v, 12, if is_act { SUB_STEP } else { sentinel });
            set_u64(v, 14, and_v);
            set_u64(v, 16, or_v);
            set_u64(v, 18, if is_act { xor_v } else { sentinel });
            set_u64(v, 20, if is_act { swp_v } else { sentinel });
            set_u64(v, 22, u_val);
            set_u64(v, 24, if is_act { u_val } else { u64::MAX });
            set_u64(v, 26, s_val as u64);
            set_u64(v, 28, if is_act { s_val as u64 } else { i64::MAX as u64 });

            if is_act {
                want_and &= and_v;
                want_or |= or_v;
                want_xor ^= xor_v;
                want_umin = want_umin.min(u_val);
                want_umax = want_umax.max(u_val);
                want_smin = want_smin.min(s_val);
                want_smax = want_smax.max(s_val);
            }
        }

        let textures = vec![
            TestTexture {
                dim: TexDim::D2, format: DXGI_FORMAT_R32G32_UINT, width: 8, height: 1, depth: 1, texel_bytes: 8, uav: true,
                data: u64_bytes(&[ADD_INIT, SUB_INIT, AND_INIT, OR_INIT, XOR_INIT, SWAP_INIT, 0x1111_2222_3333_4444, 0x5555_6666_7777_8888]),
            },
            TestTexture {
                dim: TexDim::D2Array, format: DXGI_FORMAT_R32G32_UINT, width: 2, height: 1, depth: 2, texel_bytes: 8, uav: true,
                data: u64_bytes(&[u64::MAX, 0, i64::MAX as u64, i64::MIN as u64]),
            },
        ];

        let r = runner::run_with(&dxil, &st, &[], &textures, &[]).unwrap_or_else(|e| panic!("wave64={wave64}: {e}"));
        let t0 = &r.textures[0];
        let t1 = &r.textures[1];

        assert_eq!(tex_u64(t0, 0), ADD_INIT.wrapping_add(n_act.wrapping_mul(ADD_STEP)), "wave64={wave64} ADD");
        assert_eq!(tex_u64(t0, 1), SUB_INIT.wrapping_sub(n_act.wrapping_mul(SUB_STEP)), "wave64={wave64} SUB");
        assert_eq!(tex_u64(t0, 2), want_and, "wave64={wave64} AND");
        assert_eq!(tex_u64(t0, 3), want_or, "wave64={wave64} OR");
        assert_eq!(tex_u64(t0, 4), want_xor, "wave64={wave64} XOR");
        assert_eq!(tex_u64(t0, 6), 0x1111_2222_3333_4444, "wave64={wave64} untouched guard texel 6");
        assert_eq!(tex_u64(t0, 7), 0x5555_6666_7777_8888, "wave64={wave64} untouched guard texel 7");

        assert_eq!(tex_u64(t1, 0), want_umin, "wave64={wave64} D2Array UMIN");
        assert_eq!(tex_u64(t1, 1), want_umax, "wave64={wave64} D2Array UMAX");
        assert_eq!(tex_u64(t1, 2) as i64, want_smin, "wave64={wave64} D2Array SMIN");
        assert_eq!(tex_u64(t1, 3) as i64, want_smax, "wave64={wave64} D2Array SMAX");

        // Verify GLC=1 old-value sets for ADD, SUB, SWAP and untouched VDATA for GLC=0 and EXEC=0.
        let mut add_olds: Vec<u64> = active_lanes.iter().map(|&l| get_u64(&r.vgpr[l], 10)).collect();
        add_olds.sort_unstable();
        let want_add_olds: Vec<u64> = (0..n_act).map(|k| ADD_INIT.wrapping_add(k.wrapping_mul(ADD_STEP))).collect();
        assert_eq!(add_olds, want_add_olds, "wave64={wave64} ADD GLC=1 returned old values");

        let mut sub_olds: Vec<u64> = active_lanes.iter().map(|&l| get_u64(&r.vgpr[l], 12)).collect();
        sub_olds.sort_unstable();
        let mut want_sub_olds: Vec<u64> = (0..n_act).map(|k| SUB_INIT.wrapping_sub(k.wrapping_mul(SUB_STEP))).collect();
        want_sub_olds.sort_unstable();
        assert_eq!(sub_olds, want_sub_olds, "wave64={wave64} SUB GLC=1 returned old values");

        let mut swap_seen: Vec<u64> = active_lanes.iter().map(|&l| get_u64(&r.vgpr[l], 20)).collect();
        swap_seen.push(tex_u64(t0, 5));
        swap_seen.sort_unstable();
        let mut swap_want: Vec<u64> = active_lanes.iter().map(|&l| swap_in[l]).collect();
        swap_want.push(SWAP_INIT);
        swap_want.sort_unstable();
        assert_eq!(swap_seen, swap_want, "wave64={wave64} SWAP GLC=1 multiset conservation");

        for l in 0..lanes {
            let is_act = (exec_mask >> l) & 1 != 0;
            // GLC=0 ops must never clobber VDATA
            assert_eq!(get_u64(&r.vgpr[l], 14), get_u64(&st.vgpr[l], 14), "wave64={wave64} lane {l} AND GLC=0");
            assert_eq!(get_u64(&r.vgpr[l], 16), get_u64(&st.vgpr[l], 16), "wave64={wave64} lane {l} OR GLC=0");
            assert_eq!(get_u64(&r.vgpr[l], 24), get_u64(&st.vgpr[l], 24), "wave64={wave64} lane {l} UMAX GLC=0");
            if !is_act {
                for vd in [10, 12, 18, 20, 22, 26, 28] {
                    assert_eq!(get_u64(&r.vgpr[l], vd), get_u64(&st.vgpr[l], vd), "wave64={wave64} inactive lane {l} v{vd}");
                }
            }
        }
    }
}

/// 64-bit CAS-loop image atomics (`INC`, `DEC`, `FMIN`, `FMAX`), including 64-bit wraparound
/// thresholds and IEEE-754 `f64` magnitudes exceeding `f32` range.
#[test]
fn gpu_image_atomic64_inc_dec_and_fmin_fmax_cas_loops() {
    let Some(modes) = supported_modes() else { return };

    // Tex 0 (SRSRC=4): 6x1 D2 R32G32_UINT UAV
    //   (0,0): INC non-wrapping  (v10:v11)
    //   (1,0): INC wraparound    (v12:v13)
    //   (2,0): DEC non-wrapping  (v14:v15)
    //   (3,0): DEC wraparound    (v16:v17)
    //   (4,0): FMIN f64          (v18:v19)
    //   (5,0): FMAX f64          (v20:v21)
    let mut code = Vec::new();
    for (op, vd, va) in [
        ("IMAGE_ATOMIC_INC", 10, 0),
        ("IMAGE_ATOMIC_INC", 12, 2),
        ("IMAGE_ATOMIC_DEC", 14, 4),
        ("IMAGE_ATOMIC_DEC", 16, 6),
        ("IMAGE_ATOMIC_FMIN", 18, 8),
        ("IMAGE_ATOMIC_FMAX", 20, 22),
    ] {
        code.extend(mimg(op, &[("VDATA", vd), ("VADDR", va), ("SRSRC", 4), ("DMASK", 0x3), ("DIM", 1), ("UNORM", 1), ("GLC", 1)]));
    }
    code.extend(endpgm());

    const INC_INIT: u64 = 0x0000_0001_FFFF_FFFC; // crosses 32-bit carry -> 0x0000_0002_0000_0001 (5 steps)
    const INC_LIM: u64 = 0x0000_0002_0000_FFFF;
    const INC_W_INIT: u64 = 0x0000_0001_0000_0002;
    const INC_W_LIM: u64 = 0x0000_0001_0000_0004; // 2 -> 3 -> 4 -> 0 -> 1 -> 2 (5 steps)
    const DEC_INIT: u64 = 0x0000_0002_0000_0004;
    const DEC_LIM: u64 = 0x0000_0002_0000_FFFF;
    const DEC_W_INIT: u64 = 2;
    const DEC_W_LIM: u64 = 0x0000_0003_0000_0005; // 2 -> 1 -> 0 -> LIM -> LIM-1 -> LIM-2 (5 steps)

    for (wave64, split) in modes {
        let dxil = translate_and_validate(&code, wave64, split);
        let mut st = KernelState::default();
        // Run exactly 5 active lanes (0..5) so wraparound cycles are short and exact in both wave32 and wave64.
        st.sgpr[126] = 0x1F;
        st.sgpr[127] = 0;
        st.sgpr[16..24].copy_from_slice(&tsharp(0, 62, 9));

        let fmin_vals: [f64; 5] = [1.0e200, -3.5e150, f64::NAN, -9.25e250, 42.5];
        let fmax_vals: [f64; 5] = [-1.0e200, 3.5e150, f64::NAN, 9.25e250, -42.5];

        for l in 0..5usize {
            let v = &mut st.vgpr[l];
            for (va, x) in [(0, 0), (2, 1), (4, 2), (6, 3), (8, 4), (22, 5)] {
                v[va] = x;
                v[va + 1] = 0;
            }
            set_u64(v, 10, INC_LIM);
            set_u64(v, 12, INC_W_LIM);
            set_u64(v, 14, DEC_LIM);
            set_u64(v, 16, DEC_W_LIM);
            set_u64(v, 18, fmin_vals[l].to_bits());
            set_u64(v, 20, fmax_vals[l].to_bits());
        }

        let textures = vec![TestTexture {
            dim: TexDim::D2, format: DXGI_FORMAT_R32G32_UINT, width: 6, height: 1, depth: 1, texel_bytes: 8, uav: true,
            data: u64_bytes(&[INC_INIT, INC_W_INIT, DEC_INIT, DEC_W_INIT, (5.0e200_f64).to_bits(), (-5.0e200_f64).to_bits()]),
        }];

        let r = runner::run_with(&dxil, &st, &[], &textures, &[]).unwrap_or_else(|e| panic!("wave64={wave64}: {e}"));
        let t0 = &r.textures[0];

        assert_eq!(tex_u64(t0, 0), INC_INIT + 5, "wave64={wave64} INC 64-bit carry");
        let mut inc_olds: Vec<u64> = (0..5).map(|l| get_u64(&r.vgpr[l], 10)).collect();
        inc_olds.sort_unstable();
        assert_eq!(inc_olds, (0..5).map(|k| INC_INIT + k).collect::<Vec<_>>(), "wave64={wave64} INC carry old values");
        assert_eq!(tex_u64(t0, 1), 2, "wave64={wave64} INC wraparound (2->3->4->0->1->2)");
        assert_eq!(tex_u64(t0, 2), DEC_INIT - 5, "wave64={wave64} DEC 64-bit borrow");
        assert_eq!(tex_u64(t0, 3), DEC_W_LIM - 2, "wave64={wave64} DEC wraparound (2->1->0->LIM->LIM-1->LIM-2)");
        assert_eq!(tex_u64(t0, 4), (-9.25e250_f64).to_bits(), "wave64={wave64} FMIN f64");
        assert_eq!(tex_u64(t0, 5), (9.25e250_f64).to_bits(), "wave64={wave64} FMAX f64");

        let mut inc_w_olds: Vec<u64> = (0..5).map(|l| get_u64(&r.vgpr[l], 12)).collect();
        inc_w_olds.sort_unstable();
        let mut want_inc_w = vec![INC_W_INIT, INC_W_INIT + 1, INC_W_LIM, 0, 1];
        want_inc_w.sort_unstable();
        assert_eq!(inc_w_olds, want_inc_w, "wave64={wave64} INC wrap old values");

        let mut dec_w_olds: Vec<u64> = (0..5).map(|l| get_u64(&r.vgpr[l], 16)).collect();
        dec_w_olds.sort_unstable();
        let mut want_dec_w = vec![2, 1, 0, DEC_W_LIM, DEC_W_LIM - 1];
        want_dec_w.sort_unstable();
        assert_eq!(dec_w_olds, want_dec_w, "wave64={wave64} DEC wrap old values");
    }
}

/// 64-bit `IMAGE_ATOMIC_CMPSWAP` (`DMASK=0xF`, `Texture2D`) vs `IMAGE_ATOMIC_FCMPSWAP`
/// (`DMASK=0xF`, `Texture2DArray`) across IEEE-754 edge cases (`-0.0 == +0.0`, `qNaN`/`sNaN`,
/// subnormal, high-dword mismatch), plus a 32-bit `IMAGE_ATOMIC_CMPSWAP` (`DMASK=0x3`) control.
#[test]
fn gpu_image_atomic64_cmpswap_fcmpswap_and_32bit_control() {
    let Some(modes) = supported_modes() else { return };

    // (init64, cmp64, float_swaps, int_swaps)
    const CASES: [(u64, u64, bool, bool); 9] = [
        (0x8000_0000_0000_0000, 0x0000_0000_0000_0000, true, false),  // -0.0 vs +0.0
        (0x0000_0000_0000_0000, 0x8000_0000_0000_0000, true, false),  // +0.0 vs -0.0
        (0x7FF8_0000_0000_0000, 0x7FF8_0000_0000_0000, false, true),  // qNaN vs qNaN
        (0x7FF0_0000_0000_0001, 0x7FF0_0000_0000_0001, false, true),  // sNaN vs sNaN
        (0x0000_0000_0000_0001, 0x0000_0000_0000_0000, false, false), // subnormal vs +0.0
        (0x3FF8_0000_0000_0000, 0x3FF8_0000_0000_0000, true, true),   // 1.5 vs 1.5
        (0x3FF8_0000_0000_0000, 0x4004_0000_0000_0000, false, false), // 1.5 vs 2.5
        (0xFFF0_0000_0000_0000, 0xFFF0_0000_0000_0000, true, true),   // -inf vs -inf
        (0x0000_0001_1234_5678, 0x0000_0002_1234_5678, false, false), // high-32-bit mismatch only
    ];
    const NEW64: u64 = 0x3FF0_0000_0000_0000; // 1.0_f64
    const NEW32: u32 = 0xABCD_1234;

    let mut code = Vec::new();
    // Tex 0 (SRSRC=4): 16x1 D2 R32G32_UINT -> 64-bit CMPSWAP (DMASK=0xF, v10..v13)
    code.extend(mimg("IMAGE_ATOMIC_CMPSWAP", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 1), ("UNORM", 1), ("GLC", 1)]));
    // Tex 1 (SRSRC=6): 16x1x2 D2Array R32G32_UINT -> 64-bit FCMPSWAP on slice 1 (DMASK=0xF, v14..v17)
    code.extend(mimg("IMAGE_ATOMIC_FCMPSWAP", &[("VDATA", 14), ("VADDR", 2), ("SRSRC", 6), ("DMASK", 0xF), ("DIM", 5), ("UNORM", 1), ("GLC", 1)]));
    // Tex 2 (SRSRC=8): 16x1 D2 R32_UINT -> 32-bit CMPSWAP control (DMASK=0x3, v18..v19)
    code.extend(mimg("IMAGE_ATOMIC_CMPSWAP", &[("VDATA", 18), ("VADDR", 0), ("SRSRC", 8), ("DMASK", 0x3), ("DIM", 1), ("UNORM", 1), ("GLC", 1)]));
    code.extend(endpgm());

    for (wave64, split) in modes {
        let dxil = translate_and_validate(&code, wave64, split);
        let mut st = KernelState::default();
        st.sgpr[126] = (1u32 << CASES.len()) - 1;
        st.sgpr[127] = 0;
        st.sgpr[16..24].copy_from_slice(&tsharp(0, 62, 9));
        st.sgpr[24..32].copy_from_slice(&tsharp(2, 62, 13));
        st.sgpr[32..40].copy_from_slice(&tsharp(4, 20, 9));

        let mut d2_init = vec![0u64; 16];
        let mut arr_init = vec![0x7777_8888_9999_AAAAu64; 32]; // slice 0 guard + slice 1 active
        let mut d32_init = vec![0u32; 16];

        for (l, &(init64, cmp64, _, _)) in CASES.iter().enumerate() {
            d2_init[l] = init64;
            arr_init[16 + l] = init64;
            let init32 = 0x100 + l as u32;
            let cmp32 = if l % 2 == 0 { init32 } else { init32 ^ 1 };
            d32_init[l] = init32;

            let v = &mut st.vgpr[l];
            v[0] = l as u32;
            v[1] = 0;
            v[2] = l as u32;
            v[3] = 0;
            v[4] = 1; // slice 1
            set_u64(v, 10, NEW64);
            set_u64(v, 12, cmp64);
            set_u64(v, 14, NEW64);
            set_u64(v, 16, cmp64);
            v[18] = NEW32;
            v[19] = cmp32;
        }

        let textures = vec![
            TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32_UINT, width: 16, height: 1, depth: 1, texel_bytes: 8, uav: true, data: u64_bytes(&d2_init) },
            TestTexture { dim: TexDim::D2Array, format: DXGI_FORMAT_R32G32_UINT, width: 16, height: 1, depth: 2, texel_bytes: 8, uav: true, data: u64_bytes(&arr_init) },
            TestTexture {
                dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 16, height: 1, depth: 1, texel_bytes: 4, uav: true,
                data: d32_init.iter().flat_map(|w| w.to_le_bytes()).collect(),
            },
        ];

        let r = runner::run_with(&dxil, &st, &[], &textures, &[]).unwrap_or_else(|e| panic!("wave64={wave64}: {e}"));
        let (t0, t1, t2) = (&r.textures[0], &r.textures[1], &r.textures[2]);

        for l in 0..16usize {
            assert_eq!(tex_u64(t1, l), 0x7777_8888_9999_AAAA, "wave64={wave64} slice 0 guard texel {l}");
        }
        for (l, &(init64, cmp64, fswap, iswap)) in CASES.iter().enumerate() {
            let v = &r.vgpr[l];
            assert_eq!(get_u64(v, 10), init64, "wave64={wave64} case {l} CMPSWAP old");
            assert_eq!(get_u64(v, 12), cmp64, "wave64={wave64} case {l} CMPSWAP cmp untouched");
            assert_eq!(tex_u64(t0, l), if iswap { NEW64 } else { init64 }, "wave64={wave64} case {l} CMPSWAP texel");

            assert_eq!(get_u64(v, 14), init64, "wave64={wave64} case {l} FCMPSWAP old");
            assert_eq!(get_u64(v, 16), cmp64, "wave64={wave64} case {l} FCMPSWAP cmp untouched");
            assert_eq!(tex_u64(t1, 16 + l), if fswap { NEW64 } else { init64 }, "wave64={wave64} case {l} FCMPSWAP texel");

            let init32 = 0x100 + l as u32;
            let got32 = u32::from_le_bytes(t2[l * 4..l * 4 + 4].try_into().unwrap());
            assert_eq!(v[18], init32, "wave64={wave64} case {l} 32-bit CMPSWAP old");
            assert_eq!(got32, if l % 2 == 0 { NEW32 } else { init32 }, "wave64={wave64} case {l} 32-bit CMPSWAP texel");
        }
    }
}
