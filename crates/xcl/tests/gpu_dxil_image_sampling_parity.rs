//! GPU differential parity tests between the direct RDNA2 → DXIL translator
//! (`xcl::gpu::dxil::kernel::translate_full`, validated with `dxil.dll`) and the HLSL + DXC path
//! (`xcl::gpu::recomp::translate` + `dxc::compile`) for `lower_image.rs` MIMG operations:
//! sampling across 1DArray/2D/2DArray/3D/Cube with offsets/gradients/comparison/`A16`/`D16`/`UNORM`,
//! `image_gather4_*` (including explicit-LOD S# rules), `image_load`/`image_load_mip`/`image_store`/
//! `image_get_resinfo`, 32-bit float/CAS `image_atomic_*`, and T# descriptor-heap guards.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use xcl::gpu::dxil::kernel::{translate_full, KernelOptions};
use xcl::gpu::dxil::validate;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner::{self, KernelResult, KernelState, TestSampler, TestTexture, TexDim};
use xcl::gpu::recomp::{dxc, translate, Mode, Options};
use xcl::hle::d3d12x::console_heap;

fn gpu_ready() -> bool {
    match (runner::adapter(), runner::wave64_programs_supported(), dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => true,
        (a, _, d) => {
            runner::skip_test(if a.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" });
            false
        }
    }
}

/// Wave configurations supported by this adapter: `(label, wave64, split)`.
fn wave_configs() -> Vec<(&'static str, bool, bool)> {
    let mut out = Vec::new();
    if runner::wave32_supported() == Ok(true) {
        out.push(("wave32", false, false));
        out.push(("split-wave64", true, true));
    }
    if runner::wave64_supported() == Ok(true) {
        out.push(("native-wave64", true, false));
    }
    out
}

fn prog(ops: &[(&str, &str, &[(&str, u64)])]) -> Vec<u8> {
    let mut code = Vec::new();
    for &(name, enc, fields) in ops {
        code.extend(encode(name, enc, fields, None).unwrap_or_else(|e| panic!("encode {name}: {e}")));
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    code
}

fn tsharp(index: u32, format: u32, ty: u32) -> [u32; 8] {
    [index, format << 20, 0, ty << 28, 0, 0, 0, 0]
}

fn set_tsharps(st: &mut KernelState, start_srsrc: usize, defs: &[(u32, u32, u32)]) {
    for (i, &(idx, fmt, ty)) in defs.iter().enumerate() {
        let base = (start_srsrc + i * 2) * 4;
        st.sgpr[base..base + 8].copy_from_slice(&tsharp(idx, fmt, ty));
    }
}

fn f(x: f32) -> u32 {
    x.to_bits()
}

fn ssharp(addr_u: u32, addr_v: u32, cmp: u32, lod_bias_s58: i32, min_lod_u48: u32, max_lod_u48: u32, mip_filter: u32, border: u32, heap_idx: u32) -> [u32; 4] {
    let w0 = (addr_u & 7) | ((addr_v & 7) << 3) | ((cmp & 7) << 12);
    let w1 = (min_lod_u48 & 0xFFF) | ((max_lod_u48 & 0xFFF) << 12);
    let w2 = ((lod_bias_s58 as u32) & 0x3FFF) | ((mip_filter & 3) << 26);
    let w3 = (heap_idx & 0xFFF) | ((border & 3) << 30);
    [w0, w1, w2, w3]
}

fn f16_bits(x: f32) -> u32 {
    let sign = if x.is_sign_negative() { 0x8000u32 } else { 0 };
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    let bits = a.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits & 0x7F_FFFF) >> 13;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7C00
    } else {
        sign | ((exp as u32) << 10) | mant
    }
}

fn pack_f16(lo: f32, hi: f32) -> u32 {
    f16_bits(lo) | (f16_bits(hi) << 16)
}

fn run_direct(code: &[u8], st: &KernelState, tex: &[TestTexture], samp: &[TestSampler], wave64: bool, split: bool) -> KernelResult {
    let k = translate_full(code, KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 }).expect("direct DXIL translate");
    validate::validate(&k.dxil).expect("direct DXIL validate");
    runner::run_with_statics(&k.dxil, st, &[], tex, samp, &k.static_samplers).expect("direct DXIL GPU run")
}

fn run_hlsl(code: &[u8], st: &KernelState, tex: &[TestTexture], samp: &[TestSampler], wave64: bool, split: bool) -> KernelResult {
    let opt = Options { mode: Mode::Kernel, wave64, split_wave64: split, ..Default::default() };
    let t = translate(code, &opt).expect("HLSL translate");
    assert!(t.complete(), "HLSL unsupported: {:?}", t.unsupported);
    let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).expect("DXC compile");
    runner::run_with_statics(&dxil, st, &[], tex, samp, &t.static_samplers).expect("HLSL GPU run")
}

fn assert_parity(name: &str, code: &[u8], st: &KernelState, tex: &[TestTexture], samp: &[TestSampler], vregs: &[usize], check: impl Fn(&str, usize, &KernelResult)) {
    for (label, wave64, split) in wave_configs() {
        let lanes = if wave64 { 64 } else { 32 };
        let direct = run_direct(code, st, tex, samp, wave64, split);
        let hlsl = run_hlsl(code, st, tex, samp, wave64, split);
        for l in 0..lanes {
            for &r in vregs {
                assert_eq!(direct.vgpr[l][r], hlsl.vgpr[l][r], "{name} ({label}) lane {l} v{r} direct={:#x} hlsl={:#x}", direct.vgpr[l][r], hlsl.vgpr[l][r]);
            }
        }
        assert_eq!(direct.textures, hlsl.textures, "{name} ({label}) UAV texture bytes differ");
        check(label, lanes, &direct);
    }
}

fn tex_u32(res: &KernelResult, i: usize) -> u32 {
    u32::from_le_bytes(res.textures[i][..4].try_into().unwrap())
}

fn pt_sampler(comparison: Option<D3D12_COMPARISON_FUNC>) -> TestSampler {
    let filter = if comparison.is_some() { D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT } else { D3D12_FILTER_MIN_MAG_MIP_POINT };
    TestSampler { filter, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison }
}

fn rgba32f_tex(dim: TexDim, width: u32, height: u32, depth: u32, pixel: impl Fn(u32, u32, u32) -> [f32; 4]) -> TestTexture {
    let mut data = Vec::with_capacity((width * height * depth * 16) as usize);
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                for c in pixel(x, y, z) {
                    data.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
    }
    TestTexture { dim, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width, height, depth, texel_bytes: 16, data, uav: false }
}

fn uav_1x1(format: DXGI_FORMAT, init_bits: u32) -> TestTexture {
    TestTexture { dim: TexDim::D2, format, width: 1, height: 1, depth: 1, texel_bytes: 4, data: init_bits.to_le_bytes().to_vec(), uav: true }
}

/// 1DArray, 2D, 2DArray, 3D, and CubeArray sampling with explicit LOD, texel offsets (`_o`),
/// gradients (`_d`), depth comparison (`_c_lz`), `A16` / `D16`, and non-cube `UNORM=1`.
#[test]
fn direct_dxil_image_sample_dimensions_cube_offset_and_unorm_parity() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // 1DArray sample_lz (v[0:1] = u, slice) -> v[20:23]
        ("IMAGE_SAMPLE_LZ", "ENC_MIMG", &[("VDATA", 20), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 16), ("DMASK", 0xF), ("DIM", 4)]),
        // 2D sample_l_o with texel offset (+1, -1) (v[2:5] = off, u, v, lod) -> v[24:27]
        ("IMAGE_SAMPLE_L_O", "ENC_MIMG", &[("VDATA", 24), ("VADDR", 2), ("SRSRC", 6), ("SSAMP", 16), ("DMASK", 0xF), ("DIM", 1)]),
        // 2DArray sample_lz with fractional layer rounding (v[6:8] = u, v, slice) -> v[28:31]
        ("IMAGE_SAMPLE_LZ", "ENC_MIMG", &[("VDATA", 28), ("VADDR", 6), ("SRSRC", 8), ("SSAMP", 16), ("DMASK", 0xF), ("DIM", 5)]),
        // 3D sample_d with explicit gradients (v[9:17] = dux, duy, duz, dvx, dvy, dvz, u, v, w) -> v[32:35]
        ("IMAGE_SAMPLE_D", "ENC_MIMG", &[("VDATA", 32), ("VADDR", 9), ("SRSRC", 10), ("SSAMP", 16), ("DMASK", 0xF), ("DIM", 2)]),
        // CubeArray sample_lz UNORM=1 (v[47:49] = s, t, face + 8*cube) -> v[36:39]
        ("IMAGE_SAMPLE_LZ", "ENC_MIMG", &[("VDATA", 36), ("VADDR", 47), ("SRSRC", 12), ("SSAMP", 16), ("DMASK", 0xF), ("DIM", 3), ("UNORM", 1)]),
        // 2D comparison sample_c_lz on R32_FLOAT (v[40:42] = ref, u, v) -> v[44]
        ("IMAGE_SAMPLE_C_LZ", "ENC_MIMG", &[("VDATA", 44), ("VADDR", 40), ("SRSRC", 14), ("SSAMP", 17), ("DMASK", 0x1), ("DIM", 1)]),
        // 2D sample_lz A16=1 D16=1 (v[45] = packed u|v f16) -> v[46] (packed r|g f16)
        ("IMAGE_SAMPLE_LZ", "ENC_MIMG", &[("VDATA", 46), ("VADDR", 45), ("SRSRC", 6), ("SSAMP", 16), ("DMASK", 0x3), ("DIM", 1), ("A16", 1), ("D16", 1)]),
    ]);

    let mut st = KernelState::default();
    set_tsharps(&mut st, 4, &[(0, 77, 12), (2, 77, 9), (4, 77, 13), (6, 77, 10), (8, 77, 11), (10, 22, 9)]);
    st.sgpr[64..68].copy_from_slice(&[0, 0, 0, 0]); // heap sampler 0 (point clamp)
    st.sgpr[68..72].copy_from_slice(&[0, 0, 0, 1]); // heap sampler 1 (cmp less)

    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, y, slice) = ((l % 4) as f32, ((l / 4) % 4) as f32, (l % 3) as f32);
        let (u, v) = ((x + 0.5) / 4.0, (y + 0.5) / 4.0);
        let (face, cube) = (l % 6, (l / 6) % 2);
        r[0..10].copy_from_slice(&[f(u), f(slice + if l % 2 == 0 { 0.2 } else { -0.2 }), 1 | (0x3F << 8), f(u), f(v), 0, f(u), f(v), f(slice + 0.25), f(0.25)]);
        r[14..18].copy_from_slice(&[f(0.25), f(u), f(v), f(((l % 2) as f32 + 0.5) / 2.0)]);
        r[40..43].copy_from_slice(&[f(5.5), f(u), f(v)]);
        r[45] = pack_f16(u, v);
        r[47..50].copy_from_slice(&[f(1.5), f(1.5), f((face + 8 * cube) as f32)]);
    }

    let r32f_data: Vec<u8> = (0..16u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let tex = [
        rgba32f_tex(TexDim::D1Array, 4, 1, 3, |x, _, z| [x as f32, z as f32, (x + 10 * z) as f32, 1.0]),
        rgba32f_tex(TexDim::D2, 4, 4, 1, |x, y, _| [(x + 4 * y) as f32, y as f32, x as f32, 1.0]),
        rgba32f_tex(TexDim::D2Array, 4, 4, 3, |x, y, z| [x as f32, y as f32, z as f32, 2.0]),
        rgba32f_tex(TexDim::D3, 4, 4, 2, |x, y, z| [x as f32, y as f32, z as f32, 3.0]),
        rgba32f_tex(TexDim::Cube, 2, 2, 12, |_, _, z| [(z % 6) as f32, (z / 6) as f32, z as f32, 1.0]),
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 4, data: r32f_data, uav: false },
    ];
    let samp = [pt_sampler(None), pt_sampler(Some(D3D12_COMPARISON_FUNC_LESS))];
    let vregs: Vec<usize> = (20..40).chain([44, 46]).collect();
    assert_parity("image_sample_parity", &code, &st, &tex, &samp, &vregs, |label, lanes, res| {
        for l in 0..lanes {
            let v = &res.vgpr[l];
            let x = (l % 4) as f32;
            let y = ((l / 4) % 4) as f32;
            let slice = (l % 3) as f32;
            assert_eq!(f32::from_bits(v[22]), x + 10.0 * slice, "{label} lane {l} 1DArray");
            let ox = (x + 1.0).min(3.0);
            let oy = (y - 1.0).max(0.0);
            assert_eq!(f32::from_bits(v[24]), ox + 4.0 * oy, "{label} lane {l} 2D sample_l_o");
            assert_eq!(f32::from_bits(v[30]), slice, "{label} lane {l} 2DArray");
            assert_eq!(f32::from_bits(v[34]), (l % 2) as f32, "{label} lane {l} 3D sample_d");
            assert_eq!(f32::from_bits(v[36]), (l % 6) as f32, "{label} lane {l} Cube face");
            assert_eq!(f32::from_bits(v[37]), ((l / 6) % 2) as f32, "{label} lane {l} Cube slice");
            let want_cmp = if 5.5 < (x + 4.0 * y) { 1.0f32 } else { 0.0 };
            assert_eq!(f32::from_bits(v[44]), want_cmp, "{label} lane {l} sample_c_lz");
            assert_eq!(v[46], pack_f16(x + 4.0 * y, y), "{label} lane {l} sample_lz A16+D16");
        }
    });

    // Non-cube UNORM=1 sampling is direct-DXIL-only (HLSL marks "MIMG unorm" unsupported); verify both.
    let unorm_code = prog(&[
        ("IMAGE_SAMPLE_LZ", "ENC_MIMG", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 8), ("DMASK", 0xF), ("DIM", 1), ("UNORM", 1)]),
        ("IMAGE_SAMPLE_L_O", "ENC_MIMG", &[("VDATA", 14), ("VADDR", 2), ("SRSRC", 6), ("SSAMP", 8), ("DMASK", 0xF), ("DIM", 5), ("UNORM", 1)]),
    ]);
    let ut = translate(&unorm_code, &Options::default()).unwrap();
    assert!(!ut.complete() && ut.unsupported_why.keys().all(|k| k.ends_with("MIMG unorm")), "expected MIMG unorm unsupported, got {:?}", ut.unsupported_why);
    let mut ust = KernelState::default();
    set_tsharps(&mut ust, 4, &[(2, 77, 9), (4, 77, 13)]);
    for (l, r) in ust.vgpr.iter_mut().enumerate() {
        let (x, y) = ((l % 4) as f32, ((l / 4) % 4) as f32);
        r[0..7].copy_from_slice(&[f(x + 0.5), f(y + 0.5), 1 | (1 << 8), f(x + 0.5), f(y + 0.5), f((l % 3) as f32), 0]);
    }
    for (label, wave64, split) in wave_configs() {
        let res = run_direct(&unorm_code, &ust, &tex, &samp, wave64, split);
        for l in 0..if wave64 { 64 } else { 32 } {
            let x = (l % 4) as f32;
            let y = ((l / 4) % 4) as f32;
            assert_eq!(f32::from_bits(res.vgpr[l][10]), x + 4.0 * y, "{label} lane {l} 2D UNORM=1");
            assert_eq!(f32::from_bits(res.vgpr[l][14]), (x + 1.0).min(3.0), "{label} lane {l} 2DArray UNORM=1 offset x");
            assert_eq!(f32::from_bits(res.vgpr[l][15]), (y + 1.0).min(3.0), "{label} lane {l} 2DArray UNORM=1 offset y");
            assert_eq!(f32::from_bits(res.vgpr[l][16]), (l % 3) as f32, "{label} lane {l} 2DArray UNORM=1 slice");
        }
    }
}

/// `image_gather4_*` parity (LZ / L / C_L / _O, RGBA channels, 2D & 2DArray, A16/D16) plus
/// direct DXIL `mimg_gather4_lod` S# wrap / mirror / border / `UNORM=1` execution checks.
#[test]
fn direct_dxil_image_gather4_lod_and_sampler_rules_parity() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // 2D gather4_lz Green + offset (-1, +1) (v[0:2] = off, u, v) -> v[20:23]
        ("IMAGE_GATHER4_LZ_O", "ENC_MIMG", &[("VDATA", 20), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 10), ("DMASK", 0x2), ("DIM", 1)]),
        // 2D gather4_l Alpha (v[3:5] = u, v, lod=0) -> v[24:27]
        ("IMAGE_GATHER4_L", "ENC_MIMG", &[("VDATA", 24), ("VADDR", 3), ("SRSRC", 4), ("SSAMP", 10), ("DMASK", 0x8), ("DIM", 1)]),
        // 2DArray gather4_l_o Blue (v[6:9] = off, u, v, slice, lod=0) -> v[28:31]
        ("IMAGE_GATHER4_L_O", "ENC_MIMG", &[("VDATA", 28), ("VADDR", 6), ("SRSRC", 6), ("SSAMP", 10), ("DMASK", 0x4), ("DIM", 5)]),
        // 2D gather4_c_l Red on R32_FLOAT A16=1 D16=1 (v[10] = ref, v[11] = u|v f16, v[12] = lod=0 f16) -> v[32:33]
        ("IMAGE_GATHER4_C_L", "ENC_MIMG", &[("VDATA", 32), ("VADDR", 10), ("SRSRC", 8), ("SSAMP", 11), ("DMASK", 0x1), ("DIM", 1), ("A16", 1), ("D16", 1)]),
    ]);

    let mut st = KernelState::default();
    set_tsharps(&mut st, 4, &[(0, 77, 9), (2, 77, 13), (4, 22, 9)]);
    st.sgpr[40..44].copy_from_slice(&ssharp(2, 2, 0, 0, 0, 0xFFF, 1, 0, 0)); // clamp point, idx 0
    st.sgpr[44..48].copy_from_slice(&ssharp(2, 2, 4, 0, 0, 0xFFF, 1, 0, 1)); // clamp cmp GREATER, idx 1

    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (u, v) = ((1.0 + (l % 3) as f32) / 4.0, (1.0 + ((l / 3) % 3) as f32) / 4.0);
        r[0..13].copy_from_slice(&[0x3F | (1 << 8), f(u), f(v), f(u), f(v), 0, 1, f(u), f(v), f((l % 2) as f32), f(2.5), pack_f16(u, v), pack_f16(0.0, 0.0)]);
    }

    let r32f_cmp: Vec<u8> = (0..16u32).flat_map(|i| ((i + 1) as f32).to_le_bytes()).collect();
    let tex = [
        rgba32f_tex(TexDim::D2, 4, 4, 1, |x, y, _| [x as f32, y as f32, (x + 10 * y) as f32, (x + 1 + 4 * y) as f32]),
        rgba32f_tex(TexDim::D2Array, 4, 4, 2, |x, y, z| [x as f32, y as f32, (z * 100 + x + 4 * y) as f32, 1.0]),
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 4, data: r32f_cmp, uav: false },
    ];
    let samp = [pt_sampler(None), pt_sampler(Some(D3D12_COMPARISON_FUNC_GREATER))];
    let vregs: Vec<usize> = (20..34).collect();
    assert_parity("image_gather4_parity", &code, &st, &tex, &samp, &vregs, |label, lanes, res| {
        for l in 0..lanes {
            let v = &res.vgpr[l];
            let (x, y, s) = (l % 3, (l / 3) % 3, l % 2);
            let y1o = (y + 2).min(3) as f32;
            let g_off: Vec<f32> = (20..24).map(|i| f32::from_bits(v[i])).collect();
            assert_eq!(g_off, [y1o, y1o, (y + 1) as f32, (y + 1) as f32], "{label} lane {l} gather4_lz_o green");
            let a_lod: Vec<f32> = (24..28).map(|i| f32::from_bits(v[i])).collect();
            let alpha = |tx: usize, ty: usize| (tx + 1 + 4 * ty) as f32;
            assert_eq!(a_lod, [alpha(x, y + 1), alpha(x + 1, y + 1), alpha(x + 1, y), alpha(x, y)], "{label} lane {l} gather4_l alpha");
            let x0o = (x + 1) as f32;
            let x1o = (x + 2).min(3) as f32;
            let b_off: Vec<f32> = (28..32).map(|i| f32::from_bits(v[i])).collect();
            assert_eq!(b_off, [(s * 100) as f32 + x0o + 4.0 * (y + 1) as f32, (s * 100) as f32 + x1o + 4.0 * (y + 1) as f32, (s * 100) as f32 + x1o + 4.0 * y as f32, (s * 100) as f32 + x0o + 4.0 * y as f32], "{label} lane {l} gather4_l_o blue");
            let cmp = |cx: usize, cy: usize| if 2.5 > (cx + 1 + 4 * cy) as f32 { 1.0f32 } else { 0.0 };
            assert_eq!(v[32], pack_f16(cmp(x, y + 1), cmp(x + 1, y + 1)), "{label} lane {l} gather4_c_l d16 lo");
            assert_eq!(v[33], pack_f16(cmp(x + 1, y), cmp(x, y)), "{label} lane {l} gather4_c_l d16 hi");
        }
    });

    // Exercise direct-DXIL mimg_gather4_lod S# border color, wrap/mirror, and UNORM=1.
    let s_rules = prog(&[
        // Border color opaque white (border=2, addr=6 ClampBorder) at out-of-bounds u=-0.5 -> [1,1,1,1]
        ("IMAGE_GATHER4_L", "ENC_MIMG", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)]),
        // Wrap (addr=0) + Mirror (addr=1) with UNORM=1 texel center (x=3.5 -> wraps x1=4 to 0)
        ("IMAGE_GATHER4_L", "ENC_MIMG", &[("VDATA", 14), ("VADDR", 3), ("SRSRC", 4), ("SSAMP", 7), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]),
    ]);
    let mut sst = KernelState::default();
    set_tsharps(&mut sst, 4, &[(0, 77, 9)]);
    sst.sgpr[24..28].copy_from_slice(&ssharp(6, 6, 0, 0, 0, 0xFFF, 1, 2, 0)); // border white
    sst.sgpr[28..32].copy_from_slice(&ssharp(0, 1, 0, 0, 0, 0xFFF, 1, 0, 0)); // wrap X, mirror Y
    for r in sst.vgpr.iter_mut() {
        r[0..6].copy_from_slice(&[f(-0.5), f(-0.5), 0, f(4.0), f(1.0), 0]);
    }
    for (label, wave64, split) in wave_configs() {
        let res = run_direct(&s_rules, &sst, &tex, &samp, wave64, split);
        for l in 0..if wave64 { 64 } else { 32 } {
            let b: Vec<f32> = (10..14).map(|i| f32::from_bits(res.vgpr[l][i])).collect();
            assert_eq!(b, [1.0, 1.0, 1.0, 1.0], "{label} lane {l} gather4_l opaque white border");
            let w: Vec<f32> = (14..18).map(|i| f32::from_bits(res.vgpr[l][i])).collect();
            assert_eq!(w, [3.0, 0.0, 0.0, 3.0], "{label} lane {l} gather4_l wrap+UNORM");
        }
    }
}

/// `image_load`, `image_load_mip`, `image_store`, `image_get_resinfo` across 1DArray/2D/2DArray/3D/Cube,
/// typed `SINT`/`FLOAT`, `D16` pack/unpack, partial `DMASK`, and `EXEC` masking.
#[test]
fn direct_dxil_image_load_mip_store_resinfo_and_d16_parity() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // 2D image_load_mip at mip 0, DMASK=0x5 (x, z) -> v[20:21]
        ("IMAGE_LOAD_MIP", "ENC_MIMG", &[("VDATA", 20), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0x5), ("DIM", 1)]),
        // 3D image_load D16=1 float -> v[22:23] (4 f16s packed in 2 dwords)
        ("IMAGE_LOAD", "ENC_MIMG", &[("VDATA", 22), ("VADDR", 3), ("SRSRC", 6), ("DMASK", 0xF), ("DIM", 2), ("D16", 1)]),
        // 2D R32_SINT image_load -> v[24]
        ("IMAGE_LOAD", "ENC_MIMG", &[("VDATA", 24), ("VADDR", 0), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1)]),
        // Resinfo across 1DArray (4), 2D (1), 2DArray (5), 3D (2), Cube (3)
        ("IMAGE_GET_RESINFO", "ENC_MIMG", &[("VDATA", 28), ("VADDR", 6), ("SRSRC", 10), ("DMASK", 0xF), ("DIM", 4)]),
        ("IMAGE_GET_RESINFO", "ENC_MIMG", &[("VDATA", 32), ("VADDR", 6), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 1)]),
        ("IMAGE_GET_RESINFO", "ENC_MIMG", &[("VDATA", 36), ("VADDR", 6), ("SRSRC", 12), ("DMASK", 0xF), ("DIM", 5)]),
        ("IMAGE_GET_RESINFO", "ENC_MIMG", &[("VDATA", 40), ("VADDR", 6), ("SRSRC", 6), ("DMASK", 0xF), ("DIM", 2)]),
        ("IMAGE_GET_RESINFO", "ENC_MIMG", &[("VDATA", 44), ("VADDR", 6), ("SRSRC", 14), ("DMASK", 0xF), ("DIM", 3)]),
        // Partial EXEC mask (lanes 0..16 active via inline const 16 = 144): store to UAV and read back
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 8)]),
        ("S_AND_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 106)]),
        ("IMAGE_STORE", "ENC_MIMG", &[("VDATA", 7), ("VADDR", 0), ("SRSRC", 16), ("DMASK", 0x1), ("DIM", 1)]),
        ("IMAGE_LOAD", "ENC_MIMG", &[("VDATA", 25), ("VADDR", 0), ("SRSRC", 16), ("DMASK", 0x1), ("DIM", 1)]),
    ]);

    let mut st = KernelState::default();
    set_tsharps(&mut st, 4, &[(0, 77, 9), (2, 77, 10), (4, 21, 9), (6, 77, 12), (8, 77, 13), (10, 77, 11), (12, 21, 9)]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, y) = ((l % 4) as u32, ((l / 4) % 4) as u32);
        r[0..9].copy_from_slice(&[x, y, 0, x, (l % 2) as u32, ((l / 2) % 2) as u32, 0, (-(l as i32) - 10) as u32, l as u32]);
        r[25] = 0xDEAD_BEEF; // inactive-lane sentinel
    }

    let sint_data: Vec<u8> = (0..16i32).flat_map(|i| (-100 - i).to_le_bytes()).collect();
    let uav_sint: Vec<u8> = vec![0x77; 8 * 8 * 4];
    let tex = [
        rgba32f_tex(TexDim::D2, 4, 4, 1, |x, y, _| [x as f32, y as f32, (x + 10 * y) as f32, 1.0]),
        rgba32f_tex(TexDim::D3, 4, 2, 2, |x, y, z| [x as f32, y as f32, z as f32, -2.0]),
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_SINT, width: 4, height: 4, depth: 1, texel_bytes: 4, data: sint_data, uav: false },
        rgba32f_tex(TexDim::D1Array, 5, 1, 3, |_, _, _| [0.0; 4]),
        rgba32f_tex(TexDim::D2Array, 4, 2, 3, |_, _, _| [0.0; 4]),
        rgba32f_tex(TexDim::Cube, 2, 2, 12, |_, _, _| [0.0; 4]),
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_SINT, width: 8, height: 8, depth: 1, texel_bytes: 4, data: uav_sint, uav: true },
    ];
    let vregs: Vec<usize> = [20, 21, 22, 23, 24, 25].into_iter().chain(28..48).collect();
    assert_parity("image_load_resinfo_parity", &code, &st, &tex, &[], &vregs, |label, lanes, res| {
        for l in 0..lanes {
            let v = &res.vgpr[l];
            let x = (l % 4) as f32;
            let y = ((l / 4) % 4) as f32;
            assert_eq!([f32::from_bits(v[20]), f32::from_bits(v[21])], [x, x + 10.0 * y], "{label} lane {l} load_mip");
            assert_eq!(v[22], pack_f16(x, (l % 2) as f32), "{label} lane {l} 3D load D16 lo");
            assert_eq!(v[23], pack_f16(((l / 2) % 2) as f32, -2.0), "{label} lane {l} 3D load D16 hi");
            assert_eq!(v[24] as i32, -100 - ((l % 4 + 4 * ((l / 4) % 4)) as i32), "{label} lane {l} R32_SINT load");
            assert_eq!((v[28], v[29], v[31]), (5, 3, 1), "{label} lane {l} 1DArray resinfo");
            assert_eq!(&v[32..36], &[4, 4, 0, 1], "{label} lane {l} 2D resinfo");
            assert_eq!(&v[36..40], &[4, 2, 3, 1], "{label} lane {l} 2DArray resinfo");
            assert_eq!(&v[40..44], &[4, 2, 2, 1], "{label} lane {l} 3D resinfo");
            assert_eq!(&v[44..48], &[2, 2, 12, 1], "{label} lane {l} Cube resinfo");
            if l < 16 {
                assert_eq!(v[25] as i32, -(l as i32) - 10, "{label} lane {l} active UAV readback");
            } else {
                assert_eq!(v[25], 0xDEAD_BEEF, "{label} lane {l} inactive sentinel");
            }
        }
    });
}

/// 32-bit float / CAS `image_atomic_*` (`fmin`, `fmax`, `fcmpswap`, `sub`, `dec`)
/// and T# descriptor guards (null type `< 8`, non-zero `tw1[7:0]`, and `tw0 >= GLOBAL_MAX - 1`).
#[test]
fn direct_dxil_image_atomics_and_descriptor_guard_parity() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // Float / CAS atomics on 1x1 R32_FLOAT / R32_UINT UAVs
        ("IMAGE_ATOMIC_FMIN", "ENC_MIMG", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 1)]),
        ("IMAGE_ATOMIC_FMAX", "ENC_MIMG", &[("VDATA", 11), ("VADDR", 0), ("SRSRC", 6), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 1)]),
        ("IMAGE_ATOMIC_FCMPSWAP", "ENC_MIMG", &[("VDATA", 12), ("VADDR", 0), ("SRSRC", 8), ("DMASK", 0x3), ("DIM", 1), ("UNORM", 1), ("GLC", 1)]),
        ("IMAGE_ATOMIC_SUB", "ENC_MIMG", &[("VDATA", 14), ("VADDR", 0), ("SRSRC", 10), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 0)]),
        ("IMAGE_ATOMIC_DEC", "ENC_MIMG", &[("VDATA", 15), ("VADDR", 0), ("SRSRC", 12), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1), ("GLC", 1)]),
        // Descriptor-heap guards: null type, dirty tw1[7:0], and OOB tw0 must return 0 safely
        ("IMAGE_LOAD", "ENC_MIMG", &[("VDATA", 20), ("VADDR", 0), ("SRSRC", 14), ("DMASK", 0xF), ("DIM", 1)]),
        ("IMAGE_SAMPLE_LZ", "ENC_MIMG", &[("VDATA", 24), ("VADDR", 2), ("SRSRC", 16), ("SSAMP", 20), ("DMASK", 0xF), ("DIM", 1)]),
        ("IMAGE_GATHER4_LZ", "ENC_MIMG", &[("VDATA", 28), ("VADDR", 2), ("SRSRC", 18), ("SSAMP", 20), ("DMASK", 0x1), ("DIM", 1)]),
    ]);

    let mut st = KernelState::default();
    set_tsharps(&mut st, 4, &[(0, 22, 9), (2, 22, 9), (4, 20, 9), (6, 20, 9), (8, 20, 9)]);
    st.sgpr[56..64].copy_from_slice(&[0; 8]);                                        // null T# (type=0 < 8)
    st.sgpr[64..72].copy_from_slice(&[0, (77 << 20) | 1, 0, 9 << 28, 0, 0, 0, 0]);  // dirty tw1[7:0]
    st.sgpr[72..80].copy_from_slice(&tsharp(console_heap::GLOBAL_MAX - 1, 77, 9));  // OOB tw0

    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0..4].copy_from_slice(&[0, 0, f(0.5), f(0.5)]);
        r[10..16].copy_from_slice(&[f((l as f32) - 5.0), f(l as f32), f(42.0), if l == 7 { 0x8000_0000 } else { f(999.0) }, 1, 3]);
        r[20..32].fill(0xCAFE_BABE);
    }

    let tex = [
        uav_1x1(DXGI_FORMAT_R32_FLOAT, f(100.0)),
        uav_1x1(DXGI_FORMAT_R32_FLOAT, f(-10.0)),
        uav_1x1(DXGI_FORMAT_R32_UINT, f(0.0)),
        uav_1x1(DXGI_FORMAT_R32_UINT, 100),
        uav_1x1(DXGI_FORMAT_R32_UINT, 1),
    ];
    let samp = [pt_sampler(None)];

    // Only assert deterministic final state and guard zeroes (atomic race order across lanes may differ).
    let guard_vregs: Vec<usize> = [14usize].into_iter().chain(20..32).collect();
    assert_parity("image_atomic_guard_parity", &code, &st, &tex, &samp, &guard_vregs, |label, lanes, res| {
        assert_eq!(f32::from_bits(tex_u32(res, 0)), -5.0, "{label} fmin");
        assert_eq!(f32::from_bits(tex_u32(res, 1)), (lanes - 1) as f32, "{label} fmax");
        assert_eq!(f32::from_bits(tex_u32(res, 2)), 42.0, "{label} fcmpswap -0.0==+0.0");
        assert_eq!(tex_u32(res, 3), 100 - lanes as u32, "{label} sub");
        assert_eq!(tex_u32(res, 4), 1, "{label} dec");
        for l in 0..lanes {
            assert_eq!(res.vgpr[l][14], 1, "{label} lane {l} GLC=0 preserved v14");
            assert_eq!(&res.vgpr[l][20..32], &[0u32; 12], "{label} lane {l} invalid T# returned zero");
        }
    });
}
