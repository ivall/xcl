//! Game-free end-to-end GPU tests for textured console graphics pipelines through the
//! D3D12.X `CreateGraphicsPipelineState` hook:
//! - decoded console T# / S# descriptors with per-draw user-SGPR descriptor table switching
//!   and T# component swizzle in a single command list,
//! - static samplers extracted from SGPR constants into the pipeline's ABI root signature
//!   with no dynamic sampler heap entries.

use std::ffi::CStr;
use std::sync::{Mutex, MutexGuard};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::{pso, runner};
use xcl::gpu::{desc, xbox};
use xcl::hle::d3d12x::{pso_async, real};

const SIZE: u32 = 64;
const CLEAR: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
const V: u64 = 256; // VALU source code of v0

static SERIAL: Mutex<()> = Mutex::new(());

// (semantic name, system_value, register, mask); index = 0, component_type = 3 (float)
const VS_IN: [(&str, u32, u32, u8); 3] = [("POSITION", 0, 0, 0x7), ("COLOR", 0, 1, 0xF), ("TEXCOORD", 0, 2, 0x3)];
const VS_OUT_PS_IN: [(&str, u32, u32, u8); 3] = [("SV_Position", 1, 0, 0xF), ("COLOR", 0, 1, 0xF), ("TEXCOORD", 0, 2, 0x3)];
const PS_OUT: [(&str, u32, u32, u8); 1] = [("SV_Target", 64, 0, 0xF)];
const VERTEX_LAYOUT: [(&CStr, DXGI_FORMAT, u32); 3] = [
    (c"POSITION", DXGI_FORMAT_R32G32B32_FLOAT, 0),
    (c"COLOR", DXGI_FORMAT_R32G32B32A32_FLOAT, 12),
    (c"TEXCOORD", DXGI_FORMAT_R32G32_FLOAT, 28),
];

fn le(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn sig_part(elems: &[(&str, u32, u32, u8)]) -> Vec<u8> {
    let mut d = Vec::new();
    le(&mut d, elems.len() as u32);
    le(&mut d, 8);
    let mut names = Vec::new();
    let base = (8 + elems.len() * 32) as u32;
    for &(name, sv, reg, mask) in elems {
        let name_off = base + names.len() as u32;
        names.extend_from_slice(name.as_bytes());
        names.push(0);
        for w in [0, name_off, 0, sv, 3, reg] {
            le(&mut d, w);
        }
        d.extend_from_slice(&[mask, mask, 0, 0]);
        le(&mut d, 0);
    }
    d.extend_from_slice(&names);
    d.resize(d.len().next_multiple_of(4), 0);
    d
}

/// Retail-layout console VS/PS container: DXBC with ISG1, OSG1, and DXIL/XDAT -> XBCH -> FSSM.
fn console_shader(kind: u32, hw: u8, api: u8, code: &[u8], isg1: &[(&str, u32, u32, u8)], osg1: &[(&str, u32, u32, u8)], ps_ena: u32) -> Vec<u8> {
    let code_at = 0x308usize;
    let mut fssm = vec![0u8; code_at];
    fssm[0..4].copy_from_slice(b"FSSM");
    fssm[4..8].copy_from_slice(&0x101u32.to_le_bytes());
    fssm[0xC0..0xC8].copy_from_slice(&((code_at - 0x10) as u64).to_le_bytes());
    fssm[0xC8..0xD0].copy_from_slice(&(code.len() as u64).to_le_bytes());
    let stage = 0x0000_00C0_0002_0000u64 | ((api as u64) << 8) | (hw as u64);
    fssm[0x100..0x108].copy_from_slice(&stage.to_le_bytes());
    if hw == 5 {
        fssm[0x248..0x24C].copy_from_slice(&ps_ena.to_le_bytes());
    }
    fssm.extend_from_slice(code);
    let fssm_size = (fssm.len() - 0x10) as u32;
    fssm[8..12].copy_from_slice(&fssm_size.to_le_bytes());

    let z = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 6);
    let mut xbch = b"XBCH".to_vec();
    for x in [3, z.len() as u32, 0, fssm.len() as u32, 0] {
        le(&mut xbch, x);
    }
    xbch.extend_from_slice(&z);

    let (table, data_at) = (0x18 + 12, 0x18 + 36);
    let mut d = Vec::new();
    for x in [kind << 16 | 0x66, 0, u32::from_le_bytes(*b"DXIL"), 0x106, 0, 4] {
        le(&mut d, x);
    }
    d.extend_from_slice(b"XDAT");
    for x in [2, table as u32, 0x12, data_at as u32, 16, 0x0d, (data_at + 16) as u32, xbch.len() as u32] {
        le(&mut d, x);
    }
    d.extend_from_slice(&[0x5a; 16]);
    d.extend_from_slice(&xbch);
    d.resize(d.len().next_multiple_of(4), 0);
    let nobc = d.len();
    d.extend_from_slice(b"NOBC");
    let dwords = (d.len() / 4) as u32;
    d[4..8].copy_from_slice(&dwords.to_le_bytes());
    d[16..20].copy_from_slice(&((nobc - 8) as u32).to_le_bytes());

    let (isg, osg) = (sig_part(isg1), sig_part(osg1));
    let parts: [(&[u8; 4], &[u8]); 3] = [(b"ISG1", &isg), (b"OSG1", &osg), (b"DXIL", &d)];
    let mut v = vec![0u8; 32 + parts.len() * 4];
    v[0..4].copy_from_slice(b"DXBC");
    v[20..24].copy_from_slice(&1u32.to_le_bytes());
    v[28..32].copy_from_slice(&(parts.len() as u32).to_le_bytes());
    for (i, (fourcc, data)) in parts.iter().enumerate() {
        let off = v.len() as u32;
        v[32 + i * 4..36 + i * 4].copy_from_slice(&off.to_le_bytes());
        v.extend_from_slice(*fourcc);
        le(&mut v, data.len() as u32);
        v.extend_from_slice(data);
    }
    let total = v.len() as u32;
    v[24..28].copy_from_slice(&total.to_le_bytes());
    assert!(xbox::is_xbox(&v), "expected a console container");
    v
}

fn push_op(code: &mut Vec<u8>, name: &str, enc: &str, fields: &[(&str, u64)], lit: Option<u32>) {
    code.extend(encode(name, enc, fields, lit).unwrap_or_else(|e| panic!("{name}: {e}")));
}

fn push_interp(code: &mut Vec<u8>, vdst: u64, attr: u64, chans: u64) {
    for c in 0..chans {
        push_op(code, "V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", vdst + c), ("VSRC", 0), ("ATTR", attr), ("ATTRCHAN", c)], None);
        push_op(code, "V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", vdst + c), ("VSRC", 1), ("ATTR", attr), ("ATTRCHAN", c)], None);
    }
}

fn setup() -> Option<(ID3D12Device, MutexGuard<'static, ()>)> {
    let serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if xcl::gpu::recomp::dxc::location().is_err() {
        runner::skip_test("no DXC");
        return None;
    }
    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real {
        runner::skip_test("no GPU");
        return None;
    }
    if runner::wave64_programs_supported() != Ok(true) {
        runner::skip_test("no wave64");
        return None;
    }
    let raw = real::device_raw();
    println!("backend device: {}", real::stats().adapter);
    Some((unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap().clone(), serial))
}

fn textured_vs() -> Vec<u8> {
    let mut code = Vec::new();
    push_op(&mut code, "V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)], None); // w = 1.0
    push_op(&mut code, "EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None);
    push_op(&mut code, "EXP", "ENC_EXP", &[("TGT", 33), ("EN", 0x3), ("VSRC0", 12), ("VSRC1", 13), ("VSRC2", 12), ("VSRC3", 13)], None);
    push_op(&mut code, "EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1)], None);
    push_op(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
    console_shader(1, 4, 0, &code, &VS_IN, &VS_OUT_PS_IN, 0)
}

fn pso_through_hook(dev: &ID3D12Device, vs: &[u8], ps: &[u8]) -> (ID3D12PipelineState, ID3D12PipelineState, ID3D12RootSignature) {
    let game_rs: ID3D12RootSignature = unsafe {
        let desc = D3D12_ROOT_SIGNATURE_DESC { Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, ..Default::default() };
        let mut ser = None;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).unwrap();
        let ser = ser.unwrap();
        dev.CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize())).unwrap()
    };
    let elems = VERTEX_LAYOUT.map(|(name, fmt, off)| D3D12_INPUT_ELEMENT_DESC {
        SemanticName: windows::core::PCSTR(name.as_ptr() as *const u8),
        Format: fmt,
        AlignedByteOffset: off,
        InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
        ..Default::default()
    });
    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(game_rs)),
        VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
        PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() },
        SampleMask: u32::MAX,
        InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elems.as_ptr(), NumElements: elems.len() as u32 },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = true.into();
    desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;

    let before = real::stats();
    let pso: ID3D12PipelineState = unsafe { dev.CreateGraphicsPipelineState(&desc) }.expect("CreateGraphicsPipelineState");
    unsafe { std::mem::ManuallyDrop::drop(&mut desc.pRootSignature) };
    let real_raw = pso_async::resolve_wait(pso.as_raw() as usize);
    let after = real::stats();
    assert_eq!(after.translated_pipelines, before.translated_pipelines + 1, "PSO not translated: {:?}", after.translation_errors);
    assert_eq!(after.fallback_pipelines, before.fallback_pipelines, "fallback shaders used");

    let (real_ptr, rs_raw) = (real_raw as *mut std::ffi::c_void, real::pso_root_signature(real_raw).expect("ABI root signature") as *mut std::ffi::c_void);
    let real_pso = unsafe { ID3D12PipelineState::from_raw_borrowed(&real_ptr) }.unwrap().clone();
    let rs = unsafe { ID3D12RootSignature::from_raw_borrowed(&rs_raw) }.unwrap().clone();
    (pso, real_pso, rs)
}

fn console_tsharp(address: u64, width: u32, height: u32, dst_sel: [u32; 4]) -> [u32; 8] {
    let (w1, h1) = (width - 1, height - 1);
    [
        (address >> 8) as u32,
        ((address >> 40) as u32 & 0xFF) | 56 << 20 | (w1 & 3) << 30,
        (w1 >> 2) | h1 << 14,
        dst_sel[0] | dst_sel[1] << 3 | dst_sel[2] << 6 | dst_sel[3] << 9 | 9 << 28,
        0, 0, 0, 0,
    ]
}

fn user_data(table: u32) -> Vec<u32> {
    let mut io = vec![0u32; 128];
    io[8] = table;
    io
}

fn full_quad(color: [f32; 4], u_scale: f32, v_scale: f32) -> Vec<f32> {
    let mut verts = Vec::with_capacity(6 * 9);
    for (x, y) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        verts.extend_from_slice(&[x, y, 0.5, color[0], color[1], color[2], color[3], (x + 1.0) * 0.5 * u_scale, (1.0 - y) * 0.5 * v_scale]);
    }
    verts
}

fn texel_a(x: usize, y: usize) -> [u8; 4] {
    [(x * 4) as u8, (y * 4) as u8, if (x / 8 + y / 8).is_multiple_of(2) { 255 } else { 0 }, 255]
}

fn texel_b(x: usize, y: usize) -> [u8; 4] {
    [(255 - y * 4) as u8, 128, (x * 4) as u8, 200]
}

/// Textured console VS+PS through `CreateGraphicsPipelineState`: PS loads its T# (+0x20) and
/// dynamic S# (+0x40) from the descriptor table at `s[8:9]`, runs `image_sample`, and
/// multiplies by vertex COLOR. Two decoded console T#s (identity vs BGRA swizzle) are drawn
/// in one command list, switching only `s[8]` between draws.
#[test]
fn console_textured_ps_decoded_descriptors_and_switch_between_draws() {
    let Some((dev, _serial)) = setup() else { return };

    let mut ps_code = Vec::new();
    push_op(&mut ps_code, "S_LOAD_DWORDX8", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 4), ("OFFSET", 0x20), ("SOFFSET", 125)], None);
    push_op(&mut ps_code, "S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 24), ("SBASE", 4), ("OFFSET", 0x40), ("SOFFSET", 125)], None);
    push_interp(&mut ps_code, 4, 0, 4); // COLOR -> v[4..7]
    push_interp(&mut ps_code, 8, 1, 2); // TEXCOORD -> v[8..9]
    push_op(&mut ps_code, "IMAGE_SAMPLE", "ENC_MIMG", &[("VDATA", 12), ("VADDR", 8), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1)], None);
    for c in 0..4u64 {
        push_op(&mut ps_code, "V_MUL_F32", "ENC_VOP2", &[("VDST", 12 + c), ("SRC0", V + 12 + c), ("VSRC1", 4 + c)], None);
    }
    push_op(&mut ps_code, "EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 12), ("VSRC1", 13), ("VSRC2", 14), ("VSRC3", 15), ("DONE", 1), ("VM", 1)], None);
    push_op(&mut ps_code, "S_ENDPGM", "ENC_SOPP", &[], None);

    let ps = console_shader(0, 5, 1, &ps_code, &VS_OUT_PS_IN, &PS_OUT, 2);
    let (_handle, pso, rs) = pso_through_hook(&dev, &textured_vs(), &ps);

    let (addr_a, addr_b) = (0x1_0000u64, 0x2_0000u64);
    let mut mem = vec![0u32; 0x2_4000 / 4];
    for (at, texel) in [(addr_a, texel_a as fn(usize, usize) -> [u8; 4]), (addr_b, texel_b)] {
        for y in 0..SIZE as usize {
            for x in 0..SIZE as usize {
                mem[at as usize / 4 + y * SIZE as usize + x] = u32::from_le_bytes(texel(x, y));
            }
        }
    }
    let t_a = console_tsharp(addr_a, SIZE, SIZE, [4, 5, 6, 7]);
    let t_b = console_tsharp(addr_b, SIZE, SIZE, [6, 5, 4, 7]); // BGRA swizzle: r <- z, b <- x
    let s_dyn = [2 | 2 << 3 | 2 << 6, 0xFFF << 12, 1 << 20 | 1 << 22 | 2 << 26, 0]; // linear, clamp

    let mut textures = Vec::new();
    for t in [t_a, t_b] {
        let img = desc::image(&t).expect("T#");
        assert_eq!(img.swizzle_mode, 0);
        let (res, srv) = desc::image_d3d12(&img).expect("DXGI format");
        assert_eq!((res.Width, res.Height, res.Format), (SIZE as u64, SIZE, DXGI_FORMAT_R8G8B8A8_UNORM));
        let at = img.address as usize;
        let data: Vec<u8> = mem[at / 4..at / 4 + (SIZE * SIZE) as usize].iter().flat_map(|w| w.to_le_bytes()).collect();
        textures.push(runner::ViewTexture { res, srv, data, row_bytes: SIZE as usize * 4 });
    }
    let sampler = desc::sampler_d3d12(&s_dyn, false);
    assert_eq!((sampler.Filter, sampler.AddressU), (D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_CLAMP));

    for (table, t, heap_index) in [(0x100usize, t_a, 0u32), (0x200, t_b, 2)] {
        mem[(table + 0x20) / 4..(table + 0x20) / 4 + 8].copy_from_slice(&desc::xcl_image_words(&t, heap_index));
        mem[(table + 0x40) / 4..(table + 0x40) / 4 + 4].copy_from_slice(&desc::xcl_sampler_words(&s_dyn, 0));
    }

    let tint = [0.75f32, 0.5, 1.0, 0.5];
    let half = SIZE as i32 / 2;
    let draws = [
        runner::Draw { sgprs: user_data(0x100), scissor: [0, 0, half, SIZE as i32] },
        runner::Draw { sgprs: user_data(0x200), scissor: [half, 0, SIZE as i32, SIZE as i32] },
    ];
    let px = runner::draw_on_multi(&dev, &pso, &rs, &full_quad(tint, 1.0, 1.0), 36, &draws, &mem, &textures, &[sampler], SIZE, CLEAR).expect("draw");

    let unorm = |t: [u8; 4]| [0, 1, 2, 3].map(|k| (t[k] as f32 / 255.0) * tint[k]);
    let mut changed = 0usize;
    for y in 0..SIZE as usize {
        for x in 0..SIZE as usize {
            let got = px[y * SIZE as usize + x];
            let a = unorm(texel_a(x, y));
            let b_raw = texel_b(x, y);
            let b = unorm([b_raw[2], b_raw[1], b_raw[0], b_raw[3]]);
            let want = if x < half as usize { a } else { b };
            assert!((0..4).all(|k| (got[k] - want[k]).abs() < 0.005), "pixel ({x},{y}): {got:?}, expected {want:?}");
            if x >= half as usize && (0..4).any(|k| (got[k] - a[k]).abs() > 0.05) {
                changed += 1;
            }
        }
    }
    let right = (SIZE * SIZE / 2) as usize;
    assert!(changed > right * 9 / 10, "descriptor table switch did not affect right half: {changed} / {right}");
}

/// Static sampler extraction through `CreateGraphicsPipelineState`: PS constructs two S#
/// descriptors from SGPR constants (`POINT + BORDER(opaque white)` and `LINEAR + WRAP`) and
/// samples a 4x4 texture over `u in [0, 2]` with `&[]` dynamic samplers, proving both static
/// samplers are wired into the pipeline's ABI root signature.
#[test]
fn console_ps_static_sampler_from_constants_in_abi_root_signature() {
    let Some((dev, _serial)) = setup() else { return };

    let s_border_point = [6 | 6 << 3 | 6 << 6, 0xFFF << 12, 0, 2 << 30];
    let s_wrap_linear = [0, 0xFFF << 12, 1 << 20 | 1 << 22, 0];

    let mut ps_code = Vec::new();
    push_op(&mut ps_code, "S_LOAD_DWORDX8", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 4), ("OFFSET", 0x20), ("SOFFSET", 125)], None);
    for (base, s) in [(24u64, s_border_point), (28, s_wrap_linear)] {
        for (k, &w) in s.iter().enumerate() {
            let (src, lit) = if w == 0 { (128, None) } else { (255, Some(w)) };
            push_op(&mut ps_code, "S_MOV_B32", "ENC_SOP1", &[("SDST", base + k as u64), ("SSRC0", src)], lit);
        }
    }
    push_interp(&mut ps_code, 4, 1, 2); // TEXCOORD -> v[4..5]
    push_op(&mut ps_code, "IMAGE_SAMPLE", "ENC_MIMG", &[("VDATA", 8), ("VADDR", 4), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1)], None);
    push_op(&mut ps_code, "IMAGE_SAMPLE", "ENC_MIMG", &[("VDATA", 12), ("VADDR", 4), ("SRSRC", 4), ("SSAMP", 7), ("DMASK", 0xF), ("DIM", 1)], None);
    push_op(&mut ps_code, "EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 12), ("VSRC3", 13), ("DONE", 1), ("VM", 1)], None);
    push_op(&mut ps_code, "S_ENDPGM", "ENC_SOPP", &[], None);

    let ps = console_shader(0, 5, 1, &ps_code, &VS_OUT_PS_IN, &PS_OUT, 2);
    let (_handle, pso, rs) = pso_through_hook(&dev, &textured_vs(), &ps);
    let compiled = pso::console_to_dxil_for(&ps, pso::Stage::Pixel, 0, pso::graphics_wave32_for_device(&dev)).expect("compiled PS");
    assert_eq!(compiled.samplers, vec![(s_border_point, false), (s_wrap_linear, false)], "static samplers extracted from PS constants");

    let levels = [0u8, 85, 170, 255];
    let level_f = levels.map(|c| c as f32 / 255.0);
    let tex_bytes: Vec<u8> = levels.iter().flat_map(|&gy| levels.iter().flat_map(move |&rx| [rx, gy, 0, 255])).collect();
    let t_raw = console_tsharp(0x1000, 4, 4, [4, 5, 6, 7]);
    let (res, srv) = desc::image_d3d12(&desc::image(&t_raw).unwrap()).unwrap();
    let tex = runner::ViewTexture { res, srv, data: tex_bytes, row_bytes: 4 * 4 };

    let mut mem = vec![0u32; 0x100];
    mem[(0x100 + 0x20) / 4..(0x100 + 0x20) / 4 + 8].copy_from_slice(&desc::xcl_image_words(&t_raw, 0));
    let draws = [runner::Draw { sgprs: user_data(0x100), scissor: [0, 0, SIZE as i32, SIZE as i32] }];
    let px = runner::draw_on_multi(&dev, &pso, &rs, &full_quad([1.0; 4], 2.0, 1.0), 36, &draws, &mem, &[tex], &[], SIZE, CLEAR).expect("draw");

    let linear_wrap_1d = |c: f32| {
        let p = c.rem_euclid(1.0) * 4.0 - 0.5;
        let i0 = p.floor() as i32;
        let f = p - i0 as f32;
        level_f[i0.rem_euclid(4) as usize] * (1.0 - f) + level_f[(i0 + 1).rem_euclid(4) as usize] * f
    };

    let (mut filter_diffs, mut border_hits) = (0usize, 0usize);
    for y in 0..SIZE as usize {
        for x in 0..SIZE as usize {
            let (u, v) = ((x as f32 + 0.5) / 32.0, (y as f32 + 0.5) / 64.0);
            let got = px[y * SIZE as usize + x];
            let (want_lr, want_lg) = (linear_wrap_1d(u), linear_wrap_1d(v));
            assert!((got[2] - want_lr).abs() < 0.01 && (got[3] - want_lg).abs() < 0.01, "pixel ({x},{y}) static linear+wrap: {got:?}, expected ({want_lr}, {want_lg})");

            let (want_pr, want_pg) = if u > 1.0 {
                border_hits += 1;
                (1.0, 1.0)
            } else {
                (level_f[(u * 4.0).floor() as usize], level_f[(v * 4.0).floor() as usize])
            };
            assert!((got[0] - want_pr).abs() < 0.01 && (got[1] - want_pg).abs() < 0.01, "pixel ({x},{y}) static point+border: {got:?}, expected ({want_pr}, {want_pg})");

            if u < 1.0 && ((got[0] - got[2]).abs() > 0.05 || (got[1] - got[3]).abs() > 0.05) {
                filter_diffs += 1;
            }
        }
    }
    assert_eq!(border_hits, (SIZE * SIZE / 2) as usize, "right half must sample opaque white border");
    assert!(filter_diffs > 500, "static point and linear samplers must produce distinct filtering in left half: {filter_diffs}");
}
