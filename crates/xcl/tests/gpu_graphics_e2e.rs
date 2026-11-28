//! End-to-end GPU tests for the D3D12.X graphics pipeline hooks without retail game files.
//!
//! Hand-encoded console containers (DXBC with ISG1 / OSG1 and a DXIL part carrying
//! XDAT -> XBCH -> FSSM) go through `CreateGraphicsPipelineState` and
//! `ID3D12Device2::CreatePipelineState`, compile to DXIL, draw on the adapter and check
//! pixels. Pipeline counters are verified so fallback stand-ins cannot pass.

use std::ffi::{c_void, CStr};
use std::mem::ManuallyDrop;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::{dxc, runner};
use xcl::hle::d3d12x::{pso_async, real};

static SERIAL: Mutex<()> = Mutex::new(());

const V: u64 = 256;
const CLEAR: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
const E_INVALIDARG: i32 = 0x8007_0057u32 as i32;
const LAYOUT: [(&CStr, DXGI_FORMAT, u32); 2] = [
    (c"POSITION", DXGI_FORMAT_R32G32_FLOAT, 0),
    (c"COLOR", DXGI_FORMAT_R32G32B32A32_FLOAT, 8),
];

fn le(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

type Op<'a> = (&'a str, &'a str, &'a [(&'a str, u64)], Option<u32>);

fn asm(list: &[Op<'_>]) -> Vec<u8> {
    list.iter()
        .flat_map(|(n, e, f, l)| encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")))
        .collect()
}

fn dxbc(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let mut out = vec![0u8; 32 + 4 * parts.len()];
    out[0..4].copy_from_slice(b"DXBC");
    out[20..24].copy_from_slice(&1u32.to_le_bytes());
    out[28..32].copy_from_slice(&(parts.len() as u32).to_le_bytes());
    for (i, (fourcc, data)) in parts.iter().enumerate() {
        let off = out.len() as u32;
        out[32 + 4 * i..36 + 4 * i].copy_from_slice(&off.to_le_bytes());
        out.extend_from_slice(*fourcc);
        le(&mut out, data.len() as u32);
        out.extend_from_slice(data);
    }
    let total = out.len() as u32;
    out[24..28].copy_from_slice(&total.to_le_bytes());
    out
}

/// ISG1 / OSG1 part from `(name, index, system_value, component_type, register, mask)`.
fn sig_part(elems: &[(&str, u32, u32, u32, u32, u8)]) -> Vec<u8> {
    let (mut out, mut names) = (Vec::new(), Vec::new());
    le(&mut out, elems.len() as u32);
    le(&mut out, 8);
    let base = 8 + elems.len() * 32;
    for &(name, index, sv, ct, reg, mask) in elems {
        let off = (base + names.len()) as u32;
        names.extend_from_slice(name.as_bytes());
        names.push(0);
        for w in [0, off, index, sv, ct, reg] {
            le(&mut out, w);
        }
        out.extend_from_slice(&[mask, mask, 0, 0]);
        le(&mut out, 0);
    }
    out.extend_from_slice(&names);
    out.resize((out.len() + 3) & !3, 0);
    out
}

/// Console shader container: DXBC with `extra_parts` and a DXIL part holding XDAT -> XBCH -> FSSM.
fn console_shader(dxil_kind: u32, hw: u8, api: u8, code: &[u8], patch: impl FnOnce(&mut [u8]), extra_parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let code_at = 0x308usize;
    let mut fssm = vec![0u8; code_at];
    fssm[0..4].copy_from_slice(b"FSSM");
    fssm[4..8].copy_from_slice(&0x101u32.to_le_bytes());
    fssm[0xC0..0xC8].copy_from_slice(&((code_at - 0x10) as u64).to_le_bytes());
    fssm[0xC8..0xD0].copy_from_slice(&(code.len() as u64).to_le_bytes());
    let stage = 0x0000_00C0_0002_0000u64 | ((api as u64) << 8) | (hw as u64);
    fssm[0x100..0x108].copy_from_slice(&stage.to_le_bytes());
    patch(&mut fssm);
    fssm.extend_from_slice(code);
    let fssm_size = (fssm.len() - 0x10) as u32;
    fssm[8..12].copy_from_slice(&fssm_size.to_le_bytes());

    let z = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 6);
    let mut xbch = b"XBCH".to_vec();
    for x in [3, z.len() as u32, 0, fssm.len() as u32, 0] {
        le(&mut xbch, x);
    }
    xbch.extend_from_slice(&z);

    let (table, data_at) = (0x24u32, 0x3Cu32);
    let mut d = Vec::new();
    for x in [dxil_kind << 16 | 0x66, 0, u32::from_le_bytes(*b"DXIL"), 0x106, 0, 4] {
        le(&mut d, x);
    }
    d.extend_from_slice(b"XDAT");
    for x in [2, table, 0x12, data_at, 16, 0x0d, data_at + 16, xbch.len() as u32] {
        le(&mut d, x);
    }
    d.extend_from_slice(&[0x5a; 16]);
    d.extend_from_slice(&xbch);
    d.resize((d.len() + 3) & !3, 0);
    let nobc = d.len();
    d.extend_from_slice(b"NOBC");
    let dwords = (d.len() / 4) as u32;
    d[4..8].copy_from_slice(&dwords.to_le_bytes());
    d[16..20].copy_from_slice(&((nobc - 8) as u32).to_le_bytes());

    let mut parts = extra_parts.to_vec();
    parts.push((b"DXIL", &d));
    dxbc(&parts)
}

fn require_gpu() -> Option<ID3D12Device> {
    if dxc::location().is_err() {
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
    Some(unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap().clone())
}

fn empty_rs(dev: &ID3D12Device) -> ID3D12RootSignature {
    let desc = D3D12_ROOT_SIGNATURE_DESC { Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, ..Default::default() };
    unsafe {
        let mut ser = None;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).unwrap();
        let ser = ser.unwrap();
        dev.CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize())).unwrap()
    }
}

fn input_elements() -> Vec<D3D12_INPUT_ELEMENT_DESC> {
    LAYOUT
        .iter()
        .map(|(name, fmt, off)| D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::PCSTR(name.as_ptr() as *const u8),
            Format: *fmt,
            AlignedByteOffset: *off,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            ..Default::default()
        })
        .collect()
}

fn create_graphics_pso(dev: &ID3D12Device, vs: &[u8], ps: &[u8], topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE) -> ID3D12PipelineState {
    let elems = input_elements();
    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(empty_rs(dev))),
        VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
        PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() },
        SampleMask: u32::MAX,
        InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elems.as_ptr(), NumElements: elems.len() as u32 },
        PrimitiveTopologyType: topology,
        NumRenderTargets: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = true.into();
    desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    let pso: Result<ID3D12PipelineState, _> = unsafe { dev.CreateGraphicsPipelineState(&desc) };
    unsafe { ManuallyDrop::drop(&mut desc.pRootSignature) };
    pso.expect("CreateGraphicsPipelineState through the backend hook")
}

struct ColdCache(std::path::PathBuf, Option<std::ffi::OsString>);

impl ColdCache {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("xcl-{tag}-{}", std::process::id()));
        let prev = std::env::var_os("XCL_DXIL_CACHE");
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var("XCL_DXIL_CACHE", &dir);
        Self(dir, prev)
    }
}

impl Drop for ColdCache {
    fn drop(&mut self) {
        match &self.1 {
            Some(prev) => std::env::set_var("XCL_DXIL_CACHE", prev),
            None => std::env::remove_var("XCL_DXIL_CACHE"),
        }
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn vs_ps_sigs() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        sig_part(&[("POSITION", 0, 0, 3, 0, 0x3), ("COLOR", 0, 0, 3, 1, 0xF)]),
        sig_part(&[("SV_Position", 0, 1, 3, 0, 0xF), ("COLOR", 0, 0, 3, 1, 0xF)]),
        sig_part(&[("SV_Target", 0, 64, 3, 0, 0xF)]),
    )
}

fn passthrough_vs(vs_in: &[u8], vs_out: &[u8]) -> Vec<u8> {
    let code = asm(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
        ("S_ENDPGM", "ENC_SOPP", &[], None),
    ]);
    console_shader(1, 4, 0, &code, |_| {}, &[(b"ISG1", vs_in), (b"OSG1", vs_out)])
}

/// Interpolating PS, optionally discarding pixels with `pos.x <= clip_x` and scaling red.
fn color_ps(ps_in: &[u8], ps_out: &[u8], clip_x: Option<f32>, red_scale: f32) -> Vec<u8> {
    let mut code = Vec::new();
    if let Some(x) = clip_x {
        code.extend(encode("V_CMPX_LT_F32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 2)], Some(x.to_bits())).unwrap());
    }
    for ch in 0..4u64 {
        code.extend(encode("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 4 + ch), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", ch)], None).unwrap());
        code.extend(encode("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4 + ch), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", ch)], None).unwrap());
    }
    code.extend(asm(&[
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 255), ("VSRC1", 4)], Some(red_scale.to_bits())),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
        ("S_ENDPGM", "ENC_SOPP", &[], None),
    ]));
    console_shader(0, 5, 1, &code, |f| f[0x248..0x24C].copy_from_slice(&0x302u32.to_le_bytes()), &[(b"ISG1", ps_in), (b"OSG1", ps_out)])
}

fn borrow_pso_and_rs(real_raw: usize) -> (ID3D12PipelineState, ID3D12RootSignature) {
    let rs_raw = real::pso_root_signature(real_raw).expect("ABI root signature") as *mut c_void;
    unsafe {
        (
            ID3D12PipelineState::from_raw_borrowed(&(real_raw as *mut c_void)).unwrap().clone(),
            ID3D12RootSignature::from_raw_borrowed(&rs_raw).unwrap().clone(),
        )
    }
}

fn resolve_translated(before: (u64, u64), pso: &ID3D12PipelineState, rect_gs: bool) -> (ID3D12PipelineState, ID3D12RootSignature) {
    let real_raw = pso_async::resolve_wait(pso.as_raw() as usize);
    let after = real::stats();
    assert_eq!(after.translated_pipelines, before.0 + 1, "{:?}", after.translation_errors);
    assert_eq!(after.fallback_pipelines, before.1);
    assert!(real::is_translated_graphics(real_raw));
    assert_eq!(real::has_rect_list_gs(real_raw), rect_gs);
    borrow_pso_and_rs(real_raw)
}

fn pso_counts() -> (u64, u64) {
    let s = real::stats();
    (s.translated_pipelines, s.fallback_pipelines)
}

fn assert_rgba(got: [f32; 4], want: [f32; 4], label: &str) {
    assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4), "{label}: got {got:?}, want {want:?}");
}

/// Console VS (reading a scale+bias V# from user SGPR `s[8:9]`) + console PS (clipping
/// `pos.x <= 16` and scaling red) go through `CreateGraphicsPipelineState` and read back pixels.
#[test]
fn console_vs_ps_through_the_pso_hook_render_and_read_back() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = require_gpu() else { return };
    let (vs_in, shared, ps_out) = vs_ps_sigs();

    // s[8:9] -> descriptor table with one V# -> float4(sx, sy, bx, by) in s[16:19]
    let vs_code = asm(&[
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 6), ("OFFSET", 0), ("SOFFSET", 125)], None),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 0), ("SRC0", 16), ("VSRC1", 4)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 0), ("SRC0", 18), ("VSRC1", 0)], None),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 17), ("VSRC1", 5)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 19), ("VSRC1", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
        ("S_ENDPGM", "ENC_SOPP", &[], None),
    ]);
    let vs = console_shader(1, 4, 0, &vs_code, |_| {}, &[(b"ISG1", &vs_in), (b"OSG1", &shared)]);
    let ps = color_ps(&shared, &ps_out, Some(16.0), 0.5);
    assert!(xcl::gpu::xbox::is_xbox(&vs) && xcl::gpu::xbox::is_xbox(&ps));

    let before = pso_counts();
    let pso = create_graphics_pso(&dev, &vs, &ps, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    let (real_pso, rs) = resolve_translated(before, &pso, false);

    let mut io = vec![0u32; 128];
    io[8] = 0x100;
    let mut mem = vec![0u32; 0x100];
    mem[0x100 / 4..0x100 / 4 + 4].copy_from_slice(&[0x200, 0, 0x100, 0]);
    for (k, v) in [0.5f32, 0.5, 0.25, 0.0].into_iter().enumerate() {
        mem[0x200 / 4 + k] = v.to_bits();
    }
    // Maps [-1, 1]^2 to NDC x in [-0.25, 0.75] (px 12..28), y in [-0.5, 0.5] (px 8..24); PS clips x < 16.
    let verts: Vec<f32> = [[-1.0f32, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
        .iter()
        .flat_map(|p| [p[0], p[1], 0.8, 0.6, 0.4, 1.0])
        .collect();
    let px = runner::draw_on(&dev, &real_pso, &rs, &verts, 24, &io, &mem, 32, CLEAR).expect("draw");
    for y in 0..32usize {
        for x in 0..32usize {
            let want = if (16..28).contains(&x) && (8..24).contains(&y) { [0.4, 0.6, 0.4, 1.0] } else { CLEAR };
            assert_rgba(px[y * 32 + x], want, &format!("({x},{y})"));
        }
    }
}

/// Console rect-list topology (19) through `CreateGraphicsPipelineState`: the hook attaches
/// `rect_list_gs` and sets `has_rect_list_gs`; 3 corners expand across the implied 4th corner.
#[test]
fn console_rect_list_pso_through_the_hook_draws_fourth_corner() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = require_gpu() else { return };
    let (vs_in, shared, ps_out) = vs_ps_sigs();

    let before = pso_counts();
    let pso = create_graphics_pso(&dev, &passthrough_vs(&vs_in, &shared), &color_ps(&shared, &ps_out, None, 1.0), D3D12_PRIMITIVE_TOPOLOGY_TYPE(19));
    let (real_pso, rs) = resolve_translated(before, &pso, true);

    let verts: Vec<f32> = [
        ([-1.0f32, 1.0], [0.0f32, 0.0, 0.0, 1.0]),
        ([-1.0, -1.0], [0.0, 1.0, 0.0, 1.0]),
        ([1.0, -1.0], [1.0, 1.0, 0.0, 1.0]),
    ]
    .iter()
    .flat_map(|(p, c)| [p[0], p[1], c[0], c[1], c[2], c[3]])
    .collect();
    let n = 16usize;
    let px = runner::draw_on(&dev, &real_pso, &rs, &verts, 24, &[], &[], n as u32, CLEAR).expect("draw");
    for y in 0..n {
        for x in 0..n {
            assert_rgba(px[y * n + x], [(x as f32 + 0.5) / n as f32, (y as f32 + 0.5) / n as f32, 0.0, 1.0], &format!("({x},{y})"));
        }
    }
}

/// Cold DXIL cache: `CreateGraphicsPipelineState` returns a stand-in and builds the real PSO in
/// the background; dropping the stand-in forgets and releases the real PSO.
#[test]
fn async_graphics_pso_registers_real_pso_and_cleans_up_on_drop() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = require_gpu() else { return };
    if !pso_async::enabled() {
        runner::skip_test("XCL_ASYNC_PSO=0");
        return;
    }
    let _cold = ColdCache::new("async-graphics-pso");
    let (vs_in, shared, ps_out) = vs_ps_sigs();
    let pso = create_graphics_pso(&dev, &passthrough_vs(&vs_in, &shared), &color_ps(&shared, &ps_out, None, 0.75), D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    let standin_raw = pso.as_raw() as usize;
    assert!(!real::is_translated_graphics(standin_raw));
    let real_raw = pso_async::resolve_wait(standin_raw);
    assert_ne!(real_raw, standin_raw, "cold cache must build graphics PSO behind a stand-in");
    assert!(real::is_translated_graphics(real_raw) && !real::is_translated_graphics(standin_raw));
    assert_eq!(real::pso_root_signature(standin_raw), real::pso_root_signature(real_raw));

    let (real_pso, rs) = borrow_pso_and_rs(real_raw);
    let verts: Vec<f32> = [[-1.0f32, -1.0], [-1.0, 3.0], [3.0, -1.0]].iter().flat_map(|p| [p[0], p[1], 0.8, 0.2, 0.6, 1.0]).collect();
    for p in runner::draw_on(&dev, &real_pso, &rs, &verts, 24, &[], &[], 8, CLEAR).expect("draw") {
        assert_rgba(p, [0.6, 0.2, 0.6, 1.0], "async pixel");
    }

    drop(real_pso);
    drop(pso);
    assert!(!real::is_translated_graphics(real_raw));
    assert_eq!(pso_async::resolve(standin_raw), standin_raw);
}

#[repr(C, align(8))]
struct Sub<T> {
    ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    v: T,
}

#[repr(C)]
struct GraphicsStream {
    rs: Sub<*mut c_void>,
    vs: Sub<D3D12_SHADER_BYTECODE>,
    ps: Sub<D3D12_SHADER_BYTECODE>,
    blend: Sub<D3D12_BLEND_DESC>,
    mask: Sub<u32>,
    rast: Sub<D3D12_RASTERIZER_DESC>,
    il: Sub<D3D12_INPUT_LAYOUT_DESC>,
    topo: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    rt: Sub<D3D12_RT_FORMAT_ARRAY>,
    samples: Sub<DXGI_SAMPLE_DESC>,
}

#[repr(C)]
struct AsSlotStream {
    rs: Sub<*mut c_void>,
    as_shader: Sub<D3D12_SHADER_BYTECODE>,
    rt: Sub<D3D12_RT_FORMAT_ARRAY>,
}

/// `ID3D12Device2::CreatePipelineState` stream hook: a console VS+PS graphics stream translates
/// and renders, while a console MS container placed in the AS slot is refused (`E_INVALIDARG`).
#[test]
fn console_pipeline_stream_routes_graphics_and_refuses_ms_in_as_slot() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = require_gpu() else { return };
    let (vs_in, shared, ps_out) = vs_ps_sigs();
    let vs = passthrough_vs(&vs_in, &shared);
    let ps = color_ps(&shared, &ps_out, None, 0.25);
    let ms = console_shader(13, 7, 6, &asm(&[("S_ENDPGM", "ENC_SOPP", &[], None)]), |_| {}, &[(b"OSG1", &shared)]);

    let game_rs = empty_rs(&dev);
    let d2: ID3D12Device2 = dev.cast().unwrap();
    let bc = |b: &[u8]| D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as _, BytecodeLength: b.len() };
    let mut rt = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
    rt.RTFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;

    let bad = AsSlotStream {
        rs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, v: game_rs.as_raw() },
        as_shader: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, v: bc(&ms) },
        rt: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, v: rt },
    };
    let bad_desc = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: std::mem::size_of::<AsSlotStream>(), pPipelineStateSubobjectStream: &bad as *const _ as *mut _ };
    assert_eq!(unsafe { d2.CreatePipelineState::<ID3D12PipelineState>(&bad_desc) }.unwrap_err().code().0, E_INVALIDARG);

    let elems = input_elements();
    let mut blend = D3D12_BLEND_DESC::default();
    blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    let rast = D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() };
    let st = GraphicsStream {
        rs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, v: game_rs.as_raw() },
        vs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, v: bc(&vs) },
        ps: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, v: bc(&ps) },
        blend: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, v: blend },
        mask: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, v: u32::MAX },
        rast: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, v: rast },
        il: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, v: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elems.as_ptr(), NumElements: elems.len() as u32 } },
        topo: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, v: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE },
        rt: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, v: rt },
        samples: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, v: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 } },
    };
    let before = pso_counts();
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: std::mem::size_of::<GraphicsStream>(), pPipelineStateSubobjectStream: &st as *const _ as *mut _ };
    let pso: ID3D12PipelineState = unsafe { d2.CreatePipelineState(&desc) }.expect("graphics stream through hook");
    let (real_pso, rs) = resolve_translated(before, &pso, false);

    let verts: Vec<f32> = [[-1.0f32, -1.0], [-1.0, 3.0], [3.0, -1.0]].iter().flat_map(|p| [p[0], p[1], 0.8, 0.5, 0.25, 1.0]).collect();
    for p in runner::draw_on(&dev, &real_pso, &rs, &verts, 24, &[], &[], 8, CLEAR).expect("draw") {
        assert_rgba(p, [0.2, 0.5, 0.25, 1.0], "stream pixel");
    }
}
