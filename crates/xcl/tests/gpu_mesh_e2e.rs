//! Game-free end-to-end GPU tests for XCL's two-pass Mesh (NGG) and legacy Geometry (ES+GS)
//! D3D12.X pipeline hooks: hand-encoded console containers go through `CreatePipelineState`
//! and `CreateGraphicsPipelineState`, both passes execute on the backend GPU (`Dispatch` +
//! `DispatchMesh`), and rendered pixels plus two-pass drop cleanup are verified.

use std::ffi::c_void;
use std::mem::{size_of, size_of_val, transmute_copy, ManuallyDrop};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::{runner, IO_GS_PRIMS, IO_GS_VERTICES, MESH_HEADER};
use xcl::hle::d3d12x::{pso_async, real};

const SIZE: u32 = 32;
const CLEAR: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
const V: u64 = 256; // VALU source code of v0
static SERIAL: Mutex<()> = Mutex::new(());

type Op<'a> = (&'a str, &'a str, &'a [(&'a str, u64)], Option<u32>);

fn put32(v: &mut Vec<u8>, xs: &[u32]) { xs.iter().for_each(|x| v.extend_from_slice(&x.to_le_bytes())); }
fn emit(out: &mut Vec<u8>, n: &str, e: &str, f: &[(&str, u64)], l: Option<u32>) { out.extend(encode(n, e, f, l).unwrap_or_else(|x| panic!("{n}: {x}"))); }
fn asm(ops: &[Op<'_>]) -> Vec<u8> { let mut o = Vec::new(); ops.iter().for_each(|&(n, e, f, l)| emit(&mut o, n, e, f, l)); o }
fn bc(b: &[u8]) -> D3D12_SHADER_BYTECODE { D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as _, BytecodeLength: b.len() } }

fn dxbc(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let mut v = vec![0u8; 32 + parts.len() * 4];
    v[0..4].copy_from_slice(b"DXBC");
    v[20..24].copy_from_slice(&1u32.to_le_bytes());
    v[28..32].copy_from_slice(&(parts.len() as u32).to_le_bytes());
    for (i, (cc, data)) in parts.iter().enumerate() {
        let off = v.len() as u32;
        v[32 + i * 4..36 + i * 4].copy_from_slice(&off.to_le_bytes());
        v.extend_from_slice(*cc);
        put32(&mut v, &[data.len() as u32]);
        v.extend_from_slice(data);
    }
    let total = v.len() as u32;
    v[24..28].copy_from_slice(&total.to_le_bytes());
    v
}

/// ISG1 / OSG1 signature part: `(name, system_value, component_type, register, mask)`.
fn sig_part(elems: &[(&str, u32, u32, u32, u8)]) -> Vec<u8> {
    let (mut d, mut names, base) = (Vec::new(), Vec::new(), 8 + elems.len() * 32);
    put32(&mut d, &[elems.len() as u32, 8]);
    for &(name, sv, ct, reg, mask) in elems {
        let off = (base + names.len()) as u32;
        names.extend_from_slice(name.as_bytes());
        names.push(0);
        put32(&mut d, &[0, off, 0, sv, ct, reg]);
        d.extend_from_slice(&[mask, mask, 0, 0, 0, 0, 0, 0]);
    }
    d.extend_from_slice(&names);
    d
}

/// Console DXIL part wrapping `code` in FSSM (`hw`, `api`) -> XBCH -> XDAT -> NOBC.
fn console_dxil(kind: u32, code: &[u8], hw: u8, api: u8, ps_ena: Option<u32>) -> Vec<u8> {
    let mut fssm = vec![0u8; 0x308];
    fssm[0..4].copy_from_slice(b"FSSM");
    fssm[4..8].copy_from_slice(&0x101u32.to_le_bytes());
    fssm[0xC0..0xC8].copy_from_slice(&0x2F8u64.to_le_bytes());
    fssm[0xC8..0xD0].copy_from_slice(&(code.len() as u64).to_le_bytes());
    fssm[0x100..0x108].copy_from_slice(&(0x0000_00C0_0002_0000u64 | (api as u64) << 8 | hw as u64).to_le_bytes());
    if let Some(ena) = ps_ena { fssm[0x248..0x24C].copy_from_slice(&ena.to_le_bytes()); }
    fssm.extend_from_slice(code);
    let fssm_sz = (fssm.len() - 0x10) as u32;
    fssm[8..12].copy_from_slice(&fssm_sz.to_le_bytes());
    let z = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 6);
    let mut xbch = b"XBCH".to_vec();
    put32(&mut xbch, &[3, z.len() as u32, 0, fssm.len() as u32, 0]);
    xbch.extend_from_slice(&z);
    let mut d = Vec::new();
    put32(&mut d, &[kind << 16 | 0x66, 0, u32::from_le_bytes(*b"DXIL"), 0x106, 0, 4]);
    d.extend_from_slice(b"XDAT");
    put32(&mut d, &[2, 0x24, 0x12, 0x3C, 16, 0x0d, 0x4C, xbch.len() as u32]);
    d.extend_from_slice(&[0x5a; 16]);
    d.extend_from_slice(&xbch);
    d.resize((d.len() + 3) & !3, 0);
    let nobc = d.len();
    d.extend_from_slice(b"NOBC");
    let dwords = (d.len() / 4) as u32;
    d[4..8].copy_from_slice(&dwords.to_le_bytes());
    d[16..20].copy_from_slice(&((nobc - 8) as u32).to_le_bytes());
    d
}

fn psv_part(info: &[u8]) -> Vec<u8> {
    let mut p = (info.len() as u32).to_le_bytes().to_vec();
    p.extend_from_slice(info);
    put32(&mut p, &[0]);
    p
}

/// Console PS reading interpolated `COLOR0` (`attr 0`) and writing `SV_Target0`.
fn console_ps(halve_red: bool) -> Vec<u8> {
    let mut code = Vec::new();
    for c in 0..4u64 {
        emit(&mut code, "V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 4 + c), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", c)], None);
        emit(&mut code, "V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4 + c), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", c)], None);
    }
    if halve_red { emit(&mut code, "V_MUL_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 240), ("VSRC1", 4)], None); }
    emit(&mut code, "EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None);
    emit(&mut code, "S_ENDPGM", "ENC_SOPP", &[], None);
    let isg = sig_part(&[("SV_Position", 1, 3, 0, 0xF), ("COLOR", 0, 3, 1, 0xF)]);
    let osg = sig_part(&[("SV_Target", 64, 3, 0, 0xF)]);
    dxbc(&[(b"ISG1", &isg), (b"OSG1", &osg), (b"DXIL", &console_dxil(0, &code, 5, 1, Some(0x2)))])
}

fn setup() -> Option<(ID3D12Device, std::sync::MutexGuard<'static, ()>)> {
    let serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for (ok, reason) in [
        (xcl::gpu::recomp::dxc::location().is_ok(), "no DXC"),
        (xcl::hle::d3d12x::backend() == xcl::hle::d3d12x::Backend::Real, "no GPU"),
        (runner::wave64_programs_supported() == Ok(true), "no wave64"),
    ] {
        if !ok {
            println!("{reason} – skipping");
            runner::skip_test(reason);
            return None;
        }
    }
    let raw = real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap().clone();
    let mut o7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    if unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &mut o7 as *mut _ as _, size_of_val(&o7) as u32) }.is_err()
        || o7.MeshShaderTier.0 < D3D12_MESH_SHADER_TIER_1.0
    {
        println!("no mesh shaders – skipping");
        runner::skip_test("no mesh shaders");
        return None;
    }
    println!("backend device: {}", real::stats().adapter);
    Some((dev, serial))
}

fn empty_rs(dev: &ID3D12Device) -> ID3D12RootSignature {
    unsafe {
        let mut ser = None;
        D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).unwrap();
        let ser = ser.unwrap();
        dev.CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize())).unwrap()
    }
}

fn resolve_two_pass(pso: &ID3D12PipelineState, before: (u64, u64), kind: &str, expect_async: bool) -> (usize, ID3D12PipelineState, Arc<real::TwoPassPso>) {
    let raw = pso.as_raw() as usize;
    let real_raw = pso_async::resolve_wait(raw);
    if expect_async && pso_async::enabled() {
        assert_ne!(real_raw, raw, "cold cache must build {kind} PSO behind an async stand-in");
    }
    let after = real::stats();
    assert_eq!(after.translated_pipelines, before.0 + 1, "{kind} not translated: {:?}", after.translation_errors);
    assert_eq!(after.fallback_pipelines, before.1, "fallback shaders were used");
    let tp = real::two_pass_of(raw).expect("two-pass entry");
    assert_eq!(tp.kind, kind);
    let resolved = unsafe { ID3D12PipelineState::from_raw_borrowed(&(real_raw as *mut c_void)) }.unwrap().clone();
    (raw, resolved, tp)
}

fn assert_drop_frees_two_pass(raw: usize, tp: Arc<real::TwoPassPso>, resolved: ID3D12PipelineState, pso: ID3D12PipelineState) {
    let count = real::two_pass_count();
    drop((tp, resolved, pso));
    assert_eq!(real::two_pass_count(), count - 1, "dropping PSO must free its two-pass entry");
    assert_eq!(pso_async::resolve(raw), raw);
    assert!(real::two_pass_of(raw).is_none());
}

/// Execute both passes of a hook-created two-pass pipeline (`tp.compute` via `Dispatch`,
/// then `mesh_pso` via `DispatchMesh`) on `dev` and return `(pixels, records)`.
fn run_two_pass(dev: &ID3D12Device, mesh_pso: &ID3D12PipelineState, tp: &real::TwoPassPso, groups: u32, sgprs: &[u32], mem: &[u32]) -> (Vec<[f32; 4]>, Vec<u32>) {
    unsafe {
        let mk_res = |heap, desc: &D3D12_RESOURCE_DESC, state, cv: Option<&D3D12_CLEAR_VALUE>| -> ID3D12Resource {
            let mut out = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, desc, state, cv.map(|c| c as *const _), &mut out).unwrap();
            out.unwrap()
        };
        let mk_buf = |bytes: u64, heap, state, uav: bool| mk_res(heap, &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes.max(256), Height: 1, DepthOrArraySize: 1, MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE }, ..Default::default()
        }, state, None);
        let tr = |r: &ID3D12Resource, before, after| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: before, StateAfter: after }) },
        };
        let mk_heap = |ty, n, fl| dev.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: n, Flags: fl, NodeMask: 0 }).unwrap();
        let compute_pso = ID3D12PipelineState::from_raw_borrowed(&(tp.compute as *mut c_void)).unwrap().clone();
        let compute_rs = ID3D12RootSignature::from_raw_borrowed(&(tp.compute_rs as *mut c_void)).unwrap().clone();
        let mesh_rs = ID3D12RootSignature::from_raw_borrowed(&(tp.mesh_rs as *mut c_void)).unwrap().clone();

        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let tex = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: SIZE as u64, Height: SIZE, DepthOrArraySize: 1, MipLevels: 1,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ..Default::default()
        };
        let cv = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, Anonymous: D3D12_CLEAR_VALUE_0 { Color: CLEAR } };
        let rt = mk_res(D3D12_HEAP_TYPE_DEFAULT, &tex, D3D12_RESOURCE_STATE_RENDER_TARGET, Some(&cv));
        let rtv_heap = mk_heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
        let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
        dev.CreateRenderTargetView(&rt, None, rtv);
        let (srv_heap, smp_heap) = (mk_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE), mk_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE));

        let (mut io, mut m) = (sgprs.to_vec(), mem.to_vec());
        io.resize(128, 0);
        m.resize(m.len().max(64), 0);
        let (io_b, mem_b, res_len) = ((io.len() * 4) as u64, (m.len() * 4) as u64, tp.record.record_dwords() * groups as usize);
        let res_b = (res_len * 4) as u64;
        let upload = mk_buf(io_b + mem_b, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false);
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).unwrap();
        std::ptr::copy_nonoverlapping(io.as_ptr(), p as *mut u32, io.len());
        std::ptr::copy_nonoverlapping(m.as_ptr(), (p as *mut u32).add(io.len()), m.len());
        upload.Unmap(0, None);
        let io_buf = mk_buf(io_b, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, true);
        let mem_buf = mk_buf(mem_b, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, true);
        let res_buf = mk_buf(res_b, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true);
        let (mut fp, mut total) = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(), 0u64);
        dev.GetCopyableFootprints(&tex, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
        let readback = mk_buf(total + res_b, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);

        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &compute_pso).unwrap();
        list.CopyBufferRegion(&io_buf, 0, &upload, 0, io_b);
        list.CopyBufferRegion(&mem_buf, 0, &upload, io_b, mem_b);
        list.ResourceBarrier(&[
            tr(&io_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            tr(&mem_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            tr(&res_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        list.SetDescriptorHeaps(&[Some(srv_heap), Some(smp_heap)]);
        list.SetComputeRootSignature(&compute_rs);
        for (i, b) in [&io_buf, &res_buf, &mem_buf].into_iter().enumerate() { list.SetComputeRootUnorderedAccessView(i as u32, b.GetGPUVirtualAddress()); }
        list.Dispatch(groups, 1, 1);
        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: transmute_copy(&res_buf) }) },
        }]);
        list.SetPipelineState(mesh_pso);
        list.SetGraphicsRootSignature(&mesh_rs);
        for (i, b) in [&io_buf, &res_buf, &mem_buf].into_iter().enumerate() { list.SetGraphicsRootUnorderedAccessView(i as u32, b.GetGPUVirtualAddress()); }
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.ClearRenderTargetView(rtv, &CLEAR, None);
        list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: SIZE as f32, Height: SIZE as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: SIZE as i32, bottom: SIZE as i32 }]);
        list.cast::<ID3D12GraphicsCommandList6>().unwrap().DispatchMesh(groups * tp.mesh_groups_per_record, 1, 1);
        list.ResourceBarrier(&[
            tr(&rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE),
            tr(&res_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        ]);
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&rt), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.CopyBufferRegion(&readback, total, &res_buf, 0, res_b);
        list.Close().unwrap();

        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        queue.Signal(&fence, 1).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while fence.GetCompletedValue() < 1 {
            assert!(std::time::Instant::now() < deadline, "GPU fence timed out");
            std::thread::yield_now();
        }
        let mut q = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut q)).unwrap();
        let pitch = fp.Footprint.RowPitch as usize;
        let mut px = Vec::with_capacity((SIZE * SIZE) as usize);
        for y in 0..SIZE as usize {
            for x in 0..SIZE as usize {
                let c = (q as *const u8).add(y * pitch + x * 16) as *const f32;
                px.push([*c, *c.add(1), *c.add(2), *c.add(3)]);
            }
        }
        let recs = std::slice::from_raw_parts((q as *const u8).add(total as usize) as *const u32, res_len).to_vec();
        readback.Unmap(0, None);
        (px, recs)
    }
}

#[test]
fn console_ms_ps_stream_hook_executes_two_passes_and_cleans_up() {
    let Some((dev, _serial)) = setup() else { return };

    struct ColdCache(std::path::PathBuf, Option<std::ffi::OsString>);
    impl Drop for ColdCache {
        fn drop(&mut self) {
            match &self.1 { Some(p) => std::env::set_var("XCL_DXIL_CACHE", p), None => std::env::remove_var("XCL_DXIL_CACHE") }
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let cold = ColdCache(std::env::temp_dir().join(format!("xcl-mesh-e2e-{}-{nonce}", std::process::id())), std::env::var_os("XCL_DXIL_CACHE"));
    let _ = std::fs::remove_dir_all(&cold.0);
    std::env::set_var("XCL_DXIL_CACHE", &cold.0);

    let mut ms_code = asm(&[
        ("S_BFE_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x40018)),
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 193), ("SRC1", V)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 0), ("SRC1", 134), ("SRC2", V)], None),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("S_LSHR_B32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 135)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some(1 << 12 | 3)),
        ("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 9)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 131), ("VSRC1", 0)], None),
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 20), ("SRC0", 1)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 20), ("VSRC1", 10)], None),
    ]);
    for (vd, src, lit) in [(14, V + 20, None), (15, 240, None), (16, 255, Some(0.25f32.to_bits())), (17, 242, None)] {
        emit(&mut ms_code, "V_MOV_B32", "ENC_VOP1", &[("VDST", vd), ("SRC0", src)], lit);
    }
    ms_code.extend(asm(&[
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 10), ("VSRC1", 11), ("VSRC2", 12), ("VSRC3", 13), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 14), ("VSRC1", 15), ("VSRC2", 16), ("VSRC3", 17), ("DONE", 0)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)], None),
        ("V_CMPX_EQ_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 21), ("SRC0", 255)], Some(1 << 10 | 2 << 20)),
        ("EXP", "ENC_EXP", &[("TGT", 20), ("EN", 1), ("VSRC0", 21), ("VSRC1", 0), ("VSRC2", 0), ("VSRC3", 0), ("DONE", 1)], None),
        ("S_ENDPGM", "ENC_SOPP", &[], None),
    ]));
    let mut psv = vec![0u8; 48];
    psv[12..16].copy_from_slice(&[128, 0, 128, 0]);
    for (o, val) in [(16, 64u32), (20, 64), (36, 128), (40, 1), (44, 1)] { psv[o..o + 4].copy_from_slice(&val.to_le_bytes()); }
    (psv[24], psv[27]) = (13, 2); // MS, triangles
    let ms_osg = sig_part(&[("SV_Position", 1, 3, 0, 0xF), ("COLOR", 0, 3, 1, 0xF)]);
    let mut ms_blob = dxbc(&[(b"OSG1", &ms_osg), (b"PSV0", &psv_part(&psv)), (b"DXIL", &console_dxil(13, &ms_code, 7, 6, None))]);
    let mut ps_blob = console_ps(false);
    assert!(xcl::gpu::xbox::is_xbox(&ms_blob) && xcl::gpu::xbox::is_xbox(&ps_blob));

    #[repr(C, align(8))]
    struct Sub<T> { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, v: T }
    #[repr(C)]
    struct Stream {
        rs: Sub<*mut c_void>, ms: Sub<D3D12_SHADER_BYTECODE>, ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>, mask: Sub<u32>, rast: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC>, rt: Sub<D3D12_RT_FORMAT_ARRAY>, samples: Sub<DXGI_SAMPLE_DESC>,
    }
    let game_rs = empty_rs(&dev);
    let mut blend = D3D12_BLEND_DESC::default();
    blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    let mut rt = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
    rt.RTFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    let mut st = Stream {
        rs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, v: game_rs.as_raw() },
        ms: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, v: bc(&ms_blob) },
        ps: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, v: bc(&ps_blob) },
        blend: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, v: blend },
        mask: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, v: u32::MAX },
        rast: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, v: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() } },
        depth: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, v: D3D12_DEPTH_STENCIL_DESC::default() },
        rt: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, v: rt },
        samples: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, v: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 } },
    };
    let d2: ID3D12Device2 = dev.cast().unwrap();
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: size_of::<Stream>(), pPipelineStateSubobjectStream: &mut st as *mut _ as *mut _ };
    let s = real::stats();
    let pso: ID3D12PipelineState = unsafe { d2.CreatePipelineState(&desc) }.expect("CreatePipelineState (MS+PS)");
    // Overwrite stack stream and shader buffers immediately after return (`own_stream` deep copy).
    unsafe { std::ptr::write_bytes(&mut st as *mut Stream, 0xCD, 1) };
    ms_blob.fill(0xCD);
    ps_blob.fill(0xCD);
    let (raw, resolved, tp) = resolve_two_pass(&pso, (s.translated_pipelines, s.fallback_pipelines), "MS", true);
    assert_eq!((tp.record.threads, tp.record.max_vertices, tp.record.max_primitives), (128, 128, 128));

    let mut mem = vec![0u32; 256];
    for (i, (x, y)) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (0.0, -1.0)].iter().enumerate() {
        mem[0x40 + 4 * i..0x40 + 4 * i + 4].copy_from_slice(&[x.to_bits(), y.to_bits(), 0.0f32.to_bits(), 1.0f32.to_bits()]);
    }
    let mut sg = vec![0u32; 128];
    sg[16..20].copy_from_slice(&[0x100, 16 << 16, 3, 0]);
    let (px, recs) = run_two_pass(&dev, &resolved, &tp, 2, &sg, &mem);
    let (rd, vd) = (tp.record.record_dwords(), tp.record.vert_dwords());
    for g in 0..2usize {
        let r = &recs[g * rd..(g + 1) * rd];
        assert_eq!((r[0], r[1], r[MESH_HEADER + 128 * vd]), (3, 1, 1 << 10 | 2 << 20), "group {g} record");
    }
    let at = |x: usize, y: usize| px[y * SIZE as usize + x];
    assert_eq!((at(3, 29), at(19, 29), at(29, 3)), ([0.0, 0.5, 0.25, 1.0], [1.0, 0.5, 0.25, 1.0], CLEAR));
    let left = px.iter().filter(|p| **p == [0.0, 0.5, 0.25, 1.0]).count();
    let right = px.iter().filter(|p| **p == [1.0, 0.5, 0.25, 1.0]).count();
    let outside = px.iter().filter(|p| **p == CLEAR).count();
    assert!(left > 100 && right > 100 && outside > 400, "unexpected coverage: {left}/{right}/{outside}");
    assert_drop_frees_two_pass(raw, tp, resolved, pso);
}

#[test]
fn console_es_gs_graphics_hook_executes_two_passes_and_cleans_up() {
    let Some((dev, _serial)) = setup() else { return };

    let es_code = asm(&[
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 21), ("SRC0", 133), ("VSRC1", 20)], None),
        ("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 21), ("DATA0", 9), ("DATA1", 10), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        ("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 21), ("DATA0", 11), ("DATA1", 12), ("OFFSET0", 2), ("OFFSET1", 3)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None),
        ("S_ENDPGM", "ENC_SOPP", &[], None),
    ]);
    let mut gs_code = asm(&[
        ("S_BFE_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x80008)),
        ("S_BFM_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 0), ("SSRC1", 128)], None),
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 255), ("VSRC1", 30)], Some(0x3FFFC)),
        ("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 31), ("SRC0", 144), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 31), ("SRC0", 130), ("VSRC1", 31)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 32), ("SRC0", 130), ("VSRC1", 1)], None),
    ]);
    for (vd, addr) in [(40, 30), (42, 31), (44, 32)] {
        emit(&mut gs_code, "DS_READ2_B32", "ENC_DS", &[("VDST", vd), ("ADDR", addr), ("OFFSET0", 0), ("OFFSET1", 1)], None);
    }
    for (d, a, b) in [(46u64, 42u64, 44u64), (47, 43, 45)] {
        emit(&mut gs_code, "V_ADD_F32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + a), ("VSRC1", b)], None);
        emit(&mut gs_code, "V_SUB_F32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + d), ("VSRC1", d - 6)], None);
    }
    emit(&mut gs_code, "V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 50), ("SRC0", V + 20), ("SRC1", 255), ("SRC2", 2)], Some(128));
    for (vd, src, lit) in [(51, 240, None), (52, 242, None), (53, 255, Some(0.5f32.to_bits())), (54, 255, Some(0.75f32.to_bits()))] {
        emit(&mut gs_code, "V_MOV_B32", "ENC_VOP1", &[("VDST", vd), ("SRC0", src)], lit);
    }
    for (k, (x, y)) in [(40u64, 41u64), (42, 43), (44, 45), (46, 47)].into_iter().enumerate() {
        let o = 8 * k as u64;
        for (d0, d1, off) in [(x, y, o), (51, 52, o + 2), (53, 54, o + 4), (51, 52, o + 6)] {
            emit(&mut gs_code, "DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", d0), ("DATA1", d1), ("OFFSET0", off), ("OFFSET1", off + 1)], None);
        }
        emit(&mut gs_code, "S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x22)], None);
    }
    for &(n, f) in &[("S_SENDMSG", &[("SIMM16", 0x12u64)][..]), ("S_SENDMSG", &[("SIMM16", 0x3)][..]), ("S_ENDPGM", &[])] {
        emit(&mut gs_code, n, "ENC_SOPP", f, None);
    }

    let mut psv = vec![0u8; 36];
    for (o, val) in [(0, 3u32), (4, 5), (16, 64), (20, 64)] { psv[o..o + 4].copy_from_slice(&val.to_le_bytes()); }
    (psv[24], psv[26]) = (2, 4); // GS, max_vertex_count = 4
    let es_isg = sig_part(&[("POSITION", 0, 3, 0, 0xF)]);
    let es_osg = sig_part(&[("SV_Position", 1, 3, 0, 0xF), ("TEXCOORD", 0, 3, 1, 0xF)]);
    let es_blob = dxbc(&[(b"ISG1", &es_isg), (b"OSG1", &es_osg), (b"DXIL", &console_dxil(1, &es_code, 2, 0, None))]);
    let gs_osg = sig_part(&[("SV_Position", 1, 3, 0, 0xF), ("COLOR", 0, 3, 1, 0xF)]);
    let gs_blob = dxbc(&[(b"OSG1", &gs_osg), (b"PSV0", &psv_part(&psv)), (b"DXIL", &console_dxil(2, &gs_code, 3, 2, None))]);
    let ps_blob = console_ps(true);
    assert!(xcl::gpu::xbox::is_xbox(&es_blob) && xcl::gpu::xbox::is_xbox(&gs_blob) && xcl::gpu::xbox::is_xbox(&ps_blob));

    let game_rs = empty_rs(&dev);
    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(game_rs)), VS: bc(&es_blob), GS: bc(&gs_blob), PS: bc(&ps_blob),
        SampleMask: u32::MAX, PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default()
    };
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    desc.RasterizerState = D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() };
    desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;

    let s = real::stats();
    let pso: Result<ID3D12PipelineState, _> = unsafe { dev.CreateGraphicsPipelineState(&desc) };
    unsafe { ManuallyDrop::drop(&mut desc.pRootSignature) };
    let pso = pso.expect("CreateGraphicsPipelineState (ES+GS+PS)");
    let (raw, resolved, tp) = resolve_two_pass(&pso, (s.translated_pipelines, s.fallback_pipelines), "GS", false);
    assert_eq!((tp.record.max_vertices, tp.record.max_primitives), (84, 42));

    // One input triangle (-0.5,-0.5), (0.5,-0.5), (-0.5,0.5) expanded by the GS into [-0.5, 0.5]^2.
    let mut mem = vec![0u32; 0x1000 / 4 + 12];
    for (i, v) in [[-0.5f32, -0.5, 0.5, 1.0], [0.5, -0.5, 0.5, 1.0], [-0.5, 0.5, 0.5, 1.0]].iter().enumerate() {
        mem[0x1000 / 4 + i * 4..0x1000 / 4 + i * 4 + 4].copy_from_slice(&v.map(f32::to_bits));
    }
    let mut io = vec![0u32; 128];
    io[IO_GS_PRIMS] = 1;
    io[IO_GS_VERTICES] = 0x1000;
    let (px, recs) = run_two_pass(&dev, &resolved, &tp, 1, &io, &mem);
    let tris = MESH_HEADER + tp.record.max_vertices as usize * tp.record.vert_dwords();
    assert_eq!((recs[0], recs[1], recs[tris], recs[tris + 1]), (4, 2, 1 << 10 | 2 << 20, 2 | 1 << 10 | 3 << 20), "GS record");

    let want = [0.25f32, 0.75, 0.5, 1.0]; // red halved from 0.5 by console_ps(true)
    let at = |x: usize, y: usize| px[y * SIZE as usize + x];
    assert_eq!((at(10, 22), at(22, 10), at(2, 2)), (want, want, CLEAR));
    let inside = px.iter().filter(|p| **p == want).count();
    let outside = px.iter().filter(|p| **p == CLEAR).count();
    assert!(inside >= 200 && outside >= 700, "unexpected square coverage: {inside}/{outside}");
    assert_drop_frees_two_pass(raw, tp, resolved, pso);
}
