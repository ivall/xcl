//! End-to-end GPU tests for direct DXIL compute dispatch splitting (#1269):
//! 1. Real RDNA2 compute programs (wave32 `[32, 1, 1]` and split wave64 `[64, 1, 1]` with
//!    `v_readlane_b32` inter-half exchanges, scalar loop SCCs, uniform forward skips across
//!    intermediate pieces, lane-divergent selection, and persistent cross-piece values) are
//!    translated via `kernel::translate_compute` into `DispatchChain`s, validated, and run
//!    across 3D grids and slab limits against unsplit DXIL and a CPU reference.
//! 2. Retail console containers (`DXBC -> DXIL/XDAT -> XBCH -> FSSM`) go through
//!    `real::create_device` -> `CreateComputePipelineState` -> `dispatch_chain::build` ->
//!    console command list `Dispatch` (back-to-back chained dispatches reusing `SHARED`
//!    scratch) -> `ID3D12CommandAllocator::Reset` -> second recording, verifying UMA output.

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock};

use windows::core::{Interface, GUID};
use windows::Win32::Foundation::{CloseHandle, S_OK, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows_sys::Win32::System::Diagnostics::Debug::RtlAddFunctionTable;
use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE};

use xcl::gpu::dxil::kernel::{self, DispatchChain, KernelOptions};
use xcl::gpu::dxil::validate;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::{runner, CsInterface};
use xcl::hle::d3d12x::{bind, pso_async, real};
use xcl::modules::{self, MappedModule};
use xcl::pe::OwnedExports;

const V: u64 = 256; // VALU source encoding for v0
const GRID: [u32; 3] = [2, 3, 2]; // 12 groups across all 3 dispatch axes
static SERIAL: Mutex<()> = Mutex::new(());

fn init_env() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        for (k, v) in [
            ("XCL_DIRECT_DXIL", "1"), ("XCL_DIRECT_SPLIT_INSTRUCTIONS", "80"),
            ("XCL_SPLIT_SCRATCH_MB", "1"), ("XCL_UMA_MB", "64"),
            ("XCL_TRANSLATED_CACHE", "0"), ("XCL_PSO_DISK_CACHE", "0"),
        ] { std::env::set_var(k, v); }
    });
}

fn emit(code: &mut Vec<u8>, name: &str, enc: &str, fields: &[(&str, u64)], lit: Option<u32>) {
    code.extend(encode(name, enc, fields, lit).unwrap_or_else(|e| panic!("{name}: {e}")));
}

fn patch_branch(code: &mut [u8], at: usize, target: usize) {
    let rel = ((target as isize - at as isize - 4) / 4) as i16 as u16;
    let w = u32::from_le_bytes(code[at..at + 4].try_into().unwrap());
    code[at..at + 4].copy_from_slice(&((w & 0xFFFF_0000) | rel as u32).to_le_bytes());
}

/// Real RDNA2 compute program for `[32, 1, 1]` (wave32) or `[64, 1, 1]` (wave64), no LDS.
/// Inputs: `s4` = buffer base lo, `s5` = base hi, `s6` = salt, `s7..s9` = TGID x/y/z, `v0` = tid.
fn build_split_program(wave64: bool) -> Vec<u8> {
    let shift = if wave64 { 6u64 } else { 5u64 };
    let mut c = Vec::new();
    let prologue: &[(&str, &str, &[(&str, u64)], Option<u32>)] = &[
        ("S_MUL_I32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 9), ("SSRC1", 131)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 8)], None),
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 7)], None),
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 10), ("SSRC1", 128 + shift)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 11), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 130), ("VSRC1", 1)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 4), ("VSRC1", 2)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 5)], None),
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 5), ("ADDR", 3), ("SADDR", 0x7D), ("OFFSET", 0)], None),
        // Persistent value v6 = orig + (salt ^ 0xA5A5_5A5A) kept live across all pieces; acc v7 = orig ^ salt
        ("S_XOR_B32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 6), ("SSRC1", 255)], Some(0xA5A5_5A5A)),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 12), ("VSRC1", 5)], None),
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 6), ("VSRC1", 5)], None),
    ];
    for &(n, e, f, l) in prologue { emit(&mut c, n, e, f, l); }
    if wave64 {
        emit(&mut c, "V_READLANE_B32", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 5), ("SRC1", 128)], None);
        emit(&mut c, "V_READLANE_B32", "ENC_VOP3", &[("VDST", 15), ("SRC0", V + 5), ("SRC1", 128 + 47)], None);
        emit(&mut c, "S_XOR_B32", "ENC_SOP2", &[("SDST", 14), ("SSRC0", 14), ("SSRC1", 15)], None);
        emit(&mut c, "V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 14), ("VSRC1", 7)], None);
    }
    // Scalar loop SCC: 4 trips of acc = acc * 3 + (idx ^ r) + 0x1111_0001
    emit(&mut c, "S_MOV_B32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 128)], None);
    let loop_head = c.len();
    emit(&mut c, "V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 7), ("SRC0", V + 7), ("SRC1", 131)], None);
    emit(&mut c, "V_XOR_B32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 12), ("VSRC1", 1)], None);
    emit(&mut c, "V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", V + 7), ("VSRC1", 8)], None);
    emit(&mut c, "V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 255), ("VSRC1", 7)], Some(0x1111_0001));
    emit(&mut c, "S_ADD_U32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 12), ("SSRC1", 129)], None);
    emit(&mut c, "S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 12), ("SSRC1", 132)], None);
    let loop_br = c.len();
    emit(&mut c, "S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)], None);
    patch_branch(&mut c, loop_br, loop_head);
    // Uniform forward skip: even groups skip the multi-block middle section across pieces.
    emit(&mut c, "S_AND_B32", "ENC_SOP2", &[("SDST", 13), ("SSRC0", 10), ("SSRC1", 129)], None);
    let skip_br = c.len();
    emit(&mut c, "S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 0)], None);
    for round in 1u32..=18 {
        emit(&mut c, "V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", V + 5), ("VSRC1", 7)], None);
        emit(&mut c, "V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 255), ("VSRC1", 7)], Some(round * 0x0101_0001));
        emit(&mut c, "V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 131), ("VSRC1", 7)], None);
        emit(&mut c, "V_XOR_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", V + 8), ("VSRC1", 7)], None);
        if round % 6 == 0 {
            emit(&mut c, "S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 13), ("SSRC1", 128)], None);
            let br = c.len();
            emit(&mut c, "S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)], None);
            let next = c.len();
            patch_branch(&mut c, br, next);
        }
    }
    let join = c.len();
    patch_branch(&mut c, skip_br, join);
    // Lane-divergent selection: odd lanes XOR 0x55AA_33CC.
    emit(&mut c, "V_AND_B32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 129), ("VSRC1", 0)], None);
    emit(&mut c, "V_XOR_B32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 255), ("VSRC1", 7)], Some(0x55AA_33CC));
    emit(&mut c, "V_CMP_NE_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 8)], None);
    emit(&mut c, "V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", V + 7), ("VSRC1", 9)], None);
    if wave64 {
        emit(&mut c, "V_READLANE_B32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 7), ("SRC1", 128 + 35)], None);
        emit(&mut c, "V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 16), ("VSRC1", 7)], None);
    }
    emit(&mut c, "V_XOR_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", V + 6), ("VSRC1", 7)], None);
    emit(&mut c, "GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 3), ("DATA", 7), ("SADDR", 0x7D), ("OFFSET", 0)], None);
    emit(&mut c, "S_ENDPGM", "ENC_SOPP", &[], None);
    c
}

fn eval_split_program(wave64: bool, initial: &[u32], salt: u32) -> Vec<u32> {
    let threads = if wave64 { 64usize } else { 32usize };
    let mut out = initial.to_vec();
    for g in 0..(GRID[0] * GRID[1] * GRID[2]) as usize {
        let base = g * threads;
        let peer = if wave64 { initial[base] ^ initial[base + 47] } else { 0 };
        let (mut accs, mut saveds) = (vec![0u32; threads], vec![0u32; threads]);
        for lane in 0..threads {
            let (idx, orig) = ((base + lane) as u32, initial[base + lane]);
            saveds[lane] = orig.wrapping_add(salt ^ 0xA5A5_5A5A);
            let mut acc = (orig ^ salt).wrapping_add(peer);
            for r in 0u32..4 { acc = acc.wrapping_mul(3).wrapping_add(idx ^ r).wrapping_add(0x1111_0001); }
            if g & 1 != 0 {
                for round in 1u32..=18 {
                    acc = acc.wrapping_add(orig).wrapping_add(round * 0x0101_0001);
                    acc ^= acc >> 3;
                }
            }
            accs[lane] = if lane & 1 != 0 { acc ^ 0x55AA_33CC } else { acc };
        }
        let lane35 = if wave64 { accs[35] } else { 0 };
        for lane in 0..threads { out[base + lane] = accs[lane].wrapping_add(lane35) ^ saveds[lane]; }
    }
    out
}

fn dispatch_slabs(grid: [u32; 3], most: u64) -> Vec<([u32; 3], [u32; 3])> {
    let [x, y, z] = grid;
    let fit = |n: u32, of: u64| (n as u64).min((most / of.max(1)).max(1)) as u32;
    let sx = fit(x, 1);
    let sy = if sx == x { fit(y, x as u64) } else { 1 };
    let sz = if sx == x && sy == y { fit(z, x as u64 * y as u64) } else { 1 };
    let at = |n: u32, step: u32| (0..n).step_by(step.max(1) as usize).map(move |k| (k, step.min(n - k)));
    let mut out = Vec::new();
    for (oz, nz) in at(z, sz) { for (oy, ny) in at(y, sy) { for (ox, nx) in at(x, sx) { out.push(([ox, oy, oz], [nx, ny, nz])); } } }
    out
}

fn make_buffer(dev: &ID3D12Device, size: u64, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES, uav: bool) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: size.max(256), Height: 1, DepthOrArraySize: 1, MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE }, ..Default::default()
    };
    let mut r = None;
    unsafe { dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).unwrap(); }
    r.unwrap()
}

fn barrier_transition(r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(r) }, Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: before, StateAfter: after,
            }),
        },
    }
}

fn wait_fence(queue: &ID3D12CommandQueue, fence: &ID3D12Fence, val: &mut u64) {
    *val += 1;
    unsafe {
        queue.Signal(fence, *val).unwrap();
        let ev = CreateEventW(None, false, false, None).unwrap();
        fence.SetEventOnCompletion(*val, ev).unwrap();
        assert_eq!(WaitForSingleObject(ev, 10_000), WAIT_OBJECT_0, "GPU fence timed out");
        let _ = CloseHandle(ev);
    }
}

/// Runs a `DispatchChain` on `dev` with slab limit `most`, matching `pso::record_dispatch_chain`.
fn run_dispatch_chain(dev: &ID3D12Device, chain: &DispatchChain, grid: [u32; 3], sgprs: &[u32], mem: &[u32], most: u64) -> Vec<u32> {
    unsafe {
        let uav = |reg, space| D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            Anonymous: D3D12_ROOT_PARAMETER1_0 { Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: reg, RegisterSpace: space, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE } },
        };
        let mut params: Vec<_> = (0..6).map(|i| uav(i, 0)).collect();
        params.push(uav(0, 2));
        params.push(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            Anonymous: D3D12_ROOT_PARAMETER1_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 2, Num32BitValues: 8 } },
        });
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32, pParameters: params.as_ptr(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED, ..Default::default()
                },
            },
        };
        let mut blob: Option<ID3DBlob> = None;
        D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).unwrap();
        let blob = blob.unwrap();
        let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).unwrap();
        let psos: Vec<ID3D12PipelineState> = chain.pieces.iter().map(|dxil| {
            dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&rs), CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() }, ..Default::default()
            }).unwrap()
        }).collect();
        let mut io = sgprs.to_vec();
        io.resize(128, 0);
        let (io_bytes, mem_bytes) = ((io.len() * 4) as u64, (mem.len() * 4) as u64);
        let upload = make_buffer(dev, io_bytes + mem_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false);
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).unwrap();
        std::ptr::copy_nonoverlapping(io.as_ptr(), p as *mut u32, io.len());
        std::ptr::copy_nonoverlapping(mem.as_ptr(), (p as *mut u32).add(io.len()), mem.len());
        upload.Unmap(0, None);
        let io_buf = make_buffer(dev, io_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true);
        let res_buf = make_buffer(dev, 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true);
        let mem_buf = make_buffer(dev, mem_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, true);
        let readback = make_buffer(dev, mem_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false);
        let slabs = dispatch_slabs(grid, most);
        let scratch_bytes = slabs[0].1.iter().fold(4 * u64::from(chain.dwords_per_group), |n, &g| n * u64::from(g));
        let scratch = make_buffer(dev, scratch_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);

        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &psos[0]).unwrap();
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        list.CopyBufferRegion(&io_buf, 0, &upload, 0, io_bytes);
        list.CopyBufferRegion(&mem_buf, 0, &upload, io_bytes, mem_bytes);
        list.ResourceBarrier(&[
            barrier_transition(&io_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            barrier_transition(&mem_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            barrier_transition(&res_buf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        list.SetComputeRootSignature(&rs);
        for (k, b) in [(0, &io_buf), (1, &res_buf), (2, &mem_buf), (6, &scratch)] { list.SetComputeRootUnorderedAccessView(k, b.GetGPUVirtualAddress()); }
        let uav_b = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }) },
        };
        for (slab_idx, ([ox, oy, oz], [nx, ny, nz])) in slabs.into_iter().enumerate() {
            for (i, pso) in psos.iter().enumerate() {
                if i != 0 || slab_idx != 0 { list.ResourceBarrier(&[uav_b.clone()]); }
                let c = [nx, ny, nz, i as u32, ox, oy, oz, 0];
                list.SetComputeRoot32BitConstants(7, 8, c.as_ptr().cast(), 0);
                list.SetPipelineState(pso);
                list.Dispatch(nx, ny, nz);
            }
        }
        list.ResourceBarrier(&[barrier_transition(&mem_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyBufferRegion(&readback, 0, &mem_buf, 0, mem_bytes);
        list.Close().unwrap();
        let cl: ID3D12CommandList = list.cast().unwrap();
        queue.ExecuteCommandLists(&[Some(cl)]);
        wait_fence(&queue, &fence, &mut 0);
        let mut q = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut q)).unwrap();
        let out = std::slice::from_raw_parts(q as *const u32, mem.len()).to_vec();
        readback.Unmap(0, None);
        out
    }
}

/// Retail console CS container (`DXBC -> DXIL/XDAT -> XBCH -> FSSM`), header `flags` (+0xD4; bit 5 = wave32).
fn console_cs_flags(code: &[u8], rsrc2: u32, threads: [u32; 3], flags: u32) -> Vec<u8> {
    let le = |v: &mut Vec<u8>, x: u32| v.extend_from_slice(&x.to_le_bytes());
    let mut fssm = vec![0u8; 0x308];
    fssm[0..4].copy_from_slice(b"FSSM");
    fssm[4..8].copy_from_slice(&0x101u32.to_le_bytes());
    fssm[0xC0..0xC8].copy_from_slice(&0x2F8u64.to_le_bytes());
    fssm[0xC8..0xD0].copy_from_slice(&(code.len() as u64).to_le_bytes());
    fssm[0xD4..0xD8].copy_from_slice(&flags.to_le_bytes());
    fssm[0x100..0x108].copy_from_slice(&0x0000_00C0_0002_0106u64.to_le_bytes());
    for (k, x) in [rsrc2, threads[0], threads[1], threads[2]].into_iter().enumerate() {
        fssm[0x138 + 4 * k..0x13C + 4 * k].copy_from_slice(&x.to_le_bytes());
    }
    fssm.extend_from_slice(code);
    let fssm_sz = (fssm.len() - 0x10) as u32;
    fssm[8..12].copy_from_slice(&fssm_sz.to_le_bytes());
    let z = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 6);
    let mut xbch = b"XBCH".to_vec();
    for x in [3, z.len() as u32, 0, fssm.len() as u32, 0] { le(&mut xbch, x); }
    xbch.extend_from_slice(&z);
    let mut d = Vec::new();
    for x in [5 << 16 | 0x66, 0, u32::from_le_bytes(*b"DXIL"), 0x106, 0, 4] { le(&mut d, x); }
    d.extend_from_slice(b"XDAT");
    for x in [2, 0x24, 0x12, 0x3C, 16, 0x0d, 0x4C, xbch.len() as u32] { le(&mut d, x); }
    d.extend_from_slice(&[0x5a; 16]);
    d.extend_from_slice(&xbch);
    while d.len() % 4 != 0 { d.push(0); }
    let nobc = d.len();
    d.extend_from_slice(b"NOBC");
    let dwords = (d.len() / 4) as u32;
    d[4..8].copy_from_slice(&dwords.to_le_bytes());
    d[16..20].copy_from_slice(&((nobc - 8) as u32).to_le_bytes());
    let mut v = vec![0u8; 36];
    v[0..4].copy_from_slice(b"DXBC");
    for (off, val) in [(20, 1u32), (28, 1), (32, 36)] { v[off..off + 4].copy_from_slice(&val.to_le_bytes()); }
    v.extend_from_slice(b"DXIL");
    le(&mut v, d.len() as u32);
    v.extend_from_slice(&d);
    let total = v.len() as u32;
    v[24..28].copy_from_slice(&total.to_le_bytes());
    v
}

type TrampolineFn = unsafe extern "system" fn(usize, usize, usize, usize, usize, usize, usize, usize, usize) -> usize;

fn console_caller() -> TrampolineFn {
    static TRAMPOLINE: OnceLock<TrampolineFn> = OnceLock::new();
    *TRAMPOLINE.get_or_init(|| unsafe {
        let base = VirtualAlloc(std::ptr::null(), 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
        assert!(!base.is_null(), "VirtualAlloc failed");
        let code: &[u8] = &[
            0x48, 0x83, 0xEC, 0x48, 0x48, 0x89, 0xC8, 0x48, 0x89, 0xD1, 0x4C, 0x89, 0xC2, 0x4D, 0x89, 0xC8,
            0x4C, 0x8B, 0x4C, 0x24, 0x70, 0x4C, 0x8B, 0x54, 0x24, 0x78, 0x4C, 0x89, 0x54, 0x24, 0x20,
            0x4C, 0x8B, 0x94, 0x24, 0x80, 0x00, 0x00, 0x00, 0x4C, 0x89, 0x54, 0x24, 0x28,
            0x4C, 0x8B, 0x94, 0x24, 0x88, 0x00, 0x00, 0x00, 0x4C, 0x89, 0x54, 0x24, 0x30,
            0x4C, 0x8B, 0x94, 0x24, 0x90, 0x00, 0x00, 0x00, 0x4C, 0x89, 0x54, 0x24, 0x38,
            0xFF, 0xD0, 0x48, 0x83, 0xC4, 0x48, 0xC3,
        ];
        std::ptr::copy_nonoverlapping(code.as_ptr(), base, code.len());
        let unwind: [u8; 8] = [0x01, 0x04, 0x01, 0x00, 0x04, 0x82, 0x00, 0x00];
        std::ptr::copy_nonoverlapping(unwind.as_ptr(), base.add(0x100), unwind.len());
        let rf: [u32; 3] = [0, code.len() as u32, 0x100];
        std::ptr::copy_nonoverlapping(rf.as_ptr(), base.add(0x110) as *mut u32, rf.len());
        assert!(RtlAddFunctionTable(base.add(0x110) as *const _, 1, base as u64));
        modules::register_mapped(Arc::new(MappedModule {
            name: "test_gpu_dispatch_split_caller.exe".into(), path: PathBuf::from("test_gpu_dispatch_split_caller.exe"),
            base: base as usize, size: 0x1000, entry_rva: 0, is_dll: false, exports: OwnedExports::default(),
            tls_callbacks: Mutex::new(Vec::new()), no_thread_calls: AtomicBool::new(false), initialized: AtomicBool::new(true), stack_reserve: 0,
        }));
        std::mem::transmute(base)
    })
}

unsafe fn call_console(obj: *mut c_void, slot: usize, args: &[usize]) -> usize {
    let mut a = [0usize; 7];
    a[..args.len()].copy_from_slice(args);
    (console_caller())(*(*(obj as *const *const usize)).add(slot), obj as usize, a[0], a[1], a[2], a[3], a[4], a[5], a[6])
}

#[test]
fn direct_compute_dispatch_split_matches_unsplit_across_wave32_and_wave64() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    init_env();
    if runner::wave32_supported() != Ok(true) { runner::skip_test("no wave32 GPU"); return; }
    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real { runner::skip_test("no GPU"); return; }
    let raw_dev = real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw_dev) }.expect("backend device").clone();

    for wave64 in [false, true] {
        let threads = if wave64 { 64u32 } else { 32u32 };
        let cs = CsInterface { threads: [threads, 1, 1], user_sgprs: 7, tgid: [true, true, true], tg_size: false, tid_components: 0 };
        let k = kernel::translate_compute(&build_split_program(wave64), KernelOptions { wave64, split: wave64, mem_windows: 1, sampler_base: 0 }, &cs).unwrap();
        validate::validate(&k.dxil).expect("whole DXIL validates");
        let chain = k.dispatch_chain.as_ref().expect("expected DispatchChain for split compute");
        assert!(chain.pieces.len() >= 3, "wave64={wave64}: expected >= 3 pieces, got {}", chain.pieces.len());
        for (i, piece) in chain.pieces.iter().enumerate() { validate::validate(piece).unwrap_or_else(|e| panic!("wave64={wave64} piece {i}: {e}")); }

        let words = (threads * GRID[0] * GRID[1] * GRID[2]) as usize;
        let salt = if wave64 { 0xCAFE_BABEu32 } else { 0x1357_9BDFu32 };
        let initial: Vec<u32> = (0..words + 16).map(|i| if i < words { (i as u32).wrapping_mul(0x9E37_79B1) ^ 0x1234_5678 } else { 0xDEAD_BEEF }).collect();
        let expected = eval_split_program(wave64, &initial, salt);
        let sgprs = [0, 0, 0, 0, 0, 0, salt];

        assert_eq!(runner::dispatch(&k.dxil, GRID, &sgprs, &initial).unwrap(), expected, "wave64={wave64}: unsplit");
        for most in [1u64, 2, 3, 7, 12] {
            assert_eq!(run_dispatch_chain(&dev, chain, GRID, &sgprs, &initial, most), expected, "wave64={wave64}, most={most}");
        }
    }
}

#[test]
fn console_split_compute_pso_dispatches_through_d3d12x_hook_and_reuses_scratch() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    init_env();
    if runner::wave32_supported() != Ok(true) { runner::skip_test("no wave32 GPU"); return; }
    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real { runner::skip_test("no GPU"); return; }
    let Some(uma) = xcl::gpu::uma::get() else { runner::skip_test("no UMA"); return; };
    if real::uma_gpu_va() == 0 { runner::skip_test("no GPU UMA"); return; }
    let mut raw_dev: *mut c_void = std::ptr::null_mut();
    assert_eq!(unsafe { real::create_device(&ID3D12Device::IID as *const GUID as *const c_void, &mut raw_dev) }, S_OK.0);
    let dev = unsafe { ID3D12Device::from_raw(raw_dev) };

    let param = D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 3 } },
    };
    let rs_desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: 1, pParameters: &param, ..Default::default() };
    let game_rs: ID3D12RootSignature = unsafe {
        let mut ser = None;
        D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).unwrap();
        let ser = ser.unwrap();
        dev.CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize())).unwrap()
    };
    let queue: ID3D12CommandQueue = unsafe { dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap() };
    let alloc: ID3D12CommandAllocator = unsafe { dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap() };
    let fence: ID3D12Fence = unsafe { dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap() };
    let mut fval = 0u64;

    let mut list_raw: *mut c_void = std::ptr::null_mut();
    let hr = unsafe {
        call_console(dev.as_raw(), 12, &[0, D3D12_COMMAND_LIST_TYPE_DIRECT.0 as usize, alloc.as_raw() as usize, 0,
            &ID3D12GraphicsCommandList::IID as *const GUID as usize, &mut list_raw as *mut *mut c_void as usize])
    } as i32;
    assert_eq!(hr, S_OK.0, "console CreateCommandList failed: {hr:#x}");

    let uma_buf = uma.alloc(0x2000, 0x1000).expect("UMA buffer");
    let (base_lo, base_hi) = (uma_buf as u64 as u32, ((uma_buf as u64) >> 32) as u32);
    let plain_abi_rs = real::abi_rs_cached().map(|p| p as usize).unwrap_or(0);
    let cases: &[(bool, u32)] = if runner::split_wave64() { &[(false, 0x20), (true, 0)] } else { &[(false, 0x20)] };

    for (case_idx, &(wave64, flags)) in cases.iter().enumerate() {
        let threads = if wave64 { 64u32 } else { 32u32 };
        let blob = console_cs_flags(&build_split_program(wave64), (7 << 1) | (1 << 7) | (1 << 8) | (1 << 9), [threads, 1, 1], flags);
        let mut pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(game_rs.clone())),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: blob.as_ptr().cast(), BytecodeLength: blob.len() }, ..Default::default()
        };
        let pso: ID3D12PipelineState = unsafe {
            let r = dev.CreateComputePipelineState(&pso_desc);
            std::mem::ManuallyDrop::drop(&mut pso_desc.pRootSignature);
            r.expect("CreateComputePipelineState for split console CS")
        };
        let real_pso = pso_async::resolve_wait(pso.as_raw() as usize);
        assert!(real::is_translated_compute(real_pso), "wave64={wave64}: split PSO must be registered");
        assert_ne!(real::pso_root_signature(real_pso).unwrap(), plain_abi_rs, "wave64={wave64}: expected dedicated dispatch_chain RS");

        let words = (threads * GRID[0] * GRID[1] * GRID[2]) as usize;
        let initial: Vec<u32> = (0..words + 16).map(|i| if i < words { (i as u32).wrapping_mul(0x45D9_F3B) ^ (case_idx as u32 + 1) } else { 0xCAFE_F00D }).collect();
        unsafe { std::ptr::copy_nonoverlapping(initial.as_ptr(), uma_buf as *mut u32, initial.len()); }
        if case_idx > 0 { assert_eq!(unsafe { call_console(list_raw, 10, &[alloc.as_raw() as usize, 0]) } as i32, S_OK.0); }

        let dispatches_before = bind::activity().1;
        let (salt1, salt2, salt3) = (0x1122_3344u32, 0x5566_7788u32, 0x99AA_BBCCu32);
        let (c1, c2, c3) = ([base_lo, base_hi, salt1], [base_lo, base_hi, salt2], [base_lo, base_hi, salt3]);
        unsafe {
            call_console(list_raw, 33, &[game_rs.as_raw() as usize]);
            call_console(list_raw, 25, &[pso.as_raw() as usize]);
            call_console(list_raw, 39, &[0, 3, c1.as_ptr() as usize, 0]);
            call_console(list_raw, 14, &[GRID[0] as usize, GRID[1] as usize, GRID[2] as usize]);
            call_console(list_raw, 39, &[0, 3, c2.as_ptr() as usize, 0]);
            call_console(list_raw, 14, &[GRID[0] as usize, GRID[1] as usize, GRID[2] as usize]);
            assert_eq!(call_console(list_raw, 9, &[]) as i32, S_OK.0);
            call_console(queue.as_raw(), 10, &[1, (&list_raw as *const *mut c_void) as usize]);
            wait_fence(&queue, &fence, &mut fval);
            assert_eq!(bind::activity().1, dispatches_before + 2);
            let want2 = eval_split_program(wave64, &eval_split_program(wave64, &initial, salt1), salt2);
            assert_eq!(std::slice::from_raw_parts(uma_buf as *const u32, initial.len()), want2.as_slice(), "wave64={wave64}: pass 1+2");

            alloc.Reset().unwrap();
            assert_eq!(call_console(list_raw, 10, &[alloc.as_raw() as usize, 0]) as i32, S_OK.0);
            call_console(list_raw, 33, &[game_rs.as_raw() as usize]);
            call_console(list_raw, 25, &[pso.as_raw() as usize]);
            call_console(list_raw, 39, &[0, 3, c3.as_ptr() as usize, 0]);
            call_console(list_raw, 14, &[GRID[0] as usize, GRID[1] as usize, GRID[2] as usize]);
            assert_eq!(call_console(list_raw, 9, &[]) as i32, S_OK.0);
            call_console(queue.as_raw(), 10, &[1, (&list_raw as *const *mut c_void) as usize]);
            wait_fence(&queue, &fence, &mut fval);
            let want3 = eval_split_program(wave64, &want2, salt3);
            assert_eq!(std::slice::from_raw_parts(uma_buf as *const u32, initial.len()), want3.as_slice(), "wave64={wave64}: pass 3");
            alloc.Reset().unwrap();
        }
    }
    unsafe { call_console(list_raw, 2, &[]); }
}
