//! Synthetic compute -> indirect draw through the title's actual command-list entry points.

use super::*;

/// Own RDNA2 program: s[4:5] = buffer address, s6 = draw count. Only lane 0 writes.
/// Container layout follows XCL's existing `gpu_compute_e2e::console_cs` fixture builder.
fn argument_writer() -> Vec<u8> {
    let words: [u32; 11] = [
        0xBEFE_0481, // s_mov_b64 exec, 1
        0x7E00_0280, // v_mov_b32 v0, 0 (byte offset from s[4:5])
        0x7E02_0283, // v_mov_b32 v1, 3 (VertexCountPerInstance)
        0x7E04_0281, // v_mov_b32 v2, 1 (InstanceCount)
        0x7E06_0280, // v_mov_b32 v3, 0 (StartVertexLocation)
        0x7E08_0280, // v_mov_b32 v4, 0 (StartInstanceLocation)
        0xDC78_8010, 0x0004_0100, // global_store_dwordx4 v0, v[1:4], s[4:5] offset:16
        0x7E0A_0206, // v_mov_b32 v5, s6
        0xDC70_8030, 0x0004_0500, // global_store_dword v0, v5, s[4:5] offset:48
    ];
    let mut code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    code.extend_from_slice(&0xBF81_0000u32.to_le_bytes()); // s_endpgm
    let mut fssm = vec![0u8; 0x308];
    fssm[..4].copy_from_slice(b"FSSM");
    fssm[4..8].copy_from_slice(&0x101u32.to_le_bytes());
    fssm[0xC0..0xC8].copy_from_slice(&0x2F8u64.to_le_bytes());
    fssm[0xC8..0xD0].copy_from_slice(&(code.len() as u64).to_le_bytes());
    fssm[0x100..0x108].copy_from_slice(&0x0000_00C0_0002_0106u64.to_le_bytes());
    for (k, v) in [7u32 << 1, 64, 1, 1].into_iter().enumerate() {
        fssm[0x138 + 4 * k..0x13C + 4 * k].copy_from_slice(&v.to_le_bytes());
    }
    fssm.extend_from_slice(&code);
    let size = (fssm.len() - 0x10) as u32;
    fssm[8..12].copy_from_slice(&size.to_le_bytes());
    let z = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 6);
    let mut xbch = b"XBCH".to_vec();
    for x in [3, z.len() as u32, 0, fssm.len() as u32, 0] { le(&mut xbch, x); }
    xbch.extend_from_slice(&z);
    let mut dxil = Vec::new();
    for x in [5 << 16 | 0x66, 0, u32::from_le_bytes(*b"DXIL"), 0x106, 0, 4] { le(&mut dxil, x); }
    dxil.extend_from_slice(b"XDAT");
    for x in [2, 0x24, 0x12, 0x3C, 16, 0x0D, 0x4C, xbch.len() as u32] { le(&mut dxil, x); }
    dxil.extend_from_slice(&[0x5A; 16]);
    dxil.extend_from_slice(&xbch);
    dxil.resize((dxil.len() + 3) & !3, 0);
    let nobc = dxil.len();
    dxil.extend_from_slice(b"NOBC");
    let len = (dxil.len() / 4) as u32;
    dxil[4..8].copy_from_slice(&len.to_le_bytes());
    dxil[16..20].copy_from_slice(&((nobc - 8) as u32).to_le_bytes());
    let mut blob = vec![0u8; 36];
    blob[..4].copy_from_slice(b"DXBC");
    blob[20..24].copy_from_slice(&1u32.to_le_bytes());
    blob[28..32].copy_from_slice(&1u32.to_le_bytes());
    blob[32..36].copy_from_slice(&36u32.to_le_bytes());
    blob.extend_from_slice(b"DXIL");
    le(&mut blob, dxil.len() as u32);
    blob.extend_from_slice(&dxil);
    let len = blob.len() as u32;
    blob[24..28].copy_from_slice(&len.to_le_bytes());
    blob
}

pub fn compute_generated_indirect_draw() -> Result<(), String> {
    unsafe {
        let dev = device()?;
        let mut waves = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &mut waves as *mut _ as _, std::mem::size_of_val(&waves) as u32).map_err(e)?;
        if !waves.WaveOps.as_bool() || ![32, 64].iter().any(|&n| waves.WaveLaneCountMin <= n && n <= waves.WaveLaneCountMax) {
            println!("SKIPPED compute_generated_indirect_draw: no wave64");
            return Ok(());
        }
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).map_err(e)?;
        let graphics_rs = root_signature(&dev)?;
        let vs = compile("float4 main(uint id : SV_VertexID) : SV_Position { float2 p[3] = { float2(-0.5,-0.5), float2(0,0.5), float2(0.5,-0.5) }; return float4(p[id],0,1); }", "vs_5_0")?;
        let ps = compile("float4 main() : SV_Target { return float4(0,1,0,1); }", "ps_5_0")?;
        let mut gd = pso_desc(&graphics_rs, bc(&vs), bc(&ps));
        gd.pRootSignature = ManuallyDrop::new(Some(graphics_rs.clone()));
        let graphics: Result<ID3D12PipelineState, _> = dev.CreateGraphicsPipelineState(&gd);
        ManuallyDrop::drop(&mut gd.pRootSignature);
        let graphics = graphics.map_err(e)?;

        let param = D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 3 } },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let rd = D3D12_ROOT_SIGNATURE_DESC { NumParameters: 1, pParameters: &param, ..Default::default() };
        let serialize: D3D12SerializeRootSignatureFn = xs("D3D12SerializeRootSignature")?;
        let mut raw = std::ptr::null_mut();
        ensure!(serialize(&rd as *const _ as _, 1, &mut raw, std::ptr::null_mut()) == 0, "serialize compute root signature");
        let serialized = ID3DBlob::from_raw(raw);
        let compute_rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(serialized.GetBufferPointer() as *const u8, serialized.GetBufferSize())).map_err(e)?;
        let shader = argument_writer();
        let mut cd = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(compute_rs.clone())),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: shader.as_ptr() as _, BytecodeLength: shader.len() },
            ..Default::default()
        };
        // Console slot 62 has two additional parameters before riid/ppv.
        type ComputeX = unsafe extern "system" fn(*mut c_void, *const D3D12_COMPUTE_PIPELINE_STATE_DESC, usize, usize, *const windows::core::GUID, *mut *mut c_void) -> HRESULT;
        let create: ComputeX = transmute_copy(&*(*(dev.as_raw() as *const *const usize)).add(62));
        let mut raw = std::ptr::null_mut();
        let hr = create(dev.as_raw(), &cd, 0, 0, &ID3D12PipelineState::IID, &mut raw);
        ManuallyDrop::drop(&mut cd.pRootSignature);
        ensure!(hr >= 0 && !raw.is_null(), "compute PSO hr={hr:#x}");
        let compute = ID3D12PipelineState::from_raw(raw);

        let buffer_desc = |bytes, flags| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes, Height: 1,
            DepthOrArraySize: 1, MipLevels: 1, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Flags: flags, ..Default::default()
        };
        let make = |desc: &D3D12_RESOURCE_DESC, heap, state| -> Result<ID3D12Resource, String> {
            let mut out = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, desc, state, None, &mut out).map_err(e)?;
            out.ok_or("no resource".into())
        };
        let args = make(&buffer_desc(64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
        // The shader receives the console VA; ExecuteIndirect receives the resource + offsets.
        let va = args.GetGPUVirtualAddress();
        let tex = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 16, Height: 16,
            DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ..Default::default()
        };
        let rt = make(&tex, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET)?;
        let heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).map_err(e)?;
        let start: unsafe extern "system" fn(*mut c_void) -> usize = transmute_copy(&*(*(heap.as_raw() as *const *const usize)).add(9));
        let rtv = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start(heap.as_raw()) };
        dev.CreateRenderTargetView(&rt, None, rtv);
        let (mut footprint, mut bytes) = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(), 0u64);
        dev.GetCopyableFootprints(&tex, 0, 1, 0, Some(&mut footprint), None, None, Some(&mut bytes));
        let rb = make(&buffer_desc(bytes + 64, D3D12_RESOURCE_FLAG_NONE), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST)?;
        let argument = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, ..Default::default() };
        let mut sig: Option<ID3D12CommandSignature> = None;
        dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 16, NumArgumentDescs: 1, pArgumentDescs: &argument, NodeMask: 0 }, None, &mut sig).map_err(e)?;
        let sig = sig.ok_or("no command signature")?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        let transition = |r: &ID3D12Resource, before, after| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before, StateAfter: after,
            }) }
        };
        // Positive first: stale arguments/count must not make the zero-count control draw.
        for (run, count) in [1u32, 0].into_iter().enumerate() {
            if run != 0 {
                alloc.Reset().map_err(e)?;
                list.Reset(&alloc, None).map_err(e)?;
                list.ResourceBarrier(&[
                    transition(&args, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                    transition(&rt, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET),
                ]);
            }
            list.SetPipelineState(&compute);
            xslot::<SetRootSignature>(&list, 33)(list.as_raw(), compute_rs.as_raw());
            type Constants = unsafe extern "system" fn(*mut c_void, u32, u32, *const u32, u32);
            xslot::<Constants>(&list, 39)(list.as_raw(), 0, 3, [va as u32, (va >> 32) as u32, count].as_ptr(), 0);
            list.Dispatch(1, 1, 1);
            // Also orders writes through XCL's UMA UAV alias, rather than just this resource.
            list.ResourceBarrier(&[transition(&args, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)]);
            list.SetPipelineState(&graphics);
            xslot::<SetRootSignature>(&list, X_SET_GRAPHICS_ROOT_SIGNATURE)(list.as_raw(), graphics_rs.as_raw());
            xslot::<OmSetRenderTargets>(&list, X_OM_SET_RENDER_TARGETS)(list.as_raw(), 1, &rtv, 0, std::ptr::null());
            xslot::<ClearRtv>(&list, X_CLEAR_RENDER_TARGET_VIEW)(list.as_raw(), rtv, &[0.0, 0.0, 1.0, 1.0], 0, std::ptr::null());
            list.RSSetViewports(&[D3D12_VIEWPORT { Width: 16.0, Height: 16.0, MinDepth: 0.0, MaxDepth: 1.0, ..Default::default() }]);
            list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: 16, bottom: 16 }]);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.ExecuteIndirect(&sig, 1, &args, 16, &args, 48);
            list.ResourceBarrier(&[
                transition(&rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(&args, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE),
            ]);
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&rt), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.CopyBufferRegion(&rb, bytes, &args, 0, 64);
            list.Close().map_err(e)?;
            queue.ExecuteCommandLists(&[Some(list.cast().map_err(e)?)]);
            wait(&queue, &fence, run as u64 + 1)?;
            let mut p = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut p)).map_err(e)?;
            let data = std::slice::from_raw_parts(p as *const u8, (bytes + 64) as usize);
            let at = footprint.Offset as usize + 8 * footprint.Footprint.RowPitch as usize + 8 * 4;
            let center = data[at..at + 4].to_vec();
            let corner = data[footprint.Offset as usize..footprint.Offset as usize + 4].to_vec();
            let generated = data[bytes as usize + 16..bytes as usize + 32].to_vec();
            let generated_count = u32::from_le_bytes(data[bytes as usize + 48..bytes as usize + 52].try_into().unwrap());
            rb.Unmap(0, None);
            let expected: Vec<u8> = [3u32, 1, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
            ensure!(generated == expected && generated_count == count, "compute readback: args={generated:?}, count={generated_count}, expected count={count}");
            ensure!(corner == [0, 0, 255, 255], "count={count}: background={corner:?}");
            let want = if count == 0 { [0, 0, 255, 255] } else { [0, 255, 0, 255] };
            ensure!(center == want, "count={count}: center={center:?}, expected {want:?}");
        }
        Ok(())
    }
}
