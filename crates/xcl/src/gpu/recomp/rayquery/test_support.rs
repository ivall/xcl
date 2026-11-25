//! Local retail sample only; no title code or data is stored in the repository.

use std::mem::{transmute_copy, ManuallyDrop};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use crate::gpu::bvh;
use super::super::{dxc, runner, translate, KERNEL_RES_DWORDS, KERNEL_SCC_SLOT, KERNEL_SGPRS};

unsafe fn buffer(dev: &ID3D12Device5, bytes: usize, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes.div_ceil(256) as u64 * 256,
        Height: 1, DepthOrArraySize: 1, MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if heap == D3D12_HEAP_TYPE_DEFAULT { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
        ..Default::default()
    };
    let mut resource = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE,
        &desc, state, None, &mut resource).unwrap();
    resource.unwrap()
}

unsafe fn upload(dev: &ID3D12Device5, words: &[u32]) -> ID3D12Resource {
    let resource = buffer(dev, words.len() * 4, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    let mut mapped = std::ptr::null_mut();
    resource.Map(0, None, Some(&mut mapped)).unwrap();
    std::ptr::copy_nonoverlapping(words.as_ptr(), mapped.cast(), words.len());
    resource.Unmap(0, None);
    resource
}

unsafe fn transition(resource: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
            pResource: transmute_copy(resource), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            StateBefore: before, StateAfter: after,
        }) },
        ..Default::default()
    }
}

pub(crate) unsafe fn scene(dev: &ID3D12Device5, queue: &ID3D12CommandQueue, triangles: &[bvh::Triangle]) -> (ID3D12Resource, Vec<ID3D12Resource>) {
    let words: Vec<u32> = triangles.iter().flatten().flatten().map(|v| v.to_bits()).collect();
    let vertices = upload(dev, &words);
    let geometry = D3D12_RAYTRACING_GEOMETRY_DESC {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES, Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 { Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
            VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT, VertexCount: triangles.len() as u32 * 3,
            VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE { StartAddress: vertices.GetGPUVirtualAddress(), StrideInBytes: 12 },
            ..Default::default()
        } },
    };
    let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL, NumDescs: 1, DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { pGeometryDescs: &geometry }, ..Default::default()
    };
    let allocator: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
    let list: ID3D12GraphicsCommandList4 = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
    let mut retained = Vec::new();
    let mut build = |inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS| {
        let mut size = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        dev.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut size);
        let result = buffer(dev, size.ResultDataMaxSizeInBytes as usize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
        let scratch = buffer(dev, size.ScratchDataSizeInBytes as usize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list.BuildRaytracingAccelerationStructure(&D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: result.GetGPUVirtualAddress(), ScratchAccelerationStructureData: scratch.GetGPUVirtualAddress(), Inputs: inputs, ..Default::default()
        }, None);
        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: transmute_copy(&result) }) }, ..Default::default() }]);
        retained.push(scratch); retained.push(result.clone()); result
    };
    let bottom = build(inputs);
    let mut instance = [0u32; 16];
    for i in [0, 5, 10] { instance[i] = 1f32.to_bits(); }
    instance[12] = 0xff000000;
    instance[14] = bottom.GetGPUVirtualAddress() as u32;
    instance[15] = (bottom.GetGPUVirtualAddress() >> 32) as u32;
    let instances = upload(dev, &instance);
    let top = build(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL, NumDescs: 1, DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { InstanceDescs: instances.GetGPUVirtualAddress() }, ..Default::default()
    });
    list.Close().unwrap(); queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap(); queue.Signal(&fence, 1).unwrap();
    crate::gpu::fence::wait(&fence, 1, &dev.cast().unwrap(), None, 30000).unwrap();
    // A TLAS references its BLAS even after the build has completed.
    (top, retained)
}

pub(crate) unsafe fn compare_native(dev: &ID3D12Device5, queue: &ID3D12CommandQueue, tlas: &ID3D12Resource) {
    let path = std::env::var_os("XCL_SAMPLE_RAYQUERY").unwrap();
    let blob = std::fs::read(path).unwrap();
    assert_eq!(crate::gpu::shader::fnv64(&blob), super::SHADOW_RECORD);
    let sh = crate::gpu::xbox::parse(&blob).unwrap();
    let f = sh.fssm.iter().find(|f| f.stages.iter().any(|s| s.hw == 6)).unwrap();
    let p = f.programs.first().unwrap();
    let mut code = f.data[p.offset + 0x13d4..p.offset + 0x1c6c].to_vec();
    code.extend_from_slice(&0xbf810000u32.to_le_bytes());
    let mut parameters: Vec<D3D12_ROOT_PARAMETER> = (0..6).map(|index| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: index, RegisterSpace: 0 } },
        ..Default::default()
    }).collect();
    parameters.push(D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
        Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 2 } }, ..Default::default() });
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: parameters.len() as u32, pParameters: parameters.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
        ..Default::default()
    };
    let mut serialized = None;
    D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut serialized, None).unwrap();
    let serialized = serialized.unwrap();
    let root: ID3D12RootSignature = if crate::hle::d3d12x::host_bvh::query_requested() {
        super::super::pso::abi_root_signature(&dev.cast().unwrap(), false).unwrap()
    } else {
        dev.CreateRootSignature(0, std::slice::from_raw_parts(serialized.GetBufferPointer().cast(), serialized.GetBufferSize())).unwrap()
    };
    let mut waves = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&mut waves as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS1).cast(), std::mem::size_of_val(&waves) as u32).unwrap();
    let mut pipelines = Vec::new();
    for native in [false, true] {
        let mut opt = super::super::Options { split_wave64: waves.WaveLaneCountMax < 64, ..Default::default() };
        opt.host_rayquery = native;
        let translation = translate(&code, &opt).unwrap();
        assert!(translation.complete(), "{:?}", translation.unsupported);
        assert_eq!(translation.hlsl.contains("TraceRayInline"), native);
        let dxil = dxc::compile(&translation.hlsl, "main", "cs_6_6", &[]).unwrap();
        let pipeline: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() },
            ..Default::default()
        }).unwrap();
        pipelines.push(pipeline);
    }
    let triangle = [[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]];
    let lower = |z: f32| triangle.map(|mut p| { p[2] = z; p });
    for (scene_index, triangles) in [vec![triangle], vec![triangle, lower(-1.)], vec![lower(-1.), triangle], vec![triangle, lower(-0.5), lower(-2.)]].into_iter().enumerate() {
    let built = (scene_index != 0).then(|| scene(dev, queue, &triangles));
    let tlas = built.as_ref().map_or(tlas, |built| &built.0);
    let bottom = bvh::build(&triangles, 0);
    let top = bvh::build_top(&[bvh::Instance {
        transform: [[1.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,1.,0.]], id: 0, mask: 0xff, contribution: 0, flags: 0,
        blas_address: 0x4000, blas_root: bottom.root, blas_bounds: bottom.bounds(),
    }]);
    let mut memory = vec![0u8; 0x8000];
    memory[0x1000..0x1000 + top.data.len()].copy_from_slice(&top.data);
    memory[0x4000..0x4000 + bottom.data.len()].copy_from_slice(&bottom.data);
    let mut memory: Vec<u32> = memory.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
    for case in 0..7 {
        let mut state = runner::KernelState::default();
        state.sgpr[4] = 0x1000;
        state.sgpr[super::super::IO_HOST_AS_LO] = if case == 5 { 0x2000 } else { 0x1000 };
        state.sgpr[super::super::IO_HOST_AS_HI] = 0;
        state.sgpr[super::super::IO_HOST_AS_MAGIC] = if case == 1 { 0 } else { super::MAGIC };
        if case == 2 { state.sgpr[126] = 0xaaaaaaaa; state.sgpr[127] = 0xaaaaaaaa; }
        for (lane,v) in state.vgpr.iter_mut().enumerate() {
            let (origin, direction, limit) = match if case == 4 { 0 } else { lane % 8 } {
                0 => ([0.25f32,0.25,1.], [0.,0.,-1.], 10.),
                1 => ([2.,2.,1.], [0.,0.,-1.], 10.),
                2 => ([0.25,0.25,-1.], [0.,0.,1.], 10.),
                3 => ([0.25,0.25,1.], [0.,0.,-1.], 1.),
                4 => ([0.25,0.25,1.], [0.,0.,-1.], 0.5),
                5 => ([0.25,0.25,0.], [0.,0.,-1.], 10.),
                6 => ([0.75,0.25,1.], [0.,0.,-1.], 10.),
                _ => ([0.25,0.25,1.], [0.,0.,-2.], 10.),
            };
            for (r,x) in [(6,origin[0]),(66,origin[1]),(62,origin[2]),(47,origin[0]),(48,origin[1]),(49,origin[2]),
                (11,direction[0]),(12,direction[1]),(52,direction[2]),(50,direction[0]),(51,direction[1]),(5,direction[2]),(8,limit)] { v[r] = x.to_bits(); }
            for r in [35,36,37] { v[r] = u32::MAX; }
        }
        if case == 6 { state.sgpr[126] = 0; state.sgpr[127] = 0; }
        if case == 3 { state.vgpr[17][47] = f32::NAN.to_bits(); state.vgpr[17][6] = f32::NAN.to_bits(); }
        let mut io = state.sgpr.clone(); io.resize(KERNEL_SGPRS,0); io[KERNEL_SCC_SLOT] = state.scc;
        for v in &state.vgpr { io.extend_from_slice(v); }
        let mut results = Vec::new();
        for pipeline in &pipelines {
            let allocator: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
            let uploads = [upload(dev,&io),upload(dev,&memory),upload(dev,&vec![0;KERNEL_RES_DWORDS+2])];
            let sizes = [io.len()*4,memory.len()*4,(KERNEL_RES_DWORDS+2)*4];
            let buffers: Vec<_> = sizes.iter().map(|&bytes| buffer(dev,bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST)).collect();
            for i in 0..3 {
                list.CopyBufferRegion(&buffers[i],0,&uploads[i],0,sizes[i] as u64);
                list.ResourceBarrier(&[transition(&buffers[i],D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
            }
            let readback = buffer(dev,sizes[2],D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
            list.SetComputeRootSignature(&root); list.SetPipelineState(pipeline);
            list.SetComputeRootUnorderedAccessView(0,buffers[0].GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(1,buffers[2].GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2,buffers[1].GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(6,tlas.GetGPUVirtualAddress());
            list.Dispatch(1,1,1);
            list.ResourceBarrier(&[transition(&buffers[2],D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyBufferRegion(&readback,0,&buffers[2],0,sizes[2] as u64); list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0,D3D12_FENCE_FLAG_NONE).unwrap(); queue.Signal(&fence,1).unwrap();
            crate::gpu::fence::wait(&fence,1,&dev.cast().unwrap(),None,30000).unwrap();
            let mut mapped = std::ptr::null_mut(); readback.Map(0,None,Some(&mut mapped)).unwrap();
            let words = std::slice::from_raw_parts(mapped.cast::<u32>(),sizes[2]/4).to_vec(); readback.Unmap(0,None);
            results.push(words);
        }
        for lane in 0..64 {
            for reg in [8,28,31] {
                let at = KERNEL_SGPRS + lane * 256 + reg;
                assert_eq!(results[1][at],results[0][at],"scene {scene_index}, case {case}, lane {lane}, v{reg}: native {} vs guest {}",
                    f32::from_bits(results[1][at]),f32::from_bits(results[0][at]));
            }
        }
        let expected = [64,0,32,0,64,0,0][case];
        assert_eq!(results[1][KERNEL_RES_DWORDS],expected,"native trace count for case {case}");
        let limits = results[1][KERNEL_RES_DWORDS+1];
        let minimum = [24,0,8,0,64,0,0][case];
        assert!(limits >= minimum && limits <= expected, "case {case}: only {limits} native search limits, expected at least {minimum}");
        println!("Retail region vs RayQuery scene {scene_index}, case {case}: 64 lane results matched, {expected} hardware traces, {limits} native search limits");
    }
    }
}
