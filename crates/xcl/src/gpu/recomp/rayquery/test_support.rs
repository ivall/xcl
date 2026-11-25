//! Native scene fixtures for the upcoming sample-gated traversal comparison.
#![allow(dead_code, unused_imports)]

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
