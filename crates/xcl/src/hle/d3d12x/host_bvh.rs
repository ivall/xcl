//! Bounded input preparation for a future opt-in native acceleration-structure path.
#![allow(dead_code, unused_imports)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::{size_of, ManuallyDrop, MaybeUninit};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use super::real;

const LIMIT: u64 = 256 << 20;
const MAX_DESCS: usize = 8192;
// All T at the call sites are POD descriptor records or integer words.
unsafe fn read_items<T: Copy>(address: usize, count: usize) -> Result<Vec<T>, String> {
    let bytes = count.checked_mul(size_of::<T>()).ok_or("descriptor size overflow")?;
    if count > MAX_DESCS || address.checked_add(bytes).is_none() {
        return Err("descriptor capture limit exceeded".into());
    }
    if bytes == 0 {
        return Ok(Vec::new());
    }
    let mut values = Vec::<MaybeUninit<T>>::new();
    values.try_reserve_exact(count).map_err(|_| "descriptor allocation failed")?;
    values.resize_with(count, MaybeUninit::uninit);
    let mut done = 0;
    if address == 0 || ReadProcessMemory(
        GetCurrentProcess(), address as *const c_void, values.as_mut_ptr().cast(), bytes, &mut done,
    ) == 0 || done != bytes {
        return Err("descriptor memory is unreadable".into());
    }
    let mut values = ManuallyDrop::new(values);
    Ok(Vec::from_raw_parts(values.as_mut_ptr().cast(), values.len(), values.capacity()))
}

unsafe fn native_vtable<T: Interface>(object: &T) -> &T::Vtable {
    if real::is_adopted(object.as_raw()) {
        let table = *(object.as_raw() as *const *const usize);
        &*(*table.sub(1) as *const T::Vtable)
    } else {
        object.vtable()
    }
}

#[derive(Clone)]
struct HostAs {
    resource: ID3D12Resource,
    bytes: u64,
    kind: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE,
}

struct Prepared {
    resources: Vec<ID3D12Resource>,
    copies: Vec<(ID3D12Resource, u64, ID3D12Resource, u64)>,
    bytes: u64,
    limit: u64,
}

impl Prepared {
    fn new(limit: u64) -> Self {
        Self { resources: Vec::new(), copies: Vec::new(), bytes: 0, limit }
    }

    unsafe fn buffer(&mut self, dev: &ID3D12Device5, bytes: u64, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> Result<ID3D12Resource, String> {
        let bytes = bytes.checked_add(255).map(|n| n & !255).filter(|&n| n > 0).ok_or("invalid native buffer size")?;
        let committed = bytes.checked_add(65535).map(|n| n & !65535).ok_or("native allocation size overflow")?;
        if committed > self.limit.saturating_sub(self.bytes) {
            return Err("256 MiB shadow DXR capture budget reached".into());
        }
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes, Height: 1,
            DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: if heap == D3D12_HEAP_TYPE_DEFAULT && state != D3D12_RESOURCE_STATE_COPY_DEST {
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else { D3D12_RESOURCE_FLAG_NONE },
            ..Default::default()
        };
        let properties = D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() };
        let base = &*(native_vtable(dev) as *const _ as *const ID3D12Device_Vtbl);
        let mut resource = None::<ID3D12Resource>;
        (base.CreateCommittedResource)(
            dev.as_raw(), &properties, D3D12_HEAP_FLAG_NONE, &desc, state, std::ptr::null(),
            &ID3D12Resource::IID, (&mut resource as *mut Option<ID3D12Resource>).cast(),
        ).ok().map_err(|e| e.to_string())?;
        let resource = resource.ok_or("native buffer creation returned null")?;
        self.bytes += committed;
        self.resources.push(resource.clone());
        Ok(resource)
    }

    unsafe fn input(
        &mut self, dev: &ID3D12Device5, address: u64, bytes: u64,
        source: &dyn Fn(u64, u64) -> Option<(ID3D12Resource, u64)>,
    ) -> Result<u64, String> {
        let (original, offset) = source(address, bytes).ok_or("geometry range is outside a mapped UMA window")?;
        let buffer = self.buffer(dev, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST)?;
        let address = buffer.GetGPUVirtualAddress();
        self.copies.push((original, offset, buffer, bytes));
        Ok(address)
    }
}

fn vertex_bytes(format: DXGI_FORMAT) -> Option<u64> {
    match format {
        DXGI_FORMAT_R32G32B32_FLOAT => Some(12),
        DXGI_FORMAT_R32G32_FLOAT => Some(8),
        // DXR ignores A16, and packed positions may use a six-byte stride.
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_SNORM |
        DXGI_FORMAT_R16G16B16A16_UNORM => Some(6),
        DXGI_FORMAT_R16G16_FLOAT | DXGI_FORMAT_R16G16_SNORM |
        DXGI_FORMAT_R16G16_UNORM | DXGI_FORMAT_R10G10B10A2_UNORM => Some(4),
        _ => None,
    }
}

fn vertex_span(triangle: &D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC) -> Result<u64, String> {
    let element = vertex_bytes(triangle.VertexFormat).ok_or("unsupported DXR vertex format")?;
    u64::from(triangle.VertexCount).checked_sub(1)
        .and_then(|n| n.checked_mul(triangle.VertexBuffer.StrideInBytes))
        .and_then(|n| n.checked_add(element)).ok_or_else(|| "invalid DXR vertex span".into())
}

unsafe fn geometry(
    prepared: &mut Prepared, dev: &ID3D12Device5, inputs: &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
    source: &dyn Fn(u64, u64) -> Option<(ID3D12Resource, u64)>,
) -> Result<Vec<D3D12_RAYTRACING_GEOMETRY_DESC>, String> {
    let mut geometries = match inputs.DescsLayout {
        D3D12_ELEMENTS_LAYOUT_ARRAY => read_items(inputs.Anonymous.pGeometryDescs as usize, inputs.NumDescs as usize)?,
        D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS => {
            let pointers: Vec<usize> = read_items(inputs.Anonymous.ppGeometryDescs as usize, inputs.NumDescs as usize)?;
            pointers.into_iter().map(|p| read_items::<D3D12_RAYTRACING_GEOMETRY_DESC>(p, 1).map(|mut v| v.remove(0))).collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err("unsupported geometry descriptor layout".into()),
    };
    for geometry in &mut geometries {
        if geometry.Type != D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES {
            return Err("shadow DXR currently accepts triangle geometry".into());
        }
        let triangle = &mut geometry.Anonymous.Triangles;
        let bytes = vertex_span(triangle)?;
        triangle.VertexBuffer.StartAddress = prepared.input(dev, triangle.VertexBuffer.StartAddress, bytes, source)?;
        if triangle.IndexCount != 0 {
            let width = match triangle.IndexFormat {
                DXGI_FORMAT_R16_UINT => 2,
                DXGI_FORMAT_R32_UINT => 4,
                _ => return Err("unsupported DXR index format".into()),
            };
            triangle.IndexBuffer = prepared.input(dev, triangle.IndexBuffer, u64::from(triangle.IndexCount) * width, source)?;
        }
        if triangle.Transform3x4 != 0 {
            triangle.Transform3x4 = prepared.input(dev, triangle.Transform3x4, 48, source)?;
        }
    }
    Ok(geometries)
}

unsafe fn instances(
    prepared: &mut Prepared, dev: &ID3D12Device5, inputs: &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
    structures: &HashMap<u64, HostAs>,
) -> Result<u64, String> {
    if inputs.DescsLayout != D3D12_ELEMENTS_LAYOUT_ARRAY {
        return Err("shadow TLAS requires a contiguous instance array".into());
    }
    let address = inputs.Anonymous.InstanceDescs as usize;
    let mut instances: Vec<[u8; 64]> = read_items(address, inputs.NumDescs as usize).or_else(|error| {
        let cpu = crate::hle::kernelx::mirrors::source(address).ok_or(error)?;
        read_items(cpu, inputs.NumDescs as usize)
    })?;
    for instance in &mut instances {
        let console = u64::from_le_bytes(instance[56..64].try_into().unwrap());
        let native = structures.get(&console)
            .filter(|s| s.kind == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL)
            .ok_or("TLAS references a BLAS without a shadow build")?;
        instance[56..64].copy_from_slice(&native.resource.GetGPUVirtualAddress().to_le_bytes());
    }
    let bytes = size_of::<[u8; 64]>() * instances.len();
    let upload = prepared.buffer(dev, bytes as u64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ)?;
    let mut mapped = std::ptr::null_mut();
    upload.Map(0, None, Some(&mut mapped)).map_err(|e| e.to_string())?;
    std::ptr::copy_nonoverlapping(instances.as_ptr().cast::<u8>(), mapped.cast(), bytes);
    upload.Unmap(0, None);
    Ok(upload.GetGPUVirtualAddress())
}

unsafe fn transition(resource: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
            pResource: std::mem::transmute_copy(resource), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            StateBefore: before, StateAfter: after,
        }) },
        ..Default::default()
    }
}

unsafe fn uav_barrier(resource: Option<&ID3D12Resource>) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
            pResource: ManuallyDrop::new(resource.map(|resource| std::mem::transmute_copy(resource))),
        }) },
        ..Default::default()
    }
}

unsafe fn barriers(list: &ID3D12GraphicsCommandList4, barriers: &[D3D12_RESOURCE_BARRIER]) {
    let base = &*(native_vtable(list) as *const _ as *const ID3D12GraphicsCommandList_Vtbl);
    (base.ResourceBarrier)(list.as_raw(), barriers.len() as u32, barriers.as_ptr());
}

#[cfg(test)]
mod tests;
