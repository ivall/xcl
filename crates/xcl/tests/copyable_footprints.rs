//! Compare the D3D12.X GetCopyableFootprints slot with the original D3D12 method on the same
//! device, then use the translated footprints for a texture upload and readback.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::ptr::null_mut;
use std::sync::{Arc, Once};

use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;
use windows_sys::Win32::System::Diagnostics::Debug::RtlLookupFunctionEntry;
use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{self, real};
use xcl::modules::{self, MappedModule};

type GetFootprints = unsafe extern "system" fn(
    *mut c_void,
    *const D3D12_RESOURCE_DESC,
    u32,
    u32,
    u64,
    *mut D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    *mut u32,
    *mut u64,
    *mut u64,
);

#[derive(Clone, Copy)]
struct Subresources {
    first: u32,
    count: u32,
    base: u64,
}

#[derive(Debug, PartialEq)]
struct Footprints {
    layouts: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    row_bytes: Vec<u64>,
    total: u64,
}

impl Footprints {
    unsafe fn query(
        dev: &ID3D12Device,
        get: GetFootprints,
        desc: &D3D12_RESOURCE_DESC,
        slice: Subresources,
        with_total: bool,
    ) -> Self {
        let count = slice.count as usize;
        let mut out = Self {
            layouts: vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count],
            rows: vec![0; count],
            row_bytes: vec![0; count],
            total: u64::MAX,
        };
        get(
            dev.as_raw(), desc, slice.first, slice.count, slice.base,
            out.layouts.as_mut_ptr(), out.rows.as_mut_ptr(), out.row_bytes.as_mut_ptr(),
            if with_total { &mut out.total } else { null_mut() },
        );
        out
    }

    fn row_ranges(&self, sub: usize) -> impl Iterator<Item = Range<usize>> {
        let layout = self.layouts[sub];
        let count = self.rows[sub] as usize * layout.Footprint.Depth as usize;
        let row_bytes = self.row_bytes[sub] as usize;
        (0..count).map(move |row| {
            let start = layout.Offset as usize + row * layout.Footprint.RowPitch as usize;
            start..start + row_bytes
        })
    }
}

fn device() -> Option<ID3D12Device> {
    if d3d12x::backend() != d3d12x::Backend::Real {
        runner::skip_test("no GPU");
        return None;
    }
    let raw = real::device_raw();
    println!("backend device: {}", real::stats().adapter);
    Some(unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap().clone())
}

unsafe fn queries(dev: &ID3D12Device) -> (GetFootprints, GetFootprints) {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // Mark only the fixture function as title code. The slot distinguishes title callers
        // from host callers by the return address, so a direct call from the test is native.
        let mut image_base = 0;
        let address = title_get_footprints as *const () as usize as u64;
        let entry = RtlLookupFunctionEntry(address, &mut image_base, null_mut());
        let entry = entry.as_ref().expect("unwind information for the fixture function");
        let base = image_base as usize + entry.BeginAddress as usize;
        let size = (entry.EndAddress - entry.BeginAddress) as usize;
        modules::register_mapped(Arc::new(MappedModule {
            name: "footprints_fixture.exe".into(),
            path: "footprints_fixture.exe".into(),
            base,
            size,
            entry_rva: 0,
            is_dll: false,
            exports: Default::default(),
            tls_callbacks: Default::default(),
            no_thread_calls: false.into(),
            initialized: true.into(),
            stack_reserve: 0,
        }));
    });
    let native = real::orig_fn(dev.as_raw(), 38);
    (title_get_footprints, std::mem::transmute::<usize, GetFootprints>(native))
}

#[inline(never)]
unsafe extern "system" fn title_get_footprints(
    dev: *mut c_void,
    desc: *const D3D12_RESOURCE_DESC,
    first: u32,
    count: u32,
    base: u64,
    layouts: *mut D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    rows: *mut u32,
    row_bytes: *mut u64,
    total: *mut u64,
) {
    let vtable = &**dev.cast::<*const ID3D12Device_Vtbl>();
    (vtable.GetCopyableFootprints)(dev, desc, first, count, base, layouts, rows, row_bytes, total);
    // Keep the return address inside this fixture even in an optimized test build.
    std::hint::black_box(total);
}

#[test]
fn footprints_for_a_partial_bc1_array_match_native_d3d12() {
    let Some(dev) = device() else { return };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 68,
        Height: 36,
        DepthOrArraySize: 2,
        MipLevels: 4,
        Format: DXGI_FORMAT_BC1_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    // Start at the last mip of array slice 0 and continue into slice 1. The base exercises
    // 64-bit offsets without allocating a buffer of that size.
    let slice = Subresources { first: 3, count: 4, base: (1 << 32) + 512 };
    unsafe {
        let (hooked, native) = queries(&dev);
        let expected = Footprints::query(&dev, native, &desc, slice, true);
        assert_ne!(expected.total, u64::MAX, "driver rejected the BC1 array");
        assert!(expected.layouts.iter().all(|layout| layout.Offset >= slice.base));
        assert_eq!(Footprints::query(&dev, hooked, &desc, slice, true), expected,
            "partial BC1 array with a base above 4 GiB");
    }
}

#[test]
fn console_volume_footprints_match_native_and_copy_on_gpu() {
    let Some(dev) = device() else { return };
    let pc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        Width: 19,
        Height: 11,
        DepthOrArraySize: 7,
        MipLevels: 4,
        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    // XCL represents the console's R10G10B10_7E3_A2_FLOAT texture with four half floats.
    let console = D3D12_RESOURCE_DESC { Format: DXGI_FORMAT(116), ..pc };
    let slice = Subresources { first: 1, count: 3, base: 1024 };
    unsafe {
        let (hooked, native) = queries(&dev);
        let mut invalid = 0;
        native(dev.as_raw(), &console, slice.first, slice.count, slice.base,
            null_mut(), null_mut(), null_mut(), &mut invalid);
        assert_eq!(invalid, u64::MAX, "console format must exercise the retry path");

        let expected = Footprints::query(&dev, native, &pc, slice, true);
        assert_ne!(expected.total, u64::MAX, "driver rejected the host volume");
        let upload = Footprints::query(&dev, hooked, &console, slice, true);
        assert_eq!(upload, expected, "console format with total bytes requested");

        let without_total = Footprints::query(&dev, hooked, &console, slice, false);
        assert_eq!(without_total, Footprints { total: u64::MAX, ..expected },
            "console format without total bytes requested");

        let download = Footprints::query(&dev, native, &pc, Subresources { base: 0, ..slice }, true);
        assert_ne!(download.total, u64::MAX);
        assert_eq!(download.rows, upload.rows);
        assert_eq!(download.row_bytes, upload.row_bytes);
        for (src, dst) in upload.layouts.iter().zip(&download.layouts) {
            assert_eq!(src.Footprint, dst.Footprint);
        }
        copy_volume(&dev, &pc, slice, &upload, &download);
    }
}

unsafe fn resource(
    dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let mut out = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() },
        D3D12_HEAP_FLAG_NONE, desc, state, None, &mut out).unwrap();
    out.unwrap()
}

unsafe fn copy_volume(
    dev: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    slice: Subresources,
    upload: &Footprints,
    download: &Footprints,
) {
    let buffer = |width| D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: width, Height: 1,
        DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default()
    };
    let texture = resource(dev, desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    let src_bytes = (slice.base + upload.total) as usize;
    let dst_bytes = download.total as usize;
    let src = resource(dev, &buffer(src_bytes as u64), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    let dst = resource(dev, &buffer(dst_bytes as u64), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    let mut expected = Vec::new();
    let mut data = vec![0xA5; src_bytes];
    for sub in 0..slice.count as usize {
        let payload: Vec<u8> = (0..upload.row_ranges(sub).map(|range| range.len()).sum())
            .map(|i: usize| (i * 17 + i / 251 + sub * 53) as u8).collect();
        for (range, row) in upload.row_ranges(sub).zip(payload.chunks_exact(upload.row_bytes[sub] as usize)) {
            data[range].copy_from_slice(row);
        }
        expected.push(payload);
    }
    let mut mapped = null_mut();
    src.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut mapped)).unwrap();
    std::ptr::copy_nonoverlapping(data.as_ptr(), mapped.cast(), src_bytes);
    src.Unmap(0, None);

    let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
    // The descriptors borrow resources kept alive through the fence wait.
    let tex_region = |sub| D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::transmute_copy(&texture),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
    };
    let buf_region = |res: &ID3D12Resource, layout| D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::transmute_copy(res),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: layout },
    };
    for (i, layout) in upload.layouts.iter().enumerate() {
        list.CopyTextureRegion(&tex_region(slice.first + i as u32), 0, 0, 0, &buf_region(&src, *layout), None);
    }
    let transition = D3D12_RESOURCE_TRANSITION_BARRIER {
        pResource: std::mem::transmute_copy(&texture),
        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
    };
    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: ManuallyDrop::new(transition) },
        ..Default::default()
    }]);
    for (i, layout) in download.layouts.iter().enumerate() {
        list.CopyTextureRegion(&buf_region(&dst, *layout), 0, 0, 0, &tex_region(slice.first + i as u32), None);
    }
    list.Close().unwrap();
    queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
    queue.Signal(&fence, 1).unwrap();
    let event = CreateEventW(None, false, false, None).unwrap();
    let waited = fence.SetEventOnCompletion(1, event).map(|()| WaitForSingleObject(event, 10_000));
    CloseHandle(event).unwrap();
    assert_eq!(waited.unwrap(), WAIT_OBJECT_0, "GPU copy did not finish");

    dst.Map(0, None, Some(&mut mapped)).unwrap();
    let readback = std::slice::from_raw_parts(mapped.cast::<u8>(), dst_bytes);
    for (sub, payload) in expected.iter().enumerate() {
        for (row, (range, bytes)) in download.row_ranges(sub)
            .zip(payload.chunks_exact(download.row_bytes[sub] as usize)).enumerate()
        {
            assert_eq!(&readback[range], bytes, "mip {}, row {row}", slice.first + sub as u32);
        }
    }
    dst.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
    println!("GPU round trip checked {} bytes across {} mips", expected.iter().map(Vec::len).sum::<usize>(), slice.count);
}
