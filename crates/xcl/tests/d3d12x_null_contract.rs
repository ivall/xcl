#![allow(non_snake_case)]

use std::ffi::c_void;
use std::ptr::{null, null_mut};

use windows_sys::Win32::Foundation::CloseHandle;
use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

use xcl::hle::d3d12x::null::{
    capture_compute_desc, capture_graphics_desc, capture_stream_desc, subobject_payload,
    D3D10CreateBlob, D3D12CreateDevice, D3D12CreateRootSignatureDeserializer,
    D3D12CreateVersionedRootSignatureDeserializer, D3D12EnableExperimentalFeatures,
    D3D12GetDebugInterface, D3D12GetInterface, D3D12SerializeRootSignature,
    D3D12SerializeVersionedRootSignature, D3D12XboxCreateDevice, D3D12XboxSetProcessDebugFlags,
};
use xcl::util::{E_INVALIDARG, E_NOINTERFACE, E_NOTIMPL, HRESULT, S_FALSE, S_OK};

const WAIT_OBJECT_0: u32 = 0;

#[repr(C, align(8))]
struct Aligned<const N: usize>([u8; N]);

unsafe fn slot(obj: *mut c_void, i: usize) -> usize {
    *(*(obj as *const *const usize)).add(i)
}

#[test]
fn subobject_payload_known_and_unknown_types() {
    let expected: &[(u32, (usize, usize))] = &[
        (0, (8, 8)), (1, (16, 8)), (2, (16, 8)), (3, (16, 8)), (4, (16, 8)),
        (5, (16, 8)), (6, (16, 8)), (7, (32, 8)), (8, (328, 4)), (9, (4, 4)),
        (10, (44, 4)), (11, (52, 4)), (12, (16, 8)), (13, (4, 4)), (14, (4, 4)),
        (15, (36, 4)), (16, (4, 4)), (17, (8, 4)), (18, (4, 4)), (19, (16, 8)),
        (20, (4, 4)), (21, (56, 4)), (22, (24, 8)), (24, (16, 8)), (25, (16, 8)),
        (26, (60, 4)), (27, (44, 4)), (28, (40, 4)),
    ];
    for &(ty, layout) in expected {
        assert_eq!(subobject_payload(ty), Some(layout), "subobject type {ty}");
    }
    for unknown in [23, 29, 100, u32::MAX] {
        assert_eq!(subobject_payload(unknown), None, "unknown type {unknown}");
    }
}

#[test]
fn capture_desc_and_stream_guards() {
    unsafe {
        // Null and unreadable pointers must be ignored without faulting.
        capture_graphics_desc(null());
        capture_graphics_desc(0x1000 as *const u8);
        capture_compute_desc(null());
        capture_compute_desc(0x1000 as *const u8);
        capture_stream_desc(null());
        capture_stream_desc(0x1000 as *const u8);

        // Stream desc with unreadable stream base or short size.
        let bad_base: [usize; 2] = [32, 0x1000];
        capture_stream_desc(bad_base.as_ptr() as *const u8);
        let mut buf = Aligned([0u8; 64]);
        let short_size: [usize; 2] = [4, buf.0.as_ptr() as usize];
        capture_stream_desc(short_size.as_ptr() as *const u8);

        // Valid stream: SAMPLE_MASK (type 9, 4B payload @ +4, stride 8)
        // followed by unknown type 99 (stops parse before trailing bytes).
        buf.0.fill(0);
        buf.0[0..4].copy_from_slice(&9u32.to_le_bytes());
        buf.0[4..8].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        buf.0[8..12].copy_from_slice(&99u32.to_le_bytes());
        let stream_desc: [usize; 2] = [buf.0.len(), buf.0.as_ptr() as usize];
        capture_stream_desc(stream_desc.as_ptr() as *const u8);
    }
}

#[test]
fn entry_points_and_blob_contract() {
    unsafe {
        assert_eq!(D3D12CreateDevice(null_mut(), 0xC100, null(), null_mut()), S_FALSE);
        assert_eq!(D3D12XboxCreateDevice(null_mut(), null(), null(), null_mut()), S_FALSE);

        let mut dev1 = null_mut();
        let mut dev2 = null_mut();
        assert_eq!(D3D12CreateDevice(null_mut(), 0xC100, null(), &mut dev1), S_OK);
        assert_eq!(D3D12XboxCreateDevice(null_mut(), null(), null(), &mut dev2), S_OK);
        assert!(!dev1.is_null());
        assert_eq!(dev1, dev2, "null device is a process-wide singleton");

        // D3D10CreateBlob: invalid out, zero-size, and writable non-zero buffer.
        assert_eq!(D3D10CreateBlob(16, null_mut()), E_INVALIDARG);
        let mut empty_blob = null_mut();
        assert_eq!(D3D10CreateBlob(0, &mut empty_blob), S_OK);
        let get_ptr: unsafe extern "system" fn(*mut c_void) -> *mut c_void = std::mem::transmute(slot(empty_blob, 3));
        let get_size: unsafe extern "system" fn(*mut c_void) -> usize = std::mem::transmute(slot(empty_blob, 4));
        assert!(!get_ptr(empty_blob).is_null());
        assert_eq!(get_size(empty_blob), 0);

        let mut blob = null_mut();
        assert_eq!(D3D10CreateBlob(64, &mut blob), S_OK);
        let ptr = get_ptr(blob) as *mut u8;
        assert!(!ptr.is_null());
        assert_eq!(get_size(blob), 64);
        std::ptr::write_bytes(ptr, 0x5A, 64);
        assert_eq!(*ptr.add(63), 0x5A);

        // Debug, experimental, and root-signature entry points.
        let mut out = blob;
        assert_eq!(D3D12GetDebugInterface(null(), &mut out), E_NOINTERFACE);
        assert!(out.is_null());

        out = blob;
        assert_eq!(D3D12GetInterface(null(), null(), &mut out), E_NOINTERFACE);
        assert!(out.is_null());

        assert_eq!(D3D12EnableExperimentalFeatures(0, null(), null_mut(), null_mut()), E_NOINTERFACE);

        out = blob;
        assert_eq!(D3D12CreateRootSignatureDeserializer(null(), 0, null(), &mut out), E_NOTIMPL);
        assert!(out.is_null());

        out = blob;
        assert_eq!(D3D12CreateVersionedRootSignatureDeserializer(null(), 0, null(), &mut out), E_NOTIMPL);
        assert!(out.is_null());

        let mut rs_blob = null_mut();
        let mut rs_err = blob;
        assert_eq!(D3D12SerializeRootSignature(null(), 1, &mut rs_blob, &mut rs_err), S_OK);
        assert!(rs_err.is_null());
        assert_eq!(get_size(rs_blob), 16);

        let mut vrs_blob = null_mut();
        let mut vrs_err = blob;
        assert_eq!(D3D12SerializeVersionedRootSignature(null(), &mut vrs_blob, &mut vrs_err), S_OK);
        assert!(vrs_err.is_null());
        assert_eq!(get_size(vrs_blob), 16);

        assert_eq!(D3D12XboxSetProcessDebugFlags(0), S_OK);
    }
}

#[test]
fn device_queries_and_feature_support() {
    unsafe {
        let mut dev = null_mut();
        assert_eq!(D3D12XboxCreateDevice(null_mut(), null(), null(), &mut dev), S_OK);

        let get_node_count: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(slot(dev, 7));
        assert_eq!(get_node_count(dev), 1);

        let get_luid: unsafe extern "system" fn(*mut c_void, *mut u64) -> *mut u64 = std::mem::transmute(slot(dev, 43));
        let mut luid = 0u64;
        assert_eq!(get_luid(dev, &mut luid), &mut luid as *mut u64);
        assert_eq!(luid, 0x5843_4C00);

        let get_inc: unsafe extern "system" fn(*mut c_void, u32) -> u32 = std::mem::transmute(slot(dev, 15));
        assert_eq!(get_inc(dev, 0), 32);

        // GetCustomHeapProperties: DEFAULT (1) -> NOT_AVAILABLE (1); UPLOAD (2) / READBACK (3) -> WRITE_BACK (3).
        let get_custom_heap: unsafe extern "system" fn(*mut c_void, *mut u32, u32, u32) -> *mut u32 =
            std::mem::transmute(slot(dev, 26));
        for (ty, expected_cpu_prop) in [(1u32, 1u32), (2, 3), (3, 3)] {
            let mut props = [0u32; 5];
            get_custom_heap(dev, props.as_mut_ptr(), 0, ty);
            assert_eq!(props[0], 4, "heap type CUSTOM for ty={ty}");
            assert_eq!(props[1], expected_cpu_prop, "cpu page property for ty={ty}");
            assert_eq!(props[2], 1, "memory pool L0 for ty={ty}");
        }

        // GetResourceAllocationInfo: 64 KiB alignment across buffer + texture2d descs.
        let mut descs = Aligned([0u8; 56 * 2]);
        // Desc 0: BUFFER (dim=1), Width = 1000 -> 64 KiB (0x10000).
        descs.0[0..4].copy_from_slice(&1u32.to_le_bytes());
        descs.0[16..24].copy_from_slice(&1000u64.to_le_bytes());
        // Desc 1: TEXTURE2D (dim=3), 128x128x1 -> approx 87381 -> 128 KiB (0x20000).
        let d1 = &mut descs.0[56..112];
        d1[0..4].copy_from_slice(&3u32.to_le_bytes());
        d1[16..24].copy_from_slice(&128u64.to_le_bytes());
        d1[24..28].copy_from_slice(&128u32.to_le_bytes());
        d1[28..30].copy_from_slice(&1u16.to_le_bytes());

        let get_alloc_info: unsafe extern "system" fn(*mut c_void, *mut u64, u32, u32, *const u8) -> *mut u64 =
            std::mem::transmute(slot(dev, 25));
        let mut alloc_info = [0u64; 2];
        get_alloc_info(dev, alloc_info.as_mut_ptr(), 0, 2, descs.0.as_ptr());
        assert_eq!(alloc_info[0], 0x3_0000);
        assert_eq!(alloc_info[1], 0x1_0000);

        // GetCopyableFootprints: 256-byte row-pitch alignment.
        let mut tex_desc = Aligned([0u8; 56]);
        tex_desc.0[0..4].copy_from_slice(&3u32.to_le_bytes());
        tex_desc.0[16..24].copy_from_slice(&100u64.to_le_bytes());
        tex_desc.0[24..28].copy_from_slice(&64u32.to_le_bytes());
        tex_desc.0[28..30].copy_from_slice(&1u16.to_le_bytes());
        tex_desc.0[32..36].copy_from_slice(&28u32.to_le_bytes());

        let get_footprints: unsafe extern "system" fn(
            *mut c_void, *const u8, u32, u32, u64, *mut u8, *mut u32, *mut u64, *mut u64,
        ) = std::mem::transmute(slot(dev, 38));
        let mut layout = Aligned([0u8; 32]);
        let (mut num_rows, mut row_size, mut total) = (0u32, 0u64, 0u64);
        get_footprints(
            dev, tex_desc.0.as_ptr(), 0, 1, 0x200,
            layout.0.as_mut_ptr(), &mut num_rows, &mut row_size, &mut total,
        );
        assert_eq!(u64::from_le_bytes(layout.0[0..8].try_into().unwrap()), 0x200);
        assert_eq!(u32::from_le_bytes(layout.0[8..12].try_into().unwrap()), 28);
        assert_eq!(u32::from_le_bytes(layout.0[12..16].try_into().unwrap()), 100);
        assert_eq!(u32::from_le_bytes(layout.0[16..20].try_into().unwrap()), 64);
        assert_eq!(u32::from_le_bytes(layout.0[20..24].try_into().unwrap()), 1);
        let row_pitch = u32::from_le_bytes(layout.0[24..28].try_into().unwrap());
        assert_eq!(row_pitch, 512);
        assert_eq!(row_pitch % 256, 0);
        assert_eq!(num_rows, 64);
        assert_eq!(row_size, 400);
        assert!(total >= (row_pitch as u64) * 64);

        // CheckFeatureSupport: OPTIONS5 (27) writes RaytracingTier at +8.
        let check_feature: unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32) -> HRESULT =
            std::mem::transmute(slot(dev, 13));
        let mut opts5 = [0xFFFF_FFFFu32; 3];
        assert_eq!(check_feature(dev, 27, opts5.as_mut_ptr() as *mut c_void, 12), S_OK);
        assert_eq!(opts5[2], 0);
    }
}

#[test]
fn resource_fence_device_child_and_refcount() {
    unsafe {
        let mut dev = null_mut();
        assert_eq!(D3D12CreateDevice(null_mut(), 0xC100, null(), &mut dev), S_OK);

        // CreateCommittedResource (#27) & CreateReservedResource (#30).
        let mut desc = Aligned([0u8; 56]);
        desc.0[0..4].copy_from_slice(&1u32.to_le_bytes()); // BUFFER
        desc.0[16..24].copy_from_slice(&4096u64.to_le_bytes());

        let create_committed: unsafe extern "system" fn(
            *mut c_void, *const c_void, u32, *const u8, u32, *const c_void, *const c_void, *mut *mut c_void,
        ) -> HRESULT = std::mem::transmute(slot(dev, 27));
        let create_reserved: unsafe extern "system" fn(
            *mut c_void, *const u8, u32, *const c_void, *const c_void, *mut *mut c_void,
        ) -> HRESULT = std::mem::transmute(slot(dev, 30));

        let mut r1 = null_mut();
        let mut r2 = null_mut();
        assert_eq!(create_committed(dev, null(), 0, desc.0.as_ptr(), 0, null(), null(), &mut r1), S_OK);
        assert_eq!(create_reserved(dev, desc.0.as_ptr(), 0, null(), null(), &mut r2), S_OK);

        // GetDesc (#10) roundtrips the 56-byte descriptor.
        let get_desc: unsafe extern "system" fn(*mut c_void, *mut u8) -> *mut u8 = std::mem::transmute(slot(r1, 10));
        let mut readback = [0u8; 56];
        get_desc(r1, readback.as_mut_ptr());
        assert_eq!(readback, desc.0);

        // GetHeapProperties (#14) returns DEFAULT (1) and flags = 0.
        let get_heap_props: unsafe extern "system" fn(*mut c_void, *mut u32, *mut u32) -> HRESULT =
            std::mem::transmute(slot(r1, 14));
        let mut heap_props = [0u32; 5];
        let mut heap_flags = 99u32;
        assert_eq!(get_heap_props(r1, heap_props.as_mut_ptr(), &mut heap_flags), S_OK);
        assert_eq!(heap_props[0], 1);
        assert_eq!(heap_flags, 0);

        // GetGPUVirtualAddress (#11): non-zero, 64 KiB aligned, distinct.
        let get_va: unsafe extern "system" fn(*mut c_void) -> u64 = std::mem::transmute(slot(r1, 11));
        let va1 = get_va(r1);
        let va2 = get_va(r2);
        assert_ne!(va1, 0);
        assert_ne!(va2, 0);
        assert_eq!(va1 & 0xFFFF, 0);
        assert_eq!(va2 & 0xFFFF, 0);
        assert_ne!(va1, va2);

        // Map (#8) allocates once and persists writes across subsequent Map calls.
        let map: unsafe extern "system" fn(*mut c_void, u32, *const c_void, *mut *mut c_void) -> HRESULT =
            std::mem::transmute(slot(r1, 8));
        let mut p1 = null_mut();
        let mut p2 = null_mut();
        assert_eq!(map(r1, 0, null(), &mut p1), S_OK);
        assert!(!p1.is_null());
        std::ptr::write_bytes(p1 as *mut u8, 0xCD, 4096);
        assert_eq!(map(r1, 0, null(), &mut p2), S_OK);
        assert_eq!(p1, p2);
        assert_eq!(*(p2 as *const u8).add(4095), 0xCD);

        // Fence (#36): initial value, SetEventOnCompletion (already signaled & future), Signal.
        let create_fence: unsafe extern "system" fn(*mut c_void, u64, u32, *const c_void, *mut *mut c_void) -> HRESULT =
            std::mem::transmute(slot(dev, 36));
        let mut fence = null_mut();
        assert_eq!(create_fence(dev, 3, 0, null(), &mut fence), S_OK);

        let completed: unsafe extern "system" fn(*mut c_void) -> u64 = std::mem::transmute(slot(fence, 8));
        let set_event: unsafe extern "system" fn(*mut c_void, u64, *mut c_void) -> HRESULT =
            std::mem::transmute(slot(fence, 9));
        let fence_signal: unsafe extern "system" fn(*mut c_void, u64) -> HRESULT =
            std::mem::transmute(slot(fence, 10));

        assert_eq!(completed(fence), 3);

        let ev_already = CreateEventW(null(), 1, 0, null());
        assert!(!ev_already.is_null());
        assert_eq!(set_event(fence, 3, ev_already), S_OK);
        assert_eq!(WaitForSingleObject(ev_already, 0), WAIT_OBJECT_0);
        CloseHandle(ev_already);

        let ev_future = CreateEventW(null(), 1, 0, null());
        assert!(!ev_future.is_null());
        assert_eq!(set_event(fence, 10, ev_future), S_OK);
        assert_eq!(completed(fence), 10);
        assert_eq!(WaitForSingleObject(ev_future, 0), WAIT_OBJECT_0);
        CloseHandle(ev_future);

        assert_eq!(fence_signal(fence, 42), S_OK);
        assert_eq!(completed(fence), 42);

        // DeviceChild::GetDevice (#7) returns the singleton device.
        let get_device: unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> HRESULT =
            std::mem::transmute(slot(fence, 7));
        assert_eq!(get_device(fence, null(), null_mut()), E_INVALIDARG);
        let mut child_dev = null_mut();
        assert_eq!(get_device(fence, null(), &mut child_dev), S_OK);
        assert_eq!(child_dev, dev);

        // COM QueryInterface / AddRef / Release (#0, #1, #2).
        let qi: unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> HRESULT =
            std::mem::transmute(slot(fence, 0));
        let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(slot(fence, 1));
        let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(slot(fence, 2));

        assert_eq!(qi(fence, null(), null_mut()), E_INVALIDARG);
        let mut qi_out = null_mut();
        assert_eq!(qi(fence, null(), &mut qi_out), S_OK);
        assert_eq!(qi_out, fence);
        assert_eq!(add_ref(fence), 3);
        assert_eq!(release(fence), 2);
        assert_eq!(release(fence), 1);
        assert_eq!(release(fence), 0);
        assert_eq!(release(fence), 0);
    }
}
