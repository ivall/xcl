use super::*;

#[test]
fn descriptor_capture_checks_spans_and_preserves_records() {
    unsafe {
        let data = [17u64, 29];
        assert_eq!(read_items::<u64>(data.as_ptr() as usize, 2).unwrap(), data);
        assert!(read_items::<u64>(0, 1).is_err());
        assert!(read_items::<u64>(usize::MAX - 2, 2).is_err());
        assert!(read_items::<u64>(data.as_ptr() as usize, MAX_DESCS + 1).is_err());
        assert!(read_items::<u64>(0, 0).unwrap().is_empty());
    }
}

#[test]
fn packed_vertex_spans_exclude_unused_alpha_and_check_overflow() {
    let mut triangle = D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
        VertexFormat: DXGI_FORMAT_R16G16B16A16_FLOAT,
        VertexCount: 3,
        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE { StartAddress: 256, StrideInBytes: 6 },
        ..Default::default()
    };
    assert_eq!(vertex_span(&triangle).unwrap(), 18);
    triangle.VertexCount = 0;
    assert!(vertex_span(&triangle).is_err());
    triangle.VertexCount = 3;
    triangle.VertexBuffer.StrideInBytes = u64::MAX;
    assert!(vertex_span(&triangle).is_err());
}

#[test]
fn gpu_native_input_budget_and_adopted_device() {
    use crate::gpu::recomp::{runner};
    use windows::Win32::Graphics::Direct3D::*;
    use windows::Win32::Graphics::Dxgi::*;

    unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory6>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            runner::skip_test("no DXGI factory");
            return;
        };
        let selector = crate::gpu::adapter::Selector::from_env_or("auto");
        let Some((adapter, desc)) = crate::gpu::adapter::candidates(&factory, &selector).into_iter().next() else {
            runner::skip_test("no GPU");
            return;
        };
        let Ok(base) = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_1) else {
            runner::skip_test("no D3D12 device");
            return;
        };
        let Ok(dev) = base.cast::<ID3D12Device5>() else {
            runner::skip_test("no DXR interface");
            return;
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS5).cast(), size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() as u32).unwrap();
        if options.RaytracingTier.0 < D3D12_RAYTRACING_TIER_1_1.0 {
            runner::skip_test("no DXR 1.1");
            return;
        }
        println!("Host DXR fixture on {}", crate::gpu::adapter::name(&desc));
        let mut prepared = Prepared::new(2 * 65536);
        let buffer = prepared.buffer(&dev, 36, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON).unwrap();
        assert_eq!(prepared.bytes, 65536);
        let source = |address, bytes| (address == 17 && bytes == 36).then(|| (buffer.clone(), 0));
        let copied = prepared.input(&dev, 17, 36, &source).unwrap();
        assert_ne!(copied, buffer.GetGPUVirtualAddress());
        assert_eq!(prepared.bytes, 2 * 65536);
        assert_eq!(prepared.copies.len(), 1);
        assert!(prepared.input(&dev, 17, 36, &source).is_err());
        assert_eq!(prepared.bytes, 2 * 65536);
        assert!(prepared.input(&dev, 18, 36, &source).is_err());
        let native = native_vtable(&dev) as *const _;
        real::adopt(dev.as_raw(), real::Kind::Device);
        assert_eq!(native_vtable(&dev) as *const _, native);
        println!("Native input copies respect the committed allocation cap on an adopted device");
    }
}
