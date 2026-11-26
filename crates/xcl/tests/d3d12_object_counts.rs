use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use xcl::hle::d3d12x::{present, real};
use real::{D3d12ObjectCounts, Kind};

#[test]
fn tracked_objects_follow_native_release_and_gpu_completion() {
    // This integration test has its own process and sets the options before creating a device.
    std::env::set_var("XCL_D3D12_OBJECT_COUNTS", "1");
    std::env::set_var("XCL_LATE_RELEASE", "1");
    unsafe {
        let Ok(device) = xcl::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
            return xcl::gpu::recomp::runner::skip_test("no D3D12 GPU");
        };
        assert_eq!(real::object_counts(), Some(D3d12ObjectCounts::default()));

        let properties = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };
        let mut heap: Option<ID3D12Heap> = None;
        device.CreateHeap(&D3D12_HEAP_DESC { SizeInBytes: 65_536, Properties: properties, ..Default::default() }, &mut heap).unwrap();
        let heap = heap.unwrap();
        real::adopt(heap.as_raw(), Kind::Heap);

        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 4096,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut resource).unwrap();
        let resource = resource.unwrap();
        real::adopt(resource.as_raw(), Kind::Resource);
        real::adopt(resource.as_raw(), Kind::Resource);
        drop(resource.clone());
        assert_eq!(real::object_counts(), Some(D3d12ObjectCounts { resources: 1, heaps: 1, deferred_resources: 0 }));

        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let gate: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Wait(&gate, 1).unwrap();
        present::note_execute(queue.as_raw(), &[]);
        drop(resource);
        let held = real::object_counts();
        gate.Signal(1).unwrap();
        assert_eq!(held, Some(D3d12ObjectCounts { resources: 1, heaps: 1, deferred_resources: 1 }));

        let start = Instant::now();
        let drained = Some(D3d12ObjectCounts { resources: 0, heaps: 1, deferred_resources: 0 });
        while real::object_counts() != drained {
            assert!(start.elapsed() < Duration::from_secs(5), "resource still counted after the queue completed");
            std::thread::sleep(Duration::from_millis(5));
        }
        drop(heap);
        assert_eq!(real::object_counts(), Some(D3d12ObjectCounts::default()));
    }
}
