use super::*;
use super::super::{bind, title_list};
use crate::gpu::recomp::{dxc, runner};
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST};
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory6};

#[test]
fn deferred_graphics_keeps_render_targets_after_descriptor_overwrite() {
    deferred_render_target(false);
}

#[test]
fn deferred_graphics_keeps_render_targets_after_descriptor_heap_release() {
    deferred_render_target(true);
}

fn deferred_render_target(release_heap: bool) {
    if dxc::location().is_err() { runner::skip_test("no DXC"); return; }
    unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory1().unwrap();
        let selector = crate::gpu::adapter::Selector::from_env_or("auto");
        let adapter: IDXGIAdapter1 = if selector == crate::gpu::adapter::Selector::Warp {
            factory.EnumWarpAdapter().unwrap()
        } else {
            let Some((adapter, _)) = crate::gpu::adapter::candidates(&factory, &selector).into_iter().next() else {
                runner::skip_test(&crate::gpu::adapter::no_match(&factory, &selector)); return;
            };
            adapter
        };
        let dev = crate::gpu::agility::create_device(&adapter, D3D_FEATURE_LEVEL_11_0).unwrap();
        println!("deferred RTV adapter: {}", crate::gpu::adapter::name(&adapter.GetDesc1().unwrap()));
        let vs = dxc::compile(r#"[RootSignature("RootFlags(0)")] float4 main(uint i : SV_VertexID) : SV_Position {
            return float4(i == 2 ? 3 : -1, i == 1 ? 3 : -1, 0, 1); }"#, "main", "vs_6_0", &[]).unwrap();
        let ps = dxc::compile("float4 main() : SV_Target { return float4(1,0,0,1); }", "main", "ps_6_0", &[]).unwrap();
        let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &vs).unwrap();
        let mut pd = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: transmute_copy(&rs),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default()
        };
        pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pd.BlendState.RenderTarget[0].RenderTargetWriteMask = 15;
        let pso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&pd).unwrap();
        let make = |desc: &D3D12_RESOURCE_DESC, heap, state| {
            let mut out: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, desc, state, None, &mut out).unwrap();
            out.unwrap()
        };
        let texture = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 1, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ..Default::default()
        };
        let targets = [0; 2].map(|_| make(&texture, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET));
        let buffer = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 512, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default()
        };
        let readback = make(&buffer, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        let args = make(&buffer, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let mut mapped = std::ptr::null_mut();
        args.Map(0, None, Some(&mut mapped)).unwrap();
        std::ptr::copy_nonoverlapping([3u32, 1, 0, 0].as_ptr(), mapped as *mut u32, 4);
        args.Unmap(0, None);
        let hd = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() };
        let heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&hd).unwrap();
        let handle = heap.GetCPUDescriptorHandleForHeapStart();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let raw: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).unwrap();
        // Clear both through the source slot before recording the binding we must preserve.
        for target in &targets {
            dev.CreateRenderTargetView(target, None, handle);
            raw.ClearRenderTargetView(handle, &[0., 0., 0., 0.], None);
        }
        dev.CreateRenderTargetView(&targets[0], None, handle);
        let list = raw.into_raw();
        real::adopt(list, real::Kind::CommandList);
        on_create(list, alloc.as_raw() as usize, pso.as_raw() as usize, D3D12_COMMAND_LIST_TYPE_DIRECT);
        bind::reset(list, alloc.as_raw() as usize, pso.as_raw() as usize);
        let wrapper = title_list::wrap(list);
        bind::set_graphics_root_signature(list, rs.as_raw());
        ia_set_topology_x(list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST.0 as u32);
        rs_set_viewports_x(list, 1, &D3D12_VIEWPORT { Width: 1., Height: 1., MaxDepth: 1., ..Default::default() });
        rs_set_scissors_x(list, 1, &windows::Win32::Foundation::RECT { right: 1, bottom: 1, ..Default::default() });
        pc::<unsafe extern "system" fn(*mut c_void, u32, *const usize, i32, *const usize)>(list, PC_OM_SET_RENDER_TARGETS)(list, 1, &handle.ptr, 1, std::ptr::null());
        note_render_targets(list, 1, &handle.ptr, 1, std::ptr::null());
        let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, ..Default::default() };
        let mut signature: Option<ID3D12CommandSignature> = None;
        dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 16, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 }, None, &mut signature).unwrap();
        let signature = signature.unwrap();
        // Enter the exact deferred path directly; no build timing is needed to force it.
        assert!(defer(list, Deferred::Indirect(signature.as_raw() as usize, 1, args.as_raw() as usize, 0, 0, 0)));
        dev.CreateRenderTargetView(&targets[1], None, handle);
        if release_heap { drop(heap); }
        let mut segments = Vec::new();
        assert!(title_list::push_segments(wrapper, &mut segments));
        let next = *segments.last().unwrap();
        let tail = ID3D12GraphicsCommandList::from_raw_borrowed(&next).unwrap();
        for (i, target) in targets.iter().enumerate() {
            tail.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: transmute_copy(target), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET, StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                }) }, ..Default::default()
            }]);
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(target), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, ..Default::default() };
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: i as u64 * 256,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: texture.Format, Width: 1, Height: 1, Depth: 1, RowPitch: 256 } } } };
            tail.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        }
        let close: unsafe extern "system" fn(*mut c_void) -> i32 = std::mem::transmute(*(*(wrapper as *const *const usize)).add(9));
        assert_eq!(close(wrapper), 0);
        record_deferred(wrapper);
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let execute: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) = std::mem::transmute(*(*(queue.as_raw() as *const *const usize)).add(PC_QUEUE_EXECUTE_COMMAND_LISTS));
        execute(queue.as_raw(), segments.len() as u32, segments.as_ptr());
        let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&done, 1).unwrap();
        crate::gpu::fence::wait(&done, 1, &dev, None, 5000).expect("deferred draw completes");
        readback.Map(0, None, Some(&mut mapped)).unwrap();
        let a = std::slice::from_raw_parts(mapped as *const u8, 4).to_vec();
        let b = std::slice::from_raw_parts((mapped as *const u8).add(256), 4).to_vec();
        readback.Unmap(0, None);
        let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(wrapper as *const *const usize)).add(2));
        assert_eq!(release(wrapper), 0);
        assert_eq!((a, b), (vec![255, 0, 0, 255], vec![0, 0, 0, 0]), "the recorded draw must target A and leave B untouched");
    }
}
