//! Uploads into a texture whose subresources are in different states (#104): the barriers
//! `gpu::texture::upload_batch` records start from each subresource's own state and return to
//! it, checked by the D3D12 debug layer's state tracking. Its own process: the debug layer has
//! to be on before the first device exists.

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

use xcl::gpu::texture::{self, Subresource};

const SRV: D3D12_RESOURCE_STATES = texture::SHADER_READ;
// D3D12 can reuse the same device and its diagnostic queue within this process.
static DIAGNOSTICS: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn enable_debug_layer() -> Result<(), &'static windows::core::Error> {
    static ENABLED: std::sync::OnceLock<windows::core::Result<()>> = std::sync::OnceLock::new();
    ENABLED.get_or_init(|| {
        let debug = xcl::gpu::agility::configuration_interface::<ID3D12Debug>(&CLSID_D3D12Debug)?;
        unsafe { debug.EnableDebugLayer() };
        Ok(())
    }).as_ref().copied()
}

/// The debug layer's error and corruption messages since the last call (then cleared).
unsafe fn errors(q: &ID3D12InfoQueue) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..q.GetNumStoredMessages() {
        let mut len = 0usize;
        if q.GetMessage(i, None, &mut len).is_err() || len == 0 {
            continue;
        }
        let mut buf = vec![0u64; len.div_ceil(8)];
        let m = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
        if q.GetMessage(i, Some(m), &mut len).is_ok() && matches!((*m).Severity, D3D12_MESSAGE_SEVERITY_ERROR | D3D12_MESSAGE_SEVERITY_CORRUPTION) {
            let d = std::slice::from_raw_parts((*m).pDescription as *const u8, (*m).DescriptionByteLength.saturating_sub(1));
            out.push(String::from_utf8_lossy(d).into_owned());
        }
    }
    q.ClearStoredMessages();
    out
}

unsafe fn transition(r: &ID3D12Resource, sub: u32, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: sub, StateBefore: before, StateAfter: after }),
        },
    }
}

/// A texture in shader-resource states with mip 1 moved to COPY_SOURCE (as a title moves one
/// mip to copy from it), then uploaded again: with each subresource's state no error; with the
/// creation state for all of them (what uploads used before), the debug layer sees the
/// mismatch – so the check is a real one.
#[test]
fn uploads_start_from_each_subresources_state() {
    let _diagnostics = DIAGNOSTICS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    unsafe {
        if let Err(e) = enable_debug_layer() {
            println!("SKIPPED uploads_start_from_each_subresources_state: no D3D12 debug layer ({e})");
            return;
        }
        let Ok(dev) = xcl::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
            println!("SKIPPED uploads_start_from_each_subresources_state: no GPU");
            return;
        };
        let Ok(q) = dev.cast::<ID3D12InfoQueue>() else {
            println!("SKIPPED uploads_start_from_each_subresources_state: the device has no debug layer");
            return;
        };
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let mips = 4u16;
        let rd = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: mips,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r).unwrap();
        let r = r.unwrap();
        let subs = |v: u8| -> Vec<Subresource> { (0..mips as usize).map(|m| Subresource { rows: 64 >> m, row_bytes: (64 >> m) * 4, depth: 1, data: vec![v; (64 >> m) * (64 >> m) * 4] }).collect() };

        // made in COPY_DEST, left in the shader-resource states
        let up = |states: Vec<D3D12_RESOURCE_STATES>, v: u8| texture::upload_batch(&dev, &queue, &[(r.clone(), subs(v), states)]).unwrap();
        assert!(up(vec![D3D12_RESOURCE_STATE_COPY_DEST], 1).is_empty());
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        list.ResourceBarrier(&[transition(&r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, SRV)]);
        list.ResourceBarrier(&[transition(&r, 1, SRV, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        texture::wait(&dev, &queue).unwrap();
        let setup = errors(&q);
        assert!(setup.is_empty(), "setup: {setup:?}");

        // each subresource's state: no error, and it is left in it
        let mut states = vec![SRV; mips as usize];
        states[1] = D3D12_RESOURCE_STATE_COPY_SOURCE;
        assert!(up(states.clone(), 2).is_empty());
        let right = errors(&q);
        println!("per-subresource states: {} debug-layer errors", right.len());
        assert!(right.is_empty(), "uploads with each subresource's state: {right:?}");
        assert!(up(states, 3).is_empty());
        assert!(errors(&q).is_empty(), "the upload did not leave the subresources in their states");

        // one state for all of them, as uploads did before: mip 1 is not in it
        assert!(up(vec![SRV], 4).is_empty());
        let wrong = errors(&q);
        println!("one state for all: {} debug-layer errors, first: {:?}", wrong.len(), wrong.first());
        assert!(!wrong.is_empty(), "the debug layer did not see a transition from the wrong state – the check above proves nothing");
    }
}

#[test]
fn first_render_target_clear_depends_on_submitted_state() {
    let _diagnostics = DIAGNOSTICS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    unsafe {
        if enable_debug_layer().is_err() {
            xcl::gpu::recomp::runner::skip_test("no D3D12 debug layer");
            return;
        }
        let warp = xcl::gpu::adapter::Selector::from_env_or("auto") == xcl::gpu::adapter::Selector::Warp;
        let adapter = if warp {
            use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory4};
            let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
            let Ok(adapter) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                xcl::gpu::recomp::runner::skip_test("WARP is unavailable");
                return;
            };
            Some(adapter)
        } else {
            None
        };
        let device = match adapter.as_ref() {
            Some(adapter) => xcl::gpu::agility::create_device(adapter, D3D_FEATURE_LEVEL_11_0),
            None => xcl::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0),
        };
        let Ok(dev) = device else {
            xcl::gpu::recomp::runner::skip_test("no GPU");
            return;
        };
        if warp {
            println!("Using WARP");
        }
        let q: ID3D12InfoQueue = dev.cast().unwrap();
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let rtv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default()
        }).unwrap();
        let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
        let common = D3D12_RESOURCE_STATE_COMMON;
        let target = D3D12_RESOURCE_STATE_RENDER_TARGET;
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum ClearBarriers { None, Bracket, Restore }
        use ClearBarriers::{Bracket, Restore};
        let mut original_errors = 0;
        for (name, simultaneous, preceding_list, barriers, trailing_barrier, follow_up, expect_error) in [
            ("plain initial clear", false, false, ClearBarriers::None, true, None, true),
            ("implicit clear with later transition", true, false, ClearBarriers::None, true, None, true),
            ("implicit clear alone", true, false, ClearBarriers::None, false, None, false),
            ("explicit transitions around clear", false, false, Bracket, true, Some(target), false),
            ("clear after an earlier list prepared the resource", false, true, ClearBarriers::None, false, Some(target), false),
            ("unconditional transitions after an earlier list", false, true, Bracket, false, None, true),
            ("preparation with the redundant transition retained", false, true, ClearBarriers::None, true, None, true),
            ("restoring the creation state after the clear list", false, true, ClearBarriers::None, false, Some(common), false),
            ("restoring before the title transition", false, true, Restore, true, Some(target), false),
            ("restoring without a preparation list", false, false, Restore, true, Some(target), true),
        ] {
            let mut flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            if simultaneous {
                flags |= D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
            }
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 64, Height: 64,
                DepthOrArraySize: 1, MipLevels: 1, Flags: flags,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let mut resource: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, common, None, &mut resource).unwrap();
            let resource = resource.unwrap();
            dev.CreateRenderTargetView(&resource, None, rtv);
            assert!(errors(&q).is_empty(), "{name}: setup produced errors");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let to_target = || transition(&resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, common, target);
            if barriers == Bracket {
                list.ResourceBarrier(&[to_target()]);
            }
            list.ClearRenderTargetView(rtv, &[0.25, 0.5, 0.75, 1.0], None);
            if barriers != ClearBarriers::None {
                list.ResourceBarrier(&[transition(&resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, target, common)]);
            }
            if trailing_barrier {
                list.ResourceBarrier(&[to_target()]);
            }
            list.Close().unwrap();
            // The preparing list is recorded after the clear list, but it is submitted first.
            let mut submission = Vec::new();
            if preceding_list {
                let prep: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                prep.ResourceBarrier(&[to_target()]);
                prep.Close().unwrap();
                submission.push(Some(prep.cast().unwrap()));
            }
            submission.push(Some(list.cast().unwrap()));
            if follow_up == Some(common) {
                let restore: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                restore.ResourceBarrier(&[transition(&resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, target, common)]);
                restore.Close().unwrap();
                submission.push(Some(restore.cast().unwrap()));
            }
            queue.ExecuteCommandLists(&submission);
            texture::wait(&dev, &queue).unwrap();
            let observed = errors(&q);
            println!("{name}: {} debug-layer errors", observed.len());
            assert_eq!(!observed.is_empty(), expect_error, "{name}: {observed:?}");
            if name == "plain initial clear" {
                original_errors = observed.len();
            } else if barriers == Restore && !preceding_list {
                assert!(observed.len() <= original_errors, "{name}: {observed:?}");
            }
            if let Some(expected) = follow_up {
                let next: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                if expected == common {
                    next.ResourceBarrier(&[to_target()]);
                }
                next.ClearRenderTargetView(rtv, &[0.75, 0.5, 0.25, 1.0], None);
                next.ResourceBarrier(&[transition(&resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, target, common)]);
                next.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(next.cast().unwrap())]);
                texture::wait(&dev, &queue).unwrap();
                let actual = errors(&q);
                assert!(actual.is_empty(), "{name}: transition was not preserved across submissions: {actual:?}");
            }
        }
    }
}
