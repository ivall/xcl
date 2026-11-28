//! A title's queue may GPU-wait for a fence another of its queues signals only in a later
//! submit (async compute does). Uploading CPU writes into the sparse windows
//! (`sparse::before_execute`) must not make that wait for itself.

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

use xcl::hle::d3d12x::sparse;

/// Queue A waits for fence F, then submits after a CPU write; queue B submits after another
/// CPU write and only then signals F. Both queues finish.
#[test]
fn a_wait_for_a_later_signal_on_another_queue_does_not_hang() {
    unsafe {
        std::env::set_var("XCL_UMA_MB", "64");
        std::env::set_var("XCL_COHERENT_MB", "64");
        let Ok(dev) = xcl::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
            xcl::gpu::recomp::runner::skip_test("no GPU");
            return;
        };
        let windows = sparse::init(&dev);
        let Some(&(k, _, _)) = windows.first() else {
            xcl::gpu::recomp::runner::skip_test("no sparse windows");
            return;
        };
        let (a, _) = sparse::alloc_in(k, 0x2_0000, 0x1_0000).expect("sparse allocation");
        let qa: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let qb: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
        let empty = |ty| {
            let al: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
            let l: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &al, None).unwrap();
            l.Close().unwrap();
            (al, l.cast::<ID3D12CommandList>().unwrap())
        };
        let (_aa, la) = empty(D3D12_COMMAND_LIST_TYPE_DIRECT);
        // a list is not submitted again before its first run has finished
        let (_aa2, la2) = empty(D3D12_COMMAND_LIST_TYPE_DIRECT);
        let (_ab, lb) = empty(D3D12_COMMAND_LIST_TYPE_COMPUTE);
        let f: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

        // A: earlier work, then a wait for what B signals later
        sparse::before_execute(&qa, &[], 0x1000);
        qa.ExecuteCommandLists(&[Some(la.clone())]);
        qa.Wait(&f, 1).unwrap();
        // the CPU writes, A submits
        std::ptr::write_bytes(a as *mut u8, 1, 0x1000);
        sparse::before_execute(&qa, &[a], 0x1000);
        qa.ExecuteCommandLists(&[Some(la2)]);
        qa.Signal(&done, 1).unwrap();
        // the CPU writes again, B submits and signals F
        std::ptr::write_bytes((a + 0x1_0000) as *mut u8, 2, 0x1000);
        sparse::before_execute(&qb, &[a + 0x1_0000], 0x1000);
        qb.ExecuteCommandLists(&[Some(lb)]);
        qb.Signal(&f, 1).unwrap();

        let ev = CreateEventW(None, false, false, None).unwrap();
        done.SetEventOnCompletion(1, ev).unwrap();
        let finished = WaitForSingleObject(ev, 5000) == WAIT_OBJECT_0;
        if !finished {
            // let the queues go so the process can end
            f.Signal(1).unwrap();
        }
        let _ = CloseHandle(ev);
        assert!(finished, "queue A did not finish in 5 s: the upload waits for A, which waits for B, which waits for the upload");
    }
}
