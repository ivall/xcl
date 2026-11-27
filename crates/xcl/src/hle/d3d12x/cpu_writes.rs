//! Write32/64BitValueBottomOfPipeX into memory the GPU does not see: done by the CPU once the GPU
//! has finished the work before them.
//!
//! A console title marks its GPU progress with bottom-of-pipe writes into its own memory and
//! waits on them (a retail game: plain `VirtualAlloc` memory, outside XCL's unified memory).
//! The PC GPU cannot write there; dropped, the title waited for ever after its second frame.
//! Here each queue gets a fence of XCL's: after a submission whose lists carry such writes the
//! queue signals it, and a worker thread does the writes, in submission order, when the GPU is
//! there – what bottom of pipe means.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Mutex};

use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Threading as th;

/// A bottom-of-pipe write into memory outside unified memory, applied by the CPU once the
/// queue's fence reaches the submission that recorded it.
#[derive(Clone)]
pub(super) enum Write {
    /// Immediate dwords to store starting at `va`.
    Immediate(usize, Vec<u32>),
    /// An immediate 64-bit value to store at `va` (8-byte aligned).
    ImmediateU64(usize, u64),
    /// A 64-bit value resolved by the GPU into a persistently mapped READBACK buffer slot
    /// (`src`), copied atomically to `va` once the queue's fence completes. Holding `_buf` keeps
    /// the mapping alive until the worker finishes.
    ReadbackU64 { va: usize, src: usize, _buf: ID3D12Resource },
}

struct Job {
    fence: ID3D12Fence,
    value: u64,
    writes: Vec<Write>,
}

// SAFETY: the fence is a free-threaded COM object; the writes are plain data
unsafe impl Send for Job {}

/// queue → its fence, last value signalled, and dedicated worker channel
struct QueueState {
    fence: ID3D12Fence,
    value: u64,
    tx: mpsc::Sender<Job>,
}

unsafe impl Send for QueueState {}

static QUEUES: Mutex<Option<HashMap<usize, QueueState>>> = Mutex::new(None);

/// how long the worker waits for one submission before writing anyway (a removed device, a
/// queue that never gets there)
const WAIT_MS: u32 = 10_000;

fn spawn_queue_worker(queue_id: usize) -> mpsc::Sender<Job> {
    let (tx, rx) = mpsc::channel::<Job>();
    let thread_name = format!("xcl-cpu-writes-{queue_id:x}");
    let spawned = std::thread::Builder::new().name(thread_name).spawn(move || unsafe {
        let event = th::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
        for job in rx {
            if job.fence.GetCompletedValue() < job.value && job.fence.SetEventOnCompletion(job.value, HANDLE(event as *mut c_void)).is_ok() {
                th::WaitForSingleObject(event, WAIT_MS);
                if job.fence.GetCompletedValue() < job.value {
                    static LATE: AtomicU64 = AtomicU64::new(0);
                    let k = LATE.fetch_add(1, Ordering::Relaxed) + 1;
                    if k.is_power_of_two() {
                        crate::xlog_warn!(
                            "D3D12.X: {k} writes after the GPU so far done although queue {queue_id:#x} had not got there after {WAIT_MS} ms (fence at {}, wanted {}; {} writes)",
                            job.fence.GetCompletedValue(),
                            job.value,
                            job.writes.len()
                        );
                    }
                }
            }
            for w in &job.writes {
                match w {
                    Write::Immediate(va, words) => write_now(*va, words),
                    Write::ImmediateU64(va, val) => write_u64_now(*va, *val),
                    Write::ReadbackU64 { va, src, .. } => {
                        let value = (*(*src as *const AtomicU64)).load(Ordering::Acquire);
                        write_u64_now(*va, value);
                    }
                }
            }
        }
        if !event.is_null() {
            windows_sys::Win32::Foundation::CloseHandle(event);
        }
    });
    if let Err(e) = spawned {
        crate::xlog_error!("D3D12.X: no thread for writes after queue {queue_id:#x}: {e}");
    }
    tx
}

/// The dwords at `va`, if the memory is still there and writable.
unsafe fn write_now(va: usize, words: &[u32]) {
    let bytes = 4 * words.len();
    if !crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, bytes)) {
        use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_READWRITE};
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT {
            return;
        }
        if m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) == 0 || (m.BaseAddress as usize + m.RegionSize) < va + bytes {
            return;
        }
    }
    // Commit 8-byte aligned pairs atomically as 64-bit values to prevent torn reads.
    let mut i = 0;
    while i + 1 < words.len() && (va + 4 * i) % 8 == 0 {
        let v = (words[i] as u64) | ((words[i + 1] as u64) << 32);
        (*((va + 4 * i) as *const AtomicU64)).store(v, Ordering::Release);
        i += 2;
    }
    while i < words.len() {
        (*((va + 4 * i) as *const AtomicU32)).store(words[i], Ordering::Release);
        i += 1;
    }
}

/// Whether `[va, va + 8)` is 8-byte aligned, committed, and writable without `PAGE_GUARD`.
unsafe fn u64_writable_now(va: usize) -> bool {
    if va == 0 || !va.is_multiple_of(8) {
        return false;
    }
    let Some(end) = va.checked_add(8) else { return false };
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, 8)) {
        return true;
    }
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_READWRITE};
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT {
        return false;
    }
    m.Protect & PAGE_GUARD == 0
        && m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0
        && (m.BaseAddress as usize).saturating_add(m.RegionSize) >= end
}

/// An 8-byte aligned 64-bit value at `va`, if the memory is still there and writable.
unsafe fn write_u64_now(va: usize, value: u64) {
    if u64_writable_now(va) {
        (*(va as *const AtomicU64)).store(value, Ordering::Release);
    }
}

/// Advance an 8-byte aligned 64-bit value at `va` to at least `value` (`fetch_max`), if the memory
/// is still committed and writable. Best-effort check before the atomic write (same TOCTOU
/// trade-off as `write_u64_now`: a concurrent title decommit between check and store is unguarded).
pub(super) unsafe fn max_u64_now(va: usize, value: u64) {
    if u64_writable_now(va) {
        (*(va as *const AtomicU64)).fetch_max(value, Ordering::AcqRel);
    }
}

/// Whether `[va, va + bytes)` is memory the CPU can write for the GPU later.
pub(super) fn writable(va: usize, bytes: usize) -> bool {
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, bytes)) {
        return true;
    }
    crate::hle::kernelx::lazy::ensure(va, bytes);
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_READWRITE};
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let found = unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0;
    found
        && m.State == MEM_COMMIT
        && m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0
        && m.BaseAddress as usize + m.RegionSize >= va + bytes
}

/// `writes` of lists just submitted to `queue`: done when the GPU has finished them.
pub(super) unsafe fn after(queue: *mut c_void, writes: Vec<Write>) {
    if writes.is_empty() || queue.is_null() {
        return;
    }
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let (tx, job) = {
        let mut g = QUEUES.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        let entry = match m.entry(queue as usize) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let mut dev: Option<ID3D12Device> = None;
                if q.GetDevice(&mut dev).is_err() {
                    return;
                }
                let Some(dev) = dev else { return };
                let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return };
                let tx = spawn_queue_worker(queue as usize);
                v.insert(QueueState { fence, value: 0, tx })
            }
        };
        entry.value += 1;
        if q.Signal(&entry.fence, entry.value).is_err() {
            return;
        }
        (entry.tx.clone(), Job { fence: entry.fence.clone(), value: entry.value, writes })
    };
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| crate::xlog_info!("D3D12.X: bottom-of-pipe writes outside unified memory are done by the CPU after the GPU ({} in the first submission)", job.writes.len()));
    let _ = tx.send(job);
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Dxgi::*;

    /// The value lands after the queue has done what came before it.
    #[test]
    fn writes_land_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let cell = Box::new([0u32; 2]);
            let va = cell.as_ptr() as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Write::Immediate(va, vec![7, 9])]);
            let t = std::time::Instant::now();
            while (*(va as *const AtomicU32)).load(Ordering::Acquire) != 7 {
                assert!(t.elapsed().as_secs() < 5, "never written");
                std::thread::yield_now();
            }
            assert_eq!(*cell, [7, 9]);
        }
    }

    /// A ReadbackU64 write copies the 64-bit value from the mapped readback buffer after the queue finishes.
    #[test]
    fn readback_u64_lands_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut rb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
            let rb = rb.unwrap();
            let mut ptr: *mut c_void = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut ptr)).unwrap();
            let expected = 0x1234_5678_9ABC_DEF0u64;
            (*(ptr as *const AtomicU64)).store(expected, Ordering::Release);

            let dst = Box::new(0u64);
            let va = dst.as_ref() as *const u64 as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Write::ReadbackU64 { va, src: ptr as usize, _buf: rb }]);
            let t = std::time::Instant::now();
            while (*(va as *const AtomicU64)).load(Ordering::Acquire) != expected {
                assert!(t.elapsed().as_secs() < 5, "readback value never written");
                std::thread::yield_now();
            }
            assert_eq!(*dst, expected);
        }
    }

    fn test_device() -> Option<ID3D12Device> {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory = CreateDXGIFactory1::<IDXGIFactory4>().ok()?;
                let warp = factory.EnumWarpAdapter::<IDXGIAdapter>().ok()?;
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    return None;
                }
            }
            dev
        }
    }

    /// Atomic 64-bit writes do not tear between high and low dwords.
    #[test]
    fn atomic_64bit_write_ordering() {
        let cell = Box::new(0u64);
        let va = cell.as_ref() as *const u64 as usize;
        assert!(writable(va, 8));

        unsafe {
            write_u64_now(va, 0x1122_3344_5566_7788);
        }
        assert_eq!(*cell, 0x1122_3344_5566_7788);

        // Immediate write with 2 words on 8-byte aligned address
        let lo = 0xAABB_CCDDu32;
        let hi = 0xEEFF_0011u32;
        unsafe {
            write_now(va, &[lo, hi]);
        }
        assert_eq!(*cell, (lo as u64) | ((hi as u64) << 32));

        // Concurrent atomic tear test:
        // Worker repeatedly writes matching pairs (k, k).
        // Polling thread repeatedly loads u64 and verifies lo == hi (no tearing).
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        let stop = Arc::new(AtomicBool::new(false));
        let stop_reader = Arc::clone(&stop);
        let cell_ptr = cell.as_ref() as *const u64 as usize;

        let reader = std::thread::spawn(move || {
            let mut torn = false;
            while !stop_reader.load(Ordering::Acquire) {
                let val = unsafe { (*(cell_ptr as *const AtomicU64)).load(Ordering::Acquire) };
                let lo = val as u32;
                let hi = (val >> 32) as u32;
                if lo != hi {
                    torn = true;
                    break;
                }
                std::thread::yield_now();
            }
            torn
        });

        for k in 1..=20_000u32 {
            unsafe {
                write_now(cell_ptr, &[k, k]);
            }
        }
        stop.store(true, Ordering::Release);
        let torn = reader.join().unwrap();
        assert!(!torn, "64-bit immediate write exhibited torn read!");
    }

    /// Independent command queues have dedicated worker channels and do not head-of-line block.
    #[test]
    fn independent_queues_do_not_head_of_line_block() {
        let Some(dev) = test_device() else {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        };
        unsafe {
            let q1: ID3D12CommandQueue = dev
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let q2: ID3D12CommandQueue = dev
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                    ..Default::default()
                })
                .unwrap();

            // Stall Queue 1 on the GPU via an unsignalled fence
            let f_block: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            q1.Wait(&f_block, 1).unwrap();

            let cell1 = Box::new(0u64);
            let va1 = cell1.as_ref() as *const u64 as usize;
            assert!(writable(va1, 8));
            after(q1.as_raw(), vec![Write::ImmediateU64(va1, 0x1111_1111_1111_1111)]);

            // Queue 2 is unblocked and submitted after Queue 1
            let cell2 = Box::new(0u64);
            let va2 = cell2.as_ref() as *const u64 as usize;
            assert!(writable(va2, 8));
            after(q2.as_raw(), vec![Write::ImmediateU64(va2, 0x2222_2222_2222_2222)]);

            // Queue 2's write must complete without being blocked behind Queue 1's stall
            let t = std::time::Instant::now();
            while (*(va2 as *const AtomicU64)).load(Ordering::Acquire) != 0x2222_2222_2222_2222 {
                assert!(t.elapsed().as_secs() < 3, "Queue 2 was head-of-line blocked by Queue 1!");
                std::thread::yield_now();
            }
            assert_eq!(*cell2, 0x2222_2222_2222_2222);
            assert_eq!(*cell1, 0, "Queue 1 must remain stalled until unblocked");

            // Now unblock Queue 1
            f_block.Signal(1).unwrap();
            let t1 = std::time::Instant::now();
            while (*(va1 as *const AtomicU64)).load(Ordering::Acquire) != 0x1111_1111_1111_1111 {
                assert!(t1.elapsed().as_secs() < 5, "Queue 1 failed to complete after unblocking");
                std::thread::yield_now();
            }
            assert_eq!(*cell1, 0x1111_1111_1111_1111);
        }
    }
}
