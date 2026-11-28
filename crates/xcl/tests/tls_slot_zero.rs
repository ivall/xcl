//! Process-isolated tests for `xcl::tls::take_slot_zero` and `MOVED` tracking.
//!
//! Each test spawns a child copy of this binary so `_tls_index` starts at 0 and
//! `.CRT$XLY` receives real OS `DLL_THREAD_ATTACH` / `DLL_THREAD_DETACH` callbacks.

use std::cell::Cell;
use std::collections::HashSet;
use std::ffi::c_void;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows_sys::Win32::System::Memory::{GetProcessHeap, HeapValidate};
use windows_sys::Win32::System::Threading::{
    CreateThread, Sleep, TerminateThread, WaitForSingleObject, INFINITE,
};

const CHILD_ENV: &str = "XCL_TLS_SLOT_ZERO_CHILD";
const HOST_MAGIC: u64 = 0x1122_3344_5566_7788;
const HOST_MUTATED: u64 = 0xCAFE_BABE_DEAD_BEEF;

static CALLBACK_ACTIVE: AtomicBool = AtomicBool::new(false);

thread_local! {
    static HOST_SENTINEL: Cell<u64> = const { Cell::new(HOST_MAGIC) };
}

unsafe extern "system" fn test_tls_callback(_h: *mut c_void, reason: u32, _r: *mut c_void) {
    if !CALLBACK_ACTIVE.load(Ordering::Acquire) {
        return;
    }
    match reason {
        2 => xcl::tls::on_thread_attach(),
        3 => xcl::tls::on_thread_detach(),
        _ => {}
    }
}

#[used]
#[link_section = ".CRT$XLY"]
static TEST_TLS_CALLBACK: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) =
    test_tls_callback;

unsafe fn read_slot_zero_ptr() -> *mut u8 {
    let teb = xcl::tls::current_teb();
    assert!(!teb.is_null());
    let arr = *(teb.add(0x58) as *const *mut *mut u8);
    assert!(!arr.is_null());
    *arr
}

fn run_in_child_process(test_name: &str, body: impl FnOnce()) {
    unsafe { std::ptr::read_volatile(&TEST_TLS_CALLBACK) };
    if std::env::var_os(CHILD_ENV).is_some() {
        CALLBACK_ACTIVE.store(true, Ordering::Release);
        body();
        return;
    }
    let out = Command::new(std::env::current_exe().expect("current_exe"))
        .args(["--exact", test_name, "--nocapture"])
        .env(CHILD_ENV, "1")
        .output()
        .expect("spawn child test process");
    assert!(
        out.status.success(),
        "child test `{test_name}` failed (status {:?})\nstdout:\n{}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

struct SleeperArg {
    teb: AtomicUsize,
    ready: AtomicBool,
}

unsafe extern "system" fn park_forever(param: *mut c_void) -> u32 {
    let arg = &*(param as *const SleeperArg);
    assert_eq!(HOST_SENTINEL.with(|c| c.get()), HOST_MAGIC);
    arg.teb.store(xcl::tls::current_teb() as usize, Ordering::Release);
    arg.ready.store(true, Ordering::Release);
    Sleep(INFINITE);
    0
}

unsafe fn spawn_parked_threads(n: usize) -> (Vec<Box<SleeperArg>>, Vec<usize>) {
    let mut args = Vec::with_capacity(n);
    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let arg = Box::new(SleeperArg { teb: AtomicUsize::new(0), ready: AtomicBool::new(false) });
        let ptr = (&*arg as *const SleeperArg) as *mut c_void;
        let h = CreateThread(std::ptr::null(), 0, Some(park_forever), ptr, 0, std::ptr::null_mut());
        assert!(!h.is_null(), "CreateThread failed");
        args.push(arg);
        handles.push(h as usize);
    }
    for arg in &args {
        while !arg.ready.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
    }
    (args, handles)
}

#[test]
fn reused_teb_after_terminate_thread_still_exchanges_slots() {
    run_in_child_process("reused_teb_after_terminate_thread_still_exchanges_slots", || unsafe {
        let dir = std::env::temp_dir().join(format!("xcl_tls_s0_reuse_{}", std::process::id()));
        let template: &'static [u8; 64] = Box::leak(Box::new([0xA5u8; 64]));
        let mut title_index = u32::MAX;

        // 1. Spawn threads recorded in `MOVED` by `take_slot_zero`, then terminate them
        //    without DLL_THREAD_DETACH.
        let (args, handles) = spawn_parked_threads(16);
        let victim_tebs: HashSet<usize> = args.iter().map(|a| a.teb.load(Ordering::Acquire)).collect();
        assert_eq!(victim_tebs.len(), 16);

        // 2. Install carrier and take slot 0 for the title.
        let k = xcl::tls::install("reuse_test.exe", template.as_ptr() as usize, 64, 0x2000, &mut title_index, &dir)
            .expect("tls::install");
        xcl::tls::take_slot_zero("reuse_test.exe", &mut title_index, k).expect("tls::take_slot_zero");
        assert_eq!(title_index, 0);
        assert_eq!(HOST_SENTINEL.with(|c| c.get()), HOST_MAGIC);
        assert_eq!(std::slice::from_raw_parts(read_slot_zero_ptr(), 64), &template[..]);

        // 3. Kill all victims via TerminateThread and close our handles so Windows recycles TEBs.
        for h_raw in handles {
            let h = h_raw as HANDLE;
            assert_ne!(TerminateThread(h, 0), 0, "TerminateThread failed");
            assert_eq!(WaitForSingleObject(h, 5_000), WAIT_OBJECT_0);
            CloseHandle(h);
        }
        drop(args);

        // 4. Spawn fresh threads until Windows reuses victim TEBs; every thread must still swap.
        let mut reused = 0usize;
        for i in 0..2048 {
            let (teb, host_ok, title_ok, isolated_ok) = std::thread::spawn(|| {
                let teb = xcl::tls::current_teb() as usize;
                let host_before = HOST_SENTINEL.with(|c| c.get());
                let s0 = read_slot_zero_ptr();
                let title_bytes_ok = std::slice::from_raw_parts(s0, 64) == &[0xA5u8; 64];
                HOST_SENTINEL.with(|c| c.set(HOST_MUTATED));
                std::ptr::write_bytes(s0, 0x55, 64);
                let host_after = HOST_SENTINEL.with(|c| c.get());
                (teb, host_before == HOST_MAGIC, title_bytes_ok, host_after == HOST_MUTATED)
            })
            .join()
            .expect("fresh thread panicked");
            if victim_tebs.contains(&teb) {
                reused += 1;
            }
            assert!(host_ok, "host TLS corrupted on new thread (teb={teb:#x}, iter={i}, reused={reused})");
            assert!(title_ok, "slot 0 missing title template (teb={teb:#x}, iter={i}, reused={reused})");
            assert!(isolated_ok, "slot 0 and host slot overlapped (teb={teb:#x}, iter={i}, reused={reused})");
            if reused >= 4 {
                break;
            }
        }
        assert!(reused > 0, "expected Windows to reuse at least one terminated TEB");
        assert_ne!(HeapValidate(GetProcessHeap(), 0, std::ptr::null()), 0);
        let _ = std::fs::remove_dir_all(&dir);
    });
}

#[test]
fn take_slot_zero_survives_concurrent_thread_exit_and_termination() {
    run_in_child_process("take_slot_zero_survives_concurrent_thread_exit_and_termination", || unsafe {
        let dir = std::env::temp_dir().join(format!("xcl_tls_s0_churn_{}", std::process::id()));
        let template: &'static [u8; 64] = Box::leak(Box::new([0x3Cu8; 64]));
        let mut title_index = u32::MAX;

        let k = xcl::tls::install("churn_test.exe", template.as_ptr() as usize, 64, 0x2000, &mut title_index, &dir)
            .expect("tls::install");

        const N_CHURNERS: usize = 4;
        let start = Arc::new(Barrier::new(N_CHURNERS + 3));
        let done_swap = Arc::new(AtomicBool::new(false));
        let stop_churn = Arc::new(AtomicBool::new(false));

        let (surv_start, surv_done) = (start.clone(), done_swap.clone());
        let survivor = std::thread::spawn(move || {
            assert_eq!(HOST_SENTINEL.with(|c| c.get()), HOST_MAGIC);
            surv_start.wait();
            while !surv_done.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(HOST_SENTINEL.with(|c| c.get()), HOST_MAGIC);
            HOST_SENTINEL.with(|c| c.set(HOST_MUTATED));
            let s0 = read_slot_zero_ptr();
            assert_eq!(std::slice::from_raw_parts(s0, 64), &[0x3Cu8; 64]);
            std::ptr::write_bytes(s0, 0x77, 64);
            assert_eq!(HOST_SENTINEL.with(|c| c.get()), HOST_MUTATED);
        });

        let mut churners = Vec::with_capacity(N_CHURNERS);
        for _ in 0..N_CHURNERS {
            let (b, stop) = (start.clone(), stop_churn.clone());
            churners.push(std::thread::spawn(move || {
                b.wait();
                while !stop.load(Ordering::Acquire) {
                    let _ = std::thread::spawn(|| {
                        assert_eq!(HOST_SENTINEL.with(|c| c.get()), HOST_MAGIC);
                    })
                    .join();
                }
            }));
        }

        let (victim_args, victim_handles) = spawn_parked_threads(16);
        let killer_start = start.clone();
        let killer = std::thread::spawn(move || {
            killer_start.wait();
            for (i, h_raw) in victim_handles.into_iter().enumerate() {
                if i % 2 == 1 {
                    std::thread::sleep(Duration::from_millis(2));
                }
                let h = h_raw as HANDLE;
                let _ = TerminateThread(h, 0);
                let _ = WaitForSingleObject(h, 5_000);
                CloseHandle(h);
            }
        });

        start.wait();
        std::thread::sleep(Duration::from_millis(2));
        xcl::tls::take_slot_zero("churn_test.exe", &mut title_index, k).expect("tls::take_slot_zero");
        assert_eq!(title_index, 0);
        done_swap.store(true, Ordering::Release);
        stop_churn.store(true, Ordering::Release);

        survivor.join().expect("survivor thread panicked");
        killer.join().expect("killer thread panicked");
        for j in churners {
            j.join().expect("churner thread panicked");
        }
        drop(victim_args);

        assert_eq!(HOST_SENTINEL.with(|c| c.get()), HOST_MAGIC);
        assert_eq!(std::slice::from_raw_parts(read_slot_zero_ptr(), 64), &template[..]);
        assert_ne!(HeapValidate(GetProcessHeap(), 0, std::ptr::null()), 0);
        let _ = std::fs::remove_dir_all(&dir);
    });
}
