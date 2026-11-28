//! The file calls XCL hooks commit the title's lazy pages (`kernelx::lazy`: mapped, still only
//! reserved) they read into or write from: the kernel fails on such a page (ReadFileEx:
//! ERROR_NOACCESS, WriteFile: ERROR_INVALID_USER_BUFFER) rather than faulting it in.

use std::cell::Cell;
use std::ffi::c_void;
use std::fs;
use std::path::PathBuf;
use std::ptr::{null, null_mut};
use std::sync::Once;
use std::time::{SystemTime, UNIX_EPOCH};
use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, ReadFileEx, ReadFileScatter, WriteFile, WriteFileEx, WriteFileGather, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING,
    FILE_FLAG_OVERLAPPED, FILE_SEGMENT_ELEMENT, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
};
use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE};
use windows_sys::Win32::System::Threading::{CreateEventW, SleepEx};
use windows_sys::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};
use xcl::hle::kernelx::lazy;
use xcl::util::wide;

/// a lazy page (64 KiB)
const PAGE: usize = 0x1_0000;
/// a system page: one scatter / gather segment
const SEG: usize = 0x1000;

fn hooks() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| xcl::vfs_hooks::install().unwrap());
}

/// `n` reserved pages marked lazy; marks and memory go even when the test fails.
struct Lazy(usize, usize);

impl Lazy {
    fn new(n: usize) -> Self {
        let r = unsafe { VirtualAlloc(null(), n * PAGE, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(r, 0);
        for i in 0..n {
            lazy::mark(r + i * PAGE);
        }
        Lazy(r, n)
    }
}

impl Drop for Lazy {
    fn drop(&mut self) {
        lazy::ensure(self.0, self.1 * PAGE);
        unsafe { VirtualFree(self.0 as *mut c_void, 0, MEM_RELEASE) };
    }
}

struct Handle(HANDLE);

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { CloseHandle(self.0) };
    }
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("xcl_lazy_{tag}_{}_{nonce}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn open(path: &PathBuf, access: u32, disposition: u32, flags: u32) -> Handle {
    let h = unsafe {
        CreateFileW(wide(&path.to_string_lossy()).as_ptr(), access, FILE_SHARE_READ | FILE_SHARE_WRITE, null(), disposition, flags, null_mut())
    };
    assert_ne!(h, INVALID_HANDLE_VALUE, "CreateFileW({}) failed with {}", path.display(), unsafe { GetLastError() });
    Handle(h)
}

fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 7 + 1) as u8).collect()
}

thread_local! {
    static DONE: Cell<Option<(u32, u32)>> = const { Cell::new(None) };
}

unsafe extern "system" fn done(err: u32, n: u32, _: *mut OVERLAPPED) {
    DONE.with(|d| d.set(Some((err, n))));
}

#[test]
fn read_file_ex_into_a_lazy_page_reads_the_data() {
    hooks();
    let dir = TempDir::new("readex");
    let path = dir.0.join("data.bin");
    let data = pattern(4096);
    fs::write(&path, &data).unwrap();
    let page = Lazy::new(1);
    let h = open(&path, GENERIC_READ, OPEN_EXISTING, FILE_FLAG_OVERLAPPED);
    let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
    DONE.with(|d| d.set(None));
    let ok = unsafe { ReadFileEx(h.0, page.0 as *mut u8, data.len() as u32, &mut ov, Some(done)) };
    let err = if ok == 0 { unsafe { GetLastError() } } else { 0 };
    for _ in 0..50 {
        if ok == 0 || DONE.with(|d| d.get()).is_some() {
            break;
        }
        unsafe { SleepEx(100, 1) };
    }
    assert_eq!((ok, err), (1, 0), "ReadFileEx into a lazy page");
    assert_eq!(DONE.with(|d| d.get()), Some((0, data.len() as u32)), "completion (error, bytes)");
    assert_eq!(unsafe { std::slice::from_raw_parts(page.0 as *const u8, data.len()) }, &data[..]);
}

#[test]
fn write_file_from_a_lazy_page_writes_zeros() {
    hooks();
    let dir = TempDir::new("write");
    let path = dir.0.join("out.bin");
    let page = Lazy::new(1);
    let mut written = 0u32;
    let (ok, err) = {
        let h = open(&path, GENERIC_WRITE, CREATE_ALWAYS, 0);
        let ok = unsafe { WriteFile(h.0, (page.0 + 0x100) as *const u8, 4096, &mut written, null_mut()) };
        (ok, if ok == 0 { unsafe { GetLastError() } } else { 0 })
    };
    assert_eq!((ok, err), (1, 0), "WriteFile from a lazy page");
    assert_eq!(written, 4096);
    assert_eq!(fs::read(&path).unwrap(), vec![0u8; 4096]);
}

#[test]
fn write_file_ex_from_a_lazy_page_writes_zeros() {
    hooks();
    let dir = TempDir::new("writeex");
    let path = dir.0.join("out.bin");
    let page = Lazy::new(1);
    let (ok, err) = {
        let h = open(&path, GENERIC_WRITE, CREATE_ALWAYS, FILE_FLAG_OVERLAPPED);
        let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
        DONE.with(|d| d.set(None));
        let ok = unsafe { WriteFileEx(h.0, page.0 as *const u8, 4096, &mut ov, Some(done)) };
        let err = if ok == 0 { unsafe { GetLastError() } } else { 0 };
        for _ in 0..50 {
            if ok == 0 || DONE.with(|d| d.get()).is_some() {
                break;
            }
            unsafe { SleepEx(100, 1) };
        }
        (ok, err)
    };
    assert_eq!((ok, err), (1, 0), "WriteFileEx from a lazy page");
    assert_eq!(DONE.with(|d| d.get()), Some((0, 4096)), "completion (error, bytes)");
    assert_eq!(fs::read(&path).unwrap(), vec![0u8; 4096]);
}

/// An unbuffered, overlapped handle (what scatter / gather need) and the segments of two
/// system pages in two different lazy pages.
fn segments(page: &Lazy) -> [FILE_SEGMENT_ELEMENT; 3] {
    let seg = |p: usize| FILE_SEGMENT_ELEMENT { Buffer: p as *mut c_void };
    [seg(page.0), seg(page.0 + PAGE + 3 * SEG), seg(0)]
}

/// Waits for an overlapped call: (bytes, error).
fn finish(h: &Handle, ov: &mut OVERLAPPED, ok: i32) -> (u32, u32) {
    let err = if ok == 0 { unsafe { GetLastError() } } else { 0 };
    if ok == 0 && err != 997 {
        return (0, err);
    }
    let mut n = 0u32;
    if unsafe { GetOverlappedResult(h.0, ov, &mut n, 1) } == 0 {
        return (n, unsafe { GetLastError() });
    }
    (n, 0)
}

#[test]
fn scatter_and_gather_with_lazy_segments_move_the_data() {
    hooks();
    let dir = TempDir::new("scatter");
    let path = dir.0.join("data.bin");
    let data = pattern(2 * SEG);
    fs::write(&path, &data).unwrap();
    let page = Lazy::new(2);
    let seg = segments(&page);
    let event = Handle(unsafe { CreateEventW(null(), 1, 0, null()) });
    let read = {
        let h = open(&path, GENERIC_READ, OPEN_EXISTING, FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING);
        let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
        ov.hEvent = event.0;
        let ok = unsafe { ReadFileScatter(h.0, seg.as_ptr(), data.len() as u32, null_mut(), &mut ov) };
        finish(&h, &mut ov, ok)
    };
    assert_eq!(read, (data.len() as u32, 0), "ReadFileScatter into lazy pages (bytes, error)");
    let at = |p: usize| unsafe { std::slice::from_raw_parts(p as *const u8, SEG) };
    assert_eq!(at(page.0), &data[..SEG]);
    assert_eq!(at(page.0 + PAGE + 3 * SEG), &data[SEG..]);

    let page = Lazy::new(2);
    let seg = segments(&page);
    let out = dir.0.join("out.bin");
    let write = {
        let h = open(&out, GENERIC_WRITE, CREATE_ALWAYS, FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING);
        let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
        ov.hEvent = event.0;
        let ok = unsafe { WriteFileGather(h.0, seg.as_ptr(), data.len() as u32, null_mut(), &mut ov) };
        finish(&h, &mut ov, ok)
    };
    assert_eq!(write, (data.len() as u32, 0), "WriteFileGather from lazy pages (bytes, error)");
    assert_eq!(fs::read(&out).unwrap(), vec![0u8; data.len()]);
}

#[test]
fn scatter_and_gather_reject_invalid_segment_array_without_crashing() {
    hooks();
    let dir = TempDir::new("scatter_bad");
    let path = dir.0.join("data.bin");
    fs::write(&path, pattern(SEG)).unwrap();
    let noaccess = unsafe { VirtualAlloc(null(), SEG, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS) };
    assert!(!noaccess.is_null());
    let event = Handle(unsafe { CreateEventW(null(), 1, 0, null()) });

    for bad in [0x1000usize as *const FILE_SEGMENT_ELEMENT, noaccess as *const FILE_SEGMENT_ELEMENT] {
        let h_r = open(&path, GENERIC_READ, OPEN_EXISTING, FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING);
        let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
        ov.hEvent = event.0;
        let ok = unsafe { ReadFileScatter(h_r.0, bad, SEG as u32, null_mut(), &mut ov) };
        let err = if ok == 0 { unsafe { GetLastError() } } else { 0 };
        assert_eq!(ok, 0, "ReadFileScatter must reject unreadable segment array pointer {bad:p}");
        assert_ne!(err, 0);

        let out = dir.0.join("out.bin");
        let h_w = open(&out, GENERIC_WRITE, CREATE_ALWAYS, FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING);
        let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
        ov.hEvent = event.0;
        let ok = unsafe { WriteFileGather(h_w.0, bad, SEG as u32, null_mut(), &mut ov) };
        let err = if ok == 0 { unsafe { GetLastError() } } else { 0 };
        assert_eq!(ok, 0, "WriteFileGather must reject unreadable segment array pointer {bad:p}");
        assert_ne!(err, 0);
    }

    unsafe { VirtualFree(noaccess, 0, MEM_RELEASE) };
}
