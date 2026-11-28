//! `xcl.exe` carries the manifest that makes the segment heap its process heap (`xcl.manifest`,
//! embedded by `build.rs` on MSVC).

#[cfg(target_env = "msvc")]
#[test]
fn xcl_exe_asks_for_the_segment_heap() {
    use windows_sys::Win32::System::LibraryLoader::{FindResourceW, LoadLibraryExW, LOAD_LIBRARY_AS_DATAFILE};
    let exe: Vec<u16> = env!("CARGO_BIN_EXE_xcl").encode_utf16().chain([0]).collect();
    let module = unsafe { LoadLibraryExW(exe.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_AS_DATAFILE) };
    assert!(!module.is_null(), "xcl.exe cannot be loaded as a data file");
    // RT_MANIFEST (24), CREATEPROCESS_MANIFEST_RESOURCE_ID (1): the manifest Windows reads at start
    let found = unsafe { FindResourceW(module, 1 as _, 24 as _) };
    assert!(!found.is_null(), "xcl.exe has no process manifest");
    let bytes = std::fs::read(env!("CARGO_BIN_EXE_xcl")).expect("read xcl.exe");
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("<heapType") && text.contains(">SegmentHeap</heapType>"), "the manifest does not ask for the segment heap");
}
