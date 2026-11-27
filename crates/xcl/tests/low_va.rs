//! `xcl.exe` and the crate's integration tests opt out of high-entropy ASLR, so the title's
//! memory stays below 2^40 (#585, #2263).

use std::path::Path;
use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};

/// IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA
const HIGH_ENTROPY_VA: u16 = 0x20;
/// Upper bound of the low-VA address space required by console BVH addresses (#585).
const LOW_VA_CEILING: usize = 1usize << 40;

fn assert_no_high_entropy_va(path: &Path) {
    let image = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let pe = u32::from_le_bytes(image[0x3c..0x40].try_into().unwrap()) as usize;
    assert_eq!(&image[pe..pe + 4], b"PE\0\0", "{}: not a PE image", path.display());
    let optional = pe + 24;
    assert_eq!(u16::from_le_bytes(image[optional..optional + 2].try_into().unwrap()), 0x20b, "{}: expected PE32+", path.display());
    // DllCharacteristics is at +70 of the PE32+ optional header
    let flags = u16::from_le_bytes(image[optional + 70..optional + 72].try_into().unwrap());
    assert_eq!(flags & HIGH_ENTROPY_VA, 0, "{}: DllCharacteristics {flags:#x}: high-entropy VA is on", path.display());
}

#[test]
fn xcl_exe_is_linked_without_high_entropy_va() {
    assert_no_high_entropy_va(Path::new(env!("CARGO_BIN_EXE_xcl")));
}

#[test]
fn integration_tests_are_linked_without_high_entropy_va() {
    let exe = std::env::current_exe().unwrap();
    assert_no_high_entropy_va(&exe);
    let addr = unsafe {
        let p = VirtualAlloc(std::ptr::null(), 64 * 1024, MEM_RESERVE, PAGE_READWRITE);
        assert!(!p.is_null(), "VirtualAlloc(NULL, 64 KiB, MEM_RESERVE) failed");
        let addr = p as usize;
        assert_ne!(VirtualFree(p, 0, MEM_RELEASE), 0, "VirtualFree({addr:#x}) failed");
        addr
    };
    assert!(addr < LOW_VA_CEILING, "VirtualAlloc(NULL) landed at {addr:#x} (>= 2^40)");
}
