//! xcl.exe opts out of high-entropy ASLR, so the title's memory stays below 2^40 (#585),
//! and validates essential PE link-time characteristics.

/// IMAGE_FILE_LARGE_ADDRESS_AWARE (Characteristics in COFF file header)
const LARGE_ADDRESS_AWARE: u16 = 0x0020;
/// IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA
const HIGH_ENTROPY_VA: u16 = 0x0020;
/// IMAGE_SUBSYSTEM_WINDOWS_CUI (Console application)
const SUBSYSTEM_WINDOWS_CUI: u16 = 3;

#[test]
fn xcl_exe_is_linked_without_high_entropy_va() {
    let image = std::fs::read(env!("CARGO_BIN_EXE_xcl")).unwrap();
    let pe = u32::from_le_bytes(image[0x3c..0x40].try_into().unwrap()) as usize;
    assert_eq!(&image[pe..pe + 4], b"PE\0\0");
    let optional = pe + 24;
    assert_eq!(u16::from_le_bytes(image[optional..optional + 2].try_into().unwrap()), 0x20b, "PE32+");
    // DllCharacteristics is at +70 of the PE32+ optional header
    let flags = u16::from_le_bytes(image[optional + 70..optional + 72].try_into().unwrap());
    assert_eq!(flags & HIGH_ENTROPY_VA, 0, "DllCharacteristics {flags:#x}: high-entropy VA is on");
}

#[test]
fn xcl_exe_is_linked_with_large_address_aware_and_cui_subsystem() {
    let image = std::fs::read(env!("CARGO_BIN_EXE_xcl")).unwrap();
    let pe = u32::from_le_bytes(image[0x3c..0x40].try_into().unwrap()) as usize;
    assert_eq!(&image[pe..pe + 4], b"PE\0\0");

    // Machine type is at pe + 4 (IMAGE_FILE_MACHINE_AMD64 = 0x8664)
    let machine = u16::from_le_bytes(image[pe + 4..pe + 6].try_into().unwrap());
    assert_eq!(machine, 0x8664, "expected x64 (AMD64) machine header");

    // Characteristics is at pe + 22
    let characteristics = u16::from_le_bytes(image[pe + 22..pe + 24].try_into().unwrap());
    assert_ne!(
        characteristics & LARGE_ADDRESS_AWARE,
        0,
        "Characteristics {characteristics:#x}: missing IMAGE_FILE_LARGE_ADDRESS_AWARE"
    );

    let optional = pe + 24;
    // Subsystem is at optional + 68
    let subsystem = u16::from_le_bytes(image[optional + 68..optional + 70].try_into().unwrap());
    assert_eq!(
        subsystem, SUBSYSTEM_WINDOWS_CUI,
        "Subsystem {subsystem}: expected Windows CUI (console)"
    );
}

#[test]
fn xcl_exe_contains_valid_tls_and_exception_directories() {
    let image = std::fs::read(env!("CARGO_BIN_EXE_xcl")).unwrap();
    let parsed = xcl::pe::Pe::parse(&image, xcl::pe::Layout::File).expect("parse xcl.exe as PE");

    assert_eq!(parsed.machine, xcl::pe::IMAGE_FILE_MACHINE_AMD64);
    assert!(!parsed.is_dll());
    assert_eq!(parsed.subsystem, SUBSYSTEM_WINDOWS_CUI);

    // TLS directory is required for host TLS hook callbacks (.CRT$XLY)
    let tls_dir = parsed.dir(xcl::pe::DIR_TLS);
    assert!(
        tls_dir.size > 0 && tls_dir.rva > 0,
        "xcl.exe must have a non-empty TLS data directory for TLS callbacks"
    );

    // Exception directory (.pdata) is required for x64 SEH unwinding
    let ex_dir = parsed.dir(xcl::pe::DIR_EXCEPTION);
    assert!(
        ex_dir.size > 0 && ex_dir.rva > 0,
        "xcl.exe must have an exception directory for x64 stack unwinding"
    );
}
