use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use xcl::modules::{
    describe_address, find_hle_handle, find_mapped_by_addr, find_mapped_by_base,
    find_mapped_by_name, hle_handle, hle_name_by_handle, image_base_for_addr, main_module,
    mapped_modules, register_mapped, set_main, unregister_mapped, MappedModule,
};
use xcl::pe::OwnedExports;

/// Serializes tests that inspect process-wide global module/main state.
static GLOBAL_LOCK: Mutex<()> = Mutex::new(());

fn make_module(name: &str, base: usize, size: usize, is_dll: bool) -> Arc<MappedModule> {
    Arc::new(MappedModule {
        name: name.to_string(),
        path: PathBuf::from(format!("C:\\game\\{name}")),
        base,
        size,
        entry_rva: 0x1000,
        is_dll,
        exports: OwnedExports::default(),
        tls_callbacks: Mutex::new(Vec::new()),
        no_thread_calls: AtomicBool::new(false),
        initialized: AtomicBool::new(false),
        stack_reserve: 0x10_0000,
    })
}

#[test]
fn mapped_module_contains_is_half_open() {
    let m = make_module("bounds.dll", 0x1000_0000, 0x2000, true);
    assert!(!m.contains(0x0fff_ffff));
    assert!(m.contains(0x1000_0000));
    assert!(m.contains(0x1000_1fff));
    assert!(!m.contains(0x1000_2000));

    let empty = make_module("empty.dll", 0x2000_0000, 0, true);
    assert!(!empty.contains(0x2000_0000));
}

#[test]
fn register_lookup_and_unregister_mapped_module() {
    let _guard = GLOBAL_LOCK.lock().unwrap();

    let base = 0x5100_0000_0000usize;
    let size = 0x4000usize;
    let m = make_module("alpha_mod.dll", base, size, true);
    register_mapped(Arc::clone(&m));

    assert!(mapped_modules().iter().any(|entry| entry.base == base));

    // Case-insensitive and ".dll" normalization via util::dll_key.
    assert_eq!(find_mapped_by_name("alpha_mod.dll").unwrap().base, base);
    assert_eq!(find_mapped_by_name("ALPHA_MOD.DLL").unwrap().base, base);
    assert_eq!(find_mapped_by_name("alpha_mod").unwrap().base, base);
    assert_eq!(find_mapped_by_name("C:\\Xbox\\ALPHA_MOD").unwrap().base, base);
    assert!(find_mapped_by_name("missing_mod.dll").is_none());

    assert_eq!(find_mapped_by_base(base).unwrap().name, "alpha_mod.dll");
    assert!(find_mapped_by_base(base + 0x100).is_none());

    assert_eq!(find_mapped_by_addr(base).unwrap().base, base);
    assert_eq!(find_mapped_by_addr(base + size - 1).unwrap().base, base);
    assert!(find_mapped_by_addr(base - 1).is_none());
    assert!(find_mapped_by_addr(base + size).is_none());

    // Lock-free RANGE_BASE/RANGE_SIZE lookup.
    assert_eq!(image_base_for_addr(base), Some(base));
    assert_eq!(image_base_for_addr(base + 0x123), Some(base));
    assert_eq!(image_base_for_addr(base + size - 1), Some(base));
    assert_eq!(image_base_for_addr(base - 1), None);
    assert_eq!(image_base_for_addr(base + size), None);

    assert_eq!(describe_address(base + 0x40), "alpha_mod.dll+0x40");

    // Unregister clears both MAPPED and the lock-free range table.
    unregister_mapped(base);
    assert!(!mapped_modules().iter().any(|entry| entry.base == base));
    assert!(find_mapped_by_name("alpha_mod.dll").is_none());
    assert!(find_mapped_by_base(base).is_none());
    assert!(find_mapped_by_addr(base + 0x40).is_none());
    assert_eq!(image_base_for_addr(base + 0x40), None);
    assert_eq!(describe_address(base + 0x40), format!("{:#x}", base + 0x40));

    // Re-registering at the same base after unregister succeeds.
    let m2 = make_module("alpha_reloaded.dll", base, size, true);
    register_mapped(m2);
    assert_eq!(image_base_for_addr(base + 0x10), Some(base));
    assert_eq!(find_mapped_by_name("alpha_reloaded.dll").unwrap().base, base);
    unregister_mapped(base);
}

#[test]
fn set_main_and_main_module_track_latest_main() {
    let _guard = GLOBAL_LOCK.lock().unwrap();

    assert!(main_module().is_none());

    let first = make_module("first_game.exe", 0x5200_0000_0000, 0x1000, false);
    set_main(Arc::clone(&first));
    assert_eq!(main_module().unwrap().name, "first_game.exe");

    let second = make_module("second_game.exe", 0x5300_0000_0000, 0x1000, false);
    set_main(Arc::clone(&second));
    assert_eq!(main_module().unwrap().name, "second_game.exe");
}

#[test]
fn hle_handle_allocates_mz_page_and_is_idempotent() {
    assert!(find_hle_handle("xcl_test_hle_alpha.dll").is_none());

    let h1 = hle_handle("xcl_test_hle_alpha.dll");
    assert_ne!(h1, 0);

    let magic = unsafe { std::slice::from_raw_parts(h1 as *const u8, 2) };
    assert_eq!(magic, b"MZ");

    assert_eq!(hle_handle("XCL_TEST_HLE_ALPHA.DLL"), h1);
    assert_eq!(hle_handle("xcl_test_hle_alpha"), h1);
    assert_eq!(find_hle_handle("XCL_TEST_HLE_ALPHA"), Some(h1));
    assert_eq!(
        hle_name_by_handle(h1).as_deref(),
        Some("xcl_test_hle_alpha.dll")
    );
    assert!(hle_name_by_handle(0xdead_beef).is_none());
}

#[test]
fn describe_address_formats_mapped_and_host_modules() {
    let _guard = GLOBAL_LOCK.lock().unwrap();

    let base = 0x5400_0000_0000usize;
    let m = make_module("gamma_mod.dll", base, 0x2000, true);
    register_mapped(m);

    assert_eq!(describe_address(base), "gamma_mod.dll+0x0");
    assert_eq!(describe_address(base + 0x123), "gamma_mod.dll+0x123");
    unregister_mapped(base);

    assert_eq!(describe_address(0), "0x0");

    // Real host module (kernel32.dll) resolved via GetModuleHandleExW.
    unsafe {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        let k32 = GetModuleHandleA(c"kernel32.dll".as_ptr() as *const u8);
        assert!(!k32.is_null());
        let proc = GetProcAddress(k32, c"GetCurrentProcessId".as_ptr() as *const u8)
            .expect("GetCurrentProcessId must exist in kernel32.dll");
        let desc = describe_address(proc as usize);
        let lower = desc.to_ascii_lowercase();
        assert!(
            lower.starts_with("kernel32.dll+0x"),
            "unexpected describe_address output for kernel32 proc: {desc}"
        );
    }
}
