//! Exercise the native A/W hooks without loading a title or initializing graphics.

use std::fs;
use std::path::PathBuf;
use std::ptr::{null, null_mut};
use std::time::{SystemTime, UNIX_EPOCH};
use windows_sys::Win32::Foundation::{
    GetLastError, SetLastError, ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND,
};
use windows_sys::Win32::Globalization::GetOEMCP;
use windows_sys::Win32::Storage::FileSystem::{
    AreFileApisANSI, GetDiskFreeSpaceA, GetDiskFreeSpaceExA, GetDiskFreeSpaceExW,
    GetDiskFreeSpaceW, GetFileAttributesA, GetFileAttributesExW, GetFileAttributesW,
    GetFileExInfoStandard, SetFileApisToANSI, SetFileApisToOEM, INVALID_FILE_ATTRIBUTES,
    WIN32_FILE_ATTRIBUTE_DATA,
};
use xcl::config::{Boot, Config};
use xcl::util::wide;

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn disk_free_space_ex_reports_pls_quota() {
    const MIB: u64 = 1 << 20;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("xcl_pls_quota_{}_{nonce}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let root = TempDir(dir);
    let game = root.0.join("game");
    let data = root.0.join("data");
    let pls = data.join("pls");
    fs::create_dir_all(&game).unwrap();
    fs::create_dir_all(pls.join("世界")).unwrap();
    fs::create_dir(pls.join("café")).unwrap();
    fs::create_dir(data.join("pls-other")).unwrap();
    fs::write(pls.join("existing.bin"), [0; 123]).unwrap();
    fs::write(pls.join("世界").join("nested.bin"), [0; 456]).unwrap();
    fs::write(
        game.join("MicrosoftGame.Config"),
        "<Game><PersistentLocalStorage><SizeMB>1</SizeMB></PersistentLocalStorage></Game>",
    )
    .unwrap();

    let mut config = Config::default();
    config.resolve(xcl::manifest::load(&game).as_ref());
    assert_eq!(config.pls_mb, 1);
    config.mounts.insert("P:".into(), "{data}\\pls".into());
    config
        .mounts
        .insert("T:".into(), "{data}\\pls-other".into());
    let boot = Boot {
        title: "PLS test".into(),
        exe_path: game.join("unused.exe"),
        game_root: game,
        data_dir: data,
        config,
    };
    let mounts = boot
        .config
        .mounts
        .iter()
        .map(|(k, v)| (k.clone(), boot.expand(v)))
        .collect();
    xcl::vfs::init(xcl::vfs::Vfs::new(&boot.game_root, &boot.data_dir, mounts));
    xcl::state::set_boot(boot);
    xcl::vfs_hooks::install().unwrap();

    let host = wide(&pls.to_string_lossy());
    let (mut available, mut total, mut free) = (0, 0, 0);
    unsafe {
        SetLastError(9123);
        let result = GetDiskFreeSpaceExW(host.as_ptr(), &mut available, &mut total, &mut free);
        let host_last_error = GetLastError();
        assert_ne!(result, 0);
        let host_total = total;
        assert!(
            available >= MIB && free >= MIB,
            "test requires 1 MiB of host space"
        );

        let world = wide(r"s:/世界");
        SetLastError(9123);
        let result = GetDiskFreeSpaceExW(world.as_ptr(), &mut available, &mut total, &mut free);
        let error = GetLastError();
        assert_ne!(result, 0);
        assert_eq!(error, host_last_error);
        assert_eq!((available, total, free), (MIB - 579, MIB, MIB - 579));

        let drive = b"S:\\\0";
        let (mut ansi_available, mut ansi_free) = (0, 0);
        assert_ne!(
            GetDiskFreeSpaceExA(
                drive.as_ptr(),
                &mut ansi_available,
                null_mut(),
                &mut ansi_free
            ),
            0
        );
        assert_eq!((ansi_available, ansi_free), (available, free));

        if GetOEMCP() == 437 {
            let was_ansi = AreFileApisANSI() != 0;
            SetFileApisToOEM();
            let result = GetDiskFreeSpaceExA(
                c"S:\\caf\x82".as_ptr().cast(),
                &mut ansi_available,
                null_mut(),
                null_mut(),
            );
            if was_ansi {
                SetFileApisToANSI();
            }
            assert_ne!(result, 0);
            assert_eq!(ansi_available, available);
        }

        let alias = wide(r"P:\");
        assert_ne!(
            GetDiskFreeSpaceExW(alias.as_ptr(), null_mut(), &mut total, null_mut()),
            0
        );
        assert_eq!(total, MIB);
        for path in [&host, &wide(r"T:\"), &wide(r"S:\..\pls-other")] {
            assert_ne!(
                GetDiskFreeSpaceExW(path.as_ptr(), null_mut(), &mut total, null_mut()),
                0
            );
            assert_eq!(total, host_total);
        }
        assert_ne!(
            GetDiskFreeSpaceExW(null(), null_mut(), null_mut(), null_mut()),
            0
        );

        let missing = wide(r"S:\missing");
        SetLastError(0);
        let result = GetDiskFreeSpaceExW(missing.as_ptr(), &mut available, &mut total, &mut free);
        let error = GetLastError();
        assert_eq!((result, error), (0, ERROR_PATH_NOT_FOUND));
        assert_eq!((available, total, free), (MIB - 579, host_total, MIB - 579));

        // Test GetDiskFreeSpaceW and GetDiskFreeSpaceA on mounted alias P:\ and host directory
        let (mut spc, mut bps, mut free_clusters, mut total_clusters) = (0u32, 0u32, 0u32, 0u32);
        let res_w = GetDiskFreeSpaceW(
            alias.as_ptr(),
            &mut spc,
            &mut bps,
            &mut free_clusters,
            &mut total_clusters,
        );
        assert_ne!(res_w, 0, "GetDiskFreeSpaceW on alias failed");
        assert!(spc > 0 && bps > 0 && total_clusters > 0);

        let (mut ansi_spc, mut ansi_bps, mut ansi_free_clusters, mut ansi_total_clusters) =
            (0u32, 0u32, 0u32, 0u32);
        let res_a = GetDiskFreeSpaceA(
            b"P:\\\0".as_ptr(),
            &mut ansi_spc,
            &mut ansi_bps,
            &mut ansi_free_clusters,
            &mut ansi_total_clusters,
        );
        assert_ne!(res_a, 0, "GetDiskFreeSpaceA on alias failed");
        assert_eq!((ansi_spc, ansi_bps), (spc, bps));

        // Test GetFileAttributesW and GetFileAttributesA on existing and missing files
        let existing = wide(r"P:\existing.bin");
        let attrs_w = GetFileAttributesW(existing.as_ptr());
        assert_ne!(attrs_w, INVALID_FILE_ATTRIBUTES);

        let attrs_a = GetFileAttributesA(b"P:\\existing.bin\0".as_ptr());
        assert_ne!(attrs_a, INVALID_FILE_ATTRIBUTES);
        assert_eq!(attrs_w, attrs_a);

        let nested = wide(r"s:/世界/nested.bin");
        let attrs_nested = GetFileAttributesW(nested.as_ptr());
        assert_ne!(attrs_nested, INVALID_FILE_ATTRIBUTES);

        let missing_file = wide(r"P:\missing.bin");
        SetLastError(0);
        let missing_attrs = GetFileAttributesW(missing_file.as_ptr());
        let err = GetLastError();
        assert_eq!(missing_attrs, INVALID_FILE_ATTRIBUTES);
        assert_eq!(err, ERROR_FILE_NOT_FOUND);

        // Test GetFileAttributesExW on existing files (existing.bin = 123 B, nested.bin = 456 B)
        let mut info: WIN32_FILE_ATTRIBUTE_DATA = std::mem::zeroed();
        let res_ex = GetFileAttributesExW(
            existing.as_ptr(),
            GetFileExInfoStandard,
            &mut info as *mut _ as *mut std::ffi::c_void,
        );
        assert_ne!(res_ex, 0, "GetFileAttributesExW on existing.bin failed");
        assert_eq!(info.nFileSizeLow, 123);
        assert_eq!(info.nFileSizeHigh, 0);

        let mut nested_info: WIN32_FILE_ATTRIBUTE_DATA = std::mem::zeroed();
        let res_nested_ex = GetFileAttributesExW(
            nested.as_ptr(),
            GetFileExInfoStandard,
            &mut nested_info as *mut _ as *mut std::ffi::c_void,
        );
        assert_ne!(res_nested_ex, 0, "GetFileAttributesExW on nested.bin failed");
        assert_eq!(nested_info.nFileSizeLow, 456);
        assert_eq!(nested_info.nFileSizeHigh, 0);

        // Test GetFileAttributesExW on missing file
        let mut missing_info: WIN32_FILE_ATTRIBUTE_DATA = std::mem::zeroed();
        SetLastError(0);
        let res_missing_ex = GetFileAttributesExW(
            missing_file.as_ptr(),
            GetFileExInfoStandard,
            &mut missing_info as *mut _ as *mut std::ffi::c_void,
        );
        let ex_err = GetLastError();
        assert_eq!(res_missing_ex, 0);
        assert_eq!(ex_err, ERROR_FILE_NOT_FOUND);

        fs::File::create(pls.join("fills-quota.bin"))
            .unwrap()
            .set_len(MIB)
            .unwrap();
        assert_ne!(
            GetDiskFreeSpaceExA(drive.as_ptr(), &mut available, &mut total, &mut free),
            0
        );
        assert_eq!((available, total, free), (0, MIB, 0));
        assert_ne!(
            GetDiskFreeSpaceExA(drive.as_ptr(), null_mut(), null_mut(), null_mut()),
            0
        );
    }
}
