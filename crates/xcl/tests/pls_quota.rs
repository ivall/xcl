//! Exercise the native A/W hooks without loading a title or initializing graphics.

use std::fs;
use std::path::PathBuf;
use std::ptr::{null, null_mut};
use std::time::{SystemTime, UNIX_EPOCH};
use windows_sys::Win32::Foundation::{GetLastError, SetLastError, ERROR_PATH_NOT_FOUND};
use windows_sys::Win32::Globalization::GetOEMCP;
use windows_sys::Win32::Storage::FileSystem::{
    AreFileApisANSI, GetDiskFreeSpaceA, GetDiskFreeSpaceExA, GetDiskFreeSpaceExW, GetDiskFreeSpaceW,
    SetFileApisToANSI, SetFileApisToOEM,
};
use xcl::config::{Boot, Config};
use xcl::util::wide;

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

unsafe fn query_clusters(query: impl FnOnce(*mut u32, *mut u32, *mut u32, *mut u32) -> i32) -> (i32, u32, [u32; 4]) {
    let (mut sectors, mut bytes, mut free, mut total) = (17, 19, 23, 29);
    SetLastError(9123);
    let ok = query(&mut sectors, &mut bytes, &mut free, &mut total);
    (ok, GetLastError(), [sectors, bytes, free, total])
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

        let legacy_w = |path: &[u16]| {
            query_clusters(|spc, bps, free, total| GetDiskFreeSpaceW(path.as_ptr(), spc, bps, free, total))
        };
        let legacy_a = |path: &[u8]| {
            query_clusters(|spc, bps, free, total| GetDiskFreeSpaceA(path.as_ptr(), spc, bps, free, total))
        };
        let (ok, native_error, [sectors, bytes, host_free, host_clusters]) = legacy_w(&host);
        assert_ne!(ok, 0);
        let cluster = u64::from(sectors) * u64::from(bytes);
        assert!(cluster != 0 && u64::from(host_free) * cluster >= MIB);
        let expected_total = u64::from(host_clusters).min(MIB / cluster) as u32;
        let expected_free = u64::from(host_free).min((MIB - 579) / cluster) as u32;
        let expected = (ok, native_error, [sectors, bytes, expected_free, expected_total]);
        for (t_ok, _, [t_sectors, t_bytes, t_free, t_total]) in [legacy_w(&wide(r"T:\")), legacy_a(b"T:\\\0")] {
            assert_ne!(t_ok, 0);
            assert_eq!([t_sectors, t_bytes, t_total], [sectors, bytes, host_clusters]);
            assert!(u64::from(t_free) * cluster >= MIB);
        }

        let host_missing = wide(&pls.join("missing").to_string_lossy());
        let native_failure = legacy_w(&host_missing);
        assert_eq!(native_failure.0, 0);
        assert_eq!(legacy_w(&missing), native_failure);
        assert_eq!(legacy_a(b"S:\\missing\0"), native_failure);
        assert_eq!(
            [legacy_w(&alias), legacy_a(drive)],
            [expected; 2],
            "legacy PLS quota before fill (wide, ANSI)"
        );

        fs::File::create(pls.join("fills-quota.bin"))
            .unwrap()
            .set_len(MIB)
            .unwrap();
        assert_ne!(
            GetDiskFreeSpaceExA(drive.as_ptr(), &mut available, &mut total, &mut free),
            0
        );
        assert_eq!((available, total, free), (0, MIB, 0));
        assert_eq!(
            [legacy_w(&alias), legacy_a(drive)],
            [(ok, native_error, [sectors, bytes, 0, expected_total]); 2],
            "legacy PLS quota at capacity (wide, ANSI)"
        );
        for (outside_ok, _, [outside_sectors, outside_bytes, outside_free, outside_total]) in
            [legacy_w(&host), legacy_a(b"T:\\\0")]
        {
            assert_ne!(outside_ok, 0);
            assert_eq!([outside_sectors, outside_bytes, outside_total], [sectors, bytes, host_clusters]);
            assert!(outside_free > 0);
        }
        assert_ne!(
            GetDiskFreeSpaceExA(drive.as_ptr(), null_mut(), null_mut(), null_mut()),
            0
        );
    }
}
