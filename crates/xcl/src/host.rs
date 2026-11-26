//! The host process: started by the launcher with the game's command line. Sets up
//! logging, the VFS, hooks and HLE, maps the title and jumps to its entry point.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::Boot;

pub static HOST_MODE: AtomicBool = AtomicBool::new(false);

pub fn host_main(boot_path: &Path) -> i32 {
    let boot: Boot = match std::fs::read_to_string(boot_path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string())) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("xcl host: cannot read {}: {e}", boot_path.display());
            return 0x7C1_0001;
        }
    };
    crate::log::init(
        Some(&boot.logs_dir().join("xcl.log")),
        crate::log::Level::parse(&boot.config.log_level),
        boot.config.log_to_stderr,
    );
    crate::log::log_panics();
    crate::xlog_info!("XCL {} – host for '{}'", env!("CARGO_PKG_VERSION"), boot.title);
    crate::xlog_info!("game: {} (read-only folder: {})", boot.exe_path.display(), boot.game_root.display());
    crate::xlog_info!("XCL data: {}", boot.data_dir.display());
    crate::xlog_info!(
        "console: {} (title memory {} MiB, mode {})",
        if boot.config.is_series_s() { "Xbox Series S (lockhart)" } else { "Xbox Series X (anaconda)" },
        boot.config.title_memory_mb,
        boot.config.title_memory
    );

    if std::env::var_os("XCL_UMA_MB").is_none() && boot.config.uma_mb != 0 {
        std::env::set_var("XCL_UMA_MB", boot.config.uma_mb.to_string());
    }
    crate::state::set_boot(boot.clone());
    let _ = std::thread::Builder::new().name("xcl-split-counts".into()).spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_secs(10));
        crate::hle::d3d12x::real::log_dispatch_chain_counts();
    });
    // `XCL_ACTIVITY=1`: every 2 s what the title did since (a timeline of where it stops)
    if std::env::var_os("XCL_ACTIVITY").is_some() {
        let _ = std::thread::Builder::new().name("xcl-activity".into()).spawn(|| {
            let snap = || {
                let (p, e) = crate::hle::d3d12x::present::activity();
                let (d, x) = crate::hle::d3d12x::bind::activity();
                let f = crate::hle::d3d12x::frame::activity();
                let (r, b, fail) = crate::hle::dstorage::counters();
                [p, e, d, x, f[0], f[1], f[2], f[3], r, b >> 20, fail]
            };
            let mut last = snap();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let now = snap();
                let d: Vec<u64> = now.iter().zip(last).map(|(a, b)| a - b).collect();
                crate::xlog_info!(
                    "activity (2 s): {} presents, {} submits, {} draws, {} dispatches, {} frame-event waits ({} timed out), {} frame statistics ({} not flipped); DirectStorage {} requests, {} MB, {} failed",
                    d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10]
                );
                // where the memory is: the process's commit, video memory, and XCL's parts of it
                let private = unsafe {
                    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
                    let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
                    c.cb = std::mem::size_of_val(&c) as u32;
                    K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut c as *mut _ as *mut _, c.cb);
                    c.PrivateUsage
                };
                let (local, non_local) = crate::hle::d3d12x::real::video_memory().unwrap_or((0, 0));
                let (arenas, arenas_made) = crate::hle::d3d12x::vram::arena_bytes();
                let (pools, pools_mapped) = crate::hle::d3d12x::sparse::pool_bytes();
                let (pages, page_pools) = crate::hle::d3d12x::vram::page_bytes();
                let mb = |b: u64| b >> 20;
                crate::xlog_info!(
                    "memory: {} MB committed by the process; video memory {} MB local, {} MB non-local; target pages {} MB (pools {} MB); arenas {} MB in use, {} MB made; sparse pools {} MB ({} MB mapped)",
                    mb(private as u64),
                    mb(local),
                    mb(non_local),
                    mb(pages as u64),
                    mb(page_pools as u64),
                    mb(arenas as u64),
                    mb(arenas_made as u64),
                    mb(pools as u64),
                    mb(pools_mapped as u64)
                );
                last = now;
            }
        });
    }
    // `XCL_THREAD_DUMP_AT=s1,s2…`: where every thread is at those seconds (a stalled title)
    if let Ok(v) = std::env::var("XCL_THREAD_DUMP_AT") {
        let at: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let _ = std::thread::Builder::new().name("xcl-thread-dump".into()).spawn(move || {
            let t0 = std::time::Instant::now();
            for s in at {
                let left = s - t0.elapsed().as_secs_f64();
                if left > 0.0 {
                    std::thread::sleep(std::time::Duration::from_secs_f64(left));
                }
                crate::eh::dump_threads();
            }
        });
    }
    // `XCL_STALL_DUMP=n`: once the title has presented frames and then presents none for n
    // seconds (a stalled frame loop), where every thread is goes to the log – twice, a second
    // apart, to tell stuck threads from moving ones – and the title ends as for
    // `XCL_RUN_SECONDS`: a stall diagnosed without waiting for the title's own watchdog.
    if let Some(secs) = stall_seconds(std::env::var("XCL_STALL_DUMP").ok().as_deref()) {
        let _ = std::thread::Builder::new().name("xcl-stall-dump".into()).spawn(move || {
            let mut last = crate::hle::d3d12x::present::presents();
            let mut since = std::time::Instant::now();
            loop {
                std::thread::sleep(std::time::Duration::from_millis(250));
                let now = crate::hle::d3d12x::present::presents();
                if now != last {
                    (last, since) = (now, std::time::Instant::now());
                    continue;
                }
                if !stalled(last, since.elapsed(), secs) {
                    continue;
                }
                crate::xlog_warn!("XCL_STALL_DUMP: no frame presented for {secs} s after frame {last} – where every thread is, then the end");
                crate::eh::dump_threads();
                std::thread::sleep(std::time::Duration::from_secs(1));
                crate::eh::dump_threads();
                crate::report::write_final(None);
                crate::log::flush();
                crate::hle::d3d12x::real::end_title(0);
            }
        });
    }
    // `XCL_RUN_SECONDS=n`: the title ends itself after n seconds – the GPU drained, the log and
    // report written (a process killed from outside with GPU work in flight can hang in the driver)
    if let Some(secs) = std::env::var("XCL_RUN_SECONDS").ok().and_then(|v| v.parse::<u64>().ok()) {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            let (shown, presented) = crate::hle::d3d12x::present::counters();
            crate::xlog_info!("XCL_RUN_SECONDS: {secs} s – ending the title ({shown} frames shown, {presented} PresentX calls)");
            crate::report::write_final(None);
            crate::log::flush();
            crate::hle::d3d12x::real::end_title(0);
        });
    }
    crate::stubs::init_db(&boot.config.stub_returns);
    crate::hle::init(&boot.config);
    crate::hle::kernelx::fine_timer();
    crate::loader::set_cache_dir(boot.cache_dir());
    let mut dirs = vec![boot.game_root.clone()];
    if let Some(p) = boot.exe_path.parent() {
        if p != boot.game_root {
            dirs.insert(0, p.to_path_buf());
        }
    }
    crate::resolver::set_search_dirs(dirs);

    let mounts: Vec<(String, String)> = boot.config.mounts.iter().map(|(k, v)| (k.clone(), boot.expand(v))).collect();
    // (the shader cache is XCL's too: on a drive a console mount names it must stay where it is)
    crate::vfs::init(crate::vfs::Vfs::new(&boot.game_root, &boot.data_dir, mounts).own_folder(&crate::gpu::recomp::dxc::cache_dir()));
    let _ = std::fs::create_dir_all(boot.data_dir.join("temp"));

    if let Err(e) = crate::vfs_hooks::install() {
        crate::xlog_error!("VFS: not all hooks installed: {e}");
    }
    if let Err(e) = crate::eh::install_pc_hook() {
        crate::xlog_error!("hook RtlPcToFileHeader: {e} – C++ exceptions in the game may not work");
    }
    crate::eh::install_crash_handler();
    crate::eh::start_probes();
    HOST_MODE.store(true, Ordering::Release);

    let main = match crate::loader::load_main(&boot.exe_path) {
        Ok(m) => m,
        Err(e) => {
            crate::xlog_error!("loading the game failed: {e}");
            crate::report::write_final(Some(format!("loader: {e}")));
            return 0x7C1_0002;
        }
    };
    unsafe { crate::hle::kernelx::adopt_title_hooks(&main) };
    crate::report::write_final(None);
    let code = crate::loader::run_main(&main);
    crate::xlog_info!("game entry point returned code {code}");
    crate::report::write_final(None);
    crate::log::flush();
    code as i32
}

/// `XCL_STALL_DUMP`'s value: seconds without a frame, if it is a positive number.
fn stall_seconds(value: Option<&str>) -> Option<f64> {
    value.and_then(|v| v.trim().parse::<f64>().ok()).filter(|s| s.is_finite() && *s > 0.0)
}

/// Whether the title has stalled for `XCL_STALL_DUMP`: it has presented a frame (`last` is the
/// count seen) and no other for `secs`. A title still at its first frame has not.
fn stalled(last: u64, idle: std::time::Duration, secs: f64) -> bool {
    last > 0 && idle.as_secs_f64() >= secs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn stall_dump_takes_a_positive_number_of_seconds() {
        assert_eq!(stall_seconds(Some("6")), Some(6.0));
        assert_eq!(stall_seconds(Some(" 2.5 ")), Some(2.5));
        for v in [None, Some(""), Some("0"), Some("-3"), Some("abc"), Some("inf"), Some("NaN")] {
            assert_eq!(stall_seconds(v), None, "{v:?}");
        }
    }

    #[test]
    fn a_stall_needs_a_presented_frame_and_the_full_wait() {
        assert!(!stalled(0, Duration::from_secs(600), 6.0), "no frame presented yet");
        assert!(!stalled(940, Duration::from_millis(5_999), 6.0));
        assert!(stalled(940, Duration::from_secs(6), 6.0));
        assert!(stalled(1, Duration::from_secs(60), 6.0));
    }
}
