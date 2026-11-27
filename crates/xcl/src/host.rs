//! The host process: started by the launcher with the game's command line. Sets up
//! logging, the VFS, hooks and HLE, maps the title and jumps to its entry point.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::config::Boot;

pub static HOST_MODE: AtomicBool = AtomicBool::new(false);
static ACTIVE_COMMIT_CAP_MB: AtomicU64 = AtomicU64::new(0);

/// Active `XCL_TITLE_COMMIT_CAP_MB` limit in MiB (`0` when disabled or not applied).
pub fn active_commit_cap_mb() -> u64 {
    ACTIVE_COMMIT_CAP_MB.load(Ordering::Acquire)
}

/// Current process private commit in bytes (`PrivateUsage`, falling back to `PagefileUsage`
/// on Wine). Returns `0` if the query fails.
pub fn current_process_commit_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    unsafe {
        let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        c.cb = std::mem::size_of_val(&c) as u32;
        if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c as *mut _ as *mut _, c.cb) != 0 {
            if c.PrivateUsage != 0 { c.PrivateUsage as u64 } else { c.PagefileUsage as u64 }
        } else {
            0
        }
    }
}

/// Current process private commit in MiB.
pub fn current_process_commit_mb() -> u64 {
    current_process_commit_bytes() >> 20
}

pub fn format_commit_cap_oom(site: &str, cap_mb: u64, commit_mb: u64) -> String {
    format!(
        "{site}: allocation failed while {}={cap_mb} MB is active (process commit {commit_mb} MB)",
        crate::config::ENV_TITLE_COMMIT_CAP_MB
    )
}

/// When `XCL_TITLE_COMMIT_CAP_MB` is active, logs the cap and current process commit in MiB.
pub fn log_commit_cap_if_active(site: &str) {
    let cap_mb = active_commit_cap_mb();
    if cap_mb > 0 {
        let commit_mb = current_process_commit_mb();
        crate::xlog_warn!("{}", format_commit_cap_oom(site, cap_mb, commit_mb));
    }
}

pub fn host_main(boot_path: &Path) -> i32 {
    host_main_with(boot_path, true)
}

fn host_main_with(boot_path: &Path, probe_child: bool) -> i32 {
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
    crate::report::init_emergency_meta(&boot);
    crate::eh::install_alloc_error_hook();

    // Keep the job object handle alive for the lifetime of the host run.
    let _commit_cap = match crate::config::title_commit_cap_mb_from_env() {
        Ok(Some(cap_mb)) => match apply_commit_cap_mb_with(cap_mb, || check_commit_cap_enforcement(probe_child)) {
            Ok(job) => {
                crate::xlog_info!("process commit cap: {cap_mb} MB ({})", crate::config::ENV_TITLE_COMMIT_CAP_MB);
                Some(job)
            }
            Err(e) => {
                crate::xlog_warn!("process commit cap ({cap_mb} MB) not applied: {e}");
                None
            }
        },
        Ok(None) => None,
        Err(e) => {
            crate::xlog_warn!("{e}");
            None
        }
    };

    if std::env::var_os("XCL_UMA_MB").is_none() && boot.config.uma_mb != 0 {
        std::env::set_var("XCL_UMA_MB", boot.config.uma_mb.to_string());
    }
    crate::state::set_boot(boot.clone());
    if let Err(e) = crate::hle::d3d12x::present::preparation::before_guest(&boot) {
        crate::xlog_error!("Shader preparation: {e}");
        eprintln!("xcl: {e}");
        return 2;
    }
    #[cfg(test)]
    if std::env::var_os(tests::HOST_EARLY_OOM_BOOT_ENV).is_some() {
        tests::trigger_early_oom();
    }
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
                let private = current_process_commit_bytes();
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
    crate::hle::d3d12x::backend_before_guest();
    HOST_MODE.store(true, Ordering::Release);

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

/// Live Windows Job Object handle enforcing `XCL_TITLE_COMMIT_CAP_MB` for the host run.
pub struct CommitCapJob(windows_sys::Win32::Foundation::HANDLE);

unsafe impl Send for CommitCapJob {}
unsafe impl Sync for CommitCapJob {}

impl Drop for CommitCapJob {
    fn drop(&mut self) {
        ACTIVE_COMMIT_CAP_MB.store(0, Ordering::Release);
        if !self.0.is_null() {
            unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0) };
        }
    }
}

const UNSUPPORTED_COMMIT_CAP_MSG: &str =
    "host does not enforce Windows Job Object memory limits (JOB_OBJECT_LIMIT_PROCESS_MEMORY); \
     use Linux cgroup MemoryMax or an external free-RAM guard instead";

fn check_commit_cap_enforcement(probe_child: bool) -> Result<(), String> {
    if crate::util::under_wine() {
        return Err(format!("Wine {UNSUPPORTED_COMMIT_CAP_MSG}"));
    }
    if probe_child {
        probe_commit_cap_enforcement()?;
    }
    Ok(())
}

/// Decodes the probe child exit status (`Some(0)` is the only accepted enforcement proof).
fn interpret_commit_cap_probe_outcome(outcome: std::io::Result<Option<i32>>) -> Result<(), String> {
    let code = outcome.map_err(|e| format!("commit-cap probe: child launch failed: {e}"))?;
    match code {
        Some(0) => Ok(()),
        Some(2) => Err(format!(
            "Job Object assignment succeeded without enforcing VirtualAlloc commit limits; {UNSUPPORTED_COMMIT_CAP_MSG}"
        )),
        other => Err(format!(
            "commit-cap probe could not verify enforcement (exit {other:?}); {UNSUPPORTED_COMMIT_CAP_MSG}"
        )),
    }
}

/// Runs the commit-cap enforcement probe using injectable executable lookup and child runner.
fn probe_commit_cap_enforcement_with(
    current_exe: impl FnOnce() -> std::io::Result<std::path::PathBuf>,
    run_probe: impl FnOnce(&Path) -> std::io::Result<Option<i32>>,
) -> Result<(), String> {
    let exe = current_exe().map_err(|e| format!("commit-cap probe: current_exe failed: {e}"))?;
    if !exe
        .file_stem()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("xcl"))
    {
        return Err("commit-cap enforcement probe needs xcl.exe".into());
    }
    interpret_commit_cap_probe_outcome(run_probe(&exe))
}

fn probe_commit_cap_enforcement() -> Result<(), String> {
    probe_commit_cap_enforcement_with(std::env::current_exe, |exe| {
        std::process::Command::new(exe)
            .env("XCL_COMMIT_CAP_PROBE", "1")
            .env_remove("XCL_HOST")
            .env_remove(crate::config::ENV_TITLE_COMMIT_CAP_MB)
            .status()
            .map(|status| status.code())
    })
}

/// Child-process entrypoint (`XCL_COMMIT_CAP_PROBE=1`) that checks whether the OS kernel
/// enforces `JOB_OBJECT_LIMIT_PROCESS_MEMORY` on `VirtualAlloc`.
/// Returns `0` when enforced, `2` when Job Object setup succeeded without enforcement, or `3`
/// when Job Object setup itself could not be completed.
pub fn commit_cap_probe_main() -> i32 {
    use windows_sys::Win32::Foundation::{GetLastError, ERROR_COMMITMENT_LIMIT, ERROR_NOT_ENOUGH_MEMORY};
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE,
    };

    let cap_mb = current_process_commit_mb().saturating_add(16).max(64);
    let Some(alloc_mb) = cap_mb.checked_add(64) else {
        return 3;
    };
    let Ok(alloc_bytes) = alloc_mb
        .checked_mul(1024 * 1024)
        .and_then(|b| usize::try_from(b).ok())
        .ok_or(())
    else {
        return 3;
    };
    let Ok(_job) = apply_commit_cap_mb_unchecked(cap_mb) else {
        return 3;
    };
    unsafe {
        let ptr = VirtualAlloc(std::ptr::null(), alloc_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if ptr.is_null() {
            let err = GetLastError();
            if err == ERROR_NOT_ENOUGH_MEMORY || err == ERROR_COMMITMENT_LIMIT {
                return 0;
            }
            return 3;
        }
        VirtualFree(ptr, 0, MEM_RELEASE);
        2
    }
}

fn validate_commit_cap_bytes(cap_mb: u64) -> Result<usize, String> {
    if cap_mb == 0 {
        return Err("commit cap must be > 0 MiB".into());
    }
    cap_mb
        .checked_mul(1024 * 1024)
        .and_then(|b| usize::try_from(b).ok())
        .ok_or_else(|| format!("{}: {} MB overflows process commit limit", crate::config::ENV_TITLE_COMMIT_CAP_MB, cap_mb))
}

fn apply_commit_cap_mb_unchecked(cap_mb: u64) -> Result<CommitCapJob, String> {
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, IsProcessInJob, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let cap_bytes = validate_commit_cap_bytes(cap_mb)?;

    unsafe {
        let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if handle.is_null() {
            let err = GetLastError();
            return Err(format!("CreateJobObjectW failed: error {err} ({err:#x})"));
        }
        let job = CommitCapJob(handle);

        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.ProcessMemoryLimit = cap_bytes;
        if SetInformationJobObject(
            job.0,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            std::mem::size_of_val(&info) as u32,
        ) == 0
        {
            let err = GetLastError();
            return Err(format!(
                "SetInformationJobObject(JOB_OBJECT_LIMIT_PROCESS_MEMORY = {cap_mb} MB) failed: error {err} ({err:#x})"
            ));
        }

        if AssignProcessToJobObject(job.0, GetCurrentProcess()) == 0 {
            let err = GetLastError();
            let mut in_job = 0i32;
            let nested_note = if IsProcessInJob(GetCurrentProcess(), std::ptr::null_mut(), &mut in_job) != 0 && in_job != 0 {
                " (process is already in a job object that forbids nesting or breakaway)"
            } else {
                ""
            };
            return Err(format!(
                "AssignProcessToJobObject failed: error {err} ({err:#x}){nested_note}"
            ));
        }

        ACTIVE_COMMIT_CAP_MB.store(cap_mb, Ordering::Release);
        Ok(job)
    }
}

fn apply_commit_cap_mb_with(
    cap_mb: u64,
    check_enforcement: impl FnOnce() -> Result<(), String>,
) -> Result<CommitCapJob, String> {
    let _ = validate_commit_cap_bytes(cap_mb)?;
    check_enforcement()?;
    apply_commit_cap_mb_unchecked(cap_mb)
}

/// Applies a Windows Job Object commit cap (`JOB_OBJECT_LIMIT_PROCESS_MEMORY`) of `cap_mb` MiB
/// to the current process, failing closed if the host does not enforce Job Object limits.
pub fn apply_commit_cap_mb(cap_mb: u64) -> Result<CommitCapJob, String> {
    apply_commit_cap_mb_with(cap_mb, || check_commit_cap_enforcement(false))
}

/// Reads `XCL_TITLE_COMMIT_CAP_MB` and, when set to a positive MiB value, applies a Windows Job
/// Object commit cap (`JOB_OBJECT_LIMIT_PROCESS_MEMORY`) to the current process.
pub fn apply_commit_cap_from_env() -> Result<Option<CommitCapJob>, String> {
    match crate::config::title_commit_cap_mb_from_env()? {
        Some(cap_mb) => apply_commit_cap_mb(cap_mb).map(Some),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_and_overflow_commit_cap() {
        assert!(apply_commit_cap_mb(0).is_err());
        assert!(apply_commit_cap_mb(u64::MAX).is_err());
    }

    #[test]
    fn commit_cap_query_and_oom_formatting() {
        assert_eq!(active_commit_cap_mb(), 0);
        assert!(current_process_commit_mb() > 0);
        assert_eq!(
            format_commit_cap_oom("D3D12.X: title-texture page pool CreateHeap", 24_000, 23_980),
            "D3D12.X: title-texture page pool CreateHeap: allocation failed while XCL_TITLE_COMMIT_CAP_MB=24000 MB is active (process commit 23980 MB)"
        );
    }

    #[test]
    fn unsupported_commit_cap_enforcement_fails_closed() {
        let err = apply_commit_cap_mb_with(64, || Err(format!("Wine {UNSUPPORTED_COMMIT_CAP_MSG}")))
            .err()
            .expect("unsupported host enforcement must fail closed");
        assert!(err.contains("MemoryMax"), "expected Linux cgroup guidance in error: {err}");
        assert_eq!(active_commit_cap_mb(), 0);
    }

    #[test]
    fn commit_cap_probe_requires_exit_zero() {
        use std::io::{Error, ErrorKind};
        use std::path::PathBuf;

        assert_eq!(interpret_commit_cap_probe_outcome(Ok(Some(0))), Ok(()));
        assert_eq!(
            probe_commit_cap_enforcement_with(
                || Ok(PathBuf::from(r"C:\xcl\XCL.exe")),
                |exe| {
                    assert_eq!(exe, Path::new(r"C:\xcl\XCL.exe"));
                    Ok(Some(0))
                },
            ),
            Ok(())
        );

        let xcl_exe = || Ok(PathBuf::from(r"C:\xcl\xcl.exe"));
        let cases: [(&str, Result<(), String>, &str); 7] = [
            (
                "exit 2",
                probe_commit_cap_enforcement_with(xcl_exe, |_| Ok(Some(2))),
                "without enforcing VirtualAlloc commit limits",
            ),
            (
                "exit 3",
                probe_commit_cap_enforcement_with(xcl_exe, |_| Ok(Some(3))),
                "exit Some(3)",
            ),
            (
                "terminated (None)",
                probe_commit_cap_enforcement_with(xcl_exe, |_| Ok(None)),
                "exit None",
            ),
            (
                "launch io::Error",
                probe_commit_cap_enforcement_with(xcl_exe, |_| Err(Error::new(ErrorKind::NotFound, "missing child"))),
                "child launch failed",
            ),
            (
                "missing current_exe",
                probe_commit_cap_enforcement_with(
                    || Err(Error::new(ErrorKind::NotFound, "missing exe")),
                    |_| unreachable!("probe child must not run when current_exe fails"),
                ),
                "current_exe failed",
            ),
            (
                "non-xcl executable",
                probe_commit_cap_enforcement_with(
                    || Ok(PathBuf::from(r"C:\xcl\other.exe")),
                    |_| unreachable!("probe child must not run for non-xcl executable"),
                ),
                "commit-cap enforcement probe needs xcl.exe",
            ),
            (
                "executable without stem",
                probe_commit_cap_enforcement_with(
                    || Ok(PathBuf::from(r"C:\")),
                    |_| unreachable!("probe child must not run without xcl file stem"),
                ),
                "commit-cap enforcement probe needs xcl.exe",
            ),
        ];

        for (label, outcome, expected_substr) in cases {
            let err = outcome
                .clone()
                .err()
                .unwrap_or_else(|| panic!("{label} unexpectedly passed probe"));
            assert!(
                err.contains(expected_substr),
                "{label}: expected {expected_substr:?} in {err:?}"
            );
            let applied = apply_commit_cap_mb_with(64, || outcome)
                .err()
                .unwrap_or_else(|| panic!("{label}: apply_commit_cap_mb_with must fail closed"));
            assert_eq!(applied, err, "{label}: error mismatch");
            assert_eq!(active_commit_cap_mb(), 0, "{label}: active_commit_cap_mb must stay 0");
        }
    }

    #[test]
    fn commit_cap_probe_reports_host_enforcement() {
        const CHILD_MODE: &str = "XCL_TEST_COMMIT_CAP_PROBE";
        if std::env::var_os(CHILD_MODE).is_some() {
            std::process::exit(commit_cap_probe_main());
        }
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new(&exe)
            .args(["--exact", "host::tests::commit_cap_probe_reports_host_enforcement", "--nocapture"])
            .env(CHILD_MODE, "1")
            .output()
            .unwrap();
        let code = out.status.code();
        if crate::util::under_wine() {
            assert_eq!(code, Some(2), "expected probe code 2 (un-enforced) under Wine, got {code:?}");
        } else {
            assert_eq!(code, Some(0), "expected probe code 0 (enforced) on native Windows, got {code:?}");
        }
    }

    #[test]
    fn title_commit_cap_enforced_only_when_positive() {
        const CHILD_MODE: &str = "XCL_TEST_COMMIT_CAP_MODE";
        if let Ok(mode) = std::env::var(CHILD_MODE) {
            use windows_sys::Win32::Foundation::{GetLastError, ERROR_COMMITMENT_LIMIT, ERROR_NOT_ENOUGH_MEMORY};
            use windows_sys::Win32::System::Memory::{
                VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE,
            };

            const ALLOC_BYTES: usize = 128 * 1024 * 1024;
            match mode.as_str() {
                "unset" | "zero" => {
                    let job = apply_commit_cap_from_env().expect("disabled cap should not error");
                    assert!(job.is_none(), "expected no job handle when cap is disabled ({mode})");
                    assert_eq!(active_commit_cap_mb(), 0);
                    unsafe {
                        let ptr = VirtualAlloc(std::ptr::null(), ALLOC_BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
                        assert!(!ptr.is_null(), "uncapped 128 MB VirtualAlloc failed with {}", GetLastError());
                        assert_ne!(VirtualFree(ptr, 0, MEM_RELEASE), 0);
                    }
                }
                "capped" => {
                    if crate::util::under_wine() {
                        let err = apply_commit_cap_from_env()
                            .err()
                            .expect("Wine host should refuse un-enforced commit cap");
                        assert!(err.contains("MemoryMax"), "expected Linux cgroup guidance: {err}");
                        assert_eq!(active_commit_cap_mb(), 0);
                    } else {
                        let _job = apply_commit_cap_from_env()
                            .expect("64 MB cap should succeed")
                            .expect("expected live job handle for 64 MB cap");
                        assert_eq!(active_commit_cap_mb(), 64);
                        unsafe {
                            let ptr = VirtualAlloc(std::ptr::null(), ALLOC_BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
                            let err = GetLastError();
                            assert!(ptr.is_null(), "128 MB VirtualAlloc unexpectedly succeeded under 64 MB cap");
                            assert!(
                                err == ERROR_NOT_ENOUGH_MEMORY || err == ERROR_COMMITMENT_LIMIT,
                                "expected ERROR_NOT_ENOUGH_MEMORY ({ERROR_NOT_ENOUGH_MEMORY}) or ERROR_COMMITMENT_LIMIT ({ERROR_COMMITMENT_LIMIT}), got {err}"
                            );
                            if !ptr.is_null() {
                                VirtualFree(ptr, 0, MEM_RELEASE);
                            }
                        }
                        let mut buf = Vec::<u8>::new();
                        assert!(
                            buf.try_reserve(ALLOC_BYTES).is_err(),
                            "128 MB heap reservation unexpectedly succeeded under 64 MB cap"
                        );
                    }
                }
                other => panic!("unexpected child mode: {other}"),
            }
            return;
        }

        let exe = std::env::current_exe().unwrap();
        let test_name = "host::tests::title_commit_cap_enforced_only_when_positive";

        for (mode, env_val) in [("unset", None), ("zero", Some("0")), ("capped", Some("64"))] {
            let mut cmd = std::process::Command::new(&exe);
            cmd.args(["--exact", test_name, "--nocapture"]).env(CHILD_MODE, mode);
            match env_val {
                Some(v) => {
                    cmd.env(crate::config::ENV_TITLE_COMMIT_CAP_MB, v);
                }
                None => {
                    cmd.env_remove(crate::config::ENV_TITLE_COMMIT_CAP_MB);
                }
            }
            let out = cmd.output().unwrap();
            assert!(
                out.status.success(),
                "child mode {mode} failed (status {:?}):\nstdout:\n{}\nstderr:\n{}",
                out.status.code(),
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }

    pub(super) const HOST_EARLY_OOM_BOOT_ENV: &str = "XCL_TEST_HOST_EARLY_OOM_BOOT";

    pub(super) fn trigger_early_oom() -> ! {
        if crate::util::under_wine() {
            assert_eq!(active_commit_cap_mb(), 0);
            std::alloc::handle_alloc_error(
                std::alloc::Layout::from_size_align(128 * 1024 * 1024, 1).unwrap(),
            );
        } else {
            assert!(active_commit_cap_mb() > 0, "expected active commit cap before early OOM trigger");
            let v = vec![0u8; 128 * 1024 * 1024];
            std::hint::black_box(v);
            panic!("allocation unexpectedly succeeded under active commit cap");
        }
    }

    #[test]
    fn host_main_early_commit_cap_oom_writes_log_and_report() {
        if let Some(boot_path) = std::env::var_os(HOST_EARLY_OOM_BOOT_ENV) {
            let code = host_main_with(Path::new(&boot_path), false);
            panic!("host_main unexpectedly returned {code} without early OOM abort");
        }

        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let dir = std::env::temp_dir().join(format!("xcl_host_early_oom_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let _cleanup = Cleanup(dir.clone());
        let game_root = dir.join("game");
        let data_dir = dir.join("data");
        std::fs::create_dir_all(&game_root).unwrap();
        std::fs::create_dir_all(&data_dir).unwrap();

        let boot = Boot {
            title: "host-early-oom-test".into(),
            exe_path: game_root.join("game.exe"),
            game_root: game_root.clone(),
            data_dir: data_dir.clone(),
            config: crate::config::Config::default(),
        };
        let boot_path = dir.join("boot.json");
        std::fs::write(&boot_path, serde_json::to_string_pretty(&boot).unwrap()).unwrap();

        let exe = std::env::current_exe().unwrap();
        let test_name = "host::tests::host_main_early_commit_cap_oom_writes_log_and_report";
        let out = std::process::Command::new(&exe)
            .args(["--exact", test_name, "--nocapture"])
            .env(HOST_EARLY_OOM_BOOT_ENV, &boot_path)
            .env(crate::config::ENV_TITLE_COMMIT_CAP_MB, "64")
            .env_remove("XCL_COMMIT_CAP_PROBE")
            .env_remove("XCL_HOST")
            .env_remove("XCL_PREPARATION_PARENT")
            .env_remove("XCL_PREPARATION_MENU")
            .output()
            .unwrap();

        let code = out.status.code().map(|c| c as u32);
        assert_eq!(
            code,
            Some(0xC000_0409),
            "expected STATUS_STACK_BUFFER_OVERRUN (0xc0000409), got {code:?}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );

        let log_text = std::fs::read_to_string(data_dir.join("logs").join("xcl.log")).expect("xcl.log exists");
        assert!(log_text.contains("memory allocation of 134217728 bytes"), "log missing byte count:\n{log_text}");
        assert!(log_text.contains("process commit "), "log missing process commit:\n{log_text}");

        let report_path = data_dir.join("logs").join("report.json");
        let report_raw = std::fs::read_to_string(&report_path).expect("report.json exists");
        let rep: crate::report::Report = serde_json::from_str(&report_raw)
            .unwrap_or_else(|e| panic!("report.json invalid JSON: {e}\n{report_raw}"));
        assert_eq!(rep.title, "host-early-oom-test");
        assert_eq!(rep.exe, game_root.join("game.exe").to_string_lossy());
        assert_eq!(rep.exit_code, Some(0xC000_0409_u32 as i32 as i64));
        let crash = rep.crash.expect("report.crash recorded");
        assert!(crash.contains("134217728 bytes"), "crash missing byte count: {crash}");
        assert!(crash.contains("process commit "), "crash missing process commit: {crash}");
        assert_eq!(rep.notes, vec![crash.clone()]);

        if crate::util::under_wine() {
            assert!(
                !log_text.contains("XCL_TITLE_COMMIT_CAP_MB=64 MB"),
                "log must not advertise inactive cap on Wine:\n{log_text}"
            );
            assert!(
                !crash.contains(crate::config::ENV_TITLE_COMMIT_CAP_MB),
                "crash must not advertise inactive cap on Wine: {crash}"
            );
        } else {
            assert!(
                log_text.contains("process commit cap: 64 MB (XCL_TITLE_COMMIT_CAP_MB)"),
                "log missing startup cap line:\n{log_text}"
            );
            assert!(
                log_text.contains("XCL_TITLE_COMMIT_CAP_MB=64 MB"),
                "log missing emergency alloc cap:\n{log_text}"
            );
            assert!(
                crash.contains("XCL_TITLE_COMMIT_CAP_MB=64 MB"),
                "crash missing cap: {crash}"
            );
        }
    }
}
