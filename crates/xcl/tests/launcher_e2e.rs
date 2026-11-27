use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct TempDir(PathBuf);

impl TempDir {
    fn new(prefix: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn unreadable_shader_pack_stops_launch_before_host_start() {
    let game = TempDir::new("xcl-preparation-failed");
    std::fs::write(game.0.join("title.exe"), b"must never be loaded").unwrap();
    std::fs::write(game.0.join("shaders.rpf"), b"9FPR").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .args(["run"]).arg(&game.0)
        .env("XCL_DATA_ROOT", game.0.join("user"))
        .env("XCL_SKIP_CPU_CHECK", "1")
        .output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("No readable shader records"));
    assert!(!String::from_utf8_lossy(&output.stdout).contains("Log:"));
}

#[test]
fn missing_pipeline_device_skips_shader_preparation_and_reaches_host_loader() {
    let game = TempDir::new("xcl-preparation-no-device");
    std::fs::write(game.0.join("title.exe"), b"not a PE image").unwrap();
    std::fs::create_dir(game.0.join("shaders")).unwrap();
    std::fs::write(game.0.join("shaders").join("vertex.dxbc"), b"DXBC").unwrap();
    let data = game.0.join("user");
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .args(["run"]).arg(&game.0)
        .env("XCL_DATA_ROOT", &data)
        .env("XCL_ADAPTER", "xcl-no-such-adapter-2958")
        .env("XCL_SKIP_CPU_CHECK", "1")
        .output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(0x7C1_0002), "expected loader exit, got {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}", output.status.code());
    assert!(!stderr.contains("No pipeline device"), "stderr: {stderr}");
    let title_dir = std::fs::read_dir(&data).unwrap().flatten().map(|e| e.path()).find(|p| p.is_dir()).expect("title data dir");
    let log = std::fs::read_to_string(title_dir.join("logs").join("xcl.log")).unwrap();
    assert!(log.contains("Shader preparation skipped (no pipeline device:"), "log:\n{log}");
    assert!(log.contains("loading the game failed:"), "log:\n{log}");
}

#[test]
fn shaders_cli_precompile_pipelines_fails_without_device() {
    let dir = TempDir::new("xcl-shaders-no-device");
    std::fs::write(dir.0.join("vertex.dxbc"), b"DXBC").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .args(["shaders"]).arg(&dir.0).args(["--precompile", "--pipelines"])
        .env("XCL_ADAPTER", "xcl-no-such-adapter-2958")
        .output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "stderr: {stderr}");
    assert!(stderr.contains("No pipeline device:"), "stderr: {stderr}");
}

#[test]
fn launcher_missing_target_argument_exits_code_two() {
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("run")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("Usage:"));
    assert!(text.contains("xcl run"));
}

#[test]
fn launcher_nonexistent_target_exits_code_two() {
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("run")
        .arg("nonexistent_game_directory_or_file")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(err.contains("xcl:"));
}

#[test]
fn launcher_rejects_xcl_binary_itself() {
    let exe = env!("CARGO_BIN_EXE_xcl");
    let output = Command::new(exe)
        .arg("run")
        .arg(exe)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(err.contains("is XCL itself, not a game"));
}

#[test]
fn launcher_directory_without_executables_exits_code_two() {
    let dir = TempDir::new("xcl-no-exe");
    std::fs::write(dir.0.join("readme.txt"), b"no executables here").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("run")
        .arg(&dir.0)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(err.contains("no game .exe file in"));
}

#[test]
fn launcher_directory_with_ambiguous_executables_exits_code_two() {
    let dir = TempDir::new("xcl-multi-exe");
    std::fs::write(dir.0.join("game_a.exe"), b"executable a").unwrap();
    std::fs::write(dir.0.join("game_b.exe"), b"executable b").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("run")
        .arg(&dir.0)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(err.contains("several .exe files"));
    assert!(err.contains("point to the game exe directly"));
}

#[test]
fn launcher_directory_with_only_xcl_copy_warns_self() {
    let dir = TempDir::new("xcl-self-copy");
    let self_exe = env!("CARGO_BIN_EXE_xcl");
    let dest_xcl = dir.0.join("xcl.exe");
    std::fs::copy(self_exe, &dest_xcl).unwrap();

    let output = Command::new(self_exe)
        .arg("run")
        .arg(&dir.0)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let err = String::from_utf8(output.stderr).unwrap();
    assert!(err.contains("only xcl.exe – that is XCL itself; do not copy it into the game folder"));
}
