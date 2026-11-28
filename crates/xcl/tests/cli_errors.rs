//! The command line's contract with scripts and CI: exit codes, and which stream
//! each message goes to. Nothing here needs a game, a GPU or the test title, so
//! these cases run wherever the binary does.

use std::path::PathBuf;
use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    let data = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("cli_errors_data");
    Command::new(env!("CARGO_BIN_EXE_xcl"))
        .args(args)
        .env("XCL_DATA_ROOT", &data)
        .output()
        .unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// A scratch file next to the other test output; the name is the test's own.
fn scratch(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("cli_errors");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn version_is_a_command_and_a_typo_is_a_usage_error() {
    let version = run(&["--version"]);
    assert_eq!(version.status.code(), Some(0));
    assert_eq!(
        text(&version.stdout).trim(),
        format!("xcl {}", env!("CARGO_PKG_VERSION"))
    );

    let help = run(&["frobnicate"]);
    assert_eq!(
        help.status.code(),
        Some(2),
        "an unknown command is a usage error"
    );
    let stdout = text(&help.stdout);
    assert!(stdout.contains("Usage:"), "{stdout}");
    assert!(
        stdout.contains("xcl inspect <file.exe|file.dll>"),
        "{stdout}"
    );
}

#[test]
fn inspect_tells_a_missing_file_from_a_file_that_is_not_a_pe() {
    let missing = run(&["inspect", "no-such-file.exe"]);
    assert_eq!(missing.status.code(), Some(2), "{}", text(&missing.stderr));
    let stderr = text(&missing.stderr);
    assert!(stderr.contains("xcl inspect: cannot read"), "{stderr}");
    assert!(stderr.contains("no-such-file.exe"), "{stderr}");
    assert!(
        text(&missing.stdout).is_empty(),
        "stdout stays clean: {stderr}"
    );

    let not_pe = scratch("not-a-pe.bin", b"this is not a PE file");
    let output = run(&["inspect", not_pe.to_str().unwrap()]);
    let stderr = text(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(3),
        "not a PE has its own code: {stderr}"
    );
    assert!(stderr.contains("not a PE file"), "{stderr}");
    assert!(
        stderr.contains("XVC/MSIXVC"),
        "the packaging hint: {stderr}"
    );
}

#[test]
fn disasm_only_fails_on_an_offset_past_the_end_not_on_a_blob_without_shaders() {
    let blob = scratch("no-shaders.bin", &[0x11, 0x22, 0x33, 0x44, 0x55]);
    let path = blob.to_str().unwrap();

    let plain = run(&["disasm", path]);
    let stdout = text(&plain.stdout);
    assert_eq!(
        plain.status.code(),
        Some(0),
        "a file with no shader is not a failure: {stdout}"
    );
    assert!(stdout.contains("no RDNA2 programs found"), "{stdout}");

    let past = run(&["disasm", path, "--offset", "4096"]);
    let stderr = text(&past.stderr);
    assert_eq!(past.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("past the end of the file"), "{stderr}");
}

#[test]
fn rpf_refuses_a_file_that_is_not_an_rpf9_package() {
    let file = scratch("not-an-archive.rpf", b"definitely not an archive header");
    let output = run(&["rpf", file.to_str().unwrap()]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("is not an RPF9 package"), "{stderr}");
    assert!(
        text(&output.stdout).is_empty(),
        "no listing before the check: {stderr}"
    );
}

#[test]
fn run_says_what_is_wrong_with_the_target_and_exits_2() {
    let missing = run(&["run", "no-such-folder"]);
    let stderr = text(&missing.stderr);
    assert_eq!(missing.status.code(), Some(2), "{stderr}");
    assert!(stderr.starts_with("xcl: "), "{stderr}");
    assert!(
        stderr.contains("no-such-folder"),
        "the message names the target: {stderr}"
    );

    let empty = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("cli_errors/empty-game");
    std::fs::create_dir_all(&empty).unwrap();
    let output = run(&["run", empty.to_str().unwrap()]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("no game .exe file in"), "{stderr}");
}
