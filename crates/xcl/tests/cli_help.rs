use std::process::Command;

/// `xcl <command> --help` prints the usage, as `xcl --help` does: the flag was read as the file
/// to open (`xcl shaders: --help: File not found.`, exit 2).
#[test]
fn help_after_a_command_prints_the_usage() {
    for command in ["run", "inspect", "disasm", "shaders", "rpf"] {
        for flag in ["--help", "-h"] {
            let out = Command::new(env!("CARGO_BIN_EXE_xcl")).args([command, flag]).output().unwrap();
            assert!(out.status.success(), "xcl {command} {flag}: {out:?}");
            assert!(String::from_utf8_lossy(&out.stdout).contains("Usage:"), "xcl {command} {flag}: {out:?}");
        }
    }
}
