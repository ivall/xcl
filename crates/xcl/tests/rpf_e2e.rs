use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn full_scan_and_streaming_probe_agree_on_header_evidence() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let archive = std::env::temp_dir().join(format!("xcl-rpf-{}-{nonce}.rpf", std::process::id()));
    let mut data = vec![0; 0x860];
    data[..4].copy_from_slice(b"9FPR");
    data[0x800..0x804].copy_from_slice(b"9FPR");
    std::fs::write(&archive, data).unwrap();

    let run = |mode| {
        Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("rpf")
            .arg(&archive)
            .arg(mode)
            .output()
    };
    let full = run("--full");
    let probe = run("--probe");
    std::fs::remove_file(&archive).unwrap();

    let summary = |output: std::process::Output| {
        assert!(output.status.success(), "{:?}", output);
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .skip(1) // The banner describes the selected mode.
            .collect::<Vec<_>>()
            .join("\n")
    };
    let full = summary(full.unwrap());
    let probe = summary(probe.unwrap());
    assert_eq!(full, probe);
    assert!(full.contains("RPF9 archives (headers every 2 KB): 2"));
    assert!(full.contains("TOC encryption: unknown"));
}

#[test]
fn explicit_probe_rejects_conflicting_options() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let archive = std::env::temp_dir().join(format!(
        "xcl-rpf-options-{}-{nonce}.rpf",
        std::process::id()
    ));
    std::fs::write(&archive, b"9FPR").unwrap();

    let options: [&[&str]; 4] = [
        &["--full"],
        &["--find", "record"],
        &["--extract", "record"],
        &["--out", "unused"],
    ];
    let outputs = options.map(|args| {
        Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("rpf")
            .arg(&archive)
            .arg("--probe")
            .args(args)
            .output()
    });
    std::fs::remove_file(&archive).unwrap();

    for (args, output) in options.into_iter().zip(outputs) {
        let output = output.unwrap();
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        assert!(String::from_utf8(output.stderr).unwrap().contains(args[0]));
    }
}

#[test]
fn rpf_missing_arguments_prints_help_and_exits_two() {
    let out = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("rpf")
        .output()
        .expect("spawn xcl rpf");
    assert_eq!(out.status.code(), Some(2));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("xcl rpf <package.rpf>"));
}

#[test]
fn rpf_nonexistent_file_and_invalid_magic_report_errors() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp = std::env::temp_dir();
    let missing = temp.join(format!("xcl-rpf-missing-{nonce}.rpf"));
    let out = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("rpf")
        .arg(&missing)
        .output()
        .expect("spawn xcl rpf missing");
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("xcl rpf:"));

    let bad_magic = temp.join(format!("xcl-rpf-bad-magic-{nonce}.rpf"));
    std::fs::write(&bad_magic, b"NOT_AN_RPF9_ARCHIVE").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("rpf")
        .arg(&bad_magic)
        .output()
        .expect("spawn xcl rpf bad_magic");
    std::fs::remove_file(&bad_magic).unwrap();
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("is not an RPF9 package"));
}

#[test]
fn rpf_header_with_no_records_summarizes_and_exits_zero() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let archive = std::env::temp_dir().join(format!("xcl-rpf-empty-toc-{nonce}.rpf"));
    let mut data = vec![0u8; 0x860];
    data[..4].copy_from_slice(b"9FPR");
    std::fs::write(&archive, &data).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("rpf")
        .arg(&archive)
        .output()
        .expect("spawn xcl rpf empty");
    std::fs::remove_file(&archive).unwrap();

    assert_eq!(out.status.code(), Some(0));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("no records with plain names"));
    assert!(stdout.contains("RPF9 archives (headers every 2 KB): 1"));
}
