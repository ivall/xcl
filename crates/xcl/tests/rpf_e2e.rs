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
fn rpf_rejects_unknown_options_missing_values_and_invalid_numbers() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let archive = std::env::temp_dir().join(format!(
        "xcl-rpf-badopts-{}-{nonce}.rpf",
        std::process::id()
    ));
    std::fs::write(&archive, b"9FPR").unwrap();

    let cases: &[(&[&str], &str)] = &[
        (&["--extract"], "--extract needs a value"),
        (&["--extract", "--out", "dir"], "--extract needs a value"),
        (&["--find"], "--find needs a value"),
        (&["--out"], "--out needs a value"),
        (&["--top"], "--top needs a value"),
        (&["--top", "abc"], "invalid value for --top: abc"),
        (&["--top", "-1"], "invalid value for --top: -1"),
        (&["--fnid", "foo"], "unknown option: --fnid"),
        (&["-z"], "unknown option: -z"),
        (&["extra"], "unexpected argument: extra"),
    ];
    for (args, expected) in cases {
        let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("rpf")
            .arg(&archive)
            .args(*args)
            .output()
            .expect("spawn xcl rpf");
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("xcl rpf:"), "{args:?}: {stderr}");
        assert!(stderr.contains(expected), "{args:?}: expected {expected:?} in {stderr:?}");
        assert!(output.stdout.is_empty(), "{args:?}: unexpected stdout {:?}", String::from_utf8_lossy(&output.stdout));
    }

    std::fs::remove_file(&archive).unwrap();
}

#[test]
fn probe_rejects_file_without_rpf9_header() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let archive = std::env::temp_dir().join(format!("xcl-rpf-garbage-{}-{nonce}.rpf", std::process::id()));
    std::fs::write(&archive, vec![0u8; 0x860]).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("rpf")
        .arg(&archive)
        .arg("--probe")
        .output();
    std::fs::remove_file(&archive).unwrap();

    let output = output.unwrap();
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("is not an RPF9 package"),
        "stderr did not mention the RPF9 rejection: {stderr:?}"
    );
    assert!(
        !stdout.contains("RPF9 archives"),
        "probe printed a header summary for a non-RPF9 file: {stdout:?}"
    );
}

#[test]
fn probe_rejects_missing_file() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let archive = std::env::temp_dir().join(format!("xcl-rpf-missing-{}-{nonce}.rpf", std::process::id()));
    assert!(!archive.exists());

    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("rpf")
        .arg(&archive)
        .arg("--probe")
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains(&*archive.to_string_lossy()),
        "stderr did not name the missing path: {stderr:?}"
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        !stdout.contains("RPF9 archives"),
        "probe printed a summary for a missing path: {stdout:?}"
    );
}
