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
fn extract_writes_the_matched_records_to_the_out_folder() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let archive = std::env::temp_dir().join(format!(
        "xcl-rpf-extract-{}-{nonce}.rpf",
        std::process::id()
    ));
    let out = std::env::temp_dir().join(format!("xcl-rpf-extract-{}-{nonce}", std::process::id()));
    let payload = b"DXBCtest";
    let record = |name: &str| {
        let z = miniz_oxide::deflate::compress_to_vec(payload, 6);
        let mut v = vec![(name.len() + 1) as u8];
        v.extend_from_slice(name.as_bytes());
        v.push(0);
        v.push(50);
        for x in [z.len() as u32, 0, z.len() as u32, payload.len() as u32] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v.extend_from_slice(&z);
        v
    };
    let mut data = vec![0; 0x40];
    data[..4].copy_from_slice(b"9FPR");
    data.extend(record("CS_one_11111111_Wrapped"));
    data.extend(record("CS_two_22222222_Wrapped"));
    std::fs::write(&archive, data).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("rpf")
        .arg(&archive)
        .arg("--extract")
        .arg("two")
        .arg("--out")
        .arg(&out)
        .output()
        .unwrap();
    std::fs::remove_file(&archive).unwrap();

    assert!(output.status.success(), "{:?}", output);
    let written = std::fs::read_to_string(out.join("CS_two_22222222_Wrapped")).unwrap();
    assert_eq!(written, "DXBCtest");
    assert!(!out.join("CS_one_11111111_Wrapped").exists());
    std::fs::remove_dir_all(&out).unwrap();
}
