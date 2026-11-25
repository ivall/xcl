use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path(prefix: &str, ext: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{nonce}{ext}", std::process::id()))
}

fn make_record(name: &str, payload: &[u8]) -> Vec<u8> {
    let z = miniz_oxide::deflate::compress_to_vec(payload, 6);
    let mut v = vec![(name.len() + 1) as u8];
    v.extend_from_slice(name.as_bytes());
    v.push(0);
    v.push(50);
    for x in [z.len() as u32, 0xddce7, z.len() as u32, payload.len() as u32] {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v.extend_from_slice(&z);
    v
}

#[test]
fn full_scan_and_streaming_probe_agree_on_header_evidence() {
    let archive = temp_path("xcl-rpf", ".rpf");
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
    let _ = std::fs::remove_file(&archive);

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
    let archive = temp_path("xcl-rpf-options", ".rpf");
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
    let _ = std::fs::remove_file(&archive);

    for (args, output) in options.into_iter().zip(outputs) {
        let output = output.unwrap();
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        assert!(String::from_utf8(output.stderr).unwrap().contains(args[0]));
    }
}

#[test]
fn rpf_missing_arguments_prints_help_and_exits_two() {
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("rpf")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("xcl rpf <package.rpf>"), "{stdout}");
}

#[test]
fn rpf_missing_file_reports_error_and_exits_two() {
    let nonexistent = temp_path("xcl-rpf-nonexistent", ".rpf");
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .args(["rpf", nonexistent.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("xcl rpf:"), "{stderr}");
}

#[test]
fn rpf_non_rpf9_file_reports_error_and_exits_two() {
    let non_rpf = temp_path("xcl-rpf-invalid", ".rpf");
    std::fs::write(&non_rpf, b"NOT_AN_RPF9_PACKAGE_PAYLOAD").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .args(["rpf", non_rpf.to_str().unwrap()])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&non_rpf);

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("is not an RPF9 package"), "{stderr}");
}

#[test]
fn rpf_listing_filtering_and_extraction() {
    let archive = temp_path("xcl-rpf-synthetic", ".rpf");
    let out_dir = temp_path("xcl-rpf-extracted", "");

    let vs_payload = b"TEST_VERTEX_SHADER_DATA_1234";
    let ps_payload = b"TEST_PIXEL_SHADER_DATA_5678";

    let mut pack = b"9FPR".to_vec();
    pack.resize(0x40, 0xAA);
    pack.extend(make_record("VS_Main", vs_payload));
    pack.extend(make_record("PS_Main", ps_payload));
    std::fs::write(&archive, pack).unwrap();

    // 1. Plain listing
    let list_output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .args(["rpf", archive.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(list_output.status.success());
    let list_str = String::from_utf8(list_output.stdout).unwrap();
    assert!(list_str.contains("2 records"), "{list_str}");
    assert!(list_str.contains("VS_Main"), "{list_str}");
    assert!(list_str.contains("PS_Main"), "{list_str}");
    assert!(list_str.contains("extensions (records, unpacked size):"), "{list_str}");

    // 2. Listing filtered with --find VS
    let find_output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .args(["rpf", archive.to_str().unwrap(), "--find", "VS"])
        .output()
        .unwrap();
    assert!(find_output.status.success());
    let find_str = String::from_utf8(find_output.stdout).unwrap();
    assert!(find_str.contains("VS_Main"), "{find_str}");
    assert!(!find_str.contains("PS_Main"), "{find_str}");

    // 3. Extraction with --extract PS --out <out_dir>
    let extract_output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .args([
            "rpf",
            archive.to_str().unwrap(),
            "--extract",
            "PS",
            "--out",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(extract_output.status.success());
    let extract_str = String::from_utf8(extract_output.stdout).unwrap();
    assert!(extract_str.contains("extracted 1 to"), "{extract_str}");

    let extracted_ps = std::fs::read(out_dir.join("PS_Main")).unwrap();
    assert_eq!(extracted_ps, ps_payload);
    assert!(!out_dir.join("VS_Main").exists());

    let _ = std::fs::remove_file(&archive);
    let _ = std::fs::remove_dir_all(&out_dir);
}
