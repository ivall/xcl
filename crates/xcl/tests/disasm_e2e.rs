use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn disasm_honors_byte_range_options() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let shader =
        std::env::temp_dir().join(format!("xcl-disasm-{}-{nonce}.bin", std::process::id()));
    std::fs::write(&shader, [0, 0, 0x80, 0xbf, 0, 0, 0x81, 0xbf]).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("disasm")
            .arg(&shader)
            .args(args)
            .output()
    };
    let length = run(&["--length", "4"]);
    let offset = run(&["--offset", "0x4", "--length", "0x4"]);
    let malformed = run(&["--offset", "bogus"]);
    let missing = run(&["--length"]);
    std::fs::remove_file(&shader).unwrap();

    let length = length.unwrap();
    assert!(length.status.success());
    let listing = String::from_utf8(length.stdout).unwrap();
    assert!(listing.contains("00000000: s_nop"), "{listing}");
    assert!(!listing.contains("s_endpgm"), "{listing}");

    let offset = offset.unwrap();
    assert!(offset.status.success());
    let listing = String::from_utf8(offset.stdout).unwrap();
    assert!(listing.contains("00000004: s_endpgm"), "{listing}");
    assert!(!listing.contains("s_nop"), "{listing}");

    for (output, flag) in [(malformed, "--offset"), (missing, "--length")] {
        let output = output.unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8(output.stderr).unwrap().contains(flag));
    }
}

#[test]
fn disasm_reports_fssm_write_failure() {
    let mut fssm = vec![0u8; 0x108];
    fssm[..4].copy_from_slice(b"FSSM");
    fssm[4..8].copy_from_slice(&0x101u32.to_le_bytes());
    fssm[8..12].copy_from_slice(&0xF8u32.to_le_bytes());
    fssm[0xC0..0xC8].copy_from_slice(&0xF0u64.to_le_bytes());
    fssm[0xC8..0xD0].copy_from_slice(&8u64.to_le_bytes());
    fssm[0x100..].copy_from_slice(&[0, 0, 0x80, 0xBF, 0, 0, 0x81, 0xBF]); // s_nop; s_endpgm
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 6);
    let mut xbch = vec![0u8; 0x18];
    xbch[..4].copy_from_slice(b"XBCH");
    xbch[4..8].copy_from_slice(&3u32.to_le_bytes());
    xbch[8..12].copy_from_slice(&(compressed.len() as u32).to_le_bytes());
    xbch[16..20].copy_from_slice(&(fssm.len() as u32).to_le_bytes());
    xbch.extend_from_slice(&compressed);

    let mut dxil = vec![0u8; 0x30];
    dxil[..4].copy_from_slice(&0x0005_0066u32.to_le_bytes());
    dxil[8..12].copy_from_slice(b"DXIL");
    dxil[12..16].copy_from_slice(&0x106u32.to_le_bytes());
    dxil[20..24].copy_from_slice(&4u32.to_le_bytes());
    dxil[24..28].copy_from_slice(b"XDAT");
    dxil[28..32].copy_from_slice(&1u32.to_le_bytes());
    dxil[32..36].copy_from_slice(&36u32.to_le_bytes());
    dxil[36..40].copy_from_slice(&0x0Du32.to_le_bytes());
    dxil[40..44].copy_from_slice(&48u32.to_le_bytes());
    dxil[44..48].copy_from_slice(&(xbch.len() as u32).to_le_bytes());
    dxil.extend_from_slice(&xbch);
    dxil.resize((dxil.len() + 3) & !3, 0);
    let bitcode_offset = dxil.len() as u32 - 8;
    dxil.extend_from_slice(b"NOBC");
    let words = dxil.len() as u32 / 4;
    dxil[4..8].copy_from_slice(&words.to_le_bytes());
    dxil[16..20].copy_from_slice(&bitcode_offset.to_le_bytes());

    let total = 44 + dxil.len();
    let mut container = vec![0u8; 44];
    container[..4].copy_from_slice(b"DXBC");
    container[20..24].copy_from_slice(&1u32.to_le_bytes());
    container[24..28].copy_from_slice(&(total as u32).to_le_bytes());
    container[28..32].copy_from_slice(&1u32.to_le_bytes());
    container[32..36].copy_from_slice(&36u32.to_le_bytes());
    container[36..40].copy_from_slice(b"DXIL");
    container[40..44].copy_from_slice(&(dxil.len() as u32).to_le_bytes());
    container.extend_from_slice(&dxil);

    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("xcl-disasm-fssm-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let shader = dir.join("shader.bin");
    let output = dir.join("shader.fssm");
    std::fs::write(&shader, container).unwrap();
    let run = |out: &std::path::Path| {
        Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("disasm")
            .arg(&shader)
            .arg("--fssm")
            .arg(out)
            .output()
    };
    let success = run(&output);
    let saved = std::fs::read(&output);
    let failure = run(&dir); // A directory cannot be overwritten with FSSM bytes.
    std::fs::remove_dir_all(&dir).unwrap();

    let success = success.unwrap();
    assert!(success.status.success(), "{}", String::from_utf8_lossy(&success.stderr));
    assert_eq!(saved.unwrap(), fssm);
    let failure = failure.unwrap();
    assert_eq!(failure.status.code(), Some(2), "{}", String::from_utf8_lossy(&failure.stderr));
    assert!(String::from_utf8(failure.stdout).unwrap().contains("s_endpgm"));
}
