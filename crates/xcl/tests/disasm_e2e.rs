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

fn le(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

/// Console CS container: DXBC with one DXIL part (program kind CS, no bitcode) whose XDAT
/// holds XBCH (zlib) → FSSM with `code` and a CS stage descriptor. The construction follows
/// tests/gpu_compute_e2e.rs (`console_cs`), with the code payload passed in and four trailer
/// bytes after it so `--hex` has something to dump.
fn console_cs(code: &[u8]) -> Vec<u8> {
    let code_at = 0x308usize;
    let mut fssm = vec![0u8; code_at];
    fssm[0..4].copy_from_slice(b"FSSM");
    fssm[4..8].copy_from_slice(&0x101u32.to_le_bytes());
    fssm[0xC0..0xC8].copy_from_slice(&((code_at - 0x10) as u64).to_le_bytes());
    fssm[0xC8..0xD0].copy_from_slice(&(code.len() as u64).to_le_bytes());
    fssm[0x100..0x108].copy_from_slice(&0x0000_00C0_0002_0106u64.to_le_bytes()); // CS, API stage 1
    fssm[0x138..0x144].copy_from_slice(&[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]); // rsrc2 + numthreads
    fssm.extend_from_slice(code);
    fssm.extend_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd]);
    let size = (fssm.len() - 0x10) as u32;
    fssm[8..12].copy_from_slice(&size.to_le_bytes());
    let z = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 6);
    let mut xbch = b"XBCH".to_vec();
    for x in [3, z.len() as u32, 0, fssm.len() as u32, 0] {
        le(&mut xbch, x);
    }
    xbch.extend_from_slice(&z);
    // DXIL part: program header (kind 5 = CS, 6.6), XDAT, its entries, NOBC placeholder
    let table = 0x18 + 12;
    let data_at = table + 2 * 12;
    let mut d = Vec::new();
    for x in [5 << 16 | 0x66, 0, u32::from_le_bytes(*b"DXIL"), 0x106, 0, 4] {
        le(&mut d, x);
    }
    d.extend_from_slice(b"XDAT");
    for x in [2, table as u32, 0x12, data_at as u32, 16, 0x0d, (data_at + 16) as u32, xbch.len() as u32] {
        le(&mut d, x);
    }
    d.extend_from_slice(&[0x5a; 16]);
    d.extend_from_slice(&xbch);
    while d.len() % 4 != 0 {
        d.push(0);
    }
    let nobc = d.len();
    d.extend_from_slice(b"NOBC");
    let dwords = (d.len() / 4) as u32;
    d[4..8].copy_from_slice(&dwords.to_le_bytes());
    d[16..20].copy_from_slice(&((nobc - 8) as u32).to_le_bytes());
    let mut v = vec![0u8; 36];
    v[0..4].copy_from_slice(b"DXBC");
    v[20..24].copy_from_slice(&1u32.to_le_bytes());
    v[28..32].copy_from_slice(&1u32.to_le_bytes());
    v[32..36].copy_from_slice(&36u32.to_le_bytes());
    v.extend_from_slice(b"DXIL");
    le(&mut v, d.len() as u32);
    v.extend_from_slice(&d);
    let total = v.len() as u32;
    v[24..28].copy_from_slice(&total.to_le_bytes());
    v
}

#[test]
fn hex_flag_dumps_the_fssm_data_after_the_code() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let container =
        std::env::temp_dir().join(format!("xcl-disasm-hex-{}-{nonce}.bin", std::process::id()));
    std::fs::write(&container, console_cs(&[0, 0, 0x80, 0xbf, 0, 0, 0x81, 0xbf])).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("disasm")
            .arg(&container)
            .args(args)
            .output()
    };
    let plain = run(&[]);
    let hex = run(&["--hex"]);
    std::fs::remove_file(&container).unwrap();

    let plain = plain.unwrap();
    assert!(plain.status.success(), "{:?}", plain);
    let out = String::from_utf8(plain.stdout).unwrap();
    assert!(out.contains("up to +0x140; --hex: all"), "{out}");
    assert!(!out.contains("data after the code"), "{out}");

    let hex = hex.unwrap();
    assert!(hex.status.success(), "{:?}", hex);
    let out = String::from_utf8(hex.stdout).unwrap();
    assert!(!out.contains("up to +0x140"), "{out}");
    assert!(out.contains("data after the code"), "{out}");
    assert!(out.contains("aa bb cc dd"), "{out}");
}
