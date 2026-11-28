//! Retail BCPack chunks decoded by XCL against a reference decoder's blocks.
//! Set `XCL_SAMPLE_BCPACK` to a directory of `<name>.bc<N>` + `<name>.vic<N>` files;
//! without it the test does nothing.

use xcl::hle::bcpack::{block_size, can_decode, decode, Error};

#[test]
fn mode_block_sizes_and_support_invariants() {
    assert_eq!(block_size(1), Some(8), "BC1 block size must be 8 bytes");
    assert_eq!(block_size(4), Some(8), "BC4 block size must be 8 bytes");
    assert_eq!(block_size(3), Some(16), "BC3 block size must be 16 bytes");
    assert_eq!(block_size(5), Some(16), "BC5 block size must be 16 bytes");
    assert_eq!(block_size(6), Some(16), "BC6H block size must be 16 bytes");
    assert_eq!(block_size(7), Some(16), "BC7 block size must be 16 bytes");

    for mode in [0, 2, 8, 9, 10, 255] {
        assert_eq!(block_size(mode), None, "mode {mode} has no block size");
    }

    for mode in 0..=255u8 {
        assert_eq!(
            can_decode(mode),
            block_size(mode).is_some(),
            "can_decode parity mismatch for mode {mode}"
        );
    }
}

#[test]
fn error_display_and_kind_invariants() {
    let unsupp = Error::Unsupported;
    assert_eq!(unsupp.kind(), "unsupported");
    assert_eq!(format!("{unsupp}"), "no decoder for this mode");

    let out_size = Error::OutputSize;
    assert_eq!(out_size.kind(), "output size");
    assert_eq!(
        format!("{out_size}"),
        "the output size is not a whole number of blocks"
    );

    let dec = Error::Decode("truncated stream");
    assert_eq!(dec.kind(), "decode");
    assert_eq!(format!("{dec}"), "truncated stream");

    let copy = dec;
    assert_eq!(copy, dec);
    assert_ne!(unsupp, out_size);
    assert_ne!(out_size, dec);
}

#[test]
fn decode_input_and_output_size_validation() {
    for mode in [0, 2, 8, 255] {
        assert_eq!(decode(mode, &[], 8), Err(Error::Unsupported));
        assert_eq!(decode(mode, &[1, 2, 3, 4], 16), Err(Error::Unsupported));
    }

    for &mode in &[1, 4] {
        assert_eq!(decode(mode, &[0; 16], 0), Err(Error::OutputSize));
        for bad_len in [1, 2, 7, 9, 15] {
            assert_eq!(decode(mode, &[0; 16], bad_len), Err(Error::OutputSize));
        }
    }

    for &mode in &[3, 5, 6, 7] {
        assert_eq!(decode(mode, &[0; 16], 0), Err(Error::OutputSize));
        for bad_len in [1, 2, 8, 15, 17, 31] {
            assert_eq!(decode(mode, &[0; 16], bad_len), Err(Error::OutputSize));
        }
    }

    for &mode in &[1, 3, 4, 5, 6, 7] {
        let block = block_size(mode).unwrap();
        assert!(matches!(decode(mode, &[], block), Err(Error::Decode(_))));
    }
}

#[test]
fn decode_malformed_and_truncated_streams_fail_safely() {
    for &mode in &[1, 3, 4, 5, 6, 7] {
        let block = block_size(mode).unwrap();

        for cut in [1, 2, 3, 4] {
            let truncated = vec![0x5Au8; cut];
            let res = decode(mode, &truncated, block);
            assert!(
                matches!(res, Err(Error::Decode(_))),
                "mode {mode} with {cut} bytes did not return Error::Decode: {res:?}"
            );
        }

        // Verify decoding arbitrary/fuzzed streams never panics
        let mut x = 0x8765_4321u32.wrapping_add(mode as u32);
        let junk: Vec<u8> = (0..128)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x >> 24) as u8
            })
            .collect();
        let _ = decode(mode, &junk, block * 4);
    }
}

#[test]
fn retail_chunks_match_the_reference() {
    let Some(dir) = std::env::var_os("XCL_SAMPLE_BCPACK") else { return };
    let dir = std::path::Path::new(&dir);
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("XCL_SAMPLE_BCPACK {}: {e}", dir.display()));
    let mut cases = 0;
    let mut identical = 0;
    let mut bad_bytes = 0u64;
    let mut total_bytes = 0u64;
    for entry in entries {
        let path = entry.unwrap().path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else { continue };
        let Some(mode) = ext.strip_prefix("bc").and_then(|m| m.parse::<u8>().ok()) else { continue };
        let vic = path.with_extension(format!("vic{mode}"));
        let src = std::fs::read(&path).unwrap();
        let want = if vic.exists() {
            std::fs::read(&vic).unwrap()
        } else {
            std::fs::read(path.with_extension("out")).unwrap()
        };
        let got = match xcl::hle::bcpack::decode(mode, &src, want.len()) {
            Ok(got) => got,
            Err(e) => {
                println!("{}: mode {mode} failed: {e}", path.display());
                cases += 1;
                continue;
            }
        };
        cases += 1;
        total_bytes += want.len() as u64;
        if got == want {
            identical += 1;
        } else {
            let n = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            bad_bytes += n as u64;
            println!("{}: mode {mode}, {n} of {} bytes differ", path.file_name().unwrap().to_string_lossy(), want.len());
        }
    }
    println!(
        "cases {cases}, byte-identical {identical}, differing bytes {bad_bytes} of {total_bytes} ({:.4}%)",
        100.0 * bad_bytes as f64 / total_bytes.max(1) as f64
    );
    assert!(cases > 0);
    assert_eq!(identical, cases, "every case must match the reference");
}
