#[path = "common/shader_container.rs"]
mod shader_container;

use shader_container::{container, dxil_part, fssm};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// s_mov_b32 s0, s1 ; v_add_f32 v0, v1, v5 ; s_endpgm
const CODE: [u8; 12] = [0x01, 0x03, 0x80, 0xbe, 0x01, 0x0b, 0x00, 0x06, 0x00, 0x00, 0x81, 0xbf];

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("xcl-shaders-{tag}-{}-{nonce}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn xbox_shader(kind: u32, fssms: &[Vec<u8>]) -> Vec<u8> {
    container(&[(b"SFI0", &[0; 8]), (b"HASH", &[0; 20]), (b"DXIL", &dxil_part(kind, fssms)), (b"PRIV", &[7; 16])])
}

fn rpf9_record(name: &str, payload: &[u8], usize_override: Option<u32>) -> Vec<u8> {
    let z = miniz_oxide::deflate::compress_to_vec(payload, 6);
    let mut v = vec![(name.len() + 1) as u8];
    v.extend_from_slice(name.as_bytes());
    v.push(0);
    v.push(50);
    for x in [z.len() as u32, 0xddce7, z.len() as u32, usize_override.unwrap_or(payload.len() as u32)] {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v.extend_from_slice(&z);
    v
}

fn run_shaders(target: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("shaders")
        .arg(target)
        .args(extra)
        .env_remove("XCL_DUMP_HLSL")
        .env_remove("XCL_RECOMP_ONLY")
        .env_remove("XCL_EXAMPLE_HLSL")
        .output()
        .expect("spawn xcl shaders")
}

fn run_shaders_with_cache(target: &Path, extra: &[&str], cache_dir: &Path) -> Output {
    run_shaders_with_cache_and_env(target, extra, cache_dir, &[])
}

fn run_shaders_with_cache_and_env(
    target: &Path,
    extra: &[&str],
    cache_dir: &Path,
    envs: &[(&str, &str)],
) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xcl"));
    cmd.arg("shaders")
        .arg(target)
        .args(extra)
        .env("XCL_DXIL_CACHE", cache_dir)
        .env_remove("XCL_DIRECT_DXIL")
        .env_remove("XCL_SAFE_BARRIER")
        .env_remove("XCL_RECOMP_NO_STRUCTURED")
        .env_remove("XCL_DUMP_HLSL")
        .env_remove("XCL_RECOMP_ONLY")
        .env_remove("XCL_EXAMPLE_HLSL");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.output().expect("spawn xcl shaders")
}

#[test]
fn help_lists_shader_precompile_options() {
    let out = Command::new(env!("CARGO_BIN_EXE_xcl")).arg("--help").output().unwrap();
    assert!(out.status.success(), "{out:?}");
    let help = String::from_utf8(out.stdout).unwrap();
    for flag in ["--precompile", "--only", "--pipelines", "--used-only"] {
        assert!(help.contains(flag), "missing {flag} in xcl --help");
    }
}

#[test]
fn shaders_single_file_reports_text_top_json_recomp_and_dxc_compile() {
    let dir = TempDir::new("single");
    let file = dir.path().join("ps.bin");
    std::fs::write(&file, xbox_shader(0, &[fssm(&CODE, &[(5, 1)])])).unwrap();

    // Default text output
    let text_out = run_shaders(&file, &["--recomp"]);
    assert!(text_out.status.success(), "{text_out:?}");
    let text = String::from_utf8(text_out.stdout).unwrap();
    assert!(text.contains("single file"), "{text}");
    assert!(text.contains("shaders: 1"), "{text}");
    assert!(text.contains("formats: xbox-native 1"), "{text}");
    assert!(text.contains("stages (DXIL): PS 1"), "{text}");
    assert!(text.contains("FSSM variants by hardware stage: PS 1"), "{text}");
    assert!(text.contains("RDNA2 instructions: 3, undecoded dwords: 0 (0.0000%), distinct mnemonics: 3"), "{text}");
    assert!(text.contains("top 3 mnemonics"), "{text}");
    assert!(text.contains("recompiler (kernel mode): programs 1, fully translated 1 (100.0%), translation errors 0"), "{text}");

    // --top 1 limits the printed mnemonic table
    let top_out = run_shaders(&file, &["--top", "1"]);
    assert!(top_out.status.success(), "{top_out:?}");
    let top_text = String::from_utf8(top_out.stdout).unwrap();
    assert!(top_text.contains("top 1 mnemonics"), "{top_text}");
    let ranked_lines: Vec<&str> = top_text
        .lines()
        .skip_while(|l| !l.starts_with("top 1 mnemonics"))
        .skip(1)
        .take_while(|l| l.starts_with("  "))
        .collect();
    assert_eq!(ranked_lines.len(), 1, "{top_text}");

    // --json --recomp --compile 1
    let json_out = run_shaders(&file, &["--json", "--recomp", "--compile", "1"]);
    assert!(json_out.status.success(), "{json_out:?}");
    let json: serde_json::Value = serde_json::from_slice(&json_out.stdout).unwrap();
    assert_eq!(json["source"], "single file");
    assert_eq!(json["shaders"], 1);
    assert_eq!(json["inflate_errors"], 0);
    assert_eq!(json["by_kind"]["xbox-native"], 1);
    assert_eq!(json["by_dxil_stage"]["PS"], 1);
    assert_eq!(json["by_hw_stage"]["PS"], 1);
    assert_eq!(json["fssm"], 1);
    assert_eq!(json["programs"], 1);
    assert_eq!(json["parse_problems"], 0);
    assert_eq!(json["instructions"], 3);
    assert_eq!(json["undecoded_dwords"], 0);
    assert_eq!(json["distinct_mnemonics"], 3);
    assert_eq!(json["mnemonics"].as_array().unwrap().len(), 3);
    let rc = &json["recomp"];
    assert_eq!(rc["programs"], 1, "{json}");
    assert_eq!(rc["complete"], 1, "{json}");
    assert_eq!(rc["translate_errors"], 0, "{json}");
    assert!(rc["hlsl_bytes"].as_u64().unwrap() > 0, "{json}");
    assert_eq!(rc["dxc_compiled"], 1, "{json}");
    assert_eq!(rc["dxc_failed"].as_array().unwrap().len(), 0, "{json}");
}

#[test]
fn shaders_folder_recurses_and_skips_side_files() {
    let dir = TempDir::new("folder");
    let nested = dir.path().join("nested");
    std::fs::create_dir_all(&nested).unwrap();

    std::fs::write(dir.path().join("ps.bin"), xbox_shader(0, &[fssm(&CODE, &[(5, 1)])])).unwrap();
    std::fs::write(
        nested.join("vs.bin"),
        xbox_shader(1, &[fssm(&CODE, &[(2, 0)]), fssm(&CODE, &[(4, 0)])]),
    )
    .unwrap();
    std::fs::write(dir.path().join("plain.bin"), b"not a shader").unwrap();
    // Side files written by XCL itself in <data>\shaders\ must be skipped by collect_files.
    std::fs::write(dir.path().join("listing.txt"), b"disassembly side file").unwrap();
    std::fs::write(nested.join("dump.fssm.bin"), xbox_shader(0, &[fssm(&CODE, &[(5, 1)])])).unwrap();

    let out = run_shaders(dir.path(), &["--json"]);
    assert!(out.status.success(), "{out:?}");
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["source"], "folder, 3 files", "{json}");
    assert_eq!(json["shaders"], 3);
    assert_eq!(json["by_kind"]["xbox-native"], 2);
    assert_eq!(json["by_kind"]["unknown"], 1);
    assert_eq!(json["by_dxil_stage"]["PS"], 1);
    assert_eq!(json["by_dxil_stage"]["VS"], 1);
    assert_eq!(json["by_hw_stage"]["ES"], 1);
    assert_eq!(json["by_hw_stage"]["VS"], 1);
    assert_eq!(json["by_hw_stage"]["PS"], 1);
    assert_eq!(json["fssm"], 3);
    assert_eq!(json["programs"], 3);
    assert_eq!(json["instructions"], 9);
    assert_eq!(json["parse_problems"], 0);
}

#[test]
fn shaders_rpf9_pack_scans_records_and_counts_inflate_errors() {
    let dir = TempDir::new("rpf9");
    let pack_path = dir.path().join("shaders.rpf");
    let ps = xbox_shader(0, &[fssm(&CODE, &[(5, 1)])]);
    let mut pack = b"9FPR".to_vec();
    pack.resize(0x40, 0xAA);
    pack.extend(rpf9_record("PS_Synthetic_1234abcd_Wrapped", &ps, None));
    pack.extend_from_slice(&[0x13, 0x37, 0, 1, 2, 3]);
    pack.extend(rpf9_record("VS_PlainBlob", b"DXBC-not-console", None));
    // Mismatched uncompressed size forces pack::inflate to return None -> inflate_errors == 1.
    pack.extend(rpf9_record("CS_BadInflate", b"synthetic payload", Some(999)));
    pack.extend_from_slice(&[0u8; 40]);
    std::fs::write(&pack_path, &pack).unwrap();

    let out = run_shaders(&pack_path, &["--json"]);
    assert!(out.status.success(), "{out:?}");
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(
        json["source"].as_str().unwrap().starts_with("RPF9 package "),
        "{json}"
    );
    assert_eq!(json["shaders"], 2, "{json}");
    assert_eq!(json["inflate_errors"], 1, "{json}");
    assert_eq!(json["by_kind"]["xbox-native"], 1);
    assert_eq!(json["by_kind"]["unknown"], 1);
    assert_eq!(json["by_dxil_stage"]["PS"], 1);
    assert_eq!(json["instructions"], 3);
    assert_eq!(json["undecoded_dwords"], 0);
}

#[test]
fn shaders_handles_non_console_file_and_missing_path() {
    let dir = TempDir::new("edge");
    let plain = dir.path().join("plain.bin");
    std::fs::write(&plain, b"not a shader").unwrap();

    let out = run_shaders(&plain, &[]);
    assert!(out.status.success(), "{out:?}");
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("shaders: 1"), "{text}");
    assert!(text.contains("no console containers (xbox-native)"), "{text}");

    let missing = dir.path().join("does-not-exist.bin");
    let err_out = run_shaders(&missing, &[]);
    assert_eq!(err_out.status.code(), Some(2), "{err_out:?}");
    let stderr = String::from_utf8(err_out.stderr).unwrap();
    assert!(stderr.contains("xcl shaders:"), "{stderr}");
}

#[test]
fn shaders_missing_arguments_prints_help_and_exits_two() {
    let output = Command::new(env!("CARGO_BIN_EXE_xcl"))
        .arg("shaders")
        .output()
        .expect("spawn xcl shaders");
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("xcl shaders <file | folder | package.rpf>"), "{stdout}");
}

#[test]
fn shaders_required_options_reject_missing_values() {
    let dir = TempDir::new("val");
    let file = dir.path().join("plain.bin");
    std::fs::write(&file, b"not a shader").unwrap();

    let cases: &[(&[&str], &str)] = &[
        (&["--export-hlsl"], "--export-hlsl needs a value"),
        (&["--export-hlsl", "--record-hash", "DEADBEEF"], "--export-hlsl needs a value"),
        (&["--record-hash"], "--record-hash needs a value"),
        (&["--record-hash", "--measure-pso", "CAFE"], "--record-hash needs a value"),
        (&["--measure-pso"], "--measure-pso needs a value"),
        (&["--measure-pso", "--json"], "--measure-pso needs a value"),
    ];

    for (args, expected_msg) in cases {
        let out = run_shaders(&file, args);
        assert_eq!(out.status.code(), Some(2), "args {args:?} unexpected status: {out:?}");
        let stderr = String::from_utf8(out.stderr).unwrap();
        assert!(stderr.contains(expected_msg), "args {args:?}: {stderr}");
    }
}

#[test]
fn shaders_rejects_unknown_options_missing_values_and_mistyped_values() {
    let dir = TempDir::new("badopts");
    let file = dir.path().join("plain.bin");
    std::fs::write(&file, b"not a shader").unwrap();

    let cases: &[(&[&str], &str)] = &[
        (&["--fnid", "foo"], "unknown option: --fnid"),
        (&["-x"], "unknown option: -x"),
        (&["stray"], "unexpected argument: stray"),
        (&["--top"], "--top needs a value"),
        (&["--top", "abc"], "invalid value for --top: abc"),
        (&["--compile", "nope"], "invalid value for --compile: nope"),
        (&["--windows", "four"], "invalid value for --windows: four"),
        (&["--split", "2"], "invalid value for --split: 2"),
        (&["--min-kb", "abc"], "invalid value for --min-kb: abc"),
        (&["--count", "abc"], "invalid value for --count: abc"),
        (&["--repeat", "abc"], "invalid value for --repeat: abc"),
        (&["--limit", "abc"], "invalid value for --limit: abc"),
        (&["--only"], "--only needs a value"),
        (&["--only", "--pipelines"], "--only needs a value"),
        (&["--example"], "--example needs a value"),
        (&["--direct", "--sample", "abc"], "invalid value for --sample: abc"),
        (&["--direct", "--seed", "abc"], "invalid value for --seed: abc"),
        (&["--direct", "--threads", "abc"], "invalid value for --threads: abc"),
        (&["--direct", "--max-bytes", "abc"], "invalid value for --max-bytes: abc"),
        (&["--direct", "--timeout", "abc"], "invalid value for --timeout: abc"),
        (&["--direct", "--hash"], "--hash needs a value"),
        (&["--direct", "--hash", "0xZZ"], "--hash:"),
        (&["--direct", "--dump"], "--dump needs a value"),
    ];

    for (args, expected_msg) in cases {
        let out = run_shaders(&file, args);
        assert_eq!(out.status.code(), Some(2), "args {args:?} unexpected status: {out:?}");
        let stderr = String::from_utf8(out.stderr).unwrap();
        assert!(stderr.contains("xcl shaders:"), "args {args:?}: {stderr}");
        assert!(stderr.contains(expected_msg), "args {args:?}: expected {expected_msg:?} in {stderr:?}");
        assert!(out.stdout.is_empty(), "args {args:?}: unexpected stdout {:?}", String::from_utf8_lossy(&out.stdout));
    }
}

#[test]
fn shaders_precompile_recovers_corrupt_same_length_cached_dxil() {
    let shader_dir = TempDir::new("precompile-corrupt");
    let cache_dir = TempDir::new("precompile-corrupt-cache");
    std::fs::write(shader_dir.path().join("ps.bin"), xbox_shader(0, &[fssm(&CODE, &[(5, 1)])])).unwrap();

    // 1. Cold precompile writes the cached DXIL container and warm index.
    let cold = run_shaders_with_cache(shader_dir.path(), &["--precompile"], cache_dir.path());
    assert!(cold.status.success(), "{cold:?}");
    let cold_stdout = String::from_utf8(cold.stdout).unwrap();
    assert!(cold_stdout.contains("compiled 1, already cached 0, errors 0"), "{cold_stdout}");

    let mut dxil_files = Vec::new();
    let mut index_files = Vec::new();
    for entry in std::fs::read_dir(cache_dir.path()).unwrap().flatten() {
        let name = entry.file_name();
        let s = name.to_string_lossy();
        if s.ends_with(".dxil") {
            dxil_files.push(entry.path());
        } else if s.starts_with("precompile-index-v1-") && s.ends_with(".bin") {
            index_files.push(entry.path());
        }
    }
    assert_eq!(dxil_files.len(), 1, "expected 1 .dxil file: {dxil_files:?}");
    assert_eq!(index_files.len(), 1, "expected 1 precompile index file: {index_files:?}");
    let dxil_path = dxil_files.pop().unwrap();
    let orig_bytes = std::fs::read(&dxil_path).unwrap();
    assert!(orig_bytes.starts_with(b"DXBC"), "unexpected DXIL header");

    // 2. Warm valid run hits the index without recompiling.
    let warm = run_shaders_with_cache(shader_dir.path(), &["--precompile"], cache_dir.path());
    assert!(warm.status.success(), "{warm:?}");
    let warm_stdout = String::from_utf8(warm.stdout).unwrap();
    assert!(warm_stdout.contains("compiled 0, already cached 1, errors 0"), "{warm_stdout}");
    assert_eq!(std::fs::read(&dxil_path).unwrap(), orig_bytes);

    // 3. Overwriting the cached DXIL with same-length zeros must be caught and restored.
    std::fs::write(&dxil_path, vec![0u8; orig_bytes.len()]).unwrap();
    let recovered = run_shaders_with_cache(shader_dir.path(), &["--precompile"], cache_dir.path());
    assert!(recovered.status.success(), "{recovered:?}");
    let recovered_stdout = String::from_utf8(recovered.stdout).unwrap();
    assert!(recovered_stdout.contains("compiled 1, already cached 0, errors 0"), "{recovered_stdout}");
    assert_eq!(std::fs::read(&dxil_path).unwrap(), orig_bytes);
}

#[test]
fn shaders_precompile_scopes_safe_barrier_and_bypasses_diagnostic_overrides() {
    let shader_dir = TempDir::new("precompile-scope");
    let cache_dir = TempDir::new("precompile-scope-cache");
    std::fs::write(shader_dir.path().join("ps.bin"), xbox_shader(0, &[fssm(&CODE, &[(5, 1)])])).unwrap();

    let count_indices = |dir: &Path| {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .filter(|e| {
                let name = e.file_name();
                let s = name.to_string_lossy();
                s.starts_with("precompile-index-v1-") && s.ends_with(".bin")
            })
            .count()
    };

    // 1. Default (XCL_SAFE_BARRIER unset) -> 1 index, then XCL_SAFE_BARRIER=1 / " 1 " hit the same scope.
    let cold = run_shaders_with_cache_and_env(shader_dir.path(), &["--precompile"], cache_dir.path(), &[]);
    assert!(cold.status.success(), "{cold:?}");
    let cold_stdout = String::from_utf8(cold.stdout).unwrap();
    assert!(cold_stdout.contains("compiled 1, already cached 0, errors 0"), "{cold_stdout}");
    assert_eq!(count_indices(cache_dir.path()), 1);

    for val in ["1", " 1 "] {
        let warm = run_shaders_with_cache_and_env(
            shader_dir.path(),
            &["--precompile"],
            cache_dir.path(),
            &[("XCL_SAFE_BARRIER", val)],
        );
        assert!(warm.status.success(), "{warm:?}");
        let warm_stdout = String::from_utf8(warm.stdout).unwrap();
        assert!(warm_stdout.contains("compiled 0, already cached 1, errors 0"), "{val:?}: {warm_stdout}");
        assert_eq!(count_indices(cache_dir.path()), 1);
    }

    // 2. XCL_SAFE_BARRIER=0 uses a separate scope; whitespace-padded " 0 " hits that same scope.
    let sb0 = run_shaders_with_cache_and_env(
        shader_dir.path(),
        &["--precompile"],
        cache_dir.path(),
        &[("XCL_SAFE_BARRIER", "0")],
    );
    assert!(sb0.status.success(), "{sb0:?}");
    assert_eq!(count_indices(cache_dir.path()), 2);

    let sb0_pad = run_shaders_with_cache_and_env(
        shader_dir.path(),
        &["--precompile"],
        cache_dir.path(),
        &[("XCL_SAFE_BARRIER", " 0 ")],
    );
    assert!(sb0_pad.status.success(), "{sb0_pad:?}");
    let sb0_pad_stdout = String::from_utf8(sb0_pad.stdout).unwrap();
    assert!(sb0_pad_stdout.contains("compiled 0, already cached 1, errors 0"), "{sb0_pad_stdout}");
    assert_eq!(count_indices(cache_dir.path()), 2);

    // 3. Diagnostic recomp overrides disable the warm record index cleanly.
    let diag = run_shaders_with_cache_and_env(
        shader_dir.path(),
        &["--precompile"],
        cache_dir.path(),
        &[("XCL_RECOMP_NO_STRUCTURED", "1")],
    );
    assert!(diag.status.success(), "{diag:?}");
    let diag_stdout = String::from_utf8(diag.stdout).unwrap();
    assert!(diag_stdout.contains("errors 0"), "{diag_stdout}");
    assert_eq!(count_indices(cache_dir.path()), 2);
}

#[test]
fn shaders_precompile_concurrent_processes_preserve_both_index_entries() {
    /// s_mov_b32 s0, s1 ; v_add_f32 v0, v2, v5 ; s_endpgm
    const CODE_B: [u8; 12] = [0x01, 0x03, 0x80, 0xbe, 0x02, 0x0b, 0x00, 0x06, 0x00, 0x00, 0x81, 0xbf];
    let dir_a = TempDir::new("concurrent-a");
    let dir_b = TempDir::new("concurrent-b");
    std::fs::write(dir_a.path().join("ps_a.bin"), xbox_shader(0, &[fssm(&CODE, &[(5, 1)])])).unwrap();
    std::fs::write(dir_b.path().join("ps_b.bin"), xbox_shader(0, &[fssm(&CODE_B, &[(5, 1)])])).unwrap();

    for trial in 0..3 {
        let cache_dir = TempDir::new(&format!("concurrent-cache-{trial}"));
        let child_a = Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("shaders")
            .arg(dir_a.path())
            .arg("--precompile")
            .env("XCL_DXIL_CACHE", cache_dir.path())
            .env_remove("XCL_DIRECT_DXIL")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn child a");
        let child_b = Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("shaders")
            .arg(dir_b.path())
            .arg("--precompile")
            .env("XCL_DXIL_CACHE", cache_dir.path())
            .env_remove("XCL_DIRECT_DXIL")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn child b");

        let out_a = child_a.wait_with_output().unwrap();
        let out_b = child_b.wait_with_output().unwrap();
        assert!(out_a.status.success(), "trial {trial} A: {out_a:?}");
        assert!(out_b.status.success(), "trial {trial} B: {out_b:?}");

        // Verify directly from disk BEFORE any serial revisit that both records survived in the index.
        let indices: Vec<PathBuf> = std::fs::read_dir(cache_dir.path())
            .unwrap()
            .flatten()
            .filter_map(|e| {
                let s = e.file_name().to_string_lossy().into_owned();
                (s.starts_with("precompile-index-v1-") && s.ends_with(".bin")).then(|| e.path())
            })
            .collect();
        assert_eq!(indices.len(), 1, "trial {trial}: expected 1 shared index, got {indices:?}");
        let raw = std::fs::read(&indices[0]).unwrap();
        let mut rest = raw.strip_prefix(b"XCLPIDX1").expect("valid index magic");
        let mut entry_hashes = Vec::new();
        while rest.len() >= 12 {
            let rec_hash = u64::from_le_bytes(rest[..8].try_into().unwrap());
            let count = u32::from_le_bytes(rest[8..12].try_into().unwrap()) as usize;
            let need = 12 + count * 8;
            assert!(rest.len() >= need, "trial {trial}: truncated index entry");
            entry_hashes.push(rec_hash);
            rest = &rest[need..];
        }
        assert!(rest.is_empty(), "trial {trial}: trailing bytes {}", rest.len());
        assert_eq!(entry_hashes.len(), 2, "trial {trial}: expected 2 index entries on disk before revisit: {entry_hashes:?}");

        let warm_a = run_shaders_with_cache(dir_a.path(), &["--precompile"], cache_dir.path());
        assert!(warm_a.status.success(), "trial {trial} warm_a: {warm_a:?}");
        let stdout_a = String::from_utf8(warm_a.stdout).unwrap();
        assert!(stdout_a.contains("compiled 0, already cached 1, errors 0"), "trial {trial} A: {stdout_a}");

        let warm_b = run_shaders_with_cache(dir_b.path(), &["--precompile"], cache_dir.path());
        assert!(warm_b.status.success(), "trial {trial} warm_b: {warm_b:?}");
        let stdout_b = String::from_utf8(warm_b.stdout).unwrap();
        assert!(stdout_b.contains("compiled 0, already cached 1, errors 0"), "trial {trial} B: {stdout_b}");
    }
}

#[test]
fn shaders_precompile_direct_dxil_warm_reports_cached_and_limit_skips_unselected_keys() {
    /// s_mov_b32 s0, s1 ; v_add_f32 v0, v2, v5 ; s_endpgm
    const CODE_B: [u8; 12] = [0x01, 0x03, 0x80, 0xbe, 0x02, 0x0b, 0x00, 0x06, 0x00, 0x00, 0x81, 0xbf];

    // 1. Direct DXIL graphics warm precompile reports `compiled 0, already cached 1`.
    let gfx_dir = TempDir::new("precompile-direct-gfx");
    let gfx_cache = TempDir::new("precompile-direct-gfx-cache");
    std::fs::write(gfx_dir.path().join("ps.bin"), xbox_shader(0, &[fssm(&CODE, &[(5, 1)])])).unwrap();

    let cold_direct = run_shaders_with_cache_and_env(
        gfx_dir.path(),
        &["--precompile"],
        gfx_cache.path(),
        &[("XCL_DIRECT_DXIL", "all")],
    );
    assert!(cold_direct.status.success(), "{cold_direct:?}");
    let cold_direct_stdout = String::from_utf8(cold_direct.stdout).unwrap();
    assert!(cold_direct_stdout.contains("compiled 1, already cached 0, errors 0"), "{cold_direct_stdout}");

    let warm_direct = run_shaders_with_cache_and_env(
        gfx_dir.path(),
        &["--precompile"],
        gfx_cache.path(),
        &[("XCL_DIRECT_DXIL", "all")],
    );
    assert!(warm_direct.status.success(), "{warm_direct:?}");
    let warm_direct_stdout = String::from_utf8(warm_direct.stdout).unwrap();
    assert!(warm_direct_stdout.contains("compiled 0, already cached 1, errors 0"), "{warm_direct_stdout}");

    // 2. Lazy per-key validation with --limit only checks selected records.
    //    Replace an unrelated `.dxil` with unreadable directory/invalid state and confirm --limit 1 hits cache cleanly.
    let limit_dir = TempDir::new("precompile-lazy");
    let limit_cache = TempDir::new("precompile-lazy-cache");
    std::fs::write(limit_dir.path().join("00_first.bin"), xbox_shader(0, &[fssm(&CODE, &[(5, 1)])])).unwrap();
    std::fs::write(limit_dir.path().join("01_second.bin"), xbox_shader(0, &[fssm(&CODE_B, &[(5, 1)])])).unwrap();

    let cold_all = run_shaders_with_cache(limit_dir.path(), &["--precompile"], limit_cache.path());
    assert!(cold_all.status.success(), "{cold_all:?}");
    let cold_all_stdout = String::from_utf8(cold_all.stdout).unwrap();
    assert!(cold_all_stdout.contains("compiled 2, already cached 0, errors 0"), "{cold_all_stdout}");

    let warm_limit = run_shaders_with_cache(limit_dir.path(), &["--precompile", "--limit", "1"], limit_cache.path());
    assert!(warm_limit.status.success(), "{warm_limit:?}");
    let warm_limit_stdout = String::from_utf8(warm_limit.stdout).unwrap();
    assert!(warm_limit_stdout.contains("compiled 0, already cached 1, errors 0"), "{warm_limit_stdout}");
}
