//! End to end without a game executable: a console compute shader in the retail container
//! layout (DXBC → DXIL part with XDAT → XBCH → FSSM, CS stage descriptor) goes through the
//! D3D12.X backend's CreateComputePipelineState hook as a title would hand it over →
//! recompiled → D3D12 pipeline on the adapter → dispatched → memory read back and checked.
//!
//! The program is wave64, as console compute programs are: on an adapter without 64-lane
//! waves (NVIDIA) the backend has to run it as two wave32 halves, or the driver refuses the
//! pipeline. The pipeline counters are checked, so the test cannot pass on a fallback path.

use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{pso_async, real};

static SERIAL: Mutex<()> = Mutex::new(());

fn le(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

/// Console CS container: DXBC with one DXIL part (program kind CS, no bitcode) whose XDAT
/// holds XBCH (zlib) → FSSM with `code` and a CS stage descriptor: COMPUTE_PGM_RSRC2
/// `rsrc2` at +0x38, numthreads at +0x3C.
fn console_cs(code: &[u8], rsrc2: u32, threads: [u32; 3]) -> Vec<u8> {
    console_cs_flags(code, rsrc2, threads, 0)
}

/// `console_cs` with the FSSM header's flags word (+0xD4; bit 5 = wave32).
fn console_cs_flags(code: &[u8], rsrc2: u32, threads: [u32; 3], flags: u32) -> Vec<u8> {
    let code_at = 0x308usize;
    let mut fssm = vec![0u8; code_at];
    fssm[0..4].copy_from_slice(b"FSSM");
    fssm[4..8].copy_from_slice(&0x101u32.to_le_bytes());
    fssm[0xC0..0xC8].copy_from_slice(&((code_at - 0x10) as u64).to_le_bytes());
    fssm[0xC8..0xD0].copy_from_slice(&(code.len() as u64).to_le_bytes());
    fssm[0xD4..0xD8].copy_from_slice(&flags.to_le_bytes());
    fssm[0x100..0x108].copy_from_slice(&0x0000_00C0_0002_0106u64.to_le_bytes()); // CS, API stage 1
    for (k, x) in [rsrc2, threads[0], threads[1], threads[2]].into_iter().enumerate() {
        fssm[0x138 + 4 * k..0x13C + 4 * k].copy_from_slice(&x.to_le_bytes());
    }
    fssm.extend_from_slice(code);
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

const V: u64 = 256; // VALU source code of v0

/// Make several guarded arithmetic blocks so the same shader has both whole and split forms.
fn guarded_adds() -> Vec<u8> {
    let mut code = Vec::new();
    let mut op = |name, enc, fields: &[(&str, u64)]| code.extend(encode(name, enc, fields, None).unwrap());
    op("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)]);
    op("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)]);
    for add in 1..=5 {
        op("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128 + add)]);
        op("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 2)]);
        op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 128 + add), ("VSRC1", 2)]);
        op("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 135), ("VSRC1", 2)]);
    }
    op("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 130), ("VSRC1", 0)]);
    op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 0), ("VSRC1", 4)]);
    op("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 128)]);
    op("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 4), ("DATA", 2), ("SADDR", 0x7d)]);
    op("S_ENDPGM", "ENC_SOPP", &[]);
    code
}

#[test]
fn split_compute_prefers_the_offline_whole_pipeline() {
    use std::{fs, process::Command};
    let blob = console_cs(&guarded_adds(), 2 << 1 | 1 << 7, [32,1,1]);
    if let Ok(case) = std::env::var("XCL_TEST_WHOLE_CASE") {
        assert_eq!(xcl::hle::d3d12x::backend(), xcl::hle::d3d12x::Backend::Real);
        let raw = real::device_raw();
        let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap();
        let compiled = xcl::gpu::recomp::pso::console_to_dxil_for(&blob, xcl::gpu::recomp::pso::Stage::Compute, 0, true).unwrap();
        assert!(compiled.dispatch_chain.is_some());
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: blob.as_ptr().cast(), BytecodeLength: blob.len(),
        }, ..Default::default() };
        let pso: ID3D12PipelineState = unsafe { dev.CreateComputePipelineState(&desc) }.unwrap();
        let resolved = pso_async::resolve_wait(pso.as_raw() as usize);
        assert!(real::is_translated_compute(resolved));
        let rs = real::pso_root_signature(resolved).unwrap() as *mut std::ffi::c_void;
        assert_eq!(Some(rs) == real::abi_rs_cached(), case == "hit", "selected root signature");
        if case == "hit" {
            let raw = resolved as *mut std::ffi::c_void;
            let real = unsafe { ID3D12PipelineState::from_raw_borrowed(&raw) }.unwrap();
            let rs = unsafe { ID3D12RootSignature::from_raw_borrowed(&rs) }.unwrap();
            let mut expected = vec![0xdeadbeef; 64];
            let out = runner::dispatch_on(dev, real, rs, [1,1,1], &[32,3], &expected).unwrap();
            for lane in 0..32 {
                expected[8+lane] = [1,2,4,5].into_iter().fold(lane as u32, |v, add| (v + add) ^ 7);
            }
            assert_eq!(out, expected);
        }
        return;
    }
    if xcl::gpu::recomp::dxc::location().is_err() || !runner::wave32_supported().unwrap_or(false) {
        return runner::skip_test("requires DXC and a wave32 GPU");
    }
    let dir = std::env::temp_dir().join(format!("xcl-guarded-adds-{}-{}", std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    let input = dir.join("input");
    let cache = dir.join("cache");
    fs::create_dir_all(&input).unwrap();
    fs::write(input.join("guarded.bin"), blob).unwrap();
    let run = |cmd: &mut Command| {
        let out = cmd.env("XCL_DIRECT_DXIL","1")
            .env("XCL_DXIL_CACHE",&cache).env("XCL_PSO_DISK_CACHE","1").output().unwrap();
        assert!(out.status.success(), "{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    };
    // Seed the ordinary whole PSO offline, then force the runtime to choose between it and a split chain.
    run(Command::new(env!("CARGO_BIN_EXE_xcl")).arg("shaders").arg(&input).args(["--precompile","--pipelines"])
        .env("XCL_DIRECT_SPLIT_INSTRUCTIONS","12500"));
    let mut files = fs::read_dir(cache.join("driver")).unwrap();
    let blob = files.next().unwrap().unwrap().path();
    assert!(files.next().is_none() && blob.metadata().unwrap().len() > 0);
    for case in ["hit","stale","miss"] {
        if case == "stale" { fs::write(&blob, b"rejected driver cache fixture").unwrap(); }
        run(Command::new(std::env::current_exe().unwrap()).args(["--exact","split_compute_prefers_the_offline_whole_pipeline","--nocapture"])
            .env("XCL_DIRECT_SPLIT_INSTRUCTIONS","12").env("XCL_TEST_WHOLE_CASE",case));
        assert_eq!(blob.exists(), case == "hit");
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn offline_compute_pipeline_is_reused_by_a_second_process() {
    use std::{fs, process::Command};
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if xcl::gpu::recomp::dxc::location().is_err() {
        runner::skip_test("no DXC");
        return;
    }
    if runner::wave64_programs_supported() != Ok(true) {
        runner::skip_test("no GPU with 32- or 64-lane waves");
        return;
    }
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("xcl-offline-pso-{}-{nonce}", std::process::id()));
    let input = dir.join("input");
    let cache = dir.join("cache");
    fs::create_dir_all(&input).unwrap();
    fs::write(input.join("kernel.bin"), console_cs(&program(), 2 << 1 | 1 << 7 | 1 << 8 | 1 << 11, [16, 8, 1])).unwrap();
    let run = |enabled: bool| {
        let out = Command::new(env!("CARGO_BIN_EXE_xcl")).arg("shaders").arg(&input)
            .args(["--precompile", "--pipelines"]).env("XCL_DXIL_CACHE", &cache)
            .env("TMP", &dir).env("TEMP", &dir).env("XCL_DXIL_DUMP", "all")
            .env("XCL_PSO_DISK_CACHE", if enabled { "1" } else { "0" }).output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(out.status.success() && text.contains("pipelines made on the GPU 1, refused 0"), "{text}\n{}", String::from_utf8_lossy(&out.stderr));
        text
    };
    run(false);
    assert!(!cache.join("driver").exists());
    run(true);
    let direct = dir.join("xcl-dxil-dump");
    let dxil_dir = if direct.is_dir() { &direct } else { &cache };
    let dxil_file = fs::read_dir(dxil_dir).unwrap().map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|ext| ext == "dxil")).expect("compiled fixture DXIL");
    let dxil = fs::read(dxil_file).unwrap();
    let name = format!("{:016x}-{:x}-0.pso", xcl::gpu::shader::fnv64(&dxil), dxil.len());
    let driver_file = cache.join("driver").join(name);
    assert!(driver_file.metadata().unwrap().len() > 0);
    let warm = run(true);
    // A blob from an earlier driver can be refused. The normal build must still succeed.
    fs::write(&driver_file, b"invalid cached pipeline").unwrap();
    let stale = run(true);
    assert!(stale.contains("pipelines loaded from the driver cache 0"), "{stale}");
    let repaired = run(true);
    assert!(repaired.contains("pipelines loaded from the driver cache 1"), "{repaired}");
    let _ = fs::remove_dir_all(&dir);
    assert!(warm.contains("pipelines loaded from the driver cache 1"), "{warm}");
}

/// Wave32 compute program with conditional branches across distinct basic blocks so direct
/// DXIL splitting (`XCL_DIRECT_SPLIT_INSTRUCTIONS`) partitions it into multiple pieces.
fn split_program() -> Vec<u8> {
    let list: &[(&str, &str, &[(&str, u64)], Option<u32>)] = &[
        // Mark wave32 (`detect_wave64` -> false) so `[32, 1, 1]` splits on every wave32 GPU.
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 6)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 129), ("VSRC1", 2)], None),
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 131), ("VSRC1", 2)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 133), ("VSRC1", 2)], None),
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 135), ("VSRC1", 2)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 137), ("VSRC1", 2)], None),
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 139), ("VSRC1", 2)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 2), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 6)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 141), ("VSRC1", 2)], None),
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 143), ("VSRC1", 2)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 145), ("VSRC1", 2)], None),
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 147), ("VSRC1", 2)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 149), ("VSRC1", 2)], None),
        ("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 151), ("VSRC1", 2)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 0), ("VSRC1", 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 4), ("DATA", 2), ("SADDR", 0x7D)], None),
        ("S_ENDPGM", "ENC_SOPP", &[], None),
    ];
    list.iter().flat_map(|(n, e, f, l)| encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}"))).collect()
}

#[test]
fn offline_split_compute_pipeline_is_reused_by_a_second_process() {
    use std::{fs, process::Command};
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if xcl::gpu::recomp::dxc::location().is_err() {
        runner::skip_test("no DXC");
        return;
    }
    if !runner::wave32_supported().unwrap_or(false) {
        runner::skip_test("no wave32 GPU");
        return;
    }
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("xcl-offline-split-pso-{}-{nonce}", std::process::id()));
    let input = dir.join("input");
    let cache = dir.join("cache");
    fs::create_dir_all(&input).unwrap();
    fs::write(input.join("split.bin"), console_cs(&split_program(), 2 << 1 | 1 << 7, [32, 1, 1])).unwrap();
    let run = || {
        let out = Command::new(env!("CARGO_BIN_EXE_xcl"))
            .arg("shaders")
            .arg(&input)
            .args(["--precompile", "--pipelines", "--windows", "1"])
            .env("XCL_DXIL_CACHE", &cache)
            .env("XCL_DIRECT_DXIL", "1")
            .env("XCL_DIRECT_SPLIT_INSTRUCTIONS", "12")
            .env("XCL_PSO_DISK_CACHE", "1")
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(out.status.success(), "{text}\n{}", String::from_utf8_lossy(&out.stderr));
        text
    };
    let cold = run();
    let mut files: Vec<_> = fs::read_dir(cache.join("driver")).unwrap().flatten().map(|e| e.path()).collect();
    files.sort();
    let pieces = files.len();
    assert!(pieces >= 2, "expected >= 2 split piece blobs in driver cache, got {files:?}\n{cold}");
    assert!(
        files.iter().all(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(".chain.pso"))),
        "split compute precompile must write only .chain.pso files, got {files:?}"
    );
    assert!(files.iter().all(|p| p.metadata().unwrap().len() > 0));
    assert!(cold.contains(&format!("pipelines made on the GPU {pieces}, refused 0")), "{cold}");
    assert!(cold.contains(&format!("split compute programs: 1 ({pieces} pieces)")), "{cold}");

    let warm = run();
    assert!(warm.contains(&format!("pipelines loaded from the driver cache {pieces}")), "{warm}");

    // A corrupt .chain.pso is rejected and rebuilt cleanly while sibling pieces stay reused.
    fs::write(&files[0], b"invalid cached chain pipeline").unwrap();
    let recovered = run();
    assert!(recovered.contains(&format!("pipelines made on the GPU {pieces}, refused 0")), "{recovered}");
    assert!(recovered.contains(&format!("pipelines loaded from the driver cache {}", pieces - 1)), "{recovered}");
    assert!(files[0].metadata().unwrap().len() > b"invalid cached chain pipeline".len() as u64);
    let _ = fs::remove_dir_all(&dir);
}

/// 16×8 threads per group (two wave64s), s[0:1] = output base and row width, TGID_X/Y in
/// s2/s3, thread ids in v0/v1: each thread writes its global (x, y), and through LDS +
/// s_barrier the (x, y) of the thread at the mirrored index, which is in the other wave.
fn program() -> Vec<u8> {
    let list: &[(&str, &str, &[(&str, u64)], Option<u32>)] = &[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 132), ("VSRC1", 1)], None), // y · 16
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 2), ("VSRC1", 0)], None), // index in group
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 2), ("SSRC1", 132)], None), // group x · 16
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 4), ("VSRC1", 0)], None), // global x
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 3), ("SSRC1", 131)], None), // group y · 8
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 5), ("VSRC1", 1)], None), // global y
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 144), ("VSRC1", 4)], None),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", V + 5), ("VSRC1", 3)], None), // x | y << 16
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 2)], None),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 6), ("DATA0", 5)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 255), ("VSRC1", 2)], Some(127)), // mirrored thread
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 130), ("VSRC1", 7)], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 12), ("ADDR", 7)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 5)], None),
        ("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 4), ("SRC1", 1)], None), // y · width
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", V + 9), ("VSRC1", 3)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 131), ("VSRC1", 9)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 0), ("VSRC1", 9)], None), // + output base
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORDX2", "ENC_FLAT_GLBL", &[("ADDR", 9), ("DATA", 11), ("SADDR", 0x7D)], None),
        ("S_ENDPGM", "ENC_SOPP", &[], None),
    ];
    list.iter().flat_map(|(n, e, f, l)| encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}"))).collect()
}

#[test]
fn console_wave64_compute_through_the_pso_hook_dispatch_and_read_back() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if xcl::gpu::recomp::dxc::location().is_err() {
        println!("no DXC – skipping");
        runner::skip_test("no DXC");
        return;
    }
    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real {
        println!("no real D3D12 device – skipping");
        runner::skip_test("no GPU");
        return;
    }
    // 32- or 64-lane waves (not WARP / the Basic Render Driver: 4): the driver rejects the
    // translated program otherwise
    if runner::wave64_programs_supported() != Ok(true) {
        println!("no 32- or 64-lane waves – skipping");
        runner::skip_test("no wave64");
        return;
    }
    let raw = real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap().clone();
    println!("backend device: {}", real::stats().adapter);
    mirror_program_through_the_hook(&dev);
}

/// Hand the wave64 mirror program to the CreateComputePipelineState hook, dispatch the
/// pipeline it built and check every thread's output.
fn mirror_program_through_the_hook(dev: &ID3D12Device) {
    // user SGPRs 2, TGID_X + TGID_Y, thread ids v0 + v1
    let blob = console_cs(&program(), 2 << 1 | 1 << 7 | 1 << 8 | 1 << 11, [16, 8, 1]);
    assert!(xcl::gpu::xbox::is_xbox(&blob), "expected a console container");
    let game_rs: ID3D12RootSignature = unsafe {
        let mut ser = None;
        D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).unwrap();
        let ser = ser.unwrap();
        dev.CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize())).unwrap()
    };
    let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(game_rs)),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: blob.as_ptr() as _, BytecodeLength: blob.len() },
        ..Default::default()
    };
    let before = real::stats();
    let pso: Result<ID3D12PipelineState, _> = unsafe { dev.CreateComputePipelineState(&desc) };
    unsafe { std::mem::ManuallyDrop::drop(&mut desc.pRootSignature) };
    let pso = pso.expect("CreateComputePipelineState through the backend hook");
    // built in the background behind a stand-in when its shader was not compiled yet
    let real_raw = xcl::hle::d3d12x::pso_async::resolve_wait(pso.as_raw() as usize) as *mut std::ffi::c_void;
    let after = real::stats();
    assert_eq!(after.translated_pipelines, before.translated_pipelines + 1, "PSO was not built from the translated shader: {:?}", after.translation_errors);
    assert!(
        real::is_translated_compute(real_raw as usize),
        "resolved compute PSO must be registered in TRANSLATED_COMPUTE"
    );
    let pso = unsafe { ID3D12PipelineState::from_raw_borrowed(&real_raw) }.unwrap().clone();
    let rs_raw = real::pso_root_signature(pso.as_raw() as usize).expect("root signature of the translated PSO") as *mut std::ffi::c_void;
    let rs = unsafe { ID3D12RootSignature::from_raw_borrowed(&rs_raw) }.unwrap().clone();

    let (w, h, base) = (32usize, 24usize, 0x100usize);
    let mut io = vec![0u32; 128];
    io[0] = base as u32;
    io[1] = w as u32;
    let out = runner::dispatch_on(&dev, &pso, &rs, [2, 3, 1], &io, &vec![0u32; (base + w * h * 8) / 4]).expect("dispatch");
    for y in 0..h {
        for x in 0..w {
            let at = (base + (y * w + x) * 8) / 4;
            assert_eq!(out[at], (x | y << 16) as u32, "({x}, {y}): own value");
            let (gx, gy, i) = (x / 16, y / 8, (y % 8) * 16 + x % 16);
            let (px, py) = (gx * 16 + (127 - i) % 16, gy * 8 + (127 - i) / 16);
            assert_eq!(out[at + 1], (px | py << 16) as u32, "({x}, {y}): value of thread {} through LDS", 127 - i);
        }
    }
}

/// A driver that refuses a pipeline with heap windows (E_OUTOFMEMORY, as NVIDIA 616.92 does
/// with some big retail programs, #156) gets the program again in the switch form, and the
/// pipeline from it runs correctly. The refusals are injected (`real::inject_heap_window_oom`).
#[test]
fn heap_window_out_of_memory_falls_back_to_the_switch_form() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if xcl::gpu::recomp::dxc::location().is_err() {
        println!("no DXC – skipping");
        runner::skip_test("no DXC");
        return;
    }
    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real {
        println!("no real D3D12 device – skipping");
        runner::skip_test("no GPU");
        return;
    }
    let raw = real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap().clone();
    if xcl::gpu::recomp::pso::mem_windows() & xcl::gpu::recomp::HEAP_WINDOWS == 0 {
        println!("no heap windows (opt-in: XCL_HEAP_WINDOWS=1; or no shared heap) – skipping");
        runner::skip_test("no heap windows");
        return;
    }
    struct StopInjecting;
    impl Drop for StopInjecting {
        fn drop(&mut self) {
            real::inject_heap_window_oom(0);
        }
    }
    let _stop = StopInjecting;
    // every heap-window build fails, also the background one behind a stand-in
    real::inject_heap_window_oom(u64::MAX);
    let before = real::switch_form_retries();
    mirror_program_through_the_hook(&dev);
    assert!(real::switch_form_retries() > before, "the pipeline was not rebuilt in the switch form");
}

/// Cold DXIL cache: `CreateComputePipelineState` returns a stand-in immediately and compiles
/// the real compute PSO on the background worker. The real PSO (and NOT the stand-in) must be
/// registered in `TRANSLATED_COMPUTE` once the background build completes.
#[test]
fn async_compute_pso_registers_real_pso_and_not_standin() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if xcl::gpu::recomp::dxc::location().is_err() {
        println!("no DXC – skipping");
        runner::skip_test("no DXC");
        return;
    }
    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real {
        println!("no real D3D12 device – skipping");
        runner::skip_test("no GPU");
        return;
    }
    // 32- or 64-lane waves (not WARP / the Basic Render Driver: 4): the driver rejects the
    // translated program otherwise
    if runner::wave64_programs_supported() != Ok(true) {
        println!("no 32- or 64-lane waves – skipping");
        runner::skip_test("no wave64");
        return;
    }
    if !pso_async::enabled() {
        println!("XCL_ASYNC_PSO=0 – skipping");
        return;
    }

    struct ColdCache(std::path::PathBuf, Option<std::ffi::OsString>);
    impl Drop for ColdCache {
        fn drop(&mut self) {
            match &self.1 {
                Some(prev) => std::env::set_var("XCL_DXIL_CACHE", prev),
                None => std::env::remove_var("XCL_DXIL_CACHE"),
            }
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let cold = ColdCache(
        std::env::temp_dir().join(format!("xcl-async-compute-pso-{}", std::process::id())),
        std::env::var_os("XCL_DXIL_CACHE"),
    );
    let _ = std::fs::remove_dir_all(&cold.0);
    std::env::set_var("XCL_DXIL_CACHE", &cold.0);

    let raw = real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap().clone();

    // Distinct minimal console CS so neither disk nor in-memory DXIL cache already holds it.
    let code: Vec<u8> = [
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 0)][..], None),
        ("S_ENDPGM", "ENC_SOPP", &[][..], None),
    ]
    .iter()
    .flat_map(|(n, e, f, l)| encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")))
    .collect();
    let blob = console_cs(&code, 2 << 1 | 1 << 7, [8, 8, 1]);

    let game_rs: ID3D12RootSignature = unsafe {
        let mut ser = None;
        D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).unwrap();
        let ser = ser.unwrap();
        dev.CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize())).unwrap()
    };
    let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(game_rs)),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: blob.as_ptr() as _, BytecodeLength: blob.len() },
        ..Default::default()
    };
    let pso: Result<ID3D12PipelineState, _> = unsafe { dev.CreateComputePipelineState(&desc) };
    unsafe { std::mem::ManuallyDrop::drop(&mut desc.pRootSignature) };
    let pso = pso.expect("CreateComputePipelineState stand-in");
    let standin_raw = pso.as_raw() as usize;

    assert!(!real::is_translated_compute(standin_raw), "stand-in compute PSO must not be registered in TRANSLATED_COMPUTE");
    let real_raw = pso_async::resolve_wait(standin_raw);
    assert_ne!(real_raw, standin_raw, "cold DXIL cache must build compute PSO asynchronously behind a stand-in");
    assert!(real::is_translated_compute(real_raw), "background-built real compute PSO must be registered in TRANSLATED_COMPUTE");
    assert!(!real::is_translated_compute(standin_raw), "stand-in must remain unregistered after background build");
    assert!(real::pso_root_signature(real_raw).is_some(), "real compute PSO must have an ABI root signature");

    // Releasing the stand-in must release and forget the background-built real PSO too.
    drop(pso);
    assert!(!real::is_translated_compute(real_raw), "forgetting released async compute PSO must clean up TRANSLATED_COMPUTE");
}

/// A compute program without an EXEC write has nothing in its code that tells its wave size:
/// the FSSM header's wave32 flag does. 64 threads each store `v_readfirstlane` of the thread
/// id: in 32-lane waves threads 32..63 read thread 32, in a 64-lane wave all read thread 0.
#[test]
fn console_compute_without_an_exec_write_has_the_wave_size_of_its_header() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if xcl::gpu::recomp::dxc::location().is_err() {
        runner::skip_test("no DXC");
        return;
    }
    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real {
        runner::skip_test("no GPU");
        return;
    }
    if runner::wave64_programs_supported() != Ok(true) {
        runner::skip_test("no wave64");
        return;
    }
    let raw = real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.unwrap().clone();
    let list: &[(&str, &str, &[(&str, u64)], Option<u32>)] = &[
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V)], None), // s4 = first thread of the wave
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 131), ("VSRC1", 0)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 0), ("VSRC1", 9)], None), // + output base
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORDX2", "ENC_FLAT_GLBL", &[("ADDR", 9), ("DATA", 11), ("SADDR", 0x7D)], None),
        ("S_ENDPGM", "ENC_SOPP", &[], None),
    ];
    let code: Vec<u8> = list.iter().flat_map(|(n, e, f, l)| encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}"))).collect();
    assert!(xcl::gpu::recomp::detect_wave64(&code), "the code alone reads as wave64");
    for (flags, upper) in [(0u32, 0u32), (0x20, 32)] {
        // user SGPRs 2, thread ids in v0
        let blob = console_cs_flags(&code, 2 << 1 | 1 << 11, [64, 1, 1], flags);
        let game_rs: ID3D12RootSignature = unsafe {
            let mut ser = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).unwrap();
            let ser = ser.unwrap();
            dev.CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize())).unwrap()
        };
        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(game_rs)),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: blob.as_ptr() as _, BytecodeLength: blob.len() },
            ..Default::default()
        };
        let pso: Result<ID3D12PipelineState, _> = unsafe { dev.CreateComputePipelineState(&desc) };
        unsafe { std::mem::ManuallyDrop::drop(&mut desc.pRootSignature) };
        let pso = pso.expect("CreateComputePipelineState through the backend hook");
        let real_raw = pso_async::resolve_wait(pso.as_raw() as usize) as *mut std::ffi::c_void;
        assert!(real::is_translated_compute(real_raw as usize), "flags {flags:#x}: not built from the translated shader: {:?}", real::stats().translation_errors);
        let pso = unsafe { ID3D12PipelineState::from_raw_borrowed(&real_raw) }.unwrap().clone();
        let rs_raw = real::pso_root_signature(pso.as_raw() as usize).expect("root signature of the translated PSO") as *mut std::ffi::c_void;
        let rs = unsafe { ID3D12RootSignature::from_raw_borrowed(&rs_raw) }.unwrap().clone();
        let base = 0x100usize;
        let mut io = vec![0u32; 128];
        io[0] = base as u32;
        let out = runner::dispatch_on(&dev, &pso, &rs, [1, 1, 1], &io, &vec![0xFFFF_FFFFu32; (base + 64 * 8) / 4]).expect("dispatch");
        for t in 0..64usize {
            let at = base / 4 + t * 2;
            assert_eq!((out[at], out[at + 1]), (if t < 32 { 0 } else { upper }, t as u32), "flags {flags:#x}, thread {t}");
        }
    }
}

/// Split-eligible wave32 shader writing each lane's guarded arithmetic result to 64-bit UMA
/// address `(s4, s5)` with branch selector in `s6` (console inline root constants start at `s4`).
fn guarded_adds_64() -> Vec<u8> {
    let mut code = Vec::new();
    let mut op = |name, enc, fields: &[(&str, u64)]| code.extend(encode(name, enc, fields, None).unwrap());
    op("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)]);
    op("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)]);
    for add in 1..=5 {
        op("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 6), ("SSRC1", 128 + add)]);
        op("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 2)]);
        op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 128 + add), ("VSRC1", 2)]);
        op("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 135), ("VSRC1", 2)]);
    }
    op("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 130), ("VSRC1", 0)]);
    op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 4), ("VSRC1", 4)]);
    op("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 5)]);
    op("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 4), ("DATA", 2), ("SADDR", 0x7d)]);
    op("S_ENDPGM", "ENC_SOPP", &[]);
    code
}

/// Distinct split-eligible sibling wave32 shader writing to 64-bit UMA address `(s4, s5)` with
/// branch selectors `s6` (root constant) and `s7` (`TGID_X`).
fn split_program_64() -> Vec<u8> {
    let mut code = Vec::new();
    let mut op = |name, enc, fields: &[(&str, u64)]| code.extend(encode(name, enc, fields, None).unwrap());
    op("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)]);
    op("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)]);
    op("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 6), ("SSRC1", 128)]);
    op("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 6)]);
    op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 129), ("VSRC1", 2)]);
    op("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 131), ("VSRC1", 2)]);
    op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 133), ("VSRC1", 2)]);
    op("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 135), ("VSRC1", 2)]);
    op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 137), ("VSRC1", 2)]);
    op("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 139), ("VSRC1", 2)]);
    op("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 7), ("SSRC1", 128)]);
    op("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 6)]);
    op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 141), ("VSRC1", 2)]);
    op("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 143), ("VSRC1", 2)]);
    op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 145), ("VSRC1", 2)]);
    op("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 147), ("VSRC1", 2)]);
    op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 149), ("VSRC1", 2)]);
    op("V_XOR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 151), ("VSRC1", 2)]);
    op("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 130), ("VSRC1", 0)]);
    op("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 4), ("VSRC1", 4)]);
    op("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 5)]);
    op("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 4), ("DATA", 2), ("SADDR", 0x7D)]);
    op("S_ENDPGM", "ENC_SOPP", &[]);
    code
}

/// #2741 Steps 2–4 end-to-end proof: selective offline replay warms ONLY the demanded
/// `.chain.pso` whole blob (distinct from ordinary `.pso`, split pieces, and unselected sibling),
/// and fresh processes exercising both the real runtime indirect (`ExecuteIndirect`) and direct
/// fallback (`Dispatch` without recording allocator) paths reuse that exact blob with zero cold
/// retry (`hits == 1, stale == 0`) while producing exact GPU dispatch output.
#[test]
fn selective_chain_whole_replay_reuses_blob_in_fresh_process() {
    use std::{ffi::c_void, fs, path::PathBuf, process::Command};

    let selected_blob = console_cs(&guarded_adds_64(), 7 << 1 | 1 << 7, [32, 1, 1]);
    let sibling_blob = console_cs(&split_program_64(), 7 << 1 | 1 << 7, [32, 1, 1]);

    if let Ok(case) = std::env::var("XCL_TEST_CHAIN_REPLAY_CASE") {
        if case == "seed_capture" {
            let out_path = PathBuf::from(std::env::var_os("XCL_TEST_CAPTURE_OUT").expect("capture out path"));
            let compiled = xcl::gpu::recomp::pso::console_to_dxil_in(
                &selected_blob,
                xcl::gpu::recomp::pso::Stage::Compute,
                0,
                true,
                4,
            )
            .expect("translate selected split compute shader");
            let chain = compiled.dispatch_chain.as_deref().expect("selected shader must split into a dispatch chain");
            assert!(chain.pieces.len() >= 2, "expected >= 2 split pieces");

            // Confirm ordinary whole, chain whole, and every chain piece have distinct disk names.
            let dxil_hash = xcl::gpu::shader::fnv64(&compiled.dxil);
            let ord_whole_name = format!("{dxil_hash:016x}-{:x}-0.pso", compiled.dxil.len());
            let chain_whole_name = format!("{dxil_hash:016x}-{:x}-0.chain.pso", compiled.dxil.len());
            assert_ne!(ord_whole_name, chain_whole_name);
            for piece in &chain.pieces {
                let piece_name = format!("{:016x}-{:x}-0.chain.pso", xcl::gpu::shader::fnv64(piece), piece.len());
                assert_ne!(piece_name, chain_whole_name);
                assert_ne!(piece_name, ord_whole_name);
            }
            assert!(compiled.samplers.is_empty(), "synthetic guarded_adds_64 uses no static samplers");

            // Write schema_version=1 capture sidecar matching Part 1 (`chain_whole_capture.rs`).
            let rec_hash = xcl::gpu::shader::fnv64(&selected_blob);
            let header = serde_json::json!({
                "schema_version": 1,
                "kind": "header",
                "xcl_version": "0.1.0",
                "translator_source_id": "0000000000000000",
                "direct_dxil_source_id": "0000000000000000",
                "direct_dxil_translator_version": 9,
                "per_lane_source_id": "0000000000000000",
                "chain_rs_abi_version": 1,
                "sequence_cutoff": 64,
                "captured_at_unix_ms": 1_700_000_000_000u64,
                "adapter": null,
                "agility_runtime": { "status": "not_found", "version": null },
                "dxc": {
                    "dxcompiler": { "file_version": "", "major": 0, "minor": 0, "commit": "" },
                    "dxil": { "file_version": "", "major": 0, "minor": 0, "commit": "" }
                },
                "translation_options": {
                    "direct_dxil": true,
                    "direct_compact": null,
                    "direct_compact_nest": null,
                    "per_lane": false,
                    "safe_barrier": true,
                    "heap_windows": false,
                    "split_wave64_forced": true,
                    "dispatch_piece_limit": 12,
                    "dxil_no_opt": [],
                    "x64_all_masks": false,
                    "wave64_spin": true,
                    "recomp_structured": true
                }
            });
            let demand = serde_json::json!({
                "schema_version": 1,
                "kind": "demand",
                "seq": 1,
                "elapsed_ms": 1,
                "timestamp_unix_ms": 1_700_000_000_001u64,
                "present_count": 1,
                "reason": "indirect",
                "already_ready": false,
                "source_record_hash": format!("{rec_hash:016x}"),
                "whole_dxil_hash": format!("{dxil_hash:016x}"),
                "whole_dxil_len": compiled.dxil.len(),
                "effective_flags": 0,
                "node_mask": 0,
                "chain_rs_abi_version": 1,
                "chain_rs_id": "73fbfe4eb45ed583",
                "static_samplers": [],
                "disk_cache_name": chain_whole_name,
                "mem_windows": 4,
                "split_wave64": true
            });
            fs::write(&out_path, format!("{header}\n{demand}\n")).expect("write capture sidecar");
            return;
        }

        // Fresh runtime child exercising hooked ExecuteIndirect or direct fallback (allocator == 0).
        assert_eq!(xcl::hle::d3d12x::backend(), xcl::hle::d3d12x::Backend::Real);
        let raw = real::device_raw();
        let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.expect("real D3D12 device");
        let uma = xcl::gpu::uma::get().expect("UMA open in child process");
        assert_eq!(xcl::gpu::recomp::pso::mem_windows(), 4, "expected default 4-window runtime UMA layout");

        let is_sibling = case == "unselected_sibling";
        let blob = if is_sibling { &sibling_blob } else { &selected_blob };

        // 3-constant root signature (s0 = output addr lo, s1 = output addr hi, s2 = shader param).
        let param = D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 3,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: 1,
            pParameters: &param,
            ..Default::default()
        };
        let game_rs: ID3D12RootSignature = unsafe {
            let mut ser = None;
            D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).expect("serialize RS");
            let ser = ser.unwrap();
            dev.CreateRootSignature(
                0,
                std::slice::from_raw_parts(ser.GetBufferPointer().cast::<u8>(), ser.GetBufferSize()),
            )
            .expect("CreateRootSignature through hook")
        };

        // Create split compute PSO synchronously (`XCL_ASYNC_PSO=0`): builds pieces (uncached),
        // leaving whole form deferred until fallback.
        let mut pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(game_rs.clone())),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: blob.as_ptr().cast(), BytecodeLength: blob.len() },
            ..Default::default()
        };
        let pso: ID3D12PipelineState = unsafe { dev.CreateComputePipelineState(&pso_desc) }.expect("create split PSO");
        unsafe { std::mem::ManuallyDrop::drop(&mut pso_desc.pRootSignature) };
        assert!(real::is_translated_compute(pso.as_raw() as usize), "split PSO must be registered");

        let q: ID3D12CommandQueue = unsafe {
            dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
        }
        .expect("queue");
        let alloc: ID3D12CommandAllocator =
            unsafe { dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }.expect("alloc");
        let list: ID3D12GraphicsCommandList =
            unsafe { dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None) }.expect("list");

        // Allocate coherent UMA output buffer (committed window 3) and seed sentinel values.
        let out_ptr = uma.alloc(256, 256).expect("allocate coherent UMA output") as *mut u32;
        let out_addr = out_ptr as usize as u64;
        unsafe {
            for i in 0..32usize {
                out_ptr.add(i).write_volatile(0xDEAD_BEEF);
            }
        }
        let arg1: u32 = if is_sibling { 1 } else { 3 };
        let consts: [u32; 3] = [out_addr as u32, (out_addr >> 32) as u32, arg1];

        // Register a tiny RWX caller page as a mapped title image so vtable guard thunks
        // route console command-list slots (#33, #25, #39, #14, #59) to XCL's real `bind` hooks.
        struct TitleCaller {
            base: usize,
            next: usize,
        }
        impl TitleCaller {
            fn new() -> Self {
                use std::sync::{atomic::AtomicBool, Arc, Mutex};
                use windows_sys::Win32::System::Memory::{
                    VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE,
                };
                let base = unsafe {
                    VirtualAlloc(std::ptr::null(), 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
                } as usize;
                assert_ne!(base, 0, "VirtualAlloc for TitleCaller failed");
                xcl::modules::register_mapped(Arc::new(xcl::modules::MappedModule {
                    name: "chain_whole_replay_test.exe".to_string(),
                    path: PathBuf::from("chain_whole_replay_test.exe"),
                    base,
                    size: 0x1000,
                    entry_rva: 0,
                    is_dll: false,
                    exports: xcl::pe::OwnedExports::default(),
                    tls_callbacks: Mutex::new(Vec::new()),
                    no_thread_calls: AtomicBool::new(true),
                    initialized: AtomicBool::new(true),
                    stack_reserve: 0,
                }));
                Self { base, next: base }
            }
            fn emit(&mut self, target: usize, stack_args: usize) -> usize {
                use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
                use windows_sys::Win32::System::Threading::GetCurrentProcess;
                const FRAME: u8 = 0x68;
                let mut code: Vec<u8> = Vec::with_capacity(96);
                code.extend_from_slice(&[0x48, 0x83, 0xEC, FRAME]);
                for k in 0..stack_args {
                    let src = (FRAME as u32) + 0x28 + (8 * k as u32);
                    let dst = 0x20u8 + (8 * k as u8);
                    code.extend_from_slice(&[0x48, 0x8B, 0x84, 0x24]);
                    code.extend_from_slice(&src.to_le_bytes());
                    code.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, dst]);
                }
                code.extend_from_slice(&[0x48, 0xB8]);
                code.extend_from_slice(&(target as u64).to_le_bytes());
                code.extend_from_slice(&[0xFF, 0xD0, 0x48, 0x83, 0xC4, FRAME, 0xC3]);
                let entry = (self.next + 15) & !15;
                assert!(entry + code.len() <= self.base + 0x1000);
                unsafe {
                    std::ptr::copy_nonoverlapping(code.as_ptr(), entry as *mut u8, code.len());
                    FlushInstructionCache(GetCurrentProcess(), entry as *const _, code.len());
                }
                self.next = entry + code.len();
                entry
            }
        }
        impl Drop for TitleCaller {
            fn drop(&mut self) {
                use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
                xcl::modules::unregister_mapped(self.base);
                unsafe { VirtualFree(self.base as *mut _, 0, MEM_RELEASE) };
            }
        }
        let mut caller = TitleCaller::new();

        let stats_before = real::driver_cache_stats();
        let mut _keep_indirect: Option<(ID3D12CommandSignature, ID3D12Resource)> = None;
        unsafe {
            // Adopted console command list slots: #33 SetComputeRootSignature, #25 SetPipelineState,
            // #39 SetComputeRoot32BitConstants, #14 Dispatch, #59 ExecuteIndirect.
            let vt = *(list.as_raw() as *const *const usize);
            let set_rs: unsafe extern "system" fn(*mut c_void, *mut c_void) =
                std::mem::transmute(caller.emit(*vt.add(33), 0));
            let set_pso: unsafe extern "system" fn(*mut c_void, *mut c_void) =
                std::mem::transmute(caller.emit(*vt.add(25), 0));
            let set_consts: unsafe extern "system" fn(*mut c_void, u32, u32, *const u32, u32) =
                std::mem::transmute(caller.emit(*vt.add(39), 1));
            set_rs(list.as_raw(), game_rs.as_raw());
            set_pso(list.as_raw(), pso.as_raw());
            set_consts(list.as_raw(), 0, 3, consts.as_ptr(), 0);

            if case == "direct_fallback" {
                // Created via raw CreateCommandList so bind::State.allocator == 0 -> direct fallback to Chain::whole.
                let dispatch: unsafe extern "system" fn(*mut c_void, u32, u32, u32) =
                    std::mem::transmute(caller.emit(*vt.add(14), 0));
                dispatch(list.as_raw(), 1, 1, 1);
            } else {
                // Real hooked ExecuteIndirect -> bind.rs -> dispatch_chain::whole.
                let arg_desc = D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
                    ..Default::default()
                };
                let sig_desc = D3D12_COMMAND_SIGNATURE_DESC {
                    ByteStride: 12,
                    NumArgumentDescs: 1,
                    pArgumentDescs: &arg_desc,
                    NodeMask: 0,
                };
                let mut cmd_sig: Option<ID3D12CommandSignature> = None;
                dev.CreateCommandSignature(&sig_desc, None, &mut cmd_sig).expect("CreateCommandSignature");
                let cmd_sig = cmd_sig.unwrap();

                let buf_desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 256,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                let mut arg_buf: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &buf_desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut arg_buf,
                )
                .expect("indirect argument upload buffer");
                let arg_buf = arg_buf.unwrap();
                let mut mapped: *mut c_void = std::ptr::null_mut();
                arg_buf.Map(0, None, Some(&mut mapped)).expect("map arg buffer");
                std::ptr::copy_nonoverlapping([1u32, 1, 1].as_ptr(), mapped.cast::<u32>(), 3);
                arg_buf.Unmap(0, None);

                let exec_indirect: unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64) =
                    std::mem::transmute(caller.emit(*vt.add(59), 3));
                exec_indirect(list.as_raw(), cmd_sig.as_raw() as usize, 1, arg_buf.as_raw() as usize, 0, 0, 0);
                _keep_indirect = Some((cmd_sig, arg_buf));
            }
        }
        let stats_after = real::driver_cache_stats();
        let (dhits, dstale) = (stats_after.0 - stats_before.0, stats_after.1 - stats_before.1);
        if is_sibling {
            assert_eq!((dhits, dstale), (0, 0), "unselected sibling must not hit prewarmed cache (case={case})");
        } else {
            assert_eq!(
                (dhits, dstale),
                (1, 0),
                "selected shader must reuse prewarmed .chain.pso without cold retry (case={case})"
            );
        }

        unsafe {
            list.Close().expect("Close command list");
            let cmd: ID3D12CommandList = list.cast().expect("cast ID3D12CommandList");
            q.ExecuteCommandLists(&[Some(cmd)]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            q.Signal(&fence, 1).expect("signal");
            while fence.GetCompletedValue() < 1 {
                assert!(dev.GetDeviceRemovedReason().is_ok(), "device removed: {:?}", dev.GetDeviceRemovedReason());
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            for lane in 0..32usize {
                let got = out_ptr.add(lane).read_volatile();
                let expected = if is_sibling {
                    (((((lane as u32).wrapping_add(1) ^ 3).wrapping_add(5)) ^ 7).wrapping_add(9)) ^ 11
                } else {
                    [1u32, 2, 4, 5].into_iter().fold(lane as u32, |v, add| v.wrapping_add(add) ^ 7)
                };
                assert_eq!(got, expected, "case={case} lane={lane} mismatch");
            }
        }
        return;
    }

    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if xcl::gpu::recomp::dxc::location().is_err() || !runner::wave32_supported().unwrap_or(false) {
        return runner::skip_test("requires DXC and a wave32 GPU");
    }

    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("xcl-chain-whole-replay-{}-{nonce}", std::process::id()));
    let input = dir.join("input");
    let cache = dir.join("cache");
    let capture = dir.join("capture.jsonl");
    fs::create_dir_all(&input).unwrap();
    fs::write(input.join("selected.bin"), &selected_blob).unwrap();
    fs::write(input.join("sibling.bin"), &sibling_blob).unwrap();

    // Isolate child processes from ambient translation/cache env vars and pin wave64 splitting
    // + 4-window UMA layout so the test is deterministic on both wave32 and wave64 GPUs.
    let isolate = |cmd: &mut Command| {
        for var in [
            "XCL_HEAP_WINDOWS",
            "XCL_DXIL_NO_OPT",
            "XCL_X64_ALL_MASKS",
            "XCL_WAVE64_SPIN",
            "XCL_RECOMP_NO_STRUCTURED",
            "XCL_PER_LANE",
            "XCL_DIRECT_COMPACT",
            "XCL_DIRECT_COMPACT_NEST",
            "XCL_PIPELINE_MAX_KB",
            "XCL_PIPELINE_THREADS",
            "XCL_SAFE_BARRIER",
        ] {
            cmd.env_remove(var);
        }
        cmd.env("XCL_DXIL_CACHE", &cache)
            .env("XCL_DIRECT_DXIL", "1")
            .env("XCL_DIRECT_SPLIT_INSTRUCTIONS", "12")
            .env("XCL_UMA_WINDOWS", "4")
            .env("XCL_SPLIT_WAVE64", "1");
    };

    // 1. Generate a schema_version=1 capture sidecar demanding ONLY `selected.bin`.
    let mut seed_cmd = Command::new(std::env::current_exe().unwrap());
    seed_cmd
        .args(["--exact", "selective_chain_whole_replay_reuses_blob_in_fresh_process", "--nocapture"])
        .env("XCL_TEST_CHAIN_REPLAY_CASE", "seed_capture")
        .env("XCL_TEST_CAPTURE_OUT", &capture)
        .env("XCL_PSO_DISK_CACHE", "1");
    isolate(&mut seed_cmd);
    let seed = seed_cmd.output().unwrap();
    assert!(seed.status.success(), "{}\n{}", String::from_utf8_lossy(&seed.stdout), String::from_utf8_lossy(&seed.stderr));

    // 2. Run selective offline replay into isolated cache: only selected.bin's whole `.chain.pso` is persisted.
    let run_replay = |disk_cache: &str| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_xcl"));
        cmd.arg("shaders")
            .arg(&input)
            .args(["--replay-chain-whole"])
            .arg(&capture)
            .args(["--windows", "4", "--split", "1"])
            .env("XCL_PSO_DISK_CACHE", disk_cache);
        isolate(&mut cmd);
        cmd.output().unwrap()
    };

    // Disabled disk cache must report incomplete (exit 1) and persist nothing.
    let disabled = run_replay("0");
    let disabled_out = String::from_utf8_lossy(&disabled.stdout);
    assert_eq!(disabled.status.code(), Some(1), "disabled cache must be incomplete:\n{disabled_out}");
    assert!(disabled_out.contains("cache_disabled 1") && disabled_out.contains("complete: false"), "{disabled_out}");
    assert!(!cache.join("driver").exists());

    // Cold selective replay persists exactly 1 `.chain.pso` blob for selected.bin (not sibling, not pieces, not .pso).
    let cold = run_replay("1");
    let cold_out = String::from_utf8_lossy(&cold.stdout);
    assert!(cold.status.success(), "{cold_out}\n{}", String::from_utf8_lossy(&cold.stderr));
    assert!(
        cold_out.contains("created_fresh 1") && cold_out.contains("persisted 1") && cold_out.contains("complete: true"),
        "{cold_out}"
    );
    let driver_files: Vec<PathBuf> = fs::read_dir(cache.join("driver")).unwrap().flatten().map(|e| e.path()).collect();
    assert_eq!(driver_files.len(), 1, "expected exactly 1 persisted .chain.pso blob, got {driver_files:?}");
    assert!(driver_files[0].file_name().unwrap().to_string_lossy().ends_with(".chain.pso"));

    // Warm replay reuses the persisted blob cleanly.
    let warm = run_replay("1");
    let warm_out = String::from_utf8_lossy(&warm.stdout);
    assert!(warm.status.success(), "{warm_out}\n{}", String::from_utf8_lossy(&warm.stderr));
    assert!(
        warm_out.contains("created_with_supplied_blob 1") && warm_out.contains("created_fresh 0") && warm_out.contains("complete: true"),
        "{warm_out}"
    );

    // 3. Fresh Windows processes exercising real indirect, direct fallback, and unselected sibling paths.
    for case in ["indirect", "direct_fallback", "unselected_sibling"] {
        let mut child = Command::new(std::env::current_exe().unwrap());
        child
            .args(["--exact", "selective_chain_whole_replay_reuses_blob_in_fresh_process", "--nocapture"])
            .env("XCL_TEST_CHAIN_REPLAY_CASE", case)
            .env("XCL_PSO_DISK_CACHE", "1")
            .env("XCL_ASYNC_PSO", "0");
        isolate(&mut child);
        let out = child.output().unwrap();
        assert!(
            out.status.success(),
            "fresh child case={case} failed:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let _ = fs::remove_dir_all(&dir);
}
