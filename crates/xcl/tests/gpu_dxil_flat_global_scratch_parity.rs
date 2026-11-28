//! GPU differential parity tests comparing the direct RDNA2 → DXIL translator
//! (`xcl::gpu::dxil::kernel`) against HLSL + DXC (`xcl::gpu::recomp`) for
//! `GLOBAL`, `FLAT`, and `SCRATCH` memory instructions across wave32 and wave64
//! (split-wave64 on wave32-only GPUs).

use xcl::gpu::dxil::kernel::{translate_full, KernelOptions};
use xcl::gpu::dxil::validate::validate;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner::KernelState;
use xcl::gpu::recomp::{dxc, runner, translate, Mode, Options};

const SADDR_NULL: u64 = 0x7D;

fn enc(name: &str, kind: &str, fields: &[(&str, u64)]) -> Vec<u8> {
    encode(name, kind, fields, None).unwrap_or_else(|e| panic!("{name} ({kind}): {e}"))
}

fn off12(off: i32) -> u64 {
    ((off as u32) & 0xFFF) as u64
}

fn push_ops(code: &mut Vec<u8>, prefix: &str, kind: &str, ops: &[(&str, &[(&str, u64)])]) {
    for &(op, fields) in ops {
        code.extend(enc(&format!("{prefix}_{op}"), kind, fields));
    }
}

/// Mask `EXEC` to `lane < 16`, run `body`, and restore `EXEC`.
fn push_exec_guard(code: &mut Vec<u8>, lane_reg: u64, body: &[(&str, &str, &[(&str, u64)])]) {
    code.extend(enc("S_MOV_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 126)]));
    code.extend(enc("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 128 + 16), ("VSRC1", lane_reg)]));
    for &(name, kind, fields) in body {
        code.extend(enc(name, kind, fields));
    }
    code.extend(enc("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 20)]));
}

/// Store and load `DWORDX4`..`DWORD` into `v[12..=21]` using `v[6..=9]` as payload.
fn push_vec_ld_st(code: &mut Vec<u8>, prefix: &str, kind: &str, addr: u64, saddr: u64, offs: [i32; 4]) {
    let widths = [("DWORDX4", 6, 12), ("DWORDX3", 6, 16), ("DWORDX2", 8, 19), ("DWORD", 6, 21)];
    for (&(w, data, _), &off) in widths.iter().zip(&offs) {
        let f = [("ADDR", addr), ("DATA", data), ("SADDR", saddr), ("OFFSET", off12(off))];
        code.extend(enc(&format!("{prefix}_STORE_{w}"), kind, &f));
    }
    for (&(w, _, vdst), &off) in widths.iter().zip(&offs) {
        let f = [("VDST", vdst), ("ADDR", addr), ("SADDR", saddr), ("OFFSET", off12(off))];
        code.extend(enc(&format!("{prefix}_LOAD_{w}"), kind, &f));
    }
}

/// Exercise byte/short and `D16`/`D16_HI` stores and loads into `v[22..=29, 39, 40]`.
fn push_subdword_and_d16(code: &mut Vec<u8>, prefix: &str, kind: &str, addr: u64, base_off: i32) {
    let emit = |code: &mut Vec<u8>, op: &str, rf: &str, r: u64, off: i32| {
        let f = [("ADDR", addr), (rf, r), ("SADDR", SADDR_NULL), ("OFFSET", off12(base_off + off))];
        code.extend(enc(&format!("{prefix}_{op}"), kind, &f));
    };
    for (op, off) in [("STORE_BYTE", 0), ("STORE_BYTE_D16_HI", 1), ("STORE_SHORT_D16_HI", 2), ("STORE_SHORT", 4)] {
        emit(code, op, "DATA", 11, off);
    }
    for (op, vdst, off) in [
        ("LOAD_UBYTE", 22, 0), ("LOAD_SBYTE", 23, 1), ("LOAD_USHORT", 24, 2), ("LOAD_SSHORT", 25, 2),
        ("LOAD_UBYTE_D16", 26, 0), ("LOAD_UBYTE_D16_HI", 39, 0), ("LOAD_SBYTE_D16", 40, 1),
        ("LOAD_SBYTE_D16_HI", 27, 1), ("LOAD_SHORT_D16", 28, 2), ("LOAD_SHORT_D16_HI", 29, 2),
    ] {
        emit(code, op, "VDST", vdst, off);
    }
}

fn init_common_lanes(st: &mut KernelState, lanes: usize, d16_regs: &[usize]) {
    for l in 0..lanes {
        let u = l as u32;
        let v = &mut st.vgpr[l];
        v[6] = 0x1122_3344 ^ u;
        v[7] = 0x5566_7788 + u;
        v[8] = 0x99AA_BBCC ^ (u << 8);
        v[9] = 0xDDEE_FF00 + u;
        v[11] = 0x80F1_00A5;
        for &r in d16_regs {
            v[r] = 0xABCD_1234;
        }
    }
}

/// Translate and run `code` via both Direct DXIL (validated) and HLSL+DXC across
/// all supported wave configurations, asserting bit-exact memory and VGPR parity.
fn assert_gpu_parity(
    mem: &[u32],
    code: &[u8],
    check_regs: &[usize],
    setup: impl Fn(&mut KernelState, usize),
    verify: impl Fn(&str, usize, &runner::KernelResult),
) {
    if dxc::location().is_err() {
        return runner::skip_test("no DXC");
    }
    let (has_w32, has_w64) = match (runner::wave32_supported(), runner::wave64_supported()) {
        (Ok(a), Ok(b)) => (a, b),
        _ => return runner::skip_test("no GPU"),
    };
    if !has_w32 && !has_w64 {
        return runner::skip_test("no wave64");
    }
    let mut full = code.to_vec();
    full.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));

    let mut runs: Vec<(&str, bool, bool)> = Vec::new();
    if has_w32 { runs.extend([("wave32", false, false), ("split-wave64", true, true)]); }
    if has_w64 { runs.push(("native-wave64", true, false)); }
    for (label, wave64, split) in runs {
        let ko = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
        let direct = translate_full(&full, ko).unwrap_or_else(|e| panic!("{label} direct DXIL: {e}"));
        validate(&direct.dxil).unwrap_or_else(|e| panic!("{label} validator: {e}"));

        let opt = Options { mode: Mode::Kernel, wave64, split_wave64: split, ..Default::default() };
        let hlsl = translate(&full, &opt).unwrap_or_else(|e| panic!("{label} HLSL translate: {e}"));
        assert!(hlsl.complete(), "{label} incomplete HLSL: {:?} {:?}", hlsl.unsupported, hlsl.unsupported_why);
        let hlsl_dxil = dxc::compile(&hlsl.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{label} DXC: {e}\n{}", hlsl.hlsl));

        let lanes = if wave64 { 64 } else { 32 };
        let mut st = KernelState::default();
        if !wave64 { st.sgpr[127] = 0; }
        setup(&mut st, lanes);
        let hlsl_res = runner::run(&hlsl_dxil, &st, mem).unwrap_or_else(|e| panic!("{label} HLSL run: {e}"));
        let direct_res = runner::run(&direct.dxil, &st, mem).unwrap_or_else(|e| panic!("{label} direct run: {e}"));
        assert_eq!(direct_res.mem, hlsl_res.mem, "{label}: memory mismatch");
        for l in 0..lanes {
            for &r in check_regs {
                assert_eq!(direct_res.vgpr[l][r], hlsl_res.vgpr[l][r], "{label} lane {l} v{r}: {:#010x} != {:#010x}", direct_res.vgpr[l][r], hlsl_res.vgpr[l][r]);
            }
        }
        verify(label, lanes, &direct_res);
    }
}

/// Verify `GLOBAL_{LOAD,STORE}_*` across 64-bit `VADDR`, SGPR `SADDR`, signed offsets,
/// sub-dword/`D16`/`D16_HI`, `ADDTID`, and `EXEC` masking.
#[test]
fn direct_dxil_global_load_store_subdword_d16_and_saddr_parity() {
    let mut code = Vec::new();
    push_ops(&mut code, "GLOBAL", "ENC_FLAT_GLBL", &[
        ("STORE_DWORDX4", &[("ADDR", 0), ("DATA", 6), ("SADDR", SADDR_NULL)]),
        ("STORE_DWORDX3", &[("ADDR", 2), ("DATA", 6), ("SADDR", 2), ("OFFSET", off12(16))]),
        ("STORE_DWORDX2", &[("ADDR", 2), ("DATA", 8), ("SADDR", 2), ("OFFSET", off12(-16))]),
        ("STORE_DWORD", &[("ADDR", 2), ("DATA", 6), ("SADDR", 2), ("OFFSET", off12(-4))]),
        ("LOAD_DWORDX4", &[("VDST", 12), ("ADDR", 0), ("SADDR", SADDR_NULL)]),
        ("LOAD_DWORDX3", &[("VDST", 16), ("ADDR", 2), ("SADDR", 2), ("OFFSET", off12(16))]),
        ("LOAD_DWORDX2", &[("VDST", 19), ("ADDR", 2), ("SADDR", 2), ("OFFSET", off12(-16))]),
        ("LOAD_DWORD", &[("VDST", 21), ("ADDR", 2), ("SADDR", 2), ("OFFSET", off12(-4))]),
    ]);
    push_subdword_and_d16(&mut code, "GLOBAL", "ENC_FLAT_GLBL", 4, 0);
    push_ops(&mut code, "GLOBAL", "ENC_FLAT_GLBL", &[
        ("STORE_DWORD_ADDTID", &[("DATA", 6), ("SADDR", 4), ("OFFSET", off12(4))]),
        ("LOAD_DWORD_ADDTID", &[("VDST", 30), ("SADDR", 4), ("OFFSET", off12(4))]),
    ]);
    push_exec_guard(&mut code, 34, &[
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 36), ("DATA", 35), ("SADDR", SADDR_NULL)]),
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 38), ("ADDR", 36), ("SADDR", SADDR_NULL)]),
    ]);

    let mut mem = vec![0x55AA_55AAu32; 0x2200 / 4];
    for l in 0..64 {
        mem[(0x1E00 + 4 * l) / 4] = 0x1111_0000 + l as u32;
    }
    let regs: Vec<usize> = (12..=30).chain([38, 39, 40]).collect();
    assert_gpu_parity(&mem, &code, &regs, |st, lanes| {
        st.sgpr[2] = 0x0500;
        st.sgpr[4] = 0x1A00;
        init_common_lanes(st, lanes, &[26, 27, 28, 29, 39, 40]);
        for l in 0..lanes {
            let u = l as u32;
            let v = &mut st.vgpr[l];
            v[0] = 16 * u;
            v[2] = 64 * u; // stride 64 keeps [0x04F0 + 64*u .. 0x051C + 64*u) disjoint
            v[4] = 0x1600 + 8 * u;
            v[34] = u;
            v[35] = 0xCAFE_BABE;
            v[36] = 0x1E00 + 4 * u;
            v[38] = 0xDEAD_BEEF;
        }
    }, |label, lanes, r| {
        let last = lanes - 1;
        let lu = last as u32;
        assert_eq!(r.vgpr[0][12..16], [0x1122_3344, 0x5566_7788, 0x99AA_BBCC, 0xDDEE_FF00], "{label} x4");
        assert_eq!(r.vgpr[last][16..19], [0x1122_3344 ^ lu, 0x5566_7788 + lu, 0x99AA_BBCC ^ (lu << 8)], "{label} x3");
        assert_eq!(r.vgpr[last][19..22], [0x99AA_BBCC ^ (lu << 8), 0xDDEE_FF00 + lu, 0x1122_3344 ^ lu], "{label} x2+dw");
        assert_eq!((r.vgpr[0][22], r.vgpr[0][23], r.vgpr[0][24], r.vgpr[0][25]), (0xA5, 0xFFFF_FFF1, 0x80F1, 0xFFFF_80F1), "{label} sub");
        assert_eq!((r.vgpr[0][26], r.vgpr[0][39], r.vgpr[0][40], r.vgpr[0][27], r.vgpr[0][28], r.vgpr[0][29]),
            (0xABCD_00A5, 0x00A5_1234, 0xABCD_FFF1, 0xFFF1_1234, 0xABCD_80F1, 0x80F1_1234), "{label} d16");
        assert_eq!(r.vgpr[1][30], 0x1122_3344 ^ 1, "{label} addtid");
        assert_eq!((r.vgpr[0][38], r.vgpr[last][38]), (0xCAFE_BABE, 0xDEAD_BEEF), "{label} exec vdst");
        assert_eq!(r.mem[(0x1E00 + 4 * last) / 4], 0x1111_0000 + lu, "{label} exec mem");
    });
}

/// Verify all 32-bit and 64-bit `GLOBAL_ATOMIC_*` ops (`GLC=1`, `SADDR`, negative offsets,
/// CAS hit/miss, `INC`/`DEC`/`CSUB` wrap/clamp, positive/negative float atomics, `EXEC`).
#[test]
fn direct_dxil_global_atomics_32_and_64_parity() {
    const OPS32: &[&str] = &[
        "SWAP", "CMPSWAP", "ADD", "SUB", "SMIN", "UMIN", "SMAX", "UMAX",
        "AND", "OR", "XOR", "INC", "DEC", "CSUB", "FMIN", "FMAX", "FCMPSWAP",
    ];
    const OPS64: &[&str] = &[
        "SWAP_X2", "CMPSWAP_X2", "ADD_X2", "SUB_X2", "SMIN_X2", "UMIN_X2", "SMAX_X2", "UMAX_X2",
        "AND_X2", "OR_X2", "XOR_X2", "INC_X2", "DEC_X2", "FMIN_X2", "FMAX_X2", "FCMPSWAP_X2",
    ];
    let mut code = Vec::new();
    for (i, &op) in OPS32.iter().enumerate() {
        let data = if op.contains("CMPSWAP") { 6 } else if matches!(op, "FMIN" | "FMAX") { 5 } else { 4 };
        let saddr = if i % 2 == 0 { 2 } else { SADDR_NULL };
        code.extend(enc(&format!("GLOBAL_ATOMIC_{op}"), "ENC_FLAT_GLBL", &[
            ("VDST", 10 + i as u64), ("ADDR", if i % 2 == 0 { 2 } else { 0 }), ("DATA", data),
            ("SADDR", saddr), ("OFFSET", off12((i as i32) * 4 - 32)), ("GLC", 1),
        ]));
    }
    for (i, &op) in OPS64.iter().enumerate() {
        let data = if op.contains("FCMPSWAP") { 104 } else if op.contains("CMPSWAP") { 108 }
            else if matches!(op, "FMIN_X2" | "FMAX_X2") { 102 } else { 100 };
        code.extend(enc(&format!("GLOBAL_ATOMIC_{op}"), "ENC_FLAT_GLBL", &[
            ("VDST", 30 + 2 * i as u64), ("ADDR", 2), ("DATA", data),
            ("SADDR", 4), ("OFFSET", off12((i as i32) * 8 - 64)), ("GLC", 1),
        ]));
    }
    push_exec_guard(&mut code, 66, &[
        ("GLOBAL_ATOMIC_ADD", "ENC_FLAT_GLBL", &[("VDST", 63), ("ADDR", 68), ("DATA", 4), ("SADDR", SADDR_NULL), ("GLC", 1)]),
    ]);

    let mut mem = vec![0u32; 0x5000 / 4];
    for l in 0..64 {
        let f32_init = [0x8000_0000, 0xC020_0000, 0x3FC0_0000][l % 3]; // -0.0, -2.5, +1.5
        for (i, &op) in OPS32.iter().enumerate() {
            let addr = ((0x0400 + 0x80 * l) as isize + (i as isize) * 4 - 32) as usize;
            mem[addr / 4] = if op.starts_with('F') { f32_init }
                else if op == "DEC" && l % 3 == 0 { 0 }
                else if l % 2 == 0 { 4 } else { 10 + l as u32 };
        }
        let f64_init: u64 = [0x8000_0000_0000_0000, 0xC004_0000_0000_0000, 0x3FF8_0000_0000_0000][l % 3];
        for (i, &op) in OPS64.iter().enumerate() {
            let addr = ((0x2440 + 0x80 * l) as isize + (i as isize) * 8 - 64) as usize;
            let val: u64 = if op.starts_with('F') { f64_init }
                else if op == "DEC_X2" && l % 3 == 0 { 0 }
                else if l % 2 == 0 { 0x0001_0000_0000_0004 } else { 0x0003_0000_0000_000A + l as u64 };
            mem[addr / 4] = val as u32;
            mem[addr / 4 + 1] = (val >> 32) as u32;
        }
        mem[(0x4600 + 4 * l) / 4] = 100 + l as u32;
    }

    let regs: Vec<usize> = (10..27).chain(30..62).chain([63]).collect();
    assert_gpu_parity(&mem, &code, &regs, |st, lanes| {
        st.sgpr[2] = 0x0400;
        st.sgpr[4] = 0x2440;
        for l in 0..lanes {
            let u = l as u32;
            let v = &mut st.vgpr[l];
            v[0] = 0x0400 + 0x80 * u;
            v[2] = 0x80 * u;
            v[4] = 7;
            v[5] = if l % 2 == 0 { 0xBFC0_0000 } else { 0x4020_0000 }; // -1.5f / +2.5f
            v[6] = 0x3F80_0000; // new 1.0f
            v[7] = if l % 3 == 0 { 0 } else { 4 }; // +0.0f for FCMPSWAP (-0.0==+0.0), 4 for CMPSWAP
            let d64 = 0x0002_0000_0000_0007u64;
            let f64_op = if l % 2 == 0 { 0xBFF8_0000_0000_0000u64 } else { 0x4004_0000_0000_0000u64 };
            let cmp64 = 0x0001_0000_0000_0004u64;
            (v[100], v[101]) = (d64 as u32, (d64 >> 32) as u32);
            (v[102], v[103]) = (f64_op as u32, (f64_op >> 32) as u32);
            (v[104], v[105], v[106], v[107]) = (0, 0x3FF0_0000, 0, 0); // fcmpswap_x2 new=1.0, cmp=+0.0
            (v[108], v[109], v[110], v[111]) = (0x5566_7788, 0x1122_3344, cmp64 as u32, (cmp64 >> 32) as u32);
            v[63] = 0xDEAD_BEEF;
            v[66] = u;
            v[68] = 0x4600 + 4 * u;
        }
    }, |label, lanes, r| {
        assert_eq!(r.vgpr[0][10], 4, "{label} swap old");
        assert_eq!(r.mem[(0x0400 + 14 * 4 - 32) / 4], 0xBFC0_0000, "{label} fmin neg");
        assert_eq!(r.mem[(0x0400 + 11 * 4 - 32) / 4], 5, "{label} inc <7");
        assert_eq!(r.mem[(0x0400 + 0x80 + 11 * 4 - 32) / 4], 0, "{label} inc >=7 wrap");
        assert_eq!(r.mem[(0x0400 + 13 * 4 - 32) / 4], 0, "{label} csub clamp");
        assert_eq!(r.mem[(0x0400 + 16 * 4 - 32) / 4], 0x3F80_0000, "{label} fcmpswap -0.0==+0.0");
        assert_eq!((r.vgpr[0][63], r.vgpr[lanes - 1][63]), (100, 0xDEAD_BEEF), "{label} exec vdst");
        assert_eq!(r.mem[(0x4600 + 4 * (lanes - 1)) / 4], 100 + (lanes - 1) as u32, "{label} exec mem");
    });
}

/// Verify `FLAT_{LOAD,STORE}_*` (including sub-dword/`D16`) and 32/64-bit `FLAT_ATOMIC_*`.
#[test]
fn direct_dxil_flat_load_store_and_atomic_parity() {
    let mut code = Vec::new();
    push_vec_ld_st(&mut code, "FLAT", "ENC_FLAT", 0, SADDR_NULL, [0, 16, 32, 40]);
    push_subdword_and_d16(&mut code, "FLAT", "ENC_FLAT", 2, 0);
    const FLAT_ATOMICS: &[(&str, u64, u64)] = &[
        ("SWAP", 30, 60), ("CMPSWAP", 31, 48), ("ADD", 32, 60), ("SUB", 33, 60),
        ("SMIN", 34, 60), ("UMAX", 35, 60), ("XOR", 36, 60), ("INC", 37, 60),
        ("DEC", 38, 60), ("FMIN", 58, 62), ("FMAX", 59, 62),
        ("SWAP_X2", 42, 60), ("ADD_X2", 44, 60), ("SUB_X2", 46, 60),
        ("AND_X2", 64, 60), ("FCMPSWAP_X2", 50, 52),
    ];
    for (i, &(op, vdst, data)) in FLAT_ATOMICS.iter().enumerate() {
        code.extend(enc(&format!("FLAT_ATOMIC_{op}"), "ENC_FLAT", &[
            ("VDST", vdst), ("ADDR", 4), ("DATA", data), ("SADDR", SADDR_NULL), ("OFFSET", (i * 8) as u64), ("GLC", 1),
        ]));
    }
    push_exec_guard(&mut code, 56, &[
        ("FLAT_STORE_DWORD", "ENC_FLAT", &[("ADDR", 66), ("DATA", 68), ("SADDR", SADDR_NULL)]),
        ("FLAT_LOAD_DWORD", "ENC_FLAT", &[("VDST", 57), ("ADDR", 0), ("SADDR", SADDR_NULL)]),
    ]);

    let mut mem = vec![0u32; 0x4800 / 4];
    for l in 0..64 {
        for (i, &(op, _, _)) in FLAT_ATOMICS.iter().enumerate() {
            let a = (0x2000 + 0x80 * l + 8 * i) / 4;
            if op.starts_with('F') {
                mem[a] = if l % 2 == 0 { 0x8000_0000 } else { 0x3FC0_0000 };
                mem[a + 1] = if l % 2 == 0 { 0x8000_0000 } else { 0x3FF8_0000 };
            } else {
                mem[a] = if l % 2 == 0 { 4 } else { 10 + l as u32 };
                mem[a + 1] = 0x0001_0000;
            }
        }
        mem[(0x4200 + 4 * l) / 4] = 0x2222_0000 + l as u32;
    }
    let regs: Vec<usize> = (12..=40).chain(42..=47).chain([50, 51, 57, 58, 59, 64, 65]).collect();
    assert_gpu_parity(&mem, &code, &regs, |st, lanes| {
        init_common_lanes(st, lanes, &[26, 27, 28, 29, 39, 40]);
        for l in 0..lanes {
            let u = l as u32;
            let v = &mut st.vgpr[l];
            v[0] = 64 * u;
            v[2] = 0x1000 + 8 * u;
            v[4] = 0x2000 + 0x80 * u;
            (v[48], v[49]) = (0x7777_8888, 10 + u);
            (v[52], v[53], v[54], v[55]) = (0, 0x3FF0_0000, 0x3FC0_0000, 0x3FF8_0000); // hit odd, miss even
            (v[60], v[61]) = (7, 0x0002_0000);
            v[62] = if l % 2 == 0 { 0xBFC0_0000 } else { 0x4020_0000 };
            v[56] = u;
            v[57] = 0xDEAD_BEEF;
            v[66] = 0x4200 + 4 * u;
            v[68] = 0xCAFE_BABE;
        }
    }, |label, lanes, r| {
        let last = lanes - 1;
        assert_eq!((r.vgpr[0][23], r.vgpr[0][27], r.vgpr[0][29]), (0xFFFF_FFF1, 0xFFF1_1234, 0x80F1_1234), "{label} flat d16");
        assert_eq!(r.vgpr[0][30], 4, "{label} flat_atomic_swap");
        assert_eq!((r.vgpr[0][57], r.vgpr[last][57]), (0x1122_3344, 0xDEAD_BEEF), "{label} exec vdst");
        assert_eq!(r.mem[(0x4200 + 4 * last) / 4], 0x2222_0000 + last as u32, "{label} exec mem");
    });
}

/// Verify per-lane `SCRATCH_{LOAD,STORE}_*` with SGPR `SADDR`, signed 12-bit offsets,
/// sub-dword/`D16`/`D16_HI`, and `EXEC` masking.
#[test]
fn direct_dxil_scratch_load_store_subdword_d16_and_exec_parity() {
    let mut code = Vec::new();
    push_ops(&mut code, "SCRATCH", "ENC_FLAT_SCRATCH", &[
        ("STORE_DWORD", &[("ADDR", 0), ("DATA", 10), ("SADDR", 4), ("OFFSET", off12(-4))]),
        ("LOAD_DWORD", &[("VDST", 41), ("ADDR", 0), ("SADDR", 4), ("OFFSET", off12(-4))]),
    ]);
    push_vec_ld_st(&mut code, "SCRATCH", "ENC_FLAT_SCRATCH", 0, SADDR_NULL, [0, 16, -16, -8]);
    code.extend(enc("SCRATCH_STORE_DWORD", "ENC_FLAT_SCRATCH", &[("ADDR", 0), ("DATA", 6), ("SADDR", SADDR_NULL), ("OFFSET", off12(-20))]));
    push_subdword_and_d16(&mut code, "SCRATCH", "ENC_FLAT_SCRATCH", 0, -24);
    code.extend(enc("SCRATCH_STORE_DWORD", "ENC_FLAT_SCRATCH", &[("ADDR", 0), ("DATA", 32), ("SADDR", SADDR_NULL), ("OFFSET", off12(-4))]));
    push_exec_guard(&mut code, 33, &[
        ("SCRATCH_STORE_DWORD", "ENC_FLAT_SCRATCH", &[("ADDR", 0), ("DATA", 34), ("SADDR", SADDR_NULL), ("OFFSET", off12(-4))]),
        ("SCRATCH_LOAD_DWORD", "ENC_FLAT_SCRATCH", &[("VDST", 35), ("ADDR", 0), ("SADDR", SADDR_NULL), ("OFFSET", off12(-4))]),
    ]);
    code.extend(enc("SCRATCH_LOAD_DWORD", "ENC_FLAT_SCRATCH", &[("VDST", 36), ("ADDR", 0), ("SADDR", SADDR_NULL), ("OFFSET", off12(-4))]));

    let mem = vec![0u32; 64];
    let regs: Vec<usize> = (12..=29).chain([35, 36, 39, 40, 41]).collect();
    assert_gpu_parity(&mem, &code, &regs, |st, lanes| {
        st.sgpr[4] = 8;
        init_common_lanes(st, lanes, &[26, 27, 28, 29, 39, 40]);
        for l in 0..lanes {
            let u = l as u32;
            let v = &mut st.vgpr[l];
            v[0] = 32 + 64 * u;
            v[10] = 0xA1B2_C3D4;
            v[32] = 0x55AA_0000 + u;
            v[33] = u;
            v[34] = 0xCAFE_BABE;
            v[35] = 0xDEAD_BEEF;
        }
    }, |label, lanes, r| {
        let last = lanes as u32 - 1;
        assert_eq!(r.vgpr[0][41], 0xA1B2_C3D4, "{label} scratch saddr");
        assert_eq!(
            r.vgpr[lanes - 1][12..16],
            [0x1122_3344 ^ last, 0x5566_7788 + last, 0x99AA_BBCC ^ (last << 8), 0xDDEE_FF00 + last],
            "{label} scratch x4"
        );
        assert_eq!((r.vgpr[0][22], r.vgpr[0][23], r.vgpr[0][26], r.vgpr[0][27], r.vgpr[0][29]),
            (0xA5, 0xFFFF_FFF1, 0xABCD_00A5, 0xFFF1_1234, 0x80F1_1234), "{label} scratch d16");
        assert_eq!((r.vgpr[0][35], r.vgpr[lanes - 1][35], r.vgpr[lanes - 1][36]),
            (0xCAFE_BABE, 0xDEAD_BEEF, 0x55AA_0000 + last), "{label} scratch exec");
    });
}
