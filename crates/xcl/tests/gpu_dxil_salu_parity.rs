//! GPU differential parity tests between the direct RDNA2 → DXIL translator
//! (`xcl::gpu::dxil::kernel::translate`, validated with `dxil.dll`) and the HLSL + DXC path
//! (`xcl::gpu::recomp`) for `lower.rs` scalar ALU instructions: shift-add, half packing,
//! min/max + SCC tie semantics, signed overflow + carry/borrow chains, conditional moves,
//! 32-bit and 64-bit bit-manipulation/bitfield ops, and 64-bit shifts.

use std::ops::Range;

use xcl::gpu::dxil::kernel::{translate as direct_translate, KernelOptions};
use xcl::gpu::dxil::validate;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner::{self, KernelResult, KernelState};
use xcl::gpu::recomp::{dxc, translate as hlsl_translate, Options};

macro_rules! cap_scc {
    ($sdst:expr) => {
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", $sdst), ("SSRC0", 129), ("SSRC1", 128)][..], None)
    };
}

fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    let mut v = Vec::new();
    for (n, e, f, l) in list {
        v.extend(encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    v
}

fn gpu_modes() -> Vec<(bool, bool)> {
    match (runner::adapter(), runner::wave64_programs_supported(), dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => {
            if runner::wave64_supported() == Ok(true) {
                vec![(false, false), (true, false)]
            } else {
                vec![(false, false), (true, true)]
            }
        }
        (a, w, d) => {
            runner::skip_test(if a.is_err() || w.is_err() {
                "no GPU"
            } else if d.is_err() {
                "no DXC"
            } else {
                "no wave64"
            });
            Vec::new()
        }
    }
}

fn run_parity(
    label: &str,
    code: &[u8],
    st: &KernelState,
    wave64: bool,
    split: bool,
    sgprs: Range<usize>,
) -> KernelResult {
    let hlsl_opt = Options { wave64, split_wave64: split, ..Default::default() };
    let t = hlsl_translate(code, &hlsl_opt).unwrap_or_else(|e| panic!("{label}: hlsl translate: {e}"));
    assert!(t.complete(), "{label}: hlsl incomplete: {:?}", t.unsupported_why);
    let hlsl_dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[])
        .unwrap_or_else(|e| panic!("{label}: dxc compile: {e}"));

    let direct_dxil = direct_translate(
        code,
        KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 },
    )
    .unwrap_or_else(|e| panic!("{label} (w64={wave64} split={split}): direct translate: {e}"));
    validate::validate(&direct_dxil)
        .unwrap_or_else(|e| panic!("{label} (w64={wave64} split={split}): dxil validate: {e}"));

    let r_hlsl = runner::run(&hlsl_dxil, st, &[]).unwrap();
    let r_direct = runner::run(&direct_dxil, st, &[]).unwrap();

    for s in sgprs {
        assert_eq!(
            r_direct.sgpr[s], r_hlsl.sgpr[s],
            "{label} (w64={wave64} split={split}) s{s}: direct={:#x} hlsl={:#x}",
            r_direct.sgpr[s], r_hlsl.sgpr[s]
        );
    }
    assert_eq!(r_direct.scc, r_hlsl.scc, "{label} (w64={wave64} split={split}) scc");
    r_direct
}

#[test]
fn direct_dxil_salu_shift_add_pack_minmax_and_mul_hi_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    let code = prog(&[
        // Shift-add 1..4 (s0 = 0x1000_0005, s1 = 0xE000_0003)
        ("S_LSHL1_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 0), ("SSRC1", 1)], None),
        cap_scc!(11),
        ("S_LSHL2_ADD_U32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 0), ("SSRC1", 1)], None),
        cap_scc!(13),
        ("S_LSHL3_ADD_U32", "ENC_SOP2", &[("SDST", 14), ("SSRC0", 0), ("SSRC1", 1)], None),
        ("S_LSHL4_ADD_U32", "ENC_SOP2", &[("SDST", 15), ("SSRC0", 0), ("SSRC1", 1)], None),
        // Half packing (s2 = 0xAABB_CCDD, s3 = 0x1122_3344); must preserve SCC
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 128), ("SSRC1", 128)], None), // SCC = 1
        ("S_PACK_LL_B32_B16", "ENC_SOP2", &[("SDST", 16), ("SSRC0", 2), ("SSRC1", 3)], None),
        ("S_PACK_LH_B32_B16", "ENC_SOP2", &[("SDST", 17), ("SSRC0", 2), ("SSRC1", 3)], None),
        ("S_PACK_HH_B32_B16", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 2), ("SSRC1", 3)], None),
        cap_scc!(19),
        // Min / Max signed & unsigned + tie SCC=0 rule (s4 = -10, s5 = 5)
        ("S_MIN_I32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 4), ("SSRC1", 5)], None),
        cap_scc!(21), // S0 chosen (-10 < 5) -> SCC=1
        ("S_MIN_U32", "ENC_SOP2", &[("SDST", 22), ("SSRC0", 4), ("SSRC1", 5)], None),
        cap_scc!(23), // S1 chosen (5 < 0xFFFFFFF6) -> SCC=0
        ("S_MAX_I32", "ENC_SOP2", &[("SDST", 24), ("SSRC0", 5), ("SSRC1", 5)], None),
        cap_scc!(25), // tie -> SCC=0
        ("S_MAX_U32", "ENC_SOP2", &[("SDST", 26), ("SSRC0", 4), ("SSRC1", 5)], None),
        cap_scc!(27), // S0 chosen -> SCC=1
        // Absdiff, sign-extend, mul_hi (s6 = 0x7FFF_FFFF, s7 = 0x8000_0000)
        ("S_ABSDIFF_I32", "ENC_SOP2", &[("SDST", 28), ("SSRC0", 6), ("SSRC1", 7)], None),
        cap_scc!(29),
        ("S_SEXT_I32_I8", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 2)], None),  // 0xDD -> 0xFFFF_FFDD
        ("S_SEXT_I32_I16", "ENC_SOP1", &[("SDST", 31), ("SSRC0", 2)], None), // 0xCCDD -> 0xFFFF_CCDD
        ("S_MUL_HI_U32", "ENC_SOP2", &[("SDST", 32), ("SSRC0", 6), ("SSRC1", 7)], None),
        ("S_MUL_HI_I32", "ENC_SOP2", &[("SDST", 33), ("SSRC0", 6), ("SSRC1", 6)], None),
    ]);

    let mut st = KernelState::default();
    st.sgpr[0] = 0x1000_0005;
    st.sgpr[1] = 0xE000_0003;
    st.sgpr[2] = 0xAABB_CCDD;
    st.sgpr[3] = 0x1122_3344;
    st.sgpr[4] = (-10i32) as u32;
    st.sgpr[5] = 5;
    st.sgpr[6] = 0x7FFF_FFFF;
    st.sgpr[7] = 0x8000_0000;

    for (wave64, split) in modes {
        let r = run_parity("salu_shift_pack_minmax", &code, &st, wave64, split, 10..34);
        assert_eq!(r.sgpr[10], 0x0000_000D, "s_lshl1_add_u32 wrap");
        assert_eq!(r.sgpr[11], 1, "s_lshl1_add_u32 carry SCC");
        assert_eq!(r.sgpr[12], 0x2000_0017, "s_lshl2_add_u32");
        assert_eq!(r.sgpr[16], 0x3344_CCDD, "s_pack_ll");
        assert_eq!(r.sgpr[17], 0x1122_CCDD, "s_pack_lh");
        assert_eq!(r.sgpr[18], 0x1122_AABB, "s_pack_hh");
        assert_eq!(r.sgpr[19], 1, "s_pack_* preserved SCC");
        assert_eq!((r.sgpr[20], r.sgpr[21]), ((-10i32) as u32, 1), "s_min_i32");
        assert_eq!((r.sgpr[22], r.sgpr[23]), (5, 0), "s_min_u32");
        assert_eq!((r.sgpr[24], r.sgpr[25]), (5, 0), "s_max_i32 tie SCC=0");
        assert_eq!((r.sgpr[26], r.sgpr[27]), ((-10i32) as u32, 1), "s_max_u32");
        assert_eq!((r.sgpr[28], r.sgpr[29]), (1, 1), "s_absdiff_i32(INT_MAX, INT_MIN) = 1");
        assert_eq!(r.sgpr[30], 0xFFFF_FFDD, "s_sext_i32_i8");
        assert_eq!(r.sgpr[31], 0xFFFF_CCDD, "s_sext_i32_i16");
        assert_eq!(r.sgpr[32], 0x3FFF_FFFF, "s_mul_hi_u32");
        assert_eq!(r.sgpr[33], 0x3FFF_FFFF, "s_mul_hi_i32");
    }
}

#[test]
fn direct_dxil_salu_overflow_carry_borrow_and_cmov_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    let code = prog(&[
        // Signed overflow SCC on S_ADD_I32 / S_SUB_I32 (s0 = 0x7FFF_FFFF, s1 = 1, s2 = 0x8000_0000)
        ("S_ADD_I32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 0), ("SSRC1", 1)], None),
        cap_scc!(11), // overflow -> 1
        ("S_SUB_I32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 2), ("SSRC1", 1)], None),
        cap_scc!(13), // overflow -> 1
        ("S_ADD_I32", "ENC_SOP2", &[("SDST", 14), ("SSRC0", 1), ("SSRC1", 1)], None),
        cap_scc!(15), // no overflow -> 0
        // 64-bit chained scalar add & sub (s[4:5] + s[6:7], s[4:5] - s[6:7])
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 16), ("SSRC0", 4), ("SSRC1", 6)], None),
        ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 17), ("SSRC0", 5), ("SSRC1", 7)], None),
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 18), ("SSRC0", 4), ("SSRC1", 6)], None),
        ("S_SUBB_U32", "ENC_SOP2", &[("SDST", 19), ("SSRC0", 5), ("SSRC1", 7)], None),
        // S_CMOV_B32 / S_CMOVK_I32 / S_CMOV_B64 when SCC=1 vs SCC=0
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 128), ("SSRC1", 128)], None), // SCC=1
        ("S_CMOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 0)], None),
        ("S_CMOVK_I32", "ENC_SOPK", &[("SDST", 21), ("SIMM16", 0xFFFB)], None), // -5
        ("S_CMOV_B64", "ENC_SOP1", &[("SDST", 22), ("SSRC0", 4)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 128), ("SSRC1", 129)], None), // SCC=0
        ("S_CMOV_B32", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 0)], None),       // keep old
        ("S_CMOVK_I32", "ENC_SOPK", &[("SDST", 25), ("SIMM16", 7)], None),     // keep old
        ("S_CMOV_B64", "ENC_SOP1", &[("SDST", 26), ("SSRC0", 4)], None),       // keep old
    ]);

    let mut st = KernelState::default();
    st.sgpr[0] = 0x7FFF_FFFF;
    st.sgpr[1] = 1;
    st.sgpr[2] = 0x8000_0000;
    let a = 0x0000_0005_0000_0010u64;
    let b = 0x0000_0002_FFFF_FFF8u64;
    st.sgpr[4] = a as u32;
    st.sgpr[5] = (a >> 32) as u32;
    st.sgpr[6] = b as u32;
    st.sgpr[7] = (b >> 32) as u32;
    for s in 20..28 {
        st.sgpr[s] = 0xCAFE_0000 | s as u32;
    }

    for (wave64, split) in modes {
        let r = run_parity("salu_overflow_carry_cmov", &code, &st, wave64, split, 10..28);
        assert_eq!((r.sgpr[10], r.sgpr[11]), (0x8000_0000, 1));
        assert_eq!((r.sgpr[12], r.sgpr[13]), (0x7FFF_FFFF, 1));
        assert_eq!((r.sgpr[14], r.sgpr[15]), (2, 0));
        let sum = a.wrapping_add(b);
        let diff = a.wrapping_sub(b);
        assert_eq!((r.sgpr[16], r.sgpr[17]), (sum as u32, (sum >> 32) as u32));
        assert_eq!((r.sgpr[18], r.sgpr[19]), (diff as u32, (diff >> 32) as u32));
        assert_eq!(r.sgpr[20], 0x7FFF_FFFF);
        assert_eq!(r.sgpr[21], (-5i32) as u32);
        assert_eq!((r.sgpr[22], r.sgpr[23]), (a as u32, (a >> 32) as u32));
        assert_eq!(r.sgpr[24], 0xCAFE_0018);
        assert_eq!(r.sgpr[25], 0xCAFE_0019);
        assert_eq!((r.sgpr[26], r.sgpr[27]), (0xCAFE_001A, 0xCAFE_001B));
    }
}

#[test]
fn direct_dxil_salu_32bit_logic_bitops_and_bfe_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    let code = prog(&[
        ("S_ANDN2_B32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 0), ("SSRC1", 1)], None),
        ("S_ORN2_B32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 0), ("SSRC1", 1)], None),
        ("S_NAND_B32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 0), ("SSRC1", 1)], None),
        ("S_NOR_B32", "ENC_SOP2", &[("SDST", 13), ("SSRC0", 0), ("SSRC1", 1)], None),
        ("S_XNOR_B32", "ENC_SOP2", &[("SDST", 14), ("SSRC0", 0), ("SSRC1", 1)], None),
        ("S_ASHR_I32", "ENC_SOP2", &[("SDST", 15), ("SSRC0", 2), ("SSRC1", 128 + 4)], None),
        ("S_BITSET0_B32", "ENC_SOP1", &[("SDST", 16), ("SSRC0", 128 + 5)], None),
        ("S_BITSET1_B32", "ENC_SOP1", &[("SDST", 17), ("SSRC0", 128 + 7)], None),
        ("S_BITCMP0_B32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128 + 3)], None),
        cap_scc!(18),
        ("S_BITCMP1_B32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128 + 4)], None),
        cap_scc!(19),
        ("S_BCNT0_I32_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 0)], None),
        ("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 21), ("SSRC0", 0)], None),
        ("S_FF0_I32_B32", "ENC_SOP1", &[("SDST", 22), ("SSRC0", 3)], None),
        ("S_FF1_I32_B32", "ENC_SOP1", &[("SDST", 23), ("SSRC0", 0)], None),
        ("S_FLBIT_I32_B32", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 0)], None),
        ("S_FLBIT_I32", "ENC_SOP1", &[("SDST", 25), ("SSRC0", 2)], None),
        ("S_BREV_B32", "ENC_SOP1", &[("SDST", 26), ("SSRC0", 0)], None),
        ("S_BFM_B32", "ENC_SOP2", &[("SDST", 27), ("SSRC0", 128 + 6), ("SSRC1", 128 + 4)], None),
        // BFE offset=4, width=8 -> control word = 4 | (8 << 16) in s4
        ("S_BFE_U32", "ENC_SOP2", &[("SDST", 28), ("SSRC0", 0), ("SSRC1", 4)], None),
        ("S_BFE_I32", "ENC_SOP2", &[("SDST", 29), ("SSRC0", 0), ("SSRC1", 4)], None),
    ]);

    let mut st = KernelState::default();
    st.sgpr[0] = 0x0F00_08F0;
    st.sgpr[1] = 0x00FF_00FF;
    st.sgpr[2] = 0xF000_0080;
    st.sgpr[3] = 0x0000_00FF;
    st.sgpr[4] = 4 | (8 << 16);
    st.sgpr[16] = 0xFFFF_FFFF;
    st.sgpr[17] = 0;

    for (wave64, split) in modes {
        let r = run_parity("salu_32bit_bitops_bfe", &code, &st, wave64, split, 10..30);
        assert_eq!(r.sgpr[15], 0xFF00_0008, "s_ashr_i32");
        assert_eq!(r.sgpr[16], 0xFFFF_FFDF, "s_bitset0_b32 bit 5");
        assert_eq!(r.sgpr[17], 0x0000_0080, "s_bitset1_b32 bit 7");
        assert_eq!(r.sgpr[18], 1, "s_bitcmp0 bit 3 of 0x0F00_08F0 is 0");
        assert_eq!(r.sgpr[19], 1, "s_bitcmp1 bit 4 of 0x0F00_08F0 is 1");
        assert_eq!(r.sgpr[20], 23, "s_bcnt0_i32_b32");
        assert_eq!(r.sgpr[21], 9, "s_bcnt1_i32_b32");
        assert_eq!(r.sgpr[22], 8, "s_ff0_i32_b32(0xFF) = 8");
        assert_eq!(r.sgpr[23], 4, "s_ff1_i32_b32(0x0F00_08F0) = 4");
        assert_eq!(r.sgpr[24], 4, "s_flbit_i32_b32");
        assert_eq!(r.sgpr[25], 4, "s_flbit_i32(0xF000_0080) = 4");
        assert_eq!(r.sgpr[26], 0x0F00_08F0u32.reverse_bits());
        assert_eq!(r.sgpr[27], 0x0000_03F0, "s_bfm_b32(6, 4)");
        assert_eq!(r.sgpr[28], 0x0000_008F, "s_bfe_u32");
        assert_eq!(r.sgpr[29], 0xFFFF_FF8F, "s_bfe_i32 sign-extended");
    }
}

#[test]
fn direct_dxil_salu_64bit_shifts_bitops_and_bfe_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    let code = prog(&[
        // 64-bit shifts across the 32-bit boundary (s[0:1], shift 12)
        ("S_LSHL_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 0), ("SSRC1", 128 + 12)], None),
        ("S_LSHR_B64", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 0), ("SSRC1", 128 + 12)], None),
        ("S_ASHR_I64", "ENC_SOP2", &[("SDST", 14), ("SSRC0", 0), ("SSRC1", 128 + 12)], None),
        // 64-bit bitset / bitcmp in high word (bit 40)
        ("S_BITSET0_B64", "ENC_SOP1", &[("SDST", 16), ("SSRC0", 128 + 40)], None),
        ("S_BITSET1_B64", "ENC_SOP1", &[("SDST", 18), ("SSRC0", 128 + 40)], None),
        ("S_BITCMP0_B64", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128 + 40)], None),
        cap_scc!(20),
        ("S_BITCMP1_B64", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128 + 63)], None),
        cap_scc!(21),
        // 64-bit bit counts / scans / reverse / bfm / bfe across the 32-bit boundary
        ("S_BCNT0_I32_B64", "ENC_SOP1", &[("SDST", 22), ("SSRC0", 0)], None),
        ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 23), ("SSRC0", 0)], None),
        ("S_FF0_I32_B64", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 2)], None),
        ("S_FF1_I32_B64", "ENC_SOP1", &[("SDST", 25), ("SSRC0", 0)], None),
        ("S_FLBIT_I32_B64", "ENC_SOP1", &[("SDST", 26), ("SSRC0", 2)], None),
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 27), ("SSRC0", 0)], None),
        ("S_BREV_B64", "ENC_SOP1", &[("SDST", 28), ("SSRC0", 0)], None),
        ("S_BFM_B64", "ENC_SOP2", &[("SDST", 30), ("SSRC0", 128 + 16), ("SSRC1", 128 + 24)], None),
        // BFE64 offset=28, width=12 (crosses bit 32; s4 = 28 | (12 << 16))
        ("S_BFE_U64", "ENC_SOP2", &[("SDST", 32), ("SSRC0", 0), ("SSRC1", 4)], None),
        ("S_BFE_I64", "ENC_SOP2", &[("SDST", 34), ("SSRC0", 0), ("SSRC1", 4)], None),
    ]);

    let x: u64 = 0xF000_00A0_8000_00F0;
    let y: u64 = 0x0000_000F_FFFF_FFFF;
    let mut st = KernelState::default();
    st.sgpr[0] = x as u32;
    st.sgpr[1] = (x >> 32) as u32;
    st.sgpr[2] = y as u32;
    st.sgpr[3] = (y >> 32) as u32;
    st.sgpr[4] = 28 | (12 << 16);
    st.sgpr[16] = u32::MAX;
    st.sgpr[17] = u32::MAX;
    st.sgpr[18] = 0;
    st.sgpr[19] = 0;

    for (wave64, split) in modes {
        let r = run_parity("salu_64bit_shifts_bitops_bfe", &code, &st, wave64, split, 10..36);
        let pair = |i: usize| (r.sgpr[i] as u64) | ((r.sgpr[i + 1] as u64) << 32);
        assert_eq!(pair(10), x << 12, "s_lshl_b64");
        assert_eq!(pair(12), x >> 12, "s_lshr_b64");
        assert_eq!(pair(14), ((x as i64) >> 12) as u64, "s_ashr_i64");
        assert_eq!(pair(16), u64::MAX ^ (1u64 << 40), "s_bitset0_b64 bit 40");
        assert_eq!(pair(18), 1u64 << 40, "s_bitset1_b64 bit 40");
        assert_eq!(r.sgpr[20], 1, "s_bitcmp0_b64 bit 40");
        assert_eq!(r.sgpr[21], 1, "s_bitcmp1_b64 bit 63");
        assert_eq!(r.sgpr[22], x.count_zeros(), "s_bcnt0_i32_b64");
        assert_eq!(r.sgpr[23], x.count_ones(), "s_bcnt1_i32_b64");
        assert_eq!(r.sgpr[24], 36, "s_ff0_i32_b64");
        assert_eq!(r.sgpr[25], 4, "s_ff1_i32_b64");
        assert_eq!(r.sgpr[26], 28, "s_flbit_i32_b64");
        assert_eq!(r.sgpr[27], 4, "s_flbit_i32_i64");
        assert_eq!(pair(28), x.reverse_bits(), "s_brev_b64");
        assert_eq!(pair(30), 0xFFFFu64 << 24, "s_bfm_b64");
        let u_bfe = (x >> 28) & 0xFFF;
        let i_bfe = (((u_bfe as i64) << 52) >> 52) as u64;
        assert_eq!(pair(32), u_bfe, "s_bfe_u64 cross-dword");
        assert_eq!(pair(34), i_bfe, "s_bfe_i64 cross-dword sign-extended");
    }
}
