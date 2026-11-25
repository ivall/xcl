//! GPU differential parity tests between the direct RDNA2 → DXIL translator
//! (`xcl::gpu::dxil::kernel::translate`, validated with `dxil.dll`) and the HLSL + DXC path
//! (`xcl::gpu::recomp`) for `lower_valu.rs`: 32-bit carry/borrow chains + integer clamp under
//! partial EXEC, f32/i32/u64 V_CMP/V_CMPX predicates, VOP3 source/output modifiers + CNDMASK,
//! and VALU float/int/packed conversions across wave32 and wave64 (split wave64 on wave32 GPUs).

use std::ops::Range;

use xcl::gpu::dxil::kernel::{translate as direct_translate, KernelOptions};
use xcl::gpu::dxil::validate;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner::{self, KernelResult, KernelState};
use xcl::gpu::recomp::{dxc, translate as hlsl_translate, Options};

const V: u64 = 256;

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
    sgprs: &[usize],
    vgprs: Range<usize>,
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
    let lanes = if wave64 { 64 } else { 32 };

    for &s in sgprs {
        assert_eq!(
            r_direct.sgpr[s], r_hlsl.sgpr[s],
            "{label} (w64={wave64} split={split}) s{s}: direct={:#x} hlsl={:#x}",
            r_direct.sgpr[s], r_hlsl.sgpr[s]
        );
    }
    for l in 0..lanes {
        for v in vgprs.clone() {
            assert_eq!(
                r_direct.vgpr[l][v], r_hlsl.vgpr[l][v],
                "{label} (w64={wave64} split={split}) lane {l} v{v}: direct={:#x} hlsl={:#x}",
                r_direct.vgpr[l][v], r_hlsl.vgpr[l][v]
            );
        }
    }
    r_direct
}

#[test]
fn direct_dxil_valu_carries_borrows_and_int_clamp_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    // Narrow EXEC to lanes with (lane & 3) != 3 so inactive lanes verify:
    //  1) VGPR destinations keep sentinel 0xDEAD_BEEF
    //  2) Carry/borrow SDST bits for inactive lanes are forced to 0
    let code = prog(&[
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 128 + 3), ("VSRC1", 0)], None),
        ("V_CMPX_NE_U32", "ENC_VOPC", &[("SRC0", 128 + 3), ("VSRC1", 9)], None),
        // 64-bit add: v[1:2] + v[3:4] -> v[10:11], carry-out in s[20:21]
        ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 10), ("SDST", 20), ("SRC0", V + 1), ("SRC1", V + 3)], None),
        ("V_ADD_CO_CI_U32", "VOP3_SDST_ENC", &[("VDST", 11), ("SDST", 22), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", 20)], None),
        // 64-bit sub: v[1:2] - v[3:4] -> v[12:13], borrow-out in s[24:25]
        ("V_SUB_CO_U32", "VOP3_SDST_ENC", &[("VDST", 12), ("SDST", 24), ("SRC0", V + 1), ("SRC1", V + 3)], None),
        ("V_SUB_CO_CI_U32", "VOP3_SDST_ENC", &[("VDST", 13), ("SDST", 26), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", 24)], None),
        // 64-bit subrev: v[3:4] - v[1:2] -> v[14:15], borrow-out in s[28:29]
        ("V_SUBREV_CO_U32", "VOP3_SDST_ENC", &[("VDST", 14), ("SDST", 28), ("SRC0", V + 1), ("SRC1", V + 3)], None),
        ("V_SUBREV_CO_CI_U32", "VOP3_SDST_ENC", &[("VDST", 15), ("SDST", 30), ("SRC0", V + 2), ("SRC1", V + 4), ("SRC2", 28)], None),
        // Integer clamp saturation (unsigned & signed)
        ("V_ADD_NC_U32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 1), ("SRC1", V + 3), ("CLAMP", 1)], None),
        ("V_SUB_NC_U32", "ENC_VOP3", &[("VDST", 17), ("SRC0", V + 1), ("SRC1", V + 3), ("CLAMP", 1)], None),
        ("V_SUBREV_NC_U32", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 1), ("SRC1", V + 3), ("CLAMP", 1)], None),
        ("V_ADD_NC_I32", "ENC_VOP3", &[("VDST", 19), ("SRC0", V + 5), ("SRC1", V + 6), ("CLAMP", 1)], None),
        ("V_SUB_NC_I32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 5), ("SRC1", V + 6), ("CLAMP", 1)], None),
    ]);

    let pairs_u64: [(u64, u64); 4] = [
        (0x0000_0001_FFFF_FFF0, 0x0000_0002_0000_0020), // low carry, no high carry
        (0xFFFF_FFFF_0000_0005, 0x0000_0001_0000_0010), // low borrow, high carry
        (0x0000_0000_0000_0010, 0x0000_0000_0000_0005), // no borrow in sub, borrow in subrev
        (0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF), // inactive lane (must not leak carry)
    ];
    let pairs_i32: [(i32, i32); 4] = [
        (0x7FFF_FFF0, 0x20),        // add saturates +INT_MAX
        (-0x7FFF_FFF0, -0x20),      // add saturates INT_MIN
        (i32::MIN + 5, 0x10),       // sub saturates INT_MIN
        (0x1234, 0x5678),           // inactive lane
    ];

    let mut st = KernelState::default();
    for s in 20..32 {
        st.sgpr[s] = u32::MAX; // pre-poison masks to confirm inactive lanes clear to 0
    }
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        let (a, b) = pairs_u64[l & 3];
        r[1] = a as u32;
        r[2] = (a >> 32) as u32;
        r[3] = b as u32;
        r[4] = (b >> 32) as u32;
        let (si0, si1) = pairs_i32[l & 3];
        r[5] = si0 as u32;
        r[6] = si1 as u32;
        for reg in 10..21 {
            r[reg] = 0xDEAD_BEEF;
        }
    }

    for (wave64, split) in modes {
        // In wave32, carry/borrow masks write only the low SGPR of each even slot; in wave64 they write pairs.
        let watched_sgprs: Vec<usize> = if wave64 {
            (20..32).collect()
        } else {
            vec![20, 22, 24, 26, 28, 30]
        };
        let r = run_parity("valu_carries_clamp", &code, &st, wave64, split, &watched_sgprs, 10..21);
        let lanes = if wave64 { 64 } else { 32 };
        for l in 0..lanes {
            if (l & 3) == 3 {
                assert_eq!(&r.vgpr[l][10..21], &[0xDEAD_BEEF; 11], "inactive lane {l}");
                let bit = 1u32 << (l & 31);
                let idx = if l < 32 { 20 } else { 21 };
                assert_eq!(r.sgpr[idx] & bit, 0, "inactive lane {l} carry bit must be 0");
                continue;
            }
            let (a, b) = pairs_u64[l & 3];
            let sum = a.wrapping_add(b);
            let diff = a.wrapping_sub(b);
            let rdiff = b.wrapping_sub(a);
            assert_eq!((r.vgpr[l][10], r.vgpr[l][11]), (sum as u32, (sum >> 32) as u32));
            assert_eq!((r.vgpr[l][12], r.vgpr[l][13]), (diff as u32, (diff >> 32) as u32));
            assert_eq!((r.vgpr[l][14], r.vgpr[l][15]), (rdiff as u32, (rdiff >> 32) as u32));
            let (si0, si1) = pairs_i32[l & 3];
            assert_eq!(r.vgpr[l][19], si0.saturating_add(si1) as u32);
            assert_eq!(r.vgpr[l][20], si0.saturating_sub(si1) as u32);
        }
    }
}

#[test]
fn direct_dxil_valu_compares_and_vcmpx_vcc_preservation_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    let code = prog(&[
        // f32 ordered & unordered compares (v1 vs v2: includes NaN, -0/+0, ±inf)
        ("V_CMP_EQ_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 1), ("SRC1", V + 2)], None),
        ("V_CMP_LT_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", V + 2)], None),
        ("V_CMP_O_F32", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 1), ("SRC1", V + 2)], None),
        ("V_CMP_U_F32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 1), ("SRC1", V + 2)], None),
        ("V_CMP_NEQ_F32", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 1), ("SRC1", V + 2)], None),
        // VOP3 ABS on src0, NEG on src1
        ("V_CMP_LT_F32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 1), ("SRC1", V + 2), ("ABS", 1), ("NEG", 2)], None),
        // 64-bit integer compares (v[3:4] vs v[5:6], signed & unsigned)
        ("V_CMP_EQ_U64", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 3), ("SRC1", V + 5)], None),
        ("V_CMP_LT_I64", "ENC_VOP3", &[("VDST", 24), ("SRC0", V + 3), ("SRC1", V + 5)], None),
        ("V_CMP_GT_U64", "ENC_VOP3", &[("VDST", 26), ("SRC0", V + 3), ("SRC1", V + 5)], None),
        // Seed VCC sentinel, then narrow EXEC with V_CMPX_LT_I32 (must keep VCC untouched)
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 106), ("SSRC0", 255)], Some(0x96A5_3C69)),
        ("V_CMPX_LT_I32", "ENC_VOPC", &[("SRC0", V + 7), ("VSRC1", 8)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 28), ("SSRC0", 126)], None), // exec_lo
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 29), ("SSRC0", 106)], None), // vcc_lo sentinel
    ]);

    let f32_pairs: [(f32, f32); 4] = [
        (-0.0, 0.0),          // equal, ordered
        (-2.0, 1.5),          // lt, ordered; |a| > -b
        (f32::NAN, 1.0),      // unordered; neq true, eq/lt/o false
        (f32::NEG_INFINITY, f32::INFINITY),
    ];
    let u64_pairs: [(u64, u64); 4] = [
        (0, 0),
        (0x8000_0000_0000_0000, 1), // signed <, unsigned >
        (0x0000_0001_0000_0000, 0x0000_0000_FFFF_FFFF),
        (0xFFFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000),
    ];

    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (fa, fb) = f32_pairs[l & 3];
        r[1] = fa.to_bits();
        r[2] = fb.to_bits();
        let (ua, ub) = u64_pairs[l & 3];
        r[3] = ua as u32;
        r[4] = (ua >> 32) as u32;
        r[5] = ub as u32;
        r[6] = (ub >> 32) as u32;
        r[7] = if (l & 1) == 0 { (-5i32) as u32 } else { 5 };
        r[8] = 0;
    }

    for (wave64, split) in modes {
        let watched_sgprs: Vec<usize> = if wave64 {
            (10..30).collect()
        } else {
            vec![10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29]
        };
        let r = run_parity("valu_compares_vcmpx", &code, &st, wave64, split, &watched_sgprs, 0..0);
        // Lane (l & 3) == 0 (-0 == +0) -> 0x1111_1111 in low 32 lanes
        assert_eq!(r.sgpr[10], 0x1111_1111, "v_cmp_eq_f32");
        assert_eq!(r.sgpr[14], 0xBBBB_BBBB, "v_cmp_o_f32 (all except r=2 NaN)");
        assert_eq!(r.sgpr[16], 0x4444_4444, "v_cmp_u_f32 (r=2 NaN only)");
        assert_eq!(r.sgpr[22], 0x1111_1111, "v_cmp_eq_u64");
        assert_eq!(r.sgpr[24], 0x2222_2222, "v_cmp_lt_i64");
        assert_eq!(r.sgpr[28], 0x5555_5555, "v_cmpx_lt_i32 narrowed exec_lo");
        assert_eq!(r.sgpr[29], 0x96A5_3C69, "v_cmpx must leave vcc_lo sentinel untouched");
    }
}

#[test]
fn direct_dxil_valu_vop3_modifiers_and_cndmask_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    let f = |x: f32| x.to_bits();
    let code = prog(&[
        // OMOD *2 (1), *4 (2), /2 (3), and OMOD *2 before CLAMP [0, 1]
        ("V_ADD_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 1), ("SRC1", V + 2), ("ABS", 1), ("NEG", 2)], None),
        ("V_MUL_F32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 1), ("SRC1", V + 2), ("OMOD", 1)], None),
        ("V_MUL_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", V + 2), ("OMOD", 2)], None),
        ("V_ADD_F32", "ENC_VOP3", &[("VDST", 13), ("SRC0", V + 1), ("SRC1", V + 2), ("OMOD", 3)], None),
        // 0.35 + 0.35 = 0.70; *2 -> 1.40; clamp -> 1.0 (proves OMOD runs before CLAMP)
        ("V_ADD_F32", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 3), ("SRC1", V + 3), ("OMOD", 1), ("CLAMP", 1)], None),
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 15), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", V + 3), ("ABS", 1), ("NEG", 4), ("CLAMP", 1)], None),
        ("V_MIN_F32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 1), ("SRC1", V + 2), ("NEG", 1)], None),
        ("V_MAX_F32", "ENC_VOP3", &[("VDST", 17), ("SRC0", V + 1), ("SRC1", V + 2), ("ABS", 3)], None),
        // V_CNDMASK_B32 VOP3 with SGPR mask s[20:21] and VOP2 with VCC
        ("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 0), ("SRC1", 128 + 15)], None),
        ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 1), ("SRC1", V + 2), ("SRC2", 20)], None),
        // V_CVT_F16_F32 with OMOD *2 and OP_SEL writing high half while preserving low half
        ("V_CVT_F16_F32", "ENC_VOP3", &[("VDST", 19), ("SRC0", V + 3), ("OMOD", 1), ("OP_SEL", 8)], None),
    ]);

    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[1] = f(if (l & 1) == 0 { -1.5 } else { 2.25 });
        r[2] = f(0.5);
        r[3] = f(0.35);
        r[19] = 0x0000_5A5A; // low half must be preserved when OP_SEL bit 3 is set
    }

    for (wave64, split) in modes {
        let watched_sgprs: &[usize] = if wave64 { &[20, 21] } else { &[20] };
        let r = run_parity("valu_vop3_mods_cndmask", &code, &st, wave64, split, watched_sgprs, 10..20);
        assert_eq!(f32::from_bits(r.vgpr[0][10]), 1.0, "|-1.5| + (-0.5) = 1.0");
        assert_eq!(f32::from_bits(r.vgpr[0][14]), 1.0, "(0.35 + 0.35) * 2 clamped to 1.0");
        assert_eq!(f32::from_bits(r.vgpr[0][18]), -1.5, "lane 0 cndmask picks src0");
        assert_eq!(f32::from_bits(r.vgpr[16][18]), 0.5, "lane 16 cndmask picks src1");
        assert_eq!(r.vgpr[0][19] & 0xFFFF, 0x5A5A, "v_cvt_f16_f32 op_sel[3] preserves low half");
    }
}

#[test]
fn direct_dxil_valu_conversions_and_pack_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    let f = |x: f32| x.to_bits();
    let code = prog(&[
        ("V_CVT_FLR_I32_F32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 1)], None),
        ("V_CVT_RPI_I32_F32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 1)], None),
        ("V_CVT_I32_F32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 1)], None),
        ("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 13), ("SRC0", V + 2)], None),
        ("V_CVT_F32_I32", "ENC_VOP1", &[("VDST", 14), ("SRC0", V + 3)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 15), ("SRC0", V + 3)], None),
        ("V_CVT_F32_UBYTE0", "ENC_VOP1", &[("VDST", 16), ("SRC0", V + 4)], None),
        ("V_CVT_F32_UBYTE1", "ENC_VOP1", &[("VDST", 17), ("SRC0", V + 4)], None),
        ("V_CVT_F32_UBYTE2", "ENC_VOP1", &[("VDST", 18), ("SRC0", V + 4)], None),
        ("V_CVT_F32_UBYTE3", "ENC_VOP1", &[("VDST", 19), ("SRC0", V + 4)], None),
        ("V_CVT_F32_F16", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 5), ("OP_SEL", 0)], None),
        ("V_CVT_F32_F16", "ENC_VOP3", &[("VDST", 21), ("SRC0", V + 5), ("OP_SEL", 1)], None),
        ("V_CVT_PK_I16_I32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 6), ("SRC1", V + 7)], None),
        ("V_CVT_PK_U16_U32", "ENC_VOP3", &[("VDST", 23), ("SRC0", V + 6), ("SRC1", V + 7)], None),
    ]);

    let floats = [-2.6f32, -1.5, -0.4, 0.5, 1.5, 2.75, 0.0, 100.25];
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[1] = f(floats[l & 7]);
        r[2] = f(floats[l & 7].abs());
        r[3] = if (l & 1) == 0 { (-42i32) as u32 } else { 0x0001_0000 + l as u32 };
        r[4] = 0x10_20_30_40u32.wrapping_add(l as u32);
        r[5] = 0xC000_3C00; // low = +1.0, high = -2.0
        r[6] = (-40_000i32 + (l as i32) * 1000) as u32;
        r[7] = (30_000i32 + (l as i32) * 500) as u32;
    }

    for (wave64, split) in modes {
        let r = run_parity("valu_conversions_pack", &code, &st, wave64, split, &[], 10..24);
        // Lane 0: -2.6 -> flr=-3, rpi=-3, trunc=-2
        assert_eq!(r.vgpr[0][10], (-3i32) as u32);
        assert_eq!(r.vgpr[0][11], (-3i32) as u32);
        assert_eq!(r.vgpr[0][12], (-2i32) as u32);
        // Lane 1: -1.5 -> flr=-2, rpi=floor(-1.5 + 0.5) = -1
        assert_eq!(r.vgpr[1][10], (-2i32) as u32);
        assert_eq!(r.vgpr[1][11], (-1i32) as u32);
        // UBYTE0..3 of lane 0 (0x10_20_30_40) -> 64.0, 48.0, 32.0, 16.0
        assert_eq!(f32::from_bits(r.vgpr[0][16]), 64.0);
        assert_eq!(f32::from_bits(r.vgpr[0][17]), 48.0);
        assert_eq!(f32::from_bits(r.vgpr[0][18]), 32.0);
        assert_eq!(f32::from_bits(r.vgpr[0][19]), 16.0);
        // F32_F16 low/high half
        assert_eq!(f32::from_bits(r.vgpr[0][20]), 1.0);
        assert_eq!(f32::from_bits(r.vgpr[0][21]), -2.0);
        // PK_I16_I32 lane 0: clamp(-40000) -> -32768 (0x8000), 30000 (0x7530)
        assert_eq!(r.vgpr[0][22], 0x7530_8000);
    }
}
