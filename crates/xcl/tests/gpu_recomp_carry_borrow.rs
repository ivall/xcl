//! GPU verification tests for RDNA2 VALU carry/borrow arithmetic (`V_ADD_CO_U32`,
//! `V_SUB_CO_U32`, `V_SUBREV_CO_U32`, `V_ADD_CO_CI_U32`, `V_SUB_CO_CI_U32`,
//! `V_SUBREV_CO_CI_U32`) and `V_MAD_U64_U32`.
//!
//! Covers both `ENC_VOP2` (implicit `VCC`) and `VOP3_SDST_ENC` (explicit `SDST` / `SRC2`),
//! multi-word 96-bit add/subtract chains (including `x64_masks` half-local intermediate
//! carry/borrow masks in split wave64), in-place `VDST == SRC0` / `VDST == SRC2` and
//! `SDST == SRC2` aliasing, SGPR / inline-constant / 32-bit literal operands, divergent
//! `EXEC` masks, and all supported wave execution modes (native wave64, split wave64,
//! wave32, and wave32 on 64-lane hardware).

use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner::{self, KernelState};
use xcl::gpu::recomp::{dxc, translate, Options};

const V: u64 = 256; // VALU source encoding base for VGPRs

type Insn = (
    &'static str,
    &'static str,
    Vec<(&'static str, u64)>,
    Option<u32>,
);

fn vop3b(name: &'static str, vd: u64, sd: u64, s0: u64, s1: u64, lit: Option<u32>) -> Insn {
    (
        name,
        "VOP3_SDST_ENC",
        vec![("VDST", vd), ("SDST", sd), ("SRC0", s0), ("SRC1", s1)],
        lit,
    )
}

fn vop3b_ci(name: &'static str, vd: u64, sd: u64, s0: u64, s1: u64, s2: u64) -> Insn {
    (
        name,
        "VOP3_SDST_ENC",
        vec![
            ("VDST", vd),
            ("SDST", sd),
            ("SRC0", s0),
            ("SRC1", s1),
            ("SRC2", s2),
        ],
        None,
    )
}

fn vop2(name: &'static str, vd: u64, s0: u64, vs1: u64) -> Insn {
    (
        name,
        "ENC_VOP2",
        vec![("VDST", vd), ("SRC0", s0), ("VSRC1", vs1)],
        None,
    )
}

fn s_mov_b64(sd: u64, s0: u64) -> Insn {
    (
        "S_MOV_B64",
        "ENC_SOP1",
        vec![("SDST", sd), ("SSRC0", s0)],
        None,
    )
}

fn v_cndmask3(vd: u64, s0: u64, s1: u64, s2: u64) -> Insn {
    (
        "V_CNDMASK_B32",
        "ENC_VOP3",
        vec![("VDST", vd), ("SRC0", s0), ("SRC1", s1), ("SRC2", s2)],
        None,
    )
}

fn prog(list: &[Insn]) -> Vec<u8> {
    let mut code = Vec::new();
    for (name, enc, fields, lit) in list {
        code.extend(encode(name, enc, fields, *lit).unwrap_or_else(|e| panic!("{name}: {e}")));
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    code
}

fn opt_mode(wave64: bool, split_wave64: bool, hw_wave: u32) -> Options {
    Options {
        wave64,
        split_wave64,
        hw_wave,
        ..Default::default()
    }
}

fn gpu_ready() -> bool {
    match (
        runner::adapter(),
        runner::wave64_programs_supported(),
        dxc::location(),
    ) {
        (Ok(_), Ok(true), Ok(_)) => true,
        (a, w, d) => {
            println!(
                "GPU/DXC unavailable – skipping ({:?} / {:?} / {:?})",
                a.as_ref().err(),
                w.as_ref().ok(),
                d.as_ref().err()
            );
            let reason = if a.is_err() || w.is_err() {
                "no GPU"
            } else if d.is_err() {
                "no DXC"
            } else {
                "no wave64"
            };
            runner::skip_test(reason);
            false
        }
    }
}

fn wave_configs() -> Vec<(&'static str, Options)> {
    if !gpu_ready() {
        return Vec::new();
    }
    let hw64 = runner::wave64_supported() == Ok(true);
    let mut cfgs = Vec::new();
    if hw64 {
        cfgs.push(("wave64", opt_mode(true, false, 0)));
    }
    cfgs.push(("split-wave64", opt_mode(true, true, 0)));
    cfgs.push(("wave32", opt_mode(false, false, 0)));
    if hw64 {
        cfgs.push(("wave32@hw64", opt_mode(false, false, 64)));
    }
    cfgs
}

fn compile_kernel(code: &[u8], opt: &Options) -> Vec<u8> {
    let t = translate(code, opt).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl))
}

fn active_lanes_for(opt: &Options) -> usize {
    if opt.wave64 || opt.hw_wave == 64 {
        64
    } else {
        32
    }
}

fn is_active(lane: usize, lo: u32, hi: u32, opt: &Options) -> bool {
    let (mask, bit) = if opt.wave64 {
        ((lo as u64) | ((hi as u64) << 32), lane)
    } else {
        (lo as u64, lane & 31)
    };
    ((mask >> bit) & 1) != 0
}

/// Verifies `SDST` / `VCC` readback in `r.sgpr`. In `wave32`, `write_mask` updates only the low
/// 32-bit SGPR; `hi_w32` is `reg + 1` (preloaded sentinel if referenced, else `0`).
fn check_mask(
    r: &runner::KernelResult,
    reg: usize,
    want: u64,
    hi_w32: u32,
    opt: &Options,
    ctx: &str,
) {
    assert_eq!(r.sgpr[reg], want as u32, "{ctx} lo (s{reg})");
    let want_hi = if opt.wave64 {
        (want >> 32) as u32
    } else {
        hi_w32
    };
    assert_eq!(r.sgpr[reg + 1], want_hi, "{ctx} hi (s{})", reg + 1);
}

#[derive(Clone, Copy)]
enum CoOp {
    Add,
    Sub,
    SubRev,
}

fn eval_co(op: CoOp, a: u32, b: u32, cin: bool) -> (u32, bool) {
    let c = cin as u64;
    match op {
        CoOp::Add => {
            let s = (a as u64) + (b as u64) + c;
            (s as u32, (s >> 32) != 0)
        }
        CoOp::Sub => (
            a.wrapping_sub(b).wrapping_sub(c as u32),
            (a as u64) < (b as u64) + c,
        ),
        CoOp::SubRev => (
            b.wrapping_sub(a).wrapping_sub(c as u32),
            (b as u64) < (a as u64) + c,
        ),
    }
}

fn test_pairs() -> [(u32, u32); 64] {
    const EDGES: [(u32, u32); 16] = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0xFFFF_FFFF, 0),
        (0xFFFF_FFFF, 1),
        (0xFFFF_FFFF, 0xFFFF_FFFF),
        (0, 0xFFFF_FFFF),
        (0x8000_0000, 0x8000_0000),
        (0x7FFF_FFFF, 0x8000_0000),
        (0x7FFF_FFFF, 0x8000_0001),
        (0x8000_0000, 0x7FFF_FFFF),
        (0xFFFF_FFFE, 1),
        (0xFFFF_FFFE, 2),
        (0xDEAD_BEEF, 0xDEAD_BEEF),
        (0x1234_5678, 0xEDCB_A988),
    ];
    let mut seed = 0x9E37_79B9u32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed
    };
    std::array::from_fn(|l| {
        if l < EDGES.len() {
            EDGES[l]
        } else {
            (next(), next())
        }
    })
}

#[test]
fn carry_borrow_translate_and_compile_with_dxc() {
    let code = prog(&[
        // Intermediate carry/borrow in s[20:21] consumed only lane-by-lane, then overwritten.
        vop3b("V_ADD_CO_U32", 10, 20, V, V + 1, None),
        vop3b_ci("V_ADD_CO_CI_U32", 11, 20, V + 2, V + 3, 20),
        vop3b("V_SUB_CO_U32", 12, 20, V, V + 1, None),
        vop3b_ci("V_SUB_CO_CI_U32", 13, 20, V + 2, V + 3, 20),
        vop3b("V_SUBREV_CO_U32", 14, 20, V, V + 1, None),
        vop3b_ci("V_SUBREV_CO_CI_U32", 15, 20, V + 2, V + 3, 20),
        s_mov_b64(20, 128),
        // VOP2 forms using implicit VCC, plus final whole-live V_MAD_U64_U32 -> s[22:23].
        vop2("V_ADD_CO_CI_U32", 16, V, 1),
        vop2("V_SUB_CO_CI_U32", 17, V, 1),
        vop2("V_SUBREV_CO_CI_U32", 18, V, 1),
        vop3b_ci("V_MAD_U64_U32", 20, 22, V, V + 1, V + 4),
    ]);

    let has_dxc = dxc::location().is_ok();
    if !has_dxc {
        println!("no DXC – skipping DXIL compilation");
        runner::skip_test("no DXC");
    }

    for (label, opt) in [
        ("wave64", opt_mode(true, false, 0)),
        ("split-wave64", opt_mode(true, true, 0)),
        ("wave32", opt_mode(false, false, 0)),
        ("wave32@hw64", opt_mode(false, false, 64)),
    ] {
        let t = translate(&code, &opt).unwrap();
        assert!(t.complete(), "{label}: {:?}\n{}", t.unsupported_why, t.hlsl);
        if opt.wave64 && opt.split_wave64 {
            assert!(
                t.hlsl.contains("xcl_x_ballot_local("),
                "{label}: expected xcl_x_ballot_local\n{}",
                t.hlsl
            );
            assert!(
                t.hlsl.contains("xcl_x_ballot("),
                "{label}: expected xcl_x_ballot\n{}",
                t.hlsl
            );
        }
        if has_dxc {
            dxc::compile(&t.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{label}: {e}\n{}", t.hlsl));
        }
    }
}

#[test]
fn gpu_carry_borrow_single_ops_and_edge_cases() {
    let configs = wave_configs();
    if configs.is_empty() {
        return;
    }

    let pairs = test_pairs();
    let (cin_add_lo, cin_add_hi) = (0x5A3C_96A5u32, 0xC35A_693Cu32);
    let (cin_sub_lo, cin_sub_hi) = (0xA5C3_695Au32, 0x3CA5_96C3u32);
    let (cin_rev_lo, cin_rev_hi) = (0x0F0F_F0F0u32, 0x55AA_AA55u32);
    let (vcc_in_lo, vcc_in_hi) = (0x6969_9696u32, 0xF00F_0FF0u32);
    let s0_const = 0x8000_0001u32;
    let lit_val = 0x7FFF_FFFFu32;

    let code = prog(&[
        vop3b("V_ADD_CO_U32", 10, 10, V, V + 1, None),
        vop3b("V_SUB_CO_U32", 11, 12, V, V + 1, None),
        vop3b("V_SUBREV_CO_U32", 12, 14, V, V + 1, None),
        vop3b_ci("V_ADD_CO_CI_U32", 13, 16, V, V + 1, 2),
        vop3b_ci("V_SUB_CO_CI_U32", 14, 18, V, V + 1, 4),
        vop3b_ci("V_SUBREV_CO_CI_U32", 15, 20, V, V + 1, 6),
        vop3b("V_SUB_CO_U32", 16, 22, 0, V, None),
        vop3b("V_SUBREV_CO_U32", 17, 24, 193, V, None),
        vop3b("V_ADD_CO_U32", 18, 26, 255, V, Some(lit_val)),
        vop2("V_ADD_CO_CI_U32", 20, V, 1),
        s_mov_b64(28, 106),
        s_mov_b64(106, 8),
        vop2("V_SUB_CO_CI_U32", 21, V, 1),
        s_mov_b64(30, 106),
        s_mov_b64(106, 8),
        vop2("V_SUBREV_CO_CI_U32", 22, V, 1),
        // Capture per-lane mask bits so wave32@hw64 verifies upper 32 hardware lanes too.
        v_cndmask3(23, 128, 129, 14),
        vop2("V_CNDMASK_B32", 24, 128, 2),
    ]);

    let mut st = KernelState::default();
    st.sgpr[0] = s0_const;
    st.sgpr[2..10].copy_from_slice(&[
        cin_add_lo, cin_add_hi, cin_sub_lo, cin_sub_hi, cin_rev_lo, cin_rev_hi, vcc_in_lo,
        vcc_in_hi,
    ]);
    st.sgpr[106] = vcc_in_lo;
    st.sgpr[107] = vcc_in_hi;
    for reg in [10usize, 12, 14, 16, 18, 20, 22, 24, 26] {
        st.sgpr[reg + 1] = 0xCAFE_0000 | (reg as u32 + 1);
    }
    for (l, &(a, b)) in pairs.iter().enumerate() {
        st.vgpr[l][0] = a;
        st.vgpr[l][1] = b;
        st.vgpr[l][2] = 1;
    }

    for (label, opt) in configs {
        let dxil = compile_kernel(&code, &opt);
        let r = runner::run(&dxil, &st, &[]).unwrap();
        let active_lanes = active_lanes_for(&opt);

        let mut want_masks = [0u64; 12];
        for l in 0..active_lanes {
            let (a, b) = pairs[l];
            let v = &r.vgpr[l];
            let vcc_bit = is_active(l, vcc_in_lo, vcc_in_hi, &opt);
            let res = [
                eval_co(CoOp::Add, a, b, false),
                eval_co(CoOp::Sub, a, b, false),
                eval_co(CoOp::SubRev, a, b, false),
                eval_co(CoOp::Add, a, b, is_active(l, cin_add_lo, cin_add_hi, &opt)),
                eval_co(CoOp::Sub, a, b, is_active(l, cin_sub_lo, cin_sub_hi, &opt)),
                eval_co(
                    CoOp::SubRev,
                    a,
                    b,
                    is_active(l, cin_rev_lo, cin_rev_hi, &opt),
                ),
                eval_co(CoOp::Sub, s0_const, a, false),
                eval_co(CoOp::SubRev, u32::MAX, a, false),
                eval_co(CoOp::Add, lit_val, a, false),
                eval_co(CoOp::Add, a, b, vcc_bit),
                eval_co(CoOp::Sub, a, b, vcc_bit),
                eval_co(CoOp::SubRev, a, b, vcc_bit),
            ];
            let dsts = [10usize, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22];
            for (&d, &(want, _)) in dsts.iter().zip(&res) {
                assert_eq!(v[d], want, "{label} lane {l} v{d}");
            }
            assert_eq!(v[23], res[2].1 as u32, "{label} lane {l} s14 cndmask");
            assert_eq!(v[24], res[11].1 as u32, "{label} lane {l} vcc cndmask");
            if opt.wave64 || l < 32 {
                for (k, &(_, bit)) in res.iter().enumerate() {
                    if bit {
                        want_masks[k] |= 1u64 << l;
                    }
                }
            }
        }

        for (idx, &reg) in [10usize, 12, 14, 16, 18, 20, 22, 24, 26].iter().enumerate() {
            let hi_w32 = if reg == 14 { 0xCAFE_000F } else { 0 };
            check_mask(
                &r,
                reg,
                want_masks[idx],
                hi_w32,
                &opt,
                &format!("{label} op{idx}"),
            );
        }
        check_mask(
            &r,
            28,
            want_masks[9],
            vcc_in_hi,
            &opt,
            &format!("{label} vop2 add"),
        );
        check_mask(
            &r,
            30,
            want_masks[10],
            vcc_in_hi,
            &opt,
            &format!("{label} vop2 sub"),
        );
        check_mask(
            &r,
            106,
            want_masks[11],
            vcc_in_hi,
            &opt,
            &format!("{label} vop2 subrev"),
        );
    }
}

#[test]
fn gpu_multiword_chains_in_place_and_divergent_exec() {
    let configs = wave_configs();
    if configs.is_empty() {
        return;
    }

    let (exec_lo, exec_hi) = (0x5A3C_96A5u32, 0x3C69_A55Au32);
    let code = prog(&[
        // Chain 1: A + B -> v[10:12], local-only intermediate s[20:21], final carry -> s[10:11]
        vop3b("V_ADD_CO_U32", 10, 20, V, V + 3, None),
        vop3b_ci("V_ADD_CO_CI_U32", 11, 20, V + 1, V + 4, 20),
        vop3b_ci("V_ADD_CO_CI_U32", 12, 10, V + 2, V + 5, 20),
        // Chain 2: A - B -> v[13:15], final borrow -> s[12:13]
        vop3b("V_SUB_CO_U32", 13, 20, V, V + 3, None),
        vop3b_ci("V_SUB_CO_CI_U32", 14, 106, V + 1, V + 4, 20),
        vop2("V_SUB_CO_CI_U32", 15, V + 2, 5),
        s_mov_b64(12, 106),
        // Chain 3: B - A -> v[16:18], final borrow -> VCC (s[106:107])
        vop3b("V_SUBREV_CO_U32", 16, 20, V, V + 3, None),
        vop3b_ci("V_SUBREV_CO_CI_U32", 17, 106, V + 1, V + 4, 20),
        vop2("V_SUBREV_CO_CI_U32", 18, V + 2, 5),
        // Overwrite s[20:21] so x64_masks keeps its earlier writes half-local.
        s_mov_b64(20, 128),
        // In-place VDST == SRC0 and SDST == SRC2: v[30:32] = B - v[30:32]
        vop3b("V_SUBREV_CO_U32", 30, 14, V + 30, V + 3, None),
        vop3b_ci("V_SUBREV_CO_CI_U32", 31, 14, V + 31, V + 4, 14),
        vop3b_ci("V_SUBREV_CO_CI_U32", 32, 14, V + 32, V + 5, 14),
    ]);

    let mut st = KernelState::default();
    st.sgpr[126] = exec_lo;
    st.sgpr[127] = exec_hi;

    let mut limbs_a = [[0u32; 3]; 64];
    let mut limbs_b = [[0u32; 3]; 64];
    let mut seed = 0x1357_2468u32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed
    };

    for l in 0..64 {
        let (a, b) = match l % 8 {
            0 => ([0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF], [1, 0, 0]),
            1 => ([0, 0, 0], [1, 0, 0]),
            2 => ([1, 0, 0], [1, 0, 0]),
            3 => ([0xFFFF_FFFF, 0xFFFF_FFFF, 0], [1, 0, 0]),
            4 => ([0, 0, 1], [1, 0, 0]),
            _ => ([next(), next(), next()], [next(), next(), next()]),
        };
        limbs_a[l] = a;
        limbs_b[l] = b;
        st.vgpr[l][0..3].copy_from_slice(&a);
        st.vgpr[l][3..6].copy_from_slice(&b);
        st.vgpr[l][30..33].copy_from_slice(&a);
        for (off, slot) in st.vgpr[l][10..=18].iter_mut().enumerate() {
            *slot = 0xDEAD_0000 | (((10 + off) as u32) << 8) | l as u32;
        }
    }

    let as_u96 = |w: [u32; 3]| (w[0] as u128) | ((w[1] as u128) << 32) | ((w[2] as u128) << 64);
    let to_limbs = |x: u128| [x as u32, (x >> 32) as u32, (x >> 64) as u32];
    let mask96 = (1u128 << 96) - 1;

    for (label, opt) in configs {
        let dxil = compile_kernel(&code, &opt);
        let r = runner::run(&dxil, &st, &[]).unwrap();
        let active_lanes = active_lanes_for(&opt);

        let (mut add_c_mask, mut sub_b_mask, mut rev_b_mask) = (0u64, 0u64, 0u64);
        for l in 0..active_lanes {
            let v = &r.vgpr[l];
            if !is_active(l, exec_lo, exec_hi, &opt) {
                for (off, &got) in v[10..=18].iter().enumerate() {
                    let sentinel = 0xDEAD_0000 | (((10 + off) as u32) << 8) | l as u32;
                    assert_eq!(got, sentinel, "{label} inactive lane {l} v{}", 10 + off);
                }
                assert_eq!(
                    &v[30..33],
                    &limbs_a[l],
                    "{label} inactive lane {l} in-place"
                );
                continue;
            }

            let (a, b) = (as_u96(limbs_a[l]), as_u96(limbs_b[l]));
            let add_sum = a + b;
            let rev_limbs = to_limbs(b.wrapping_sub(a) & mask96);
            assert_eq!(
                &v[10..13],
                &to_limbs(add_sum),
                "{label} lane {l} 96-bit add"
            );
            assert_eq!(
                &v[13..16],
                &to_limbs(a.wrapping_sub(b) & mask96),
                "{label} lane {l} 96-bit sub"
            );
            assert_eq!(&v[16..19], &rev_limbs, "{label} lane {l} 96-bit subrev");
            assert_eq!(&v[30..33], &rev_limbs, "{label} lane {l} 96-bit in-place");

            if opt.wave64 || l < 32 {
                if (add_sum >> 96) != 0 {
                    add_c_mask |= 1u64 << l;
                }
                if a < b {
                    sub_b_mask |= 1u64 << l;
                }
                if b < a {
                    rev_b_mask |= 1u64 << l;
                }
            }
        }

        check_mask(&r, 10, add_c_mask, 0, &opt, &format!("{label} 96-bit add"));
        check_mask(&r, 12, sub_b_mask, 0, &opt, &format!("{label} 96-bit sub"));
        check_mask(
            &r,
            14,
            rev_b_mask,
            0,
            &opt,
            &format!("{label} 96-bit in-place"),
        );
        check_mask(&r, 106, rev_b_mask, 0, &opt, &format!("{label} 96-bit vcc"));
        assert_eq!(
            (r.sgpr[20], r.sgpr[21]),
            (0, 0),
            "{label} cleared temp mask"
        );
    }
}

#[test]
fn gpu_v_mad_u64_u32_carry_and_aliasing() {
    let configs = wave_configs();
    if configs.is_empty() {
        return;
    }

    let (exec_lo, exec_hi) = (0x7E5A_C396u32, 0xA596_3C7Eu32);
    let sgpr_m = 0xFFFF_FFFFu32;
    let lit_m = 0x8000_0001u32;

    let code = prog(&[
        vop3b_ci("V_MAD_U64_U32", 10, 10, V, V + 1, V + 2),
        v_cndmask3(22, 128, 129, 10),
        vop3b_ci("V_MAD_U64_U32", 12, 12, 0, 130, 193),
        (
            "V_MAD_U64_U32",
            "VOP3_SDST_ENC",
            vec![
                ("VDST", 14),
                ("SDST", 14),
                ("SRC0", 255),
                ("SRC1", V + 1),
                ("SRC2", 2),
            ],
            Some(lit_m),
        ),
        vop3b_ci("V_MAD_U64_U32", 20, 16, V + 20, V + 1, V + 20),
    ]);

    const EDGES: [(u32, u32, u64); 8] = [
        (0, 0, 0),
        (0xFFFF_FFFF, 0xFFFF_FFFF, 0),
        (0xFFFF_FFFF, 0xFFFF_FFFF, 0x0000_0001_FFFF_FFFE),
        (0xFFFF_FFFF, 0xFFFF_FFFF, 0x0000_0001_FFFF_FFFF),
        (0xFFFF_FFFF, 0xFFFF_FFFF, u64::MAX),
        (1, 1, u64::MAX),
        (0x8000_0000, 2, 0xFFFF_FFFF_0000_0000),
        (0x8000_0000, 2, 0xFFFF_FFFE_FFFF_FFFF),
    ];

    let mut st = KernelState::default();
    st.sgpr[0] = sgpr_m;
    st.sgpr[2] = 0xFFFF_FFFF;
    st.sgpr[3] = 0xBFFF_FFFF;
    st.sgpr[11] = 0xCAFE_000B;
    st.sgpr[126] = exec_lo;
    st.sgpr[127] = exec_hi;

    let mut seed = 0xCAFE_BABEu32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed
    };

    let mut cases = [(0u32, 0u32, 0u64); 64];
    for l in 0..64 {
        let (a, b, c) = if l < EDGES.len() {
            EDGES[l]
        } else {
            (next(), next(), ((next() as u64) << 32) | (next() as u64))
        };
        cases[l] = (a, b, c);
        st.vgpr[l][0] = a;
        st.vgpr[l][1] = b;
        st.vgpr[l][2] = c as u32;
        st.vgpr[l][3] = (c >> 32) as u32;
        st.vgpr[l][20] = c as u32;
        st.vgpr[l][21] = (c >> 32) as u32;
        for (off, slot) in st.vgpr[l][10..=15].iter_mut().enumerate() {
            *slot = 0xDEAD_0000 | (((10 + off) as u32) << 8) | l as u32;
        }
        st.vgpr[l][22] = 0xDEAD_1600 | l as u32;
    }

    let sgpr_c = (st.sgpr[2] as u64) | ((st.sgpr[3] as u64) << 32);

    for (label, opt) in configs {
        let dxil = compile_kernel(&code, &opt);
        let r = runner::run(&dxil, &st, &[]).unwrap();
        let active_lanes = active_lanes_for(&opt);

        let mut want_masks = [0u64; 4];
        for l in 0..active_lanes {
            let (a, b, c) = cases[l];
            let v = &r.vgpr[l];
            if !is_active(l, exec_lo, exec_hi, &opt) {
                for (off, &got) in v[10..=15].iter().enumerate() {
                    let sentinel = 0xDEAD_0000 | (((10 + off) as u32) << 8) | l as u32;
                    assert_eq!(got, sentinel, "{label} inactive lane {l} v{}", 10 + off);
                }
                assert_eq!(
                    v[22],
                    0xDEAD_1600 | l as u32,
                    "{label} inactive lane {l} v22"
                );
                assert_eq!(
                    (v[20], v[21]),
                    (c as u32, (c >> 32) as u32),
                    "{label} inactive lane {l} in-place"
                );
                continue;
            }

            let eval = |x: u32, y: u32, z: u64| -> (u32, u32, bool) {
                let s = (x as u128) * (y as u128) + (z as u128);
                (s as u32, (s >> 32) as u32, (s >> 64) != 0)
            };

            let (w0l, w0h, c0) = eval(a, b, c);
            let (w1l, w1h, c1) = eval(sgpr_m, 2, u64::MAX);
            let (w2l, w2h, c2) = eval(lit_m, b, sgpr_c);
            let (w3l, w3h, c3) = eval(c as u32, b, c);

            assert_eq!((v[10], v[11]), (w0l, w0h), "{label} lane {l} v_mad_u64_u32");
            assert_eq!(v[22], c0 as u32, "{label} lane {l} v_mad_u64_u32 carry");
            assert_eq!((v[12], v[13]), (w1l, w1h), "{label} lane {l} inline -1");
            assert_eq!((v[14], v[15]), (w2l, w2h), "{label} lane {l} literal");
            assert_eq!((v[20], v[21]), (w3l, w3h), "{label} lane {l} in-place");

            if opt.wave64 || l < 32 {
                for (k, &bit) in [c0, c1, c2, c3].iter().enumerate() {
                    if bit {
                        want_masks[k] |= 1u64 << l;
                    }
                }
            }
        }

        for (idx, reg) in [10usize, 12, 14, 16].iter().enumerate() {
            let hi_w32 = if *reg == 10 { 0xCAFE_000B } else { 0 };
            check_mask(
                &r,
                *reg,
                want_masks[idx],
                hi_w32,
                &opt,
                &format!("{label} mad_u64 {idx}"),
            );
        }
    }
}
