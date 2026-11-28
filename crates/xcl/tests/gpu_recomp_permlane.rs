//! GPU verification tests for RDNA2 `V_PERMLANE16_B32` and `V_PERMLANEX16_B32`.
//!
//! Covers row-local vs cross-row (16-lane halves inside each 32-lane half) permutations,
//! SGPR / inline-constant / 32-bit literal select masks, in-place `VDST == SRC0` aliasing,
//! all four `(FI, BOUND_CTRL)` flag combinations (`OP_SEL[0..1]`) under a divergent `EXEC`
//! mask, and all supported wave execution modes (native wave64, split wave64, wave32, and
//! wave32 on 64-lane hardware).

use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner::{self, KernelState};
use xcl::gpu::recomp::{dxc, translate, Options};

const V: u64 = 256; // VALU source encoding base for VGPRs

type Insn<'a> = (&'a str, &'a str, &'a [(&'a str, u64)], Option<u32>);

#[derive(Clone, Copy)]
struct PermSpec<'a> {
    cross: bool,
    fi: bool,
    bound_ctrl: bool,
    exec_lo: u32,
    exec_hi: u32,
    src0: &'a [u32; 64],
    src1: u32,
    src2: u32,
    init_vdst: &'a [u32; 64],
}

fn prog(list: &[Insn<'_>]) -> Vec<u8> {
    let mut code = Vec::new();
    for &(name, enc, fields, lit) in list {
        code.extend(encode(name, enc, fields, lit).unwrap_or_else(|e| panic!("{name}: {e}")));
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

/// AMD RDNA 2 ISA reference for `V_PERMLANE16_B32` (`cross == false`) and
/// `V_PERMLANEX16_B32` (`cross == true`).
fn expected_permlane(lane: usize, spec: &PermSpec<'_>, opt: &Options) -> u32 {
    let (eff_lane, half_base, exec_mask, wave_mask) = if opt.wave64 {
        (
            lane,
            0usize,
            (spec.exec_lo as u64) | ((spec.exec_hi as u64) << 32),
            63usize,
        )
    } else {
        // In wave32@hw64 (`hw_wave == 64`), the upper 32 hardware lanes execute as a second
        // independent 32-lane wave sharing the same 32-bit EXEC (`exec_lo`).
        (lane & 31, lane & 32, spec.exec_lo as u64, 31usize)
    };
    if (exec_mask >> eff_lane) & 1 == 0 {
        return spec.init_vdst[lane];
    }
    let row_pos = eff_lane & 15;
    let sel_word = if row_pos < 8 { spec.src1 } else { spec.src2 };
    let target = ((sel_word >> (4 * (row_pos & 7))) & 0xF) as usize;
    let row_base = if spec.cross {
        (eff_lane & !15) ^ 16
    } else {
        eff_lane & !15
    };
    let src_eff = row_base | target;
    let src_active = (exec_mask >> (src_eff & wave_mask)) & 1 != 0;
    if spec.fi || src_active {
        spec.src0[half_base + src_eff]
    } else if spec.bound_ctrl {
        0
    } else {
        spec.init_vdst[lane]
    }
}

#[test]
fn permlane16_and_permlanex16_translate_and_compile_with_dxc() {
    let mut code = Vec::new();
    for (i, (name, op_sel)) in [false, true]
        .into_iter()
        .flat_map(|c| {
            (0u64..4).map(move |op| {
                (
                    if c {
                        "V_PERMLANEX16_B32"
                    } else {
                        "V_PERMLANE16_B32"
                    },
                    op,
                )
            })
        })
        .enumerate()
    {
        let fields = [
            ("VDST", 10 + i as u64),
            ("SRC0", V),
            ("SRC1", 0),
            ("SRC2", 1),
            ("OP_SEL", op_sel),
        ];
        code.extend(encode(name, "ENC_VOP3", &fields, None).unwrap());
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

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
        let expected_rla = if opt.wave64 && opt.split_wave64 {
            "xcl_x_rla_half("
        } else {
            "rla("
        };
        assert!(
            t.hlsl.contains(expected_rla),
            "{label}: expected {expected_rla}\n{}",
            t.hlsl
        );
        assert!(
            t.hlsl.contains("^ 16u"),
            "{label}: expected cross-row XOR for V_PERMLANEX16_B32\n{}",
            t.hlsl
        );
        assert!(
            t.hlsl.contains("or(true, lanebit_at("),
            "{label}: expected FI=1 check\n{}",
            t.hlsl
        );
        assert!(
            t.hlsl.contains("or(false, lanebit_at("),
            "{label}: expected FI=0 check\n{}",
            t.hlsl
        );
        if has_dxc {
            dxc::compile(&t.hlsl, "main", "cs_6_6", &[])
                .unwrap_or_else(|e| panic!("{label}: {e}\n{}", t.hlsl));
        }
    }
}

#[test]
fn gpu_permlane_row_permutations_and_inplace_aliasing() {
    let configs = wave_configs();
    if configs.is_empty() {
        return;
    }

    let (sel1_a, sel2_a) = (0x5A2F_90E3u32, 0x1B74_C8D6u32);
    let (sel1_x, sel2_x) = (0xD84E_12A7u32, 0x9F03_6B5Cu32);
    let lit_sel1 = 0x3141_5926u32;

    let code = prog(&[
        (
            "V_PERMLANE16_B32",
            "ENC_VOP3",
            &[("VDST", 10), ("SRC0", V), ("SRC1", 0), ("SRC2", 1)],
            None,
        ),
        (
            "V_PERMLANEX16_B32",
            "ENC_VOP3",
            &[("VDST", 11), ("SRC0", V), ("SRC1", 2), ("SRC2", 3)],
            None,
        ),
        // Inline constants: SRC1 = 0 (lane 0 for 0..7), SRC2 = -1 (lane 15 for 8..15)
        (
            "V_PERMLANE16_B32",
            "ENC_VOP3",
            &[("VDST", 12), ("SRC0", V), ("SRC1", 128), ("SRC2", 193)],
            None,
        ),
        (
            "V_PERMLANEX16_B32",
            "ENC_VOP3",
            &[("VDST", 13), ("SRC0", V), ("SRC1", 255), ("SRC2", 3)],
            Some(lit_sel1),
        ),
        // In-place aliasing: VDST == SRC0
        (
            "V_PERMLANE16_B32",
            "ENC_VOP3",
            &[("VDST", 20), ("SRC0", V + 20), ("SRC1", 0), ("SRC2", 1)],
            None,
        ),
        (
            "V_PERMLANEX16_B32",
            "ENC_VOP3",
            &[("VDST", 21), ("SRC0", V + 21), ("SRC1", 2), ("SRC2", 3)],
            None,
        ),
    ]);

    let mut st = KernelState::default();
    st.sgpr[0..4].copy_from_slice(&[sel1_a, sel2_a, sel1_x, sel2_x]);
    let mut src0 = [0u32; 64];
    for (l, slot) in src0.iter_mut().enumerate() {
        let val = 0x1000_0000 | ((l as u32) << 8) | (l as u32 ^ 0x5A);
        *slot = val;
        st.vgpr[l][0] = val;
        st.vgpr[l][20] = val;
        st.vgpr[l][21] = val;
    }

    let base_spec = PermSpec {
        cross: false,
        fi: false,
        bound_ctrl: false,
        exec_lo: u32::MAX,
        exec_hi: u32::MAX,
        src0: &src0,
        src1: sel1_a,
        src2: sel2_a,
        init_vdst: &src0,
    };

    for (label, opt) in configs {
        let dxil = compile_kernel(&code, &opt);
        let r = runner::run(&dxil, &st, &[]).unwrap();
        let active_lanes = if opt.wave64 || opt.hw_wave == 64 {
            64
        } else {
            32
        };
        for (l, v) in r.vgpr.iter().take(active_lanes).enumerate() {
            let w10 = expected_permlane(l, &base_spec, &opt);
            let w11 = expected_permlane(
                l,
                &PermSpec {
                    cross: true,
                    src1: sel1_x,
                    src2: sel2_x,
                    ..base_spec
                },
                &opt,
            );
            let w12 = expected_permlane(
                l,
                &PermSpec {
                    src1: 0,
                    src2: u32::MAX,
                    ..base_spec
                },
                &opt,
            );
            let w13 = expected_permlane(
                l,
                &PermSpec {
                    cross: true,
                    src1: lit_sel1,
                    src2: sel2_x,
                    ..base_spec
                },
                &opt,
            );
            assert_eq!(v[10], w10, "{label} lane {l} v_permlane16_b32");
            assert_eq!(v[11], w11, "{label} lane {l} v_permlanex16_b32");
            assert_eq!(
                v[12], w12,
                "{label} lane {l} v_permlane16_b32 inline consts"
            );
            assert_eq!(v[13], w13, "{label} lane {l} v_permlanex16_b32 literal");
            assert_eq!(v[20], w10, "{label} lane {l} v_permlane16_b32 in-place");
            assert_eq!(v[21], w11, "{label} lane {l} v_permlanex16_b32 in-place");
        }
    }
}

#[test]
fn gpu_permlane_fi_bound_ctrl_and_divergent_exec() {
    let configs = wave_configs();
    if configs.is_empty() {
        return;
    }

    // Asymmetric active masks so every 16-lane row has both active and inactive lanes, and
    // row-local / cross-row lookups hit both active and inactive source lanes.
    let (exec_lo, exec_hi) = (0x5A3C_96A5u32, 0x3C69_A55Au32);
    let (sel1, sel2) = (0xF1B3_8E40u32, 0x2D69_C075u32);

    // (dst_vgpr, cross, fi, bound_ctrl, init_idx)
    let cases: [(usize, bool, bool, bool, usize); 9] = [
        (10, false, false, false, 0),
        (11, false, false, true, 1),
        (12, false, true, false, 2),
        (13, false, true, true, 3),
        (14, true, false, false, 4),
        (15, true, false, true, 5),
        (16, true, true, false, 6),
        (17, true, true, true, 7),
        (18, true, false, true, 8), // in-place VDST == SRC0 with FI=0, BOUND_CTRL=1
    ];

    let mut code = Vec::new();
    for &(dst, cross, fi, bound_ctrl, _) in &cases {
        let name = if cross {
            "V_PERMLANEX16_B32"
        } else {
            "V_PERMLANE16_B32"
        };
        let src0 = if dst == 18 { V + 18 } else { V };
        let op_sel = (fi as u64) | ((bound_ctrl as u64) << 1);
        let fields = [
            ("VDST", dst as u64),
            ("SRC0", src0),
            ("SRC1", 0),
            ("SRC2", 1),
            ("OP_SEL", op_sel),
        ];
        code.extend(encode(name, "ENC_VOP3", &fields, None).unwrap());
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let mut st = KernelState::default();
    st.sgpr[0] = sel1;
    st.sgpr[1] = sel2;
    st.sgpr[126] = exec_lo;
    st.sgpr[127] = exec_hi;

    let mut src0 = [0u32; 64];
    let mut init_regs = [[0u32; 64]; 9];
    for l in 0..64 {
        let s = 0x2000_0000 | ((l as u32) << 12) | (l as u32 * 3 + 7);
        src0[l] = s;
        st.vgpr[l][0] = s;
        st.vgpr[l][18] = s;
        init_regs[8][l] = s;
        for (idx, dst) in (10..=17).enumerate() {
            let sentinel = 0xDEAD_0000 | ((dst as u32) << 8) | l as u32;
            init_regs[idx][l] = sentinel;
            st.vgpr[l][dst] = sentinel;
        }
    }

    for (label, opt) in configs {
        let dxil = compile_kernel(&code, &opt);
        let r = runner::run(&dxil, &st, &[]).unwrap();
        let active_lanes = if opt.wave64 || opt.hw_wave == 64 {
            64
        } else {
            32
        };
        for (l, v) in r.vgpr.iter().take(active_lanes).enumerate() {
            for &(dst, cross, fi, bound_ctrl, init_idx) in &cases {
                let spec = PermSpec {
                    cross,
                    fi,
                    bound_ctrl,
                    exec_lo,
                    exec_hi,
                    src0: &src0,
                    src1: sel1,
                    src2: sel2,
                    init_vdst: &init_regs[init_idx],
                };
                let want = expected_permlane(l, &spec, &opt);
                assert_eq!(
                    v[dst], want,
                    "{label} lane {l} v{dst} (cross={cross}, fi={fi}, bound_ctrl={bound_ctrl})"
                );
            }
        }
    }
}
