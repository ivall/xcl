//! GPU verification tests for RDNA2 DPP16, DPP8, and SDWA VALU/VOPC encoding modifiers:
//! - DPP16 (`VOP1_VOP_DPP16`, `VOP2_VOP_DPP16`): `quad_perm`, `row_shl`, `row_shr`, `row_ror`,
//!   `row_mirror`, `row_half_mirror`, `row_share`, `row_xmask`, `ROW_MASK`, `BANK_MASK`,
//!   `BOUND_CTRL` (0 vs 1), `FI` (fetch-inactive under divergent `EXEC`), float `SRC*_ABS`/`NEG`,
//!   and 16-bit `VALU` (`V_ADD_F16`) under DPP16.
//! - DPP8 (`VOP1_VOP_DPP8`, `VOP2_VOP_DPP8`): asymmetric `LANE_SEL_0..7` permutations,
//!   two-operand VALU, and `FI=0` (`0xE9`) vs `FI=1` (`0xEA`) under divergent `EXEC`.
//! - SDWA (`VOP1_VOP_SDWA`, `VOP2_VOP_SDWA`, `VOPC_VOP_SDWA_SDST_ENC`): `SRC0_SEL`/`SRC1_SEL`
//!   (`BYTE_0..3`, `WORD_0..1`, `DWORD`) with `SRC*_SEXT`, scalar `S0`/`S1` SGPR and inline
//!   constants, `DST_SEL` across all byte/word/dword positions with `DST_UNUSED` (`UNUSED_PAD`,
//!   `UNUSED_SEXT`, `UNUSED_PRESERVE`), float `ABS`/`NEG`/`OMOD`/`CLAMP`, 16-bit `VALU` under SDWA,
//!   `V_CMP_*` with `SD=0` (`VCC`), `SD=1` (`SDST` / `null`), and `V_CMPX_*` updating `EXEC`.
//!
//! Verified across all four wave execution modes (`wave64`, `split-wave64`, `wave32`, `wave32@hw64`).

use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner::{self, KernelState};
use xcl::gpu::recomp::{dxc, translate, Options};

type Insn = (
    &'static str,
    &'static str,
    Vec<(&'static str, u64)>,
    Option<u32>,
);

const UNUSED_PAD: u64 = 0;
const UNUSED_SEXT: u64 = 1;
const UNUSED_PRESERVE: u64 = 2;

const SEL_BYTE_0: u64 = 0;
const SEL_BYTE_1: u64 = 1;
const SEL_BYTE_2: u64 = 2;
const SEL_BYTE_3: u64 = 3;
const SEL_WORD_0: u64 = 4;
const SEL_WORD_1: u64 = 5;
const SEL_DWORD: u64 = 6;

const EXEC_LO: u64 = 126;
const NULL_REG: u64 = 125;

fn insn(name: &'static str, enc: &'static str, fields: &[(&'static str, u64)]) -> Insn {
    (name, enc, fields.to_vec(), None)
}

fn s_mov(wave64: bool, sdst: u64, ssrc0: u64) -> Insn {
    let op = if wave64 { "S_MOV_B64" } else { "S_MOV_B32" };
    insn(op, "ENC_SOP1", &[("SDST", sdst), ("SSRC0", ssrc0)])
}

fn vop1_dpp16(
    name: &'static str,
    vdst: u64,
    vsrc0: u64,
    ctrl: u64,
    extra: &[(&'static str, u64)],
) -> Insn {
    let mut fields = vec![
        ("VDST", vdst),
        ("SRC0", 0xFA),
        ("VSRC0", vsrc0),
        ("DPP_CTRL", ctrl),
        ("ROW_MASK", 0xF),
        ("BANK_MASK", 0xF),
    ];
    fields.extend_from_slice(extra);
    (name, "VOP1_VOP_DPP16", fields, None)
}

fn vop2_dpp16(
    name: &'static str,
    vdst: u64,
    vsrc0: u64,
    vsrc1: u64,
    ctrl: u64,
    extra: &[(&'static str, u64)],
) -> Insn {
    let mut fields = vec![
        ("VDST", vdst),
        ("SRC0", 0xFA),
        ("VSRC0", vsrc0),
        ("VSRC1", vsrc1),
        ("DPP_CTRL", ctrl),
        ("ROW_MASK", 0xF),
        ("BANK_MASK", 0xF),
    ];
    fields.extend_from_slice(extra);
    (name, "VOP2_VOP_DPP16", fields, None)
}

fn dpp8_fields(sels: [u64; 8]) -> Vec<(&'static str, u64)> {
    vec![
        ("LANE_SEL_0", sels[0]),
        ("LANE_SEL_1", sels[1]),
        ("LANE_SEL_2", sels[2]),
        ("LANE_SEL_3", sels[3]),
        ("LANE_SEL_4", sels[4]),
        ("LANE_SEL_5", sels[5]),
        ("LANE_SEL_6", sels[6]),
        ("LANE_SEL_7", sels[7]),
    ]
}

fn vop1_dpp8(name: &'static str, vdst: u64, vsrc0: u64, fi: bool, sels: [u64; 8]) -> Insn {
    let mut fields = vec![
        ("VDST", vdst),
        ("SRC0", if fi { 0xEA } else { 0xE9 }),
        ("VSRC0", vsrc0),
    ];
    fields.extend(dpp8_fields(sels));
    (name, "VOP1_VOP_DPP8", fields, None)
}

fn vop2_dpp8(
    name: &'static str,
    vdst: u64,
    vsrc0: u64,
    vsrc1: u64,
    fi: bool,
    sels: [u64; 8],
) -> Insn {
    let mut fields = vec![
        ("VDST", vdst),
        ("SRC0", if fi { 0xEA } else { 0xE9 }),
        ("VSRC0", vsrc0),
        ("VSRC1", vsrc1),
    ];
    fields.extend(dpp8_fields(sels));
    (name, "VOP2_VOP_DPP8", fields, None)
}

fn vop1_sdwa(name: &'static str, vdst: u64, vsrc0: u64, extra: &[(&'static str, u64)]) -> Insn {
    let mut fields = vec![
        ("VDST", vdst),
        ("SRC0", 0xF9),
        ("VSRC0", vsrc0),
        ("DST_SEL", SEL_DWORD),
        ("DST_UNUSED", UNUSED_PAD),
        ("SRC0_SEL", SEL_DWORD),
    ];
    fields.extend_from_slice(extra);
    (name, "VOP1_VOP_SDWA", fields, None)
}

fn vop2_sdwa(
    name: &'static str,
    vdst: u64,
    vsrc0: u64,
    vsrc1: u64,
    extra: &[(&'static str, u64)],
) -> Insn {
    let mut fields = vec![
        ("VDST", vdst),
        ("SRC0", 0xF9),
        ("VSRC0", vsrc0),
        ("VSRC1", vsrc1),
        ("DST_SEL", SEL_DWORD),
        ("DST_UNUSED", UNUSED_PAD),
        ("SRC0_SEL", SEL_DWORD),
        ("SRC1_SEL", SEL_DWORD),
    ];
    fields.extend_from_slice(extra);
    (name, "VOP2_VOP_SDWA", fields, None)
}

fn vopc_sdwa(name: &'static str, vsrc0: u64, vsrc1: u64, extra: &[(&'static str, u64)]) -> Insn {
    let mut fields = vec![
        ("SRC0", 0xF9),
        ("VSRC0", vsrc0),
        ("VSRC1", vsrc1),
        ("SRC0_SEL", SEL_DWORD),
        ("SRC1_SEL", SEL_DWORD),
    ];
    fields.extend_from_slice(extra);
    (name, "VOPC_VOP_SDWA_SDST_ENC", fields, None)
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

fn all_modes() -> [(&'static str, Options); 4] {
    [
        ("wave64", opt_mode(true, false, 0)),
        ("split-wave64", opt_mode(true, true, 0)),
        ("wave32", opt_mode(false, false, 0)),
        ("wave32@hw64", opt_mode(false, false, 64)),
    ]
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
                "GPU/DXC unavailable - skipping ({:?} / {:?} / {:?})",
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
    if opt.wave64 {
        64
    } else {
        32
    }
}

fn run_in_mode(code: &[u8], st: &KernelState, opt: &Options) -> (runner::KernelResult, usize) {
    let dxil = compile_kernel(code, opt);
    let r = runner::run(&dxil, st, &[]).unwrap();
    (r, active_lanes_for(opt))
}

fn quad_perm_ctrl(p: [u64; 4]) -> u64 {
    p[0] | (p[1] << 2) | (p[2] << 4) | (p[3] << 6)
}

fn mask64_for_lanes(opt: &Options, mask64: u64) -> (u32, u32) {
    if opt.wave64 {
        (mask64 as u32, (mask64 >> 32) as u32)
    } else {
        (mask64 as u32, 0)
    }
}

/// Verifies HLSL emission (including SDWA helpers and split-wave64 half-wave reads) and DXC
/// compilation across all four wave configurations without requiring a GPU device.
#[test]
fn dpp_sdwa_translate_and_compile_with_dxc() {
    if dxc::location().is_err() {
        runner::skip_test("no DXC");
        return;
    }

    let code = prog(&[
        vop1_dpp16("V_MOV_B32", 10, 0, 0x101, &[("BOUND_CTRL", 1), ("FI", 1)]),
        vop2_dpp16(
            "V_ADD_F32",
            11,
            0,
            1,
            0x157,
            &[
                ("SRC0_ABS", 1),
                ("SRC1_NEG", 1),
                ("ROW_MASK", 0xA),
                ("BANK_MASK", 0x5),
            ],
        ),
        vop2_dpp8("V_ADD_NC_U32", 12, 0, 1, true, [7, 6, 5, 4, 3, 2, 1, 0]),
        vop1_sdwa(
            "V_CVT_F32_I32",
            13,
            0,
            &[
                ("SRC0_SEL", SEL_BYTE_2),
                ("SRC0_SEXT", 1),
                ("OMOD", 1),
                ("CLAMP", 1),
            ],
        ),
        vop2_sdwa(
            "V_ADD_NC_U32",
            14,
            4,
            1,
            &[
                ("S0", 1),
                ("SRC0_SEL", SEL_WORD_1),
                ("SRC1_SEL", SEL_BYTE_1),
                ("DST_SEL", SEL_BYTE_1),
                ("DST_UNUSED", UNUSED_SEXT),
            ],
        ),
        vopc_sdwa(
            "V_CMP_LT_I32",
            0,
            1,
            &[
                ("SRC0_SEL", SEL_BYTE_0),
                ("SRC0_SEXT", 1),
                ("SRC1_SEL", SEL_BYTE_1),
                ("SRC1_SEXT", 1),
                ("SD", 1),
                ("SDST", 10),
            ],
        ),
        vopc_sdwa(
            "V_CMPX_EQ_U32",
            0,
            1,
            &[("SRC0_SEL", SEL_BYTE_0), ("SRC1_SEL", SEL_BYTE_0)],
        ),
    ]);

    for (mode, opt) in all_modes() {
        let t = translate(&code, &opt).unwrap();
        assert!(t.complete(), "{mode}: {:?}\n{}", t.unsupported_why, t.hlsl);
        assert!(
            t.hlsl.contains("sdwa_sel("),
            "{mode}: missing sdwa_sel\n{}",
            t.hlsl
        );
        assert!(
            t.hlsl.contains("sdwa_dst("),
            "{mode}: missing sdwa_dst\n{}",
            t.hlsl
        );
        if opt.split_wave64 {
            assert!(
                t.hlsl.contains("xcl_x_rla_half("),
                "{mode}: missing xcl_x_rla_half\n{}",
                t.hlsl
            );
        } else {
            assert!(t.hlsl.contains("rla("), "{mode}: missing rla\n{}", t.hlsl);
        }
        dxc::compile(&t.hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("{mode}: DXC failed: {e}\n{}", t.hlsl));
    }
}

/// Verifies DPP16 row/quad/share/xmask permutations, row/bank masks, `BOUND_CTRL`, `FI` under
/// divergent `EXEC`, float input modifiers, and 16-bit `VALU` preservation across all wave modes.
#[test]
fn gpu_dpp16_controls_masks_and_fetch_inactive() {
    let cfgs = wave_configs();
    if cfgs.is_empty() {
        return;
    }

    let qperm = [3u64, 1, 0, 2];
    for (mode, opt) in cfgs {
        let code = prog(&[
            // v10: quad_perm [3, 1, 0, 2]
            vop1_dpp16("V_MOV_B32", 10, 0, quad_perm_ctrl(qperm), &[]),
            // v11: quad_perm [0, 0, 0, 0], BANK_MASK = 0b0101 (banks 0 and 2 only)
            vop1_dpp16(
                "V_MOV_B32",
                11,
                0,
                quad_perm_ctrl([0, 0, 0, 0]),
                &[("BANK_MASK", 0x5)],
            ),
            // v12: row_shl:1 with BOUND_CTRL=0 (lane%16 == 15 preserves sentinel)
            vop1_dpp16("V_MOV_B32", 12, 0, 0x101, &[("BOUND_CTRL", 0)]),
            // v13: row_shl:5 with BOUND_CTRL=1 (lane%16 >= 11 reads 0)
            vop1_dpp16("V_MOV_B32", 13, 0, 0x105, &[("BOUND_CTRL", 1)]),
            // v14: row_shr:3 with BOUND_CTRL=0 (lane%16 < 3 preserves sentinel)
            vop1_dpp16("V_MOV_B32", 14, 0, 0x113, &[("BOUND_CTRL", 0)]),
            // v15: row_shr:3 with BOUND_CTRL=1 (lane%16 < 3 reads 0)
            vop1_dpp16("V_MOV_B32", 15, 0, 0x113, &[("BOUND_CTRL", 1)]),
            // v16: row_ror:4 (rotates right 4 within each 16-lane row)
            vop1_dpp16("V_MOV_B32", 16, 0, 0x124, &[]),
            // v17: row_half_mirror (mirrors within each 8-lane half-row)
            vop1_dpp16("V_MOV_B32", 17, 0, 0x141, &[]),
            // v18: row_share:11 (broadcasts lane 11 of each 16-lane row)
            vop1_dpp16("V_MOV_B32", 18, 0, 0x15B, &[]),
            // v19: row_xmask:6 with ROW_MASK = 0b0101 (rows 0 and 2) and BANK_MASK = 0b1011 (banks 0, 1, 3)
            vop1_dpp16(
                "V_MOV_B32",
                19,
                0,
                0x166,
                &[("ROW_MASK", 0x5), ("BANK_MASK", 0xB)],
            ),
            // v23: row_shl:15 with BOUND_CTRL=0 (only col 0 writes v0[row*16 + 15]; cols 1..15 preserve sentinel)
            vop1_dpp16("V_MOV_B32", 23, 0, 0x10F, &[("BOUND_CTRL", 0)]),
            // v24: row_shr:15 with BOUND_CTRL=1 (only col 15 reads v0[row*16 + 0]; cols 0..14 read 0)
            vop1_dpp16("V_MOV_B32", 24, 0, 0x11F, &[("BOUND_CTRL", 1)]),
            // v20: VOP2 row_ror:1 (src0 rotated + v1 of destination lane)
            vop2_dpp16("V_ADD_NC_U32", 20, 0, 1, 0x121, &[]),
            // v21: VOP2 float modifiers + BOUND_CTRL=1: -abs(src0) + abs(v3)
            vop2_dpp16(
                "V_ADD_F32",
                21,
                2,
                3,
                0x101,
                &[
                    ("BOUND_CTRL", 1),
                    ("SRC0_ABS", 1),
                    ("SRC0_NEG", 1),
                    ("SRC1_ABS", 1),
                ],
            ),
            // v22: 16-bit VALU under DPP16 row_ror:1 preserves upper 16 bits of VDST
            vop2_dpp16("V_ADD_F16", 22, 4, 5, 0x121, &[]),
            // Divergent EXEC: keep only even lanes active (odd source lanes become inactive)
            s_mov(opt.wave64, 2, EXEC_LO),
            s_mov(opt.wave64, EXEC_LO, 0),
            // v30: row_shl:1, FI=0, BOUND_CTRL=0 -> active even lane reads inactive odd -> preserves v30
            vop1_dpp16("V_MOV_B32", 30, 0, 0x101, &[("FI", 0), ("BOUND_CTRL", 0)]),
            // v31: row_shl:1, FI=0, BOUND_CTRL=1 -> active even lane gets 0
            vop1_dpp16("V_MOV_B32", 31, 0, 0x101, &[("FI", 0), ("BOUND_CTRL", 1)]),
            // v32: VOP2 row_shl:1, FI=0, BOUND_CTRL=1 -> active even lane computes 0 + v1[lane]
            vop2_dpp16(
                "V_ADD_NC_U32",
                32,
                0,
                1,
                0x101,
                &[("FI", 0), ("BOUND_CTRL", 1)],
            ),
            // v33: row_shl:1, FI=1, BOUND_CTRL=0 -> active even lane fetches inactive v0[lane + 1]
            vop1_dpp16("V_MOV_B32", 33, 0, 0x101, &[("FI", 1), ("BOUND_CTRL", 0)]),
            // v34: VOP2 row_shl:1, FI=0, BOUND_CTRL=0 -> active even lane reads inactive odd -> preserves v34
            vop2_dpp16(
                "V_ADD_NC_U32",
                34,
                0,
                1,
                0x101,
                &[("FI", 0), ("BOUND_CTRL", 0)],
            ),
            s_mov(opt.wave64, EXEC_LO, 2),
        ]);

        let mut st = KernelState::default();
        st.sgpr[0] = 0x5555_5555;
        st.sgpr[1] = 0x5555_5555;
        for (l, r) in st.vgpr.iter_mut().enumerate() {
            let l32 = l as u32;
            r[0] = 100 + l32;
            r[1] = 1000 + l32;
            r[2] = if l % 2 == 0 { -2.5f32 } else { 3.5f32 }.to_bits();
            r[3] = (-1.25f32).to_bits();
            r[4] = if l % 2 == 0 { 0xDEAD_3C00 } else { 0xDEAD_4000 }; // lo = 1.0h / 2.0h
            r[5] = 0xCAFE_3800; // lo = 0.5h
            for dst in [11usize, 12, 14, 19, 23, 30, 31, 32, 33, 34] {
                r[dst] = 0xDEAD_0000 | l32;
            }
            r[22] = 0xBEEF_0000;
        }

        let (r, lanes) = run_in_mode(&code, &st, &opt);
        for l in 0..lanes {
            let v = &r.vgpr[l];
            let sentinel = 0xDEAD_0000 | (l as u32);
            let row = l / 16;
            let col = l % 16;
            let bank = (l / 4) % 4;
            let base_row = l & !15;

            assert_eq!(
                v[10],
                100 + ((l & !3) | qperm[l & 3] as usize) as u32,
                "{mode} lane {l} quad_perm"
            );
            assert_eq!(
                v[11],
                if bank == 0 || bank == 2 {
                    100 + (l & !3) as u32
                } else {
                    sentinel
                },
                "{mode} lane {l} bank_mask"
            );
            assert_eq!(
                v[12],
                if col < 15 {
                    100 + (l + 1) as u32
                } else {
                    sentinel
                },
                "{mode} lane {l} row_shl:1 bound_ctrl:0"
            );
            assert_eq!(
                v[13],
                if col < 11 { 100 + (l + 5) as u32 } else { 0 },
                "{mode} lane {l} row_shl:5 bound_ctrl:1"
            );
            assert_eq!(
                v[14],
                if col >= 3 {
                    100 + (l - 3) as u32
                } else {
                    sentinel
                },
                "{mode} lane {l} row_shr:3 bound_ctrl:0"
            );
            assert_eq!(
                v[15],
                if col >= 3 { 100 + (l - 3) as u32 } else { 0 },
                "{mode} lane {l} row_shr:3 bound_ctrl:1"
            );
            let ror4_src = base_row | ((col + 16 - 4) & 15);
            assert_eq!(v[16], 100 + ror4_src as u32, "{mode} lane {l} row_ror:4");
            let hmir_src = (l & !7) | (7 - (l & 7));
            assert_eq!(
                v[17],
                100 + hmir_src as u32,
                "{mode} lane {l} row_half_mirror"
            );
            assert_eq!(
                v[18],
                100 + (base_row | 11) as u32,
                "{mode} lane {l} row_share:11"
            );
            assert_eq!(
                v[19],
                if row % 2 == 0 && bank != 2 {
                    100 + (l ^ 6) as u32
                } else {
                    sentinel
                },
                "{mode} lane {l} row_xmask:6 row_mask+bank_mask"
            );
            assert_eq!(
                v[23],
                if col == 0 {
                    100 + (base_row | 15) as u32
                } else {
                    sentinel
                },
                "{mode} lane {l} row_shl:15 bound_ctrl:0"
            );
            assert_eq!(
                v[24],
                if col == 15 { 100 + base_row as u32 } else { 0 },
                "{mode} lane {l} row_shr:15 bound_ctrl:1"
            );

            let ror1_src = base_row | ((col + 15) & 15);
            assert_eq!(
                v[20],
                (100 + ror1_src as u32) + (1000 + l as u32),
                "{mode} lane {l} vop2 row_ror:1"
            );
            let s0_abs = if col < 15 {
                if (l + 1) % 2 == 0 {
                    2.5f32
                } else {
                    3.5f32
                }
            } else {
                0.0f32
            };
            assert_eq!(
                f32::from_bits(v[21]),
                -s0_abs + 1.25f32,
                "{mode} lane {l} v_add_f32 dpp16 modifiers"
            );
            let want_v22_lo = if ror1_src % 2 == 0 {
                0x3E00u32
            } else {
                0x4100u32
            }; // 1.5h / 2.5h
            assert_eq!(
                v[22],
                0xBEEF_0000 | want_v22_lo,
                "{mode} lane {l} v_add_f16 dpp16 preserves hi16"
            );

            // Divergent EXEC: even lanes active, reading odd inactive neighbors
            assert_eq!(
                v[30], sentinel,
                "{mode} lane {l} fi=0 bound_ctrl=0 preserves"
            );
            assert_eq!(
                v[31],
                if l % 2 == 0 { 0 } else { sentinel },
                "{mode} lane {l} fi=0 bound_ctrl=1 zeroes inactive src"
            );
            assert_eq!(
                v[32],
                if l % 2 == 0 {
                    1000 + l as u32
                } else {
                    sentinel
                },
                "{mode} lane {l} vop2 fi=0 bound_ctrl=1"
            );
            assert_eq!(
                v[33],
                if l % 2 == 0 {
                    100 + (l + 1) as u32
                } else {
                    sentinel
                },
                "{mode} lane {l} fi=1 fetches inactive"
            );
            assert_eq!(
                v[34], sentinel,
                "{mode} lane {l} vop2 fi=0 bound_ctrl=0 preserves"
            );
        }
    }
}

/// Verifies DPP8 `VOP1`/`VOP2` 8-lane permutations and `FI=0` (`0xE9`) vs `FI=1` (`0xEA`)
/// semantics under divergent `EXEC` across all wave execution modes.
#[test]
fn gpu_dpp8_permutations_and_fetch_inactive() {
    let cfgs = wave_configs();
    if cfgs.is_empty() {
        return;
    }

    let perm = [3u64, 3, 0, 7, 1, 6, 4, 2];
    let ror1 = [7u64, 0, 1, 2, 3, 4, 5, 6];
    let mixed = [1u64, 0, 2, 3, 5, 4, 6, 7];

    for (mode, opt) in cfgs {
        let code = prog(&[
            vop1_dpp8("V_MOV_B32", 10, 0, false, perm),
            vop2_dpp8("V_ADD_NC_U32", 11, 0, 1, false, ror1),
            // Divergent EXEC: even lanes active, odd lanes inactive
            s_mov(opt.wave64, 2, EXEC_LO),
            s_mov(opt.wave64, EXEC_LO, 0),
            // v12: FI=0 (0xE9) -> lanes 0,4 read inactive 1,5 (get 0); lanes 2,6 read active 2,6
            vop2_dpp8("V_XOR_B32", 12, 0, 1, false, mixed),
            // v13: FI=1 (0xEA) -> lanes 0,4 fetch inactive 1,5; lanes 2,6 read active 2,6
            vop2_dpp8("V_ADD_NC_U32", 13, 0, 1, true, mixed),
            s_mov(opt.wave64, EXEC_LO, 2),
        ]);

        let mut st = KernelState::default();
        st.sgpr[0] = 0x5555_5555;
        st.sgpr[1] = 0x5555_5555;
        for (l, r) in st.vgpr.iter_mut().enumerate() {
            let l32 = l as u32;
            r[0] = 0x1000 + l32;
            r[1] = 0x2000 + l32;
            r[12] = 0xDEAD_0000 | l32;
            r[13] = 0xDEAD_0000 | l32;
        }

        let (r, lanes) = run_in_mode(&code, &st, &opt);
        for l in 0..lanes {
            let v = &r.vgpr[l];
            let base8 = l & !7;
            let sentinel = 0xDEAD_0000 | (l as u32);
            let p_src = base8 | perm[l & 7] as usize;
            let r_src = base8 | ror1[l & 7] as usize;

            assert_eq!(v[10], 0x1000 + p_src as u32, "{mode} lane {l} dpp8 perm");
            assert_eq!(
                v[11],
                (0x1000 + r_src as u32) + (0x2000 + l as u32),
                "{mode} lane {l} vop2 dpp8 ror1"
            );
            let m_src = base8 | mixed[l & 7] as usize;
            let src_active = m_src.is_multiple_of(2);
            assert_eq!(
                v[12],
                if l % 2 == 0 {
                    let s0 = if src_active { 0x1000 + m_src as u32 } else { 0 };
                    s0 ^ (0x2000 + l as u32)
                } else {
                    sentinel
                },
                "{mode} lane {l} dpp8 fi=0 mixed active/inactive source"
            );
            assert_eq!(
                v[13],
                if l % 2 == 0 {
                    (0x1000 + m_src as u32) + (0x2000 + l as u32)
                } else {
                    sentinel
                },
                "{mode} lane {l} dpp8 fi=1 fetches inactive source"
            );
        }
    }
}

/// Verifies SDWA source/destination selectors (`BYTE_0..3`, `WORD_0..1`, `DWORD`), `SRC*_SEXT`,
/// `S0`/`S1` scalar sources, `DST_UNUSED` (`PAD`, `SEXT`, `PRESERVE`), float `ABS`/`NEG`/`OMOD`/
/// `CLAMP`, 16-bit VALU under SDWA, and `VOPC_VOP_SDWA_SDST_ENC` (`SD=0`, `SD=1`, `SDST=null`,
/// and `V_CMPX_*`) across all wave execution modes.
#[test]
fn gpu_sdwa_selectors_unused_modifiers_and_vopc() {
    let cfgs = wave_configs();
    if cfgs.is_empty() {
        return;
    }

    for (mode, opt) in cfgs {
        let code = prog(&[
            // --- Source selectors + sign extension + scalar sources + DST_SEL x DST_UNUSED ---
            // v10: byte0 -> dst_sel:BYTE_2, dst_unused:PAD (zeroes outside [23:16])
            vop1_sdwa(
                "V_MOV_B32",
                10,
                0,
                &[
                    ("SRC0_SEL", SEL_BYTE_0),
                    ("DST_SEL", SEL_BYTE_2),
                    ("DST_UNUSED", UNUSED_PAD),
                ],
            ),
            // v11: byte1(sext) -> dst_sel:BYTE_1, dst_unused:SEXT (sign-extends bit 15..31, bits 7:0 = 0)
            vop1_sdwa(
                "V_MOV_B32",
                11,
                0,
                &[
                    ("SRC0_SEL", SEL_BYTE_1),
                    ("DST_SEL", SEL_BYTE_1),
                    ("DST_UNUSED", UNUSED_SEXT),
                ],
            ),
            // v12: byte3 -> dst_sel:BYTE_3, dst_unused:PRESERVE (keeps low 24 bits of v12)
            vop1_sdwa(
                "V_MOV_B32",
                12,
                0,
                &[
                    ("SRC0_SEL", SEL_BYTE_3),
                    ("DST_SEL", SEL_BYTE_3),
                    ("DST_UNUSED", UNUSED_PRESERVE),
                ],
            ),
            // v13: V_ADD_NC_U32 with S0=1 (s4 WORD_1 sext) + S1=1 (inline -1) -> WORD_0 UNUSED_SEXT
            vop2_sdwa(
                "V_ADD_NC_U32",
                13,
                4,
                193,
                &[
                    ("S0", 1),
                    ("S1", 1),
                    ("SRC0_SEL", SEL_WORD_1),
                    ("SRC0_SEXT", 1),
                    ("SRC1_SEL", SEL_DWORD),
                    ("DST_SEL", SEL_WORD_0),
                    ("DST_UNUSED", UNUSED_SEXT),
                ],
            ),
            // v14: V_CVT_F32_I32 from sext(v0.byte2) -> DWORD
            vop1_sdwa(
                "V_CVT_F32_I32",
                14,
                0,
                &[("SRC0_SEL", SEL_BYTE_2), ("SRC0_SEXT", 1)],
            ),
            // v15: V_ADD_F32 with SRC0_ABS, SRC0_NEG, SRC1_ABS, OMOD=1 (*2.0), CLAMP=0
            vop2_sdwa(
                "V_ADD_F32",
                15,
                1,
                2,
                &[
                    ("SRC0_ABS", 1),
                    ("SRC0_NEG", 1),
                    ("SRC1_ABS", 1),
                    ("OMOD", 1),
                ],
            ),
            // v16: V_MUL_F32 with OMOD=2 (*4.0), CLAMP=1 (saturate after OMOD)
            vop2_sdwa("V_MUL_F32", 16, 1, 2, &[("OMOD", 2), ("CLAMP", 1)]),
            // v17: V_SUB_F32 with OMOD=3 (*0.5)
            vop2_sdwa("V_SUB_F32", 17, 2, 1, &[("OMOD", 3)]),
            // v18: V_CVT_F16_F32 with SRC0_NEG, OMOD=1 (*2.0), CLAMP=1 -> WORD_1 PRESERVE
            vop1_sdwa(
                "V_CVT_F16_F32",
                18,
                1,
                &[
                    ("SRC0_NEG", 1),
                    ("OMOD", 1),
                    ("CLAMP", 1),
                    ("DST_SEL", SEL_WORD_1),
                    ("DST_UNUSED", UNUSED_PRESERVE),
                ],
            ),
            // v19: 16-bit VALU (V_CVT_U16_F16) with SRC0_SEL=WORD_0, SRC0_ABS=1 -> WORD_1 PRESERVE
            vop1_sdwa(
                "V_CVT_U16_F16",
                19,
                3,
                &[
                    ("SRC0_SEL", SEL_WORD_0),
                    ("SRC0_ABS", 1),
                    ("DST_SEL", SEL_WORD_1),
                    ("DST_UNUSED", UNUSED_PRESERVE),
                ],
            ),
            // v20: 16-bit float (V_ADD_F16) with SRC0_ABS, SRC1_NEG, OMOD=3 (*0.5), CLAMP=1 -> WORD_0 PAD
            vop2_sdwa(
                "V_ADD_F16",
                20,
                3,
                3,
                &[
                    ("SRC0_SEL", SEL_WORD_0),
                    ("SRC1_SEL", SEL_WORD_1),
                    ("SRC0_ABS", 1),
                    ("SRC1_NEG", 1),
                    ("OMOD", 3),
                    ("CLAMP", 1),
                    ("DST_SEL", SEL_WORD_0),
                    ("DST_UNUSED", UNUSED_PAD),
                ],
            ),
            // v21: byte2(sext) + byte1(sext) -> dst_sel:BYTE_2, dst_unused:SEXT
            // (verifies SRC1_SEXT and zeroing of lower [15:0] alongside sign-extension of [31:24])
            vop2_sdwa(
                "V_ADD_NC_U32",
                21,
                0,
                0,
                &[
                    ("SRC0_SEL", SEL_BYTE_2),
                    ("SRC0_SEXT", 1),
                    ("SRC1_SEL", SEL_BYTE_1),
                    ("SRC1_SEXT", 1),
                    ("DST_SEL", SEL_BYTE_2),
                    ("DST_UNUSED", UNUSED_SEXT),
                ],
            ),
            // --- VOPC_VOP_SDWA_SDST_ENC ---
            // SD=0 writes VCC: compare sext(v0.byte1) < sext(s4.byte0) (-16)
            vopc_sdwa(
                "V_CMP_LT_I32",
                0,
                4,
                &[
                    ("S1", 1),
                    ("SRC0_SEL", SEL_BYTE_1),
                    ("SRC0_SEXT", 1),
                    ("SRC1_SEL", SEL_BYTE_0),
                    ("SRC1_SEXT", 1),
                    ("SD", 0),
                ],
            ),
            // Save VCC into s[6:7] (or s6 in wave32) before subsequent SD=1 / SDST=null tests
            s_mov(opt.wave64, 6, 106),
            // SD=1, SDST=10 writes s[10:11] (and leaves VCC untouched): abs(v1) > -v2
            vopc_sdwa(
                "V_CMP_GT_F32",
                1,
                2,
                &[("SRC0_ABS", 1), ("SRC1_NEG", 1), ("SD", 1), ("SDST", 10)],
            ),
            // SD=1, SDST=125 (null): must leave both VCC and s[10:11] untouched
            vopc_sdwa(
                "V_CMP_EQ_U32",
                0,
                0,
                &[
                    ("SRC0_SEL", SEL_BYTE_0),
                    ("SRC1_SEL", SEL_BYTE_0),
                    ("SD", 1),
                    ("SDST", NULL_REG),
                ],
            ),
            // Read s11 into s16 so wave32 includes s11 in KernelResult and proves SDST=10 left s11 intact
            insn("S_MOV_B32", "ENC_SOP1", &[("SDST", 16), ("SSRC0", 11)]),
            // Count active bits in s[10:11] to exercise cross-half SDWA SDST consumption in split-wave64
            insn(
                if opt.wave64 {
                    "S_BCNT1_I32_B64"
                } else {
                    "S_BCNT1_I32_B32"
                },
                "ENC_SOP1",
                &[("SDST", 17), ("SSRC0", 10)],
            ),
            // Save EXEC, run V_CMPX_EQ_U32_sdwa comparing (v0.byte0 & 0xFF) == 0, mask v29, restore EXEC
            s_mov(opt.wave64, 12, EXEC_LO),
            vopc_sdwa(
                "V_CMPX_EQ_U32",
                0,
                128,
                &[("S1", 1), ("SRC0_SEL", SEL_BYTE_0), ("SRC1_SEL", SEL_DWORD)],
            ),
            insn("V_MOV_B32", "ENC_VOP1", &[("VDST", 29), ("SRC0", 256)]),
            s_mov(opt.wave64, 14, EXEC_LO),
            s_mov(opt.wave64, EXEC_LO, 12),
        ]);

        let mut st = KernelState::default();
        st.sgpr[4] = 0xFFFE_00F0; // WORD_1 = -2 (0xFFFE), BYTE_0 = 0xF0 (-16 as i8)
        st.sgpr[10] = 0xDEAD_BEEF;
        st.sgpr[11] = 0xDEAD_BEEF;
        for (l, r) in st.vgpr.iter_mut().enumerate() {
            // byte0 = l % 4 (0 when l % 4 == 0, so V_CMPX_EQ_U32_sdwa activates every 4th lane)
            // byte1 alternates signed negative (0x80..0x8F) and positive (0x10..0x1F)
            // byte2 alternates signed negative (0xFE) and positive (0x05)
            // byte3 = 0x40 + l
            let b0 = (l as u32) & 3;
            let b1 = if l % 2 == 0 {
                0x80 + (l as u32 & 0xF)
            } else {
                0x10 + (l as u32 & 0xF)
            };
            let b2 = if l % 3 == 0 { 0xFE } else { 0x05 };
            let b3 = 0x40 + (l as u32);
            r[0] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            r[1] = if l % 2 == 0 { -0.5f32 } else { 0.75f32 }.to_bits();
            r[2] = if l % 2 == 0 { -1.25f32 } else { -0.25f32 }.to_bits();
            // v3: WORD_0 = -3.5h (0xC300), WORD_1 = -2.0h (0xC000) -> abs(-3.5) - (-2.0) = 5.5 * 0.5 = 2.75 -> clamp = 1.0h (0x3C00)
            // odd lanes: WORD_0 = -1.5h (0xBE00), WORD_1 = 0.25h (0x3400) -> abs(-1.5) - 0.25 = 1.25 * 0.5 = 0.625h (0x3900)
            r[3] = if l % 2 == 0 { 0xC000_C300 } else { 0x3400_BE00 };
            r[12] = 0x1122_3344;
            r[18] = 0x0000_CAFE;
            r[19] = 0x0000_BEEF;
            r[20] = 0xDEAD_FFFF;
            r[29] = 0x5555_0000 | (l as u32);
        }

        let (r, lanes) = run_in_mode(&code, &st, &opt);
        let mut want_lt_i32 = 0u64;
        let mut want_gt_f32 = 0u64;
        let mut want_cmpx = 0u64;

        for l in 0..lanes {
            let v = &r.vgpr[l];
            let v0 = st.vgpr[l][0];
            let b0 = v0 & 0xFF;
            let b1 = (v0 >> 8) & 0xFF;
            let b2 = (v0 >> 16) & 0xFF;
            let b3 = (v0 >> 24) & 0xFF;

            assert_eq!(v[10], b0 << 16, "{mode} lane {l} dst_sel:BYTE_2 UNUSED_PAD");
            let b1_sext = if b1 & 0x80 != 0 {
                0xFFFF_0000 | (b1 << 8)
            } else {
                b1 << 8
            };
            assert_eq!(v[11], b1_sext, "{mode} lane {l} dst_sel:BYTE_1 UNUSED_SEXT");
            assert_eq!(
                v[12],
                (b3 << 24) | 0x0022_3344,
                "{mode} lane {l} dst_sel:BYTE_3 PRESERVE"
            );
            // s4.word1 sext (-2) + (-1) = -3 (0xFFFF_FFFD) -> WORD_0 UNUSED_SEXT = 0xFFFF_FFFD
            assert_eq!(
                v[13], 0xFFFF_FFFD,
                "{mode} lane {l} s0/s1 scalar + WORD_0 SEXT"
            );
            let b2_i8 = b2 as u8 as i8 as f32;
            assert_eq!(
                f32::from_bits(v[14]),
                b2_i8,
                "{mode} lane {l} v_cvt_f32_i32 byte2 sext"
            );

            let f1 = f32::from_bits(st.vgpr[l][1]);
            let f2 = f32::from_bits(st.vgpr[l][2]);
            let want_v15 = (-f1.abs() + f2.abs()) * 2.0f32;
            assert_eq!(
                f32::from_bits(v[15]),
                want_v15,
                "{mode} lane {l} v_add_f32 sdwa omod"
            );
            let want_v16 = (f1 * f2 * 4.0f32).clamp(0.0, 1.0);
            assert_eq!(
                f32::from_bits(v[16]),
                want_v16,
                "{mode} lane {l} v_mul_f32 sdwa omod+clamp"
            );
            let want_v17 = (f2 - f1) * 0.5f32;
            assert_eq!(
                f32::from_bits(v[17]),
                want_v17,
                "{mode} lane {l} v_sub_f32 sdwa omod"
            );

            // v18: -f1 * 2.0 clamped to [0, 1]: even lanes -(-0.5)*2 = 1.0 (0x3C00), odd lanes -0.75*2 = -1.5 -> 0.0 (0x0000)
            let want_f16_bits = if l % 2 == 0 { 0x3C00u32 } else { 0x0000u32 };
            assert_eq!(
                v[18],
                (want_f16_bits << 16) | 0xCAFE,
                "{mode} lane {l} v_cvt_f16_f32 sdwa word1 preserve"
            );

            // v19: abs(-3.5h) -> 3 (even), abs(-1.5h) -> 1 (odd)
            assert_eq!(
                v[19],
                ((if l % 2 == 0 { 3u32 } else { 1u32 }) << 16) | 0xBEEF,
                "{mode} lane {l} v_cvt_u16_f16 sdwa word1 preserve"
            );
            assert_eq!(
                v[20],
                if l % 2 == 0 { 0x3C00 } else { 0x3900 },
                "{mode} lane {l} v_add_f16 sdwa word0 pad"
            );
            let sum_b2_b1 = ((b2 as u8 as i8 as i32) + (b1 as u8 as i8 as i32)) as u32 & 0xFF;
            let want_v21 = if sum_b2_b1 & 0x80 != 0 {
                0xFF00_0000 | (sum_b2_b1 << 16)
            } else {
                sum_b2_b1 << 16
            };
            assert_eq!(
                v[21], want_v21,
                "{mode} lane {l} src1_sext + dst_sel:BYTE_2 UNUSED_SEXT"
            );

            assert_eq!(
                v[29],
                if b0 == 0 {
                    v0
                } else {
                    0x5555_0000 | (l as u32)
                },
                "{mode} lane {l} v_cmpx_eq_u32 sdwa exec mask"
            );

            if (b1 as u8 as i8 as i32) < -16 {
                want_lt_i32 |= 1u64 << l;
            }
            if f1.abs() > -f2 {
                want_gt_f32 |= 1u64 << l;
            }
            if b0 == 0 {
                want_cmpx |= 1u64 << l;
            }
        }

        let want_vcc = mask64_for_lanes(&opt, want_lt_i32);
        let want_sd10 = mask64_for_lanes(&opt, want_gt_f32);
        let want_exec = mask64_for_lanes(&opt, want_cmpx);
        assert_eq!(
            r.sgpr[17],
            want_gt_f32.count_ones(),
            "{mode} s_bcnt1 over SDWA SDST"
        );
        assert_eq!(
            (r.sgpr[6], r.sgpr[7]),
            want_vcc,
            "{mode} v_cmp_lt_i32_sdwa vcc"
        );
        assert_eq!(
            (r.sgpr[106], r.sgpr[107]),
            want_vcc,
            "{mode} SD=1 and SDST=null must leave VCC untouched"
        );
        if opt.wave64 {
            assert_eq!(
                (r.sgpr[10], r.sgpr[11]),
                want_sd10,
                "{mode} v_cmp_gt_f32_sdwa sdst"
            );
            assert_eq!(r.sgpr[16], want_sd10.1, "{mode} v_cmp_gt_f32_sdwa s11 copy");
            assert_eq!(
                (r.sgpr[14], r.sgpr[15]),
                want_exec,
                "{mode} v_cmpx_eq_u32_sdwa exec"
            );
        } else {
            assert_eq!(
                r.sgpr[10], want_sd10.0,
                "{mode} wave32 v_cmp_gt_f32_sdwa sdst"
            );
            assert_eq!(
                r.sgpr[11], 0xDEAD_BEEF,
                "{mode} wave32 v_cmp_gt_f32_sdwa preserves s11"
            );
            assert_eq!(
                r.sgpr[16], 0xDEAD_BEEF,
                "{mode} wave32 v_cmp_gt_f32_sdwa s11 copy"
            );
            assert_eq!(
                r.sgpr[14], want_exec.0,
                "{mode} wave32 v_cmpx_eq_u32_sdwa exec"
            );
        }
    }
}
