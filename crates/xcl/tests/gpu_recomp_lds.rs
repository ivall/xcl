//! GPU verification tests for RDNA2 LDS (`DS_*`) instructions:
//! - Wide loads/stores (`DS_WRITE_B64`, `DS_READ_B64`, `DS_WRITE_B96`, `DS_READ_B96`,
//!   `DS_WRITE_B128`, `DS_READ_B128`) and `DS_NOP`
//! - Dual-offset and stride-64 loads/stores (`DS_WRITE2_B64`, `DS_READ2_B64`,
//!   `DS_WRITE2ST64_B32`, `DS_READ2ST64_B32`, `DS_WRITE2ST64_B64`, `DS_READ2ST64_B64`)
//! - Dual atomic exchanges (`DS_WRXCHG2_RTN_B32`, `DS_WRXCHG2ST64_RTN_B32`) and 32-bit LDS
//!   atomics (`DS_INC_RTN_U32`, `DS_DEC_RTN_U32`, `DS_RSUB_RTN_U32`, `DS_WRXCHG_RTN_B32`,
//!   `DS_MSKOR_RTN_B32`)
//! - D16 sub-dword accesses (`DS_WRITE_B8_D16_HI`, `DS_WRITE_B16_D16_HI`, `DS_READ_U8_D16`,
//!   `DS_READ_U8_D16_HI`, `DS_READ_I8_D16`, `DS_READ_I8_D16_HI`, `DS_READ_U16_D16`,
//!   `DS_READ_U16_D16_HI`)
//!
//! Covers 16-bit `(OFFSET1 << 8) | OFFSET0` byte offsets, `VDST == ADDR` / overlapping register
//! hazards, atomic sibling byte/halfword preservation, 8-to-16 sign extension and untouched half
//! preservation, divergent `EXEC` masking, split-wave64 `XCL_X_LDS_TOP` coexistence with
//! cross-half lane exchange, and all four wave execution modes.

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

fn ds(name: &'static str, fields: &[(&'static str, u64)]) -> Insn {
    (name, "ENC_DS", fields.to_vec(), None)
}

fn s_barrier() -> Insn {
    ("S_BARRIER", "ENC_SOPP", Vec::new(), None)
}

fn s_mov_b64(sd: u64, s0: u64) -> Insn {
    (
        "S_MOV_B64",
        "ENC_SOP1",
        vec![("SDST", sd), ("SSRC0", s0)],
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

#[test]
fn lds_wide_dual_d16_translate_and_compile_with_dxc() {
    let code = prog(&[
        // Include a cross-half lane read so split-wave64 places exchange slots in xcl_lds.
        (
            "V_READFIRSTLANE_B32",
            "ENC_VOP1",
            vec![("VDST", 10), ("SRC0", V)],
            None,
        ),
        ds("DS_NOP", &[]),
        ds(
            "DS_WRITE_B64",
            &[("ADDR", 0), ("DATA0", 2), ("OFFSET0", 0x10), ("OFFSET1", 1)],
        ),
        ds(
            "DS_READ_B64",
            &[("VDST", 4), ("ADDR", 0), ("OFFSET0", 0x10), ("OFFSET1", 1)],
        ),
        ds("DS_WRITE_B96", &[("ADDR", 0), ("DATA0", 6)]),
        ds("DS_READ_B96", &[("VDST", 9), ("ADDR", 0)]),
        ds("DS_WRITE_B128", &[("ADDR", 0), ("DATA0", 12)]),
        ds("DS_READ_B128", &[("VDST", 16), ("ADDR", 0)]),
        ds(
            "DS_WRITE2_B64",
            &[
                ("ADDR", 0),
                ("DATA0", 2),
                ("DATA1", 4),
                ("OFFSET0", 1),
                ("OFFSET1", 3),
            ],
        ),
        ds(
            "DS_READ2_B64",
            &[("VDST", 20), ("ADDR", 0), ("OFFSET0", 1), ("OFFSET1", 3)],
        ),
        ds(
            "DS_WRITE2ST64_B32",
            &[
                ("ADDR", 0),
                ("DATA0", 2),
                ("DATA1", 3),
                ("OFFSET0", 1),
                ("OFFSET1", 2),
            ],
        ),
        ds(
            "DS_READ2ST64_B32",
            &[("VDST", 24), ("ADDR", 0), ("OFFSET0", 1), ("OFFSET1", 2)],
        ),
        ds(
            "DS_WRITE2ST64_B64",
            &[
                ("ADDR", 0),
                ("DATA0", 2),
                ("DATA1", 4),
                ("OFFSET0", 1),
                ("OFFSET1", 2),
            ],
        ),
        ds(
            "DS_READ2ST64_B64",
            &[("VDST", 26), ("ADDR", 0), ("OFFSET0", 1), ("OFFSET1", 2)],
        ),
        ds(
            "DS_WRXCHG2_RTN_B32",
            &[
                ("VDST", 30),
                ("ADDR", 0),
                ("DATA0", 2),
                ("DATA1", 3),
                ("OFFSET0", 1),
                ("OFFSET1", 2),
            ],
        ),
        ds(
            "DS_WRXCHG2ST64_RTN_B32",
            &[
                ("VDST", 32),
                ("ADDR", 0),
                ("DATA0", 2),
                ("DATA1", 3),
                ("OFFSET0", 1),
                ("OFFSET1", 2),
            ],
        ),
        ds("DS_WRITE_B8_D16_HI", &[("ADDR", 0), ("DATA0", 2)]),
        ds(
            "DS_WRITE_B16_D16_HI",
            &[("ADDR", 0), ("DATA0", 2), ("OFFSET0", 2)],
        ),
        ds("DS_READ_U8_D16", &[("VDST", 34), ("ADDR", 0)]),
        ds("DS_READ_U8_D16_HI", &[("VDST", 34), ("ADDR", 0)]),
        ds("DS_READ_I8_D16", &[("VDST", 35), ("ADDR", 0)]),
        ds("DS_READ_I8_D16_HI", &[("VDST", 35), ("ADDR", 0)]),
        ds(
            "DS_READ_U16_D16",
            &[("VDST", 36), ("ADDR", 0), ("OFFSET0", 2)],
        ),
        ds(
            "DS_READ_U16_D16_HI",
            &[("VDST", 36), ("ADDR", 0), ("OFFSET0", 2)],
        ),
        s_barrier(),
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
        assert!(
            t.hlsl.contains("groupshared uint xcl_lds[8192];"),
            "{label}: missing xcl_lds\n{}",
            t.hlsl
        );
        if opt.wave64 && opt.split_wave64 {
            assert!(
                t.hlsl.contains("XCL_X_LDS_TOP"),
                "{label}: expected XCL_X_LDS_TOP in split-wave64\n{}",
                t.hlsl
            );
            assert!(
                t.hlsl.contains("xcl_x_rla_uniform("),
                "{label}: expected xcl_x_rla_uniform\n{}",
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
fn gpu_lds_wide_b64_b96_b128_and_addr_overlap() {
    let configs = wave_configs();
    if configs.is_empty() {
        return;
    }

    // Region layout in LDS (bytes, all below XCL_X_LDS_TOP):
    //   0x0110..0x0310 : b64  per-lane (written at v0 + 0x0110, v0 = 8 * lane; low word also
    //                    cross-checked via DS_READ_B32 at v7 = 0x0110 + 8 * lane, offset 0)
    //   0x0400..0x0700 : b96  per-lane (v1 + 0x0080, v1 = 0x0380 + 12 * lane)
    //   0x0800..0x0C00 : b128 per-lane (v2 + 0x0104 -> 0x0800 + 16 * lane; word 0 also
    //                    cross-checked via DS_READ_B32 at v8 = 0x0800 + 16 * lane, offset 0)
    //   0x1000..0x1400 : b128 EXEC-masked sentinel region (v4 = 0x1000 + 16 * lane)
    let (mask_lo, mask_hi) = (0x5555_AAAAu32, 0xF0F0_0F0Fu32);

    let code = prog(&[
        // Cross-half exchange to exercise split-wave64 exchange inside xcl_lds alongside LDS ops.
        (
            "V_READFIRSTLANE_B32",
            "ENC_VOP1",
            vec![("VDST", 11), ("SRC0", V)],
            None,
        ),
        ds("DS_NOP", &[]),
        // b64 with 16-bit byte offset 0x0110 (OFFSET0 = 0x10, OFFSET1 = 0x01)
        ds(
            "DS_WRITE_B64",
            &[
                ("ADDR", 0),
                ("DATA0", 10),
                ("OFFSET0", 0x10),
                ("OFFSET1", 1),
            ],
        ),
        // b96 with 16-bit byte offset 0x0080
        ds(
            "DS_WRITE_B96",
            &[
                ("ADDR", 1),
                ("DATA0", 12),
                ("OFFSET0", 0x80),
                ("OFFSET1", 0),
            ],
        ),
        // b128 with 16-bit byte offset 0x0104 (OFFSET0 = 0x04, OFFSET1 = 0x01)
        ds(
            "DS_WRITE_B128",
            &[
                ("ADDR", 2),
                ("DATA0", 16),
                ("OFFSET0", 0x04),
                ("OFFSET1", 1),
            ],
        ),
        s_barrier(),
        // Neighbor reads across (lane ^ 1)
        ds(
            "DS_READ_B64",
            &[("VDST", 20), ("ADDR", 3), ("OFFSET0", 0x10), ("OFFSET1", 1)],
        ),
        ds(
            "DS_READ_B96",
            &[("VDST", 22), ("ADDR", 5), ("OFFSET0", 0x80), ("OFFSET1", 0)],
        ),
        ds(
            "DS_READ_B128",
            &[("VDST", 26), ("ADDR", 6), ("OFFSET0", 0x04), ("OFFSET1", 1)],
        ),
        // Independent offset-0 cross-checks proving (OFFSET1 << 8) | OFFSET0 landed at the exact byte offset
        ds("DS_READ_B32", &[("VDST", 48), ("ADDR", 7)]),
        ds("DS_READ_B32", &[("VDST", 49), ("ADDR", 8)]),
        // Hazard 1: VDST + 1 == ADDR on DS_READ_B64 (v[30:31] reads from v31 + 0x0110)
        ds(
            "DS_READ_B64",
            &[
                ("VDST", 30),
                ("ADDR", 31),
                ("OFFSET0", 0x10),
                ("OFFSET1", 1),
            ],
        ),
        // Hazard 2: VDST == ADDR on DS_READ_B128 (v[32:35] reads from v32 + 0x0104)
        ds(
            "DS_READ_B128",
            &[
                ("VDST", 32),
                ("ADDR", 32),
                ("OFFSET0", 0x04),
                ("OFFSET1", 1),
            ],
        ),
        // Divergent EXEC test: seed all lanes' sentinel region, then overwrite/read under mask
        ds("DS_WRITE_B128", &[("ADDR", 4), ("DATA0", 36)]),
        s_barrier(),
        s_mov_b64(126, 2), // EXEC = s[2:3]
        ds("DS_WRITE_B128", &[("ADDR", 4), ("DATA0", 16)]),
        s_barrier(),
        ds("DS_READ_B128", &[("VDST", 40), ("ADDR", 4)]),
        s_mov_b64(126, 193), // EXEC = -1 (all lanes)
        s_barrier(),
        // Read final state of sentinel region with all lanes active
        ds("DS_READ_B128", &[("VDST", 44), ("ADDR", 4)]),
    ]);

    let mut st = KernelState::default();
    st.sgpr[2] = mask_lo;
    st.sgpr[3] = mask_hi;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        let nb = (l ^ 1) as u32;
        r[0] = 8 * l32;
        r[1] = 0x0380 + 12 * l32;
        r[2] = 0x06FC + 16 * l32; // + 0x0104 = 0x0800 + 16 * l
        r[3] = 8 * nb;
        r[4] = 0x1000 + 16 * l32;
        r[5] = 0x0380 + 12 * nb;
        r[6] = 0x06FC + 16 * nb;
        r[7] = 0x0110 + 8 * l32;
        r[8] = 0x0800 + 16 * l32;

        // Overlap address registers (read own lane's data)
        r[31] = 8 * l32;
        r[32] = 0x06FC + 16 * l32;

        // Distinct payloads
        r[10] = 0xB640_0000 | l32;
        r[11] = 0xB641_0000 | (l32 * 3 + 1);
        r[12] = 0xB960_0000 | l32;
        r[13] = 0xB961_0000 | (l32 * 5 + 2);
        r[14] = 0xB962_0000 | (l32 * 7 + 3);
        r[16] = 0xB128_0000 | l32;
        r[17] = 0xB129_0000 | (l32 * 11 + 4);
        r[18] = 0xB12A_0000 | (l32 * 13 + 5);
        r[19] = 0xB12B_0000 | (l32 * 17 + 6);

        // Sentinels for EXEC masking
        r[36] = 0xDEAD_0000 | l32;
        r[37] = 0xDEAD_1000 | l32;
        r[38] = 0xDEAD_2000 | l32;
        r[39] = 0xDEAD_3000 | l32;
        r[40] = 0xCAFE_0000 | l32;
        r[41] = 0xCAFE_1000 | l32;
        r[42] = 0xCAFE_2000 | l32;
        r[43] = 0xCAFE_3000 | l32;
    }

    for (label, opt) in configs {
        let dxil = compile_kernel(&code, &opt);
        let r = runner::run(&dxil, &st, &[]).unwrap();
        assert_eq!(
            r.sgpr[11], 0,
            "{label}: V_READFIRSTLANE_B32 across LDS shader"
        );
        let lanes = active_lanes_for(&opt);
        for (l, v) in r.vgpr.iter().enumerate().take(lanes) {
            let l32 = l as u32;
            let nb = (l ^ 1) as u32;
            assert_eq!(
                [v[20], v[21]],
                [0xB640_0000 | nb, 0xB641_0000 | (nb * 3 + 1)],
                "{label} lane {l}: ds_read_b64 neighbor"
            );
            assert_eq!(
                [v[22], v[23], v[24]],
                [
                    0xB960_0000 | nb,
                    0xB961_0000 | (nb * 5 + 2),
                    0xB962_0000 | (nb * 7 + 3),
                ],
                "{label} lane {l}: ds_read_b96 neighbor"
            );
            assert_eq!(
                [v[26], v[27], v[28], v[29]],
                [
                    0xB128_0000 | nb,
                    0xB129_0000 | (nb * 11 + 4),
                    0xB12A_0000 | (nb * 13 + 5),
                    0xB12B_0000 | (nb * 17 + 6),
                ],
                "{label} lane {l}: ds_read_b128 neighbor"
            );
            assert_eq!(
                v[48],
                0xB640_0000 | l32,
                "{label} lane {l}: ds_write_b64 exact off16 byte address"
            );
            assert_eq!(
                v[49],
                0xB128_0000 | l32,
                "{label} lane {l}: ds_write_b128 exact off16 byte address"
            );
            assert_eq!(
                [v[30], v[31]],
                [0xB640_0000 | l32, 0xB641_0000 | (l32 * 3 + 1)],
                "{label} lane {l}: ds_read_b64 VDST+1==ADDR overlap"
            );
            assert_eq!(
                [v[32], v[33], v[34], v[35]],
                [
                    0xB128_0000 | l32,
                    0xB129_0000 | (l32 * 11 + 4),
                    0xB12A_0000 | (l32 * 13 + 5),
                    0xB12B_0000 | (l32 * 17 + 6),
                ],
                "{label} lane {l}: ds_read_b128 VDST==ADDR overlap"
            );

            let active = is_active(l, mask_lo, mask_hi, &opt);
            let want_masked_read = if active {
                [
                    0xB128_0000 | l32,
                    0xB129_0000 | (l32 * 11 + 4),
                    0xB12A_0000 | (l32 * 13 + 5),
                    0xB12B_0000 | (l32 * 17 + 6),
                ]
            } else {
                [
                    0xCAFE_0000 | l32,
                    0xCAFE_1000 | l32,
                    0xCAFE_2000 | l32,
                    0xCAFE_3000 | l32,
                ]
            };
            assert_eq!(
                [v[40], v[41], v[42], v[43]],
                want_masked_read,
                "{label} lane {l}: masked ds_read_b128"
            );

            let want_lds = if active {
                [
                    0xB128_0000 | l32,
                    0xB129_0000 | (l32 * 11 + 4),
                    0xB12A_0000 | (l32 * 13 + 5),
                    0xB12B_0000 | (l32 * 17 + 6),
                ]
            } else {
                [
                    0xDEAD_0000 | l32,
                    0xDEAD_1000 | l32,
                    0xDEAD_2000 | l32,
                    0xDEAD_3000 | l32,
                ]
            };
            assert_eq!(
                [v[44], v[45], v[46], v[47]],
                want_lds,
                "{label} lane {l}: masked ds_write_b128"
            );
        }
    }
}

#[test]
fn gpu_lds_dual_stride64_and_atomics() {
    let configs = wave_configs();
    if configs.is_empty() {
        return;
    }

    // LDS layout (bytes, staying safely below XCL_X_LDS_TOP ~ 32 KB):
    //   0x0000..0x0400 : DS_WRITE2_B64 / DS_READ2_B64 (v0 = 16*lane; o0=1 (+8), o1=3 (+24))
    //                    Wait: stride between lanes must avoid overlap -> v0 = 32 * lane:
    //                    slot0 at 32*lane + 8, slot1 at 32*lane + 24 -> 0..2048 bytes.
    //   0x0800..0x0C00 : DS_WRITE2ST64_B32 / DS_READ2ST64_B32 (v1 = 0x0700 + 4*lane;
    //                    o0=1 (+256 -> 0x0800+4*lane), o1=3 (+768 -> 0x0A00+4*lane))
    //   0x1000..0x1800 : DS_WRITE2ST64_B64 / DS_READ2ST64_B64 (v2 = 0x0C00 + 8*lane;
    //                    o0=2 (+1024 -> 0x1000+8*lane), o1=5 (+2560 -> 0x1600+8*lane))
    //   0x2000..0x2600 : DS_WRXCHG2_RTN_B32 per-lane (v3 = 0x2000 + 8*lane; o0=0 (+0), o1=1 (+4))
    //                    and DS_WRXCHG2ST64_RTN_B32 (v4 = 0x2200 + 4*lane; o0=1 (+256), o1=3 (+768))
    //   0x2800..0x2818 : Shared contention slots for DS_WRXCHG2_RTN_B32 (v5 = 0x2800; o0=1, o1=4)
    //   0x2900..0x2D00 : 32-bit atomics per-lane (INC, DEC, RSUB, WRXCHG, MSKOR)
    let (mask_lo, mask_hi) = (0xA55A_33CCu32, 0x0FF0_F00Fu32);

    let code = prog(&[
        // 1. DS_WRITE2_B64 (stride 8 bytes)
        ds(
            "DS_WRITE2_B64",
            &[
                ("ADDR", 0),
                ("DATA0", 10),
                ("DATA1", 12),
                ("OFFSET0", 1),
                ("OFFSET1", 3),
            ],
        ),
        // 2. DS_WRITE2ST64_B32 (stride 256 bytes)
        ds(
            "DS_WRITE2ST64_B32",
            &[
                ("ADDR", 1),
                ("DATA0", 14),
                ("DATA1", 15),
                ("OFFSET0", 1),
                ("OFFSET1", 3),
            ],
        ),
        // 3. DS_WRITE2ST64_B64 (stride 512 bytes)
        ds(
            "DS_WRITE2ST64_B64",
            &[
                ("ADDR", 2),
                ("DATA0", 16),
                ("DATA1", 18),
                ("OFFSET0", 2),
                ("OFFSET1", 5),
            ],
        ),
        // 4. Seed per-lane dual-exchange slots & shared contention slots & 32-bit atomic slots
        ds(
            "DS_WRITE2_B32",
            &[
                ("ADDR", 3),
                ("DATA0", 20),
                ("DATA1", 21),
                ("OFFSET0", 0),
                ("OFFSET1", 1),
            ],
        ),
        ds(
            "DS_WRITE2ST64_B32",
            &[
                ("ADDR", 4),
                ("DATA0", 22),
                ("DATA1", 23),
                ("OFFSET0", 1),
                ("OFFSET1", 3),
            ],
        ),
        // Seed shared dual-exchange slots at 0x2804 and 0x2810 (from lane 0 via v5)
        ds(
            "DS_WRITE2_B32",
            &[
                ("ADDR", 5),
                ("DATA0", 24),
                ("DATA1", 25),
                ("OFFSET0", 1),
                ("OFFSET1", 4),
            ],
        ),
        // Seed 32-bit atomic slots at v6 + 0, +4, +8, +12, +16
        ds(
            "DS_WRITE_B32",
            &[("ADDR", 6), ("DATA0", 26), ("OFFSET0", 0)],
        ),
        ds(
            "DS_WRITE_B32",
            &[("ADDR", 6), ("DATA0", 27), ("OFFSET0", 4)],
        ),
        ds(
            "DS_WRITE_B32",
            &[("ADDR", 6), ("DATA0", 28), ("OFFSET0", 8)],
        ),
        ds(
            "DS_WRITE_B32",
            &[("ADDR", 6), ("DATA0", 29), ("OFFSET0", 12)],
        ),
        ds(
            "DS_WRITE_B32",
            &[("ADDR", 6), ("DATA0", 30), ("OFFSET0", 16)],
        ),
        s_barrier(),
        // Cross-check DS_READ2_B64 (with VDST+1 == ADDR overlap on v[36..39], ADDR = v37 = v0)
        ds(
            "DS_READ2_B64",
            &[("VDST", 36), ("ADDR", 37), ("OFFSET0", 1), ("OFFSET1", 3)],
        ),
        // Verify exact byte offsets of DS_WRITE2_B64 using single DS_READ_B64
        ds("DS_READ_B64", &[("VDST", 40), ("ADDR", 0), ("OFFSET0", 24)]),
        // Read DS_READ2ST64_B32 + verify exact byte address 0x0A00 (= 0x0700 + 768)
        ds(
            "DS_READ2ST64_B32",
            &[("VDST", 42), ("ADDR", 1), ("OFFSET0", 1), ("OFFSET1", 3)],
        ),
        ds(
            "DS_READ_B32",
            &[("VDST", 44), ("ADDR", 1), ("OFFSET0", 0), ("OFFSET1", 3)], // +768 bytes
        ),
        // Read DS_READ2ST64_B64 (VDST == ADDR overlap: v[46..49], ADDR = v46 = v2) + single DS_READ_B64 check
        ds(
            "DS_READ2ST64_B64",
            &[("VDST", 46), ("ADDR", 46), ("OFFSET0", 2), ("OFFSET1", 5)],
        ),
        ds(
            "DS_READ_B64",
            &[("VDST", 50), ("ADDR", 2), ("OFFSET0", 0), ("OFFSET1", 10)], // +2560 bytes (0x0A00)
        ),
        // Dual atomic exchanges: per-lane with VDST == ADDR overlap (v[52:53], ADDR = v52 = v3)
        ds(
            "DS_WRXCHG2_RTN_B32",
            &[
                ("VDST", 52),
                ("ADDR", 52),
                ("DATA0", 14),
                ("DATA1", 15),
                ("OFFSET0", 0),
                ("OFFSET1", 1),
            ],
        ),
        ds(
            "DS_READ2_B32",
            &[("VDST", 54), ("ADDR", 3), ("OFFSET0", 0), ("OFFSET1", 1)],
        ),
        ds(
            "DS_WRXCHG2ST64_RTN_B32",
            &[
                ("VDST", 56),
                ("ADDR", 4),
                ("DATA0", 14),
                ("DATA1", 15),
                ("OFFSET0", 1),
                ("OFFSET1", 3),
            ],
        ),
        ds(
            "DS_READ2ST64_B32",
            &[("VDST", 58), ("ADDR", 4), ("OFFSET0", 1), ("OFFSET1", 3)],
        ),
        s_barrier(),
        // Shared multi-lane contention on DS_WRXCHG2_RTN_B32 (all active lanes race on 0x2804 & 0x2810)
        ds(
            "DS_WRXCHG2_RTN_B32",
            &[
                ("VDST", 60),
                ("ADDR", 5),
                ("DATA0", 7),
                ("DATA1", 8),
                ("OFFSET0", 1),
                ("OFFSET1", 4),
            ],
        ),
        s_barrier(),
        ds(
            "DS_READ2_B32",
            &[("VDST", 62), ("ADDR", 5), ("OFFSET0", 1), ("OFFSET1", 4)],
        ),
        // 32-bit LDS atomics: INC, DEC, RSUB, WRXCHG, MSKOR
        ds(
            "DS_INC_RTN_U32",
            &[("VDST", 64), ("ADDR", 6), ("DATA0", 31), ("OFFSET0", 0)],
        ),
        ds(
            "DS_DEC_RTN_U32",
            &[("VDST", 65), ("ADDR", 6), ("DATA0", 31), ("OFFSET0", 4)],
        ),
        ds(
            "DS_RSUB_RTN_U32",
            &[("VDST", 66), ("ADDR", 6), ("DATA0", 32), ("OFFSET0", 8)],
        ),
        ds(
            "DS_WRXCHG_RTN_B32",
            &[("VDST", 67), ("ADDR", 6), ("DATA0", 33), ("OFFSET0", 12)],
        ),
        ds(
            "DS_MSKOR_RTN_B32",
            &[
                ("VDST", 68),
                ("ADDR", 6),
                ("DATA0", 34),
                ("DATA1", 35),
                ("OFFSET0", 16),
            ],
        ),
        ds("DS_READ_B32", &[("VDST", 69), ("ADDR", 6), ("OFFSET0", 0)]),
        ds("DS_READ_B32", &[("VDST", 70), ("ADDR", 6), ("OFFSET0", 4)]),
        ds("DS_READ_B32", &[("VDST", 71), ("ADDR", 6), ("OFFSET0", 8)]),
        ds("DS_READ_B32", &[("VDST", 72), ("ADDR", 6), ("OFFSET0", 12)]),
        ds("DS_READ_B32", &[("VDST", 73), ("ADDR", 6), ("OFFSET0", 16)]),
        // Divergent EXEC check on DS_WRXCHG2_RTN_B32 and DS_READ2ST64_B64
        s_mov_b64(126, 2),
        ds(
            "DS_WRXCHG2_RTN_B32",
            &[
                ("VDST", 74),
                ("ADDR", 3),
                ("DATA0", 20),
                ("DATA1", 21),
                ("OFFSET0", 0),
                ("OFFSET1", 1),
            ],
        ),
        ds(
            "DS_READ2ST64_B64",
            &[("VDST", 76), ("ADDR", 2), ("OFFSET0", 2), ("OFFSET1", 5)],
        ),
        s_mov_b64(126, 193),
        s_barrier(),
        ds(
            "DS_READ2_B32",
            &[("VDST", 80), ("ADDR", 3), ("OFFSET0", 0), ("OFFSET1", 1)],
        ),
    ]);

    let mut st = KernelState::default();
    st.sgpr[2] = mask_lo;
    st.sgpr[3] = mask_hi;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[0] = 32 * l32;
        r[37] = 32 * l32;
        r[1] = 0x0700 + 4 * l32;
        r[2] = 0x0C00 + 8 * l32;
        r[46] = 0x0C00 + 8 * l32;
        r[3] = 0x2000 + 8 * l32;
        r[52] = 0x2000 + 8 * l32;
        r[4] = 0x2200 + 4 * l32;
        r[5] = 0x2800; // shared by all lanes
        r[6] = 0x2900 + 32 * l32;

        // Shared contention tags
        r[7] = 0x7000_0000 | l32;
        r[8] = 0x8000_0000 | l32;

        // Data for dual writes
        r[10] = 0xD264_0000 | l32;
        r[11] = 0xD264_1000 | l32;
        r[12] = 0xD264_2000 | l32;
        r[13] = 0xD264_3000 | l32;
        r[14] = 0x5632_0000 | l32;
        r[15] = 0x5632_1000 | l32;
        r[16] = 0x5664_0000 | l32;
        r[17] = 0x5664_1000 | l32;
        r[18] = 0x5664_2000 | l32;
        r[19] = 0x5664_3000 | l32;

        // Initial values for per-lane wrxchg2
        r[20] = 0x1111_0000 | l32;
        r[21] = 0x2222_0000 | l32;
        r[22] = 0x3333_0000 | l32;
        r[23] = 0x4444_0000 | l32;

        // Initial values for shared slots
        r[24] = 0xAAAA_0000;
        r[25] = 0xBBBB_0000;

        // 32-bit atomic initial values & operands
        // INC wrap threshold = 9; test wrap at 9 -> 0 on odd lanes, increment on even lanes
        r[26] = if l % 2 == 0 { 4 } else { 9 };
        // DEC threshold = 9; test wrap at 0 -> 9 on even lanes, out-of-range 12 -> 9 on lanes % 4 == 1, else l-1
        r[27] = match l % 4 {
            0 => 0,
            1 => 12,
            _ => 5,
        };
        r[28] = 100 + l32; // RSUB init: 1000 - (100 + l) = 900 - l
        r[29] = 0x6666_0000 | l32; // WRXCHG init
        r[30] = 0xF0F0_0F0F; // MSKOR init: (old & ~mask) | or_val
        r[31] = 9; // INC/DEC limit
        r[32] = 1000; // RSUB minuend
        r[33] = 0x7777_0000 | l32; // WRXCHG new
        r[34] = 0x00FF_00FF; // MSKOR clear mask
        r[35] = 0x00A5_005A; // MSKOR or value

        // Sentinels for EXEC-masked outputs
        for reg in [74, 75, 76, 77, 78, 79] {
            r[reg] = 0xFACE_0000 | ((reg as u32) << 8) | l32;
        }
    }

    for (label, opt) in configs {
        let dxil = compile_kernel(&code, &opt);
        let r = runner::run(&dxil, &st, &[]).unwrap();
        let lanes = active_lanes_for(&opt);

        for (l, v) in r.vgpr.iter().enumerate().take(lanes) {
            let l32 = l as u32;

            // DS_READ2_B64 & exact offset DS_READ_B64
            assert_eq!(
                [v[36], v[37], v[38], v[39]],
                [
                    0xD264_0000 | l32,
                    0xD264_1000 | l32,
                    0xD264_2000 | l32,
                    0xD264_3000 | l32,
                ],
                "{label} lane {l}: ds_read2_b64 (with overlap)"
            );
            assert_eq!(
                [v[40], v[41]],
                [0xD264_2000 | l32, 0xD264_3000 | l32],
                "{label} lane {l}: ds_write2_b64 byte offset check"
            );

            // DS_READ2ST64_B32 & byte offset check
            assert_eq!(
                [v[42], v[43], v[44]],
                [0x5632_0000 | l32, 0x5632_1000 | l32, 0x5632_1000 | l32],
                "{label} lane {l}: ds_read2st64_b32"
            );

            // DS_READ2ST64_B64 & byte offset check
            assert_eq!(
                [v[46], v[47], v[48], v[49]],
                [
                    0x5664_0000 | l32,
                    0x5664_1000 | l32,
                    0x5664_2000 | l32,
                    0x5664_3000 | l32,
                ],
                "{label} lane {l}: ds_read2st64_b64 (with overlap)"
            );
            assert_eq!(
                [v[50], v[51]],
                [0x5664_2000 | l32, 0x5664_3000 | l32],
                "{label} lane {l}: ds_write2st64_b64 byte offset check"
            );

            // Per-lane DS_WRXCHG2_RTN_B32 and DS_WRXCHG2ST64_RTN_B32
            assert_eq!(
                [v[52], v[53]],
                [0x1111_0000 | l32, 0x2222_0000 | l32],
                "{label} lane {l}: ds_wrxchg2_rtn_b32 old"
            );
            assert_eq!(
                [v[54], v[55]],
                [0x5632_0000 | l32, 0x5632_1000 | l32],
                "{label} lane {l}: ds_wrxchg2_rtn_b32 stored"
            );
            assert_eq!(
                [v[56], v[57]],
                [0x3333_0000 | l32, 0x4444_0000 | l32],
                "{label} lane {l}: ds_wrxchg2st64_rtn_b32 old"
            );
            assert_eq!(
                [v[58], v[59]],
                [0x5632_0000 | l32, 0x5632_1000 | l32],
                "{label} lane {l}: ds_wrxchg2st64_rtn_b32 stored"
            );

            // 32-bit atomics old + new
            let want_inc_old = if l % 2 == 0 { 4 } else { 9 };
            let want_inc_new = if l % 2 == 0 { 5 } else { 0 };
            let (want_dec_old, want_dec_new) = match l % 4 {
                0 => (0, 9),
                1 => (12, 9),
                _ => (5, 4),
            };
            assert_eq!(
                (v[64], v[69]),
                (want_inc_old, want_inc_new),
                "{label} lane {l}: ds_inc_rtn_u32"
            );
            assert_eq!(
                (v[65], v[70]),
                (want_dec_old, want_dec_new),
                "{label} lane {l}: ds_dec_rtn_u32"
            );
            assert_eq!(
                (v[66], v[71]),
                (100 + l32, 900 - l32),
                "{label} lane {l}: ds_rsub_rtn_u32"
            );
            assert_eq!(
                (v[67], v[72]),
                (0x6666_0000 | l32, 0x7777_0000 | l32),
                "{label} lane {l}: ds_wrxchg_rtn_b32"
            );
            assert_eq!(
                (v[68], v[73]),
                (0xF0F0_0F0F, 0xF0A5_0F5A),
                "{label} lane {l}: ds_mskor_rtn_b32"
            );

            // Divergent EXEC masking on DS_WRXCHG2_RTN_B32 and DS_READ2ST64_B64
            let active = is_active(l, mask_lo, mask_hi, &opt);
            if active {
                assert_eq!(
                    [v[74], v[75]],
                    [0x5632_0000 | l32, 0x5632_1000 | l32],
                    "{label} lane {l}: active masked ds_wrxchg2_rtn_b32"
                );
                assert_eq!(
                    [v[80], v[81]],
                    [0x1111_0000 | l32, 0x2222_0000 | l32],
                    "{label} lane {l}: active masked ds_wrxchg2_rtn_b32 stored"
                );
                assert_eq!(
                    [v[76], v[77], v[78], v[79]],
                    [
                        0x5664_0000 | l32,
                        0x5664_1000 | l32,
                        0x5664_2000 | l32,
                        0x5664_3000 | l32,
                    ],
                    "{label} lane {l}: active masked ds_read2st64_b64"
                );
            } else {
                assert_eq!(
                    [v[74], v[75]],
                    [0xFACE_4A00 | l32, 0xFACE_4B00 | l32],
                    "{label} lane {l}: inactive masked ds_wrxchg2_rtn_b32"
                );
                assert_eq!(
                    [v[80], v[81]],
                    [0x5632_0000 | l32, 0x5632_1000 | l32],
                    "{label} lane {l}: inactive masked ds_wrxchg2_rtn_b32 kept LDS"
                );
                assert_eq!(
                    [v[76], v[77], v[78], v[79]],
                    [
                        0xFACE_4C00 | l32,
                        0xFACE_4D00 | l32,
                        0xFACE_4E00 | l32,
                        0xFACE_4F00 | l32,
                    ],
                    "{label} lane {l}: inactive masked ds_read2st64_b64"
                );
            }
        }

        // Verify shared-slot multi-lane DS_WRXCHG2_RTN_B32 linearizability:
        // {initial} U {written tags} == {returned olds} U {final LDS value}
        let final0 = r.vgpr[0][62];
        let final1 = r.vgpr[0][63];
        let mut expected0: Vec<u32> = (0..lanes as u32).map(|l| 0x7000_0000 | l).collect();
        expected0.push(0xAAAA_0000);
        expected0.sort_unstable();
        let mut got0: Vec<u32> = r.vgpr[..lanes].iter().map(|v| v[60]).collect();
        got0.push(final0);
        got0.sort_unstable();
        assert_eq!(
            got0, expected0,
            "{label}: ds_wrxchg2_rtn_b32 slot 0 permutation"
        );

        let mut expected1: Vec<u32> = (0..lanes as u32).map(|l| 0x8000_0000 | l).collect();
        expected1.push(0xBBBB_0000);
        expected1.sort_unstable();
        let mut got1: Vec<u32> = r.vgpr[..lanes].iter().map(|v| v[61]).collect();
        got1.push(final1);
        got1.sort_unstable();
        assert_eq!(
            got1, expected1,
            "{label}: ds_wrxchg2_rtn_b32 slot 1 permutation"
        );
    }
}

#[test]
fn gpu_lds_d16_subdword_and_sign_extension() {
    let configs = wave_configs();
    if configs.is_empty() {
        return;
    }

    // LDS layout (bytes):
    //   0x0000..0x0040 : 16 shared dwords written concurrently by 64 lanes via DS_WRITE_B8_D16_HI
    //   0x0100..0x0180 : 32 shared dwords written concurrently by 64 lanes via DS_WRITE_B16_D16_HI
    //   0x0300..0x0400 : preloaded edge bytes/halfwords read at base + off16 (0x0104 = 260)
    //   0x0500..0x0540 : EXEC-masked sub-dword write sentinel region
    let (mask_lo, mask_hi) = (0x0F0F_5555u32, 0xAAAA_F0F0u32);

    let code = prog(&[
        // 1. Concurrent 4-bytes-per-dword writes from neighboring lanes
        ds("DS_WRITE_B8_D16_HI", &[("ADDR", 0), ("DATA0", 10)]),
        // 2. Concurrent 2-halfwords-per-dword writes from neighboring lanes
        ds("DS_WRITE_B16_D16_HI", &[("ADDR", 1), ("DATA0", 11)]),
        // 3. Seed edge-value dword region at v6 (= 0x0300 + 4*lane, offset 0) and read at v2 + 0x0104..0x0107
        ds("DS_WRITE_B32", &[("ADDR", 6), ("DATA0", 12)]),
        // Seed EXEC-masked region
        ds("DS_WRITE_B32", &[("ADDR", 3), ("DATA0", 13)]),
        s_barrier(),
        // Read back shared byte-written dword and halfword-written dword
        ds("DS_READ_B32", &[("VDST", 14), ("ADDR", 4)]),
        ds("DS_READ_B32", &[("VDST", 15), ("ADDR", 5)]),
        // D16 reads across all 4 byte alignments + nonzero off16 (0x0104 + b)
        // v16..v21 are preloaded with distinct 32-bit sentinels so both half preservations are checked
        ds(
            "DS_READ_U8_D16",
            &[("VDST", 16), ("ADDR", 2), ("OFFSET0", 0x04), ("OFFSET1", 1)],
        ),
        ds(
            "DS_READ_U8_D16_HI",
            &[("VDST", 17), ("ADDR", 2), ("OFFSET0", 0x05), ("OFFSET1", 1)],
        ),
        ds(
            "DS_READ_I8_D16",
            &[("VDST", 18), ("ADDR", 2), ("OFFSET0", 0x06), ("OFFSET1", 1)],
        ),
        ds(
            "DS_READ_I8_D16_HI",
            &[("VDST", 19), ("ADDR", 2), ("OFFSET0", 0x07), ("OFFSET1", 1)],
        ),
        ds(
            "DS_READ_U16_D16",
            &[("VDST", 20), ("ADDR", 2), ("OFFSET0", 0x04), ("OFFSET1", 1)],
        ),
        ds(
            "DS_READ_U16_D16_HI",
            &[("VDST", 21), ("ADDR", 2), ("OFFSET0", 0x06), ("OFFSET1", 1)],
        ),
        // VDST == ADDR overlap on D16 read: v22 holds address (0x01FC + 4*lane), reads into v22[15:0]
        ds(
            "DS_READ_I8_D16",
            &[
                ("VDST", 22),
                ("ADDR", 22),
                ("OFFSET0", 0x07),
                ("OFFSET1", 1),
            ],
        ),
        // Divergent EXEC masking on D16 write & read
        s_mov_b64(126, 2),
        ds("DS_WRITE_B8_D16_HI", &[("ADDR", 3), ("DATA0", 10)]),
        ds(
            "DS_WRITE_B16_D16_HI",
            &[("ADDR", 3), ("DATA0", 11), ("OFFSET0", 2)],
        ),
        ds(
            "DS_READ_U16_D16_HI",
            &[("VDST", 23), ("ADDR", 2), ("OFFSET0", 0x04), ("OFFSET1", 1)],
        ),
        s_mov_b64(126, 193),
        s_barrier(),
        ds("DS_READ_B32", &[("VDST", 24), ("ADDR", 3)]),
    ]);

    const EDGE_BYTES: [[u8; 4]; 4] = [
        [0x00, 0x7F, 0x80, 0xFF],
        [0x7F, 0x80, 0xFF, 0x01],
        [0x80, 0xFF, 0x00, 0x81],
        [0xFF, 0x01, 0x7F, 0xFE],
    ];

    let mut st = KernelState::default();
    st.sgpr[2] = mask_lo;
    st.sgpr[3] = mask_hi;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[0] = l32; // 4 lanes share each dword at 0x0000 + (l & !3)
        r[1] = 0x0100 + 2 * l32; // 2 lanes share each dword at 0x0100 + 4*(l/2)
        r[2] = 0x01FC + 4 * l32; // + 0x0104 = 0x0300 + 4*l
        r[3] = 0x0500 + 4 * l32;
        r[4] = l32 & !3; // dword base of lane l's byte group
        r[5] = 0x0100 + 4 * (l32 / 2); // dword base of lane l's halfword pair
        r[6] = 0x0300 + 4 * l32; // offset-0 seed address for v2 + 0x0104..0x0107 reads

        // Pack target byte into bits [23:16] with dirty bits in [31:24] and [15:0]
        let b = (0x40 + l32) & 0xFF;
        r[10] = 0xAA00_BEEF | (b << 16);
        // Pack target halfword into bits [31:16] with dirty bits in [15:0]
        let hw = 0x9000 + l32;
        r[11] = (hw << 16) | 0xCAFE;

        let eb = EDGE_BYTES[l % 4];
        r[12] = u32::from_le_bytes(eb);
        r[13] = 0xDEAD_BEEF;

        // Preload half-preservation sentinels in v16..v21 and v23
        r[16] = 0xA1A1_1111;
        r[17] = 0xB2B2_2222;
        r[18] = 0xC3C3_3333;
        r[19] = 0xD4D4_4444;
        r[20] = 0xE5E5_5555;
        r[21] = 0xF6F6_6666;
        r[22] = 0x01FC + 4 * l32; // VDST == ADDR overlap
        r[23] = 0x1357_2468;
    }

    for (label, opt) in configs {
        let dxil = compile_kernel(&code, &opt);
        let r = runner::run(&dxil, &st, &[]).unwrap();
        let lanes = active_lanes_for(&opt);

        for (l, v) in r.vgpr.iter().enumerate().take(lanes) {
            let l32 = l as u32;

            // 4 neighboring lanes' bytes combined into one dword
            let gbase = l32 & !3;
            let want_b8_dword = ((0x40 + gbase) & 0xFF)
                | (((0x40 + gbase + 1) & 0xFF) << 8)
                | (((0x40 + gbase + 2) & 0xFF) << 16)
                | (((0x40 + gbase + 3) & 0xFF) << 24);
            assert_eq!(
                v[14], want_b8_dword,
                "{label} lane {l}: ds_write_b8_d16_hi concurrent sibling bytes"
            );

            // 2 neighboring lanes' halfwords combined into one dword
            let pbase = l32 & !1;
            let want_b16_dword = (0x9000 + pbase) | ((0x9000 + pbase + 1) << 16);
            assert_eq!(
                v[15], want_b16_dword,
                "{label} lane {l}: ds_write_b16_d16_hi concurrent sibling halfwords"
            );

            let eb = EDGE_BYTES[l % 4];
            let b0 = eb[0] as u32;
            let b1 = eb[1] as u32;
            let b2_sext = (eb[2] as i8 as i16 as u16) as u32;
            let b3_sext = (eb[3] as i8 as i16 as u16) as u32;
            let hw0 = u16::from_le_bytes([eb[0], eb[1]]) as u32;
            let hw1 = u16::from_le_bytes([eb[2], eb[3]]) as u32;

            assert_eq!(v[16], 0xA1A1_0000 | b0, "{label} lane {l}: ds_read_u8_d16");
            assert_eq!(
                v[17],
                (b1 << 16) | 0x2222,
                "{label} lane {l}: ds_read_u8_d16_hi"
            );
            assert_eq!(
                v[18],
                0xC3C3_0000 | b2_sext,
                "{label} lane {l}: ds_read_i8_d16"
            );
            assert_eq!(
                v[19],
                (b3_sext << 16) | 0x4444,
                "{label} lane {l}: ds_read_i8_d16_hi"
            );
            assert_eq!(
                v[20],
                0xE5E5_0000 | hw0,
                "{label} lane {l}: ds_read_u16_d16"
            );
            assert_eq!(
                v[21],
                (hw1 << 16) | 0x6666,
                "{label} lane {l}: ds_read_u16_d16_hi"
            );

            // VDST == ADDR overlap: upper half of v22 stays untouched from the address value
            let overlap_hi = (0x01FC + 4 * l32) & 0xFFFF_0000;
            assert_eq!(
                v[22],
                overlap_hi | b3_sext,
                "{label} lane {l}: ds_read_i8_d16 VDST==ADDR"
            );

            let active = is_active(l, mask_lo, mask_hi, &opt);
            let want_v23 = if active {
                (hw0 << 16) | 0x2468
            } else {
                0x1357_2468
            };
            assert_eq!(
                v[23], want_v23,
                "{label} lane {l}: masked ds_read_u16_d16_hi"
            );

            let want_lds_masked = if active {
                let b = (0x40 + l32) & 0xFF;
                let hw = 0x9000 + l32;
                (0xDEAD_BE00 | b) & 0x0000_FFFF | (hw << 16)
            } else {
                0xDEAD_BEEF
            };
            assert_eq!(
                v[24], want_lds_masked,
                "{label} lane {l}: masked ds_write_b8_d16_hi + b16_d16_hi"
            );
        }
    }
}
