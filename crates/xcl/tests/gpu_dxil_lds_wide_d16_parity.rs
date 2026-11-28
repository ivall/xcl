//! GPU differential parity tests between the direct RDNA2 → DXIL translator
//! (`xcl::gpu::dxil::kernel::translate`, validated with `dxil.dll`) and the HLSL + DXC path
//! (`xcl::gpu::recomp`) for `lower_mem.rs` LDS loads and stores: wide 64/96/128-bit accesses,
//! two-address `DS_READ2`/`DS_WRITE2` (b32/b64 and `ST64`, including `VDST == ADDR`),
//! sub-dword and `D16`/`D16_HI` reads & writes, and `DS_WRITE/READ_ADDTID_B32` with `M0` + `EXEC`.

use std::ops::Range;

use xcl::gpu::dxil::kernel::{translate as direct_translate, KernelOptions};
use xcl::gpu::dxil::validate;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner::{self, KernelResult, KernelState};
use xcl::gpu::recomp::{dxc, translate as hlsl_translate, Options};

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
fn direct_dxil_lds_wide_64_96_128_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    // Each lane writes b64 at 0x0000 + 8*l, b96 at 0x0400 + 12*l, b128 at 0x0800 + 16*l,
    // synchronizes with S_BARRIER, then reads back from its partner lane (l ^ 1) while lanes >= 24 are inactive.
    let code = prog(&[
        ("DS_WRITE_B64", "ENC_DS", &[("ADDR", 1), ("DATA0", 4)], None),
        ("DS_WRITE_B96", "ENC_DS", &[("ADDR", 2), ("DATA0", 6), ("OFFSET1", 4)], None), // +0x400
        ("DS_WRITE_B128", "ENC_DS", &[("ADDR", 3), ("DATA0", 10), ("OFFSET1", 8)], None), // +0x800
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 128 + 24), ("VSRC1", 0)], None),
        ("DS_READ_B64", "ENC_DS", &[("VDST", 20), ("ADDR", 14)], None),
        ("DS_READ_B96", "ENC_DS", &[("VDST", 22), ("ADDR", 15), ("OFFSET1", 4)], None),
        ("DS_READ_B128", "ENC_DS", &[("VDST", 25), ("ADDR", 16), ("OFFSET1", 8)], None),
    ]);

    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        let p = l32 ^ 1;
        r[0] = l32;
        r[1] = 8 * l32;
        r[2] = 12 * l32;
        r[3] = 16 * l32;
        r[14] = 8 * p;
        r[15] = 12 * p;
        r[16] = 16 * p;
        r[4] = 0x1000_0000 | l32;
        r[5] = 0x2000_0000 | l32;
        r[6] = 0x3000_0000 | l32;
        r[7] = 0x4000_0000 | l32;
        r[8] = 0x5000_0000 | l32;
        r[10] = 0x6000_0000 | l32;
        r[11] = 0x7000_0000 | l32;
        r[12] = 0x8000_0000 | l32;
        r[13] = 0x9000_0000 | l32;
        for reg in 20..29 {
            r[reg] = 0xDEAD_BEEF;
        }
    }

    for (wave64, split) in modes {
        let r = run_parity("lds_wide_64_96_128", &code, &st, wave64, split, 20..29);
        let lanes = if wave64 { 64 } else { 32 };
        for l in 0..lanes {
            if l >= 24 {
                assert_eq!(&r.vgpr[l][20..29], &[0xDEAD_BEEF; 9], "inactive lane {l}");
                continue;
            }
            let p = (l ^ 1) as u32;
            assert_eq!(&r.vgpr[l][20..22], &[0x1000_0000 | p, 0x2000_0000 | p]);
            assert_eq!(&r.vgpr[l][22..25], &[0x3000_0000 | p, 0x4000_0000 | p, 0x5000_0000 | p]);
            assert_eq!(
                &r.vgpr[l][25..29],
                &[0x6000_0000 | p, 0x7000_0000 | p, 0x8000_0000 | p, 0x9000_0000 | p]
            );
        }
    }
}

#[test]
fn direct_dxil_lds_two_address_and_st64_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    // Exercises DS_WRITE2/READ2_B32 (including VDST==ADDR overlap), DS_WRITE2/READ2_B64,
    // DS_WRITE2ST64/READ2ST64_B32 (256-byte stride), and DS_WRITE2ST64/READ2ST64_B64 (512-byte stride).
    let code = prog(&[
        // b32 paired at v1 = 8*l, offsets 0 & 1 -> read back into v[1:2] (overlaps ADDR v1!)
        ("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 4), ("DATA1", 5), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        ("DS_READ2_B32", "ENC_DS", &[("VDST", 1), ("ADDR", 1), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        // b64 paired at v11 = 0x400 + 16*l, offsets 0 & 1 (8-byte units) -> read into v[20:23]
        ("DS_WRITE2_B64", "ENC_DS", &[("ADDR", 11), ("DATA0", 6), ("DATA1", 8), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        ("DS_READ2_B64", "ENC_DS", &[("VDST", 20), ("ADDR", 11), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        // st64 b32 at v3 = 0x1000 + 4*l, offsets 0 & 1 (256-byte units) -> read into v[24:25]
        ("DS_WRITE2ST64_B32", "ENC_DS", &[("ADDR", 3), ("DATA0", 4), ("DATA1", 5), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        ("DS_READ2ST64_B32", "ENC_DS", &[("VDST", 24), ("ADDR", 3), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        // st64 b64 at v10 = 0x2000 + 8*l, offsets 0 & 1 (512-byte units) -> read into v[26:29]
        ("DS_WRITE2ST64_B64", "ENC_DS", &[("ADDR", 10), ("DATA0", 6), ("DATA1", 8), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        ("DS_READ2ST64_B64", "ENC_DS", &[("VDST", 26), ("ADDR", 10), ("OFFSET0", 0), ("OFFSET1", 1)], None),
    ]);

    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[1] = 8 * l32;
        r[3] = 0x1000 + 4 * l32;
        r[10] = 0x2000 + 8 * l32;
        r[11] = 0x400 + 16 * l32;
        r[4] = 0xA000_0000 | l32;
        r[5] = 0xB000_0000 | l32;
        r[6] = 0xC000_0000 | l32;
        r[7] = 0xC100_0000 | l32;
        r[8] = 0xD000_0000 | l32;
        r[9] = 0xD100_0000 | l32;
    }

    for (wave64, split) in modes {
        let r = run_parity("lds_two_address_st64", &code, &st, wave64, split, 1..30);
        let lanes = if wave64 { 64 } else { 32 };
        for l in 0..lanes {
            let l32 = l as u32;
            assert_eq!((r.vgpr[l][1], r.vgpr[l][2]), (0xA000_0000 | l32, 0xB000_0000 | l32), "lane {l} read2_b32 overlap");
            assert_eq!(
                &r.vgpr[l][20..24],
                &[0xC000_0000 | l32, 0xC100_0000 | l32, 0xD000_0000 | l32, 0xD100_0000 | l32]
            );
            assert_eq!((r.vgpr[l][24], r.vgpr[l][25]), (0xA000_0000 | l32, 0xB000_0000 | l32));
            assert_eq!(
                &r.vgpr[l][26..30],
                &[0xC000_0000 | l32, 0xC100_0000 | l32, 0xD000_0000 | l32, 0xD100_0000 | l32]
            );
        }
    }
}

#[test]
fn direct_dxil_lds_subdword_and_d16_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    // Neighbouring lanes share dwords:
    //  - bytes written at 0x000 + l (DS_WRITE_B8) and 0x100 + l (DS_WRITE_B8_D16_HI)
    //  - shorts written at 0x200 + 2*l (DS_WRITE_B16) and 0x300 + 2*l (DS_WRITE_B16_D16_HI)
    // After S_BARRIER, each lane reads back u8/i8/u16/i16 and all 6 D16 / D16_HI forms into seeded VGPRs.
    let code = prog(&[
        ("DS_WRITE_B8", "ENC_DS", &[("ADDR", 0), ("DATA0", 2)], None),
        ("DS_WRITE_B8_D16_HI", "ENC_DS", &[("ADDR", 0), ("DATA0", 3), ("OFFSET1", 1)], None), // +0x100
        ("DS_WRITE_B16", "ENC_DS", &[("ADDR", 1), ("DATA0", 4), ("OFFSET1", 2)], None),       // +0x200
        ("DS_WRITE_B16_D16_HI", "ENC_DS", &[("ADDR", 1), ("DATA0", 5), ("OFFSET1", 3)], None), // +0x300
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ_U8", "ENC_DS", &[("VDST", 10), ("ADDR", 0)], None),
        ("DS_READ_I8", "ENC_DS", &[("VDST", 11), ("ADDR", 0)], None),
        ("DS_READ_U8", "ENC_DS", &[("VDST", 12), ("ADDR", 0), ("OFFSET1", 1)], None),
        ("DS_READ_U16", "ENC_DS", &[("VDST", 13), ("ADDR", 1), ("OFFSET1", 2)], None),
        ("DS_READ_I16", "ENC_DS", &[("VDST", 14), ("ADDR", 1), ("OFFSET1", 2)], None),
        ("DS_READ_U16", "ENC_DS", &[("VDST", 15), ("ADDR", 1), ("OFFSET1", 3)], None),
        // D16 / D16_HI reads must preserve the untouched 16-bit half of VDST (seeded 0x5A5A_1234)
        ("DS_READ_U8_D16", "ENC_DS", &[("VDST", 16), ("ADDR", 0)], None),
        ("DS_READ_U8_D16_HI", "ENC_DS", &[("VDST", 17), ("ADDR", 0)], None),
        ("DS_READ_I8_D16", "ENC_DS", &[("VDST", 18), ("ADDR", 0)], None),
        ("DS_READ_I8_D16_HI", "ENC_DS", &[("VDST", 19), ("ADDR", 0)], None),
        ("DS_READ_U16_D16", "ENC_DS", &[("VDST", 20), ("ADDR", 1), ("OFFSET1", 2)], None),
        ("DS_READ_U16_D16_HI", "ENC_DS", &[("VDST", 21), ("ADDR", 1), ("OFFSET1", 3)], None),
    ]);

    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        let b = (0x78 + l32) & 0xFF;     // crosses 0x80 at lane 8
        let b_hi = (0xC0 + l32) & 0xFF;
        let s = (0x7FF8 + l32) & 0xFFFF; // crosses 0x8000 at lane 8
        let s_hi = (0xA000 + l32) & 0xFFFF;
        r[0] = l32;
        r[1] = 2 * l32;
        r[2] = 0xDEAD_0000 | b;
        r[3] = (b_hi << 16) | 0xBEEF;
        r[4] = 0xCAFE_0000 | s;
        r[5] = (s_hi << 16) | 0x1111;
        for reg in 16..22 {
            r[reg] = 0x5A5A_1234;
        }
    }

    for (wave64, split) in modes {
        let r = run_parity("lds_subdword_d16", &code, &st, wave64, split, 10..22);
        let lanes = if wave64 { 64 } else { 32 };
        for l in 0..lanes {
            let l32 = l as u32;
            let b = (0x78 + l32) & 0xFF;
            let b_hi = (0xC0 + l32) & 0xFF;
            let s = (0x7FF8 + l32) & 0xFFFF;
            let s_hi = (0xA000 + l32) & 0xFFFF;
            let ib16 = ((b as i8) as i16) as u16 as u32;
            assert_eq!(r.vgpr[l][10], b, "lane {l} ds_read_u8");
            assert_eq!(r.vgpr[l][11], (b as i8) as i32 as u32, "lane {l} ds_read_i8");
            assert_eq!(r.vgpr[l][12], b_hi, "lane {l} ds_write_b8_d16_hi");
            assert_eq!(r.vgpr[l][13], s, "lane {l} ds_read_u16");
            assert_eq!(r.vgpr[l][14], (s as i16) as i32 as u32, "lane {l} ds_read_i16");
            assert_eq!(r.vgpr[l][15], s_hi, "lane {l} ds_write_b16_d16_hi");
            assert_eq!(r.vgpr[l][16], 0x5A5A_0000 | b, "lane {l} ds_read_u8_d16");
            assert_eq!(r.vgpr[l][17], (b << 16) | 0x1234, "lane {l} ds_read_u8_d16_hi");
            assert_eq!(r.vgpr[l][18], 0x5A5A_0000 | ib16, "lane {l} ds_read_i8_d16");
            assert_eq!(r.vgpr[l][19], (ib16 << 16) | 0x1234, "lane {l} ds_read_i8_d16_hi");
            assert_eq!(r.vgpr[l][20], 0x5A5A_0000 | s, "lane {l} ds_read_u16_d16");
            assert_eq!(r.vgpr[l][21], (s_hi << 16) | 0x1234, "lane {l} ds_read_u16_d16_hi");
        }
    }
}

#[test]
fn direct_dxil_lds_addtid_and_exec_mask_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    // M0 = 0xDEAD_0100 -> M0[15:0] = 0x100, offset = 0x0240 -> base = 0x340 + 4 * lane
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some(0xDEAD_0100)),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 2)], None),
        ("DS_READ_ADDTID_B32", "ENC_DS", &[("VDST", 10), ("OFFSET0", 0x40), ("OFFSET1", 0x02)], None),
        ("DS_WRITE_ADDTID_B32", "ENC_DS", &[("DATA0", 3), ("OFFSET0", 0x40), ("OFFSET1", 0x02)], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 11), ("ADDR", 1)], None),
        // Mask EXEC to even lanes (v0 & 1 == 0), overwrite LDS with v4 and read into v12
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 129), ("VSRC1", 0)], None),
        ("V_CMPX_EQ_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 9)], None),
        ("DS_WRITE_ADDTID_B32", "ENC_DS", &[("DATA0", 4), ("OFFSET0", 0x40), ("OFFSET1", 0x02)], None),
        ("DS_READ_ADDTID_B32", "ENC_DS", &[("VDST", 12), ("OFFSET0", 0x40), ("OFFSET1", 0x02)], None),
    ]);

    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[0] = l32;
        r[1] = 0x340 + 4 * l32;
        r[2] = 0xBEEF_0000 | l32;
        r[3] = 0x1000_0000 | l32;
        r[4] = 0x2000_0000 | l32;
        r[12] = 0xCAFE_0000 | l32;
    }

    for (wave64, split) in modes {
        let r = run_parity("lds_addtid_exec", &code, &st, wave64, split, 10..13);
        let lanes = if wave64 { 64 } else { 32 };
        for l in 0..lanes {
            let l32 = l as u32;
            assert_eq!(r.vgpr[l][10], 0xBEEF_0000 | l32, "lane {l} explicit -> addtid");
            assert_eq!(r.vgpr[l][11], 0x1000_0000 | l32, "lane {l} addtid -> explicit");
            let want12 = if (l & 1) == 0 { 0x2000_0000 | l32 } else { 0xCAFE_0000 | l32 };
            assert_eq!(r.vgpr[l][12], want12, "lane {l} masked addtid");
        }
    }
}
