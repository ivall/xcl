use std::collections::BTreeSet;

use xcl::gpu::rdna2::{self, Decoded, encode};
use xcl::gpu::recomp::upper_lanes::{self, Read, analyze, slot};

const V: u64 = 256;
type Inst = (&'static str, &'static str, &'static [(&'static str, u64)], Option<u32>);

const WWM: Inst = ("S_OR_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 193)], None);
const RESTORE: Inst = ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 30)], None);

fn program(lines: &[Inst]) -> Vec<Decoded> {
    let mut bytes: Vec<u8> = lines
        .iter()
        .flat_map(|(n, e, f, l)| encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n} ({e}): {x}")))
        .collect();
    bytes.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    decode_all(&bytes)
}

fn decode_all(bytes: &[u8]) -> Vec<Decoded> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off < bytes.len() {
        let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
        off += d.len;
        out.push(d);
    }
    rdna2::resolve_far_jumps(&mut out);
    out
}

fn reads(lines: &[Inst]) -> Vec<Read> {
    let p = program(lines);
    let a = analyze(&p);
    p.iter().filter_map(|d| a.reads.get(&d.pc).copied()).collect()
}

#[test]
fn slot_extracts_constant_lanes_and_rejects_sgpr_or_other_insts() {
    let p = program(&[
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 7), ("SRC1", 128 + 37)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 7), ("SRC1", 255)], Some(100)),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 7), ("SRC1", 193)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 7), ("SRC1", 125)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 7), ("SRC1", 4)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", 12), ("SRC1", 128 + 5)], None),
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", 20), ("SRC1", 128 + 40)], None),
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", 20), ("SRC1", 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 128)], None),
        ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 1), ("ADDR", 2), ("DATA0", 9)], None),
    ]);
    assert_eq!(slot(&p[0]), Some((7, 37)));
    assert_eq!(slot(&p[1]), Some((7, 36))); // 100 & 63
    assert_eq!(slot(&p[2]), Some((7, 63))); // -1 & 63
    assert_eq!(slot(&p[3]), Some((7, 0))); // NULL -> 0
    assert_eq!(slot(&p[4]), None); // SGPR lane
    assert_eq!(slot(&p[5]), None); // non-VGPR SRC0
    assert_eq!(slot(&p[6]), Some((9, 40)));
    assert_eq!(slot(&p[7]), None); // SGPR lane
    assert_eq!(slot(&p[8]), None); // V_MOV_B32
    assert_eq!(slot(&p[9]), None); // DS_BPERMUTE_B32
    assert_eq!(slot(&p[10]), None); // S_ENDPGM
}

#[test]
fn hazards_and_resolved_classify_all_read_kinds_and_ds_permute() {
    // Clean program producing Native, Const, and Shadow resolutions (zero hazards)
    let clean = program(&[
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 4), ("SRC0", 20), ("SRC1", 128 + 35)], None),
        WWM,
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 128 + 11)], None),
        RESTORE,
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 4), ("SRC1", 128 + 35)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 5), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 6), ("SRC1", 128 + 33)], None),
    ]);
    let a = analyze(&clean);
    assert!(a.hazards(&clean).is_empty(), "unexpected hazards: {:?}", a.hazards(&clean));
    assert_eq!(
        a.resolved(),
        BTreeSet::from([
            "lanes ≥ 32 never written (or inactive)",
            "lane ≥ 32 holds a constant (whole-wave identity)",
            "spill slot from its v_writelane shadow",
        ])
    );
    assert_eq!(a.shadows, BTreeSet::from([(4, 35)]));

    // Program hitting all three hazard buckets
    let hazardous = program(&[
        WWM,
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", 20)], None),
        RESTORE,
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 8), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 8), ("SRC1", 4)], None),
        ("DS_PERMUTE_B32", "ENC_DS", &[("VDST", 1), ("ADDR", 2), ("DATA0", 3)], None),
    ]);
    let h = analyze(&hazardous);
    assert_eq!(
        h.hazards(&hazardous),
        BTreeSet::from([
            "v_readlane: lane ≥ 32",
            "v_readlane: lane number from an SGPR",
            "ds_(b)permute: arbitrary lane",
        ])
    );
    assert!(h.resolved().is_empty());

    // DS_BPERMUTE_B32 vs DS_PERMUTE_B32 under exec_hi == 0 and under whole-wave constant
    assert_eq!(
        reads(&[
            ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 1), ("ADDR", 2), ("DATA0", 5)], None),
            ("DS_PERMUTE_B32", "ENC_DS", &[("VDST", 1), ("ADDR", 2), ("DATA0", 5)], None),
        ]),
        vec![Read::Native, Read::Unknown]
    );
    assert_eq!(
        reads(&[
            WWM,
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 128 + 42)], None),
            ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 1), ("ADDR", 2), ("DATA0", 5)], None),
            ("DS_PERMUTE_B32", "ENC_DS", &[("VDST", 1), ("ADDR", 2), ("DATA0", 5)], None),
            RESTORE,
        ]),
        vec![Read::Const(42), Read::Unknown]
    );
}

#[test]
fn cfg_indirect_jumps_vs_fetch_swappc_and_resolved_far_jump() {
    // Indirect S_SETPC_B64 triggers conservative fallback: all upper-reaching reads become Unknown,
    // low-lane reads are not added, and shadows remain empty.
    let indirect_setpc = program(&[
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 4), ("SRC0", 20), ("SRC1", 128 + 5)], None),
        ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 4), ("SRC1", 128 + 5)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 4), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 4), ("SRC1", 4)], None),
        ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 1), ("ADDR", 2), ("DATA0", 4)], None),
    ]);
    let a = analyze(&indirect_setpc);
    assert!(a.shadows.is_empty());
    assert_eq!(a.reads.get(&indirect_setpc[2].pc), None);
    assert_eq!(a.reads.get(&indirect_setpc[3].pc), Some(&Read::Unknown));
    assert_eq!(a.reads.get(&indirect_setpc[4].pc), Some(&Read::Unknown));
    assert_eq!(a.reads.get(&indirect_setpc[5].pc), Some(&Read::Unknown));

    // Non-fetch S_SWAPPC_B64 (SDST != SSRC0) also falls back conservatively
    let non_fetch = program(&[
        ("S_SWAPPC_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 4)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)], None),
    ]);
    assert_eq!(analyze(&non_fetch).reads.get(&non_fetch[1].pc), Some(&Read::Unknown));

    // Fetch-shader S_SWAPPC_B64 (SDST == SSRC0) falls through cleanly
    let fetch = program(&[
        ("S_SWAPPC_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 4)], None),
        WWM,
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 128 + 9)], None),
        RESTORE,
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)], None),
    ]);
    let fa = analyze(&fetch);
    assert_eq!(fa.reads.get(&fetch[4].pc), Some(&Read::Const(9)));
    assert!(fa.hazards(&fetch).is_empty());

    // Resolved far jump (s_getpc_b64 + s_add_u32 + s_addc_u32 + s_setpc_b64) skips an unknown write:
    // PCs: 0x00 GETPC(4), 0x04 ADD(8 with lit), 0x0C ADDC(4), 0x10 SETPC(4),
    //      0x14 WWM(4), 0x18 V_MOV unknown(4), 0x1C RESTORE(4), 0x20 V_READLANE(8)
    // GETPC yields 0x04; 0x04 + 0x1C = 0x20.
    let far = program(&[
        ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 4)], None),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 255)], Some(0x1C)),
        ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 5), ("SSRC1", 128)], None),
        ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 4)], None),
        WWM,
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 20)], None),
        RESTORE,
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)], None),
    ]);
    assert_eq!(far[3].far_target, Some(far[7].pc));
    let far_res = analyze(&far);
    assert_eq!(far_res.reads.get(&far[7].pc), Some(&Read::Native));
    assert!(far_res.hazards(&far).is_empty());
}

#[test]
fn binary_valu_ops_absorbing_constants_vop3_modifiers_and_dpp16_masks() {
    // Binary ops with constant operands (signed, unsigned, bitwise, wrapping add)
    let bin = reads(&[
        WWM,
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 194)], None), // -2
        ("V_MIN_I32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 128 + 5), ("VSRC1", 1)], None),
        ("V_MAX_I32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 128 + 5), ("VSRC1", 1)], None),
        ("V_MAX_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 128 + 5), ("VSRC1", 1)], None), // -2u > 5u
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 128 + 16), ("VSRC1", 2)], None), // 16 | -2 = -2
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 128 + 5), ("VSRC1", 1)], None), // -2 + 5 = 3
        RESTORE,
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 2), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 3), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 4), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 5), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)], None),
    ]);
    assert_eq!(
        bin,
        vec![
            Read::Const((-2i32) as u32),
            Read::Const(5),
            Read::Const((-2i32) as u32),
            Read::Const((-2i32) as u32),
            Read::Const(3),
        ]
    );

    // Absorbing constants decide unwritten operands; non-absorbing keeps Unwritten (Native)
    let abs_const = reads(&[
        WWM,
        ("V_MIN_I32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 255), ("VSRC1", 9)], Some(i32::MIN as u32)),
        ("V_MAX_I32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 9)], Some(i32::MAX as u32)),
        ("V_MAX_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 193), ("VSRC1", 9)], None),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 193), ("VSRC1", 9)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 128 + 1), ("VSRC1", 9)], None),
        RESTORE,
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 2), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 3), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 4), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 5), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)], None),
    ]);
    assert_eq!(
        abs_const,
        vec![
            Read::Const(i32::MIN as u32),
            Read::Const(i32::MAX as u32),
            Read::Const(u32::MAX),
            Read::Const(u32::MAX),
            Read::Native,
        ]
    );

    // VOP3 modifiers (CLAMP / OMOD / ABS / NEG) bail out of evaluate -> Unknown
    let vop3_mods = reads(&[
        WWM,
        ("V_ADD_NC_U32", "ENC_VOP3", &[("VDST", 2), ("SRC0", 128 + 3), ("SRC1", 128 + 4)], None),
        ("V_ADD_NC_U32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 128 + 3), ("SRC1", 128 + 4), ("CLAMP", 1)], None),
        ("V_ADD_NC_U32", "ENC_VOP3", &[("VDST", 4), ("SRC0", 128 + 3), ("SRC1", 128 + 4), ("OMOD", 1)], None),
        ("V_ADD_NC_U32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 128 + 3), ("SRC1", 128 + 4), ("ABS", 1)], None),
        ("V_ADD_NC_U32", "ENC_VOP3", &[("VDST", 6), ("SRC0", 128 + 3), ("SRC1", 128 + 4), ("NEG", 1)], None),
        RESTORE,
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 2), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 3), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 4), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 5), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)], None),
    ]);
    assert_eq!(
        vop3_mods,
        vec![Read::Const(7), Read::Unknown, Read::Unknown, Read::Unknown, Read::Unknown]
    );

    // DPP16 partial ROW_MASK (row 2 only: 0b0100) and BANK_MASK (bank 0 only: 0b0001):
    // lane 32 (row 2, bank 0) updates to 1 + 1 = 2; lane 36 (row 2, bank 1) and lane 48 (row 3, bank 0) keep 1.
    let dpp = reads(&[
        WWM,
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 128 + 1)], None),
        (
            "V_ADD_NC_U32",
            "VOP2_VOP_DPP16",
            &[
                ("VDST", 6),
                ("SRC0", 0xFA),
                ("VSRC0", 6),
                ("VSRC1", 6),
                ("DPP_CTRL", 0x140),
                ("ROW_MASK", 0b0100),
                ("BANK_MASK", 0b0001),
            ],
            None,
        ),
        RESTORE,
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 36)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 48)], None),
    ]);
    assert_eq!(dpp, vec![Read::Const(2), Read::Const(1), Read::Const(1)]);
}

#[test]
fn movreld_sgpr_writelane_mimg_dmask_and_unreachable_code() {
    // S_MOVRELD_B32 preserves exec_hi (0) while clobbering SGPRs (s[10:11] becomes Any).
    let movreld = reads(&[
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None),
        ("S_MOVRELD_B32", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 20)], None),
        // exec_hi is still 0, so this V_MOV_B32 leaves v4's upper lanes unwritten
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 20)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 4), ("SRC1", 128 + 32)], None),
        // Under WWM, s[10:11] is now Any, so V_CNDMASK_B32 joins SRC0(const) and SRC1(unknown) -> Unknown
        WWM,
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 20)], None),
        ("V_CNDMASK_B32", "ENC_VOP3", &[("VDST", 6), ("SRC0", 128 + 7), ("SRC1", V + 5), ("SRC2", 10)], None),
        RESTORE,
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 6), ("SRC1", 128 + 32)], None),
    ]);
    assert_eq!(movreld, vec![Read::Native, Read::Unknown]);

    // SGPR-lane V_WRITELANE_B32 invalidates both lo shadows and all upper lanes of that VGPR
    let wl_sgpr = program(&[
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", 20), ("SRC1", 128 + 5)], None),
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", 21), ("SRC1", 128 + 36)], None),
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", 22), ("SRC1", 4)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 9), ("SRC1", 128 + 5)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 9), ("SRC1", 128 + 36)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 9), ("SRC1", 128 + 37)], None),
    ]);
    let wla = analyze(&wl_sgpr);
    assert!(wla.shadows.is_empty());
    assert_eq!(wla.reads.get(&wl_sgpr[3].pc), None); // lo shadow cleared
    assert_eq!(wla.reads.get(&wl_sgpr[4].pc), Some(&Read::Unknown));
    assert_eq!(wla.reads.get(&wl_sgpr[5].pc), Some(&Read::Unknown));

    // MIMG DMASK truncates written VGPR count (2 channels -> v14, v15 only; v16 shadow survives;
    // D16 packs 2 channels into 1 VGPR -> v20 only; v21 shadow survives; TFE adds 1 -> v25 clobbered).
    let mimg = program(&[
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 14), ("SRC0", 20), ("SRC1", 128 + 5)], None),
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 15), ("SRC0", 20), ("SRC1", 128 + 5)], None),
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 16), ("SRC0", 20), ("SRC1", 128 + 5)], None),
        ("IMAGE_LOAD", "ENC_MIMG", &[("VDATA", 14), ("VADDR", 3), ("SRSRC", 4), ("DMASK", 0x5), ("DIM", 1), ("UNORM", 1)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 14), ("SRC1", 128 + 5)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 15), ("SRC1", 128 + 5)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 16), ("SRC1", 128 + 5)], None),
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 20), ("SRC1", 128 + 5)], None),
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 21), ("SRC0", 20), ("SRC1", 128 + 5)], None),
        ("IMAGE_LOAD", "ENC_MIMG", &[("VDATA", 20), ("VADDR", 3), ("SRSRC", 4), ("DMASK", 0x3), ("DIM", 1), ("D16", 1)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 20), ("SRC1", 128 + 5)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 21), ("SRC1", 128 + 5)], None),
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 24), ("SRC0", 20), ("SRC1", 128 + 5)], None),
        ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 25), ("SRC0", 20), ("SRC1", 128 + 5)], None),
        ("IMAGE_LOAD", "ENC_MIMG", &[("VDATA", 24), ("VADDR", 3), ("SRSRC", 4), ("DMASK", 0x1), ("DIM", 1), ("TFE", 1)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 24), ("SRC1", 128 + 5)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 25), ("SRC1", 128 + 5)], None),
    ]);
    let ma = analyze(&mimg);
    assert_eq!(ma.reads.get(&mimg[4].pc), None);
    assert_eq!(ma.reads.get(&mimg[5].pc), None);
    assert_eq!(ma.reads.get(&mimg[6].pc), Some(&Read::Shadow));
    assert_eq!(ma.reads.get(&mimg[10].pc), None);
    assert_eq!(ma.reads.get(&mimg[11].pc), Some(&Read::Shadow));
    assert_eq!(ma.reads.get(&mimg[15].pc), None);
    assert_eq!(ma.reads.get(&mimg[16].pc), None);
    assert_eq!(ma.shadows, BTreeSet::from([(16, 5), (21, 5)]));

    // Empty input and unreachable code after S_ENDPGM produce no reads or hazards
    let empty = upper_lanes::analyze(&[]);
    assert!(empty.reads.is_empty());
    assert!(empty.shadows.is_empty());
    assert!(empty.hazards(&[]).is_empty());
    assert!(empty.resolved().is_empty());

    let unreachable = program(&[
        ("S_ENDPGM", "ENC_SOPP", &[], None),
        ("DS_PERMUTE_B32", "ENC_DS", &[("VDST", 1), ("ADDR", 2), ("DATA0", 3)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 6), ("SRC1", 128 + 32)], None),
    ]);
    let ua = analyze(&unreachable);
    assert!(ua.reads.is_empty());
    assert!(ua.hazards(&unreachable).is_empty());
}
