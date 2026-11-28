use xcl::gpu::rdna2::{self, Decoded};
use xcl::gpu::recomp::mesh_wave::{
    self, BranchMask, LaneRead, LaneReadKind, LaneValueProvenance, MaskBranch,
};
use xcl::gpu::recomp::{dxc, translate, x64_masks, CsInterface, Mode, Options};

const V: u64 = 256;

fn one(name: &str, enc: &str, fields: &[(&str, u64)], lit: Option<u32>) -> Vec<u8> {
    rdna2::encode(name, enc, fields, lit).unwrap_or_else(|e| panic!("{name} ({enc}): {e}"))
}

fn prog(insts: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    let mut code = Vec::new();
    for &(name, enc, fields, lit) in insts {
        code.extend(one(name, enc, fields, lit));
    }
    code.extend(one("S_ENDPGM", "ENC_SOPP", &[], None));
    code
}

fn decode_all(code: &[u8]) -> Vec<Decoded> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off < code.len() {
        let d = rdna2::decode(&code[off..], off as u64).unwrap();
        off += d.len;
        out.push(d);
    }
    out
}

fn split_cs_options() -> Options {
    Options {
        mode: Mode::Compute(CsInterface {
            threads: [64, 1, 1],
            user_sgprs: 4,
            tid_components: 1,
            ..Default::default()
        }),
        wave64: true,
        split_wave64: true,
        ..Default::default()
    }
}

fn compile_if_dxc_available(hlsl: &str) {
    if dxc::location().is_ok() {
        dxc::compile(hlsl, "main", "cs_6_6", &[])
            .unwrap_or_else(|e| panic!("DXC failed on emitted HLSL: {e}\n{hlsl}"));
    }
}

#[test]
fn split_wave64_sub_borrow_chains_and_null_operands_stay_half_local() {
    // 64-bit subtraction via V_SUB_CO_U32 -> V_SUB_CO_CI_U32 (VCC) and
    // V_SUBREV_CO_U32 -> V_SUBREV_CO_CI_U32 (s[20:21]), plus a scalar NULL (125) read.
    let code = prog(&[
        (
            "V_SUB_CO_U32",
            "VOP3_SDST_ENC",
            &[("VDST", 2), ("SDST", 106), ("SRC0", V), ("SRC1", 129)],
            None,
        ),
        (
            "V_SUB_CO_CI_U32",
            "ENC_VOP2",
            &[("VDST", 3), ("SRC0", V + 1), ("VSRC1", 0)],
            None,
        ),
        (
            "V_SUBREV_CO_U32",
            "VOP3_SDST_ENC",
            &[("VDST", 4), ("SDST", 20), ("SRC0", 130), ("SRC1", V + 2)],
            None,
        ),
        (
            "V_SUBREV_CO_CI_U32",
            "VOP3_SDST_ENC",
            &[
                ("VDST", 5),
                ("SDST", 22),
                ("SRC0", V + 3),
                ("SRC1", V + 4),
                ("SRC2", 20),
            ],
            None,
        ),
        (
            "V_ADD_NC_U32",
            "ENC_VOP2",
            &[("VDST", 6), ("SRC0", 125), ("VSRC1", 5)],
            None,
        ),
        (
            "BUFFER_STORE_DWORD",
            "ENC_MUBUF",
            &[("VDATA", 6), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)],
            None,
        ),
    ]);

    let decoded = decode_all(&code);
    let live_by_pc = x64_masks::by_pc(&decoded, false);
    let vcc_bits = 3u128 << 106;
    let s20_bits = 3u128 << 20;
    assert_eq!(
        live_by_pc[&decoded[0].pc] & vcc_bits,
        0,
        "V_SUB_CO_CI_U32 reads VCC per lane only"
    );
    assert_eq!(
        live_by_pc[&decoded[2].pc] & s20_bits,
        0,
        "V_SUBREV_CO_CI_U32 reads s[20:21] per lane only"
    );
    assert_eq!(
        live_by_pc[&decoded[3].pc] & (1u128 << 125),
        0,
        "NULL (125) is not an SGPR"
    );

    let t = translate(&code, &split_cs_options()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    assert_eq!(
        t.hlsl.matches("= xcl_x_ballot(and(").count(),
        0,
        "borrow masks must not trigger cross-half exchanges:\n{}",
        t.hlsl
    );
    assert!(
        t.hlsl.contains("= xcl_x_ballot_local(and("),
        "expected half-local ballot emission:\n{}",
        t.hlsl
    );
    compile_if_dxc_available(&t.hlsl);
}

#[test]
fn split_wave64_src_vccz_src_execz_and_swappc_whole_mask_liveness() {
    // src_execz (252) reads whole EXEC, leaving an otherwise per-lane VCC compare half-local.
    let execz_only = prog(&[
        (
            "V_CMP_GT_U32",
            "ENC_VOPC",
            &[("SRC0", V), ("VSRC1", 0)],
            None,
        ),
        (
            "V_CNDMASK_B32",
            "ENC_VOP2",
            &[("VDST", 1), ("SRC0", 128), ("VSRC1", 0)],
            None,
        ),
        (
            "V_ADD_NC_U32",
            "ENC_VOP2",
            &[("VDST", 2), ("SRC0", 252), ("VSRC1", 1)],
            None,
        ),
    ]);
    let dec_execz = decode_all(&execz_only);
    assert_eq!(x64_masks::by_pc(&dec_execz, false)[&0] & (3u128 << 106), 0);
    let t_execz = translate(&execz_only, &split_cs_options()).unwrap();
    assert!(t_execz.complete(), "{:?}", t_execz.unsupported_why);
    assert_eq!(t_execz.hlsl.matches("= xcl_x_ballot(and(").count(), 0);
    assert_eq!(t_execz.hlsl.matches("= xcl_x_ballot_local(and(").count(), 1);
    compile_if_dxc_available(&t_execz.hlsl);

    // src_vccz (251) depends on all 64 bits of VCC: the preceding V_CMP must exchange halves.
    let vccz_read = prog(&[
        (
            "V_CMP_GT_U32",
            "ENC_VOPC",
            &[("SRC0", V), ("VSRC1", 0)],
            None,
        ),
        (
            "V_ADD_NC_U32",
            "ENC_VOP2",
            &[("VDST", 2), ("SRC0", 251), ("VSRC1", 0)],
            None,
        ),
    ]);
    let dec_vccz = decode_all(&vccz_read);
    assert_eq!(
        x64_masks::by_pc(&dec_vccz, false)[&0] & (3u128 << 106),
        3u128 << 106
    );
    let t_vccz = translate(&vccz_read, &split_cs_options()).unwrap();
    assert!(t_vccz.complete(), "{:?}", t_vccz.unsupported_why);
    assert_eq!(t_vccz.hlsl.matches("= xcl_x_ballot(and(").count(), 1);
    assert_eq!(t_vccz.hlsl.matches("= xcl_x_ballot_local(and(").count(), 0);
    compile_if_dxc_available(&t_vccz.hlsl);

    // An unresolved S_SWAPPC_B64 has unknown target: every SGPR not overwritten by SDST (s[30:31])
    // is conservatively whole-live at pc 0, including s[20:21].
    let swappc = prog(&[
        (
            "V_CMP_LT_U32",
            "ENC_VOP3",
            &[("VDST", 20), ("SRC0", V), ("SRC1", 129)],
            None,
        ),
        (
            "S_SWAPPC_B64",
            "ENC_SOP1",
            &[("SDST", 30), ("SSRC0", 2)],
            None,
        ),
    ]);
    let dec_swappc = decode_all(&swappc);
    assert_eq!(x64_masks::by_pc(&dec_swappc, false)[&0], !(3u128 << 30));
}

#[test]
fn mesh_wave_lane_provenance_and_write_lane_sources() {
    let mut code = Vec::new();
    // 0: unread v9 -> Entry
    code.extend(one(
        "V_READFIRSTLANE_B32",
        "ENC_VOP1",
        &[("VDST", 0), ("SRC0", V + 9)],
        None,
    ));
    // 1: SGPR-sourced V_WRITELANE_B32 (SRC0 = s4, SRC1 = s0 -> VDST = v10): no VGPR read -> None / Entry,
    //    and v10's writer becomes V_WRITELANE_B32 (Arithmetic).
    code.extend(one(
        "V_WRITELANE_B32",
        "ENC_VOP3",
        &[("VDST", 10), ("SRC0", 4), ("SRC1", 0)],
        None,
    ));
    // 2: read v10 written by V_WRITELANE_B32 -> Arithmetic
    code.extend(one(
        "V_READLANE_B32",
        "ENC_VOP3",
        &[("VDST", 1), ("SRC0", V + 10), ("SRC1", 0)],
        None,
    ));
    // 3: BUFFER_LOAD_DWORD -> v2 -> Memory
    code.extend(one(
        "BUFFER_LOAD_DWORD",
        "ENC_MUBUF",
        &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 0), ("SOFFSET", 128)],
        None,
    ));
    code.extend(one(
        "V_READFIRSTLANE_B32",
        "ENC_VOP1",
        &[("VDST", 2), ("SRC0", V + 2)],
        None,
    ));
    // 4: V_CNDMASK_B32 -> v3 -> MaskSelect
    code.extend(one(
        "V_CNDMASK_B32",
        "ENC_VOP2",
        &[("VDST", 3), ("SRC0", 128), ("VSRC1", 2)],
        None,
    ));
    code.extend(one(
        "V_READFIRSTLANE_B32",
        "ENC_VOP1",
        &[("VDST", 3), ("SRC0", V + 3)],
        None,
    ));
    // 5: V_MOV_B32 -> v4 -> Move
    code.extend(one(
        "V_MOV_B32",
        "ENC_VOP1",
        &[("VDST", 4), ("SRC0", 129)],
        None,
    ));
    code.extend(one(
        "V_READFIRSTLANE_B32",
        "ENC_VOP1",
        &[("VDST", 4), ("SRC0", V + 4)],
        None,
    ));
    // 6: V_MBCNT_LO_U32_B32 -> v5 -> PrefixCount
    code.extend(one(
        "V_MBCNT_LO_U32_B32",
        "ENC_VOP3",
        &[("VDST", 5), ("SRC0", 193), ("SRC1", 128)],
        None,
    ));
    code.extend(one(
        "V_READFIRSTLANE_B32",
        "ENC_VOP1",
        &[("VDST", 5), ("SRC0", V + 5)],
        None,
    ));
    // 7: V_ADD_NC_U32 -> v6, read by V_READLANE_B32 -> Arithmetic
    code.extend(one(
        "V_ADD_NC_U32",
        "ENC_VOP2",
        &[("VDST", 6), ("SRC0", 130), ("VSRC1", 4)],
        None,
    ));
    code.extend(one(
        "V_READLANE_B32",
        "ENC_VOP3",
        &[("VDST", 7), ("SRC0", V + 6), ("SRC1", 0)],
        None,
    ));
    code.extend(one("S_ENDPGM", "ENC_SOPP", &[], None));

    let report = mesh_wave::analyze(&code);
    let got: Vec<(LaneReadKind, Option<u64>, LaneValueProvenance)> = report
        .lane_reads
        .iter()
        .map(
            |LaneRead {
                 kind,
                 source_vgpr,
                 provenance,
                 ..
             }| (*kind, *source_vgpr, *provenance),
        )
        .collect();
    assert_eq!(
        got,
        vec![
            (LaneReadKind::ReadFirst, Some(9), LaneValueProvenance::Entry),
            (LaneReadKind::WriteLane, None, LaneValueProvenance::Entry),
            (
                LaneReadKind::ReadLane,
                Some(10),
                LaneValueProvenance::Arithmetic
            ),
            (
                LaneReadKind::ReadFirst,
                Some(2),
                LaneValueProvenance::Memory
            ),
            (
                LaneReadKind::ReadFirst,
                Some(3),
                LaneValueProvenance::MaskSelect
            ),
            (LaneReadKind::ReadFirst, Some(4), LaneValueProvenance::Move),
            (
                LaneReadKind::ReadFirst,
                Some(5),
                LaneValueProvenance::PrefixCount
            ),
            (
                LaneReadKind::ReadLane,
                Some(6),
                LaneValueProvenance::Arithmetic
            ),
        ]
    );
}

#[test]
fn mesh_wave_branch_masks_sendmsg_nibble_and_unreachable_scc() {
    let mut code = Vec::new();
    // pc 0..16: all four whole-mask conditional branches jumping to S_ENDPGM at pc 36
    code.extend(one("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 8)], None)); // pc 0 -> 36
    code.extend(one("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", 7)], None)); // pc 4 -> 36
    code.extend(one("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 6)], None)); // pc 8 -> 36
    code.extend(one("S_CBRANCH_VCCNZ", "ENC_SOPP", &[("SIMM16", 5)], None)); // pc 12 -> 36
                                                                             // pc 16..28: S_SENDMSG nibble filtering (low 4 bits == 9 -> GS_ALLOC_REQ)
    code.extend(one("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 9)], None)); // pc 16 (match)
    code.extend(one("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 1)], None)); // pc 20 (ignored)
    code.extend(one("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x19)], None)); // pc 24 (match: 0x19 & 0xf == 9)
                                                                          // pc 28..36: taint SCC from EXEC, then terminate before an unreachable S_CBRANCH_SCC1
    code.extend(one(
        "S_AND_B64",
        "ENC_SOP2",
        &[("SDST", 8), ("SSRC0", 126), ("SSRC1", 106)],
        None,
    )); // pc 28 (8 bytes? No, SOP2 without literal is 4 bytes -> pc 28)
    code.extend(one("S_ENDPGM", "ENC_SOPP", &[], None)); // pc 32
    code.extend(one("S_ENDPGM", "ENC_SOPP", &[], None)); // pc 36 (branch target)
    code.extend(one(
        "S_CBRANCH_SCC1",
        "ENC_SOPP",
        &[("SIMM16", 0xFFFF)],
        None,
    )); // pc 40 (unreachable -> pc 40)

    let report = mesh_wave::analyze(&code);
    assert!(
        report.invalid_branch_targets.is_empty(),
        "{:?}",
        report.invalid_branch_targets
    );
    assert_eq!(
        report.whole_mask_branches,
        vec![
            MaskBranch {
                pc: 0,
                target: 36,
                mask: BranchMask::ExecZero
            },
            MaskBranch {
                pc: 4,
                target: 36,
                mask: BranchMask::ExecNonZero
            },
            MaskBranch {
                pc: 8,
                target: 36,
                mask: BranchMask::VccZero
            },
            MaskBranch {
                pc: 12,
                target: 36,
                mask: BranchMask::VccNonZero
            },
        ]
    );
    assert_eq!(report.allocation_messages, vec![16, 24]);
    assert!(
        report.mask_scc_branches.is_empty(),
        "unreachable S_CBRANCH_SCC1 after S_ENDPGM has no reachable taint state"
    );
}
