use xcl::gpu::pack::{self, Record, RPF9_MAGIC};
use xcl::gpu::xbox::{
    Fssm, HwStage, InstStats, Program, XDAT_HASH, XDAT_NESTED_HEADER, XDAT_XBCH,
    api_stage_name, header_qwords, hw_stage_name, is_data_block, is_xbox, parse,
};

/// `s_mov_b32 s0, s1 ; v_add_f32 v0, v1, v5 ; s_endpgm`
const CODE: [u8; 12] = [0x01, 0x03, 0x80, 0xbe, 0x01, 0x0b, 0x00, 0x06, 0x00, 0x00, 0x81, 0xbf];
const S_ENDPGM: u32 = 0xBF81_0000;
const S_CODE_END: u32 = 0xBF9F_0000;

fn le32s(v: &mut Vec<u8>, xs: &[u32]) {
    for &x in xs {
        v.extend_from_slice(&x.to_le_bytes());
    }
}

fn container(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let header = 32 + parts.len() * 4;
    let mut v = vec![0u8; header];
    v[0..4].copy_from_slice(b"DXBC");
    v[20..24].copy_from_slice(&1u32.to_le_bytes());
    v[28..32].copy_from_slice(&(parts.len() as u32).to_le_bytes());
    for (i, (cc, data)) in parts.iter().enumerate() {
        let off = v.len() as u32;
        v[32 + i * 4..36 + i * 4].copy_from_slice(&off.to_le_bytes());
        v.extend_from_slice(*cc);
        v.extend_from_slice(&(data.len() as u32).to_le_bytes());
        v.extend_from_slice(data);
    }
    let total = v.len() as u32;
    v[24..28].copy_from_slice(&total.to_le_bytes());
    v
}

fn fssm_blob(code: &[u8], stages: &[(u8, u8)]) -> Vec<u8> {
    let code_at = 0x308usize;
    let mut v = vec![0u8; code_at];
    v[0..4].copy_from_slice(b"FSSM");
    v[4..8].copy_from_slice(&0x101u32.to_le_bytes());
    for q in (0x18..0xC0).step_by(8) {
        v[q..q + 8].copy_from_slice(&0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
    }
    v[0xC0..0xC8].copy_from_slice(&((code_at - 0x10) as u64).to_le_bytes());
    v[0xC8..0xD0].copy_from_slice(&(code.len() as u64 + 8).to_le_bytes());
    for (i, &(hw, api)) in stages.iter().enumerate() {
        let o = 0x100 + i * 0x88;
        let q = 0x0000_00C0_0002_0000u64 | (api as u64) << 8 | hw as u64;
        v[o..o + 8].copy_from_slice(&q.to_le_bytes());
    }
    v.extend_from_slice(code);
    v.extend_from_slice(&S_CODE_END.to_le_bytes());
    v.extend_from_slice(&S_CODE_END.to_le_bytes());
    v.extend_from_slice(b"\x03\0\0\0metadata\0\0\0\0");
    let size = (v.len() - 0x10) as u32;
    v[8..12].copy_from_slice(&size.to_le_bytes());
    v
}

fn xbch_blob(fssm: &[u8]) -> Vec<u8> {
    let z = miniz_oxide::deflate::compress_to_vec_zlib(fssm, 6);
    let mut v = b"XBCH".to_vec();
    le32s(&mut v, &[3, z.len() as u32, 0, fssm.len() as u32, 0]);
    v.extend_from_slice(&z);
    v
}

fn program_header(kind: u32, size_dw: u32, bitcode_off: u32) -> Vec<u8> {
    let mut v = Vec::new();
    le32s(&mut v, &[kind << 16 | 0x66, size_dw]);
    v.extend_from_slice(b"DXIL");
    le32s(&mut v, &[0x106, bitcode_off, 4]);
    v
}

fn dxil_part_with_entries(kind: u32, blobs: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let table = 0x18 + 12;
    let mut data_at = table + blobs.len() * 12;
    let mut xdat = b"XDAT".to_vec();
    le32s(&mut xdat, &[blobs.len() as u32, table as u32]);
    for (id, b) in blobs {
        le32s(&mut xdat, &[*id, data_at as u32, b.len() as u32]);
        data_at += b.len();
    }
    for (_, b) in blobs {
        xdat.extend_from_slice(b);
    }
    while (0x18 + xdat.len()) % 4 != 0 {
        xdat.push(0);
    }
    let nobc_at = 0x18 + xdat.len();
    let mut v = program_header(kind, ((nobc_at + 4) / 4) as u32, (nobc_at - 8) as u32);
    v.extend_from_slice(&xdat);
    v.extend_from_slice(b"NOBC");
    v
}

fn xbox_sample(kind: u32, fssms: &[Vec<u8>]) -> Vec<u8> {
    let mut blobs: Vec<(u32, Vec<u8>)> = vec![(XDAT_HASH, (0u8..16).collect())];
    blobs.extend(fssms.iter().map(|f| (XDAT_XBCH, xbch_blob(f))));
    let mut nested = program_header(kind, 0x8000_0007, 0x10);
    nested.extend_from_slice(b"NOBC");
    blobs.push((XDAT_NESTED_HEADER, nested));
    container(&[
        (b"SFI0", &[0; 8]), (b"HASH", &[0; 20]),
        (b"DXIL", &dxil_part_with_entries(kind, &blobs)), (b"PRIV", &[7; 16]),
    ])
}

fn pack_record(name: &str, flag: u8, payload: &[u8]) -> Vec<u8> {
    let z = miniz_oxide::deflate::compress_to_vec(payload, 6);
    let mut v = vec![(name.len() + 1) as u8];
    v.extend_from_slice(name.as_bytes());
    v.extend_from_slice(&[0, flag]);
    le32s(&mut v, &[z.len() as u32, 0xddce7, z.len() as u32, payload.len() as u32]);
    v.extend_from_slice(&z);
    v
}

#[test]
fn pack_roundtrip_ms_and_cs_into_xbox_parse() {
    let ms_blob = xbox_sample(13, &[fssm_blob(&CODE, &[(7, 6)])]);
    let cs_blob = xbox_sample(5, &[fssm_blob(&CODE, &[(6, 3)])]);

    let mut archive = RPF9_MAGIC.to_vec();
    archive.resize(0x40, 0xCC);
    archive.extend(pack_record("meshPass:MS_ClusterCull_abcd1234", 42, &ms_blob));
    archive.extend_from_slice(&[0xDE, 0xAD, 0x00, 0x01, 0x02]);
    archive.extend(pack_record("CS_Reduce_99887766", 7, &cs_blob));
    archive.extend_from_slice(&[0u8; 40]);

    let recs = pack::scan(&archive, 0x40);
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[0].name, "meshPass:MS_ClusterCull_abcd1234");
    assert_eq!(recs[0].flag, 42);
    assert_eq!(recs[1].name, "CS_Reduce_99887766");
    assert_eq!(recs[1].flag, 7);

    assert_eq!(pack::name_stage(&recs[0].name), Some("MS"));
    assert_eq!(pack::name_stage(&recs[1].name), Some("CS"));
    for (name, expected) in [
        ("ps_main", Some("PS")), ("VS_Skin", Some("VS")), ("GS_Wire", Some("GS")),
        ("HS_Patch", Some("HS")), ("DS_Eval", Some("DS")), ("eff:as_amp", Some("AS")),
        ("unknown_shader", None), ("A", None),
    ] {
        assert_eq!(pack::name_stage(name), expected, "name={name}");
    }

    let ms_inflated = pack::inflate(&archive, &recs[0]).expect("inflate MS");
    let cs_inflated = pack::inflate(&archive, &recs[1]).expect("inflate CS");
    assert!(is_xbox(&ms_inflated));
    assert!(is_xbox(&cs_inflated));

    let ms_sh = parse(&ms_inflated).expect("parse MS");
    assert_eq!(ms_sh.stage, "MS");
    assert_eq!(ms_sh.shader_model, (6, 6));
    assert!(ms_sh.notes.is_empty(), "{:?}", ms_sh.notes);
    assert_eq!(ms_sh.fssm[0].stage_label(), "NGG");
    assert_eq!(ms_sh.fssm[0].stages, vec![HwStage { offset: 0x100, hw: 7, api: 6 }]);
    assert_eq!(ms_sh.fssm[0].stages[0].label(), "NGG (API MS)");
    assert_eq!(ms_sh.fssm[0].programs, vec![Program { offset: 0x308, len: CODE.len() }]);
    assert_eq!(ms_sh.fssm[0].code_bytes().len(), CODE.len() + 8);
    assert!(ms_sh.fssm[0].code_bytes().starts_with(&CODE));

    let cs_sh = parse(&cs_inflated).expect("parse CS");
    assert_eq!(cs_sh.stage, "CS");
    assert!(cs_sh.notes.is_empty(), "{:?}", cs_sh.notes);
    assert_eq!(cs_sh.fssm[0].stage_label(), "CS");
    assert_eq!(cs_sh.fssm[0].stages[0].label(), "CS (API CS)");
    assert_eq!(cs_sh.fssm[0].programs, vec![Program { offset: 0x308, len: CODE.len() }]);
    assert!(cs_sh.fssm[0].code_bytes().starts_with(&CODE));

    // Non-Xbox containers and raw bytes must not be recognized as Xbox shaders
    let pc_dxbc = container(&[(b"DXIL", b"\x66\0\0\0\x04\0\0\0DXIL\0\0\0\0")]);
    assert!(!is_xbox(&pc_dxbc));
    assert!(!is_xbox(b"not a dxbc container"));

    // Corrupt record stream fails inflate cleanly
    let bad_rec = Record {
        name: "CS_Bad".into(), offset: 0, flag: 0,
        data: recs[1].data, csize: recs[1].csize, usize: recs[1].usize + 10,
    };
    assert_eq!(pack::inflate(&archive, &bad_rec), None);
}

#[test]
fn stage_names_and_fssm_helpers() {
    for (hw, name) in [(0,"LS"),(1,"HS"),(2,"ES"),(3,"GS"),(4,"VS"),(5,"PS"),(6,"CS"),(7,"NGG"),(8,"hw#8")] {
        assert_eq!(hw_stage_name(hw), name);
    }
    for (api, name) in [(0,"VS"),(1,"PS"),(2,"GS"),(3,"CS"),(4,"api#4"),(5,"api#5"),(6,"MS")] {
        assert_eq!(api_stage_name(api), name);
    }

    let s = HwStage { offset: 0x100, hw: 8, api: 4 };
    assert_eq!(s.label(), "hw#8 (API api#4)");

    let mut f = Fssm::default();
    assert!(f.code_bytes().is_empty());
    assert_eq!(f.stage_label(), "?");

    f.data = vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
    f.code = Some((1, 3));
    assert_eq!(f.code_bytes(), &[0xBB, 0xCC, 0xDD]);
    f.stages = vec![HwStage { offset: 0x100, hw: 3, api: 2 }, HwStage { offset: 0x188, hw: 4, api: 2 }];
    assert_eq!(f.stage_label(), "GS+VS");
}

#[test]
fn xdat_relative_offsets_non_0x18_xdat_dxil_kinds_and_stats() {
    // 1. XDAT-relative entry offsets + non-0x18 XDAT offset (+0x20) + AS kind (14)
    let f = fssm_blob(&CODE, &[(2, 0)]);
    let xbch = xbch_blob(&f);
    let xdat_at = 0x20usize;
    let table_in_xdat = 12usize;
    let data_in_xdat = table_in_xdat + 12; // 24 (< xdat_at + table_in_xdat = 56, so base=0 fails and base=x succeeds)
    let mut xdat = b"XDAT".to_vec();
    // invalid stated table_off (0) -> falls back to x + 12
    le32s(&mut xdat, &[1, 0, XDAT_XBCH, data_in_xdat as u32, xbch.len() as u32]);
    xdat.extend_from_slice(&xbch);
    while (xdat_at + xdat.len()) % 4 != 0 {
        xdat.push(0);
    }
    let nobc_at = xdat_at + xdat.len();
    let mut dxil = program_header(14, ((nobc_at + 4) / 4) as u32, (nobc_at - 8) as u32);
    dxil.resize(xdat_at, 0);
    dxil.extend_from_slice(&xdat);
    dxil.extend_from_slice(b"NOBC");

    let sh = parse(&container(&[(b"DXIL", &dxil)])).expect("parse XDAT-relative");
    assert_eq!(sh.stage, "AS");
    assert_eq!(sh.xdat_offset, sh.dxil_offset + 0x20);
    assert!(sh.notes.iter().any(|n| n.contains("XDAT at +0x20")), "{:?}", sh.notes);
    assert_eq!(sh.entries.len(), 1);
    assert_eq!(sh.entries[0].offset, sh.xdat_offset + data_in_xdat);
    assert_eq!(sh.fssm.len(), 1);
    assert_eq!(sh.fssm[0].error, None);
    assert_eq!(sh.fssm[0].stage_label(), "ES");

    // 2. Unknown DXIL kind (e.g. 22 -> "kind#22")
    let unk_sh = parse(&xbox_sample(22, &[fssm_blob(&CODE, &[(5, 1)])])).expect("parse unknown kind");
    assert_eq!(unk_sh.stage, "kind#22");

    // 3. InstStats::merge / sorted and header_qwords
    let mut s1 = InstStats::default();
    s1.add_code(&CODE); // s_mov_b32, v_add_f32, s_endpgm
    let mut with_bad = CODE.to_vec();
    with_bad.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // undecodable VOP2 op 0
    let mut s2 = InstStats::default();
    s2.add_code(&with_bad);

    s1.merge(&s2);
    assert_eq!(s1.instructions, 6);
    assert_eq!(s1.undecoded, 1);

    let sorted = s1.sorted();
    assert_eq!(sorted.len(), 3);
    assert!(sorted.iter().all(|&(_, count)| count == 2));
    // Tie-break is ascending by mnemonic name
    assert_eq!(sorted[0].0, "s_endpgm");
    assert_eq!(sorted[1].0, "s_mov_b32");
    assert_eq!(sorted[2].0, "v_add_f32");

    let qwords = header_qwords(&sh.fssm[0], 0x110);
    // 0x00 has "FSSM" + version, 0x08 has size, 0xC0 has code off, 0xC8 has code len, 0x100 has stage;
    // all 0xDEAD_BEEF_DEAD_BEEF and 0 qwords are filtered out.
    assert!(qwords.iter().any(|&(off, _)| off == 0x00));
    assert!(qwords.iter().any(|&(off, _)| off == 0xC0));
    assert!(qwords.iter().any(|&(off, _)| off == 0x100));
    assert!(!qwords.iter().any(|&(off, _)| off >= 0x110));
    assert!(!qwords.iter().any(|&(_, val)| val == 0 || val == 0xDEAD_BEEF_DEAD_BEEF));
}

#[test]
fn xbch_and_fssm_error_and_fallback_paths() {
    // 1. Corrupt zlib stream
    let mut corrupt_xbch = b"XBCH".to_vec();
    le32s(&mut corrupt_xbch, &[3, 8, 0, 64, 0]);
    corrupt_xbch.extend_from_slice(&[0x78, 0x9c, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00]);

    // 2. rsize mismatch (valid zlib of FSSM, but header rsize is larger than actual)
    let valid_fssm = fssm_blob(&CODE, &[(5, 1)]);
    let mut rsize_mismatch = xbch_blob(&valid_fssm);
    let wrong_rsize = (valid_fssm.len() + 128) as u32;
    rsize_mismatch[0x10..0x14].copy_from_slice(&wrong_rsize.to_le_bytes());

    // 3. Missing zlib header
    let mut no_zlib = b"XBCH".to_vec();
    no_zlib.resize(0x30, 0xAA);

    // 4. Non-0x18 zlib header offset (zlib starts at +0x20)
    let z = miniz_oxide::deflate::compress_to_vec_zlib(&valid_fssm, 6);
    let mut shifted_xbch = b"XBCH".to_vec();
    le32s(&mut shifted_xbch, &[3, z.len() as u32, 0, valid_fssm.len() as u32]);
    shifted_xbch.resize(0x20, 0x00);
    shifted_xbch.extend_from_slice(&z);

    // 5. Missing FSSM magic with fallback code scan
    let mut no_magic = vec![0u8; 0x30];
    no_magic[0..4].copy_from_slice(b"XXXX");
    no_magic[0x2C..0x30].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    no_magic.extend_from_slice(&CODE);
    no_magic.extend_from_slice(&S_CODE_END.to_le_bytes());
    let no_magic_xbch = xbch_blob(&no_magic);

    // 6. Unexpected XBCH entry id (0x99) & 7. XDAT_XBCH entry missing XBCH header
    let dxil = dxil_part_with_entries(
        0,
        &[
            (XDAT_XBCH, corrupt_xbch),
            (XDAT_XBCH, rsize_mismatch),
            (XDAT_XBCH, no_zlib),
            (XDAT_XBCH, shifted_xbch),
            (XDAT_XBCH, no_magic_xbch),
            (0x99, xbch_blob(&valid_fssm)),
            (XDAT_XBCH, b"NOPE_NOT_XBCH".to_vec()),
        ],
    );
    let sh = parse(&container(&[(b"DXIL", &dxil)])).expect("parse error/fallback container");
    assert_eq!(sh.fssm.len(), 6);

    assert!(sh.fssm[0].error.as_deref().unwrap_or("").starts_with("XBCH: zlib error"));
    assert!(sh.fssm[0].data.is_empty());

    let e1 = sh.fssm[1].error.as_deref().unwrap_or("");
    assert!(e1.contains(&format!("the header says {wrong_rsize} B")), "{e1}");
    assert_eq!(sh.fssm[1].programs.len(), 1);

    assert_eq!(sh.fssm[2].error.as_deref(), Some("XBCH: zlib stream not found"));

    assert_eq!(sh.fssm[3].error, None);
    assert_eq!(sh.fssm[3].programs, vec![Program { offset: 0x308, len: CODE.len() }]);
    assert!(sh.fssm[3].code_bytes().starts_with(&CODE));

    assert_eq!(sh.fssm[4].error.as_deref(), Some("no FSSM header"));
    assert!(!sh.fssm[4].code_from_header);
    assert_eq!(sh.fssm[4].programs, vec![Program { offset: 0x30, len: CODE.len() }]);

    assert_eq!(sh.fssm[5].error, None);
    assert!(sh.notes.iter().any(|n| n == "XBCH in an entry with unexpected id 0x99"), "{:?}", sh.notes);
    assert!(sh.notes.iter().any(|n| n == "entry 0xd without an XBCH header"), "{:?}", sh.notes);

    // Container with only a non-XBCH entry triggers the "no FSSM" note
    let empty_dxil = dxil_part_with_entries(0, &[(XDAT_XBCH, b"BAD!".to_vec())]);
    let empty_sh = parse(&container(&[(b"DXIL", &empty_dxil)])).expect("parse no-fssm");
    assert!(empty_sh.fssm.is_empty());
    assert!(empty_sh.notes.iter().any(|n| n == "no FSSM (native code) in the container"));
}

#[test]
fn is_data_block_distinguishes_code_and_data() {
    // Counted data block: u32 N (=2), 2 dwords, s_endpgm
    let counted: Vec<u8> = [2u32, 0x3f80_0000, 0x1234_5678, S_ENDPGM]
        .iter().flat_map(|w| w.to_le_bytes()).collect();
    assert!(is_data_block(&counted));

    // Undecodable first dword (VOP2 opcode 0 is unassigned)
    let mut bad_start = 0u32.to_le_bytes().to_vec();
    bad_start.extend_from_slice(&CODE);
    assert!(is_data_block(&bad_start));

    // >= 25% undecodable: 3 valid dwords + 1 undecodable dword (1/4 = 25%)
    let mut quarter_bad = CODE.to_vec();
    le32s(&mut quarter_bad, &[0]);
    assert!(is_data_block(&quarter_bad));

    // < 25% undecodable: 4 valid dwords + 1 undecodable dword (1/5 = 20%) -> false
    let mut mostly_valid = CODE.to_vec();
    le32s(&mut mostly_valid, &[0xBF80_0000, 0]); // s_nop 0 + undecodable
    assert!(!is_data_block(&mostly_valid));

    // Pure valid code -> false
    assert!(!is_data_block(&CODE));
}
