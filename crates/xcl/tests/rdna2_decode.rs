//! Structured decoding of RDNA2 microcode through `gpu::rdna2`'s public API: instruction
//! length per encoding class, encoding / opcode identification, literal and extension dwords,
//! MIMG NSA, program walking, truncated and random input, and the instruction tables.
//!
//! Every instruction word is assembled by hand from the field layouts of AMD's "RDNA 2
//! Instruction Set Architecture" reference guide, chapter "Microcode Formats" (section named
//! per case). Nothing here looks at disassembly text, whose format is still changing.

use std::collections::{HashMap, HashSet};
use std::panic::catch_unwind;

use xcl::gpu::rdna2::{
    all_instructions, branch_target, decode, disassemble, encode, field_value, find_programs, DecodeError, Variant,
    F_BRANCH, F_CONDITIONAL, F_INDIRECT, F_TERMINATOR,
};

/// One hand-assembled instruction and what the decoder must make of it.
struct Case {
    words: &'static [u32],
    name: &'static str,
    /// the primary encoding the decoder identifies
    enc: &'static str,
    len: usize,
    variant: Variant,
    literal: Option<u32>,
}

const fn c(words: &'static [u32], name: &'static str, enc: &'static str, len: usize) -> Case {
    Case { words, name, enc, len, variant: Variant::Plain, literal: None }
}
const fn lit(words: &'static [u32], name: &'static str, enc: &'static str, len: usize, l: u32) -> Case {
    Case { words, name, enc, len, variant: Variant::Plain, literal: Some(l) }
}
const fn ext(words: &'static [u32], name: &'static str, enc: &'static str, variant: Variant) -> Case {
    Case { words, name, enc, len: 8, variant, literal: None }
}

const V: u32 = 256; // VGPR base in a 9-bit source field
const L: u32 = 0x1234_5678; // a literal

#[rustfmt::skip]
static CASES: &[Case] = &[
    // SOP1: ENCODING[31:23]=0x17D SDST[22:16] OP[15:8] SSRC0[7:0]
    c(&[0xBE80_0000 | 5 << 16 | 0x03 << 8 | 7], "S_MOV_B32", "ENC_SOP1", 4),
    lit(&[0xBE80_0000 | 2 << 16 | 0x04 << 8 | 255, L], "S_MOV_B64", "ENC_SOP1", 8, L), // 64-bit op, 32-bit literal
    // SOP2: ENCODING[31:30]=2 OP[29:23] SDST[22:16] SSRC1[15:8] SSRC0[7:0]
    c(&[0x8000_0000 | 1 << 16 | 3 << 8 | 2], "S_ADD_U32", "ENC_SOP2", 4),
    lit(&[0x8000_0000 | 0x04 << 23 | 1 << 16 | 255 << 8 | 1, L], "S_ADDC_U32", "ENC_SOP2", 8, L),
    lit(&[0x8000_0000 | 1 << 16 | 255 << 8 | 255, L], "S_ADD_U32", "ENC_SOP2", 8, L), // both sources share one literal
    // SOPK: ENCODING[31:28]=0xB OP[27:23] SDST[22:16] SIMM16[15:0]
    c(&[0xB000_0000 | 4 << 16 | 0xFFFF], "S_MOVK_I32", "ENC_SOPK", 4), // SIMM16 0xFFFF is no source code
    lit(&[0xB000_0000 | 0x15 << 23 | 0x0001, L], "S_SETREG_IMM32_B32", "ENC_SOPK", 8, L),
    // SOPC: ENCODING[31:23]=0x17E OP[22:16] SSRC1[15:8] SSRC0[7:0]
    c(&[0xBF00_0000 | 0x06 << 16 | 2 << 8 | 1], "S_CMP_EQ_U32", "ENC_SOPC", 4),
    lit(&[0xBF00_0000 | 0x06 << 16 | 255 << 8 | 1, L], "S_CMP_EQ_U32", "ENC_SOPC", 8, L),
    // SOPP: ENCODING[31:23]=0x17F OP[22:16] SIMM16[15:0]
    c(&[0xBF80_0000 | 0x0C << 16], "S_WAITCNT", "ENC_SOPP", 4),
    c(&[0xBF80_0000 | 0x02 << 16 | 0x00FF], "S_BRANCH", "ENC_SOPP", 4),
    // SMEM: ENCODING[31:26]=0x3D OP[25:18] GLC[16] DLC[14] SDATA[12:6] SBASE[5:0]; SOFFSET[63:57] OFFSET[52:32]
    c(&[0xF400_0000 | 0x02 << 18 | 8 << 6 | 1, 125 << 25 | 0x10], "S_LOAD_DWORDX4", "ENC_SMEM", 8),
    c(&[0xF400_0000 | 0x08 << 18 | 4 << 6 | 2, 0x7F << 25 | 0x1F_FFFF], "S_BUFFER_LOAD_DWORD", "ENC_SMEM", 8),
    // VOP1: ENCODING[31:25]=0x3F VDST[24:17] OP[16:9] SRC0[8:0]
    c(&[0x7E00_0000 | 1 << 17 | 0x01 << 9 | (V + 2)], "V_MOV_B32", "ENC_VOP1", 4),
    lit(&[0x7E00_0000 | 1 << 17 | 0x01 << 9 | 255, L], "V_MOV_B32", "ENC_VOP1", 8, L),
    lit(&[0x7E00_0000 | 255, L], "V_NOP", "ENC_VOP1", 8, L), // v_nop reads no source, the literal is still fetched
    // VOP2: ENCODING[31]=0 OP[30:25] VDST[24:17] VSRC1[16:9] SRC0[8:0]
    c(&[0x03 << 25 | 1 << 17 | 3 << 9 | (V + 2)], "V_ADD_F32", "ENC_VOP2", 4),
    c(&[0x03 << 25 | 1 << 17 | 255 << 9 | (V + 2)], "V_ADD_F32", "ENC_VOP2", 4), // VSRC1 255 is v255
    lit(&[0x03 << 25 | 1 << 17 | 3 << 9 | 255, L], "V_ADD_F32", "ENC_VOP2", 8, L),
    lit(&[0x2C << 25 | 1 << 17 | 3 << 9 | (V + 2), L], "V_FMAMK_F32", "ENC_VOP2", 8, L), // K is always there
    lit(&[0x2D << 25 | 1 << 17 | 3 << 9 | 255, L], "V_FMAAK_F32", "ENC_VOP2", 8, L), // K and src0 share it
    lit(&[0x37 << 25 | 1 << 17 | 3 << 9 | (V + 2), L], "V_FMAMK_F16", "ENC_VOP2", 8, L), // 16-bit K, full dword
    // VOPC: ENCODING[31:25]=0x3E OP[24:17] VSRC1[16:9] SRC0[8:0]
    c(&[0x7C00_0000 | 0x01 << 17 | 2 << 9 | (V + 1)], "V_CMP_LT_F32", "ENC_VOPC", 4),
    lit(&[0x7C00_0000 | 0x11 << 17 | 2 << 9 | 255, L], "V_CMPX_LT_F32", "ENC_VOPC", 8, L),
    // SDWA / DPP16 / DPP8 (section "VOP SDWA", "VOP DPP"): SRC0 = 0xF9 / 0xFA / 0xE9 / 0xEA, one extension dword
    ext(&[0x7E00_0000 | 1 << 17 | 0x01 << 9 | 0xF9, 0x0006_0602], "V_MOV_B32", "ENC_VOP1", Variant::Sdwa),
    ext(&[0x03 << 25 | 1 << 17 | 3 << 9 | 0xF9, 0x0606_0602], "V_ADD_F32", "ENC_VOP2", Variant::Sdwa),
    ext(&[0x7C00_0000 | 0x01 << 17 | 2 << 9 | 0xF9, 0x0606_0001], "V_CMP_LT_F32", "ENC_VOPC", Variant::Sdwa),
    ext(&[0x7E00_0000 | 1 << 17 | 0x01 << 9 | 0xFA, 0xFF00_1B03], "V_MOV_B32", "ENC_VOP1", Variant::Dpp16),
    ext(&[0x03 << 25 | 1 << 17 | 3 << 9 | 0xFA, 0xFF00_0002], "V_ADD_F32", "ENC_VOP2", Variant::Dpp16),
    ext(&[0x7E00_0000 | 1 << 17 | 0x01 << 9 | 0xE9, 0x00FA_C688], "V_MOV_B32", "ENC_VOP1", Variant::Dpp8),
    ext(&[0x03 << 25 | 1 << 17 | 3 << 9 | 0xEA, 0x00FA_C688], "V_ADD_F32", "ENC_VOP2", Variant::Dpp8), // DPP8 FI
    // VOP3: ENCODING[31:26]=0x35 OP[25:16] CLAMP[15] OP_SEL[14:11] ABS[10:8] VDST[7:0]; NEG OMOD SRC2[58:50] SRC1[49:41] SRC0[40:32]
    c(&[0xD400_0000 | 0x14B << 16, (V + 1) | (V + 2) << 9 | (V + 3) << 18], "V_FMA_F32", "ENC_VOP3", 8),
    lit(&[0xD400_0000 | 0x14B << 16, (V + 1) | (V + 2) << 9 | 255 << 18, L], "V_FMA_F32", "ENC_VOP3", 12, L),
    lit(&[0xD400_0000 | 0x14B << 16, 255 | 255 << 9 | 255 << 18, L], "V_FMA_F32", "ENC_VOP3", 12, L),
    lit(&[0xD400_0000 | 0x164 << 16 | 4, (V + 2) | 255 << 9, L], "V_ADD_F64", "ENC_VOP3", 12, L),
    lit(&[0xD400_0000 | 0x180 << 16, 255 << 18, L], "V_NOP", "ENC_VOP3", 12, L),
    c(&[0xD400_0000 | 0x103 << 16 | 1, 0xF9 | (V + 3) << 9], "V_ADD_F32", "ENC_VOP3", 8), // no SDWA in VOP3
    c(&[0xD400_0000 | 0x103 << 16 | 1, 0xFA | (V + 3) << 9], "V_ADD_F32", "ENC_VOP3", 8), // nor DPP
    c(&[0xD400_0000 | 0x001 << 16 | 106, (V + 1) | (V + 2) << 9], "V_CMP_LT_F32", "ENC_VOP3", 8),
    // VOP3 with SDST[14:8] in place of ABS / OP_SEL
    c(&[0xD400_0000 | 0x30F << 16 | 106 << 8 | 1, (V + 2) | (V + 3) << 9], "V_ADD_CO_U32", "VOP3_SDST_ENC", 8),
    lit(&[0xD400_0000 | 0x30F << 16 | 106 << 8 | 1, 255 | (V + 3) << 9, L], "V_ADD_CO_U32", "VOP3_SDST_ENC", 12, L),
    // VOP3P: ENCODING[31:26]=0x33 OP[22:16] OP_SEL_HI_2[14] ...; OP_SEL_HI[60:59] SRC2 SRC1 SRC0
    c(&[0xCC00_0000 | 0x0E << 16 | 1 << 14, (V + 1) | (V + 2) << 9 | (V + 3) << 18 | 3 << 27], "V_PK_FMA_F16", "ENC_VOP3P", 8),
    lit(&[0xCC00_0000 | 0x0E << 16 | 1 << 14, (V + 1) | (V + 2) << 9 | 255 << 18 | 3 << 27, L], "V_PK_FMA_F16", "ENC_VOP3P", 12, L),
    // VINTRP: ENCODING[31:26]=0x32 VDST[25:18] OP[17:16] ATTR[15:10] ATTRCHAN[9:8] VSRC[7:0]
    c(&[0xC800_0000 | 2 << 18 | 3 << 10 | 1 << 8], "V_INTERP_P1_F32", "ENC_VINTRP", 4),
    c(&[0xC800_0000 | 2 << 18 | 0x2 << 16 | 63 << 10 | 3 << 8 | 2], "V_INTERP_MOV_F32", "ENC_VINTRP", 4),
    // DS: ENCODING[31:26]=0x36 OP[25:18] GDS[17] OFFSET1[15:8] OFFSET0[7:0]; VDST[63:56] DATA1 DATA0 ADDR[39:32]
    c(&[0xD800_0000 | 0x36 << 18 | 0xFFFF, 2 | 1 << 24], "DS_READ_B32", "ENC_DS", 8),
    c(&[0xD800_0000 | 0x0D << 18, 2 | 3 << 8], "DS_WRITE_B32", "ENC_DS", 8),
    // MUBUF: OP is [24:18] plus bit 25 as its MSB; SOFFSET[63:56] takes no literal
    c(&[0xE000_0000 | 0x0C << 18 | 1 << 12, 2 | 1 << 8 | 1 << 16 | 255 << 24], "BUFFER_LOAD_DWORD", "ENC_MUBUF", 8),
    c(&[0xE000_0000 | 1 << 25, 2 | 1 << 8 | 1 << 16], "BUFFER_LOAD_FORMAT_D16_X", "ENC_MUBUF", 8), // opcode 0x80
    // MTBUF: OP is [18:16] plus bit 53 (word 1 bit 21) as its MSB, FORMAT[25:19]
    c(&[0xE800_0000 | 0x16 << 19, 2 | 1 << 8 | 1 << 16], "TBUFFER_LOAD_FORMAT_X", "ENC_MTBUF", 8),
    c(&[0xE800_0000 | 0x16 << 19, 2 | 1 << 8 | 1 << 16 | 1 << 21], "TBUFFER_LOAD_FORMAT_D16_X", "ENC_MTBUF", 8), // opcode 8
    c(&[0xE800_0000 | 0x7 << 16, 2 | 1 << 8 | 1 << 16 | 1 << 21], "TBUFFER_STORE_FORMAT_D16_XYZW", "ENC_MTBUF", 8),
    // MIMG: OP is [24:18] plus bit 0 as its MSB, NSA[2:1] = extra address dwords; SSAMP[57:53] SRSRC[52:48]
    c(&[0xF000_0000 | 0x20 << 18 | 0xF << 8 | 1 << 3, 4 | 2 << 16 | 4 << 21], "IMAGE_SAMPLE", "ENC_MIMG", 8),
    c(&[0xF000_0000 | 0x20 << 18 | 0xF << 8 | 1 << 3 | 1 << 1, 4 | 2 << 16 | 4 << 21, 0x0706_0504], "IMAGE_SAMPLE", "ENC_MIMG", 12),
    c(&[0xF000_0000 | 0x20 << 18 | 0xF << 8 | 2 << 3 | 2 << 1, 4 | 2 << 16 | 4 << 21, 5, 6], "IMAGE_SAMPLE", "ENC_MIMG", 16),
    c(&[0xF000_0000 | 0x66 << 18 | 1 | 3 << 1, 4 | 8 << 8 | 2 << 16, 1, 2, 3], "IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", 20), // opcode 0xE6
    c(&[0xF000_0000 | 0x67 << 18 | 1, 4 | 8 << 8 | 2 << 16], "IMAGE_BVH64_INTERSECT_RAY", "ENC_MIMG", 8),
    // EXP: ENCODING[31:26]=0x3E VM[12] DONE[11] COMPR[10] TGT[9:4] EN[3:0]; VSRC3..0, no OP field
    c(&[0xF800_0000 | 1 << 12 | 1 << 11 | 0xF, 0x0302_0100], "EXP", "ENC_EXP", 8),
    // FLAT / GLOBAL / SCRATCH: ENCODING[31:26]=0x37 OP[24:18] SLC GLC SEG[15:14] LDS DLC OFFSET[11:0]
    c(&[0xDC00_0000 | 0x0C << 18, 2 | 0x7D << 16 | 1 << 24], "FLAT_LOAD_DWORD", "ENC_FLAT", 8),
    c(&[0xDC00_0000 | 0x0C << 18 | 2 << 14 | 0xFFF, 2 | 0x7D << 16 | 1 << 24], "GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", 8),
    c(&[0xDC00_0000 | 0x1C << 18 | 2 << 14, 2 | 3 << 8 | 4 << 16], "GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", 8),
    c(&[0xDC00_0000 | 0x0C << 18 | 1 << 14, 2 | 0x7D << 16 | 1 << 24], "SCRATCH_LOAD_DWORD", "ENC_FLAT_SCRATCH", 8),
    // console-only (rdna2_xbox.rs): the gfx10.1 non-fused multiply-adds
    c(&[0x1F << 25 | 6 << 17 | 21 << 9 | 40], "V_MAC_F32", "ENC_VOP2", 4),
    c(&[0xD400_0000 | 0x11F << 16 | 6, 40 | (V + 21) << 9], "V_MAC_F32", "ENC_VOP3", 8),
    lit(&[0x20 << 25 | 1 << 17 | 3 << 9 | (V + 2), 0x3F80_0000], "V_MADMK_F32", "ENC_VOP2", 8, 0x3F80_0000),
    lit(&[0x21 << 25 | 1 << 17 | 3 << 9 | (V + 2), 0x3F80_0000], "V_MADAK_F32", "ENC_VOP2", 8, 0x3F80_0000),
    lit(&[0x21 << 25 | 1 << 17 | 3 << 9 | 255, 0x4000_0000], "V_MADAK_F32", "ENC_VOP2", 8, 0x4000_0000),
    c(&[0xD400_0000 | 0x141 << 16 | 7, 42 | 13 << 9 | (V + 24) << 18], "V_MAD_F32", "ENC_VOP3", 8),
    lit(&[0xD400_0000 | 0x141 << 16 | 7, 42 | 255 << 9 | (V + 24) << 18, L], "V_MAD_F32", "ENC_VOP3", 12, L),
    // terminator last, so the walker can leave it out of its pool
    c(&[0xBF81_0000], "S_ENDPGM", "ENC_SOPP", 4),
];

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Deterministic random numbers (Knuth's MMIX LCG, high bits).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// Instruction length of an encoding class without literal or extension dwords
/// (chapter "Microcode Formats": the scalar ALU, VOP1 / VOP2 / VOPC and VINTRP formats are one
/// dword, everything else two).
fn class_len(enc: &str) -> usize {
    match enc {
        "ENC_SOP1" | "ENC_SOP2" | "ENC_SOPK" | "ENC_SOPC" | "ENC_SOPP" | "ENC_VOP1" | "ENC_VOP2" | "ENC_VOPC" | "ENC_VINTRP" => 4,
        _ => 8,
    }
}

/// Every hand-assembled case decodes to the instruction, encoding, length, variant and literal
/// the ISA gives it, with the raw dwords and the PC kept, and extra trailing bytes ignored.
#[test]
fn hand_assembled_instructions_decode_by_class() {
    for (i, k) in CASES.iter().enumerate() {
        assert_eq!(k.words.len() * 4, k.len, "case {i} ({}) is assembled with the wrong dword count", k.name);
        let mut code = bytes(k.words);
        code.extend_from_slice(&[0xAA; 8]); // the next instruction must not be read
        let pc = 0x1000 + 4 * i as u64;
        let d = decode(&code, pc).unwrap_or_else(|e| panic!("case {i} {} {:08x?}: {e:?}", k.name, k.words));
        let what = format!("case {i} {:08x?}", k.words);
        assert_eq!(d.inst.name, k.name, "{what}");
        assert_eq!(d.enc.name, k.enc, "{what}");
        assert_eq!((d.len, d.variant, d.literal), (k.len, k.variant, k.literal), "{what} ({})", k.name);
        assert_eq!(d.words, k.words, "{what}");
        assert_eq!(d.pc, pc);
        assert!(d.inst.encs.iter().any(|e| std::ptr::eq(e, d.ienc)), "{what}: ienc is not one of the instruction's");
        if let Some(op) = field_value(d.enc, &d.words, "OP") {
            assert_eq!(op, d.ienc.opcode as u64, "{what}: OP field");
        }
        // a literal sits right after the class's base dwords
        if let Some(l) = d.literal {
            assert_eq!(l, d.words[class_len(k.enc) / 4], "{what}");
        }
    }
}

/// `encode` puts each field where the ISA's layout has it: its bytes match the hand-assembled
/// words, including split opcode fields (MUBUF bit 25, MTBUF bit 53) and the SDWA / DPP layouts.
#[test]
fn encoder_matches_hand_assembly() {
    let checks: &[(&str, &str, &[(&str, u64)], Option<u32>, &[u32])] = &[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 7)], None, &[0xBE85_0307]),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 255), ("SSRC1", 255)], Some(L), &[0x8001_FFFF, L]),
        ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SIMM16", 1)], Some(L), &[0xBA80_0001, L]),
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 1), ("SOFFSET", 125), ("OFFSET", 0x10)], None, &[0xF408_0201, 0xFA00_0010]),
        ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 4), ("SBASE", 2), ("SOFFSET", 0x7F), ("OFFSET", 0x1F_FFFF)], None, &[0xF420_0102, 0xFE1F_FFFF]),
        ("V_MOV_B32", "VOP1_VOP_SDWA", &[("VDST", 1), ("SRC0", 0xF9), ("VSRC0", 2), ("DST_SEL", 6), ("SRC0_SEL", 6)], None, &[0x7E02_02F9, 0x0006_0602]),
        ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 1), ("SRC0", 0xFA), ("VSRC0", 3), ("DPP_CTRL", 0x1B), ("BANK_MASK", 0xF), ("ROW_MASK", 0xF)], None, &[0x7E02_02FA, 0xFF00_1B03]),
        ("V_FMA_F32", "ENC_VOP3", &[("SRC0", V as u64 + 1), ("SRC1", V as u64 + 2), ("SRC2", 255)], Some(L), &[0xD54B_0000, 0x03FE_0501, L]),
        ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 1), ("SDST", 106), ("SRC0", V as u64 + 2), ("SRC1", V as u64 + 3)], None, &[0xD70F_6A01, 0x0002_0702]),
        ("DS_READ_B32", "ENC_DS", &[("OFFSET0", 0xFF), ("OFFSET1", 0xFF), ("ADDR", 2), ("VDST", 1)], None, &[0xD8D8_FFFF, 0x0100_0002]),
        ("BUFFER_LOAD_FORMAT_D16_X", "ENC_MUBUF", &[("VADDR", 2), ("VDATA", 1), ("SRSRC", 1)], None, &[0xE200_0000, 0x0001_0102]),
        ("TBUFFER_LOAD_FORMAT_D16_X", "ENC_MTBUF", &[("FORMAT", 0x16), ("VADDR", 2), ("VDATA", 1), ("SRSRC", 1)], None, &[0xE8B0_0000, 0x0021_0102]),
        ("IMAGE_BVH64_INTERSECT_RAY", "ENC_MIMG", &[("VADDR", 4), ("VDATA", 8), ("SRSRC", 2)], None, &[0xF19C_0001, 0x0002_0804]),
        ("EXP", "ENC_EXP", &[("EN", 0xF), ("DONE", 1), ("VM", 1), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3)], None, &[0xF800_180F, 0x0302_0100]),
        ("SCRATCH_LOAD_DWORD", "ENC_FLAT_SCRATCH", &[("ADDR", 2), ("SADDR", 0x7D), ("VDST", 1)], None, &[0xDC30_4000, 0x017D_0002]),
        ("V_MADAK_F32", "ENC_VOP2", &[("VDST", 1), ("VSRC1", 3), ("SRC0", 255)], Some(0x4000_0000), &[0x4202_06FF, 0x4000_0000]),
    ];
    for &(name, enc, fields, l, want) in checks {
        let got = encode(name, enc, fields, l).unwrap_or_else(|e| panic!("encode {name} {enc}: {e}"));
        assert_eq!(got, bytes(want), "{name} {enc}");
        assert_eq!(decode(&got, 0).unwrap().inst.name, name);
    }
}

/// Opcodes the RDNA2 spec leaves unassigned decode as `Unknown` with the first dword, also
/// when the word's prefix would fit a lower-priority encoding (SOP1 → SOPK / SOP2, VOP1 /
/// VOPC → VOP2, VOP3 → VOP3 with SDST); FLAT SEG = 3 is reserved; all-zero padding is no code.
#[test]
fn unassigned_opcodes_are_unknown() {
    let tables = encoding_indices();
    let used: HashSet<(u16, u16)> = all_instructions().flat_map(|i| i.encs.iter().map(|e| (e.encoding, e.opcode))).collect();
    // (encoding, fixed bits, OP shift); the first opcode the tables do not use there
    let classes: &[(&str, u32, u32)] = &[
        ("ENC_SOP1", 0xBE80_0000, 8),
        ("ENC_SOPC", 0xBF00_0000, 16),
        ("ENC_SOPP", 0xBF80_0000, 16),
        ("ENC_SOPK", 0xB000_0000, 23),
        ("ENC_SOP2", 0x8000_0000, 23),
        ("ENC_SMEM", 0xF400_0000, 18),
        ("ENC_VOP1", 0x7E00_0000, 9),
        ("ENC_VOP2", 0x0000_0000, 25),
        ("ENC_VOP3P", 0xCC00_0000, 16),
        ("ENC_DS", 0xD800_0000, 18),
        ("ENC_FLAT", 0xDC00_0000, 18),
    ];
    for &(enc, fixed, shift) in classes {
        let ei = tables[enc];
        let op = (0..128u16).find(|&o| !used.contains(&(ei, o))).unwrap_or_else(|| panic!("{enc}: no free opcode"));
        let w = fixed | (op as u32) << shift;
        assert_eq!(decode(&bytes(&[w, 0]), 0).err(), Some(DecodeError::Unknown(w)), "{enc} opcode {op:#x}");
    }
    // VOP3 opcode with neither a VOP3 nor a VOP3-with-SDST instruction
    let (v3, sd) = (tables["ENC_VOP3"], tables["VOP3_SDST_ENC"]);
    let op = (0..1024u16).find(|&o| !used.contains(&(v3, o)) && !used.contains(&(sd, o))).unwrap();
    let w = 0xD400_0000 | (op as u32) << 16;
    assert_eq!(decode(&bytes(&[w, 0]), 0).err(), Some(DecodeError::Unknown(w)), "VOP3 opcode {op:#x}");
    // FLAT with SEG = 3
    let w = 0xDC00_0000 | 0x0C << 18 | 3 << 14;
    assert_eq!(decode(&bytes(&[w, 0]), 0).err(), Some(DecodeError::Unknown(w)));
    // zero dwords (VOP2 opcode 0): padding between programs
    assert_eq!(decode(&[0; 8], 0).err(), Some(DecodeError::Unknown(0)));
}

/// Encoding indices of the tables (`InstEnc::encoding`), learnt by decoding one hand-assembled
/// instruction of each primary encoding.
fn encoding_indices() -> HashMap<&'static str, u16> {
    let mut m = HashMap::new();
    for k in CASES {
        let d = decode(&bytes(k.words), 0).unwrap();
        let prev = m.insert(d.enc.name, d.ienc.encoding);
        assert!(prev.is_none() || prev == Some(d.ienc.encoding), "{} has two indices", d.enc.name);
    }
    m
}

/// The instruction tables: names are unique, every (encoding, opcode) belongs to one
/// instruction (VOP3 and VOP3-with-SDST share one opcode space), console extensions only fill
/// opcodes the spec leaves free, label operands are direct branches, and every encoding of
/// every instruction assembles with `encode` and decodes back to itself at the class's length.
#[test]
fn every_table_entry_is_unique_and_decodes_to_itself() {
    let idx = encoding_indices();
    let names: HashMap<u16, &str> = idx.iter().map(|(&n, &i)| (i, n)).collect();
    let mut seen: HashMap<(u16, u16), &str> = HashMap::new();
    let mut inst_names = HashSet::new();
    let (v3, sd) = (idx["ENC_VOP3"], idx["VOP3_SDST_ENC"]);
    let mut checked = 0;
    for inst in all_instructions() {
        assert!(inst_names.insert(inst.name), "{} is listed twice", inst.name);
        assert!(!inst.encs.is_empty(), "{} has no encoding", inst.name);
        if inst.flags & F_CONDITIONAL != 0 {
            assert!(inst.flags & F_BRANCH != 0, "{}: conditional but no branch", inst.name);
        }
        assert_eq!(inst.flags & F_TERMINATOR != 0, inst.name.starts_with("S_ENDPGM"), "{} terminator flag", inst.name);
        for e in inst.encs {
            // VOP3 with SDST overlays VOP3: key both by the VOP3 index
            let key = (if e.encoding == sd { v3 } else { e.encoding }, e.opcode);
            if let Some(other) = seen.insert(key, inst.name) {
                panic!("{} and {other} share encoding {} opcode {:#x}", inst.name, e.encoding, e.opcode);
            }
            if e.operands.iter().any(|o| o.ty == "OPR_LABEL") {
                assert!(inst.flags & F_BRANCH != 0 && inst.flags & F_INDIRECT == 0, "{}: label but not a direct branch", inst.name);
            }
            let enc = *names.get(&e.encoding).unwrap_or_else(|| panic!("{}: encoding {} not reached", inst.name, e.encoding));
            let l = inst.always_literal.then_some(0xC0DE_F00D);
            let code = encode(inst.name, enc, &[], l).unwrap_or_else(|err| panic!("encode {} {enc}: {err}", inst.name));
            let d = decode(&code, 0).unwrap_or_else(|err| panic!("{} {enc} {:02x?}: {err:?}", inst.name, code));
            assert_eq!((d.inst.name, d.enc.name, d.ienc.opcode), (inst.name, enc, e.opcode), "{:02x?}", code);
            assert_eq!(d.len, class_len(enc) + if l.is_some() { 4 } else { 0 }, "{} {enc}", inst.name);
            assert_eq!((d.len, d.literal), (code.len(), l), "{} {enc}", inst.name);
            checked += 1;
        }
    }
    assert!(checked >= 1400, "only {checked} table entries");
}

/// A stream of random instructions of every class, then `s_endpgm`, then more code: decoding
/// from each boundary lands on the next, the walk ends on the terminator, and `disassemble`
/// finds the same boundaries, stops after `s_endpgm` when asked to and goes on when not.
#[test]
fn program_walker_finds_every_boundary() {
    let pool = &CASES[..CASES.len() - 1];
    let endpgm = CASES.last().unwrap();
    let mut rng = Lcg(0x5EED_0002);
    for round in 0..20 {
        let mut code = Vec::new();
        let mut starts = Vec::new();
        for _ in 0..(20 + rng.next() % 100) {
            let k = &pool[rng.next() as usize % pool.len()];
            starts.push(code.len());
            code.extend_from_slice(&bytes(k.words));
        }
        let end = code.len();
        starts.push(end);
        code.extend_from_slice(&bytes(endpgm.words));
        let body_len = code.len();
        for k in &pool[..8] {
            code.extend_from_slice(&bytes(k.words));
        }
        let base = 0x8000;
        let (mut off, mut seen) = (0usize, Vec::new());
        loop {
            let d = decode(&code[off..], base + off as u64).unwrap_or_else(|e| panic!("round {round} at {off:#x}: {e:?}"));
            seen.push(off);
            off += d.len;
            if d.inst.flags & F_TERMINATOR != 0 {
                assert_eq!(d.inst.name, "S_ENDPGM");
                break;
            }
        }
        assert_eq!(seen, starts, "round {round}");
        assert_eq!(off, body_len);
        let lines = disassemble(&code, base, true);
        assert_eq!(lines.iter().map(|l| (l.pc - base) as usize).collect::<Vec<_>>(), starts, "round {round}");
        assert!(lines.iter().all(|l| l.ok));
        assert_eq!(lines.iter().map(|l| l.bytes.len()).sum::<usize>(), body_len);
        assert_eq!(lines.last().unwrap().flags & F_TERMINATOR, F_TERMINATOR);
        assert_eq!(disassemble(&code, base, false).len(), lines.len() + 8, "round {round}: no stop");
        assert_eq!(find_programs(&code[..body_len]), vec![(0, body_len)], "round {round}");
    }
}

/// Every prefix shorter than an instruction is `Truncated` (a literal, extension or NSA dword
/// missing included), never a shorter instruction, and nothing panics.
#[test]
fn truncated_prefixes_are_reported() {
    for k in CASES {
        let code = bytes(k.words);
        for n in 0..code.len() {
            let r = catch_unwind(|| decode(&code[..n], 0)).unwrap_or_else(|_| panic!("decode panicked on {:02x?}", &code[..n]));
            assert_eq!(r.err(), Some(DecodeError::Truncated), "{} cut to {n} of {} bytes", k.name, code.len());
        }
    }
}

/// Primary-encoding prefixes of chapter "Microcode Formats" (mask, value), to steer random words
/// into every decoder path.
const PREFIXES: &[(u32, u32)] = &[
    (0xFF80_0000, 0xBE80_0000), (0xFF80_0000, 0xBF00_0000), (0xFF80_0000, 0xBF80_0000), (0xF000_0000, 0xB000_0000),
    (0xC000_0000, 0x8000_0000), (0xFC00_0000, 0xF400_0000), (0xFE00_0000, 0x7E00_0000), (0xFE00_0000, 0x7C00_0000),
    (0x8000_0000, 0x0000_0000), (0xFC00_0000, 0xC800_0000), (0xFC00_0000, 0xD400_0000), (0xFC00_0000, 0xCC00_0000),
    (0xFC00_0000, 0xD800_0000), (0xFC00_0000, 0xE000_0000), (0xFC00_0000, 0xE800_0000), (0xFC00_0000, 0xF000_0000),
    (0xFC00_0000, 0xF800_0000), (0xFC00_0000, 0xDC00_0000), (0, 0),
];

/// Random words never panic the decoder, and whatever decodes is consistent: the dwords are the
/// input's, the length fits the input and only depends on its own bytes, the fixed bits match the
/// encoding, the literal follows the base dwords, SDWA / DPP only on VOP1 / VOP2 / VOPC.
/// `disassemble` and `find_programs` cover random streams without panicking or overlapping.
#[test]
fn random_words_never_panic() {
    let mut rng = Lcg(0xD15A_5E3B);
    let mut ok = 0;
    for i in 0..20_000 {
        let (mask, value) = PREFIXES[i % PREFIXES.len()];
        let n = 1 + rng.next() as usize % 24;
        let mut code: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
        if n >= 4 {
            let w = (u32::from_le_bytes([code[0], code[1], code[2], code[3]]) & !mask) | value;
            code[..4].copy_from_slice(&w.to_le_bytes());
        }
        let Ok(r) = catch_unwind(|| decode(&code, 0)) else { panic!("decode panicked on {code:02x?}") };
        let Ok(d) = r else { continue };
        ok += 1;
        let what = format!("{code:02x?} as {}", d.inst.name);
        assert!(d.len % 4 == 0 && (4..=20).contains(&d.len) && d.len <= code.len(), "{what}: len {}", d.len);
        assert_eq!(bytes(&d.words), &code[..d.len], "{what}");
        let w64 = d.words[0] as u64 | (*d.words.get(1).unwrap_or(&0) as u64) << 32;
        assert_eq!(w64 & d.enc.fixed_mask, d.enc.fixed_value, "{what}");
        if let Some(l) = d.literal {
            assert_eq!(l, d.words[d.enc.bits as usize / 32], "{what}");
        }
        if d.variant != Variant::Plain {
            assert!(matches!(d.enc.name, "ENC_VOP1" | "ENC_VOP2" | "ENC_VOPC"), "{what}: {:?}", d.variant);
        }
        let again = decode(&code[..d.len], 0).unwrap_or_else(|e| panic!("{what}: own bytes give {e:?}"));
        assert_eq!((again.inst.name, again.len), (d.inst.name, d.len), "{what}");
        assert_eq!(decode(&code[..d.len - 1], 0).err(), Some(DecodeError::Truncated), "{what}");
    }
    assert!(ok > 5_000, "only {ok} random words decoded");
    for seed in 0..50 {
        let mut rng = Lcg(seed);
        let n = rng.next() as usize % 2048;
        let blob: Vec<u8> = (0..n).map(|_| if rng.next() % 4 == 0 { 0 } else { rng.next() as u8 }).collect();
        let r = catch_unwind(|| (disassemble(&blob, 0, false), find_programs(&blob)));
        let Ok((lines, progs)) = r else { panic!("panicked on seed {seed}") };
        let mut at = 0u64;
        for l in &lines {
            assert_eq!(l.pc, at, "seed {seed}");
            assert_eq!(l.bytes, blob[at as usize..at as usize + l.bytes.len()], "seed {seed}");
            at += l.bytes.len() as u64;
        }
        assert_eq!(at as usize, n / 4 * 4, "seed {seed}: lines do not cover the stream");
        let mut prev = 0;
        for &(s, len) in &progs {
            assert!(s >= prev && s + len <= n && len >= 4, "seed {seed}: program {s:#x}+{len:#x}");
            prev = s + len;
        }
    }
}

/// Direct branch targets: PC + 4 + SIMM16 * 4 with SIMM16 signed (section "SOPP"), the same for
/// SOPK's s_call_b64; indirect jumps and non-branches have none.
#[test]
fn sopp_branch_targets() {
    let at = |w: u32, pc: u64| branch_target(&decode(&w.to_le_bytes(), pc).unwrap());
    assert_eq!(at(0xBF82_FFFF, 0x100), Some(0x100)); // s_branch -1: to itself
    assert_eq!(at(0xBF82_8000, 0x20_0000), Some(0x20_0004 - 0x2_0000)); // most negative
    assert_eq!(at(0xBF82_7FFF, 0), Some(4 + 0x7FFF * 4));
    assert_eq!(at(0xBF88_0003, 0x100), Some(0x110)); // s_cbranch_execz +3
    assert_eq!(at(0xBF84_0000, 0x40), Some(0x44)); // s_cbranch_scc0 +0: falls through
    assert_eq!(at(0xB000_0000 | 0x16 << 23 | 30 << 16 | 0x0010, 0x200), Some(0x244)); // s_call_b64 s[30:31]
    assert_eq!(at(0xBE80_0000 | 0x20 << 8 | 30, 0x100), None); // s_setpc_b64 s[30:31]
    assert_eq!(at(0xBF9F_0000, 0x100), None); // s_code_end
    assert_eq!(at(0xBF81_0000, 0x100), None); // s_endpgm
    assert_eq!(at(0x0602_0501, 0x100), None); // v_add_f32
}

/// `find_programs` splits a blob at each `s_endpgm`: two programs back to back come out as two.
#[test]
fn find_programs_splits_at_program_ends() {
    // s_mov_b32 s5, s7; v_add_f32 v1, v2, v3; s_endpgm
    let a = bytes(&[0xBE85_0307, 0x0602_0702, 0xBF81_0000]);
    // v_fma_f32 v0, v1, v2, 0x12345678; s_endpgm
    let b = bytes(&[0xD54B_0000, 0x03FE_0501, L, 0xBF81_0000]);
    let blob = [a.clone(), b.clone()].concat();
    assert_eq!(find_programs(&blob), vec![(0, a.len()), (a.len(), b.len())]);
}

