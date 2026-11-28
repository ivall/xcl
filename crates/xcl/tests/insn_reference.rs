//! The console-only instruction decoder and the SSE4a bit-field helpers (`insn.rs`) against
//! reference models written from AMD's manuals: EXTRQ / INSERTQ from the AMD64 Architecture
//! Programmer's Manual, Volume 4 (128-bit media instructions), and the ModRM / SIB operand
//! lengths from Volume 3, section 1.4 (ModRM and SIB bytes). The exception handler emulates
//! and steps over what `decode` returns, so a wrong length or register runs the title off the
//! rails on a PC without these instructions.

use xcl::insn::{decode, extrq, hex, insertq, ConsoleOnly};

/// A fixed LCG, so failures repeat.
fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s ^ (s >> 29)
    }
}

/// EXTRQ bit by bit (APM vol. 4, EXTRQ): bits idx..idx+len-1 of the source moved to bit 0,
/// the rest zero; a length of 0 means 64.
fn extrq_ref(v: u64, len: u8, idx: u8) -> u64 {
    let len = if len & 63 == 0 { 64 } else { (len & 63) as u32 };
    let idx = (idx & 63) as u32;
    (0..len).filter(|&b| idx + b < 64 && v >> (idx + b) & 1 == 1).fold(0, |acc, b| acc | 1 << b)
}

/// INSERTQ bit by bit (APM vol. 4, INSERTQ): the low len bits of the source replace bits
/// idx..idx+len-1 of the destination; the other destination bits stay.
fn insertq_ref(dst: u64, src: u64, len: u8, idx: u8) -> u64 {
    let len = if len & 63 == 0 { 64 } else { (len & 63) as u32 };
    let idx = (idx & 63) as u32;
    let mut out = dst;
    for b in 0..len {
        if idx + b < 64 {
            out = (out & !(1 << (idx + b))) | (src >> b & 1) << (idx + b);
        }
    }
    out
}

/// Every defined (length, index) pair, which is length + index <= 64 (APM: a larger field is
/// undefined), on edge values and seeded random ones. The helpers mask both to six bits, so
/// the upper two bits of the immediates must not matter either.
#[test]
fn sse4a_fields_match_the_manual_for_every_defined_length_and_index() {
    let mut r = lcg(0x5EE4A);
    let edges = [0u64, u64::MAX, 1, 1 << 63, 0x8000_0000_0000_0001, 0xAAAA_AAAA_AAAA_AAAA, 0x0123_4567_89AB_CDEF];
    for len in 0u8..64 {
        let width = if len == 0 { 64 } else { len as u32 };
        for idx in 0u8..64 {
            if width + idx as u32 > 64 {
                continue;
            }
            let values: Vec<u64> = edges.iter().copied().chain((0..8).map(|_| r())).collect();
            for (k, v) in values.into_iter().enumerate() {
                let d = if k % 2 == 0 { r() } else { !v };
                assert_eq!(extrq(v, len, idx), extrq_ref(v, len, idx), "extrq({v:#x}, len {len}, idx {idx})");
                assert_eq!(insertq(d, v, len, idx), insertq_ref(d, v, len, idx), "insertq({d:#x}, {v:#x}, len {len}, idx {idx})");
                assert_eq!(extrq(v, len | 0xC0, idx | 0xC0), extrq(v, len, idx), "upper immediate bits, len {len} idx {idx}");
                assert_eq!(insertq(d, v, len | 0x40, idx | 0x80), insertq(d, v, len, idx), "upper immediate bits, len {len} idx {idx}");
            }
        }
    }
}

/// A field past bit 63 is undefined on the CPU; XCL must still give an answer and not panic
/// (an overflowing shift is a panic in a debug build).
#[test]
fn sse4a_fields_past_bit_63_do_not_panic() {
    for len in 0u8..=255 {
        for idx in 0u8..=255 {
            let _ = extrq(u64::MAX, len, idx);
            let _ = insertq(0, u64::MAX, len, idx);
        }
    }
}

/// The operand length of a ModRM byte with a memory operand (APM vol. 3, 1.4): SIB when
/// rm = 100b; disp32 for mod 00 with rm = 101b (RIP-relative) or a SIB base of 101b; disp8 for
/// mod 01, disp32 for mod 10. REX.B does not change either special case.
fn modrm_mem_len(modrm: u8, sib: u8) -> usize {
    let (md, rm) = (modrm >> 6, modrm & 7);
    let sib_len = usize::from(rm == 4);
    let disp = match md {
        0 if rm == 5 || (rm == 4 && sib & 7 == 5) => 4,
        0 => 0,
        1 => 1,
        _ => 4,
    };
    1 + sib_len + disp
}

/// CLWB (66 0F AE /6) for every memory ModRM byte, every SIB base and with or without REX.B:
/// the length the exception handler steps over is the instruction's real length.
#[test]
fn clwb_steps_over_every_memory_operand_form() {
    for rex in [None, Some(0x41u8), Some(0x49)] {
        for modrm in (0u8..=255).filter(|m| (m >> 3) & 7 == 6 && m >> 6 != 3) {
            for sib in [0x00u8, 0x24, 0x25, 0x05, 0xE5, 0x9C] {
                let mut code = vec![0x66];
                code.extend(rex);
                code.extend([0x0F, 0xAE, modrm, sib, 0x11, 0x22, 0x33, 0x44, 0x55]);
                let want = code.len() - 9 + modrm_mem_len(modrm, sib) + 2;
                assert_eq!(decode(&code), Some((ConsoleOnly::Clwb, want)), "{}", hex(&code));
                // the operand cut short by one byte: nothing to step over
                assert_eq!(decode(&code[..want - 1]), None, "truncated {}", hex(&code[..want - 1]));
            }
        }
    }
}

/// The ModRM reg field /6 with a register operand is not CLWB (66 0F AE F0+r is TPAUSE
/// territory and must reach the title as it is), and neither is any other /r.
#[test]
fn only_clwb_decodes_under_66_0f_ae() {
    for modrm in 0u8..=255 {
        let code = [0x66, 0x0F, 0xAE, modrm, 0x24, 0, 0, 0, 0, 0];
        let clwb = (modrm >> 3) & 7 == 6 && modrm >> 6 != 3;
        assert_eq!(decode(&code).is_some(), clwb, "{}", hex(&code));
    }
}

/// RDPID (F3 0F C7 /7, register form) names its register in CONTEXT order with REX.B; REX.W
/// and REX.R do not change it, and the memory form is something else.
#[test]
fn rdpid_takes_its_register_from_rm_and_rex_b() {
    for rex in [None, Some(0x40u8), Some(0x41), Some(0x48), Some(0x4C), Some(0x4D)] {
        for rm in 0u8..8 {
            let mut code = vec![0xF3];
            code.extend(rex);
            code.extend([0x0F, 0xC7, 0xF8 | rm]);
            let reg = rm | (rex.unwrap_or(0) & 1) << 3;
            assert_eq!(decode(&code), Some((ConsoleOnly::Rdpid(reg), code.len())), "{}", hex(&code));
        }
    }
    for modrm in (0u8..0xC0).filter(|m| (m >> 3) & 7 == 7) {
        assert_eq!(decode(&[0xF3, 0x0F, 0xC7, modrm, 0, 0, 0, 0, 0]), None, "memory form {modrm:#x}");
    }
}

/// SSE4a register numbers for all ModRM register forms and all 16 REX prefixes (APM vol. 4:
/// EXTRQ xmm1, imm8, imm8 is 66 0F 78 /0 ib ib; EXTRQ xmm1, xmm2 is 66 0F 79 /r; INSERTQ is
/// F2 0F 78 /r ib ib and F2 0F 79 /r). REX.R extends the reg field, REX.B the rm field; a memory
/// operand is #UD and must not be emulated.
#[test]
fn sse4a_registers_for_every_rex_and_modrm() {
    for rex in 0x40u8..=0x4F {
        let r = (rex & 4) << 1;
        let b = (rex & 1) << 3;
        for modrm in 0u8..=255 {
            let (reg, rm) = ((modrm >> 3) & 7 | r, modrm & 7 | b);
            let reg_form = modrm >> 6 == 3;
            let imm = [0x0Cu8, 0x14];
            let at = |p: u8, op: u8, tail: &[u8]| {
                let mut c = vec![p, rex, 0x0F, op, modrm];
                c.extend_from_slice(tail);
                c
            };

            let c = at(0x66, 0x78, &imm);
            let want = (reg_form && (modrm >> 3) & 7 == 0).then_some((ConsoleOnly::Extrq { dst: rm, src: None, len: 0x0C, idx: 0x14 }, 7));
            assert_eq!(decode(&c), want, "{}", hex(&c));

            let c = at(0x66, 0x79, &[]);
            let want = reg_form.then_some((ConsoleOnly::Extrq { dst: reg, src: Some(rm), len: 0, idx: 0 }, 5));
            assert_eq!(decode(&c), want, "{}", hex(&c));

            let c = at(0xF2, 0x78, &imm);
            let want = reg_form.then_some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: false, len: 0x0C, idx: 0x14 }, 7));
            assert_eq!(decode(&c), want, "{}", hex(&c));

            let c = at(0xF2, 0x79, &[]);
            let want = reg_form.then_some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: true, len: 0, idx: 0 }, 5));
            assert_eq!(decode(&c), want, "{}", hex(&c));
        }
    }
}

/// The CLZERO loop's end register for each of the 16 registers XCL can compare against
/// (`cmp rax, r64` is REX.W [REX.R] 39 /r with rm = rax), and the single CLZERO when the bytes
/// after it are not exactly that loop.
#[test]
fn clzero_loop_end_register_for_every_register() {
    for end in 0u8..16 {
        let rex = 0x48 | (end >> 3) << 2;
        let code = [0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, rex, 0x39, 0xC0 | (end & 7) << 3, 0x72, 0xF4];
        let want = if end == 0 { (ConsoleOnly::Clzero, 3) } else { (ConsoleOnly::ClzeroLoop { end_reg: end }, 12) };
        assert_eq!(decode(&code), Some(want), "{}", hex(&code));
        // a different step, a different branch target, or a jae: one CLZERO only
        for (at, byte) in [(6, 0x80), (11, 0xF2), (10, 0x73), (4, 0x81)] {
            let mut other = code;
            other[at] = byte;
            assert_eq!(decode(&other), Some((ConsoleOnly::Clzero, 3)), "{}", hex(&other));
        }
        // the loop cut short is one CLZERO too
        for n in 3..12 {
            assert_eq!(decode(&code[..n]), Some((ConsoleOnly::Clzero, 3)), "{}", hex(&code[..n]));
        }
    }
}

/// Whatever the bytes, `decode` never panics and never steps over more bytes than it was given;
/// a recognized instruction keeps its decoding when bytes are appended after it.
#[test]
fn decode_is_total_and_prefix_stable_on_random_bytes() {
    let mut r = lcg(0xC0DE);
    // the opcode bytes the decoder knows, so random input reaches every arm
    let heads: [&[u8]; 9] = [&[0x0F, 0x01], &[0x66, 0x0F, 0xAE], &[0xF3, 0x0F, 0xC7], &[0x66, 0x0F, 0x78], &[0x66, 0x0F, 0x79], &[0xF2, 0x0F, 0x78], &[0xF2, 0x0F, 0x79], &[0x0F, 0x0B], &[]];
    let prefixes = [0x66u8, 0xF2, 0xF3, 0x67, 0x2E, 0x41, 0x48, 0x4C, 0x65];
    for case in 0..20_000 {
        let mut code = Vec::new();
        for _ in 0..r() % 4 {
            code.push(prefixes[(r() % prefixes.len() as u64) as usize]);
        }
        code.extend_from_slice(heads[case % heads.len()]);
        let tail = (r() % 14) as usize;
        code.extend((0..tail).map(|_| r() as u8));
        let got = std::panic::catch_unwind(|| decode(&code)).unwrap_or_else(|_| panic!("decode panicked on {}", hex(&code)));
        if let Some((insn, len)) = got {
            assert!(len >= 2 && len <= code.len(), "{insn:?} length {len} for {}", hex(&code));
            let mut longer = code.clone();
            longer.extend([0x90; 16]);
            assert_eq!(decode(&longer), got, "appending bytes changed the decoding of {}", hex(&code));
        }
        for n in 0..code.len() {
            if let Some((_, len)) = decode(&code[..n]) {
                assert!(len <= n, "length {len} past {n} bytes of {}", hex(&code));
            }
        }
    }
}
