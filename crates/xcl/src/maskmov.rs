//! AVX / AVX2 masked stores (VMASKMOVPS, VMASKMOVPD, VPMASKMOVD, VPMASKMOVQ with a memory
//! destination) of a title, replaced when its image is loaded on a host whose translator
//! cannot resume them after a fault.
//!
//! Rosetta 2 runs a masked store an element at a time and can restart it only from its first
//! element: a fault there is delivered and resumed like any other, a fault on a later element
//! ends the process ("rosetta error: unexpectedly need to EmulateForward on a synchronous
//! exception"). That is a store which starts on a page that can be written and goes on into one
//! that cannot – what a write watch kept by page protection (Wine on macOS) makes of every
//! page boundary inside a buffer the title fills with such stores.
//!
//! Each store becomes `call helper` (five bytes; these instructions have at least five) and
//! NOPs. The helper stores the selected elements one at a time with ordinary `mov`s, which
//! fault and resume as any store does, and changes nothing else: no register, no flag
//! (it has no instruction that writes one), no byte of an element whose mask bit is clear –
//! such an element is not touched at all, so a store whose tail is masked out may end at the
//! last byte of a mapping as before.
//!
//! This module holds the parts that do not depend on Windows. So far:
//!
//! * `insn_len` – the length of an x86-64 instruction, to walk a function from its start.
//! * `decode` – the four instructions with any memory operand.
//!
//! The helpers a site calls, the plan that finds the sites of an image and the loader step that
//! applies it come next; nothing calls this module yet.

/// Length of the x86-64 instruction at the start of `c`. `None`: the bytes run out, or it is
/// something this walk does not know and must not guess at – EVEX, XOP, 3DNow!, an opcode
/// that is not valid in 64-bit mode (which in a function means data).
pub fn insn_len(c: &[u8]) -> Option<usize> {
    let (mut i, mut p66, mut p67, mut pf2, mut legacy) = (0usize, false, false, false, false);
    loop {
        match *c.get(i)? {
            0x66 => p66 = true,
            0x67 => p67 = true,
            0xF2 => pf2 = true,
            0xF3 | 0xF0 => legacy = true,
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 => {}
            _ => break,
        }
        i += 1;
        if i > 14 {
            return None;
        }
    }
    let (mut rex, mut rex_w) = (false, false);
    if *c.get(i)? & 0xF0 == 0x40 {
        rex = true;
        rex_w = *c.get(i)? & 8 != 0;
        i += 1;
    }
    let op = *c.get(i)?;
    i += 1;
    // an immediate of the operand size, at most four bytes
    let immz = if rex_w || !p66 { 4 } else { 2 };
    // (has ModRM, immediate bytes)
    let (modrm, mut imm): (bool, usize) = match op {
        0xC4 | 0xC5 => {
            // VEX takes no REX, LOCK or 66 / F2 / F3 in front
            if rex || p66 || pf2 || legacy {
                return None;
            }
            let map = if op == 0xC5 {
                i += 1;
                1
            } else {
                i += 2;
                *c.get(i - 2)? & 0x1F
            };
            let o = *c.get(i)?;
            i += 1;
            match map {
                1 => (o != 0x77, matches!(o, 0x70..=0x73 | 0xC2 | 0xC4..=0xC6) as usize),
                2 => (true, 0),
                3 => (true, 1),
                _ => return None,
            }
        }
        0x0F => {
            let o = *c.get(i)?;
            i += 1;
            match o {
                0x38 | 0x3A => {
                    c.get(i)?;
                    i += 1;
                    (true, (o == 0x3A) as usize)
                }
                0x05..=0x09 | 0x0B | 0x30..=0x35 | 0x37 | 0x77 | 0xA0..=0xA2 | 0xA8..=0xAA | 0xC8..=0xCF => (false, 0),
                0x80..=0x8F => (false, 4),
                0x70..=0x73 | 0xA4 | 0xAC | 0xBA | 0xC2 | 0xC4..=0xC6 => (true, 1),
                // SSE4a EXTRQ / INSERTQ with length and index bytes
                0x78 if p66 || pf2 => (true, 2),
                0x04 | 0x0A | 0x0C | 0x0E | 0x0F | 0x24..=0x27 | 0x36 | 0x39 | 0x3B..=0x3F | 0x7A | 0x7B | 0xA6 | 0xA7 | 0xFF => return None,
                _ => (true, 0),
            }
        }
        0x00..=0x3F => match op & 7 {
            0..=3 => (true, 0),
            4 => (false, 1),
            5 => (false, immz),
            _ => return None,
        },
        0x50..=0x5F | 0x6C..=0x6F | 0x90..=0x99 | 0x9B..=0x9F | 0xA4..=0xA7 | 0xAA..=0xAF => (false, 0),
        0xC3 | 0xC9 | 0xCB | 0xCC | 0xCF | 0xD7 | 0xEC..=0xEF | 0xF1 | 0xF4 | 0xF5 | 0xF8..=0xFD => (false, 0),
        // MOV to or from a segment register: there are six
        0x8C | 0x8E if (*c.get(i)? >> 3) & 7 > 5 => return None,
        0x63 | 0x84..=0x8E | 0xD0..=0xD3 | 0xD8..=0xDF | 0xFE | 0xFF => (true, 0),
        // POP r/m; any other /r is XOP
        0x8F if *c.get(i)? & 0x38 == 0 => (true, 0),
        0x68 | 0xA9 => (false, immz),
        0x69 | 0x81 | 0xC7 => (true, immz),
        0x6A | 0x70..=0x7F | 0xA8 | 0xB0..=0xB7 | 0xCD | 0xE0..=0xE7 | 0xEB => (false, 1),
        0x6B | 0x80 | 0x83 | 0xC0 | 0xC1 | 0xC6 => (true, 1),
        0xA0..=0xA3 => (false, if p67 { 4 } else { 8 }),
        0xB8..=0xBF => (false, if rex_w { 8 } else { immz }),
        0xC2 | 0xCA => (false, 2),
        0xC8 => (false, 3),
        0xE8 | 0xE9 => (false, 4),
        // TEST r/m, imm is /0 and /1 of the group
        0xF6 => (true, (*c.get(i)? & 0x30 == 0) as usize),
        0xF7 => (true, if *c.get(i)? & 0x30 == 0 { immz } else { 0 }),
        _ => return None,
    };
    if modrm {
        let m = *c.get(i)?;
        i += 1;
        let (md, rm) = (m >> 6, m & 7);
        if md != 3 {
            let mut base = rm;
            if rm == 4 {
                base = *c.get(i)? & 7;
                i += 1;
            }
            imm += match md {
                0 if base == 5 => 4,
                0 => 0,
                1 => 1,
                _ => 4,
            };
        }
    }
    i += imm;
    (i <= 15 && i <= c.len()).then_some(i)
}

/// Which of the four instructions: the element size and what the log calls it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// VEX.66.0F38.W0 2E /r: VMASKMOVPS m, mask, src
    Ps,
    /// VEX.66.0F38.W0 2F /r: VMASKMOVPD m, mask, src
    Pd,
    /// VEX.66.0F38.W0 8E /r: VPMASKMOVD m, mask, src
    D,
    /// VEX.66.0F38.W1 8E /r: VPMASKMOVQ m, mask, src
    Q,
}

impl Kind {
    pub const ALL: [Kind; 4] = [Kind::Ps, Kind::Pd, Kind::D, Kind::Q];

    pub fn name(self) -> &'static str {
        match self {
            Kind::Ps => "VMASKMOVPS",
            Kind::Pd => "VMASKMOVPD",
            Kind::D => "VPMASKMOVD",
            Kind::Q => "VPMASKMOVQ",
        }
    }

    /// Bytes of one element.
    pub fn element(self) -> usize {
        match self {
            Kind::Ps | Kind::D => 4,
            Kind::Pd | Kind::Q => 8,
        }
    }
}

/// The memory operand: base + index * 2^scale + disp, or the next instruction's address + disp.
/// Registers are the hardware's numbers (4 RSP, 8–15 R8–R15).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mem {
    pub base: Option<u8>,
    pub index: Option<u8>,
    pub scale: u8,
    pub disp: i32,
    pub rip: bool,
}

/// A masked store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Store {
    pub kind: Kind,
    /// Length in bytes (5..=10).
    pub len: u8,
    /// 256 bits (YMM); else 128 (XMM).
    pub wide: bool,
    /// The register stored from (ModRM reg) and the mask register (VEX.vvvv).
    pub src: u8,
    pub mask: u8,
    pub mem: Mem,
}

impl Store {
    /// Number of elements.
    pub fn elements(&self) -> usize {
        (if self.wide { 32 } else { 16 }) / self.kind.element()
    }
}

/// The first five bytes of every masked store: three-byte VEX with map 0F38 and prefix 66, one
/// of the three opcodes, a ModRM byte with a memory operand.
pub fn is_candidate(c: &[u8]) -> bool {
    matches!(c, [0xC4, b1, b2, 0x2E | 0x2F | 0x8E, m, ..] if b1 & 0x1F == 2 && b2 & 3 == 1 && m >> 6 != 3)
}

/// Decodes the masked store at the start of `c` (`None`: not one, or cut short). A prefix in
/// front (a segment override, 32-bit addressing) makes it another instruction start, which
/// this does not take.
pub fn decode(c: &[u8]) -> Option<Store> {
    if !is_candidate(c) {
        return None;
    }
    let (b1, b2, m) = (c[1], c[2], c[4]);
    let (r, x, b, w) = ((!b1 >> 7) & 1, (!b1 >> 6) & 1, (!b1 >> 5) & 1, b2 >> 7);
    let kind = match (c[3], w) {
        (0x2E, 0) => Kind::Ps,
        (0x2F, 0) => Kind::Pd,
        (0x8E, 0) => Kind::D,
        (0x8E, _) => Kind::Q,
        _ => return None, // W1 of the floating-point forms is not an instruction
    };
    let (md, rm) = (m >> 6, m & 7);
    let mut i = 5;
    let (mut base, mut index, mut scale, mut rip) = (None, None, 0, false);
    let mut disp_len = match md {
        1 => 1,
        2 => 4,
        _ => 0,
    };
    if rm == 4 {
        let sib = *c.get(i)?;
        i += 1;
        scale = sib >> 6;
        let ix = ((sib >> 3) & 7) | x << 3;
        if ix != 4 {
            index = Some(ix); // index 4 without VEX.X: none (R12 is an index)
        }
        if sib & 7 == 5 && md == 0 {
            disp_len = 4; // disp32 without a base
        } else {
            base = Some((sib & 7) | b << 3);
        }
    } else if rm == 5 && md == 0 {
        rip = true;
        disp_len = 4;
    } else {
        base = Some(rm | b << 3);
    }
    let disp = match disp_len {
        1 => *c.get(i)? as i8 as i32,
        4 => i32::from_le_bytes([*c.get(i)?, *c.get(i + 1)?, *c.get(i + 2)?, *c.get(i + 3)?]),
        _ => 0,
    };
    i += disp_len;
    Some(Store { kind, len: i as u8, wide: b2 & 4 != 0, src: ((m >> 3) & 7) | r << 3, mask: (!b2 >> 3) & 15, mem: Mem { base, index, scale, disp, rip } })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lengths as an assembler gives them, for the kinds of instruction a function has on the
    /// way to a masked store – and `None` for what must stop a walk.
    #[test]
    fn instruction_lengths() {
        let cases: &[(&[u8], usize)] = &[
            (&[0x90], 1),                                                       // nop
            (&[0xC3], 1),                                                       // ret
            (&[0x55], 1),                                                       // push rbp
            (&[0x41, 0x57], 2),                                                 // push r15
            (&[0x48, 0x83, 0xEC, 0x28], 4),                                     // sub rsp, 0x28
            (&[0x48, 0x81, 0xEC, 0x88, 0x00, 0x00, 0x00], 7),                   // sub rsp, 0x88
            (&[0x48, 0x8B, 0x05, 1, 2, 3, 4], 7),                               // mov rax, [rip + disp32]
            (&[0x48, 0x8B, 0x44, 0x24, 0x08], 5),                               // mov rax, [rsp + 8]
            (&[0x4A, 0x8D, 0x04, 0xC5, 0, 0, 0, 0], 8),                         // lea rax, [r8 * 8 + disp32]
            (&[0x48, 0xB8, 1, 2, 3, 4, 5, 6, 7, 8], 10),                        // mov rax, imm64
            (&[0xB8, 1, 2, 3, 4], 5),                                           // mov eax, imm32
            (&[0x66, 0xB8, 1, 2], 4),                                           // mov ax, imm16
            (&[0x66, 0xC7, 0x00, 1, 2], 5),                                     // mov word [rax], imm16
            (&[0xC7, 0x04, 0x25, 0, 0, 0, 0, 1, 2, 3, 4], 11),                  // mov dword [0], imm32
            (&[0xF6, 0xC1, 0x01], 3),                                           // test cl, 1
            (&[0xF6, 0xD9], 2),                                                 // neg cl
            (&[0xF7, 0xC1, 1, 2, 3, 4], 6),                                     // test ecx, imm32
            (&[0x48, 0xF7, 0xD8], 3),                                           // neg rax
            (&[0xE8, 1, 2, 3, 4], 5),                                           // call
            (&[0x74, 0x10], 2),                                                 // je
            (&[0x0F, 0x84, 1, 2, 3, 4], 6),                                     // je rel32
            (&[0xF2, 0xC3], 2),                                                 // bnd ret
            (&[0xFF, 0x15, 1, 2, 3, 4], 6),                                     // call [rip + disp32]
            (&[0x48, 0xA1, 1, 2, 3, 4, 5, 6, 7, 8], 10),                        // mov rax, [moffs64]
            (&[0x66, 0x66, 0x2E, 0x0F, 0x1F, 0x84, 0x00, 0, 0, 0, 0], 11),      // nop word cs:[rax + rax + disp32]
            (&[0x0F, 0x1F, 0x44, 0x00, 0x00], 5),                               // nop dword [rax + rax + 0]
            (&[0x0F, 0xBA, 0xE0, 0x07], 4),                                     // bt eax, 7
            (&[0x66, 0x0F, 0x3A, 0x0F, 0xC1, 0x08], 6),                         // palignr xmm0, xmm1, 8
            (&[0xF2, 0x0F, 0x78, 0xC1, 0x08, 0x10], 6),                         // insertq xmm0, xmm1, 8, 16
            (&[0x0F, 0x01, 0xFC], 3),                                           // clzero
            (&[0xF3, 0x48, 0x0F, 0xB8, 0xC1], 5),                               // popcnt rax, rcx
            (&[0xC5, 0xF8, 0x77], 3),                                           // vzeroupper
            (&[0xC5, 0xFE, 0x7F, 0x04, 0x24], 5),                               // vmovdqu [rsp], ymm0
            (&[0xC5, 0xF9, 0x70, 0xC1, 0x1B], 5),                               // vpshufd xmm0, xmm1, 0x1B
            (&[0xC4, 0xE2, 0x7D, 0x21, 0x05, 1, 2, 3, 4], 9),                   // vpmovsxbd ymm0, [rip + disp32]
            (&[0xC4, 0xE3, 0x7D, 0x39, 0xC1, 0x01], 6),                         // vextracti128 xmm1, ymm0, 1
            (&[0xC4, 0xE3, 0x75, 0x4C, 0xC2, 0x30], 6),                         // vpblendvb ymm0, ymm1, ymm2, ymm3
            (&[0xC4, 0xE2, 0x4D, 0x8E, 0x04, 0x90], 6),                         // vpmaskmovd [rax + 4 * rdx], ymm6, ymm0
            (&[0xC4, 0xE2, 0xF8, 0xF7, 0xC1], 5),                               // bextr rax, rcx, rax
        ];
        for (code, len) in cases {
            assert_eq!(insn_len(code), Some(*len), "{code:02x?}");
            // followed by other bytes it is no longer, cut short it is nothing
            let mut more = code.to_vec();
            more.extend_from_slice(&[0xC4; 16]);
            assert_eq!(insn_len(&more), Some(*len), "{code:02x?} with a tail");
            assert_eq!(insn_len(&code[..len - 1]), None, "{code:02x?} cut short");
        }
        // EVEX, an opcode 64-bit mode does not have, XOP, 3DNow!, a far jump, REX before VEX, and
        // the first word of a jump table a function kept in its code (8E /6: no such register)
        for code in [&[0x62, 0xF1, 0x7C, 0x48, 0x10, 0x00][..], &[0x06], &[0x8F, 0xE8, 0x78, 0xC2, 0xC1, 0x04], &[0x0F, 0x0F, 0xC1, 0x9E], &[0xEA], &[0x41, 0xC5, 0xF8, 0x77], &[0x8E, 0xB2, 0x5E, 0x05, 0x98, 0xB2]] {
            assert_eq!(insn_len(code), None, "{code:02x?}");
        }
    }

    const NO_MEM: Mem = Mem { base: None, index: None, scale: 0, disp: 0, rip: false };

    /// Every shape of the instruction a retail game has (1,130 sites: lengths 5, 6, 7, 9 and
    /// 10; RSP, R12, R13 as bases; XMM forms), plus the ones it has not: RIP-relative, an
    /// address without a base.
    #[test]
    fn decodes_the_masked_stores() {
        let mem = |base, index, scale, disp| Mem { base, index, scale, disp, rip: false };
        let cases: &[(&[u8], Kind, bool, u8, u8, Mem)] = &[
            // vpmaskmovq [rsp + r8 + 0x40], ymm7, ymm8
            (&[0xC4, 0x22, 0xC5, 0x8E, 0x44, 0x04, 0x40], Kind::Q, true, 8, 7, mem(Some(4), Some(8), 0, 0x40)),
            // vmaskmovps [rcx + 4 * rdx], ymm4, ymm5
            (&[0xC4, 0xE2, 0x5D, 0x2E, 0x2C, 0x91], Kind::Ps, true, 5, 4, mem(Some(1), Some(2), 2, 0)),
            // vpmaskmovd [r12 + 4 * r8], ymm5, ymm6
            (&[0xC4, 0x82, 0x55, 0x8E, 0x34, 0x84], Kind::D, true, 6, 5, mem(Some(12), Some(8), 2, 0)),
            // vpmaskmovd [rcx], ymm7, ymm6
            (&[0xC4, 0xE2, 0x45, 0x8E, 0x31], Kind::D, true, 6, 7, mem(Some(1), None, 0, 0)),
            // vpmaskmovq [rax + 8 * r9], ymm4, ymm5
            (&[0xC4, 0xA2, 0xDD, 0x8E, 0x2C, 0xC8], Kind::Q, true, 5, 4, mem(Some(0), Some(9), 3, 0)),
            // vpmaskmovd [rsi + rcx], ymm12, ymm13
            (&[0xC4, 0x62, 0x1D, 0x8E, 0x2C, 0x0E], Kind::D, true, 13, 12, mem(Some(6), Some(1), 0, 0)),
            // vpmaskmovd [rsp + 4 * rcx + 0x4e0], ymm7, ymm0
            (&[0xC4, 0xE2, 0x45, 0x8E, 0x84, 0x8C, 0xE0, 0x04, 0x00, 0x00], Kind::D, true, 0, 7, mem(Some(4), Some(1), 2, 0x4E0)),
            // vpmaskmovd [rsp + 0x90], ymm1, ymm0;  vpmaskmovd [rsp], ymm1, ymm0
            (&[0xC4, 0xE2, 0x75, 0x8E, 0x84, 0x24, 0x90, 0x00, 0x00, 0x00], Kind::D, true, 0, 1, mem(Some(4), None, 0, 0x90)),
            (&[0xC4, 0xE2, 0x75, 0x8E, 0x04, 0x24], Kind::D, true, 0, 1, mem(Some(4), None, 0, 0)),
            // vmaskmovpd [rdx + 8 * rcx], ymm5, ymm6
            (&[0xC4, 0xE2, 0x55, 0x2F, 0x34, 0xCA], Kind::Pd, true, 6, 5, mem(Some(2), Some(1), 3, 0)),
            // vpmaskmovq [r10 + 8 * r11 - 8], ymm4, ymm5
            (&[0xC4, 0x82, 0xDD, 0x8E, 0x6C, 0xDA, 0xF8], Kind::Q, true, 5, 4, mem(Some(10), Some(11), 3, -8)),
            // vpmaskmovd [r13], xmm2, xmm1 (R13 needs a displacement byte);  vpmaskmovd [rbp + 0x10], xmm15, xmm9
            (&[0xC4, 0xC2, 0x69, 0x8E, 0x4D, 0x00], Kind::D, false, 1, 2, mem(Some(13), None, 0, 0)),
            (&[0xC4, 0x62, 0x01, 0x8E, 0x4D, 0x10], Kind::D, false, 9, 15, mem(Some(5), None, 0, 0x10)),
            // vpmaskmovd [rip + 0x1000], ymm1, ymm0
            (&[0xC4, 0xE2, 0x75, 0x8E, 0x05, 0x00, 0x10, 0x00, 0x00], Kind::D, true, 0, 1, Mem { disp: 0x1000, rip: true, ..NO_MEM }),
            // vmaskmovps [4 * rcx + 0x12345678], xmm1, xmm0;  vpmaskmovq [0x12345678], ymm1, ymm0
            (&[0xC4, 0xE2, 0x71, 0x2E, 0x04, 0x8D, 0x78, 0x56, 0x34, 0x12], Kind::Ps, false, 0, 1, mem(None, Some(1), 2, 0x1234_5678)),
            (&[0xC4, 0xE2, 0xF5, 0x8E, 0x04, 0x25, 0x78, 0x56, 0x34, 0x12], Kind::Q, true, 0, 1, mem(None, None, 0, 0x1234_5678)),
        ];
        for (code, kind, wide, src, mask, mem) in cases {
            let want = Store { kind: *kind, len: code.len() as u8, wide: *wide, src: *src, mask: *mask, mem: *mem };
            assert_eq!(decode(code), Some(want), "{code:02x?}");
            assert_eq!(insn_len(code), Some(code.len()), "{code:02x?}");
            assert_eq!(decode(&code[..code.len() - 1]), None, "{code:02x?} cut short");
        }
        assert_eq!(decode(&cases[0].0[..]).unwrap().elements(), 4);
        assert_eq!(decode(&cases[11].0[..]).unwrap().elements(), 4);
        assert_eq!(decode(&cases[14].0[..]).unwrap().elements(), 4);
        assert_eq!(decode(&cases[15].0[..]).unwrap().elements(), 4);
        assert_eq!(decode(&cases[1].0[..]).unwrap().elements(), 8);
        // the loads (8C, 2C, 2D), a register "destination", VMASKMOVPS with W1, another map: not stores
        for code in [[0xC4, 0xE2, 0x4D, 0x8C, 0x04, 0x90], [0xC4, 0xE2, 0x5D, 0x2C, 0x2C, 0x91], [0xC4, 0xE2, 0x4D, 0x8E, 0xC1, 0x90], [0xC4, 0xE2, 0xDD, 0x2E, 0x2C, 0x91], [0xC4, 0xE3, 0x4D, 0x8E, 0x04, 0x90], [0xC4, 0xE2, 0x4E, 0x8E, 0x04, 0x90]] {
            assert_eq!(decode(&code), None, "{code:02x?}");
        }
    }
}
