//! Walking RDNA2 code without decoding it: an instruction's length, opcode and direct branch
//! target, from tables made of the decoder's own (`rdna2_spec`, `rdna2_xbox`).
//!
//! `rdna2::decode` builds a `Decoded` (its words on the heap, operands by field name) for the
//! recompiler. Finding the programs of a console container (`xbox::parse`) and counting their
//! instructions (`xbox::InstStats`) need none of that, yet decoded every program three times:
//! 31 % of the samples of a retail game's running threads in its first 40 s. `step` answers as
//! `decode` does, instruction by instruction (`tests`).

use std::sync::OnceLock;

use super::rdna2::{self, Field, Inst, InstEnc, Variant, F_BRANCH, F_INDIRECT};
use super::rdna2_spec::ENCODINGS;

/// A field in the first 64 bits: the parts `((raw >> shift) & mask) << to`.
#[derive(Clone, Copy, Default)]
struct Bits([(u8, u16, u8); 2]);

impl Bits {
    fn of(f: &Field) -> Bits {
        assert!(f.ranges.len() <= 2, "{}: more than two parts", f.name);
        let (mut out, mut to) = (Bits::default(), 0u8);
        for (part, &(off, n)) in out.0.iter_mut().zip(f.ranges) {
            assert!(off + n <= 64 && to + n <= 16, "{}: past the first 64 bits, or wider than 16", f.name);
            *part = (off, ((1u32 << n) - 1) as u16, to);
            to += n;
        }
        out
    }

    fn get(&self, raw: u64) -> u64 {
        self.0.iter().fold(0, |v, &(shift, mask, to)| v | (raw >> shift & mask as u64) << to)
    }
}

/// One opcode of one encoding.
pub struct Op {
    pub inst: &'static Inst,
    /// the number of `inst` in `rdna2::all_instructions()`
    pub index: usize,
    pub getpc: bool,
    pub setpc: bool,
    base: u8,
    /// VOP1 / VOP2 / VOPC: SRC0 may select an extension dword
    src0: Option<Bits>,
    /// the fields that select a literal: (the field's bits, their value there); an unused one
    /// never matches
    literal: [(u64, u64); 4],
    /// MIMG: extra address dwords
    nsa: Option<Bits>,
    /// the offset of a direct branch (dwords, signed)
    label: Option<Bits>,
}

/// A primary encoding: its opcodes are `slots[first + OP]`.
struct Enc {
    mask: u64,
    value: u64,
    op: Bits,
    first: usize,
}

struct Tables {
    /// the top 9 bits of the first dword → the encodings it may be, in the decoder's order
    by_top: Vec<Vec<Enc>>,
    /// index in `ops`, or `NONE`
    slots: Vec<u16>,
    ops: Vec<Op>,
}

const NONE: u16 = u16::MAX;

fn make_op(inst: &'static Inst, index: usize, ie: &'static InstEnc) -> Op {
    let enc = &ENCODINGS[ie.encoding as usize];
    let field = |n: &str| enc.field(n).map(Bits::of);
    let mut sources: Vec<&str> = ie.operands.iter().filter(|o| !o.implicit && rdna2::is_src_type(o.ty)).filter_map(|o| o.field).collect();
    if rdna2::is_valu(enc) {
        sources.extend(["SRC0", "SRC1", "SRC2"]);
    }
    sources.sort_unstable();
    sources.dedup();
    let mut literal = [(0, 1); 4];
    for (k, f) in sources.into_iter().filter_map(|n| enc.field(n)).enumerate() {
        let &[(off, n)] = f.ranges else { panic!("{}: a source field in parts", f.name) };
        assert!(n >= 8 && off + n <= 64, "{}: cannot be 255", f.name);
        literal[k] = (((1 << n) - 1) << off, 255 << off);
    }
    let direct = inst.flags & F_BRANCH != 0 && inst.flags & F_INDIRECT == 0;
    Op {
        inst,
        index,
        getpc: inst.name == "S_GETPC_B64",
        setpc: inst.name == "S_SETPC_B64",
        base: (enc.bits / 8) as u8,
        src0: if rdna2::is_valu32(enc) { field("SRC0") } else { None },
        literal,
        nsa: if enc.name == "ENC_MIMG" { field("NSA") } else { None },
        label: if direct { ie.operands.iter().find(|o| o.ty == "OPR_LABEL").and_then(|o| o.field).and_then(field) } else { None },
    }
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut primary: Vec<usize> = (0..ENCODINGS.len()).filter(|&i| ENCODINGS[i].primary).collect();
        primary.sort_by_key(|&i| ENCODINGS[i].order);
        // where each encoding's opcodes start (one without an OP field has a single slot)
        let mut first = vec![0usize; ENCODINGS.len()];
        let mut n = 0;
        for &e in &primary {
            first[e] = n;
            n += 1 << ENCODINGS[e].field_width("OP");
        }
        let (mut slots, mut ops) = (vec![NONE; n], Vec::new());
        for (index, inst) in rdna2::all_instructions().enumerate() {
            for ie in inst.encs {
                let e = ie.encoding as usize;
                if !ENCODINGS[e].primary {
                    continue;
                }
                let op = if ENCODINGS[e].field("OP").is_some() { ie.opcode as usize } else { 0 };
                // (an opcode two instructions claim is the first one's)
                if slots[first[e] + op] == NONE {
                    slots[first[e] + op] = ops.len() as u16;
                    ops.push(make_op(inst, index, ie));
                }
            }
        }
        let of = |e: usize| Enc {
            mask: ENCODINGS[e].fixed_mask,
            value: ENCODINGS[e].fixed_value,
            op: ENCODINGS[e].field("OP").map(Bits::of).unwrap_or_default(),
            first: first[e],
        };
        let by_top = (0..512u64)
            .map(|top| primary.iter().copied().map(of).filter(|e| (top << 23 ^ e.value) & e.mask & 0xFF80_0000 == 0).collect())
            .collect();
        Tables { by_top, slots, ops }
    })
}

/// An instruction as `rdna2::decode` would give it.
pub struct Step {
    pub op: &'static Op,
    pub variant: Variant,
    pub len: usize,
    raw: u64,
}

impl Step {
    /// `rdna2::branch_target` of the instruction at `pc`.
    pub fn branch_target(&self, pc: u64) -> Option<u64> {
        let v = self.op.label.as_ref()?.get(self.raw);
        Some((pc as i64 + 4 + (v as u16 as i16 as i64) * 4) as u64)
    }
}

/// The instruction `code` starts with; None where `rdna2::decode` fails.
pub fn step(code: &[u8]) -> Option<Step> {
    let word = |o: usize| code.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    let w0 = word(0)?;
    let raw = w0 as u64 | (word(4).unwrap_or(0) as u64) << 32;
    let t = tables();
    let slot = |e: &Enc| t.slots[e.first + e.op.get(raw) as usize];
    let slot = t.by_top[(w0 >> 23) as usize].iter().filter(|e| raw & e.mask == e.value).map(slot).find(|&s| s != NONE)?;
    let op = &t.ops[slot as usize];
    let variant = match op.src0.map(|f| f.get(raw)) {
        Some(0xF9) => Variant::Sdwa,
        Some(0xFA) => Variant::Dpp16,
        Some(0xE9 | 0xEA) => Variant::Dpp8,
        _ => Variant::Plain,
    };
    let literal = op.inst.always_literal || variant == Variant::Plain && op.literal.iter().fold(false, |any, &(mask, value)| any | (raw & mask == value));
    let extra = (variant != Variant::Plain) as usize + literal as usize + op.nsa.map_or(0, |f| f.get(raw) as usize);
    let len = op.base as usize + 4 * extra;
    (len <= code.len()).then_some(Step { op, variant, len, raw })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn check(code: &[u8]) {
        match (step(code), rdna2::decode(code, 0x1000)) {
            (Some(s), Ok(d)) => {
                assert!(std::ptr::eq(s.op.inst, d.inst), "{} / {}: {code:02x?}", s.op.inst.name, d.inst.name);
                assert!(std::ptr::eq(rdna2::all_instructions().nth(s.op.index).unwrap(), d.inst));
                assert_eq!((s.len, s.variant), (d.len, d.variant), "{}: {code:02x?}", d.inst.name);
                assert_eq!(s.branch_target(0x1000), rdna2::branch_target(&d), "{}: {code:02x?}", d.inst.name);
                assert_eq!((s.op.getpc, s.op.setpc), (d.inst.name == "S_GETPC_B64", d.inst.name == "S_SETPC_B64"));
            }
            (None, Err(_)) => {}
            (s, d) => panic!("step {:?}, decode {:?}: {code:02x?}", s.map(|s| s.op.inst.name), d.map(|d| d.inst.name)),
        }
    }

    pub(crate) fn rng() -> impl FnMut() -> u64 {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    /// Every opcode of every primary encoding, assigned or not, with random other bits and with
    /// the source fields selecting an extension or a literal; whole and cut short.
    #[test]
    fn every_opcode_steps_as_it_decodes() {
        let mut r = rng();
        for enc in ENCODINGS.iter().filter(|e| e.primary) {
            let op = enc.field("OP").map(Bits::of).unwrap_or_default();
            for opcode in 0..1u64 << enc.field_width("OP") {
                for k in 0..24u64 {
                    let mut raw = r() & !enc.fixed_mask | enc.fixed_value;
                    for (shift, mask, to) in op.0 {
                        raw = raw & !((mask as u64) << shift) | (opcode >> to & mask as u64) << shift;
                    }
                    let special = [255u64, 0xF9, 0xFA, 0xE9, 0xEA][k as usize % 5];
                    for (i, n) in ["SRC0", "SRC1", "SRC2", "SSRC0", "SSRC1"].iter().enumerate() {
                        if let (Some(f), true) = (enc.field(n), k >> i & 1 == 1) {
                            let (off, cnt) = f.ranges[0];
                            raw = raw & !(((1u64 << cnt) - 1) << off) | special << off;
                        }
                    }
                    let mut code = raw.to_le_bytes().to_vec();
                    code.extend((0..4).flat_map(|_| (r() as u32).to_le_bytes()));
                    check(&code);
                    check(&code[..r() as usize % 24]);
                }
            }
        }
    }

    #[test]
    fn random_dwords_step_as_they_decode() {
        let mut r = rng();
        for _ in 0..100_000 {
            let code: Vec<u8> = (0..3).flat_map(|_| r().to_le_bytes()).collect();
            check(&code);
        }
    }
}
