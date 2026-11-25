//! LLVM bitstream writer: fixed and VBR fields, nested blocks with
//! back-patched lengths, and unabbreviated records.
//!
//! DXIL accepts the plain bitstream container without any abbreviations,
//! which keeps the writer small; size is not a concern for shader modules
//! (the driver re-encodes them anyway).

/// Bit-level writer, little-endian within 32-bit words as LLVM defines.
pub struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    nbits: u32,
    /// Open blocks: (byte offset of the length word, outer abbreviation width).
    blocks: Vec<(usize, u32)>,
    width: u32,
}

/// Abbreviation width used for every block we open (IDs 0..3 need 2 bits).
pub const ABBREV_WIDTH: u32 = 3;

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(bytes: usize) -> Self {
        BitWriter { out: Vec::with_capacity(bytes), acc: 0, nbits: 0, blocks: Vec::new(), width: 2 }
    }

    #[inline]
    pub fn emit(&mut self, value: u64, width: u32) {
        debug_assert!(width <= 32);
        debug_assert!(width == 64 || value >> width == 0, "value {value:#x} does not fit {width} bits");
        self.acc |= value << self.nbits;
        self.nbits += width;
        if self.nbits >= 32 {
            self.out.extend_from_slice(&(self.acc as u32).to_le_bytes());
            self.acc >>= 32;
            self.nbits -= 32;
        }
    }

    #[inline]
    pub fn vbr(&mut self, mut value: u64, width: u32) {
        let hi = 1u64 << (width - 1);
        if value < hi {
            self.emit(value, width);
            return;
        }
        let mask = hi - 1;
        let shift = width - 1;
        while value >= hi {
            self.emit((value & mask) | hi, width);
            value >>= shift;
        }
        self.emit(value, width);
    }

    pub fn align32(&mut self) {
        while self.nbits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
        if self.nbits > 0 {
            self.out.push(self.acc as u8);
            self.acc = 0;
            self.nbits = 0;
        }
        while self.out.len() % 4 != 0 {
            self.out.push(0);
        }
    }

    pub fn enter_block(&mut self, id: u32) {
        self.emit(1, self.width); // ENTER_SUBBLOCK
        self.vbr(id as u64, 8);
        self.vbr(ABBREV_WIDTH as u64, 4);
        self.align32();
        let at = self.out.len();
        self.out.extend_from_slice(&[0; 4]);
        self.blocks.push((at, self.width));
        self.width = ABBREV_WIDTH;
    }

    pub fn end_block(&mut self) {
        self.emit(0, self.width); // END_BLOCK
        self.align32();
        let (at, outer) = self.blocks.pop().expect("end_block without enter_block");
        let words = ((self.out.len() - at - 4) / 4) as u32;
        self.out[at..at + 4].copy_from_slice(&words.to_le_bytes());
        self.width = outer;
    }

    /// An unabbreviated record.
    pub fn record(&mut self, code: u32, ops: &[u64]) {
        self.emit(3, self.width); // UNABBREV_RECORD
        self.vbr(code as u64, 6);
        self.vbr(ops.len() as u64, 6);
        for &op in ops {
            self.vbr(op, 6);
        }
    }

    /// A record whose operands are the bytes of a string.
    pub fn record_str(&mut self, code: u32, prefix: &[u64], s: &str) {
        self.emit(3, self.width); // UNABBREV_RECORD
        self.vbr(code as u64, 6);
        self.vbr((prefix.len() + s.len()) as u64, 6);
        for &op in prefix {
            self.vbr(op, 6);
        }
        for b in s.bytes() {
            self.vbr(b as u64, 6);
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        assert!(self.blocks.is_empty(), "unterminated bitstream block");
        self.align32();
        self.out
    }
}

/// LLVM's signed VBR operand encoding: sign in bit 0.
pub fn signed(v: i64) -> u64 {
    if v >= 0 {
        (v as u64) << 1
    } else if v == i64::MIN {
        1 // "-0": LLVM's encoding of INT64_MIN
    } else {
        ((v.unsigned_abs()) << 1) | 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vbr_and_fixed_fields_pack_little_endian() {
        let mut w = BitWriter::new();
        w.emit(0b10, 2);
        w.vbr(5, 3); // 0b101 needs continuation in 3-bit VBR: chunks 01|1, 01|0... check value round trip below
        w.align32();
        let bytes = w.finish();
        assert_eq!(bytes.len(), 4);
        // Decode back: 2 bits, then VBR3.
        let word = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        assert_eq!(word & 3, 0b10);
        let (c0, c1) = ((word >> 2) & 7, (word >> 5) & 7);
        assert_eq!(c0, 0b101); // low 2 bits of 5 = 01, continuation bit set
        assert_eq!(c1, 0b001); // remaining 1, no continuation
    }

    #[test]
    fn block_lengths_are_back_patched_in_words() {
        let mut w = BitWriter::new();
        w.enter_block(8);
        w.record(1, &[1]);
        w.end_block();
        let b = w.finish();
        let len = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
        assert_eq!(b.len(), 8 + len * 4);
    }

    #[test]
    fn signed_vbr_operands() {
        assert_eq!(signed(0), 0);
        assert_eq!(signed(3), 6);
        assert_eq!(signed(-3), 7);
        assert_eq!(signed(i64::MIN), 1);
    }

    #[test]
    fn emit_word_flush_and_align32_match_byte_by_byte() {
        // Exercise every residual bit count 0..=31 before a 32-bit emit and before align32.
        for lead in 0..=31u32 {
            let mut w = BitWriter::with_capacity(16);
            let mut bits = Vec::new();
            let push_bits = |bits: &mut Vec<u8>, v: u64, n: u32| {
                for i in 0..n {
                    bits.push(((v >> i) & 1) as u8);
                }
            };
            if lead > 0 {
                let pat = (0xA5A5_A5A5u64) & ((1u64 << lead) - 1);
                w.emit(pat, lead);
                push_bits(&mut bits, pat, lead);
            }
            let full = 0xDEAD_BEEFu64;
            w.emit(full, 32);
            push_bits(&mut bits, full, 32);
            w.vbr(17, 6);
            push_bits(&mut bits, 17, 6);
            w.vbr(0x1234_5678, 6);
            let mut v = 0x1234_5678u64;
            while v >= 32 {
                push_bits(&mut bits, (v & 31) | 32, 6);
                v >>= 5;
            }
            push_bits(&mut bits, v, 6);
            w.align32();
            while bits.len() % 32 != 0 {
                bits.push(0);
            }
            let expected: Vec<u8> = bits
                .chunks_exact(8)
                .map(|c| c.iter().enumerate().fold(0u8, |acc, (i, &b)| acc | (b << i)))
                .collect();
            assert_eq!(w.finish(), expected, "lead={lead}");
        }
    }

    #[test]
    fn record_str_matches_record_bytes() {
        let mut a = BitWriter::new();
        let mut b = BitWriter::new();
        a.enter_block(14);
        b.enter_block(14);
        a.record_str(1, &[7, 42], "dx.op.rawBufferLoad.i32");
        let mut ops = vec![7u64, 42];
        ops.extend("dx.op.rawBufferLoad.i32".bytes().map(u64::from));
        b.record(1, &ops);
        a.end_block();
        b.end_block();
        assert_eq!(a.finish(), b.finish());
    }
}
