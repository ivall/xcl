//! BCPack: lossless packing of BC1-BC7 block streams, as the console's
//! DirectStorage hardware unpacks them. Decode only; BC2 is left out (no
//! title data uses it). Ported from VIctory's victory-formats/src/bcpack.rs on 2026-11-23
//! (https://github.com/Marvisak/VIctory, GPL-3.0, see THIRD_PARTY.md).
//!
//! A packed stream has no header. It is 12 or 16 interleaved rANS *lanes*;
//! each lane is a run of fragments (header byte, length, payload) coded with
//! one of 128 static models, and a fixed schedule decides which lane takes
//! the next input byte. The lanes' symbols form one record per block: two
//! back references (to one of the last 256 blocks, or to a per-format
//! dictionary that stands in for blocks -1 to -255) and the residuals of the
//! endpoints and indices against them. The models, the index rank tables,
//! the BC7 control codes and the dictionaries are constants of the format,
//! kept in `bcpack-tables.bin` beside this file ([`Tables::builtin`]).

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub &'static str);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BCPack: {}", self.0)
    }
}

impl std::error::Error for Error {}

const HISTORY: usize = 256;
/// rANS: decode while the state is above this, read while below.
const LOW: u32 = 1 << 24;
/// Input bytes per schedule round.
const ROUND_BYTES: usize = 8;
/// At most 16 lanes with two symbols per block.
const MAX_RECORD_BYTES: usize = 32;
/// Keep the lane loop on a small batch before reconstructing its blocks.
const RECORD_BATCH: usize = 64;

use std::sync::OnceLock;

use super::tables::{self, Bc6Ep, Bc6Ix, Bc7Ep, Bc7Ix, Plain};

/// The format's constant tables, initialised once from `tables.rs`.
pub struct Tables {
    pub freq: &'static [[u32; tables::BUCKETS]; tables::MODELS],
    pub start: [[u32; tables::BUCKETS]; tables::MODELS],
    pub recip: [[u64; tables::BUCKETS]; tables::MODELS],
    pub rank1: &'static [u8; 256],
    pub rank3a: &'static [u8; 64],
    pub rank3c: &'static [u8; 256],
    pub rank4: &'static [u8; 64],
    pub rank5: &'static [u8; 64],
    pub ctrl7: &'static [u16; 512],
    pub bc1: &'static [Plain],
    pub bc3: &'static [Plain],
    pub bc4: &'static [Plain],
    pub bc5: &'static [Plain],
    pub bc6: &'static [(Bc6Ep, Bc6Ix)],
    pub bc7: &'static [(Bc7Ep, Bc7Ix)],
}

impl Tables {
    pub fn builtin() -> &'static Tables {
        static T: OnceLock<Tables> = OnceLock::new();
        T.get_or_init(|| {
            let mut start = [[0u32; tables::BUCKETS]; tables::MODELS];
            let mut recip = [[0u64; tables::BUCKETS]; tables::MODELS];
            for m in 0..tables::MODELS {
                let used = m / 16 + 2;
                let mut at = 0u32;
                for b in 0..used {
                    start[m][b] = at;
                    recip[m][b] = (1u64 << 32).div_ceil(u64::from(tables::FREQ[m][b]));
                    at += tables::FREQ[m][b] << b.saturating_sub(1);
                }
                start[m][used..].fill(1 << 16);
            }
            Tables {
                freq: &tables::FREQ,
                start,
                recip,
                rank1: &tables::RANK1,
                rank3a: &tables::RANK3A,
                rank3c: &tables::RANK3C,
                rank4: &tables::RANK4,
                rank5: &tables::RANK5,
                ctrl7: &tables::CTRL7,
                bc1: &tables::BC1,
                bc3: &tables::BC3,
                bc4: &tables::BC4,
                bc5: &tables::BC5,
                bc6: &tables::BC6,
                bc7: &tables::BC7,
            }
        })
    }
}


/// Unpacks `src` (BC number `bc`, 1 and 3-7) into `dst`, whose length gives
/// the block count. The stream carries no size: bytes after the last one the
/// lanes need are ignored, a stream that runs out is an error. The caller
/// must discard `dst` on error.
pub fn decode(bc: u8, src: &[u8], dst: &mut [u8], tables: &Tables) -> Result<(), Error> {
    let block = match bc {
        1 | 4 => 8,
        3 | 5 | 6 | 7 => 16,
        _ => return Err(Error("unsupported BC format")),
    };
    if dst.is_empty() || !dst.len().is_multiple_of(block) {
        return Err(Error("destination is not a whole number of blocks"));
    }
    let blocks = dst.len() / block;
    let (widths, per) = lane_layout(bc);
    let mut records = RecordStream::new(tables, widths, per, blocks, src);
    let out = dst.chunks_exact_mut(block);
    let t = tables;
    let result = match bc {
        1 => plain(&mut records, out, &t.bc1, &BC1, |s| bc1_residual(t, s), bc1_pack),
        3 => plain(&mut records, out, &t.bc3, &BC3, |s| bc3_residual(t, s), bc3_pack),
        4 => plain(&mut records, out, &t.bc4, &BC4, |s| bc4_residual(t, s), bc4_pack),
        5 => plain(&mut records, out, &t.bc5, &BC5, |s| bc5_residual(t, s), bc5_pack),
        6 => bc6(&mut records, out, &t.bc6),
        _ => bc7(&mut records, out, t),
    };
    // The caller discards dst on error. Keep the old error precedence: a bad
    // lane stream takes precedence over a bad block reconstructed from it.
    if result.is_err() {
        records.try_for_each(|record| record.map(|_| ()))?;
    }
    result
}

// Lanes.

/// Format `bc`'s lanes (symbol bits - 1 of each) and the symbols each lane gives per block.
fn lane_layout(bc: u8) -> (&'static [u8], usize) {
    match bc {
        1 => (&[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1),
        3 => (&[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7], 2),
        4 => (&[7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 1),
        5 => (&[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 2),
        6 => (&[3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4], 2),
        _ => (&[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], 2),
    }
}

/// The lane stage alone (tests): the records of `blocks` blocks of format `bc` in `src`.
#[cfg(test)]
pub(super) fn records(bc: u8, src: &[u8], blocks: usize) -> Result<(Vec<u8>, usize, usize), Error> {
    let (widths, per) = lane_layout(bc);
    let stride = widths.len() * per;
    let mut records = Vec::with_capacity(blocks * stride);
    for record in RecordStream::new(Tables::builtin(), widths, per, blocks, src) {
        records.extend_from_slice(&record?[..stride]);
    }
    Ok((records, widths.len(), per))
}

#[derive(Clone, Copy, Default)]
struct Lane {
    /// Symbol bits - 1 at the full alphabet.
    width: u8,
    /// 0 header, 1 length (low byte), 2 length high byte, 3 payload.
    phase: u8,
    long: bool,
    reset: bool,
    model: u8,
    need: bool,
    latch: Option<u8>,
    out: Option<u8>,
    stalled: bool,
    x: u32,
    left: u32,
}

impl Lane {
    /// One step: take the latched byte (header, length or payload) and
    /// decode at most one symbol.
    fn step(&mut self, t: &Tables) -> Result<(), Error> {
        if self.phase != 3 {
            let Some(b) = self.latch.take() else {
                return Ok(());
            };
            self.need = true;
            match self.phase {
                0 => {
                    // The top two bits shrink the alphabet; width >= 3.
                    let group = self.width - (b >> 6);
                    self.model = group * 16 + (b & 15);
                    self.reset = b & 0x10 != 0;
                    self.long = b & 0x20 != 0;
                    self.left = 0;
                    self.phase = 1;
                }
                1 if self.long => {
                    self.left = b.into();
                    self.phase = 2;
                }
                1 => {
                    self.left = u32::from(b) + 1;
                    self.phase = 3;
                }
                _ => {
                    self.left = (u32::from(b) << 8 | self.left) + 257;
                    self.phase = 3;
                }
            }
            return Ok(());
        }
        let mut x = self.x;
        if x < LOW {
            if let Some(b) = self.latch.take() {
                self.left = self.left.wrapping_sub(1);
                x = x << 8 | u32::from(b);
            }
        }
        self.need = self.left != 0 && self.latch.is_none();
        if x > LOW {
            let model = usize::from(self.model);
            let freq = &t.freq[model];
            let start = &t.start[model];
            let slot = x & 0xffff;
            let (sym, f, c) = if slot < start[1] {
                // Symbol zero is common in sparse records and owns slots 1 .. f[0] - 1.
                (0, freq[0] - 1, 1)
            } else {
                // Starts are ascending and 65536 past the model's last bucket.
                let b = 1 + start[2..].iter().filter(|&&s| s <= slot).count();
                let f = freq[b];
                // Bucket 1 is one symbol. In larger buckets the reciprocal
                // divides the offset exactly because it is below 2^16.
                let offset = if b == 1 {
                    0
                } else {
                    ((u64::from(slot - start[b]) * t.recip[model][b]) >> 32) as u32
                };
                let sym = (1 << (b - 1)) + offset as u8;
                (sym, f, start[b] + offset * f)
            };
            self.x = (x >> 16).wrapping_mul(f).wrapping_sub(c).wrapping_add(slot);
            self.out = Some(sym);
        } else if self.left != 0 {
            // At exactly 2^24 the lane neither reads nor decodes again; only a
            // lane that has delivered its symbols may end up there.
            self.x = x;
        } else {
            // The fragment ends; a byte read in this step does not reach the carried state.
            self.phase = 0;
            self.need = self.latch.is_none();
            if self.reset {
                self.x = 0;
            }
        }
        Ok(())
    }
}

/// Yields block records from a bounded batch, keeping lane state and
/// read-ahead between batches. Each round the lanes step in order and share
/// eight bytes.
struct RecordStream<'a> {
    tables: &'a Tables,
    lanes: [Lane; 16],
    lane_count: usize,
    per: usize,
    remaining: usize,
    src: std::slice::Iter<'a, u8>,
    batch: [[u8; MAX_RECORD_BYTES]; RECORD_BATCH],
    ready: std::ops::Range<usize>,
}

impl<'a> RecordStream<'a> {
    fn new(tables: &'a Tables, widths: &[u8], per: usize, blocks: usize, src: &'a [u8]) -> Self {
        let mut lanes = [Lane::default(); 16];
        for (lane, &width) in lanes.iter_mut().zip(widths) {
            lane.width = width;
            lane.need = true;
        }
        Self {
            tables,
            lanes,
            lane_count: widths.len(),
            per,
            remaining: blocks,
            src: src.iter(),
            batch: [[0; MAX_RECORD_BYTES]; RECORD_BATCH],
            ready: 0..0,
        }
    }

    /// Runs the schedule until `blocks` records are in `batch`.
    // shortcut: one lane step at a time (BC1 roughly 30-60 MB/s, BC7 15-20 MB/s on one core,
    // measured before the batches); if texture streaming needs more, run the schedule only to
    // split the input per lane and decode the lanes apart.
    fn fill_batch<const PER: usize>(&mut self, blocks: usize) -> Result<(), Error> {
        let full = (1u32 << self.lane_count) - 1;
        let lanes = &mut self.lanes[..self.lane_count];
        let (mut pos, mut done, mut idle) = (0, 0u32, 0);
        while pos < blocks * PER {
            let row = &mut self.batch[pos / PER];
            let (mut taken, mut moved) = (0, false);
            for (j, lane) in lanes.iter_mut().enumerate() {
                let bit = 1 << j;
                let before = taken;
                if !lane.stalled {
                    lane.step(self.tables)?;
                }
                let waiting = lane.out.is_some() && done & bit != 0;
                if lane.need && !waiting && taken < ROUND_BYTES {
                    // Past the end nothing arrives; only finished lanes ask there in a valid stream.
                    if let Some(&b) = self.src.next() {
                        lane.latch = Some(b);
                        taken += 1;
                    }
                }
                lane.stalled = (lane.need && before >= ROUND_BYTES) || waiting;
                if !lane.stalled && lane.out.is_some() {
                    row[j * PER + pos % PER] = lane.out.take().unwrap();
                    done |= bit;
                    moved = true;
                }
            }
            if done == full {
                pos += 1;
                done = 0;
                moved = true;
            }
            // No lane can go on after several rounds without input or output.
            idle = if taken > 0 || moved { 0 } else { idle + 1 };
            if idle > 3 {
                return Err(Error("the stream ends early or is corrupt"));
            }
        }
        Ok(())
    }
}

impl Iterator for RecordStream<'_> {
    type Item = Result<[u8; MAX_RECORD_BYTES], Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ready.is_empty() {
            if self.remaining == 0 {
                return None;
            }
            let blocks = self.remaining.min(RECORD_BATCH);
            let result = if self.per == 1 {
                self.fill_batch::<1>(blocks)
            } else {
                self.fill_batch::<2>(blocks)
            };
            if let Err(error) = result {
                self.remaining = 0;
                return Some(Err(error));
            }
            self.remaining -= blocks;
            self.ready = 0..blocks;
        }
        self.ready.next().map(|i| Ok(self.batch[i]))
    }
}

// Blocks.

/// The record's endpoint and index references.
fn references(s: &[u8], paired: bool) -> (usize, usize) {
    let (d, ix) = if paired {
        (interleave(s[0], s[1]), interleave(s[2], s[3]))
    } else {
        (s[0], s[1])
    };
    let delta = (d >> 1) ^ (d & 1).wrapping_neg();
    (usize::from(ix.wrapping_add(delta)), usize::from(ix))
}

/// Two 4-bit symbols, the first on the even bits.
fn interleave(a: u8, b: u8) -> u8 {
    let spread = |v: u8| (0..4).fold(0, |acc, k| acc | (v >> k & 1) << (2 * k));
    spread(a) | spread(b) << 1
}

/// Block `i`'s reference `r`: 0 none, else `r` blocks back, before the first
/// block into the dictionary.
fn pick<'a, T>(i: usize, r: usize, history: &'a [T], dictionary: &'a [T], zero: &'a T) -> &'a T {
    match r {
        0 => zero,
        r if r <= i => &history[(i - r) % HISTORY],
        r => &dictionary[r - i - 1],
    }
}

/// Zigzag residual (0, -1, 1, -2, ...) modulo 2^bits.
fn zz(v: u8, bits: u32) -> u8 {
    let mask = ((1u32 << bits) - 1) as u8;
    ((v >> 1) ^ if v & 1 != 0 { mask } else { 0 }) & mask
}

/// Element widths of the plain formats' endpoints and indices.
struct Widths {
    ep: [u8; 8],
    ix: [u8; 32],
    paired: bool,
}

const fn widths(ep: [u8; 8], ix3: usize, ix2: usize, paired: bool) -> Widths {
    let mut ix = [0; 32];
    let mut k = 0;
    while k < ix3 + ix2 {
        ix[k] = if k < ix3 { 3 } else { 2 };
        k += 1;
    }
    Widths { ep, ix, paired }
}

const BC1: Widths = widths([5, 6, 5, 5, 6, 5, 0, 0], 0, 16, false);
const BC3: Widths = widths([5, 6, 5, 5, 6, 5, 8, 8], 16, 16, true);
const BC4: Widths = widths([8, 8, 0, 0, 0, 0, 0, 0], 16, 0, false);
const BC5: Widths = widths([8, 8, 8, 8, 0, 0, 0, 0], 32, 0, true);

fn plain<'a>(
    records: &mut RecordStream<'_>,
    out: impl Iterator<Item = &'a mut [u8]>,
    dictionary: &[Plain],
    w: &Widths,
    residual: impl Fn(&[u8]) -> Plain,
    pack: fn(&Plain, &mut [u8]),
) -> Result<(), Error> {
    let mut history = vec![Plain::default(); HISTORY];
    let zero = Plain::default();
    for (i, (s, block)) in records.zip(out).enumerate() {
        let s = &s?;
        let (ep_ref, ix_ref) = references(s, w.paired);
        let ep_ref = &pick(i, ep_ref, &history, dictionary, &zero).ep;
        let ix_ref = &pick(i, ix_ref, &history, dictionary, &zero).ix;
        let mut v = residual(s);
        for ((v, r), bits) in v.ep.iter_mut().zip(ep_ref).zip(w.ep) {
            *v = v.wrapping_add(*r) & ((1u16 << bits) - 1) as u8;
        }
        for ((v, r), bits) in v.ix.iter_mut().zip(ix_ref).zip(w.ix) {
            *v = v.wrapping_add(*r) & ((1u8 << bits) - 1);
        }
        history[i % HISTORY] = v;
        pack(&v, block);
    }
    Ok(())
}

/// 2-bit index residuals: one symbol per 2x2 quad.
fn quads(rank: &[u8; 256], s: &[u8], ix: &mut [u8]) {
    for (q, &sym) in s.iter().enumerate() {
        let v = rank[usize::from(sym)];
        let base = (q >> 1) * 8 + (q & 1) * 2;
        for (k, p) in [0, 1, 4, 5].into_iter().enumerate() {
            ix[base + p] = v >> (2 * k) & 3;
        }
    }
}

/// 3-bit index residuals: one symbol per two pixels in raster order.
fn pairs(rank: &[u8; 64], s: &[u8], ix: &mut [u8]) {
    for (k, &sym) in s.iter().enumerate() {
        let v = rank[usize::from(sym & 63)];
        ix[2 * k] = v & 7;
        ix[2 * k + 1] = v >> 4 & 7;
    }
}

fn bc1_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain::default();
    for (k, bits) in [5, 6, 5, 5, 6, 5].into_iter().enumerate() {
        v.ep[k] = zz(s[2 + k], bits);
    }
    quads(&t.rank1, &s[8..12], &mut v.ix);
    v
}

fn bc3_residual(t: &Tables, s: &[u8]) -> Plain {
    // Colour lanes carry each channel's two endpoints together.
    let mut v = Plain {
        ep: [
            zz(s[14], 5),
            zz(s[16], 6),
            zz(s[18], 5),
            zz(s[15], 5),
            zz(s[17], 6),
            zz(s[19], 5),
            zz(s[4], 8),
            zz(s[5], 8),
        ],
        ..Default::default()
    };
    pairs(&t.rank3a, &s[6..14], &mut v.ix[..16]);
    quads(&t.rank3c, &s[20..24], &mut v.ix[16..]);
    v
}

fn bc4_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain::default();
    v.ep[..2].copy_from_slice(&[zz(s[2], 8), zz(s[3], 8)]);
    pairs(&t.rank4, &s[4..12], &mut v.ix[..16]);
    v
}

fn bc5_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain::default();
    for k in 0..4 {
        v.ep[k] = zz(s[4 + k], 8);
    }
    pairs(&t.rank5, &s[8..24], &mut v.ix);
    v
}

/// Indices along the endpoint line -> BC1 and BC4 codes.
const BC1_CODE: [u32; 4] = [0, 2, 3, 1];
const BC4_CODE: [u64; 8] = [0, 2, 3, 4, 5, 6, 7, 1];

fn bc1_color(ep: &[u8], ix: &[u8], out: &mut [u8]) {
    let c0 = u16::from(ep[0]) << 11 | u16::from(ep[1]) << 5 | u16::from(ep[2]);
    let c1 = u16::from(ep[3]) << 11 | u16::from(ep[4]) << 5 | u16::from(ep[5]);
    let bits = (0..16).fold(0, |acc, p| {
        acc | BC1_CODE[usize::from(ix[p] & 3)] << (2 * p)
    });
    out[..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&bits.to_le_bytes());
}

/// BC4 half: endpoints are kept 128 below their stored value.
fn bc4_alpha(e0: u8, e1: u8, ix: &[u8], out: &mut [u8]) {
    let bits = (0..16).fold(0u64, |acc, p| {
        acc | BC4_CODE[usize::from(ix[p] & 7)] << (16 + 3 * p)
    });
    let v = u64::from(e0 ^ 0x80) | u64::from(e1 ^ 0x80) << 8 | bits;
    out[..8].copy_from_slice(&v.to_le_bytes());
}

fn bc1_pack(v: &Plain, out: &mut [u8]) {
    bc1_color(&v.ep, &v.ix, out);
}

fn bc3_pack(v: &Plain, out: &mut [u8]) {
    bc4_alpha(v.ep[6], v.ep[7], &v.ix[..16], &mut out[..8]);
    bc1_color(&v.ep, &v.ix[16..], &mut out[8..]);
}

fn bc4_pack(v: &Plain, out: &mut [u8]) {
    bc4_alpha(v.ep[0], v.ep[1], &v.ix, out);
}

fn bc5_pack(v: &Plain, out: &mut [u8]) {
    bc4_alpha(v.ep[0], v.ep[1], &v.ix[..16], &mut out[..8]);
    bc4_alpha(v.ep[2], v.ep[3], &v.ix[16..], &mut out[8..]);
}

/// A byte's even and odd bits as two zigzag-coded 4-bit residuals.
fn nibbles(b: u8, odd_first: bool) -> [i16; 2] {
    let half = |v: u8| (0..4).fold(0u8, |acc, k| acc | (v >> (2 * k) & 1) << k);
    let z = |u: u8| i16::from(u >> 1) ^ -i16::from(u & 1);
    let (even, odd) = (z(half(b)), z(half(b >> 1)));
    if odd_first { [odd, even] } else { [even, odd] }
}

/// Index residual plus reference, both left-aligned to 16 bits at their own
/// widths, back to `bits` bits (signed).
fn add_index(cur: i16, bits: u8, refv: i16, ref_bits: u8) -> i16 {
    let up = |v: i16, b: u8| {
        if b == 0 || b > 16 {
            0
        } else {
            (v as u16) << (16 - b)
        }
    };
    let sum = up(cur, bits).wrapping_add(up(refv, ref_bits)) as i16;
    if bits == 0 { 0 } else { sum >> (16 - bits) }
}

/// Little-endian bit writer for one 128-bit block.
#[derive(Default)]
struct Bits {
    v: u128,
    at: u32,
}

impl Bits {
    fn put(&mut self, value: u64, n: u32) {
        if n > 0 && self.at < 128 {
            self.v |= (u128::from(value) & ((1u128 << n) - 1)) << self.at;
        }
        self.at += n;
    }
}

// BC7.

/// Public BC7 anchor indices: second subset of two; second and third of three.
const ANCHOR2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
];
const ANCHOR3A: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5,
    15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5,
    10, 8, 13, 15, 12, 3, 3,
];
const ANCHOR3B: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6,
    10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];

/// Endpoint bits of R G B A per mode (public), before p-bits and rotation.
const BC7_BITS: [[u8; 4]; 8] = [
    [4, 4, 4, 0],
    [6, 6, 6, 0],
    [5, 5, 5, 0],
    [7, 7, 7, 0],
    [5, 5, 5, 6],
    [7, 7, 7, 8],
    [7, 7, 7, 7],
    [5, 5, 5, 5],
];
/// Index bits of the primary and secondary sets per mode.
const BC7_INDEX: [[u8; 2]; 8] = [
    [3, 0],
    [3, 0],
    [2, 0],
    [2, 0],
    [3, 2],
    [2, 2],
    [4, 0],
    [2, 0],
];

fn pbits(mode: u16) -> bool {
    matches!(mode, 0 | 1 | 3 | 6 | 7)
}

/// Channel precisions: the p-bit is red's lowest bit; rotation swaps alpha
/// with the rotated channel.
fn bc7_precision(ctrl: u16) -> [u8; 4] {
    let (mode, rot) = (ctrl & 7, usize::from(ctrl >> 3 & 3));
    let mut p = BC7_BITS[usize::from(mode)];
    if rot != 0 {
        p.swap(3, rot - 1);
    } else if pbits(mode) {
        p[0] += 1;
    }
    p
}

fn bc7<'a>(
    records: &mut RecordStream<'_>,
    out: impl Iterator<Item = &'a mut [u8]>,
    t: &Tables,
) -> Result<(), Error> {
    let mut history = vec![(Bc7Ep::default(), Bc7Ix::default()); HISTORY];
    let zero = (Bc7Ep::default(), Bc7Ix::default());
    for (i, (s, block)) in records.zip(out).enumerate() {
        let s = &s?;
        let (ep_ref, ix_ref) = references(s, true);
        let ctrl = t.ctrl7[usize::from(s[31] & 15) << 5 | usize::from(s[30] & 31)];
        if ctrl == 0 {
            return Err(Error("unused BC7 control code"));
        }
        let mode = usize::from(ctrl & 7);
        let subsets = usize::from((ctrl & 0xff) >> 6);
        let prec = bc7_precision(ctrl);
        let three = subsets == 3;
        let or0 = |v: u8| if three { 0 } else { v };
        let if3 = |v: u8| if three { v } else { 0 };
        let stored = [
            [s[4], s[5], s[6], s[7], s[8], s[9], or0(s[10]), or0(s[11])],
            [
                s[12],
                s[13],
                s[14],
                s[15],
                s[16],
                s[17],
                or0(s[18]),
                or0(s[19]),
            ],
            [
                s[20],
                s[21],
                if3(s[10]),
                if3(s[11]),
                if3(s[18]),
                if3(s[19]),
                0,
                0,
            ],
        ];
        let r = &pick(i, ep_ref, &history, &t.bc7, &zero).0;
        let mut ep = Bc7Ep {
            ctrl,
            ..Default::default()
        };
        // Rows the reference lacks build on its first row.
        let common = subsets.min(usize::from((r.ctrl & 0xff) >> 6));
        let columns = if prec[3] > 0 { 8 } else { 6 };
        for c in 0..columns {
            let p = prec[c / 2];
            if p == 0 {
                continue;
            }
            let mask = 0xffffu16 << (16 - p);
            for (row, st) in stored.iter().enumerate().take(subsets) {
                let v = st[c];
                let residual =
                    (u16::from(v >> 1) ^ (v & 1).wrapping_neg() as i8 as u16) << (16 - p);
                let base = r.rows[if row < common { row } else { 0 }][c] & mask;
                ep.rows[row][c] = base.wrapping_add(residual);
            }
        }
        let bits = BC7_INDEX[mode];
        let rx = &pick(i, ix_ref, &history, &t.bc7, &zero).1;
        let mut ix = Bc7Ix {
            bits,
            ..Default::default()
        };
        for (set, offsets) in [[22, 24, 26, 28], [12, 14, 16, 20]].into_iter().enumerate() {
            if bits[set] == 0 {
                continue;
            }
            for (k, o) in offsets.into_iter().enumerate() {
                for (b, &byte) in s[o..o + 2].iter().enumerate() {
                    let pair = nibbles(byte, k < 3);
                    for (h, &v) in pair.iter().enumerate() {
                        let p = 4 * k + 2 * b + h;
                        ix.ix[set][p] = add_index(v, bits[set], rx.ix[set][p], rx.bits[set]);
                    }
                }
            }
        }
        history[i % HISTORY] = (ep, ix);
        bc7_pack(&ep, &ix, prec, block);
    }
    Ok(())
}

fn bc7_pack(ep: &Bc7Ep, ix: &Bc7Ix, prec: [u8; 4], out: &mut [u8]) {
    let ctrl = ep.ctrl;
    let mode = ctrl & 7;
    let subsets = usize::from((ctrl & 0xff) >> 6);
    let part = usize::from(ctrl >> 8 & 63);
    let rot = usize::from(ctrl >> 3 & 3);
    let mut w = Bits::default();
    w.put(1 << mode, u32::from(mode) + 1);
    match mode {
        0 => w.put(part as u64, 4),
        1 | 2 | 3 | 7 => w.put(part as u64, 6),
        4 => {
            w.put(rot as u64, 2);
            w.put(u64::from(ctrl >> 5 & 1), 1);
        }
        5 => w.put(rot as u64, 2),
        _ => {}
    }
    let ends = 2 * subsets;
    let field = |row: usize, col: usize, p: u8| u64::from(ep.rows[row][col] >> (16 - p));
    let mut order = [0, 1, 2, 3];
    if rot != 0 {
        order.swap(3, rot - 1);
    }
    for ch in order {
        let p = prec[ch];
        if p == 0 {
            continue;
        }
        let drop = u32::from(ch == 0 && pbits(mode));
        for e in 0..ends {
            w.put(field(e / 2, 2 * ch + e % 2, p) >> drop, u32::from(p) - drop);
        }
    }
    match mode {
        1 => (0..subsets).for_each(|row| w.put(field(row, 0, prec[0]) & 1, 1)),
        m if pbits(m) => (0..ends).for_each(|e| w.put(field(e / 2, e % 2, prec[0]) & 1, 1)),
        _ => {}
    }
    let anchors: &[u8] = match subsets {
        2 => &[0, ANCHOR2[part]],
        3 => &[0, ANCHOR3A[part], ANCHOR3B[part]],
        _ => &[0],
    };
    let mut indices = |set: usize| {
        let bits = ix.bits[set];
        for (p, &v) in ix.ix[set].iter().enumerate() {
            let n = u32::from(bits) - u32::from(anchors.contains(&(p as u8)));
            w.put(u64::from(v as u16), n);
        }
    };
    match mode {
        4 => {
            indices(1);
            indices(0);
        }
        5 => {
            indices(0);
            indices(1);
        }
        _ => indices(0),
    }
    out.copy_from_slice(&w.v.to_le_bytes());
}

// BC6H.

/// Bits of endpoint 0, then of the other endpoints' R, G, B, per mode 1-14.
const BC6_BITS: [[u8; 4]; 15] = [
    [0; 4],
    [10, 5, 5, 5],
    [7, 6, 6, 6],
    [11, 5, 4, 4],
    [11, 4, 5, 4],
    [11, 4, 4, 5],
    [9, 5, 5, 5],
    [8, 6, 5, 5],
    [8, 5, 6, 5],
    [8, 5, 5, 6],
    [6, 6, 6, 6],
    [10, 10, 10, 10],
    [11, 9, 9, 9],
    [12, 8, 8, 8],
    [16, 4, 4, 4],
];
/// Mode code and its width, modes 1-14.
const BC6_CODE: [(u64, u32); 15] = [
    (0, 0),
    (0, 2),
    (1, 2),
    (2, 5),
    (6, 5),
    (10, 5),
    (14, 5),
    (18, 5),
    (22, 5),
    (26, 5),
    (30, 5),
    (3, 5),
    (7, 5),
    (11, 5),
    (15, 5),
];

/// A run of the public BC6H layout: `len` bits of channel `ch` (0-2; 3 is
/// the partition) of endpoint `ep`, from bit `first` up.
#[derive(Clone, Copy)]
struct Run {
    ch: u8,
    ep: u8,
    first: u8,
    len: u8,
}

const fn r(ch: u8, ep: u8, first: u8, len: u8) -> Run {
    Run { ch, ep, first, len }
}

const R: u8 = 0;
const G: u8 = 1;
const B: u8 = 2;
const D: Run = r(3, 0, 0, 5);

/// The header after the mode bits, per mode 1-14. Modes 13 and 14 write the
/// high base bits in ascending order (what the console's packer does).
#[rustfmt::skip]
const BC6_LAYOUT: [&[Run]; 15] = [
    &[],
    &[
        r(G, 2, 4, 1), r(B, 2, 4, 1), r(B, 3, 4, 1), r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10),
        r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4), r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4),
        r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5),
        r(B, 3, 3, 1), D,
    ],
    &[
        r(G, 2, 5, 1), r(G, 3, 4, 1), r(G, 3, 5, 1), r(R, 0, 0, 7), r(B, 3, 0, 1), r(B, 3, 1, 1),
        r(B, 2, 4, 1), r(G, 0, 0, 7), r(B, 2, 5, 1), r(B, 3, 2, 1), r(G, 2, 4, 1), r(B, 0, 0, 7),
        r(B, 3, 3, 1), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 6),
        r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 6), r(R, 3, 0, 6), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 5), r(R, 0, 10, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 4), r(G, 0, 10, 1), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 4), r(B, 0, 10, 1),
        r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 1), r(G, 3, 4, 1),
        r(G, 2, 0, 4), r(G, 1, 0, 5), r(G, 0, 10, 1), r(G, 3, 0, 4), r(B, 1, 0, 4), r(B, 0, 10, 1),
        r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 4), r(B, 3, 0, 1), r(B, 3, 2, 1), r(R, 3, 0, 4),
        r(G, 2, 4, 1), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 1), r(B, 2, 4, 1),
        r(G, 2, 0, 4), r(G, 1, 0, 4), r(G, 0, 10, 1), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 5),
        r(B, 0, 10, 1), r(B, 2, 0, 4), r(R, 2, 0, 4), r(B, 3, 1, 1), r(B, 3, 2, 1), r(R, 3, 0, 4),
        r(B, 3, 4, 1), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 9), r(B, 2, 4, 1), r(G, 0, 0, 9), r(G, 2, 4, 1), r(B, 0, 0, 9), r(B, 3, 4, 1),
        r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4), r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4),
        r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5),
        r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 8), r(G, 3, 4, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(B, 3, 2, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(B, 3, 3, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 5),
        r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 6),
        r(R, 3, 0, 6), D,
    ],
    &[
        r(R, 0, 0, 8), r(B, 3, 0, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(G, 2, 5, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(G, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 6), r(G, 3, 0, 4), r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5),
        r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 8), r(B, 3, 1, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(B, 2, 5, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 5),
        r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 6), r(G, 3, 4, 1), r(B, 3, 0, 1), r(B, 3, 1, 1), r(B, 2, 4, 1), r(G, 0, 0, 6),
        r(G, 2, 5, 1), r(B, 2, 5, 1), r(B, 3, 2, 1), r(G, 2, 4, 1), r(B, 0, 0, 6), r(G, 3, 5, 1),
        r(B, 3, 3, 1), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 6),
        r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 6), r(R, 3, 0, 6), D,
    ],
    &[r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 10), r(G, 1, 0, 10), r(B, 1, 0, 10)],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 9), r(R, 0, 10, 1), r(G, 1, 0, 9),
        r(G, 0, 10, 1), r(B, 1, 0, 9), r(B, 0, 10, 1),
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 8), r(R, 0, 10, 2), r(G, 1, 0, 8),
        r(G, 0, 10, 2), r(B, 1, 0, 8), r(B, 0, 10, 2),
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 6), r(G, 1, 0, 4),
        r(G, 0, 10, 6), r(B, 1, 0, 4), r(B, 0, 10, 6),
    ],
];

fn bc6_index_bits(mode: u8) -> u8 {
    match mode {
        1..=10 => 3,
        11..=14 => 4,
        _ => 0,
    }
}

fn bc6<'a>(
    records: &mut RecordStream<'_>,
    out: impl Iterator<Item = &'a mut [u8]>,
    dictionary: &[(Bc6Ep, Bc6Ix)],
) -> Result<(), Error> {
    let mut history = vec![(Bc6Ep::default(), Bc6Ix::default()); HISTORY];
    let zero = (
        Bc6Ep::default(),
        Bc6Ix {
            mode: 11,
            ..Default::default()
        },
    );
    let z = |v: u16| (v >> 1) ^ (v & 1).wrapping_neg();
    let wide = |a: u8, b: u8| {
        (0..8).fold(0u16, |acc, k| {
            acc | u16::from(a >> k & 1) << (2 * k) | u16::from(b >> k & 1) << (2 * k + 1)
        })
    };
    for (i, (s, block)) in records.zip(out).enumerate() {
        let s = &s?;
        let (ep_ref, ix_ref) = references(s, true);
        let mode = s[30].wrapping_add(1);
        if !(1..=14).contains(&mode) {
            return Err(Error("BC6H mode out of range"));
        }
        let two = mode < 11;
        let shift = BC6_BITS[usize::from(mode)].map(|b| 16 - u32::from(b));
        let e0 = [wide(s[4], s[5]), wide(s[6], s[7]), wide(s[8], s[9])].map(z);
        let e1 = [wide(s[10], s[11]), wide(s[12], s[13]), wide(s[14], s[15])].map(z);
        let (e2, e3) = if two {
            (
                [s[16], s[18], s[20]].map(|v| z(v.into())),
                [s[17], s[19], s[21]].map(|v| z(v.into())),
            )
        } else {
            (e0, e1)
        };
        let r = &pick(i, ep_ref, &history, dictionary, &zero).0;
        let mut ep = Bc6Ep::default();
        // Endpoint 0 at the base precision, the others per channel; the
        // reference is cut by the mask after the sum.
        let mask = |sh: u32| 0xffffu16 << sh;
        for ((e, &r), res) in ep.e[0].iter_mut().zip(&r.e[0]).zip(e0) {
            *e = r.wrapping_add(res << shift[0]) & mask(shift[0]);
        }
        for (k, res) in [e1, e2, e3]
            .iter()
            .enumerate()
            .take(if two { 3 } else { 1 })
        {
            for c in 0..3 {
                ep.e[k + 1][c] =
                    r.e[k + 1][c].wrapping_add(res[c] << shift[1 + c]) & mask(shift[1 + c]);
            }
        }
        if !two {
            for c in 0..3 {
                ep.e[2][c] = (ep.e[0][c] >> shift[0]) << shift[1];
            }
            ep.e[3] = ep.e[1];
        }
        let rx = &pick(i, ix_ref, &history, dictionary, &zero).1;
        let bits = bc6_index_bits(mode);
        let mut ix = Bc6Ix {
            mode,
            ..Default::default()
        };
        for k in 0..8 {
            for (h, v) in nibbles(s[22 + k], k < 6).into_iter().enumerate() {
                let p = 2 * k + h;
                ix.ix[p] = add_index(v, bits, rx.ix[p], bc6_index_bits(rx.mode));
            }
        }
        history[i % HISTORY] = (ep, ix);
        // Fields: the working value has its top bit flipped.
        let field = |k: usize, c: usize| {
            let sh = if k == 0 { shift[0] } else { shift[1 + c] };
            u64::from((ep.e[k][c] ^ 0x8000) >> sh)
        };
        let mut w = Bits::default();
        let (code, n) = BC6_CODE[usize::from(mode)];
        w.put(code, n);
        let part = s[31] & 31;
        for run in BC6_LAYOUT[usize::from(mode)] {
            if run.ch == 3 {
                w.put(part.into(), 5);
            } else {
                w.put(
                    field(run.ep.into(), run.ch.into()) >> run.first,
                    run.len.into(),
                );
            }
        }
        let anchor = if two { ANCHOR2[usize::from(part)] } else { 0 };
        for (p, &v) in ix.ix.iter().enumerate() {
            let short = p == 0 || p == usize::from(anchor);
            let v = (v as u16).wrapping_add(1 << (bits - 1));
            w.put(v.into(), u32::from(bits) - u32::from(short));
        }
        block.copy_from_slice(&w.v.to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod streaming_tests {
    use super::super::{decode, Error};

    #[test]
    fn decoded_blocks_match_the_previous_decoder() {
        // Reference bytes from the two-pass decoder on the encoder's synthetic fixtures.
        let cases: [(u8, &[u8], &str); 6] = [
            (1, include_bytes!("../../../../../tests/data/bcpack/bc1-16-1.bin"), concat!(
                "9db2fc612faaeafb4c4c8300fa401aa821135e088aff7b9b09684010635320e0241d4339dfd3fded109c8f6baf3e2f5d811c62084e4b6f45a942433192a9ab1f",
                "4678a44916d85a2be5a54149f54d99e949596231e2dd1f7de76783398a40608562f6cc735f3b3b7faa3a6a42d333ddbd0428a429abf78b0aa237404176f8e095",
            )),
            (3, include_bytes!("../../../../../tests/data/bcpack/bc3-16-1.bin"), concat!(
                "a3197dbe01c046b50e74e58462229830121168e76bfecc1a4f6bfb0ea2a03df5698f5ef2b700f59331470520656faa0ec37d4a9bb788546e128d526dfea1246e",
                "6f6a7be4ba79c7bd261ef6f00402ac665b7341d052e42f499311bb100e4e81578847bc5497bf9f6cba47352e5fa0e88c6e448b68bf09b06d17b7a2fe1b524f91",
            )),
            (4, include_bytes!("../../../../../tests/data/bcpack/bc4-16-1.bin"), concat!(
                "994835c43474933d310b6a55db7eaaf25624ff1faf9ad5de9e3528cc4f97d5d2cb4de2a4df661c27b28c6a43b38e9ec99304e666ffd2221d7d7f98b911912f",
                "a1b5725b226d5f485a17a65d0b513e1abfd0a4aecdbe77c68a727d8704487e709058645b8be59f75e9f766ff16d497c5a59878b04d6db97d11b4d88e030a448b0d",
            )),
            (5, include_bytes!("../../../../../tests/data/bcpack/bc5-16-1.bin"), concat!(
                "505d3049c2f336030a16a46b40edfc3be9ef365843341bca5c67edd0a540c01368b4c90a2c892acdf381369c82e99ba4cd669ebdb930a07a0d8975fedb208945",
                "73b1798c04524d622d1cd11d473fbe0e0d2a9b247d76d5beef3e2537c280c56492613832b431a06c8046d2c5a252f34a96cc44165f652ef6a8bfa31a33ed5a0f",
            )),
            (6, include_bytes!("../../../../../tests/data/bcpack/bc6-16-1.bin"), concat!(
                "81611a7d7d4662e88e1d50c12c57d55d264f0ac0a3b7df44ec2f604102495de62d5063808181ce5bd026707d9cfb4e4ba04f44b8499cc8802465e434294746",
                "20263b073755dcaf23ee4e6c7a6e6b82d3a21b9f2e618b3918ad6efcbda1785385c2d15dad53160e2c310d0c101ee03df5558fb747c05fd465955784541e9b8572",
            )),
            (7, include_bytes!("../../../../../tests/data/bcpack/bc7-16-1.bin"), concat!(
                "803de0abb1d3de4875d5e82ddff8c7cc802d04ac22bdaa75b47913c2a7be436980f91114024551c9a7624ce93a5ba82880baa8520e2d67e8866f33d66fe75aa68",
                "0df265426d966be430581487ec360dd8060eb8b0a25d1cd95f40f8239446b38803ba3889ef316b6893b90e31fafe4f8803427e826fd2c55915f5c42129f6d2c",
            )),
        ];
        for (mode, src, hex) in cases {
            let expected: Vec<u8> = (0..hex.len()).step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect();
            assert_eq!(expected.len(), 128);
            assert_eq!(decode(mode, src, 128).unwrap(), expected, "BC{mode}");
        }
    }

    #[test]
    fn batch_boundaries_preserve_lane_read_ahead() {
        let src = include_bytes!("../../../../../tests/data/bcpack/bc7-8192-3.bin");
        let expected = [0x80, 0x3f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(decode(7, src, 4096 * 16).unwrap(), expected.repeat(4096));
        assert_eq!(decode(7, src, 65 * 16).unwrap(), expected.repeat(65));
    }

    /// FNV-1a, 64 bits.
    fn fnv(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// Output that varies from block to block, across the batches of 64 records. The hashes are
    /// of what the two-pass decoder of `main` (47efa2e) gave for the same synthetic fixtures
    /// (`tests/data/bcpack/generate.py`), so a record dropped, repeated or left unwritten at a
    /// boundary shows. BC1 has no block errors, so random symbols decode.
    #[test]
    fn batch_boundaries_match_the_previous_decoder() {
        let bc1_700 = include_bytes!("../../../../../tests/data/bcpack/bc1-700-2.bin");
        for (blocks, hash) in [
            (63, 0x7e84_51a2_111e_a50e),
            (64, 0x1a27_ded4_beb3_46cd),
            (65, 0x558e_76a5_56fe_a2ea),
            (127, 0x1414_519a_6c0b_f625),
            (128, 0xb296_1a17_d993_e007),
            (129, 0x782f_2402_991a_e524),
        ] {
            let out = decode(1, bc1_700, blocks * 8).unwrap();
            assert_eq!(fnv(&out), hash, "bc1-700-2, {blocks} blocks");
        }
        // 700 blocks: one hash for each batch of 64 blocks (512 bytes), the last one of 60.
        let want = [
            0x1a27_ded4_beb3_46cd,
            0xae91_a1a1_5304_38af,
            0x6156_c01a_8c61_00a5,
            0x70db_84c1_ca4e_0084,
            0x4183_85c2_a165_1eff,
            0x1b48_5c02_2cc6_9307,
            0xa191_9c35_81e6_46f4,
            0xcdf4_f556_c294_3a40,
            0x9285_541f_4dea_fcf4,
            0x4680_ef24_f794_c2bb,
            0x2b3c_9844_1c46_1bb3,
        ];
        let out = decode(1, bc1_700, 700 * 8).unwrap();
        let got: Vec<u64> = out.chunks(64 * 8).map(fnv).collect();
        assert_eq!(got, want, "bc1-700-2, 700 blocks");
        // 8192 blocks, mostly equal with a few different ones.
        let bc1_8192 = include_bytes!("../../../../../tests/data/bcpack/bc1-8192-3.bin");
        let out = decode(1, bc1_8192, 8192 * 8).unwrap();
        assert_eq!(fnv(&out), 0xa0d8_a27e_a25c_b660, "bc1-8192-3");
    }

    #[test]
    fn truncated_lane_stream_takes_precedence_over_invalid_block() {
        let src = include_bytes!("../../../../../tests/data/bcpack/bc7-700-2.bin");
        let prefix = &src[..src.len() / 2];
        let invalid_block = Error::Decode("unused BC7 control code");
        assert_eq!(decode(7, prefix, 2 * 16), Err(invalid_block));
        assert_eq!(decode(7, src, 350 * 16), Err(invalid_block));
        assert_eq!(decode(7, prefix, 350 * 16), Err(Error::Decode("the stream ends early or is corrupt")));
    }
}

