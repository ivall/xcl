//! Console shader containers as a title hands them to D3D12.X: the DXBC part table
//! (`gpu::shader::container_parts`) and the XDAT / XBCH / FSSM layers inside the DXIL part
//! (`gpu::xbox`). The blobs are built here from the layouts the parsers document; malformed ones
//! must give `None` or an error, never a panic.

use std::panic::{catch_unwind, AssertUnwindSafe};

use xcl::gpu::shader::{container_parts, Part};
use xcl::gpu::xbox::{self, header_qwords, is_xbox, listing, InstStats, XboxShader, XDAT_HASH, XDAT_NESTED_HEADER, XDAT_XBCH};

fn le32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn put32(v: &mut [u8], o: usize, x: u32) {
    v[o..o + 4].copy_from_slice(&x.to_le_bytes());
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Run `f`; a panic fails the test with the input that caused it.
fn guarded<T>(what: &str, input: &[u8], f: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => panic!("{what} panicked on {} B: {}", input.len(), hex(input)),
    }
}

/// Small deterministic generator (Knuth's MMIX LCG).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n.max(1)
    }
}

/// DXBC container: "DXBC", 16-byte hash, u32 version, u32 total size, u32 part count,
/// part offsets; each part is FourCC, u32 size, data.
fn container(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let mut v = vec![0u8; 32 + parts.len() * 4];
    v[0..4].copy_from_slice(b"DXBC");
    put32(&mut v, 20, 1);
    put32(&mut v, 28, parts.len() as u32);
    for (i, (cc, data)) in parts.iter().enumerate() {
        let off = v.len() as u32;
        put32(&mut v, 32 + i * 4, off);
        v.extend_from_slice(*cc);
        le32(&mut v, data.len() as u32);
        v.extend_from_slice(data);
    }
    let total = v.len() as u32;
    put32(&mut v, 24, total);
    v
}

/// Header with `count` and the given part offsets, then `body` (offsets are absolute).
fn raw_container(count: u32, offsets: &[u32], body: &[u8]) -> Vec<u8> {
    let mut v = vec![0u8; 32];
    v[0..4].copy_from_slice(b"DXBC");
    put32(&mut v, 28, count);
    for &o in offsets {
        le32(&mut v, o);
    }
    v.extend_from_slice(body);
    v
}

/// Reference reading of the part table: every part header and its data inside the blob, at most
/// 64 parts. FourCC bytes are kept raw.
fn ref_parts(b: &[u8]) -> Option<Vec<([u8; 4], usize, usize)>> {
    let rd = |o: usize| -> Option<usize> { Some(u32::from_le_bytes(b.get(o..o.checked_add(4)?)?.try_into().ok()?) as usize) };
    if b.get(0..4)? != b"DXBC" || b.len() < 32 {
        return None;
    }
    let count = rd(28)?;
    if count > 64 {
        return None;
    }
    let mut out = Vec::new();
    for i in 0..count {
        let off = rd(32 + 4 * i)?;
        let cc: [u8; 4] = b.get(off..off.checked_add(4)?)?.try_into().ok()?;
        let size = rd(off.checked_add(4)?)?;
        if off.checked_add(8)?.checked_add(size)? > b.len() {
            return None;
        }
        out.push((cc, off + 8, size));
    }
    Some(out)
}

/// A FourCC label: ASCII graphic characters kept, anything else shown as '?'.
fn sane_fourcc(s: &str) -> bool {
    (1..=4).contains(&s.chars().count()) && s.chars().all(|c| c.is_ascii_graphic())
}

/// The table is read in order, one `Part` per entry, offsets pointing past the 8-byte part header.
#[test]
fn part_table_round_trips() {
    let blob = container(&[(b"SFI0", &[1; 8]), (b"EMPT", &[]), (b"DXIL", &[2; 13]), (b"PRIV", &[3; 3])]);
    let parts = container_parts(&blob).unwrap();
    let want: Vec<Part> = [("SFI0", 56, 8), ("EMPT", 72, 0), ("DXIL", 80, 13), ("PRIV", 101, 3)]
        .iter()
        .map(|&(f, offset, size)| Part { fourcc: f.into(), offset, size })
        .collect();
    assert_eq!(parts, want);
    assert_eq!(&blob[parts[2].offset..parts[2].offset + parts[2].size], &[2; 13]);
    // the header's total size and version are not consulted
    let mut odd = blob.clone();
    put32(&mut odd, 20, 0xFFFF_FFFF);
    put32(&mut odd, 24, 7);
    assert_eq!(container_parts(&odd).unwrap(), want);
}

/// Count 0 is an empty table, 64 parts are read, 65 refuse the container.
#[test]
fn part_count_limits() {
    assert_eq!(container_parts(&raw_container(0, &[], &[])), Some(vec![]));
    let data = [9u8; 3];
    let many: Vec<(&[u8; 4], &[u8])> = (0..65).map(|_| (b"PART", &data[..])).collect();
    let at64 = container(&many[..64]);
    let p = container_parts(&at64).unwrap();
    assert_eq!(p.len(), 64);
    assert!(p.iter().all(|p| p.fourcc == "PART" && p.size == 3));
    assert_eq!(container_parts(&container(&many)), None);
    // a count the table does not hold
    let mut short = at64.clone();
    put32(&mut short, 28, 64);
    short.truncate(32 + 10 * 4);
    assert_eq!(container_parts(&short), None);
    assert_eq!(container_parts(&raw_container(u32::MAX, &[], &[0; 64])), None);
}

/// Parts are not checked against each other: aliased and overlapping parts, and a part that
/// reads the container header, all come back as written.
#[test]
fn overlapping_parts_are_kept() {
    let mut body = b"AAAA".to_vec();
    le32(&mut body, 12);
    body.extend_from_slice(b"BBBB");
    le32(&mut body, 0);
    body.extend_from_slice(&[0; 4]);
    // part 0 at 48, part 1 at 56 inside part 0's data, part 2 = part 0 again, part 3 the header
    let blob = raw_container(4, &[48, 56, 48, 0], &body);
    let p = container_parts(&blob).unwrap();
    assert_eq!(p[0], Part { fourcc: "AAAA".into(), offset: 56, size: 12 });
    assert_eq!(p[1], Part { fourcc: "BBBB".into(), offset: 64, size: 0 });
    assert_eq!(p[2], p[0]);
    // the hash bytes (zero here) read as the size
    assert_eq!(p[3], Part { fourcc: "DXBC".into(), offset: 8, size: 0 });
}

/// A part whose header or data runs past the blob refuses the whole container.
#[test]
fn part_ranges_past_the_blob_are_rejected() {
    let blob = container(&[(b"SFI0", &[0; 8]), (b"DXIL", &[1; 16])]);
    let last = 32 + 2 * 4 + 16;
    assert!(container_parts(&blob).is_some());
    assert_eq!(container_parts(&blob[..blob.len() - 1]), None);
    for size in [17u32, 0x8000_0000, u32::MAX] {
        let mut b = blob.clone();
        put32(&mut b, last + 4, size);
        assert_eq!(container_parts(&b), None, "size {size:#x}");
    }
    let len = blob.len() as u32;
    for off in [len, len - 4, len - 7, len + 1, u32::MAX - 3, u32::MAX] {
        let mut b = blob.clone();
        put32(&mut b, 36, off);
        assert_eq!(container_parts(&b), None, "offset {off:#x}");
    }
    // a part of size 0 that ends exactly at the end of the blob
    let p = container_parts(&container(&[(b"SFI0", &[0; 8]), (b"ZERO", &[])])).unwrap();
    assert_eq!(p[1], Part { fourcc: "ZERO".into(), offset: 64, size: 0 });
    assert_eq!(container_parts(&b"DXBC"[..]), None);
    assert_eq!(container_parts(&[0u8; 31]), None);
    let mut wrong = blob.clone();
    wrong[0..4].copy_from_slice(b"DXBD");
    assert_eq!(container_parts(&wrong), None);
}

/// FourCC bytes that are not ASCII graphic characters show as '?', the rest as they are.
#[test]
fn fourcc_is_sanitized() {
    for b0 in 0..=255u8 {
        let cc = [b0, b'X', b'I', b'L'];
        let p = container_parts(&container(&[(&cc, &[])])).unwrap();
        let want = if b0.is_ascii_graphic() { format!("{}XIL", b0 as char) } else { "?XIL".into() };
        assert_eq!(p[0].fourcc, want, "byte {b0:#04x}");
    }
    let p = container_parts(&container(&[(b" \0\x7f~", &[]), (b"\xff\xfe\x80\x81", &[])])).unwrap();
    assert_eq!((p[0].fourcc.as_str(), p[1].fourcc.as_str()), ("???~", "????"));
}

/// Random mutations of a container (bytes, table words) never panic, and the table agrees with
/// the reference reading; FourCCs are always printable.
#[test]
fn part_table_matches_reference_under_mutation() {
    let base = container(&[(b"SFI0", &[0; 8]), (b"ISG1", &[1; 40]), (b"DXIL", &[2; 64]), (b"HASH", &[3; 20])]);
    let mut rng = Lcg(0x5eed_0001);
    for _ in 0..4000 {
        let mut b = base.clone();
        match rng.below(3) {
            0 => {
                for _ in 0..1 + rng.below(4) {
                    let i = rng.below(b.len());
                    b[i] = rng.next() as u8;
                }
            }
            1 => {
                let o = rng.below(b.len() / 4) * 4;
                let vals = [0, 1, 64, 65, b.len() as u32, b.len() as u32 - 8, u32::MAX, 0x8000_0000, rng.next()];
                put32(&mut b, o, vals[rng.below(vals.len())]);
            }
            _ => b.truncate(rng.below(b.len() + 1)),
        }
        let got = guarded("container_parts", &b, || container_parts(&b));
        let want = ref_parts(&b);
        assert_eq!(got.is_some(), want.is_some(), "{}", hex(&b));
        for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
            assert_eq!((g.offset, g.size), (w.1, w.2), "{}", hex(&b));
            assert!(sane_fourcc(&g.fourcc), "{:?}", g.fourcc);
            if w.0.iter().all(|c| c.is_ascii()) {
                let s: String = w.0.iter().map(|&c| if c.is_ascii_graphic() { c as char } else { '?' }).collect();
                assert_eq!(g.fourcc, s);
            }
        }
    }
}

// ---- console containers (gpu::xbox) ----

/// s_mov_b32 s0, s1 ; v_add_f32 v0, v1, v5 ; s_endpgm
const CODE: [u8; 12] = [0x01, 0x03, 0x80, 0xbe, 0x01, 0x0b, 0x00, 0x06, 0x00, 0x00, 0x81, 0xbf];
const CODE_AT: usize = 0x308;

/// FSSM: magic, version 0x101, size; code pointer (relative to +0x10) and size at +0xC0/+0xC8;
/// stage descriptors 0x000000C0_0002_AAHH from +0x100 every 0x88 B; code, s_code_end, name.
fn fssm(code: &[u8], stages: &[(u8, u8)]) -> Vec<u8> {
    let mut v = vec![0u8; CODE_AT];
    v[0..4].copy_from_slice(b"FSSM");
    put32(&mut v, 4, 0x101);
    for q in (0x18..0xC0).step_by(4) {
        put32(&mut v, q, 0xDEAD_BEEF);
    }
    v[0xC0..0xC8].copy_from_slice(&((CODE_AT - 0x10) as u64).to_le_bytes());
    v[0xC8..0xD0].copy_from_slice(&(code.len() as u64).to_le_bytes());
    for (i, &(hw, api)) in stages.iter().enumerate() {
        let q = 0x0000_00C0_0002_0000u64 | (api as u64) << 8 | hw as u64;
        v[0x100 + i * 0x88..0x108 + i * 0x88].copy_from_slice(&q.to_le_bytes());
    }
    v.extend_from_slice(code);
    v.extend_from_slice(&[0x00, 0x00, 0x9f, 0xbf, 0x00, 0x00, 0x9f, 0xbf]);
    v.extend_from_slice(b"PS_Main\0");
    let size = (v.len() - 0x10) as u32;
    put32(&mut v, 8, size);
    v
}

/// XBCH: magic, version 3, compressed size, raw size at +0x10; zlib stream at `at` (0x18 normally).
fn xbch_with(fssm: &[u8], csize: Option<u32>, rsize: u32, at: usize) -> Vec<u8> {
    let z = miniz_oxide::deflate::compress_to_vec_zlib(fssm, 6);
    let mut v = b"XBCH".to_vec();
    le32(&mut v, 3);
    le32(&mut v, csize.unwrap_or(z.len() as u32));
    le32(&mut v, 0);
    le32(&mut v, rsize);
    le32(&mut v, 0);
    v.resize(at, 0);
    v.extend_from_slice(&z);
    v
}

fn xbch(fssm: &[u8]) -> Vec<u8> {
    xbch_with(fssm, None, fssm.len() as u32, 0x18)
}

/// DXIL part: program header (kind/SM word, size, "DXIL", version, bitcode offset and size),
/// XDAT at `x` with its table right after its 12-byte header, entry data (offsets relative to
/// the program header), "NOBC".
fn dxil_part(pv: u32, x: usize, blobs: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut v = Vec::new();
    le32(&mut v, pv);
    le32(&mut v, 0);
    v.extend_from_slice(b"DXIL");
    le32(&mut v, 0x106);
    le32(&mut v, 0);
    le32(&mut v, 4);
    v.resize(x, 0);
    v.extend_from_slice(b"XDAT");
    le32(&mut v, blobs.len() as u32);
    le32(&mut v, (x + 12) as u32);
    let mut at = x + 12 + blobs.len() * 12;
    for (id, b) in blobs {
        le32(&mut v, *id);
        le32(&mut v, at as u32);
        le32(&mut v, b.len() as u32);
        at += b.len();
    }
    for (_, b) in blobs {
        v.extend_from_slice(b);
    }
    v.extend_from_slice(b"NOBC");
    v
}

fn hash() -> Vec<u8> {
    (0u8..16).collect()
}

fn nested() -> Vec<u8> {
    let mut n = dxil_part(0x66, 0x18, &[]);
    n.truncate(0x18);
    n.extend_from_slice(b"NOBC");
    n
}

fn standard_entries(xbchs: &[Vec<u8>]) -> Vec<(u32, Vec<u8>)> {
    let mut e = vec![(XDAT_HASH, hash())];
    e.extend(xbchs.iter().map(|x| (XDAT_XBCH, x.clone())));
    e.push((XDAT_NESTED_HEADER, nested()));
    e
}

fn wrap(dxil: &[u8]) -> Vec<u8> {
    container(&[(b"SFI0", &[0; 8]), (b"HASH", &[0; 20]), (b"DXIL", dxil), (b"PRIV", &[7; 16])])
}

fn sample(kind: u32, fssms: &[Vec<u8>]) -> Vec<u8> {
    let x: Vec<Vec<u8>> = fssms.iter().map(|f| xbch(f)).collect();
    wrap(&dxil_part(kind << 16 | 0x66, 0x18, &standard_entries(&x)))
}

/// Every view of a parsed shader stays inside its buffers; the consumers do not panic.
fn check_shader(blob: &[u8], sh: &XboxShader) {
    for e in &sh.entries {
        assert!(e.offset >= sh.dxil_offset && e.offset + e.size <= blob.len(), "{e:?}");
    }
    for f in &sh.fssm {
        if let Some((o, l)) = f.code {
            assert!(o + l <= f.data.len(), "code {o:#x}+{l:#x} of {:#x}", f.data.len());
            for p in f.programs.iter().chain(&f.data_blocks) {
                assert!(p.offset >= o && p.offset + p.len <= o + l, "{p:?} outside the code");
            }
        } else {
            assert!(f.programs.is_empty() && f.data_blocks.is_empty());
        }
        for s in &f.stages {
            assert!(s.offset + 8 <= f.data.len() && s.hw <= 7);
        }
        let _ = (f.code_bytes().len(), f.stage_label(), header_qwords(f, usize::MAX));
    }
    let _ = listing(sh);
    let mut st = InstStats::default();
    st.add_shader(sh);
}

fn parse_checked(blob: &[u8]) -> Option<XboxShader> {
    guarded("xbox::parse", blob, || {
        let sh = xbox::parse(blob);
        assert_eq!(is_xbox(blob), sh.is_some());
        if let Some(sh) = &sh {
            check_shader(blob, sh);
        }
        sh
    })
}

/// A two-variant vertex shader (ES + VS) parses back to exactly what was built: entry ranges,
/// the hash, both FSSMs byte for byte, their stage descriptors and programs.
#[test]
fn console_container_round_trips() {
    let es = fssm(&CODE, &[(2, 0)]);
    let vs = fssm(&CODE, &[(4, 0), (5, 1)]);
    let blob = sample(1, &[es.clone(), vs.clone()]);
    let sh = parse_checked(&blob).unwrap();
    let dxil = container_parts(&blob).unwrap().into_iter().find(|p| p.fourcc == "DXIL").unwrap();
    assert_eq!((sh.stage.as_str(), sh.shader_model), ("VS", (6, 6)));
    assert_eq!((sh.dxil_offset, sh.xdat_offset), (dxil.offset, dxil.offset + 0x18));
    assert!(sh.notes.is_empty(), "{:?}", sh.notes);
    assert_eq!(sh.entries.iter().map(|e| e.id).collect::<Vec<_>>(), vec![XDAT_HASH, XDAT_XBCH, XDAT_XBCH, XDAT_NESTED_HEADER]);
    let table = dxil.offset + 0x18 + 12 + 4 * 12;
    assert_eq!((sh.entries[0].offset, sh.entries[0].size), (table, 16));
    assert_eq!(sh.entries[1].offset, table + 16);
    assert!(blob[sh.entries[1].offset..].starts_with(b"XBCH"));
    assert_eq!(sh.hash, Some(core::array::from_fn(|i| i as u8)));
    assert_eq!(sh.fssm.len(), 2);
    for (f, want, label) in [(&sh.fssm[0], &es, "ES"), (&sh.fssm[1], &vs, "VS+PS")] {
        assert_eq!(&f.data, want);
        assert_eq!((f.error.as_deref(), f.version, f.xbch_version), (None, 0x101, 3));
        assert_eq!(f.stage_label(), label);
        assert!(f.code_from_header);
        assert_eq!(f.code, Some((CODE_AT, CODE.len())));
        assert_eq!(f.code_bytes(), &CODE);
        assert_eq!(f.programs.len(), 1);
        assert_eq!((f.programs[0].offset, f.programs[0].len), (CODE_AT, 12));
    }
    assert_eq!(sh.fssm[0].xbch_offset, sh.entries[1].offset);
    assert_eq!(sh.fssm[1].stages.iter().map(|s| s.offset).collect::<Vec<_>>(), vec![0x100, 0x188]);
    assert_eq!(sh.fssm[1].stages[1].label(), "PS (API PS)");
}

/// The program header word gives the API stage (DXIL shader kind, bits 16+) and shader model
/// (major bits 4-7, minor bits 0-3).
#[test]
fn program_header_kind_and_shader_model() {
    let kinds = ["PS", "VS", "GS", "HS", "DS", "CS", "Library", "RayGen", "Intersection", "AnyHit", "ClosestHit", "Miss", "Callable", "MS", "AS", "Node"];
    let f = xbch(&fssm(&CODE, &[(6, 3)]));
    for k in 0..18u32 {
        let pv = k << 16 | 0x60 | (k % 9);
        let sh = xbox::parse(&wrap(&dxil_part(pv, 0x18, &standard_entries(&[f.clone()])))).unwrap();
        let want = kinds.get(k as usize).map(|s| s.to_string()).unwrap_or(format!("kind#{k}"));
        assert_eq!((sh.stage, sh.shader_model), (want, (6, (k % 9) as u8)));
    }
}

/// The XDAT table count: 0 and over 64 leave the table unread (with a note), 64 is read.
#[test]
fn xdat_count_limits() {
    let f = xbch(&fssm(&CODE, &[(5, 1)]));
    let mut d = dxil_part(0x66, 0x18, &standard_entries(&[f.clone()]));
    put32(&mut d, 0x18 + 4, 0);
    let sh = parse_checked(&wrap(&d)).unwrap();
    assert!(sh.entries.is_empty() && sh.fssm.is_empty());
    assert!(sh.notes.iter().any(|n| n.contains("XDAT table")), "{:?}", sh.notes);

    let mut many = vec![(XDAT_HASH, hash()), (XDAT_XBCH, f.clone())];
    many.extend((2..64).map(|i| (0x40 + i, vec![i as u8; 4])));
    let sh = parse_checked(&wrap(&dxil_part(0x66, 0x18, &many))).unwrap();
    assert_eq!((sh.entries.len(), sh.fssm.len()), (64, 1));
    assert!(sh.notes.is_empty(), "{:?}", sh.notes);

    many.push((0x99, vec![0; 4]));
    let sh = parse_checked(&wrap(&dxil_part(0x66, 0x18, &many))).unwrap();
    assert!(sh.entries.is_empty() && sh.fssm.is_empty());
}

/// Where the table and the entries may sit: the stated table offset or right after the XDAT
/// header; entry offsets relative to the program header or to XDAT; never over the table or
/// past the DXIL part.
#[test]
fn xdat_table_and_entry_ranges() {
    let f = xbch(&fssm(&CODE, &[(5, 1)]));
    let base = dxil_part(0x66, 0x18, &standard_entries(&[f.clone()]));
    let entry = |i: usize| 0x18 + 12 + i * 12;
    let table_end = entry(3);
    let entries = |d: &[u8]| parse_checked(&wrap(d)).unwrap().entries.len();
    assert_eq!(entries(&base), 3);

    // a stated table offset that does not check out: the table right after the header is used
    for t in [0u32, 0x10, 0x1000, u32::MAX] {
        let mut d = base.clone();
        put32(&mut d, 0x18 + 8, t);
        assert_eq!(entries(&d), 3, "table offset {t:#x}");
    }
    // entry offsets relative to XDAT
    let mut d = base.clone();
    for i in 0..3 {
        let o = u32::from_le_bytes(d[entry(i) + 4..entry(i) + 8].try_into().unwrap());
        put32(&mut d, entry(i) + 4, o - 0x18);
    }
    let sh = parse_checked(&wrap(&d)).unwrap();
    assert_eq!((sh.entries.len(), sh.fssm.len(), sh.hash.is_some()), (3, 1, true));

    let len = base.len() as u32;
    let hash_at = table_end as u32;
    for (what, off, size) in [
        ("over the table", hash_at - 4, 16),
        ("over the XDAT header", 0x18, 16),
        ("size past the part", hash_at, len - hash_at + 1),
        ("offset past the part", len + 4, 0),
        ("offset at u32::MAX", u32::MAX, 16),
        ("size at u32::MAX", hash_at, u32::MAX),
    ] {
        let mut d = base.clone();
        put32(&mut d, entry(0) + 4, off);
        put32(&mut d, entry(0) + 8, size);
        let sh = parse_checked(&wrap(&d)).unwrap();
        assert!(sh.entries.is_empty() && sh.fssm.is_empty(), "{what}: {:?}", sh.entries);
    }
    // an entry that ends exactly at the end of the part, and two entries on the same XBCH
    let mut d = base.clone();
    put32(&mut d, entry(0) + 8, len - hash_at);
    assert_eq!(entries(&d), 3);
    let mut d = base.clone();
    d[entry(2)..entry(3)].copy_from_slice(&base[entry(1)..entry(2)].to_vec());
    let sh = parse_checked(&wrap(&d)).unwrap();
    assert_eq!(sh.fssm.len(), 2);
    assert_eq!(sh.fssm[0].data, sh.fssm[1].data);
}

/// XDAT is looked for on dword boundaries in the first 256 bytes of the DXIL part.
#[test]
fn xdat_position() {
    let x = vec![xbch(&fssm(&CODE, &[(5, 1)]))];
    let sh = parse_checked(&wrap(&dxil_part(0x66, 0x1C, &standard_entries(&x)))).unwrap();
    assert_eq!(sh.xdat_offset - sh.dxil_offset, 0x1C);
    assert!(sh.notes.iter().any(|n| n.contains("expected +0x18")), "{:?}", sh.notes);
    assert_eq!(sh.fssm.len(), 1);
    assert!(parse_checked(&wrap(&dxil_part(0x66, 0xFC, &standard_entries(&x)))).is_some());
    assert!(parse_checked(&wrap(&dxil_part(0x66, 0x100, &standard_entries(&x)))).is_none());
    // not on a dword boundary
    let mut d = dxil_part(0x66, 0x18, &standard_entries(&x));
    d.insert(0x18, 0);
    assert!(parse_checked(&wrap(&d)).is_none());
    // no DXIL part, or a PC DXIL part
    assert!(parse_checked(&container(&[(b"DXIL", b"\x66\0\0\0\x08\0\0\0DXIL\0\0\0\0BC\xc0\xde")])).is_none());
    let mut renamed = wrap(&dxil_part(0x66, 0x18, &standard_entries(&x)));
    let p = container_parts(&renamed).unwrap()[2].offset;
    renamed[p - 8..p - 4].copy_from_slice(b"DXIM");
    assert!(parse_checked(&renamed).is_none());
}

/// The XBCH raw size bounds decompression: exact is clean, smaller stops inflation with an
/// error and no data, larger (or over the 64 MiB cap) keeps the data with a size note, 0 is
/// "unknown".
#[test]
fn xbch_raw_size_limits() {
    let f = fssm(&CODE, &[(5, 1)]);
    let n = f.len() as u32;
    let one = |rsize: u32| {
        let blob = wrap(&dxil_part(0x66, 0x18, &standard_entries(&[xbch_with(&f, None, rsize, 0x18)])));
        parse_checked(&blob).unwrap().fssm.remove(0)
    };
    let exact = one(n);
    assert_eq!((exact.error, exact.data.len()), (None, f.len()));
    let unknown = one(0);
    assert_eq!((unknown.error, &unknown.data), (None, &f));
    for small in [n - 1, 1] {
        let s = one(small);
        assert!(s.data.is_empty() && s.programs.is_empty(), "raw size {small}");
        assert!(s.error.as_deref().is_some_and(|e| e.contains("zlib error")), "{:?}", s.error);
    }
    for big in [n + 1, 64 << 20, (64 << 20) + 1, u32::MAX] {
        let b = one(big);
        assert_eq!(b.data, f, "raw size {big}");
        assert_eq!(b.error, Some(format!("XBCH: decompressed {n} B, the header says {big} B")));
        assert_eq!(b.programs.len(), 1);
    }
}

/// The compressed size and where the zlib stream starts.
#[test]
fn xbch_compressed_size_and_stream_search() {
    let f = fssm(&CODE, &[(5, 1)]);
    let one = |x: Vec<u8>| parse_checked(&wrap(&dxil_part(0x66, 0x18, &standard_entries(&[x])))).unwrap().fssm.remove(0);
    let z = miniz_oxide::deflate::compress_to_vec_zlib(&f, 6).len() as u32;
    // a size past the entry, or 0: the rest of the entry is the stream
    for c in [0, z + 1, u32::MAX] {
        let r = one(xbch_with(&f, Some(c), f.len() as u32, 0x18));
        assert_eq!((r.error, r.compressed, &r.data), (None, z as usize, &f), "csize {c:#x}");
    }
    let r = one(xbch_with(&f, Some(z - 4), f.len() as u32, 0x18));
    assert_eq!(r.compressed, z as usize - 4);
    assert!(r.error.is_some() && r.data.is_empty(), "a stream cut before its checksum: {:?}", r.error);
    // the zlib header is searched for in the first 64 bytes
    let r = one(xbch_with(&f, None, f.len() as u32, 0x30));
    assert_eq!((r.error, &r.data), (None, &f));
    let r = one(xbch_with(&f, None, f.len() as u32, 0x40));
    assert_eq!(r.error.as_deref(), Some("XBCH: zlib stream not found"));
    // a corrupt stream, a stream that is not an FSSM
    let mut x = xbch(&f);
    x[0x18 + 6] ^= 0xFF;
    assert!(one(x).error.is_some());
    let r = one(xbch_with(b"not an fssm", None, 11, 0x18));
    assert_eq!((r.error.as_deref(), r.code), (Some("no FSSM header"), None));
}

/// A header code pointer that does not check out falls back to scanning the FSSM for s_endpgm.
#[test]
fn fssm_code_pointer_edges() {
    let base = fssm(&CODE, &[(5, 1)]);
    let total = base.len() as u64;
    for (off, len) in [(0x2F8u64, 0u64), (0x2F9, 12), (total, 4), (0x2F8, total), (u64::MAX - 0xF, 12), (u64::MAX, u64::MAX), (0x2F8, 8)] {
        let mut f = base.clone();
        f[0xC0..0xC8].copy_from_slice(&off.to_le_bytes());
        f[0xC8..0xD0].copy_from_slice(&len.to_le_bytes());
        let sh = parse_checked(&sample(0, &[f])).unwrap();
        let f = &sh.fssm[0];
        assert!(!f.code_from_header, "pointer {off:#x}+{len:#x}");
        assert_eq!(f.code, Some((CODE_AT, 12)), "pointer {off:#x}+{len:#x}");
        assert_eq!(f.programs.len(), 1);
    }
    // no s_endpgm anywhere: no code region at all
    let mut f = fssm(&[0x01, 0x03, 0x80, 0xbe], &[(5, 1)]);
    f[0xC8..0xD0].copy_from_slice(&4u64.to_le_bytes());
    let f = &parse_checked(&sample(0, &[f])).unwrap().fssm[0];
    assert_eq!((f.code, f.code_bytes().len()), (None, 0));
    assert!(header_qwords(f, usize::MAX).iter().all(|&(o, _)| o + 8 <= f.data.len()));
}

/// Stage descriptors are qwords 0x000000C0_0002_AAHH with HH <= 7, on the 8-byte grid from +0x10
/// up to the code; anything else is not a stage. `header_qwords` lists the non-filler qwords
/// in front of the code.
#[test]
fn stage_descriptors_and_header_qwords() {
    let mut f = fssm(&CODE, &[(5, 1), (8, 1), (0, 9)]);
    f[0x10..0x18].copy_from_slice(&0x0000_00C0_0003_0105u64.to_le_bytes()); // wrong tag
    f[0x2A4..0x2AC].copy_from_slice(&0x0000_00C0_0002_0105u64.to_le_bytes()); // off the grid
    f.extend_from_slice(&[0; 4]);
    f.extend_from_slice(&0x0000_00C0_0002_0004u64.to_le_bytes()); // after the code
    let sh = parse_checked(&sample(0, &[f])).unwrap();
    let st = &sh.fssm[0].stages;
    assert_eq!(st.iter().map(|s| (s.offset, s.hw, s.api)).collect::<Vec<_>>(), vec![(0x100, 5, 1), (0x210, 0, 9)]);
    assert_eq!((st[0].label(), st[1].label()), ("PS (API PS)".to_string(), "LS (API api#9)".to_string()));
    assert_eq!(sh.fssm[0].stage_label(), "PS+LS");
    let hw: Vec<String> = (0..9).map(xbox::hw_stage_name).collect();
    assert_eq!(hw, ["LS", "HS", "ES", "GS", "VS", "PS", "CS", "NGG", "hw#8"]);
    let api: Vec<String> = (0..8).map(xbox::api_stage_name).collect();
    assert_eq!(api, ["VS", "PS", "GS", "CS", "api#4", "api#5", "MS", "api#7"]);
    let q: Vec<usize> = header_qwords(&sh.fssm[0], usize::MAX).iter().map(|&(o, _)| o).collect();
    assert_eq!(q, [0, 8, 0x10, 0xC0, 0xC8, 0x100, 0x188, 0x210, 0x2A0, 0x2A8]);
    assert_eq!(header_qwords(&sh.fssm[0], 0x100).last().map(|&(o, _)| o), Some(0xC8));
}

/// Every truncation of the DXIL part, of an XBCH entry and of the FSSM inside it parses
/// without a panic; a cut XBCH always reports an error.
#[test]
fn truncations_never_panic() {
    let f = fssm(&CODE, &[(4, 0)]);
    let x = xbch(&f);
    let d = dxil_part(0x10066, 0x18, &standard_entries(&[x.clone()]));
    for n in 0..=d.len() {
        parse_checked(&wrap(&d[..n]));
    }
    let blob = wrap(&d);
    for n in 0..blob.len() {
        assert!(parse_checked(&blob[..n]).is_none(), "prefix {n}");
    }
    for n in 0..x.len() {
        let sh = parse_checked(&wrap(&dxil_part(0x10066, 0x18, &standard_entries(&[x[..n].to_vec()])))).unwrap();
        if n < 4 {
            assert!(sh.fssm.is_empty() && sh.notes.iter().any(|m| m.contains("without an XBCH header")), "{n}: {:?}", sh.notes);
        } else {
            assert!(sh.fssm[0].error.is_some(), "XBCH cut to {n} B");
        }
    }
    for n in 0..=f.len() {
        let cut = xbch_with(&f[..n], None, n as u32, 0x18);
        parse_checked(&wrap(&dxil_part(0x10066, 0x18, &standard_entries(&[cut]))));
    }
}

/// Seeded random mutations of the container, the XDAT table and the decompressed FSSM never
/// panic, and whatever parses keeps every range inside its buffer.
#[test]
fn mutations_never_panic() {
    let mut rng = Lcg(0x5eed_0003);
    let mut two = CODE.to_vec();
    two.extend_from_slice(&[0x00, 0x00, 0x9f, 0xbf]);
    two.extend_from_slice(&CODE);
    let f = fssm(&two, &[(3, 2), (4, 2)]);
    let blob = sample(2, &[f.clone()]);
    let dxil = container_parts(&blob).unwrap()[2].offset;
    let words = [0, 1, 4, 0x18, 63, 64, 65, 0xDEAD_BEEF, 0xBF81_0000, 0xBF9F_0000, 0x7FFF_FFFF, u32::MAX];
    for i in 0..3000 {
        let mut b = blob.clone();
        for _ in 0..1 + rng.below(3) {
            // most mutations land in the DXIL part's headers, where the parser makes decisions
            let o = if i % 2 == 0 { dxil + rng.below(0x80) } else { rng.below(b.len()) };
            if rng.below(2) == 0 {
                b[o] = rng.next() as u8;
            } else if o + 4 <= b.len() {
                put32(&mut b, o & !3, words[rng.below(words.len())]);
            }
        }
        parse_checked(&b);
    }
    for _ in 0..600 {
        let mut m = f.clone();
        for _ in 0..1 + rng.below(6) {
            let o = if rng.below(2) == 0 { CODE_AT - 8 + rng.below(m.len() - CODE_AT + 8) } else { rng.below(m.len()) };
            if rng.below(2) == 0 {
                m[o] = rng.next() as u8;
            } else if o + 4 <= m.len() {
                put32(&mut m, o & !3, words[rng.below(words.len())]);
            }
        }
        parse_checked(&sample(2, &[m]));
    }
}
