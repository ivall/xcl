//! The parts of a shader blob that link stages on PC (`gpu::dxbc`: ISGN/ISG1/OSGN/OSG1
//! signatures and PSV0) and the packed shader archives titles ship (`gpu::pack`: plain records
//! inside RPF9 packs). Inputs are built here from the documented layouts; malformed ones must give
//! `None`, never a panic.

use std::panic::{catch_unwind, AssertUnwindSafe};

use xcl::gpu::dxbc::{psv, signature, signatures};
use xcl::gpu::pack::{inflate, name_stage, scan, Record, RPF9_MAGIC};

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

// ---- signatures and PSV0 ----

struct El<'a> {
    name: &'a str,
    index: u32,
    sv: u32,
    ty: u32,
    reg: u32,
    mask: u8,
    rw: u8,
    stream: u32,
}

/// Signature part: u32 count, u32 offset of the first element (8), elements, names. An element
/// is name offset, semantic index, system value, component type, register, mask, rw mask (24 B);
/// the v1 parts (ISG1/OSG1/PSG1) add a stream before and a min precision after (32 B).
fn sig_part(v1: bool, els: &[El]) -> Vec<u8> {
    let size = if v1 { 32 } else { 24 };
    let mut names = Vec::new();
    let mut v = Vec::new();
    le32(&mut v, els.len() as u32);
    le32(&mut v, 8);
    for e in els {
        if v1 {
            le32(&mut v, e.stream);
        }
        le32(&mut v, (8 + els.len() * size + names.len()) as u32);
        names.extend_from_slice(e.name.as_bytes());
        names.push(0);
        for x in [e.index, e.sv, e.ty, e.reg] {
            le32(&mut v, x);
        }
        v.extend_from_slice(&[e.mask, e.rw, 0, 0]);
        if v1 {
            le32(&mut v, 0);
        }
    }
    v.extend_from_slice(&names);
    v
}

fn els() -> Vec<El<'static>> {
    vec![
        El { name: "SV_Position", index: 0, sv: 1, ty: 3, reg: 0, mask: 0xF, rw: 0xF, stream: 0 },
        El { name: "TEXCOORD", index: 3, sv: 0, ty: 3, reg: 1, mask: 0x3, rw: 0x1, stream: 2 },
        El { name: "", index: 7, sv: 0, ty: 1, reg: 9, mask: 0x8, rw: 0, stream: 0 },
    ]
}

/// v0 and v1 signature parts read back element for element; `signatures` prefers v1 parts and
/// falls back to v0.
#[test]
fn signatures_round_trip() {
    let want = els();
    for (v1, cc) in [(false, b"OSGN"), (true, b"OSG1")] {
        let s = signature(&container(&[(cc, &sig_part(v1, &want))]), std::str::from_utf8(cc).unwrap()).unwrap();
        assert_eq!(s.len(), 3);
        for (g, w) in s.iter().zip(&want) {
            let stream = if v1 { w.stream } else { 0 };
            assert_eq!(
                (g.name.as_str(), g.index, g.system_value, g.component_type, g.register, g.mask, g.rw_mask, g.stream),
                (w.name, w.index, w.sv, w.ty, w.reg, w.mask, w.rw, stream)
            );
        }
        assert_eq!(s[1].semantic(), "TEXCOORD3");
    }
    let v0 = sig_part(false, &want[..1]);
    let v1 = sig_part(true, &want[1..]);
    let both = container(&[(b"ISGN", &v0), (b"ISG1", &v1), (b"OSGN", &v0)]);
    let (i, o) = signatures(&both);
    assert_eq!((i.len(), i[0].name.as_str(), o.len(), o[0].name.as_str()), (2, "TEXCOORD", 1, "SV_Position"));
    assert_eq!(signatures(&container(&[(b"SFI0", &[0; 8])])), (vec![], vec![]));
    assert_eq!(signature(&container(&[(b"ISG1", &sig_part(true, &[]))]), "ISG1"), Some(vec![]));
}

/// Element and name ranges are bounds-checked: an element past the part gives None, a name
/// offset past it gives an empty name, an unterminated name ends at the part's end.
#[test]
fn signature_ranges() {
    let p = sig_part(true, &els());
    let sig = |p: &[u8]| guarded("signature", p, || signature(&container(&[(b"ISG1", p)]), "ISG1"));
    for n in 0..p.len() {
        let s = sig(&p[..n]);
        // the last element is read up to its rw mask (+25); the names after it may be cut
        assert_eq!(s.is_some(), n >= 8 + 2 * 32 + 26, "cut to {n}");
    }
    let mut q = p.clone();
    put32(&mut q, 8 + 4, 0xFFFF_FFF0);
    assert_eq!(sig(&q).unwrap()[0].name, "");
    let mut q = p.clone();
    q.pop();
    assert_eq!(sig(&q).unwrap()[2].name, "");
    q.truncate(q.len() - 1);
    assert_eq!(sig(&q).unwrap()[1].name, "TEXCOORD");
    for first in [0xFFFF_FFFFu32, p.len() as u32] {
        let mut q = p.clone();
        put32(&mut q, 4, first);
        assert_eq!(sig(&q), None);
    }
}

/// PSV0 part: u32 size of PSVRuntimeInfo, then the info. Layout from DXC's
/// DxilPipelineStateValidation.h: v0 (24 B) is a 16-byte stage union (GS: input primitive,
/// output topology; MS: groupshared, -, payload, max vertices u16, max primitives u16) and the
/// min / max wave lane counts; v1 (36 B) adds the shader stage at 24, a u16 at 26 (GS max vertex
/// count; MS: primitive vectors u8, output topology u8 at 27); v2 (48 B) adds numthreads at 36.
fn psv_part(size: usize, stage: u8, union: [u32; 4], at26: u16, threads: [u32; 3]) -> Vec<u8> {
    let mut r = vec![0u8; size.max(48)];
    for (i, x) in union.iter().enumerate() {
        put32(&mut r, i * 4, *x);
    }
    put32(&mut r, 16, 4);
    put32(&mut r, 20, 64);
    r[24] = stage;
    r[26..28].copy_from_slice(&at26.to_le_bytes());
    for (i, x) in threads.iter().enumerate() {
        put32(&mut r, 36 + i * 4, *x);
    }
    r.truncate(size);
    let mut v = (size as u32).to_le_bytes().to_vec();
    v.extend_from_slice(&r);
    le32(&mut v, 0); // resource count
    v
}

/// Each PSVRuntimeInfo version yields exactly the fields it has; stage-specific fields only
/// for their stage.
#[test]
fn psv_versions_and_stages() {
    let read = |p: &[u8]| guarded("psv", p, || psv(&container(&[(b"PSV0", p)])));
    let union = [1024, 7, 256, 126 << 16 | 128];
    for size in [24, 36, 48, 52] {
        let ms = read(&psv_part(size, 13, union, 2 << 8 | 5, [32, 2, 1])).unwrap();
        assert_eq!((ms.min_wave, ms.max_wave), (4, 64));
        if size < 36 {
            assert_eq!((ms.kind, ms.ms_max_vertices, ms.ms_groupshared), (None, 0, 0), "v0 has no stage");
        } else {
            assert_eq!(ms.kind, Some(13));
            assert_eq!((ms.ms_groupshared, ms.ms_payload, ms.ms_max_vertices, ms.ms_max_primitives, ms.ms_topology), (1024, 256, 128, 126, 2));
            assert_eq!(ms.gs_max_vertex_count, 0);
        }
        assert_eq!(ms.numthreads, (size >= 48).then_some([32, 2, 1]));
        let gs = read(&psv_part(size, 2, [3, 5, 1, 0], 18, [0; 3])).unwrap();
        if size >= 36 {
            assert_eq!((gs.gs_input_primitive, gs.gs_output_topology, gs.gs_max_vertex_count), (3, 5, 18));
            assert_eq!((gs.ms_groupshared, gs.ms_topology), (0, 0));
        }
        let cs = read(&psv_part(size, 5, union, 0, [8, 8, 1])).unwrap();
        assert_eq!((cs.ms_groupshared, cs.gs_input_primitive), (0, 0));
    }
    assert_eq!(read(&psv_part(23, 0, [0; 4], 0, [0; 3])), None);
    let mut p = psv_part(48, 5, [0; 4], 0, [64, 1, 1]);
    p.truncate(4 + 47);
    assert_eq!(read(&p), None, "info past the part");
    let mut p = psv_part(48, 5, [0; 4], 0, [64, 1, 1]);
    put32(&mut p, 0, u32::MAX);
    assert_eq!(read(&p), None);
    assert_eq!(psv(&container(&[(b"ISG1", &[0; 8])])), None);
}

/// Truncations and random mutations of signature and PSV0 parts never panic.
#[test]
fn dxbc_parts_never_panic() {
    let sig = sig_part(true, &els());
    let ps = psv_part(48, 13, [1024, 0, 16, 64 << 16 | 64], 2 << 8, [128, 1, 1]);
    let mut rng = Lcg(0x5eed_0004);
    for (cc, part) in [(b"ISG1", &sig), (b"OSGN", &sig_part(false, &els())), (b"PSV0", &ps)] {
        let name = std::str::from_utf8(cc).unwrap();
        for n in 0..=part.len() {
            let b = container(&[(cc, &part[..n])]);
            guarded("dxbc", &b, || (signature(&b, name), psv(&b), signatures(&b)));
        }
        // a PSV0 part is not read as a signature: its first word is not an element count
        let sig_of = |b: &[u8]| if name == "PSV0" { None } else { signature(b, name) };
        for _ in 0..1500 {
            let mut p = part.clone();
            for _ in 0..1 + rng.below(4) {
                let o = rng.below(p.len());
                p[o] = rng.next() as u8;
                if rng.below(3) == 0 && o + 4 <= p.len() {
                    // keep the element count small enough to read (see signature_huge_count_is_none)
                    let w = [0, 1, 0x30, 0x7FFF_FFFF, u32::MAX][rng.below(5)];
                    put32(&mut p, o.max(4) & !3, w);
                }
            }
            if name != "PSV0" {
                put32(&mut p, 0, rng.below(8) as u32);
            }
            let b = container(&[(cc, &p)]);
            guarded("dxbc", &b, || (sig_of(&b), psv(&b), signatures(&b)));
        }
    }
}

// ---- packed shader archives ----

/// Record header: u8 len, name[len-1], 0x00, u8 flag, u32 csize, u32 unknown, u32 csize,
/// u32 usize, then `payload` (a raw deflate stream, unless the test passes something else).
fn record_raw(name: &[u8], flag: u8, c1: u32, c2: u32, u: u32, payload: &[u8]) -> Vec<u8> {
    let mut v = vec![(name.len() + 1) as u8];
    v.extend_from_slice(name);
    v.push(0);
    v.push(flag);
    for x in [c1, 0xddce7, c2, u] {
        le32(&mut v, x);
    }
    v.extend_from_slice(payload);
    v
}

fn record(name: &str, payload: &[u8]) -> Vec<u8> {
    let z = miniz_oxide::deflate::compress_to_vec(payload, 6);
    record_raw(name.as_bytes(), 50, z.len() as u32, z.len() as u32, payload.len() as u32, &z)
}

fn pack_head() -> Vec<u8> {
    let mut p = RPF9_MAGIC.to_vec();
    p.resize(0x40, 0xAA);
    p
}

/// A shader blob the size of a small real one, so its record is longer than the scan's 32-byte tail.
fn shader_blob(seed: u8) -> Vec<u8> {
    let code: Vec<u8> = (0..200u32).map(|i| (i * 7 + seed as u32) as u8).collect();
    container(&[(b"SFI0", &[0; 8]), (b"DXIL", &code)])
}

/// Every record a scan returns lies inside the pack, after the previous one, with a valid
/// header; inflating it never panics and gives exactly `usize` bytes when it succeeds.
fn check_records(d: &[u8], from: usize, recs: &[Record]) {
    let mut end = from;
    for r in recs {
        assert!(r.offset >= end && r.data + r.csize <= d.len(), "{r:?}");
        assert_eq!(r.data, r.offset + r.name.len() + 1 + 18, "{r:?}");
        assert!(r.name.len() >= 3 && r.name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b':'));
        assert!(r.csize > 0 && r.csize < 16 << 20 && r.usize > 0 && r.usize < 64 << 20);
        if let Some(u) = inflate(d, r) {
            assert_eq!(u.len(), r.usize);
        }
        let _ = name_stage(&r.name);
        end = r.data + r.csize;
    }
}

/// Records with the shortest and longest names the length byte allows, an effect-qualified
/// name, and a last record that ends exactly at the end of the pack all come back whole.
#[test]
fn records_round_trip() {
    let long = format!("PS_{}", "x".repeat(251));
    let names = ["VSa", long.as_str(), "fx:MS_Mesh_0badf00d_Wrapped", "CS_Last"];
    let blobs: Vec<Vec<u8>> = (0..4).map(|i| shader_blob(i as u8)).collect();
    let mut pack = pack_head();
    let mut starts = Vec::new();
    for (n, b) in names.iter().zip(&blobs) {
        starts.push(pack.len());
        pack.extend(record(n, b));
        pack.extend_from_slice(&[0x13, 0x37]);
    }
    pack.truncate(pack.len() - 2);
    let recs = guarded("scan", &pack, || scan(&pack, 0x40));
    check_records(&pack, 0x40, &recs);
    assert_eq!(recs.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), names);
    assert_eq!(recs.iter().map(|r| r.offset).collect::<Vec<_>>(), starts);
    assert_eq!(recs[3].data + recs[3].csize, pack.len());
    for (r, b) in recs.iter().zip(&blobs) {
        assert_eq!((r.flag, r.usize), (50, b.len()));
        assert_eq!(inflate(&pack, r).as_ref(), Some(b));
    }
    assert_eq!(recs.iter().map(|r| name_stage(&r.name)).collect::<Vec<_>>(), [Some("VS"), Some("PS"), Some("MS"), Some("CS")]);
    // from inside the long name, its suffix 'x' * 119 reads as a record with the same header
    // and payload (the alias that resuming after each payload skips); the next record follows
    let inner = scan(&pack, starts[1] + 1);
    assert_eq!((inner[0].name.clone(), inner[0].data), ("x".repeat(119), recs[1].data));
    assert_eq!(inner[1].offset, starts[2]);
    assert_eq!(scan(&pack, starts[3]).len(), 1);
    assert!(scan(&pack, pack.len()).is_empty());
}

/// Each header check on its own turns a record down: name characters, the terminator, the two
/// csize copies, zero sizes and the size caps (16 MiB compressed, 64 MiB raw).
#[test]
fn header_checks() {
    let z = miniz_oxide::deflate::compress_to_vec(&shader_blob(0), 6);
    let (c, u) = (z.len() as u32, shader_blob(0).len() as u32);
    let found = |rec: Vec<u8>| {
        let mut p = pack_head();
        p.extend(rec);
        p.extend_from_slice(&[0; 40]);
        let r = guarded("scan", &p, || scan(&p, 0x40));
        check_records(&p, 0x40, &r);
        r.iter().any(|r| r.offset == 0x40)
    };
    assert!(found(record_raw(b"PS_Ok", 0, c, c, u, &z)));
    assert!(found(record_raw(b"PSa", 0, c, c, u, &z)));
    assert!(!found(record_raw(b"PS", 0, c, c, u, &z)), "names are at least 3 bytes");
    for bad in [&b"PS-Dash"[..], b"PS.dot", b"9PS_Digit", b":PS_Colon", b"PS Space", b"PS_\xc3\xa9"] {
        assert!(!found(record_raw(bad, 0, c, c, u, &z)), "{:?}", String::from_utf8_lossy(bad));
    }
    let mut unterminated = record_raw(b"PS_Unterminated", 0, c, c, u, &z);
    unterminated[16] = b'x';
    assert!(!found(unterminated));
    assert!(!found(record_raw(b"PS_Mismatch", 0, c, c + 1, u, &z)));
    assert!(!found(record_raw(b"PS_Zero", 0, 0, 0, u, &z)));
    assert!(!found(record_raw(b"PS_ZeroRaw", 0, c, c, 0, &z)));
    assert!(found(record_raw(b"PS_BigRaw", 0, c, c, (64 << 20) - 1, &z)));
    assert!(!found(record_raw(b"PS_TooBigRaw", 0, c, c, 64 << 20, &z)));
    assert!(!found(record_raw(b"PS_PastEnd", 0, c + 41, c + 41, u, &z)));
    // the compressed cap, with that much data behind the header
    let big = vec![0u8; 16 << 20];
    assert!(found(record_raw(b"PS_BigC", 0, (16 << 20) - 1, (16 << 20) - 1, u, &big)));
    assert!(!found(record_raw(b"PS_TooBigC", 0, 16 << 20, 16 << 20, u, &big)));
}

/// `inflate` gives the payload only when the stream inflates to exactly `usize` bytes.
#[test]
fn inflate_size_limits() {
    let blob = shader_blob(3);
    let z = miniz_oxide::deflate::compress_to_vec(&blob, 6);
    let mut d = z.clone();
    d.extend_from_slice(&[0xEE; 16]);
    let r = Record { name: "PS_X".into(), offset: 0, flag: 0, data: 0, csize: z.len(), usize: blob.len() };
    let with = |f: &dyn Fn(&mut Record)| {
        let mut r = r.clone();
        f(&mut r);
        guarded("inflate", &d, || inflate(&d, &r))
    };
    assert_eq!(with(&|_| {}), Some(blob.clone()));
    assert_eq!(with(&|r| r.usize -= 1), None, "a stream past the stated size");
    assert_eq!(with(&|r| r.usize = 1), None);
    assert_eq!(with(&|r| r.usize += 1), None);
    assert_eq!(with(&|r| r.usize = usize::MAX), None);
    assert_eq!(with(&|r| r.usize = 0), None);
    assert_eq!(with(&|r| r.csize -= 3), None, "a cut stream");
    // bytes after the final block are not read
    assert_eq!(with(&|r| r.csize += 16), Some(blob.clone()));
    assert_eq!(with(&|r| r.csize += 17), None);
    assert_eq!(with(&|r| r.data = usize::MAX - 1), None);
    let mut bad = d.clone();
    bad[0] = 0xFF; // reserved block type 3
    let rr = r.clone();
    assert_eq!(guarded("inflate", &bad, || inflate(&bad, &rr)), None);
}

/// The stage is the first two characters of the name after its last ':', in any case.
#[test]
fn name_stage_edges() {
    for (name, want) in [
        ("ps_lower", Some("PS")),
        ("Vs", Some("VS")),
        ("a:b:GS_Inner", Some("GS")),
        ("HSx", Some("HS")),
        ("DS:AS_Amp", Some("AS")),
        ("fx:", None),
        ("", None),
        ("P", None),
        ("XS_Foo", None),
        ("P\u{e9}", None),
        ("\u{e9}S", None),
        ("CS:\u{1F600}", None),
    ] {
        assert_eq!(guarded("name_stage", name.as_bytes(), || name_stage(name)), want, "{name:?}");
    }
}

/// Seeded mutations of a pack (bytes and header words) never panic, and what the scan returns
/// stays consistent.
#[test]
fn mutated_packs_never_panic() {
    let mut base = pack_head();
    for (i, n) in ["PS_A_Wrapped", "fx:CS_B", "VS_C"].iter().enumerate() {
        base.extend(record(n, &shader_blob(i as u8)));
    }
    base.extend_from_slice(&[0; 8]);
    let words = [0u32, 1, 3, 0x30, (16 << 20) - 1, 64 << 20, 0x7FFF_FFFF, u32::MAX];
    let mut rng = Lcg(0x5eed_0005);
    for _ in 0..3000 {
        let mut p = base.clone();
        for _ in 0..1 + rng.below(4) {
            let o = 0x40 + rng.below(p.len() - 0x40);
            if rng.below(2) == 0 {
                p[o] = rng.next() as u8;
            } else if o + 4 <= p.len() {
                put32(&mut p, o, words[rng.below(words.len())]);
            }
        }
        if rng.below(4) == 0 {
            p.truncate(rng.below(p.len() + 1));
        }
        let from = [0x40, 0, rng.below(p.len() + 1)][rng.below(3)];
        guarded("scan", &p, || check_records(&p, from, &scan(&p, from)));
    }
}
