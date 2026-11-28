//! `xcl::pe` on a hand-built PE32+ image instead of system DLLs: every directory the loader reads
//! (imports, delay imports, relocations, TLS, exports with forwarders), checked in both layouts of
//! the same image – the file on disk (`Layout::File`) and the image mapped at its RVAs
//! (`Layout::Mapped`). Offsets follow the PE format spec ("PE Format", Microsoft Learn): COFF
//! file header, optional header (PE32+), section table, `.idata`, `.edata`, `.reloc`, `.tls`.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use xcl::pe::*;

const BASE: u64 = 0x1_8000_0000;
const HDRS: u32 = 0x400;
const IMAGE_SIZE: u32 = 0x8000;
const NT: usize = 0x80;
const OPT: usize = NT + 24;
const OPT_SIZE: usize = 112 + 16 * 8;
const SEC_TABLE: usize = OPT + OPT_SIZE;
/// The end of the section table: the last byte `Pe::parse` reads.
const HEADER_END: usize = SEC_TABLE + SECTIONS.len() * 40;

/// (name, va, vsize, raw_size, characteristics). `.text` has a zero-filled tail (vsize > raw),
/// `.data` a large one (.bss-like), `.strings` an 8-byte name with no NUL, `.reloc` vsize 0.
const SECTIONS: [(&[u8; 8], u32, u32, u32, u32); 5] = [
    (b".text\0\0\0", 0x1000, 0x300, 0x200, SCN_MEM_EXECUTE | SCN_MEM_READ | 0x20),
    (b".rdata\0\0", 0x2000, 0x1000, 0x1000, SCN_MEM_READ | 0x40),
    (b".data\0\0\0", 0x3000, 0x2000, 0x200, SCN_MEM_READ | SCN_MEM_WRITE | 0x40),
    (b".strings", 0x5000, 0x1200, 0x1200, SCN_MEM_READ | 0x40),
    (b".reloc\0\0", 0x7000, 0, 0x200, SCN_MEM_READ | 0x0200_0040),
];

fn raw_ptr(i: usize) -> u32 {
    HDRS + SECTIONS[..i].iter().map(|s| s.3).sum::<u32>()
}

fn put(v: &mut [u8], at: u32, b: &[u8]) {
    v[at as usize..at as usize + b.len()].copy_from_slice(b);
}
fn p16(v: &mut [u8], at: u32, x: u16) { put(v, at, &x.to_le_bytes()) }
fn p32(v: &mut [u8], at: u32, x: u32) { put(v, at, &x.to_le_bytes()) }
fn p64(v: &mut [u8], at: u32, x: u64) { put(v, at, &x.to_le_bytes()) }
fn hint_name(v: &mut [u8], at: u32, hint: u16, name: &str) {
    p16(v, at, hint);
    put(v, at + 2, name.as_bytes());
}
fn set_dir(v: &mut [u8], dir: usize, rva: u32, size: u32) {
    let o = (OPT + 112 + dir * 8) as u32;
    p32(v, o, rva);
    p32(v, o + 4, size);
}

const ORD: u64 = 1 << 63;

/// The image as mapped (RVA == offset, headers at 0); `files` derives the file from it.
/// Contents are written only inside each section's raw part, so the zero tails stay zero.
fn virtual_image() -> Vec<u8> {
    let mut v = vec![0u8; IMAGE_SIZE as usize];
    // DOS + NT headers
    put(&mut v, 0, b"MZ");
    p32(&mut v, 0x3C, NT as u32);
    put(&mut v, NT as u32, b"PE\0\0");
    let coff = NT as u32 + 4;
    p16(&mut v, coff, IMAGE_FILE_MACHINE_AMD64);
    p16(&mut v, coff + 2, SECTIONS.len() as u16);
    p32(&mut v, coff + 4, 0x5F5E_1234);
    p16(&mut v, coff + 16, OPT_SIZE as u16);
    p16(&mut v, coff + 18, IMAGE_FILE_DLL | 0x0022);
    let opt = OPT as u32;
    p16(&mut v, opt, 0x20B);
    p32(&mut v, opt + 16, 0x1000); // AddressOfEntryPoint
    p64(&mut v, opt + 24, BASE);
    p32(&mut v, opt + 32, 0x1000);
    p32(&mut v, opt + 36, 0x200);
    p32(&mut v, opt + 56, IMAGE_SIZE);
    p32(&mut v, opt + 60, HDRS);
    p16(&mut v, opt + 68, 3);
    p16(&mut v, opt + 70, 0x8160);
    p64(&mut v, opt + 72, 0x10_0000);
    p64(&mut v, opt + 80, 0x1000);
    p32(&mut v, opt + 108, 16);
    for (i, &(name, va, vsize, raw, ch)) in SECTIONS.iter().enumerate() {
        let o = (SEC_TABLE + i * 40) as u32;
        put(&mut v, o, name);
        p32(&mut v, o + 8, vsize);
        p32(&mut v, o + 12, va);
        p32(&mut v, o + 16, raw);
        p32(&mut v, o + 20, raw_ptr(i));
        p32(&mut v, o + 36, ch);
    }

    // imports: three descriptors + the null one
    set_dir(&mut v, DIR_IMPORT, 0x2000, 0x50);
    set_dir(&mut v, DIR_IAT, 0x2200, 0x50);
    for (i, (int, name, iat)) in [(0x2100, 0x2300, 0x2200), (0x2120, 0x2310, 0x2220), (0, 0x2320, 0x2240)].into_iter().enumerate() {
        let d = 0x2000 + i as u32 * 20;
        p32(&mut v, d, int);
        p32(&mut v, d + 12, name);
        p32(&mut v, d + 16, iat);
    }
    put(&mut v, 0x2300, b"KERNEL32.dll");
    put(&mut v, 0x2310, b"xapi.dll");
    put(&mut v, 0x2320, b"d3d12.dll");
    // INT and its unbound IAT copy: by name, by name / by ordinal, name, ordinal
    for (i, t) in [0x2400, 0x2410, 0].into_iter().enumerate() {
        p64(&mut v, 0x2100 + i as u32 * 8, t);
        p64(&mut v, 0x2200 + i as u32 * 8, t);
    }
    for (i, t) in [ORD | 5, 0x2420, ORD | 0x1234, 0].into_iter().enumerate() {
        p64(&mut v, 0x2120 + i as u32 * 8, t);
        p64(&mut v, 0x2220 + i as u32 * 8, t);
    }
    p64(&mut v, 0x2240, 0x2430); // d3d12: no INT, names from the IAT
    hint_name(&mut v, 0x2400, 0x0123, "CreateFileW");
    hint_name(&mut v, 0x2410, 7, "CloseHandle");
    hint_name(&mut v, 0x2420, 0, "XMemAlloc");
    hint_name(&mut v, 0x2430, 0x55, "D3D12CreateDevice");

    // delay imports: one RVA-based descriptor (attrs bit 0) + the null one
    set_dir(&mut v, DIR_DELAY_IMPORT, 0x2500, 0x40);
    p32(&mut v, 0x2500, 1);
    p32(&mut v, 0x2504, 0x2560); // name
    p32(&mut v, 0x2508, 0x3100); // HMODULE slot
    p32(&mut v, 0x250C, 0x3120); // IAT
    p32(&mut v, 0x2510, 0x2540); // INT
    p64(&mut v, 0x2540, 0x2590);
    p64(&mut v, 0x2548, ORD | 42);
    put(&mut v, 0x2560, b"xgameruntime.dll");
    hint_name(&mut v, 0x2590, 3, "XGameGetXboxTitleId");
    // the delay IAT points at the load thunks until first call
    p64(&mut v, 0x3120, BASE + 0x1040);
    p64(&mut v, 0x3128, BASE + 0x1048);

    // exports: base 10, five functions (one unused slot, two forwarders), three unsorted names
    set_dir(&mut v, DIR_EXPORT, 0x2600, 0x130);
    p32(&mut v, 0x2600 + 12, 0x2700);
    p32(&mut v, 0x2600 + 16, 10);
    p32(&mut v, 0x2600 + 20, 5);
    p32(&mut v, 0x2600 + 24, 3);
    p32(&mut v, 0x2600 + 28, 0x2640);
    p32(&mut v, 0x2600 + 32, 0x2660);
    p32(&mut v, 0x2600 + 36, 0x2670);
    for (i, f) in [0x1010, 0, 0x26A0, 0x1020, 0x26C0].into_iter().enumerate() {
        p32(&mut v, 0x2640 + i as u32 * 4, f);
    }
    for (i, (name, idx)) in [(0x2710, 0u16), (0x2718, 3), (0x2720, 2)].into_iter().enumerate() {
        p32(&mut v, 0x2660 + i as u32 * 4, name);
        p16(&mut v, 0x2670 + i as u32 * 2, idx);
    }
    put(&mut v, 0x26A0, b"KERNEL32.CreateFileW");
    put(&mut v, 0x26C0, b"NTDLL.#12");
    put(&mut v, 0x2700, b"synthetic.dll");
    put(&mut v, 0x2710, b"Zeta");
    put(&mut v, 0x2718, b"Alpha");
    put(&mut v, 0x2720, b"Fwd");

    // TLS: a 0x40-byte template at .data + 0x20 bytes of zero fill, one callback
    set_dir(&mut v, DIR_TLS, 0x2800, 40);
    p64(&mut v, 0x2800, BASE + 0x3000);
    p64(&mut v, 0x2808, BASE + 0x3040);
    p64(&mut v, 0x2810, BASE + 0x3080);
    p64(&mut v, 0x2818, BASE + 0x2840);
    p32(&mut v, 0x2820, 0x20);
    p32(&mut v, 0x2824, 0x0050_0000); // IMAGE_SCN_ALIGN_16BYTES
    p64(&mut v, 0x2840, BASE + 0x1030);
    put(&mut v, 0x3000, &[0x11; 0x40]);

    // relocations: three blocks, ABSOLUTE padding in two of them
    set_dir(&mut v, DIR_BASERELOC, 0x7000, 0x28);
    let dir64 = |o: u16| (IMAGE_REL_BASED_DIR64 << 12) | o;
    let blocks: [(u32, &[u16]); 3] = [
        (0x1000, &[dir64(0x010), dir64(0x018), 0x0123, dir64(0xFF8)]),
        (0x3000, &[dir64(0x000), dir64(0x008)]),
        (0x2000, &[(IMAGE_REL_BASED_HIGHLOW << 12) | 0x004, 0]),
    ];
    let mut at = 0x7000;
    for (page, entries) in blocks {
        p32(&mut v, at, page);
        p32(&mut v, at + 4, 8 + 2 * entries.len() as u32);
        for (i, &e) in entries.iter().enumerate() {
            p16(&mut v, at + 8 + 2 * i as u32, e);
        }
        at += 8 + 2 * entries.len() as u32;
    }

    // an unterminated string ending at .text's raw end, and 4096 bytes before a NUL in .strings
    put(&mut v, 0x11F8, b"NOTERMIN");
    put(&mut v, 0x5000, &[b'A'; 4096]);
    v
}

/// The same image in both layouts: the file (headers, then each section's raw bytes at its
/// PointerToRawData) and the mapped image.
fn files(v: &[u8]) -> [(Layout, Vec<u8>); 2] {
    let len = raw_ptr(SECTIONS.len()) as usize;
    let mut f = vec![0u8; len];
    f[..HDRS as usize].copy_from_slice(&v[..HDRS as usize]);
    for (i, &(_, va, _, raw, _)) in SECTIONS.iter().enumerate() {
        let p = raw_ptr(i) as usize;
        f[p..p + raw as usize].copy_from_slice(&v[va as usize..(va + raw) as usize]);
    }
    [(Layout::File, f), (Layout::Mapped, v.to_vec())]
}

fn image() -> [(Layout, Vec<u8>); 2] {
    files(&virtual_image())
}

/// The headers' fields and the section table, as written.
#[test]
fn header_fields_and_sections() {
    for (layout, data) in image() {
        let pe = Pe::parse(&data, layout).unwrap();
        assert_eq!(pe.machine, IMAGE_FILE_MACHINE_AMD64);
        assert_eq!(pe.characteristics, IMAGE_FILE_DLL | 0x0022);
        assert!(pe.is_dll() && pe.has_relocs());
        assert_eq!((pe.image_base, pe.entry_rva, pe.timestamp), (BASE, 0x1000, 0x5F5E_1234));
        assert_eq!((pe.section_alignment, pe.file_alignment), (0x1000, 0x200));
        assert_eq!((pe.size_of_image, pe.size_of_headers), (IMAGE_SIZE, HDRS));
        assert_eq!((pe.subsystem, pe.dll_characteristics), (3, 0x8160));
        assert_eq!((pe.stack_reserve, pe.stack_commit), (0x10_0000, 0x1000));
        assert_eq!(pe.dir(DIR_EXPORT), DataDir { rva: 0x2600, size: 0x130 });
        assert_eq!(pe.dir(DIR_IAT), DataDir { rva: 0x2200, size: 0x50 });
        assert_eq!(pe.dir(DIR_RESOURCE), DataDir::default());
        let names: Vec<&str> = pe.sections.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, [".text", ".rdata", ".data", ".strings", ".reloc"]);
        for (i, (s, &(_, va, vsize, raw, ch))) in pe.sections.iter().zip(&SECTIONS).enumerate() {
            assert_eq!((s.va, s.vsize, s.raw_size, s.raw_ptr, s.characteristics), (va, vsize, raw, raw_ptr(i), ch));
            assert_eq!(s.mapped_size(), if vsize == 0 { raw } else { vsize });
        }
        assert_eq!(pe.data().len(), data.len());
    }
}

/// Header variants: stripped relocations, fewer data directories than 16 (the rest read as
/// empty), a SizeOfOptionalHeader that holds fewer, more than 16 (clamped), x86 or PE32 headers.
#[test]
fn header_variants() {
    let patch = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut v = virtual_image();
        f(&mut v);
        files(&v)
    };
    for (layout, data) in patch(&|v| p16(v, NT as u32 + 22, IMAGE_FILE_DLL | IMAGE_FILE_RELOCS_STRIPPED)) {
        let pe = Pe::parse(&data, layout).unwrap();
        assert!(!pe.has_relocs());
        assert_eq!(pe.relocations().unwrap().len(), 6, "the directory itself is still read");
    }
    for (layout, data) in patch(&|v| p32(v, OPT as u32 + 108, 6)) {
        let pe = Pe::parse(&data, layout).unwrap();
        assert_eq!(pe.dir(DIR_BASERELOC).rva, 0x7000);
        assert_eq!(pe.dir(DIR_TLS), DataDir::default());
        assert_eq!(pe.tls().unwrap(), None);
        assert!(pe.delay_imports().unwrap().is_empty());
    }
    for (layout, data) in patch(&|v| p32(v, OPT as u32 + 108, 0xFFFF_FFFF)) {
        assert_eq!(Pe::parse(&data, layout).unwrap().dir(DIR_DELAY_IMPORT).rva, 0x2500);
    }
    // SizeOfOptionalHeader = 112 + 2 * 8: only exports and imports; the section table moves
    // up, so this image's sections no longer parse as written, but the headers do
    for (layout, data) in patch(&|v| p16(v, NT as u32 + 20, 112 + 16)) {
        let pe = Pe::parse(&data, layout).unwrap();
        assert_eq!(pe.dir(DIR_IMPORT).rva, 0x2000);
        assert_eq!(pe.dir(DIR_RESOURCE), DataDir::default());
    }
    for (layout, data) in patch(&|v| p16(v, NT as u32 + 4, 0x014C)) {
        assert_eq!(Pe::parse(&data, layout).err(), Some(PeError::NotX64(0x014C)));
    }
    for (layout, data) in patch(&|v| p16(v, OPT as u32, 0x10B)) {
        assert_eq!(Pe::parse(&data, layout).err(), Some(PeError::BadMagic("PE32+ (0x20B)")));
    }
    for (layout, data) in patch(&|v| { p16(v, OPT as u32, 0x10B); p16(v, NT as u32 + 4, 0x01C4) }) {
        assert_eq!(Pe::parse(&data, layout).err(), Some(PeError::NotX64(0x01C4)));
    }
    for (layout, data) in patch(&|v| p32(v, 0x3C, 0x7FFF_0000)) {
        assert_eq!(Pe::parse(&data, layout).err(), Some(PeError::Truncated("NT")));
    }
}

/// RVA → offset against a model from the section table: in the file, headers map to themselves
/// and a section's first SizeOfRawData bytes to PointerToRawData; its zero-filled tail, the gaps
/// between sections and everything past the image are not in the file. Mapped: RVA == offset
/// inside the image. Every RVA reads the same byte through both layouts wherever both have it.
#[test]
fn rva_to_offset_matches_the_section_table() {
    let v = virtual_image();
    let [(_, file), (_, mapped)] = image();
    let fpe = Pe::parse(&file, Layout::File).unwrap();
    let mpe = Pe::parse(&mapped, Layout::Mapped).unwrap();
    let model = |rva: u32| -> Option<usize> {
        if rva < HDRS {
            return Some(rva as usize);
        }
        SECTIONS.iter().enumerate().find(|(_, s)| rva >= s.1 && rva < s.1 + s.3).map(|(i, s)| (raw_ptr(i) + rva - s.1) as usize)
    };
    for rva in 0..IMAGE_SIZE + 0x40 {
        assert_eq!(fpe.rva_to_offset(rva), model(rva), "file RVA {rva:#x}");
        let m = (rva < IMAGE_SIZE).then_some(rva as usize);
        assert_eq!(mpe.rva_to_offset(rva), m, "mapped RVA {rva:#x}");
        if let Some(off) = model(rva) {
            assert_eq!(file[off], v[rva as usize], "RVA {rva:#x}");
            assert_eq!(fpe.bytes_at(rva, 1), Some(&v[rva as usize..rva as usize + 1]));
        }
        if rva < IMAGE_SIZE {
            assert_eq!(mpe.bytes_at(rva, 1), Some(&v[rva as usize..rva as usize + 1]));
        }
    }
    // the edges by name: .text's raw end, its tail, the gap after it, .reloc (vsize 0)
    assert_eq!(fpe.rva_to_offset(0x1000), Some(0x400));
    assert_eq!(fpe.rva_to_offset(0x11FF), Some(0x5FF));
    for rva in [0x1200, 0x12FF, 0x1300, 0x1FFF, 0x3200, 0x4FFF, 0x7200, IMAGE_SIZE, u32::MAX] {
        assert_eq!(fpe.rva_to_offset(rva), None, "{rva:#x}");
    }
    assert_eq!(fpe.rva_to_offset(0x71FF), Some(0x2BFF));
    assert_eq!(mpe.rva_to_offset(0x1300), Some(0x1300));
    assert_eq!(mpe.rva_to_offset(u32::MAX), None);
}

/// Reads that straddle the end of the bytes a layout has: the file stops at a section's raw end
/// and at SizeOfHeaders; the mapped image goes on into the zero tail.
#[test]
fn reads_straddling_a_raw_end() {
    let [(_, file), (_, mapped)] = image();
    let f = Pe::parse(&file, Layout::File).unwrap();
    let m = Pe::parse(&mapped, Layout::Mapped).unwrap();
    // "NOTERMIN" at 0x11F8..0x1200
    for pe in [&f, &m] {
        assert_eq!(pe.u32_at(0x11FC), Some(u32::from_le_bytes(*b"RMIN")));
        assert_eq!(pe.u64_at(0x11F8), Some(u64::from_le_bytes(*b"NOTERMIN")));
        assert_eq!(pe.bytes_at(0x11F8, 8), Some(&b"NOTERMIN"[..]));
        assert_eq!(pe.bytes_at(0x11F8, 0), Some(&[][..]));
        assert_eq!(pe.u16_at(0x3C), Some(NT as u16));
        assert_eq!(pe.u32_at(IMAGE_SIZE - 2), None);
    }
    assert_eq!(f.u32_at(0x11FE), None);
    assert_eq!(f.u16_at(0x11FF), None);
    assert_eq!(f.bytes_at(0x11F8, 9), None);
    assert_eq!(f.u32_at(HDRS - 2), None, "file headers end at SizeOfHeaders");
    assert_eq!(f.u32_at(0x1200), None);
    assert_eq!(m.u32_at(0x11FE), Some(u32::from_le_bytes([b'I', b'N', 0, 0])));
    assert_eq!(m.bytes_at(0x11F8, 9), Some(&b"NOTERMIN\0"[..]));
    assert_eq!(m.u32_at(HDRS - 2), Some(0));
    assert_eq!(m.u32_at(0x1200), Some(0));
    assert_eq!(m.u32_at(IMAGE_SIZE - 4), Some(0));
}

/// `cstr_at`: a string running into a section's raw end has no terminator in the file, but the
/// mapped image's zero tail ends it; a string needs its NUL within 4096 bytes.
#[test]
fn cstr_terminators_and_the_4096_limit() {
    let [(_, file), (_, mapped)] = image();
    let f = Pe::parse(&file, Layout::File).unwrap();
    let m = Pe::parse(&mapped, Layout::Mapped).unwrap();
    assert_eq!(f.cstr_at(0x11F8), None);
    assert_eq!(m.cstr_at(0x11F8).as_deref(), Some("NOTERMIN"));
    for pe in [&f, &m] {
        assert_eq!(pe.cstr_at(0x2300).as_deref(), Some("KERNEL32.dll"));
        assert_eq!(pe.cstr_at(0x2400 + 2).as_deref(), Some("CreateFileW"));
        assert_eq!(pe.cstr_at(0x230C).as_deref(), Some(""));
        // 4096 'A's then a NUL: the NUL is the 4097th byte from 0x5000, the 4096th from 0x5001
        assert_eq!(pe.cstr_at(0x5000), None);
        assert_eq!(pe.cstr_at(0x5001).map(|s| s.len()), Some(4095));
        assert_eq!(pe.cstr_at(IMAGE_SIZE), None);
    }
    assert_eq!(f.cstr_at(0x1300), None);
    assert_eq!(m.cstr_at(0x1300).as_deref(), Some(""));
}

type ImportView = Vec<(String, bool, u32, Vec<(u32, ImportName)>)>;

fn view(dlls: Vec<ImportDll>) -> ImportView {
    dlls.into_iter()
        .map(|d| (d.dll, d.delay, d.module_handle_rva, d.entries.into_iter().map(|e| (e.iat_rva, e.what)).collect()))
        .collect()
}

fn name(hint: u16, name: &str) -> ImportName {
    ImportName::Name { hint, name: name.into() }
}

fn expected_imports() -> ImportView {
    vec![
        ("KERNEL32.dll".into(), false, 0, vec![(0x2200, name(0x0123, "CreateFileW")), (0x2208, name(7, "CloseHandle"))]),
        ("xapi.dll".into(), false, 0, vec![(0x2220, ImportName::Ordinal(5)), (0x2228, name(0, "XMemAlloc")), (0x2230, ImportName::Ordinal(0x1234))]),
        ("d3d12.dll".into(), false, 0, vec![(0x2240, name(0x55, "D3D12CreateDevice"))]),
    ]
}

fn expected_delay_imports() -> ImportView {
    vec![("xgameruntime.dll".into(), true, 0x3100, vec![(0x3120, name(3, "XGameGetXboxTitleId")), (0x3128, ImportName::Ordinal(42))])]
}

/// Imports by name (with hint) and by ordinal (bit 63), several DLLs, the IAT slot of each entry,
/// a descriptor without an INT read from its IAT; delay imports with RVA-based attributes.
#[test]
fn imports_and_delay_imports() {
    for (layout, data) in image() {
        let pe = Pe::parse(&data, layout).unwrap();
        assert_eq!(view(pe.imports().unwrap()), expected_imports(), "{layout:?}");
        assert_eq!(view(pe.delay_imports().unwrap()), expected_delay_imports(), "{layout:?}");
        let shown: Vec<String> = pe.imports().unwrap()[1].entries.iter().map(|e| e.what.to_string()).collect();
        assert_eq!(shown, ["#5", "XMemAlloc", "#4660"]);
    }
    // a by-name thunk whose hint/name RVA is outside the image
    let mut v = virtual_image();
    p64(&mut v, 0x2108, 0x7FFF_0000);
    for (layout, data) in files(&v) {
        let pe = Pe::parse(&data, layout).unwrap();
        assert_eq!(pe.imports().err(), Some(PeError::BadRva(0x7FFF_0000)), "{layout:?}");
    }
}

/// Relocations: several blocks, ABSOLUTE padding skipped (whatever its offset), DIR64 and
/// HIGHLOW kinds; a block larger than what is left of the directory is an error; a block size
/// below 8 ends the table; a directory size cut at a block boundary reads only the blocks in it.
#[test]
fn relocations() {
    let dir64 = |rva| Reloc { rva, kind: IMAGE_REL_BASED_DIR64 };
    let all = vec![dir64(0x1010), dir64(0x1018), dir64(0x1FF8), dir64(0x3000), dir64(0x3008), Reloc { rva: 0x2004, kind: IMAGE_REL_BASED_HIGHLOW }];
    for (layout, data) in image() {
        assert_eq!(Pe::parse(&data, layout).unwrap().relocations().unwrap(), all, "{layout:?}");
    }
    let cases: [(&dyn Fn(&mut Vec<u8>), Result<Vec<Reloc>, PeError>); 4] = [
        (&|v| p32(v, 0x7000 + 28 + 4, 16), Err(PeError::BadRva(0x7000 + 28))),
        (&|v| p32(v, 0x7000 + 16 + 4, 4), Ok(all[..3].to_vec())),
        (&|v| set_dir(v, DIR_BASERELOC, 0x7000, 28), Ok(all[..5].to_vec())),
        (&|v| set_dir(v, DIR_BASERELOC, 0x7000, 0x28 + 6), Ok(all.clone())),
    ];
    for (i, (f, want)) in cases.iter().enumerate() {
        let mut v = virtual_image();
        f(&mut v);
        for (layout, data) in files(&v) {
            assert_eq!(&Pe::parse(&data, layout).unwrap().relocations(), want, "case {i} {layout:?}");
        }
    }
}

/// IMAGE_TLS_DIRECTORY64: absolute VAs as stored, template and total length, VA → RVA.
#[test]
fn tls_directory() {
    for (layout, data) in image() {
        let pe = Pe::parse(&data, layout).unwrap();
        let tls = pe.tls().unwrap().unwrap();
        assert_eq!(tls, TlsDir {
            start_raw: BASE + 0x3000,
            end_raw: BASE + 0x3040,
            address_of_index: BASE + 0x3080,
            address_of_callbacks: BASE + 0x2840,
            size_of_zero_fill: 0x20,
            characteristics: 0x0050_0000,
        });
        assert_eq!((tls.template_len(), tls.total_len()), (0x40, 0x60));
        assert_eq!(pe.va_to_rva(tls.start_raw), Some(0x3000));
        let cb = pe.va_to_rva(tls.address_of_callbacks).unwrap();
        assert_eq!(pe.u64_at(cb), Some(BASE + 0x1030));
        assert_eq!(pe.u64_at(cb + 8), Some(0));
        assert_eq!(pe.bytes_at(0x3000, 0x40), Some(&[0x11u8; 0x40][..]));
        assert_eq!(pe.va_to_rva(BASE - 1), None);
        assert_eq!(pe.va_to_rva(BASE + u32::MAX as u64), Some(u32::MAX));
        assert_eq!(pe.va_to_rva(BASE + (1 << 32)), None);
    }
    let odd = TlsDir { start_raw: 0x2000, end_raw: 0x1000, size_of_zero_fill: 8, ..TlsDir::default() };
    assert_eq!((odd.template_len(), odd.total_len()), (0, 8));
}

/// Exports with ordinal base 10, an unused slot, unsorted names and two forwarders (by name and
/// by ordinal) inside the export directory: the raw table, `export_target` and `OwnedExports`.
#[test]
fn exports_and_forwarders() {
    for (layout, data) in image() {
        let pe = Pe::parse(&data, layout).unwrap();
        let ex = pe.exports().unwrap().unwrap();
        assert_eq!(ex.dll_name, "synthetic.dll");
        assert_eq!(ex.ordinal_base, 10);
        assert_eq!(ex.functions, [0x1010, 0, 0x26A0, 0x1020, 0x26C0]);
        assert_eq!(ex.names, [("Zeta".to_string(), 0), ("Alpha".to_string(), 3), ("Fwd".to_string(), 2)]);
        assert_eq!(ex.dir, DataDir { rva: 0x2600, size: 0x130 });
        assert_eq!(ex.by_name_raw("Alpha"), Some(0x1020));
        assert_eq!(ex.by_name_raw("Zeta"), Some(0x1010));
        assert_eq!(ex.by_name_raw("alpha"), None);
        assert_eq!(ex.by_ordinal_raw(9), None);
        assert_eq!(ex.by_ordinal_raw(10), Some(0x1010));
        assert_eq!(ex.by_ordinal_raw(11), Some(0));
        assert_eq!(ex.by_ordinal_raw(15), None);
        assert!(ex.is_forwarder_rva(0x2600) && ex.is_forwarder_rva(0x272F));
        assert!(!ex.is_forwarder_rva(0x25FF) && !ex.is_forwarder_rva(0x2730));
        assert_eq!(pe.export_target(&ex, 0x1010), Some(ExportTarget::Rva(0x1010)));
        assert_eq!(pe.export_target(&ex, 0), None);
        assert_eq!(pe.export_target(&ex, 0x26A0), Some(ExportTarget::Forwarder("KERNEL32.CreateFileW".into())));

        let owned = OwnedExports::from_pe(&pe).unwrap();
        let fwd = |s: &str| ExportTarget::Forwarder(s.into());
        let by_ordinal: HashMap<u32, ExportTarget> = [
            (10, ExportTarget::Rva(0x1010)),
            (12, fwd("KERNEL32.CreateFileW")),
            (13, ExportTarget::Rva(0x1020)),
            (14, fwd("NTDLL.#12")),
        ].into_iter().collect();
        assert_eq!(owned.by_ordinal, by_ordinal, "{layout:?}");
        let by_name: HashMap<String, ExportTarget> = [
            ("Zeta".to_string(), ExportTarget::Rva(0x1010)),
            ("Alpha".to_string(), ExportTarget::Rva(0x1020)),
            ("Fwd".to_string(), fwd("KERNEL32.CreateFileW")),
        ].into_iter().collect();
        assert_eq!(owned.by_name, by_name, "{layout:?}");
    }
    // a name index past the function table names nothing
    let mut v = virtual_image();
    p16(&mut v, 0x2670, 9);
    for (layout, data) in files(&v) {
        let owned = OwnedExports::from_pe(&Pe::parse(&data, layout).unwrap()).unwrap();
        assert!(!owned.by_name.contains_key("Zeta") && owned.by_name.len() == 2, "{layout:?}");
    }
    // an export directory under 40 bytes is malformed
    let mut v = virtual_image();
    set_dir(&mut v, DIR_EXPORT, 0x2600, 39);
    for (layout, data) in files(&v) {
        assert_eq!(Pe::parse(&data, layout).unwrap().exports().err(), Some(PeError::BadRva(0x2600)));
    }
}

/// Every prefix of the image, in both layouts: the headers are an error until the section table
/// is complete, never a panic; past that, each directory reads either an error or exactly what
/// the whole image reads (exports may lose names whose strings were cut, never gain any).
#[test]
fn every_truncation_is_an_error_or_a_prefix_of_the_truth() {
    for (layout, full) in image() {
        let pe = Pe::parse(&full, layout).unwrap();
        let imports = view(pe.imports().unwrap());
        let delay = view(pe.delay_imports().unwrap());
        let relocs = pe.relocations().unwrap();
        let tls = pe.tls().unwrap();
        let ex = pe.exports().unwrap().unwrap();
        let owned = OwnedExports::from_pe(&pe).unwrap();
        for len in 0..full.len() {
            let cut = &full[..len];
            let r = catch_unwind(AssertUnwindSafe(|| {
                let pe = match Pe::parse(cut, layout) {
                    Err(e) => {
                        assert!(len < HEADER_END, "{e:?}");
                        return;
                    }
                    Ok(pe) => pe,
                };
                assert!(len >= HEADER_END);
                assert_eq!(pe.sections.len(), SECTIONS.len());
                if let Ok(i) = pe.imports() { assert_eq!(view(i), imports) }
                if let Ok(d) = pe.delay_imports() { assert_eq!(view(d), delay) }
                if let Ok(r) = pe.relocations() { assert_eq!(r, relocs) }
                if let Ok(t) = pe.tls() { assert_eq!(t, tls) }
                if let Ok(e) = pe.exports() {
                    let e = e.expect("the export directory is still there");
                    assert_eq!((e.ordinal_base, &e.functions), (ex.ordinal_base, &ex.functions));
                    assert!(e.dll_name == ex.dll_name || e.dll_name.is_empty());
                    let mut rest = ex.names.iter();
                    assert!(e.names.iter().all(|n| rest.any(|m| m == n)), "{:?}", e.names);
                }
                if let Ok(o) = OwnedExports::from_pe(&pe) {
                    assert!(o.by_ordinal.keys().all(|k| owned.by_ordinal.contains_key(k)));
                    assert!(o.by_name.keys().all(|k| owned.by_name.contains_key(k)));
                }
                for rva in [0x3C, 0x11F8, 0x2300, 0x26A0, 0x7000] {
                    let _ = (pe.cstr_at(rva), pe.u64_at(rva), pe.rva_to_offset(rva));
                }
            }));
            assert!(r.is_ok(), "{layout:?}: truncated to {len:#x} bytes");
        }
    }
}
