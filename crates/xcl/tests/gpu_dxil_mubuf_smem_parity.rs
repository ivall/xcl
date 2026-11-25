//! GPU differential parity tests between the direct RDNA2 → DXIL translator
//! (`xcl::gpu::dxil::kernel::translate`) and the HLSL + DXC path (`xcl::gpu::recomp`) for
//! `lower_mem.rs`: MUBUF raw/sub-dword loads & stores, MUBUF/TBUFFER typed format & D16 ops,
//! 32-bit MUBUF atomics, and SMEM (`S_LOAD` / `S_BUFFER_LOAD`) alignment and bounds.

use std::ops::Range;

use xcl::gpu::dxil::kernel::{translate as direct_translate, KernelOptions};
use xcl::gpu::dxil::validate;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner::{self, KernelResult, KernelState};
use xcl::gpu::recomp::{dxc, translate as hlsl_translate, Options};

fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    let mut v = Vec::new();
    for (n, e, f, l) in list {
        v.extend(encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    v
}

/// Wave configurations supported on this adapter: `(wave64, split)`.
/// Skips cleanly when no D3D12 SM 6.6 GPU or DXC is available.
fn gpu_modes() -> Vec<(bool, bool)> {
    match (runner::adapter(), runner::wave64_programs_supported(), dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => {
            if runner::wave64_supported() == Ok(true) {
                vec![(false, false), (true, false)]
            } else {
                vec![(false, false), (true, true)]
            }
        }
        (a, w, d) => {
            runner::skip_test(if a.is_err() || w.is_err() {
                "no GPU"
            } else if d.is_err() {
                "no DXC"
            } else {
                "no wave64"
            });
            Vec::new()
        }
    }
}

/// Translates `code` through both HLSL+DXC and Direct DXIL (validating the direct container),
/// runs both on the GPU, asserts exact parity on the watched registers and memory range, and
/// returns the direct result for ISA assertions.
fn run_parity(
    label: &str,
    code: &[u8],
    st: &KernelState,
    mem: &[u32],
    wave64: bool,
    split: bool,
    sgprs: Range<usize>,
    vgprs: Range<usize>,
    mem_dwords: Range<usize>,
) -> KernelResult {
    let hlsl_opt = Options { wave64, split_wave64: split, ..Default::default() };
    let t = hlsl_translate(code, &hlsl_opt).unwrap_or_else(|e| panic!("{label}: hlsl translate: {e}"));
    assert!(t.complete(), "{label}: hlsl incomplete: {:?}", t.unsupported_why);
    let hlsl_dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[])
        .unwrap_or_else(|e| panic!("{label}: dxc compile: {e}"));

    let direct_dxil = direct_translate(
        code,
        KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 },
    )
    .unwrap_or_else(|e| panic!("{label} (w64={wave64} split={split}): direct translate: {e}"));
    validate::validate(&direct_dxil)
        .unwrap_or_else(|e| panic!("{label} (w64={wave64} split={split}): dxil validate: {e}"));

    let r_hlsl = runner::run(&hlsl_dxil, st, mem).unwrap();
    let r_direct = runner::run(&direct_dxil, st, mem).unwrap();
    let lanes = if wave64 { 64 } else { 32 };

    for s in sgprs {
        assert_eq!(
            r_direct.sgpr[s], r_hlsl.sgpr[s],
            "{label} (w64={wave64} split={split}) s{s}: direct={:#x} hlsl={:#x}",
            r_direct.sgpr[s], r_hlsl.sgpr[s]
        );
    }
    for l in 0..lanes {
        for v in vgprs.clone() {
            assert_eq!(
                r_direct.vgpr[l][v], r_hlsl.vgpr[l][v],
                "{label} (w64={wave64} split={split}) lane {l} v{v}: direct={:#x} hlsl={:#x}",
                r_direct.vgpr[l][v], r_hlsl.vgpr[l][v]
            );
        }
    }
    for idx in mem_dwords {
        assert_eq!(
            r_direct.mem[idx], r_hlsl.mem[idx],
            "{label} (w64={wave64} split={split}) mem[{idx}]: direct={:#x} hlsl={:#x}",
            r_direct.mem[idx], r_hlsl.mem[idx]
        );
    }
    r_direct
}

#[test]
fn direct_dxil_mubuf_raw_subdword_bounds_and_exec_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    // V# s[12:15]: structured stride 12, 20 records at 0x100 (IDXEN)
    // V# s[16:19]: raw buffer, 0x80 bytes at 0x200 (OFFEN dword/x2/x4 + OOB)
    // V# s[20:23]: raw buffer, 0x80 bytes at 0x300 (sub-dword byte/short stores & loads)
    // s24 = 4 (SOFFSET for raw load)
    let code = prog(&[
        // Narrow EXEC to lanes 0..23 so upper lanes verify EXEC preservation
        ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 128 + 24), ("VSRC1", 0)], None),
        // Structured dwordx3 load (lanes 0..19 in range, 20..23 OOB -> 0, 24+ inactive -> sentinel)
        ("BUFFER_LOAD_DWORDX3", "ENC_MUBUF", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
        // Raw dwordx2 load with OFFEN + SOFFSET (s24 = 4) + OFFSET = 4
        ("BUFFER_LOAD_DWORDX2", "ENC_MUBUF", &[("VDATA", 13), ("VADDR", 1), ("SRSRC", 4), ("SOFFSET", 24), ("OFFEN", 1), ("OFFSET", 4)], None),
        // Raw dwordx4 load at 0x200 + 8 * lane (lanes 0..14 in range, 15+ OOB)
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", &[("VDATA", 15), ("VADDR", 1), ("SRSRC", 4), ("SOFFSET", 128), ("OFFEN", 1)], None),
        // Sub-dword stores (byte at 0x300 + lane, short at 0x340 + 2*lane) and loads back
        ("BUFFER_STORE_BYTE", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 5), ("SOFFSET", 128), ("OFFEN", 1)], None),
        ("BUFFER_STORE_SHORT", "ENC_MUBUF", &[("VDATA", 3), ("VADDR", 4), ("SRSRC", 5), ("SOFFSET", 128), ("OFFEN", 1), ("OFFSET", 0x40)], None),
        ("BUFFER_LOAD_UBYTE", "ENC_MUBUF", &[("VDATA", 19), ("VADDR", 0), ("SRSRC", 5), ("SOFFSET", 128), ("OFFEN", 1)], None),
        ("BUFFER_LOAD_SBYTE", "ENC_MUBUF", &[("VDATA", 20), ("VADDR", 0), ("SRSRC", 5), ("SOFFSET", 128), ("OFFEN", 1)], None),
        ("BUFFER_LOAD_USHORT", "ENC_MUBUF", &[("VDATA", 21), ("VADDR", 4), ("SRSRC", 5), ("SOFFSET", 128), ("OFFEN", 1), ("OFFSET", 0x40)], None),
        ("BUFFER_LOAD_SSHORT", "ENC_MUBUF", &[("VDATA", 22), ("VADDR", 4), ("SRSRC", 5), ("SOFFSET", 128), ("OFFEN", 1), ("OFFSET", 0x40)], None),
        // OOB store into s[20:23] at offset 0x90 (past 0x80) must leave mem untouched
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 3), ("VADDR", 0), ("SRSRC", 5), ("SOFFSET", 128), ("OFFEN", 1), ("OFFSET", 0x90)], None),
    ]);

    let mut mem = vec![0u32; 512];
    for (i, w) in mem.iter_mut().enumerate() {
        *w = 0x2000_0000 | i as u32;
    }
    let mut st = KernelState::default();
    st.sgpr[12..16].copy_from_slice(&[0x100, 12 << 16, 20, 0]);
    st.sgpr[16..20].copy_from_slice(&[0x200, 0, 0x80, 0]);
    st.sgpr[20..24].copy_from_slice(&[0x300, 0, 0x80, 0]);
    st.sgpr[24] = 4;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[0] = l32;
        r[1] = 8 * l32;
        r[2] = 0x78 + l32; // crosses 0x80 sign boundary at lane 8
        r[3] = 0x7FF8 + l32; // crosses 0x8000 sign boundary at lane 8
        r[4] = 2 * l32;
        for reg in 10..23 {
            r[reg] = 0xDEAD_BEEF;
        }
    }

    for (wave64, split) in modes {
        let r = run_parity(
            "mubuf_raw_subdword",
            &code,
            &st,
            &mem,
            wave64,
            split,
            0..0,
            10..23,
            0x300 / 4..0x400 / 4,
        );
        let lanes = if wave64 { 64 } else { 32 };
        for l in 0..lanes {
            // Check key ISA semantics on top of Direct-vs-HLSL parity
            let want_b = (0x78 + l as u32) & 0xFF;
            let want_s = (0x7FF8 + l as u32) & 0xFFFF;
            if l < 24 {
                assert_eq!(r.vgpr[l][19], want_b, "lane {l} ubyte");
                assert_eq!(r.vgpr[l][20], (want_b as i8) as i32 as u32, "lane {l} sbyte");
                assert_eq!(r.vgpr[l][21], want_s, "lane {l} ushort");
                assert_eq!(r.vgpr[l][22], (want_s as i16) as i32 as u32, "lane {l} sshort");
                let want_x3_0 = if l < 20 { 0x2000_0000 | (0x100 / 4 + 3 * l as u32) } else { 0 };
                assert_eq!(r.vgpr[l][10], want_x3_0, "lane {l} structured dwordx3[0]");
            } else {
                assert_eq!(r.vgpr[l][10], 0xDEAD_BEEF, "inactive lane {l} kept sentinel");
                assert_eq!(r.vgpr[l][19], 0xDEAD_BEEF, "inactive lane {l} ubyte kept sentinel");
            }
        }
        // OOB store at 0x390 must not clobber memory
        assert_eq!(r.mem[0x390 / 4], 0x2000_0000 | (0x390 / 4) as u32, "OOB store untouched");
    }
}

#[test]
fn direct_dxil_mubuf_and_tbuffer_format_d16_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    let ident = 4 | (5 << 3) | (6 << 6) | (7 << 9);
    let bgra = 6 | (5 << 3) | (4 << 6) | (7 << 9);
    let vs = |base: u32, fmt: u32, sel: u32| [base, 0, 0x100, sel | (fmt << 12)];
    let f = |x: f32| x.to_bits();

    let code = prog(&[
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 3), ("SOFFSET", 128)], None),
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 14), ("SRSRC", 4), ("SOFFSET", 128)], None),
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 18), ("SRSRC", 5), ("SOFFSET", 128), ("OFFSET", 2)], None),
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 19), ("SRSRC", 5), ("SOFFSET", 128), ("OFFSET", 0x200)], None),
        ("TBUFFER_LOAD_FORMAT_XY", "ENC_MTBUF", &[("VDATA", 20), ("SRSRC", 6), ("SOFFSET", 128), ("FORMAT", 64)], None),
        ("BUFFER_LOAD_FORMAT_D16_XY", "ENC_MUBUF", &[("VDATA", 22), ("SRSRC", 4), ("SOFFSET", 128)], None),
        ("BUFFER_LOAD_FORMAT_D16_X", "ENC_MUBUF", &[("VDATA", 23), ("SRSRC", 5), ("SOFFSET", 128), ("OFFSET", 2)], None),
        ("TBUFFER_LOAD_FORMAT_D16_XY", "ENC_MTBUF", &[("VDATA", 24), ("SRSRC", 6), ("SOFFSET", 128), ("FORMAT", 64)], None),
        ("BUFFER_STORE_FORMAT_D16_XY", "ENC_MUBUF", &[("VDATA", 25), ("SRSRC", 7), ("SOFFSET", 128)], None),
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 26), ("SRSRC", 8), ("SOFFSET", 128)], None),
        ("BUFFER_STORE_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 26), ("SRSRC", 9), ("SOFFSET", 128)], None),
    ]);

    let mut mem = vec![0u32; 512];
    mem[0x100 / 4] = u32::from_le_bytes([0x00, 0x80, 0xFF, 0x40]); // 8_8_8_8_UNORM
    mem[0x110 / 4] = 0x3C00 | (0xC000 << 16); // 16_16_FLOAT (1.0, -2.0)
    mem[0x118 / 4] = 0x8001 << 16; // 16_SNORM at byte +2
    mem[0x120 / 4] = f(1.5);
    mem[0x124 / 4] = f(-3.25); // 32_32_FLOAT
    mem[0x140 / 4..0x150 / 4].copy_from_slice(&[f(1.25), f(-2.5), f(3.75), f(-4.5)]);

    let mut st = KernelState::default();
    st.sgpr[12..16].copy_from_slice(&vs(0x100, 56, bgra)); // SRSRC 3
    st.sgpr[16..20].copy_from_slice(&vs(0x110, 29, ident)); // SRSRC 4
    st.sgpr[20..24].copy_from_slice(&vs(0x118, 8, ident)); // SRSRC 5
    st.sgpr[24..28].copy_from_slice(&vs(0x120, 0, ident)); // SRSRC 6 (TBUFFER fmt 64)
    st.sgpr[28..32].copy_from_slice(&vs(0x130, 29, ident)); // SRSRC 7 (D16 store)
    st.sgpr[32..36].copy_from_slice(&vs(0x140, 77, 0 | (1 << 3) | (4 << 6) | (7 << 9))); // SRSRC 8
    st.sgpr[36..40].copy_from_slice(&vs(0x150, 77, ident)); // SRSRC 9 (XYZW store)
    for r in st.vgpr.iter_mut() {
        r[25] = 0xB800_4000; // packed f16 (+2.0, -0.5)
    }

    for (wave64, split) in modes {
        let r = run_parity(
            "mubuf_tbuffer_format_d16",
            &code,
            &st,
            &mem,
            wave64,
            split,
            0..0,
            10..30,
            0x130 / 4..0x160 / 4,
        );
        assert_eq!(r.vgpr[0][19], 0, "OOB format load -> 0");
        assert_eq!(r.vgpr[0][22], 0xC000_3C00, "d16 16_16_float");
        assert_eq!(r.vgpr[0][23], 0x0000_BC00, "d16 16_snorm unaligned");
        assert_eq!(r.vgpr[0][24], 0xC280_3E00, "d16 tbuffer 32_32_float");
        assert_eq!(r.mem[0x130 / 4], 0xB800_4000, "d16 store 16_16_float");
        assert_eq!(&r.mem[0x150 / 4..0x160 / 4], &[f(0.0), f(1.0), f(1.25), f(-4.5)]);
    }
}

#[test]
fn direct_dxil_mubuf_atomics_and_bounds_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    // Lane 0 runs all 32-bit MUBUF atomics with GLC=1 on distinct slots in V# s[12:15] (0x200, len 0x40),
    // plus an OOB atomic at offset 0x40 (must return 0 and leave mem untouched); lanes 1+ are inactive
    // and must keep their VDATA sentinels.
    let f = |x: f32| x.to_bits();
    let code = prog(&[
        ("V_CMPX_EQ_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 0)], None), // lane 0 only
        ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 0), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_SUB", "ENC_MUBUF", &[("VDATA", 11), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 4), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_SMIN", "ENC_MUBUF", &[("VDATA", 12), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 8), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_UMIN", "ENC_MUBUF", &[("VDATA", 13), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 12), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_SMAX", "ENC_MUBUF", &[("VDATA", 14), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 16), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_UMAX", "ENC_MUBUF", &[("VDATA", 15), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 20), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_AND", "ENC_MUBUF", &[("VDATA", 16), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 24), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_OR", "ENC_MUBUF", &[("VDATA", 17), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 28), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_XOR", "ENC_MUBUF", &[("VDATA", 18), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 32), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_SWAP", "ENC_MUBUF", &[("VDATA", 19), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 36), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_CMPSWAP", "ENC_MUBUF", &[("VDATA", 20), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 40), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_INC", "ENC_MUBUF", &[("VDATA", 22), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 44), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_DEC", "ENC_MUBUF", &[("VDATA", 23), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 48), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_CSUB", "ENC_MUBUF", &[("VDATA", 24), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 52), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_FMIN", "ENC_MUBUF", &[("VDATA", 25), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 56), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_FMAX", "ENC_MUBUF", &[("VDATA", 26), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 60), ("GLC", 1)], None),
        // OOB atomic at offset 0x40 (num_records == 0x40)
        ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("VDATA", 27), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 64), ("GLC", 1)], None),
    ]);

    let mut mem = vec![0u32; 256];
    let base = 0x200 / 4;
    let init = [
        100, 100, 5, 5, (-5i32) as u32, 5, 0xFF00_AA55, 0x0F00_00F0,
        0xAAAA_5555, 0x1111_2222, 0x0000_0077, 10, 0, 3, f(2.5), f(-1.5), 0xCAFE_BABE,
    ];
    mem[base..base + init.len()].copy_from_slice(&init);

    let mut st = KernelState::default();
    st.sgpr[12..16].copy_from_slice(&[0x200, 0, 0x40, 0]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        if l == 0 {
            r[10] = 25;                  // add -> 125
            r[11] = 30;                  // sub -> 70
            r[12] = (-10i32) as u32;     // smin(5, -10) -> -10
            r[13] = (-10i32) as u32;     // umin(5, 0xFFFFFFF6) -> 5
            r[14] = 3;                   // smax(-5, 3) -> 3
            r[15] = (-10i32) as u32;     // umax(5, 0xFFFFFFF6) -> 0xFFFFFFF6
            r[16] = 0x00FF_FFFF;         // and -> 0x0000_AA55
            r[17] = 0x00F0_0F00;         // or -> 0x0FF0_0FF0
            r[18] = 0xFFFF_0000;         // xor -> 0x5555_5555
            r[19] = 0x9999_8888;         // swap -> 0x9999_8888
            r[20] = 0x0000_0099;         // cmpswap data
            r[21] = 0x0000_0077;         // cmpswap cmp (hit!)
            r[22] = 10;                  // inc(10, limit 10) -> wrap to 0
            r[23] = 9;                   // dec(0, limit 9) -> wrap to 9
            r[24] = 10;                  // csub(3, 10) -> clamped to 0
            r[25] = f(-0.5);             // fmin(2.5, -0.5) -> -0.5
            r[26] = f(3.25);             // fmax(-1.5, 3.25) -> 3.25
            r[27] = 999;                 // OOB add -> returns 0, mem unchanged
        } else {
            for reg in 10..28 {
                r[reg] = 0xBEEF_0000 | reg as u32;
            }
        }
    }

    for (wave64, split) in modes {
        let r = run_parity(
            "mubuf_atomics_bounds",
            &code,
            &st,
            &mem,
            wave64,
            split,
            0..0,
            10..28,
            base..base + init.len(),
        );
        let want_mem = [
            125, 70, (-10i32) as u32, 5, 3, (-10i32) as u32, 0x0000_AA55, 0x0FF0_0FF0,
            0x5555_5555, 0x9999_8888, 0x0000_0099, 0, 9, 0, f(-0.5), f(3.25), 0xCAFE_BABE,
        ];
        assert_eq!(&r.mem[base..base + want_mem.len()], &want_mem);
        assert_eq!(r.vgpr[0][27], 0, "OOB atomic returns 0");
        assert_eq!(r.vgpr[1][10], 0xBEEF_000A, "inactive lane preserves VDATA");
    }
}

#[test]
fn direct_dxil_smem_loads_alignment_and_bounds_parity() {
    let modes = gpu_modes();
    if modes.is_empty() {
        return;
    }
    // s[0:1] = 0x100 (flat pointer for S_LOAD with negative signed offset and unaligned offset)
    // s[4:7] = V# base 0x100, num_records = 20 bytes (5 dwords: dwordx8 gives 5 data + 3 zero)
    // s[8:11] = V# base 0x200, stride = 4, num_records = 6 (24 bytes; dwordx4 at soffset 16 gives 2 data + 2 zero)
    // s12 = 16 (SOFFSET)
    let code = prog(&[
        // Unaligned S_LOAD_DWORD (+0x23 -> truncates to +0x20)
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 0), ("OFFSET", 0x23), ("SOFFSET", 125)], None),
        // Negative signed 21-bit offset (-16 = 0x1FFFF0) on S_LOAD_DWORDX2 from base 0x100 -> reads 0xF0..0xF8
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 18), ("SBASE", 0), ("OFFSET", 0x1F_FFF0), ("SOFFSET", 125)], None),
        // S_LOAD_DWORDX4 with SGPR SOFFSET (s12 = 16) + OFFSET = 4 -> 0x114
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 0), ("OFFSET", 4), ("SOFFSET", 12)], None),
        // S_BUFFER_LOAD_DWORDX8 across num_records = 20 (5 in range, 3 zero)
        ("S_BUFFER_LOAD_DWORDX8", "ENC_SMEM", &[("SDATA", 24), ("SBASE", 2), ("OFFSET", 0), ("SOFFSET", 125)], None),
        // S_BUFFER_LOAD_DWORDX4 with stride=4, num_records=6 (24 bytes), soffset=16 -> 2 in range, 2 zero
        ("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 32), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 12)], None),
        // S_BUFFER_LOAD_DWORD completely OOB -> 0
        ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 36), ("SBASE", 2), ("OFFSET", 0x40), ("SOFFSET", 125)], None),
    ]);

    let mut mem = vec![0u32; 256];
    for (i, w) in mem.iter_mut().enumerate() {
        *w = 0x5000_0000 | i as u32;
    }
    let mut st = KernelState::default();
    st.sgpr[0] = 0x100;
    st.sgpr[4..8].copy_from_slice(&[0x100, 0, 20, 0]);
    st.sgpr[8..12].copy_from_slice(&[0x200, 4 << 16, 6, 0]);
    st.sgpr[12] = 16;
    for s in 16..37 {
        st.sgpr[s] = 0xDEAD_C0DE;
    }

    for (wave64, split) in modes {
        let r = run_parity(
            "smem_align_bounds",
            &code,
            &st,
            &mem,
            wave64,
            split,
            16..37,
            0..0,
            0..0,
        );
        assert_eq!(r.sgpr[16], 0x5000_0000 | (0x120 / 4));
        assert_eq!((r.sgpr[18], r.sgpr[19]), (0x5000_0000 | (0xF0 / 4), 0x5000_0000 | (0xF4 / 4)));
        for i in 0..4 {
            assert_eq!(r.sgpr[20 + i], 0x5000_0000 | (0x114 / 4 + i as u32));
        }
        for i in 0..8 {
            let want = if i < 5 { 0x5000_0000 | (0x100 / 4 + i as u32) } else { 0 };
            assert_eq!(r.sgpr[24 + i], want, "s_buffer_load_dwordx8[{i}]");
        }
        assert_eq!(&r.sgpr[32..36], &[0x5000_0000 | (0x210 / 4), 0x5000_0000 | (0x214 / 4), 0, 0]);
        assert_eq!(r.sgpr[36], 0, "OOB s_buffer_load_dword -> 0");
    }
}
