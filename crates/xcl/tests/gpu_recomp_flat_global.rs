//! GPU verification tests for RDNA2 `ENC_FLAT` (`FLAT_*`) and `ENC_FLAT_GLBL` (`GLOBAL_*`):
//! multi-dword loads/stores (`DWORD`..`DWORDX4`), `VDST == ADDR`, signed/unsigned 12-bit offsets,
//! `SADDR == off` vs `SGPR`, sub-dword and D16 loads/stores across unaligned boundaries,
//! 32/64-bit integer atomics + `CSUB` (`GLC=0/1`, alignment truncation, contention), and `EXEC` masking.

use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner::{self, KernelState};
use xcl::gpu::recomp::{dxc, translate, Options};

const V: u64 = 256;
const EXEC_LO_MASK: u32 = 0xA55A_3C93;
const EXEC_HI_MASK: u32 = 0x5AA5_C339;
const SENTINEL32: u32 = 0xDEAD_BEEF;
const SENTINEL64: u64 = 0xDEAD_BEEF_CAFE_BABE;
type Insn = (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>);
type AtomicCase<T> = (&'static str, &'static str, u64, u64, T, T, T, T, bool);

fn flat(name: &'static str, fields: &[(&'static str, u64)]) -> Insn {
    let mut f = fields.to_vec();
    f.push(("SADDR", 0x7D));
    (name, "ENC_FLAT", f, None)
}
fn glbl(name: &'static str, fields: &[(&'static str, u64)]) -> Insn { (name, "ENC_FLAT_GLBL", fields.to_vec(), None) }
fn insn(name: &'static str, enc: &'static str, fields: &[(&'static str, u64)]) -> Insn { (name, enc, fields.to_vec(), None) }
fn keep_vgpr(v: u64) -> Insn { insn("V_OR_B32", "ENC_VOP2", &[("VDST", v), ("SRC0", 128), ("VSRC1", v)]) }

fn prog(list: &[Insn]) -> Vec<u8> {
    let mut code = Vec::new();
    for (name, enc, fields, lit) in list { code.extend(encode(name, enc, fields, *lit).unwrap_or_else(|e| panic!("{name}: {e}"))); }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    code
}

fn gpu_ready() -> bool {
    match (runner::adapter(), runner::wave64_programs_supported(), dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => true,
        (a, w, d) => {
            runner::skip_test(if a.is_err() || w.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" });
            false
        }
    }
}

fn wave_configs() -> Vec<(&'static str, Options)> {
    if !gpu_ready() { return Vec::new(); }
    let hw64 = runner::wave64_supported() == Ok(true);
    let opt = |wave64, split_wave64, hw_wave| Options { wave64, split_wave64, hw_wave, ..Default::default() };
    let mut cfgs = Vec::new();
    if hw64 { cfgs.push(("wave64", opt(true, false, 0))); }
    cfgs.push(("split-wave64", opt(true, true, 0)));
    cfgs.push(("wave32", opt(false, false, 0)));
    if hw64 { cfgs.push(("wave32@hw64", opt(false, false, 64))); }
    cfgs
}

fn compile_kernel(code: &[u8], opt: &Options) -> Vec<u8> {
    let t = translate(code, opt).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl))
}

fn active_lanes_for(opt: &Options) -> usize { if opt.wave64 || opt.hw_wave == 64 { 64 } else { 32 } }
fn is_active(lane: usize, opt: &Options) -> bool {
    let mask = if opt.wave64 { (EXEC_LO_MASK as u64) | ((EXEC_HI_MASK as u64) << 32) } else { EXEC_LO_MASK as u64 };
    ((mask >> if opt.wave64 { lane } else { lane & 31 }) & 1) != 0
}
fn put64(r: &mut [u32], at: usize, v: u64) { r[at] = v as u32; r[at + 1] = (v >> 32) as u32; }
fn get64(r: &[u32], at: usize) -> u64 { (r[at] as u64) | ((r[at + 1] as u64) << 32) }
fn set_mem64(m: &mut [u32], a: usize, v: u64) { m[a / 4] = v as u32; m[a / 4 + 1] = (v >> 32) as u32; }
fn get_mem64(m: &[u32], a: usize) -> u64 { (m[a / 4] as u64) | ((m[a / 4 + 1] as u64) << 32) }

#[test]
fn flat_and_global_translate_and_compile_with_dxc() {
    let contrast = prog(&[
        flat("FLAT_LOAD_DWORD", &[("VDST", 4), ("ADDR", 0), ("OFFSET", 0x800)]),
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 5), ("ADDR", 2), ("SADDR", 0x7F), ("OFFSET", 0x800)]),
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 6), ("ADDR", 8), ("SADDR", 4), ("OFFSET", 0xFF0)]),
    ]);
    let t1 = translate(&contrast, &Options::default()).unwrap();
    assert!(t1.complete(), "{:?}\n{}", t1.unsupported_why, t1.hlsl);
    assert!(t1.hlsl.contains("0x00000800u") && t1.hlsl.contains("0xFFFFF800u"), "{}", t1.hlsl);
    let t2 = translate(&contrast, &Options { mem_windows: 2, ..Default::default() }).unwrap();
    assert!(t2.complete() && t2.hlsl.contains("v1") && t2.hlsl.contains("v3") && t2.hlsl.contains("s5"), "{}", t2.hlsl);

    let mut all: Vec<Insn> = [
        "FLAT_STORE_DWORD", "FLAT_STORE_DWORDX2", "FLAT_STORE_DWORDX3", "FLAT_STORE_DWORDX4",
        "FLAT_STORE_BYTE", "FLAT_STORE_SHORT", "FLAT_STORE_BYTE_D16_HI", "FLAT_STORE_SHORT_D16_HI",
    ].into_iter().map(|n| flat(n, &[("ADDR", 0), ("DATA", 10)])).collect();
    for n in [
        "FLAT_LOAD_DWORDX2", "FLAT_LOAD_DWORDX3", "FLAT_LOAD_DWORDX4", "FLAT_LOAD_UBYTE", "FLAT_LOAD_SBYTE",
        "FLAT_LOAD_USHORT", "FLAT_LOAD_SSHORT", "FLAT_LOAD_UBYTE_D16", "FLAT_LOAD_UBYTE_D16_HI",
        "FLAT_LOAD_SBYTE_D16", "FLAT_LOAD_SBYTE_D16_HI", "FLAT_LOAD_SHORT_D16", "FLAT_LOAD_SHORT_D16_HI",
    ] { all.push(flat(n, &[("VDST", 14), ("ADDR", 0)])); }
    for n in [
        "FLAT_ATOMIC_SWAP", "FLAT_ATOMIC_CMPSWAP", "FLAT_ATOMIC_ADD", "FLAT_ATOMIC_SUB", "FLAT_ATOMIC_SMIN",
        "FLAT_ATOMIC_UMIN", "FLAT_ATOMIC_SMAX", "FLAT_ATOMIC_UMAX", "FLAT_ATOMIC_AND", "FLAT_ATOMIC_OR",
        "FLAT_ATOMIC_XOR", "FLAT_ATOMIC_INC", "FLAT_ATOMIC_DEC",
        "FLAT_ATOMIC_SWAP_X2", "FLAT_ATOMIC_CMPSWAP_X2", "FLAT_ATOMIC_ADD_X2", "FLAT_ATOMIC_SUB_X2",
        "FLAT_ATOMIC_SMIN_X2", "FLAT_ATOMIC_UMIN_X2", "FLAT_ATOMIC_SMAX_X2", "FLAT_ATOMIC_UMAX_X2",
        "FLAT_ATOMIC_AND_X2", "FLAT_ATOMIC_OR_X2", "FLAT_ATOMIC_XOR_X2", "FLAT_ATOMIC_INC_X2", "FLAT_ATOMIC_DEC_X2",
    ] { all.push(flat(n, &[("VDST", 20), ("ADDR", 0), ("DATA", 10), ("GLC", 1)])); }
    all.extend([
        glbl("GLOBAL_STORE_DWORDX3", &[("ADDR", 0), ("DATA", 10), ("SADDR", 4)]),
        glbl("GLOBAL_LOAD_DWORDX3", &[("VDST", 14), ("ADDR", 0), ("SADDR", 4)]),
        glbl("GLOBAL_LOAD_DWORDX4", &[("VDST", 14), ("ADDR", 0), ("SADDR", 4)]),
        glbl("GLOBAL_ATOMIC_CSUB", &[("VDST", 14), ("ADDR", 0), ("DATA", 10), ("SADDR", 4), ("GLC", 1)]),
    ]);
    if dxc::location().is_err() { return runner::skip_test("no DXC"); }
    let code = prog(&all);
    for (w64, split, hw) in [(true, false, 0), (true, true, 0), (false, false, 0), (false, false, 64)] {
        compile_kernel(&code, &Options { wave64: w64, split_wave64: split, hw_wave: hw, ..Default::default() });
    }
}

#[test]
fn gpu_flat_and_global_multi_dword_and_offsets() {
    let cfgs = wave_configs();
    if cfgs.is_empty() { return; }
    let code = prog(&[
        flat("FLAT_STORE_DWORD", &[("ADDR", 0), ("DATA", 10), ("OFFSET", 0)]),
        flat("FLAT_STORE_DWORDX2", &[("ADDR", 0), ("DATA", 11), ("OFFSET", 8)]),
        flat("FLAT_STORE_DWORDX3", &[("ADDR", 0), ("DATA", 13), ("OFFSET", 16)]),
        flat("FLAT_STORE_DWORDX4", &[("ADDR", 0), ("DATA", 16), ("OFFSET", 32)]),
        flat("FLAT_LOAD_DWORD", &[("VDST", 20), ("ADDR", 0), ("OFFSET", 0)]),
        flat("FLAT_LOAD_DWORDX3", &[("VDST", 21), ("ADDR", 0), ("OFFSET", 16)]),
        insn("V_MOV_B32", "ENC_VOP1", &[("VDST", 24), ("SRC0", V)]),
        insn("V_MOV_B32", "ENC_VOP1", &[("VDST", 25), ("SRC0", V + 1)]),
        flat("FLAT_LOAD_DWORDX2", &[("VDST", 24), ("ADDR", 24), ("OFFSET", 8)]),
        insn("V_MOV_B32", "ENC_VOP1", &[("VDST", 26), ("SRC0", V)]),
        insn("V_MOV_B32", "ENC_VOP1", &[("VDST", 27), ("SRC0", V + 1)]),
        flat("FLAT_LOAD_DWORDX4", &[("VDST", 26), ("ADDR", 26), ("OFFSET", 32)]),
        flat("FLAT_STORE_DWORD", &[("ADDR", 2), ("DATA", 10), ("OFFSET", 0x800)]),
        flat("FLAT_LOAD_DWORD", &[("VDST", 30), ("ADDR", 2), ("OFFSET", 0x800)]),
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 31), ("ADDR", 2), ("SADDR", 0x7F), ("OFFSET", 0x800)]),
        glbl("GLOBAL_STORE_DWORD", &[("ADDR", 4), ("DATA", 11), ("SADDR", 0x7D), ("OFFSET", 0xFF0)]),
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 32), ("ADDR", 4), ("SADDR", 0x7D), ("OFFSET", 0xFF0)]),
        glbl("GLOBAL_STORE_DWORDX3", &[("ADDR", 33), ("DATA", 34), ("SADDR", 4), ("OFFSET", 16)]),
        glbl("GLOBAL_LOAD_DWORDX3", &[("VDST", 37), ("ADDR", 33), ("SADDR", 4), ("OFFSET", 16)]),
        glbl("GLOBAL_LOAD_DWORDX4", &[("VDST", 40), ("ADDR", 33), ("SADDR", 4), ("OFFSET", 0)]),
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 44), ("ADDR", 44), ("SADDR", 6), ("OFFSET", 0xFF0)]),
        insn("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)]),
        flat("FLAT_STORE_DWORDX2", &[("ADDR", 46), ("DATA", 11), ("OFFSET", 0)]),
        flat("FLAT_LOAD_DWORDX2", &[("VDST", 48), ("ADDR", 46), ("OFFSET", 8)]),
        insn("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]),
    ]);

    for (label, opt) in cfgs {
        let dxil = compile_kernel(&code, &opt);
        let mut st = KernelState::default();
        (st.sgpr[0], st.sgpr[1], st.sgpr[4], st.sgpr[6]) = (EXEC_LO_MASK, EXEC_HI_MASK, 0x2400, 0x2C40);
        let mut mem = vec![0u32; 0x3200 / 4];
        let lanes = active_lanes_for(&opt);
        for l in 0..lanes {
            let l32 = l as u32;
            let r = &mut st.vgpr[l];
            (r[0], r[2], r[4], r[33], r[44], r[46]) = (0x200 + 64 * l32, 0x1A00 + 4 * l32, 0x1600 + 4 * l32, 32 * l32, 4 * l32, 0x2E00 + 16 * l32);
            r[10] = 0x1100_0000 | l32;
            for (k, &base_tag) in [0x2100_0000u32, 0x2200_0000, 0x3100_0000, 0x3200_0000, 0x3300_0000, 0x4100_0000, 0x4200_0000, 0x4300_0000, 0x4400_0000].iter().enumerate() {
                r[11 + k] = base_tag | l32;
            }
            (r[34], r[35], r[36]) = (0x6100_0000 | l32, 0x6200_0000 | l32, 0x6300_0000 | l32);
            (r[48], r[49]) = (0xDEAD_0000 | l32, 0xDEAD_1000 | l32);
            mem[(0x1200 + 4 * l) / 4] = 0xB200_0000 | l32;
            for k in 0..4 { mem[(0x2400 + 32 * l + 4 * k) / 4] = (0x7100_0000 + ((k as u32) << 24)) | l32; }
            mem[(0x2C30 + 4 * l) / 4] = 0x8800_0000 | l32;
            mem[(0x2E00 + 16 * l) / 4] = 0xCAFE_0000 | l32;
            mem[(0x2E00 + 16 * l + 4) / 4] = 0xCAFE_1000 | l32;
            mem[(0x2E00 + 16 * l + 8) / 4] = 0xBEEF_0000 | l32;
            mem[(0x2E00 + 16 * l + 12) / 4] = 0xBEEF_1000 | l32;
        }
        let r = runner::run(&dxil, &st, &mem).unwrap();
        for l in 0..lanes {
            let (l32, v, base) = (l as u32, &r.vgpr[l], (0x200 + 64 * l) / 4);
            assert_eq!(r.mem[base], 0x1100_0000 | l32, "{label} lane {l} flat_store_dword");
            assert_eq!(&r.mem[base + 2..base + 4], &[0x2100_0000 | l32, 0x2200_0000 | l32], "{label} lane {l} flat_store_dwordx2");
            assert_eq!(&r.mem[base + 4..base + 7], &[0x3100_0000 | l32, 0x3200_0000 | l32, 0x3300_0000 | l32], "{label} lane {l} flat_store_dwordx3");
            assert_eq!(&r.mem[base + 8..base + 12], &[0x4100_0000 | l32, 0x4200_0000 | l32, 0x4300_0000 | l32, 0x4400_0000 | l32], "{label} lane {l} flat_store_dwordx4");
            assert_eq!(v[20], 0x1100_0000 | l32);
            assert_eq!(&v[21..24], &[0x3100_0000 | l32, 0x3200_0000 | l32, 0x3300_0000 | l32]);
            assert_eq!(&v[24..26], &[0x2100_0000 | l32, 0x2200_0000 | l32], "{label} lane {l} flat_load_dwordx2 VDST==ADDR");
            assert_eq!(&v[26..30], &[0x4100_0000 | l32, 0x4200_0000 | l32, 0x4300_0000 | l32, 0x4400_0000 | l32], "{label} lane {l} flat_load_dwordx4 VDST==ADDR");
            assert_eq!((r.mem[(0x2200 + 4 * l) / 4], v[30], v[31]), (0x1100_0000 | l32, 0x1100_0000 | l32, 0xB200_0000 | l32), "{label} lane {l} +2048 vs -2048");
            assert_eq!((r.mem[(0x15F0 + 4 * l) / 4], v[32]), (0x2100_0000 | l32, 0x2100_0000 | l32), "{label} lane {l} global -16");
            assert_eq!(&v[37..40], &[0x6100_0000 | l32, 0x6200_0000 | l32, 0x6300_0000 | l32]);
            assert_eq!(&v[40..44], &[0x7100_0000 | l32, 0x7200_0000 | l32, 0x7300_0000 | l32, 0x7400_0000 | l32]);
            assert_eq!(v[44], 0x8800_0000 | l32, "{label} lane {l} global VDST==ADDR");
            let eb = (0x2E00 + 16 * l) / 4;
            let (want_m, want_v) = if is_active(l, &opt) {
                ([0x2100_0000 | l32, 0x2200_0000 | l32], [0xBEEF_0000 | l32, 0xBEEF_1000 | l32])
            } else {
                ([0xCAFE_0000 | l32, 0xCAFE_1000 | l32], [0xDEAD_0000 | l32, 0xDEAD_1000 | l32])
            };
            assert_eq!((&r.mem[eb..eb + 2], &v[48..50]), (want_m.as_slice(), want_v.as_slice()), "{label} lane {l} exec mask");
        }
    }
}

#[test]
fn gpu_flat_and_global_sub_dword_and_d16() {
    let cfgs = wave_configs();
    if cfgs.is_empty() { return; }
    let code = prog(&[
        flat("FLAT_STORE_BYTE", &[("ADDR", 0), ("DATA", 10), ("OFFSET", 1)]),
        flat("FLAT_STORE_BYTE_D16_HI", &[("ADDR", 0), ("DATA", 11), ("OFFSET", 2)]),
        flat("FLAT_STORE_SHORT", &[("ADDR", 0), ("DATA", 12), ("OFFSET", 7)]),
        flat("FLAT_STORE_SHORT_D16_HI", &[("ADDR", 0), ("DATA", 13), ("OFFSET", 10)]),
        flat("FLAT_LOAD_UBYTE", &[("VDST", 20), ("ADDR", 0), ("OFFSET", 17)]),
        flat("FLAT_LOAD_SBYTE", &[("VDST", 21), ("ADDR", 0), ("OFFSET", 17)]),
        flat("FLAT_LOAD_USHORT", &[("VDST", 22), ("ADDR", 0), ("OFFSET", 22)]),
        flat("FLAT_LOAD_SSHORT", &[("VDST", 23), ("ADDR", 0), ("OFFSET", 22)]),
        flat("FLAT_LOAD_UBYTE_D16", &[("VDST", 24), ("ADDR", 0), ("OFFSET", 17)]),
        flat("FLAT_LOAD_UBYTE_D16_HI", &[("VDST", 25), ("ADDR", 0), ("OFFSET", 17)]),
        flat("FLAT_LOAD_SBYTE_D16", &[("VDST", 26), ("ADDR", 0), ("OFFSET", 17)]),
        flat("FLAT_LOAD_SBYTE_D16_HI", &[("VDST", 27), ("ADDR", 0), ("OFFSET", 17)]),
        flat("FLAT_LOAD_SHORT_D16", &[("VDST", 28), ("ADDR", 0), ("OFFSET", 22)]),
        flat("FLAT_LOAD_SHORT_D16_HI", &[("VDST", 29), ("ADDR", 0), ("OFFSET", 23)]),
        glbl("GLOBAL_STORE_SHORT", &[("ADDR", 30), ("DATA", 12), ("SADDR", 4), ("OFFSET", 0xFF7)]),
        glbl("GLOBAL_STORE_BYTE_D16_HI", &[("ADDR", 30), ("DATA", 11), ("SADDR", 4), ("OFFSET", 0xFF1)]),
        glbl("GLOBAL_LOAD_USHORT", &[("VDST", 31), ("ADDR", 30), ("SADDR", 4), ("OFFSET", 6)]),
        glbl("GLOBAL_LOAD_SSHORT", &[("VDST", 32), ("ADDR", 30), ("SADDR", 4), ("OFFSET", 6)]),
        glbl("GLOBAL_LOAD_UBYTE_D16", &[("VDST", 33), ("ADDR", 30), ("SADDR", 4), ("OFFSET", 1)]),
        glbl("GLOBAL_LOAD_SBYTE_D16", &[("VDST", 34), ("ADDR", 30), ("SADDR", 4), ("OFFSET", 1)]),
        glbl("GLOBAL_LOAD_SBYTE_D16_HI", &[("VDST", 35), ("ADDR", 30), ("SADDR", 4), ("OFFSET", 1)]),
        glbl("GLOBAL_LOAD_SHORT_D16", &[("VDST", 36), ("ADDR", 30), ("SADDR", 4), ("OFFSET", 6)]),
        glbl("GLOBAL_LOAD_SHORT_D16_HI", &[("VDST", 37), ("ADDR", 30), ("SADDR", 4), ("OFFSET", 7)]),
        insn("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)]),
        flat("FLAT_STORE_BYTE", &[("ADDR", 0), ("DATA", 14), ("OFFSET", 12)]),
        flat("FLAT_LOAD_SBYTE_D16_HI", &[("VDST", 38), ("ADDR", 0), ("OFFSET", 16)]),
        insn("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]),
    ]);

    for (label, opt) in cfgs {
        let dxil = compile_kernel(&code, &opt);
        let mut st = KernelState::default();
        (st.sgpr[0], st.sgpr[1], st.sgpr[4]) = (EXEC_LO_MASK, EXEC_HI_MASK, 0xB00);
        let mut mem = vec![0u32; 0x1400 / 4];
        let lanes = active_lanes_for(&opt);
        for (l, r) in st.vgpr.iter_mut().enumerate().take(lanes) {
            let l32 = l as u32;
            (r[0], r[30]) = (0x200 + 32 * l32, 32 * l32 + 16);
            (r[10], r[11], r[12], r[13], r[14]) = (
                0xA0 | (l32 & 0xF), ((0xB0 | (l32 & 0xF)) << 16) | 0x7777, 0xC100 | l32, ((0xD100 | l32) << 16) | 0x9999, 0xEE,
            );
            for reg in (24..=29).chain(33..=38) { r[reg] = 0xA5A5_5A5A; }
            for base in [0x200 + 32 * l, 0xB00 + 32 * l] {
                mem[base / 4..base / 4 + 7].copy_from_slice(&[0x4433_2211, 0x8877_6655, 0xCCBB_AA99, 0x1122_3344, 0x01FF_807F, 0x8000_7FFF, 0x12EF_CDAB]);
            }
        }
        let r = runner::run(&dxil, &st, &mem).unwrap();
        for l in 0..lanes {
            let (l32, v, fw, gw) = (l as u32, &r.vgpr[l], (0x200 + 32 * l) / 4, (0xB00 + 32 * l) / 4);
            let (b1, b2, sh, sh_hi) = (0xA0 | (l32 & 0xF), 0xB0 | (l32 & 0xF), 0xC100 | l32, 0xD100 | l32);
            assert_eq!(r.mem[fw], (0x44 << 24) | (b2 << 16) | (b1 << 8) | 0x11, "{label} lane {l} flat bytes");
            assert_eq!(r.mem[fw + 1], ((sh & 0xFF) << 24) | 0x0077_6655, "{label} lane {l} flat short straddle lo");
            assert_eq!(r.mem[fw + 2], (sh_hi << 16) | 0xAA00 | (sh >> 8), "{label} lane {l} flat short straddle hi");
            assert_eq!((v[20], v[21], v[22], v[23]), (0x80, 0xFFFF_FF80, 0x8000, 0xFFFF_8000));
            assert_eq!((v[24], v[25], v[26], v[27], v[28], v[29]), (0xA5A5_0080, 0x0080_5A5A, 0xA5A5_FF80, 0xFF80_5A5A, 0xA5A5_8000, 0xAB80_5A5A));
            assert_eq!(r.mem[gw], 0x4433_0011 | (b2 << 8));
            assert_eq!(r.mem[gw + 1], ((sh & 0xFF) << 24) | 0x0077_6655);
            assert_eq!(r.mem[gw + 2], 0xCCBB_AA00 | (sh >> 8));
            assert_eq!((v[31], v[32], v[33], v[34], v[35], v[36], v[37]), (0x8000, 0xFFFF_8000, 0xA5A5_0080, 0xA5A5_FF80, 0xFF80_5A5A, 0xA5A5_8000, 0xAB80_5A5A));
            let want_exec = if is_active(l, &opt) { (0x1122_33EE, 0x007F_5A5A) } else { (0x1122_3344, 0xA5A5_5A5A) };
            assert_eq!((r.mem[fw + 3], v[38]), want_exec, "{label} lane {l} sub-dword exec");
        }
    }
}

#[test]
fn gpu_flat_and_global_atomics_32_and_64() {
    let cfgs = wave_configs();
    if cfgs.is_empty() { return; }
    let mut ops32 = Vec::new();
    let mut ops64 = Vec::new();
    #[rustfmt::skip]
    let t32: &[AtomicCase<u32>] = &[
        ("ENC_FLAT", "FLAT_ATOMIC_SWAP", 0x02, 1, 0x1111_2222, 0x3333_4444, 0, 0x3333_4444, true),
        ("ENC_FLAT", "FLAT_ATOMIC_CMPSWAP", 0x04, 1, 0xAAAA_5555, 0x7777_8888, 0xAAAA_5555, 0x7777_8888, true),
        ("ENC_FLAT", "FLAT_ATOMIC_CMPSWAP", 0x08, 1, 0xAAAA_5555, 0x7777_8888, 0xAAAA_0000, 0xAAAA_5555, true),
        ("ENC_FLAT", "FLAT_ATOMIC_ADD", 0x0C, 1, 100, 25, 0, 125, true),
        ("ENC_FLAT", "FLAT_ATOMIC_SUB", 0x10, 1, 100, 35, 0, 65, true),
        ("ENC_FLAT", "FLAT_ATOMIC_SMIN", 0x14, 1, 10, (-50i32) as u32, 0, (-50i32) as u32, true),
        ("ENC_FLAT", "FLAT_ATOMIC_UMIN", 0x18, 1, 0x8000_0001, 5, 0, 5, true),
        ("ENC_FLAT", "FLAT_ATOMIC_SMAX", 0x1C, 1, (-50i32) as u32, 10, 0, 10, true),
        ("ENC_FLAT", "FLAT_ATOMIC_UMAX", 0x20, 1, 5, 0x8000_0001, 0, 0x8000_0001, true),
        ("ENC_FLAT", "FLAT_ATOMIC_AND", 0x24, 1, 0xF0F0_AA55, 0x0FF0_FF0F, 0, 0x00F0_AA05, true),
        ("ENC_FLAT", "FLAT_ATOMIC_OR", 0x28, 1, 0x1200_0034, 0x0056_7800, 0, 0x1256_7834, true),
        ("ENC_FLAT", "FLAT_ATOMIC_XOR", 0x2C, 1, 0xFF00_AA55, 0x0F0F_55AA, 0, 0xF00F_FFFF, true),
        ("ENC_FLAT", "FLAT_ATOMIC_INC", 0x30, 1, 7, 9, 0, 8, true),
        ("ENC_FLAT", "FLAT_ATOMIC_INC", 0x34, 1, 9, 9, 0, 0, true),
        ("ENC_FLAT", "FLAT_ATOMIC_DEC", 0x38, 1, 0, 9, 0, 9, true),
        ("ENC_FLAT", "FLAT_ATOMIC_DEC", 0x3C, 0, 4, 9, 0, 3, false),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_SWAP", 0x00, 1, 0xDEAD_0001, 0xBEEF_0002, 0, 0xBEEF_0002, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_CMPSWAP", 0x04, 1, 0xCAFE_1111, 0xBABE_2222, 0xCAFE_1111, 0xBABE_2222, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_SMIN", 0x08, 1, (-3i32) as u32, (-9i32) as u32, 0, (-9i32) as u32, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_UMIN", 0x0C, 1, 0xFFFF_FFF0, 99, 0, 99, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_UMAX", 0x10, 1, 99, 0xFFFF_FFF0, 0, 0xFFFF_FFF0, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_AND", 0x14, 1, 0xAAAA_FFFF, 0x0F0F_F0F0, 0, 0x0A0A_F0F0, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_OR", 0x18, 1, 0x0102_0000, 0x0000_0304, 0, 0x0102_0304, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_XOR", 0x1C, 1, 0x1234_5678, 0xFFFF_FFFF, 0, 0xEDCB_A987, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_CSUB", 0x20, 1, 25, 10, 0, 15, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_CSUB", 0x24, 0, 5, 10, 0, 0, false),
    ];
    #[rustfmt::skip]
    let t64: &[AtomicCase<u64>] = &[
        ("ENC_FLAT", "FLAT_ATOMIC_SWAP_X2", 0x04, 1, 0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210, 0, 0xFEDC_BA98_7654_3210, true),
        ("ENC_FLAT", "FLAT_ATOMIC_CMPSWAP_X2", 0x08, 1, 0x1111_2222_3333_4444, 0x5555_6666_7777_8888, 0x1111_2222_3333_4444, 0x5555_6666_7777_8888, true),
        ("ENC_FLAT", "FLAT_ATOMIC_CMPSWAP_X2", 0x10, 1, 0x0001_AAAA_BBBB_CCCC, 0x9999_0000_0000_0000, 0x0002_AAAA_BBBB_CCCC, 0x0001_AAAA_BBBB_CCCC, true),
        ("ENC_FLAT", "FLAT_ATOMIC_ADD_X2", 0x18, 1, 0x0000_0000_FFFF_FFF0, 0x20, 0, 0x0000_0001_0000_0010, true),
        ("ENC_FLAT", "FLAT_ATOMIC_SUB_X2", 0x20, 1, 0x0000_0001_0000_0008, 0x10, 0, 0x0000_0000_FFFF_FFF8, true),
        ("ENC_FLAT", "FLAT_ATOMIC_SMIN_X2", 0x28, 1, 1, (-1i64 << 32) as u64, 0, (-1i64 << 32) as u64, true),
        ("ENC_FLAT", "FLAT_ATOMIC_UMIN_X2", 0x30, 1, 0x8000_0000_0000_0001, 7, 0, 7, true),
        ("ENC_FLAT", "FLAT_ATOMIC_SMAX_X2", 0x38, 1, (-1i64 << 32) as u64, 1, 0, 1, true),
        ("ENC_FLAT", "FLAT_ATOMIC_UMAX_X2", 0x40, 1, 7, 0x8000_0000_0000_0001, 0, 0x8000_0000_0000_0001, true),
        ("ENC_FLAT", "FLAT_ATOMIC_AND_X2", 0x48, 1, 0xFF00_FF00_AAAA_5555, 0x0FF0_F0F0_FFFF_00FF, 0, 0x0F00_F000_AAAA_0055, true),
        ("ENC_FLAT", "FLAT_ATOMIC_OR_X2", 0x50, 1, 0x1100_0022_3300_0044, 0x0055_6600_0077_8800, 0, 0x1155_6622_3377_8844, true),
        ("ENC_FLAT", "FLAT_ATOMIC_XOR_X2", 0x58, 1, 0xAAAA_BBBB_CCCC_DDDD, 0xFFFF_FFFF_0000_FFFF, 0, 0x5555_4444_CCCC_2222, true),
        ("ENC_FLAT", "FLAT_ATOMIC_INC_X2", 0x60, 1, 0x0000_0001_0000_0005, 0x0000_0001_0000_0005, 0, 0, true),
        ("ENC_FLAT", "FLAT_ATOMIC_DEC_X2", 0x68, 0, 0, 0x0000_0001_0000_0005, 0, 0x0000_0001_0000_0005, false),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_SWAP_X2", 0x00, 1, 0x1010_2020_3030_4040, 0x5050_6060_7070_8080, 0, 0x5050_6060_7070_8080, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_CMPSWAP_X2", 0x08, 1, 0xA1B2_C3D4_E5F6_0718, 0x0102_0304_0506_0708, 0xA1B2_C3D4_E5F6_0718, 0x0102_0304_0506_0708, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_SUB_X2", 0x10, 1, 0x0000_0002_0000_0000, 1, 0, 0x0000_0001_FFFF_FFFF, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_SMIN_X2", 0x18, 1, 42, (-100i64) as u64, 0, (-100i64) as u64, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_UMIN_X2", 0x20, 1, 0xFFFF_FFFF_0000_0000, 9, 0, 9, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_SMAX_X2", 0x28, 1, (-100i64) as u64, 42, 0, 42, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_UMAX_X2", 0x30, 1, 9, 0xFFFF_FFFF_0000_0000, 0, 0xFFFF_FFFF_0000_0000, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_AND_X2", 0x38, 1, 0xF0F0_F0F0_1234_5678, 0x0F0F_FFFF_FF00_00FF, 0, 0x0000_F0F0_1200_0078, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_OR_X2", 0x40, 1, 0x0100_0200_0300_0400, 0x0010_0020_0030_0040, 0, 0x0110_0220_0330_0440, true),
        ("ENC_FLAT_GLBL", "GLOBAL_ATOMIC_XOR_X2", 0x48, 0, 0x1122_3344_5566_7788, 0xFFFF_FFFF_FFFF_FFFF, 0, 0xEEDD_CCBB_AA99_8877, false),
    ];

    for (i, &(enc, name, off, glc, ..)) in t32.iter().enumerate() {
        let mut f = vec![("VDST", 70 + i as u64), ("ADDR", if enc == "ENC_FLAT" { 0 } else { 2 }), ("DATA", 10 + 2 * i as u64), ("OFFSET", off), ("GLC", glc)];
        if enc == "ENC_FLAT_GLBL" { f.push(("SADDR", 4)); ops32.push(insn(name, enc, &f)); } else { ops32.push(flat(name, &f)); }
        if glc == 0 { ops32.push(keep_vgpr(70 + i as u64)); }
    }
    ops32.extend([
        insn("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)]),
        flat("FLAT_ATOMIC_ADD", &[("VDST", 100), ("ADDR", 4), ("DATA", 6), ("OFFSET", 0), ("GLC", 1)]),
        flat("FLAT_ATOMIC_ADD", &[("VDST", 101), ("ADDR", 8), ("DATA", 6), ("OFFSET", 0), ("GLC", 1)]),
        flat("FLAT_ATOMIC_XOR", &[("VDST", 102), ("ADDR", 8), ("DATA", 7), ("OFFSET", 4), ("GLC", 1)]),
        flat("FLAT_ATOMIC_CMPSWAP", &[("VDST", 103), ("ADDR", 8), ("DATA", 150), ("OFFSET", 8), ("GLC", 1)]),
        insn("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]),
    ]);

    for (i, &(enc, name, off, glc, ..)) in t64.iter().enumerate() {
        let mut f = vec![("VDST", 110 + 2 * i as u64), ("ADDR", if enc == "ENC_FLAT" { 0 } else { 2 }), ("DATA", 10 + 4 * i as u64), ("OFFSET", off), ("GLC", glc)];
        if enc == "ENC_FLAT_GLBL" { f.push(("SADDR", 4)); ops64.push(insn(name, enc, &f)); } else { ops64.push(flat(name, &f)); }
        if glc == 0 { ops64.push(keep_vgpr(110 + 2 * i as u64)); ops64.push(keep_vgpr(111 + 2 * i as u64)); }
    }
    ops64.extend([
        insn("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)]),
        flat("FLAT_ATOMIC_ADD_X2", &[("VDST", 170), ("ADDR", 4), ("DATA", 6), ("OFFSET", 0), ("GLC", 1)]),
        flat("FLAT_ATOMIC_ADD_X2", &[("VDST", 172), ("ADDR", 8), ("DATA", 6), ("OFFSET", 0), ("GLC", 1)]),
        flat("FLAT_ATOMIC_CMPSWAP_X2", &[("VDST", 174), ("ADDR", 8), ("DATA", 180), ("OFFSET", 8), ("GLC", 1)]),
        insn("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]),
    ]);
    let (code32, code64) = (prog(&ops32), prog(&ops64));

    const WIN_INIT32: u32 = 0xAAAA_5555;
    const WIN_INIT64: u64 = 0x1111_2222_3333_4444;
    for (label, opt) in cfgs {
        let (dxil32, dxil64, lanes) = (compile_kernel(&code32, &opt), compile_kernel(&code64, &opt), active_lanes_for(&opt));

        let mut st32 = KernelState::default();
        (st32.sgpr[0], st32.sgpr[1], st32.sgpr[4]) = (EXEC_LO_MASK, EXEC_HI_MASK, 0x2400);
        let mut mem32 = vec![0u32; 0x3600 / 4];
        mem32[0x88 / 4] = WIN_INIT32;
        for l in 0..lanes {
            let r = &mut st32.vgpr[l];
            (r[0], r[2], r[4], r[8], r[6], r[7]) = (0x200 + 64 * l as u32, 64 * l as u32, 0x3400 + 4 * l as u32, 0x80, l as u32 + 1, 1 << (l & 15));
            (r[100], r[103], r[150], r[151]) = (SENTINEL32, SENTINEL32, 0x7777_8888, WIN_INIT32);
            mem32[(0x3400 + 4 * l) / 4] = 1000;
            for (i, &(enc, _, off, _, init, data, cmp, ..)) in t32.iter().enumerate() {
                (r[10 + 2 * i], r[11 + 2 * i], r[70 + i]) = (data, cmp, SENTINEL32);
                mem32[(if enc == "ENC_FLAT" { 0x200 } else { 0x2400 } + 64 * l + (off as usize & !3)) / 4] = init;
            }
        }
        let r32 = runner::run(&dxil32, &st32, &mem32).unwrap();
        let (mut sum32, mut xor32, mut winners32) = (0u32, 0u32, 0usize);
        for l in 0..lanes {
            let v = &r32.vgpr[l];
            for (i, &(enc, name, off, _, init, _, _, want_mem, check_old)) in t32.iter().enumerate() {
                let c = if enc == "ENC_FLAT" { 0x200 } else { 0x2400 } + 64 * l + (off as usize & !3);
                assert_eq!(r32.mem[c / 4], want_mem, "{label} lane {l} {name} mem");
                assert_eq!(v[70 + i], if check_old { init } else { SENTINEL32 }, "{label} lane {l} {name} vdst");
            }
            if is_active(l, &opt) {
                sum32 += l as u32 + 1;
                xor32 ^= 1 << (l as u32 & 15);
                assert_eq!((r32.mem[(0x3400 + 4 * l) / 4], v[100]), (1000 + l as u32 + 1, 1000));
                if v[103] == WIN_INIT32 { winners32 += 1; }
            } else {
                assert_eq!((r32.mem[(0x3400 + 4 * l) / 4], v[100], v[103]), (1000, SENTINEL32, SENTINEL32));
            }
        }
        assert_eq!((r32.mem[0x80 / 4], r32.mem[0x84 / 4], winners32), (sum32, xor32, 1), "{label} shared 32-bit");

        let mut st64 = KernelState::default();
        (st64.sgpr[0], st64.sgpr[1], st64.sgpr[4]) = (EXEC_LO_MASK, EXEC_HI_MASK, 0x2600);
        let mut mem64 = vec![0u32; 0x5200 / 4];
        set_mem64(&mut mem64, 0x88, WIN_INIT64);
        for (l, r) in st64.vgpr.iter_mut().enumerate().take(lanes) {
            (r[0], r[2], r[4], r[8]) = (0x200 + 128 * l as u32, 128 * l as u32, 0x4800 + 8 * l as u32, 0x80);
            put64(r, 6, (1u64 << 32) | (l as u64 + 1));
            for (reg, val) in [(170, SENTINEL64), (174, SENTINEL64), (180, 0x5555_6666_7777_8888), (182, WIN_INIT64)] { put64(r, reg, val); }
            set_mem64(&mut mem64, 0x4800 + 8 * l, 0x0000_0001_0000_0100);
            for (i, &(enc, _, off, _, init, data, cmp, ..)) in t64.iter().enumerate() {
                put64(r, 10 + 4 * i, data);
                put64(r, 12 + 4 * i, cmp);
                put64(r, 110 + 2 * i, SENTINEL64);
                set_mem64(&mut mem64, if enc == "ENC_FLAT" { 0x200 } else { 0x2600 } + 128 * l + (off as usize & !7), init);
            }
        }
        let r64 = runner::run(&dxil64, &st64, &mem64).unwrap();
        let (mut sum64, mut winners64) = (0u64, 0usize);
        for l in 0..lanes {
            let v = &r64.vgpr[l];
            for (i, &(enc, name, off, _, init, _, _, want_mem, check_old)) in t64.iter().enumerate() {
                let c = if enc == "ENC_FLAT" { 0x200 } else { 0x2600 } + 128 * l + (off as usize & !7);
                assert_eq!(get_mem64(&r64.mem, c), want_mem, "{label} lane {l} {name} mem");
                assert_eq!(get64(v, 110 + 2 * i), if check_old { init } else { SENTINEL64 }, "{label} lane {l} {name} vdst");
            }
            if is_active(l, &opt) {
                sum64 += (1u64 << 32) | (l as u64 + 1);
                assert_eq!((get_mem64(&r64.mem, 0x4800 + 8 * l), get64(v, 170)), (0x0000_0002_0000_0101 + l as u64, 0x0000_0001_0000_0100));
                if get64(v, 174) == WIN_INIT64 { winners64 += 1; }
            } else {
                assert_eq!((get_mem64(&r64.mem, 0x4800 + 8 * l), get64(v, 170), get64(v, 174)), (0x0000_0001_0000_0100, SENTINEL64, SENTINEL64));
            }
        }
        assert_eq!((get_mem64(&r64.mem, 0x80), winners64), (sum64, 1), "{label} shared 64-bit");
    }
}
