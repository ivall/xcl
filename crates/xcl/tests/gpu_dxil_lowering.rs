//! End-to-end GPU execution and HLSL-differential tests for the direct
//! RDNA2 -> DXIL translator (`xcl::gpu::dxil::kernel`).
//!
//! Covers `Mode::Kernel`, `Mode::Compute`, `Mode::Vertex`, `Mode::Pixel`, and
//! `Mode::Mesh` across native wave64, split-wave64, and wave32 without
//! requiring `XCL_DIRECT_DXIL` to be set.

use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32_UINT,
};
use xcl::gpu::dxbc::SigElement;
use xcl::gpu::dxil::kernel::{
    translate_compute, translate_full, translate_mesh, translate_pixel, translate_vertex,
    KernelOptions,
};
use xcl::gpu::dxil::validate;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::{
    self, dxc, mesh_passthrough,
    runner::{self, KernelResult, KernelState},
    CsInterface, Mode, MsInterface, Options, PsInterface, VsInterface, HEAP_WINDOWS, MESH_HEADER,
};

const V: u64 = 256; // VALU source encoding for v0

fn gpu_ready() -> bool {
    match (runner::adapter(), runner::wave64_programs_supported(), dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => true,
        (a, w, d) => {
            runner::skip_test(if a.is_err() || w.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" });
            false
        }
    }
}

fn mesh_ready() -> bool {
    match runner::mesh_shaders_supported() {
        Ok(true) => true,
        r => {
            runner::skip_test(if r.is_err() { "no GPU" } else { "no mesh shaders" });
            false
        }
    }
}

fn configs(include_w32: bool) -> Vec<(&'static str, bool, bool)> {
    let mut out = Vec::new();
    if runner::wave64_supported() == Ok(true) {
        out.push(("native-w64", true, false));
    }
    if runner::wave32_supported() == Ok(true) {
        out.push(("split-w64", true, true));
        if include_w32 {
            out.push(("wave32", false, false));
        }
    }
    out
}

fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    let mut code = Vec::new();
    for (name, enc, fields, lit) in list {
        code.extend(encode(name, enc, fields, *lit).unwrap_or_else(|e| panic!("{name}: {e}")));
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    code
}

fn hlsl_compile(code: &[u8], opt: &Options, target: &str) -> Vec<u8> {
    let t = recomp::translate(code, opt).unwrap_or_else(|e| panic!("hlsl translate: {e}"));
    assert!(t.complete(), "{:?}", t.unsupported_why);
    dxc::compile(&t.hlsl, "main", target, &[]).unwrap_or_else(|e| panic!("dxc {target}: {e}\n{}", t.hlsl))
}

fn sig(name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8) -> SigElement {
    SigElement { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 }
}

fn assert_kernel_matches(
    label: &str, code: &[u8], st: &KernelState, mem: &[u32], wave64: bool, split: bool,
    lanes: usize, check_sgpr: &[usize], check_vgpr: &[usize],
) -> KernelResult {
    let ko = KernelOptions { wave64, split, mem_windows: 1, sampler_base: 0 };
    let direct = translate_full(code, ko).unwrap_or_else(|e| panic!("{label}: direct: {e}"));
    validate::validate(&direct.dxil).unwrap_or_else(|e| panic!("{label}: validate: {e}"));
    let hlsl = hlsl_compile(code, &Options { mode: Mode::Kernel, wave64, split_wave64: split, ..Default::default() }, "cs_6_6");
    let got = runner::run(&direct.dxil, st, mem).unwrap_or_else(|e| panic!("{label}: run direct: {e}"));
    let want = runner::run(&hlsl, st, mem).unwrap_or_else(|e| panic!("{label}: run hlsl: {e}"));
    for &s in check_sgpr {
        assert_eq!(got.sgpr[s], want.sgpr[s], "{label}: s{s}");
    }
    assert_eq!(got.scc, want.scc, "{label}: scc");
    for l in 0..lanes {
        for &v in check_vgpr {
            assert_eq!(got.vgpr[l][v], want.vgpr[l][v], "{label}: lane {l} v{v}");
        }
    }
    assert_eq!(got.mem, want.mem, "{label}: memory");
    got
}

#[test]
fn direct_dxil_kernel_control_flow_exec_and_whole_wave() {
    if !gpu_ready() {
        return;
    }
    // Nested EXEC save/restore, cbranch_execz, cross-half wave masks (s_bcnt1, s_ff1,
    // v_mbcnt), readlane/readfirstlane, ds_swizzle, ds_bpermute, and VGPR preservation.
    let wave_prog = prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(40)),
        ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)], None),
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 106), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 107), ("SRC1", V + 5)], None),
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 106)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 1), ("VSRC1", 3)], None),
        ("S_XOR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)], None),
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 2)], None),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 1)], None),
        ("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 3), ("VSRC1", 0)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 2)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 1), ("SRC1", 1)], None),
        ("S_FF1_I32_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 106)], None),
        ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 6), ("ADDR", 1), ("OFFSET0", 0x1F), ("OFFSET1", 1 << 2)], None),
        ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 7), ("ADDR", 4), ("DATA0", 1)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[1] = 37;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[1] = 1000 + l as u32;
        r[3] = 0x5500 + l as u32;
        r[4] = (63 - l as u32) * 4;
    }
    for (name, _, split) in configs(false) {
        let r = assert_kernel_matches(name, &wave_prog, &st, &[], true, split, 64, &[0, 2, 3, 4, 5, 6], &[3, 5, 6, 7]);
        assert_eq!((r.sgpr[0], r.sgpr[4], r.sgpr[5], r.sgpr[6]), (40, 1040, 1037, 0), "{name}: scalar checks");
        assert_eq!((r.vgpr[0][6], r.vgpr[3][7]), (1001, 1060), "{name}: ds_swizzle / ds_bpermute");
    }

    // Irreducible two-entry loop CFG: s0 == 1 jumps directly into the loop body (dxil::cfg node split).
    let irred = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 129)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 2)], None),
        ("S_ADD_I32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 130), ("VSRC1", 2)], None),
        ("S_ADD_I32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 130)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 0), ("VSRC1", 2)], None),
        ("S_CMP_LT_I32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 136)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFA)], None),
    ]);
    for entry_s0 in [0u32, 1] {
        let mut ist = KernelState::default();
        ist.sgpr[0] = entry_s0;
        for (l, r) in ist.vgpr.iter_mut().enumerate() {
            r[0] = l as u32 + 1;
        }
        for (name, w64, split) in configs(true) {
            assert_kernel_matches(&format!("irred-s0={entry_s0}-{name}"), &irred, &ist, &[], w64, split, if w64 { 64 } else { 32 }, &[1], &[2]);
        }
    }
}

#[test]
fn direct_dxil_kernel_memory_lds_and_exact_valu() {
    if !gpu_ready() {
        return;
    }
    // SMEM, MUBUF (load/store/atomic+OOB), GLOBAL (load/store/atomic), FLAT (load/store),
    // LDS (write/add_rtn/add/read + barrier), and exact VALU rules (fract edge, -0 < +0,
    // NaN min/max, bitwise class abs/neg, 64-bit carry, VOP3P saturated i16 add).
    let code = prog(&[
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 0), ("OFFSET", 0x10), ("SOFFSET", 125), ("GLC", 1)], None),
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("OFFSET", 4), ("OFFEN", 1)], None),
        ("BUFFER_ATOMIC_ADD", "ENC_MUBUF", &[("VDATA", 3), ("VADDR", 1), ("SRSRC", 4), ("SOFFSET", 128), ("GLC", 1), ("OFFEN", 1)], None),
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("OFFSET", 0x100), ("OFFEN", 1)], None),
        ("GLOBAL_ATOMIC_ADD", "ENC_FLAT_GLBL", &[("VDST", 4), ("ADDR", 5), ("DATA", 6), ("SADDR", 0x7D), ("OFFSET", 0x200), ("GLC", 1)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 7), ("ADDR", 5), ("SADDR", 0x7D), ("OFFSET", 0x200)], None),
        ("FLAT_STORE_DWORD", "ENC_FLAT", &[("ADDR", 8), ("DATA", 7), ("OFFSET", 0x300)], None),
        ("FLAT_LOAD_DWORD", "ENC_FLAT", &[("VDST", 10), ("ADDR", 8), ("OFFSET", 0x300)], None),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 26), ("DATA0", 11)], None),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 27), ("DATA0", 28)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_ADD_RTN_U32", "ENC_DS", &[("VDST", 12), ("ADDR", 26), ("DATA0", 6)], None),
        ("DS_ADD_U32", "ENC_DS", &[("ADDR", 27), ("DATA0", 6)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 13), ("ADDR", 27)], None),
        ("V_FRACT_F32", "ENC_VOP1", &[("VDST", 14), ("SRC0", V + 14)], None),
        ("V_MIN_F32", "ENC_VOP2", &[("VDST", 15), ("SRC0", V + 15), ("VSRC1", 16)], None),
        ("V_MAX_F32", "ENC_VOP2", &[("VDST", 17), ("SRC0", V + 15), ("VSRC1", 16)], None),
        ("V_CMP_CLASS_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 18), ("SRC1", 19), ("ABS", 1), ("NEG", 1)], None),
        ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 20), ("SDST", 106), ("SRC0", V + 20), ("SRC1", V + 21)], None),
        ("V_ADD_CO_CI_U32", "ENC_VOP2", &[("VDST", 22), ("SRC0", V + 22), ("VSRC1", 23)], None),
        ("V_PK_ADD_I16", "ENC_VOP3P", &[("VDST", 24), ("SRC0", V + 24), ("SRC1", V + 25), ("OP_SEL_HI", 3), ("CLAMP", 1)], None),
    ]);
    let mut mem = vec![0u32; 512];
    for (i, w) in mem.iter_mut().enumerate().take(128) {
        *w = 0x100 + i as u32;
    }
    mem[0x80] = 10;
    let mut st = KernelState::default();
    st.sgpr[16..20].copy_from_slice(&[0x40, 0, 0x200, 0]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[0] = if l < 48 { l32 * 4 } else { 0x400 };
        r[1] = 0x20;
        r[3] = 1;
        r[6] = 1;
        r[8] = l32 * 4;
        r[11] = 0xAA00 + l32;
        r[14] = [f32::INFINITY, -(2.0f32.powi(-25)), 1.25 + l as f32][l % 3].to_bits();
        r[15] = if l % 2 == 0 { (-0.0f32).to_bits() } else { f32::NAN.to_bits() };
        r[16] = if l % 2 == 0 { 0.0f32.to_bits() } else { 3.5f32.to_bits() };
        r[18] = if l % 2 == 0 { 0x7F80_0001 } else { 0x0000_0001 };
        r[19] = (1 << 0) | (1 << 4);
        r[20] = if l % 2 == 0 { 0xFFFF_FFF0 } else { 5 };
        r[21] = 0x20;
        r[22] = 100;
        r[23] = 7;
        r[24] = 30000u32 | ((0xFFFF_FFF0u32 & 0xFFFF) << 16);
        r[25] = 5000u32 | ((0xFFFF_FFE0u32 & 0xFFFF) << 16);
        r[26] = l32 * 4;
        r[27] = 0x400;
    }
    for (name, w64, split) in configs(true) {
        let lanes = if w64 { 64 } else { 32 };
        let got = assert_kernel_matches(
            name, &code, &st, &mem, w64, split, lanes,
            &[8, 9, 10, 11, 12, 13], &[2, 3, 4, 7, 10, 12, 13, 14, 15, 17, 20, 22, 24],
        );
        assert!(f32::from_bits(got.vgpr[0][14]).is_nan(), "{name}: v_fract_f32(+inf) == NaN");
        assert_eq!(got.vgpr[1][14], 0x3F7F_FFFF, "{name}: v_fract_f32(-2^-25) below 1.0");
        assert_eq!(got.vgpr[0][15], (-0.0f32).to_bits(), "{name}: min(-0, +0) == -0");
        assert_eq!(got.vgpr[0][17], 0.0f32.to_bits(), "{name}: max(-0, +0) == +0");
        assert_eq!(got.vgpr[1][15], 3.5f32.to_bits(), "{name}: min(NaN, 3.5) == 3.5");
        assert_eq!(got.vgpr[0][12], 0xAA00, "{name}: ds_add_rtn_u32 old");
        assert_eq!(got.vgpr[0][13], lanes as u32, "{name}: shared LDS sum");
        assert_eq!(got.mem[0x80], 10 + lanes as u32, "{name}: global atomic sum");
        assert_eq!(got.vgpr[0][24], 0x7FFF | (0xFFD0 << 16), "{name}: v_pk_add_i16 clamp");
    }
}

#[test]
fn direct_dxil_compute_dispatch_spin_partial_and_2d() {
    if !gpu_ready() {
        return;
    }
    // 1) Multi-wave [128,1,1] with tg_size=true, cross-wave LDS + S_BARRIER, and ds_bpermute
    // (exercises Partner::Spin when split=true).
    let spin_prog = prog(&[
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 255), ("VSRC1", 0)], Some(63)),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 2), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 136), ("VSRC1", 4)], None),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 3), ("VSRC1", 4)], None),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 3), ("DATA0", 4)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 255), ("VSRC1", 0)], Some(127)),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 130), ("VSRC1", 5)], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 6), ("ADDR", 5)], None),
        ("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 255), ("VSRC1", 2)], Some(63)),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 130), ("VSRC1", 7)], None),
        ("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 7), ("ADDR", 7), ("DATA0", 4)], None),
        ("S_MUL_I32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 2), ("SSRC1", 255)], Some(128)),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 4), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 131), ("VSRC1", 8)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 0), ("VSRC1", 8)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORDX2", "ENC_FLAT_GLBL", &[("ADDR", 8), ("DATA", 6), ("SADDR", 0x7D)], None),
    ]);
    let cs_spin = CsInterface { threads: [128, 1, 1], user_sgprs: 2, tgid: [true, false, false], tg_size: true, tid_components: 1 };
    let mem = vec![0u32; (0x100 + 2 * 128 * 8) / 4];
    for (name, _, split) in configs(false) {
        let dxil = translate_compute(&spin_prog, KernelOptions { wave64: true, split, mem_windows: 1, sampler_base: 0 }, &cs_spin).unwrap_or_else(|e| panic!("spin {name}: {e}"));
        let hlsl = hlsl_compile(&spin_prog, &Options { mode: Mode::Compute(cs_spin.clone()), wave64: true, split_wave64: split, ..Default::default() }, "cs_6_6");
        assert_eq!(runner::dispatch(&dxil.dxil, [2, 1, 1], &[0x100], &mem).unwrap(), runner::dispatch(&hlsl, [2, 1, 1], &[0x100], &mem).unwrap(), "compute spin {name}");
    }

    // 2) 2D [8,4,1] (XCL_WT path), partial [40,1,1] (Partner::Present), and [16,1,1] (Partner::Missing).
    let simple_cs = prog(&[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 136), ("VSRC1", 1)], None),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 4), ("VSRC1", 0)], None),
        ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 126)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 5), ("VSRC1", 4)], None),
        ("S_MUL_I32", "ENC_SOP2", &[("SDST", 6), ("SSRC0", 2), ("SSRC1", 1)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 6), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 131), ("VSRC1", 1)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 7), ("VSRC1", 6)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 6)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 0), ("VSRC1", 6)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 5), ("SADDR", 0x7D)], None),
    ]);
    for (tag, threads, tid_comp) in [("2d-8x4", [8, 4, 1], 2), ("partial-40", [40, 1, 1], 1), ("small-16", [16, 1, 1], 1)] {
        let cs = CsInterface { threads, user_sgprs: 2, tgid: [true, false, false], tg_size: false, tid_components: tid_comp };
        let total = threads.iter().product::<u32>() as usize;
        let sg = [0x100u32, total as u32];
        let mem = vec![0u32; (0x100 + 2 * total * 4) / 4];
        for (name, w64, split) in configs(true) {
            let dxil = translate_compute(&simple_cs, KernelOptions { wave64: w64, split, mem_windows: 1, sampler_base: 0 }, &cs).unwrap_or_else(|e| panic!("{tag} {name}: {e}"));
            let hlsl = hlsl_compile(&simple_cs, &Options { mode: Mode::Compute(cs.clone()), wave64: w64, split_wave64: split, ..Default::default() }, "cs_6_6");
            assert_eq!(runner::dispatch(&dxil.dxil, [2, 1, 1], &sg, &mem).unwrap(), runner::dispatch(&hlsl, [2, 1, 1], &sg, &mem).unwrap(), "{tag} {name}");
        }
    }
}

#[test]
fn direct_dxil_graphics_interp_discard_compr_and_mixed_linking() {
    if !gpu_ready() {
        return;
    }
    let vs_code = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 1), ("VSRC1", 2), ("VSRC2", 3), ("VSRC3", 4), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
        ("EXP", "ENC_EXP", &[("TGT", 33), ("EN", 0x3), ("VSRC0", 0), ("VSRC1", 12), ("VSRC2", 0), ("VSRC3", 0)], None),
    ]);
    let vsi = VsInterface {
        inputs: vec![sig("POSITION", 0, 0, 3, 0, 3), sig("COLOR", 0, 0, 3, 1, 0xF), sig("ID", 0, 0, 1, 2, 1)],
        outputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF), sig("COLOR", 0, 0, 3, 1, 0xF), sig("TEXCOORD", 0, 0, 1, 2, 0x3)],
        attr_base: 4,
        params: None,
    };
    // PS input_ena = 0x1302: I/J in v0..1, POS_X in v2, FRONT_FACE in v4; covers f16 interp,
    // v_interp_mov_f32, SV_IsFrontFace, conditional DONE=1 discard, and COMPR=1 f16 export.
    let ps_code = prog(&[
        ("V_INTERP_P1LL_F16", "ENC_VOP3", &[("VDST", 10), ("SRC1", 257), ("SRC0", 0)], None),
        ("V_INTERP_P2_F16", "ENC_VOP3", &[("VDST", 12), ("SRC1", 257), ("SRC0", 0), ("SRC2", V + 10)], None),
        ("V_INTERP_P1LL_F16", "ENC_VOP3", &[("VDST", 11), ("SRC1", 257), ("SRC0", 1 << 6)], None),
        ("V_INTERP_P2_F16", "ENC_VOP3", &[("VDST", 12), ("SRC1", 257), ("SRC0", 1 << 6), ("SRC2", V + 11), ("OP_SEL", 8)], None),
        ("V_INTERP_MOV_F32", "ENC_VINTRP", &[("VDST", 13), ("VSRC", 2), ("ATTR", 1), ("ATTRCHAN", 1)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 13), ("SRC0", V + 13)], None),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 13), ("SRC0", V + 13), ("VSRC1", 4)], None),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 13), ("SRC0", 240), ("VSRC1", 13)], None),
        ("V_CVT_PKRTZ_F16_F32", "ENC_VOP2", &[("VDST", 14), ("SRC0", V + 13), ("VSRC1", 4)], None),
        ("V_CMPX_GT_F32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 2)], Some(6.0f32.to_bits())),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 12), ("VSRC1", 14), ("COMPR", 1), ("DONE", 1), ("VM", 1)], None),
    ]);
    let psi = PsInterface {
        inputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF), sig("COLOR", 0, 0, 3, 1, 0xF), sig("TEXCOORD", 0, 0, 1, 2, 0x3)],
        outputs: vec![sig("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x1302,
        attrs: None,
    };
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 8 },
        runner::VertexElement { semantic: "ID", index: 0, format: DXGI_FORMAT_R32_UINT, offset: 24 },
    ];
    let verts: Vec<f32> = [[-1.0f32, -1.0], [-1.0, 3.0], [3.0, -1.0]]
        .iter().flat_map(|p| [p[0], p[1], 0.25, 0.5, 0.0, 1.0, f32::from_bits(1)]).collect();
    let clear = [0.0f32, 0.0, 0.75, 1.0];
    let gw32 = runner::graphics_wave32();
    let hvs = hlsl_compile(&vs_code, &Options { mode: Mode::Vertex(vsi.clone()), wave64: true, split_wave64: gw32, ..Default::default() }, "vs_6_6");
    let hps = hlsl_compile(&ps_code, &Options { mode: Mode::Pixel(psi.clone()), wave64: true, split_wave64: gw32, ..Default::default() }, "ps_6_6");
    let want = runner::draw(&hvs, &hps, &layout, &verts, 28, &[], &[], 12, clear).unwrap();
    assert_eq!(want[2], [0.25, 0.5, 0.5, 1.0]);
    assert_eq!(want[8], clear);

    for wave64 in [true, false] {
        let ko = KernelOptions { wave64, split: false, mem_windows: 1, sampler_base: 0 };
        let dvs = translate_vertex(&vs_code, ko, &vsi).unwrap().dxil;
        let dps = translate_pixel(&ps_code, ko, &psi).unwrap().dxil;
        assert_eq!(runner::draw(&dvs, &dps, &layout, &verts, 28, &[], &[], 12, clear).unwrap(), want, "direct+direct w64={wave64}");
        assert_eq!(runner::draw(&dvs, &hps, &layout, &verts, 28, &[], &[], 12, clear).unwrap(), want, "directVS+hlslPS w64={wave64}");
        assert_eq!(runner::draw(&hvs, &dps, &layout, &verts, 28, &[], &[], 12, clear).unwrap(), want, "hlslVS+directPS w64={wave64}");
    }
}

#[test]
fn direct_dxil_mesh_two_passes() {
    if !gpu_ready() || !mesh_ready() {
        return;
    }
    let ms = MsInterface { outputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF), sig("TEXCOORD", 0, 0, 3, 1, 0xF)], threads: 128, max_vertices: 128, max_primitives: 128 };
    let code = prog(&[
        ("S_BFE_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x40018)),
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 193), ("SRC1", V)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 0), ("SRC1", 134), ("SRC2", V)], None),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("S_LSHR_B32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 135)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some(1 << 12 | 3)),
        ("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 9)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 131), ("VSRC1", 0)], None),
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 20), ("SRC0", 1)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 20), ("VSRC1", 10)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 14), ("SRC0", V + 20)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 15), ("SRC0", V)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 16), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 17), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 10), ("VSRC1", 11), ("VSRC2", 12), ("VSRC3", 13), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 14), ("VSRC1", 15), ("VSRC2", 16), ("VSRC3", 17)], None),
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)], None),
        ("V_CMPX_EQ_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 0)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 21), ("SRC0", 255)], Some(1 << 10 | 2 << 20)),
        ("EXP", "ENC_EXP", &[("TGT", 20), ("EN", 1), ("VSRC0", 21), ("VSRC1", 0), ("VSRC2", 0), ("VSRC3", 0), ("DONE", 1)], None),
    ]);
    let msb = dxc::compile(&mesh_passthrough(&ms), "main", "ms_6_6", &[]).unwrap();
    let ps = dxc::compile("struct V { float4 pos : SV_Position; float4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return v.p0; }\n", "main", "ps_6_6", &[]).unwrap();
    let mut mem = vec![0u32; 256];
    for (i, (x, y)) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (0.0, -1.0)].iter().enumerate() {
        mem[0x40 + 4 * i..0x40 + 4 * i + 4].copy_from_slice(&[x.to_bits(), y.to_bits(), 0.0f32.to_bits(), 1.0f32.to_bits()]);
    }
    let mut sg = vec![0u32; 128];
    sg[16..20].copy_from_slice(&[0x100, 16 << 16, 3, 0]);
    for (name, _, split) in configs(false) {
        let dcs = translate_mesh(&code, KernelOptions { wave64: true, split, mem_windows: 1, sampler_base: 0 }, &ms).unwrap_or_else(|e| panic!("mesh {name}: {e}"));
        let hcs = hlsl_compile(&code, &Options { mode: Mode::Mesh(ms.clone()), wave64: true, split_wave64: split, ..Default::default() }, "cs_6_6");
        let (dpx, drec) = runner::mesh_draw(&dcs.dxil, &msb, &ps, 2, ms.record_dwords(), &sg, &mem, 32, [-1.0; 4]).unwrap();
        let (hpx, hrec) = runner::mesh_draw(&hcs, &msb, &ps, 2, ms.record_dwords(), &sg, &mem, 32, [-1.0; 4]).unwrap();
        assert_eq!(dpx, hpx, "mesh {name} pixels");
        assert_eq!(drec[..MESH_HEADER], hrec[..MESH_HEADER], "mesh {name} header");
    }
}

#[test]
fn direct_dxil_rejects_heap_windows_and_reports_unsupported_op() {
    let nop = prog(&[("S_NOP", "ENC_SOPP", &[("SIMM16", 0)], None)]);
    let heap_opt = KernelOptions { wave64: true, split: false, mem_windows: 1 | HEAP_WINDOWS, sampler_base: 0 };
    let cs = CsInterface { threads: [64, 1, 1], user_sgprs: 0, tgid: [false; 3], tg_size: false, tid_components: 1 };
    let vs = VsInterface { inputs: vec![], outputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF)], attr_base: 4, params: None };
    let ps = PsInterface { inputs: vec![], outputs: vec![sig("SV_Target", 0, 64, 3, 0, 0xF)], input_ena: 0, attrs: None };
    let ms = MsInterface { outputs: vec![sig("SV_Position", 0, 1, 3, 0, 0xF)], threads: 64, max_vertices: 64, max_primitives: 64 };
    for (kind, err) in [
        ("kernel", translate_full(&nop, heap_opt).err()),
        ("compute", translate_compute(&nop, heap_opt, &cs).err()),
        ("vertex", translate_vertex(&nop, heap_opt, &vs).err()),
        ("pixel", translate_pixel(&nop, heap_opt, &ps).err()),
        ("mesh", translate_mesh(&nop, heap_opt, &ms).err()),
    ] {
        assert!(err.unwrap_or_else(|| panic!("{kind}: expected refusal")).contains("heap memory windows"), "{kind}");
    }
    let getreg = prog(&[("S_GETREG_B32", "ENC_SOPK", &[("SDST", 0), ("SIMM16", 0)], None)]);
    let err = translate_full(&getreg, KernelOptions { wave64: true, split: false, mem_windows: 1, sampler_base: 0 }).err().expect("S_GETREG_B32 refused");
    assert!(err.to_ascii_lowercase().contains("s_getreg"), "{err}");
}
