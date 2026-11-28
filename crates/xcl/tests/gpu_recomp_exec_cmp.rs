//! GPU verification tests for RDNA2 integer/64-bit comparisons, `V_CMPX_*` EXEC-modifying
//! comparisons, `S_*_SAVEEXEC_*` / `S_*_WREXEC_*` wave-mask operations, scalar mask conditional
//! branches (`S_CBRANCH_{VCCZ,VCCNZ,EXECZ,EXECNZ,SCC0,SCC1}`), and `V_CNDMASK_B32` with
//! non-VCC SGPR masks:
//! - Integer `V_CMP_*` (`I32`, `U32`, `I16`, `U16`, `I64`, `U64`) across all ISA predicates
//!   (`F`, `LT`, `EQ`, `LE`, `GT`, `NE`, `GE`, `T`), both `ENC_VOPC` (`VCC`) and `ENC_VOP3`
//!   (`SDST` / `NULL`), divergent `EXEC` masking, `wave32` high-SGPR preservation, and
//!   ignored upper 16 bits on 16-bit integer comparisons
//! - `V_CMPX_*` (`ENC_VOPC` and `ENC_VOP3`) across integer (`I32`, `U32`, `I16`, `U16`, `I64`,
//!   `U64`) and float (`F32`, `F16`, `CLASS_F32`, `CLASS_F16`) forms, verifying `EXEC_after ==
//!   EXEC_before & cond`, `VDST` non-interference in `ENC_VOP3`, and post-`V_CMPX` lane masking
//! - All 24 `S_*_SAVEEXEC_*` / `S_*_WREXEC_*` instructions (`_B64` on 64-lane waves, `_B32` on
//!   both 32- and 64-lane waves), verifying `EXEC`, `SDST` (old vs new `EXEC`), `SCC` including
//!   upper-half-only nonzero masks, and `_B32` upper-half preservation
//! - Scalar conditional branches (`S_CBRANCH_VCCZ/VCCNZ/EXECZ/EXECNZ/SCC0/SCC1`) including
//!   upper-half-only 64-bit masks, divergent `V_CMPX` + `S_CBRANCH_EXECNZ` loops, and
//!   `V_CNDMASK_B32` (`ENC_VOP3`) with arbitrary SGPR masks

use std::collections::BTreeMap;

use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::runner::{self, KernelState};
use xcl::gpu::recomp::{dxc, translate, Options};

const V: u64 = 256; // VALU source encoding base for VGPRs
const S_NULL: u64 = 125;
const S_VCC_LO: u64 = 106;
const S_EXEC_LO: u64 = 126;

fn inline_u32(v: u32) -> u64 {
    assert!(v <= 64);
    128 + v as u64
}

fn inline_i32(v: i32) -> u64 {
    assert!((-16..=-1).contains(&v));
    (192 + (-v)) as u64
}

enum Step {
    Insn {
        name: &'static str,
        enc: &'static str,
        fields: Vec<(&'static str, u64)>,
        lit: Option<u32>,
    },
    Label(&'static str),
    Branch {
        name: &'static str,
        target: &'static str,
    },
}

struct ProgBuilder {
    steps: Vec<Step>,
}

impl ProgBuilder {
    fn new() -> Self {
        Self { steps: Vec::new() }
    }

    fn push(
        &mut self,
        name: &'static str,
        enc: &'static str,
        fields: &[(&'static str, u64)],
        lit: Option<u32>,
    ) {
        self.steps.push(Step::Insn {
            name,
            enc,
            fields: fields.to_vec(),
            lit,
        });
    }

    fn label(&mut self, name: &'static str) {
        self.steps.push(Step::Label(name));
    }

    fn branch(&mut self, name: &'static str, target: &'static str) {
        self.steps.push(Step::Branch { name, target });
    }

    fn s_mov_b32(&mut self, sd: u64, src: u64, lit: Option<u32>) {
        self.push(
            "S_MOV_B32",
            "ENC_SOP1",
            &[("SDST", sd), ("SSRC0", src)],
            lit,
        );
    }

    fn s_mov_b64(&mut self, sd: u64, src: u64) {
        self.push(
            "S_MOV_B64",
            "ENC_SOP1",
            &[("SDST", sd), ("SSRC0", src)],
            None,
        );
    }

    fn s_cmp_eq_u32(&mut self, s0: u64, s1: u64, lit: Option<u32>) {
        self.push(
            "S_CMP_EQ_U32",
            "ENC_SOPC",
            &[("SSRC0", s0), ("SSRC1", s1)],
            lit,
        );
    }

    fn s_cselect_b32(&mut self, sd: u64, s0: u64, s1: u64) {
        self.push(
            "S_CSELECT_B32",
            "ENC_SOP2",
            &[("SDST", sd), ("SSRC0", s0), ("SSRC1", s1)],
            None,
        );
    }

    fn v_mov_b32(&mut self, vd: u64, src: u64, lit: Option<u32>) {
        self.push("V_MOV_B32", "ENC_VOP1", &[("VDST", vd), ("SRC0", src)], lit);
    }

    fn v_add_nc_u32(&mut self, vd: u64, s0: u64, vs1: u64, lit: Option<u32>) {
        self.push(
            "V_ADD_NC_U32",
            "ENC_VOP2",
            &[("VDST", vd), ("SRC0", s0), ("VSRC1", vs1)],
            lit,
        );
    }

    fn vopc(&mut self, name: &'static str, src0: u64, vsrc1: u64, lit: Option<u32>) {
        self.push(name, "ENC_VOPC", &[("SRC0", src0), ("VSRC1", vsrc1)], lit);
    }

    fn vop3(&mut self, name: &'static str, vd: u64, src0: u64, src1: u64) {
        self.push(
            name,
            "ENC_VOP3",
            &[("VDST", vd), ("SRC0", src0), ("SRC1", src1)],
            None,
        );
    }

    fn build(self) -> Vec<u8> {
        let mut labels = BTreeMap::new();
        let mut byte_off = 0usize;
        let mut encoded_offsets = Vec::with_capacity(self.steps.len());

        for step in &self.steps {
            encoded_offsets.push(byte_off);
            match step {
                Step::Label(name) => {
                    assert!(
                        labels.insert(*name, byte_off).is_none(),
                        "duplicate label {name}"
                    );
                }
                Step::Insn {
                    name,
                    enc,
                    fields,
                    lit,
                } => {
                    let bytes = encode(name, enc, fields, *lit)
                        .unwrap_or_else(|e| panic!("{name} ({enc}): {e}"));
                    byte_off += bytes.len();
                }
                Step::Branch { name, .. } => {
                    let bytes = encode(name, "ENC_SOPP", &[("SIMM16", 0)], None)
                        .unwrap_or_else(|e| panic!("{name}: {e}"));
                    byte_off += bytes.len();
                }
            }
        }

        let mut code = Vec::with_capacity(byte_off + 4);
        for (step, &pc) in self.steps.iter().zip(&encoded_offsets) {
            match step {
                Step::Label(_) => {}
                Step::Insn {
                    name,
                    enc,
                    fields,
                    lit,
                } => {
                    code.extend(encode(name, enc, fields, *lit).unwrap());
                }
                Step::Branch { name, target } => {
                    let target_pc = *labels
                        .get(target)
                        .unwrap_or_else(|| panic!("unknown label {target}"));
                    let delta_bytes = target_pc as isize - (pc as isize + 4);
                    assert_eq!(delta_bytes % 4, 0);
                    let simm16 =
                        i16::try_from(delta_bytes / 4).expect("branch offset out of SIMM16 range");
                    code.extend(
                        encode(name, "ENC_SOPP", &[("SIMM16", simm16 as u16 as u64)], None)
                            .unwrap(),
                    );
                }
            }
        }
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        code
    }
}

fn opt_mode(wave64: bool, split_wave64: bool, hw_wave: u32) -> Options {
    Options {
        wave64,
        split_wave64,
        hw_wave,
        ..Default::default()
    }
}

fn all_wave_options() -> [(&'static str, Options); 4] {
    [
        ("wave64", opt_mode(true, false, 0)),
        ("split-wave64", opt_mode(true, true, 0)),
        ("wave32", opt_mode(false, false, 0)),
        ("wave32@hw64", opt_mode(false, false, 64)),
    ]
}

fn gpu_ready() -> bool {
    match (
        runner::adapter(),
        runner::wave64_programs_supported(),
        dxc::location(),
    ) {
        (Ok(_), Ok(true), Ok(_)) => true,
        (a, w, d) => {
            println!(
                "GPU/DXC unavailable – skipping ({:?} / {:?} / {:?})",
                a.as_ref().err(),
                w.as_ref().ok(),
                d.as_ref().err()
            );
            let reason = if a.is_err() || w.is_err() {
                "no GPU"
            } else if d.is_err() {
                "no DXC"
            } else {
                "no wave64"
            };
            runner::skip_test(reason);
            false
        }
    }
}

fn wave_configs() -> Vec<(&'static str, Options)> {
    if !gpu_ready() {
        return Vec::new();
    }
    let hw64 = runner::wave64_supported() == Ok(true);
    let hw32 = runner::graphics_wave32();
    let mut cfgs = Vec::new();
    if hw64 {
        cfgs.push(("wave64", opt_mode(true, false, 0)));
    }
    if hw32 {
        cfgs.push(("split-wave64", opt_mode(true, true, 0)));
        cfgs.push(("wave32", opt_mode(false, false, 0)));
    }
    if hw64 {
        cfgs.push(("wave32@hw64", opt_mode(false, false, 64)));
    }
    cfgs
}

fn console_lanes_for(opt: &Options) -> usize {
    if opt.wave64 {
        64
    } else {
        32
    }
}

fn hw_lanes_for(opt: &Options) -> usize {
    if opt.wave64 || opt.hw_wave == 64 {
        64
    } else {
        32
    }
}

fn wave_mask(opt: &Options, full_mask: u64) -> u64 {
    if opt.wave64 {
        full_mask
    } else {
        full_mask & (u32::MAX as u64)
    }
}

fn compile_kernel(code: &[u8], opt: &Options) -> (String, Vec<u8>) {
    let t = translate(code, opt).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    let dxil =
        dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    (t.hlsl, dxil)
}

fn run_kernel(dxil: &[u8], st: &KernelState) -> runner::KernelResult {
    runner::run(dxil, st, &[]).unwrap()
}

fn read_sgpr64(sgpr: &[u32], base: usize) -> u64 {
    sgpr[base] as u64 | ((sgpr[base + 1] as u64) << 32)
}

fn sgpr_mask(opt: &Options, sgpr: &[u32], base: usize) -> u64 {
    wave_mask(opt, read_sgpr64(sgpr, base))
}

fn write_sgpr64(sgpr: &mut [u32], base: usize, val: u64) {
    sgpr[base] = val as u32;
    sgpr[base + 1] = (val >> 32) as u32;
}

fn assert_masked_vgpr(
    opt: &Options,
    vgpr: &[Vec<u32>],
    mask: u64,
    vreg: usize,
    vals: (u32, u32),
    ctx: &str,
) {
    let hw_lanes = hw_lanes_for(opt);
    let console_lanes = console_lanes_for(opt);
    for (l, row) in vgpr.iter().enumerate().take(hw_lanes) {
        let want = if (mask >> (l % console_lanes)) & 1 != 0 {
            vals.0
        } else {
            vals.1
        };
        assert_eq!(row[vreg], want, "{ctx} lane {l}");
    }
}

fn ord_cmp<T: Ord>(op: &str, a: T, b: T) -> bool {
    match op {
        "F" => false,
        "LT" => a < b,
        "EQ" => a == b,
        "LE" => a <= b,
        "GT" => a > b,
        "NE" => a != b,
        "GE" => a >= b,
        "T" => true,
        _ => panic!("unknown cmp op {op}"),
    }
}

fn float_cmp_eval(op: &str, a: f64, b: f64) -> bool {
    let ord = !a.is_nan() && !b.is_nan();
    match op {
        "F" => false,
        "LT" => a < b,
        "EQ" => a == b,
        "LE" => a <= b,
        "GT" => a > b,
        "LG" => ord && a != b,
        "GE" => a >= b,
        "O" => ord,
        "U" => !ord,
        "NGE" => !ord || a < b,
        "NLG" => !(ord && a != b),
        "NGT" => !ord || a <= b,
        "NLE" => !ord || a > b,
        "NEQ" => !(a == b),
        "NLT" => !ord || a >= b,
        "TRU" => true,
        _ => panic!("unknown float cmp op {op}"),
    }
}

fn f16_to_f64(h: u16) -> f64 {
    let s = ((h >> 15) & 1) as u64;
    let e = ((h >> 10) & 0x1F) as u64;
    let m = (h & 0x3FF) as u64;
    if e == 0x1F {
        if m == 0 {
            if s != 0 {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        } else {
            f64::from_bits((s << 63) | (0x7FFu64 << 52) | (m << 42))
        }
    } else if e == 0 {
        let sign = if s != 0 { -1.0 } else { 1.0 };
        sign * (m as f64) * 2.0f64.powi(-24)
    } else {
        f64::from_bits((s << 63) | ((e + 1023 - 15) << 52) | (m << 42))
    }
}

fn fclass_bit(neg: bool, exp: u32, mant: u32, exp_max: u32, qnan_mask: u32) -> u32 {
    if exp == exp_max {
        match (mant != 0, (mant & qnan_mask) != 0, neg) {
            (true, true, _) => 1,
            (true, false, _) => 0,
            (false, _, true) => 2,
            (false, _, false) => 9,
        }
    } else if exp == 0 {
        match (mant == 0, neg) {
            (true, true) => 5,
            (true, false) => 6,
            (false, true) => 4,
            (false, false) => 7,
        }
    } else if neg {
        3
    } else {
        8
    }
}

fn fclass32_bit(bits: u32) -> u32 {
    fclass_bit(
        (bits >> 31) != 0,
        (bits >> 23) & 0xFF,
        bits & 0x7F_FFFF,
        0xFF,
        0x40_0000,
    )
}

fn fclass16_bit(bits: u16) -> u32 {
    fclass_bit(
        (bits >> 15) != 0,
        ((bits >> 10) & 0x1F) as u32,
        (bits & 0x3FF) as u32,
        0x1F,
        0x200,
    )
}

fn eval_saveexec(op: &str, src: u64, exec: u64) -> u64 {
    match op {
        "AND" => src & exec,
        "OR" => src | exec,
        "XOR" => src ^ exec,
        "ANDN2" => src & !exec,
        "ORN2" => src | !exec,
        "NAND" => !(src & exec),
        "NOR" => !(src | exec),
        "XNOR" => !(src ^ exec),
        "ANDN1" => !src & exec,
        "ORN1" => !src | exec,
        _ => panic!("unknown saveexec op {op}"),
    }
}

fn leak_name(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

#[test]
fn exec_cmp_translate_and_compile_with_dxc() {
    if let Err(e) = dxc::location() {
        println!("DXC unavailable – skipping ({e})");
        runner::skip_test("no DXC");
        return;
    }

    // Program 1: lane-local vs whole-live masks in split-wave64, integer VOPC/VOP3, V_CMPX,
    // and V_CNDMASK_B32 with non-VCC SGPR mask.
    let mut b1 = ProgBuilder::new();
    // s[20:21] only read per-lane by V_CNDMASK_B32 -> uses xcl_x_ballot_local in split-wave64
    b1.vop3("V_CMP_LT_I32", 20, V, V + 1);
    b1.push(
        "V_CNDMASK_B32",
        "ENC_VOP3",
        &[("VDST", 2), ("SRC0", V + 2), ("SRC1", V + 3), ("SRC2", 20)],
        None,
    );
    // V_CMP_EQ_U64 read whole by S_CBRANCH_VCCZ -> must use full exchange xcl_x_ballot
    b1.vopc("V_CMP_EQ_U64", V + 4, 6, None);
    b1.branch("S_CBRANCH_VCCZ", "skip_vcc");
    b1.v_mov_b32(8, inline_u32(11), None);
    b1.label("skip_vcc");
    b1.vop3("V_CMPX_GE_I16", 22, V, V + 1);
    b1.branch("S_CBRANCH_EXECZ", "done");
    b1.v_add_nc_u32(8, inline_u32(1), 8, None);
    b1.label("done");
    let prog1 = b1.build();

    for (name, opt) in all_wave_options() {
        let (hlsl, dxil) = compile_kernel(&prog1, &opt);
        assert!(!dxil.is_empty(), "{name} DXIL empty");
        if name == "split-wave64" {
            assert!(
                hlsl.contains("xcl_x_ballot_local("),
                "lane-local V_CNDMASK mask should omit exchange:\n{hlsl}"
            );
            assert!(
                hlsl.contains("xcl_x_ballot("),
                "whole-live V_CMP/V_CMPX mask should exchange:\n{hlsl}"
            );
        }
    }

    // Program 2: all 12 _B32 SAVEEXEC/WREXEC ops + all 12 _B64 ops.
    let ops = [
        "AND", "OR", "XOR", "ANDN2", "ORN2", "NAND", "NOR", "XNOR", "ANDN1", "ORN1",
    ];
    let mut b32 = ProgBuilder::new();
    for (i, op) in ops.iter().enumerate() {
        b32.push(
            leak_name(format!("S_{op}_SAVEEXEC_B32")),
            "ENC_SOP1",
            &[("SDST", 10 + i as u64), ("SSRC0", 2)],
            None,
        );
    }
    b32.push(
        "S_ANDN1_WREXEC_B32",
        "ENC_SOP1",
        &[("SDST", 20), ("SSRC0", 2)],
        None,
    );
    b32.push(
        "S_ANDN2_WREXEC_B32",
        "ENC_SOP1",
        &[("SDST", 21), ("SSRC0", 2)],
        None,
    );
    let prog32 = b32.build();

    let mut b64 = ProgBuilder::new();
    for (i, op) in ops.iter().enumerate() {
        b64.push(
            leak_name(format!("S_{op}_SAVEEXEC_B64")),
            "ENC_SOP1",
            &[("SDST", 10 + 2 * i as u64), ("SSRC0", 2)],
            None,
        );
    }
    b64.push(
        "S_ANDN1_WREXEC_B64",
        "ENC_SOP1",
        &[("SDST", 30), ("SSRC0", 2)],
        None,
    );
    b64.push(
        "S_ANDN2_WREXEC_B64",
        "ENC_SOP1",
        &[("SDST", 32), ("SSRC0", 2)],
        None,
    );
    let prog64 = b64.build();

    for (name, opt) in all_wave_options() {
        let (_, dxil32) = compile_kernel(&prog32, &opt);
        assert!(!dxil32.is_empty(), "{name} _B32 DXIL empty");
        if opt.wave64 {
            let (_, dxil64) = compile_kernel(&prog64, &opt);
            assert!(!dxil64.is_empty(), "{name} _B64 DXIL empty");
        }
    }
}

#[test]
fn gpu_v_cmp_integer_and_64bit_all_predicates() {
    let cfgs = wave_configs();
    if cfgs.is_empty() {
        return;
    }

    const INT_OPS: [&str; 8] = ["F", "LT", "EQ", "LE", "GT", "NE", "GE", "T"];
    const INT16_OPS: [&str; 6] = ["LT", "EQ", "LE", "GT", "NE", "GE"];

    // Program A: 32-bit I32 & U32 (VOPC -> VCC, VOP3 -> s[dst:dst+1], NULL dst=125, inline/SGPR/literal)
    let mut pa = ProgBuilder::new();
    pa.s_mov_b64(90, S_EXEC_LO);
    pa.s_mov_b64(S_EXEC_LO, 0);
    for (i, op) in INT_OPS.iter().enumerate() {
        pa.vopc(leak_name(format!("V_CMP_{op}_I32")), V, 1, None);
        pa.s_mov_b64(10 + 2 * i as u64, S_VCC_LO);
        pa.vop3(
            leak_name(format!("V_CMP_{op}_U32")),
            30 + 2 * i as u64,
            V,
            V + 1,
        );
    }
    pa.vop3("V_CMP_LT_I32", 50, V, V + 1);
    pa.vopc("V_CMP_GT_U32", V, 1, None);
    pa.s_mov_b64(52, S_VCC_LO);
    // NULL dest: must leave s[50:51] and VCC untouched
    pa.vop3("V_CMP_NE_I32", S_NULL, V, V + 1);
    pa.s_mov_b64(54, S_VCC_LO);
    // Inline constant -1, SGPR s2, and 32-bit literal sources
    pa.vopc("V_CMP_EQ_I32", inline_i32(-1), 0, None);
    pa.s_mov_b64(56, S_VCC_LO);
    pa.vop3("V_CMP_LT_I32", 58, 2, V);
    pa.vopc("V_CMP_GE_U32", 255, 0, Some(0x8000_0000));
    pa.s_mov_b64(60, S_VCC_LO);
    // Keep odd SGPR halves in the shader's SGPR set so wave32 preservation is observable
    for i in 0..INT_OPS.len() {
        let odd = 31 + 2 * i as u64;
        pa.s_mov_b32(odd, odd, None);
    }
    pa.s_mov_b32(51, 51, None);
    pa.s_mov_b32(59, 59, None);
    pa.s_mov_b64(S_EXEC_LO, 90);
    let prog_a = pa.build();

    // Program B: 16-bit I16 & U16 + 64-bit I64 & U64
    let mut pb = ProgBuilder::new();
    pb.s_mov_b64(90, S_EXEC_LO);
    pb.s_mov_b64(S_EXEC_LO, 0);
    for (i, op) in INT16_OPS.iter().enumerate() {
        pb.vopc(leak_name(format!("V_CMP_{op}_I16")), V, 1, None);
        pb.s_mov_b64(10 + 2 * i as u64, S_VCC_LO);
        pb.vop3(
            leak_name(format!("V_CMP_{op}_U16")),
            24 + 2 * i as u64,
            V,
            V + 1,
        );
    }
    for (i, op) in INT_OPS.iter().enumerate() {
        pb.vopc(leak_name(format!("V_CMP_{op}_I64")), V + 2, 4, None);
        pb.s_mov_b64(40 + 2 * i as u64, S_VCC_LO);
        pb.vop3(
            leak_name(format!("V_CMP_{op}_U64")),
            56 + 2 * i as u64,
            V + 2,
            V + 4,
        );
    }
    pb.vop3("V_CMP_GT_I64", 72, 4, V + 2);
    pb.vopc("V_CMP_EQ_U64", inline_i32(-1), 2, None);
    pb.s_mov_b64(74, S_VCC_LO);
    for i in 0..INT16_OPS.len() {
        let odd = 25 + 2 * i as u64;
        pb.s_mov_b32(odd, odd, None);
    }
    for i in 0..INT_OPS.len() {
        let odd = 57 + 2 * i as u64;
        pb.s_mov_b32(odd, odd, None);
    }
    pb.s_mov_b32(73, 73, None);
    pb.s_mov_b64(S_EXEC_LO, 90);
    let prog_b = pb.build();

    const EXEC_MASK: u64 = 0xA55A_C33C_F00F_1235;
    let pairs32: [(u32, u32); 16] = [
        (0, 0),
        (0, 1),
        (1, 0),
        (u32::MAX, 0),
        (u32::MAX, 1),
        (u32::MAX, u32::MAX),
        (i32::MIN as u32, i32::MAX as u32),
        (i32::MAX as u32, i32::MIN as u32),
        (i32::MIN as u32, i32::MIN as u32),
        (0x7FFF_FFFF, 0x8000_0000),
        (0x8000_0000, 0x7FFF_FFFF),
        (0x8000_0000, 0x8000_0001),
        (0xFFFF_FFFF, 0xFFFF_FFFE),
        (0x1234_5678, 0x1234_5678),
        (0x0000_FFFF, 0xFFFF_0000),
        (0x0000_8000, 0x0000_7FFF),
    ];
    let pairs16: [(u16, u16); 16] = [
        (0, 0),
        (0, 1),
        (1, 0),
        (0xFFFF, 0),
        (0xFFFF, 1),
        (0xFFFF, 0xFFFF),
        (0x8000, 0x7FFF),
        (0x7FFF, 0x8000),
        (0x8000, 0x8000),
        (0x8000, 0x8001),
        (0xFFFE, 0xFFFF),
        (0x1234, 0x1234),
        (0x7FFF, 0x7FFF),
        (0xFF00, 0x00FF),
        (0x0080, 0xFF80),
        (0x0001, 0xFFFE),
    ];
    let pairs64: [(u64, u64); 16] = [
        (0, 0),
        (0, 1),
        (1, 0),
        (u64::MAX, 0),
        (u64::MAX, 1),
        (u64::MAX, u64::MAX),
        (i64::MIN as u64, i64::MAX as u64),
        (i64::MAX as u64, i64::MIN as u64),
        (0x0000_0001_0000_0000, 0x0000_0001_0000_0001),
        (0x0000_0001_FFFF_FFFF, 0x0000_0002_0000_0000),
        (0xFFFF_FFFF_0000_0000, 0x0000_0000_FFFF_FFFF),
        (0x8000_0000_0000_0000, 0x8000_0000_0000_0001),
        (0x7FFF_FFFF_FFFF_FFFF, 0x7FFF_FFFF_FFFF_FFFE),
        (0x1234_5678_9ABC_DEF0, 0x1234_5678_9ABC_DEF0),
        (0x0000_0000_8000_0000, 0x0000_0000_7FFF_FFFF),
        (0xFFFF_FFFF_8000_0000, 0xFFFF_FFFF_7FFF_FFFF),
    ];

    for (cfg_name, opt) in &cfgs {
        let (_, dxil_a) = compile_kernel(&prog_a, opt);
        let (_, dxil_b) = compile_kernel(&prog_b, opt);
        let lanes = console_lanes_for(opt);
        let wmask = wave_mask(opt, EXEC_MASK);
        const SENTINEL: u32 = 0xDEAD_BEEF;

        // Run Program A (I32 / U32)
        let mut st_a = KernelState::default();
        write_sgpr64(&mut st_a.sgpr, 0, EXEC_MASK);
        st_a.sgpr[2] = (-5i32) as u32;
        for i in 0..INT_OPS.len() {
            st_a.sgpr[31 + 2 * i] = SENTINEL;
        }
        st_a.sgpr[51] = SENTINEL;
        st_a.sgpr[59] = SENTINEL;
        for l in 0..64 {
            let (a, b) = pairs32[l % pairs32.len()];
            st_a.vgpr[l][0] = a;
            st_a.vgpr[l][1] = b;
        }
        let ra = run_kernel(&dxil_a, &st_a);

        for (i, op) in INT_OPS.iter().enumerate() {
            let got_i32 = sgpr_mask(opt, &ra.sgpr, 10 + 2 * i);
            let got_u32 = sgpr_mask(opt, &ra.sgpr, 30 + 2 * i);
            if !opt.wave64 {
                assert_eq!(
                    ra.sgpr[31 + 2 * i],
                    SENTINEL,
                    "{cfg_name}: wave32 clobbered s{}",
                    31 + 2 * i
                );
            }
            let mut want_i32 = 0u64;
            let mut want_u32 = 0u64;
            for l in 0..lanes {
                if (wmask >> l) & 1 == 0 {
                    continue;
                }
                let (a, b) = pairs32[l % pairs32.len()];
                if ord_cmp(op, a as i32, b as i32) {
                    want_i32 |= 1u64 << l;
                }
                if ord_cmp(op, a, b) {
                    want_u32 |= 1u64 << l;
                }
            }
            assert_eq!(got_i32, want_i32, "{cfg_name} V_CMP_{op}_I32 VOPC");
            assert_eq!(got_u32, want_u32, "{cfg_name} V_CMP_{op}_U32 VOP3");
        }
        let got_50 = sgpr_mask(opt, &ra.sgpr, 50);
        let got_52 = sgpr_mask(opt, &ra.sgpr, 52);
        let got_54 = sgpr_mask(opt, &ra.sgpr, 54);
        assert_eq!(
            got_50,
            sgpr_mask(opt, &ra.sgpr, 12),
            "{cfg_name} V_CMP_LT_I32 VOP3 vs VOPC"
        );
        assert_eq!(
            got_52,
            sgpr_mask(opt, &ra.sgpr, 38),
            "{cfg_name} V_CMP_GT_U32 VOPC vs VOP3"
        );
        assert_eq!(
            got_54, got_52,
            "{cfg_name} V_CMP_NE_I32 SDST=NULL must preserve VCC"
        );
        if !opt.wave64 {
            assert_eq!(ra.sgpr[51], SENTINEL, "{cfg_name}: wave32 clobbered s51");
            assert_eq!(ra.sgpr[59], SENTINEL, "{cfg_name}: wave32 clobbered s59");
        }

        let mut want_eq_m1 = 0u64;
        let mut want_lt_s2 = 0u64;
        let mut want_ge_lit = 0u64;
        for l in 0..lanes {
            if (wmask >> l) & 1 == 0 {
                continue;
            }
            let a = pairs32[l % pairs32.len()].0;
            if a as i32 == -1 {
                want_eq_m1 |= 1u64 << l;
            }
            if -5i32 < a as i32 {
                want_lt_s2 |= 1u64 << l;
            }
            if 0x8000_0000u32 >= a {
                want_ge_lit |= 1u64 << l;
            }
        }
        assert_eq!(
            sgpr_mask(opt, &ra.sgpr, 56),
            want_eq_m1,
            "{cfg_name} inline -1"
        );
        assert_eq!(
            sgpr_mask(opt, &ra.sgpr, 58),
            want_lt_s2,
            "{cfg_name} SGPR s2 < v0"
        );
        assert_eq!(
            sgpr_mask(opt, &ra.sgpr, 60),
            want_ge_lit,
            "{cfg_name} literal 0x80000000 >= v0"
        );

        // Run Program B (I16 / U16 / I64 / U64)
        let mut st_b = KernelState::default();
        write_sgpr64(&mut st_b.sgpr, 0, EXEC_MASK);
        let sgpr_i64_ref: u64 = 0xFFFF_FFFF_8000_0000;
        write_sgpr64(&mut st_b.sgpr, 4, sgpr_i64_ref);
        for i in 0..INT16_OPS.len() {
            st_b.sgpr[25 + 2 * i] = SENTINEL;
        }
        for i in 0..INT_OPS.len() {
            st_b.sgpr[57 + 2 * i] = SENTINEL;
        }
        st_b.sgpr[73] = SENTINEL;
        for (l, row) in st_b.vgpr.iter_mut().enumerate().take(64) {
            let (a16, b16) = pairs16[l % pairs16.len()];
            // Non-zero upper 16 bits that must be ignored by 16-bit integer comparisons
            let hi_a = (((l as u32) * 0x1357 + 0xA5A5) ^ 0xFFFF) << 16;
            let hi_b = (((l as u32) * 0x2468 + 0x5A5A) & 0xFFFF) << 16;
            row[0] = (a16 as u32) | hi_a;
            row[1] = (b16 as u32) | hi_b;
            let (a64, b64) = pairs64[l % pairs64.len()];
            row[2] = a64 as u32;
            row[3] = (a64 >> 32) as u32;
            row[4] = b64 as u32;
            row[5] = (b64 >> 32) as u32;
        }
        let rb = run_kernel(&dxil_b, &st_b);

        for (i, op) in INT16_OPS.iter().enumerate() {
            let got_i16 = sgpr_mask(opt, &rb.sgpr, 10 + 2 * i);
            let got_u16 = sgpr_mask(opt, &rb.sgpr, 24 + 2 * i);
            if !opt.wave64 {
                assert_eq!(
                    rb.sgpr[25 + 2 * i],
                    SENTINEL,
                    "{cfg_name}: wave32 clobbered s{}",
                    25 + 2 * i
                );
            }
            let mut want_i16 = 0u64;
            let mut want_u16 = 0u64;
            for l in 0..lanes {
                if (wmask >> l) & 1 == 0 {
                    continue;
                }
                let (a, b) = pairs16[l % pairs16.len()];
                if ord_cmp(op, a as i16, b as i16) {
                    want_i16 |= 1u64 << l;
                }
                if ord_cmp(op, a, b) {
                    want_u16 |= 1u64 << l;
                }
            }
            assert_eq!(got_i16, want_i16, "{cfg_name} V_CMP_{op}_I16 VOPC");
            assert_eq!(got_u16, want_u16, "{cfg_name} V_CMP_{op}_U16 VOP3");
        }

        for (i, op) in INT_OPS.iter().enumerate() {
            let got_i64 = sgpr_mask(opt, &rb.sgpr, 40 + 2 * i);
            let got_u64 = sgpr_mask(opt, &rb.sgpr, 56 + 2 * i);
            if !opt.wave64 {
                assert_eq!(
                    rb.sgpr[57 + 2 * i],
                    SENTINEL,
                    "{cfg_name}: wave32 clobbered s{}",
                    57 + 2 * i
                );
            }
            let mut want_i64 = 0u64;
            let mut want_u64 = 0u64;
            for l in 0..lanes {
                if (wmask >> l) & 1 == 0 {
                    continue;
                }
                let (a, b) = pairs64[l % pairs64.len()];
                if ord_cmp(op, a as i64, b as i64) {
                    want_i64 |= 1u64 << l;
                }
                if ord_cmp(op, a, b) {
                    want_u64 |= 1u64 << l;
                }
            }
            assert_eq!(got_i64, want_i64, "{cfg_name} V_CMP_{op}_I64 VOPC");
            assert_eq!(got_u64, want_u64, "{cfg_name} V_CMP_{op}_U64 VOP3");
        }
        if !opt.wave64 {
            assert_eq!(rb.sgpr[73], SENTINEL, "{cfg_name}: wave32 clobbered s73");
        }
        let mut want_gt_s4 = 0u64;
        let mut want_eq_m1_64 = 0u64;
        for l in 0..lanes {
            if (wmask >> l) & 1 == 0 {
                continue;
            }
            let a64 = pairs64[l % pairs64.len()].0;
            if (sgpr_i64_ref as i64) > (a64 as i64) {
                want_gt_s4 |= 1u64 << l;
            }
            if u64::MAX == a64 {
                want_eq_m1_64 |= 1u64 << l;
            }
        }
        assert_eq!(
            sgpr_mask(opt, &rb.sgpr, 72),
            want_gt_s4,
            "{cfg_name} V_CMP_GT_I64 SGPR pair"
        );
        assert_eq!(
            sgpr_mask(opt, &rb.sgpr, 74),
            want_eq_m1_64,
            "{cfg_name} V_CMP_EQ_U64 inline -1"
        );
    }
}

#[test]
fn gpu_v_cmpx_exec_masking_and_encodings() {
    let cfgs = wave_configs();
    if cfgs.is_empty() {
        return;
    }

    #[derive(Clone, Copy)]
    enum Kind {
        I32(&'static str),
        U32(&'static str),
        I16(&'static str),
        U16(&'static str),
        I64(&'static str),
        U64(&'static str),
        F32(&'static str),
        F16(&'static str),
        ClassF32,
        ClassF16,
    }

    let cases: [(&str, &str, Kind); 18] = [
        ("V_CMPX_LT_I32", "ENC_VOPC", Kind::I32("LT")),
        ("V_CMPX_GE_U32", "ENC_VOP3", Kind::U32("GE")),
        ("V_CMPX_T_I32", "ENC_VOPC", Kind::I32("T")),
        ("V_CMPX_F_U32", "ENC_VOP3", Kind::U32("F")),
        ("V_CMPX_LT_I16", "ENC_VOPC", Kind::I16("LT")),
        ("V_CMPX_EQ_U16", "ENC_VOP3", Kind::U16("EQ")),
        ("V_CMPX_GT_I64", "ENC_VOPC", Kind::I64("GT")),
        ("V_CMPX_LE_U64", "ENC_VOP3", Kind::U64("LE")),
        ("V_CMPX_T_I64", "ENC_VOPC", Kind::I64("T")),
        ("V_CMPX_LT_F32", "ENC_VOPC", Kind::F32("LT")),
        ("V_CMPX_LG_F32", "ENC_VOP3", Kind::F32("LG")),
        ("V_CMPX_O_F32", "ENC_VOPC", Kind::F32("O")),
        ("V_CMPX_U_F32", "ENC_VOP3", Kind::F32("U")),
        ("V_CMPX_NLT_F16", "ENC_VOPC", Kind::F16("NLT")),
        ("V_CMPX_EQ_F16", "ENC_VOP3", Kind::F16("EQ")),
        ("V_CMPX_TRU_F16", "ENC_VOPC", Kind::F16("TRU")),
        ("V_CMPX_CLASS_F32", "ENC_VOPC", Kind::ClassF32),
        ("V_CMPX_CLASS_F16", "ENC_VOP3", Kind::ClassF16),
    ];

    // Inputs sit in v0..v10; per-case outputs sit in v16..v33 and s30..s65.
    // ENC_VOP3 V_CMPX_* passes non-EXEC VDST=20 to verify s[20:21] stays untouched.
    let mut pb = ProgBuilder::new();
    pb.s_mov_b64(90, S_EXEC_LO);
    for (i, &(name, enc, kind)) in cases.iter().enumerate() {
        pb.s_mov_b64(S_EXEC_LO, 90);
        let dst_v = 16 + i as u64;
        pb.v_mov_b32(dst_v, 255, Some(0xDEAD_C0DE));
        pb.s_mov_b64(S_EXEC_LO, 0);
        let (s0, s1) = match kind {
            Kind::I32(_) | Kind::U32(_) | Kind::I16(_) | Kind::U16(_) => (0, 1),
            Kind::I64(_) | Kind::U64(_) => (2, 4),
            Kind::F32(_) => (6, 7),
            Kind::F16(_) => (8, 9),
            Kind::ClassF32 => (6, 10),
            Kind::ClassF16 => (8, 10),
        };
        if enc == "ENC_VOP3" {
            pb.vop3(name, 20, V + s0, V + s1);
        } else {
            pb.vopc(name, V + s0, s1, None);
        }
        pb.s_mov_b64(30 + 2 * i as u64, S_EXEC_LO);
        pb.v_mov_b32(dst_v, inline_u32(i as u32 + 1), None);
    }
    pb.s_mov_b32(20, 20, None);
    pb.s_mov_b32(21, 21, None);
    pb.s_mov_b64(S_EXEC_LO, 90);
    let prog = pb.build();

    const EXEC_INIT: u64 = 0xDB6D_B6DB_7777_EEEE;
    const SENTINEL_S20: u32 = 0xCAFE_BABE;
    const SENTINEL_S21: u32 = 0xFEED_FACE;

    let pairs32: [(u32, u32); 8] = [
        (0, 0),
        (i32::MIN as u32, 0),
        (i32::MAX as u32, i32::MIN as u32),
        (0xFFFF_FFFF, 1),
        (0x8000_0000, 0x7FFF_FFFF),
        (42, 42),
        (0x0000_8000, 0xFFFF_8000),
        (7, 100),
    ];
    let pairs64: [(u64, u64); 8] = [
        (0, 0),
        (i64::MIN as u64, i64::MAX as u64),
        (0x0000_0001_0000_0000, 0x0000_0000_FFFF_FFFF),
        (u64::MAX, u64::MAX),
        (0x8000_0000_0000_0000, 0),
        (0x1234_5678_9ABC_DEF0, 0x1234_5678_9ABC_DEF1),
        (100, 50),
        (0xFFFF_FFFF_FFFF_FFFE, 0xFFFF_FFFF_FFFF_FFFF),
    ];
    let pairs_f32: [(u32, u32); 8] = [
        (0.0f32.to_bits(), (-0.0f32).to_bits()),
        ((-1.5f32).to_bits(), 2.25f32.to_bits()),
        (f32::INFINITY.to_bits(), f32::INFINITY.to_bits()),
        (f32::NEG_INFINITY.to_bits(), (-1.0f32).to_bits()),
        (0x7FC0_0001, 1.0f32.to_bits()),
        (0x7F80_0001, 0x7FC0_0001),
        (3.5f32.to_bits(), 3.5f32.to_bits()),
        (0x8000_0000, 0x0000_0000),
    ];
    let pairs_f16: [(u16, u16); 8] = [
        (0x0000, 0x8000),
        (0xBC00, 0x4000),
        (0x7C00, 0x7C00),
        (0xFC00, 0xBC00),
        (0x7E01, 0x3C00),
        (0x7C01, 0x7E01),
        (0x4200, 0x4200),
        (0x0001, 0x0000),
    ];
    let class_masks: [u32; 8] = [
        (1 << 5) | (1 << 6), // ±0
        (1 << 3) | (1 << 8), // ±normal
        (1 << 2) | (1 << 9), // ±inf
        (1 << 0) | (1 << 1), // sNaN | qNaN
        1 << 0,              // sNaN
        1 << 1,              // qNaN
        (1 << 4) | (1 << 7), // ±denorm
        0x3FF,               // all classes
    ];

    for (cfg_name, opt) in &cfgs {
        let (_, dxil) = compile_kernel(&prog, opt);
        let console_lanes = console_lanes_for(opt);
        let wmask = wave_mask(opt, EXEC_INIT);

        let mut st = KernelState::default();
        write_sgpr64(&mut st.sgpr, 0, EXEC_INIT);
        st.sgpr[20] = SENTINEL_S20;
        st.sgpr[21] = SENTINEL_S21;
        for (l, row) in st.vgpr.iter_mut().enumerate().take(64) {
            let idx = l % 8;
            row[0] = pairs32[idx].0;
            row[1] = pairs32[idx].1;
            row[2] = pairs64[idx].0 as u32;
            row[3] = (pairs64[idx].0 >> 32) as u32;
            row[4] = pairs64[idx].1 as u32;
            row[5] = (pairs64[idx].1 >> 32) as u32;
            row[6] = pairs_f32[idx].0;
            row[7] = pairs_f32[idx].1;
            row[8] = (pairs_f16[idx].0 as u32) | (((l as u32) * 0x1111 + 0xABCD) << 16);
            row[9] = (pairs_f16[idx].1 as u32) | (((l as u32) * 0x2222 + 0x1234) << 16);
            row[10] = class_masks[idx];
        }
        let r = run_kernel(&dxil, &st);
        assert_eq!(
            r.sgpr[20], SENTINEL_S20,
            "{cfg_name}: VOP3 V_CMPX clobbered s20"
        );
        assert_eq!(
            r.sgpr[21], SENTINEL_S21,
            "{cfg_name}: VOP3 V_CMPX clobbered s21"
        );

        for (i, &(name, enc, kind)) in cases.iter().enumerate() {
            let mut want_exec = 0u64;
            for l in 0..console_lanes {
                if (wmask >> l) & 1 == 0 {
                    continue;
                }
                let idx = l % 8;
                let pass = match kind {
                    Kind::I32(op) => ord_cmp(op, pairs32[idx].0 as i32, pairs32[idx].1 as i32),
                    Kind::U32(op) => ord_cmp(op, pairs32[idx].0, pairs32[idx].1),
                    Kind::I16(op) => ord_cmp(
                        op,
                        pairs32[idx].0 as u16 as i16,
                        pairs32[idx].1 as u16 as i16,
                    ),
                    Kind::U16(op) => ord_cmp(
                        op,
                        (pairs32[idx].0 & 0xFFFF) as u16,
                        (pairs32[idx].1 & 0xFFFF) as u16,
                    ),
                    Kind::I64(op) => ord_cmp(op, pairs64[idx].0 as i64, pairs64[idx].1 as i64),
                    Kind::U64(op) => ord_cmp(op, pairs64[idx].0, pairs64[idx].1),
                    Kind::F32(op) => float_cmp_eval(
                        op,
                        f32::from_bits(pairs_f32[idx].0) as f64,
                        f32::from_bits(pairs_f32[idx].1) as f64,
                    ),
                    Kind::F16(op) => float_cmp_eval(
                        op,
                        f16_to_f64(pairs_f16[idx].0),
                        f16_to_f64(pairs_f16[idx].1),
                    ),
                    Kind::ClassF32 => {
                        ((class_masks[idx] >> fclass32_bit(pairs_f32[idx].0)) & 1) != 0
                    }
                    Kind::ClassF16 => {
                        ((class_masks[idx] >> fclass16_bit(pairs_f16[idx].0)) & 1) != 0
                    }
                };
                if pass {
                    want_exec |= 1u64 << l;
                }
            }
            let got_exec = sgpr_mask(opt, &r.sgpr, 30 + 2 * i);
            assert_eq!(got_exec, want_exec, "{cfg_name} {name} ({enc}) EXEC");
            assert_masked_vgpr(
                opt,
                &r.vgpr,
                want_exec,
                16 + i,
                (i as u32 + 1, 0xDEAD_C0DE),
                name,
            );
        }
    }
}

#[test]
fn gpu_saveexec_and_wrexec_all_forms() {
    let cfgs = wave_configs();
    if cfgs.is_empty() {
        return;
    }

    const ALL_OPS: [(&str, bool); 12] = [
        ("AND", false),
        ("OR", false),
        ("XOR", false),
        ("ANDN2", false),
        ("ORN2", false),
        ("NAND", false),
        ("NOR", false),
        ("XNOR", false),
        ("ANDN1", false),
        ("ORN1", false),
        ("ANDN1", true),
        ("ANDN2", true),
    ];

    fn build_saveexec_prog(wide: bool) -> Vec<u8> {
        let mut pb = ProgBuilder::new();
        let sfx = if wide { "B64" } else { "B32" };
        pb.s_mov_b64(90, S_EXEC_LO);
        for (i, &(op, is_wrexec)) in ALL_OPS.iter().enumerate() {
            let kind = if is_wrexec { "WREXEC" } else { "SAVEEXEC" };
            let ins = leak_name(format!("S_{op}_{kind}_{sfx}"));
            let sdst = 10 + 2 * i as u64;
            pb.s_mov_b64(S_EXEC_LO, 90);
            pb.v_mov_b32(10 + i as u64, 255, Some(0x55AA_0000 | i as u32));
            pb.s_mov_b64(S_EXEC_LO, 0);
            pb.push(ins, "ENC_SOP1", &[("SDST", sdst), ("SSRC0", 2)], None);
            pb.s_cselect_b32(40 + i as u64, inline_u32(1), inline_u32(0));
            pb.s_mov_b64(60 + 2 * i as u64, S_EXEC_LO);
            pb.v_mov_b32(10 + i as u64, inline_u32(0x10 + i as u32), None);
        }
        pb.s_mov_b64(S_EXEC_LO, 90);
        if !wide {
            for i in 0..ALL_OPS.len() {
                let odd = 11 + 2 * i as u64;
                pb.s_mov_b32(odd, odd, None);
            }
        }
        pb.build()
    }

    let prog32 = build_saveexec_prog(false);
    let prog64 = build_saveexec_prog(true);

    // Pairs (init_exec, src): cover all-zero, upper-half-only nonzero, lower-half-only nonzero,
    // and mixed bits across both halves.
    let masks: [(u64, u64); 4] = [
        (0, 0),
        (0x8000_0001_0000_0000, 0x0000_0003_0000_0000),
        (0x0000_0000_0000_0005, 0x0000_0000_FFFF_FFFF),
        (0xA55A_000F_3C3C_1111, 0x5AA5_F000_0F0F_8888),
    ];
    const SENTINEL_HI: u32 = 0xA1B2_C3D4;

    for (cfg_name, opt) in &cfgs {
        let (_, dxil32) = compile_kernel(&prog32, opt);
        let dxil64 = opt.wave64.then(|| compile_kernel(&prog64, opt).1);

        for &(init_exec_raw, src_raw) in &masks {
            let init_exec = wave_mask(opt, init_exec_raw);
            let src = wave_mask(opt, src_raw);

            // Check all 12 _B32 forms on every wave config
            let mut st32 = KernelState::default();
            write_sgpr64(&mut st32.sgpr, 0, init_exec);
            write_sgpr64(&mut st32.sgpr, 2, src);
            for i in 0..ALL_OPS.len() {
                st32.sgpr[11 + 2 * i] = SENTINEL_HI;
            }
            let r32 = run_kernel(&dxil32, &st32);

            for (i, &(op, is_wrexec)) in ALL_OPS.iter().enumerate() {
                let lo_res = eval_saveexec(op, src as u32 as u64, init_exec as u32 as u64) as u32;
                let want_exec = if opt.wave64 {
                    (init_exec & 0xFFFF_FFFF_0000_0000) | (lo_res as u64)
                } else {
                    lo_res as u64
                };
                let want_sdst = if is_wrexec { lo_res } else { init_exec as u32 };
                let want_scc = u32::from(lo_res != 0);

                assert_eq!(
                    r32.sgpr[10 + 2 * i],
                    want_sdst,
                    "{cfg_name} S_{op}_B32 SDST"
                );
                assert_eq!(
                    r32.sgpr[11 + 2 * i],
                    SENTINEL_HI,
                    "{cfg_name} _B32 clobbered s{}",
                    11 + 2 * i
                );
                assert_eq!(
                    r32.sgpr[40 + i],
                    want_scc,
                    "{cfg_name} S_{op}_B32 SCC (init={init_exec:#x}, src={src:#x})"
                );
                assert_eq!(
                    sgpr_mask(opt, &r32.sgpr, 60 + 2 * i),
                    want_exec,
                    "{cfg_name} S_{op}_B32 EXEC"
                );
                assert_masked_vgpr(
                    opt,
                    &r32.vgpr,
                    want_exec,
                    10 + i,
                    (0x10 + i as u32, 0x55AA_0000 | i as u32),
                    op,
                );
            }

            // Check all 12 _B64 forms on 64-lane wave configs (wave64 + split-wave64)
            if let Some(dxil64) = dxil64.as_deref() {
                let mut st64 = KernelState::default();
                write_sgpr64(&mut st64.sgpr, 0, init_exec);
                write_sgpr64(&mut st64.sgpr, 2, src);
                let r64 = run_kernel(dxil64, &st64);

                for (i, &(op, is_wrexec)) in ALL_OPS.iter().enumerate() {
                    let new_exec = eval_saveexec(op, src, init_exec);
                    let want_sdst = if is_wrexec { new_exec } else { init_exec };
                    let want_scc = u32::from(new_exec != 0);

                    assert_eq!(
                        read_sgpr64(&r64.sgpr, 10 + 2 * i),
                        want_sdst,
                        "{cfg_name} S_{op}_B64 SDST (init={init_exec:#x}, src={src:#x})"
                    );
                    assert_eq!(
                        r64.sgpr[40 + i],
                        want_scc,
                        "{cfg_name} S_{op}_B64 SCC (init={init_exec:#x}, src={src:#x})"
                    );
                    assert_eq!(
                        read_sgpr64(&r64.sgpr, 60 + 2 * i),
                        new_exec,
                        "{cfg_name} S_{op}_B64 EXEC"
                    );
                    assert_masked_vgpr(
                        opt,
                        &r64.vgpr,
                        new_exec,
                        10 + i,
                        (0x10 + i as u32, 0x55AA_0000 | i as u32),
                        op,
                    );
                }
            }
        }
    }
}

#[test]
fn gpu_cbranch_masks_and_v_cndmask_sgpr() {
    let cfgs = wave_configs();
    if cfgs.is_empty() {
        return;
    }

    // Program A: S_CBRANCH_VCCZ/VCCNZ/EXECZ/EXECNZ/SCC0/SCC1 + S_BRANCH
    // Inputs: s[0:1] = VCC, s[2:3] = EXEC_test, s4 = SCC_flag (0 or 1)
    // Outputs: s10..s15 = 1 (fallthrough) or 2 (branch taken)
    let mut pa = ProgBuilder::new();
    pa.s_mov_b64(90, S_EXEC_LO);
    pa.s_mov_b64(S_VCC_LO, 0);

    let cases_br: [(&str, &str, &str, u64); 4] = [
        ("S_CBRANCH_VCCZ", "vccz_taken", "vccz_done", 10),
        ("S_CBRANCH_VCCNZ", "vccnz_taken", "vccnz_done", 11),
        ("S_CBRANCH_EXECZ", "execz_taken", "execz_done", 12),
        ("S_CBRANCH_EXECNZ", "execnz_taken", "execnz_done", 13),
    ];
    for (idx, &(br, lbl_taken, lbl_done, dst)) in cases_br.iter().enumerate() {
        if idx == 2 {
            pa.s_mov_b64(S_EXEC_LO, 2);
        }
        pa.branch(br, lbl_taken);
        pa.s_mov_b32(dst, inline_u32(1), None);
        pa.branch("S_BRANCH", lbl_done);
        pa.label(lbl_taken);
        pa.s_mov_b32(dst, inline_u32(2), None);
        pa.label(lbl_done);
    }
    pa.s_mov_b64(S_EXEC_LO, 90);

    for (br, lbl_taken, lbl_done, dst) in [
        ("S_CBRANCH_SCC0", "scc0_taken", "scc0_done", 14u64),
        ("S_CBRANCH_SCC1", "scc1_taken", "scc1_done", 15u64),
    ] {
        pa.s_cmp_eq_u32(4, inline_u32(1), None);
        pa.branch(br, lbl_taken);
        pa.s_mov_b32(dst, inline_u32(1), None);
        pa.branch("S_BRANCH", lbl_done);
        pa.label(lbl_taken);
        pa.s_mov_b32(dst, inline_u32(2), None);
        pa.label(lbl_done);
    }
    let prog_a = pa.build();

    // Program B: divergent V_CMPX + S_CBRANCH_EXECNZ loop + V_CNDMASK_B32 (ENC_VOP3) with SGPR masks.
    // Inputs: s[0:1] = EXEC_MASK, s[26:27] = constant SGPR mask, v0 = (l & 7) + 1,
    // v1 = sentinel, v2/v3 = cmp operands, v4/v7 = sentinels, v5/v6 = cndmask false/true sources.
    let mut pb = ProgBuilder::new();
    pb.s_mov_b64(90, S_EXEC_LO);
    pb.s_mov_b64(S_EXEC_LO, 0);
    pb.v_mov_b32(1, inline_u32(0), None);
    pb.label("loop");
    pb.v_add_nc_u32(1, inline_u32(1), 1, None);
    pb.v_add_nc_u32(0, inline_i32(-1), 0, None);
    pb.push(
        "V_CMPX_LT_U32",
        "ENC_VOPC",
        &[("SRC0", inline_u32(0)), ("VSRC1", 0)],
        None,
    );
    pb.branch("S_CBRANCH_EXECNZ", "loop");
    // Restore EXEC to full, compute s[24:25] via V_CMP_GT_U32 (lane-local in split-wave64),
    // then apply s[0:1] for V_CNDMASK_B32 tests.
    pb.s_mov_b64(S_EXEC_LO, 90);
    pb.push(
        "V_CMP_GT_U32",
        "ENC_VOP3",
        &[("VDST", 24), ("SRC0", V + 2), ("SRC1", V + 3)],
        None,
    );
    pb.s_mov_b64(S_EXEC_LO, 0);
    pb.push(
        "V_CNDMASK_B32",
        "ENC_VOP3",
        &[("VDST", 4), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", 24)],
        None,
    );
    pb.push(
        "V_CNDMASK_B32",
        "ENC_VOP3",
        &[("VDST", 7), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", 26)],
        None,
    );
    pb.s_mov_b64(S_EXEC_LO, 90);
    let prog_b = pb.build();

    let branch_cases: [(u64, u64, u32); 4] = [
        (0, 0, 0),
        (0x0000_0000_0000_0001, 0x0000_0000_0000_0001, 1),
        (0x0001_0000_0000_0000, 0x0001_0000_0000_0000, 1),
        (u64::MAX, u64::MAX, 0),
    ];

    for (cfg_name, opt) in &cfgs {
        let (_, dxil_a) = compile_kernel(&prog_a, opt);
        for &(vcc_raw, ex_raw, scc_flag) in &branch_cases {
            let vcc = wave_mask(opt, vcc_raw);
            let ex = wave_mask(opt, ex_raw);
            let mut st = KernelState::default();
            write_sgpr64(&mut st.sgpr, 0, vcc);
            write_sgpr64(&mut st.sgpr, 2, ex);
            st.sgpr[4] = scc_flag;
            let r = run_kernel(&dxil_a, &st);

            assert_eq!(
                r.sgpr[10],
                if vcc == 0 { 2 } else { 1 },
                "{cfg_name} S_CBRANCH_VCCZ (vcc={vcc:#x})"
            );
            assert_eq!(
                r.sgpr[11],
                if vcc != 0 { 2 } else { 1 },
                "{cfg_name} S_CBRANCH_VCCNZ (vcc={vcc:#x})"
            );
            assert_eq!(
                r.sgpr[12],
                if ex == 0 { 2 } else { 1 },
                "{cfg_name} S_CBRANCH_EXECZ (exec={ex:#x})"
            );
            assert_eq!(
                r.sgpr[13],
                if ex != 0 { 2 } else { 1 },
                "{cfg_name} S_CBRANCH_EXECNZ (exec={ex:#x})"
            );
            assert_eq!(
                r.sgpr[14],
                if scc_flag == 0 { 2 } else { 1 },
                "{cfg_name} S_CBRANCH_SCC0 (flag={scc_flag})"
            );
            assert_eq!(
                r.sgpr[15],
                if scc_flag == 1 { 2 } else { 1 },
                "{cfg_name} S_CBRANCH_SCC1 (flag={scc_flag})"
            );
        }

        let (_, dxil_b) = compile_kernel(&prog_b, opt);
        let console_lanes = console_lanes_for(opt);
        let hw_lanes = hw_lanes_for(opt);
        const EXEC_B: u64 = 0x5A3C_96A5_1234_ABCD;
        const CONST_MASK: u64 = 0xA55A_3C3C_1234_5678;
        let wmask = wave_mask(opt, EXEC_B);

        let mut st_b = KernelState::default();
        write_sgpr64(&mut st_b.sgpr, 0, EXEC_B);
        write_sgpr64(&mut st_b.sgpr, 26, CONST_MASK);
        for (l, row) in st_b.vgpr.iter_mut().enumerate().take(64) {
            let lu = (l % console_lanes) as u32;
            row[0] = (lu & 7) + 1;
            row[1] = 0xCAFE_0000 | lu;
            row[2] = (lu.wrapping_mul(37)) % 50;
            row[3] = 24;
            row[4] = 0xBAD0_0000 | lu;
            row[5] = 0x1000_0000 | lu;
            row[6] = 0x2000_0000 | lu;
            row[7] = 0xBAD1_0000 | lu;
        }
        let rb = run_kernel(&dxil_b, &st_b);
        for (l, row) in rb.vgpr.iter().enumerate().take(hw_lanes) {
            let cl = l % console_lanes;
            let lu = cl as u32;
            let active = (wmask >> cl) & 1 != 0;
            if active {
                assert_eq!(row[0], 0, "{cfg_name} loop counter lane {l}");
                assert_eq!(row[1], (lu & 7) + 1, "{cfg_name} loop trip count lane {l}");
                let cond_local = ((lu.wrapping_mul(37)) % 50) > 24;
                assert_eq!(
                    row[4],
                    if cond_local {
                        0x2000_0000 | lu
                    } else {
                        0x1000_0000 | lu
                    },
                    "{cfg_name} V_CNDMASK lane-local SGPR lane {l}"
                );
                let cond_const = (CONST_MASK >> cl) & 1 != 0;
                assert_eq!(
                    row[7],
                    if cond_const {
                        0x2000_0000 | lu
                    } else {
                        0x1000_0000 | lu
                    },
                    "{cfg_name} V_CNDMASK constant SGPR lane {l}"
                );
            } else {
                assert_eq!(
                    row[0],
                    (lu & 7) + 1,
                    "{cfg_name} inactive lane {l} v0 modified"
                );
                assert_eq!(
                    row[1],
                    0xCAFE_0000 | lu,
                    "{cfg_name} inactive lane {l} v1 modified"
                );
                assert_eq!(
                    row[4],
                    0xBAD0_0000 | lu,
                    "{cfg_name} inactive lane {l} v4 modified"
                );
                assert_eq!(
                    row[7],
                    0xBAD1_0000 | lu,
                    "{cfg_name} inactive lane {l} v7 modified"
                );
            }
        }
    }
}
