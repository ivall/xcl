from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_rdna2 import generate, DEFAULT_OUT

SCRIPT = Path(__file__).resolve().parent / "gen_rdna2.py"

SYNTHETIC_XML = """<?xml version="1.0" encoding="utf-8"?>
<Root>
  <Document>
    <Copyright>Copyright (c) 2026 Advanced Micro Devices, Inc.</Copyright>
    <License>MIT</License>
    <ReleaseDate>2026-02-20</ReleaseDate>
  </Document>
  <ISA>
    <Architecture>
      <ArchitectureName>AMD RDNA 2</ArchitectureName>
    </Architecture>
    <Encodings>
      <Encoding Order="0">
        <EncodingName>ENC_SOP1</EncodingName>
        <BitCount>32</BitCount>
        <EncodingIdentifierMask>11111111100000000000000000000000</EncodingIdentifierMask>
        <EncodingIdentifiers>
          <EncodingIdentifier>10111110100000000000000000000000</EncodingIdentifier>
        </EncodingIdentifiers>
        <MicrocodeFormat>
          <BitMap>
            <Field>
              <FieldName>ENCODING</FieldName>
              <BitLayout>
                <Range Order="0">
                  <BitOffset>23</BitOffset>
                  <BitCount>9</BitCount>
                </Range>
              </BitLayout>
            </Field>
            <Field>
              <FieldName>OP</FieldName>
              <BitLayout>
                <Range Order="0">
                  <BitOffset>8</BitOffset>
                  <BitCount>8</BitCount>
                </Range>
              </BitLayout>
            </Field>
            <Field>
              <FieldName>SDST</FieldName>
              <BitLayout>
                <Range Order="0">
                  <BitOffset>16</BitOffset>
                  <BitCount>7</BitCount>
                </Range>
              </BitLayout>
            </Field>
            <Field>
              <FieldName>SSRC0</FieldName>
              <BitLayout>
                <Range Order="0">
                  <BitOffset>0</BitOffset>
                  <BitCount>8</BitCount>
                </Range>
              </BitLayout>
            </Field>
          </BitMap>
        </MicrocodeFormat>
      </Encoding>
      <Encoding Order="1">
        <EncodingName>ENC_VOP1</EncodingName>
        <BitCount>32</BitCount>
        <EncodingIdentifierMask>11111110000000000000000000000000</EncodingIdentifierMask>
        <EncodingIdentifiers>
          <EncodingIdentifier>01111110000000000000000000000000</EncodingIdentifier>
        </EncodingIdentifiers>
        <MicrocodeFormat>
          <BitMap>
            <Field>
              <FieldName>ENCODING</FieldName>
              <BitLayout>
                <Range Order="0">
                  <BitOffset>25</BitOffset>
                  <BitCount>7</BitCount>
                </Range>
              </BitLayout>
            </Field>
            <Field>
              <FieldName>OP</FieldName>
              <BitLayout>
                <Range Order="0">
                  <BitOffset>9</BitOffset>
                  <BitCount>8</BitCount>
                </Range>
              </BitLayout>
            </Field>
            <Field>
              <FieldName>VDST</FieldName>
              <BitLayout>
                <Range Order="0">
                  <BitOffset>17</BitOffset>
                  <BitCount>8</BitCount>
                </Range>
              </BitLayout>
            </Field>
            <Field>
              <FieldName>SRC0</FieldName>
              <BitLayout>
                <Range Order="0">
                  <BitOffset>0</BitOffset>
                  <BitCount>9</BitCount>
                </Range>
              </BitLayout>
            </Field>
          </BitMap>
        </MicrocodeFormat>
      </Encoding>
    </Encodings>
    <Instructions>
      <Instruction>
        <InstructionName>S_MOV_B32</InstructionName>
        <InstructionFlags>
          <IsBranch>FALSE</IsBranch>
          <IsConditionalBranch>FALSE</IsConditionalBranch>
          <IsIndirectBranch>FALSE</IsIndirectBranch>
          <IsProgramTerminator>FALSE</IsProgramTerminator>
          <IsImmediatelyExecuted>FALSE</IsImmediatelyExecuted>
        </InstructionFlags>
        <InstructionEncodings>
          <InstructionEncoding>
            <EncodingName>ENC_SOP1</EncodingName>
            <EncodingCondition>default</EncodingCondition>
            <Opcode>3</Opcode>
            <Operands>
              <Operand Order="0" Input="false" Output="true" IsImplicit="false">
                <FieldName>SDST</FieldName>
                <OperandType>SSRC</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
              <Operand Order="1" Input="true" Output="false" IsImplicit="false">
                <FieldName>SSRC0</FieldName>
                <OperandType>SSRC</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
            </Operands>
          </InstructionEncoding>
        </InstructionEncodings>
      </Instruction>
      <Instruction>
        <InstructionName>S_BRANCH</InstructionName>
        <InstructionFlags>
          <IsBranch>TRUE</IsBranch>
          <IsConditionalBranch>FALSE</IsConditionalBranch>
          <IsIndirectBranch>FALSE</IsIndirectBranch>
          <IsProgramTerminator>TRUE</IsProgramTerminator>
          <IsImmediatelyExecuted>FALSE</IsImmediatelyExecuted>
        </InstructionFlags>
        <InstructionEncodings>
          <InstructionEncoding>
            <EncodingName>ENC_SOP1</EncodingName>
            <EncodingCondition>default</EncodingCondition>
            <Opcode>1</Opcode>
            <Operands>
              <Operand Order="0" Input="true" Output="false" IsImplicit="false">
                <FieldName>SSRC0</FieldName>
                <OperandType>SIMM16</OperandType>
                <OperandSize>16</OperandSize>
              </Operand>
            </Operands>
          </InstructionEncoding>
        </InstructionEncodings>
      </Instruction>
      <Instruction>
        <InstructionName>V_FMAMK_F32</InstructionName>
        <InstructionFlags>
          <IsBranch>FALSE</IsBranch>
        </InstructionFlags>
        <InstructionEncodings>
          <InstructionEncoding>
            <EncodingName>VOP1_INST_LITERAL</EncodingName>
            <EncodingCondition>default</EncodingCondition>
            <Opcode>20</Opcode>
            <Operands>
              <Operand Order="0" Input="false" Output="true" IsImplicit="false">
                <FieldName>VDST</FieldName>
                <OperandType>VGPR</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
            </Operands>
          </InstructionEncoding>
        </InstructionEncodings>
      </Instruction>
    </Instructions>
  </ISA>
</Root>
"""


class GenRdna2CliTests(unittest.TestCase):
    def test_help_flag_prints_usage_and_exits_zero(self):
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                res = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), flag],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(res.returncode, 0)
                self.assertIn("Generate crates/xcl/src/gpu/rdna2_spec.rs", res.stdout)
                self.assertIn("scripts/gen_rdna2.py", res.stdout)

    def test_missing_args_prints_usage_and_exits_code_2(self):
        res = subprocess.run(
            [sys.executable, "-B", str(SCRIPT)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 2)
        self.assertIn("scripts/gen_rdna2.py", res.stdout)

    def test_nonexistent_file_exits_code_1_with_error(self):
        res = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "nonexistent_isa.xml"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("error: file not found", res.stderr)

    def test_invalid_xml_file_exits_code_1_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_file = Path(tmp) / "malformed.xml"
            bad_file.write_text("<Root><Unclosed>", encoding="utf-8")
            res = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(bad_file)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 1)
            self.assertIn("error: invalid XML", res.stderr)

    def test_missing_required_xml_elements_exits_code_1_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_file = Path(tmp) / "missing_isa.xml"
            missing_file.write_text("<Root><Document></Document></Root>", encoding="utf-8")
            res = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(missing_file)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 1)
            self.assertIn("missing required", res.stderr)

    def test_cli_custom_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml_path = Path(tmp) / "amdgpu_isa_rdna2.xml"
            xml_path.write_text(SYNTHETIC_XML, encoding="utf-8")
            out_file = Path(tmp) / "sub" / "rdna2_spec.rs"

            res = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(xml_path), str(out_file)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0)
            self.assertIn("AMD RDNA 2: 2 encodings (2 primary), 3 instructions", res.stdout)
            self.assertTrue(out_file.is_file())
            content = out_file.read_text(encoding="utf-8")
            self.assertIn('pub const ARCH: &str = "AMD RDNA 2";', content)
            self.assertIn("pub static ENCODINGS: &[Encoding] = &[", content)
            self.assertIn("pub static INSTRUCTIONS: &[Inst] = &[", content)
            self.assertIn("S_MOV_B32", content)
            self.assertIn("S_BRANCH", content)
            self.assertIn("V_FMAMK_F32", content)

    def test_default_out_path_points_to_rdna2_spec(self):
        norm = Path(DEFAULT_OUT).resolve()
        self.assertEqual(norm.name, "rdna2_spec.rs")
        self.assertEqual(norm.parent.name, "gpu")


class GenRdna2GenerateTests(unittest.TestCase):
    def test_generate_synthetic_spec(self):
        content, (arch, num_enc, num_prim, num_inst) = generate(SYNTHETIC_XML)
        self.assertEqual(arch, "AMD RDNA 2")
        self.assertEqual(num_enc, 2)
        self.assertEqual(num_prim, 2)
        self.assertEqual(num_inst, 3)
        self.assertIn("// @generated by scripts/gen_rdna2.py \u2013 do not edit by hand.", content)
        self.assertIn("use super::rdna2::{Encoding, Field, Inst, InstEnc, Operand};", content)
        self.assertIn('pub const ARCH: &str = "AMD RDNA 2";', content)

        # Check instruction flags:
        # S_BRANCH has IsBranch (bit 0 = 1) and IsProgramTerminator (bit 3 = 8) -> flags = 9
        self.assertIn('Inst { name: "S_BRANCH", flags: 9, always_literal: false, encs: &[InstEnc { encoding: 0, opcode: 1', content)
        # S_MOV_B32 has flags: 0
        self.assertIn('Inst { name: "S_MOV_B32", flags: 0, always_literal: false, encs: &[InstEnc { encoding: 0, opcode: 3', content)
        # V_FMAMK_F32 has always_literal: true mapped to base encoding ENC_VOP1 (index 1)
        self.assertIn('Inst { name: "V_FMAMK_F32", flags: 0, always_literal: true, encs: &[InstEnc { encoding: 1, opcode: 20', content)

    def test_generate_rejects_conflicting_identifiers(self):
        conflicting_xml = SYNTHETIC_XML.replace(
            "<EncodingIdentifier>10111110100000000000000000000000</EncodingIdentifier>",
            "<EncodingIdentifier>10111110100000000000000000000000</EncodingIdentifier>\n"
            "          <EncodingIdentifier>11111110100000000000000000000000</EncodingIdentifier>",
        )
        with self.assertRaises(ValueError) as cm:
            generate(conflicting_xml)
        self.assertIn("identifiers disagree outside OP", str(cm.exception))

    def test_generate_rejects_invalid_type(self):
        with self.assertRaises(TypeError):
            generate(12345)


if __name__ == "__main__":
    unittest.main()
