"""Generate crates/xcl/src/gpu/rdna2_spec.rs from AMD's Machine-Readable ISA spec.

Input:  amdgpu_isa_rdna2.xml from https://gpuopen.com/machine-readable-isa/ (MIT licence)
Usage:  python -I scripts/gen_rdna2.py <path/to/amdgpu_isa_rdna2.xml> [crates/xcl/src/gpu/rdna2_spec.rs]

Only the "primary" encodings (condition "default") are emitted with operands; the
literal / SDWA / DPP / NSA variants are derived at decode time from operand values,
exactly as the hardware does. Instructions that *always* carry a literal (e.g.
V_FMAMK_F32) are flagged.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "crates", "xcl", "src", "gpu", "rdna2_spec.rs"
)


def bits_of(text):
    text = text.strip()
    return int(text, 2), len(text)


def rust_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def generate(xml_input):
    """Parse AMD ISA XML and return (rust_source_string, stats_tuple).

    `xml_input` may be a string, bytes, or an `xml.etree.ElementTree.Element`.
    `stats_tuple` is `(arch_name, num_encodings, num_primary_used, num_instructions)`.
    """
    if isinstance(xml_input, (str, bytes)):
        root = ET.fromstring(xml_input)
    elif hasattr(xml_input, "getroot"):
        root = xml_input.getroot()
    elif isinstance(xml_input, ET.Element):
        root = xml_input
    else:
        raise TypeError(f"expected str, bytes, ElementTree, or Element, got {type(xml_input).__name__}")

    doc = root.find("Document")
    isa = root.find("ISA")
    if doc is None or isa is None:
        raise ValueError("missing required 'Document' or 'ISA' element in XML root")

    arch_elem = isa.find("Architecture")
    if arch_elem is None:
        raise ValueError("missing required 'Architecture' element in 'ISA'")
    arch = arch_elem.findtext("ArchitectureName") or ""

    enc_elem = isa.find("Encodings")
    if enc_elem is None:
        raise ValueError("missing required 'Encodings' element in 'ISA'")

    inst_elem = isa.find("Instructions")
    if inst_elem is None:
        raise ValueError("missing required 'Instructions' element in 'ISA'")

    encodings = []  # (name, bits, order, fixed_mask, fixed_value, fields)
    enc_index = {}
    for e in enc_elem:
        name = e.findtext("EncodingName")
        nbits = int(e.findtext("BitCount"))
        order = int(e.get("Order", "0"))
        mask, _ = bits_of(e.find("EncodingIdentifierMask").text)
        ids = [bits_of(i.text)[0] for i in e.find("EncodingIdentifiers")]
        fields = []
        op_bits = 0
        microcode = e.find("MicrocodeFormat")
        if microcode is not None and microcode.find("BitMap") is not None:
            for f in microcode.find("BitMap").iter("Field"):
                fname = f.findtext("FieldName")
                ranges = []
                layout = f.find("BitLayout")
                if layout is not None:
                    for r in sorted(layout.findall("Range"), key=lambda r: int(r.get("Order", "0"))):
                        off = int(r.findtext("BitOffset"))
                        cnt = int(r.findtext("BitCount"))
                        ranges.append((off, cnt))
                        if fname == "OP":
                            op_bits |= ((1 << cnt) - 1) << off
                fields.append((fname, ranges))
        # identification bits always live in the first 64 bits (NSA masks are longer)
        fixed_mask = mask & ~op_bits & ((1 << 64) - 1)
        values = {i & fixed_mask for i in ids}
        if len(values) != 1:
            raise ValueError(f"{name}: identifiers disagree outside OP: {sorted(values)}")
        encodings.append((name, nbits, order, fixed_mask, values.pop(), fields))
        enc_index[name] = len(encodings) - 1

    insts = []
    primary_used = set()
    for i in inst_elem:
        name = i.findtext("InstructionName")
        fl = i.find("InstructionFlags")
        flag = 0
        if fl is not None:
            for bit, tag in enumerate(["IsBranch", "IsConditionalBranch", "IsIndirectBranch", "IsProgramTerminator", "IsImmediatelyExecuted"]):
                if (fl.findtext(tag) or "FALSE").upper() == "TRUE":
                    flag |= 1 << bit
        encs = []
        always_lit = False
        inst_encs = i.find("InstructionEncodings")
        if inst_encs is not None:
            for ie in inst_encs:
                ename = ie.findtext("EncodingName")
                cond = ie.findtext("EncodingCondition") or ""
                primary = ename.startswith("ENC_") or ename == "VOP3_SDST_ENC"
                if "LITERAL" in ename and cond == "default":
                    # e.g. V_FMAMK_F32 / S_SETREG_IMM32_B32: the literal is part of the
                    # instruction, so the "literal" encoding is its only form – map it onto
                    # the base encoding and flag the extra dword.
                    always_lit = True
                    base = "ENC_" + ename.replace("_INST_LITERAL", "")
                    if base in enc_index:
                        ename, primary = base, True
                if not primary or not cond.startswith(("default", "Nothas")):
                    continue
                ops = []
                operands = ie.find("Operands")
                if operands is not None:
                    for o in sorted(operands, key=lambda o: int(o.get("Order", "0"))):
                        ops.append((
                            o.findtext("FieldName"),
                            o.findtext("OperandType") or "",
                            int(o.findtext("OperandSize") or "0"),
                            o.get("Input") == "true",
                            o.get("Output") == "true",
                            o.get("IsImplicit") == "true",
                        ))
                encs.append((enc_index[ename], int(ie.findtext("Opcode")), ops))
                primary_used.add(ename)
        if encs:
            insts.append((name, flag, always_lit, encs))

    out = []
    w = out.append
    w("// @generated by scripts/gen_rdna2.py – do not edit by hand.")
    w(f"// Source: AMD Machine-Readable GPU ISA Specification, {arch}")
    w(f"// {doc.findtext('Copyright')} Licence: {doc.findtext('License')}. Release {doc.findtext('ReleaseDate')}.")
    w("#![allow(clippy::all)]")
    w("")
    w("use super::rdna2::{Encoding, Field, Inst, InstEnc, Operand};")
    w("")
    w(f"pub const ARCH: &str = {rust_str(arch)};")
    w("")
    w("pub static ENCODINGS: &[Encoding] = &[")
    for (name, nbits, order, fmask, fval, fields) in encodings:
        fs = ", ".join(
            "Field { name: %s, ranges: &[%s] }" % (rust_str(fn), ", ".join(f"({o}, {c})" for o, c in rs))
            for fn, rs in fields
        )
        prim = "true" if name in primary_used else "false"
        w(f"    Encoding {{ name: {rust_str(name)}, bits: {nbits}, order: {order}, primary: {prim}, fixed_mask: {fmask:#x}, fixed_value: {fval:#x}, fields: &[{fs}] }},")
    w("];")
    w("")
    w("pub static INSTRUCTIONS: &[Inst] = &[")
    for (name, flag, always_lit, encs) in insts:
        es = []
        for (ei, opc, ops) in encs:
            os_ = ", ".join(
                "Operand { field: %s, ty: %s, size: %d, input: %s, output: %s, implicit: %s }" % (
                    ("Some(%s)" % rust_str(f)) if f else "None", rust_str(t), sz,
                    str(inp).lower(), str(outp).lower(), str(imp).lower())
                for (f, t, sz, inp, outp, imp) in ops
            )
            es.append(f"InstEnc {{ encoding: {ei}, opcode: {opc}, operands: &[{os_}] }}")
        w(f"    Inst {{ name: {rust_str(name)}, flags: {flag}, always_literal: {str(always_lit).lower()}, encs: &[{', '.join(es)}] }},")
    w("];")
    return "\n".join(out) + "\n", (arch, len(encodings), len(primary_used), len(insts))


def main(argv=None, dst=None):
    if isinstance(argv, str) and dst is not None:
        args = [argv, dst]
    elif isinstance(argv, str):
        args = [argv]
    elif argv is not None:
        args = list(argv)
    else:
        args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if (args and args[0] in ("-h", "--help")) else 2

    xml_path = args[0]
    if not os.path.isfile(xml_path):
        print(f"error: file not found: '{xml_path}'", file=sys.stderr)
        return 1

    out_path = args[1] if len(args) > 1 else DEFAULT_OUT

    try:
        with io.open(xml_path, "r", encoding="utf-8") as f:
            src_text = f.read()
        content, (arch, num_enc, num_prim, num_inst) = generate(src_text)
    except ET.ParseError as e:
        print(f"error: invalid XML in '{xml_path}': {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"error: failed to process '{xml_path}': {e}", file=sys.stderr)
        return 1

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

    print(f"{arch}: {num_enc} encodings ({num_prim} primary), {num_inst} instructions -> {os.path.normpath(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
