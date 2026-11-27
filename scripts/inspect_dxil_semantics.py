"""Print stable DXIL container semantics using DXC's disassembler.

Usage:
    python scripts/inspect_dxil_semantics.py shader.dxil [other.dxil ...] [--dxc PATH]

The JSON intentionally omits hashes, resource names, SSA names, and instruction order so
DXC and XCL containers can be compared without harmless code-generation differences.
`disassembler_dxc_version` identifies the tool used for inspection, not the container producer.
Multiple-entry-point containers are rejected because this first version reports one entry.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys


STAGE_NAMES = {
    "vs": "vertex",
    "hs": "hull",
    "ds": "domain",
    "gs": "geometry",
    "ps": "pixel",
    "cs": "compute",
    "ms": "mesh",
    "as": "amplification",
    "lib": "library",
    "raygeneration": "raygeneration",
    "intersection": "intersection",
    "anyhit": "anyhit",
    "closesthit": "closesthit",
    "miss": "miss",
    "callable": "callable",
}
OVERLOADS = {"i1", "i8", "i16", "i32", "i64", "f16", "f32", "f64", "void"}


def find_dxc(explicit: str | None = None) -> Path:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"specified dxc.exe does not exist: {candidate}")

    dxc_dir = os.environ.get("XCL_DXC_DIR")
    if dxc_dir:
        candidate = Path(dxc_dir) / "dxc.exe"
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"XCL_DXC_DIR does not contain dxc.exe: {candidate}")

    candidates: list[Path] = []
    for name in ("dxc.exe", "dxc"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    for root in (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits/10/bin",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Windows Kits/10/bin",
    ):
        versions = [p for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
        versions.sort(key=lambda p: tuple(int(n) for n in re.findall(r"\d+", p.name)), reverse=True)
        candidates.extend(p / "x64/dxc.exe" for p in versions)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("dxc.exe not found; pass --dxc or set XCL_DXC_DIR")


def container_feature_flags(data: bytes) -> int:
    """Read the SFI0 feature-bit mask from a DXBC/DXIL container."""
    if len(data) < 32 or data[:4] != b"DXBC":
        raise ValueError("input is not a DXBC/DXIL container")
    total_size = struct.unpack_from("<I", data, 24)[0]
    chunk_count = struct.unpack_from("<I", data, 28)[0]
    if total_size < 32 or total_size > len(data) or 32 + chunk_count * 4 > total_size:
        raise ValueError("invalid DXBC container header")
    for index in range(chunk_count):
        offset = struct.unpack_from("<I", data, 32 + index * 4)[0]
        if offset + 8 > total_size:
            raise ValueError("DXBC chunk offset is outside the container")
        size = struct.unpack_from("<I", data, offset + 4)[0]
        if offset + 8 + size > total_size:
            raise ValueError("DXBC chunk extends past the container")
        if data[offset : offset + 4] == b"SFI0":
            if size != 8:
                raise ValueError(f"unexpected SFI0 size: {size}")
            return struct.unpack_from("<Q", data, offset + 8)[0]
    raise ValueError("DXIL container has no SFI0 feature chunk")


def _metadata(text: str) -> dict[int, str]:
    return {int(number): body for number, body in re.findall(r"^!(\d+)\s*=\s*!\{(.*)\}\s*$", text, re.M)}


def _root_node(text: str, name: str) -> int | None:
    match = re.search(rf"^{re.escape(name)}\s*=\s*!\{{!(\d+)\}}\s*$", text, re.M)
    return int(match.group(1)) if match else None


def _entry_point_nodes(text: str) -> list[int]:
    match = re.search(r"^!dx\.entryPoints\s*=\s*!\{([^}]*)\}\s*$", text, re.M)
    return [int(value) for value in re.findall(r"!(\d+)", match.group(1))] if match else []


def _node_strings(node: str) -> list[str]:
    return re.findall(r'!"((?:[^"\\]|\\.)*)"', node)


def _node_ints(node: str, width: int | None = None) -> list[int]:
    pattern = r"\bi(?:32|64)\s+(-?\d+)" if width is None else rf"\bi{width}\s+(-?\d+)"
    return [int(value) for value in re.findall(pattern, node)]


def _resource_bindings(text: str) -> list[dict[str, object]]:
    row = re.compile(
        r"^;\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+((?:cb|[btsu])\d+(?:,space\d+)?)(?:\s+(\S+)|(unbounded))\s*$",
        re.I,
    )
    resources = []
    for line in text.splitlines():
        match = row.match(line)
        if not match:
            continue
        _, resource_type, format_name, dimension, _, binding, spaced_count, attached_unbounded = match.groups()
        count = spaced_count or attached_unbounded
        register = binding.split(",", 1)[0].lower()
        if register.startswith(("cb", "b")):
            kind = "CBUFFER"
        elif register.startswith("s"):
            kind = "SAMPLER"
        elif register.startswith("u"):
            kind = "UAV"
        else:
            kind = "TBUFFER" if "tbuffer" in (resource_type.lower(), dimension.lower()) else "SRV"
        resources.append({
            "kind": kind,
            "format": format_name,
            "dimension": dimension,
            "hlsl_binding": binding,
            "count": int(count) if count.isdigit() else count,
        })
    return sorted(
        resources,
        key=lambda item: (
            str(item["kind"]),
            str(item["hlsl_binding"]),
            str(item["format"]),
            str(item["dimension"]),
        ),
    )


def _dx_ops(text: str) -> list[dict[str, object]]:
    counts: dict[tuple[str, str | None, int], int] = {}
    call_re = re.compile(r"\bcall\b[^\n]*@dx\.op\.([A-Za-z_][\w.]*)\(")
    for line in text.splitlines():
        match = call_re.search(line)
        if not match:
            continue
        symbol = match.group(1)
        opcode_match = re.search(r"\(i32\s+(\d+)", line[match.end() - 1 :])
        if not opcode_match:
            continue
        parts = symbol.rsplit(".", 1)
        if len(parts) == 2 and parts[1] in OVERLOADS:
            name, overload = parts[0], parts[1]
        else:
            name, overload = symbol, None
        key = (name, overload, int(opcode_match.group(1)))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"name": name, "overload": overload, "opcode": opcode, "calls": count}
        for (name, overload, opcode), count in sorted(counts.items(), key=lambda item: item[0])
    ]


def _resource_annotations(text: str) -> list[dict[str, object]]:
    result = []
    for line in text.splitlines():
        if "@dx.op.annotateHandle(" not in line:
            continue
        properties = re.search(r"ResourceProperties\s*\{\s*i32\s+(-?\d+)\s*,\s*i32\s+(-?\d+)\s*\}", line)
        if not properties:
            continue
        kind = re.search(r"resource:\s*([A-Za-z0-9_]+)", line)
        result.append({
            "resource_type": kind.group(1) if kind else None,
            "properties": [int(properties.group(1)), int(properties.group(2))],
        })
    return sorted(result, key=lambda item: (str(item["resource_type"]), item["properties"]))


def _required_features(text: str) -> list[str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "Note: shader requires additional functionality:" not in line:
            continue
        features = []
        for candidate in lines[index + 1 :]:
            if candidate.startswith("; Input signature:"):
                break
            if candidate.startswith(";"):
                value = candidate[1:].strip()
                if value:
                    features.append(value)
        return sorted(set(features))
    return []


def _entry_metadata(text: str, nodes: dict[int, str]) -> dict[str, object]:
    roots = _entry_point_nodes(text)
    if len(roots) > 1:
        raise ValueError("DXIL containers with multiple entry points are not supported")
    if not roots or roots[0] not in nodes:
        return {}
    root = roots[0]
    entry = nodes[root]
    function_match = re.search(r"\*\s*@([A-Za-z_.$][\w.$]*)", entry)
    names = _node_strings(entry)
    entry: dict[str, object] = {
        "function": function_match.group(1) if function_match else None,
        "name": names[0] if names else None,
    }
    refs = [int(value) for value in re.findall(r"!(\d+)", nodes[root])]
    if refs:
        props = nodes.get(refs[-1], "")
        flags = re.search(r"i32\s+0\s*,\s*i64\s+(\d+)", props)
        group = re.search(r"i32\s+4\s*,\s*!(\d+)", props)
        if flags:
            entry["shader_flags"] = int(flags.group(1))
        if group:
            entry["thread_group_size"] = _node_ints(nodes.get(int(group.group(1)), ""), 32)
    return entry


def inspect_dxil(path: Path, dxc: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    data = path.read_bytes()
    dump = subprocess.run([str(dxc), "-dumpbin", str(path)], check=True, capture_output=True, text=True).stdout
    nodes = _metadata(dump)

    def version(field: str) -> list[int] | None:
        node_id = _root_node(dump, field)
        return _node_ints(nodes.get(node_id, ""), 32) if node_id is not None else None

    model_id = _root_node(dump, "!dx.shaderModel")
    model_node = nodes.get(model_id, "") if model_id is not None else ""
    model_values = _node_ints(model_node, 32)
    model_strings = _node_strings(model_node)
    profile = None
    stage = None
    if model_strings and len(model_values) >= 2:
        stage_code = model_strings[0]
        profile = f"{stage_code}_{model_values[0]}_{model_values[1]}"
        stage = STAGE_NAMES.get(stage_code, stage_code)

    triple_match = re.search(r'^target triple = "([^"]+)"', dump, re.M)
    layout_match = re.search(r'^target datalayout = "([^"]+)"', dump, re.M)
    threads = re.search(r"NumThreads=\((\d+),(\d+),(\d+)\)", dump)
    entry = _entry_metadata(dump, nodes)
    if threads and "thread_group_size" not in entry:
        entry["thread_group_size"] = [int(value) for value in threads.groups()]

    sfi0 = container_feature_flags(data)
    return {
        "input": path.name,
        "stage": stage,
        "shader_model": profile,
        "dxil_module_version": version("!dx.version"),
        "validator_version": version("!dx.valver"),
        "entry": entry,
        "sfi0_feature_flags": f"0x{sfi0:016x}",
        "required_features": _required_features(dump),
        "dx_ops": _dx_ops(dump),
        "resource_bindings": _resource_bindings(dump),
        "resource_annotations": _resource_annotations(dump),
        "target_triple": triple_match.group(1) if triple_match else None,
        "data_layout": layout_match.group(1) if layout_match else None,
    }


def dxc_version(dxc: Path) -> str:
    result = subprocess.run([str(dxc), "--version"], check=True, capture_output=True, text=True)
    text = result.stdout + result.stderr
    match = re.search(r"(?<!\d)(\d+\.\d+\.\d+\.\d+)(?!\d)", text)
    if not match:
        raise ValueError(f"could not read DXC version from {dxc}")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("containers", nargs="+", type=Path, help="DXIL container(s) to inspect")
    parser.add_argument("--dxc", help="path to dxc.exe (or use XCL_DXC_DIR / Windows SDK)")
    args = parser.parse_args()
    try:
        dxc = find_dxc(args.dxc)
        results = [inspect_dxil(path, dxc) for path in args.containers]
        tool_version = dxc_version(dxc)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"dxil semantics: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"schema_version": 1, "disassembler_dxc_version": tool_version, "shaders": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
