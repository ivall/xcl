#!/usr/bin/env python3
"""Linux: python3 scripts/repro_dxil_structurizer.py <dxc> <libdxil-spirv-c-shared.so>

Use Microsoft's DXC v1.9.2609. Build the converter from dxil-spirv with
CMake (Release, -DDXIL_SPIRV_CLI=OFF, target dxil-spirv-c-shared) after
initializing the spirv-headers and dxbc-spirv submodules. Compare
8b48f0e0392d76be4fdcea2a3ba3ea99dddaae5b (vkd3d-proton v3.1) with
34f0ff845eb7f1ca76ca8df5d1593d6913f19e9c (vkd3d-proton master 30b0eb9).
On the latter, --legacy selects the previous structurizer in the same binary.
Also requires prlimit, GNU timeout, and spirv-val (Vulkan 1.3 support).
"""

import argparse
import ctypes as c
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import monotonic


def convert(library, dxil, output, legacy):
    class Spirv(c.Structure):
        _fields_ = [("data", c.c_void_p), ("size", c.c_size_t)]

    class LegacyOption(c.Structure):
        # dxil_spv_option_legacy_structurizer in dxil_spirv_c.h.
        _fields_ = [("type", c.c_int), ("enabled", c.c_ubyte)]

    lib = c.CDLL(library)

    def call(name, types=(), args=(), returns=c.c_int):
        func = getattr(lib, f"dxil_spv_{name}")
        func.argtypes = types
        func.restype = returns
        if code := func(*args):
            raise SystemExit(f"{name} failed: {code}")

    blob, converter, spirv = c.c_void_p(), c.c_void_p(), Spirv()
    data = Path(dxil).read_bytes()
    ptr = c.c_void_p
    call("begin_thread_allocator_context", returns=None)
    call("parse_dxil_blob", (c.c_char_p, c.c_size_t, c.POINTER(ptr)), (data, len(data), c.byref(blob)))
    call("create_converter", (ptr, c.POINTER(ptr)), (blob, c.byref(converter)))
    if legacy == "legacy":
        call("converter_add_option", (ptr, c.POINTER(LegacyOption)), (converter, c.byref(LegacyOption(57, 1))))
    call("converter_run", (ptr,), (converter,))
    call("converter_get_compiled_spirv", (ptr, c.POINTER(Spirv)), (converter, c.byref(spirv)))
    Path(output).write_bytes(c.string_at(spirv.data, spirv.size))
    call("converter_free", (ptr,), (converter,), returns=None)
    call("parsed_blob_free", (ptr,), (blob,), returns=None)
    call("end_thread_allocator_context", returns=None)


def bounded(stage, *command):
    # A hard per-process address-space cap; no core files or machine-wide settings.
    argv = ["prlimit", "--as=536870912", "--core=0", "--", "timeout", "-k", "1s", "10s", *map(str, command)]
    start = monotonic()
    pid = os.posix_spawnp(argv[0], argv, os.environ)
    _, status, usage = os.wait4(pid, 0)
    code = os.waitstatus_to_exitcode(status)
    print(f"{stage}: exit {code}, {monotonic() - start:.3f} s, {usage.ru_maxrss / 1024:.1f} MiB peak RSS", flush=True)
    return code == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dxc")
    parser.add_argument("converter", type=Path)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "tests/data/dxil-structurizer.hlsl"
    with TemporaryDirectory(prefix="xcl-dxil-repro-") as temp:
        dxil, ir, spirv = (Path(temp) / f"shader.{ext}" for ext in ("dxil", "ll", "spv"))
        if not bounded("DXC", args.dxc, "-HV", "2021", "-O1", "-T", "cs_6_6", "-E", "main", "-Fo", dxil, "-Fc", ir, source):
            print("FAIL: DXIL compilation failed", file=sys.stderr)
            return 1
        if sum(line.startswith("  switch ") for line in ir.read_text().splitlines()) != 1:
            print("FAIL: DXC did not emit the single switch", file=sys.stderr)
            return 1
        mode = "legacy" if args.legacy else "default"
        if not bounded("dxil-spirv", sys.executable, __file__, "--convert", args.converter.resolve(), dxil, spirv, mode):
            print("FAIL: DXIL conversion did not complete within the limits", file=sys.stderr)
            return 1
        if not bounded("spirv-val", "spirv-val", "--target-env", "vulkan1.3", spirv):
            print("FAIL: SPIR-V validation failed", file=sys.stderr)
            return 1
    print("PASS: validated SPIR-V within 512 MiB and 10 s per step")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--convert"]:
        convert(*sys.argv[2:])
    else:
        raise SystemExit(main())
