"""Check the files next to a manifest written by xcl shaders --direct --dump.

Run python scripts/verify_shader_dump.py FILE.manifest.json [FILE.manifest.json ...].
"""

import argparse
import json
from pathlib import Path
import sys


def fnv64(path):
    """Match shader::fnv64 on the bytes of a dump file."""
    value = 0xCBF29CE484222325
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            for byte in chunk:
                value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def verify(path):
    suffix = ".manifest.json"
    if not path.name.endswith(suffix):
        raise ValueError(f"expected a {suffix} file")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("expected a manifest object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("expected schema_version 1")
    if manifest.get("backend") != "direct-dxil":
        raise ValueError("expected backend direct-dxil")
    prefix = path.name[:-len(suffix)]
    guest = path.with_name(prefix + ".guest.bin")
    if fnv64(guest) != manifest["guest_fnv64"]:
        raise ValueError(f"{guest.name}: guest_fnv64 mismatch")

    dxil_hash, error = manifest["dxil_fnv64"], manifest["error"]
    if error is None and isinstance(dxil_hash, str):
        dxil = path.with_name(prefix + ".direct.dxil")
        if fnv64(dxil) != dxil_hash:
            raise ValueError(f"{dxil.name}: dxil_fnv64 mismatch")
        return "guest and translated DXIL match"
    if dxil_hash is None and isinstance(error, str):
        diagnostic = path.with_name(prefix + ".direct.err.txt")
        if diagnostic.read_bytes() != error.encode("utf-8"):
            raise ValueError(f"{diagnostic.name}: error text mismatch")
        return "guest and translation error match"
    raise ValueError("expected one translation result (dxil_fnv64 or error)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path, nargs="+")
    failed = False
    for path in parser.parse_args(argv).manifest:
        try:
            print(f"{path}: {verify(path)}")
        except (OSError, ValueError, KeyError) as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
