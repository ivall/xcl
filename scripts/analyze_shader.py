"""Prepare a verified, isolated shader bundle for optional vendor diagnostics."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

from verify_shader_dump import verify


def prepare(manifest, output, source_commit=None):
    """Copy first, then verify the exact bytes an analyzer will consume."""
    output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="shader-", dir=output)).resolve()
    report = dict(schema_version=1, source_commit=source_commit,
                  manifest_path=str(manifest.resolve()), status="ERROR",
                  category="invalid_bundle", verification=None, analyzer=None)
    try:
        if not manifest.name.endswith(".manifest.json"):
            raise ValueError("expected a .manifest.json file")
        bundle = run / "input"
        bundle.mkdir()
        copied = bundle / manifest.name
        shutil.copyfile(manifest, copied)
        data = json.loads(copied.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected a manifest object")
        prefix = manifest.name[:-len(".manifest.json")]
        suffixes = [".guest.bin", ".direct.err.txt" if data.get("error") is not None else ".direct.dxil"]
        for suffix in suffixes:
            shutil.copyfile(manifest.with_name(prefix + suffix), bundle / (prefix + suffix))
        report["verification"] = verify(copied)
        report["manifest"] = data
        # SHA-256 complements the manifest's FNV identity; neither authenticates its author.
        report["artifacts"] = {}
        for path in sorted(bundle.iterdir()):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            report["artifacts"][path.name] = dict(path=str(path), sha256=digest)
        report.update(status="SKIPPED", category="analyzer_not_requested")
        if data.get("error") is not None:
            report["category"] = "translation_rejected"
    except (OSError, ValueError, KeyError) as exc:
        report["detail"] = str(exc)
    return run, report


def save_report(run, report):
    path = run / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", help="producer commit, supplied by the caller; not inferred from this checkout")
    parser.add_argument("--rga", nargs="?", const=os.environ.get("XCL_RGA", "rga.exe"),
                        help="opt in to AMD offline compute analysis; optional executable path (or XCL_RGA)")
    parser.add_argument("--target", help="explicit AMD offline target, e.g. gfx1101; not a detected host adapter")
    parser.add_argument("--timeout", type=float, default=120, help="seconds per analyzer invocation")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    try:
        run, report = prepare(args.manifest, args.out, args.source_commit)
        if args.rga is not None:
            from shader_rga import analyze
            analyze(run, report, args.rga, args.target, args.timeout)
        path = save_report(run, report)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{report['status']}: {report['category']}; {path}")
    return int(report["status"] == "ERROR")


if __name__ == "__main__":
    sys.exit(main())
