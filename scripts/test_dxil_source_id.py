"""Check that a deleted direct-DXIL source restores the original cache identity.

Runs the real crates/xcl/build.rs in a tiny temporary Cargo package so this
test checks Cargo's first build after a nested source is removed, not just the
hash function in isolation. Requires only Python, Rust, and Cargo.
"""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BUILD_RS = ROOT / "crates/xcl/build.rs"


def listed_sources(source, name):
    match = re.search(rf"const {name}:.*?= &\[(.*?)\];", source, re.S)
    if match is None:
        raise RuntimeError(f"cannot find {name} in build.rs")
    return re.findall(r'"(src/[^"\n]+\.rs)"', match.group(1))


def build_id(manifest, target):
    command = ["cargo", "run", "--offline", "--quiet", "--manifest-path", str(manifest), "--target-dir", str(target)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Cargo failed:\n{result.stdout}{result.stderr}")
    return result.stdout.strip()


def main():
    source = BUILD_RS.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="xcl-source-id-") as tmp:
        package = Path(tmp) / "package"
        package.mkdir()
        shutil.copyfile(BUILD_RS, package / "build.rs")
        (package / "Cargo.toml").write_text('[package]\nname = "xcl"\nversion = "0.0.0"\nedition = "2021"\n', encoding="utf-8")
        (package / "src").mkdir()
        shutil.copyfile(BUILD_RS.parent / "src/source_hash.rs", package / "src/source_hash.rs")
        (package / "src/main.rs").write_text('fn main() { println!("{}", env!("XCL_DIRECT_DXIL_SOURCE_ID")); }\n', encoding="utf-8")
        for relative in listed_sources(source, "TRANSLATOR_SOURCES") + listed_sources(source, "PER_LANE_SOURCES"):
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("// fixture\n", encoding="utf-8")
        direct = package / "src/gpu/dxil/module"
        direct.mkdir(parents=True)
        (package / "src/gpu/dxil/mod.rs").write_text("// fixture\n", encoding="utf-8")
        (direct / "mod.rs").write_text("// fixture\n", encoding="utf-8")

        incremental = Path(tmp) / "incremental"
        original = build_id(package / "Cargo.toml", incremental)
        added = direct / "review_temporary_input.rs"
        added.write_text("// temporary source\n", encoding="utf-8")
        with_file = build_id(package / "Cargo.toml", incremental)
        added.unlink()
        first_restored = build_id(package / "Cargo.toml", incremental)
        clean_restored = build_id(package / "Cargo.toml", Path(tmp) / "clean")
        second_restored = build_id(package / "Cargo.toml", incremental)

        print(f"original: {original}")
        print(f"with temporary file: {with_file}")
        print(f"first build after deletion: {first_restored}")
        print(f"clean build after deletion: {clean_restored}")
        print(f"second build after deletion: {second_restored}")
        if original == with_file:
            raise AssertionError("adding a source did not change the ID")
        if original != clean_restored:
            raise AssertionError("clean restoration did not reproduce the original ID")
        if first_restored != clean_restored:
            raise AssertionError("first build after deletion used a stale direct-DXIL source ID")
        if second_restored != clean_restored:
            raise AssertionError("second build after deletion did not recover")


if __name__ == "__main__":
    main()
