from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().with_name("gen_tiling.py")


class TilingTableSpacingTests(unittest.TestCase):
    def test_initializer_endings_leave_generated_tables_unchanged(self):
        modes = (
            "256_S", "256_D", "4K_S", "4K_D", "64K_S", "64K_D", "64K_S_T", "64K_D_T", "4K_S_X",
            "4K_D_X", "64K_Z_X_1xaa", "64K_S_X", "64K_D_X", "64K_R_X_1xaa", "4K_S3", "64K_S3",
            "64K_S3_T", "4K_S3_X", "64K_S3_X", "64K_D3_X",
        )
        tables = {f"GFX10_SW_PATTERN_NIBBLE{i}": "X0, Y0, 0, 0" for i in (2, 3, 4)}
        tables["GFX10_SW_PATTERN_NIBBLE01"] = "X0, Y0, Z0, S0, 0, 0, 0, 0"
        tables.update((f"GFX10_SW_{mode}_RBPLUS_PATINFO", "0, 0, 0, 0, 0") for mode in modes)
        plain = "typedef unsigned long long u64;\n#define X0 1\n#define Y0 (1ULL << 16)\n#define Z0 (1ULL << 32)\n#define S0 (1ULL << 48)\n"
        plain += "".join(
            f"const u64 {name}[][{len(values.split(', '))}] = {{\n    {{{values}}}\n}};\n"
            for name, values in tables.items()
        )
        first_mode = "const u64 GFX10_SW_256_S_RBPLUS_PATINFO[][5]"
        second_mode = "const u64 GFX10_SW_256_D_RBPLUS_PATINFO[][5]"
        one_row = "{\n    {0, 0, 0, 0, 0}\n};"
        plain = plain.replace(f"{second_mode} = {one_row}",
                              f"{second_mode} = {{\n    {{0, 0, 0, 0, 0}},\n    {{0, 0, 0, 0, 0}}\n}};")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "scripts" / SCRIPT.name
            script.parent.mkdir()
            shutil.copyfile(SCRIPT, script)
            output = root / "crates" / "xcl" / "src" / "gpu" / "tiling_tables.rs"
            output.parent.mkdir(parents=True)
            source = root / "gfx10SwizzlePattern.h"
            command = [sys.executable, "-B", str(script), str(source)]

            source.write_text(plain, encoding="utf-8")
            subprocess.run(command, capture_output=True, check=True)
            expected = output.read_bytes()
            self.assertIn(b"pub static PATTERNS_3D", expected)

            mixed = plain.replace(f"{first_mode} = {one_row}",
                                  f"{first_mode} = {{\n    {{0, 0, 0, 0, 0}} }};")
            variants = {
                "indented": plain.replace("\n};", "\n\t } ;"),
                "one inline mode": mixed,
                "all inline": plain.replace("\n};", " };"),
                "block comment": plain.replace("\n};", ", /* }; */ } ;"),
                "line comment": plain.replace("\n};", " // }; in a comment\n}\n;"),
            }
            for layout, header in variants.items():
                with self.subTest(layout=layout):
                    source.write_text(header, encoding="utf-8")
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
