import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_gdk_table import generate, hexg, DEFAULT_OUT


SCRIPT = Path(__file__).resolve().parent / "gen_gdk_table.py"


SAMPLE_INPUT = """Source: test-lib/xgameruntime.lib
NAME(XUserCloseHandle, "01acd177-91f9-4763-a38e-ccbb55ce32e0", "01acd177-91f9-4763-a38e-ccbb55ce32e0", 4)
NAME(XUserDuplicateHandle, "01acd177-91f9-4763-a38e-ccbb55ce32e0", "01acd177-91f9-4763-a38e-ccbb55ce32e0", 3)
NAME(XOtherApi, "bbbbbbbb-1111-2222-3333-444444444444", "aaaaaaaa-5555-6666-7777-888888888888", 1)
"""


class GenGdkTableCliTests(unittest.TestCase):
    def test_help_flag_prints_usage_and_exits_zero(self):
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                res = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), flag],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(res.returncode, 0)
                self.assertIn("Generate crates/xcl/src/hle/xgameruntime/gdk_table.rs", res.stdout)
                self.assertIn("usage: python scripts/gen_gdk_table.py", res.stdout)

    def test_missing_args_prints_usage_and_exits_code_2(self):
        res = subprocess.run(
            [sys.executable, "-B", str(SCRIPT)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 2)
        self.assertIn("usage: python scripts/gen_gdk_table.py", res.stdout)

    def test_nonexistent_file_exits_code_1_with_error(self):
        res = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "nonexistent_gdk_table.txt"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("error: file not found", res.stderr)

    def test_invalid_input_file_exits_code_1_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_file = Path(tmp) / "empty.txt"
            bad_file.write_text("No rows here\n", encoding="utf-8")
            res = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(bad_file)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 1)
            self.assertIn("error: no GDK API rows found", res.stderr)

    def test_cli_custom_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_file = Path(tmp) / "input.txt"
            src_file.write_text(SAMPLE_INPUT, encoding="utf-8")
            out_file = Path(tmp) / "sub" / "out_gdk.rs"

            res = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(src_file), str(out_file)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0)
            self.assertIn("3 rows ->", res.stdout)
            self.assertTrue(out_file.is_file())
            content = out_file.read_text(encoding="utf-8")
            self.assertIn('pub static API: &[(&str, u128, u128, u8)]', content)
            self.assertIn('("XUserDuplicateHandle", 0x01acd17791f94763a38eccbb55ce32e0, 0x01acd17791f94763a38eccbb55ce32e0, 3)', content)


class GenGdkTableGenerateTests(unittest.TestCase):
    def test_hexg_formatting(self):
        self.assertEqual(
            hexg("01acd177-91f9-4763-a38e-ccbb55ce32e0"),
            "0x01acd17791f94763a38eccbb55ce32e0",
        )
        self.assertEqual(
            hexg("ABCDEF01-2345-6789-ABCD-EF0123456789"),
            "0xabcdef0123456789abcdef0123456789",
        )

    def test_generate_sorting_and_content(self):
        content, rows = generate(SAMPLE_INPUT)
        self.assertEqual(len(rows), 3)
        self.assertIn("test-lib/xgameruntime.lib", content)
        # DuplicateHandle (slot 3) must appear before CloseHandle (slot 4)
        pos_dup = content.find('"XUserDuplicateHandle"')
        pos_close = content.find('"XUserCloseHandle"')
        self.assertNotEqual(pos_dup, -1)
        self.assertNotEqual(pos_close, -1)
        self.assertLess(pos_dup, pos_close)

    def test_generate_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            generate("random content without matching macro calls")


if __name__ == "__main__":
    unittest.main()
