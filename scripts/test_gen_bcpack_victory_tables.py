import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_bcpack_victory_tables import generate, DEFAULT_OUT, TABLES_BIN_SIZE, MODELS, BUCKETS, DICTIONARY


SCRIPT = Path(__file__).resolve().parent / "gen_bcpack_victory_tables.py"


def make_synthetic_tables_bin():
    b = bytearray()
    # freq: MODELS * BUCKETS u16
    b.extend(b"\x01\x00" * (MODELS * BUCKETS))
    # rank1: 256 bytes
    b.extend(bytes(range(256)))
    # rank3a: 64 bytes
    b.extend(bytes(range(64)))
    # rank3c: 256 bytes
    b.extend(bytes(range(256)))
    # rank4: 64 bytes
    b.extend(bytes(range(64)))
    # rank5: 64 bytes
    b.extend(bytes(range(64)))
    # ctrl7: 512 u16
    b.extend(b"\x07\x00" * 512)
    # bc1: 255 * (6 + 16)
    b.extend(b"\x11" * (DICTIONARY * 22))
    # bc3: 255 * (8 + 32)
    b.extend(b"\x33" * (DICTIONARY * 40))
    # bc4: 255 * (2 + 16)
    b.extend(b"\x44" * (DICTIONARY * 18))
    # bc5: 255 * (4 + 32)
    b.extend(b"\x55" * (DICTIONARY * 36))
    # bc6: 255 * (1 + 12*2 + 1 + 16*2) = 255 * 58
    b.extend(b"\x66" * (DICTIONARY * 58))
    # bc7: 255 * (2 + 24*2 + 2 + 2 + 31*2) = 255 * 116
    b.extend(b"\x77" * (DICTIONARY * 116))
    assert len(b) == TABLES_BIN_SIZE, f"synthetic size {len(b)} != {TABLES_BIN_SIZE}"
    return bytes(b)


class GenBcpackVictoryTablesCliTests(unittest.TestCase):
    def test_help_flag_prints_usage_and_exits_zero(self):
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                res = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), flag],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(res.returncode, 0)
                self.assertIn("Turn VIctory's bcpack-tables.bin into Rust constants", res.stdout)
                self.assertIn("scripts/gen_bcpack_victory_tables.py", res.stdout)

    def test_missing_args_prints_usage_and_exits_code_2(self):
        res = subprocess.run(
            [sys.executable, "-B", str(SCRIPT)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 2)
        self.assertIn("scripts/gen_bcpack_victory_tables.py", res.stdout)

    def test_nonexistent_file_exits_code_1_with_error(self):
        res = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "nonexistent_tables.bin"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("error: file not found", res.stderr)

    def test_invalid_size_file_exits_code_1_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_file = Path(tmp) / "truncated.bin"
            bad_file.write_bytes(b"\x00" * 100)
            res = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(bad_file)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 1)
            self.assertIn(f"expected {TABLES_BIN_SIZE} bytes", res.stderr)

    def test_cli_custom_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_bin = Path(tmp) / "bcpack-tables.bin"
            synthetic_bin.write_bytes(make_synthetic_tables_bin())
            out_file = Path(tmp) / "sub" / "out_tables.rs"

            res = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(synthetic_bin), str(out_file)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0)
            self.assertIn("wrote", res.stdout)
            self.assertIn("entries 255 255 255 255 255 255", res.stdout)
            self.assertTrue(out_file.is_file())
            content = out_file.read_text(encoding="utf-8")
            self.assertIn("pub const MODELS: usize = 128;", content)
            self.assertIn("pub static FREQ: [[u32; BUCKETS]; MODELS] = [", content)
            self.assertIn("pub static BC7: [(Bc7Ep, Bc7Ix); DICTIONARY] = [", content)


class GenBcpackVictoryTablesGenerateTests(unittest.TestCase):
    def test_generate_synthetic_tables_structure(self):
        blob = make_synthetic_tables_bin()
        content, counts = generate(blob)
        self.assertEqual(counts, (255, 255, 255, 255, 255, 255))
        self.assertIn("pub const DICTIONARY: usize = 255;", content)
        self.assertIn("pub static RANK1: [u8; 256] = [", content)
        self.assertIn("pub static BC1: [Plain; DICTIONARY] = [", content)

    def test_generate_rejects_incorrect_size(self):
        with self.assertRaises(ValueError):
            generate(b"short data")


if __name__ == "__main__":
    unittest.main()
