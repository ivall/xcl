import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Allow importing generate from tests/data/bcpack
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "data" / "bcpack"))

import generate


class BCPackGenerateTests(unittest.TestCase):
    def test_record_header_small(self):
        # Body size <= 256 uses 2-byte header: (mode, size - 1)
        h = generate.record_header(0x10, 100)
        self.assertEqual(len(h), 2)
        self.assertEqual(h[0], 0x10)
        self.assertEqual(h[1], 99)

    def test_record_header_large(self):
        # Body size > 256 sets bit 5 (0x20) and encodes size - 257 in 2 little-endian bytes
        h = generate.record_header(0x10, 300)
        self.assertEqual(len(h), 3)
        self.assertEqual(h[0], 0x10 | 0x20)
        # 300 - 257 = 43 -> 0x2B 0x00
        self.assertEqual(h[1:], b"\x2b\x00")

    def test_seed_channels(self):
        # Verifies seed channels generate 6 predictor values within channel bit widths
        channels = generate.seed_channels(0)
        self.assertEqual(len(channels), 6)
        for val, width in zip(channels, generate.BC1[2:8]):
            self.assertTrue(0 <= val < (1 << width))

    def test_assemble_mode1(self):
        # Synthetic symbol block of mode 1 symbols
        units = 4
        symbols = [[0] * units for _ in generate.WIDTHS[1]]
        symbols[0] = [1, 2, 3, 4]  # ref indices
        raw = generate.assemble(symbols)
        # Each BC1 block is 8 bytes (2x 16-bit colors + 1x 32-bit indices)
        self.assertEqual(len(raw), units * 8)

    def test_encoder_lane_and_encode(self):
        symbols = [[0, 1], [0, 0], [1, 0], [2, 1], [1, 1], [0, 1], [1, 0], [0, 0], [1, 1], [0, 0], [1, 0], [0, 1]]
        widths = generate.WIDTHS[1]
        reductions = [0] * len(widths)
        encoded = generate.encode(symbols, widths, reductions)
        self.assertIsInstance(encoded, bytes)
        self.assertGreater(len(encoded), 0)

    def test_generate_custom_out_dir_matches_reference(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            syms, path = generate.generate(1, 2, 0, False, out_dir=out_dir)
            self.assertEqual(path.parent, out_dir)
            self.assertTrue(path.with_suffix(".bin").is_file())
            self.assertTrue(path.with_suffix(".lanes").is_file())

            # Verify byte-identical match with committed fixture
            ref_bin = generate.ROOT / "bc1-2-0.bin"
            ref_lanes = generate.ROOT / "bc1-2-0.lanes"
            self.assertEqual(path.with_suffix(".bin").read_bytes(), ref_bin.read_bytes())
            self.assertEqual(path.with_suffix(".lanes").read_bytes(), ref_lanes.read_bytes())

    def test_generate_trim_and_carry_options(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            _, path_trim = generate.generate(1, 1, 8, False, out_dir=out_dir, trim=True)
            self.assertTrue(path_trim.with_suffix(".bin").is_file())
            ref_trim = generate.ROOT / "bc1-1-8.bin"
            self.assertEqual(path_trim.with_suffix(".bin").read_bytes(), ref_trim.read_bytes())

            _, path_carry = generate.generate(1, 16, 9, False, out_dir=out_dir, carry=True)
            self.assertTrue(path_carry.with_suffix(".bin").is_file())
            ref_carry = generate.ROOT / "bc1-16-9.bin"
            self.assertEqual(path_carry.with_suffix(".bin").read_bytes(), ref_carry.read_bytes())

    def test_main_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            with patch("sys.stdout", new=io.StringIO()):
                generate.main(["--help"])
        self.assertEqual(cm.exception.code, 0)

        with self.assertRaises(SystemExit) as cm:
            with patch("sys.stdout", new=io.StringIO()):
                generate.main(["-h"])
        self.assertEqual(cm.exception.code, 0)

    def test_main_invalid_mode_exits_code_two(self):
        with self.assertRaises(SystemExit) as cm:
            with patch("sys.stderr", new=io.StringIO()):
                generate.main(["--mode", "99"])
        self.assertEqual(cm.exception.code, 2)

    def test_main_mode_filter_selective_generation(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            code = generate.main(["--out-dir", str(out_dir), "--mode", "3"])
            self.assertEqual(code, 0)

            # Mode 3 vectors should exist
            self.assertTrue((out_dir / "bc3-16-1.bin").is_file())
            self.assertTrue((out_dir / "bc3-16-1.lanes").is_file())

            # Mode 1 or Mode 7 vectors should NOT have been generated
            self.assertFalse((out_dir / "bc1-2-0.bin").exists())
            self.assertFalse((out_dir / "bc7-2-0.bin").exists())

    def test_main_mode1_generates_bc1_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            code = generate.main(["--out-dir", str(out_dir), "--mode", "1"])
            self.assertEqual(code, 0)

            # Mode 1 includes .bc1 files from assemble()
            self.assertTrue((out_dir / "bc1-2-0.bc1").is_file())
            self.assertTrue((out_dir / "bc1-16-1.bc1").is_file())
            self.assertTrue((out_dir / "bc1-8192-3.bin").is_file())


if __name__ == "__main__":
    unittest.main()
