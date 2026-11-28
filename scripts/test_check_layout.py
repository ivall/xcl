import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "re"))
from check_layout import layout, parts, find_xcl


SCRIPT = Path(__file__).resolve().parent / "re" / "check_layout.py"


class CheckLayoutCliTests(unittest.TestCase):
    def test_help_flag_prints_usage_and_exits_zero(self):
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                res = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), flag],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(res.returncode, 0)
                self.assertIn("Check the root-argument table hypothesis", res.stdout)
                self.assertIn("usage: python check_layout.py", res.stdout)

    def test_missing_args_prints_usage_and_exits_code_2(self):
        res = subprocess.run(
            [sys.executable, "-B", str(SCRIPT)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 2)
        self.assertIn("usage: python check_layout.py", res.stdout)

    def test_nonexistent_directory_exits_code_1_with_error(self):
        res = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "nonexistent_dir_12345"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("error: directory not found", res.stderr)

    def test_missing_xcl_binary_reports_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["XCL_EXE"] = os.path.join(tmp, "missing_xcl.exe")
            res = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), tmp],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(res.returncode, 1)
            self.assertIn("error: xcl binary not found", res.stderr)


class CheckLayoutLayoutTests(unittest.TestCase):
    def test_layout_parses_constants_and_descriptors(self):
        # RTS0 header: ver=1, n=4, poff=12
        # Param 0: ty=1 (consts, 4 dwords = 16 B) -> pay=60
        # Param 1: ty=2 (CBV, 8 B) -> pay=72
        # Param 2: ty=3 (SRV, 8 B) -> pay=80
        # Param 3: ty=4 (UAV, 8 B) -> pay=88
        n = 4
        poff = 12
        r = bytearray(struct.pack("<3I", 1, n, poff))
        # Param table: ty, vis, pay
        r += struct.pack("<3I", 1, 0, 60)
        r += struct.pack("<3I", 2, 0, 72)
        r += struct.pack("<3I", 3, 0, 80)
        r += struct.pack("<3I", 4, 0, 88)
        # Pad up to offset 60
        r.extend(b"\0" * (60 - len(r)))
        # Payload 0: reg=0, space=0, num=4
        r += struct.pack("<3I", 0, 0, 4)
        # Payload 1: reg=0, space=0
        r += struct.pack("<2I", 0, 0)
        # Payload 2: reg=1, space=0
        r += struct.pack("<2I", 1, 0)
        # Payload 3: reg=2, space=0
        r += struct.pack("<2I", 2, 0)

        consts, descs, size = layout(bytes(r))
        self.assertEqual(consts, [(0, 16)])
        self.assertEqual(descs, {16: "CBV", 24: "SRV", 32: "UAV"})
        self.assertEqual(size, 40)

    def test_layout_rejects_unknown_param_type(self):
        # ty = 0 (table) -> layout() returns None
        r = bytearray(struct.pack("<3I", 1, 1, 12))
        r += struct.pack("<3I", 0, 0, 24)
        self.assertIsNone(layout(bytes(r)))


class CheckLayoutEndToEndTests(unittest.TestCase):
    def test_disasm_matching_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Build synthetic DXBC container with RTS0
            root = bytearray(struct.pack("<3I", 1, 2, 12))
            root += struct.pack("<3I", 1, 0, 36)  # consts x2 -> 8 B (0..8)
            root += struct.pack("<3I", 2, 0, 48)  # CBV -> 8 B (8..16)
            root += struct.pack("<3I", 0, 0, 2)
            root += struct.pack("<2I", 0, 0)

            chunk = b"RTS0" + struct.pack("<I", len(root)) + root
            blob = struct.pack("<4s16sIII", b"DXBC", bytes(16), 1, 36 + len(chunk), 1)
            blob += struct.pack("<I", 36) + chunk

            dxbc_path = Path(tmp) / "shaders" / "00.dxbc"
            dxbc_path.parent.mkdir(parents=True, exist_ok=True)
            dxbc_path.write_bytes(blob)

            # Mock xcl executable as a python script runner
            mock_xcl = Path(tmp) / "mock_xcl.cmd"
            mock_xcl_py = Path(tmp) / "mock_xcl.py"
            code = (
                "import sys\n"
                "if len(sys.argv) > 1 and sys.argv[1] == 'disasm':\n"
                "    sys.stdout.write('s_mov_b32 s4, s0\\ns_and_b32 s5, s1, 0xff\\ns_load_dword s2, s[4:5], 0x0\\ns_load_dwordx2 s[6:7], s[4:5], 0x8\\n')\n"
            )
            mock_xcl_py.write_text(code, encoding="utf-8")
            mock_xcl.write_text(f'@"{sys.executable}" "{mock_xcl_py}" %*\n', encoding="utf-8")

            env = os.environ.copy()
            env["XCL_EXE"] = str(mock_xcl)
            res = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(dxbc_path.parent)],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(res.returncode, 0)
            self.assertIn("loads on parameter boundaries: 2, off: 0", res.stdout)


if __name__ == "__main__":
    unittest.main()
