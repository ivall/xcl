from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import runpy
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parent / "re" / "check_layout.py"


class LayoutPointerTests(unittest.TestCase):
    def test_unrelated_mask_does_not_hide_the_root_argument_pair(self):
        root = struct.pack("<11I", 1, 1, 24, 0, 0, 0, 2, 0, 36, 0, 0)
        chunk = b"RTS0" + struct.pack("<I", len(root)) + root
        blob = struct.pack("<4s16sIIII", b"DXBC", bytes(16), 1, 36 + len(chunk), 1, 36) + chunk
        listing = (
            "s_and_b32 s13, s1, 0xff\n"
            "s_mov_b32 s4, s0\n"
            "s_and_b32 s5, s1, 0xff\n"
            "s_load_dword s6, s[4:5], 0x0\n"
            "s_load_dword s7, s[4:5], 0x8\n"
        )
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout=listing, stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "00.dxbc").write_bytes(blob)
            output = StringIO()
            with patch("sys.argv", [str(SCRIPT), tmp]), patch("subprocess.run", return_value=result), redirect_stdout(output):
                runpy.run_path(str(SCRIPT), run_name="__main__")
        self.assertIn("loads on parameter boundaries: 1, off: 1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
