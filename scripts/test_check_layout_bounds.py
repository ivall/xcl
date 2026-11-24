from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parent / "re" / "check_layout.py"


class RootArgumentBoundsTests(unittest.TestCase):
    def test_loads_must_fit_in_the_root_argument_table(self):
        # An 8-byte UAV descriptor followed by two 4-byte constants.
        root = struct.pack("<6I", 1, 2, 24, 0, 0, 0)
        root += struct.pack("<6I", 4, 0, 48, 1, 0, 56)
        root += struct.pack("<5I", 0, 0, 0, 0, 2)
        chunk = b"RTS0" + struct.pack("<I", len(root)) + root
        header = struct.pack("<4s16sIIII", b"DXBC", bytes(16), 1, 36 + len(chunk), 1, 36)
        disassembly = (
            "s_mov_b32 s4, s0\n"
            "s_and_b32 s5, s1, 0xff\n"
            "s_load_dwordx4 s[8:11], s[4:5], 0x0\n"
            "s_load_dword s8, s[4:5], 0x4\n"
            "s_load_dword s8, s[4:5], 0xc\n"
            "s_load_dwordx2 s[8:9], s[4:5], 0xc\n"
            "s_load_dwordx4 s[8:11], s[4:5], 0x4\n"
        )
        output = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "synthetic.dxbc").write_bytes(header + chunk)
            result = subprocess.CompletedProcess([], 0, stdout=disassembly, stderr="")
            with (
                patch.object(sys, "argv", [str(SCRIPT), directory]),
                patch.object(subprocess, "run", return_value=result),
                redirect_stdout(output),
            ):
                runpy.run_path(str(SCRIPT), run_name="__main__")

        self.assertIn("load 8 B @0xc extends past", output.getvalue())
        self.assertIn("load 16 B @0x4 extends past", output.getvalue())
        self.assertIn("loads on parameter boundaries: 3, off: 2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
