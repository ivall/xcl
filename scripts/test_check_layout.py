import contextlib
import io
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class CheckLayoutTests(unittest.TestCase):
    def test_disassembler_failure_stops_the_scan(self):
        script = Path(__file__).parent / "re" / "check_layout.py"
        root_signature = struct.pack("<3I", 1, 0, 12)
        dxbc = b"DXBC" + bytes(24) + struct.pack("<2I", 1, 36)
        dxbc += b"RTS0" + struct.pack("<I", len(root_signature)) + root_signature
        failed = subprocess.CompletedProcess([], 2, stdout="", stderr="invalid shader\n")
        stdout, stderr = io.StringIO(), io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "shader.dxbc").write_bytes(dxbc)
            with patch.object(sys, "argv", [str(script), directory]), patch.object(subprocess, "run", return_value=failed):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as exit_info:
                        runpy.run_path(str(script), run_name="__main__")

        self.assertEqual(exit_info.exception.code, 2)
        self.assertIn("shader.dxbc: xcl disasm exited with code 2", stderr.getvalue())
        self.assertIn("invalid shader", stderr.getvalue())
        self.assertNotIn("loads on parameter boundaries", stdout.getvalue())
