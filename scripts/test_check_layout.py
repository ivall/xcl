import contextlib
import importlib.util
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parent / "re" / "check_layout.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_layout_mod", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_dxbc(chunks):
    header_size = 32 + 4 * len(chunks)
    offsets = []
    body = bytearray()
    for cc, payload in chunks:
        offsets.append(header_size + len(body))
        body += cc.encode("latin1") + struct.pack("<I", len(payload)) + payload
    total = header_size + len(body)
    header = struct.pack("<4s16sIII", b"DXBC", bytes(16), 1, total, len(chunks))
    for off in offsets:
        header += struct.pack("<I", off)
    return bytes(header + body)


def make_rts0(params):
    """Build an RTS0 chunk from entries: ('consts', reg, num) or ('CBV'|'SRV'|'UAV', reg)."""
    kind_map = {"consts": 1, "CBV": 2, "SRV": 3, "UAV": 4}
    poff = 24
    table = bytearray()
    payloads = bytearray()
    pay_base = poff + 12 * len(params)
    for entry in params:
        kind = entry[0]
        ty = kind_map[kind]
        table += struct.pack("<3I", ty, 0, pay_base + len(payloads))
        if ty == 1:
            _, reg, num = entry
            payloads += struct.pack("<3I", reg, 0, num)
        else:
            _, reg = entry
            payloads += struct.pack("<2I", reg, 0)
    return bytes(struct.pack("<6I", 1, len(params), poff, 0, 0, 0) + table + payloads)


class CheckLayoutPartsAndLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_parts_and_layout_with_rts0(self):
        rts0 = make_rts0([("consts", 0, 3), ("CBV", 1), ("SRV", 0), ("UAV", 2)])
        blob = make_dxbc([("SHDR", b"\x01\x02\x03\x04"), ("RTS0", rts0)])
        parsed = self.mod.parts(blob)
        self.assertEqual([cc for cc, _ in parsed], ["SHDR", "RTS0"])
        rs = next((p for cc, p in parsed if cc == "RTS0"), None)
        consts, descs, total = self.mod.layout(rs)
        self.assertEqual(consts, [(0, 12)])
        self.assertEqual(descs, {12: "CBV", 20: "SRV", 28: "UAV"})
        self.assertEqual(total, 36)

    def test_parts_and_layout_without_rts0(self):
        blob = make_dxbc([("SHDR", b"\xaa\xbb")])
        parsed = self.mod.parts(blob)
        self.assertEqual(parsed, [("SHDR", b"\xaa\xbb")])
        rs = next((p for cc, p in parsed if cc == "RTS0"), None)
        self.assertIsNone(rs)
        self.assertIsNone(self.mod.layout(rs))

    def test_parts_rejects_non_dxbc_and_truncated_containers(self):
        with self.subTest("too short"):
            with self.assertRaises(ValueError):
                self.mod.parts(b"DXBC" + b"\x00" * 10)
        with self.subTest("bad magic"):
            with self.assertRaises(ValueError):
                self.mod.parts(b"NOPE" + b"\x00" * 40)
        with self.subTest("truncated offset table"):
            bad = struct.pack("<4s16sIII", b"DXBC", bytes(16), 1, 32, 4)
            with self.assertRaises(ValueError):
                self.mod.parts(bad)
        with self.subTest("chunk offset out of bounds"):
            bad = struct.pack("<4s16sIII", b"DXBC", bytes(16), 1, 40, 1) + struct.pack("<I", 100)
            with self.assertRaises(ValueError):
                self.mod.parts(bad)
        with self.subTest("truncated chunk payload"):
            bad = (
                struct.pack("<4s16sIII", b"DXBC", bytes(16), 1, 44, 1)
                + struct.pack("<I", 36)
                + b"RTS0"
                + struct.pack("<I", 64)
                + b"\x00\x00"
            )
            with self.assertRaises(ValueError):
                self.mod.parts(bad)


class CheckLayoutDirectoryAndCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def _run_main(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = self.mod.main(argv)
        return rc, stdout.getvalue(), stderr.getvalue()

    def test_directory_with_subdir_non_dxbc_and_no_rts0_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xcl = root / "xcl.exe"
            xcl.write_bytes(b"")
            (root / "subdir").mkdir()
            (root / "notes.txt").write_text("not a shader", encoding="utf-8")
            (root / "corrupt.dxbc").write_bytes(b"not dxbc bytes")
            (root / "no_rts0.dxbc").write_bytes(make_dxbc([("SHDR", b"\x00\x01\x02\x03")]))

            with patch.object(self.mod.subprocess, "run") as mock_run:
                rc, out, err = self._run_main([str(root), "--xcl", str(xcl)])

            mock_run.assert_not_called()
            self.assertEqual(rc, 0)
            self.assertIn("loads on parameter boundaries: 0, off: 0", out)
            self.assertIn("corrupt.dxbc", err)

    def test_disassembler_nonzero_exit_returns_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xcl = root / "xcl.exe"
            xcl.write_bytes(b"")
            rts0 = make_rts0([("CBV", 0)])
            (root / "00.dxbc").write_bytes(make_dxbc([("RTS0", rts0)]))

            failed = subprocess.CompletedProcess(
                args=[str(xcl), "disasm"],
                returncode=2,
                stdout="",
                stderr="xcl: invalid shader container\n",
            )
            with patch.object(self.mod.subprocess, "run", return_value=failed):
                rc, _, err = self._run_main([str(root), "--xcl", str(xcl)])

            self.assertEqual(rc, 1)
            self.assertIn("xcl: invalid shader container", err)
            self.assertIn("disassembler errors: 1", err)

    def test_missing_xcl_returns_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_xcl = Path(tmp) / "missing_xcl.exe"
            rc, _, err = self._run_main([tmp, "--xcl", str(missing_xcl)])
            self.assertEqual(rc, 1)
            self.assertIn("xcl not found", err)

    def test_on_boundary_vs_off_boundary_loads(self):
        rts0 = make_rts0([("consts", 0, 2), ("CBV", 1), ("SRV", 0)])
        dxbc_bytes = make_dxbc([("RTS0", rts0)])
        # Layout: consts [0, 8), CBV at 8 (8 B), SRV at 16 (8 B)
        on_boundary_asm = (
            "  s_mov_b32 s4, s0\n"
            "  s_and_b32 s5, s1, 0xff\n"
            "  s_load_dword s6, s[4:5], 0x0\n"
            "  s_load_dword s7, s[4:5], 0x4\n"
            "  s_load_dwordx2 s[8:9], s[4:5], 0x8\n"
            "  s_load_dword s10, s[4:5], 0x14\n"
        )
        off_boundary_asm = (
            "  s_mov_b32 s4, s0\n"
            "  s_and_b32 s5, s1, 0xff\n"
            "  s_load_dword s6, s[4:5], 0x8\n"
            "  s_load_dword s7, s[4:5], 0x18\n"
        )

        for label, listing, expected_rc, expected_summary in (
            ("on_boundary", on_boundary_asm, 0, "loads on parameter boundaries: 4, off: 0"),
            ("off_boundary", off_boundary_asm, 1, "loads on parameter boundaries: 1, off: 1"),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                xcl = root / "xcl.exe"
                xcl.write_bytes(b"")
                (root / "00.dxbc").write_bytes(dxbc_bytes)
                completed = subprocess.CompletedProcess(
                    args=[str(xcl), "disasm"],
                    returncode=0,
                    stdout=listing,
                    stderr="",
                )
                with patch.object(self.mod.subprocess, "run", return_value=completed):
                    rc, out, _ = self._run_main([str(root), "--xcl", str(xcl)])
                self.assertEqual(rc, expected_rc)
                self.assertIn(expected_summary, out)
                if expected_rc == 1:
                    self.assertIn("0x18 not on a parameter", out)

    def test_missing_cli_arg_and_nonexistent_dir_exit_nonzero(self):
        missing_arg = subprocess.run(
            [sys.executable, "-B", str(SCRIPT)],
            capture_output=True, text=True,
        )
        self.assertEqual(missing_arg.returncode, 2)

        with tempfile.TemporaryDirectory() as tmp:
            bad_dir = Path(tmp) / "does_not_exist"
            nonexistent = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(bad_dir)],
                capture_output=True, text=True,
            )
            self.assertEqual(nonexistent.returncode, 2)
            self.assertIn("not a directory", nonexistent.stderr)


if __name__ == "__main__":
    unittest.main()
