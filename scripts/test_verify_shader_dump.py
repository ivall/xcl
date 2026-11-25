import json
from pathlib import Path
import tempfile
import unittest

from verify_shader_dump import verify


class ShaderDumpTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.prefix = Path(tmp.name) / "0000000000000001_f0_24"
        self.path = self.prefix.with_suffix(".manifest.json")
        self.manifest = dict(schema_version=1, backend="direct-dxil", guest_fnv64="a430d84680aabd0b",
                             dxil_fnv64="85944171f73967e8", error=None)
        self.prefix.with_suffix(".guest.bin").write_bytes(b"hello")
        self.prefix.with_suffix(".direct.dxil").write_bytes(b"foobar")

    def check(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return verify(self.path)

    def test_translation(self):
        self.assertEqual(self.check(), "guest and translated DXIL match")
        self.prefix.with_suffix(".direct.dxil").write_bytes(b"foo")
        with self.assertRaisesRegex(ValueError, "dxil_fnv64 mismatch"):
            self.check()

    def test_hashes_every_binary_chunk(self):
        self.prefix.with_suffix(".guest.bin").write_bytes(bytes(range(256)) * 512 + b"\0")
        self.manifest["guest_fnv64"] = "9fd6a807c893b7df"
        self.assertEqual(self.check(), "guest and translated DXIL match")

    def test_error_uses_exact_bytes_even_when_old_dxil_is_present(self):
        error = "first\nsecond: \u0394"
        self.manifest.update(dxil_fnv64=None, error=error)
        diagnostic = self.prefix.with_suffix(".direct.err.txt")
        diagnostic.write_bytes(error.encode("utf-8"))
        self.assertEqual(self.check(), "guest and translation error match")
        diagnostic.write_bytes(error.replace("\n", "\r\n").encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "error text mismatch"):
            self.check()

    def test_guest_changed_or_missing(self):
        guest = self.prefix.with_suffix(".guest.bin")
        guest.write_bytes(b"HELLO")
        with self.assertRaisesRegex(ValueError, "guest_fnv64 mismatch"):
            self.check()
        guest.unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_version_and_ambiguous_result(self):
        for change in (dict(schema_version=True), dict(schema_version=2), dict(backend="other"),
                       dict(error="also an error"), dict(dxil_fnv64=None)):
            with self.subTest(change=change):
                before = dict(self.manifest)
                self.manifest.update(change)
                with self.assertRaises(ValueError):
                    self.check()
                self.manifest = before
