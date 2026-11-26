"""Synthetic artifact fixtures; these bytes are not executable DXIL."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze_shader import prepare, save_report, main


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root / "sample.manifest.json"
        self.data = dict(schema_version=1, backend="direct-dxil", stage="cs", wave64=False,
                         guest_fnv64="a430d84680aabd0b", dxil_fnv64="85944171f73967e8", error=None)
        self.path.write_text(json.dumps(self.data), encoding="utf-8")
        (self.root / "sample.guest.bin").write_bytes(b"hello")
        (self.root / "sample.direct.dxil").write_bytes(b"foobar")
        self.output = self.root / "reports"

    def test_snapshot_and_hashes(self):
        run, report = prepare(self.path, self.output, "a" * 40)
        self.assertEqual(report["category"], "analyzer_not_requested")
        self.assertEqual(report["source_commit"], "a" * 40)
        artifact = report["artifacts"]["sample.direct.dxil"]
        self.assertEqual(artifact["sha256"], hashlib.sha256(b"foobar").hexdigest())
        (self.root / "sample.direct.dxil").write_bytes(b"changed")
        self.assertEqual(Path(artifact["path"]).read_bytes(), b"foobar")
        self.assertEqual(json.loads(save_report(run, report).read_text()), report)

    def test_corruption_and_missing_artifact_fail(self):
        artifact = self.root / "sample.direct.dxil"
        artifact.write_bytes(b"corrupt")
        self.assertEqual(prepare(self.path, self.output)[1]["status"], "ERROR")
        artifact.unlink()
        self.assertEqual(main([str(self.path), "--out", str(self.output)]), 1)

    def test_rejected_translation_preserves_error(self):
        self.data.update(dxil_fnv64=None, error="unsupported\n")
        self.path.write_text(json.dumps(self.data), encoding="utf-8")
        (self.root / "sample.direct.err.txt").write_bytes(b"unsupported\n")
        run, report = prepare(self.path, self.output)
        self.assertEqual(report["category"], "translation_rejected")
        self.assertFalse((run / "input" / "sample.direct.dxil").exists())

    def test_invalid_manifest_and_schema(self):
        for value in ([], {}, dict(self.data, schema_version=True), dict(self.data, backend="other")):
            with self.subTest(value=value):
                self.path.write_text(json.dumps(value), encoding="utf-8")
                self.assertEqual(prepare(self.path, self.output)[1]["status"], "ERROR")

    def test_runs_never_reuse_outputs(self):
        a, _ = prepare(self.path, self.output)
        b, _ = prepare(self.path, self.output)
        self.assertNotEqual(a, b)
        self.assertEqual(main([str(self.path), "--out", str(self.output)]), 0)


if __name__ == "__main__":
    unittest.main()
