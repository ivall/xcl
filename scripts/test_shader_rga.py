"""Synthetic subprocess results; these tests do not establish AMD compiler coverage."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from shader_rga import analyze, execute


class RgaTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = Path(tmp.name)
        self.report = dict(category="analyzer_not_requested", status="SKIPPED",
                           manifest={"stage": "CS"}, artifacts={
                               "shader.direct.dxil": {"path": str(self.run / "shader.direct.dxil")}})

    def fake(self, command, directory, label, timeout):
        (directory / f"{label}.stdout.txt").write_text(
            "RGA test fixture" if label == "version" else "--offline --cs-blob", encoding="utf-8")
        if label == "analysis":
            (directory / "gfx1101_shader_cs.isa").write_text("synthetic ISA", encoding="utf-8")
            (directory / "gfx1101_stats_cs.csv").write_text("DEVICE,VGPRs\ngfx1101,16\n", encoding="utf-8")
        return dict(command=command, exit_code=0, timed_out=False, wall_seconds=0.01)

    def test_success_keeps_exact_argv_and_raw_metrics(self):
        with patch("shader_rga.shutil.which", return_value="C:/Tools/RGA/rga.exe"), patch("shader_rga.execute", side_effect=self.fake):
            analyze(self.run, self.report, "rga", "gfx1101")
        self.assertEqual(self.report["status"], "OK")
        info = self.report["analyzer"]
        self.assertEqual(info["statistics"][0]["rows"][0]["VGPRs"], "16")
        command = info["invocations"][-1]["command"]
        self.assertEqual(command[command.index("--cs-blob") + 1], str(self.run / "shader.direct.dxil"))
        self.assertIn("--offline", command)

    def test_absent_tool_skips(self):
        with patch("shader_rga.shutil.which", return_value=None):
            analyze(self.run, self.report, "missing.exe", "gfx1101")
        self.assertEqual(self.report["category"], "tool_missing")

    def test_unsupported_stage_never_launches(self):
        self.report["manifest"]["stage"] = "GS"
        with patch("shader_rga.execute") as launch:
            analyze(self.run, self.report, "rga", "gfx1101")
            launch.assert_not_called()
        self.assertEqual(self.report["category"], "unsupported_stage")

    def test_invalid_bundle_never_launches(self):
        self.report.update(category="invalid_bundle", status="ERROR")
        with patch("shader_rga.execute") as launch:
            analyze(self.run, self.report, "rga", "gfx1101")
            launch.assert_not_called()
        self.assertEqual(self.report["category"], "invalid_bundle")

    def test_zero_exit_without_output_is_failure(self):
        def no_output(command, directory, label, timeout):
            result = self.fake(command, directory, label, timeout)
            if label == "analysis":
                (directory / "gfx1101_shader_cs.isa").unlink()
            return result
        with patch("shader_rga.shutil.which", return_value="C:/Tools/RGA/rga.exe"), patch("shader_rga.execute", side_effect=no_output):
            analyze(self.run, self.report, "rga", "gfx1101")
        self.assertEqual(self.report["category"], "analyzer_failed")

    def test_bad_target_is_error(self):
        analyze(self.run, self.report, "rga", "--anything")
        self.assertEqual(self.report["category"], "invalid_target")

    def test_real_subprocess_exit_and_logs(self):
        result = execute([sys.executable, "-c", "import sys; print('output'); sys.exit(7)"], self.run, "child", 5)
        self.assertEqual(result["exit_code"], 7)
        self.assertIn("output", Path(result["stdout"]).read_text())

    def test_real_subprocess_timeout(self):
        result = execute([sys.executable, "-c", "import time; time.sleep(30)"], self.run, "child", 0.1)
        self.assertTrue(result["timed_out"])
        self.assertIsNotNone(result["exit_code"])


if __name__ == "__main__":
    unittest.main()
