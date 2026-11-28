import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import coverage_gate


def report(**counts):
    stage = dict(total=10, translated=10, unsupported=0,
                 validator_failed=0, panicked=0, timeout=0)
    stage.update(counts)
    return {"stages": {"CS": stage}}


class CounterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.baseline = Path(self.directory.name) / "baseline.json"
        self.saved = Path(self.directory.name) / "run.json"
        self.baseline.write_text(json.dumps(report()) + "\n", encoding="utf-8")

    def invoke(self, run, *extra):
        self.saved.write_text(json.dumps(run), encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = coverage_gate.main(["--run-json", str(self.saved),
                                       "--baseline", str(self.baseline), *extra])
        return code, out.getvalue(), err.getvalue()

    def test_nan_cannot_hide_a_translation_regression(self):
        code, out, err = self.invoke(report(translated=float("nan")))
        self.assertEqual(code, 2)
        self.assertNotIn("no regressions", out)
        self.assertIn("CS.translated", err)

    def test_invalid_counts_cannot_replace_a_baseline(self):
        original = self.baseline.read_bytes()
        for field in report()["stages"]["CS"]:
            for value in (-1, 0.5, True, False, "10", None, float("nan"),
                          float("inf"), -float("inf")):
                with self.subTest(field=field, value=value):
                    code, out, err = self.invoke(report(**{field: value}), "--update")
                    self.assertEqual(code, 2)
                    self.assertNotIn("baseline written", out)
                    self.assertIn(f"CS.{field}", err)
                    self.assertEqual(self.baseline.read_bytes(), original)

    def test_invalid_counts_cannot_create_a_baseline(self):
        self.baseline.unlink()
        for run in (report(timeout=-1), report(translated=11)):
            with self.subTest(report=run):
                code, _, _ = self.invoke(run, "--update")
                self.assertEqual(code, 2)
                self.assertFalse(self.baseline.exists())

    def test_inconsistent_totals_fail_comparison_and_update(self):
        original = self.baseline.read_bytes()
        for field, value in (("total", 11), ("translated", 11), ("unsupported", 1),
                             ("validator_failed", 1), ("panicked", 1), ("timeout", 1)):
            for extra in ((), ("--update",)):
                with self.subTest(field=field, options=extra):
                    code, out, err = self.invoke(report(**{field: value}), *extra)
                    self.assertEqual(code, 2)
                    self.assertNotIn("no regressions", out)
                    self.assertNotIn("baseline written", out)
                    self.assertIn("CS.total", err)
                    self.assertEqual(self.baseline.read_bytes(), original)

    def test_inconsistent_baseline_total_fails_comparison(self):
        for total in (9, 11):
            with self.subTest(total=total):
                self.baseline.write_text(json.dumps(report(total=total)), encoding="utf-8")
                original = self.baseline.read_bytes()
                code, out, err = self.invoke(report())
                self.assertEqual(code, 2)
                self.assertNotIn("no regressions", out)
                self.assertIn("invalid baseline counters", err)
                self.assertIn("CS.total", err)
                self.assertEqual(self.baseline.read_bytes(), original)

    def test_invalid_baseline_counts_fail_comparison(self):
        for field in report()["stages"]["CS"]:
            with self.subTest(field=field):
                self.baseline.write_text(json.dumps(report(**{field: float("nan")})), encoding="utf-8")
                code, out, err = self.invoke(report())
                self.assertEqual(code, 2)
                self.assertNotIn("no regressions", out)
                self.assertIn("baseline", err)
                self.assertIn(f"CS.{field}", err)

    def test_missing_counter_and_non_object_stage_fail_before_update(self):
        missing = report()
        del missing["stages"]["CS"]["panicked"]
        for invalid in (missing, {"stages": {"CS": []}}):
            with self.subTest(report=invalid):
                original = self.baseline.read_bytes()
                code, _, err = self.invoke(invalid, "--update")
                self.assertEqual(code, 2)
                self.assertIn("CS", err)
                self.assertEqual(self.baseline.read_bytes(), original)

    def test_live_sweep_counts_are_checked_too(self):
        original = self.baseline.read_bytes()
        for run, field in ((report(validator_failed=-1), "validator_failed"),
                           (report(translated=11), "total")):
            with self.subTest(report=run):
                output = "progress\n" + json.dumps(run) + "\n"
                child = subprocess.CompletedProcess([], 0, output, "")
                err = io.StringIO()
                with patch.object(coverage_gate.subprocess, "run", return_value=child), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                    code = coverage_gate.main(["--xcl", "synthetic-xcl", "--rpf", "synthetic-pack",
                                               "--baseline", str(self.baseline), "--update"])
                self.assertEqual(code, 2)
                self.assertIn(f"CS.{field}", err.getvalue())
                self.assertEqual(self.baseline.read_bytes(), original)

    def test_valid_mixed_outcomes_can_be_updated_and_compared(self):
        run = report(total=15, translated=1, unsupported=2, validator_failed=3,
                     panicked=4, timeout=5)
        run["stages"]["CS"]["metadata"] = "not a counter"
        self.assertEqual(self.invoke(run, "--update")[0], 0)
        self.assertEqual(json.loads(self.baseline.read_text()), run)
        self.assertEqual(self.invoke(run)[0], 0)

    def test_valid_zero_and_large_integer_counts_can_be_updated(self):
        for total in (0, 10, 1 << 40):
            with self.subTest(total=total):
                run = report(total=total, translated=total)
                run["all"] = {"metadata": "not stage counters"}
                code, _, err = self.invoke(run, "--update")
                self.assertEqual((code, err), (0, ""))
                self.assertEqual(json.loads(self.baseline.read_text()), {"stages": run["stages"]})

    def test_valid_pass_and_regression_exit_codes_are_preserved(self):
        self.assertEqual(self.invoke(report())[0], 0)
        self.assertEqual(self.invoke(report(translated=9, unsupported=1))[0], 1)


if __name__ == "__main__":
    unittest.main()
