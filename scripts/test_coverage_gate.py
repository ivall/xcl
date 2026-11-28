import json
import tempfile
import unittest
from pathlib import Path

from coverage_gate import compare, main, sweep_json

STAGES = {
    "CS": {"total": 100, "translated": 95, "unsupported": 5, "validator_failed": 0, "panicked": 0, "timeout": 0},
    "PS": {"total": 200, "translated": 200, "unsupported": 0, "validator_failed": 0, "panicked": 0, "timeout": 0},
}


def run(**over):
    stages = json.loads(json.dumps(STAGES))
    for stage, patch in over.items():
        stages[stage].update(patch)
    return {"stages": stages}


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {"stages": STAGES}

    def test_identical_sweep_passes(self):
        bad, notes = compare(self.baseline, run())
        self.assertEqual((bad, notes), ([], []))

    def test_translated_drop_is_a_regression(self):
        bad, _ = compare(self.baseline, run(PS={"translated": 168, "unsupported": 32}))
        self.assertEqual(bad, [("PS", "translated 168 < baseline 200")])

    def test_new_validator_failure_panics_and_timeouts_are_regressions(self):
        bad, _ = compare(self.baseline, run(CS={"validator_failed": 2, "panicked": 1, "timeout": 3}))
        self.assertEqual(
            sorted(bad),
            [("CS", "panicked 1 > baseline 0"), ("CS", "timeout 3 > baseline 0"), ("CS", "validator_failed 2 > baseline 0")],
        )

    def test_more_translated_and_fewer_unsupported_pass(self):
        bad, _ = compare(self.baseline, run(CS={"translated": 98, "unsupported": 1}))
        self.assertEqual(bad, [])

    def test_total_change_is_a_note_not_a_regression(self):
        bad, notes = compare(self.baseline, run(PS={"total": 210, "translated": 210}))
        self.assertEqual(bad, [])
        self.assertEqual(notes, ["PS: total 200 -> 210 (corpus changed)"])

    def test_missing_stage_is_a_regression(self):
        stages = {"stages": {"CS": STAGES["CS"]}}
        bad, _ = compare(self.baseline, stages)
        self.assertEqual(bad, [("PS", "stage missing from the sweep")])

    def test_new_stage_is_a_note(self):
        r = run()
        r["stages"]["MS"] = {"total": 3, "translated": 3, "unsupported": 0, "validator_failed": 0, "panicked": 0, "timeout": 0}
        bad, notes = compare(self.baseline, r)
        self.assertEqual(bad, [])
        self.assertEqual(notes, ["MS: new stage in the sweep"])


class SweepJsonTests(unittest.TestCase):
    def test_takes_the_last_json_line(self):
        out = "some progress\n" + json.dumps({"stages": STAGES, "all": {}}) + "\n"
        self.assertEqual(sweep_json(out)["stages"], STAGES)

    def test_raises_without_a_report(self):
        with self.assertRaises(ValueError):
            sweep_json("no json here")


class CliGateTests(unittest.TestCase):
    def test_missing_all_arguments_exits_two(self):
        with self.assertRaises(SystemExit) as cm:
            main([])
        self.assertEqual(cm.exception.code, 2)

    def test_missing_rpf_when_xcl_given_exits_two(self):
        with self.assertRaises(SystemExit) as cm:
            main(["--xcl", "fake_xcl.exe"])
        self.assertEqual(cm.exception.code, 2)

    def test_run_json_not_found_exits_two(self):
        ret = main(["--run-json", "nonexistent_file.json"])
        self.assertEqual(ret, 2)

    def test_run_json_invalid_json_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("NOT_JSON", encoding="utf-8")
            ret = main(["--run-json", str(bad)])
            self.assertEqual(ret, 2)

    def test_run_json_missing_stages_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"wrong": 123}), encoding="utf-8")
            ret = main(["--run-json", str(bad)])
            self.assertEqual(ret, 2)

    def test_xcl_binary_not_found_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            rpf = Path(tmp) / "pack.rpf"
            rpf.write_bytes(b"dummy")
            ret = main(["--xcl", "nonexistent_xcl.exe", "--rpf", str(rpf)])
            self.assertEqual(ret, 2)

    def test_rpf_file_not_found_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            xcl = Path(tmp) / "dummy_xcl.exe"
            xcl.write_bytes(b"dummy")
            ret = main(["--xcl", str(xcl), "--rpf", "nonexistent.rpf"])
            self.assertEqual(ret, 2)

    def test_baseline_not_found_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_file = Path(tmp) / "run.json"
            run_file.write_text(json.dumps({"stages": STAGES}), encoding="utf-8")
            ret = main(["--run-json", str(run_file), "--baseline", str(Path(tmp) / "absent_baseline.json")])
            self.assertEqual(ret, 2)

    def test_baseline_invalid_json_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_file = Path(tmp) / "run.json"
            run_file.write_text(json.dumps({"stages": STAGES}), encoding="utf-8")
            bad_base = Path(tmp) / "bad_base.json"
            bad_base.write_text("INVALID_JSON", encoding="utf-8")
            ret = main(["--run-json", str(run_file), "--baseline", str(bad_base)])
            self.assertEqual(ret, 2)

    def test_baseline_missing_stages_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_file = Path(tmp) / "run.json"
            run_file.write_text(json.dumps({"stages": STAGES}), encoding="utf-8")
            bad_base = Path(tmp) / "bad_base.json"
            bad_base.write_text(json.dumps({"other": {}}), encoding="utf-8")
            ret = main(["--run-json", str(run_file), "--baseline", str(bad_base)])
            self.assertEqual(ret, 2)

    def test_update_creates_baseline_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_file = Path(tmp) / "run.json"
            run_file.write_text(json.dumps({"stages": STAGES}), encoding="utf-8")
            new_base = Path(tmp) / "sub" / "baseline.json"
            ret = main(["--run-json", str(run_file), "--baseline", str(new_base), "--update"])
            self.assertEqual(ret, 0)
            self.assertTrue(new_base.is_file())
            content = json.loads(new_base.read_text(encoding="utf-8"))
            self.assertEqual(content, {"stages": STAGES})

    def test_pass_exit_code_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_file = Path(tmp) / "run.json"
            run_file.write_text(json.dumps({"stages": STAGES}), encoding="utf-8")
            base_file = Path(tmp) / "baseline.json"
            base_file.write_text(json.dumps({"stages": STAGES}), encoding="utf-8")
            ret = main(["--run-json", str(run_file), "--baseline", str(base_file)])
            self.assertEqual(ret, 0)

    def test_regression_exit_code_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_run = run(PS={"translated": 100})
            run_file = Path(tmp) / "run.json"
            run_file.write_text(json.dumps(bad_run), encoding="utf-8")
            base_file = Path(tmp) / "baseline.json"
            base_file.write_text(json.dumps({"stages": STAGES}), encoding="utf-8")
            ret = main(["--run-json", str(run_file), "--baseline", str(base_file)])
            self.assertEqual(ret, 1)


if __name__ == "__main__":
    unittest.main()
