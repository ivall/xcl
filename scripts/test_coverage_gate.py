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

    def test_new_stage_with_validator_failure_panic_or_timeout_is_a_regression(self):
        r = run()
        r["stages"]["HS"] = {"total": 6, "translated": 0, "unsupported": 0, "validator_failed": 2, "panicked": 1, "timeout": 3}
        bad, notes = compare(self.baseline, r)
        self.assertEqual(
            sorted(bad),
            [
                ("HS", "panicked 1 (stage not in baseline)"),
                ("HS", "timeout 3 (stage not in baseline)"),
                ("HS", "validator_failed 2 (stage not in baseline)"),
            ],
        )
        self.assertEqual(notes, ["HS: new stage in the sweep"])
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "baseline.json"
            run_path = Path(tmp) / "run.json"
            base_path.write_text(json.dumps(self.baseline), encoding="utf-8")
            run_path.write_text(json.dumps(r), encoding="utf-8")
            self.assertEqual(main(["--baseline", str(base_path), "--run-json", str(run_path)]), 1)

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


if __name__ == "__main__":
    unittest.main()
