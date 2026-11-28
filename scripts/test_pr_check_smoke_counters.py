"""Synthetic checker inputs, not measured title runs."""
import unittest

import pr_check


HEAD = "a" * 40
MAIN = "b" * 40
COUNTERS = ("device removals", "crashes")
RUNTIME = [("crates/xcl/src/lib.rs", "modified", False, lambda: 1)]
SECTIONS = "**Tested**\nSynthetic input.\n**Not tested**\nNo title run.\n**AI use**\nNone.\n"


def block(commit, overrides=None):
    fields = {"commit": commit, "title": "synthetic-counter-fixture", "seconds": "60",
              "exit": "0x0", "frames shown": "100 (PresentX 100)", "peak commit": "100 MB",
              "device removals": "0", "crashes": "0"}
    fields.update(overrides or {})
    lines = ["```", "xcl-smoke v1"]
    lines.extend(f"{key}: {value}" for key, value in fields.items() if value is not None)
    return "\n".join(lines + ["```", ""])


def problems(pr=None, main=None):
    body = SECTIONS + block(MAIN, main) + block(HEAD, pr)
    return pr_check.check(body, HEAD, RUNTIME, lambda sha: sha == MAIN)


class SmokeCounterTests(unittest.TestCase):
    def test_zero_counters_are_accepted(self):
        for value in ("0", "000", " 0 ", "0" * 5000):
            with self.subTest(value=value[:20]):
                self.assertEqual(problems(dict.fromkeys(COUNTERS, value)), [])

    def test_each_positive_counter_is_refused(self):
        for key in COUNTERS:
            for value in ("1", "0002", "9" * 5000):
                with self.subTest(key=key, value=value[:20]):
                    self.assertTrue(any(f"shows {key}:" in p for p in problems({key: value})))

    def test_each_missing_or_empty_counter_is_refused(self):
        for key in COUNTERS:
            for value in (None, "", " "):
                with self.subTest(key=key, value=value):
                    self.assertTrue(any(key in p for p in problems({key: value})))

    def test_each_malformed_counter_is_refused(self):
        for key in COUNTERS:
            for value in ("unknown", "n/a", "-1", "+0", "0x0", "0.0", "0 trailing", "\u0660"):
                with self.subTest(key=key, value=value):
                    self.assertTrue(any(key in p for p in problems({key: value})))

    def test_a_crashing_main_baseline_remains_allowed(self):
        self.assertEqual(problems(main={"exit": "0xc0000005", "crashes": "2", "device removals": "1"}), [])

    def test_runtime_changes_still_require_both_reports(self):
        result = pr_check.check(SECTIONS, HEAD, RUNTIME, lambda sha: sha == MAIN)
        self.assertTrue(any(HEAD[:12] in p for p in result))
        self.assertTrue(any("commit of main" in p for p in result))

    def test_scripts_do_not_require_a_title_report(self):
        files = [("scripts/pr_check.py", "modified", False, lambda: 1)]
        self.assertEqual(pr_check.check(SECTIONS, HEAD, files, lambda sha: False), [])


if __name__ == "__main__":
    unittest.main()
