"""Synthetic checker inputs, not measured regression runs."""
import unittest

import pr_check


HEAD = "a" * 40
BASE = "b" * 40
RUNTIME = [("crates/xcl/src/lib.rs", "modified", False, lambda: 1)]


def block(commit, exit_code):
    lines = ["```", "xcl-smoke v1", f"commit: {commit}", "title: synthetic-fixture",
             "seconds: 60", "frames shown: 100", "peak commit: 100 MB",
             "device removals: 0", "crashes: 0"]
    if exit_code is not None:
        lines.append(f"exit: {exit_code}")
    return "\n".join(lines + ["```"])


def check(pr_exit, base_exit="0x0", files=RUNTIME):
    body = "**Tested**\nSynthetic input.\n**Not tested**\nNo title run.\n**AI use**\nNone.\n"
    body += block(BASE, base_exit) + "\n" + block(HEAD, pr_exit)
    return pr_check.check(body, HEAD, files, lambda sha: sha == BASE)


class SmokeExitTests(unittest.TestCase):
    def test_zero_exit_is_accepted(self):
        for value in ("0x0", "0x00000000", "0X0"):
            with self.subTest(exit=value):
                self.assertEqual(check(value), [])

    def test_nonzero_exit_is_refused_even_without_logged_crashes(self):
        for value in ("0x1", "0xc0000005", "0xffffffff"):
            with self.subTest(exit=value):
                self.assertTrue(any("exit" in p for p in check(value)))

    def test_missing_or_malformed_exit_is_refused(self):
        for value in (None, "", "n/a", "0x", "0x0 error"):
            with self.subTest(exit=value):
                self.assertTrue(any("exit" in p for p in check(value)))

    def test_timeout_keeps_its_diagnostic(self):
        self.assertIn("the PR's regression run did not end by itself (killed)",
                      check("killed (no exit)"))

    def test_failing_main_is_allowed_when_pr_succeeds(self):
        self.assertEqual(check("0x0", base_exit="0xc0000005"), [])

    def test_nonruntime_change_does_not_require_smoke(self):
        files = [("scripts/example.py", "modified", False, lambda: 1)]
        self.assertEqual(check(None, files=files), [])


if __name__ == "__main__":
    unittest.main()
