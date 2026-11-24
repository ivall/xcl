"""Synthetic report/log regressions; no game files, GPU, or third-party packages."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from triage_d3d12 import analyze, render


IID = "12345678-1234-5678-9abc-123456789abc"
LOG = ("[12:34:56 WARN] [STUB] d3d12_xs.dll!ID3D12Device::#200 (not on PC) "
       "first called from synthetic.exe+0x10 → returning 0x0")


def stub(calls=1, caller="synthetic.exe+0x10", variant="not on PC", dll="d3d12_xs.dll"):
    suffix = f" ({variant})" if variant else ""
    return dict(dll=dll, function=f"ID3D12Device::#200{suffix}", calls=calls, first_caller=caller)


class TriageTests(unittest.TestCase):
    def test_guard_duplicates_sum_report_counts_and_keep_callers(self):
        result = analyze({"called_stubs": [stub(2), stub(3, "synthetic.exe+0x20")]})
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(len(result["slots"]), 1)
        self.assertEqual(result["slots"][0]["calls"], 5)
        self.assertEqual(result["slots"][0]["first_callers"], ["synthetic.exe+0x10", "synthetic.exe+0x20"])

    def test_log_duplicates_do_not_inflate_report_count(self):
        slot = analyze({"called_stubs": [stub(4)]}, [LOG, LOG])["slots"][0]
        self.assertEqual(slot["calls"], 4)
        self.assertEqual(slot["sources"], ["log", "report"])

    def test_log_only_has_unknown_count_and_unicode_caller(self):
        result = analyze(None, [LOG.replace("synthetic.exe", "合成.exe")])
        self.assertIsNone(result["slots"][0]["calls"])
        self.assertEqual(result["slots"][0]["first_callers"], ["合成.exe+0x10"])
        self.assertTrue(result["warnings"])

    def test_variants_and_null_backend_are_distinct(self):
        rows = [stub(variant=v) for v in ("", "not on PC", "older layout", "newer layout")]
        slots = analyze({"called_stubs": rows})["slots"]
        self.assertEqual([s["variant"] for s in slots], ["", "newer layout", "not on PC", "older layout"])

    def test_dll_normalization_and_unrelated_stubs(self):
        rows = [stub(dll="D3D12_XS.DLL"), stub(dll="kernelx.dll"),
                dict(stub(), function="ID3D12Device::CreateHeap")]
        slots = analyze({"called_stubs": rows})["slots"]
        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["dll"], "d3d12_xs.dll")

    def test_slot_numbers_are_scoped_to_the_interface(self):
        rows = [stub(), dict(stub(), function="ID3D12CommandQueue::#200 (not on PC)")]
        slots = analyze({"called_stubs": rows})["slots"]
        self.assertEqual([s["interface"] for s in slots], ["ID3D12CommandQueue", "ID3D12Device"])

    def test_bad_count_warns_and_preserves_valid_records(self):
        result = analyze({"called_stubs": [stub(-1), stub(True), stub("2"), stub(0), stub(2**64), stub(7)]})
        self.assertTrue(any("call count" in warning for warning in result["warnings"]))
        self.assertEqual(result["slots"][0]["calls"], 7)

    def test_iids_normalize_deduplicate_and_merge_sources(self):
        log = f"D3D12.X: QueryInterface({IID}) – interface unknown on the PC (console-only?)"
        result = analyze({"d3d12": {"unknown_iids": [IID.upper(), IID]}}, [log, log])
        self.assertEqual(result["unknown_iids"], [{"iid": IID, "sources": ["log", "report"]}])

    def test_missing_optional_fields_and_malformed_entries(self):
        self.assertEqual(analyze({})["slots"], [])
        result = analyze({"called_stubs": [None, stub()], "d3d12": {"unknown_iids": [None, "invalid", IID]}})
        self.assertEqual(len(result["slots"]), 1)
        self.assertEqual(len(result["unknown_iids"]), 1)
        self.assertTrue(result["warnings"])

    def test_input_order_does_not_change_findings(self):
        rows = [stub(2, "b.exe+0x20"), stub(3, "a.exe+0x10"), stub(variant="older layout")]
        self.assertEqual(analyze({"called_stubs": rows}, [LOG]),
                         analyze({"called_stubs": rows[::-1]}, [LOG]))

    def test_cli_falls_back_from_missing_truncated_and_non_object_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            report, log = Path(directory) / "report.json", Path(directory) / "xcl.log"
            log.write_text(LOG + "\n", encoding="utf-8")
            for content in (None, '{"called_stubs":', "[]", "[" * 10000 + "0" + "]" * 10000):
                with self.subTest(content=content):
                    if content is not None:
                        report.write_text(content, encoding="utf-8")
                    proc = self.run_cli(report, "--log", log, "--json")
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertIsNone(json.loads(proc.stdout)["slots"][0]["calls"])
                    self.assertTrue(proc.stderr.strip())

    def test_cli_missing_inputs_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = self.run_cli(Path(directory) / "missing.json", "--log", Path(directory) / "missing.log", "--json")
            self.assertEqual(proc.returncode, 2)
            self.assertTrue(proc.stderr.strip())
            self.assertNotIn("Traceback", proc.stderr)

    def test_bad_log_lines_warn_without_hiding_valid_evidence(self):
        result = analyze(None, [LOG, LOG.split(" first called")[0],
                               "D3D12.X: QueryInterface(new format) interface unknown", "\ufffd"])
        self.assertEqual(len(result["slots"]), 1)
        self.assertEqual(len(result["warnings"]), 4)

    def test_cli_report_survives_missing_optional_log(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text(json.dumps({"called_stubs": [stub(9)], "d3d12": {"unknown_iids": []}}),
                              encoding="utf-8-sig")
            proc = self.run_cli(report, "--log", Path(directory) / "absent.log", "--json")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["slots"][0]["calls"], 9)
            self.assertIn("Cannot use log", proc.stderr)

    def test_text_output_does_not_claim_support_or_causation(self):
        self.assertIn("does not establish interface support", render(analyze({})))
        text = render(analyze(None, [LOG]))
        self.assertIn("not proven crash causes", text)
        self.assertIn("calls=unknown", text)
        self.assertIn("first caller:", text)

    @staticmethod
    def run_cli(*args):
        return subprocess.run([sys.executable, str(Path(__file__).with_name("triage_d3d12.py")),
                               *map(str, args)], capture_output=True, text=True, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
