import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "re"))
import record_writers


SCRIPT = Path(__file__).resolve().parent / "re" / "record_writers.py"


class RecordWriterTests(unittest.TestCase):
    def test_branch_targets_with_and_without_hex_prefix(self):
        for target in ("180001030 <fallback>", "0x180001030 <fallback>"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                listing = Path(tmp) / "synthetic.asm"
                listing.write_text(
                    "   180001000: mov 0x18(%rcx),%rax\n"
                    "   180001004: cmp 0x30(%rcx),%rax\n"
                    f"   180001008: jae {target}\n"
                    "   18000100a: movl $0x1234,(%rax)\n"
                    "   180001010: add $0x20,%rax\n"
                    "   180001014: mov %rax,0x18(%rcx)\n"
                    "   180001018: ret\n"
                    "   180001030: call *0x40(%rbx)\n",
                    encoding="utf-8",
                )
                result = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), str(listing)],
                    capture_output=True, text=True, check=True,
                )
                self.assertIn("1 record write sites", result.stdout)
                self.assertRegex(
                    result.stdout,
                    r"header imm 1234\s+length 32 B  bound \+0x30  fallback slot 8",
                )

    def test_header_from_list_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / "list_field.asm"
            listing.write_text(
                "   180001000: mov 0x18(%r8),%r9\n"
                "   180001004: cmp 0x28(%r8),%r9\n"
                "   180001008: jae 180001028\n"
                "   18000100a: mov 0x20(%r8),%r10\n"
                "   180001010: add $0x40,%r9\n"
                "   180001014: mov %r9,0x18(%r8)\n"
                "   180001018: ret\n"
                "   180001028: call *0x50(%rbx)\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(listing)],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("1 record write sites", result.stdout)
            self.assertRegex(
                result.stdout,
                r"header list\+20\s+length 64 B  bound \+0x28  fallback slot 10",
            )

    def test_lea_length_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / "lea_len.asm"
            listing.write_text(
                "   180001000: mov 0x18(%rcx),%rax\n"
                "   180001004: cmp 0x30(%rcx),%rax\n"
                "   180001008: jae 180001028\n"
                "   18000100a: movl $0xabcd,(%rax)\n"
                "   180001010: lea 0x38(%rax),%rdx\n"
                "   180001014: mov %rdx,0x18(%rcx)\n"
                "   180001018: ret\n"
                "   180001028: call *0x30(%rbx)\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(listing)],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("1 record write sites", result.stdout)
            self.assertRegex(
                result.stdout,
                r"header imm abcd\s+length 56 B  bound \+0x30  fallback slot 6",
            )

    def test_empty_asm_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / "empty.asm"
            listing.write_text("   180001000: nop\n   180001001: ret\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(listing)],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("0 record write sites", result.stdout)


class CliHardeningTests(unittest.TestCase):
    def test_help_flags_print_usage_and_exit_zero(self):
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                result = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), flag],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0)
                self.assertIn("usage: objdump", result.stdout)
                self.assertIn("record_writers.py", result.stdout)

    def test_no_arguments_prints_usage_and_exits_two(self):
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage: objdump", result.stdout)

    def test_missing_file_reports_error_and_exits_one(self):
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "nonexistent_listing.asm"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("error: file not found: 'nonexistent_listing.asm'", result.stderr)

    def test_python_api_direct_import(self):
        lines = [
            "   180001000: mov 0x18(%rcx),%rax",
            "   180001004: cmp 0x30(%rcx),%rax",
            "   180001008: jae 180001030",
            "   18000100a: movl $0x5678,(%rax)",
            "   180001010: add $0x10,%rax",
            "   180001014: mov %rax,0x18(%rcx)",
            "   180001018: ret",
            "   180001030: call *0x20(%rbx)",
        ]
        sites = record_writers.find_record_writers(lines)
        self.assertEqual(len(sites), 1)
        addr, reg, bound, length, header, slot = sites[0]
        self.assertEqual(addr, 0x180001000)
        self.assertEqual(reg, "rcx")
        self.assertEqual(bound, "30")
        self.assertEqual(length, 16)
        self.assertEqual(header, "imm 5678")
        self.assertEqual(slot, 4)

        buf = io.StringIO()
        record_writers.dump_record_writers(lines, file=buf)
        out = buf.getvalue()
        self.assertIn("1 record write sites", out)
        self.assertIn("header imm 5678", out)

    def test_main_function_programmatic(self):
        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / "prog.asm"
            listing.write_text(
                "   180001000: mov 0x18(%rcx),%rax\n"
                "   180001004: cmp 0x30(%rcx),%rax\n"
                "   180001008: jae 180001030\n"
                "   18000100a: movl $0x9999,(%rax)\n"
                "   180001010: add $0x20,%rax\n"
                "   180001014: mov %rax,0x18(%rcx)\n"
                "   180001018: ret\n"
                "   180001030: call *0x40(%rbx)\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            ret = record_writers.main([str(listing)], out=stdout)
            self.assertEqual(ret, 0)
            self.assertIn("1 record write sites", stdout.getvalue())
            self.assertIn("header imm 9999", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
