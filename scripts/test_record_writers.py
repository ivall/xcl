from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parent / "re" / "record_writers.py"


class RecordWriterTests(unittest.TestCase):
    def test_llvm_att_listing_matches_writer_at_image_address(self):
        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / "llvm.asm"
            listing.write_text(
                "140001000: movq\t0x18(%rcx), %rax\n"
                "140001004: cmpq\t0x30(%rcx), %rax\n"
                "140001008: jae\t0x140001019 <fallback>\n"
                "14000100a: movl\t$0x1234, (%rax)  # imm = 0x1234\n"
                "140001010: addq\t$0x20, %rax\n"
                "140001014: movq\t%rax, 0x18(%rcx)\n"
                "140001018: retq\n"
                "140001019: callq\t*0x40(%rbx)\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(listing)],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("1 record write sites", result.stdout)
            self.assertRegex(result.stdout, r"header imm 1234\s+length 32 B  bound \+0x30  fallback slot 8")

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


if __name__ == "__main__":
    unittest.main()
