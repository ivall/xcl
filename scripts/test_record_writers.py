from itertools import product
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parent / "re" / "record_writers.py"


class RecordWriterTests(unittest.TestCase):
    def test_branch_target_and_header_destination_spellings(self):
        targets = ("180001030 <fallback>", "0x180001030 <fallback>")
        destinations = (("(%rax)", "imm 1234"), ("0x0(%rax)", "imm 1234"), ("0x4(%rax)", r"\?"))
        for target, (destination, header) in product(targets, destinations):
            with self.subTest(target=target, destination=destination), tempfile.TemporaryDirectory() as tmp:
                listing = Path(tmp) / "synthetic.asm"
                listing.write_text(
                    "   180001000: mov 0x18(%rcx),%rax\n"
                    "   180001004: cmp 0x30(%rcx),%rax\n"
                    f"   180001008: jae {target}\n"
                    f"   18000100a: movl $0x1234,{destination}\n"
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
                    rf"header {header}\s+length 32 B  bound \+0x30  fallback slot 8",
                )


if __name__ == "__main__":
    unittest.main()
