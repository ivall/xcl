from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().with_name("gen_gdk_table.py")


class GdkTableCommentTests(unittest.TestCase):
    def test_commented_rows_do_not_become_api_entries(self):
        row = 'NAME({}, "00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002", 3)'
        malformed = 'NAME({}, "bad//guid", "00000000-0000-0000-0000-000000000002", 3)'
        origin = "https://example.invalid/gdk"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "scripts" / SCRIPT.name
            script.parent.mkdir()
            shutil.copyfile(SCRIPT, script)
            output = root / "crates" / "xcl" / "src" / "hle" / "xgameruntime" / "gdk_table.rs"
            output.parent.mkdir(parents=True)
            source = root / "input.txt"
            source.write_text(
                f"// Source: {origin}\n"
                f"{row.format('Included')} // active entry\n"
                f'"not//a comment" {row.format("AfterQuotedSlashes")}\n'
                f"// {row.format('DisabledLine')}\n"
                f"// {malformed.format('DisabledMalformed')} {row.format('StillDisabled')}\n"
                f"/* temporary exclusion\n{row.format('DisabledBlock')}\n*/\n",
                encoding="utf-8",
            )
            subprocess.run([sys.executable, "-B", str(script), str(source)], capture_output=True, check=True)
            generated = output.read_text(encoding="utf-8")
            self.assertIn(origin, generated)
            self.assertIn('"Included"', generated)
            self.assertIn('"AfterQuotedSlashes"', generated)
            self.assertNotIn('"DisabledLine"', generated)
            self.assertNotIn('"DisabledBlock"', generated)
            self.assertNotIn('"StillDisabled"', generated)


if __name__ == "__main__":
    unittest.main()
