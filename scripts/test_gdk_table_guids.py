from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class GdkTableGuidTests(unittest.TestCase):
    def test_invalid_guid_does_not_replace_generated_table(self):
        source_script = Path(__file__).resolve().with_name("gen_gdk_table.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "scripts" / source_script.name
            script.parent.mkdir()
            shutil.copyfile(source_script, script)
            output = root / "crates" / "xcl" / "src" / "hle" / "xgameruntime" / "gdk_table.rs"
            output.parent.mkdir(parents=True)
            source = root / "input.txt"
            command = [sys.executable, "-B", str(script), str(source)]
            row = 'NAME({}, "{}", "{}", 3)\n'
            class_guid = "00000000-0000-0000-0000-000000000001"
            interface_guid = "ABCDEF00-0000-0000-0000-000000000002"
            valid = row.format("Example", class_guid, interface_guid)
            source.write_text(valid, encoding="utf-8")
            subprocess.run(command, capture_output=True, encoding="utf-8", check=True)
            generated = output.read_bytes()
            self.assertIn(b"0xabcdef00000000000000000000000002", generated)

            for cls, iface, field in (
                ("f" * 36, interface_guid, "class"),
                ("bad//guid", interface_guid, "class"),
                (class_guid, "bad/*guid*/", "interface"),
                (class_guid, interface_guid[:-1], "interface"),
                (class_guid, interface_guid.replace("-", "", 1) + "-", "interface"),
            ):
                with self.subTest(field=field, cls=cls, iface=iface):
                    source.write_text(valid + row.format("Invalid", cls, iface), encoding="utf-8")
                    result = subprocess.run(command, capture_output=True, encoding="utf-8")
                    self.assertNotEqual(0, result.returncode)
                    self.assertIn(f"Invalid: invalid {field} GUID", result.stderr)
                    self.assertEqual(generated, output.read_bytes())
