from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parent / "gen_bcpack_victory_tables.py"
TABLE_BYTES = 77_982


class BcpackTableGeneratorTests(unittest.TestCase):
    def test_trailing_byte_does_not_replace_output_with_optimization_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "tables.bin"
            output = folder / "tables.rs"
            original = b"existing output"
            output.write_bytes(original)
            source.write_bytes(bytes(TABLE_BYTES) + b"x")

            result = subprocess.run(
                [sys.executable, "-O", "-I", str(SCRIPT), str(source), str(output)],
                capture_output=True,
            )

            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn(b"trailing", result.stderr)
            self.assertEqual(output.read_bytes(), original)

    def test_exact_size_generates_output_with_either_optimization_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "tables.bin"
            source.write_bytes(bytes(TABLE_BYTES))
            outputs = []
            for optimization in ([], ["-O"]):
                output = folder / "tables.rs"
                subprocess.run(
                    [sys.executable, *optimization, "-I", str(SCRIPT), str(source), str(output)],
                    capture_output=True,
                    check=True,
                )
                outputs.append(output.read_bytes())

            self.assertIn(b"pub static BC7:", outputs[0])
            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
