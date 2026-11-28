from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parent / "re" / "bvhb_shaders.py"


def summarize_root(root, folder):
    chunk = b"RTS0" + struct.pack("<I", len(root)) + root
    blob = struct.pack("<4s16sIII", b"DXBC", bytes(16), 1, 36 + len(chunk), 1)
    blob += struct.pack("<I", 36) + chunk
    source = folder / "synthetic.dll"
    source.write_bytes(blob)
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT), str(source), str(folder / "out")],
        capture_output=True, text=True, check=True,
    ).stdout


class RootDescriptorTests(unittest.TestCase):
    def test_root_descriptors_print_their_register_classes(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                descriptors = ((2, 7), (3, 11), (4, 13))
                payload_size = 12 if version == 2 else 8
                payload_offset = 24 + 12 * len(descriptors)
                root = bytearray(struct.pack("<6I", version, len(descriptors), 24, 0, 0, 0))
                for index, (kind, register) in enumerate(descriptors):
                    root += struct.pack("<3I", kind, 0, payload_offset + index * payload_size)
                for kind, register in descriptors:
                    root += struct.pack("<2I", register, 0)
                    if version == 2:
                        root += struct.pack("<I", 0)
                self.assertIn("CBV b7 | SRV t11 | UAV u13", summarize_root(root, Path(tmp)))

    def test_descriptor_table_prints_explicit_and_append_offsets(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                root = bytearray(struct.pack("<6I", version, 1, 24, 0, 0, 0))
                root += struct.pack("<3I", 0, 0, 36)  # descriptor table parameter
                root += struct.pack("<2I", 2, 44)  # two descriptor ranges
                for kind, count, register, flags, offset in ((0, 2, 3, 4, 8), (1, 1, 7, 2, 0xFFFFFFFF)):
                    root += struct.pack("<4I", kind, count, register, 0)
                    if version == 2:
                        root += struct.pack("<I", flags)
                    root += struct.pack("<I", offset)
                self.assertIn(
                    "table[SRVx2@3 offset=8,UAVx1@7 offset=append]",
                    summarize_root(root, Path(tmp)),
                )


if __name__ == "__main__":
    unittest.main()
