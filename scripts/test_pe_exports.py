from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parent / "re" / "pe_exports.py"


def make_export_image(magic):
    image = bytearray(0x400)
    image[:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, 0x80)
    image[0x80:0x84] = b"PE\0\0"
    optional_size = 0xF0 if magic == 0x20B else 0xE0
    machine = 0x8664 if magic == 0x20B else 0x14C
    struct.pack_into("<HHIIIHH", image, 0x84, machine, 1, 0, 0, 0, optional_size, 0x2002)
    optional = 0x98
    struct.pack_into("<H", image, optional, magic)
    struct.pack_into("<II", image, optional + 32, 0x1000, 0x200)
    struct.pack_into("<II", image, optional + 56, 0x2000, 0x200)
    directory = optional + (112 if magic == 0x20B else 96)
    struct.pack_into("<I", image, directory - 4, 16)
    struct.pack_into("<II", image, directory, 0x1000, 0x100)
    section = optional + optional_size
    image[section:section + 8] = b".edata\0\0"
    struct.pack_into("<IIII", image, section + 8, 0x200, 0x1000, 0x200, 0x200)
    struct.pack_into("<I", image, section + 36, 0x40000040)
    struct.pack_into("<7I", image, 0x20C, 0x1080, 7, 3, 3, 0x1040, 0x1050, 0x1060)
    struct.pack_into("<3I", image, 0x240, 0x1100, 0x1110, 0x10C0)
    struct.pack_into("<3I", image, 0x250, 0x1090, 0x1098, 0x10A0)
    struct.pack_into("<3H", image, 0x260, 0, 0, 2)
    for offset, text in ((0x280, b"synthetic.dll\0"), (0x290, b"Alpha\0"),
                         (0x298, b"Beta\0"), (0x2A0, b"Forwarded\0"),
                         (0x2C0, b"other.Target\0")):
        image[offset:offset + len(text)] = text
    return image


class ExportAliasTests(unittest.TestCase):
    def test_export_name_in_zero_filled_section_tail_is_not_read_from_disk(self):
        image = bytearray(0x600)
        image[:2] = b"MZ"
        struct.pack_into("<I", image, 0x3C, 0x80)
        image[0x80:0x84] = b"PE\0\0"
        struct.pack_into("<HHIIIHH", image, 0x84, 0x8664, 2, 0, 0, 0, 0xF0, 0x2002)
        optional = 0x98
        struct.pack_into("<H", image, optional, 0x20B)
        struct.pack_into("<II", image, optional + 32, 0x1000, 0x200)
        struct.pack_into("<II", image, optional + 56, 0x3000, 0x200)
        struct.pack_into("<III", image, optional + 108, 16, 0x1000, 0x100)
        section = optional + 0xF0
        image[section:section + 8] = b".edata\0\0"
        struct.pack_into("<IIII", image, section + 8, 0x300, 0x1000, 0x200, 0x200)
        image[section + 40:section + 48] = b".text\0\0\0"
        struct.pack_into("<IIII", image, section + 48, 0x200, 0x2000, 0x200, 0x400)
        struct.pack_into("<7I", image, 0x20C, 0x1080, 1, 1, 1, 0x1040, 0x1050, 0x1060)
        struct.pack_into("<I", image, 0x240, 0x2000)
        struct.pack_into("<I", image, 0x250, 0x1200)
        image[0x280:0x28E] = b"synthetic.dll\0"
        # RVA 0x1200 is in the zero-filled part of .edata. Treating it as file-backed
        # would read .text's first bytes and report a name the loader never sees there.
        image[0x400:0x40D] = b"UnrelatedTag\0"

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "synthetic.dll"
            source.write_bytes(image)
            result = subprocess.run([sys.executable, "-B", str(SCRIPT), str(source)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("0x1200", result.stderr)
            self.assertNotIn("UnrelatedTag", result.stdout)

    def test_all_names_for_an_ordinal_are_printed(self):
        for magic in (0x10B, 0x20B):
            with self.subTest(magic=hex(magic)), tempfile.TemporaryDirectory() as tmp:
                image = make_export_image(magic)
                source = Path(tmp) / "synthetic.dll"
                source.write_bytes(image)
                result = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), str(source)],
                    capture_output=True, text=True, check=True,
                )
                self.assertRegex(result.stdout, r"(?m)^  #7\s+Alpha, Beta @0x1100$")
                self.assertRegex(result.stdout, r"(?m)^  #8\s+\(ordinal only\) @0x1110$")
                self.assertRegex(result.stdout, r"(?m)^  #9\s+Forwarded @0x10c0 -> other.Target$")

    def test_export_names_cannot_read_outside_section_raw_data(self):
        for name_rva, offset, raw_name, error in (
            (0x1200, 0x400, b"OverlayName\0", "is outside file-backed section data"),
            (0x11FC, 0x3FC, b"FakeName\0", "has no terminator in its section"),
        ):
            with self.subTest(name_rva=hex(name_rva)), tempfile.TemporaryDirectory() as tmp:
                image = make_export_image(0x20B)
                image.extend(bytes(0x200))
                struct.pack_into("<I", image, 0x98 + 0xF0 + 8, 0x400)  # VirtualSize exceeds SizeOfRawData.
                struct.pack_into("<I", image, 0x250, name_rva)
                image[offset:offset + len(raw_name)] = raw_name
                source = Path(tmp) / "synthetic.dll"
                source.write_bytes(image)
                result = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), str(source)],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(f"RVA {name_rva:#x} {error}", result.stderr)

    def test_export_table_entry_cannot_cross_section_raw_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = make_export_image(0x20B)
            image.extend(bytes(0x200))
            struct.pack_into("<I", image, 0x218, 1)  # one named export
            struct.pack_into("<I", image, 0x220, 0x11FE)  # name pointer table RVA
            struct.pack_into("<I", image, 0x3FE, 0x1090)  # first two bytes in the section
            source = Path(tmp) / "synthetic.dll"
            source.write_bytes(image)
            result = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(source)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("RVA 0x11fe is outside file-backed section data", result.stderr)


if __name__ == "__main__":
    unittest.main()
