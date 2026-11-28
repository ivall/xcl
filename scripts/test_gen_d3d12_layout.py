import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class D3D12LayoutTests(unittest.TestCase):
    def test_device_without_parsed_methods_leaves_output_alone(self):
        generator = Path(__file__).with_name("gen_d3d12_layout.py")
        base = (
            'MIDL_INTERFACE("00000000-0000-0000-0000-000000000001")\n'
            'ID3D12Device : public IUnknown {};\n'
            'typedef struct ID3D12DeviceVtbl {\n'
            '    void (STDMETHODCALLTYPE *QueryInterface)(void);\n'
            '} ID3D12DeviceVtbl;\n'
        )
        derived = (
            'MIDL_INTERFACE("00000000-0000-0000-0000-000000000002")\n'
            'ID3D12Device1 : public ID3D12Device {};\n'
            'typedef struct ID3D12Device1Vtbl {} ID3D12Device1Vtbl;\n'
        )
        with tempfile.TemporaryDirectory() as folder:
            header, output = Path(folder) / "device.h", Path(folder) / "layout.rs"
            output.write_bytes(b"previous layout\r\n")
            for source in (derived, base + derived):
                with self.subTest(source=source):
                    header.write_text(source, encoding="utf-8")
                    run = subprocess.run([sys.executable, "-I", generator, header, output], capture_output=True)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn(b"ID3D12Device1", run.stderr)
                    self.assertEqual(output.read_bytes(), b"previous layout\r\n")

    def test_header_without_a_device_leaves_outputs_alone(self):
        generator = Path(__file__).with_name("gen_d3d12_layout.py")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            header = root / "debug.h"
            header.write_text(
                'MIDL_INTERFACE("00000000-0000-0000-0000-000000000001")\n'
                'ID3D12Debug : public IUnknown {};\n'
                'typedef struct ID3D12DebugVtbl {\n'
                '    void (STDMETHODCALLTYPE *EnableDebugLayer)(void);\n'
                '} ID3D12DebugVtbl;\n',
                encoding="utf-8",
            )
            existing, missing = root / "existing.rs", root / "missing.rs"
            previous = b"// previously generated table\n"
            existing.write_bytes(previous)
            for output in (existing, missing):
                run = subprocess.run(
                    [sys.executable, "-I", generator, header, output], capture_output=True,
                )
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(b"expected d3d12.h", run.stderr)
            self.assertEqual(existing.read_bytes(), previous)
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
