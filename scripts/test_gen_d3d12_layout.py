import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_d3d12_layout import generate, DEFAULT_OUT


SCRIPT = Path(__file__).resolve().parent / "gen_d3d12_layout.py"


SAMPLE_HEADER = """
MIDL_INTERFACE("189819F1-1DB6-4B57-BE54-1821339B85F7")
ID3D12Device : public ID3D12Object
{
};

typedef struct ID3D12DeviceVtbl
{
    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( ID3D12Device * This, REFIID riid, void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )( ID3D12Device * This);
    ULONG ( STDMETHODCALLTYPE *Release )( ID3D12Device * This);
    HRESULT ( STDMETHODCALLTYPE *CreateCommandQueue )( ID3D12Device * This, const D3D12_COMMAND_QUEUE_DESC *pDesc, REFIID riid, void **ppCommandQueue);
} ID3D12DeviceVtbl;

MIDL_INTERFACE("77ACCE80-638E-4E65-8895-C1F23386863E")
ID3D12Device1 : public ID3D12Device
{
};

typedef struct ID3D12Device1Vtbl
{
    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( ID3D12Device1 * This, REFIID riid, void **ppvObject);
    ULONG ( STDMETHODCALLTYPE *AddRef )( ID3D12Device1 * This);
    ULONG ( STDMETHODCALLTYPE *Release )( ID3D12Device1 * This);
    HRESULT ( STDMETHODCALLTYPE *CreateCommandQueue )( ID3D12Device1 * This, const D3D12_COMMAND_QUEUE_DESC *pDesc, REFIID riid, void **ppCommandQueue);
    HRESULT ( STDMETHODCALLTYPE *CreatePipelineLibrary )( ID3D12Device1 * This, const void *pLibraryBlob, SIZE_T BlobLength, REFIID riid, void **ppPipelineLibrary);
} ID3D12Device1Vtbl;

MIDL_INTERFACE("6102DEE4-AF59-4B09-B999-B44D73F09B24")
ID3D12CommandAllocator : public ID3D12Pageable
{
};

typedef struct ID3D12CommandAllocatorVtbl
{
    HRESULT ( STDMETHODCALLTYPE *QueryInterface )( ID3D12CommandAllocator * This, REFIID riid, void **ppvObject);
    // Duplicate declaration like struct-returning methods
    ULONG ( STDMETHODCALLTYPE *AddRef )( ID3D12CommandAllocator * This);
    ULONG ( STDMETHODCALLTYPE *AddRef )( ID3D12CommandAllocator * This);
    ULONG ( STDMETHODCALLTYPE *Release )( ID3D12CommandAllocator * This);
} ID3D12CommandAllocatorVtbl;
"""


class GenD3D12LayoutCliTests(unittest.TestCase):
    def test_help_flag_prints_usage_and_exits_zero(self):
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                res = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), flag],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(res.returncode, 0)
                self.assertIn("Generate crates/xcl/src/hle/d3d12_layout.rs", res.stdout)
                self.assertIn("Usage: python -I scripts/gen_d3d12_layout.py", res.stdout)

    def test_missing_args_prints_usage_and_exits_code_2(self):
        res = subprocess.run(
            [sys.executable, "-B", str(SCRIPT)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 2)
        self.assertIn("Usage: python -I scripts/gen_d3d12_layout.py", res.stdout)

    def test_nonexistent_file_exits_code_1_with_error(self):
        res = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "nonexistent_d3d12.h"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("error: file not found", res.stderr)

    def test_invalid_input_file_exits_code_1_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_file = Path(tmp) / "empty.h"
            bad_file.write_text("// No interfaces\n", encoding="utf-8")
            res = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(bad_file)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 1)
            self.assertIn("error: no ID3D12 interfaces found", res.stderr)

    def test_cli_custom_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_file = Path(tmp) / "d3d12.h"
            src_file.write_text(SAMPLE_HEADER, encoding="utf-8")
            out_file = Path(tmp) / "sub" / "out_layout.rs"

            res = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), str(src_file), str(out_file)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0)
            self.assertIn("3 interfaces; most derived device ID3D12Device1 with 5 methods ->", res.stdout)
            self.assertTrue(out_file.is_file())
            content = out_file.read_text(encoding="utf-8")
            self.assertIn("pub static IFACES: &[Iface] = &[", content)
            self.assertIn('Iface { name: "ID3D12Device", iid: "189819f1-1db6-4b57-be54-1821339b85f7", parent: "ID3D12Object"', content)


class GenD3D12LayoutGenerateTests(unittest.TestCase):
    def test_generate_parses_interfaces_and_deduplicates_methods(self):
        content, names, dev, dev_count = generate(SAMPLE_HEADER)
        self.assertEqual(names, ["ID3D12CommandAllocator", "ID3D12Device", "ID3D12Device1"])
        self.assertEqual(dev, "ID3D12Device1")
        self.assertEqual(dev_count, 5)

        # Ensure AddRef was deduplicated from two consecutive occurrences to one
        self.assertIn('methods: &["QueryInterface", "AddRef", "Release"]', content)
        # Ensure lowercased IIDs
        self.assertIn('"77acce80-638e-4e65-8895-c1f23386863e"', content)

    def test_generate_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            generate("int x = 42;")


if __name__ == "__main__":
    unittest.main()
