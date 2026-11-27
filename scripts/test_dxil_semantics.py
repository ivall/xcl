"""Compile and inspect the minimal DXIL semantic-audit compute fixture."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "tests/data/dxil_semantic/compute_store.hlsl"
RESOURCE_FIXTURE = HERE.parent / "tests/data/dxil_semantic/resource_bindings.hlsl"
SPEC = importlib.util.spec_from_file_location("inspect_dxil_semantics", HERE / "inspect_dxil_semantics.py")
assert SPEC and SPEC.loader
INSPECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECTOR)


class SemanticFixtureTests(unittest.TestCase):
    def test_explicit_missing_dxc_does_not_fall_back(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "specified dxc.exe does not exist"):
            INSPECTOR.find_dxc(str(HERE / "missing-dxc.exe"))

    def test_invalid_dxc_directory_does_not_fall_back(self) -> None:
        with mock.patch.dict("os.environ", {"XCL_DXC_DIR": str(HERE / "missing-dxc-dir")}):
            with self.assertRaisesRegex(FileNotFoundError, "XCL_DXC_DIR does not contain dxc.exe"):
                INSPECTOR.find_dxc()

    def test_multiple_entry_points_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple entry points"):
            INSPECTOR._entry_metadata("!dx.entryPoints = !{!1, !2}", {1: "!{}", 2: "!{}"})

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.dxc = INSPECTOR.find_dxc(DXC_OVERRIDE)
        except FileNotFoundError as exc:
            if REQUIRE_DXC:
                raise
            raise unittest.SkipTest(str(exc)) from exc
        cls.temp = tempfile.TemporaryDirectory(prefix="xcl-dxil-semantics-")
        cls.container = Path(cls.temp.name) / "compute_store.dxil"
        subprocess.run(
            [str(cls.dxc), "-E", "main", "-T", "cs_6_6", "-O3", "-Fo", str(cls.container), str(FIXTURE)],
            check=True,
            capture_output=True,
            text=True,
        )
        cls.dump = subprocess.run(
            [str(cls.dxc), "-dumpbin", str(cls.container)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        cls.report = INSPECTOR.inspect_dxil(cls.container, cls.dxc)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "temp"):
            cls.temp.cleanup()

    def test_stage_model_versions_and_entry_metadata(self) -> None:
        self.assertEqual(self.report["stage"], "compute")
        self.assertEqual(self.report["shader_model"], "cs_6_6")
        self.assertEqual(self.report["dxil_module_version"], [1, 6])
        self.assertEqual(len(self.report["validator_version"]), 2)
        self.assertEqual(self.report["entry"]["function"], "main")
        self.assertEqual(self.report["entry"]["name"], "main")
        self.assertIn("shader_flags", self.report["entry"])
        self.assertEqual(self.report["entry"]["thread_group_size"], [8, 4, 2])

    def test_entry_metadata_contains_thread_group_size(self) -> None:
        nodes = INSPECTOR._metadata(self.dump)
        entry = INSPECTOR._entry_metadata(self.dump, nodes)
        self.assertEqual(entry["thread_group_size"], [8, 4, 2])

    def test_features_operations_and_resource_semantics(self) -> None:
        self.assertEqual(self.report["sfi0_feature_flags"], "0x0000000000004000")
        self.assertIn("Wave level operations", self.report["required_features"])
        operations = {(op["name"], op["overload"]): op["opcode"] for op in self.report["dx_ops"]}
        self.assertEqual(operations["waveActiveOp", "i32"], 119)
        self.assertEqual(operations["rawBufferStore", "i32"], 140)
        resources = self.report["resource_bindings"]
        self.assertEqual(len(resources), 1)
        self.assertEqual(resources[0]["kind"], "UAV")
        self.assertEqual(resources[0]["hlsl_binding"], "u0")
        self.assertEqual(resources[0]["count"], 1)
        self.assertIn(
            {"resource_type": "RWByteAddressBuffer", "properties": [4107, 0]},
            self.report["resource_annotations"],
        )

    def test_resource_bindings_include_constant_buffers_textures_and_tbuffers(self) -> None:
        container = Path(self.temp.name) / "resource_bindings.dxil"
        subprocess.run(
            [str(self.dxc), "-E", "main", "-T", "cs_6_6", "-O3", "-Fo", str(container), str(RESOURCE_FIXTURE)],
            check=True,
            capture_output=True,
            text=True,
        )
        bindings = INSPECTOR.inspect_dxil(container, self.dxc)["resource_bindings"]
        self.assertEqual(
            {(entry["kind"], entry["hlsl_binding"], entry["count"]) for entry in bindings},
            {("CBUFFER", "cb2,space3", 1), ("SRV", "t1,space3", 2), ("SRV", "t11,space15", "unbounded"), ("SAMPLER", "s4,space3", 1), ("UAV", "u5,space3", 1), ("TBUFFER", "t2,space12", 1)},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dxc", help="path to dxc.exe; otherwise use XCL_DXC_DIR or Windows SDK")
    parser.add_argument("--require-dxc", action="store_true", help="fail instead of skipping if dxc.exe is unavailable")
    args = parser.parse_args()
    global DXC_OVERRIDE
    DXC_OVERRIDE = args.dxc
    global REQUIRE_DXC
    REQUIRE_DXC = args.require_dxc
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SemanticFixtureTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


DXC_OVERRIDE: str | None = None
REQUIRE_DXC = False

if __name__ == "__main__":
    raise SystemExit(main())
