import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

from scripts.kbm_icons import build_library


class OutputTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name)
        self.game = self.root / "game"
        self.game.mkdir()
        self.source = self.game / "data.rpf"
        self.source.write_bytes(b"original archive")

    def run_guard(self, flag, destination):
        other_flag = "--out" if flag == "--movie-out" else "--movie-out"
        other_path = self.root / "other-output"
        result = subprocess.run(
            [sys.executable, "-B", build_library.__file__, "--game", str(self.game),
             flag, str(destination), other_flag, str(other_path)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(f"{flag} must be outside the game folder", result.stderr)
        self.assertFalse(other_path.exists())
        self.assertFalse(destination.exists())
        self.assertEqual(self.source.read_bytes(), b"original archive")

    def test_both_destinations_are_checked_before_writing(self):
        for flag in ("--out", "--movie-out"):
            with self.subTest(flag=flag):
                self.run_guard(flag, self.game / "new" / "output")

    @unittest.skipUnless(os.name == "nt", "requires Windows directory junctions")
    def test_both_destinations_reject_junction_into_game(self):
        subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", "game-alias", "game"],
            cwd=self.root, capture_output=True, text=True, check=True,
        )
        for flag in ("--out", "--movie-out"):
            with self.subTest(flag=flag):
                self.run_guard(flag, self.root / "game-alias" / "new" / "output")

    def test_outputs_leave_preexisting_links_to_game_file_untouched(self):
        out = self.root / "overlay.rpf"
        old_tmp = out.with_suffix(".rpf.tmp")
        movie_out = self.root / "movie.gfx"
        os.link(self.source, old_tmp)
        os.link(self.source, movie_out)
        movie = b"new movie"
        packed = zlib.compress(movie)
        args = ["build_library.py", "--game", str(self.game),
                "--out", str(out), "--movie-out", str(movie_out)]
        with (
            patch.object(sys, "argv", args),
            patch.object(build_library, "find_entry", return_value=(0, b"old movie", len(packed))),
            patch.object(build_library, "load_master"),
            patch.object(build_library, "build", return_value=(movie, len(movie), 1)),
            patch.object(build_library, "pack_exact", return_value=(packed, movie)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            build_library.main()
        self.assertEqual(out.read_bytes(), packed)
        self.assertEqual(movie_out.read_bytes(), movie)
        self.assertEqual(self.source.read_bytes(), b"original archive")
        self.assertEqual(old_tmp.read_bytes(), b"original archive")
        self.assertEqual(list(self.root.glob("*.tmp")), [old_tmp])


if __name__ == "__main__":
    unittest.main()
