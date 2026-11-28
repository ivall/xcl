---
name: xcl-report
description: >-
  Read XCL report.json and choose the next fix, and run the project's tests
  and title smoke. Use when the user pastes a report or log, asks what is
  blocking a title, or asks to run cargo test, scripts/test.ps1, scripts/smoke.ps1,
  or XCL_SAMPLE_* checks. Also use when writing a report or PR about how
  far a title got.
---

# Reports and tests

Every report or PR about a title run includes all of these. Do not guess a missing item.

- Exact commit (`git rev-parse HEAD`)
- GPU, CPU, RAM, and driver
- How far the run got: launcher, account picker, settings, main menu, in-game, or the first intro cutscene
- Exact launch command and the environment variables that were set
- The precise failure: API or method, HRESULT or exception, log lines, and the call site when one exists
- A screenshot or frame capture of the furthest screen reached (`XCL_FRAME_DUMP` is the capture XCL already writes; a present count is not a picture)
- After a fix: a rerun on that same commit's successor, and whether the screen or the failure changed

`report.rs` writes `report.json` on every run. It is the to-do list. Fields:

| Field | Meaning |
|---|---|
| `crash`, `exit_code` | Process end. Fix this before tuning frames. |
| `environment` | Preflight: DXC, GPU features (`preflight.rs`). |
| `imports_by_source`, `stubbed_imports` | Static resolution. Labels look like `hle:kernelx.dll`, `hle-cross:…`, `system`, `game-dll:…`, `stub`. A stub that was never called is not the blocker. |
| `called_stubs` | Runtime stubs, by count, with `first_caller` RVA. |
| `vfs` | `unmapped_paths`, `redirected_writes`, `whiteouts`. |
| `dstorage` | `bcpack_unpacked` (BC1, BC3, BC4, BC5, BC6H, BC7 that the decoder ran) and `bcpack_refused` (BC2, mode 0, modes above 7, swizzle). `XCL_DSTORAGE_BCPACK_FILL=1` paints a refused destination and still fails the request. |
| `notes` | Free-form warnings from the run. |
| `shaders` | Formats, hardware stages, recomp completeness, missing mnemonics. |
| `dxc` | Which `dxcompiler.dll` / `dxil.dll` loaded. |
| `d3d12` | Adapter, unknown IIDs and slots, translated vs fallback pipelines, `presents`, `last_frame_dump`. |
| `audio` | Engines, voices, formats played as silence. |
| `xgameruntime` | GDK classes asked for, XCL slots not implemented. |
| `modules` | Mapped images, base and size. |

`d3d12.presents` is a present count. A non-dark sample (`XCL_FRAME_DUMP`) does not prove rendering.

## Tests

```powershell
cargo test -p xcl --lib
cargo test --workspace -- --test-threads=4
.\scripts\test.ps1
.\scripts\smoke.ps1 -Game <folder> [-OutDir <path>] [-Title <label>]
```

`smoke.ps1` prints `title` as the first 8 hex digits of the SHA-256 of the game folder's name, so a PR does not name the folder. `-Title` prints that label instead. The two runs must use the same one. `-OutDir` copies `xcl.log`, `report.json`, and a `manifest.json`, and removes a stale `report.json` in that folder before the run.

The merge bar (`CONTRIBUTING.md`, checked by `scripts/pr_check.py`) is the workspace test with real counts, plus two `xcl-smoke v1` blocks when the runtime changed: `crates/xcl/src` outside tests, or `crates/xcl/data/`, `crates/xcl/build.rs`, `crates/xcl/Cargo.toml`, `Cargo.toml`, `Cargo.lock`. Paste them unchanged. A hand-written block or a count you did not see fails review. Smoke sets `XCL_NETWORK=0`, `XCL_RUN_SECONDS` (default 60), and `XCL_ACTIVITY=1`. `smoke.ps1` prints the adapter from `report.json` and runs on Windows PowerShell 5.1.

The two blocks must be the same title and the same `seconds`, and each needs `title`, `seconds`, and `frames shown`. Two blocks pasted in one fence both count. One commit is the PR head with no uncommitted changes. The other is a commit of `main`. On the PR block, `device removals` and `crashes` must be 0 and `exit` must not start with `killed`. The `main` block may show a crash when that is what the PR fixes. Frames shown on the PR must be at least 80% of `main`. Peak commit must be at most 125% of `main`. The checker is the copy of `pr_check.py` on `main`, not the PR's. A reviewed test-only diff skips the smoke check only. **Tested**, **Not tested**, and **AI use** need a real line; a placeholder copied from the PR template is an empty section.

GPU tests call `gpu::recomp::runner::skip_test` and still pass. Report `SKIPPED` lines; they are not coverage. `XCL_REQUIRE_GPU=wave64` or `mesh` (comma-separated) makes those skips fail. `XCL_ADAPTER=1` or a name fragment picks another adapter.

Opt-in retail data, never committed: `XCL_SAMPLE_RPF`, `XCL_SAMPLE_DLLS`, `XCL_SAMPLE_TTPF`, `XCL_SAMPLE_BK2`, `XCL_SAMPLE_BVH4`, `XCL_SAMPLE_AWC`, `XCL_SAMPLE_BCPACK` (`<name>.bc<mode>` and `<name>.vic<mode>`). Unset means the test passes without running. `scripts/test.ps1` lists which are unset.

CI (`.github/workflows/ci.yml`) runs `scripts/test.ps1 --skip test_title_runs_and_game_folder_is_untouched` on a GPU-less runner. Do not delete that e2e test; it checks the game folder stays byte-identical.

Shader compile spot-check: `xcl shaders <pack> --recomp --compile 64`. Do not compile the whole pack.

`XCL_TRACE_IMPORTS` logs arguments on resolved calls. See `xcl-imports`. It is for a short run; `all` wraps system imports too.
