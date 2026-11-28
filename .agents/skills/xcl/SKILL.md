---
name: xcl
description: >-
  Index for working on XCL, the Xbox Series X|S to Windows compatibility layer.
  Use when starting work in this repo, choosing which subsystem to change, or
  when the user mentions XCL, kernelx, D3D12.X, RDNA2, a pull request, or running
  a Series title on Windows. Read the matching granular skill before editing
  that subsystem. Contributor and agent rules: CONTRIBUTING.md and AGENTS.md.
---

# XCL

Native x64 compatibility layer. The title runs in a child process (`XCL_HOST`,
`host.rs`). XCL maps the PE, resolves imports, and recompiles RDNA2 to HLSL
then DXIL. No CPU emulation. Design: `docs/specs/2026-11-20-xcl-design.md`.
User-facing status: `README.md`. `HANDOFF.md` is a session log.

These skills live only in `.agents/skills/`. Codex and Cursor load that directory. Claude Code does not; `CLAUDE.md` tells it to read the same files. Do not copy them elsewhere.

Repo language is English. License is GPL-2.0-only. No SDK headers or binaries,
no game files, no DRM or package decryption. The game folder is read-only;
writes go to `%LOCALAPPDATA%\XCL\<title>\` (`%XCL_DATA_ROOT%` overrides).

## Which skill

Open that directory's `SKILL.md` before editing the subsystem.

| Work | Path |
|---|---|
| First intro cutscene | `.agents/skills/xcl-cutscene/SKILL.md` |
| `report.json`, tests, samples | `.agents/skills/xcl-report/SKILL.md` |
| PE map, TLS, exceptions, launch | `.agents/skills/xcl-loader/SKILL.md` |
| Imports, stubs, ordinals | `.agents/skills/xcl-imports/SKILL.md` |
| Paths, overlay, DirectStorage | `.agents/skills/xcl-vfs/SKILL.md` |
| Memory, threads, console info | `.agents/skills/xcl-kernelx/SKILL.md` |
| GDK thunks, user, saves | `.agents/skills/xcl-xgameruntime/SKILL.md` |
| Device, lists, present | `.agents/skills/xcl-d3d12/SKILL.md` |
| RDNA2 decode and HLSL | `.agents/skills/xcl-shaders/SKILL.md` |
| Direct DXIL, per-lane, split dispatch | `.agents/skills/xcl-dxil/SKILL.md` |
| GFX10 tiling, descriptors | `.agents/skills/xcl-textures/SKILL.md` |
| XAudio2, XMA, Opus | `.agents/skills/xcl-audio/SKILL.md` |
| Gamepad, keyboard | `.agents/skills/xcl-input/SKILL.md` |

## Shared rules

- One concrete startup or playability blocker at a time. Reproduce it on current `main` and name the furthest screen before writing code. What a report must include is in `xcl-cutscene` and `xcl-report`.
- A missing import is a logging stub (`stubs.rs`, `crates/xcl/data/stubdb.toml`), not a load failure. A stub that returns success so a title gets past a screen it has not actually reached is a speculative stub. Do not add one.
- Do not invent console layouts. Use public docs, existing tables, or a user-supplied report.
- Do not hand-edit generated files: `gpu/rdna2_spec.rs` (`scripts/gen_rdna2.py`), `gpu/tiling_tables.rs` (`scripts/gen_tiling.py`), `hle/d3d12_layout.rs` (`scripts/gen_d3d12_layout.py`), `hle/xgameruntime/gdk_table.rs` (`scripts/gen_gdk_table.py`), `hle/bcpack/tables.rs` (`scripts/gen_bcpack_victory_tables.py`). Each header says so.
- WinDurango is reference only. AnyPS5 ports need a provenance note (`THIRD_PARTY.md`).
- Edit with the file tools. PowerShell `Set-Content` corrupts UTF-8.
- Do not rewrite `HANDOFF.md` or `docs/specs/` in a feature change. If a statement there is now wrong, say so in the PR description. Update `README.md` when user-visible behavior changes.
- Do not run `cargo fmt` on a file or the crate. Format only the lines you change.
- One logical change. Under ~400 changed lines, excluding generated files and test data. A second bug is a second PR. Search open PRs before starting (`gh pr list -R ivall/xcl --search "<keywords>"`).
- Commit subjects are `<Area>: <what it does>`, about 72 characters, matching `git log` (`VFS`, `D3D12.X`, `Recompiler`, `kernelx`, …). No `fix:` / `feat(x):` prefixes. Commit only when the user asks. Do not push or open a PR unless they ask in this session.
- Local check while iterating: `cargo test -p xcl --lib`. Before calling the suite done: `cargo test --workspace -- --test-threads=4` and report passed / failed / ignored / `SKIPPED` from that run. `.\scripts\test.ps1` is the same build plus tests. Do not `--compile` thousands of shaders; `--compile 64` is the spot check.
- A runtime change needs two `xcl-smoke v1` blocks from `.\scripts\smoke.ps1 -Game <folder>`, pasted unchanged. Runtime means `crates/xcl/src` outside tests, and also `crates/xcl/data/`, `crates/xcl/build.rs`, `crates/xcl/Cargo.toml`, `Cargo.toml`, and `Cargo.lock`. One block is the PR head with no uncommitted changes. The other is a commit that is on `main`. Same title, same `seconds`. Each block needs `title`, `seconds`, and `frames shown`. Two blocks in one code fence both count. The PR run must not be killed, and its `device removals` and `crashes` must be 0. The `main` run may show a crash when the PR fixes one. The PR needs at least 80% of `main`'s frames and at most 125% of its peak commit. `scripts/pr_check.py` checks this on `main`'s copy of the script (`pull_request_target`); a PR cannot change the checker. A reviewed test-only diff skips only the smoke check. **Tested**, **Not tested**, and **AI use** must contain a real line; a placeholder copied from the PR template fails the check. **AI use** must not name a Qwen, QwQ, DeepSeek, Llama, Mistral (including Codestral), Gemma, or Phi model. Do not commit game files or binaries outside `tests/data/` (or any binary over 1 MB). `.gitignore` lists the usual ones.
- GPU tests that cannot run call `gpu::recomp::runner::skip_test` and return. Do not `#[ignore]` a test to make it pass.
- CI skips `test_title_runs_and_game_folder_is_untouched` (no Gaming Services, no real GPU). That test still matters on a PC.
- Disclose AI use in the PR **AI use** section: tool, model, and what it did. Do not invent the model name.
