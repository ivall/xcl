---
name: xcl-loader
description: >-
  Change XCL's PE loader, TLS, C++ exceptions, or process launch. Use when a
  title fails before main, imports do not map, TLS callbacks misbehave, or the
  user mentions loader.rs, pe.rs, tls.rs, eh.rs, host.rs, or the XCL_HOST child.
---

# Loader

`xcl.exe` is the launcher (`launcher.rs`). The title runs in a child started with `XCL_HOST` (`host.rs`, `main.rs`). Do not bind the title's imports inside the launcher process.

The launcher stops when the CPU lacks the console Zen 2 set: SSE4.2, POPCNT, AVX, AVX2, FMA, F16C, BMI1, BMI2, LZCNT, MOVBE. `XCL_SKIP_CPU_CHECK=1` prints a warning and continues. The host does not check again. A title `STATUS_ILLEGAL_INSTRUCTION` from one of those ops is `insn.rs`, logged from `eh.rs`.

`XCL_DEVICE_TYPE` overrides `device_type` for that process (`anaconda` / `series x` / `xsx` / `x`, or `lockhart` / `series s` / `xss` / `s`). An unknown value is ignored and `xcl.toml` is left as it is.

`crates/xcl/build.rs` links `xcl.exe` with high-entropy ASLR off (`/HIGHENTROPYVA:NO`) so title memory stays under 2^40. `tests/low_va.rs` checks the flag. The TLS carrier DLL sets the flag on itself; leave `xcl.exe` opted out.

`loader.rs` / `pe.rs`: PE32+ AMD64 only, relocations, delay-load, section protection, game DLLs recursive. Preferred image base is kept when free; no relocations and a taken base is a hard error.

`tls.rs`: static TLS via a generated carrier DLL. On Windows 11 the loader allocates a TLS slot only for DLLs that have imports, and `SizeOfZeroFill` is ignored (the whole template is raw data). `on_tls_event` runs from the TLS callback in `.CRT$XLY` so threads created after load get the title's TLS.

`eh.rs`: `.pdata` via `RtlAddFunctionTable`, plus a hook so `RtlPcToFileHeader` sees manually mapped images. A C++ throw and catch inside title code is the test.

`run_main` starts the entry on a thread whose stack reserve comes from the PE header.

## Checks

Extend `tests/testtitle` rather than inventing a second harness. The e2e test `test_title_runs_and_game_folder_is_untouched` must keep passing on a machine with `xgameruntime.dll`. A loader change that copies the exe into the game folder violates the read-only rule.
