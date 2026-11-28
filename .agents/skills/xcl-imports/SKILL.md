---
name: xcl-imports
description: >-
  Resolve Xbox console imports in XCL: HLE routing, system fallbacks, ordinals,
  and auto-stubs. Use when report.json lists stubbed_imports or called_stubs,
  or when editing resolver.rs, hle/mod.rs, stubs.rs, or stubdb.toml.
---

# Imports

`resolve_inner_opt` in `resolver.rs`, for `dll!name` or `dll!#ordinal`:

1. `Registry::route`. Aliases from `xcl.toml` are registered before the built-in patterns, so they win. `@stub` stubs immediately. `@system` is `LoadLibrary` on that DLL name. An HLE hit tries the module (`resolve_hle`, including its `fallback` DLLs) and, if still missing, stubs. It does not fall through to a game DLL of the same name.
2. A DLL already mapped, or a file in the game folder (`loader::load_game_dll`).
3. `LoadLibrary` / `GetProcAddress` on the imported name. There is no allowlist. Any DLL Windows can load is `Source::System`.
4. `Registry::find_any`: a known export name hosted on a different HLE module (`Source::HleOther`).
5. Auto-stub.

`HleModule.fallback`: `"@self"` (`hle::SELF`) means the DLL name the title imported. `aliases_are_system` is true only when `@self` is in that list. `kernelx` is; `GetModuleHandle("kernel32.dll")` must return the real Windows module. `xaudio2` falls back to system `xaudio2_9.dll` for unimplemented exports and does not set `aliases_are_system`. Read that flag before changing module-handle behavior.

New export: implement the function, add it with `hle_exports!` (Rust name equals the export), or an ordinal in `stubdb.toml` `[ordinals."dll"]` when the title imports by number. Unknown ordinals become `dll!#N` and show up in `inspect` and `report.json`.

`stubdb.toml`: module default return, then `[functions]` overrides (`*!Name`). GDK defaults to `E_NOTIMPL`. `S_OK` on a pointer-out parameter leaves the title's output uninitialized — do not use it to silence a stub. Per-title overrides: `xcl.toml` `[stub_returns]`.

Stubs are `mov r10, info; jmp handler` and must not clobber argument registers (`stubs.rs`). First call logs and `called_stubs` records the count and caller RVA.

`xcl inspect <exe> [--json]` is the static view. Only `called_stubs` says the title reached the function.

`XCL_TRACE_IMPORTS` wraps resolved calls (`1` or `hle` traces HLE; `all` traces system too; `dll!Prefix` includes, `-dll!Prefix` excludes). Stubs log themselves and are not wrapped.

`XCL_NETWORK=0` makes the title's WinHTTP and Winsock imports fail, and hooks the system's own `winhttp`, `wininet`, and `ws2_32` for the whole process (`hle/netoff.rs`). Loopback and local names are left alone. `scripts/smoke.ps1` sets the variable. Leave it unset unless the user asked for offline sockets. Do not fake a successful login.
