---
name: xcl-kernelx
description: >-
  Implement Xbox kernelx memory, threads, modules, and console identity in XCL.
  Use when editing hle/kernelx.rs, kernelx/lazy.rs, kernelx/placeholders.rs,
  or when a title fails on VirtualAlloc, title memory, or MapTitlePhysicalPages.
---

# kernelx

Module `kernelx.dll` in `hle/kernelx.rs`, with `lazy.rs` (commit on touch) and `placeholders.rs`. Routing in `hle/mod.rs` also sends `kernel32`, `kernelbase`, `advapi32`, `*legacy`, `minappmodelx`, and `api-ms-win-core-*` / `ext-ms-win-core-*` here. Unimplemented names fall through to `kernelbase` / `kernel32` / `advapi32` / `sechost`.

Title memory budget comes from `MicrosoftGame.Config` (`manifest.rs`) unless `xcl.toml` sets `title_memory_mb`. `device_type` is `anaconda` (Series X, 13824 MiB standard) or `lockhart` (Series S, 8192). `title_memory` is `auto`, `standard`, or `advanced`. `console_type_raw` is an override for `GetConsoleType`; the Series value is not public — do not invent a new constant without saying it is a guess.

`VirtualAlloc` (`MEM_GRAPHICS` is `0x1000_0000`):

- Commit with no fixed address: `d3d12x::sparse` first, then `gpu/uma` alloc.
- Reserve with no fixed address: a sparse window that uses no system RAM until commit.
- `MEM_64K_PAGES` (`0x2040_0000`): 64 KiB pages in a sparse window.
- Commit inside an existing sparse reservation: `commit_sparse`.
- Anything else: Windows `VirtualAlloc`.

`XMemAlloc` with `memory_type != 0` sets `MEM_GRAPHICS` and follows that path. `xmem.dll` routes here. `XMemIsToolsMemAvailable` returns false. UMA is GPU-visible in 4 GiB windows; the low 32 bits of a console address are the offset in the window. The aperture is 4 windows unless `XCL_UMA_WINDOWS` is 8 or 16 and heap windows are on (`xcl-d3d12`). Default size is 2048 MiB, or 4095 for advanced title memory; `uma_mb` and `XCL_UMA_MB` override. `OpenExistingHeapFromAddress` rejects a partial commit and a 4 GiB region. `xcl.exe` is linked without high-entropy ASLR so those addresses stay under 2^40 (`crates/xcl/build.rs`, `tests/low_va.rs`). Committed coherent windows are not asked of `VirtualQuery` on every touch.

A page that moves goes to its next address directly. If that commit cannot be made, the page stays mapped where it was. A lazy page that loses the race to commit retries (`lazy.rs`). `map_pages` calls `VirtualQuery` once per run of fresh pages. A freed sparse block is decommitted before its address is reused. Decommitted sparse tiles stay mapped until the next submit (`d3d12x/sparse.rs`).

GPU-visible memory outside the UMA windows is write-watched, so a texture placed there sees CPU stores. A GPU view of the title's CPU memory reads that CPU view. `fine_timer` asks Windows for its finest timer period and tells the process to keep it while the window is hidden; the 15.6 ms default made a title's audio mixer run at a fraction of real time. Process memory in use is `PrivateUsage`, or `PagefileUsage` when `PrivateUsage` is 0.

`placeholders.rs` runs before those branches. A placeholder that returns success without backing pages will corrupt later GPU reads. `MapTitlePhysicalPages` must use the title's addresses.

`Sleep` of 100 ms or longer does not count time the title is held (a bind waiting for a pipeline, or a PC call that keeps its threads). Shorter sleeps, a zero sleep, and `INFINITE` run as they are. `XCL_SLEEP_HOLD=<ms>` changes that threshold; `0` is the PC's `Sleep`.

Thread APIs schedule on the host's processors. Keep the host's TLS callback path (`loader::on_tls_event`) working for threads the title creates.
