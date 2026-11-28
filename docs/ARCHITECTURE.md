# XCL – architecture map

A short map of the code for new contributors. Status, measurements and findings are in
[`HANDOFF.md`](../HANDOFF.md) (long: search it); the design is in
[`specs/2026-11-20-xcl-design.md`](specs/2026-11-20-xcl-design.md); the rules for a PR are in
[`CONTRIBUTING.md`](../CONTRIBUTING.md). All paths below are relative to `crates/xcl/src/`
unless they start with `crates/`, `scripts/` or `tests/`.

## How `xcl run` works

XCL runs a console title's x64 code natively in a Windows process and stands in for the console
DLLs it imports. `xcl run <game folder | exe>` (`main.rs` → `launcher.rs`) finds the game's exe,
prepares the title's data folder and `xcl.toml`, writes a boot record (`cache\boot.json`), checks
the PC (`preflight.rs`: DXC, the system XGameRuntime, the GPU; and the console CPU features) and
starts `xcl.exe` again as the **host process**, with the game's own command line and `XCL_HOST`
pointing at the boot record. The host (`host.rs`) sets up logging, the stub database, the HLE
module registry, the VFS and its file hooks and the exception hooks, then the **PE loader**
(`loader.rs`) maps the exe: relocations, imports, static TLS, exception tables, then the entry
point. Every import goes through the **resolver** (`resolver.rs`): routing to an **HLE module**
(`hle/`, our implementation of a console DLL), else a DLL shipped in the game folder (mapped by
our loader), else a real Windows DLL, else an automatic logging stub (`stubs.rs`). When the title
creates a pipeline from console shaders, the **GPU recompiler** (`gpu/recomp/`) translates the
RDNA2 code to HLSL and DXC compiles it to DXIL (`gpu/recomp/pso.rs`, `gpu/recomp/dxc.rs`), in
the background (`hle/d3d12x/pso_async.rs`). At the end the run's `report.json` and `xcl.log` are
in the title's `logs\` folder.

## Module map

"In-file" means unit tests in the file's `#[cfg(test)] mod tests`. Integration tests are in
`crates/xcl/tests/`; `e2e.rs` runs our own console title (`tests/testtitle`, `tests/testgamedll`)
through `xcl run`, and the test title checks the areas marked "test title".

### HLE modules (console DLLs)

Which DLL names go to which module is decided in `hle/mod.rs` (the routing table; aliases in
`xcl.toml` win).

| Console DLL / API | Module | What it does | Tests |
|---|---|---|---|
| `kernelx.dll` (also `kernel32`, `kernelbase`, `advapi32`, `api-ms-win-core-*` …) | `hle/kernelx.rs`, `hle/kernelx/` | the console flavour of kernel32 – only what differs from Windows; the rest falls back to the real export. `lazy.rs`: physical pages committed on first use; `placeholders.rs`: reservations in the GPU-visible sparse address space | in-file; test title |
| XGameRuntime (`xgameruntime*.dll`, `xgameplatform.dll`) | `hle/xgameruntime/` | offline GDK system APIs: `mod.rs` by-name exports, `api.rs` the thunk protocol (`QueryApiImpl`), `gdk_table.rs` the generated function table, `taskqueue.rs`, `xasync.rs`, `gamesave.rs`, `networking.rs`, `accessibility.rs`, `xsign.rs` (offline XSTS request signing) | in-file; test title |
| D3D12.X (`d3d12_xs.dll`, `d3d12_x.dll`) | `hle/d3d12x/` | the console's D3D12 on PC D3D12 – see the table below | in-file; `gpu_compute_e2e.rs`, `gpu_pso_e2e.rs`, `predication_e2e.rs`; test title |
| PC D3D12 / DXGI layout | `hle/d3d12_layout.rs`, `hle/d3d12x/layout.rs`, `hle/d3d12x/dxgi_device.rs` | the generated table of PC D3D12 interfaces (IIDs, vtable order) and lookups over it; the console device answering as an `IDXGIDevice` | in-file (`layout.rs`) |
| DirectStorage (`xcrdapi.dll`) | `hle/dstorage.rs` | `DStorageGetFactory` → the PC DirectStorage interfaces in the console's vtable order and structure layouts | in-file; test title |
| XAudio2 (`xaudio2_9.dll`) | `hle/xaudio2.rs` | console XAudio2 on the Windows XAudio2 2.9 engine, with compressed source voices (XMA2, Opus) | in-file; test title |
| XMA1 / XMA2 | `hle/xma2/` | pure-Rust decoder (`tables.rs` ported from FFmpeg, see its header) | in-file |
| Opus | `hle/opus.rs` | Opus packet framing, decoded through Windows' `MSOpusDecoder.dll` | in-file (`XCL_SAMPLE_AWC`) |
| `acphal.dll`, `xapu.dll`, `chad.dll` | `hle/acphal.rs` | the console's audio processor: its memory, command / message channel, SHAPE contexts, decoder queue | in-file |
| `gameinput.dll` | `hle/gameinput.rs`, `hle/kbm.rs` | GameInput with one always-connected gamepad: the first XInput controller; keyboard and mouse as that gamepad (`kbm.rs`) | in-file |
| `mfplat.dll` | `hle/mfplat.rs` | Media Foundation console helpers; standard exports fall back to the system `mfplat.dll` | in-file |
| `xg_xs.dll` | `hle/xg.rs` | the console's texture layout library, answered from XCL's GFX10 model | in-file; `xg_oracle.rs` (`XCL_SAMPLE_XG`) |
| `dxcompiler_xs.dll`, `dxcompiler_x.dll` | `hle/dxcompiler.rs` | the console's DXC: titles get the PC compiler XCL uses | in-file |
| `winhttp.dll`, `ws2_32.dll` (only with `XCL_NETWORK=0`) | `hle/netoff.rs` | the title's requests and sockets fail as on a console without a network; loopback stays | in-file |
| `minuserx.dll`, `ext-ms-win-rtcore-ntuser-window-*` | `hle/user.rs` | the window functions titles use; the rest is Windows' user32 | in-file |

### D3D12.X (`hle/d3d12x/`)

| File | What it does |
|---|---|
| `mod.rs` | the two backends: `real` (default) and `null` |
| `real.rs` | genuine PC D3D12 objects from the driver, "adopted" with extended vtables |
| `null.rs` | fallback backend: objects accept every call and render nothing (`graphics = "null"` or no D3D12 device) |
| `guard.rs` | slot guards: console-only vtable slots past the PC interface |
| `device_x.rs`, `list_x.rs` | the console device's and command list's own methods |
| `console_list.rs`, `title_list.rs` | command lists in the console's format and command records written by D3D12.X code compiled into the title |
| `bind.rs`, `console_heap.rs` | root arguments of console code; console descriptor heaps (T# / V# / S#) in unified memory |
| `present.rs`, `frame.rs` | `PresentX` into XCL's window; frame clock and frame pacing |
| `pso_async.rs` | non-blocking and lazy pipeline creation (stand-in pipeline until the real one is built) |
| `title_textures.rs`, `vram.rs`, `sparse.rs` | textures in title memory (detiled copies); GPU-owned targets aliased as on the console; the title's own reservations seen by the GPU |
| `fence_x.rs`, `cpu_writes.rs` | fence completed-value cells the title polls; bottom-of-pipe writes into memory the GPU does not see |
| `dxgi_device.rs`, `layout.rs` | see the HLE table above |

### Core (not HLE)

| Module | What it does | Tests |
|---|---|---|
| `main.rs` | CLI front-end (`run`, `inspect`, `disasm`, `shaders`, `rpf`) and host-mode entry | – |
| `launcher.rs` | `xcl run`: data folders, config, boot record, preflight, starts the host process | in-file; `e2e.rs` |
| `host.rs` | the host process: logging, VFS, hooks, HLE, maps the title, runs its entry point | `e2e.rs` |
| `pe.rs` | PE32+ parser for files on disk and mapped images | in-file |
| `loader.rs` | maps console executables / DLLs and wires them to HLE modules, game DLLs and Windows DLLs | in-file; `e2e.rs`, `console_dlls_e2e.rs` (`XCL_SAMPLE_DLLS`) |
| `resolver.rs` | import resolution policy: HLE → game DLL → Windows DLL → stub | in-file; `e2e.rs` |
| `modules.rs` | process-wide table of mapped images and HLE pseudo handles; lock-free address → module lookup | in-file |
| `stubs.rs` + `crates/xcl/data/stubdb.toml` | automatic logging stubs for unimplemented functions; the database holds their default return values (per module and per function) and ordinal → name maps; `[stub_returns]` in `xcl.toml` overrides | in-file; `e2e.rs` |
| `hooks.rs` | inline hooks on real Windows functions (via `retour`) | – |
| `eh.rs` | exception handling for manually mapped images (`.pdata`, `RtlPcToFileHeader` hook, crash filter) | in-file; test title |
| `insn.rs` | emulation of console CPU instructions a PC may lack | in-file |
| `tls.rs` | static TLS for manually mapped images | in-file; test title |
| `vfs.rs`, `vfs_hooks.rs` | console paths → the game folder, read-only, with a copy-on-write overlay; Win32 file API hooks on KERNELBASE feeding it | in-file; `pls_quota.rs`; test title |
| `config.rs`, `manifest.rs` | per-title `xcl.toml` and the boot record; the few `MicrosoftGame.Config` settings XCL uses | in-file |
| `preflight.rs` | what the PC has of what a run needs (DXC, XGameRuntime, GPU), printed and kept in `report.json` | in-file |
| `report.rs` | `report.json`: imports, called stubs, unmapped console paths, redirected files | in-file; `e2e.rs` |
| `log.rs` | process-wide logger, opened before the file hooks | – |
| `rpf.rs`, `rpf_probe.rs` | `xcl rpf`: lists and extracts RPF9 package records; bounded-memory probing | in-file; `rpf_e2e.rs` |
| `shaders.rs` | `xcl shaders`: statistics over a shader corpus; `--recomp` / `--compile N` | in-file |
| `disasm.rs` | `xcl disasm`: RDNA2 disassembler, console containers unpacked | `disasm_e2e.rs` |
| `inspect.rs` | `xcl inspect`: static import analysis, read-only | – |

### GPU (`gpu/`)

| Module | What it does | Tests |
|---|---|---|
| `rdna2.rs`, `rdna2_spec.rs`, `rdna2_xbox.rs` | RDNA2 decoder and disassembler: generated tables, decoding rules, and the gfx10.1 instructions the console GPU kept | in-file |
| `xbox.rs`, `pack.rs`, `dxbc.rs`, `shader.rs` | console shader containers (DXBC with XDAT / XBCH / FSSM); blobs found in packs without a TOC; DXBC signatures; shader capture and classification | in-file |
| `desc.rs` | GFX10 T# / S# / V# descriptors in D3D12 terms | in-file |
| `recomp/` | the RDNA2 → HLSL recompiler: `mod.rs` (execution model, structured control flow), `emit.rs` (one instruction → HLSL), `pso.rs` (console shaders in pipeline states, translation cache), `dxc.rs` (in-process DXC, DXIL cache), `runner.rs` (runs kernels on the GPU, `skip_test`), `upper_lanes.rs`, `x64_masks.rs`, `mesh_wave.rs`, `mesh_dispatch.rs` (wave64 on 32-lane GPUs, see [`wave64-on-wave32.md`](wave64-on-wave32.md)) | `recomp/tests.rs` (per-instruction GPU tests); in-file; `console_shaders.rs` (`XCL_SAMPLE_DXC`), `gpu_pso_e2e.rs`, `gpu_compute_e2e.rs` |
| `texture.rs`, `tiling.rs`, `tiling_tables.rs` | console textures in console memory → PC textures; GFX10 tiling from addrlib swizzle patterns | in-file (`XCL_SAMPLE_TTPF`); `texture_states.rs` |
| `uma.rs` | the title's graphics memory, shared by CPU and GPU, in up to 4 windows D3D12 opens as heaps | in-file; `heap_probe.rs` (`--ignored`) |
| `bvh.rs` | acceleration structures in the console's RDNA2 node format | in-file; `console_dlls_e2e.rs` |
| `agility.rs` | optional D3D12 Agility SDK loading (`D3D12Core.dll`) | `agility_tests.rs` |
| `adapter.rs` | which D3D12 adapter is used, for titles and the GPU tests | `adapter_tests.rs` |
| `fence.rs` | completion checks for XCL-owned D3D12 fences | in-file |

## Generated files

These files are written by a script in `scripts/` and are never edited by hand: change the
generator and rerun it (the PR says how).

| Generator | Output | Input |
|---|---|---|
| `scripts/gen_rdna2.py` | `gpu/rdna2_spec.rs` | AMD's machine-readable RDNA2 ISA (`amdgpu_isa_rdna2.xml`) |
| `scripts/gen_tiling.py` | `gpu/tiling_tables.rs` | AMD addrlib `gfx10SwizzlePattern.h` (Mesa) |
| `scripts/gen_d3d12_layout.py` | `hle/d3d12_layout.rs` | `d3d12.h` (microsoft/DirectX-Headers) |
| `scripts/gen_gdk_table.py` | `hle/xgameruntime/gdk_table.rs` | a table of the GDK runtime's functions from the public PC GDK's `xgameruntime.lib` |

Each script's docstring gives its exact command line.

## Environment variables

For running a title:

| Variable | What it does | Read in |
|---|---|---|
| `XCL_DATA_ROOT` | data root instead of `%LOCALAPPDATA%\XCL` | `launcher.rs` |
| `XCL_NETWORK` | `0`: the title runs as a console without a network (`hle/netoff.rs` takes `winhttp` / `ws2_32`) | `hle/netoff.rs` |
| `XCL_RUN_SECONDS` | `n`: the title ends itself after n seconds, with the GPU drained and the log and report written | `host.rs` |
| `XCL_ACTIVITY` | every 2 s, log what the title did (presents, submits, draws, dispatches, frame events, DirectStorage) and where the memory is | `host.rs` |
| `XCL_SKIP_CPU_CHECK` | run even when the CPU lacks features of the console's CPU | `launcher.rs` |
| `XCL_ADAPTER` | overrides `d3d12_adapter`: `auto`, `warp`, an index, or part of the adapter's name; also for the GPU tests | `gpu/adapter.rs` |
| `XCL_DXC_DIR` | folder tried first for `dxcompiler.dll` / `dxil.dll` | `gpu/recomp/dxc.rs` |
| `XCL_DXC_THREADS` | number of compile threads (default: every core but one) | `gpu/recomp/dxc.rs` |
| `XCL_D3D12_DIR` | folder of the D3D12 Agility SDK | `gpu/agility.rs` |
| `XCL_D3D12_DEBUG` | turns on the D3D12 debug layer (like `d3d12_debug` in `xcl.toml`) | `hle/d3d12x/real.rs` |
| `XCL_RAYTRACING` | overrides `raytracing` in `xcl.toml`: `off` / `0`, `tier1_0` / `10`, `tier1_1` / `11`; anything else keeps the GPU's tier | `config.rs` |
| `XCL_UMA_MB` | size of the title's graphics memory in MB (default 2048; `uma_mb` in `xcl.toml` sets it when unset) | `gpu/uma.rs`, `host.rs` |
| `XCL_FRAME_DUMP` | `1`: every 120th shown frame as `frame_<n>.bmp` in the logs folder; `n > 1`: every n-th. `XCL_FRAME_DUMP_FULL=1`: full size | `hle/d3d12x/present.rs` |
| `XCL_DUMP_SHADERS` | `1`: every shader the title submits is written to the title's `shaders` folder with its disassembly and HLSL | `gpu/shader.rs` |
| `XCL_TRANSLATED_CACHE` | `0`: no disk cache of translated programs | `gpu/recomp/pso.rs` |
| `XCL_DXIL_CACHE` | folder of the DXIL cache (default `%LOCALAPPDATA%\xcl\dxil`) | `gpu/recomp/dxc.rs` |
| `XCL_ASYNC_PSO` | `0`: no non-blocking pipeline creation | `hle/d3d12x/pso_async.rs` |
| `XCL_LAZY_PSO` | `0`: pipelines with console shaders are built when made, not at first use | `hle/d3d12x/pso_async.rs` |
| `XCL_SPLIT_WAVE64` | `1`: the split wave64 form also on GPUs with 64-lane waves (to check the 32-lane path on RDNA) | `gpu/recomp/pso.rs` |

`XCL_HOST` is set by the launcher for the host process; do not set it yourself.

For tests (the `XCL_SAMPLE_*` tests do nothing when the variable is unset; `scripts/test.ps1`
lists the ones not set):

| Variable | What it does | Read in |
|---|---|---|
| `XCL_REQUIRE_GPU` | `wave64`, `mesh` or `wave64,mesh`: a GPU test skipped for that reason fails instead | `gpu/recomp/runner.rs` |
| `XCL_SAMPLE_RPF` | a shader pack (`shaders.rpf`) | `gpu/recomp/tests.rs`, `shaders.rs`, `tests/gpu_pso_e2e.rs` |
| `XCL_SAMPLE_DLLS` | folder of the game's console DLLs | `tests/console_dlls_e2e.rs` |
| `XCL_SAMPLE_TTPF` | a texture archive | `gpu/texture.rs` |
| `XCL_SAMPLE_BK2` | a Bink 2 movie (with `XCL_SAMPLE_DLLS`) | `tests/console_dlls_e2e.rs` |
| `XCL_SAMPLE_BVH4` | the `core_bvh4.rpf` package (with `XCL_SAMPLE_DLLS`) | `tests/console_dlls_e2e.rs` |
| `XCL_SAMPLE_AWC` | a sound | `hle/opus.rs` |
| `XCL_SAMPLE_XG` | the console's own `xg_xs.dll` | `tests/xg_oracle.rs` |
| `XCL_SAMPLE_DXC` | the console's `dxcompiler_xs.dll` (with `xbsc_xs.dll` next to it) | `tests/console_shaders.rs` |

The other `XCL_*` variables are debug and trace switches. List them with
`grep -rn '"XCL_' crates/xcl/src`; each one's doc comment is next to the code that reads it.

## Where to start

- **Tests for an area.** Pick a module and add unit tests in its `#[cfg(test)] mod tests`;
  `hooks.rs`, `log.rs`, `inspect.rs` and `disasm.rs` have none in-file.
- **A recompiler instruction.** Add a per-instruction GPU test to `gpu/recomp/tests.rs`, compared
  against an expected result like the existing ones.
- **A missing XGameRuntime API.** Run a title and look at `report.json`: the called stubs and the
  XGameRuntime classes asked for show what is missing. Implement it in `hle/xgameruntime/`;
  `gdk_table.rs` gives the class, interface and vtable slot.
- **The test title.** When a change concerns the path a title takes, extend `tests/testtitle` /
  `tests/testgamedll` so `e2e.rs` covers it.
- **Docs.** Update `README.md` when a change makes something work (or stop working) for users.

Run one area's unit tests while iterating, then the whole suite:

```powershell
cargo test -p xcl --lib hle::kbm          # one area: the filter matches test paths
.\scripts\test.ps1                        # cargo build --workspace + cargo test --workspace
```

Before opening a PR, read [`CONTRIBUTING.md`](../CONTRIBUTING.md).
