# Running XCL on Linux

XCL is a Windows program. On Linux it is cross-compiled for `x86_64-pc-windows-gnu` and `xcl.exe` runs
under Wine, which sends D3D12 to the host's Vulkan driver through vkd3d-proton. This works, with the
setup below, up to the first frames of a retail game; the picture is still black (see
[Known state](#known-state)). Requests: [#154](https://github.com/ivall/xcl/issues/154) (Linux),
[#175](https://github.com/ivall/xcl/issues/175) (what Wine and vkd3d-proton need),
[#241](https://github.com/ivall/xcl/issues/241) (the vkd3d-proton assertion).

Checked on Ubuntu 26.04 (kernel 7.0), an NVIDIA RTX 4080 SUPER, Proton Experimental 11.0, vkd3d-proton v3.1,
DXVK 3.1.1. The helper scripts are in [`scripts/linux/`](../scripts/linux).

## Prerequisites

- Rust with the target: `rustup target add x86_64-pc-windows-gnu`
- mingw-w64 (Ubuntu: `gcc-mingw-w64-x86-64-posix g++-mingw-w64-x86-64-posix`; if `x86_64-w64-mingw32-gcc`
  uses win32 threads, `CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc-posix` picks the
  POSIX one)
- To build vkd3d-proton: `git meson ninja glslang-tools mingw-w64-tools` (the last one has `widl`)
- Steam with **Proton Experimental** installed (the Wine; see below)
- A Vulkan driver with the extensions vkd3d-proton needs, and a GPU with Shader Model 6.6 and mesh
  shaders, as on Windows (README)
- Linux 6.7 or newer (`userfaultfd` write-protect, see [Wine](#wine))
- An audio device, even a silent one (see [Audio device](#audio-device))

## Build

```sh
cargo build --release --workspace --target x86_64-pc-windows-gnu    # target/x86_64-pc-windows-gnu/release/xcl.exe
```

`cargo build --workspace --all-targets --locked --target x86_64-pc-windows-gnu` builds the tests too.
The test binaries are run under Wine the same way as `xcl.exe`, with the setup below.

## DXC

XCL compiles shaders with Microsoft's DXC, as on Windows. Take `dxcompiler.dll` and `dxil.dll` (`bin/x64`) from
a [DirectXShaderCompiler release](https://github.com/microsoft/DirectXShaderCompiler/releases) (1.8 or
newer) and put them next to `xcl.exe`, or point `XCL_DXC_DIR` at their folder. `run.sh` takes a Linux path
there and converts it.

## Opus (the sound of GTA)

GTA plays its sound through the console's audio processor, not XAudio2: with #294 its Opus packets go to the emulated
decoder (xapu, `hle/acphal.rs`), whose PCM is played on the PC through XAudio2 (`hle/xaudio2.rs`). On Windows the decoder
is the system's `MSOpusDecoder.dll`. **Wine / Proton has none** (only `mfplat.dll`), so XCL loads libopus
(BSD-3-Clause) from an `opus.dll` at run time, like DXC. Without either, the decoder answers silence and the log says
`xapu: no Opus decoder`. Build it with mingw-w64 (`cmake`, `make`, `curl`; downloads and checks the official release):

```sh
scripts/linux/build-opus.sh            # -> ~/.local/share/xcl-runtime/opus/opus.dll
```

and point `XCL_OPUS_DIR` at its folder (a Windows path: `Z:\home\you\.local\share\xcl-runtime\opus`), or copy it next to
`xcl.exe`. `xcl.log` says which decoder is used (`Opus: decoder = libopus (…)`); `XCL_TRACE_APU=1` logs what a title sends
to xapu / acphal.

## Wine

XCL needs a Wine that tracks `MEM_WRITE_WATCH` memory with the kernel (`userfaultfd`, Wine's
`USE_UFFD_WRITEWATCH`). Without it vkd3d-proton cannot import XCL's unified memory: it logs
`Failed to import host memory, hr 0x8007000e`, allocates a copy of its own and still returns success, so
the CPU and the GPU each see their own memory and the GPU reads zeros (#175).

- **Proton Experimental's Wine** (`files/bin/wine`) has it: its `lib/wine/x86_64-unix/ntdll.so` contains
  "Using kernel write watches". This is the Wine the scripts use.
- WineHQ's `wine-staging` packages for current Ubuntu have it too (11.21 on Ubuntu 24.04, #257).
- Ubuntu's own `wine` package (10.0 here) does not, nor do builds made against kernel headers older than
  6.7, such as portable builds made in an old chroot (Kron4ek's). Check any other Wine with
  `grep -a "Using kernel write watches" <wine>/lib/wine/x86_64-unix/ntdll.so`.
- Proton's `proton` script is not used: it failed here, copying `nvngx_update.exe`. The scripts call its
  `wine` directly, with a prefix of their own.
- A container needs `userfaultfd` allowed (Docker's default seccomp profile blocks it).
  `vm.unprivileged_userfaultfd` does not matter: Wine passes `UFFD_USER_MODE_ONLY`.

## vkd3d-proton

Stock vkd3d-proton (the releases, Proton's, winetricks') cannot run XCL. XCL gives D3D12 objects it
creates its own extended vtable (`hle/d3d12x/real.rs`, `adopt`), and vkd3d-proton checks the vtable's identity:

- 11 `assert(iface->lpVtbl == &d3d12_..._vtbl)` in `vkd3d_private.h`. The first one a title reaches is the
  heap's (`iface->lpVtbl == &d3d12_heap_vtbl`, [#241](https://github.com/ivall/xcl/issues/241)).
- `d3d12_command_allocator_from_iface` and `d3d12_command_list_from_iface` return NULL for those
  objects, so `CreateCommandList` fails with `E_INVALIDARG`.
- Shared fences are told from plain ones by their vtable: run with `VKD3D_CONFIG=ignore_shared_fence`.

The build in `scripts/linux/` is v3.1 with a 6-line patch for the two `from_iface` functions
([`vkd3d-proton-xcl.patch`](../scripts/linux/vkd3d-proton-xcl.patch)) and `-Db_ndebug=true`, which compiles
the assertions out. Its `dxil-spirv` submodule is moved to a newer commit (`DXIL_SPIRV_REF`, default
`53aac418`): v3.1's loops without end or overflows the stack on 20 of the retail title's compute programs
during world loading ([#611](https://github.com/ivall/xcl/issues/611)), the newer one converts all of them:

```sh
scripts/linux/build-vkd3d-proton.sh ~/xcl-vkd3d      # -> ~/xcl-vkd3d/vkd3d-proton-xcl/{d3d12,d3d12core}.dll
```

This removes checks for every application: use it for XCL only, in XCL's prefix. It is a workaround; the
real fix is XCL not touching the host's objects (proxies, [#175](https://github.com/ivall/xcl/issues/175)).

## DXVK

Wine's own `dxgi.dll` crashes in wined3d when XCL presents (#175). `setup-prefix.sh` copies DXVK's `dxgi.dll`,
`d3d11.dll` and `d3d10core.dll` from Proton (`files/lib/wine/dxvk/x86_64-windows/`). DXVK reports an
NVIDIA GPU as AMD by default; `DXVK_CONFIG='dxgi.hideNvidiaGpu = False'` keeps XCL's NVIDIA paths.

## Audio device

A retail game needs an audio endpoint. With no sound device at all (a container, a headless machine) its
audio engine's device setup fails and the game deadlocks about 30 s in (#257). A silent ALSA default device
is enough, in `~/.asoundrc`:

```
pcm.!default {
    type null
}
```

A desktop with sound does not need it.

## Prefix

```sh
PROTON="$HOME/.steam/steam/steamapps/common/Proton - Experimental"
scripts/linux/setup-prefix.sh "$PROTON" ~/xcl-prefix ~/xcl-vkd3d/vkd3d-proton-xcl
```

It creates the prefix with Proton's Wine and copies into its `System32`: Proton's `libvkd3d-*.dll`
(without them `xcl.exe` does not start: `d3dcompiler_47` → wined3d → `libvkd3d-shader-1.dll`), DXVK's
three DLLs and the vkd3d-proton build. Only the prefix is written to.

## Running

```sh
scripts/linux/run.sh "$PROTON" ~/xcl-prefix /path/to/unpacked/game
```

`run.sh` runs `xcl.exe run <game>` (`XCL_EXE` picks another `xcl.exe`) with:

| Variable | Value |
|---|---|
| `WINEPREFIX` | the prefix |
| `WINEDLLOVERRIDES` | `d3d12,d3d12core,dxgi,d3d11,d3d10core=n,b`: the prefix's DLLs, not Wine's |
| `VKD3D_CONFIG` | `ignore_shared_fence` unless set |
| `XCL_NETWORK` | `0` unless set. A retail game's crash reporter sends minidumps out; leave it off |
| `XCL_DXC_DIR` | converted to a Windows path if set |
| `WINEDEBUG` | `-all` unless set |

Arguments after the game folder go to `xcl run`. The title's data and logs are in the prefix:
`drive_c/users/<user>/AppData/Local/XCL/<title>/` (`logs/xcl.log`, `logs/report.json`). A run left behind
is stopped with `WINEPREFIX=~/xcl-prefix "$PROTON/files/bin/wineserver" -k`.

## Regression runs

`scripts/smoke.ps1` (the regression run CONTRIBUTING.md asks of a PR that changes the runtime) needs
Windows. `scripts/linux/smoke.sh` is the same run under Wine: it runs the title for a fixed time with
`XCL_RUN_SECONDS`, reads XCL's log and prints the same `xcl-smoke` block, once from `main` and once
from the PR's head commit:

```sh
scripts/linux/smoke.sh "$PROTON" ~/xcl-prefix /path/to/unpacked/game 60
```

Say in the PR that the blocks come from `smoke.sh` under Wine. The `gpu:` line comes from XCL's log.
`SMOKE_GRACE` (default 120 s, as `smoke.ps1`) is how long past the run's time it may take before it is
killed: under Wine XCL's end takes about 15 s after `XCL_RUN_SECONDS` (its end watchdog), and
`PR checks` refuses a block that says `killed`.

## Start-up check

Under Wine `xcl run` also checks the libraries it got and prints the result with the other start-up checks
(and keeps it in `report.json`):

- **ERROR** when the loaded `d3d12core.dll` holds the string `iface->lpVtbl == &d3d12_heap_vtbl`: this
  vkd3d-proton stops on XCL's objects (#241). Use the build above.
- **WARNING** when the loaded `dxgi.dll` is not DXVK's: presenting crashes in wined3d.

Nothing changes on Windows.

## Known state

- A retail game (GTA VI 1.0.2.1) starts, compiles its pipelines and runs at about 30 fps with the setup
  above. **The picture stays black.** This is known upstream and not something this kit changes: on NVIDIA
  under Wine the menu's element draws collapse onto one diagonal line
  ([#175](https://github.com/ivall/xcl/issues/175), comments); the retail black image on memory the GPU does
  not see is [#186](https://github.com/ivall/xcl/issues/186).
- The vkd3d-proton build is a workaround. The correct fix is proxies for the host's objects, so vkd3d-proton
  stays untouched (proposed in #175, not implemented).
- Some tests differ under Wine (error codes, timings; #175).

Links: [#154](https://github.com/ivall/xcl/issues/154) · [#175](https://github.com/ivall/xcl/issues/175) ·
[#186](https://github.com/ivall/xcl/issues/186) · [#241](https://github.com/ivall/xcl/issues/241)
