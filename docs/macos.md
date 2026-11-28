# Building XCL on macOS and running it under CrossOver

XCL is a Windows program. On an Apple Silicon Mac the same `xcl.exe` can be cross-compiled with
mingw-w64 and run under CrossOver's Wine, with Apple's D3DMetal as the D3D12 implementation and
Rosetta 2 running the x86-64 code. This page is how to build, run and test it there. It is not a
supported platform: see [State](#state) for what works on `main` today. The work is tracked in
[#258](https://github.com/xcl-dev/xcl/issues/258).

## State

Checked on one machine: MacBook Pro (Apple M5 Max, 64 GB), macOS 27.3, CrossOver 26.3, Rust 1.100.0,
mingw-w64 14.0.0 (GCC 16.2.0), DXC 1.9.2609.5, `main` at `3ceb550`.

- **The build works.** `main` builds for `x86_64-pc-windows-gnu` without a change.
- **The tools work:** `xcl inspect`, `xcl disasm`, `xcl shaders`, `xcl rpf` (checked on `508a0c7`;
  `xcl inspect` again on `3ceb550`).
- **Tests** (`cargo test --workspace` under Wine, see [Tests](#tests)): 1,280 passed, 59 failed,
  46 ignored, 6 hung, 2 ended without a result. Most of the failures are what D3DMetal does not do:
  a program with an f64 instruction writes nothing, and `OpenExistingHeapFromAddress` returns
  `E_NOTIMPL`, which the memory-window and unified-memory tests need. The wave64 GPU tests print
  `SKIPPED: no wave64` (32-lane waves only).
- **Titles** (checked on `508a0c7`): the newer retail game from the README table starts and then shows no frame (0 frames
  in a 60 s `scripts/smoke.ps1` run, no crash, no device removal). D3DMetal has no
  `OpenExistingHeapFromAddress` (`E_NOTIMPL`), so the title's graphics memory is not shared with the
  GPU. The other titles of the table were not tried.

What this host does differently from a Windows PC, as far as it was seen here:

- The adapter is reported as `AMD Compatibility Mode`, with 32-lane waves.
- Wine reports no commit charge: `peak commit` in a smoke block and the `memory:` lines read 0 MB.
- There is no `System32\xgameruntime.dll`; XCL's own task queues and async are used.
- A crash starts Wine's debugger, which holds the process without printing anything. A crashed test
  looks like a hang until its processes are ended (see [Cleaning up](#cleaning-up)).

## What you need

- An Apple Silicon Mac with Rosetta 2 (`softwareupdate --install-rosetta`).
- [CrossOver](https://www.codeweavers.com/crossover) 26 or newer (it ships D3DMetal).
- Rust with the Windows GNU target, and mingw-w64 as its linker:

  ```sh
  brew install mingw-w64
  rustup target add x86_64-pc-windows-gnu
  ```

- DXC: `dxcompiler.dll` and `dxil.dll` (x64) from the
  [DirectXShaderCompiler releases](https://github.com/microsoft/DirectXShaderCompiler/releases),
  1.8 or newer. There is no Windows SDK to find them in.
- PowerShell, only for `scripts/smoke.ps1`: `brew install powershell`.

## Build

```sh
cargo build --release -p xcl --target x86_64-pc-windows-gnu
```

The result is `target/x86_64-pc-windows-gnu/release/xcl.exe`. Put `dxcompiler.dll` and `dxil.dll`
next to it, or point `XCL_DXC_DIR` at their folder (a Wine path, see below).

## A bottle

Create a Windows 10 64-bit bottle in CrossOver and choose D3DMetal as its graphics backend (the
bottle's settings). In the bottle's `cxbottle.conf` that is:

```
[EnvironmentVariables]
"CX_GRAPHICS_BACKEND" = "d3dmetal"
"ROSETTA_ADVERTISE_AVX" = "1"
```

`ROSETTA_ADVERTISE_AVX` makes Rosetta 2 report AVX and AVX2, which XCL's CPU check asks for.

Set the CrossOver location and your bottle name in the shell used for the examples below.

```sh
export CX="/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin"   # or under ~/Applications
export XCL_MAC_BOTTLE="My XCL Bottle"   # use the name of your bottle
```

## Run

Wine shows the Mac's file system as drive `Z:`, so a Mac path `/Users/me/Games/Title` is
`Z:\Users\me\Games\Title` to XCL. `Z:/Users/me/Games/Title` works too.

```sh
XCL_DATA_ROOT='Z:\Users\me\xcl-data' WINEDEBUG=-all \
  "$CX/wine" --bottle "$XCL_MAC_BOTTLE" "Z:$PWD/target/x86_64-pc-windows-gnu/release/xcl.exe" \
  run 'Z:\Users\me\Games\Title' < /dev/null
```

- Environment variables of the shell reach XCL (`XCL_NETWORK=0`, `XCL_RUN_SECONDS=60`, …).
- Set `XCL_DATA_ROOT`, outside the game folder. Without it the title's data, log and report go to
  `%LOCALAPPDATA%\XCL` inside the bottle (`drive_c/users/crossover/AppData/Local/XCL`).
- Keep stdin at `/dev/null`: on an error XCL waits for Enter.
- `WINEDEBUG=-all` silences Wine's own messages; XCL's log is in `<data>/<title>/logs/xcl.log`.

## Tests

The tests are Windows programs too. Build the workspace (the end-to-end test needs the test title
and its DLL) and the test executables, then let cargo start each one under Wine through a runner:

```sh
cargo build --workspace --target x86_64-pc-windows-gnu

cat > /tmp/xcl-wine-runner <<'EOF'
#!/bin/sh
case "$1" in
  /*) exe="Z:$1" ;;
  *)  exe="Z:$PWD/$1" ;;
esac
shift
exec "$CX/wine" --bottle "$XCL_MAC_BOTTLE" "$exe" "$@" < /dev/null
EOF
chmod +x /tmp/xcl-wine-runner

CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER=/tmp/xcl-wine-runner \
XCL_DXC_DIR='Z:\path\to\dxc' WINEDEBUG=-all \
  cargo test --workspace --target x86_64-pc-windows-gnu --no-fail-fast -- --test-threads=4
```

`XCL_DXC_DIR` is needed here: the test executables are in `target/…/debug/deps`, not next to
`xcl.exe`.

A test that crashes leaves its process in Wine's debugger, which prints nothing, so a test binary can
stop making progress without ending. Put a time limit on each binary; when one runs out, end the
bottle's processes (see [Cleaning up](#cleaning-up)) and run the tests it had not reported one
process each, with `--exact <test name>`. The counts under [State](#state) were made that way: each
binary with `--test-threads=4`, one with no output for 240 s ended, the rest of its tests run alone
with a 120 s limit.

## Regression run (`scripts/smoke.ps1`)

`scripts/smoke.ps1` runs unchanged under PowerShell on macOS. It starts the program given as
`-Xcl` and reads the log whose path XCL prints, so on a Mac `-Xcl` is a small wrapper that starts
`xcl.exe` under Wine, turns the game folder into a Wine path and the printed `Z:\…` paths back
into Mac paths:

```sh
#!/bin/zsh
# Put xcl-wine next to xcl.exe. It reads CX, XCL_MAC_BOTTLE and XCL_MAC_DATA (a Mac folder).
here=${0:A:h}
out=$(mktemp -d)
args=()
for a in "$@"; do
  if [[ "$a" == /* && -e "$a" ]]; then args+=("Z:${a:A}"); else args+=("$a"); fi
done
# stdout to a file, not a pipe: Wine's helper processes keep a pipe open after xcl.exe has ended
XCL_DATA_ROOT="Z:${XCL_MAC_DATA//\//\\}" WINEDEBUG=-all "$CX/wine" --bottle "$XCL_MAC_BOTTLE" \
  "Z:$here/xcl.exe" "${args[@]}" > "$out/stdout.txt" 2> "$out/stderr.txt" < /dev/null
code=$?
sed -E '/^(Log|Report):[[:space:]]+Z:[\\/]/{s/Z:[\\/]/\//;s/\\/\//g;}' "$out/stdout.txt"
exit $code
```

```sh
XCL_MAC_DATA="$HOME/xcl-data" \
  pwsh -NoProfile -File scripts/smoke.ps1 -Game /Users/me/Games/Title -Seconds 60 \
  -Xcl target/x86_64-pc-windows-gnu/release/xcl-wine
```

The block's `gpu:` line is what D3DMetal reports (`AMD Compatibility Mode`) and `peak commit` is
0 MB (Wine reports none). `title:` is the last part of the `-Game` path.

## Cleaning up

Wine keeps helper processes (`wineserver`, `winedevice.exe`, `conhost.exe`) after a run, and its
debugger keeps a crashed process. To end everything that belongs to one bottle:

```sh
WINEPREFIX="$HOME/Library/Application Support/CrossOver/Bottles/${XCL_MAC_BOTTLE:?set the bottle name}" "$CX/wineserver" -k
```
