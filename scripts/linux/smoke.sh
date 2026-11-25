#!/usr/bin/env bash
# scripts/smoke.ps1 under Wine: a regression run of a title for a pull request. It runs the title
# for a fixed time (XCL_RUN_SECONDS), reads XCL's log and prints the same `xcl-smoke` block as
# smoke.ps1, to paste into the PR's "Regression run" section – once built from main, once from the
# PR's head commit (CONTRIBUTING.md, "What a PR needs to be merged"). Say in the PR that the blocks
# come from this script under Wine; see docs/linux.md.
#
# usage: smoke.sh <proton-dir> <prefix> <game-dir> [seconds (default 60)]
# env: XCL_EXE (default: target/x86_64-pc-windows-gnu/release/xcl.exe); as run.sh: XCL_DXC_DIR,
#      VKD3D_CONFIG, WINEDEBUG; any other XCL_* variable is passed on. SMOKE_GRACE: seconds
#      the run may take past <seconds> before it is killed (default 120, as smoke.ps1). Networking is off
#      (XCL_NETWORK=0). Under Wine XCL's end takes about 15 s after XCL_RUN_SECONDS (its end watchdog),
#      so a grace under 20 s gets the run killed and the block says so.
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "usage: $0 <proton-dir> <prefix> <game-dir> [seconds]" >&2
    exit 2
fi
proton=$1
prefix=$2
game=$3
seconds=${4:-60}
files=$proton/files
[ -x "$files/bin/wine" ] || files=$proton
[ -x "$files/bin/wine" ] || { echo "no files/bin/wine in $proton" >&2; exit 1; }
[ -d "$prefix/drive_c" ] || { echo "$prefix is not a prefix (setup-prefix.sh makes one)" >&2; exit 1; }
[ -d "$game" ] || { echo "no game folder $game" >&2; exit 1; }

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
exe=${XCL_EXE:-$root/target/x86_64-pc-windows-gnu/release/xcl.exe}
[ -f "$exe" ] || { echo "no xcl.exe at $exe - build it first (docs/linux.md, Build)" >&2; exit 1; }
game=$(cd "$game" && pwd)
commit=$(git -C "$root" rev-parse HEAD)
dirty=$(git -C "$root" status --porcelain --untracked-files=no -- crates Cargo.toml Cargo.lock)

export WINEPREFIX=$prefix
export WINEDEBUG=${WINEDEBUG:--all}
export WINEDLLOVERRIDES="d3d12,d3d12core,dxgi,d3d11,d3d10core=n,b"
export VKD3D_CONFIG=${VKD3D_CONFIG:-ignore_shared_fence}
export XCL_NETWORK=0 XCL_RUN_SECONDS=$seconds XCL_ACTIVITY=1
if [ -n "${XCL_DXC_DIR:-}" ]; then
    XCL_DXC_DIR=$("$files/bin/wine" winepath -w "$XCL_DXC_DIR" 2>/dev/null | tr -d '\r')
    export XCL_DXC_DIR
fi

out=$(mktemp)
trap 'rm -f "$out"' EXIT
code=0
timeout -k 10 $((seconds + ${SMOKE_GRACE:-120})) "$files/bin/wine" "$exe" run "$game" >"$out" 2>/dev/null </dev/null || code=$?
if [ "$code" -eq 124 ] || [ "$code" -eq 137 ]; then
    exitcode='killed (no exit)'
else
    exitcode=$(printf '0x%x' "$code")
fi

winlog=$(tr -d '\r' <"$out" | sed -n 's/^Log:[[:space:]]*//p' | tail -n 1)
if [ -z "$winlog" ]; then
    # a timed run (XCL_RUN_SECONDS) can end before the launcher prints its summary under Wine:
    # the log is in the title's data folder that it printed at the start
    data=$(tr -d '\r' <"$out" | sed -n 's/^[[:space:]]*data:[[:space:]]*//p' | head -n 1)
    [ -n "$data" ] && winlog="$data\\logs\\xcl.log"
fi
[ -n "$winlog" ] || { echo "XCL printed no log path; its output:" >&2; cat "$out" >&2; exit 1; }
log=$("$files/bin/wine" winepath -u "$winlog" 2>/dev/null | tr -d '\r')
[ -f "$log" ] || { echo "no log at $log ($winlog)" >&2; exit 1; }

frames=$(grep -oE '\(([0-9]+) frames shown, ([0-9]+) PresentX calls\)' "$log" | tail -n 1 |
    sed -E 's/\(([0-9]+) frames shown, ([0-9]+) PresentX calls\)/\1 (PresentX \2)/')
[ -n "$frames" ] || frames='none logged (the run did not reach its end)'
presents=$(grep -oE 'activity \(2 s\): [0-9]+ presents' "$log" | grep -oE '[0-9]+ presents' |
    grep -oE '^[0-9]+' | tail -n 3 | paste -sd' ' -)
peak=$(grep -oE 'memory: [0-9]+ MB committed' "$log" | grep -oE '[0-9]+' | sort -n | tail -n 1)
removals=$(grep -c 'device removed' "$log" || true)
crashes=$(grep -cE 'unhandled exception|process ends with status' "$log" || true)
gpu=$(grep -oE "backend 'real' – .+ \([0-9]+ MiB VRAM\)" "$log" | head -n 1 |
    sed -E "s/^backend 'real' – (.+) \([0-9]+ MiB VRAM\)$/\1/")

echo '```'
echo 'xcl-smoke v1'
echo "commit: $commit$([ -n "$dirty" ] && echo ' (uncommitted changes)')"
echo "gpu: ${gpu:-unknown}"
echo "title: $(basename "$game")"
echo "seconds: $seconds"
echo "exit: $exitcode"
echo "frames shown: $frames"
echo "presents per 2 s (last 3): $presents"
echo "device removals: $removals"
echo "crashes: $crashes"
echo "peak commit: ${peak:-n/a} MB"
echo '```'
