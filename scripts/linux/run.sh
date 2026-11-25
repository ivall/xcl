#!/usr/bin/env bash
# Run a game with xcl.exe under Proton's Wine. See docs/linux.md ("Running").
#
# usage: run.sh <proton-dir> <prefix> <game-dir> [xcl run arguments...]
#   <proton-dir>  Proton Experimental's folder (it holds files/bin/wine)
#   <prefix>      a prefix made by setup-prefix.sh
#   <game-dir>    the unpacked game's folder
# env: XCL_EXE      xcl.exe to run (default: target/x86_64-pc-windows-gnu/release/xcl.exe)
#      XCL_DXC_DIR  a Linux folder with dxcompiler.dll and dxil.dll (default: next to xcl.exe)
#      XCL_NETWORK  0 unless set: a retail game's crash reporter uploads minidumps
#      VKD3D_CONFIG ignore_shared_fence unless set; WINEDEBUG -all unless set
#   Any other XCL_* variable is passed on to xcl.exe as it is. Nothing is written but the
#   prefix (and what xcl.exe writes there). Stop a run left behind with
#   WINEPREFIX=<prefix> <proton-dir>/files/bin/wineserver -k
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "usage: $0 <proton-dir> <prefix> <game-dir> [xcl run arguments...]" >&2
    exit 2
fi
proton=$1
prefix=$2
game=$3
shift 3
files=$proton/files
[ -x "$files/bin/wine" ] || files=$proton
[ -x "$files/bin/wine" ] || { echo "no files/bin/wine in $proton" >&2; exit 1; }
[ -d "$prefix/drive_c" ] || { echo "$prefix is not a prefix (setup-prefix.sh makes one)" >&2; exit 1; }
[ -d "$game" ] || { echo "no game folder $game" >&2; exit 1; }

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exe=${XCL_EXE:-$here/../../target/x86_64-pc-windows-gnu/release/xcl.exe}
[ -f "$exe" ] || { echo "no xcl.exe at $exe (see docs/linux.md, Build; XCL_EXE overrides)" >&2; exit 1; }
game=$(cd "$game" && pwd)

export WINEPREFIX=$prefix
export WINEDEBUG=${WINEDEBUG:--all}
export WINEDLLOVERRIDES="d3d12,d3d12core,dxgi,d3d11,d3d10core=n,b"
export VKD3D_CONFIG=${VKD3D_CONFIG:-ignore_shared_fence}
export XCL_NETWORK=${XCL_NETWORK:-0}
if [ -n "${XCL_DXC_DIR:-}" ]; then
    XCL_DXC_DIR=$("$files/bin/wine" winepath -w "$XCL_DXC_DIR" 2>/dev/null | tr -d '\r')
    export XCL_DXC_DIR
fi

exec "$files/bin/wine" "$exe" run "$game" "$@"
