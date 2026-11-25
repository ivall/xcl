#!/usr/bin/env bash
# Build vkd3d-proton v3.1 for XCL: its dxil-spirv submodule at a newer commit, the XCL patch
# applied, assertions compiled out.
# See docs/linux.md ("vkd3d-proton").
#
# usage: build-vkd3d-proton.sh <work-dir>
#   Clones vkd3d-proton into <work-dir>/vkd3d-proton, builds there, and leaves
#   d3d12.dll and d3d12core.dll in <work-dir>/vkd3d-proton-xcl/ (what setup-prefix.sh takes).
#   Nothing is written outside <work-dir>. Needs: git, meson, ninja, glslangValidator
#   (glslang-tools), x86_64-w64-mingw32-gcc / -g++ and -widl (mingw-w64, mingw-w64-tools).
# env: VKD3D_PROTON_REPO  the repository to clone (default: upstream)
#      VKD3D_PROTON_REF   the tag to build (default: v3.1, the one the patch is made for)
#      DXIL_SPIRV_REF     the dxil-spirv commit to build it with (default: 53aac41815f8;
#                         empty: the one the tag has)
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <work-dir>" >&2
    exit 2
fi
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
patch=$here/vkd3d-proton-xcl.patch
repo=${VKD3D_PROTON_REPO:-https://github.com/HansKristian-Work/vkd3d-proton}
ref=${VKD3D_PROTON_REF:-v3.1}
dxs_ref=${DXIL_SPIRV_REF-53aac41815f8203dff4e081969438ed5063dbff3}

for tool in git meson ninja glslangValidator x86_64-w64-mingw32-gcc x86_64-w64-mingw32-g++ x86_64-w64-mingw32-widl; do
    command -v "$tool" >/dev/null || { echo "missing: $tool (see docs/linux.md, Prerequisites)" >&2; exit 1; }
done

mkdir -p "$1"
work=$(cd "$1" && pwd)
src=$work/vkd3d-proton
out=$work/vkd3d-proton-xcl

if [ ! -d "$src/.git" ]; then
    git clone --branch "$ref" --recurse-submodules "$repo" "$src"
fi
cd "$src"

# apply_once <dir> <patch>: the patch applied in <dir> unless it is already
apply_once() {
    if git -C "$1" apply --check "$2" 2>/dev/null; then
        git -C "$1" apply "$2"
    elif git -C "$1" apply --check --reverse "$2" 2>/dev/null; then
        echo "$(basename "$2") already applied"
    else
        echo "$2 does not apply to $1 (is it vkd3d-proton $ref?)" >&2
        exit 1
    fi
}
apply_once "$src" "$patch"

# v3.1's dxil-spirv (8b48f0e) cannot structure 20 of a retail title's compute programs: it loops
# without end (tens of GB within minutes) or recurses until the stack runs out, and the title's
# GPU watchdog ends it during world loading (#611). Its master has fixed all of them since.
if [ -n "$dxs_ref" ]; then
    dxs=$src/subprojects/dxil-spirv
    if [ "$(git -C "$dxs" rev-parse HEAD)" != "$(git -C "$dxs" rev-parse --verify --quiet "$dxs_ref^{commit}" || true)" ]; then
        git -C "$dxs" fetch --quiet origin
        git -C "$dxs" checkout --quiet --detach "$dxs_ref"
        git -C "$dxs" submodule update --init --recursive --quiet
    fi
    echo "dxil-spirv: $(git -C "$dxs" log -1 --format='%h %s')"
fi

# b_ndebug: the vtable-identity assertions the objects XCL adopts would trip are compiled out
if [ -d build ]; then
    meson setup --reconfigure --cross-file build-win64.txt --buildtype release -Db_ndebug=true --strip build
else
    meson setup --cross-file build-win64.txt --buildtype release -Db_ndebug=true --strip build
fi
ninja -C build

mkdir -p "$out"
cp build/libs/d3d12/d3d12.dll build/libs/d3d12core/d3d12core.dll "$out/"
echo "built: $out/{d3d12.dll,d3d12core.dll}"
