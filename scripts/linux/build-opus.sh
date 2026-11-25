#!/bin/bash
# Builds libopus (the Opus decoder, BSD-3-Clause) for Windows with mingw-w64: the opus.dll XCL loads at run
# time when Windows' own decoder (MSOpusDecoder.dll) is missing, which is the case under Wine / Proton.
# XCL does not distribute it (like DXC); see docs/linux.md.
#
# usage: build-opus.sh [output dir]      default: ~/.local/share/xcl-runtime/opus
#   -> <output dir>/opus.dll             point XCL_OPUS_DIR at it (Windows path: Z:\home\you\...) or copy it
#                                        next to xcl.exe
# needs: cmake, make, curl, sha256sum, x86_64-w64-mingw32-gcc
set -euo pipefail

VERSION=1.5.2
SHA256=65c1d2f78b9f2fb20082c38cbe47c951ad5839345876e46941612ee87f9a7ce1   # downloads.xiph.org/releases/opus/SHA256SUMS.txt
OUT=${1:-$HOME/.local/share/xcl-runtime/opus}

for t in cmake make curl sha256sum x86_64-w64-mingw32-gcc; do
    command -v "$t" >/dev/null || { echo "missing: $t" >&2; exit 1; }
done

mkdir -p "$OUT/dl"
TARBALL="$OUT/dl/opus-$VERSION.tar.gz"
if [ ! -f "$TARBALL" ]; then
    curl -fsSL -o "$TARBALL" "https://downloads.xiph.org/releases/opus/opus-$VERSION.tar.gz"
fi
echo "$SHA256  $TARBALL" | sha256sum -c - >&2

SRC="$OUT/src"
rm -rf "$SRC" && mkdir -p "$SRC"
tar -xzf "$TARBALL" -C "$SRC" --strip-components=1

cmake -S "$SRC" -B "$SRC/build" \
    -DCMAKE_SYSTEM_NAME=Windows \
    -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DOPUS_BUILD_PROGRAMS=OFF -DOPUS_BUILD_TESTING=OFF \
    -DOPUS_INSTALL_PKG_CONFIG_MODULE=OFF -DOPUS_INSTALL_CMAKE_CONFIG_MODULE=OFF \
    -DOPUS_STACK_PROTECTOR=OFF -DOPUS_FORTIFY_SOURCE=OFF \
    -DCMAKE_C_FLAGS="-fno-stack-protector -U_FORTIFY_SOURCE" \
    -DCMAKE_SHARED_LIBRARY_PREFIX="" \
    -DCMAKE_SHARED_LINKER_FLAGS="-static-libgcc"
cmake --build "$SRC/build" -j"$(nproc)"

DLL=$(find "$SRC/build" -maxdepth 1 -name '*opus*.dll' | head -1)
[ -n "$DLL" ] || { echo "no DLL built" >&2; exit 1; }
cp "$DLL" "$OUT/opus.dll"
echo "built: $OUT/opus.dll (libopus $VERSION)" >&2
x86_64-w64-mingw32-objdump -p "$OUT/opus.dll" | grep -E "DLL Name|opus_decoder_create|opus_decode$" >&2 || true
