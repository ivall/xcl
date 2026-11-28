---
name: xcl-dxil
description: >-
  Translate Xbox RDNA2 straight to DXIL in XCL, without HLSL or DXC. Use when
  editing gpu/dxil/, gpu/recomp/lower_dxil.rs, or the direct path in
  gpu/recomp/pso.rs, or when the user mentions XCL_DIRECT_DXIL, XCL_PER_LANE,
  a direct DXIL fallback, or a split compute dispatch.
---

# Direct DXIL

`gpu/dxil/` emits a signed DXIL container from decoded RDNA2. No HLSL, no DXC on this path. Design: `docs/specs/2026-11-24-direct-dxil.md`. Layers, bottom up: `bitstream`, `module`, `shader` (`dx.op`), `container` (digest, so no SDK DLL at run time), `validate` (`dxil.dll` as the test oracle). A validator older than 1.8 skips validation. Builder misuse is an error, not a panic.

The runtime still translates through HLSL (`xcl-shaders`) unless a flag says otherwise.

- `XCL_DIRECT_DXIL=1`: compute, the mesh compute pass, and the geometry (ES+GS) compute pass. Each falls back to HLSL on its own, with one `[SHADER] direct DXIL fallback` line. FSPX containers stay on HLSL. The mesh passthrough shader is still HLSL.
- `XCL_DIRECT_DXIL=all`: vertex and pixel shaders too. A pixel shader with a debug hook (`XCL_DEBUG_LOD0`, `XCL_DEBUG_PS_SUB`, `XCL_DEBUG_PS_EXPR`, or a debug color) stays on HLSL. A direct stage declares the same signature as the HLSL entry, so the two link.
- A compute program whose HLSL is over `dxc::UNOPTIMISED_ABOVE` is translated here even when the flag is off (`pso::enormous`).
- `XCL_PER_LANE=1` (with the flag): compute tries the per-lane form first (`gpu/recomp/lower_dxil.rs`), then the per-instruction form, then HLSL. Its disk entries use their own identity (`xcl-per-lane-` plus `XCL_PER_LANE_SOURCE_ID`).

`XCL_DIRECT_DXIL_STATS=1` prints live direct-versus-fallback counts at the end of the run. `XCL_DXIL_DUMP=<hash>` (or `all`) writes that program's container to `%TEMP%\xcl-dxil-dump\` and needs `XCL_TRANSLATED_CACHE=0` so a cached translation is redone. `xcl shaders --direct` is the offline sweep.

Cache identity is not the HLSL one. Direct entries carry `kernel::TRANSLATOR_VERSION` and `XCL_DIRECT_DXIL_SOURCE_ID` (the hash of `gpu/dxil`, from `build.rs`). An edit under `gpu/dxil` needs no version bump. Bump `TRANSLATOR_VERSION` when `pso::direct_compute` or `direct_graphics` change the containers those functions hand back. Do not put `gpu/dxil` into `TRANSLATOR_SOURCES`. Source hashes treat CRLF as LF (`source_hash.rs`), so a checkout's line endings do not miss the cache.

A program the translator cannot lower is a fallback with the opcode named. Do not emit a stand-in instruction. `image_get_resinfo` past the last level is 0 for every size, including on AMD. A 1D, 2D, or 2D-MSAA resource reports depth 0. A 1D array reports its layers in y. `v_readfirstlane` in a pixel shader reads a live lane, and a lane-mask test there counts live lanes only.

## Split dispatches

A large direct compute program is cut at CFG edges into a `DispatchChain` (`module/split.rs`, `XCL_DIRECT_SPLIT_INSTRUCTIONS`, default 12500). The PSO the title receives is the first piece (`hle/d3d12x/real/dispatch_chain.rs`). Later pieces and a lazily built whole form belong to that PSO. The pieces share one scratch buffer, capped by `XCL_SPLIT_SCRATCH_MB` (default 256, at most 4096). Groups that do not fit run in slabs, and the cap is across every recording, not one list. `XCL_DIRECT_COMPACT=1` forces the compact form and `0` forbids it.

A GPU test of a new lowering encodes with `rdna2::encode` and checks the lane result on this path and, where the HLSL path still has the op, on that one too (`gpu/dxil/core_tests.rs`, `gpu/recomp/tests.rs`). `XCL_DIRECT_DXIL=1` is how the kernel tests select this backend.
