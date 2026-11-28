# Direct DXIL backend — design and roadmap (2026-11-24)

* **Branch:** `shaders/direct-dxil`
* **Code location:** `crates/xcl/src/gpu/dxil/`

## Goal

Translate console RDNA2 programs straight to signed DXIL containers, with no
HLSL text and no DXC at run time, while keeping the `HLSL → DXC` path as a
per-shader fallback until the direct path matches its 100% corpus coverage.

**Why:**
* **Lower compile latency:** Eliminates DXC invocation overhead on cache misses (which takes hundreds of milliseconds to seconds, and gigabytes of memory, on large translated programs).
* **Zero external runtime dependencies:** Removes reliance on the Windows SDK's `dxcompiler.dll` / `dxil.dll`.
* **Exact float semantics:** Enforces strict floating-point behavior with no implicit `fast-math`.
* **Compact intermediate representation:** Emits compact binary bitcode directly, avoiding the memory and parsing overhead of generating huge intermediate HLSL strings.

## Provenance and licensing

XCL is licensed strictly under **GPL-2.0-only**. References and clean-room boundaries:

* **LLVM 3.7 bitcode format and DXIL specifications** (`DxilOperations`, `DXIL.rst`, and container hashing) — DXC/LLVM codebase under the University of Illinois/NCSA license (GPL-2.0 compatible). Used strictly as format specifications; no third-party code with incompatible sub-licenses is copied.
* **DXC layout oracle:** DXC's *compiled binary output* for small HLSL test programs (`tests/data/dxil/`), inspected via a local bitstream reader solely to verify binary layout compliance.
* **Mesa `src/microsoft/compiler`** (MIT license): May be selectively ported with explicit file-level provenance and copyright attribution notes.

> **Not used:** The Victory project (`GPL-3.0-or-later`, incompatible). Its
> public architecture description informed the high-level approach; no code was
> read for porting or copied.

## Layers (done)

| Module | Role |
| :--- | :--- |
| `bitstream` | LLVM bitstream writer: fixed/VBR fields, blocks with back-patched lengths, and unabbreviated records. |
| `module` | LLVM 3.7 module: interned types, module-level constants, function declarations, single defined entry, instructions (`binop`, `cast`, `cmp`, `select`, `extractvalue`, `call`, `phi`, `load`/`store`/`GEP`, `atomicrmw`/`cmpxchg`, `br`/`switch`/`ret`), metadata, and bitcode writer with relative operand numbering. |
| `shader` | DXIL layer: `dx.op` calls, `createHandleFromBinding`/`FromHeap` + `annotateHandle`, raw buffer load/store, wave ops, system values, entry metadata, and shader flags ↔ `SFI0` bits. |
| `container` | DXBC container: `SFI0`, `ISG1`/`OSG1`, `PSV0` (`RuntimeInfo3`), `DXIL` program header, and DXIL retail hash (verified byte-for-byte against DXC). |
| `validate` | DXC's internal validator and disassembler (matching the pinned test SDK version) used as test oracles. |

**Verified:** The emitted sample kernel (`branch` + `phi`, raw UAV load/store, wave
read-lane, and `WaveSize`) passes the validator, disassembles as expected, and runs
on the GPU with the recompiler's kernel ABI, matching a CPU reference.

## Translator architecture (current model & target design)

The legacy HLSL backend emulates the wave via a uniform `switch (pc)` dispatcher over
register arrays with `EXEC` as an explicit mask, relying heavily on DXC's optimizer. Emitting that pattern unoptimized in DXIL would spill register arrays into scratch memory. The direct backend instead lowers registers into SSA values using a two-stage execution model:

1. **SSA construction (`dxil/ssa.rs`):** Builds a CFG from the decoded program (reusing `rdna2` decode and `recomp` analysis). Each register dword is treated as an SSA variable via the Braun et al. on-the-fly algorithm (no dominance frontiers required). `SCC`, `VCC`, `M0`, and `EXEC` are explicit operands; VGPR writes implicitly read the prior value so inactive lanes retain their state.
2. **Execution & lane-mask lowering (`dxil/lanes.rs`):**
   * *Current implementation (Phases 1–3):* Uses a faithful `EXEC`/ballot predicated model where `EXEC` and `VCC` are tracked as wave masks (via a hash-consed boolean DAG), preserving basic-block control flow while predicating vector instructions per lane. Path-sensitive dataflow elides guards where a lane's `EXEC` bit is proven active. Wave-wide queries (`s_bcnt1` on masks, `v_readlane`, ballots) lower to `WaveActiveBallot` / `WaveReadLaneAt` / `WaveActiveCountBits`.
   * *Future work:* Converting mask-driven branches (such as `s_cbranch_execz` and `vccz`) from instruction predication into true divergent per-lane control-flow branching.
3. **Whole-wave & cross-lane regions:** Code delimited by `s_or_saveexec -1` and its restore (such as DPP/permlane reductions and prefix sums) is symbolically evaluated across console lanes. Recognized associative reductions/prefixes lower to native DXIL wave ops (`WaveActive*` / `WavePrefix*`), paired with multi-wave combining (e.g., via LDS/shared state) when console `wave64` exceeds host `wave32` width. Unrecognized shuffle patterns, or cross-lane operations (`v_readlane`, `DPP`, `permlane`) addressing lane indices outside the host's native wave width without a valid DXIL equivalent, safely fall back to the HLSL path.
4. **Control flow:** DXIL validation requires a reducible CFG; irreducible inputs fall back to the HLSL path. Blocks are ordered in reverse post-order (RPO) to establish a clean dominance and emission schedule, with loop back-edges and incomplete `phi` nodes resolved via block sealing during Braun SSA construction.
5. **Lowering (`dxil/lower.rs`):** Maps machine instructions 1:1 to DXIL ops, sharing semantic tables with the HLSL emitter (float compare classes, DPP/SDWA selects, buffer formats, denorm/rounding modes). Enforces strict IEEE/RDNA semantics: no `fast-math` flags, and explicit NaN/denorm handling where RDNA2 and DXIL diverge (`V_FRACT_F32(+inf)`, `min`/`max` NaN propagation, `v_rcp` precision).
6. **Resources:** Traces descriptor addresses back to root/user-data sources (reusing `T#`/`V#`/`S#` analysis) and maps them to SM 6.6 heap indexing or root descriptors matching `pso.rs`'s ABI.
7. **Light optimization passes:** Runs constant folding, dead code elimination, copy propagation, and guard merging on the SSA IR prior to bitcode emission, leaving heavy backend optimization to the GPU driver.

## Testing strategy

* **Validator & disassembly:** Run DXC's validator and disassembler on every emitted shader in the test suite.
* **Differential GPU testing:** Extend `runner::run` in `recomp/tests.rs` (≈6k lines) to accept raw DXIL bytes alongside HLSL, running identical RDNA2 instruction tests through both backends and asserting bit-identical final registers and memory.
* **Corpus sweep:** Translate the bounded validation corpus (9,282 programs) in time-budgeted batches, logging direct vs. fallback rates and validator pass/fail metrics.
* **Live routing:** Enable direct DXIL routing progressively behind environment flags (`XCL_DIRECT_DXIL=1` for compute and mesh compute passes, and `XCL_DIRECT_DXIL=all` for VS/PS graphics passes), falling back cleanly to HLSL on any unsupported construct.

## Roadmap

| Phase | Content | Exit criterion / Remaining work |
| :--- | :--- | :--- |
| **0** ✅ | Bitstream, module, DXIL layer, container + hash, validator oracle, GPU smoke test | **Done:** Emitted kernel validates and runs |
| **1** ✅ | SSA + predicated per-lane model for straight-line and branchy compute kernels (test ABI); SALU/VALU integer + float ALU | **Done:** Differential tests pass for ALU instruction suite |
| **2** ✅ | Memory: `buffer_*` / `tbuffer_*` / `s_load` / `s_buffer_load` / flat / global, LDS (`ds_*` → `groupshared`), atomics | **Done:** Memory instruction tests pass |
| **3** ✅ | Cross-lane: DPP16/DPP8, `v_readlane`/`v_writelane`, `permlane`, whole-wave reductions | **Done:** Cross-lane tests pass; `wave64` reductions validated on `wave32` |
| **4** *(partial)* | Real compute shaders: resource tracing → heap/root bindings; `XCL_DIRECT_DXIL=1` live routing | **Remaining:** Measure CS corpus coverage and verify live title stability |
| **5** *(partial)* | Graphics: VS/PS signatures (`ISG1`/`OSG1`/PSV elements), interpolation, exports, images/samplers | **Remaining:** Complete VS/PS test coverage and fix live draw regressions |
| **6** *(partial)* | Mesh/two-pass pipelines, mesh output metadata | **Remaining:** Pass mesh pipeline tests and verify geometry output |
| **7** | Default-on for covered shaders, cache keying on backend version | **Target:** Full parity with the HLSL path |

---

### Current Status (main-based port, 2026-11-24)

* **Completed:** Phases 0–3 are complete and verified against the validator and GPU differential runner.
* **In Progress (Phases 4–6 partial):** Live routing is active behind `XCL_DIRECT_DXIL=1` (compute + mesh compute pass) and `XCL_DIRECT_DXIL=all` (VS/PS) with logged HLSL fallbacks; full corpus coverage metrics and live-title runs are pending.
* **Execution Model:** Currently uses the faithful `EXEC`/ballot predicated model over a reducible block CFG; converting mask-driven branches into true divergent per-lane control flow is planned as future work.
* **Coverage Command:** `xcl shaders <pack> --direct`
