# Compile time of translated shaders: what has been measured

Status: notes only, no code. This page collects the compile-time measurements published in
issue #156 and PRs #187 and #214. Every number is copied from the comment or commit named next
to it; none was measured for this page. All come from NVIDIA GPUs, most from synthetic compute
programs, each from one machine. A translated program is compiled twice: by DXC (HLSL → DXIL, on
the CPU) and by the driver's `Create*PipelineState` (`nvgpucomp64.dll` on NVIDIA). The DXC
version and the memory-window form change both.

## Why it matters

Issue #156 (DaftPunkk1, RTX 5070, retail title): one `XMemMapPhysicalPagesScatter` call of 6 720
pages ran for more than 23 s and the title's hang watchdog fired (`mov dword [0], 0`). That call is
not a compile, and the watchdog's timeout is not known (gw1337x hit the same site at 31.5 s, #187),
but several cold driver compiles below take longer than 23 s if a title thread waits for them.

## Driver time grows faster than DXIL size

imnicho, issue #156 (RTX 5080, driver 610.47, DXC 1.9.2602; 114 retail compute programs,
split wave64, 4 windows, one at a time, driver cache files removed): driver time per KB rose from
8.0 ms at ≤ 32 KB (0.21 s each) through 16.6 ms at ≤ 128 KB to 28.5 ms at ≤ 512 KB (8.66 s each).
Two synthetic probes in #156 agree. Kalmarv (RTX 5070 Ti, 610.74) measured 6–7 ms per KB at
60–305 KB and about 14 ms at 633 KB. nFaktorial (RTX 3070, 610.88, straight-line code) measured
10.5 s for 298 KB, 31 s for 613 KB, 55 s for 863 KB and over 2 minutes for about 1.1 MB.

Size does not predict driver time on its own. With dense memory accesses the form matters at equal
size: 447 KB of four-window `switch` form took 139.5 / 159.6 s against 20.2 / 21.3 s for 416 KB of
one-window form (Kalmarv, #187, RTX 5070 Ti, DXC 1.9). In the E2 table below, 512 KB of DXC 1.9
switch form took longer in the driver than 695 KB of DXC 1.7 one-window form.

## The DXC version changes DXIL size, so byte thresholds are version-relative

Kalmarv, #214: E2 at 700 blocks / 19,601 instructions, RTX 5070 Ti (NVIDIA 610.74). Each cell
gives DXIL bytes, the median DXC time in seconds, and the range of three driver times in seconds:

| E2 form | SDK DXC 1.7.2308.16 | DXC 1.9.2602.17 |
|---|---:|---:|
| one window | 695,172 / 4.43 / 0.05–6.62 | 190,328 / 3.97 / 0.02–2.99 |
| four, switch | 1,035,432 / 6.90 / 0.11–57.76 | 512,328 / 61.37 / 0.06–59.34 |
| four, heap | 807,024 / 4.85 / 0.04–6.97 | 237,660 / 4.84 / 0.02–3.50 |

With DXC 1.9 the same program is 2.0–3.7× smaller here, and #156's synthetic F case shrank from
633 KB to 235 KB. Not for every program: Kalmarv later found the shrink only where XCL compiles at
-O1 (HLSL over 1 MiB), and nFaktorial saw 8–28 % on straight-line code above 1 MB of HLSL and
nothing below (both in #156).

On `main`, "big" is a fixed `BIG_DRIVER_COMPILE` = 256 KiB of DXIL (`real.rs`). With SDK DXC 1.7
all three E2 forms are above it, with DXC 1.9 only the switch form (Kalmarv, #214). #214 (open)
makes it `XCL_BIG_PSO_KB`; its suggested 128 for DXC 1.9 is not measured. State any threshold
together with the DXC version it was chosen for.

## Four windows through a `switch` cost about 12× in DXC; heap windows remove most of it

#187's premise (commit `c3fb89e`, RemingtonWilcox): with more than one window, every access
"was inlined once per window (a switch over the root UAVs mem, mem1..)". #187 makes each access a
single `ResourceDescriptorHeap[NonUniformResourceIndex(MEM_HEAP_BASE + w)]` (`XCL_HEAP_WINDOWS=0`
gives back the switch).

yous0serious, #214: E2, 700 blocks, i7-13700KF, RTX 4090 (driver 32.0.16.2101), medians of 3.
The post does not say which DXC produced the DXIL column; its sizes match Kalmarv's DXC 1.9 ones.

| form | DXIL | DXC 1.8.2502 | DXC 1.9.2609 | driver (3 runs, sorted) |
|---|---:|---:|---:|---|
| 1 window | 190 KB | 3.80 s | 3.49 s | 0.00, 0.00, 3.7–3.8 s |
| 4 windows, `switch` | 512 KB | **54.2 s** | **44.2 s** | 0.00, 0.00, 7.9 s |
| 4 windows, heap (#187) | 238 KB | 4.6 s | 4.0 s | 0.00, 0.00, 1.2–1.4 s |

E2 DXC time for each form, from every published run:

| source | machine | DXC | 1 window | switch | heap | switch ÷ 1 window |
|---|---|---|---:|---:|---:|---:|
| #187 `c3fb89e` | RTX 3090 | not stated | 12.1 s | 150.7 s | 12.5 s | 12.5× |
| yous0serious, #214 | RTX 4090 | 1.8.2502 | 3.80 s | 54.2 s | 4.6 s | 14.3× |
| yous0serious, #214 | RTX 4090 | 1.9.2609 | 3.49 s | 44.2 s | 4.0 s | 12.7× |
| Kalmarv, #214 | RTX 5070 Ti | 1.9.2602.17 | 3.97 s | 61.37 s | 4.84 s | 15.5× |
| Kalmarv, #214 | RTX 5070 Ti | 1.7.2308.16 | 4.43 s | 6.90 s | 4.85 s | 1.6× |
| MiguelMachado-dev, #187 (250 blocks) | RTX 5080 | 1.8.2502.11 | 2.55 s | 21.56 s | 2.78 s | 8.5× |

- With DXC 1.8 / 1.9 (and the RTX 3090's unstated DXC) at 700 blocks, the switch costs
  12.5–15.5× the one-window DXC time. The heap form stays within 1.0–1.3× of one window.
- With SDK DXC 1.7 the switch costs only 1.6× in DXC, but its cold driver compile took 57.76 s
  against 6.62 s for one window: the cost moves into the driver, and heap windows remove it there
  too (6.97 s). Kalmarv's dense driver bench in #187 agrees (heap 3.6 s, switch 139.5 / 159.6 s).
- Retail programs (mariuslozo-boop, #156, RTX 5070 Ti, 616.92): heap windows cut driver time
  2–2.5× on MS / PS and 12–15× on 1.3 MB ray-tracing CS, but five pipelines were refused with
  `E_OUTOFMEMORY` in the heap form on that driver. #187 (closed) rebuilt those in the switch form.

## Only the cold driver number means anything

The driver keeps pipelines it has built, in the process and in its disk cache. E2 uses a random
literal per run so that the cache cannot serve it, yet two of three driver compiles per form still
came back in under 10 ms (yous0serious, #214). Only the slowest of the three is a real compile.
The low ends of Kalmarv's ranges (0.02–0.11 s) are the same effect, and his driver cache state
was not controlled. A median of three driver times is therefore a cache hit. To get a cold number,
use a fresh seed in a new process (Kalmarv, #156) or remove the driver's cache files for the test
executable (imnicho, #156). A title run with warm caches measures no compile at all
(MaurycyBartczak, #156: 0 DXC compilations).

## What is not known

- **AMD:** no compile-time numbers at all. The RX 7800 XT runs in #187 checked correctness only,
  and the AMD cost of `NonUniformResourceIndex` is unmeasured.
- **Drivers:** each table comes from one machine and one driver (4090: 32.0.16.2101, 5070 Ti:
  610.74, or 616.92 for the retail rows, 5080: 610.47, 3070: 610.88); none was repeated on another.
- **Synthetic programs:** E2 is a synthetic compute program; no graphics pipeline was timed in it.
  The only retail timings are imnicho's and mariuslozo-boop's in #156.
- **DXC versions:** why 1.9 shrinks some programs and not others (Kalmarv's -O1 finding, one machine).
- **Thresholds:** the title's watchdog timeout, and the right `XCL_BIG_PSO_KB` per DXC version.

## Reproducing

None of these three tools is on `main` yet:

- **E2 bench**: `e2_window_compile_time` in `gpu/recomp/aperture_tests.rs` (#187, closed), ignored; run it
  alone with `cargo test --release -p xcl --lib e2_window_compile_time -- --ignored --nocapture`.
  `XCL_E2_BLOCKS` (default 700) sets the size and `XCL_E2_REPS` (default 3) the runs per form;
  `XCL_DXC_DIR` picks the DXC. Report GPU, driver, CPU and DXC version; the slowest driver time
  is the cold one.
- **`XCL_TRACE_PSO_TIME=1`** (#214): logs every translated driver compile with its time and its
  wait for a slot. Without it, only compiles or waits of 2 s or more are logged.
- **`XCL_BIG_PSO_KB`** (#214): the big-pipeline threshold in KiB, default 256. On `main` it is
  the fixed `BIG_DRIVER_COMPILE`.
