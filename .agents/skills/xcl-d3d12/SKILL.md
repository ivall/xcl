---
name: xcl-d3d12
description: >-
  Change XCL's D3D12.X implementation: device, heaps, command lists, pipeline
  state, present, and UMA. Use when a title has no picture, report.json d3d12
  shows unknown slots or IIDs, or when editing hle/d3d12x/.
---

# D3D12.X

`hle/d3d12x/mod.rs`. Exports include `D3D12XboxCreateDevice` and `IID_ID3D12Device` as data.

`real.rs` is the default: PC D3D12 objects with an extended vtable. Driver vtables have internal methods past the public ones, so `guard.rs` checks the caller: a call from the title image is a console slot (log and return 0), anything else goes to the driver. `null.rs` is selected by `graphics = "null"` or when no device is created. `real::init` is idempotent.

Console recording lives in `console_list.rs`, `list_x.rs`, `device_x.rs`, `console_heap.rs`, `bind.rs`, `frame.rs`, `fence_x.rs`, `vram.rs`, `sparse.rs`, `title_textures.rs`. Public COM layout is generated: `scripts/gen_d3d12_layout.py` → `hle/d3d12_layout.rs`.

Graphics PSO creation calls `gpu/recomp/pso.rs` for native VS/PS (and the mesh/GS paths the backend already hooks). Failure falls back to replacement shaders. Counters: `translated_pipelines`, `translation_errors`, `fallback_pipelines`.

`present.rs`: XCL window and swap chain. The window title shows live FPS. `present` config is `auto` or `off`. `d3d12.presents` increments per present, including empty frames. `XCL_FRAME_DUMP=N` fills `last_frame_dump` (every 4th pixel unless `XCL_FRAME_DUMP_FULL=1`).

`CheckFeatureSupport` raytracing tier follows `raytracing` / `XCL_RAYTRACING` (`auto`, `tier1_1`, `tier1_0`, `off`). Adapter: `d3d12_adapter` / `XCL_ADAPTER`. Debug layer: `d3d12_debug` or `XCL_D3D12_DEBUG=1` (needs the Agility SDK layers on Windows 10; see README).

Also in this directory: `pso_async.rs` (pipeline builds), `alloc_pool.rs` (reusable allocators), `late_release.rs`, `shader_cache_session.rs`, `cpu_writes.rs`, `title_list.rs`.

## Memory windows

UMA is 4 GiB windows: 4 by default, or 8 or 16 with `XCL_UMA_WINDOWS` and `XCL_HEAP_WINDOWS=1` (`gpu/uma.rs`). Wider than 4 without heap windows stays at 4. Sparse VRAM windows stay capped at 3 (12 GiB). An 8-window layout has 5 coherent windows and a 16-window layout has 13; do not raise the sparse cap to fill the aperture. Shader access switches on the high bits over root UAVs `mem`, `mem1`.. (`Options::mem_windows`). `XCL_HEAP_WINDOWS=1` is opt-in: window k is descriptor `MEM_HEAP_BASE + k` (`MEM_HEAP_BASE` is 4) on the shared heap (`console_heap::set_window_views`), one access instead of that switch. Descriptors 0..4 of that heap are null views. It stays off when the views could not be made. A pipeline the driver refuses with heap windows is built again in the switch form (`pso::in_switch_form`); both forms share the root signature and the binds. A raw window UAV is at most 4 GiB − 32 bytes (`runner::raw_window_uav`). A view that ends higher removes the device on older D3D12 runtimes, and a view of 4 GiB − 64 KiB leaves the last tile unreachable. `SetDescriptorHeaps(0, NULL)` clears the list's record of bound heaps. A dispatch with heap windows still needs the shared heap when the list set none.

A copy inside one memory window reads through a scratch buffer. Scratch is capped and retired. A new command list does not take an earlier list's scratch.

A large direct compute program is several pipelines (`real/dispatch_chain.rs`, `xcl-dxil`). The title is given the first piece. The pieces share one scratch buffer (`XCL_SPLIT_SCRATCH_MB`, default 256 MiB), groups that do not fit run in slabs, and that cap is across every recording.

The newer command-list layout's `DispatchMesh` is slot 129, which is the older slot 127. Slot 128 is not `DispatchMesh`. `R9G9B9E5_SHAREDEXP` render targets and UAVs are created as `R11G11B10_FLOAT`. A list `Reset` while its last submit is still running records on a spare list. A submit that overlaps the previous one is logged. `ExecuteIndirect` reads the arguments written before it on that list. Title-texture upload batches are capped by count and 32 MiB.

## Pipelines

Builds are asynchronous (`pso_async.rs`). A bind waits until the pipeline is built unless `XCL_PSO_WAIT_MS` is a positive number of milliseconds; unset and `0` wait with no budget. The pipeline being bound is built on that thread when no worker has started it, so a long queue does not hold it. `ExecuteIndirect` while its pipeline is still building is skipped. The driver's blob of a compute program is stored under the DXC cache's `driver/` directory and reused on the next run (`XCL_PSO_DISK_CACHE=0` turns that off). Pipelines an earlier run used are built ahead of first use. `CreateShaderCacheSession` falls back to an in-memory session (`shader_cache_session.rs`, 32 MB cap) when the PC runtime refuses the real one. A root signature whose constant count would overflow gets no layout (`console_list::layout_from_blob` returns `None`). A pipeline stream cut inside a subobject payload is left as the title wrote it.

## Resources, lists, textures

A reserved resource is recorded at creation, by address and description. Ask the device for tiled support before making a GPU-owned texture reserved; a format without tiled support is not reservable, whatever an MSAA query says. `Map` of a resource the PC cannot map returns without calling the PC `Map`. `CreatePlacedResource` converts the clear value of a console format. Default SRVs of multisampled depth buffers are `Texture2DMS`. Depth and UAV buffers on the PC fallback paths start as depth writes. Allocation info for several resources is padded out to their alignment. Constant buffer views are float4 rows (stride 16).

Allocator recordings move only after a successful `Reset`. A released list drops them. A failed `Reset` leaves the allocator reusable. A released allocator leaves `alloc_pool`. A resource's last `Release` waits until the queues have passed the work still in flight. The late-release drain takes the sparse queue without holding the sparse state's lock. A sparse pool retired a moment ago may be taken back. Decommitted sparse tiles stay mapped until the next submit.

`title_list` splits a recorded list around a mid-list `Wait64` that follows a `Write64`. `Wait64BitValueX` is bridged to a GPU fence; keep the cycle break that bridge already has. DRED reports list a partially executed command list ahead of the others.

Title textures (`title_textures.rs`) are a detiled PC copy of memory the title placed. The first upload waits for the next `ExecuteCommandLists`, in batches, one list per batch. Detile and hash the title's pages in place. Skip an upload whose pages match the last one. A batch of textures is detiled on several threads, and the queue reuses its upload buffer and list. Kept-in-step textures are backed by shared VRAM pages. A read tolerates a page decommitted while it runs. GPU writes into the title's copy of that memory are not seen; the title's own draws go to the PC texture.

A new console method needs a slot test or an e2e draw (`crates/xcl/tests/gpu_pso_e2e.rs` pattern): translate, draw, read back pixels. Do not treat a present count as success.
