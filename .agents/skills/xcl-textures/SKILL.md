---
name: xcl-textures
description: >-
  Decode Xbox GFX10 tiled textures and T#, S#, V# descriptors in XCL. Use when
  textures are scrambled, xg_xs is stubbed, or when editing gpu/tiling.rs,
  gpu/desc.rs, gpu/texture.rs, or hle/xg.rs.
---

# Textures

`gpu/tiling.rs` and generated `tiling_tables.rs` (`scripts/gen_tiling.py`, AMD addrlib patterns): GFX10 2D swizzles, 3D thick modes, linear, mip tails, arrays, cubes, volume. A layout's offset tables are cached by `(is_3d, mode, elem_log2, pipe config)`. Retail check is byte-identical against `XCL_SAMPLE_TTPF`, not a visual guess. A 3D texture with a packed mip tail is detiled with that tail (`title_textures.rs`).

`gpu/desc.rs`: GFX10 image, buffer, and sampler descriptors to D3D12. Formats 0–77 match the buffer table; 128+ sRGB and packed; 134 (`1_5_5_5`) is `B5G5R5A1_UNORM`, and `gfx10_format` of that DXGI format is 134; 169–182 BC1–BC7. SRV channel mapping follows DST_SEL. Static samplers baked into shader code are still samplers (`desc::sampler_d3d12`).

`hle/xg.rs` (`xg_xs.dll`) exports `XGCreateTextureComputer` and `XGSuggestSwizzleMode` only. `optimal_mode` picks the 2D swizzle (depth, layout 2, render target, UAV size). Every other `xg` import stays `E_NOTIMPL` via `stubdb.toml`. Do not answer a surface query with success and an empty descriptor.

Placement: tiled texels are detiled into a PC resource (`hle/d3d12x/title_textures.rs`, `gpu/texture.rs`). Detile and hash the title's pages in place. A layout's high-coordinate tables hold one period. A linear level is detiled a row at a time. An array or cube the driver will not create as reserved is committed. UMA addresses in the descriptor must match `gpu/uma.rs` windows. A GPU upload test that compares texels is the bar (`crates/xcl/tests/texture_states.rs`). Upload timing and the write watch are in `xcl-d3d12`.
