//! Reserved-resource tiling parity between XCL's D3D12.X path and the host D3D12 driver (#480).
//!
//! Oracle split:
//! - The host D3D12 driver is the oracle for what the PC also executes: the same reserved
//!   resource created through XCL's `vram::place_pages` (`CreateReservedResource` with the
//!   layout rewritten to `D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE`) must return identical
//!   `GetResourceTiling` values whether queried through XCL's device vtable slot or directly
//!   through the driver's original `ID3D12Device::GetResourceTiling` entry.
//! - The resource XCL hands back must still be the tiled layout it asked for (`GetDesc`), and
//!   its tiling must satisfy the D3D12 spec invariants that hold on every vendor and driver:
//!   the standard-tile grid and the packed-mip tiles add up to the reported total, and the
//!   standard grid and the packed-mip range each fit inside that total. Drivers differ on
//!   whether a small texture is a 1x1 standard tile (AMD) or a packed-mip tile (NVIDIA reports
//!   the packed 8x8 as a 0x0 grid with one packed tile, and 0xFFFFFFFF as that subresource's
//!   start tile), so neither the grid nor that sentinel is pinned: the accounting is.
//! - Console-only D3D12.X semantics belong in `xg_oracle.rs` (`XCL_SAMPLE_XG`), not here.

use std::ffi::c_void;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::{self, layout, real, vram, Backend};

/// `GetResourceTiling` answers that every driver must agree on: the total tile count, the
/// standard-tile grid of the first (only) subresource, where those tiles start, the packed-mip
/// accounting (including where its range starts) and the tile shape in texels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Tiling {
    total_tiles: u32,
    grid: u64,
    start_tile: u32,
    standard_mips: u8,
    packed_mips: u8,
    packed_start: u32,
    packed_tiles: u32,
    tile_shape: (u32, u32, u32),
}

struct Case {
    name: &'static str,
    format: DXGI_FORMAT,
    width: u64,
    height: u32,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
    clear_depth: bool,
    /// The standard D3D12 tile shape for the format, in texels (independent of the vendor):
    /// 4-byte texel formats tile as 128x128x1, BC1/BC4 (8 bytes per 4x4 block) as 512x256x1,
    /// BC5 (16 bytes per 4x4 block) as 256x256x1.
    tile_shape: (u32, u32, u32),
}

const CASES: &[Case] = &[
    Case {
        name: "R8G8B8A8_UNORM 8x8",
        format: DXGI_FORMAT_R8G8B8A8_UNORM,
        width: 8,
        height: 8,
        flags: D3D12_RESOURCE_FLAG_NONE,
        state: D3D12_RESOURCE_STATE_COMMON,
        clear_depth: false,
        tile_shape: (128, 128, 1),
    },
    Case {
        name: "R8G8B8A8_UNORM 256x256",
        format: DXGI_FORMAT_R8G8B8A8_UNORM,
        width: 256,
        height: 256,
        flags: D3D12_RESOURCE_FLAG_NONE,
        state: D3D12_RESOURCE_STATE_COMMON,
        clear_depth: false,
        tile_shape: (128, 128, 1),
    },
    Case {
        name: "BC1_UNORM 256x256",
        format: DXGI_FORMAT_BC1_UNORM,
        width: 256,
        height: 256,
        flags: D3D12_RESOURCE_FLAG_NONE,
        state: D3D12_RESOURCE_STATE_COMMON,
        clear_depth: false,
        tile_shape: (512, 256, 1),
    },
    Case {
        name: "BC5_UNORM 512x512",
        format: DXGI_FORMAT_BC5_UNORM,
        width: 512,
        height: 512,
        flags: D3D12_RESOURCE_FLAG_NONE,
        state: D3D12_RESOURCE_STATE_COMMON,
        clear_depth: false,
        tile_shape: (256, 256, 1),
    },
    Case {
        name: "D32_FLOAT 8x8",
        format: DXGI_FORMAT_D32_FLOAT,
        width: 8,
        height: 8,
        flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        state: D3D12_RESOURCE_STATE_DEPTH_WRITE,
        clear_depth: true,
        tile_shape: (128, 128, 1),
    },
];

type GetResourceTilingFn = unsafe extern "system" fn(
    *mut c_void,
    *mut c_void,
    *mut u32,
    *mut D3D12_PACKED_MIP_INFO,
    *mut D3D12_TILE_SHAPE,
    *mut u32,
    u32,
    *mut D3D12_SUBRESOURCE_TILING,
);

/// Read the outputs of a raw `GetResourceTiling` (driver vtable entry or XCL's adopted slot).
unsafe fn read_tiling(call: GetResourceTilingFn, dev: *mut c_void, res: *mut c_void) -> Tiling {
    let mut total = 0u32;
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let mut shape = D3D12_TILE_SHAPE::default();
    let mut sub_count = 1u32;
    let mut sub = D3D12_SUBRESOURCE_TILING::default();
    call(
        dev,
        res,
        &mut total,
        &mut packed,
        &mut shape,
        &mut sub_count,
        0,
        &mut sub,
    );
    assert_eq!(sub_count, 1, "expected 1 subresource tiling entry");
    let grid = sub.WidthInTiles as u64 * sub.HeightInTiles as u64 * sub.DepthInTiles as u64;
    Tiling {
        total_tiles: total,
        grid,
        start_tile: sub.StartTileIndexInOverallResource,
        standard_mips: packed.NumStandardMips,
        packed_mips: packed.NumPackedMips,
        packed_start: packed.StartTileIndexInOverallResource,
        packed_tiles: packed.NumTilesForPackedMips,
        tile_shape: (shape.WidthInTexels, shape.HeightInTexels, shape.DepthInTexels),
    }
}

/// Query `GetResourceTiling` through XCL's adopted device vtable slot.
unsafe fn query_via_xcl_slot(dev: &ID3D12Device, res: &ID3D12Resource) -> Tiling {
    let slot = layout::slot(
        layout::find("ID3D12Device").expect("ID3D12Device in layout"),
        "GetResourceTiling",
    )
    .expect("GetResourceTiling slot");
    let f: GetResourceTilingFn = std::mem::transmute(orig_slot_fn(dev.as_raw(), slot));
    read_tiling(f, dev.as_raw(), res.as_raw())
}

/// Query `GetResourceTiling` directly via the host driver's original vtable entry on the same
/// returned `ID3D12Resource`.
unsafe fn query_via_driver(dev: &ID3D12Device, res: &ID3D12Resource) -> Tiling {
    let slot = layout::slot(
        layout::find("ID3D12Device").expect("ID3D12Device in layout"),
        "GetResourceTiling",
    )
    .expect("GetResourceTiling slot");
    let f: GetResourceTilingFn = std::mem::transmute(real::orig_fn(dev.as_raw(), slot));
    read_tiling(f, dev.as_raw(), res.as_raw())
}

/// The function pointer XCL's adopted device currently serves for a vtable slot: the override
/// when XCL hooks it, the driver's original entry otherwise. `real::orig_fn` would return the
/// *original* (pre-adoption) entry, so for the adopted slot we read the live vtable the way the
/// COM call does. This is the function XCL's own `ID3D12Device` services the method with.
unsafe fn orig_slot_fn(dev: *mut c_void, slot: usize) -> usize {
    let vtbl = *(dev as *const *const usize);
    *vtbl.add(slot)
}

#[test]
fn reserved_resource_tiling_matches_host_driver() {
    if d3d12x::backend() != Backend::Real {
        runner::skip_test("no GPU");
        return;
    }
    if std::env::var_os("XCL_VRAM_ARENAS").is_some() {
        runner::skip_test("XCL_VRAM_ARENAS disables reserved-resource page placement");
        return;
    }

    unsafe {
        let mut dev_raw: *mut c_void = std::ptr::null_mut();
        let hr = d3d12x::D3D12XboxCreateDevice(
            std::ptr::null_mut(),
            std::ptr::null(),
            &ID3D12Device::IID as *const windows::core::GUID as *const c_void,
            &mut dev_raw,
        );
        assert!(hr >= 0 && !dev_raw.is_null(), "D3D12XboxCreateDevice failed: {hr:#x}");
        let dev = ID3D12Device::from_raw(dev_raw);

        let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        if dev
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                &mut opts as *mut _ as *mut c_void,
                std::mem::size_of_val(&opts) as u32,
            )
            .is_err()
            || opts.TiledResourcesTier == D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED
        {
            runner::skip_test("no tiled resources");
            return;
        }
        if opts.ResourceHeapTier.0 < D3D12_RESOURCE_HEAP_TIER_2.0 {
            runner::skip_test("no tier 2 resource heaps");
            return;
        }

        let res_iid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;

        for (i, case) in CASES.iter().enumerate() {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: case.width,
                Height: case.height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: case.format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: case.flags,
            };
            let clear = D3D12_CLEAR_VALUE {
                Format: case.format,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 },
                },
            };
            let clear_ptr = if case.clear_depth {
                &clear as *const D3D12_CLEAR_VALUE as *const c_void
            } else {
                std::ptr::null()
            };

            let va = 0x20_0000_0000usize + i * 0x10_0000;
            let mut res_raw: *mut c_void = std::ptr::null_mut();
            let placed = vram::place_pages(
                dev.as_raw(),
                va,
                &desc,
                case.state.0 as u32,
                clear_ptr,
                None,
                res_iid,
                &mut res_raw,
            );
            let hr = placed.unwrap_or_else(|| panic!("{}: vram::place_pages returned None", case.name));
            assert!(hr >= 0 && !res_raw.is_null(), "{}: CreateReservedResource failed: {hr:#x}", case.name);
            let res = ID3D12Resource::from_raw(res_raw);

            // 1. Same resource, host driver as oracle: XCL's device slot and the driver's
            //    original GetResourceTiling must return identical tiling.
            let xcl_tiling = query_via_xcl_slot(&dev, &res);
            let driver_tiling = query_via_driver(&dev, &res);
            assert_eq!(
                xcl_tiling, driver_tiling,
                "{}: XCL slot ({xcl_tiling:?}) != host driver ({driver_tiling:?})",
                case.name
            );

            // 2. The layout rewrite stuck: the resource is a 64 KiB swizzled reserved texture.
            assert_eq!(
                res.GetDesc().Layout,
                D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
                "{}: placed resource lost the 64 KiB swizzle layout",
                case.name
            );

            // 3. Tiling invariants that hold on every D3D12 driver (the total is the standard
            //    grid plus the packed-mip tiles, and all tiles fit in the resource). A subresource
            //    in the packed mip tail reports 0xFFFFFFFF as its start tile (NVIDIA does this for
            //    a small or packed single mip); the packed range itself comes from
            //    D3D12_PACKED_MIP_INFO and is what must fit.
            let t = xcl_tiling;
            assert_eq!(t.standard_mips as u16 + t.packed_mips as u16, 1, "{}: one mip is standard or packed, not both", case.name);
            assert_eq!(t.total_tiles as u64, t.grid + t.packed_tiles as u64, "{}: total tiles = standard grid + packed mip tiles", case.name);
            if t.grid > 0 {
                assert!(t.start_tile as u64 + t.grid <= t.total_tiles as u64, "{}: standard tiles fit in the total", case.name);
            } else {
                assert!(t.packed_tiles >= 1, "{}: a packed 0x0 grid has packed tiles", case.name);
                assert!(
                    t.packed_start as u64 + t.packed_tiles as u64 <= t.total_tiles as u64,
                    "{}: the packed mip range fits in the resource (start {}, {} tiles, total {})",
                    case.name,
                    t.packed_start,
                    t.packed_tiles,
                    t.total_tiles
                );
            }

            // 4. The tile shape is the format's standard, on every vendor.
            assert_eq!(t.tile_shape, case.tile_shape, "{}: tile shape", case.name);
        }
    }
}