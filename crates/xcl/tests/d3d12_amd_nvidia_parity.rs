//! Synthetic dual-vendor (AMD vs NVIDIA) D3D12 parity & memory tests for #2688 (WS3, WS7, WS9).
//!
//! Runs with zero game data on any hardware D3D12 adapter selected by `XCL_ADAPTER` (and skips
//! cleanly on WARP / no-GPU CI hosts):
//! 1. `ws9_adapter_capabilities_and_dxgi_budget`: reports adapter identity, D3D12 feature tiers
//!    (`TiledResourcesTier`, `ResourceHeapTier`, wave sizes, mesh tier, UMA), and DXGI local /
//!    non-local budgets.
//! 2. `ws3_reserved_texture_tiling_overcommit_matrix`: queries `CreateReservedResource` +
//!    `GetResourceTiling` + `GetResourceAllocationInfo` across 3D, MSAA, small-mip, BC7, and
//!    cubemap textures, comparing PC 64 KiB tile footprints against console `SW_64KB_S_X` bytes.
//! 3. `ws7_driver_pipeline_memory_for_same_dxil`: compiles deterministic synthetic DXIL compute
//!    and graphics shaders and measures process private commit + DXGI local/non-local VRAM before
//!    creation, while live, and after `Release()` for both distinct and identical PSOs.

use std::ffi::c_void;
use std::time::Instant;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use xcl::gpu::adapter::{self, Selector};
use xcl::gpu::agility;
use xcl::gpu::desc::{Image, ImageType};
use xcl::gpu::recomp::{dxc, runner};
use xcl::gpu::texture::{self, CONSOLE_PIPES};

struct HwAdapter {
    adapter: IDXGIAdapter3,
    desc: DXGI_ADAPTER_DESC1,
    name: String,
    dev: ID3D12Device,
}

fn open_hw_adapter() -> Option<HwAdapter> {
    let sel = Selector::from_env_or("auto");
    if sel == Selector::Warp {
        runner::skip_test("WARP requested via XCL_ADAPTER");
        return None;
    }
    unsafe {
        let factory: IDXGIFactory6 = match CreateDXGIFactory1() {
            Ok(f) => f,
            Err(e) => {
                runner::skip_test(&format!("CreateDXGIFactory1 failed: {e}"));
                return None;
            }
        };
        let list = adapter::candidates(&factory, &sel);
        for (a, desc) in list {
            if desc.Flags & (DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
                continue;
            }
            if let Ok(dev) = agility::create_device(&a, D3D_FEATURE_LEVEL_11_0) {
                if let Ok(a3) = a.cast::<IDXGIAdapter3>() {
                    let name = adapter::name(&desc);
                    return Some(HwAdapter { adapter: a3, desc, name, dev });
                }
            }
        }
    }
    runner::skip_test("no hardware D3D12 adapter");
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SegmentUsage {
    current: u64,
    budget: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProcessCommit {
    bytes: u64,
    source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MemSnap {
    private_commit: u64,
    commit_source: &'static str,
    local: SegmentUsage,
    /// `None` when `QueryVideoMemoryInfo(NON_LOCAL)` returns an error so an unavailable
    /// segment is never confused with a reported `0` usage/budget.
    nonlocal: Option<SegmentUsage>,
}

fn select_process_commit(private_usage: usize, pagefile_usage: usize) -> Result<ProcessCommit, String> {
    if private_usage > 0 {
        Ok(ProcessCommit { bytes: private_usage as u64, source: "PrivateUsage" })
    } else if pagefile_usage > 0 {
        // Wine returns TRUE from K32GetProcessMemoryInfo with PrivateUsage == 0 while populating
        // PagefileUsage with the process commit charge.
        Ok(ProcessCommit { bytes: pagefile_usage as u64, source: "PagefileUsage" })
    } else {
        Err("K32GetProcessMemoryInfo returned both PrivateUsage=0 and PagefileUsage=0".into())
    }
}

fn mem_snap_from_queries<FProc, FLoc, FNonLoc>(
    proc_query: FProc,
    local_query: FLoc,
    nonlocal_query: FNonLoc,
) -> Result<MemSnap, String>
where
    FProc: FnOnce() -> Result<ProcessCommit, String>,
    FLoc: FnOnce() -> Result<SegmentUsage, String>,
    FNonLoc: FnOnce() -> Result<SegmentUsage, String>,
{
    let commit = proc_query()?;
    let local = local_query()?;
    let nonlocal = nonlocal_query().ok();
    Ok(MemSnap { private_commit: commit.bytes, commit_source: commit.source, local, nonlocal })
}

fn mem_snap(adapter: &IDXGIAdapter3) -> Result<MemSnap, String> {
    unsafe {
        mem_snap_from_queries(
            || {
                let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
                c.cb = std::mem::size_of_val(&c) as u32;
                let ok = K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c as *mut _ as *mut _, c.cb);
                if ok == 0 {
                    return Err(format!("K32GetProcessMemoryInfo failed: {}", std::io::Error::last_os_error()));
                }
                select_process_commit(c.PrivateUsage, c.PagefileUsage)
            },
            || {
                let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                adapter
                    .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local)
                    .map_err(|e| format!("QueryVideoMemoryInfo(LOCAL) failed: {e}"))?;
                Ok(SegmentUsage { current: local.CurrentUsage, budget: local.Budget })
            },
            || {
                let mut nonlocal = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                adapter
                    .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut nonlocal)
                    .map_err(|e| format!("QueryVideoMemoryInfo(NON_LOCAL) failed: {e}"))?;
                Ok(SegmentUsage { current: nonlocal.CurrentUsage, budget: nonlocal.Budget })
            },
        )
    }
}

fn signed_delta_kib(after: u64, before: u64) -> f64 {
    (after as i128 - before as i128) as f64 / 1024.0
}

fn format_optional_mib(bytes: Option<u64>) -> String {
    match bytes {
        Some(b) => (b >> 20).to_string(),
        None => "unavailable".to_string(),
    }
}

fn format_nonlocal_delta(
    before: Option<SegmentUsage>,
    live: Option<SegmentUsage>,
    after: Option<SegmentUsage>,
) -> String {
    match (before, live, after) {
        (Some(b), Some(l), Some(a)) => format!(
            "{:+6.1}->{:+6.1}",
            signed_delta_kib(l.current, b.current),
            signed_delta_kib(a.current, b.current),
        ),
        _ => "unavailable".to_string(),
    }
}

#[test]
fn mem_snap_rejects_required_query_failures_and_marks_unavailable_nonlocal() {
    let ok_commit = || Ok(ProcessCommit { bytes: 1234, source: "PrivateUsage" });
    let ok_local = || Ok(SegmentUsage { current: 4096, budget: 8192 });
    let ok_nonlocal = || Ok(SegmentUsage { current: 0, budget: 0 });

    // Windows populates PrivateUsage (preferred); Wine leaves PrivateUsage == 0 and populates PagefileUsage.
    assert_eq!(
        select_process_commit(8192, 16384),
        Ok(ProcessCommit { bytes: 8192, source: "PrivateUsage" }),
    );
    assert_eq!(
        select_process_commit(0, 16384),
        Ok(ProcessCommit { bytes: 16384, source: "PagefileUsage" }),
    );
    assert!(
        select_process_commit(0, 0).is_err(),
        "both PrivateUsage=0 and PagefileUsage=0 must fail instead of silently reporting 0 commit",
    );

    // A failed K32GetProcessMemoryInfo must fail rather than reporting 0 private commit.
    let err_proc = mem_snap_from_queries(|| Err("proc failed".into()), ok_local, ok_nonlocal);
    assert!(err_proc.is_err(), "process memory query failure must return Err");

    // A failed LOCAL QueryVideoMemoryInfo must fail rather than reporting 0 VRAM/budget.
    let err_local = mem_snap_from_queries(ok_commit, || Err("local failed".into()), ok_nonlocal);
    assert!(err_local.is_err(), "LOCAL DXGI query failure must return Err");

    // A failed NON_LOCAL query marks the segment explicitly unavailable (distinct from 0).
    let unavail = mem_snap_from_queries(ok_commit, ok_local, || Err("nonlocal unsupported".into()))
        .expect("unavailable NON_LOCAL should still return snapshot");
    assert_eq!(unavail.nonlocal, None);
    assert_eq!(format_optional_mib(unavail.nonlocal.map(|s| s.budget)), "unavailable");
    assert_eq!(format_nonlocal_delta(unavail.nonlocal, unavail.nonlocal, unavail.nonlocal), "unavailable");

    // A succeeding NON_LOCAL query reporting 0 stays distinct from unavailable.
    let zero = mem_snap_from_queries(ok_commit, ok_local, ok_nonlocal).expect("zero NON_LOCAL");
    assert_eq!(zero.nonlocal, Some(SegmentUsage { current: 0, budget: 0 }));
    assert_eq!(format_optional_mib(zero.nonlocal.map(|s| s.budget)), "0");
    assert_eq!(format_nonlocal_delta(zero.nonlocal, zero.nonlocal, zero.nonlocal), "  +0.0->  +0.0");
}

#[test]
fn ws9_adapter_capabilities_and_dxgi_budget() {
    let Some(hw) = open_hw_adapter() else { return };
    unsafe {
        let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        hw.dev
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut opts as *mut _ as *mut c_void, std::mem::size_of_val(&opts) as u32)
            .expect("D3D12_OPTIONS");

        let mut opts1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        let has_opts1 = hw
            .dev
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &mut opts1 as *mut _ as *mut c_void, std::mem::size_of_val(&opts1) as u32)
            .is_ok();

        let mut opts7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        let has_opts7 = hw
            .dev
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &mut opts7 as *mut _ as *mut c_void, std::mem::size_of_val(&opts7) as u32)
            .is_ok();

        let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
        hw.dev
            .CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &mut arch as *mut _ as *mut c_void, std::mem::size_of_val(&arch) as u32)
            .expect("ARCHITECTURE1");

        let snap = mem_snap(&hw.adapter).expect("WS9 memory snapshot");
        println!(
            "WS9_ADAPTER | name={} | vendor={:#06x} | device={:#06x} | ded_vram_mib={} | shared_sys_mib={} | uma={} | coherent_uma={} | tiled_tier={} | heap_tier={} | bind_tier={} | wave_ops={} | wave_min={} | wave_max={} | total_lanes={} | mesh_tier={} | local_budget_mib={} | local_used_mib={} | nonlocal_budget_mib={} | nonlocal_used_mib={}",
            hw.name,
            hw.desc.VendorId,
            hw.desc.DeviceId,
            hw.desc.DedicatedVideoMemory >> 20,
            hw.desc.SharedSystemMemory >> 20,
            arch.UMA.as_bool(),
            arch.CacheCoherentUMA.as_bool(),
            opts.TiledResourcesTier.0,
            opts.ResourceHeapTier.0,
            opts.ResourceBindingTier.0,
            has_opts1 && opts1.WaveOps.as_bool(),
            if has_opts1 { opts1.WaveLaneCountMin } else { 0 },
            if has_opts1 { opts1.WaveLaneCountMax } else { 0 },
            if has_opts1 { opts1.TotalLaneCount } else { 0 },
            if has_opts7 { opts7.MeshShaderTier.0 } else { 0 },
            snap.local.budget >> 20,
            snap.local.current >> 20,
            format_optional_mib(snap.nonlocal.map(|s| s.budget)),
            format_optional_mib(snap.nonlocal.map(|s| s.current)),
        );

        assert!(opts.TiledResourcesTier.0 >= 1, "expected tiled resources support on hardware adapter");
        assert!(snap.local.budget > 0, "expected non-zero DXGI local video memory budget");
    }
}

struct TexCase {
    label: &'static str,
    dim: D3D12_RESOURCE_DIMENSION,
    img_ty: ImageType,
    dxgi: DXGI_FORMAT,
    gfx_fmt: u32,
    bpp_or_blk: u32,
    blk_side: u32,
    width: u32,
    height: u32,
    depth_or_array: u16,
    mips: u16,
    samples: u32,
    flags: D3D12_RESOURCE_FLAGS,
}

fn console_bytes_for(c: &TexCase) -> (Option<usize>, Option<usize>, usize) {
    // Raw logical bytes across all mips/slices/samples
    let mut raw = 0usize;
    let slices = if c.dim == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { c.depth_or_array as usize };
    for m in 0..c.mips as u32 {
        let w = (c.width >> m).max(1).div_ceil(c.blk_side) as usize;
        let h = (c.height >> m).max(1).div_ceil(c.blk_side) as usize;
        let d = if c.dim == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            ((c.depth_or_array as u32) >> m).max(1) as usize
        } else {
            1
        };
        raw += w * h * d * (c.bpp_or_blk as usize) * (c.samples as usize);
    }
    raw *= slices;

    if c.samples > 1 {
        return (None, None, raw);
    }
    let measure = |swizzle_mode: u32| {
        let img = Image {
            address: 0,
            format: c.gfx_fmt,
            width: c.width,
            height: c.height,
            depth: c.depth_or_array as u32,
            ty: c.img_ty,
            base_level: 0,
            last_level: (c.mips as u32).saturating_sub(1),
            max_mip: (c.mips as u32).saturating_sub(1),
            base_array: 0,
            dst_sel: [4, 5, 6, 7],
            swizzle_mode,
            min_lod: 0.0,
        };
        texture::placement(&img, CONSOLE_PIPES).ok().map(|p| p.bytes)
    };
    // SW_64KB_S_X (25) and SW_4KB_S_X (21)
    (measure(25), measure(21), raw)
}

#[test]
fn ws3_reserved_texture_tiling_overcommit_matrix() {
    let Some(hw) = open_hw_adapter() else { return };
    let cases = [
        // 3D volumes (worst overcommit class cited in #2688)
        TexCase { label: "3D_33x33x33_RGBA16F_m1", dim: D3D12_RESOURCE_DIMENSION_TEXTURE3D, img_ty: ImageType::D3, dxgi: DXGI_FORMAT_R16G16B16A16_FLOAT, gfx_fmt: 71, bpp_or_blk: 8, blk_side: 1, width: 33, height: 33, depth_or_array: 33, mips: 1, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
        TexCase { label: "3D_33x33x33_RGBA16F_m6", dim: D3D12_RESOURCE_DIMENSION_TEXTURE3D, img_ty: ImageType::D3, dxgi: DXGI_FORMAT_R16G16B16A16_FLOAT, gfx_fmt: 71, bpp_or_blk: 8, blk_side: 1, width: 33, height: 33, depth_or_array: 33, mips: 6, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
        TexCase { label: "3D_64x64x64_RGBA16F_m1", dim: D3D12_RESOURCE_DIMENSION_TEXTURE3D, img_ty: ImageType::D3, dxgi: DXGI_FORMAT_R16G16B16A16_FLOAT, gfx_fmt: 71, bpp_or_blk: 8, blk_side: 1, width: 64, height: 64, depth_or_array: 64, mips: 1, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
        TexCase { label: "3D_64x64x64_RGBA16F_m7", dim: D3D12_RESOURCE_DIMENSION_TEXTURE3D, img_ty: ImageType::D3, dxgi: DXGI_FORMAT_R16G16B16A16_FLOAT, gfx_fmt: 71, bpp_or_blk: 8, blk_side: 1, width: 64, height: 64, depth_or_array: 64, mips: 7, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
        TexCase { label: "3D_160x90x64_RGBA16F_m1", dim: D3D12_RESOURCE_DIMENSION_TEXTURE3D, img_ty: ImageType::D3, dxgi: DXGI_FORMAT_R16G16B16A16_FLOAT, gfx_fmt: 71, bpp_or_blk: 8, blk_side: 1, width: 160, height: 90, depth_or_array: 64, mips: 1, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
        TexCase { label: "3D_65x65x65_R11G11B10F_m1", dim: D3D12_RESOURCE_DIMENSION_TEXTURE3D, img_ty: ImageType::D3, dxgi: DXGI_FORMAT_R11G11B10_FLOAT, gfx_fmt: 36, bpp_or_blk: 4, blk_side: 1, width: 65, height: 65, depth_or_array: 65, mips: 1, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
        TexCase { label: "3D_128x128x32_RGBA8_m8", dim: D3D12_RESOURCE_DIMENSION_TEXTURE3D, img_ty: ImageType::D3, dxgi: DXGI_FORMAT_R8G8B8A8_UNORM, gfx_fmt: 56, bpp_or_blk: 4, blk_side: 1, width: 128, height: 128, depth_or_array: 32, mips: 8, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
        // Render targets & depth
        TexCase { label: "2D_1920x1080_RGBA16F_RT", dim: D3D12_RESOURCE_DIMENSION_TEXTURE2D, img_ty: ImageType::D2, dxgi: DXGI_FORMAT_R16G16B16A16_FLOAT, gfx_fmt: 71, bpp_or_blk: 8, blk_side: 1, width: 1920, height: 1080, depth_or_array: 1, mips: 1, samples: 1, flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET },
        TexCase { label: "2D_3840x2160_RGBA16F_RT", dim: D3D12_RESOURCE_DIMENSION_TEXTURE2D, img_ty: ImageType::D2, dxgi: DXGI_FORMAT_R16G16B16A16_FLOAT, gfx_fmt: 71, bpp_or_blk: 8, blk_side: 1, width: 3840, height: 2160, depth_or_array: 1, mips: 1, samples: 1, flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET },
        TexCase { label: "2D_1920x1080_D32F_DSV", dim: D3D12_RESOURCE_DIMENSION_TEXTURE2D, img_ty: ImageType::D2, dxgi: DXGI_FORMAT_D32_FLOAT, gfx_fmt: 22, bpp_or_blk: 4, blk_side: 1, width: 1920, height: 1080, depth_or_array: 1, mips: 1, samples: 1, flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL },
        // MSAA render targets (may be unsupported as reserved on some adapters; reported cleanly)
        TexCase { label: "2D_1920x1080_RGBA8_MSAA4", dim: D3D12_RESOURCE_DIMENSION_TEXTURE2D, img_ty: ImageType::D2Msaa, dxgi: DXGI_FORMAT_R8G8B8A8_UNORM, gfx_fmt: 56, bpp_or_blk: 4, blk_side: 1, width: 1920, height: 1080, depth_or_array: 1, mips: 1, samples: 4, flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET },
        TexCase { label: "2D_512x512_RGBA16F_MSAA4", dim: D3D12_RESOURCE_DIMENSION_TEXTURE2D, img_ty: ImageType::D2Msaa, dxgi: DXGI_FORMAT_R16G16B16A16_FLOAT, gfx_fmt: 71, bpp_or_blk: 8, blk_side: 1, width: 512, height: 512, depth_or_array: 1, mips: 1, samples: 4, flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET },
        // Small mip chains & BC7 & cubemaps
        TexCase { label: "2D_256x256_RGBA8_m9", dim: D3D12_RESOURCE_DIMENSION_TEXTURE2D, img_ty: ImageType::D2, dxgi: DXGI_FORMAT_R8G8B8A8_UNORM, gfx_fmt: 56, bpp_or_blk: 4, blk_side: 1, width: 256, height: 256, depth_or_array: 1, mips: 9, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
        TexCase { label: "2D_64x64_RGBA8_m7", dim: D3D12_RESOURCE_DIMENSION_TEXTURE2D, img_ty: ImageType::D2, dxgi: DXGI_FORMAT_R8G8B8A8_UNORM, gfx_fmt: 56, bpp_or_blk: 4, blk_side: 1, width: 64, height: 64, depth_or_array: 1, mips: 7, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
        TexCase { label: "2D_32x32_RGBA16F_m6", dim: D3D12_RESOURCE_DIMENSION_TEXTURE2D, img_ty: ImageType::D2, dxgi: DXGI_FORMAT_R16G16B16A16_FLOAT, gfx_fmt: 71, bpp_or_blk: 8, blk_side: 1, width: 32, height: 32, depth_or_array: 1, mips: 6, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
        TexCase { label: "2D_1024x1024_BC7_m11", dim: D3D12_RESOURCE_DIMENSION_TEXTURE2D, img_ty: ImageType::D2, dxgi: DXGI_FORMAT_BC7_UNORM, gfx_fmt: 181, bpp_or_blk: 16, blk_side: 4, width: 1024, height: 1024, depth_or_array: 1, mips: 11, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
        TexCase { label: "2D_256x256_BC7_m9", dim: D3D12_RESOURCE_DIMENSION_TEXTURE2D, img_ty: ImageType::D2, dxgi: DXGI_FORMAT_BC7_UNORM, gfx_fmt: 181, bpp_or_blk: 16, blk_side: 4, width: 256, height: 256, depth_or_array: 1, mips: 9, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
        TexCase { label: "CUBE_128x128x6_RGBA16F_m8", dim: D3D12_RESOURCE_DIMENSION_TEXTURE2D, img_ty: ImageType::Cube, dxgi: DXGI_FORMAT_R16G16B16A16_FLOAT, gfx_fmt: 71, bpp_or_blk: 8, blk_side: 1, width: 128, height: 128, depth_or_array: 6, mips: 8, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
        TexCase { label: "CUBE_64x64x6_RGBA16F_m7", dim: D3D12_RESOURCE_DIMENSION_TEXTURE2D, img_ty: ImageType::Cube, dxgi: DXGI_FORMAT_R16G16B16A16_FLOAT, gfx_fmt: 71, bpp_or_blk: 8, blk_side: 1, width: 64, height: 64, depth_or_array: 6, mips: 7, samples: 1, flags: D3D12_RESOURCE_FLAG_NONE },
    ];

    println!("WS3_TILING_ADAPTER | {}", hw.name);
    let mut created = 0usize;
    unsafe {
        for c in &cases {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: c.dim,
                Alignment: 0,
                Width: c.width as u64,
                Height: c.height,
                DepthOrArraySize: c.depth_or_array,
                MipLevels: c.mips,
                Format: c.dxgi,
                SampleDesc: DXGI_SAMPLE_DESC { Count: c.samples, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
                Flags: c.flags,
            };
            let mut res: Option<ID3D12Resource> = None;
            let hr = hw.dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut res);
            let (console_64k, console_4k, raw_bytes) = console_bytes_for(c);
            let mut alloc_desc = desc;
            alloc_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            let alloc = hw.dev.GetResourceAllocationInfo(0, &[alloc_desc]);
            match (hr, res) {
                (Ok(()), Some(r)) => {
                    created += 1;
                    let mut total_tiles = 0u32;
                    let mut packed = D3D12_PACKED_MIP_INFO::default();
                    let mut tile_shape = D3D12_TILE_SHAPE::default();
                    let mut num_sub = 0u32;
                    hw.dev.GetResourceTiling(&r, Some(&mut total_tiles), Some(&mut packed), Some(&mut tile_shape), Some(&mut num_sub), 0, std::ptr::null_mut());
                    assert!(total_tiles > 0, "{} returned 0 tiles", c.label);
                    let pc_bytes = (total_tiles as usize) * 65536;
                    let ratio_alloc = pc_bytes as f64 / (alloc.SizeInBytes as usize).max(1) as f64;
                    let ratio_4k = pc_bytes as f64 / console_4k.unwrap_or(raw_bytes).max(1) as f64;
                    println!(
                        "WS3_ROW | {:26} | tiles={:4} | pc_kib={:7} | alloc_kib={:7} | c64k_kib={:7} | c4k_kib={:7} | raw_kib={:7} | pc/alloc={:4.2}x | pc/c4k={:4.2}x | shape={}x{}x{} | mips=std:{}/pack:{}/ptiles:{}",
                        c.label,
                        total_tiles,
                        pc_bytes / 1024,
                        alloc.SizeInBytes / 1024,
                        console_64k.map_or(0, |b| b / 1024),
                        console_4k.map_or(0, |b| b / 1024),
                        raw_bytes / 1024,
                        ratio_alloc,
                        ratio_4k,
                        tile_shape.WidthInTexels,
                        tile_shape.HeightInTexels,
                        tile_shape.DepthInTexels,
                        packed.NumStandardMips,
                        packed.NumPackedMips,
                        packed.NumTilesForPackedMips,
                    );
                }
                (Err(e), _) => {
                    println!(
                        "WS3_ROW | {:26} | RESERVED_UNSUPPORTED ({:#x}) | alloc_kib={:7} | c64k_kib={:7} | c4k_kib={:7} | raw_kib={:7}",
                        c.label,
                        e.code().0 as u32,
                        alloc.SizeInBytes / 1024,
                        console_64k.map_or(0, |b| b / 1024),
                        console_4k.map_or(0, |b| b / 1024),
                        raw_bytes / 1024,
                    );
                }
                _ => unreachable!(),
            }
        }
    }
    assert!(created >= 12, "expected at least 12 reserved textures to succeed, got {created}");
}

fn make_compute_hlsl(variant: u32, kind: &str) -> String {
    match kind {
        "small" => format!(
            "RWByteAddressBuffer dst : register(u0);\n\
             [numthreads(64,1,1)]\n\
             void main(uint tid : SV_DispatchThreadID) {{\n\
                 dst.Store(tid * 4u, tid ^ {variant}u);\n\
             }}\n"
        ),
        "medium" => {
            let mut body = String::from(
                "RWByteAddressBuffer dst : register(u0);\n\
                 [numthreads(64,1,1)]\n\
                 void main(uint tid : SV_DispatchThreadID) {\n\
                     uint x = dst.Load(tid * 4u) ^ ",
            );
            body.push_str(&format!("{variant}u;\n    float f = asfloat(x | 0x3F800000u);\n"));
            for i in 0..48u32 {
                let k = variant.wrapping_mul(2654435761).wrapping_add(i.wrapping_mul(0x9E37_79B9));
                body.push_str(&format!(
                    "    x = (x * 1664525u + {k}u) ^ (x >> 13);\n\
                     f = f * 1.000123f + asfloat((x & 0x007FFFFFu) | 0x3F000000u);\n"
                ));
            }
            body.push_str(
                "    uint w = WaveActiveSum(x) ^ asuint(f);\n\
                     dst.Store(tid * 4u, w);\n\
                 }\n",
            );
            body
        }
        "large" => {
            let mut body = String::from(
                "RWByteAddressBuffer dst : register(u0);\n\
                 groupshared uint gs[256];\n\
                 [numthreads(64,1,1)]\n\
                 void main(uint tid : SV_DispatchThreadID, uint gid : SV_GroupIndex) {\n\
                     uint a = dst.Load((tid & 255u) * 4u) ^ ",
            );
            body.push_str(&format!(
                "{variant}u;\n\
                 uint b = a ^ 0xA5A55A5Au;\n\
                 float f0 = asfloat((a & 0x007FFFFFu) | 0x3F800000u);\n\
                 float f1 = asfloat((b & 0x007FFFFFu) | 0x3F800000u);\n\
                 gs[gid] = a; gs[gid + 64u] = b; gs[gid + 128u] = a + b; gs[gid + 192u] = a ^ b;\n\
                 GroupMemoryBarrierWithGroupSync();\n"
            ));
            for i in 0..160u32 {
                let k = variant.wrapping_mul(2246822519).wrapping_add(i.wrapping_mul(3266489917));
                body.push_str(&format!(
                    "    if (((a + {i}u) & 3u) != 0u) {{\n\
                         a = (a ^ gs[(gid + {i}u) & 255u]) * 747796405u + {k}u;\n\
                         f0 = mad(f0, 0.9993f, f1 * 0.0017f + (float)({i}u) * 1e-4f);\n\
                     }} else {{\n\
                         b = (b + gs[(gid + {i}u + 3u) & 255u]) ^ ({k}u);\n\
                         f1 = mad(f1, 0.9989f, f0 * 0.0021f);\n\
                     }}\n"
                ));
            }
            body.push_str(
                "    dst.Store((tid & 255u) * 4u, a ^ b ^ asuint(f0 + f1));\n\
                 }\n",
            );
            body
        }
        _ => unreachable!(),
    }
}

fn make_gfx_hlsl(variant: u32) -> (String, String) {
    let vs = format!(
        "float4 main(uint vid : SV_VertexID) : SV_Position {{\n\
             float2 p = float2((vid & 1u) ? 1.0f : -1.0f, (vid & 2u) ? 1.0f : -1.0f);\n\
             return float4(p, 0.0f, 1.0f);\n\
         }}\n"
    );
    let ps = format!(
        "float4 main(float4 pos : SV_Position) : SV_Target0 {{\n\
             uint v = {variant}u;\n\
             float3 c = float3((v & 255u) / 255.0f, ((v >> 8) & 255u) / 255.0f, ((v >> 16) & 255u) / 255.0f);\n\
             return float4(c + float3(pos.xy * 1e-4f, 0.0f), 1.0f);\n\
         }}\n"
    );
    (vs, ps)
}

#[test]
fn ws7_driver_pipeline_memory_for_same_dxil() {
    let Some(hw) = open_hw_adapter() else { return };
    if dxc::location().is_err() {
        runner::skip_test("no DXC");
        return;
    }

    // Minimal root signature with 1 root UAV (u0) for compute, and empty root signature for graphics.
    let (cs_rs, gfx_rs) = unsafe {
        let param = D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let cs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: 1,
            pParameters: &param,
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };
        let mut blob = None;
        D3D12SerializeRootSignature(&cs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).expect("serialize cs rs");
        let b = blob.unwrap();
        let cs_rs: ID3D12RootSignature = hw
            .dev
            .CreateRootSignature(0, std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize()))
            .expect("create cs rs");

        let gfx_desc = D3D12_ROOT_SIGNATURE_DESC::default();
        let mut gblob = None;
        D3D12SerializeRootSignature(&gfx_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut gblob, None).expect("serialize gfx rs");
        let gb = gblob.unwrap();
        let gfx_rs: ID3D12RootSignature = hw
            .dev
            .CreateRootSignature(0, std::slice::from_raw_parts(gb.GetBufferPointer() as *const u8, gb.GetBufferSize()))
            .expect("create gfx rs");
        (cs_rs, gfx_rs)
    };

    // Pre-compile DXIL blobs BEFORE taking memory snapshots so DXC compiler allocations are excluded
    // from the driver Create*PipelineState deltas.
    let compile_cs_batch = |kind: &str, n: usize, distinct: bool| -> Vec<Vec<u8>> {
        if distinct {
            (0..n)
                .map(|i| dxc::compile(&make_compute_hlsl(0x1000 + i as u32, kind), "main", "cs_6_6", &[]).expect("dxc cs"))
                .collect()
        } else {
            let one = dxc::compile(&make_compute_hlsl(0x9999, kind), "main", "cs_6_6", &[]).expect("dxc cs");
            vec![one; n]
        }
    };

    let cs_small = compile_cs_batch("small", 128, true);
    let cs_medium_w1 = compile_cs_batch("medium", 128, true);
    let cs_medium_w2: Vec<Vec<u8>> = (0..128)
        .map(|i| dxc::compile(&make_compute_hlsl(0x5000 + i as u32, "medium"), "main", "cs_6_6", &[]).expect("dxc cs w2"))
        .collect();
    let cs_medium_dup = compile_cs_batch("medium", 128, false);
    let cs_large = compile_cs_batch("large", 32, true);
    let gfx_pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..128u32)
        .map(|i| {
            let (vs_src, ps_src) = make_gfx_hlsl(0x2000 + i);
            (
                dxc::compile(&vs_src, "main", "vs_6_6", &[]).expect("dxc vs"),
                dxc::compile(&ps_src, "main", "ps_6_6", &[]).expect("dxc ps"),
            )
        })
        .collect();

    // Warm up driver compiler & initial pools once before measuring batches.
    unsafe {
        let mut d = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
        d.pRootSignature = std::mem::ManuallyDrop::new(Some(cs_rs.clone()));
        d.CS = D3D12_SHADER_BYTECODE { pShaderBytecode: cs_small[0].as_ptr() as *const c_void, BytecodeLength: cs_small[0].len() };
        let warmup: ID3D12PipelineState = hw.dev.CreateComputePipelineState(&d).expect("warmup cs pso");
        let _ = std::mem::ManuallyDrop::into_inner(d.pRootSignature);
        drop(warmup);
    }

    let baseline_all = mem_snap(&hw.adapter).expect("WS7 baseline memory snapshot");
    println!("WS7_PSO_ADAPTER | {} | commit_src={}", hw.name, baseline_all.commit_source);

    for (label, blobs) in [
        ("cs_small_128_distinct", &cs_small),
        ("cs_med_128_wave1", &cs_medium_w1),
        ("cs_med_128_wave2_new", &cs_medium_w2),
        ("cs_med_128_identical", &cs_medium_dup),
        ("cs_large_32_distinct", &cs_large),
    ] {
        let avg_dxil = blobs.iter().map(Vec::len).sum::<usize>() / blobs.len();
        let before = mem_snap(&hw.adapter).expect("WS7 pre-batch memory snapshot");
        let t0 = Instant::now();
        let mut psos: Vec<ID3D12PipelineState> = Vec::with_capacity(blobs.len());
        unsafe {
            for b in blobs {
                let mut d = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
                d.pRootSignature = std::mem::ManuallyDrop::new(Some(cs_rs.clone()));
                d.CS = D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as *const c_void, BytecodeLength: b.len() };
                let pso: ID3D12PipelineState = hw.dev.CreateComputePipelineState(&d).expect("create compute PSO");
                let _ = std::mem::ManuallyDrop::into_inner(d.pRootSignature);
                psos.push(pso);
            }
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let live = mem_snap(&hw.adapter).expect("WS7 live memory snapshot");
        drop(psos);
        let after = mem_snap(&hw.adapter).expect("WS7 post-release memory snapshot");
        unsafe {
            assert!(hw.dev.GetDeviceRemovedReason().is_ok(), "device removed during {label}");
        }

        let n = blobs.len() as f64;
        assert_eq!(before.commit_source, baseline_all.commit_source);
        assert_eq!(live.commit_source, baseline_all.commit_source);
        assert_eq!(after.commit_source, baseline_all.commit_source);
        let c_live = signed_delta_kib(live.private_commit, before.private_commit);
        let c_after = signed_delta_kib(after.private_commit, before.private_commit);
        let c_cum = signed_delta_kib(after.private_commit, baseline_all.private_commit);
        let v_live = signed_delta_kib(live.local.current, before.local.current);
        let v_after = signed_delta_kib(after.local.current, before.local.current);
        let v_cum = signed_delta_kib(after.local.current, baseline_all.local.current);
        let nonloc = format_nonlocal_delta(before.nonlocal, live.nonlocal, after.nonlocal);
        println!(
            "WS7_ROW | {:22} | n={:3} | dxil_B={:6} | ms={:7.1} | commit_kib={:+8.1} ({:+.1}/pso) | after_rel={:+8.1} (cum={:+9.1}) | vram_loc={:+7.1} ({:+.1}/pso) | after_loc={:+7.1} (cum={:+8.1}) | nonloc={}",
            label,
            blobs.len(),
            avg_dxil,
            elapsed_ms,
            c_live,
            c_live / n,
            c_after,
            c_cum,
            v_live,
            v_live / n,
            v_after,
            v_cum,
            nonloc,
        );
    }

    // Graphics PSO batch (VS + PS)
    {
        let avg_dxil = gfx_pairs.iter().map(|(v, p)| v.len() + p.len()).sum::<usize>() / gfx_pairs.len();
        let before = mem_snap(&hw.adapter).expect("WS7 pre-batch gfx memory snapshot");
        let t0 = Instant::now();
        let mut psos: Vec<ID3D12PipelineState> = Vec::with_capacity(gfx_pairs.len());
        unsafe {
            for (vs, ps) in &gfx_pairs {
                let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
                d.pRootSignature = std::mem::ManuallyDrop::new(Some(gfx_rs.clone()));
                d.VS = D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as *const c_void, BytecodeLength: vs.len() };
                d.PS = D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as *const c_void, BytecodeLength: ps.len() };
                d.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
                d.SampleMask = u32::MAX;
                d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
                d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
                d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
                d.NumRenderTargets = 1;
                d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
                d.SampleDesc.Count = 1;
                let pso: ID3D12PipelineState = hw.dev.CreateGraphicsPipelineState(&d).expect("create graphics PSO");
                let _ = std::mem::ManuallyDrop::into_inner(d.pRootSignature);
                psos.push(pso);
            }
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let live = mem_snap(&hw.adapter).expect("WS7 live gfx memory snapshot");
        drop(psos);
        let after = mem_snap(&hw.adapter).expect("WS7 post-release gfx memory snapshot");
        unsafe {
            assert!(hw.dev.GetDeviceRemovedReason().is_ok(), "device removed during gfx_vs_ps_128_distinct");
        }

        let n = gfx_pairs.len() as f64;
        assert_eq!(before.commit_source, baseline_all.commit_source);
        assert_eq!(live.commit_source, baseline_all.commit_source);
        assert_eq!(after.commit_source, baseline_all.commit_source);
        let c_live = signed_delta_kib(live.private_commit, before.private_commit);
        let c_after = signed_delta_kib(after.private_commit, before.private_commit);
        let c_cum = signed_delta_kib(after.private_commit, baseline_all.private_commit);
        let v_live = signed_delta_kib(live.local.current, before.local.current);
        let v_after = signed_delta_kib(after.local.current, before.local.current);
        let v_cum = signed_delta_kib(after.local.current, baseline_all.local.current);
        let nonloc = format_nonlocal_delta(before.nonlocal, live.nonlocal, after.nonlocal);
        println!(
            "WS7_ROW | {:22} | n={:3} | dxil_B={:6} | ms={:7.1} | commit_kib={:+8.1} ({:+.1}/pso) | after_rel={:+8.1} (cum={:+9.1}) | vram_loc={:+7.1} ({:+.1}/pso) | after_loc={:+7.1} (cum={:+8.1}) | nonloc={}",
            "gfx_vs_ps_128_distinct",
            gfx_pairs.len(),
            avg_dxil,
            elapsed_ms,
            c_live,
            c_live / n,
            c_after,
            c_cum,
            v_live,
            v_live / n,
            v_after,
            v_cum,
            nonloc,
        );
    }
}
