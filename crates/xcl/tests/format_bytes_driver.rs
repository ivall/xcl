//! `gpu::desc::format_bytes` against the PC driver: for every GFX10 buffer / image format of its
//! range (0–77) that `gpu::desc::dxgi_format` maps, the element size it gives is the one the
//! device's `GetCopyableFootprints` gives for a texture in that DXGI format. Typed buffer views
//! take their element stride from it (`real::buffer_words_at`). No game data; any D3D12 device
//! answers, WARP included.

use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use xcl::gpu::desc;

/// Bytes per element of one row of a 64-texel-wide 2D texture in `format`, as the device lays it out.
unsafe fn driver_element_bytes(dev: &ID3D12Device, format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT) -> u64 {
    let rd = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 64,
        Height: 64,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    let mut row = 0u64;
    dev.GetCopyableFootprints(&rd, 0, 1, 0, None, None, Some(&mut row), None);
    row / 64
}

#[test]
fn format_bytes_matches_the_drivers_element_size() {
    let Ok(dev) = xcl::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
        xcl::gpu::recomp::runner::skip_test("no GPU");
        return;
    };
    let mut checked = 0;
    let mut wrong = Vec::new();
    for f in 0..=77u32 {
        let Some((d, _)) = desc::dxgi_format(f) else { continue };
        let driver = unsafe { driver_element_bytes(&dev, d) };
        if desc::format_bytes(f) as u64 != driver {
            wrong.push(format!("format {f} ({d:?}): format_bytes {}, driver {driver}", desc::format_bytes(f)));
        }
        checked += 1;
    }
    assert!(checked >= 40, "only {checked} formats of 0–77 have a DXGI format");
    assert!(wrong.is_empty(), "{}", wrong.join("; "));
}

/// `gfx10_format` is the inverse of `dxgi_format` for every format it maps, 0–255: no two GFX10
/// formats share a DXGI one, so a view's DXGI format names its GFX10 format back.
#[test]
fn gfx10_format_inverts_dxgi_format() {
    let mut wrong = Vec::new();
    for f in 0..=255u32 {
        if let Some((d, _)) = desc::dxgi_format(f) {
            let back = desc::gfx10_format(d);
            if back != Some(f) {
                wrong.push(format!("format {f} ({d:?}) comes back as {back:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("; "));
}
