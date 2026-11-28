use windows::core::HRESULT;
use windows::Win32::Foundation::E_NOTIMPL;
use windows::Win32::Graphics::Dxgi::{IDXGIAdapter, IDXGIFactory4};

pub(super) fn enum_warp_adapter(factory: &IDXGIFactory4) -> Option<IDXGIAdapter> {
    match unsafe { factory.EnumWarpAdapter() } {
        Ok(adapter) => Some(adapter),
        Err(err) if is_missing_warp(err.code(), crate::util::under_wine()) => {
            crate::gpu::recomp::runner::skip_test("no WARP");
            None
        }
        Err(err) => panic!("WARP: {err:?}"),
    }
}

fn is_missing_warp(code: HRESULT, under_wine: bool) -> bool {
    under_wine && code == E_NOTIMPL
}

#[test]
fn warp_adapter_absence_requires_wine_and_not_implemented() {
    use windows::Win32::Foundation::E_FAIL;
    use windows::Win32::Graphics::Dxgi::DXGI_ERROR_NOT_FOUND;

    assert!(is_missing_warp(E_NOTIMPL, true));
    assert!(!is_missing_warp(E_NOTIMPL, false));
    assert!(!is_missing_warp(E_FAIL, true));
    assert!(!is_missing_warp(DXGI_ERROR_NOT_FOUND, true));
}
