//! Pure, no-GPU contract test for the console IDXGIDevice object (`dxgi_device::answers`).
//! The object answers IDXGIObject / IDXGIDevice / IDXGIDevice1 / IDXGIDevice2 and nothing
//! else — the D3D12 interfaces, the DXGI factory, and a zero GUID all stay false.

use windows::core::{GUID, Interface};
use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12Resource};
use windows::Win32::Graphics::Dxgi::{IDXGIDevice, IDXGIDevice1, IDXGIDevice2, IDXGIFactory4, IDXGIObject};

use xcl::hle::d3d12x::dxgi_device::answers;

#[test]
fn answers_true_for_the_four_dxgi_guids() {
    assert!(answers(&IDXGIObject::IID), "IDXGIObject must be answered");
    assert!(answers(&IDXGIDevice::IID), "IDXGIDevice must be answered");
    assert!(answers(&IDXGIDevice1::IID), "IDXGIDevice1 must be answered");
    assert!(answers(&IDXGIDevice2::IID), "IDXGIDevice2 must be answered");
}

#[test]
fn answers_false_for_other_interfaces_and_zero_guid() {
    assert!(!answers(&ID3D12Device::IID), "ID3D12Device must not be answered");
    assert!(!answers(&IDXGIFactory4::IID), "IDXGIFactory4 must not be answered");
    assert!(!answers(&ID3D12Resource::IID), "ID3D12Resource must not be answered");
    assert!(!answers(&GUID::from_u128(0)), "a zero GUID must not be answered");
}
