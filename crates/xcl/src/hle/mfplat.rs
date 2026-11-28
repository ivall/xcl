//! `mfplat.dll` – Media Foundation console helpers on top of Windows Media Foundation.
//!
//! All standard Media Foundation exports (`MFStartup`, `MFCreateDXGIDeviceManager`,
//! `MFCreateMediaType`, `MFCreateSample`, `MFCreateAttributes`, …) fall back transparently
//! to the system `mfplat.dll` (see `hle::build`).
//!
//! Xbox GDK titles additionally import `MFResetDXGIDeviceManagerX`, which binds a D3D12
//! device to an `IMFDXGIDeviceManager`. On Windows PC, `IMFDXGIDeviceManager::ResetDevice`
//! expects an `ID3D11Device` (typically with video + BGRA support and multithread
//! protection enabled) and rejects real `ID3D12Device`s (while XCL's null D3D12 backend
//! answers every IID and would poison the manager if passed directly). When `pDevice`
//! implements `ID3D12Device`, we bind a cached companion D3D11 device on the same adapter
//! LUID with `ID3D10Multithread::SetMultithreadProtected(TRUE)` enabled.

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::{Mutex, OnceLock};

use windows::core::{GUID, IUnknown, Interface};
use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory4};
use windows::Win32::Media::MediaFoundation::IMFDXGIDeviceManager;

use crate::util::{E_POINTER, HRESULT};

const D3D_DRIVER_TYPE_UNKNOWN: i32 = 0;
const D3D_DRIVER_TYPE_HARDWARE: i32 = 1;
const D3D_DRIVER_TYPE_WARP: i32 = 5;

const D3D11_CREATE_DEVICE_BGRA_SUPPORT: u32 = 0x20;
const D3D11_CREATE_DEVICE_VIDEO_SUPPORT: u32 = 0x800;
const D3D11_SDK_VERSION: u32 = 7;

const D3D_FEATURE_LEVELS: [u32; 4] = [
    0xb100, // D3D_FEATURE_LEVEL_11_1
    0xb000, // D3D_FEATURE_LEVEL_11_0
    0xa100, // D3D_FEATURE_LEVEL_10_1
    0xa000, // D3D_FEATURE_LEVEL_10_0
];

/// `ID3D10Multithread`: `{9B7E4E00-342C-4106-A19F-4F2704F689F0}`.
const IID_ID3D10_MULTITHREAD: GUID = GUID::from_u128(0x9b7e4e00_342c_4106_a19f_4f2704f689f0);
const ID3D10_MULTITHREAD_SET_PROTECTED: usize = 5;

/// `ID3D11Device`: `{DB6F6DDB-AC77-4E88-8253-819DF9BBF140}`.
const IID_ID3D11_DEVICE: GUID = GUID::from_u128(0xdb6f6ddb_ac77_4e88_8253_819df9bbf140);
/// `ID3D11DeviceVtbl::GetDeviceRemovedReason` (`d3d11.h`: slots 37 = `GetFeatureLevel`, 38 = `GetCreationFlags`, 39 = `GetDeviceRemovedReason`).
const ID3D11_DEVICE_GET_REMOVED_REASON: usize = 39;

type D3D11CreateDeviceFn = unsafe extern "system" fn(
    pAdapter: *mut c_void,
    DriverType: i32,
    Software: *mut c_void,
    Flags: u32,
    pFeatureLevels: *const u32,
    FeatureLevels: u32,
    SDKVersion: u32,
    ppDevice: *mut *mut c_void,
    pFeatureLevel: *mut u32,
    ppImmediateContext: *mut *mut c_void,
) -> HRESULT;

type SetMultithreadProtectedFn = unsafe extern "system" fn(*mut c_void, i32) -> i32;

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::hle_exports![MFResetDXGIDeviceManagerX]
}

/// Enable `ID3D10Multithread::SetMultithreadProtected(TRUE)` on a D3D11 device as required
/// by `IMFDXGIDeviceManager::ResetDevice`.
unsafe fn enable_multithread_protection(d3d11_dev: &IUnknown) {
    let mut mt_raw: *mut c_void = std::ptr::null_mut();
    let hr = d3d11_dev.query(&IID_ID3D10_MULTITHREAD, &mut mt_raw);
    if hr.is_ok() && !mt_raw.is_null() {
        let mt = IUnknown::from_raw(mt_raw);
        let vtbl = *(mt.as_raw() as *const *const usize);
        let set_protected: SetMultithreadProtectedFn =
            std::mem::transmute(*vtbl.add(ID3D10_MULTITHREAD_SET_PROTECTED));
        set_protected(mt.as_raw(), 1);
    }
}

/// Find the DXGI adapter matching `luid` (if non-zero).
unsafe fn adapter_for_luid(luid: LUID) -> Option<IDXGIAdapter> {
    if luid.LowPart == 0 && luid.HighPart == 0 {
        return None;
    }
    let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
    factory.EnumAdapterByLuid::<IDXGIAdapter>(luid).ok()
}

/// Create a companion D3D11 device (with video + BGRA support and multithread protection)
/// on the given adapter LUID so Windows' `IMFDXGIDeviceManager` can drive hardware MFTs.
unsafe fn create_d3d11_bridge(luid: Option<LUID>) -> Result<IUnknown, String> {
    let d3d11 = crate::resolver::system_module("d3d11.dll")
        .ok_or_else(|| "d3d11.dll not available".to_string())?;
    let proc = crate::resolver::system_proc_in(
        d3d11,
        &crate::pe::ImportName::Name { hint: 0, name: "D3D11CreateDevice".into() },
    )
    .ok_or_else(|| "D3D11CreateDevice export missing".to_string())?;
    let create_device: D3D11CreateDeviceFn = std::mem::transmute(proc);

    let adapter = luid.and_then(|l| adapter_for_luid(l));

    let try_create = |adapter_ptr: *mut c_void, driver_type: i32, flags: u32| -> Result<IUnknown, HRESULT> {
        let mut dev: *mut c_void = std::ptr::null_mut();
        let mut ctx: *mut c_void = std::ptr::null_mut();
        let mut fl: u32 = 0;
        let hr = create_device(
            adapter_ptr,
            driver_type,
            std::ptr::null_mut(),
            flags,
            D3D_FEATURE_LEVELS.as_ptr(),
            D3D_FEATURE_LEVELS.len() as u32,
            D3D11_SDK_VERSION,
            &mut dev,
            &mut fl,
            &mut ctx,
        );
        if !ctx.is_null() {
            drop(IUnknown::from_raw(ctx));
        }
        if hr >= 0 && !dev.is_null() {
            let unk = IUnknown::from_raw(dev);
            enable_multithread_protection(&unk);
            Ok(unk)
        } else {
            if !dev.is_null() {
                drop(IUnknown::from_raw(dev));
            }
            Err(hr)
        }
    };

    let mut last_hr: HRESULT = 0;
    let flag_sets = [
        D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
    ];

    // 1. Prefer the exact adapter matching the D3D12 device's LUID.
    if let Some(ref a) = adapter {
        for flags in flag_sets {
            match try_create(a.as_raw(), D3D_DRIVER_TYPE_UNKNOWN, flags) {
                Ok(dev) => return Ok(dev),
                Err(hr) => last_hr = hr,
            }
        }
    }

    // 2. Fall back to default hardware adapter, then WARP.
    for driver_type in [D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP] {
        for flags in flag_sets {
            match try_create(std::ptr::null_mut(), driver_type, flags) {
                Ok(dev) => return Ok(dev),
                Err(hr) => last_hr = hr,
            }
        }
    }

    Err(format!("D3D11CreateDevice failed: {:#x}", last_hr as u32))
}

fn bridge_cache() -> &'static Mutex<HashMap<(u32, i32), usize>> {
    static CACHE: OnceLock<Mutex<HashMap<(u32, i32), usize>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

unsafe fn d3d11_device_alive(unk: &IUnknown) -> bool {
    let mut raw: *mut c_void = std::ptr::null_mut();
    if unk.query(&IID_ID3D11_DEVICE, &mut raw).is_err() || raw.is_null() {
        return false;
    }
    let dev = IUnknown::from_raw(raw);
    let vtbl = *(dev.as_raw() as *const *const usize);
    let get_removed: unsafe extern "system" fn(*mut c_void) -> HRESULT =
        std::mem::transmute(*vtbl.add(ID3D11_DEVICE_GET_REMOVED_REASON));
    get_removed(dev.as_raw()) >= 0
}

/// Return a cached multithread-protected companion `ID3D11Device` for `luid`, creating a fresh
/// one only on first use or if the cached device entered device-removed state.
unsafe fn d3d11_bridge_for_luid(luid: LUID) -> Result<IUnknown, String> {
    let key = (luid.LowPart, luid.HighPart);
    let cached = bridge_cache().lock().unwrap().get(&key).copied();
    if let Some(raw) = cached {
        let borrowed = ManuallyDrop::new(IUnknown::from_raw(raw as *mut c_void));
        if d3d11_device_alive(&borrowed) {
            return Ok((*borrowed).clone());
        }
        let mut map = bridge_cache().lock().unwrap();
        if map.get(&key) == Some(&raw) {
            map.remove(&key);
            drop(IUnknown::from_raw(raw as *mut c_void));
        }
    }
    let created = create_d3d11_bridge(Some(luid))?;
    let mut map = bridge_cache().lock().unwrap();
    if let Some(&existing) = map.get(&key) {
        let borrowed = ManuallyDrop::new(IUnknown::from_raw(existing as *mut c_void));
        if d3d11_device_alive(&borrowed) {
            return Ok((*borrowed).clone());
        }
        map.remove(&key);
        drop(ManuallyDrop::into_inner(borrowed));
    }
    map.insert(key, created.clone().into_raw() as usize);
    Ok(created)
}

/// Xbox GDK `MFResetDXGIDeviceManagerX(IMFDXGIDeviceManager*, IUnknown*, UINT32) -> HRESULT`.
pub unsafe extern "system" fn MFResetDXGIDeviceManagerX(
    pDeviceManager: *mut c_void,
    pDevice: *mut c_void,
    resetToken: u32,
) -> HRESULT {
    if pDeviceManager.is_null() || pDevice.is_null() {
        return E_POINTER;
    }

    // Borrow caller-owned COM pointers without taking ownership (no Release on drop).
    let manager = ManuallyDrop::new(IMFDXGIDeviceManager::from_raw(pDeviceManager));
    let device_unk = ManuallyDrop::new(IUnknown::from_raw(pDevice));

    // Xbox GDK titles pass an ID3D12Device (including XCL's null D3D12 backend, whose
    // QueryInterface answers every IID and would otherwise poison IMFDXGIDeviceManager if passed
    // directly). Check ID3D12Device first and bind a cached companion D3D11 device on the same
    // adapter LUID; genuine ID3D11Device callers do not answer ID3D12Device::IID and fall through.
    if let Ok(d3d12_dev) = device_unk.cast::<ID3D12Device>() {
        let luid = d3d12_dev.GetAdapterLuid();
        match d3d11_bridge_for_luid(luid) {
            Ok(d3d11_unk) => match manager.ResetDevice(&d3d11_unk, resetToken) {
                Ok(()) => {
                    crate::xlog_info!(
                        "mfplat: MFResetDXGIDeviceManagerX bridged ID3D12Device (LUID {:x}:{:x}) via companion D3D11 device",
                        luid.HighPart,
                        luid.LowPart
                    );
                    return 0;
                }
                Err(bridge_err) => {
                    let bridge_hr = bridge_err.code().0;
                    crate::xlog_warn!(
                        "mfplat: MFResetDXGIDeviceManagerX bridged ResetDevice failed: {:#x}",
                        bridge_hr as u32
                    );
                    return bridge_hr;
                }
            },
            Err(e) => {
                crate::xlog_warn!(
                    "mfplat: MFResetDXGIDeviceManagerX failed to create D3D11 bridge ({e}); trying direct ResetDevice"
                );
            }
        }
    }

    match manager.ResetDevice(&*device_unk, resetToken) {
        Ok(()) => {
            crate::xlog_debug!("mfplat: MFResetDXGIDeviceManagerX direct ResetDevice succeeded");
            0
        }
        Err(direct_err) => {
            let direct_hr = direct_err.code().0;
            crate::xlog_warn!(
                "mfplat: MFResetDXGIDeviceManagerX ResetDevice failed: {:#x}",
                direct_hr as u32
            );
            direct_hr
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pe::ImportName;
    use crate::resolver::Source;
    use crate::util::{E_INVALIDARG, S_OK};

    #[test]
    fn routing_and_system_fallback() {
        let name = |s: &str| ImportName::Name { hint: 0, name: s.into() };
        assert_eq!(
            crate::resolver::classify("mfplat.dll", &name("MFResetDXGIDeviceManagerX")),
            Source::Hle(crate::hle::MFPLAT.to_string())
        );
        assert_eq!(
            crate::resolver::classify("mfplat.dll", &name("MFStartup")),
            Source::System("mfplat.dll".to_string())
        );
        assert_eq!(
            crate::resolver::classify("mfplat.dll", &name("MFCreateDXGIDeviceManager")),
            Source::System("mfplat.dll".to_string())
        );
        assert_eq!(
            crate::resolver::classify("mfplat.dll", &name("MFCreateMediaType")),
            Source::System("mfplat.dll".to_string())
        );

        // LoadLibrary + GetProcAddress roundtrip for both HLE and system exports.
        let h = crate::resolver::load_library("mfplat.dll").expect("load_library(mfplat.dll)");
        let hle_fn = crate::resolver::get_proc_address(h, &name("MFResetDXGIDeviceManagerX"))
            .expect("GetProcAddress(MFResetDXGIDeviceManagerX)");
        assert_eq!(hle_fn, MFResetDXGIDeviceManagerX as usize);
        let sys_fn = crate::resolver::get_proc_address(h, &name("MFStartup"))
            .expect("GetProcAddress(MFStartup)");
        assert_ne!(sys_fn, 0);
    }

    #[test]
    fn null_pointers_return_e_pointer() {
        unsafe {
            assert_eq!(
                MFResetDXGIDeviceManagerX(std::ptr::null_mut(), std::ptr::null_mut(), 0),
                E_POINTER
            );
            assert_eq!(
                MFResetDXGIDeviceManagerX(0x1000 as *mut c_void, std::ptr::null_mut(), 1),
                E_POINTER
            );
            assert_eq!(
                MFResetDXGIDeviceManagerX(std::ptr::null_mut(), 0x1000 as *mut c_void, 1),
                E_POINTER
            );
        }
    }

    #[test]
    fn reset_dxgi_device_manager_with_d3d12_device_and_token_validation() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Media::MediaFoundation::{
            MFCreateDXGIDeviceManager, MFShutdown, MFStartup, MFSTARTUP_LITE, MF_VERSION,
        };

        unsafe {
            if MFStartup(MF_VERSION, MFSTARTUP_LITE).is_err() {
                return;
            }

            let mut reset_token: u32 = 0;
            let mut manager: Option<IMFDXGIDeviceManager> = None;
            let hr = MFCreateDXGIDeviceManager(&mut reset_token, &mut manager);
            assert!(hr.is_ok(), "MFCreateDXGIDeviceManager failed: {hr:?}");
            let manager = manager.expect("manager");

            let Ok(d3d12_dev) = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
                let _ = MFShutdown();
                return;
            };

            // Invalid reset token MUST be rejected.
            let bad_token = reset_token.wrapping_add(1);
            let hr_bad = MFResetDXGIDeviceManagerX(
                manager.as_raw(),
                d3d12_dev.as_raw(),
                bad_token,
            );
            assert_eq!(
                hr_bad, E_INVALIDARG,
                "invalid token should return E_INVALIDARG, got {:#x}",
                hr_bad as u32
            );

            // Valid reset token with ID3D12Device must succeed.
            let hr_ok = MFResetDXGIDeviceManagerX(
                manager.as_raw(),
                d3d12_dev.as_raw(),
                reset_token,
            );
            assert_eq!(hr_ok, S_OK, "MFResetDXGIDeviceManagerX failed: {:#x}", hr_ok as u32);

            // Verify the device manager now has a usable device handle.
            let handle = manager
                .OpenDeviceHandle()
                .expect("OpenDeviceHandle should succeed after MFResetDXGIDeviceManagerX");
            manager
                .TestDevice(handle)
                .expect("TestDevice should succeed on reset device");
            manager
                .CloseDeviceHandle(handle)
                .expect("CloseDeviceHandle should succeed");

            // Also verify the D3D11 bridge path works end-to-end on the same adapter LUID.
            let luid = d3d12_dev.GetAdapterLuid();
            let d3d11_unk = create_d3d11_bridge(Some(luid))
                .expect("create_d3d11_bridge should succeed");
            manager
                .ResetDevice(&d3d11_unk, reset_token)
                .expect("ResetDevice with bridged D3D11 device should succeed");
            let handle2 = manager
                .OpenDeviceHandle()
                .expect("OpenDeviceHandle should succeed on bridged D3D11 device");
            manager
                .TestDevice(handle2)
                .expect("TestDevice should succeed on bridged D3D11 device");
            manager
                .CloseDeviceHandle(handle2)
                .expect("CloseDeviceHandle should succeed");

            drop(manager);
            drop(d3d11_unk);
            drop(d3d12_dev);
            let _ = MFShutdown();
        }
    }

    #[test]
    fn null_d3d12_device_bridges_to_real_d3d11_and_reuses_companion_per_luid() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Media::MediaFoundation::{
            MFCreateDXGIDeviceManager, MFShutdown, MFStartup, MFSTARTUP_LITE, MF_VERSION,
        };

        unsafe {
            if MFStartup(MF_VERSION, MFSTARTUP_LITE).is_err() {
                return;
            }

            let mut null_dev: *mut c_void = std::ptr::null_mut();
            assert_eq!(
                crate::hle::d3d12x::null::D3D12CreateDevice(
                    std::ptr::null_mut(),
                    0xC100,
                    std::ptr::null(),
                    &mut null_dev,
                ),
                S_OK
            );

            let mut tok1: u32 = 0;
            let mut mgr1: Option<IMFDXGIDeviceManager> = None;
            MFCreateDXGIDeviceManager(&mut tok1, &mut mgr1).expect("MFCreateDXGIDeviceManager 1");
            let mgr1 = mgr1.unwrap();

            assert_eq!(MFResetDXGIDeviceManagerX(mgr1.as_raw(), null_dev, tok1), S_OK);

            let lock_identity = |mgr: &IMFDXGIDeviceManager| -> (*mut c_void, bool) {
                let h = mgr.OpenDeviceHandle().expect("OpenDeviceHandle");
                let mut locked: *mut c_void = std::ptr::null_mut();
                mgr.LockDevice(h, &IID_ID3D11_DEVICE, &mut locked, false).expect("LockDevice(IID_ID3D11Device)");
                let d11 = IUnknown::from_raw(locked);
                let answers_d3d12 = d11.cast::<ID3D12Device>().is_ok();
                let canonical = d11.cast::<IUnknown>().expect("QI(IUnknown)").as_raw();
                drop(d11);
                mgr.UnlockDevice(h, false).expect("UnlockDevice");
                mgr.CloseDeviceHandle(h).expect("CloseDeviceHandle");
                (canonical, answers_d3d12)
            };

            let (null_bridge_1, answers_d3d12) = lock_identity(&mgr1);
            assert_ne!(
                null_bridge_1, null_dev,
                "NullObj must not be stored directly inside IMFDXGIDeviceManager"
            );
            assert!(
                !answers_d3d12,
                "locked IMFDXGIDeviceManager device must be a real ID3D11Device, not a D3D12/NullObj stand-in"
            );

            let mut tok2: u32 = 0;
            let mut mgr2: Option<IMFDXGIDeviceManager> = None;
            MFCreateDXGIDeviceManager(&mut tok2, &mut mgr2).expect("MFCreateDXGIDeviceManager 2");
            let mgr2 = mgr2.unwrap();

            assert_eq!(MFResetDXGIDeviceManagerX(mgr2.as_raw(), null_dev, tok2), S_OK);
            let (null_bridge_2, _) = lock_identity(&mgr2);
            assert_eq!(null_bridge_1, null_bridge_2, "companion ID3D11Device must be reused for the same LUID");

            if let Ok(d3d12_dev) = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) {
                assert_eq!(MFResetDXGIDeviceManagerX(mgr1.as_raw(), d3d12_dev.as_raw(), tok1), S_OK);
                assert_eq!(MFResetDXGIDeviceManagerX(mgr2.as_raw(), d3d12_dev.as_raw(), tok2), S_OK);
                let (hw1, hw_is_d3d12) = lock_identity(&mgr1);
                let (hw2, _) = lock_identity(&mgr2);
                assert!(!hw_is_d3d12);
                assert_eq!(hw1, hw2, "hardware LUID companion ID3D11Device must be cached across resets");
            }

            drop(mgr2);
            drop(mgr1);
            let _ = MFShutdown();
        }
    }
}
