//! D3D12 command list stub safety and return code auditing.
//!
//! Unhandled or fallback command list methods must follow console expectations:
//! - Recording / action methods return `void` (0 in rax) and must never dereference
//!   null pointers (descriptors, barriers, view arrays, constant data, etc.).
//! - Lifecycle methods (`Close`, `Reset`) return `HRESULT` (`S_OK` on success;
//!   `E_INVALIDARG` if required pointers like `allocator` are null; `E_FAIL` on error).
//! - Unhandled optional / feature queries that return `HRESULT` (e.g. meta-commands,
//!   private data queries) must return `E_NOTIMPL` (`0x8000_4001`) or `E_NOINTERFACE`
//!   (`0x8000_4002`) rather than false `S_OK`, preventing titles from assuming
//!   unsupported features are operational.
//! - Out-pointer methods (`QueryInterface`, `GetDevice`, etc.) must validate that
//!   `ppv` is non-null before writing, returning `E_INVALIDARG` on null.

#![allow(non_snake_case)]

use std::ffi::c_void;
use windows_sys::core::GUID;

use crate::util::{E_FAIL, E_INVALIDARG, E_NOINTERFACE, E_NOTIMPL, E_POINTER, HRESULT, S_OK};

pub const DXGI_ERROR_NOT_FOUND: HRESULT = 0x887A_0002u32 as i32;

/// Categorization of command list methods by expected return code and parameter contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandListMethodKind {
    /// Action / recording method returning void (rax = 0).
    Void,
    /// Lifecycle method returning HRESULT (e.g. Close, Reset: S_OK on success).
    Lifecycle,
    /// Unhandled optional feature returning HRESULT (must return E_NOTIMPL).
    UnhandledHResult,
    /// Interface or device query writing to an out-pointer (*mut *mut c_void).
    OutPointer,
}

/// Query the expected return code for a given method name.
/// Unhandled recording methods return 0 (void).
/// Unhandled HRESULT methods return `E_NOTIMPL`.
/// Lifecycle methods return `S_OK`.
pub fn expected_return_code(method_name: &str) -> HRESULT {
    match method_name {
        "Close" | "Reset" | "ClearState" | "SetName" | "SetPrivateDataInterface" => S_OK,
        "QueryInterface" | "GetDevice" => S_OK,
        "GetPrivateData" => DXGI_ERROR_NOT_FOUND,
        "InitializeMetaCommand" | "ExecuteMetaCommand" => E_NOTIMPL,
        "OpenExistingSharedHandle" | "CreateSharedHandle" => E_NOTIMPL,
        _ => {
            if is_hresult_method(method_name) {
                E_NOTIMPL
            } else {
                0 // void recording methods
            }
        }
    }
}

/// Identifies whether a method on ID3D12GraphicsCommandList* returns an HRESULT.
pub fn is_hresult_method(method_name: &str) -> bool {
    matches!(
        method_name,
        "QueryInterface"
            | "GetPrivateData"
            | "SetPrivateData"
            | "SetPrivateDataInterface"
            | "SetName"
            | "GetDevice"
            | "Close"
            | "Reset"
            | "InitializeMetaCommand"
            | "ExecuteMetaCommand"
    )
}

/// Safely construct a slice from raw pointer and length, guarding against null.
#[inline]
pub unsafe fn safe_slice<'a, T>(ptr: *const T, len: usize) -> Option<&'a [T]> {
    if ptr.is_null() || len == 0 {
        None
    } else {
        Some(std::slice::from_raw_parts(ptr, len))
    }
}

/// Safely construct a mutable slice from raw pointer and length, guarding against null.
#[inline]
pub unsafe fn safe_slice_mut<'a, T>(ptr: *mut T, len: usize) -> Option<&'a mut [T]> {
    if ptr.is_null() || len == 0 {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(ptr, len))
    }
}

/// Safely dereference a pointer, returning None if null.
#[inline]
pub unsafe fn safe_deref<'a, T>(ptr: *const T) -> Option<&'a T> {
    if ptr.is_null() {
        None
    } else {
        Some(&*ptr)
    }
}

/// Safely dereference a mutable pointer, returning None if null.
#[inline]
pub unsafe fn safe_deref_mut<'a, T>(ptr: *mut T) -> Option<&'a mut T> {
    if ptr.is_null() {
        None
    } else {
        Some(&mut *ptr)
    }
}

/// Safely write to an out-pointer, returning E_INVALIDARG if ppv is null.
#[inline]
pub unsafe fn safe_out_ptr(ppv: *mut *mut c_void, val: *mut c_void) -> HRESULT {
    if ppv.is_null() {
        E_INVALIDARG
    } else {
        *ppv = val;
        S_OK
    }
}

// ---------------------------------------------------------------------------------------
// Audited command list stubs
// ---------------------------------------------------------------------------------------

/// Audited QueryInterface stub: rejects null `ppv` with `E_POINTER`.
pub unsafe extern "system" fn stub_query_interface(
    _this: *mut c_void,
    riid: *const GUID,
    ppv: *mut *mut c_void,
) -> HRESULT {
    if ppv.is_null() {
        return E_POINTER;
    }
    *ppv = std::ptr::null_mut();
    if riid.is_null() {
        return E_INVALIDARG;
    }
    E_NOINTERFACE
}

/// Audited GetPrivateData stub: rejects null pointers and returns not found.
pub unsafe extern "system" fn stub_get_private_data(
    _this: *mut c_void,
    guid: *const GUID,
    size: *mut u32,
    _data: *mut c_void,
) -> HRESULT {
    if guid.is_null() || size.is_null() {
        return E_INVALIDARG;
    }
    DXGI_ERROR_NOT_FOUND
}

/// Audited SetPrivateData stub: validates parameters without dereferencing null pointers.
pub unsafe extern "system" fn stub_set_private_data(
    _this: *mut c_void,
    guid: *const GUID,
    size: u32,
    data: *const c_void,
) -> HRESULT {
    if guid.is_null() || (size > 0 && data.is_null()) {
        return E_INVALIDARG;
    }
    S_OK
}

/// Audited SetPrivateDataInterface stub: validates guid parameter.
pub unsafe extern "system" fn stub_set_private_data_interface(
    _this: *mut c_void,
    guid: *const GUID,
    _iface: *const c_void,
) -> HRESULT {
    if guid.is_null() {
        return E_INVALIDARG;
    }
    S_OK
}

/// Audited SetName stub: always succeeds.
pub unsafe extern "system" fn stub_set_name(_this: *mut c_void, _name: *const u16) -> HRESULT {
    S_OK
}

/// Audited GetDevice stub: rejects null out-pointer with `E_INVALIDARG`.
pub unsafe extern "system" fn stub_get_device(
    _this: *mut c_void,
    _riid: *const GUID,
    pp_device: *mut *mut c_void,
) -> HRESULT {
    if pp_device.is_null() {
        return E_INVALIDARG;
    }
    *pp_device = std::ptr::null_mut();
    S_OK
}

/// Audited Close stub: returns `E_FAIL` if list pointer is null, otherwise `S_OK`.
pub unsafe extern "system" fn stub_close(this: *mut c_void) -> HRESULT {
    if this.is_null() {
        return E_FAIL;
    }
    S_OK
}

/// Audited Reset stub: returns `E_INVALIDARG` if allocator is null.
pub unsafe extern "system" fn stub_reset(
    this: *mut c_void,
    allocator: *mut c_void,
    _pso: *mut c_void,
) -> HRESULT {
    if this.is_null() || allocator.is_null() {
        return E_INVALIDARG;
    }
    S_OK
}

/// Audited InitializeMetaCommand stub: unhandled feature returns `E_NOTIMPL`.
pub unsafe extern "system" fn stub_initialize_meta_command(
    _this: *mut c_void,
    _meta: *mut c_void,
    _params: *const c_void,
    _size: usize,
) -> HRESULT {
    E_NOTIMPL
}

/// Audited ExecuteMetaCommand stub: unhandled feature returns `E_NOTIMPL`.
pub unsafe extern "system" fn stub_execute_meta_command(
    _this: *mut c_void,
    _meta: *mut c_void,
    _params: *const c_void,
    _size: usize,
) -> HRESULT {
    E_NOTIMPL
}

/// Generic unhandled HRESULT stub returning `E_NOTIMPL`.
pub unsafe extern "system" fn stub_unhandled_hresult(_this: *mut c_void) -> HRESULT {
    E_NOTIMPL
}

/// Safe stub for recording methods taking descriptor heaps: ignores null or empty arrays.
pub unsafe extern "system" fn stub_set_descriptor_heaps(
    _this: *mut c_void,
    n: u32,
    heaps: *const usize,
) {
    if n == 0 || heaps.is_null() {
        return;
    }
    // Safe no-op without dereference
}

/// Safe stub for recording methods taking barriers: ignores null or empty arrays.
pub unsafe extern "system" fn stub_resource_barrier(
    _this: *mut c_void,
    n: u32,
    barriers: *const c_void,
) {
    if n == 0 || barriers.is_null() {
        return;
    }
    // Safe no-op without dereference
}

/// Safe stub for root constants: ignores null data pointer or zero count.
pub unsafe extern "system" fn stub_set_root_constants(
    _this: *mut c_void,
    _index: u32,
    n: u32,
    data: *const u32,
    _offset: u32,
) {
    if n == 0 || data.is_null() {
        return;
    }
    // Safe no-op without dereference
}

/// Safe stub for ClearUnorderedAccessView: handles null values safely.
pub unsafe extern "system" fn stub_clear_uav(
    _this: *mut c_void,
    values: *const u32,
    _rect_count: u32,
    _rects: *const c_void,
) {
    let _vals = if let Some(slice) = safe_slice(values, 4) {
        [slice[0], slice[1], slice[2], slice[3]]
    } else {
        [0u32; 4]
    };
}

/// Safe stub for CopyTextureRegion: ignores null source or destination.
pub unsafe extern "system" fn stub_copy_texture_region(
    _this: *mut c_void,
    dst: *const c_void,
    _x: u32,
    _y: u32,
    _z: u32,
    src: *const c_void,
    _bx: *const c_void,
) {
    if dst.is_null() || src.is_null() {
        return;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_return_codes() {
        assert_eq!(expected_return_code("Close"), S_OK);
        assert_eq!(expected_return_code("Reset"), S_OK);
        assert_eq!(expected_return_code("InitializeMetaCommand"), E_NOTIMPL);
        assert_eq!(expected_return_code("ExecuteMetaCommand"), E_NOTIMPL);
        assert_eq!(expected_return_code("GetPrivateData"), DXGI_ERROR_NOT_FOUND);
        assert_eq!(expected_return_code("DrawInstanced"), 0);
        assert_eq!(expected_return_code("Dispatch"), 0);
        assert_eq!(expected_return_code("ResourceBarrier"), 0);
    }

    #[test]
    fn test_is_hresult_method() {
        assert!(is_hresult_method("Close"));
        assert!(is_hresult_method("Reset"));
        assert!(is_hresult_method("QueryInterface"));
        assert!(is_hresult_method("InitializeMetaCommand"));
        assert!(is_hresult_method("ExecuteMetaCommand"));
        assert!(!is_hresult_method("DrawInstanced"));
        assert!(!is_hresult_method("Dispatch"));
        assert!(!is_hresult_method("SetPipelineState"));
        assert!(!is_hresult_method("ResourceBarrier"));
    }

    #[test]
    fn test_safe_helpers_null_safety() {
        unsafe {
            assert!(safe_slice::<u32>(std::ptr::null(), 10).is_none());
            assert!(safe_slice_mut::<u32>(std::ptr::null_mut(), 10).is_none());
            assert!(safe_deref::<u32>(std::ptr::null()).is_none());
            assert!(safe_deref_mut::<u32>(std::ptr::null_mut()).is_none());

            let val: u32 = 42;
            assert_eq!(safe_deref(&val), Some(&42));
            assert_eq!(safe_slice(&val, 1), Some(&[42][..]));
            assert_eq!(safe_slice(&val, 0), None);

            assert_eq!(safe_out_ptr(std::ptr::null_mut(), 0x1234 as *mut c_void), E_INVALIDARG);
            let mut out: *mut c_void = std::ptr::null_mut();
            assert_eq!(safe_out_ptr(&mut out, 0x1234 as *mut c_void), S_OK);
            assert_eq!(out as usize, 0x1234);
        }
    }

    #[test]
    fn test_stub_qi_null_rejection() {
        unsafe {
            assert_eq!(stub_query_interface(std::ptr::null_mut(), std::ptr::null(), std::ptr::null_mut()), E_POINTER);
            let mut out = 0x1234 as *mut c_void;
            let guid = GUID { data1: 1, data2: 2, data3: 3, data4: [0; 8] };
            assert_eq!(stub_query_interface(std::ptr::null_mut(), &guid, &mut out), E_NOINTERFACE);
            assert!(out.is_null());
        }
    }

    #[test]
    fn test_stub_reset_null_allocator_rejection() {
        unsafe {
            assert_eq!(stub_reset(0x1000 as *mut c_void, std::ptr::null_mut(), std::ptr::null_mut()), E_INVALIDARG);
            assert_eq!(stub_reset(std::ptr::null_mut(), 0x2000 as *mut c_void, std::ptr::null_mut()), E_INVALIDARG);
            assert_eq!(stub_reset(0x1000 as *mut c_void, 0x2000 as *mut c_void, std::ptr::null_mut()), S_OK);
        }
    }

    #[test]
    fn test_stub_close_safety() {
        unsafe {
            assert_eq!(stub_close(std::ptr::null_mut()), E_FAIL);
            assert_eq!(stub_close(0x1000 as *mut c_void), S_OK);
        }
    }

    #[test]
    fn test_stub_get_device_null_rejection() {
        unsafe {
            assert_eq!(stub_get_device(std::ptr::null_mut(), std::ptr::null(), std::ptr::null_mut()), E_INVALIDARG);
            let mut dev = 0x5678 as *mut c_void;
            assert_eq!(stub_get_device(std::ptr::null_mut(), std::ptr::null(), &mut dev), S_OK);
            assert!(dev.is_null());
        }
    }

    #[test]
    fn test_stub_meta_commands_return_notimpl() {
        unsafe {
            assert_eq!(stub_initialize_meta_command(std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(), 0), E_NOTIMPL);
            assert_eq!(stub_execute_meta_command(std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(), 0), E_NOTIMPL);
            assert_eq!(stub_unhandled_hresult(std::ptr::null_mut()), E_NOTIMPL);
        }
    }

    #[test]
    fn test_recording_stubs_null_tolerated() {
        unsafe {
            stub_set_descriptor_heaps(std::ptr::null_mut(), 10, std::ptr::null());
            stub_set_descriptor_heaps(std::ptr::null_mut(), 0, std::ptr::null());
            stub_resource_barrier(std::ptr::null_mut(), 5, std::ptr::null());
            stub_resource_barrier(std::ptr::null_mut(), 0, std::ptr::null());
            stub_set_root_constants(std::ptr::null_mut(), 0, 4, std::ptr::null(), 0);
            stub_set_root_constants(std::ptr::null_mut(), 0, 0, std::ptr::null(), 0);
            stub_clear_uav(std::ptr::null_mut(), std::ptr::null(), 0, std::ptr::null());
            stub_copy_texture_region(std::ptr::null_mut(), std::ptr::null(), 0, 0, 0, std::ptr::null(), std::ptr::null());
        }
    }
}
