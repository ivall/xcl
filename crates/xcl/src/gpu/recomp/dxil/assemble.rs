//! A DXIL module as LLVM IR text (or bitcode) to a signed container, through the PC DXC's
//! libraries: `dxcompiler.dll`'s assembler builds the container from the module's `dx.*`
//! metadata, `dxil.dll`'s validator checks and signs it. Neither DXC's HLSL front end nor its
//! optimiser runs: a retail compute program's module (8.6 MB of IR, 1 MB of DXIL) takes 0.75 s,
//! where its HLSL takes 25 s through `dxc::compile`.
//!
//! What a module needs on top of what the validator checks (RX 7800 XT, Windows driver):
//! an `!llvm.ident`. Without one the container validates and signs, but
//! `CreateComputePipelineState` faults inside the AMD driver.

use std::ffi::c_void;

use windows::core::{Interface, GUID, HRESULT};
use windows::Win32::Graphics::Direct3D::Dxc::{
    DxcValidatorFlags_InPlaceEdit, IDxcAssembler, IDxcOperationResult, IDxcUtils, IDxcValidator, CLSID_DxcAssembler, CLSID_DxcUtils, CLSID_DxcValidator, DXC_CP_UTF8,
};

type CreateInstance = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> HRESULT;

unsafe fn blob_bytes<'a>(ptr: *const c_void, len: usize) -> Result<&'a [u8], &'static str> {
    if len == 0 {
        Ok(&[])
    } else if ptr.is_null() {
        Err("null DXC blob buffer")
    } else {
        Ok(std::slice::from_raw_parts(ptr.cast::<u8>(), len))
    }
}

fn blob_len(len: usize) -> Result<u32, String> {
    u32::try_from(len).map_err(|_| "assemble: IR larger than 4 GiB".to_string())
}

fn errors(r: &IDxcOperationResult, status: HRESULT) -> String {
    let fallback = || format!("{status:?}");
    let Ok(b) = (unsafe { r.GetErrorBuffer() }) else {
        return fallback();
    };
    match unsafe { blob_bytes(b.GetBufferPointer(), b.GetBufferSize()) } {
        Ok(bytes) => {
            let text = String::from_utf8_lossy(bytes);
            let trimmed = text.trim_end_matches('\0').trim();
            if trimmed.is_empty() { fallback() } else { trimmed.to_string() }
        }
        Err(e) => format!("{e} ({status:?})"),
    }
}

unsafe fn make<T: Interface>(create: CreateInstance, clsid: &GUID) -> Result<T, String> {
    let mut p = std::ptr::null_mut();
    create(clsid, &T::IID, &mut p).ok().map_err(|e| format!("DxcCreateInstance: {e}"))?;
    if p.is_null() {
        return Err("DxcCreateInstance: returned null interface".into());
    }
    Ok(T::from_raw(p))
}

/// The module `ir` as a validated, signed DXIL container; the assembler's or the validator's
/// messages as the error.
pub fn assemble(ir: &str) -> Result<Vec<u8>, String> {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
    let create = super::super::dxc::create_instance().ok_or("dxcompiler.dll not found")?;
    unsafe {
        let create = std::mem::transmute::<usize, CreateInstance>(create);
        // (`dxc` loaded dxil.dll by its full path, before dxcompiler.dll)
        let signer = GetModuleHandleW(crate::util::wide("dxil.dll").as_ptr());
        let sign = (!signer.is_null()).then(|| GetProcAddress(signer, b"DxcCreateInstance\0".as_ptr())).flatten().ok_or("assemble: no dxil.dll to sign with")?;
        let utils: IDxcUtils = make(create, &CLSID_DxcUtils)?;
        let blob = utils.CreateBlob(ir.as_ptr() as *const c_void, blob_len(ir.len())?, DXC_CP_UTF8).map_err(|e| format!("assemble: {e}"))?;
        let asm: IDxcAssembler = make(create, &CLSID_DxcAssembler)?;
        let r = asm.AssembleToContainer(&blob).map_err(|e| format!("assemble: {e}"))?;
        let status = r.GetStatus().map_err(|e| format!("assemble: {e}"))?;
        if status.is_err() {
            return Err(format!("assemble: {}", errors(&r, status)));
        }
        let container = r.GetResult().map_err(|e| format!("assemble: {e}"))?;
        let v: IDxcValidator = make(std::mem::transmute::<usize, CreateInstance>(sign as usize), &CLSID_DxcValidator)?;
        let r = v.Validate(&container, DxcValidatorFlags_InPlaceEdit).map_err(|e| format!("validate: {e}"))?;
        let status = r.GetStatus().map_err(|e| format!("validate: {e}"))?;
        if status.is_err() {
            return Err(format!("validate: {}", errors(&r, status)));
        }
        let bytes = blob_bytes(container.GetBufferPointer(), container.GetBufferSize()).map_err(|e| format!("assemble: {e}"))?;
        if bytes.is_empty() {
            return Err("assemble: empty DXIL container".into());
        }
        Ok(bytes.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::runner;

    /// A CS of the ABI root signature, written by hand: `mem[i] = 3 i + 1` through `u2`.
    const IR: &str = r#"target datalayout = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-ms-dx"
%dx.types.Handle = type { i8* }
%dx.types.ResBind = type { i32, i32, i32, i8 }
%dx.types.ResourceProperties = type { i32, i32 }
%struct.RWByteAddressBuffer = type { i32 }
define void @main() {
  %1 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 2, i32 2, i32 0, i8 1 }, i32 2, i1 false)
  %2 = call i32 @dx.op.threadId.i32(i32 93, i32 0)
  %3 = mul i32 %2, 3
  %4 = add i32 %3, 1
  %5 = shl i32 %2, 2
  %6 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %1, %dx.types.ResourceProperties { i32 4107, i32 0 })
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %6, i32 %5, i32 undef, i32 %4, i32 undef, i32 undef, i32 undef, i8 1, i32 4)
  ret void
}
declare i32 @dx.op.threadId.i32(i32, i32) #0
declare void @dx.op.rawBufferStore.i32(i32, %dx.types.Handle, i32, i32, i32, i32, i32, i32, i8, i32) #1
declare %dx.types.Handle @dx.op.annotateHandle(i32, %dx.types.Handle, %dx.types.ResourceProperties) #0
declare %dx.types.Handle @dx.op.createHandleFromBinding(i32, %dx.types.ResBind, i32, i1) #0
attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
!llvm.ident = !{!0}
!dx.version = !{!1}
!dx.valver = !{!2}
!dx.shaderModel = !{!3}
!dx.resources = !{!4}
!dx.entryPoints = !{!7}
!0 = !{!"xcl"}
!1 = !{i32 1, i32 6}
!2 = !{i32 1, i32 6}
!3 = !{!"cs", i32 6, i32 6}
!4 = !{null, !5, null, null}
!5 = !{!6}
!6 = !{i32 0, %struct.RWByteAddressBuffer* undef, !"", i32 0, i32 2, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!7 = !{void ()* @main, !"main", null, !4, !8}
!8 = !{i32 0, i64 16, i32 4, !9}
!9 = !{i32 64, i32 1, i32 1}
"#;

    /// The text assembles, signs and runs on the GPU; a module the validator rejects (a use
    /// before its definition) is an error, not a container.
    #[test]
    fn ir_text_assembles_signs_and_runs() {
        if super::super::super::dxc::location().is_err() {
            runner::skip_test("no DXC");
            return;
        }
        // a Windows SDK's `bin\<ver>\x64` has dxcompiler.dll without dxil.dll: nothing to sign with
        let dxil = match super::assemble(IR) {
            Err(e) if e.contains("no dxil.dll") => {
                runner::skip_test("no dxil.dll");
                return;
            }
            r => r.unwrap(),
        };
        assert_eq!(&dxil[..4], b"DXBC");
        let bad = super::assemble(&IR.replace("%3 = mul i32 %2, 3", "%3 = mul i32 %4, 3")).unwrap_err();
        assert!(bad.contains("dominate"), "{bad}");
        if runner::adapter().is_err() {
            runner::skip_test("no GPU");
            return;
        }
        let out = runner::dispatch(&dxil, [1, 1, 1], &[], &[0u32; 64]).unwrap();
        let want: Vec<u32> = (0..64).map(|i| 3 * i + 1).collect();
        assert_eq!(&out[..64], &want[..]);
    }

    #[test]
    fn blob_bytes_and_blob_len_validate_null_empty_and_bounds() {
        let sample = [1u8, 2, 3, 4];
        unsafe {
            assert_eq!(super::blob_bytes(std::ptr::null(), 0), Ok(&[][..]));
            assert_eq!(super::blob_bytes(sample.as_ptr().cast(), 0), Ok(&[][..]));
            assert!(super::blob_bytes(std::ptr::null(), 16).is_err());
            assert_eq!(super::blob_bytes(sample.as_ptr().cast(), sample.len()), Ok(&sample[..]));
        }
        assert_eq!(super::blob_len(0), Ok(0));
        assert_eq!(super::blob_len(u32::MAX as usize), Ok(u32::MAX));
        #[cfg(target_pointer_width = "64")]
        assert!(super::blob_len((u32::MAX as usize) + 1).is_err());
    }

    #[test]
    fn make_rejects_null_interface_and_propagates_hresult() {
        use windows::core::{IUnknown, GUID, HRESULT};
        use windows::Win32::Foundation::{E_FAIL, S_OK};
        unsafe extern "system" fn ok_null(_: *const GUID, _: *const GUID, out: *mut *mut std::ffi::c_void) -> HRESULT {
            *out = std::ptr::null_mut();
            S_OK
        }
        unsafe extern "system" fn fail_hr(_: *const GUID, _: *const GUID, _: *mut *mut std::ffi::c_void) -> HRESULT {
            E_FAIL
        }
        let err = unsafe { super::make::<IUnknown>(ok_null, &GUID::zeroed()) }.unwrap_err();
        assert!(err.contains("null interface"), "{err}");
        assert!(unsafe { super::make::<IUnknown>(fail_hr, &GUID::zeroed()) }.is_err());
    }

    #[test]
    fn errors_handles_null_empty_blob_trims_nuls_and_falls_back_to_hresult() {
        use std::ffi::c_void;
        use std::mem::{offset_of, size_of};
        use windows::core::{IUnknown_Vtbl, Interface, HRESULT};
        use windows::Win32::Foundation::{E_FAIL, S_OK};
        use windows::Win32::Graphics::Direct3D::Dxc::{IDxcBlobEncoding_Vtbl, IDxcBlob_Vtbl, IDxcOperationResult, IDxcOperationResult_Vtbl};
        #[repr(C)]
        struct MockBlob { vtbl: *const usize, ptr: *const c_void, len: usize }
        #[repr(C)]
        struct MockOp { vtbl: *const usize, blob: *mut c_void, hr: HRESULT }
        extern "system" fn unexpected() { std::process::abort(); }
        unsafe extern "system" fn release(_: *mut c_void) -> u32 { 0 }
        unsafe extern "system" fn buf_ptr(this: *mut c_void) -> *mut c_void { (*this.cast::<MockBlob>()).ptr as *mut c_void }
        unsafe extern "system" fn buf_len(this: *mut c_void) -> usize { (*this.cast::<MockBlob>()).len }
        unsafe extern "system" fn err_buf(this: *mut c_void, out: *mut *mut c_void) -> HRESULT {
            let op = &*this.cast::<MockOp>();
            if op.hr.is_err() { return op.hr; }
            *out = op.blob;
            S_OK
        }
        const SLOT: usize = size_of::<usize>();
        let mut bvtbl = [unexpected as usize; size_of::<IDxcBlobEncoding_Vtbl>() / SLOT];
        bvtbl[offset_of!(IUnknown_Vtbl, Release) / SLOT] = release as usize;
        bvtbl[offset_of!(IDxcBlob_Vtbl, GetBufferPointer) / SLOT] = buf_ptr as usize;
        bvtbl[offset_of!(IDxcBlob_Vtbl, GetBufferSize) / SLOT] = buf_len as usize;
        let mut ovtbl = [unexpected as usize; size_of::<IDxcOperationResult_Vtbl>() / SLOT];
        ovtbl[offset_of!(IDxcOperationResult_Vtbl, GetErrorBuffer) / SLOT] = err_buf as usize;
        let run = |ptr: *const c_void, len: usize, hr: HRESULT| -> String {
            let mut blob = MockBlob { vtbl: bvtbl.as_ptr(), ptr, len };
            let mut op = MockOp { vtbl: ovtbl.as_ptr(), blob: std::ptr::from_mut(&mut blob).cast(), hr };
            let raw = std::ptr::from_mut(&mut op).cast();
            unsafe { super::errors(IDxcOperationResult::from_raw_borrowed(&raw).unwrap(), E_FAIL) }
        };
        let hr_str = format!("{E_FAIL:?}");
        assert_eq!(run(std::ptr::null(), 0, S_OK), hr_str);
        assert_eq!(run(b"  \0\0".as_ptr().cast(), 4, S_OK), hr_str);
        assert_eq!(run(b"bad ir\n\0\0".as_ptr().cast(), 9, S_OK), "bad ir");
        let corrupt = run(std::ptr::null(), 16, S_OK);
        assert!(corrupt.contains("null DXC blob buffer") && corrupt.contains(&hr_str), "{corrupt}");
        assert_eq!(run(std::ptr::null(), 0, E_FAIL), hr_str);
    }
}
