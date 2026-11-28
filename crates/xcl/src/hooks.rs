//! Inline hooks on real Windows functions (via `retour`), used where calls into Windows
//! do not go through the game's import table: file APIs called by the CRT, exception
//! machinery, etc.

use std::sync::atomic::{AtomicUsize, Ordering};

use retour::RawDetour;

use crate::util::wide;

/// Publish `trampoline` into `orig` before enabling the detour so a concurrent caller cannot
/// observe a zero slot once patched, and restore the prior value if enabling fails so `orig`
/// never retains a dangling trampoline pointer.
fn enable_detour<E>(orig: &AtomicUsize, trampoline: usize, enable: impl FnOnce() -> Result<(), E>) -> Result<(), E> {
    let prev = orig.swap(trampoline, Ordering::AcqRel);
    if let Err(e) = enable() {
        orig.store(prev, Ordering::Release);
        return Err(e);
    }
    Ok(())
}

/// Detour `module!func` to `detour`; the trampoline (call-original) address is stored
/// in `orig`. The detour object is leaked on purpose – hooks live for the process.
pub unsafe fn hook(module: &str, func: &str, detour: usize, orig: &AtomicUsize) -> Result<(), String> {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryW};
    let mut h = GetModuleHandleW(wide(module).as_ptr());
    if h.is_null() {
        h = LoadLibraryW(wide(module).as_ptr());
    }
    if h.is_null() {
        return Err(format!("hook {module}!{func}: module not found"));
    }
    let cname = std::ffi::CString::new(func).map_err(|_| format!("hook {module}!{func}: invalid function name"))?;
    let target = GetProcAddress(h, cname.as_ptr() as *const u8).ok_or_else(|| format!("hook: {module}!{func} not found"))?;
    let d = RawDetour::new(target as *const (), detour as *const ()).map_err(|e| format!("hook {module}!{func}: {e}"))?;
    enable_detour(orig, d.trampoline() as *const () as usize, || d.enable()).map_err(|e| format!("hook {module}!{func}: {e}"))?;
    Box::leak(Box::new(d));
    crate::xlog_debug!("hook {module}!{func} active");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enable_detour_restores_orig_on_failure_and_keeps_it_on_success() {
        let orig = AtomicUsize::new(0);
        let res = enable_detour(&orig, 0xDEAD_BEEF, || {
            assert_eq!(orig.load(Ordering::Acquire), 0xDEAD_BEEF, "trampoline must be visible during enable");
            Err("enable failed")
        });
        assert_eq!(res, Err("enable failed"));
        assert_eq!(orig.load(Ordering::Acquire), 0, "orig must not keep dangling trampoline on enable failure");

        let ok = enable_detour(&orig, 0xCAFE_BABE, || Ok::<(), &str>(()));
        assert_eq!(ok, Ok(()));
        assert_eq!(orig.load(Ordering::Acquire), 0xCAFE_BABE);
    }

    #[test]
    fn hook_rejects_interior_nul_without_panicking() {
        let orig = AtomicUsize::new(0);
        let err = unsafe { hook("kernel32.dll", "Create\0FileW", 0x1000, &orig) }.unwrap_err();
        assert!(err.contains("invalid function name"), "{err}");
        assert_eq!(orig.load(Ordering::Acquire), 0);
    }
}
