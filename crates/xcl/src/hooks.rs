//! Inline hooks on real Windows functions (via `retour`), used where calls into Windows
//! do not go through the game's import table: file APIs called by the CRT, exception
//! machinery, etc.

use std::sync::atomic::{AtomicUsize, Ordering};

use retour::RawDetour;

use crate::util::wide;

/// Detour `target` to `detour`; the trampoline (call-original) address is stored in `orig`.
/// The detour object is leaked on purpose – hooks live for the process.
pub unsafe fn hook_addr(name: &str, target: usize, detour: usize, orig: &AtomicUsize) -> Result<(), String> {
    let d = RawDetour::new(target as *const (), detour as *const ()).map_err(|e| format!("hook {name}: {e}"))?;
    orig.store(d.trampoline() as *const () as usize, Ordering::Release);
    d.enable().map_err(|e| format!("hook {name}: {e}"))?;
    Box::leak(Box::new(d));
    crate::xlog_debug!("hook {name} active");
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
    let cname = std::ffi::CString::new(func).unwrap();
    let target = GetProcAddress(h, cname.as_ptr() as *const u8).ok_or_else(|| format!("hook: {module}!{func} not found"))?;
    hook_addr(&format!("{module}!{func}"), target as usize, detour, orig)
}
