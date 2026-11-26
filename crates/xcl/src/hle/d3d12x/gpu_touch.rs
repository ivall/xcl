//! `XCL_GPU_TEXTURE_TOUCH=1`: which of the textures the GPU owns (render targets, depth buffers,
//! UAV textures: `vram::place`) the title's CPU reads or writes at their console address (#2663).
//!
//! On the PC those textures live in video memory, and nothing copies them to or from the title's
//! memory. On the console the CPU and the GPU share that memory, so a title that reads such a
//! texture on the CPU (an ocean's height map that the physics samples, say) reads stale bytes here.
//! This finds which textures that is, and from where in the title.
//!
//! The committed pages wholly inside such a texture's memory get `PAGE_GUARD` when it is placed.
//! The first access to each page raises a guard-page exception, which the OS clears; the handler
//! notes it and lets the access go on. A lazily committed page (`kernelx::lazy`) is noted when its
//! first access commits it. A background thread logs every texture's first access, with the
//! title's code address, then counts every 2 s. A kernel call that reads or writes a guarded page
//! fails (`STATUS_GUARD_PAGE_VIOLATION`), so this is a diagnostic only.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

use windows_sys::Win32::Foundation::NTSTATUS;
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD};

const STATUS_GUARD_PAGE_VIOLATION: NTSTATUS = 0x8000_0001_u32 as NTSTATUS;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
const PAGE: usize = 0x1000;

/// A texture the GPU owns, at its console address.
struct Texture {
    end: usize,
    /// width, height, console format, resource flags
    what: (u64, u32, u32, u32),
    touched: bool,
}

/// start → texture
static TEXTURES: RwLock<BTreeMap<usize, Texture>> = RwLock::new(BTreeMap::new());
/// the pages this diagnostic guarded and that were not touched since: their guard exception is
/// its own to handle, whether a texture still owns the page or a newer one replaced it
static ARMED: RwLock<BTreeSet<usize>> = RwLock::new(BTreeSet::new());
/// first accesses not logged yet: (texture start, access address, code address, write, lazy page)
static FIRST: Mutex<Vec<(usize, usize, usize, bool, bool)>> = Mutex::new(Vec::new());
static GUARDED_PAGES: AtomicU64 = AtomicU64::new(0);
static TOUCHED_PAGES: AtomicU64 = AtomicU64::new(0);
static WRITES: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_GPU_TEXTURE_TOUCH").is_ok_and(|v| v.trim() == "1"))
}

fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
        let _ = std::thread::Builder::new().name("xcl-gpu-touch".into()).spawn(report);
    });
}

/// A texture the GPU owns was placed at `va` (`len` bytes of console memory; `desc` its console
/// description): watch its pages.
pub unsafe fn placed(va: usize, len: usize, desc: &[u8; 0x38]) {
    if !enabled() || len == 0 {
        return;
    }
    install();
    let u = |o: usize| u32::from_le_bytes(desc[o..o + 4].try_into().unwrap());
    watch(va, len, (u64::from_le_bytes(desc[0x10..0x18].try_into().unwrap()), u(0x18), u(0x20), u(0x30)));
}

/// `placed` without the switch.
unsafe fn watch(va: usize, len: usize, what: (u64, u32, u32, u32)) {
    let end = va.saturating_add(len);
    {
        let mut t = TEXTURES.write().unwrap_or_else(|e| e.into_inner());
        // a newer texture at overlapping memory replaces the older ones there
        let old: Vec<usize> = t.range(..end).filter(|(&s, x)| x.end > va && s < end).map(|(&s, _)| s).collect();
        for s in old {
            t.remove(&s);
        }
        t.insert(va, Texture { end, what, touched: false });
    }
    GUARDED_PAGES.fetch_add(guard(va, end) as u64, Ordering::Relaxed);
}

/// `PAGE_GUARD` on the committed pages wholly inside `[a, b)`; returns how many.
unsafe fn guard(a: usize, b: usize) -> usize {
    let (mut p, end) = ((a + PAGE - 1) & !(PAGE - 1), b & !(PAGE - 1));
    let mut n = 0;
    while p < end {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.RegionSize == 0 {
            break;
        }
        let run_end = (m.BaseAddress as usize).saturating_add(m.RegionSize).min(end);
        // (a page guarded already is someone else's: left alone)
        if m.State == MEM_COMMIT && m.Protect & PAGE_GUARD == 0 && m.Protect & 0xEE != 0 {
            let pages: Vec<usize> = (p..run_end).step_by(PAGE).collect();
            // armed before the guard is set, so an access in between is handled
            ARMED.write().unwrap_or_else(|e| e.into_inner()).extend(pages.iter().copied());
            let mut old = 0;
            if VirtualProtect(p as *const c_void, run_end - p, m.Protect | PAGE_GUARD, &mut old) != 0 {
                n += pages.len();
            } else {
                let mut a = ARMED.write().unwrap_or_else(|e| e.into_inner());
                for q in pages {
                    a.remove(&q);
                }
            }
        }
        p = run_end;
    }
    n
}

/// The title's pages `[a, a + len)` were decommitted, released or mapped anew: their guards (if
/// any) went with them, so they are not this diagnostic's any more. Otherwise, after the page is
/// committed again and someone else guards it, the handler would take that guard's exception.
pub fn forget(a: usize, len: usize) {
    let mut armed = ARMED.write().unwrap_or_else(|e| e.into_inner());
    if armed.is_empty() {
        return;
    }
    let gone: Vec<usize> = armed.range(a & !(PAGE - 1)..a.saturating_add(len)).copied().collect();
    for p in gone {
        armed.remove(&p);
    }
}

/// `kernelx::VirtualFree` of `[a, a + size)`; a release frees the whole allocation at `a`.
pub fn freed(a: usize, size: usize, release: bool) {
    if ARMED.read().unwrap_or_else(|e| e.into_inner()).is_empty() {
        return;
    }
    let len = if release || size == 0 { unsafe { allocation_len(a) } } else { size };
    forget(a, len);
}

/// Bytes of the allocation starting at `a` (its regions of one allocation base).
unsafe fn allocation_len(a: usize) -> usize {
    let mut p = a;
    loop {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.RegionSize == 0 || m.AllocationBase as usize != a {
            return p - a;
        }
        p = (m.BaseAddress as usize).saturating_add(m.RegionSize);
    }
}

/// The texture whose memory holds `at` (its start).
fn texture_at(at: usize) -> Option<usize> {
    let t = TEXTURES.read().unwrap_or_else(|e| e.into_inner());
    t.range(..=at).next_back().filter(|(_, x)| at < x.end).map(|(&s, _)| s)
}

/// A first access to `at` (from code at `rip`) in a texture the GPU owns.
fn note(at: usize, rip: usize, write: bool, lazy: bool) -> bool {
    let Some(start) = texture_at(at) else { return false };
    TOUCHED_PAGES.fetch_add(1, Ordering::Relaxed);
    if write {
        WRITES.fetch_add(1, Ordering::Relaxed);
    }
    let first = {
        let mut t = TEXTURES.write().unwrap_or_else(|e| e.into_inner());
        t.get_mut(&start).is_some_and(|x| !std::mem::replace(&mut x.touched, true))
    };
    // (no logging here: the faulting thread may hold any lock; `report` logs it)
    if first {
        if let Ok(mut f) = FIRST.try_lock() {
            if f.len() < 4096 {
                f.push((start, at, rip, write, lazy));
            }
        }
    }
    true
}

/// `kernelx::lazy` committed the page at `at` on its first access (`access`: 0 read, 1 write).
pub fn lazy_commit(at: usize, access: usize, rip: usize) {
    if enabled() {
        note(at, rip, access == 1, true);
    }
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    if r.ExceptionCode != STATUS_GUARD_PAGE_VIOLATION || r.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // a page this diagnostic guarded (the OS cleared the guard: the access goes on as it is). A
    // page whose texture a newer one replaced is still handled here, and only not noted.
    let at = r.ExceptionInformation[1];
    if ARMED.write().unwrap_or_else(|e| e.into_inner()).remove(&(at & !(PAGE - 1))) {
        note(at, r.ExceptionAddress as usize, r.ExceptionInformation[0] == 1, false);
        EXCEPTION_CONTINUE_EXECUTION
    } else {
        EXCEPTION_CONTINUE_SEARCH
    }
}

/// Every texture's first access as it comes, and the counts every 2 s while they change.
fn report() {
    let mut last = (0, 0, 0, 0);
    loop {
        std::thread::sleep(std::time::Duration::from_secs(2));
        let first: Vec<_> = std::mem::take(&mut *FIRST.lock().unwrap_or_else(|e| e.into_inner()));
        for (start, at, rip, write, lazy) in first {
            let what = TEXTURES.read().unwrap_or_else(|e| e.into_inner()).get(&start).map(|x| (x.end - start, x.what));
            let Some((len, (w, h, format, flags))) = what else { continue };
            crate::xlog_info!(
                "GPU texture touch: the CPU {} the GPU's texture at {start:#x} ({w}x{h}, format {format}, flags {flags:#x}, {} KB) at +{:#x}{}, from {}",
                if write { "wrote" } else { "read" },
                len >> 10,
                at - start,
                if lazy { " (a lazy page)" } else { "" },
                crate::modules::describe_address(rip)
            );
        }
        let (textures, touched) = {
            let t = TEXTURES.read().unwrap_or_else(|e| e.into_inner());
            (t.len(), t.values().filter(|x| x.touched).count())
        };
        let now = (textures, touched, TOUCHED_PAGES.load(Ordering::Relaxed), WRITES.load(Ordering::Relaxed));
        if now != last {
            crate::xlog_info!(
                "GPU texture touch: {} textures the GPU owns ({} pages guarded), {} of them touched by the CPU: {} pages, {} by writes",
                now.0,
                GUARDED_PAGES.load(Ordering::Relaxed),
                now.1,
                now.2,
                now.3
            );
            last = now;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};

    /// The first read and the first write of a watched texture's pages are noted, with the access
    /// address; memory outside it is not watched, and the accesses go on with the right bytes.
    #[test]
    fn the_first_cpu_access_to_a_gpu_texture_is_noted() {
        unsafe {
            let len = 4 * PAGE;
            let base = VirtualAlloc(std::ptr::null(), len + 2 * PAGE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            let va = base + PAGE;
            std::ptr::write_bytes(base as *mut u8, 7, len + 2 * PAGE);
            install();
            {
                // (`placed` without the switch)
                let mut t = TEXTURES.write().unwrap();
                t.insert(va, Texture { end: va + len, what: (64, 64, 28, 4), touched: false });
            }
            assert_eq!(guard(va, va + len), 4, "the texture's four pages guarded");
            let before = TOUCHED_PAGES.load(Ordering::Relaxed);
            // outside the texture: no guard, nothing noted
            assert_eq!(std::ptr::read_volatile(base as *const u8), 7);
            assert_eq!(std::ptr::read_volatile((va + len) as *const u8), 7);
            // a read of page 1, a write of page 2: both go on, both noted
            assert_eq!(std::ptr::read_volatile((va + PAGE + 8) as *const u8), 7);
            std::ptr::write_volatile((va + 2 * PAGE) as *mut u8, 9);
            assert_eq!(std::ptr::read_volatile((va + 2 * PAGE) as *const u8), 9);
            // a second access of the same page: the guard is gone, not noted again
            assert_eq!(std::ptr::read_volatile((va + PAGE + 16) as *const u8), 7);
            assert_eq!(TOUCHED_PAGES.load(Ordering::Relaxed) - before, 2);
            let first: Vec<_> = FIRST.lock().unwrap().iter().filter(|e| e.0 == va).cloned().collect();
            assert_eq!(first.len(), 1, "one first access per texture");
            assert_eq!((first[0].1, first[0].3), (va + PAGE + 8, false), "the first access: the read, at its address");
            assert!(TEXTURES.read().unwrap()[&va].touched);
            // a newer texture over the middle page replaces this one: the pages of the old one
            // that stay guarded are this diagnostic's still, handled (not noted, not passed on)
            watch(va + PAGE, PAGE, (64, 16, 28, 4));
            let touched = TOUCHED_PAGES.load(Ordering::Relaxed);
            assert!(TEXTURES.read().unwrap().get(&va).is_none(), "the older texture is replaced");
            assert!(ARMED.read().unwrap().contains(&(va + 3 * PAGE)), "its untouched page is still armed");
            assert_eq!(std::ptr::read_volatile((va + 3 * PAGE) as *const u8), 7, "an access there goes on");
            assert!(!ARMED.read().unwrap().contains(&(va + 3 * PAGE)));
            assert_eq!(TOUCHED_PAGES.load(Ordering::Relaxed), touched, "a page no texture owns is not noted");
            // a page someone else guarded is left to them: not armed
            let other = base + len + PAGE;
            let mut old = 0;
            assert_ne!(VirtualProtect(other as *const c_void, PAGE, PAGE_READWRITE | PAGE_GUARD, &mut old), 0);
            assert_eq!(guard(other, other + PAGE), 0);
            assert!(!ARMED.read().unwrap().contains(&other));
            assert_ne!(VirtualProtect(other as *const c_void, PAGE, PAGE_READWRITE, &mut old), 0);
            TEXTURES.write().unwrap().remove(&(va + PAGE));
            TEXTURES.write().unwrap().insert(va, Texture { end: va + len, what: (64, 64, 28, 4), touched: true });
            // a lazy page's first access (`lazy_commit`) is noted the same way: a write here
            let writes = WRITES.load(Ordering::Relaxed);
            assert!(note(va + 3 * PAGE, 0, true, true));
            assert!(!note(va + len, 0, true, true), "outside the texture");
            assert_eq!(WRITES.load(Ordering::Relaxed) - writes, 1);
            TEXTURES.write().unwrap().remove(&va);
            FIRST.lock().unwrap().retain(|e| e.0 != va);
            ARMED.write().unwrap().retain(|&q| !(base..base + len + 2 * PAGE).contains(&q));
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A page the title decommits (or releases) through kernelx is not the diagnostic's any more:
    /// committed again and guarded by someone else, that guard's exception goes to its owner.
    #[test]
    fn a_decommitted_page_is_left_to_its_next_owner() {
        use std::sync::atomic::AtomicUsize;
        use windows_sys::Win32::System::Diagnostics::Debug::RemoveVectoredExceptionHandler;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        static PAGE_AT: AtomicUsize = AtomicUsize::new(0);
        static SEEN: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "system" fn later(info: *mut EXCEPTION_POINTERS) -> i32 {
            let r = &*(*info).ExceptionRecord;
            if r.ExceptionCode == STATUS_GUARD_PAGE_VIOLATION && r.NumberParameters >= 2 && r.ExceptionInformation[1] & !(PAGE - 1) == PAGE_AT.load(Ordering::SeqCst) {
                SEEN.fetch_add(1, Ordering::SeqCst);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            EXCEPTION_CONTINUE_SEARCH
        }
        unsafe {
            install();
            let base = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            assert_eq!(guard(base, base + 2 * PAGE), 2);
            // decommitted through kernelx: no longer armed
            assert_ne!(crate::hle::kernelx::VirtualFree(base as *mut c_void, PAGE, MEM_DECOMMIT), 0);
            assert!(!ARMED.read().unwrap().contains(&base), "a decommitted page stays armed");
            assert!(ARMED.read().unwrap().contains(&(base + PAGE)), "the page left committed is still armed");
            // committed again and guarded by another owner, with its handler after this one
            assert_eq!(VirtualAlloc(base as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE) as usize, base);
            let mut old = 0;
            assert_ne!(VirtualProtect(base as *const c_void, PAGE, PAGE_READWRITE | PAGE_GUARD, &mut old), 0);
            PAGE_AT.store(base, Ordering::SeqCst);
            let h = AddVectoredExceptionHandler(0, Some(later));
            assert_eq!(std::ptr::read_volatile(base as *const u8), 0);
            RemoveVectoredExceptionHandler(h);
            assert_eq!(SEEN.load(Ordering::SeqCst), 1, "the other owner's guard exception reached it");
            // a release through kernelx frees every armed page of the allocation
            assert_ne!(crate::hle::kernelx::VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
            assert!(!ARMED.read().unwrap().contains(&(base + PAGE)), "a released page stays armed");
        }
    }
}
