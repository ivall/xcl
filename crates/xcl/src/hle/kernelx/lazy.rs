//! Physical pages mapped into the title's own reservations, committed on first use.
//!
//! A title maps its physical pages wherever it reserved address space and places resources
//! there – render targets and depth buffers too, which on the PC live in video memory
//! (`d3d12x::vram`); the CPU never touches those pages. Committed at map time they doubled
//! that memory (a retail game: 5.4 GB of mapped pages next to its video-memory targets, past
//! the system's commit). So a page whose contents are still zero is mapped as reserved address
//! space only: the CPU's first access faults, the page is committed (zeros, as the page holds)
//! and the access goes on. Kernel calls do not fault in the title – they fail – so the buffers
//! of the file calls XCL routes, and DirectStorage's destinations, are committed first
//! (`ensure`).

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Once, RwLock};

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
};

const PAGE: usize = 0x1_0000;
/// 4 MiB stripe per shard so contiguous ranges hit 1 shard while 16 threads faulting across
/// different regions commit in parallel without serializing on a single global RwLock.
const SHARDS: usize = 16;
const STRIPE_SHIFT: u32 = 22;
#[cfg(test)]
const STRIPE_SIZE: usize = 1 << STRIPE_SHIFT;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// pages (64 KiB) mapped but not committed yet, sharded by 4 MiB VA stripe
static LAZY: [RwLock<BTreeSet<usize>>; SHARDS] = [const { RwLock::new(BTreeSet::new()) }; SHARDS];
static LAZY_COUNT: AtomicUsize = AtomicUsize::new(0);
static LAZY_COMMITTED_ANY: AtomicBool = AtomicBool::new(false);

#[inline]
fn shard_index(page: usize) -> usize {
    (page >> STRIPE_SHIFT) & (SHARDS - 1)
}

fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

/// Commit `page` if it is a lazy one (true: it was).
unsafe fn commit(page: usize) -> bool {
    let mut g = LAZY[shard_index(page)].write().unwrap_or_else(|e| e.into_inner());
    if !g.remove(&page) {
        return false;
    }
    LAZY_COUNT.fetch_sub(1, Ordering::Release);
    LAZY_COMMITTED_ANY.store(true, Ordering::Release);
    !VirtualAlloc(page as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null()
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let count = LAZY_COUNT.load(Ordering::Acquire);
    let committed_any = LAZY_COMMITTED_ANY.load(Ordering::Acquire);
    if count == 0 && !committed_any {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let page = r.ExceptionInformation[1] & !(PAGE - 1);
    let lazy = count != 0 && LAZY[shard_index(page)].read().unwrap_or_else(|e| e.into_inner()).contains(&page);
    if lazy && commit(page) || (committed_any && allows(r.ExceptionInformation[0], r.ExceptionInformation[1])) {
        EXCEPTION_CONTINUE_EXECUTION
    } else {
        EXCEPTION_CONTINUE_SEARCH
    }
}

unsafe fn allows(access: usize, at: usize) -> bool {
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT || m.Protect & PAGE_GUARD != 0 {
        return false;
    }
    let writes = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    let reads = writes | PAGE_READONLY | PAGE_EXECUTE_READ;
    match access {
        0 => m.Protect & reads != 0,
        1 => m.Protect & writes != 0,
        _ => false,
    }
}

/// `page` (reserved, 64 KiB aligned) is mapped: committed on first use.
pub(crate) fn mark(page: usize) {
    install();
    if LAZY[shard_index(page)].write().unwrap_or_else(|e| e.into_inner()).insert(page) {
        LAZY_COUNT.fetch_add(1, Ordering::Release);
    }
}

/// Bulk-mark `pages` (each 64 KiB aligned) as lazy, holding each touched shard lock once per run.
pub(super) fn mark_batch(pages: &[usize]) {
    if pages.is_empty() {
        return;
    }
    install();
    let mut added = 0usize;
    let mut i = 0;
    while i < pages.len() {
        let s = shard_index(pages[i]);
        let mut g = LAZY[s].write().unwrap_or_else(|e| e.into_inner());
        while i < pages.len() && shard_index(pages[i]) == s {
            if g.insert(pages[i]) {
                added += 1;
            }
            i += 1;
        }
    }
    if added != 0 {
        LAZY_COUNT.fetch_add(added, Ordering::Release);
    }
}

/// `page` is unmapped: true if it was never used (nothing to keep of it).
pub(super) fn unmark(page: usize) -> bool {
    if LAZY_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let removed = LAZY[shard_index(page)].write().unwrap_or_else(|e| e.into_inner()).remove(&page);
    if removed {
        LAZY_COUNT.fetch_sub(1, Ordering::Release);
    }
    removed
}

/// Bulk-unmark `pages` (each 64 KiB aligned), holding each touched shard lock once per run.
pub(super) fn unmark_batch(pages: &[usize]) {
    if pages.is_empty() || LAZY_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut removed = 0usize;
    let mut i = 0;
    while i < pages.len() {
        let s = shard_index(pages[i]);
        let mut g = LAZY[s].write().unwrap_or_else(|e| e.into_inner());
        while i < pages.len() && shard_index(pages[i]) == s {
            if g.remove(&pages[i]) {
                removed += 1;
            }
            i += 1;
        }
    }
    if removed != 0 {
        LAZY_COUNT.fetch_sub(removed, Ordering::Release);
    }
}

/// Whether `page` (64 KiB aligned) is currently marked lazy (reserved, never touched, holds zeros).
pub(super) fn is_lazy(page: usize) -> bool {
    LAZY_COUNT.load(Ordering::Acquire) != 0 && LAZY[shard_index(page)].read().unwrap_or_else(|e| e.into_inner()).contains(&page)
}

/// The lazy pages of `[p, p + len)` committed – memory a kernel call or XCL writes.
pub fn ensure(p: usize, len: usize) {
    if len == 0 || LAZY_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let start = p & !(PAGE - 1);
    let end = p.saturating_add(len);
    let mut pages = Vec::new();
    let first_stripe = start >> STRIPE_SHIFT;
    let last_stripe = end.saturating_sub(1) >> STRIPE_SHIFT;
    let span = last_stripe.saturating_sub(first_stripe);
    if span >= SHARDS - 1 {
        for shard in &LAZY {
            let g = shard.read().unwrap_or_else(|e| e.into_inner());
            if !g.is_empty() {
                pages.extend(g.range(start..end).copied());
            }
        }
    } else {
        for stripe in first_stripe..=last_stripe {
            let g = LAZY[stripe & (SHARDS - 1)].read().unwrap_or_else(|e| e.into_inner());
            if !g.is_empty() {
                pages.extend(g.range(start..end).copied());
            }
        }
    }
    for page in pages {
        unsafe { commit(page) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE, MEM_RESERVE};

    /// A lazy page reads as zeros on first touch, keeps what is written; `ensure` commits it
    /// without a fault.
    #[test]
    fn a_lazy_page_is_committed_on_first_use() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            mark(r);
            mark(r + PAGE);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 0);
            std::ptr::write_volatile((r + 8) as *mut u32, 7);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 7);
            assert!(!unmark(r), "used: no longer lazy");
            ensure(r + PAGE + 4, 4);
            assert!(!is_lazy(r + PAGE));
            assert!(crate::gpu::shader::readable(r + PAGE, 4));
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn sharded_lazy_batch_spans_stripes_and_unmarks_cleanly() {
        unsafe {
            // Span 2 full 4 MiB stripes + 1 page so multiple shards are exercised.
            let n = (STRIPE_SIZE / PAGE) * 2 + 1;
            let r = VirtualAlloc(std::ptr::null(), n * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(r, 0);
            let pages: Vec<usize> = (0..n).map(|i| r + i * PAGE).collect();
            mark_batch(&pages);
            assert!(is_lazy(pages[0]) && is_lazy(pages[n / 2]) && is_lazy(pages[n - 1]));
            // Ensure only the final page across the stripe boundary.
            ensure(pages[n - 1], 16);
            assert!(!is_lazy(pages[n - 1]));
            assert!(is_lazy(pages[0]));
            unmark_batch(&pages);
            assert!(!is_lazy(pages[0]) && !is_lazy(pages[n / 2]));
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn threads_first_touching_one_lazy_page_together_all_go_on() {
        const THREADS: usize = 8;
        const PAGES: usize = 256;
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            for i in 0..PAGES {
                mark(r + i * PAGE);
            }
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
            let handles: Vec<_> = (0..THREADS)
                .map(|t| {
                    let barrier = barrier.clone();
                    std::thread::spawn(move || {
                        for i in 0..PAGES {
                            barrier.wait();
                            std::ptr::write_volatile((r + i * PAGE + t * 64) as *mut u64, (i * THREADS + t) as u64);
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            for i in 0..PAGES {
                for t in 0..THREADS {
                    assert_eq!(std::ptr::read_volatile((r + i * PAGE + t * 64) as *const u64), (i * THREADS + t) as u64);
                }
            }
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
