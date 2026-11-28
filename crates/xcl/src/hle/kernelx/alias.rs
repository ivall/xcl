//! Physical pages that show at every address they are mapped at (`XCL_PAGE_ALIAS=1`).
//!
//! Without it a physical page is kept at one address at a time and moved (copied) when the
//! title maps it elsewhere. With it a block of physical pages is a section (`sections`): the
//! title's reservations of 64 KiB pages are placeholders ("spaces"), and mapping a page there
//! maps a view of its slice, so every address the title mapped it at shows the same memory.
//! The views are guarded (`guards`): a write through one is uploaded at all of them.
//!
//! Each block also has a view of its own (its home, never given to the title) through which
//! XCL reads and writes pages by number (DirectStorage destinations). A retail title never maps
//! a page at its home, so homes are plain address space, not the GPU-visible sparse windows.
//!
//! A page mapped at an address outside a space (a placeholder region the title made in a
//! sparse window) is moved there as before: its bytes are copied there and back again when it
//! is unmapped. While it is away its views show what it held when it left.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows_sys::Win32::Foundation as fnd;
use windows_sys::Win32::System::Memory as mem;

use super::sections::{self, Section, PAGE};
use super::{guards, lazy, page_index, page_number, NEXT_PAGE_INDEX};

/// `XCL_PAGE_ALIAS=1` turns physical page aliasing on (default: off).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_PAGE_ALIAS").is_ok_and(|v| v.trim() == "1"))
}

struct Block {
    section: Arc<Section>,
    /// XCL's own view of the whole block
    home: usize,
    /// pages not freed yet
    live: usize,
    /// Individual IDs may be freed before their backing block and surviving views.
    freed: Vec<u64>,
    /// one bit a page: committed (asking Windows walked the rest of a block's uncommitted run –
    /// a retail title mapping 100 000 fresh pages one at a time waited 20 s for its first frame)
    committed: Arc<Vec<AtomicU64>>,
}

impl Block {
    fn has_page(&self, index: usize) -> bool {
        index < self.section.pages() && self.freed[index / 64] & (1 << (index % 64)) == 0
    }
}

#[derive(Default)]
struct State {
    /// first page index → block
    blocks: BTreeMap<usize, Block>,
    /// the title's reservations kept as placeholders: base → size
    spaces: BTreeMap<usize, usize>,
    /// address (64 KiB) in a space → (page number, its section, the page's index in it)
    views: BTreeMap<usize, (usize, Arc<Section>, usize)>,
    /// pages moved to an address outside the spaces: page → address, and address → page
    away: HashMap<usize, usize>,
    away_at: BTreeMap<usize, usize>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
// Serialize map/copy-back with page and space release. Fault handlers use STATE while a
// mapping is open, so STATE itself cannot be held across the native operations.
static MAPPINGS: Mutex<()> = Mutex::new(());

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(State::default))
}

impl State {
    /// (block's section, the page's index in it, its address in the home view) of page number `page`
    fn page(&self, page: usize) -> Option<(Arc<Section>, usize, usize)> {
        let index = page_index(page)?;
        let (&first, b) = self.blocks.range(..=index).next_back()?;
        let i = index - first;
        b.has_page(i).then(|| (b.section.clone(), i, b.home + i * PAGE))
    }

    /// The commit bits of page number `page`'s block and its index there.
    fn bits(&self, page: usize) -> Option<(Arc<Vec<AtomicU64>>, usize)> {
        let index = page_index(page)?;
        let (&first, b) = self.blocks.range(..=index).next_back()?;
        b.has_page(index - first).then(|| (b.committed.clone(), index - first))
    }

    fn in_space(&self, at: usize) -> bool {
        self.spaces.range(..=at).next_back().is_some_and(|(&b, &n)| at < b + n)
    }
}

/// Whether page number `page` is committed (a page never used is not).
fn committed(page: usize) -> bool {
    with(|s| s.bits(page)).is_some_and(|(b, i)| b[i / 64].load(Ordering::Acquire) & (1 << (i % 64)) != 0)
}

/// Page number `page` is committed now.
fn note_committed(page: usize) {
    if let Some((b, i)) = with(|s| s.bits(page)) {
        b[i / 64].fetch_or(1 << (i % 64), Ordering::AcqRel);
    }
}

/// The view at `view` was committed by the lazy handler (its page's first use).
pub fn committed_at(view: usize) {
    if let Some(page) = with(|s| s.views.get(&(view & !(PAGE - 1))).map(|v| v.0)) {
        note_committed(page);
    }
}

/// A reservation of 64 KiB pages the title will map physical pages into: a space. None if
/// Windows refuses the address space.
pub fn reserve_space(size: usize) -> Option<usize> {
    let len = size.checked_next_multiple_of(PAGE)?;
    // SAFETY: only address space is reserved; `free_space` releases it.
    let at = unsafe { sections::reserve(None, len) }?;
    with(|s| s.spaces.insert(at, len));
    Some(at)
}

/// Whether `p` lies in a space.
pub fn in_space(p: usize) -> bool {
    enabled() && with(|s| s.in_space(p))
}

/// `VirtualFree` of an address in a space: release (the whole space) unmaps its pages and gives
/// the address space back; decommit unmaps the pages of the range. None: not a space.
pub unsafe fn free_space(addr: usize, size: usize, t: u32) -> Option<bool> {
    // VirtualFree also reaches this for addresses outside alias spaces.
    if !with(|s| s.in_space(addr)) {
        return None;
    }
    let _mapping = MAPPINGS.lock().unwrap_or_else(|e| e.into_inner());
    let (base, len) = with(|s| s.spaces.range(..=addr).next_back().filter(|(&b, &n)| addr < b + n).map(|(&b, &n)| (b, n)))?;
    if t & mem::MEM_RELEASE != 0 {
        if addr != base || size != 0 {
            fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
            return Some(false);
        }
        unmap_range(base, len);
        with(|s| s.spaces.remove(&base));
        sections::release_all(base, len);
        return Some(true);
    }
    let end = if size == 0 { base + len } else { addr.saturating_add(size).min(base + len) };
    unmap_range(addr & !(PAGE - 1), end - (addr & !(PAGE - 1)));
    Some(true)
}

unsafe fn unmap_range(at: usize, len: usize) {
    let views: Vec<usize> = with(|s| s.views.range(at..at + len).map(|(&a, _)| a).collect());
    for a in views {
        unmap_view(a);
    }
}

/// The view at `at` (if any) unmapped, leaving a placeholder.
unsafe fn unmap_view(at: usize) {
    if with(|s| s.views.remove(&at)).is_some() {
        guards::forget(at);
        lazy::unmark(at);
        sections::unmap(at);
    }
}

/// `count` physical pages for the title; their numbers in `arr`.
pub unsafe fn allocate(n: usize, arr: *mut usize) -> bool {
    let Some(section) = Section::new(n) else {
        crate::xlog_warn!("physical pages: a section of {n} pages could not be made");
        return false;
    };
    let Some(home) = sections::reserve(None, n * PAGE) else { return false };
    if !section.map(0, n, home) {
        sections::release(home);
        return false;
    }
    let first = NEXT_PAGE_INDEX.fetch_add(n, Ordering::Relaxed);
    for i in 0..n {
        *arr.add(i) = page_number(first + i);
    }
    let committed = Arc::new((0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect());
    let freed = vec![0; n.div_ceil(64)];
    with(|s| s.blocks.insert(first, Block { section: Arc::new(section), home, live: n, freed, committed }));
    true
}

/// The title frees `pages`. Addresses that show one keep showing it (the title is meant to unmap
/// first); a block whose pages are all freed loses its home.
pub unsafe fn free(pages: &[usize]) {
    let _mapping = MAPPINGS.lock().unwrap_or_else(|e| e.into_inner());
    let mut homes = Vec::new();
    with(|s| {
        for &page in pages {
            if let Some(at) = s.away.remove(&page) {
                s.away_at.remove(&at);
            }
            let Some(index) = page_index(page) else { continue };
            let Some((&first, b)) = s.blocks.range_mut(..=index).next_back() else { continue };
            let i = index - first;
            if !b.has_page(i) {
                continue;
            }
            b.freed[i / 64] |= 1 << (i % 64);
            b.live -= 1;
            if b.live == 0 {
                homes.push(b.home);
                s.blocks.remove(&first);
            }
        }
    });
    for home in homes {
        sections::unmap(home);
        sections::release(home);
    }
}

/// Map `n` pages (`pages[i]`, 0: unmap) at `va`. Pages at an address in a space are views of
/// them; elsewhere they are moved there. False with the last error set if a page is not one of
/// the title's or Windows refuses.
pub unsafe fn map(va: usize, pages: &[usize]) -> bool {
    if !va.is_multiple_of(PAGE) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return false;
    }
    let mapping = MAPPINGS.lock().unwrap_or_else(|e| e.into_inner());
    // Reject invalid IDs in a run before changing any of its views. A single page is
    // validated by map_one.
    if pages.len() > 1 && with(|s| pages.iter().any(|&p| p != 0 && s.page(p).is_none())) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return false;
    }
    for (i, &page) in pages.iter().enumerate() {
        let at = va + i * PAGE;
        if let Err(err) = map_one(at, page, &mapping) {
            fnd::SetLastError(err);
            return false;
        }
    }
    true
}

/// Resolve the source before touching the destination. The mapping lock keeps the raw home
/// address valid through any native mapping, move, or copy-back below.
unsafe fn map_one(at: usize, page: usize, _mapping: &std::sync::MutexGuard<'_, ()>) -> Result<(), u32> {
    let source = if page == 0 {
        None
    } else {
        Some(with(|s| s.page(page)).ok_or(fnd::ERROR_INVALID_PARAMETER)?)
    };
    unmap_view(at);
    unmap_away(at);
    let Some((section, index, home)) = source else { return Ok(()) };
    // a page moved elsewhere comes home first (it is at one address outside the spaces at most,
    // as before): its views then show what it holds
    if let Some(old) = with(|s| s.away.get(&page).copied()) {
        unmap_away(old);
    }
    if with(|s| s.in_space(at)) {
        if !section.map(index, 1, at) {
            return Err(fnd::ERROR_INVALID_ADDRESS);
        }
        with(|s| s.views.insert(at, (page, section, index)));
        guards::watch(page, at);
        if !committed(page) {
            // a page never used: committed at its first touch here (or through its home)
            lazy::mark(at);
        }
        return Ok(());
    }
    move_to(at, page, home).then_some(()).ok_or(fnd::ERROR_INVALID_ADDRESS)
}

/// Page `page` moved to `at`, an address outside the spaces: committed there with its bytes.
unsafe fn move_to(at: usize, page: usize, home: usize) -> bool {
    let u = crate::gpu::uma::get();
    let ok = if u.is_some_and(|u| u.is_committed(at)) {
        true
    } else if u.is_some_and(|u| u.is_sparse(at)) {
        !super::commit_sparse(at, PAGE, mem::PAGE_READWRITE).is_null()
    } else {
        !mem::VirtualAlloc(at as *const c_void, PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null()
    };
    if !ok {
        return false;
    }
    if committed(page) {
        std::ptr::copy_nonoverlapping(home as *const u8, at as *mut u8, PAGE);
    } else {
        std::ptr::write_bytes(at as *mut u8, 0, PAGE);
    }
    with(|s| {
        s.away.insert(page, at);
        s.away_at.insert(at, page);
    });
    true
}

/// The page moved to `at` (if any) goes home with what `at` holds; `at` is decommitted.
unsafe fn unmap_away(at: usize) {
    let Some((page, home)) = with(|s| {
        let page = s.away_at.remove(&at)?;
        s.away.remove(&page);
        Some((page, s.page(page)?.2))
    }) else {
        return;
    };
    if sections::commit(home, PAGE) {
        note_committed(page);
        std::ptr::copy_nonoverlapping(at as *const u8, home as *mut u8, PAGE);
        guards::written(page);
    }
    let u = crate::gpu::uma::get();
    if u.is_some_and(|u| u.is_committed(at)) {
        return;
    }
    if u.is_some_and(|u| u.is_sparse(at)) {
        crate::hle::d3d12x::sparse::decommitted(at, PAGE);
    }
    mem::VirtualFree(at as *mut c_void, PAGE, mem::MEM_DECOMMIT);
}

/// Whether a physical page is mapped at the page holding `va` (a view, or a page moved there).
pub fn physical_page_at(va: usize) -> bool {
    let at = va & !(PAGE - 1);
    with(|s| s.views.contains_key(&at) || s.away_at.contains_key(&at))
}

/// Whether a nonempty request list names allocated section-backed page IDs.
pub(super) fn pages_live(pages: &[usize]) -> bool {
    !pages.is_empty() && with(|s| pages.iter().all(|&page| {
        let Some(index) = page_index(page) else { return false };
        s.blocks.range(..=index).next_back().is_some_and(|(&first, block)| block.has_page(index - first))
    }))
}

/// Where page `page`'s bytes are now (committing its home if `write`), or None.
unsafe fn bytes_of(page: usize, write: bool) -> Option<(usize, bool)> {
    let (away, home) = with(|s| (s.away.get(&page).copied(), s.page(page).map(|p| p.2)));
    if let Some(at) = away {
        return Some((at, true));
    }
    let home = home?;
    if committed(page) {
        return Some((home, true));
    }
    if write {
        if !sections::commit(home, PAGE) {
            return None;
        }
        note_committed(page);
        return Some((home, true));
    }
    Some((home, false))
}

/// `read_physical_pages` in alias mode.
pub unsafe fn read(pages: &[usize], offset: usize, dst: &mut [u8]) -> bool {
    copy_pages(pages, offset, dst.len(), |addr, done, n| {
        match addr {
            Some(a) => std::ptr::copy_nonoverlapping(a as *const u8, dst.as_mut_ptr().add(done), n),
            None => std::ptr::write_bytes(dst.as_mut_ptr().add(done), 0, n),
        }
    }, false)
}

/// `write_physical_pages` in alias mode.
pub unsafe fn write(pages: &[usize], offset: usize, src: &[u8]) -> bool {
    copy_pages(pages, offset, src.len(), |addr, done, n| {
        if let Some(a) = addr {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(done), a as *mut u8, n);
        }
    }, true)
}

/// Walk `len` bytes from `offset` in `pages`, calling `f(address or None for a page never used,
/// bytes done, bytes here)`. All pages are checked before any byte is touched.
unsafe fn copy_pages(pages: &[usize], offset: usize, len: usize, mut f: impl FnMut(Option<usize>, usize, usize), write: bool) -> bool {
    if len == 0 {
        return offset < PAGE;
    }
    let Some(need) = offset.checked_add(len).map(|end| end.div_ceil(PAGE)) else { return false };
    if offset >= PAGE || pages.len() < need {
        return false;
    }
    let mut addrs = Vec::with_capacity(need);
    for &p in &pages[..need] {
        let Some(a) = bytes_of(p, write) else { return false };
        addrs.push((p, a));
    }
    let (mut done, mut off) = (0, offset);
    for (page, (addr, used)) in addrs {
        let n = (PAGE - off).min(len - done);
        f(used.then_some(addr + off), done, n);
        if write && with(|s| !s.away.contains_key(&page)) {
            guards::written(page);
        }
        done += n;
        off = 0;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// alias state and guards are one per process: these tests run one at a time, and not
    /// beside the guards' own
    use super::guards::tests::SERIAL;

    unsafe fn read64(at: usize) -> u64 {
        std::ptr::read_volatile(at as *const u64)
    }

    unsafe fn write64(at: usize, v: u64) {
        std::ptr::write_volatile(at as *mut u64, v)
    }

    fn pages(n: usize) -> Vec<usize> {
        let mut arr = vec![0usize; n];
        assert!(unsafe { allocate(n, arr.as_mut_ptr()) });
        arr
    }

    fn dirty() -> Vec<usize> {
        let mut out = Vec::new();
        guards::take_dirty(&mut out, PAGE);
        out.sort_unstable();
        out
    }

    #[test]
    fn a_page_mapped_into_two_spaces_is_one_page_and_writes_reach_both() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let p = pages(4);
        let cpu = reserve_space(8 * PAGE).unwrap();
        let gpu = reserve_space(8 * PAGE).unwrap();
        unsafe {
            assert!(map(cpu + 2 * PAGE, &p[1..3]));
            assert!(map(gpu + 5 * PAGE, &[p[2]]));
            assert!(physical_page_at(gpu + 5 * PAGE + 123));
            write64(cpu + 3 * PAGE + 8, 0xABCD); // page p[2] through the first space
            assert_eq!(read64(gpu + 5 * PAGE + 8), 0xABCD, "the other space shows the same page");
            dirty();
            write64(gpu + 5 * PAGE, 1);
            assert_eq!(dirty(), vec![cpu + 3 * PAGE, gpu + 5 * PAGE], "a write through one view: both are uploaded");
            assert_eq!(free_space(cpu, 0, mem::MEM_RELEASE), Some(true));
            assert_eq!(read64(gpu + 5 * PAGE + 8), 0xABCD, "releasing one space leaves the page at the other");
            assert_eq!(free_space(gpu, 0, mem::MEM_RELEASE), Some(true));
            free(&p);
        }
    }

    #[test]
    fn scatter_without_pages_unmaps_only_the_listed_aliases() {
        use super::super::XMemMapPhysicalPagesScatter;
        if !enabled() {
            return crate::gpu::recomp::runner::skip_test("XCL_PAGE_ALIAS=1 required");
        }
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let p = pages(2);
        let first = reserve_space(2 * PAGE).unwrap();
        let survivor = reserve_space(2 * PAGE).unwrap();
        let addresses = [first as *mut c_void, (first + PAGE) as *mut c_void];
        let mut ids = [p[0] as u64, p[1] as u64];
        unsafe {
            assert_eq!(XMemMapPhysicalPagesScatter(addresses.as_ptr(), 2, ids.as_mut_ptr()), 1);
            assert!(map(survivor, &p));
            write64(first, 17);
            write64(first + PAGE, 23);

            assert_eq!(XMemMapPhysicalPagesScatter(addresses.as_ptr(), 2, std::ptr::null_mut()), 1);
            assert!(!physical_page_at(first));
            assert!(!physical_page_at(first + PAGE));
            assert_eq!((read64(survivor), read64(survivor + PAGE)), (17, 23));

            ids.swap(0, 1);
            assert_eq!(XMemMapPhysicalPagesScatter(addresses.as_ptr(), 2, ids.as_mut_ptr()), 1);
            assert_eq!((read64(first), read64(first + PAGE)), (23, 17));

            assert_eq!(free_space(first, 0, mem::MEM_RELEASE), Some(true));
            assert_eq!(free_space(survivor, 0, mem::MEM_RELEASE), Some(true));
            free(&p);
            assert!(!read(&p, 0, &mut [0; 8]), "released pages cannot be read by ID");
        }
    }

    #[test]
    fn remapping_an_address_shows_the_new_page_and_keeps_the_old_one() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let p = pages(2);
        let s = reserve_space(2 * PAGE).unwrap();
        unsafe {
            assert!(map(s, &[p[0]]));
            write64(s, 11);
            assert!(map(s, &[p[1]]));
            assert_eq!(read64(s), 0, "a page never written reads zeros");
            assert!(map(s, &[0]));
            assert!(!physical_page_at(s));
            assert!(map(s + PAGE, &[p[0]]));
            assert_eq!(read64(s + PAGE), 11, "the first page kept what was written to it");
            free_space(s, 0, mem::MEM_RELEASE);
            free(&p);
        }
    }

    #[test]
    fn pages_read_and_written_by_number_are_the_pages_the_title_sees() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let p = pages(3);
        let s = reserve_space(3 * PAGE).unwrap();
        unsafe {
            assert!(map(s, &p));
            dirty();
            // a DirectStorage destination: 16 bytes across the boundary of the first two pages
            assert!(write(&p, PAGE - 8, &[7u8; 16]));
            assert_eq!(read64(s + PAGE - 8), 0x0707_0707_0707_0707);
            assert_eq!(read64(s + PAGE), 0x0707_0707_0707_0707);
            assert_eq!(dirty(), vec![s, s + PAGE], "pages written by number are uploaded where they are mapped");
            write64(s + 2 * PAGE + 16, 5);
            let mut back = [0u8; 8];
            assert!(read(&p[2..], 16, &mut back));
            assert_eq!(u64::from_le_bytes(back), 5);
            let mut fresh = [1u8; 4];
            let q = pages(1);
            assert!(read(&q, 0, &mut fresh));
            assert_eq!(fresh, [0; 4], "a page never used reads zeros without being committed");
            assert!(!read(&p, PAGE, &mut back), "an offset past the first page is refused");
            free_space(s, 0, mem::MEM_RELEASE);
            free(&p);
            free(&q);
        }
    }

    #[test]
    fn a_page_moved_outside_the_spaces_goes_and_comes_back_with_its_bytes() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let p = pages(1);
        let s = reserve_space(PAGE).unwrap();
        // address space outside the spaces (as a title's own placeholder region would be)
        let out = unsafe { mem::VirtualAlloc(std::ptr::null(), PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) } as usize;
        unsafe {
            assert!(map(s, &p));
            write64(s, 21);
            assert!(map(out, &p));
            assert_eq!(read64(out), 21, "moved with its bytes");
            assert!(physical_page_at(out));
            write64(out, 22);
            let mut b = [0u8; 8];
            assert!(read(&p, 0, &mut b));
            assert_eq!(u64::from_le_bytes(b), 22, "read by number where it is now");
            assert!(map(out, &[0]));
            assert_eq!(read64(s), 22, "back home with what it held there");
            free_space(s, 0, mem::MEM_RELEASE);
            mem::VirtualFree(out as *mut c_void, 0, mem::MEM_RELEASE);
            free(&p);
        }
    }

    #[test]
    fn a_freed_page_id_is_rejected_while_its_neighbor_and_existing_view_survive() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let p = pages(2);
        let at = reserve_space(2 * PAGE).unwrap();
        unsafe {
            assert!(pages_live(&p));
            assert!(map(at, &p));
            write64(at, 17);
            write64(at + PAGE, 23);
            free(&[p[0], p[0]]);

            assert!(!pages_live(&[p[0]]) && pages_live(&[p[1]]));
            assert!(!read(&[p[0]], 0, &mut [0; 8]));
            assert!(!write(&[p[0]], 0, &[0; 8]));
            assert!(!map(at + PAGE, &[p[0]]));
            assert_eq!((read64(at), read64(at + PAGE)), (17, 23));

            assert_eq!(free_space(at, 0, mem::MEM_RELEASE), Some(true));
            free(&[p[1]]);
            assert!(!pages_live(&p));
        }
    }

    #[test]
    fn a_source_freed_after_validation_leaves_the_destination_mapped() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let p = pages(2);
        let last = pages(1);
        let at = reserve_space(2 * PAGE).unwrap();
        unsafe {
            assert!(map(at, &[p[0]]));
            write64(at, 0xABCD);
            // Each source was valid at preflight. Reject both an individual release and
            // a release that destroys the source block at the point of mapping.
            assert!(pages_live(&[p[1], last[0]]));
            free(&[p[1], last[0]]);
            let mapping = MAPPINGS.lock().unwrap_or_else(|e| e.into_inner());
            for source in [p[1], last[0]] {
                assert_eq!(map_one(at, source, &mapping), Err(fnd::ERROR_INVALID_PARAMETER));
                assert!(physical_page_at(at));
                assert_eq!(read64(at), 0xABCD);
            }
            drop(mapping);
            assert!(!map(at, &[p[1]]));
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);
            assert!(!map(at, &[0, last[0]]));
            assert_eq!(fnd::GetLastError(), fnd::ERROR_INVALID_PARAMETER);
            assert_eq!(read64(at), 0xABCD);

            assert_eq!(free_space(at, 0, mem::MEM_RELEASE), Some(true));
            free(&p[..1]);
        }
    }

    #[test]
    fn freeing_pages_at_the_bitmap_boundary_leaves_the_other_bits_alone() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let p = pages(65);
        unsafe {
            assert!(pages_live(&p));
            free(&p[63..64]);
            assert!(!pages_live(&p[63..64]));
            assert!(pages_live(&[p[0], p[62], p[64]]));
            free(&[p[64], p[63], p[64]]);
            assert!(!pages_live(&p[64..]));
            assert!(pages_live(&p[..63]));
            free(&p[..63]);
            assert!(!pages_live(&[p[0]]));
            assert!(!pages_live(&[p[62]]));
        }
    }

    #[test]
    fn a_page_not_from_the_title_is_refused() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = reserve_space(PAGE).unwrap();
        unsafe {
            assert!(!map(s, &[page_number(1 << 40)]));
            assert!(!map(s + 4, &[0]), "an address not on a page boundary is refused");
            free_space(s, 0, mem::MEM_RELEASE);
        }
    }
}
