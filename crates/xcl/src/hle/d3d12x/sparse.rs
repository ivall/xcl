//! The title's own memory, seen by the GPU: sparse windows.
//!
//! A console title keeps GPU data in memory it reserves and commits itself (XDefiant: one
//! 8 GB reservation of 64 KiB pages it commits in steps and places buffers in) – on the
//! console all of it is GPU-visible at the same addresses. XCL gives such reservations
//! addresses in the *sparse windows* of the unified-memory block (`gpu::uma`: windows 0..2
//! in `for_runtime`), so translated shaders reach them through `mem`…`mem2` like any unified
//! memory. On the GPU each sparse window is a reserved (tiled) buffer of 4 GiB whose 64 KiB
//! tiles are mapped onto video-memory heaps as the title commits pages.
//! The CPU's view stays the title's memory: what the CPU wrote (write-watch) is copied into
//! the tiles before the title's next ExecuteCommandLists via an asynchronous ring-buffered
//! staging upload queue, and the title's queues GPU-wait on the upload fence (no CPU stall).
//! Only the bytes the CPU changed go (64-byte chunks against a hash of what was sent last,
//! `changed_chunks`): the rest of a written page may hold what the GPU wrote there (counters,
//! indirect arguments) and must not be overwritten with the CPU's stale view of it.
//! Buffers the title places there get reserved resources over the same tiles, so PC-side
//! copies, views and barriers see the same memory.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::c_void;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_DECOMMIT, PAGE_GUARD,
    PAGE_NOACCESS, PAGE_READWRITE,
};

pub mod gpu_view;

const TILE: usize = 0x1_0000;
/// tiles of a video-memory pool heap (4 MiB). A heap goes back only when every tile of it is
/// free (`State::retire_free_pools`), and a title frees its tiles in runs spread over all its
/// heaps: with heaps of 64 MiB a retail title's world load unmapped 1.5 GB of its 4.6 GB of
/// tiles and 3 of 78 heaps went back (0.2 GB); with heaps of 4 MiB 1.25 GB went back.
const POOL_TILES: u32 = 64;

/// `XCL_SPARSE_VRAM_CAP_MB=<N>` (unset by default): keep at most `N` MiB of sparse pool heaps in
/// local VRAM (`DEFAULT`), and allocate additional sparse pools in `CUSTOM` L0 write-back system
/// memory so 8 GB cards stay under the DXGI local VRAM budget.
fn sparse_vram_cap_pools() -> Option<usize> {
    static CAP: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let mb: usize = std::env::var("XCL_SPARSE_VRAM_CAP_MB").ok()?.trim().parse().ok()?;
        let pool_bytes = POOL_TILES as usize * TILE;
        let pools = (mb << 20) / pool_bytes;
        crate::xlog_info!("D3D12.X: XCL_SPARSE_VRAM_CAP_MB={mb} - sparse pools past {pools} ({} MiB) in L0 system memory", (pools * pool_bytes) >> 20);
        Some(pools)
    })
}
/// tiles decommitted while no title work was submitted that stay mapped (`State::parked`) before
/// they are unmapped together: 256 MB of video memory
const PARKED_MAX: usize = 4096;
/// bytes of the ring-buffered upload staging buffer
const STAGING: usize = 16 << 20;
/// alignment of sub-allocations inside the upload staging ring (write-combined cacheline)
const RING_ALIGN: usize = 64;
/// number of in-flight upload command allocators / lists
const UPLOAD_SLOTS: usize = 16;
/// bytes of the dedicated pre-zeroed buffer for tile/hole clears (1 MiB)
const ZERO_BUF: usize = 1 << 20;
/// bytes of the GPU→CPU readback staging buffer and maximum per readback job
const READBACK_STAGING: usize = 4 << 20;
const QUEUE_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

const PAGE: usize = 0x1000;
const PAGES_PER_TILE: usize = TILE / PAGE;
const CHUNK: usize = 64;
const CHUNKS: usize = PAGE / CHUNK;

/// Fast non-cryptographic integer hasher for page/tile lookup maps on the hot submit path.
#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn finish(&self) -> u64 {
        // Tile addresses are 64 KiB aligned. Move mixed bits into the low bucket-index bits.
        self.0.rotate_right(23)
    }
}

type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

struct TileEntry {
    pool: (u32, u32),
    /// bit `i` set => 4 KiB page `i` of this 64 KiB tile is uncommitted (a hole)
    holes: u16,
    /// per-4 KiB-page hashes of the 64-byte chunks last sent to the GPU (`None` = all zero)
    sent: [Option<Box<[u64; CHUNKS]>>; PAGES_PER_TILE],
}

impl TileEntry {
    fn with_holes(pool: (u32, u32), holes: u16) -> Self {
        Self { pool, holes, sent: std::array::from_fn(|_| None) }
    }
}

/// Query actual OS page commitment across the 64 KiB tile at `tile`.
/// Bit `i` in the returned mask is set whenever 4 KiB page `tile + i * PAGE` is not committed
/// and accessible (`MEM_RESERVE`, `MEM_FREE`, `PAGE_NOACCESS`, `PAGE_GUARD`, etc.).
unsafe fn query_tile_holes(tile: usize) -> u16 {
    let mut holes = 0xFFFFu16;
    let end = tile.saturating_add(TILE);
    let mut cur = tile;
    while cur < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        // SAFETY: `VirtualQuery` inspects process virtual memory metadata without dereferencing `cur`.
        let n = VirtualQuery(cur as *const c_void, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>());
        if n == 0 || mbi.RegionSize == 0 {
            break;
        }
        let region_end = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize).min(end);
        let accessible = mbi.State == MEM_COMMIT && mbi.Protect != 0 && (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
        if accessible {
            let mut pg = cur;
            while pg < region_end {
                holes &= !(1u16 << ((pg - tile) / PAGE));
                pg += PAGE;
            }
        }
        if region_end <= cur {
            break;
        }
        cur = region_end;
    }
    holes
}

/// Update `mapped`, `free`, and `to_zero` for a decommit of `[a, a + n)`.
/// Returns the base addresses of tiles that became completely uncommitted (`holes == 0xFFFF`)
/// and were removed from `mapped` (ready to be unmapped from their sparse window buffers).
fn decommit_tiles(
    mapped: &mut FxHashMap<usize, TileEntry>,
    pinned: &FxHashMap<(u32, u32), u32>,
    free: &mut Vec<(u32, u32)>,
    to_zero: &mut Vec<(usize, usize)>,
    a: usize,
    n: usize,
) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let mut unmapped: Vec<usize> = Vec::new();
    let mut pg = a & !(PAGE - 1);
    let end = a.saturating_add(n);
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = t.saturating_add(TILE).min(end);
        if let Some(entry) = mapped.get_mut(&t) {
            let mut newly_holed = 0u16;
            let mut p = pg;
            while p < tile_end {
                let idx = (p - t) / PAGE;
                entry.sent[idx] = None;
                let bit = 1u16 << idx;
                if entry.holes & bit == 0 {
                    entry.holes |= bit;
                    newly_holed |= bit;
                }
                p += PAGE;
            }
            if entry.holes == 0xFFFF {
                unmapped.push(t);
            } else if newly_holed != 0 {
                for idx in 0..PAGES_PER_TILE {
                    if newly_holed & (1u16 << idx) != 0 {
                        to_zero.push((t + idx * PAGE, PAGE));
                    }
                }
            }
        }
        if tile_end <= pg {
            break;
        }
        pg = tile_end;
    }
    for &t in &unmapped {
        if let Some(entry) = mapped.remove(&t) {
            if !pinned.contains_key(&entry.pool) {
                free.push(entry.pool);
            }
        }
    }
    if !unmapped.is_empty() {
        to_zero.retain(|&(za, _)| mapped.contains_key(&(za & !(TILE - 1))));
    }
    unmapped
}

struct UploadSlot {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence_val: u64,
}

struct RetiredTiles {
    tiles: Vec<(u32, u32)>,
    /// Fence marks on `self.queue` and every known title queue (`None` = signal failed, keep held).
    marks: Option<Vec<(ID3D12Fence, u64)>>,
}

struct PendingUnmap {
    seq: u64,
    entries: Vec<(usize, (u32, u32))>,
    /// Fence marks on every known title queue (`None` = signal failed, keep mapped/held).
    marks: Option<Vec<(ID3D12Fence, u64)>>,
}

struct State {
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    /// reusable auto-reset Win32 event for fence waits (no per-wait CreateEventW/CloseHandle)
    event: HANDLE,
    /// dedicated readback queue (NEVER waited on by title queues, so waiting on title queues
    /// here cannot form a circular GPU deadlock `title_qA -> title_qB -> sparse_q -> title_qA`)
    rb_queue: ID3D12CommandQueue,
    /// fence signalled on title queues so `rb_queue` waits before copying GPU writes back to CPU
    rb_fence: ID3D12Fence,
    rb_value: u64,
    /// fence signalled on `rb_queue` for CPU-side readback completion waits
    rb_done_fence: ID3D12Fence,
    rb_done_value: u64,
    /// (window start, reserved buffer) of each sparse window
    windows: Vec<(usize, ID3D12Resource)>,
    /// pool heaps (None: given back, its index free for the next one)
    pools: Vec<Option<ID3D12Heap>>,
    /// parallel to `pools`: true when pool `i` lives in L0 system memory (`XCL_SPARSE_VRAM_CAP_MB`)
    pool_is_sys: Vec<bool>,
    free: Vec<(u32, u32)>,
    /// pools all of whose tiles came free: released once the GPU is past `(value, rb_done_value)`
    /// and a moment went by (titles' queues and `rb_queue` may still read what was mapped there)
    retired: Vec<(ID3D12Heap, bool, u64, u64, std::time::Instant)>,
    /// tile address → pool tile + uncommitted page hole mask + per-page sent chunk hashes
    mapped: FxHashMap<usize, TileEntry>,
    /// pool tiles also used by placed buffers (not freed while they are)
    pinned: FxHashMap<(u32, u32), u32>,
    /// tile address → pool tile of tiles decommitted since the last title submit, still mapped
    /// in their window: unmapped together (`flush_parked`) before the next submit, when the free
    /// list runs out, or past `PARKED_MAX`; a commit there before that takes its tile back with
    /// no new mapping. (A retail title moving physical pages one at a time while it loads
    /// decommits and commits again thousands of tiles a second, and each `UpdateTileMappings`
    /// took ~0.15 ms.)
    parked: FxHashMap<usize, (u32, u32)>,
    /// parked tiles awaiting in-flight title queues before `UpdateTileMappings(NULL)` executes
    pending_unmaps: VecDeque<PendingUnmap>,
    /// tile VA → `(pool_tile, unmap_seq)` for tiles still mapped in hardware inside `pending_unmaps`;
    /// re-committing a VA cancels its pending unmap with no GPU tile-mapping change
    pending_unmap_va: FxHashMap<usize, ((u32, u32), u64)>,
    /// monotonic sequence id preventing ABA cancellation across repeated decommit/recommit cycles
    unmap_seq: u64,
    /// unmapped parked pool tiles awaiting `self.queue` and in-flight title queues before entering `free`
    quarantined: VecDeque<RetiredTiles>,
    /// unpinned unused pool tiles queued from `forget_in` (cannot signal queues under D3D12 release locks)
    /// until `flush_pending_quarantine` places them behind GPU fences
    pending_quarantine: Vec<(u32, u32)>,
    /// per-title-queue fences used to mark in-flight GPU work (non-owning raw queue key → fence + value)
    queue_fences: FxHashMap<usize, (ID3D12Fence, u64)>,
    /// placed buffers → the pool tiles they use
    buffers: FxHashMap<usize, Vec<(u32, u32)>>,
    /// reused tiles / decommitted partial pages to clear: (address, bytes)
    to_zero: Vec<(usize, usize)>,
    /// ring-buffered upload staging heap (`STAGING` bytes, persistently mapped)
    staging: Option<(ID3D12Resource, *mut u8)>,
    /// monotonic ring-buffer write/submitted/retired byte cursors
    ring_write: u64,
    ring_submitted: u64,
    ring_free: u64,
    /// submitted ring chunks awaiting GPU completion: `(fence_val, ring_end_pos)`
    inflight: VecDeque<(u64, u64)>,
    /// dedicated 1 MiB pre-zeroed buffer for tile/hole clears (zero staging bandwidth)
    zero_buf: Option<ID3D12Resource>,
    /// ring of async upload command allocators + lists
    slots: Vec<UploadSlot>,
    next_slot: usize,
    /// dedicated readback staging buffer + command allocator / list
    readback: Option<(ID3D12Resource, *mut u8)>,
    rb_alloc: ID3D12CommandAllocator,
    rb_list: ID3D12GraphicsCommandList,
    rb_last_fence: u64,
    /// the fence value each title queue last waited for
    waited: FxHashMap<usize, u64>,
    /// tile mappings were queued since the last `signal` (`map_tiles` / `unmap_tiles`)
    mapping_queued: std::sync::atomic::AtomicBool,
    /// the fence value of the last `signal` that came after tile mappings
    mapped_value: u64,
    /// `mapped` as sorted (start, end) runs (None: to be made again)
    runs: Option<Vec<(usize, usize)>>,
    /// reusable buffer for dirty runs in `before_execute` (avoids per-submit heap alloc)
    scratch_runs: Vec<(usize, usize)>,
    /// Write-watch may already have been reset for pages with unsent CPU writes.
    retry_pages: Vec<usize>,
}

impl Drop for State {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

const K0: u64 = 0x9E37_79B9_7F4A_7C15;
const K1: u64 = 0xBF58_476D_1CE4_E5B9;
const K2: u64 = 0x94D0_49BB_1331_11EB;
const K3: u64 = 0x243F_6A88_85A3_08D3;

/// 4-way parallel hash over 8 `u64` words (64 bytes): breaks serial multiply latency chain.
#[inline(always)]
const fn hash_words(w: [u64; 8]) -> u64 {
    let a = (w[0] ^ K3).wrapping_mul(K0) ^ w[4].wrapping_mul(K1);
    let b = (w[1] ^ K0).wrapping_mul(K1) ^ w[5].wrapping_mul(K2);
    let c = (w[2] ^ K1).wrapping_mul(K2) ^ w[6].wrapping_mul(K3);
    let d = (w[3] ^ K2).wrapping_mul(K3) ^ w[7].wrapping_mul(K0);
    let h = a.rotate_left(27).wrapping_add(b.rotate_left(13)) ^ c.rotate_left(41).wrapping_add(d.rotate_left(53));
    h.wrapping_mul(K0) ^ (h >> 32)
}

const ZERO_HASH: u64 = hash_words([0; 8]);
const ZERO_HASHES: [u64; CHUNKS] = [ZERO_HASH; CHUNKS];

#[inline(always)]
fn read_chunk_words(c: &[u8]) -> [u64; 8] {
    debug_assert_eq!(c.len(), CHUNK);
    let p = c.as_ptr() as *const [u8; 8];
    unsafe {
        [
            u64::from_le_bytes(p.add(0).read_unaligned()),
            u64::from_le_bytes(p.add(1).read_unaligned()),
            u64::from_le_bytes(p.add(2).read_unaligned()),
            u64::from_le_bytes(p.add(3).read_unaligned()),
            u64::from_le_bytes(p.add(4).read_unaligned()),
            u64::from_le_bytes(p.add(5).read_unaligned()),
            u64::from_le_bytes(p.add(6).read_unaligned()),
            u64::from_le_bytes(p.add(7).read_unaligned()),
        ]
    }
}

/// hash of 64 bytes
#[inline(always)]
fn chunk_hash(c: &[u8]) -> u64 {
    hash_words(read_chunk_words(c))
}

#[inline(always)]
fn push_run(out: &mut Vec<(usize, usize)>, addr: usize, len: usize) {
    match out.last_mut() {
        Some(last) if last.0 + last.1 == addr => last.1 += len,
        _ => out.push((addr, len)),
    }
}

#[inline]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static AVX2: OnceLock<bool> = OnceLock::new();
        *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn is_chunk_zero_avx2(c: &[u8]) -> bool {
    use std::arch::x86_64::{_mm256_loadu_si256, _mm256_or_si256, _mm256_testz_si256, __m256i};
    debug_assert_eq!(c.len(), CHUNK);
    let p = c.as_ptr() as *const __m256i;
    let v0 = _mm256_loadu_si256(p);
    let v1 = _mm256_loadu_si256(p.add(1));
    let any = _mm256_or_si256(v0, v1);
    _mm256_testz_si256(any, any) != 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_page_avx2(page: &[u8], pg: usize, slot: &mut Option<Box<[u64; CHUNKS]>>, out: &mut Vec<(usize, usize)>) {
    use std::arch::x86_64::{_mm256_loadu_si256, _mm256_or_si256, _mm256_testz_si256, __m256i};
    if let Some(sent) = slot.as_deref_mut() {
        for (k, c) in page.chunks_exact(CHUNK).enumerate() {
            if sent[k] == ZERO_HASH && is_chunk_zero_avx2(c) {
                continue;
            }
            let h = chunk_hash(c);
            if h == sent[k] {
                continue;
            }
            sent[k] = h;
            push_run(out, pg + k * CHUNK, CHUNK);
        }
        return;
    }
    // Fresh all-zero page fast path: scan 128 bytes (2 chunks) per iteration with AVX2.
    let ptr = page.as_ptr() as *const __m256i;
    let pairs = page.len() / (CHUNK * 2);
    let mut first_nz_chunk = None;
    for pair in 0..pairs {
        let base = ptr.add(pair * 4);
        let a = _mm256_or_si256(_mm256_loadu_si256(base), _mm256_loadu_si256(base.add(1)));
        let b = _mm256_or_si256(_mm256_loadu_si256(base.add(2)), _mm256_loadu_si256(base.add(3)));
        let any = _mm256_or_si256(a, b);
        if _mm256_testz_si256(any, any) == 0 {
            let k0 = pair * 2;
            first_nz_chunk = Some(if _mm256_testz_si256(a, a) == 0 { k0 } else { k0 + 1 });
            break;
        }
    }
    let Some(start_k) = first_nz_chunk else { return };
    for (k, c) in page.chunks_exact(CHUNK).enumerate().skip(start_k) {
        if is_chunk_zero_avx2(c) {
            continue;
        }
        let h = chunk_hash(c);
        if h == ZERO_HASH {
            continue;
        }
        let sent = slot.get_or_insert_with(|| Box::new(ZERO_HASHES));
        sent[k] = h;
        push_run(out, pg + k * CHUNK, CHUNK);
    }
}

#[inline]
fn scan_page_scalar(page: &[u8], pg: usize, slot: &mut Option<Box<[u64; CHUNKS]>>, out: &mut Vec<(usize, usize)>) {
    if let Some(sent) = slot.as_deref_mut() {
        for (k, c) in page.chunks_exact(CHUNK).enumerate() {
            let h = chunk_hash(c);
            if h == sent[k] {
                continue;
            }
            sent[k] = h;
            push_run(out, pg + k * CHUNK, CHUNK);
        }
        return;
    }
    // Page has never had non-zero data sent: skip zero chunks with a cheap OR-reduction and
    // allocate the hash array only if at least one chunk is non-zero.
    let mut chunks = page.chunks_exact(CHUNK).enumerate();
    while let Some((k, c)) = chunks.next() {
        let w = read_chunk_words(c);
        if (w[0] | w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) == 0 {
            continue;
        }
        let h = hash_words(w);
        if h == ZERO_HASH {
            continue;
        }
        let sent = slot.insert(Box::new(ZERO_HASHES));
        sent[k] = h;
        push_run(out, pg + k * CHUNK, CHUNK);
        for (k2, c2) in chunks {
            let h2 = chunk_hash(c2);
            if h2 == ZERO_HASH {
                continue;
            }
            sent[k2] = h2;
            push_run(out, pg + k2 * CHUNK, CHUNK);
        }
        break;
    }
}

/// Scan a 4 KiB page at `pg`, appending any changed 64-byte chunk runs directly to `out`
/// (zero per-page heap allocation; skips allocating `slot` completely when a fresh zero page
/// is still all zeros; uses AVX2 zero-skipping when available).
#[inline]
fn scan_page(page: &[u8], pg: usize, slot: &mut Option<Box<[u64; CHUNKS]>>, out: &mut Vec<(usize, usize)>) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && page.len() == PAGE {
        unsafe {
            scan_page_avx2(page, pg, slot, out);
        }
        return;
    }
    scan_page_scalar(page, pg, slot, out);
}

/// Minimum dirty pages in one `before_execute` before fanning out `scan_page` across `ScanPool`.
const PARALLEL_SCAN_PAGES: usize = 64;
const MAX_SCAN_THREADS: usize = 6;

fn scan_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        if let Some(n) = std::env::var("XCL_SPARSE_SCAN_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
            return n.clamp(1, 16);
        }
        std::thread::available_parallelism().map_or(1, |n| n.get() / 2).clamp(1, MAX_SCAN_THREADS)
    })
}

struct ScanPoolState {
    gen: u64,
    active_helpers: usize,
    pending_helpers: usize,
    func: Option<unsafe fn(*const c_void)>,
    ctx: usize,
}

/// Persistent worker pool for parallel dirty-page chunk scanning in `before_execute` (avoids
/// per-submit thread creation overhead while keeping tile hash state lock-free per page).
struct ScanPool {
    run_lock: Mutex<()>,
    state: Arc<(Mutex<ScanPoolState>, Condvar, Condvar)>,
    helpers: usize,
}

impl ScanPool {
    fn new(max_threads: usize) -> Self {
        let helpers = max_threads.saturating_sub(1);
        let state = Arc::new((
            Mutex::new(ScanPoolState { gen: 0, active_helpers: 0, pending_helpers: 0, func: None, ctx: 0 }),
            Condvar::new(),
            Condvar::new(),
        ));
        let mut spawned = 0usize;
        for _ in 0..helpers {
            let id = spawned;
            let shared = Arc::clone(&state);
            if std::thread::Builder::new()
                .name(format!("xcl-sparse-scan-{id}"))
                .spawn(move || {
                    let mut seen = 0u64;
                    loop {
                        let job = {
                            let mut g = shared.0.lock().unwrap();
                            while g.gen == seen {
                                g = shared.1.wait(g).unwrap();
                            }
                            seen = g.gen;
                            if id < g.active_helpers { g.func.map(|f| (f, g.ctx)) } else { None }
                        };
                        if let Some((func, ctx)) = job {
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { func(ctx as *const c_void) }));
                            let mut g = shared.0.lock().unwrap();
                            g.pending_helpers -= 1;
                            if g.pending_helpers == 0 {
                                shared.2.notify_one();
                            }
                        }
                    }
                })
                .is_ok()
            {
                spawned += 1;
            }
        }
        ScanPool { run_lock: Mutex::new(()), state, helpers: spawned }
    }

    fn run<F: Fn() + Sync>(&self, total_threads: usize, f: &F) {
        let helpers = total_threads.saturating_sub(1).min(self.helpers);
        if helpers == 0 {
            f();
            return;
        }
        unsafe fn trampoline<F: Fn() + Sync>(ctx: *const c_void) {
            (*(ctx as *const F))();
        }
        let _serial = self.run_lock.lock().unwrap();
        {
            let mut g = self.state.0.lock().unwrap();
            g.func = Some(trampoline::<F>);
            g.ctx = (f as *const F) as usize;
            g.active_helpers = helpers;
            g.pending_helpers = helpers;
            g.gen = g.gen.wrapping_add(1);
            self.state.1.notify_all();
        }
        let caller_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        {
            let mut g = self.state.0.lock().unwrap();
            while g.pending_helpers > 0 {
                g = self.state.2.wait(g).unwrap();
            }
            g.func = None;
            g.ctx = 0;
            g.active_helpers = 0;
        }
        if let Err(payload) = caller_res {
            std::panic::resume_unwind(payload);
        }
    }
}

fn scan_pool() -> &'static ScanPool {
    static POOL: OnceLock<ScanPool> = OnceLock::new();
    POOL.get_or_init(|| ScanPool::new(scan_threads()))
}

/// Disjoint page work item for parallel `scan_page` execution.
#[derive(Clone, Copy)]
struct ScanItem {
    pg: usize,
    slot: *mut Option<Box<[u64; CHUNKS]>>,
}

// SAFETY: `scan_dirty_pages_into` deduplicates sorted `pg` values before constructing `ScanItem`s,
// so every item owns a unique `(tile, idx)` hash slot and reads immutable title memory at `pg`.
unsafe impl Sync for ScanItem {}
unsafe impl Send for ScanItem {}

/// Scan all write-watched `dirty` pages (sorted ascending, `page` bytes each) overlapping `mapped`,
/// appending coalesced changed 64-byte runs into `out` in ascending address order. Uses the
/// persistent `ScanPool` when at least `PARALLEL_SCAN_PAGES` mapped pages are dirty.
unsafe fn scan_dirty_pages_into(
    mapped: &mut FxHashMap<usize, TileEntry>,
    dirty: &[usize],
    page: usize,
    max_threads: usize,
    out: &mut Vec<(usize, usize)>,
) {
    if dirty.is_empty() {
        return;
    }
    // Small dirty sets scan directly with zero intermediate allocation.
    if dirty.len() < PARALLEL_SCAN_PAGES || max_threads <= 1 {
        let mut i = 0;
        while i < dirty.len() {
            let a = dirty[i];
            let mut n = page;
            i += 1;
            while i < dirty.len() && dirty[i] == a + n {
                n += page;
                i += 1;
            }
            let mut x = a;
            while x < a + n {
                let t = x & !(TILE - 1);
                let end = (t + TILE).min(a + n);
                if let Some(entry) = mapped.get_mut(&t) {
                    let mut pg = x & !(PAGE - 1);
                    while pg < end {
                        let idx = (pg - t) / PAGE;
                        if entry.holes & (1u16 << idx) == 0 {
                            let bytes = std::slice::from_raw_parts(pg as *const u8, PAGE);
                            scan_page(bytes, pg, &mut entry.sent[idx], out);
                        }
                        pg += PAGE;
                    }
                }
                x = end;
            }
        }
        return;
    }

    // Collect deduplicated, ascending `(pg, &mut entry.sent[idx])` pairs once under `&mut mapped`.
    let mut items: Vec<ScanItem> = Vec::with_capacity(dirty.len());
    let mut last_pg = usize::MAX;
    let mut i = 0;
    while i < dirty.len() {
        let a = dirty[i];
        let mut n = page;
        i += 1;
        while i < dirty.len() && dirty[i] == a + n {
            n += page;
            i += 1;
        }
        let mut x = a;
        while x < a + n {
            let t = x & !(TILE - 1);
            let end = (t + TILE).min(a + n);
            if let Some(entry) = mapped.get_mut(&t) {
                let mut pg = x & !(PAGE - 1);
                while pg < end {
                    if pg != last_pg {
                        last_pg = pg;
                        let idx = (pg - t) / PAGE;
                        if entry.holes & (1u16 << idx) == 0 {
                            items.push(ScanItem { pg, slot: &mut entry.sent[idx] as *mut _ });
                        }
                    }
                    pg += PAGE;
                }
            }
            x = end;
        }
    }

    if items.len() < PARALLEL_SCAN_PAGES {
        for item in items {
            let bytes = std::slice::from_raw_parts(item.pg as *const u8, PAGE);
            scan_page(bytes, item.pg, &mut *item.slot, out);
        }
        return;
    }

    let threads = max_threads.min(items.len() / 16).max(1);
    let chunk_len = items.len().div_ceil(threads);
    let chunks: Vec<&[ScanItem]> = items.chunks(chunk_len).collect();
    let buckets: Vec<Mutex<Vec<(usize, usize)>>> = (0..chunks.len()).map(|_| Mutex::new(Vec::new())).collect();
    let next = AtomicUsize::new(0);
    scan_pool().run(chunks.len(), &|| loop {
        let b = next.fetch_add(1, Ordering::Relaxed);
        let Some(&slice) = chunks.get(b) else { break };
        let mut local = Vec::new();
        for &item in slice {
            let bytes = std::slice::from_raw_parts(item.pg as *const u8, PAGE);
            scan_page(bytes, item.pg, &mut *item.slot, &mut local);
        }
        *buckets[b].lock().unwrap() = local;
    });
    for b in buckets {
        for (addr, len) in b.into_inner().unwrap() {
            push_run(out, addr, len);
        }
    }
}

/// The byte ranges (offset, length) of `page` whose 64-byte chunks differ from `sent`
/// (updated to the page's chunks).
pub fn changed_chunks(page: &[u8], sent: &mut [u64; CHUNKS]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (k, c) in page.chunks_exact(CHUNK).enumerate() {
        let h = chunk_hash(c);
        if h == sent[k] {
            continue;
        }
        sent[k] = h;
        push_run(&mut out, k * CHUNK, CHUNK);
    }
    out
}

/// Plan a contiguous sub-allocation `(aligned_pos, chunk_len)` inside a ring buffer of `cap`
/// bytes starting at monotonic cursor `ring_write`, keeping each chunk `<= cap / 2` and
/// avoiding mid-copy wrap-around without wasting large tail gaps.
#[inline]
fn plan_ring_chunk(ring_write: u64, want: usize, cap: usize) -> (u64, usize) {
    let mut chunk = want.min(cap / 2);
    let mut aligned = (ring_write + (RING_ALIGN as u64 - 1)) & !(RING_ALIGN as u64 - 1);
    let off = (aligned as usize) % cap;
    if off + chunk > cap {
        let tail = cap - off;
        if tail >= PAGE {
            chunk = tail;
        } else {
            aligned += tail as u64;
        }
    }
    (aligned, chunk)
}

// D3D12 objects are free-threaded; staging pointers and event handle are only used under STATE lock
unsafe impl Send for State {}

/// Wait for `fence` to reach `target` using a reusable auto-reset `event`, bounded by `timeout`.
/// Resets `event` before arming and loops until `GetCompletedValue() >= target` or the deadline
/// expires, so a late completion from an earlier timed-out wait cannot satisfy a newer wait early.
/// (As in #1419 by gridoe and #2821 for `main`, same signature, so the two merge as one.)
unsafe fn wait_fence_until(fence: &ID3D12Fence, event: HANDLE, target: u64, timeout: std::time::Duration) -> bool {
    let completed = fence.GetCompletedValue();
    if completed >= target {
        return completed != u64::MAX;
    }
    let _ = windows::Win32::System::Threading::ResetEvent(event);
    let _ = fence.SetEventOnCompletion(target, event);
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let completed = fence.GetCompletedValue();
        if completed >= target {
            return completed != u64::MAX;
        }
        let rem = deadline.saturating_duration_since(std::time::Instant::now());
        if rem.is_zero() {
            return false;
        }
        let ms = rem.as_millis().clamp(1, u32::MAX as u128) as u32;
        windows::Win32::System::Threading::WaitForSingleObject(event, ms);
    }
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// The state's queue again, outside `STATE` (`queue`). `STATE` is held around D3D12 calls (an
/// upload's allocator `Reset` in `begin_slot`), and `late_release` asks for the queue from a
/// resource's last `Release`, which D3D12 can make inside a call of its own on another thread
/// (a list's `Reset`, with `XCL_DRED=1` on every start): the two then waited for each other.
static QUEUE: Mutex<Option<ID3D12CommandQueue>> = Mutex::new(None);

/// (bytes of video memory in pool heaps, of those mapped) – `XCL_ACTIVITY`
/// Bytes of the pool heaps given back but not released yet (titles' queues may still read
/// them): memory accounting, #2688.
pub fn retired_bytes() -> usize {
    lock_state().as_ref().map_or(0, |s| s.retired.iter().filter(|r| !r.1).count() * POOL_TILES as usize * TILE)
}

/// Bytes of the title's pages committed in the sparse windows (their CPU side): memory
/// accounting, #2688.
pub fn committed_bytes() -> usize {
    lock_state().as_ref().map_or(0, |s| s.committed_bytes())
}

pub fn pool_bytes() -> (usize, usize) {
    let g = lock_state();
    let Some(s) = g.as_ref() else { return (0, 0) };
    let vram_pools = s
        .pools
        .iter()
        .enumerate()
        .filter(|(i, p)| p.is_some() && !s.pool_is_sys.get(*i).copied().unwrap_or(false))
        .count()
        * POOL_TILES as usize
        * TILE;
    let vram_free = s.free.iter().filter(|&&(h, _)| !s.pool_is_sys.get(h as usize).copied().unwrap_or(false)).count() * TILE;
    (vram_pools, vram_pools.saturating_sub(vram_free))
}
/// commits before the device existed: the pages committed, as disjoint page ranges (start →
/// end) merged with their neighbours, so a commit or decommit costs O(log n) and not a scan of
/// every page committed so far. Like `committed` / `decommitted` with a state, a range counts
/// every page it touches; `init` commits them once the windows exist.
static EARLY: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

/// The pages `[a, a + n)` touches, as `[start, end)`.
fn page_span(a: usize, n: usize) -> (usize, usize) {
    (a & !(PAGE - 1), a.saturating_add(n).saturating_add(PAGE - 1) & !(PAGE - 1))
}

/// Add the pages `[a, b)` to `early`, merged with the ranges they overlap or touch.
fn early_add(early: &mut BTreeMap<usize, usize>, mut a: usize, mut b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..=a).next_back() {
        if e >= a {
            a = s;
            b = b.max(e);
            early.remove(&s);
        }
    }
    loop {
        let Some((s, e)) = early.range(a..=b).next().map(|(&s, &e)| (s, e)) else { break };
        b = b.max(e);
        early.remove(&s);
    }
    early.insert(a, b);
}

/// Take the pages `[a, b)` out of `early`; a range that overlaps it keeps the rest.
fn early_remove(early: &mut BTreeMap<usize, usize>, a: usize, b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..a).next_back() {
        if e > a {
            early.insert(s, a);
            if e > b {
                early.insert(b, e);
                return;
            }
        }
    }
    loop {
        let Some((s, e)) = early.range(a..b).next().map(|(&s, &e)| (s, e)) else { break };
        early.remove(&s);
        if e > b {
            early.insert(b, e);
            break;
        }
    }
}
/// generation counter incremented whenever the set of mapped tiles changes (for watch_ranges cache)
static MAPPED_GEN: AtomicU64 = AtomicU64::new(0);

/// Generation counter of `mapped_ranges()` (changes only when tiles are mapped or unmapped).
pub fn mapped_gen() -> u64 {
    MAPPED_GEN.load(Ordering::Acquire)
}

fn buffer_desc(bytes: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

/// The sparse windows' buffers on `dev`: `(window_index, buffer_raw, gpu_va)` for each
/// sparse window in `gpu::uma` (empty: no sparse windows).
pub unsafe fn init(dev: &ID3D12Device) -> Vec<(usize, usize, u64)> {
    let Some(u) = crate::gpu::uma::get() else { return Vec::new() };
    let mut out = Vec::new();
    let mut windows = Vec::new();
    for k in 0..u.shader_windows() {
        if !u.is_sparse_window(k) {
            continue;
        }
        let (base, size) = u.window(k);
        let mut r: Option<ID3D12Resource> = None;
        if let Err(e) = dev.CreateReservedResource(&buffer_desc(size as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut r) {
            crate::xlog_warn!("sparse window {k}: CreateReservedResource: {e}");
            return Vec::new();
        }
        let Some(r) = r else { return Vec::new() };
        if super::real::dred_on() {
            super::real::name_resource(r.as_raw(), &format!("xcl sparse window {k}"));
        }
        out.push((k, r.clone().into_raw() as usize, r.GetGPUVirtualAddress()));
        windows.push((base, r));
    }
    if out.is_empty() {
        return out;
    }
    let made = (|| -> windows::core::Result<State> {
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())?;
        let rb_queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let rb_done_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None)?;
        let mut slots = Vec::with_capacity(UPLOAD_SLOTS);
        for _ in 0..UPLOAD_SLOTS {
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
            list.Close()?;
            let _ = list.SetName(windows::core::w!("xcl sparse upload list"));
            let _ = alloc.SetName(windows::core::w!("xcl sparse upload allocator"));
            slots.push(UploadSlot { alloc, list, fence_val: 0 });
        }
        let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None)?;
        rb_list.Close()?;
        let _ = rb_list.SetName(windows::core::w!("xcl sparse readback list"));
        let _ = rb_alloc.SetName(windows::core::w!("xcl sparse readback allocator"));
        let _ = queue.SetName(windows::core::w!("xcl sparse queue"));
        let _ = rb_queue.SetName(windows::core::w!("xcl sparse readback queue"));
        Ok(State {
            dev: dev.clone(),
            queue,
            fence,
            value: 0,
            event,
            rb_queue,
            rb_fence,
            rb_value: 0,
            rb_done_fence,
            rb_done_value: 0,
            windows,
            pools: Vec::new(),
            pool_is_sys: Vec::new(),
            free: Vec::new(),
            retired: Vec::new(),
            mapped: FxHashMap::default(),
            pinned: FxHashMap::default(),
            parked: FxHashMap::default(),
            pending_unmaps: VecDeque::new(),
            pending_unmap_va: FxHashMap::default(),
            unmap_seq: 0,
            quarantined: VecDeque::new(),
            pending_quarantine: Vec::new(),
            queue_fences: FxHashMap::default(),
            buffers: FxHashMap::default(),
            to_zero: Vec::new(),
            staging: None,
            ring_write: 0,
            ring_submitted: 0,
            ring_free: 0,
            inflight: VecDeque::new(),
            zero_buf: None,
            slots,
            next_slot: 0,
            readback: None,
            rb_alloc,
            rb_list,
            rb_last_fence: 0,
            waited: FxHashMap::default(),
            mapping_queued: std::sync::atomic::AtomicBool::new(false),
            mapped_value: 0,
            runs: None,
            scratch_runs: Vec::new(),
            retry_pages: Vec::new(),
        })
    })();
    match made {
        Ok(s) => {
            crate::xlog_info!("D3D12.X: {} sparse window(s) for the title's own memory (tiled buffers on video memory)", out.len());
            // a state this replaces (the windows set up again) goes after the lock: the release of
            // its last buffers reaches `forget_buffer`, which takes the lock (#344)
            // (the queue first: `queue()` never answers with one older than the state's)
            let old_queue = QUEUE.lock().unwrap().replace(s.queue.clone());
            let old = STATE.lock().unwrap().replace(s);
            drop(old);
            drop(old_queue);
            for (a, b) in std::mem::take(&mut *EARLY.lock().unwrap()) {
                committed(a, b - a);
            }
            out
        }
        Err(e) => {
            crate::xlog_warn!("sparse windows: {e}");
            Vec::new()
        }
    }
}

impl State {
    #[inline]
    fn window_index(&self, a: usize) -> Option<usize> {
        self.windows.iter().position(|(b, _)| a >= *b && a < *b + crate::gpu::uma::WINDOW)
    }

    unsafe fn take_tile(&mut self) -> Option<((u32, u32), bool)> {
        self.flush_pending_quarantine();
        self.drain_quarantined();
        if self.free.is_empty() && self.flush_parked() {
            self.drain_quarantined();
        }
        if let Some(t) = self.free.pop() {
            return Some((t, true));
        }
        // a pool retired a moment ago (its last tile came free) is taken back before a new
        // heap is made: freeing and allocating a tile in turn made a new pool each time,
        // up to 18 GB of retired heaps in 2 s (#913). Its tiles are reused as freed tiles are
        // (zeroed), so this is no less safe than reusing a tile of a pool in use.
        let live_vram = self
            .pools
            .iter()
            .enumerate()
            .filter(|(i, p)| p.is_some() && !self.pool_is_sys.get(*i).copied().unwrap_or(false))
            .count();
        let want_sys = sparse_vram_cap_pools().is_some_and(|cap| live_vram >= cap);
        if let Some(pos) = self.retired.iter().rposition(|r| r.1 == want_sys) {
            let (heap, is_sys, _, _, _) = self.retired.swap_remove(pos);
            let h = self.add_pool(heap, is_sys);
            return Some(((h, 0), true));
        }
        if want_sys {
            let sys_desc = D3D12_HEAP_DESC {
                SizeInBytes: POOL_TILES as u64 * TILE as u64,
                Properties: D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_CUSTOM,
                    CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                    MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                    ..Default::default()
                },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
            };
            let mut heap: Option<ID3D12Heap> = None;
            if super::title_textures::own_work(|| self.dev.CreateHeap(&sys_desc, &mut heap)).is_ok() {
                if let Some(heap) = heap {
                    let h = self.add_pool(heap, true);
                    return Some(((h, 0), false));
                }
            }
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_TILES as u64 * TILE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        if let Err(e) = super::title_textures::own_work(|| self.dev.CreateHeap(&desc, &mut heap)) {
            crate::xlog_warn!("sparse windows: no video memory for more tiles ({e})");
            return None;
        }
        let heap = heap?;
        let h = self.add_pool(heap, false);
        Some(((h, 0), false))
    }

    /// `heap` becomes a pool (in a given-back pool's index if there is one); its tiles but the
    /// first, which the caller takes, go to the free list.
    fn add_pool(&mut self, heap: ID3D12Heap, is_sys: bool) -> u32 {
        let h = match self.pools.iter().position(|p| p.is_none()) {
            Some(i) => {
                self.pools[i] = Some(heap);
                if i < self.pool_is_sys.len() {
                    self.pool_is_sys[i] = is_sys;
                } else {
                    self.pool_is_sys.resize(i + 1, false);
                    self.pool_is_sys[i] = is_sys;
                }
                i as u32
            }
            None => {
                self.pools.push(Some(heap));
                self.pool_is_sys.push(is_sys);
                self.pools.len() as u32 - 1
            }
        };
        self.free.extend((1..POOL_TILES).rev().map(|t| (h, t)));
        h
    }

    /// Pools whose every tile is free again go (their video memory – and as much of the
    /// system's commit: a retail game's homes of physical pages left 0.75 GB of pools for
    /// 0.29 GB of tiles in use). Call after their tiles are unmapped and signalled.
    fn retire_free_pools(&mut self) {
        let now = std::time::Instant::now();
        let done = unsafe { self.fence.GetCompletedValue() };
        let rb_done = unsafe { self.rb_done_fence.GetCompletedValue() };
        self.retired.retain(|(_, _, v, rb_v, at)| !(done >= *v && rb_done >= *rb_v && now.duration_since(*at).as_secs() >= 2));
        if self.free.len() < POOL_TILES as usize {
            return;
        }
        let mut count = vec![0u32; self.pools.len()];
        for &(h, _) in &self.free {
            count[h as usize] += 1;
        }
        // (a flag a pool: a title's world load frees hundreds of pools' worth of tiles at once)
        let gone: Vec<bool> = (0..count.len()).map(|h| count[h] == POOL_TILES && self.pools[h].is_some()).collect();
        if !gone.contains(&true) {
            return;
        }
        self.free.retain(|&(h, _)| !gone[h as usize]);
        for h in (0..gone.len()).filter(|&h| gone[h]) {
            if let Some(heap) = self.pools[h].take() {
                let is_sys = self.pool_is_sys.get(h).copied().unwrap_or(false);
                self.retired.push((heap, is_sys, self.value, self.rb_done_value, now));
            }
        }
    }

    /// Map `tiles` (tile addresses) of `res` (starting at `res_base`) onto pool tiles,
    /// batched per pool heap with contiguous runs merged into single regions.
    unsafe fn map_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[(usize, (u32, u32))]) {
        if tiles.is_empty() {
            return;
        }
        self.mapping_queued.store(true, Ordering::Relaxed);
        let mut by_heap: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
        for &(a, (h, t)) in tiles {
            by_heap.entry(h).or_default().push((((a - res_base) / TILE) as u32, t));
        }
        for (h, mut pairs) in by_heap {
            pairs.sort_unstable_by_key(|&(x, _)| x);
            let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(pairs.len());
            let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(pairs.len());
            let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(pairs.len());
            let mut offsets: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut counts: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut i = 0;
            while i < pairs.len() {
                let (x0, t0) = pairs[i];
                let mut run = 1u32;
                while i + (run as usize) < pairs.len() {
                    let (nx, nt) = pairs[i + run as usize];
                    if nx == x0 + run && nt == t0 + run {
                        run += 1;
                    } else {
                        break;
                    }
                }
                coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
                regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
                flags.push(D3D12_TILE_RANGE_FLAG_NONE);
                offsets.push(t0);
                counts.push(run);
                i += run as usize;
            }
            let flight_t = std::time::Instant::now();
            self.queue.UpdateTileMappings(
                res,
                coords.len() as u32,
                Some(coords.as_ptr()),
                Some(regions.as_ptr()),
                self.pools[h as usize].as_ref().expect("a pool tile of a live pool"),
                flags.len() as u32,
                Some(flags.as_ptr()),
                Some(offsets.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
            super::flight::tile_map("sparse", windows::core::Interface::as_raw(&self.queue) as usize, (counts.iter().map(|&c| c as u64).sum::<u64>()) as u64, flight_t.elapsed());
        }
    }

    unsafe fn unmap_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[usize]) {
        if tiles.is_empty() {
            return;
        }
        self.mapping_queued.store(true, Ordering::Relaxed);
        let mut xs: Vec<u32> = tiles.iter().map(|&a| ((a - res_base) / TILE) as u32).collect();
        xs.sort_unstable();
        let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(xs.len());
        let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(xs.len());
        let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(xs.len());
        let mut counts: Vec<u32> = Vec::with_capacity(xs.len());
        let mut i = 0;
        while i < xs.len() {
            let x0 = xs[i];
            let mut run = 1u32;
            while i + (run as usize) < xs.len() && xs[i + run as usize] == x0 + run {
                run += 1;
            }
            coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
            regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
            flags.push(D3D12_TILE_RANGE_FLAG_NULL);
            counts.push(run);
            i += run as usize;
        }
        let flight_t = std::time::Instant::now();
        self.queue.UpdateTileMappings(
            res,
            coords.len() as u32,
            Some(coords.as_ptr()),
            Some(regions.as_ptr()),
            None,
            flags.len() as u32,
            Some(flags.as_ptr()),
            None,
            Some(counts.as_ptr()),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
        super::flight::tile_map("sparse", windows::core::Interface::as_raw(&self.queue) as usize, (counts.iter().map(|&c| c as u64).sum::<u64>()) as u64, flight_t.elapsed());
    }

    /// Bytes of the title's pages committed in the windows: the pages of the mapped tiles that
    /// are not holes. A tile the title decommitted whole is not mapped any more, though its pool
    /// tile may still be parked (memory accounting, #2688).
    fn committed_bytes(&self) -> usize {
        self.mapped.values().map(|e| PAGES_PER_TILE - e.holes.count_ones() as usize).sum::<usize>() * PAGE
    }

    /// tiles of `[a, a + n)` get video memory (mapped in their window); `initial_holes` supplies
    /// the initial 16-bit hole mask for each newly mapped 64 KiB tile.
    unsafe fn ensure_mapped(&mut self, a: usize, n: usize, mut initial_holes: impl FnMut(usize) -> u16) {
        if n == 0 {
            return;
        }
        let first = a & !(TILE - 1);
        let end = a.saturating_add(n);
        let mut new: Vec<(usize, (u32, u32))> = Vec::new();
        let mut unparked = false;
        let mut t = first;
        while t < end {
            if !self.mapped.contains_key(&t) && self.window_index(t).is_some() {
                // still mapped there: no new mapping, what it held is cleared
                if let Some(tile) = self.parked.remove(&t).or_else(|| self.pending_unmap_va.remove(&t).map(|(tile, _)| tile)) {
                    self.mapped.insert(t, TileEntry::with_holes(tile, initial_holes(t)));
                    self.runs = None;
                    self.to_zero.push((t, TILE));
                    unparked = true;
                } else {
                    let Some((tile, reused)) = self.take_tile() else { break };
                    self.mapped.insert(t, TileEntry::with_holes(tile, initial_holes(t)));
                    self.runs = None;
                    if reused {
                        self.to_zero.push((t, TILE));
                    }
                    new.push((t, tile));
                }
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        if new.is_empty() {
            if unparked {
                MAPPED_GEN.fetch_add(1, Ordering::Release);
            }
            return;
        }
        MAPPED_GEN.fetch_add(1, Ordering::Release);
        // per window
        let mut by_window: FxHashMap<usize, Vec<(usize, (u32, u32))>> = FxHashMap::default();
        for x in new {
            if let Some(wi) = self.window_index(x.0) {
                by_window.entry(wi).or_default().push(x);
            }
        }
        for (wi, tiles) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            self.map_tiles(&r, b, &tiles);
        }
        self.signal();
    }

    #[inline]
    unsafe fn signal(&mut self) {
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
        if self.mapping_queued.swap(false, Ordering::Relaxed) {
            self.mapped_value = self.value;
        }
        super::flight::signaled("sparse", windows::core::Interface::as_raw(&self.queue) as usize, &self.fence, self.value);
    }

    #[cfg(test)]
    unsafe fn wait_for(&self, target: u64) -> bool {
        self.wait_for_upload(target, QUEUE_WAIT_TIMEOUT)
    }

    unsafe fn wait_for_upload(&self, target: u64, timeout: std::time::Duration) -> bool {
        if wait_fence_until(&self.fence, self.event, target, timeout) {
            return true;
        }
        crate::xlog_warn!("sparse: the queue has not reached fence value {target} after {} ms (at {})", timeout.as_millis(), self.fence.GetCompletedValue());
        super::real::stop_if_device_removed();
        false
    }

    #[inline]
    unsafe fn order_rb_after_uploads(&self) {
        if self.value > 0 && self.fence.GetCompletedValue() < self.value {
            let _ = self.rb_queue.Wait(&self.fence, self.value);
        }
    }

    #[inline]
    unsafe fn signal_rb(&mut self) {
        self.rb_done_value += 1;
        let _ = self.rb_queue.Signal(&self.rb_done_fence, self.rb_done_value);
        super::flight::signaled("sparse readback", windows::core::Interface::as_raw(&self.rb_queue) as usize, &self.rb_done_fence, self.rb_done_value);
    }

    /// Whether `rb_done_fence` reached `target` within 5 s: if not, the staging bytes are not the
    /// readback yet and must not be copied out.
    #[inline]
    unsafe fn wait_for_rb(&self, target: u64) -> bool {
        if wait_fence_until(&self.rb_done_fence, self.event, target, QUEUE_WAIT_TIMEOUT) {
            return true;
        }
        crate::xlog_warn!(
            "sparse: the readback queue has not reached fence value {target} after 5 s (at {})",
            self.rb_done_fence.GetCompletedValue()
        );
        super::real::stop_if_device_removed();
        false
    }

    #[inline]
    unsafe fn reclaim_ring(&mut self) {
        if self.inflight.is_empty() {
            return;
        }
        let completed = self.fence.GetCompletedValue();
        while let Some(&(fv, end_pos)) = self.inflight.front() {
            if fv <= completed {
                self.ring_free = end_pos;
                self.inflight.pop_front();
            } else {
                break;
            }
        }
    }

    /// Acquire the next upload command slot, waiting only if that slot's previous submission is
    /// still executing on the GPU, then reset its allocator and list.
    unsafe fn begin_slot(&mut self, timeout: std::time::Duration) -> Option<usize> {
        let idx = self.next_slot;
        let fv = self.slots.get(idx)?.fence_val;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        if fv > 0 {
            if !self.wait_for_upload(fv, timeout) {
                return None;
            }
            self.reclaim_ring();
        }
        let slot = &self.slots[idx];
        slot.alloc.Reset().ok()?;
        slot.list.Reset(&slot.alloc, None).ok()?;
        Some(idx)
    }

    /// Close and submit upload slot `idx` asynchronously (NO CPU wait — callers order GPU work
    /// via `queue.Wait(&self.fence, self.value)`).
    unsafe fn submit_slot(&mut self, idx: usize) -> bool {
        let list = &self.slots[idx].list;
        let Ok(l) = list.Close().and_then(|()| list.cast::<ID3D12CommandList>()) else {
            self.ring_write = self.ring_submitted;
            return false;
        };
        super::title_textures::own_work(|| {
            self.queue.ExecuteCommandLists(&[Some(l)]);
        });
        self.signal();
        self.slots[idx].fence_val = self.value;
        if self.ring_write > self.ring_submitted {
            self.inflight.push_back((self.value, self.ring_write));
            self.ring_submitted = self.ring_write;
        }
        true
    }

    unsafe fn staging_ptr(&mut self) -> Option<*mut u8> {
        if self.staging.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            if super::real::dred_on() {
                super::real::name_resource(r.as_raw(), "xcl sparse staging");
            }
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            self.staging = Some((r, p as *mut u8));
        }
        self.staging.as_ref().map(|&(_, p)| p)
    }

    unsafe fn ensure_zero_buf(&mut self) -> bool {
        if self.zero_buf.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(ZERO_BUF as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            if self
                .dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .is_err()
            {
                return false;
            }
            let Some(r) = r else { return false };
            if super::real::dred_on() {
                super::real::name_resource(r.as_raw(), "xcl sparse zero");
            }
            let mut p = std::ptr::null_mut();
            if r.Map(0, None, Some(&mut p)).is_err() {
                return false;
            }
            std::ptr::write_bytes(p as *mut u8, 0, ZERO_BUF);
            r.Unmap(0, None);
            self.zero_buf = Some(r);
        }
        true
    }

    unsafe fn readback_buf(&mut self) -> Option<(ID3D12Resource, *mut u8)> {
        if self.readback.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(READBACK_STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            if super::real::dred_on() {
                super::real::name_resource(r.as_raw(), "xcl sparse readback");
            }
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            self.readback = Some((r, p as *mut u8));
        }
        self.readback.clone()
    }

    /// Clear `sent` hashes for every page overlapping `[a, a + n)`.
    fn clear_sent_range(&mut self, a: usize, n: usize) {
        let mut pg = a & !(PAGE - 1);
        let end = a + n;
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    entry.sent[(p - t) / PAGE] = None;
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
    }

    /// Make the scanned hashes of unsent chunks differ on the next scan. Leave the hashes of
    /// chunks already sent untouched, even when they share a CPU page with a retry.
    fn invalidate_sent_range(&mut self, a: usize, n: usize) {
        let end = a + n;
        let mut at = a;
        while at < end {
            let page = at & !(PAGE - 1);
            let tile = page & !(TILE - 1);
            let page_end = (page + PAGE).min(end);
            if let Some(sent) = self.mapped.get_mut(&tile).and_then(|entry| entry.sent[(page - tile) / PAGE].as_deref_mut()) {
                for hash in &mut sent[(at - page) / CHUNK..(page_end - page).div_ceil(CHUNK)] {
                    *hash ^= 1;
                }
            }
            at = page_end;
        }
    }

    fn retry_dirty_from(&mut self, runs: &[(usize, usize)], start: (usize, usize)) {
        for (a, n) in unsent_runs(runs, start) {
            self.invalidate_sent_range(a, n);
            self.retry_pages.extend((a & !(PAGE - 1)..a + n).step_by(PAGE));
        }
    }

    unsafe fn scan_retry_pages(&mut self, out: &mut Vec<(usize, usize)>) {
        if self.retry_pages.is_empty() {
            return;
        }
        self.retry_pages.sort_unstable();
        self.retry_pages.dedup();
        scan_dirty_pages_into(&mut self.mapped, &self.retry_pages, PAGE, scan_threads(), out);
        self.retry_pages.clear();
    }

    /// Update `sent` hashes for all 64-byte chunks overlapping `[va, va + len)` so the CPU
    /// copy-back we just wrote to `va` is not re-uploaded by the next `before_execute`.
    unsafe fn sync_sent_range(&mut self, va: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut pg = va & !(PAGE - 1);
        let end = va + len;
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    let idx = (p - t) / PAGE;
                    if entry.holes & (1u16 << idx) == 0 {
                        let bytes = std::slice::from_raw_parts(p as *const u8, PAGE);
                        let sent = entry.sent[idx].get_or_insert_with(|| Box::new(ZERO_HASHES));
                        let lo = va.saturating_sub(p) / CHUNK;
                        let hi = (end.min(p + PAGE) - p).div_ceil(CHUNK).min(CHUNKS);
                        for k in lo..hi {
                            sent[k] = chunk_hash(&bytes[k * CHUNK..(k + 1) * CHUNK]);
                        }
                    }
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
    }

    /// Copy `ranges` (address, bytes) from sparse VRAM back into the CPU view and refresh
    /// `sent` hashes. Caller has already ordered `self.queue` after any GPU writes if needed.
    unsafe fn readback_ranges(&mut self, ranges: &[(usize, usize)]) {
        let Some((rb, p)) = self.readback_buf() else { return };
        // (va, rb_off, len, pre-wait 64-byte chunk hashes of committed pages)
        let mut pending: Vec<(usize, usize, usize, Vec<Option<u64>>)> = Vec::new();
        let mut used = 0usize;
        let mut open = false;
        let flush_and_copy = |s: &mut State, open: &mut bool, pending: &mut Vec<(usize, usize, usize, Vec<Option<u64>>)>, used: &mut usize, p: *mut u8| -> bool {
            if *open {
                *open = false;
                s.order_rb_after_uploads();
                let Ok(l) = s.rb_list.Close().and_then(|()| s.rb_list.cast::<ID3D12CommandList>()) else { return false };
                super::title_textures::own_work(|| {
                    s.rb_queue.ExecuteCommandLists(&[Some(l)]);
                });
                s.signal_rb();
                s.rb_last_fence = s.rb_done_value;
                if !s.wait_for_rb(s.rb_done_value) {
                    return false;
                }
                s.reclaim_ring();
            }
            for (dst, off, len, snap) in pending.iter() {
                let (dst, off, len) = (*dst, *off, *len);
                let snap_base = dst & !(CHUNK - 1);
                // Copy only committed (non-hole) 64-byte chunks whose host memory did not change
                // during the GPU readback wait, so concurrent CPU writes are never clobbered.
                let mut at = dst;
                let mut src_off = off;
                let end = dst + len;
                while at < end {
                    let pg = at & !(PAGE - 1);
                    let t = pg & !(TILE - 1);
                    let idx = (pg - t) / PAGE;
                    let page_end = (pg + PAGE).min(end);
                    if let Some(entry) = s.mapped.get_mut(&t).filter(|e| e.holes & (1u16 << idx) == 0) {
                        while at < page_end {
                            let chunk_va = at & !(CHUNK - 1);
                            let chunk_end = (chunk_va + CHUNK).min(page_end);
                            let step = chunk_end - at;
                            let snap_i = (chunk_va - snap_base) / CHUNK;
                            let unchanged = snap.get(snap_i).copied().flatten().is_some_and(|expected| {
                                chunk_hash(std::slice::from_raw_parts(chunk_va as *const u8, CHUNK)) == expected
                            });
                            if unchanged {
                                std::ptr::copy_nonoverlapping(p.add(src_off), at as *mut u8, step);
                                let new_h = chunk_hash(std::slice::from_raw_parts(chunk_va as *const u8, CHUNK));
                                let k = (chunk_va - pg) / CHUNK;
                                if let Some(sent) = entry.sent[idx].as_deref_mut() {
                                    sent[k] = new_h;
                                } else if new_h != ZERO_HASH {
                                    let sent = entry.sent[idx].insert(Box::new(ZERO_HASHES));
                                    sent[k] = new_h;
                                }
                            }
                            at += step;
                            src_off += step;
                        }
                    } else {
                        let step = page_end - at;
                        at += step;
                        src_off += step;
                    }
                }
            }
            pending.clear();
            *used = 0;
            true
        };
        for &(a, n) in ranges {
            if n == 0 {
                continue;
            }
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let tile = cur & !(TILE - 1);
                let Some(wi) = self.window_index(cur) else { break };
                let b = self.windows[wi].0;
                let to_tile_end = (tile + TILE) - cur;
                if !self.mapped.contains_key(&tile) {
                    done += (n - done).min(to_tile_end);
                    continue;
                }
                if used == READBACK_STAGING && !flush_and_copy(self, &mut open, &mut pending, &mut used, p) {
                    return;
                }
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - done).min(READBACK_STAGING - used).min(to_tile_end).min(to_win_end);
                if !open {
                    if !self.wait_for_rb(self.rb_last_fence) || self.rb_alloc.Reset().is_err() || self.rb_list.Reset(&self.rb_alloc, None).is_err() {
                        return;
                    }
                    open = true;
                }
                let snap_base = cur & !(CHUNK - 1);
                let snap_end = (cur + chunk).saturating_add(CHUNK - 1) & !(CHUNK - 1);
                let mut snap = Vec::with_capacity((snap_end - snap_base) / CHUNK);
                let mut c = snap_base;
                while c < snap_end {
                    let pg = c & !(PAGE - 1);
                    let t = pg & !(TILE - 1);
                    let idx = (pg - t) / PAGE;
                    if self.mapped.get(&t).is_some_and(|e| e.holes & (1u16 << idx) == 0) {
                        snap.push(Some(chunk_hash(std::slice::from_raw_parts(c as *const u8, CHUNK))));
                    } else {
                        snap.push(None);
                    }
                    c += CHUNK;
                }
                self.rb_list.CopyBufferRegion(&rb, used as u64, &self.windows[wi].1, (cur - b) as u64, chunk as u64);
                pending.push((cur, used, chunk, snap));
                used += chunk;
                done += chunk;
            }
        }
        if open {
            flush_and_copy(self, &mut open, &mut pending, &mut used, p);
        }
    }

    /// Asynchronously zero `runs` (address, bytes) in sparse VRAM using the pre-zeroed `zero_buf`
    /// (zero staging ring bytes consumed, zero CPU stall).
    unsafe fn upload_zeros(&mut self, runs: &[(usize, usize)], timeout: std::time::Duration) -> bool {
        if runs.is_empty() {
            return true;
        }
        if !self.ensure_zero_buf() {
            return false;
        }
        let Some(slot_idx) = self.begin_slot(timeout) else { return false };
        let mut last_wi: Option<usize> = None;
        let zero_buf = self.zero_buf.as_ref().unwrap();
        let list = &self.slots[slot_idx].list;
        for &(a, n) in runs {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => break,
                    },
                };
                let b = self.windows[wi].0;
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - done).min(ZERO_BUF).min(to_win_end);
                list.CopyBufferRegion(&self.windows[wi].1, (cur - b) as u64, zero_buf, 0, chunk as u64);
                done += chunk;
            }
        }
        self.submit_slot(slot_idx)
    }

    /// Asynchronously copy `runs` (address, bytes) from the CPU view into sparse VRAM tiles via
    /// the ring staging buffer. Preserve any unsent chunks when a submission stalls.
    unsafe fn upload_dirty_with_timeout(&mut self, runs: &[(usize, usize)], timeout: std::time::Duration) -> bool {
        if let Err(start) = self.upload_runs(runs, timeout, &|at, p, n| {
            std::ptr::copy_nonoverlapping(at as *const u8, p, n);
            true
        }) {
            self.retry_dirty_from(runs, start);
            return false;
        }
        true
    }

    /// Copies `runs` (address in a window, bytes) into video memory; `read(address, to, bytes)`
    /// gives their bytes (false: the run's piece is left as it is).
    unsafe fn upload_runs(&mut self, runs: &[(usize, usize)], timeout: std::time::Duration, read: &dyn Fn(usize, *mut u8, usize) -> bool) -> Result<(), (usize, usize)> {
        if runs.is_empty() {
            return Ok(());
        }
        let p = self.staging_ptr().ok_or((0, 0))?;
        // A failed Close invalidates this whole batch, including earlier runs recorded in it.
        let mut open_slot: Option<(usize, (usize, usize))> = None;
        let mut last_wi: Option<usize> = None;
        for (index, &(a, n)) in runs.iter().enumerate() {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => {
                            if let Some((slot, start)) = open_slot.take() {
                                if !self.submit_slot(slot) {
                                    return Err(start);
                                }
                            }
                            return Err((index, done));
                        }
                    },
                };
                let b = self.windows[wi].0;
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let (aligned, chunk) = plan_ring_chunk(self.ring_write, (n - done).min(to_win_end), STAGING);
                if (aligned + chunk as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                    self.reclaim_ring();
                    if (aligned + chunk as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                        if let Some((slot_idx, start)) = open_slot.take() {
                            if !self.submit_slot(slot_idx) {
                                return Err(start);
                            }
                            self.reclaim_ring();
                        }
                        while (aligned + chunk as u64).saturating_sub(self.ring_free) > STAGING as u64 {
                            let &(fv, _) = self.inflight.front().ok_or((index, done))?;
                            if !self.wait_for_upload(fv, timeout) {
                                return Err((index, done));
                            }
                            self.reclaim_ring();
                        }
                    }
                }
                let slot_idx = match open_slot {
                    Some((idx, _)) => idx,
                    None => {
                        let idx = self.begin_slot(timeout).ok_or((index, done))?;
                        open_slot = Some((idx, (index, done)));
                        idx
                    }
                };
                let dst_off = (aligned as usize) % STAGING;
                if !read(cur, p.add(dst_off), chunk) {
                    done += chunk;
                    continue;
                }
                let staging_res = &self.staging.as_ref().unwrap().0;
                self.slots[slot_idx].list.CopyBufferRegion(&self.windows[wi].1, (cur - b) as u64, staging_res, dst_off as u64, chunk as u64);
                self.ring_write = aligned + chunk as u64;
                done += chunk;
            }
        }
        if let Some((slot_idx, start)) = open_slot {
            if !self.submit_slot(slot_idx) {
                return Err(start);
            }
        }
        Ok(())
    }
}

fn unsent_runs(runs: &[(usize, usize)], (index, done): (usize, usize)) -> impl Iterator<Item = (usize, usize)> + '_ {
    runs.iter().skip(index).enumerate().filter_map(move |(i, &(a, n))| {
        let sent = if i == 0 { done } else { 0 };
        (sent < n).then(|| (a + sent, n - sent))
    })
}

/// Allocate a zeroed, write-watched block in the sparse VRAM windows (never straddling a
/// 4 GiB window boundary when `size <= 4 GiB`). Commits the CPU pages and maps VRAM tiles
/// (or queues them until the D3D12 device exists). Returns `(address, rounded_size)`.
pub unsafe fn alloc(size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc(size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// `alloc` for the title's own memory: its CPU pages are committed when the CPU first touches
/// them (`commit_title`).
pub unsafe fn alloc_title(size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc(size, align)?;
    if commit_title(addr, rounded, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    Some((addr, rounded))
}

/// The title commits `[a, a + n)` in a sparse window with the protection `protect`: as
/// `committed`, but the CPU's copy of a whole 64 KiB page is made when the CPU first touches it
/// (`kernelx::lazy`). Returns the address, or null as VirtualAlloc does.
///
/// A title page in a sparse window was committed on the CPU as well as mapped to a tile of
/// video memory. A retail title's render targets, depth buffers and GPU-written buffers there
/// took system memory and commit the CPU never touched, and each write-watch look walked them
/// (2-4 GB of tiles, ~5 ms a look, ~90 looks a second). Such a page's tile is now video memory
/// only (all holes: the look leaves it out and no upload writes it) until the CPU's first
/// access commits the page (zeros, as a new commit holds); from then on its writes are watched
/// and uploaded as before. Pages at the ends that do not fill a 64 KiB page, pages already
/// committed, another protection than read-write, and commits before the device exists are
/// committed now, as before.
pub unsafe fn commit_title(a: usize, n: usize, protect: u32) -> *mut c_void {
    let eager = |a: usize, n: usize| -> bool {
        if n == 0 {
            return true;
        }
        if VirtualAlloc(a as *const c_void, n, MEM_COMMIT, protect).is_null() {
            return false;
        }
        committed(a, n);
        true
    };
    let (lo, hi) = ((a + TILE - 1) & !(TILE - 1), a.saturating_add(n) & !(TILE - 1));
    if protect != PAGE_READWRITE || hi <= lo || lock_state().is_none() {
        return if eager(a, n) { a as *mut c_void } else { std::ptr::null_mut() };
    }
    if !eager(a, lo - a) || !eager(hi, a + n - hi) {
        return std::ptr::null_mut();
    }
    // the pages committed already (the title commits a range again) stay as they are
    let mut lazy: Vec<(usize, usize)> = Vec::new();
    let mut at = lo;
    while at < hi {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 {
            return std::ptr::null_mut();
        }
        let end = (m.BaseAddress as usize).saturating_add(m.RegionSize).min(hi).max(at + PAGE);
        if m.State == MEM_COMMIT {
            committed(at, end - at);
        } else {
            // whole 64 KiB pages lazy; a part of one at either end now
            let s = ((at + TILE - 1) & !(TILE - 1)).min(end);
            let e = (end & !(TILE - 1)).max(s);
            if !eager(at, s - at) {
                return std::ptr::null_mut();
            }
            if e > s {
                lazy.push((s, e - s));
            }
            if !eager(e, end - e) {
                return std::ptr::null_mut();
            }
        }
        at = end;
    }
    {
        let mut g = lock_state();
        let Some(s) = g.as_mut() else { return std::ptr::null_mut() };
        for &(p, len) in &lazy {
            s.ensure_mapped(p, len, |_| 0xFFFF);
        }
    }
    // (marked after their tiles are mapped: a first touch finds them, `drain_lazy_commits`)
    for &(p, len) in &lazy {
        for page in (p..p + len).step_by(TILE) {
            crate::hle::kernelx::lazy::mark(page);
        }
        LAZY_BYTES.fetch_add(len as u64, Ordering::Relaxed);
    }
    a as *mut c_void
}

/// Bytes of the title's commits in the sparse windows that `commit_title` left lazy so far
/// (some of them committed since by a first touch, some decommitted).
pub fn lazy_title_bytes() -> u64 {
    LAZY_BYTES.load(Ordering::Relaxed)
}

static LAZY_BYTES: AtomicU64 = AtomicU64::new(0);

/// Allocate a zeroed, write-watched block strictly inside sparse window `window` (e.g.
/// window 0 for 32-bit-addressed descriptor heaps). Returns `(address, rounded_size)`.
pub unsafe fn alloc_in(window: usize, size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc_in(window, size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// Release a sparse allocation from `alloc` / `alloc_in` (false: not a sparse allocation).
pub unsafe fn free(p: usize) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    if !u.is_sparse(p) {
        return false;
    }
    // decommitted before the address goes back: once released, another thread may get it,
    // commit it and write there
    let Some((start, len)) = u.sparse_reservation(p).filter(|&(start, _)| start == p) else { return false };
    decommitted(start, len);
    VirtualFree(p as *mut c_void, len, MEM_DECOMMIT);
    u.sparse_release(p).is_some()
}

/// 64 KiB sparse pages lazily committed by `kernelx::lazy` when the title first CPU-touches a
/// `alloc_vram_only` buffer (queued without locking `STATE` so the VEH can never deadlock).
static LAZY_COMMITS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static HAS_LAZY_COMMITS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Record that `kernelx::lazy` just CPU-committed 64 KiB page `page`. Safe inside the VEH:
/// never acquires `STATE`.
pub(crate) fn note_lazy_commit(page: usize) {
    if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(page)) {
        LAZY_COMMITS.lock().unwrap_or_else(|e| e.into_inner()).push(page);
        HAS_LAZY_COMMITS.store(true, Ordering::Release);
    }
}

/// Allocate a VRAM-only sparse block (zero upfront CPU commit, `holes = 0xFFFF` so CPU
/// write-watch never overwrites GPU UAV writes) for GPU-writable buffers placed outside the UMA
/// windows. Each 64 KiB page is marked with `kernelx::lazy` so if the title later CPU-writes or
/// reads it (via `GetGPUVirtualAddress()` or `Map()`, #2582), the page commits on demand and
/// subsequent CPU writes upload cleanly to sparse VRAM.
pub unsafe fn alloc_vram_only(size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc(size, align.max(TILE))?;
    let ok = {
        let mut g = lock_state();
        if let Some(s) = g.as_mut() {
            s.ensure_mapped(addr, rounded, |_| 0xFFFF);
            let last_tile = (addr + rounded - 1) & !(TILE - 1);
            if s.mapped.contains_key(&addr) && s.mapped.contains_key(&last_tile) {
                true
            } else {
                s.decommit_range(addr, rounded);
                false
            }
        } else {
            false
        }
    };
    if !ok {
        u.sparse_release(addr);
        return None;
    }
    for page in (addr..addr + rounded).step_by(TILE) {
        crate::hle::kernelx::lazy::mark(page);
    }
    Some((addr, rounded))
}

/// Release a VRAM-only sparse block from `alloc_vram_only` (unmarks lazy pages, defers tile
/// unmap behind GPU fences, and decommits any pages that were lazily committed by CPU access).
pub unsafe fn free_vram_only(p: usize) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    if !u.is_sparse(p) {
        return false;
    }
    let Some((start, len)) = u.sparse_reservation(p).filter(|&(start, _)| start == p) else { return false };
    for page in (start..start + len).step_by(TILE) {
        crate::hle::kernelx::lazy::unmark(page);
    }
    decommitted(start, len);
    VirtualFree(start as *mut c_void, len, MEM_DECOMMIT);
    u.sparse_release(p).is_some()
}

/// After `queue.ExecuteCommandLists`: copy GPU-written sparse ranges (e.g. `Write32/64`,
/// `FillMemoryWith32BitValueX`, `CopyMemoryToMemoryX`) back to their CPU view so CPU reads
/// and spin-waits see the GPU writes without clobbering `sent` hashes.
/// The queue tile mappings go out on, once there is one. Answered without `STATE` (`QUEUE`):
/// a caller may be inside a D3D12 call that a holder of `STATE` waits for.
pub fn queue() -> Option<ID3D12CommandQueue> {
    QUEUE.lock().unwrap().clone()
}

pub unsafe fn readback_after_execute(queue: &ID3D12CommandQueue, ranges: &[(usize, usize)]) {
    if ranges.is_empty() {
        return;
    }
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return };
    s.drain_lazy_commits();
    let mut filtered: Vec<(usize, usize)> = ranges.iter().copied().filter(|&(a, n)| n > 0 && s.window_index(a).is_some()).collect();
    if filtered.is_empty() {
        return;
    }
    filtered.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(filtered.len());
    for (a, n) in filtered {
        match merged.last_mut() {
            Some(last) if a <= last.0.saturating_add(last.1) => {
                let end = (last.0 + last.1).max(a + n);
                last.1 = end - last.0;
            }
            _ => merged.push((a, n)),
        }
    }
    if let Some((f, v)) = readback_after(&s.rb_queue, queue) {
        super::list_x::note_internal_wait(s.rb_queue.as_raw() as usize, "sparse readback behind a title queue", &f, v);
    }
    s.readback_ranges(&merged);
}

/// The sparse readback queue `rb_queue` waits until `q` has done what it was given so far, on a
/// fence of `q`'s own: (fence, value), None if `q` could not signal it.
///
/// One fence for all the title's queues, each signalling the next value, went back down: a queue
/// signals 11 and is done, another one still busy signals 10 after it – and the readback queue's
/// wait for 11 never ended. A retail title's frames stopped behind it (every title queue idle,
/// the readback queue stuck with work pending until the device was removed). A queue runs its
/// work in order, so the values of its own fence only go up.
unsafe fn readback_after(rb_queue: &ID3D12CommandQueue, q: &ID3D12CommandQueue) -> Option<(ID3D12Fence, u64)> {
    static FENCES: Mutex<Option<HashMap<usize, (ID3D12Fence, u64)>>> = Mutex::new(None);
    let key = q.as_raw() as usize;
    let mut g = FENCES.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if !m.contains_key(&key) {
        let mut dev: Option<ID3D12Device> = None;
        q.GetDevice(&mut dev).ok()?;
        let f: ID3D12Fence = dev?.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        m.insert(key, (f, 0));
    }
    let e = m.get_mut(&key)?;
    e.1 += 1;
    q.Signal(&e.0, e.1).ok()?;
    let _ = rb_queue.Wait(&e.0, e.1);
    Some((e.0.clone(), e.1))
}

/// Pooled per-`Map` synchronous readback resources so concurrent `ID3D12Resource::Map` readbacks
/// record, submit, and GPU-wait OUTSIDE the global `STATE` lock (avoiding `STATE` convoy stalls).
struct ReadbackJob {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    buf: ID3D12Resource,
    event: HANDLE,
    mapped: usize,
    cap: usize,
    dev_key: usize,
}

unsafe impl Send for ReadbackJob {}

impl Drop for ReadbackJob {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            unsafe { let _ = windows::Win32::Foundation::CloseHandle(self.event); }
        }
    }
}

const MAX_RB_JOBS: usize = 16;
const MIN_RB_JOB_BYTES: usize = 256 << 10;
static FREE_RB_JOBS: Mutex<Vec<ReadbackJob>> = Mutex::new(Vec::new());

/// A job whose readback was still queued on the GPU when `readback_now` gave up waiting: its
/// buffer, allocator and list stay alive until `fence` reaches `value`. Dropped earlier, the copy
/// still runs later and writes into freed memory (under RADV: a GPU page fault, CPG write).
struct InFlightRbJob {
    job: ReadbackJob,
    fence: ID3D12Fence,
    value: u64,
}

// SAFETY: ID3D12Fence is a free-threaded COM object; `ReadbackJob` is `Send`.
unsafe impl Send for InFlightRbJob {}

static IN_FLIGHT_RB_JOBS: Mutex<Vec<InFlightRbJob>> = Mutex::new(Vec::new());

/// Keep `job` until its GPU readback (`fence` >= `value`) is done.
fn keep_in_flight_readback_job(job: ReadbackJob, fence: ID3D12Fence, value: u64) {
    IN_FLIGHT_RB_JOBS.lock().unwrap().push(InFlightRbJob { job, fence, value });
}

/// Jobs kept by `keep_in_flight_readback_job` whose readback has finished go back to the pool.
unsafe fn reclaim_in_flight_readback_jobs() {
    let done: Vec<ReadbackJob> = {
        let mut v = IN_FLIGHT_RB_JOBS.lock().unwrap();
        if v.is_empty() {
            return;
        }
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut *v).into_iter().partition(|j| j.fence.GetCompletedValue() >= j.value);
        *v = keep;
        done.into_iter().map(|j| j.job).collect()
    };
    for job in done {
        return_readback_job(job);
    }
}

unsafe fn take_readback_job(dev: &ID3D12Device, need: usize) -> Option<ReadbackJob> {
    let dev_key = dev.as_raw() as usize;
    let want = need.max(MIN_RB_JOB_BYTES).min(READBACK_STAGING);
    reclaim_in_flight_readback_jobs();
    {
        let mut free = FREE_RB_JOBS.lock().unwrap();
        if let Some(i) = free.iter().position(|j| j.dev_key == dev_key && j.cap >= want) {
            return Some(free.swap_remove(i));
        }
    }
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).ok()?;
    let _ = list.Close();
    let mut d = buffer_desc(want as u64);
    d.Flags = D3D12_RESOURCE_FLAG_NONE;
    let mut r: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &d,
        D3D12_RESOURCE_STATE_COPY_DEST,
        None,
        &mut r,
    )
    .ok()?;
    let buf = r?;
    let mut p = std::ptr::null_mut();
    buf.Map(0, None, Some(&mut p)).ok()?;
    if p.is_null() {
        return None;
    }
    let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).ok()?;
    Some(ReadbackJob { alloc, list, buf, event, mapped: p as usize, cap: want, dev_key })
}

fn return_readback_job(job: ReadbackJob) {
    let mut free = FREE_RB_JOBS.lock().unwrap();
    if free.len() < MAX_RB_JOBS {
        free.push(job);
    }
}

/// Apply GPU readback bytes for `[dst, dst + len)` into host memory, comparing each 64-byte
/// chunk's current host bytes against its Phase-1 `snap_hashes` entry so any chunk modified by
/// the CPU during the GPU wait is NOT clobbered (and its `sent` hash remains stale so the next
/// `before_execute` uploads the CPU's write). Updates `snap_hashes` for copied chunks so a
/// multi-batch readback splitting a 64-byte chunk across `job.cap` preserves continuity.
unsafe fn apply_readback_preserving_cpu_writes(
    dst: usize,
    src: *const u8,
    len: usize,
    snap_base: usize,
    snap_hashes: &mut [u64],
    mapped: &mut FxHashMap<usize, TileEntry>,
) {
    let mut at = dst;
    let mut src_off = 0usize;
    let end = dst + len;
    while at < end {
        let pg = at & !(PAGE - 1);
        let t = pg & !(TILE - 1);
        let page_idx = (pg - t) / PAGE;
        let page_end = (pg + PAGE).min(end);
        if let Some(entry) = mapped.get_mut(&t).filter(|e| e.holes & (1u16 << page_idx) == 0) {
            while at < page_end {
                let chunk_va = at & !(CHUNK - 1);
                let chunk_end = (chunk_va + CHUNK).min(page_end);
                let step = chunk_end - at;
                let snap_i = (chunk_va - snap_base) / CHUNK;
                if let Some(expected) = snap_hashes.get_mut(snap_i) {
                    let cur_slice = std::slice::from_raw_parts(chunk_va as *const u8, CHUNK);
                    if chunk_hash(cur_slice) == *expected {
                        std::ptr::copy_nonoverlapping(src.add(src_off), at as *mut u8, step);
                        let new_h = chunk_hash(std::slice::from_raw_parts(chunk_va as *const u8, CHUNK));
                        *expected = new_h;
                        let k = (chunk_va - pg) / CHUNK;
                        if let Some(sent) = entry.sent[page_idx].as_deref_mut() {
                            sent[k] = new_h;
                        } else if new_h != ZERO_HASH {
                            let sent = entry.sent[page_idx].insert(Box::new(ZERO_HASHES));
                            sent[k] = new_h;
                        }
                    }
                }
                at += step;
                src_off += step;
            }
        } else {
            let step = page_end - at;
            at += step;
            src_off += step;
        }
    }
}

/// Synchronize `[va, va + len)` in a sparse window for a CPU read (`ID3D12Resource::Map` with
/// a non-empty read range, or `Map(0, NULL)` on a sparse readback resource): flushes pending
/// tile clears and unsubmitted CPU-changed chunks first so CPU writes are preserved, orders
/// `s.queue` behind active title queues via `rb_fence` before copying, and copies the range
/// back from VRAM into CPU memory using a per-call pooled `ReadbackJob` whose GPU fence wait
/// runs OUTSIDE the global `STATE` lock without clobbering concurrent CPU writes.
pub unsafe fn readback_now(va: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = va.saturating_add(len);
    let snap_base = va & !(CHUNK - 1);
    let snap_end = end.saturating_add(CHUNK - 1) & !(CHUNK - 1);
    let snap_count = (snap_end - snap_base) / CHUNK;

    // 1. Flush pending clears & dirty CPU chunks under `STATE`, snapshot Phase-1 64-byte chunk
    //    hashes, collect mapped spans, then release `STATE` before recording/waiting so other
    //    `Map` threads and `before_execute` never stall.
    let (dev, spans, mut snap_hashes): (ID3D12Device, Vec<(usize, ID3D12Resource, u64, usize)>, Vec<u64>) = {
        let mut g = lock_state();
        let Some(s) = g.as_mut() else { return };
        if s.window_index(va).is_none() {
            return;
        }
        s.drain_lazy_commits();
        let zeroed = std::mem::take(&mut s.to_zero);
        if !zeroed.is_empty() {
            for &(a, n) in &zeroed {
                s.clear_sent_range(a, n);
            }
            if !s.upload_zeros(&zeroed, QUEUE_WAIT_TIMEOUT) {
                s.to_zero = zeroed;
                return;
            }
        }
        // Preserve any unsubmitted CPU writes on pages overlapping `[va, va + len)` and snapshot
        // each 64-byte chunk's Phase-1 hash so Phase 3 never clobbers concurrent CPU writes.
        let mut dirty_runs: Vec<(usize, usize)> = Vec::new();
        let mut snap_hashes = vec![ZERO_HASH; snap_count];
        let mut pg = va & !(PAGE - 1);
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = s.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    let idx = (p - t) / PAGE;
                    if entry.holes & (1u16 << idx) == 0 {
                        let bytes = std::slice::from_raw_parts(p as *const u8, PAGE);
                        scan_page(bytes, p, &mut entry.sent[idx], &mut dirty_runs);
                        if let Some(sent) = entry.sent[idx].as_deref() {
                            let lo = va.saturating_sub(p) / CHUNK;
                            let hi = (end.min(p + PAGE) - p).div_ceil(CHUNK).min(CHUNKS);
                            for k in lo..hi {
                                let snap_i = (p + k * CHUNK - snap_base) / CHUNK;
                                if let Some(dst_h) = snap_hashes.get_mut(snap_i) {
                                    *dst_h = sent[k];
                                }
                            }
                        }
                    }
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
        if !s.upload_dirty_with_timeout(&dirty_runs, QUEUE_WAIT_TIMEOUT) {
            return;
        }
        let mut spans = Vec::new();
        let mut done = 0usize;
        while done < len {
            let cur = va + done;
            let tile = cur & !(TILE - 1);
            let Some(wi) = s.window_index(cur) else { break };
            let b = s.windows[wi].0;
            let to_tile_end = (tile + TILE) - cur;
            if !s.mapped.contains_key(&tile) {
                done += (len - done).min(to_tile_end);
                continue;
            }
            let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
            let chunk = (len - done).min(to_tile_end).min(to_win_end);
            spans.push((cur, s.windows[wi].1.clone(), (cur - b) as u64, chunk));
            done += chunk;
        }
        if spans.is_empty() {
            return;
        }
        (s.dev.clone(), spans, snap_hashes)
    };

    let total_need: usize = spans.iter().map(|s| s.3).sum();
    let Some(job) = take_readback_job(&dev, total_need) else {
        let mut g = lock_state();
        if let Some(s) = g.as_mut() {
            s.wait_title_queues_for_readback();
            s.readback_ranges(&[(va, len)]);
        }
        return;
    };

    let mut idx = 0usize;
    let mut sub_off = 0usize;
    let mut pending: Vec<(usize, usize, usize)> = Vec::new();
    let mut waited_title_queues = false;
    while idx < spans.len() {
        let mut used = 0usize;
        pending.clear();
        if job.alloc.Reset().is_err() || job.list.Reset(&job.alloc, None).is_err() {
            break;
        }
        while idx < spans.len() && used < job.cap {
            let (dst, ref win_buf, src_base, span_len) = spans[idx];
            let rem = span_len - sub_off;
            let take = rem.min(job.cap - used);
            job.list.CopyBufferRegion(&job.buf, used as u64, win_buf, src_base + sub_off as u64, take as u64);
            pending.push((dst + sub_off, used, take));
            used += take;
            sub_off += take;
            if sub_off >= span_len {
                idx += 1;
                sub_off = 0;
            }
        }
        if pending.is_empty() || job.list.Close().is_err() {
            break;
        }
        let Ok(cmd_list) = job.list.cast::<ID3D12CommandList>() else { break };
        // 2. Order `s.rb_queue` behind pending sparse uploads and active title queues via
        //    `rb_fence`, then submit & signal atomically under `STATE` and release `STATE` BEFORE
        //    waiting on `rb_done_fence`. (`s.queue` never waits on title queues, preventing a
        //    `title_qA -> title_qB -> s.queue -> title_qA` circular GPU deadlock.)
        let wait_target = {
            let mut g = lock_state();
            let Some(s) = g.as_mut() else { break };
            s.order_rb_after_uploads();
            if !waited_title_queues {
                s.wait_title_queues_for_readback();
                waited_title_queues = true;
            }
            super::title_textures::own_work(|| {
                s.rb_queue.ExecuteCommandLists(&[Some(cmd_list)]);
            });
            s.signal_rb();
            (s.rb_done_fence.clone(), s.rb_done_value)
        };
        if wait_target.1 > 0 && wait_target.0.GetCompletedValue() < wait_target.1 {
            if !wait_fence_until(&wait_target.0, job.event, wait_target.1, std::time::Duration::from_secs(5)) {
                crate::xlog_warn!(
                    "sparse: readback_now queue has not reached fence value {} after 5 s (at {})",
                    wait_target.1,
                    wait_target.0.GetCompletedValue()
                );
                super::real::stop_if_device_removed();
                // the copy is still queued: keep its buffer alive until it has run
                keep_in_flight_readback_job(job, wait_target.0.clone(), wait_target.1);
                return;
            }
        }
        // 3. Copy only still-committed non-hole 64-byte chunks that were NOT modified by the CPU
        //    during the GPU wait, and sync `sent` hashes only for copied chunks.
        let mut g = lock_state();
        let Some(s) = g.as_mut() else { break };
        let p = job.mapped as *const u8;
        for &(dst, off, chunk) in &pending {
            apply_readback_preserving_cpu_writes(dst, p.add(off), chunk, snap_base, &mut snap_hashes, &mut s.mapped);
        }
        s.reclaim_ring();
    }
    return_readback_job(job);
}

/// The title committed `[a, a + n)` in a sparse window. The pages of tiles mapped for it that
/// lie outside the range are not committed (holes); the range's pages are.
pub unsafe fn committed(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    let mut g = lock_state();
    let Some(s) = g.as_mut() else {
        let (start, end) = page_span(a, n);
        early_add(&mut EARLY.lock().unwrap(), start, end);
        return;
    };
    s.commit_range(a, n);
}

/// The title decommitted `[a, a + n)`: tiles whose pages are all uncommitted (whether in one
/// full-tile decommit or incrementally via sub-tile decommits) give their video memory back
/// (unless a placed buffer still pins the pool tile).
pub unsafe fn decommitted(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    // (pages `commit_title` left lazy go with them)
    crate::hle::kernelx::lazy::unmark_range(a, n);
    let mut g = lock_state();
    let Some(s) = g.as_mut() else {
        let (start, end) = page_span(a, n);
        early_remove(&mut EARLY.lock().unwrap(), start, end);
        return;
    };
    s.decommit_range(a, n);
}

/// Clear per-page hole bits in `[a, a + n)` for already-mapped tiles; if any VRAM-only tile
/// transitions from all-holes (`0xFFFF`) to watchable (`!= 0xFFFF`), invalidate cached watchable
/// tile runs and bump `MAPPED_GEN`.
fn unmask_committed_holes(
    mapped: &mut FxHashMap<usize, TileEntry>,
    runs: &mut Option<Vec<(usize, usize)>>,
    a: usize,
    n: usize,
) -> bool {
    let mut pg = a & !(PAGE - 1);
    let end = a.saturating_add(n);
    let mut unmasked = false;
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = t.saturating_add(TILE).min(end);
        if let Some(entry) = mapped.get_mut(&t) {
            let was_all_holes = entry.holes == 0xFFFF;
            let mut p = pg;
            while p < tile_end {
                entry.holes &= !(1u16 << ((p - t) / PAGE));
                p += PAGE;
            }
            if was_all_holes && entry.holes != 0xFFFF {
                unmasked = true;
            }
        }
        if tile_end <= pg {
            break;
        }
        pg = tile_end;
    }
    if unmasked {
        *runs = None;
        MAPPED_GEN.fetch_add(1, Ordering::Release);
    }
    unmasked
}

impl State {
    /// `committed` with the state locked
    unsafe fn commit_range(&mut self, a: usize, n: usize) {
        self.ensure_mapped(a, n, |_| 0xFFFF);
        unmask_committed_holes(&mut self.mapped, &mut self.runs, a, n);
    }

    /// Clear hole masks for 64 KiB tiles lazily CPU-committed by `kernelx::lazy` (#2582).
    /// Leaves `to_zero` untouched so existing GPU VRAM contents remain intact while subsequent
    /// write-watch dirty 4 KiB pages upload cleanly from the CPU view.
    fn drain_lazy_commits(&mut self) {
        if !HAS_LAZY_COMMITS.swap(false, Ordering::AcqRel) {
            return;
        }
        let pages = std::mem::take(&mut *LAZY_COMMITS.lock().unwrap_or_else(|e| e.into_inner()));
        let mut unmasked = false;
        for page in pages {
            if let Some(entry) = self.mapped.get_mut(&(page & !(TILE - 1))) {
                if entry.holes == 0xFFFF {
                    unmasked = true;
                }
                entry.holes = 0;
            }
        }
        if unmasked {
            self.runs = None;
            MAPPED_GEN.fetch_add(1, Ordering::Release);
        }
    }

    /// `decommitted` with the state locked: the tiles that go are parked (`flush_parked` unmaps
    /// them and frees their pool tiles)
    unsafe fn decommit_range(&mut self, a: usize, n: usize) {
        self.drain_lazy_commits();
        let mut pools: Vec<(usize, (u32, u32))> = Vec::new();
        let mut t = a & !(TILE - 1);
        let end = a.saturating_add(n);
        while t < end {
            if let Some(e) = self.mapped.get(&t) {
                pools.push((t, e.pool));
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        let tiles = decommit_tiles(&mut self.mapped, &self.pinned, &mut Vec::new(), &mut self.to_zero, a, n);
        if tiles.is_empty() {
            return;
        }
        self.runs = None;
        for &t in &tiles {
            if let Ok(i) = pools.binary_search_by_key(&t, |p| p.0) {
                self.parked.insert(t, pools[i].1);
            }
        }
        MAPPED_GEN.fetch_add(1, Ordering::Release);
        if self.parked.len() >= PARKED_MAX && self.flush_parked() {
            self.drain_quarantined();
            self.retire_free_pools();
        }
    }

    /// Queue a pool tile that no window maps, none is parked on, and no buffer pins for GPU fence
    /// quarantine. Never pushes straight to `free` or signals queues here (called from `forget_in`
    /// during D3D12 `Release`); `flush_pending_quarantine` attaches completion fences before reuse.
    fn release_if_unused(&mut self, tile: (u32, u32)) {
        if !self.pinned.contains_key(&tile)
            && !self.mapped.values().any(|e| e.pool == tile)
            && !self.parked.values().any(|&p| p == tile)
            && !self.pending_unmap_va.values().any(|&(p, _)| p == tile)
            && !self.quarantined.iter().any(|b| b.tiles.contains(&tile))
            && !self.pending_quarantine.contains(&tile)
            && !self.free.contains(&tile)
        {
            self.pending_quarantine.push(tile);
        }
    }

    /// Move any `pending_quarantine` tiles (unpinned when a placed buffer was released) into the
    /// GPU fence quarantine behind `self.fence` and every known title queue.
    unsafe fn flush_pending_quarantine(&mut self) -> bool {
        self.flush_pending_quarantine_with(None)
    }

    unsafe fn flush_pending_quarantine_with(&mut self, extra: Option<&ID3D12CommandQueue>) -> bool {
        if self.pending_quarantine.is_empty() {
            return false;
        }
        let raw_ptrs = collect_known_queue_ptrs(extra);
        let borrowed: Vec<&ID3D12CommandQueue> = raw_ptrs.iter().filter_map(|p| ID3D12CommandQueue::from_raw_borrowed(p)).collect();
        self.flush_pending_quarantine_for(&borrowed)
    }

    unsafe fn flush_pending_quarantine_for(&mut self, queues: &[&ID3D12CommandQueue]) -> bool {
        if self.pending_quarantine.is_empty() {
            return false;
        }
        let tiles: Vec<(u32, u32)> = self
            .pending_quarantine
            .drain(..)
            .filter(|tile| {
                !self.pinned.contains_key(tile)
                    && !self.mapped.values().any(|e| e.pool == *tile)
                    && !self.parked.values().any(|&p| p == *tile)
                    && !self.pending_unmap_va.values().any(|&(p, _)| p == *tile)
                    && !self.quarantined.iter().any(|b| b.tiles.contains(tile))
            })
            .collect();
        if tiles.is_empty() {
            return false;
        }
        self.quarantine_tiles_for(tiles, queues);
        true
    }

    /// Order `self.queue` behind every active same-device title queue via `self.rb_fence`
    /// so a subsequent sparse readback sees all submitted GPU writes.
    unsafe fn wait_title_queues_for_readback(&mut self) {
        let self_q = self.queue.as_raw() as usize;
        let rb_q = self.rb_queue.as_raw() as usize;
        let self_dev = self.dev.as_raw();
        let mut seen: Vec<usize> = Vec::new();
        for raw in super::present::known_queues() {
            if raw == 0 || raw == self_q || raw == rb_q || seen.contains(&raw) {
                continue;
            }
            let ptr = raw as *mut c_void;
            let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&ptr) else {
                continue;
            };
            let mut qdev: Option<ID3D12Device> = None;
            if q.GetDevice(&mut qdev).is_err() || qdev.as_ref().is_none_or(|d| d.as_raw() != self_dev) {
                continue;
            }
            seen.push(raw);
            if let Some((f, v)) = readback_after(&self.rb_queue, q) {
                super::list_x::note_internal_wait(rb_q, "sparse readback behind a title queue", &f, v);
            }
        }
    }

    unsafe fn signal_title_queues_for(&mut self, queues: &[&ID3D12CommandQueue]) -> Option<Vec<(ID3D12Fence, u64)>> {
        // Pass 1 (before signaling): deduplicate live same-device title queues and prune stale
        // queue fences unconditionally so an early error cannot prune another live queue.
        let self_q = self.queue.as_raw() as usize;
        let rb_q = self.rb_queue.as_raw() as usize;
        let mut active: Vec<(usize, &ID3D12CommandQueue)> = Vec::with_capacity(queues.len());
        let mut live_keys: Vec<usize> = Vec::with_capacity(queues.len());
        for &q in queues {
            let key = q.as_raw() as usize;
            if key == 0 || key == self_q || key == rb_q || live_keys.contains(&key) {
                continue;
            }
            let mut qdev: Option<ID3D12Device> = None;
            if q.GetDevice(&mut qdev).is_err() || qdev.as_ref().is_none_or(|d| d.as_raw() != self.dev.as_raw()) {
                continue;
            }
            live_keys.push(key);
            active.push((key, q));
        }
        self.queue_fences.retain(|k, _| live_keys.contains(k));
        let mut marks = Vec::with_capacity(active.len());
        for (key, q) in active {
            let entry = match self.queue_fences.entry(key) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => match self.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) {
                    Ok(fence) => e.insert((fence, 0)),
                    Err(_) => return None,
                },
            };
            let next = entry.1 + 1;
            if q.Signal(&entry.0, next).is_ok() {
                entry.1 = next;
                marks.push((entry.0.clone(), next));
            } else {
                return None;
            }
        }
        Some(marks)
    }

    unsafe fn quarantine_tiles_for(&mut self, tiles: Vec<(u32, u32)>, queues: &[&ID3D12CommandQueue]) {
        if tiles.is_empty() {
            return;
        }
        let marks = self.signal_title_queues_for(queues).map(|mut title_marks| {
            title_marks.push((self.fence.clone(), self.value));
            title_marks
        });
        self.quarantined.push_back(RetiredTiles { tiles, marks });
    }

    /// Unmap `entries` on `self.queue` and quarantine their unpinned pool tiles behind `self.fence`.
    /// Caller guarantees every active title queue has already passed the work that could touch them.
    unsafe fn unmap_entries_and_quarantine(&mut self, entries: Vec<(usize, (u32, u32))>) {
        if entries.is_empty() {
            return;
        }
        let mut by_window: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        let mut tiles: Vec<(u32, u32)> = Vec::new();
        for (va, tile) in entries {
            if let Some(wi) = self.window_index(va) {
                by_window.entry(wi).or_default().push(va);
            }
            if !self.pinned.contains_key(&tile)
                && !self.mapped.values().any(|e| e.pool == tile)
                && !self.parked.values().any(|&p| p == tile)
                && !self.pending_unmap_va.values().any(|&(p, _)| p == tile)
                && !self.quarantined.iter().any(|b| b.tiles.contains(&tile))
                && !tiles.contains(&tile)
            {
                tiles.push(tile);
            }
        }
        for (wi, ts) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            self.unmap_tiles(&r, b, &ts);
        }
        self.signal();
        if !tiles.is_empty() {
            let mut marks = vec![(self.fence.clone(), self.value)];
            if self.rb_done_value > 0 {
                marks.push((self.rb_done_fence.clone(), self.rb_done_value));
            }
            self.quarantined.push_back(RetiredTiles {
                tiles,
                marks: Some(marks),
            });
        }
    }

    /// Execute deferred `UpdateTileMappings(NULL)` batches once all title queues have passed their
    /// completion marks, skipping any VA that was re-committed in the meantime.
    unsafe fn drain_pending_unmaps(&mut self) -> bool {
        let mut unmapped = false;
        while let Some(front) = self.pending_unmaps.front() {
            let ready = front.marks.as_ref().is_some_and(|marks| marks.iter().all(|(f, v)| f.GetCompletedValue() >= *v));
            if !ready {
                break;
            }
            let batch = self.pending_unmaps.pop_front().unwrap();
            let mut ready_entries = Vec::with_capacity(batch.entries.len());
            for (va, tile) in batch.entries {
                if self.pending_unmap_va.get(&va).is_some_and(|&(_, s)| s == batch.seq) {
                    self.pending_unmap_va.remove(&va);
                    ready_entries.push((va, tile));
                }
            }
            if !ready_entries.is_empty() {
                self.unmap_entries_and_quarantine(ready_entries);
                unmapped = true;
            }
        }
        unmapped
    }

    unsafe fn drain_quarantined(&mut self) -> bool {
        self.drain_pending_unmaps();
        let mut freed = false;
        while let Some(front) = self.quarantined.front() {
            let ready = front.marks.as_ref().is_some_and(|marks| marks.iter().all(|(f, v)| f.GetCompletedValue() >= *v));
            if !ready {
                break;
            }
            let batch = self.quarantined.pop_front().unwrap();
            for tile in batch.tiles {
                if !self.pinned.contains_key(&tile) {
                    self.free.push(tile);
                    freed = true;
                }
            }
        }
        freed
    }

    /// Queue parked tiles for unmapping once all active title queues finish in-flight work (never
    /// issuing `UpdateTileMappings(NULL)` concurrently with executing title queues, and never
    /// inserting a cross-queue GPU `Wait` on `self.queue`). When no title queues are busy, unmaps
    /// immediately on `self.queue` and quarantines unpinned pool tiles behind `self.fence`.
    unsafe fn flush_parked(&mut self) -> bool {
        self.flush_parked_with(None)
    }

    unsafe fn flush_parked_with(&mut self, extra: Option<&ID3D12CommandQueue>) -> bool {
        self.drain_pending_unmaps();
        if self.parked.is_empty() {
            return false;
        }
        let raw_ptrs = collect_known_queue_ptrs(extra);
        let borrowed: Vec<&ID3D12CommandQueue> = raw_ptrs.iter().filter_map(|p| ID3D12CommandQueue::from_raw_borrowed(p)).collect();
        self.flush_parked_for(&borrowed)
    }

    unsafe fn flush_parked_for(&mut self, queues: &[&ID3D12CommandQueue]) -> bool {
        self.drain_pending_unmaps();
        if self.parked.is_empty() {
            return false;
        }
        let entries: Vec<(usize, (u32, u32))> = std::mem::take(&mut self.parked).into_iter().collect();
        let mut title_marks = self.signal_title_queues_for(queues);
        if self.rb_done_value > 0 {
            if let Some(marks) = title_marks.as_mut() {
                marks.push((self.rb_done_fence.clone(), self.rb_done_value));
            }
        }
        if title_marks.as_ref().is_some_and(|marks| marks.iter().all(|(f, v)| f.GetCompletedValue() >= *v)) {
            self.unmap_entries_and_quarantine(entries);
        } else {
            self.unmap_seq = self.unmap_seq.wrapping_add(1);
            let seq = self.unmap_seq;
            for &(va, tile) in &entries {
                self.pending_unmap_va.insert(va, (tile, seq));
            }
            self.pending_unmaps.push_back(PendingUnmap { seq, entries, marks: title_marks });
        }
        true
    }
}

fn collect_known_queue_ptrs(extra: Option<&ID3D12CommandQueue>) -> Vec<*mut c_void> {
    let mut raw_ptrs: Vec<*mut c_void> =
        super::present::known_queues().into_iter().filter(|&raw| raw != 0).map(|raw| raw as *mut c_void).collect();
    if let Some(q) = extra {
        let p = q.as_raw();
        if !p.is_null() && !raw_ptrs.contains(&p) {
            raw_ptrs.push(p);
        }
    }
    raw_ptrs
}

fn watchable_tile_runs(mapped: &FxHashMap<usize, TileEntry>) -> Vec<(usize, usize)> {
    let mut tiles: Vec<usize> = mapped.iter().filter(|(_, e)| e.holes != 0xFFFF).map(|(&t, _)| t).collect();
    tiles.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for t in tiles {
        match runs.last_mut() {
            Some(last) if last.1 == t => last.1 = t + TILE,
            _ => runs.push((t, t + TILE)),
        }
    }
    runs
}

/// Drain any pending VEH lazy commits so `mapped_gen()` / `mapped_ranges()` include newly
/// CPU-committed tiles on the very first submit (zero-lock fast path when none are pending).
pub(super) fn drain_lazy_commits_if_pending() {
    if HAS_LAZY_COMMITS.load(Ordering::Acquire) {
        if let Some(s) = lock_state().as_mut() {
            s.drain_lazy_commits();
        }
    }
}

/// The sparse windows' tiles with video memory as sorted (start, end) runs: where a CPU write
/// has to reach the GPU.
pub fn mapped_ranges() -> Vec<(usize, usize)> {
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return Vec::new() };
    if s.runs.is_none() {
        s.runs = Some(watchable_tile_runs(&s.mapped));
    }
    s.runs.clone().unwrap_or_default()
}

/// time spent and pages found by the write-watch scans (`refresh_once`), for the timing line
static WATCH: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// [lock wait, chunk scan, upload] in µs, runs uploaded
static PHASES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

pub fn note_watch(t: std::time::Duration, pages: usize) {
    WATCH[0].fetch_add(t.as_micros() as u64, Ordering::Relaxed);
    WATCH[1].fetch_add(pages as u64, Ordering::Relaxed);
}

/// Before the title's ExecuteCommandLists on `queue`: the pages the CPU wrote (`dirty`, 4 KiB
/// pages from the write-watch, sorted) and the reused tiles are copied asynchronously into video
/// memory, and `queue` GPU-waits for that and for new tile mappings.
pub unsafe fn before_execute(queue: &ID3D12CommandQueue, dirty: &[usize], page: usize) {
    let t0 = std::time::Instant::now();
    // Phase 1 of GPU-view sync runs BEFORE `lock_state()` so `kernelx::AWAY` / `mirrors` locks
    // are never acquired while holding `sparse::STATE` (ABBA deadlock with `XMemMapPhysicalPages`).
    let gpu_view_prep = if gpu_view::on() { gpu_view::prepare_sync() } else { None };
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return };
    s.drain_lazy_commits();
    // the title's work sees what was decommitted unmapped, as it was
    let flushed = s.flush_parked_with(Some(queue));
    let quarantined = s.flush_pending_quarantine_with(Some(queue));
    let drained = s.drain_quarantined();
    if flushed || quarantined || drained {
        s.retire_free_pools();
    }
    let t_locked = std::time::Instant::now();
    PHASES[0].fetch_add((t_locked - t0).as_micros() as u64, Ordering::Relaxed);
    let zero_runs = std::mem::take(&mut s.to_zero);
    for &(a, n) in &zero_runs {
        s.clear_sent_range(a, n);
    }
    let mut dirty_runs = std::mem::take(&mut s.scratch_runs);
    dirty_runs.clear();
    scan_dirty_pages_into(&mut s.mapped, dirty, page, scan_threads(), &mut dirty_runs);
    s.scan_retry_pages(&mut dirty_runs);
    let t_scanned = std::time::Instant::now();
    PHASES[1].fetch_add((t_scanned - t_locked).as_micros() as u64, Ordering::Relaxed);
    PHASES[3].fetch_add((zero_runs.len() + dirty_runs.len()) as u64, Ordering::Relaxed);
    // Submit tile clears before CPU writes so overlapping CopyBufferRegion destinations have no WAW hazard.
    let uploaded = if s.upload_zeros(&zero_runs, QUEUE_WAIT_TIMEOUT) {
        s.upload_dirty_with_timeout(&dirty_runs, QUEUE_WAIT_TIMEOUT)
    } else {
        s.to_zero.extend_from_slice(&zero_runs);
        s.retry_dirty_from(&dirty_runs, (0, 0));
        false
    };
    if uploaded && gpu_view::on() {
        gpu_view::apply_sync(s, dirty, gpu_view_prep);
    }
    PHASES[2].fetch_add(t_scanned.elapsed().as_micros() as u64, Ordering::Relaxed);
    {
        static CALLS: AtomicU64 = AtomicU64::new(0);
        static BYTES: AtomicU64 = AtomicU64::new(0);
        static MICROS: AtomicU64 = AtomicU64::new(0);
        let bytes: usize = if uploaded { zero_runs.iter().chain(dirty_runs.iter()).map(|r| r.1).sum() } else { 0 };
        let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        let b = BYTES.fetch_add(bytes as u64, Ordering::Relaxed) + bytes as u64;
        let us = MICROS.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed) + t0.elapsed().as_micros() as u64;
        if n.is_power_of_two() && n >= 16 {
            crate::xlog_info!("sparse windows: {n} submits, {} MB of CPU writes uploaded, {} ms spent, {} tiles mapped", b >> 20, us / 1000, s.mapped.len());
            let ms = |x: &AtomicU64| x.load(Ordering::Relaxed) / 1000;
            crate::xlog_info!(
                "sparse windows timing: write-watch {} ms ({} dirty pages), lock wait {} ms, chunk scan {} ms, upload {} ms ({} runs)",
                ms(&WATCH[0]),
                WATCH[1].load(Ordering::Relaxed),
                ms(&PHASES[0]),
                ms(&PHASES[1]),
                ms(&PHASES[2]),
                PHASES[3].load(Ordering::Relaxed)
            );
        }
    }
    dirty_runs.clear();
    s.scratch_runs = dirty_runs;
    // the tile mappings of buffers placed since the last signal (`place_buffer`)
    if s.mapping_queued.load(Ordering::Relaxed) {
        s.signal();
    }
    let q = queue.as_raw() as usize;
    let waited_before = s.waited.get(&q).copied().unwrap_or(0);
    if s.value > 0 && waited_before < s.value {
        let (fence, value) = (s.fence.clone(), s.value);
        // new tile mappings since this queue last waited: on the CPU (`sparse_cpu_wait`); only
        // copies: the GPU wait as before (a CPU wait there would hold every submit for them)
        let mappings = s.mapped_value > waited_before;
        s.waited.insert(q, value);
        // (the wait below is without the state lock: the sparse queue's own work needs no lock to
        // finish, and other submits must not stand behind this one meanwhile)
        drop(g);
        if !(mappings && cpu_waited_for_mappings(&fence, value, "sparse uploads and tile mappings")) {
            let _ = queue.Wait(&fence, value);
            super::list_x::note_internal_wait(q, "sparse uploads and tile mappings", &fence, value);
        }
    }
}

/// Fast-path wait for any sparse-window tile mappings queued after `title_textures::refresh`
/// (for example while `record_deferred` waited on PSO builds).
pub(super) unsafe fn wait_pending_mappings(queue: &ID3D12CommandQueue) {
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return };
    if s.mapping_queued.load(Ordering::Relaxed) {
        s.signal();
    }
    let q = queue.as_raw() as usize;
    let waited_before = s.waited.get(&q).copied().unwrap_or(0);
    if s.value > 0 && waited_before < s.value {
        let (fence, value) = (s.fence.clone(), s.value);
        let mappings = s.mapped_value > waited_before;
        s.waited.insert(q, value);
        drop(g);
        if !(mappings && cpu_waited_for_mappings(&fence, value, "sparse uploads and tile mappings")) {
            let _ = queue.Wait(&fence, value);
            super::list_x::note_internal_wait(q, "sparse uploads and tile mappings", &fence, value);
        }
    }
}

/// `XCL_SPARSE_CPU_WAIT_MS` (1 000 by default, 0: off): how long a title submit waits on the
/// CPU for a mapping queue's tile mappings (the sparse windows', the video memory pages') before
/// its lists go, instead of the title's queue waiting for them on the GPU.
///
/// With the GPU wait every queue of a retail title held its queued work behind the sparse
/// queue's tile mappings. On an AMD RX 9070 (16 GB, at its memory budget) the sparse queue's
/// fence then stopped short of the value they waited for, for 15 s – while AMD's crash analysis
/// found no command buffer in flight – and the device was removed (five runs in a row, 63–140 s
/// into the world). A tile mapping is the video memory manager's work: it can have to evict, and
/// it cannot take memory from queues whose work is already queued behind that very mapping. With
/// the wait on the CPU no queue holds work behind a mapping; past the limit the GPU wait is used
/// as before (logged).
fn sparse_cpu_wait() -> std::time::Duration {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    std::time::Duration::from_millis(*MS.get_or_init(|| {
        std::env::var("XCL_SPARSE_CPU_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1000)
    }))
}

/// Whether `fence` (of the mapping queue `what`) reached `value` within `sparse_cpu_wait`.
pub(super) unsafe fn cpu_waited_for_mappings(fence: &ID3D12Fence, value: u64, what: &str) -> bool {
    let most = sparse_cpu_wait();
    if most.is_zero() {
        return false;
    }
    if fence.GetCompletedValue() >= value {
        return true;
    }
    thread_local! {
        static EVENT: HANDLE = unsafe { windows::Win32::System::Threading::CreateEventW(None, false, false, None).unwrap_or_default() };
    }
    let t = std::time::Instant::now();
    let reached = EVENT.with(|&ev| {
        if ev.is_invalid() || fence.SetEventOnCompletion(value, ev).is_err() {
            return false;
        }
        windows::Win32::System::Threading::WaitForSingleObject(ev, most.as_millis() as u32);
        fence.GetCompletedValue() >= value
    });
    let waited = t.elapsed();
    super::flight::event("mapping cpu wait", value, waited.as_micros() as u64, reached as u64);
    static SLOW: AtomicU64 = AtomicU64::new(0);
    if !reached {
        crate::xlog_warn!(
            "D3D12.X: {what} did not reach {value} in {} ms (at {}): the title's queue waits for it on the GPU",
            waited.as_millis(),
            fence.GetCompletedValue()
        );
    } else if waited >= std::time::Duration::from_millis(50) {
        let n = SLOW.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 8 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: a submit waited {} ms for {what} ({n} such waits)", waited.as_millis());
        }
    }
    reached
}

/// A buffer the title places at `va` in a sparse window (`bytes` long, `desc` the PC
/// description): a reserved resource over the window's tiles there, starting at the 64 KiB
/// below `va` (returned: its distance from there).
pub unsafe fn place_buffer(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<(crate::util::HRESULT, usize)> {
    let start = va & !(TILE - 1);
    let delta = va - start;
    let bytes = desc.Width as usize;
    if bytes == 0 {
        return None;
    }
    let last = va.checked_add(bytes)?.checked_sub(1)?;
    {
        let g = lock_state();
        let s = g.as_ref()?;
        s.window_index(start)?;
        s.window_index(last)?;
    }
    let mut d = *desc;
    d.Width = (delta + bytes) as u64;
    d.Alignment = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // Reserved buffers only accept NONE or ALLOW_UNORDERED_ACCESS (strip console flags and ALLOW_CROSS_ADAPTER).
    d.Flags &= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if state & (D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32 | D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32) != 0 {
        d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    // Create the reserved resource without holding STATE so concurrent final Release -> forget_buffer
    // cannot queue a reused COM pointer in FORGOTTEN and later evict this live buffer (#2457).
    let hr = super::real::dev_create_reserved_resource(this, &d as *const _ as usize, state as usize, 0, riid, ppv);
    if hr < 0 || ppv.is_null() || (*ppv).is_null() {
        return None;
    }
    let mut g = lock_state();
    let s = g.as_mut()?;
    // its tiles: the window's video memory at the same addresses; probe actual OS commitment
    // so uncommitted pages (e.g. [start, va) or reserved tails) stay marked as holes
    s.ensure_mapped(start, delta + bytes, |t| query_tile_holes(t));
    let tiles: Vec<(usize, (u32, u32))> = (start..va + bytes).step_by(TILE).filter_map(|t| s.mapped.get(&t).map(|e| (t, e.pool))).collect();
    let raw = *ppv;
    if super::real::dred_on() {
        super::real::name_resource(raw, &format!("xcl sparse_buf va={va:#x} sz={bytes:#x}"));
    }
    let res = ID3D12Resource::from_raw_borrowed(&raw)?.clone();
    s.map_tiles(&res, start, &tiles);
    for &(_, tile) in &tiles {
        *s.pinned.entry(tile).or_default() += 1;
    }
    s.buffers.insert(raw as usize, tiles.iter().map(|x| x.1).collect());
    // (no signal of its own: the next submit's `before_execute` gives one for all the mappings
    // queued since – a retail title placed 165 buffers in 6 ms, each with a mapping and a signal,
    // right before its GPU stopped)
    Some((hr, delta))
}

/// Placed buffers released while `STATE` was held, let go by the next `lock_state`. Held only
/// to push or take, never across a call out.
static FORGOTTEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// `STATE`, with the buffers released while it was held let go first.
fn lock_state() -> std::sync::MutexGuard<'static, Option<State>> {
    let mut g = STATE.lock().unwrap();
    forget_released(&mut g);
    g
}

fn forget_released(g: &mut Option<State>) {
    let gone = std::mem::take(&mut *FORGOTTEN.lock().unwrap());
    let Some(s) = g.as_mut() else { return };
    for res in gone {
        forget_in(s, res);
    }
}

/// A placed buffer is gone: its tiles are not pinned by it any more. This runs from the
/// resource's last Release, which D3D12 can make while it holds a lock of its own (in a title's
/// command list Reset), and `before_execute` calls D3D12 with `STATE` held: waiting for `STATE`
/// here deadlocked the two threads. So it does not wait: with `STATE` held, the holder's next
/// `lock_state` lets the buffer go.
pub fn forget_buffer(res: usize) {
    FORGOTTEN.lock().unwrap().push(res);
    match STATE.try_lock() {
        Ok(mut g) => forget_released(&mut g),
        Err(std::sync::TryLockError::WouldBlock) => {}
        Err(e) => panic!("sparse state: {e}"),
    }
}

fn forget_in(s: &mut State, res: usize) {
    let Some(tiles) = s.buffers.remove(&res) else { return };
    for tile in tiles {
        if let Some(c) = s.pinned.get_mut(&tile) {
            *c -= 1;
            if *c == 0 {
                s.pinned.remove(&tile);
                s.release_if_unused(tile);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `queue()` answers while the state's lock is held: a resource's last `Release` asks for
    /// it (`late_release::drain_known`) from inside a D3D12 call that the lock's holder can be
    /// waiting for (a list's `Reset` against an upload's allocator `Reset`, with `XCL_DRED=1`).
    #[test]
    fn the_queue_is_answered_while_the_state_is_locked() {
        let held = STATE.lock().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let asked = std::thread::spawn(move || {
            let _ = queue();
            let _ = tx.send(());
        });
        let answered = rx.recv_timeout(std::time::Duration::from_secs(5)).is_ok();
        drop(held);
        asked.join().unwrap();
        assert!(answered, "queue() waited for the sparse state's lock");
    }

    /// A block freed on one thread while another allocates: its address goes back to the
    /// windows only once it is decommitted (released first, the decommit came after another
    /// thread had the same address again: that thread's writes faulted).
    #[test]
    fn a_freed_block_is_decommitted_before_its_address_is_reused() {
        if crate::gpu::uma::get().is_none_or(|u| !u.has_sparse()) {
            return;
        }
        let threads: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| unsafe {
                    for i in 0..2000u64 {
                        let Some((p, n)) = alloc(TILE, TILE) else { continue };
                        std::ptr::write_volatile(p as *mut u64, i);
                        std::ptr::write_volatile((p + n - 8) as *mut u64, i);
                        assert_eq!(std::ptr::read_volatile(p as *const u64), i);
                        assert!(free(p));
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    }

    /// `f` on another thread while this one holds the sparse state (as `before_execute` does
    /// around its D3D12 calls, which can wait for that thread): did `f` finish without waiting
    /// for the state? The thread runs before the state is taken, so its start is not timed.
    fn returns_while_the_state_is_held(f: impl FnOnce() + Send + 'static) -> bool {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let (tx, rx) = std::sync::mpsc::channel();
        let t = std::thread::spawn(move || {
            let _ = ready_tx.send(());
            let _ = go_rx.recv();
            f();
            let _ = tx.send(());
        });
        ready_rx.recv().unwrap();
        let held = STATE.lock().unwrap();
        go_tx.send(()).unwrap();
        let done = rx.recv_timeout(std::time::Duration::from_secs(5));
        drop(held);
        let joined = t.join();
        assert!(joined.is_ok(), "the thread panicked: {joined:?}");
        done.is_ok()
    }

    /// A placed buffer released while another thread holds the sparse state: the release does
    /// not wait for the state, and the next `lock_state` lets the buffer go.
    #[test]
    fn a_buffer_released_while_the_state_is_held_does_not_wait_for_it() {
        const RES: usize = 0x7ff0_dead_0000;
        assert!(returns_while_the_state_is_held(|| forget_buffer(RES)), "forget_buffer waited for the sparse state's lock");
        drop(lock_state());
        assert!(!FORGOTTEN.lock().unwrap().contains(&RES));
    }


    /// Freeing the last tile in use of a pool and taking a tile again takes the retired pool
    /// back, not a new heap each time (#913: 288 pools of 64 MB, 18 GB asked of a 12 GB card).
    #[test]
    fn a_pool_retired_a_moment_ago_is_taken_back() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let rb_queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("rb_queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let rb_done_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_done_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");
            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_queue,
                rb_fence,
                rb_value: 0,
                rb_done_fence,
                rb_done_value: 0,
                windows: Vec::new(),
                pools: Vec::new(),
                pool_is_sys: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                parked: FxHashMap::default(),
                pending_unmaps: VecDeque::new(),
                pending_unmap_va: FxHashMap::default(),
                unmap_seq: 0,
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                quarantined: VecDeque::new(),
                pending_quarantine: Vec::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: Vec::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
            mapping_queued: std::sync::atomic::AtomicBool::new(false),
            mapped_value: 0,
                runs: None,
                scratch_runs: Vec::new(),
                retry_pages: Vec::new(),
            };

            let (first, reused) = s.take_tile().expect("a tile from a new pool");
            assert!(!reused, "a new pool's tile needs no clearing");
            s.free.push(first);
            s.retire_free_pools();
            assert_eq!(s.retired.len(), 1, "an all-free pool is retired");
            for round in 0..8 {
                let (tile, reused) = s.take_tile().expect("a tile");
                assert!(reused, "round {round}: a tile of a retired pool is cleared before use");
                s.free.push(tile);
                s.retire_free_pools();
                assert_eq!(s.retired.len(), 1, "round {round}: the retired pool was taken back, not a new one made");
                assert_eq!(s.pools.iter().filter(|p| p.is_some()).count(), 0, "round {round}");
            }
        }
    }

    #[test]
    fn aligned_tile_addresses_spread_across_hash_buckets() {
        use std::hash::BuildHasher;

        let hasher = BuildHasherDefault::<FxHasher>::default();
        let mut used = [false; 256];
        for tile in 0..used.len() {
            let address = (1usize << 32) + tile * TILE;
            let bucket = hasher.hash_one(address) as usize & (used.len() - 1);
            used[bucket] = true;
        }
        let occupied = used.iter().filter(|&&bucket| bucket).count();
        assert!(occupied >= used.len() / 2, "256 aligned addresses used {occupied} buckets");
    }

    /// Only the 64-byte chunks the CPU changed are sent again – the rest of the page may hold
    /// what the GPU wrote.
    #[test]
    fn only_changed_chunks_are_sent() {
        let mut page = vec![0u8; PAGE];
        let mut sent = ZERO_HASHES;
        assert!(changed_chunks(&page, &mut sent).is_empty(), "a zero page is what the GPU has");
        page[10] = 1;
        page[200] = 2;
        page[255] = 3;
        page[4000] = 4;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(0, 64), (192, 64), (3968, 64)]);
        assert!(changed_chunks(&page, &mut sent).is_empty(), "nothing changed since");
        page[256] = 5; // next to the chunk at 192: one run
        page[200] = 9;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(192, 128)]);
    }

    /// `scan_page` skips allocating a hash box on all-zero pages, allocates on the first write,
    /// and coalesces adjacent dirty chunks across consecutive pages.
    #[test]
    fn scan_page_lazy_alloc_and_coalesce() {
        let mut page = vec![0u8; PAGE];
        let mut slot: Option<Box<[u64; CHUNKS]>> = None;
        let mut out = Vec::new();
        scan_page(&page, 0x1_0000, &mut slot, &mut out);
        assert!(slot.is_none(), "all-zero page must not allocate hash box");
        assert!(out.is_empty());

        page[PAGE - 64] = 0xAA;
        scan_page(&page, 0x1_0000, &mut slot, &mut out);
        assert!(slot.is_some());
        assert_eq!(out, vec![(0x1_0FC0, 64)]);

        // Adjacent chunk at the start of the next page coalesces into a single run.
        let mut next = vec![0u8; PAGE];
        next[0] = 0xBB;
        let mut next_slot: Option<Box<[u64; CHUNKS]>> = None;
        scan_page(&next, 0x1_1000, &mut next_slot, &mut out);
        assert_eq!(out, vec![(0x1_0FC0, 128)]);
    }

    #[test]
    fn scan_page_avx2_matches_scalar_bit_for_bit() {
        let mut page = vec![0u8; PAGE];
        let mut slot_a: Option<Box<[u64; CHUNKS]>> = None;
        let mut slot_b: Option<Box<[u64; CHUNKS]>> = None;
        let (mut out_a, mut out_b) = (Vec::new(), Vec::new());

        // Step 1: all-zero fresh page (neither allocates)
        scan_page(&page, 0x2_0000, &mut slot_a, &mut out_a);
        scan_page_scalar(&page, 0x2_0000, &mut slot_b, &mut out_b);
        assert_eq!(out_a, out_b);
        assert!(slot_a.is_none() && slot_b.is_none());

        // Step 2: sparse non-zero writes across odd/even 64B chunks (including last byte)
        for &off in &[0usize, 63, 64, 191, 512, 2048, PAGE - 1] {
            page[off] = (off as u8).wrapping_mul(37).wrapping_add(1);
        }
        scan_page(&page, 0x2_0000, &mut slot_a, &mut out_a);
        scan_page_scalar(&page, 0x2_0000, &mut slot_b, &mut out_b);
        assert_eq!(out_a, out_b);
        assert_eq!(slot_a.as_deref(), slot_b.as_deref());

        // Step 3: re-scan unchanged page (no runs)
        out_a.clear();
        out_b.clear();
        scan_page(&page, 0x2_0000, &mut slot_a, &mut out_a);
        scan_page_scalar(&page, 0x2_0000, &mut slot_b, &mut out_b);
        assert_eq!(out_a, out_b);
        assert!(out_a.is_empty());

        // Step 4: zero out some previously non-zero chunks and mutate another
        page[64] = 0;
        page[512] = 0;
        page[1024] = 0x5A;
        scan_page(&page, 0x2_0000, &mut slot_a, &mut out_a);
        scan_page_scalar(&page, 0x2_0000, &mut slot_b, &mut out_b);
        assert_eq!(out_a, out_b);
        assert_eq!(slot_a.as_deref(), slot_b.as_deref());
    }

    #[test]
    fn parallel_scan_dirty_pages_matches_serial_bit_for_bit() {
        // Allocate a 64 KiB-aligned backing buffer of 16 tiles (256 pages = 1 MiB) via VirtualAlloc
        // so `t = x & !(TILE - 1)` aligns cleanly with `MappedTile` boundaries.
        const NUM_TILES: usize = 16;
        const TOTAL_BYTES: usize = NUM_TILES * TILE;
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), TOTAL_BYTES, MEM_COMMIT | 0x2000, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            assert_eq!(base % TILE, 0);
            let mem = std::slice::from_raw_parts_mut(base as *mut u8, TOTAL_BYTES);
            // Seed deterministic pattern across every other page + boundary-spanning chunks.
            for p in 0..(NUM_TILES * PAGES_PER_TILE) {
                if p % 3 == 0 {
                    mem[p * PAGE] = (p as u8).wrapping_add(1);
                    mem[p * PAGE + PAGE - 1] = (p as u8).wrapping_mul(13).wrapping_add(7);
                } else if p % 3 == 1 {
                    // Adjacent start chunk to force boundary coalescing across worker slices
                    mem[p * PAGE] = 0xEE;
                }
            }
            let make_map = || -> FxHashMap<usize, TileEntry> {
                let mut m = FxHashMap::default();
                for ti in 0..NUM_TILES {
                    // Mark page 5 as a hole on odd tiles
                    let holes = if ti % 2 == 1 { 1u16 << 5 } else { 0 };
                    m.insert(base + ti * TILE, TileEntry::with_holes((0, ti as u32), holes));
                }
                m
            };
            let mut dirty: Vec<usize> = (0..(NUM_TILES * PAGES_PER_TILE)).map(|p| base + p * PAGE).collect();
            // Inject a duplicate entry to verify deduplication safety.
            dirty.insert(40, dirty[40]);

            let mut serial_map = make_map();
            let mut parallel_map = make_map();
            let (mut out_serial, mut out_parallel) = (Vec::new(), Vec::new());

            scan_dirty_pages_into(&mut serial_map, &dirty, PAGE, 1, &mut out_serial);
            scan_dirty_pages_into(&mut parallel_map, &dirty, PAGE, 6, &mut out_parallel);
            assert_eq!(out_parallel, out_serial, "parallel scan must match serial scan bit-for-bit");
            assert!(!out_parallel.is_empty());

            // Second pass with no changes must emit zero runs on both.
            out_serial.clear();
            out_parallel.clear();
            scan_dirty_pages_into(&mut serial_map, &dirty, PAGE, 1, &mut out_serial);
            scan_dirty_pages_into(&mut parallel_map, &dirty, PAGE, 6, &mut out_parallel);
            assert!(out_serial.is_empty() && out_parallel.is_empty());

            // Mutate one byte on every 4th page and re-verify parity.
            for p in (0..(NUM_TILES * PAGES_PER_TILE)).step_by(4) {
                mem[p * PAGE + 128] ^= 0xFF;
            }
            scan_dirty_pages_into(&mut serial_map, &dirty, PAGE, 1, &mut out_serial);
            scan_dirty_pages_into(&mut parallel_map, &dirty, PAGE, 6, &mut out_parallel);
            assert_eq!(out_parallel, out_serial, "second mutation pass must match serial bit-for-bit");

            VirtualFree(base as *mut c_void, 0, 0x8000);
        }
    }

    /// Ring-buffer planner keeps every chunk contiguous, 64-byte aligned, <= cap/2, and splits
    /// cleanly across the end of the ring buffer without wrapping mid-copy.
    #[test]
    fn ring_planner_handles_wrap_and_large_chunks() {
        let cap = 64 << 20;
        // Normal aligned chunk
        let (a, n) = plan_ring_chunk(0, 4096, cap);
        assert_eq!((a, n), (0, 4096));
        // Unaligned cursor aligns up to 64
        let (a, n) = plan_ring_chunk(17, 256, cap);
        assert_eq!((a, n), (64, 256));
        // Oversized chunk capped at cap / 2
        let (a, n) = plan_ring_chunk(0, cap * 2, cap);
        assert_eq!((a, n), (0, cap / 2));
        // Near end of ring with >= 4 KiB left: shrinks to fill tail
        let near = (cap - 8192) as u64;
        let (a, n) = plan_ring_chunk(near, 16384, cap);
        assert_eq!((a, n), (near, 8192));
        assert_eq!((a as usize % cap) + n, cap);
        // Tiny tail (< 4 KiB): wraps cleanly to 0 of next cycle
        let tiny = (cap - 512) as u64;
        let (a, n) = plan_ring_chunk(tiny, 4096, cap);
        assert_eq!((a, n), (cap as u64, 4096));
        assert_eq!(a as usize % cap, 0);
    }

    /// `query_tile_holes` marks reserved, free, and `PAGE_NOACCESS` pages as holes while clearing
    /// bits only for committed, accessible 4 KiB pages.
    #[test]
    fn query_tile_holes_detects_reserved_noaccess_and_committed_pages() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            // SAFETY: Reserving 2 tiles (128 KiB) with VirtualAlloc returns a 64 KiB-aligned VA.
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);

            // Initially every page in both reserved tiles is a hole.
            assert_eq!(query_tile_holes(base), 0xFFFF);
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 4..8 as PAGE_READWRITE and page 9 as PAGE_NOACCESS.
            let p4 = VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p4.is_null());
            let p9 = VirtualAlloc((base + 9 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_NOACCESS);
            assert!(!p9.is_null());

            // Only bits 4..8 are cleared; reserved [0..4), PAGE_NOACCESS [9], and reserved [10..16) remain holes.
            assert_eq!(query_tile_holes(base), 0xFFFF & !(0b1111u16 << 4));
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 0..4 and confirm bits 0..4 clear too.
            let p0 = VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p0.is_null());
            assert_eq!(query_tile_holes(base), 0xFF00);

            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
            // Released (MEM_FREE) memory reports all holes.
            assert_eq!(query_tile_holes(base), 0xFFFF);
        }
    }

    /// Mapping tiles for a placed buffer over partially-committed/reserved memory must preserve
    /// uncommitted hole bits so `readback_ranges` / `sync_sent_range` do not access-violate,
    /// and later commits clear those hole bits so subsequent writes scan cleanly.
    #[test]
    fn ensure_mapped_with_query_holes_skips_uncommitted_pages_on_readback() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");

            // Reserve 2 tiles; commit only [base + 0x4000, base + 0x8000) to simulate a
            // non-64KiB-aligned placed buffer with uncommitted prefix [base, base + 0x4000) and
            // uncommitted second tile [base + TILE, base + 2*TILE).
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            assert!(!VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());

            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((2 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
                .expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let rb_queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("rb_queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let rb_done_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_done_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");

            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_queue,
                rb_fence,
                rb_value: 0,
                rb_done_fence,
                rb_done_value: 0,
                windows: vec![(base, reserved)],
                pools: Vec::new(),
                pool_is_sys: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                pending_unmaps: VecDeque::new(),
                pending_unmap_va: FxHashMap::default(),
                unmap_seq: 0,
                quarantined: VecDeque::new(),
                pending_quarantine: Vec::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: Vec::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
            mapping_queued: std::sync::atomic::AtomicBool::new(false),
            mapped_value: 0,
                runs: None,
                scratch_runs: Vec::new(),
                retry_pages: Vec::new(),
            };

            // Mirrors `place_buffer`: map both tiles probing real OS page commitment.
            s.ensure_mapped(base, 2 * TILE, |t| query_tile_holes(t));
            assert_eq!(s.mapped[&base].holes, 0xFF0F, "uncommitted prefix and suffix pages must stay holes");
            assert_eq!(s.mapped[&(base + TILE)].holes, 0xFFFF, "fully reserved tile must have all 16 holes set");

            // Reading back the full 2-tile span must not fault on uncommitted pages.
            s.readback_ranges(&[(base, 2 * TILE)]);

            // Now incrementally commit [base, base + 4*PAGE) and clear its hole bits (as `committed` does).
            assert!(!VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            let entry = s.mapped.get_mut(&base).unwrap();
            for i in 0..4 {
                entry.holes &= !(1u16 << i);
            }
            assert_eq!(entry.holes, 0xFF00);

            // Writing to the newly committed page is detected by `scan_page`.
            *(base as *mut u8) = 0x5A;
            let mut dirty = Vec::new();
            let page0 = std::slice::from_raw_parts(base as *const u8, PAGE);
            scan_page(page0, base, &mut entry.sent[0], &mut dirty);
            assert_eq!(dirty, vec![(base, CHUNK)]);

            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A tile decommitted and committed again before the next submit gets its pool tile back
    /// with no new tile mapping (cleared instead). One that is not is unmapped and freed by
    /// `flush_parked` (before a submit), or when the free list runs out; a placed buffer's
    /// release does not free it while it is parked.
    #[test]
    fn a_tile_decommitted_and_committed_again_keeps_its_video_memory() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((2 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
                .expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let rb_queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("rb_queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let rb_done_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_done_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");

            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_queue,
                rb_fence,
                rb_value: 0,
                rb_done_fence,
                rb_done_value: 0,
                windows: vec![(base, reserved)],
                pools: Vec::new(),
                pool_is_sys: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                pending_unmaps: VecDeque::new(),
                pending_unmap_va: FxHashMap::default(),
                unmap_seq: 0,
                quarantined: VecDeque::new(),
                pending_quarantine: Vec::new(),
                queue_fences: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: Vec::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
            mapping_queued: std::sync::atomic::AtomicBool::new(false),
            mapped_value: 0,
                runs: None,
                scratch_runs: Vec::new(),
                retry_pages: Vec::new(),
            };

            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            let v = s.value;
            assert!(v > 0, "a new tile is mapped");

            // decommitted: parked, still mapped, nothing sent to the GPU
            s.decommit_range(base, TILE);
            assert!(!s.mapped.contains_key(&base));
            assert_eq!(s.parked.get(&base), Some(&tile));
            assert!(!s.free.contains(&tile));
            assert_eq!(s.value, v, "no tile mapping changed");
            s.release_if_unused(tile);
            assert!(!s.free.contains(&tile), "a parked tile is not free");

            // committed again: the same tile, cleared, no new mapping
            s.to_zero.clear();
            let gen = mapped_gen();
            s.commit_range(base, TILE);
            assert_eq!(s.mapped[&base].pool, tile);
            assert_eq!(s.mapped[&base].holes, 0);
            assert!(s.parked.is_empty());
            assert_eq!(s.to_zero, vec![(base, TILE)]);
            assert_eq!(s.value, v, "no tile mapping changed");
            assert!(mapped_gen() > gen, "readers of `mapped` see it changed");

            // decommitted and flushed (as before a submit): unmapped and free once unmap fence completes
            s.decommit_range(base, TILE);
            assert!(s.flush_parked());
            assert!(s.parked.is_empty());
            assert!(s.value > v, "the unmapping is sent");
            assert!(!s.flush_parked(), "nothing left to flush");
            s.wait_for(s.value);
            assert!(s.drain_quarantined());
            assert!(s.free.contains(&tile));

            // no free tile left: the parked one is flushed and reused once its unmap fence finishes
            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked());
            s.wait_for(s.value);
            s.commit_range(base + TILE, TILE);
            assert!(s.parked.is_empty());
            assert_eq!(s.mapped[&(base + TILE)].pool, tile);
            assert!(!s.mapped.contains_key(&base));

            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    struct SignalOnDrop(ID3D12Fence, u64);

    impl Drop for SignalOnDrop {
        fn drop(&mut self) {
            unsafe {
                let _ = self.0.Signal(self.1);
            }
        }
    }

    /// A title frees its tiles in runs spread over all its pools: the pools of the freed runs go
    /// back. With pools of 64 MiB none of these did (every pool kept the tiles still in use).
    /// (From upstream PR #2630.)
    #[test]
    fn pools_of_freed_runs_go_back() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            // 192 MiB of tiles, taken in order as a title's first commits take them
            const TILES: usize = 3072;
            let tiles: Vec<(u32, u32)> = (0..TILES).map(|_| s.take_tile().expect("a tile").0).collect();
            let made = s.pools.iter().filter(|p| p.is_some()).count();
            assert_eq!(made * POOL_TILES as usize, TILES, "the tiles fill whole pools");
            // of every 64 MiB the title keeps the first 4 MiB and frees the rest
            let (mut kept, mut freed) = (0, 0);
            for (i, &t) in tiles.iter().enumerate() {
                if i % 1024 < 64 {
                    kept += 1;
                } else {
                    s.free.push(t);
                    freed += 1;
                }
            }
            s.retire_free_pools();
            let left = s.pools.iter().filter(|p| p.is_some()).count() * POOL_TILES as usize;
            assert_eq!(s.free.len() + s.retired.len() * POOL_TILES as usize, freed, "every freed tile is free or its pool retired");
            assert!(left <= kept * 2, "{left} tiles of video memory stay in pools for {kept} tiles in use ({} MiB for {} MiB)", left * TILE >> 20, kept * TILE >> 20);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    unsafe fn test_state_with_reserved(tiles: usize) -> (usize, State) {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
        let mut dev: Option<ID3D12Device> = None;
        if std::env::var("XCL_WARP").ok().as_deref() == Some("1")
            || D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err()
        {
            let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
            D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
        }
        let dev = dev.expect("D3D12 device");
        test_state_with_device(dev, tiles)
    }

    unsafe fn test_state_with_device(dev: ID3D12Device, tiles: usize) -> (usize, State) {
        use windows_sys::Win32::System::Memory::MEM_RESERVE;

        let base = VirtualAlloc(std::ptr::null(), tiles * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
        assert!(base != 0 && base % TILE == 0);
        let mut reserved: Option<ID3D12Resource> = None;
        dev.CreateReservedResource(&buffer_desc((tiles * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
            .expect("CreateReservedResource");
        let reserved = reserved.expect("reserved buffer");
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
        let rb_queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("rb_queue");
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
        let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
        let rb_done_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_done_fence");
        let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
        let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
        let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
        rb_list.Close().expect("rb_list close");

        let s = State {
            dev,
            queue,
            fence,
            value: 0,
            event,
            rb_queue,
            rb_fence,
            rb_value: 0,
            rb_done_fence,
            rb_done_value: 0,
            windows: vec![(base, reserved)],
            pools: Vec::new(),
            pool_is_sys: Vec::new(),
            free: Vec::new(),
            retired: Vec::new(),
            mapped: FxHashMap::default(),
            pinned: FxHashMap::default(),
            parked: FxHashMap::default(),
            pending_unmaps: VecDeque::new(),
            pending_unmap_va: FxHashMap::default(),
            unmap_seq: 0,
            quarantined: VecDeque::new(),
            pending_quarantine: Vec::new(),
            queue_fences: FxHashMap::default(),
            buffers: FxHashMap::default(),
            to_zero: Vec::new(),
            staging: None,
            ring_write: 0,
            ring_submitted: 0,
            ring_free: 0,
            inflight: VecDeque::new(),
            zero_buf: None,
            slots: Vec::new(),
            next_slot: 0,
            readback: None,
            rb_alloc,
            rb_list,
            rb_last_fence: 0,
            waited: FxHashMap::default(),
            mapping_queued: std::sync::atomic::AtomicBool::new(false),
            mapped_value: 0,
            runs: None,
            scratch_runs: Vec::new(),
            retry_pages: Vec::new(),
        };
        (base, s)
    }

    #[test]
    fn sparse_stalled_upload_retries_only_unsent_chunks_without_a_new_cpu_write() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            assert_eq!(VirtualAlloc(base as *const c_void, TILE, MEM_COMMIT, PAGE_READWRITE) as usize, base);
            std::ptr::write_bytes(base as *mut u8, 0x5a, PAGE);
            s.commit_range(base, TILE);
            s.wait_for(s.value);
            s.sync_sent_range(base, PAGE);
            let before = *s.mapped[&base].sent[0].as_deref().unwrap();

            s.ring_write = STAGING as u64;
            s.ring_submitted = s.ring_write;
            s.inflight.push_back((s.value + 1, s.ring_write));
            let runs = [(base + CHUNK, 2 * CHUNK)];
            assert!(!s.upload_dirty_with_timeout(&runs, std::time::Duration::from_millis(20)));
            let after = s.mapped[&base].sent[0].as_deref().unwrap();
            assert_eq!(after[0], before[0]);
            assert_eq!(after[3], before[3]);

            let mut pending = Vec::new();
            s.scan_retry_pages(&mut pending);
            assert_eq!(pending, runs);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn sparse_staging_bounds_memory_and_transfers_larger_ranges() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let Some(dev) = test_device() else { return };
            let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut opts as *mut _ as *mut c_void, std::mem::size_of_val(&opts) as u32).is_err()
                || opts.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_1.0
            {
                crate::gpu::recomp::runner::skip_test("no tiled resources");
                return;
            }

            let bytes = (16 << 20) + TILE;
            let (base, mut s) = test_state_with_device(dev, bytes / TILE);
            let alloc: ID3D12CommandAllocator = s.dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = s.dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            list.Close().unwrap();
            s.slots.push(UploadSlot { alloc, list, fence_val: 0 });
            assert_eq!(VirtualAlloc(base as *const c_void, bytes, MEM_COMMIT, PAGE_READWRITE) as usize, base);
            s.commit_range(base, bytes);

            let mut expected = vec![0; bytes];
            for (i, tile) in expected.chunks_exact_mut(TILE).enumerate() {
                tile.fill((i % 251 + 1) as u8);
            }
            s.upload_runs(&[(base, bytes)], QUEUE_WAIT_TIMEOUT, &|at, dst, len| {
                expected[at - base..at - base + len].as_ptr().copy_to_nonoverlapping(dst, len);
                true
            }).unwrap();
            s.readback_ranges(&[(base, bytes)]);
            let copied = std::slice::from_raw_parts(base as *const u8, bytes);
            assert!(copied == expected, "bytes must survive across upload wraps and readback batches");

            let upload_bytes = s.staging.as_ref().unwrap().0.GetDesc().Width;
            let readback_bytes = s.readback.as_ref().unwrap().0.GetDesc().Width;
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
            assert_eq!((upload_bytes, readback_bytes), (16 << 20, 4 << 20));
        }
    }

    /// `flush_parked` must keep unmapped pool tiles out of `free` (and `take_tile` must not reuse
    /// them) while the sparse queue's unmap fence is still in flight on the GPU.
    #[test]
    fn flush_parked_does_not_free_or_reuse_tiles_while_unmap_is_in_flight() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.wait_for(s.value);

            let gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate fence");
            let _unblock = SignalOnDrop(gate.clone(), 1);
            s.queue.Wait(&gate, 1).expect("gate sparse queue");

            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked());
            assert!(!s.free.contains(&tile), "unmapped tile must remain quarantined until unmap fence completes");

            // While the unmap is still blocked on `gate`, committing another tile must not steal `tile`.
            s.commit_range(base + TILE, TILE);
            assert_ne!(s.mapped[&(base + TILE)].pool, tile, "in-flight unmapped tile must not be reused before GPU completion");

            gate.Signal(1).expect("release sparse queue");
            s.wait_for(s.value);
            assert!(s.drain_quarantined(), "completed unmap batch should drain after fence completion");
            assert!(s.free.contains(&tile), "unmapped tile must return to free list once unmap fence completes");
            drop(_unblock);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// `flush_parked` must keep unmapped pool tiles quarantined until every active title queue has
    /// finished its prior work, even after `s.queue` itself finishes unmapping.
    #[test]
    fn flush_parked_quarantines_tiles_until_title_queues_finish() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;

        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            let title_q: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");

            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.wait_for(s.value);

            let title_gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("title gate fence");
            let _unblock = SignalOnDrop(title_gate.clone(), 1);
            title_q.Wait(&title_gate, 1).expect("gate title queue");

            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked_for(&[&title_q]));

            // Even after `s.queue` finishes `unmap_tiles`, `title_q` is still gated behind `title_gate`.
            s.wait_for(s.value);
            assert!(!s.drain_quarantined(), "tile must remain quarantined while title queue has work in flight");
            assert!(!s.free.contains(&tile));

            s.commit_range(base + TILE, TILE);
            assert_ne!(s.mapped[&(base + TILE)].pool, tile, "must not reuse tile still in use by title queue");

            title_gate.Signal(1).expect("release title queue");
            let start = std::time::Instant::now();
            while !s.drain_quarantined() && start.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            assert!(s.free.contains(&tile), "tile must return to free list after title queue completes");
            drop(_unblock);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    unsafe fn com_refcount(q: &ID3D12CommandQueue) -> u32 {
        let raw = q.as_raw() as *mut *const *const c_void;
        let vt = *raw;
        let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(1));
        let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(2));
        let _ = add_ref(raw as *mut c_void);
        release(raw as *mut c_void)
    }

    #[test]
    fn flush_parked_does_not_retain_title_queue_com_refs() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;
        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            let title_q: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            let before = com_refcount(&title_q);
            s.commit_range(base, TILE);
            s.wait_for(s.value);
            s.decommit_range(base, TILE);
            assert!(s.flush_parked_for(&[&title_q]));
            assert_eq!(com_refcount(&title_q), before, "flush_parked must not retain a strong COM ref on title queues");
            assert_eq!(s.queue_fences.len(), 1);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn flush_parked_prunes_stale_queue_fences_while_inflight_batches_stay_gated() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;
        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            let title_q: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            let title_gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate fence");
            let _unblock = SignalOnDrop(title_gate.clone(), 1);
            title_q.Wait(&title_gate, 1).expect("gate title queue");

            s.commit_range(base, TILE);
            let tile_a = s.mapped[&base].pool;
            s.commit_range(base + TILE, TILE);
            let tile_b = s.mapped[&(base + TILE)].pool;
            s.wait_for(s.value);

            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked_for(&[&title_q]));
            assert_eq!(s.queue_fences.len(), 1);

            // Simulate title_q release before the next flush: stale fence entry is pruned,
            // while batch A keeps its own fence clone in `marks` and stays gated.
            s.decommit_range(base + TILE, TILE);
            assert!(s.flush_parked_for(&[]));
            assert!(s.queue_fences.is_empty(), "stale queue fence must be pruned");
            s.wait_for(s.value);
            s.drain_quarantined();
            assert!(!s.free.contains(&tile_a) && !s.pending_unmaps.is_empty(), "in-flight batch A must stay gated");

            title_gate.Signal(1).expect("release title queue");
            let start = std::time::Instant::now();
            while (!s.free.contains(&tile_a) || !s.free.contains(&tile_b)) && start.elapsed() < std::time::Duration::from_secs(5) {
                s.drain_quarantined();
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            assert!(s.free.contains(&tile_a) && s.free.contains(&tile_b), "both batches must drain after title_q completes");
            drop(_unblock);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn repeated_queue_quarantine_and_release_does_not_leak_queues_or_fences() {
        use windows_sys::Win32::System::Memory::MEM_RELEASE;
        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            for i in 0..80usize {
                let q: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap_or_else(|e| panic!("queue {i}: {e}"));
                let before = com_refcount(&q);
                s.commit_range(base, TILE);
                s.decommit_range(base, TILE);
                assert!(s.flush_parked_for(&[&q]));
                assert_eq!(com_refcount(&q), before, "queue {i} refcount leaked");
                drop(q);
                s.commit_range(base + TILE, TILE);
                s.decommit_range(base + TILE, TILE);
                assert!(s.flush_parked_for(&[]));
                assert!(s.queue_fences.is_empty(), "stale fence leaked at iteration {i}");
                s.wait_for(s.value);
                let start = std::time::Instant::now();
                while (!s.quarantined.is_empty() || !s.pending_unmaps.is_empty()) && start.elapsed() < std::time::Duration::from_secs(5) {
                    s.drain_quarantined();
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                assert!(s.quarantined.is_empty() && s.pending_unmaps.is_empty(), "quarantine failed to drain at iteration {i}");
            }
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// Sub-tile decommits (two 32 KiB halves, sixteen 4 KiB pages, or decommitting the last
    /// remaining committed page) must unmap the tile and return unpinned pool tiles to `free`
    /// once `holes` reaches `0xFFFF`, pruning stale `to_zero` entries for unmapped tiles.
    #[test]
    fn decommit_sub_tile_reclaims_pool_tile_when_all_pages_become_holes() {
        let mut mapped: FxHashMap<usize, TileEntry> = FxHashMap::default();
        let mut pinned: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut free: Vec<(u32, u32)> = Vec::new();
        let mut to_zero: Vec<(usize, usize)> = Vec::new();

        // 1. Two 32 KiB decommits across a full 64 KiB tile.
        let t0 = 0x10_0000usize;
        mapped.insert(t0, TileEntry::with_holes((0, 10), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0, TILE / 2).is_empty());
        assert_eq!(mapped[&t0].holes, 0x00FF);
        assert_eq!(to_zero.len(), 8);
        assert!(free.is_empty());

        let unmapped = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0 + TILE / 2, TILE / 2);
        assert_eq!(unmapped, vec![t0]);
        assert!(!mapped.contains_key(&t0));
        assert_eq!(free, vec![(0, 10)]);
        assert!(to_zero.is_empty(), "stale to_zero entries for the unmapped tile must be pruned");

        // 2. Sixteen individual 4 KiB decommits in reverse order.
        let t1 = 0x20_0000usize;
        free.clear();
        mapped.insert(t1, TileEntry::with_holes((0, 11), 0));
        for i in (1..PAGES_PER_TILE).rev() {
            let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1 + i * PAGE, PAGE);
            assert!(u.is_empty(), "tile still has committed pages after step {i}");
        }
        assert!(free.is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1, PAGE);
        assert_eq!(u, vec![t1]);
        assert!(!mapped.contains_key(&t1));
        assert_eq!(free, vec![(0, 11)]);
        assert!(to_zero.is_empty());

        // 3. Pinned tile decommitted in two halves: unmapped from `mapped`, NOT returned to `free`.
        let t2 = 0x30_0000usize;
        free.clear();
        mapped.insert(t2, TileEntry::with_holes((0, 12), 0));
        pinned.insert((0, 12), 1);
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2, TILE / 2).is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2 + TILE / 2, TILE / 2);
        assert_eq!(u, vec![t2]);
        assert!(!mapped.contains_key(&t2));
        assert!(free.is_empty(), "pinned pool tile must not return to free list");
        assert!(to_zero.is_empty());

        // 4. Single-committed-page tile decommit reclaims tile without dropping another tile's to_zero.
        let t3 = 0x40_0000usize;
        let t4 = 0x50_0000usize;
        mapped.insert(t3, TileEntry::with_holes((0, 13), 0xFFFF & !(1u16 << 2)));
        mapped.insert(t4, TileEntry::with_holes((0, 14), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t4, PAGE).is_empty());
        assert_eq!(to_zero, vec![(t4, PAGE)]);
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t3 + 2 * PAGE, PAGE);
        assert_eq!(u, vec![t3]);
        assert_eq!(free, vec![(0, 13)]);
        assert_eq!(to_zero, vec![(t4, PAGE)], "unrelated mapped tile's to_zero must remain");
    }

    /// The pages in `early`, checking that its ranges are page-aligned, not empty, disjoint and
    /// merged (no two touch).
    fn early_pages(early: &BTreeMap<usize, usize>) -> std::collections::BTreeSet<usize> {
        let mut last_end = None;
        let mut pages = std::collections::BTreeSet::new();
        for (&a, &b) in early {
            assert!(a % PAGE == 0 && b % PAGE == 0 && a < b, "range {a:#x}..{b:#x}");
            assert!(last_end.is_none_or(|e| e < a), "range {a:#x}..{b:#x} touches the one before");
            last_end = Some(b);
            pages.extend((a..b).step_by(PAGE));
        }
        pages
    }

    /// xorshift64*: the same sequence on every run
    fn next(x: &mut u64) -> u64 {
        *x ^= *x >> 12;
        *x ^= *x << 25;
        *x ^= *x >> 27;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Before the device exists, commits and decommits leave the pages a plain page set would
    /// hold, with the live path's rule (a range counts every page it touches): random ranges,
    /// unaligned too, over 300 pages.
    #[test]
    fn early_commits_are_the_pages_a_page_set_holds() {
        let base = 0x4_2000_0000usize;
        for seed in 1..=8u64 {
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let (mut early, mut model) = (BTreeMap::new(), std::collections::BTreeSet::new());
            for _ in 0..4000 {
                let a = base + (next(&mut x) % (300 * PAGE as u64)) as usize;
                let n = match next(&mut x) % 4 {
                    0 => PAGE,
                    1 => 1 + (next(&mut x) % 64) as usize,
                    _ => 1 + (next(&mut x) % (24 * PAGE as u64)) as usize,
                };
                let (start, end) = page_span(a, n);
                let pages = (start..end).step_by(PAGE);
                if next(&mut x) % 3 == 0 {
                    early_remove(&mut early, start, end);
                    pages.for_each(|p| {
                        model.remove(&p);
                    });
                } else {
                    early_add(&mut early, start, end);
                    model.extend(pages);
                }
                assert_eq!(early_pages(&early), model, "seed {seed}");
            }
        }
    }

    /// The same pages as the list this replaced, for what that list handled exactly: one
    /// entry per committed page (`XMemMapPhysicalPagesScatter`, `home_again`) and page-aligned
    /// decommits of any length.
    #[test]
    fn early_commits_match_the_old_list_for_page_entries() {
        let base = 0x4_2000_0000usize;
        let mut x = 0x1234_5678_9ABC_DEF1u64;
        let (mut early, mut old): (BTreeMap<usize, usize>, Vec<(usize, usize)>) = (BTreeMap::new(), Vec::new());
        for _ in 0..20000 {
            let a = base + (next(&mut x) % 512) as usize * PAGE;
            if next(&mut x) % 2 == 0 {
                old.push((a, PAGE));
                early_add(&mut early, a, a + PAGE);
            } else {
                let n = (1 + next(&mut x) % 8) as usize * PAGE;
                old.retain(|&(y, m)| y + m <= a || y >= a + n);
                early_remove(&mut early, a, a + n);
            }
        }
        let old_pages: std::collections::BTreeSet<usize> = old.iter().map(|&(y, _)| y).collect();
        assert_eq!(early_pages(&early), old_pages);
    }

    /// A range committed in one call keeps its other pages when one page of it is decommitted
    /// (`commit_homes` commits several pages, `move_home_away` gives one back). The list this
    /// replaced dropped the whole entry, so `init` left those still-committed pages as holes.
    #[test]
    fn early_decommit_of_one_page_keeps_the_rest() {
        let a = 0x4_2000_0000usize;
        let mut early = BTreeMap::new();
        early_add(&mut early, a, a + 4 * TILE);
        early_remove(&mut early, a + TILE, a + 2 * TILE);
        assert_eq!(early.into_iter().collect::<Vec<_>>(), vec![(a, a + TILE), (a + 2 * TILE, a + 4 * TILE)]);
    }

    /// A late completion from an earlier timed-out `SetEventOnCompletion` leaves the reusable
    /// auto-reset event signalled (or fires it mid-wait). `wait_fence_until` must ignore both
    /// and wait until the fence actually reaches `target`. (The test of #2821, for `main`.)
    #[test]
    fn wait_fence_until_ignores_stale_event_signal() {
        use std::time::{Duration, Instant};
        use windows::Win32::System::Threading::{CreateEventW, SetEvent};
        unsafe {
            let Some(dev) = test_device() else { return };
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let event = CreateEventW(None, false, false, None).expect("event");
            // a stale signal already set, then an earlier completion (2) firing mid-wait for 5
            SetEvent(event).expect("stale SetEvent");
            fence.SetEventOnCompletion(2, event).expect("stale completion arm");
            let f2 = fence.clone();
            let t = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                let _ = f2.Signal(2);
                std::thread::sleep(Duration::from_millis(50));
                let _ = f2.Signal(5);
            });
            let start = Instant::now();
            let ok = wait_fence_until(&fence, event, 5, Duration::from_secs(2));
            let elapsed = start.elapsed();
            t.join().unwrap();
            let _ = CloseHandle(event);
            assert!(ok, "wait_fence_until timed out instead of observing Signal(5)");
            assert!(fence.GetCompletedValue() >= 5, "returned early at {}", fence.GetCompletedValue());
            assert!(elapsed >= Duration::from_millis(50), "returned too early ({elapsed:?}) on a stale event signal");
        }
    }

    /// A `readback_now` job that timed out with its copy still queued must not be handed out
    /// (or freed) before that copy has run, then goes back to the pool.
    #[test]
    fn a_timed_out_readback_job_is_kept_until_its_copy_ran() {
        unsafe {
            let Some(dev) = test_device() else { return };
            let job = take_readback_job(&dev, 1).expect("readback job");
            let buf = job.buf.as_raw() as usize;
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            keep_in_flight_readback_job(job, fence.clone(), 1);
            let other = take_readback_job(&dev, 1).expect("another job");
            assert_ne!(other.buf.as_raw() as usize, buf, "a job whose copy is queued was handed out again");
            assert!(IN_FLIGHT_RB_JOBS.lock().unwrap().iter().any(|j| j.job.buf.as_raw() as usize == buf), "the queued job was dropped");
            return_readback_job(other);
            fence.Signal(1).expect("signal");
            reclaim_in_flight_readback_jobs();
            assert!(!IN_FLIGHT_RB_JOBS.lock().unwrap().iter().any(|j| j.job.buf.as_raw() as usize == buf), "the job stayed in flight after its copy");
            assert!(FREE_RB_JOBS.lock().unwrap().iter().any(|j| j.buf.as_raw() as usize == buf), "the job did not go back to the pool");
        }
    }

    /// A D3D12 device for the tests above (hardware, else WARP), or `None` after a skip notice.
    unsafe fn test_device() -> Option<ID3D12Device> {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_ok() {
            return dev;
        }
        let warp = CreateDXGIFactory1::<IDXGIFactory4>().ok().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>().ok());
        if let Some(warp) = warp {
            if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_ok() {
                return dev;
            }
        }
        crate::gpu::recomp::runner::skip_test("no GPU");
        None
    }

    /// Time for n 64 KiB pages committed one at a time and decommitted one at a time in
    /// reverse, before the device exists (`cargo test --release -p xcl --lib --
    /// --ignored --exact hle::d3d12x::sparse::tests::early_commit_timing --nocapture`,
    /// `XCL_EARLY_PAGES=16384,32768`). Run it alone: a test that made the device keeps a state.
    #[test]
    #[ignore]
    fn early_commit_timing() {
        if STATE.lock().unwrap().is_some() {
            return eprintln!("early_commit_timing: a sparse state exists – run it alone");
        }
        let sizes: Vec<usize> = std::env::var("XCL_EARLY_PAGES").ok().map_or(vec![4096, 8192, 16384, 32768], |v| v.split(',').map(|s| s.trim().parse().unwrap()).collect());
        let base = 0x4_2000_0000usize;
        for n in sizes {
            let t = std::time::Instant::now();
            for i in 0..n {
                unsafe { committed(base + i * TILE, TILE) };
            }
            let commit = t.elapsed();
            let t = std::time::Instant::now();
            for i in (0..n).rev() {
                unsafe { decommitted(base + i * TILE, TILE) };
            }
            let decommit = t.elapsed();
            println!("early: {n} pages: commit {:.1} ms, decommit {:.1} ms", commit.as_secs_f64() * 1e3, decommit.as_secs_f64() * 1e3);
        }
    }

    /// Releasing a placed buffer that was the last pin on an already-unmapped tile must queue the
    /// tile in `pending_quarantine` (never directly in `free`) and keep it out of `free` until
    /// `flush_pending_quarantine` + active title-queue fences complete (#2457).
    #[test]
    fn gpu_unpinned_unused_tile_waits_in_quarantine_before_free() {
        use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            let title_queue: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            let gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate fence");
            let _release_gate = SignalOnDrop(gate.clone(), 1);

            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            const BUF: usize = 0xDEAD_BEE0;
            s.pinned.insert(tile, 1);
            s.buffers.insert(BUF, vec![tile]);

            // Decommit and flush while still pinned by BUF: tile is unmapped from window once
            // idle title_queue passes its fence, NOT freed.
            s.decommit_range(base, TILE);
            assert!(s.flush_parked_for(&[&title_queue]));
            let start = std::time::Instant::now();
            while !s.pending_unmaps.is_empty() && start.elapsed() < std::time::Duration::from_secs(5) {
                s.drain_pending_unmaps();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(s.pending_unmaps.is_empty());
            assert!(!s.free.contains(&tile));
            assert!(s.quarantined.is_empty());

            // Stall title_queue behind `gate` to simulate in-flight GPU work reading BUF.
            title_queue.Wait(&gate, 1).expect("wait on gate");

            // Buffer release unpins tile: must enter pending_quarantine, NOT free.
            forget_in(&mut s, BUF);
            assert!(!s.pinned.contains_key(&tile));
            assert!(!s.free.contains(&tile), "unpinned tile must not bypass quarantine into free");
            assert_eq!(s.pending_quarantine, vec![tile]);

            // Flushing pending_quarantine attaches title_queue fence; tile stays quarantined until gate signals.
            assert!(s.flush_pending_quarantine_for(&[&title_queue]));
            assert!(s.pending_quarantine.is_empty());
            assert!(!s.drain_quarantined(), "tile must remain quarantined while title queue is blocked");
            assert!(!s.free.contains(&tile));

            gate.Signal(1).expect("signal gate");
            let start = std::time::Instant::now();
            while !s.drain_quarantined() && start.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(s.free.contains(&tile), "tile should enter free once title queue fence completes");

            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// `flush_parked` must defer `UpdateTileMappings(NULL)` until active title queues finish
    /// their prior work (preventing concurrent PTE unmapping while shaders run), and re-committing
    /// the VA before that fence completes must cancel the deferred unmap cleanly (including ABA).
    #[test]
    fn gpu_flush_parked_defers_null_unmap_and_recommit_cancels_stale_unmap() {
        use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
        unsafe {
            let (base, mut s) = test_state_with_reserved(2);
            let title_q: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");

            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.wait_for(s.value);
            let v_before = s.value;

            let gate1: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate1");
            let _unblock1 = SignalOnDrop(gate1.clone(), 1);
            title_q.Wait(&gate1, 1).expect("gate title_q 1");

            // Decommit + flush while title_q is busy: NULL unmap must be deferred, NOT queued on s.queue yet.
            s.decommit_range(base, TILE);
            assert!(s.flush_parked_for(&[&title_q]));
            assert_eq!(s.value, v_before, "UpdateTileMappings(NULL) must not be issued while title queue is in flight");
            assert_eq!(s.pending_unmaps.len(), 1);
            assert!(s.pending_unmap_va.contains_key(&base));

            // Re-commit same VA before gate1 signals: cancels pending unmap, keeps tile, zeros it, no GPU remap.
            s.to_zero.clear();
            s.commit_range(base, TILE);
            assert_eq!(s.mapped[&base].pool, tile);
            assert!(!s.pending_unmap_va.contains_key(&base));
            assert_eq!(s.to_zero, vec![(base, TILE)]);
            assert_eq!(s.value, v_before);

            // ABA check: decommit & flush again behind a second gate; when gate1 finishes first,
            // batch #1 must NOT unmap batch #2's tile.
            let gate2: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate2");
            let _unblock2 = SignalOnDrop(gate2.clone(), 1);
            title_q.Wait(&gate2, 1).expect("gate title_q 2");
            s.decommit_range(base, TILE);
            assert!(s.flush_parked_for(&[&title_q]));
            assert_eq!(s.pending_unmaps.len(), 2);

            gate1.Signal(1).expect("signal gate1");
            let start = std::time::Instant::now();
            while s.pending_unmaps.len() == 2 && start.elapsed() < std::time::Duration::from_secs(5) {
                s.drain_pending_unmaps();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert_eq!(s.pending_unmaps.len(), 1, "batch #1 drained cleanly");
            assert_eq!(s.value, v_before, "stale batch #1 must not unmap VA owned by batch #2");
            assert!(s.pending_unmap_va.contains_key(&base));

            // Releasing gate2 executes batch #2's deferred NULL unmap and returns tile to free.
            gate2.Signal(1).expect("signal gate2");
            let start = std::time::Instant::now();
            while !s.drain_quarantined() && start.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(s.value > v_before, "batch #2 must emit UpdateTileMappings(NULL) once title_q completes");
            assert!(s.pending_unmaps.is_empty() && s.pending_unmap_va.is_empty());
            assert!(s.free.contains(&tile));

            drop(_unblock2);
            drop(_unblock1);
            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn lazy_commit_clears_vram_only_holes_without_clobbering_to_zero() {
        let mut mapped = BTreeMap::new();
        let tile_a = 0x4_6eb2_0000usize;
        let tile_unmapped = 0x4_6eb3_0000usize;
        mapped.insert(tile_a, TileEntry { pool: (0, 0), holes: 0xFFFF, sent: std::array::from_fn(|_| None) });
        {
            let mut g = LAZY_COMMITS.lock().unwrap_or_else(|e| e.into_inner());
            g.clear();
            g.push(tile_a + 0x5920);
            g.push(tile_unmapped);
        }
        HAS_LAZY_COMMITS.store(true, Ordering::Release);

        // Drain via the exact same logic used by `State::drain_lazy_commits`:
        // mapped tile clears all holes (`0xFFFF -> 0`), unmapped tile is ignored cleanly.
        assert!(HAS_LAZY_COMMITS.swap(false, Ordering::AcqRel));
        let pages = std::mem::take(&mut *LAZY_COMMITS.lock().unwrap_or_else(|e| e.into_inner()));
        for page in pages {
            if let Some(entry) = mapped.get_mut(&(page & !(TILE - 1))) {
                entry.holes = 0;
            }
        }
        assert_eq!(mapped[&tile_a].holes, 0);
        assert!(!mapped.contains_key(&tile_unmapped));
        assert!(!HAS_LAZY_COMMITS.load(Ordering::Acquire));

        // Direct CPU write onto a reserved-only 64 KiB page marked with `kernelx::lazy::mark`
        // (matching `alloc_vram_only` + title clear loop at `+0x501eb50`) commits cleanly.
        unsafe {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
            let p = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(p, 0);
            crate::hle::kernelx::lazy::mark(p);
            let slice = std::slice::from_raw_parts_mut((p + 0x5920) as *mut u32, 0x100);
            slice.fill(0xFFFF_FFFF);
            assert_eq!(slice[0], 0xFFFF_FFFF);
            assert_eq!(slice[0xFF], 0xFFFF_FFFF);
            assert!(!crate::hle::kernelx::lazy::unmark(p), "page must already have been committed by VEH");
            assert_ne!(VirtualFree(p as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn readback_now_preserves_concurrent_cpu_writes_and_unaligned_batches() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        unsafe {
            // Allocate a real 64 KiB aligned tile with page 0 committed and page 1 left as a hole.
            let tile = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(tile != 0 && tile % TILE == 0);
            assert!(!VirtualAlloc(tile as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());

            let host = std::slice::from_raw_parts_mut(tile as *mut u8, PAGE);
            host[0..64].fill(0x11); // chunk 0: untouched during GPU wait -> should take GPU readback
            host[64..128].fill(0x22); // chunk 1: modified by CPU during GPU wait -> must NOT be clobbered
            host[128..192].fill(0x33); // chunk 2: split across two unaligned batches -> both halves should apply

            let mut mapped: FxHashMap<usize, TileEntry> = FxHashMap::default();
            mapped.insert(tile, TileEntry::with_holes((0, 0), 0xFFFF & !1u16));
            let mut dirty = Vec::new();
            scan_page(host, tile, &mut mapped.get_mut(&tile).unwrap().sent[0], &mut dirty);
            assert_eq!(dirty, vec![(tile, 192)]);

            // Phase-1 snapshot hashes for chunks 0..3.
            let mut snap_hashes = vec![
                chunk_hash(&host[0..64]),
                chunk_hash(&host[64..128]),
                chunk_hash(&host[128..192]),
            ];

            // Simulate concurrent CPU write to chunk 1 during the GPU wait.
            host[64 + 7] = 0xEE;

            // GPU readback buffer containing new GPU values for chunks 0..3.
            let mut gpu_rb = vec![0u8; 192];
            gpu_rb[0..64].fill(0xA1);
            gpu_rb[64..128].fill(0xB2);
            gpu_rb[128..160].fill(0xC3);
            gpu_rb[160..192].fill(0xD4);

            // Batch 1 covers [0..160) (splitting chunk 2 in half); Batch 2 covers [160..192).
            apply_readback_preserving_cpu_writes(tile, gpu_rb.as_ptr(), 160, tile, &mut snap_hashes, &mut mapped);
            apply_readback_preserving_cpu_writes(tile + 160, gpu_rb.as_ptr().add(160), 32, tile, &mut snap_hashes, &mut mapped);

            // Chunk 0 received GPU readback; chunk 1 preserved CPU write; chunk 2 received both batch halves.
            assert!(host[0..64].iter().all(|&b| b == 0xA1));
            assert_eq!(host[64 + 7], 0xEE, "concurrent CPU write in chunk 1 must not be clobbered");
            assert_eq!(host[64], 0x22, "rest of modified chunk 1 must stay intact");
            assert!(host[128..160].iter().all(|&b| b == 0xC3));
            assert!(host[160..192].iter().all(|&b| b == 0xD4));

            // Next `scan_page` must detect ONLY chunk 1 as dirty (chunks 0 and 2 had their `sent` hashes synced).
            dirty.clear();
            scan_page(host, tile, &mut mapped.get_mut(&tile).unwrap().sent[0], &mut dirty);
            assert_eq!(dirty, vec![(tile + 64, 64)]);

            assert_ne!(VirtualFree(tile as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn commit_range_unmasking_vram_only_tile_invalidates_runs_and_gen() {
        let mut mapped: FxHashMap<usize, TileEntry> = FxHashMap::default();
        let t0 = 0x2000_0000usize;
        mapped.insert(t0, TileEntry::with_holes((0, 0), 0xFFFF));

        // Prime cached watchable runs while tile is VRAM-only (all holes).
        let mut runs = Some(watchable_tile_runs(&mapped));
        assert_eq!(runs, Some(Vec::new()));

        let gen_before = MAPPED_GEN.load(Ordering::Acquire);
        assert!(unmask_committed_holes(&mut mapped, &mut runs, t0, PAGE));
        assert_eq!(mapped[&t0].holes, 0xFFFE);
        assert!(runs.is_none(), "cached runs must be invalidated when a VRAM-only tile becomes watchable");
        assert!(MAPPED_GEN.load(Ordering::Acquire) > gen_before, "MAPPED_GEN must increment on unmask");

        // Recomputing watchable runs now includes the unmasked tile.
        let recomputed = watchable_tile_runs(&mapped);
        assert_eq!(recomputed, vec![(t0, t0 + TILE)]);

        // A subsequent page commit on an already-watchable tile does not spuriously invalidate runs.
        runs = Some(recomputed);
        assert!(!unmask_committed_holes(&mut mapped, &mut runs, t0 + PAGE, PAGE));
        assert_eq!(mapped[&t0].holes, 0xFFFC);
        assert!(runs.is_some());
    }

    #[test]
    fn watchable_tile_runs_omits_vram_only_holes_until_unmasked() {
        let mut mapped: FxHashMap<usize, TileEntry> = FxHashMap::default();
        let t0 = 0x1000_0000usize;
        let t1 = t0 + TILE;
        let t2 = t1 + TILE;
        mapped.insert(t0, TileEntry::with_holes((0, 0), 0));
        mapped.insert(t1, TileEntry::with_holes((0, 1), 0xFFFF));
        mapped.insert(t2, TileEntry::with_holes((0, 2), 0x00FF));

        assert_eq!(watchable_tile_runs(&mapped), vec![(t0, t0 + TILE), (t2, t2 + TILE)]);

        mapped.get_mut(&t1).unwrap().holes = 0;
        assert_eq!(watchable_tile_runs(&mapped), vec![(t0, t2 + TILE)]);
    }

    /// Regression test for the circular GPU deadlock:
    /// `title_qA -> title_qB -> sparse.queue -> title_qA`.
    /// When `rb_queue` waits on a gated title queue `title_qA` (`rb_fence`), `sparse.queue`
    /// (uploads + tile mappings) must remain unblocked so `title_qB` waiting on `sparse.queue`
    /// can complete and release `title_qA`, unblocking the readback cleanly.
    #[test]
    fn readback_on_dedicated_queue_never_blocks_sparse_uploads_behind_gated_title_queue() {
        unsafe {
            let (base, mut s) = test_state_with_reserved(1);
            let title_q_a: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title_q_a");
            let title_q_b: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title_q_b");
            let cross_fence: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("cross_fence");

            // 1. title_q_a waits on title_q_b via cross_fence (not yet signalled by title_q_b).
            title_q_a.Wait(&cross_fence, 1).expect("qA waits on qB");

            // 2. A readback orders `rb_queue` behind title_q_a via `rb_fence` and signals `rb_done_fence`.
            s.rb_value += 1;
            let rb_v = s.rb_value;
            title_q_a.Signal(&s.rb_fence, rb_v).expect("qA signals rb_fence");
            s.rb_queue.Wait(&s.rb_fence, rb_v).expect("rb_queue waits on qA");
            s.signal_rb();
            let rb_target = s.rb_done_value;

            // 3. Now title_q_b submits work that waits on `sparse.queue` (uploads), then signals cross_fence.
            //    With the old shared-queue design, `sparse.queue` was stuck behind qA, causing a 3-way deadlock!
            s.signal();
            let upload_target = s.value;
            title_q_b.Wait(&s.fence, upload_target).expect("qB waits on sparse.queue");
            title_q_b.Signal(&cross_fence, 1).expect("qB releases qA");

            // Both `sparse.queue` AND `rb_queue` must complete well within 5 s (no deadlock!).
            s.wait_for(upload_target);
            assert!(s.fence.GetCompletedValue() >= upload_target, "sparse.queue must not be blocked by rb_queue");
            s.wait_for_rb(rb_target);
            assert!(s.rb_done_fence.GetCompletedValue() >= rb_target, "rb_queue must complete once qB unblocks qA");

            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, windows_sys::Win32::System::Memory::MEM_RELEASE), 0);
        }
    }
}
