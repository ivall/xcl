//! 4 KiB sub-allocated default heaps for small (< 64 KiB) sampled textures.
//!
//! A console title packs thousands of small mip-tails, icons, decals and LUTs at 256 B – 4 KiB
//! offsets inside its texture pools. Backing each one with a 64 KiB reserved resource in
//! `vram::Pages` wastes up to 16× video memory (one full 64 KiB `extra_pages` tile per small
//! texture) and submits a `CreateReservedResource` + `GetResourceTiling` + `UpdateTileMappings`
//! round trip for every placement.
//!
//! Direct3D 12 supports 4 KiB placement (`D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT`) for
//! single-sample non-RT/DS textures whose allocation info fits under 64 KiB. Placing those in
//! shared 16 MiB default heaps packs up to 4 096 textures into one heap with zero tile mappings,
//! shares the exact `(heap, offset)` across repeated placements at the same `(va, size)` so
//! same-address aliases see each other's uploads, and recycles released blocks by size class
//! after `IDLE` + `COOL`. `XCL_VRAM_SMALL=0` disables this path.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

/// Allocation unit: 4 KiB (`D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT`).
const UNIT: usize = 0x1000;
/// Largest small-resource size handled here: 60 KiB (`15 * 4 KiB < 64 KiB`).
const MAX_UNITS: u32 = 15;
/// Units per shared default heap: 16 MiB (`4096 * 4 KiB`).
const POOL_UNITS: u32 = 4096;
/// Keep an unreferenced `(va, units)` slot this long so immediate re-placements reuse it.
const IDLE: Duration = Duration::from_secs(2);
/// Wait before reusing a freed block for a different address (in-flight GPU work).
const COOL: Duration = Duration::from_secs(2);

/// Key identifying a shared small-texture backing slot: `(title_va, size_in_4k_units)`.
type SlotKey = (usize, u32);

/// Block location inside a device's small pools: `(pool_index, offset_in_4k_units, size_in_4k_units)`.
type Block = (u32, u32, u32);

/// GPU fence snapshot captured when a pooled block enters `cooling`. A block cannot be recycled
/// until BOTH wall-clock `COOL` has elapsed AND every captured title-queue fence has completed,
/// preventing premature aliasing across multi-second pipeline-compile stalls.
#[derive(Default)]
struct GpuMarks(Vec<(ID3D12Fence, u64)>);

unsafe impl Send for GpuMarks {}

impl GpuMarks {
    fn capture() -> Self {
        Self(super::alloc_pool::pending_gpu_marks())
    }

    fn is_complete(&self) -> bool {
        self.0.iter().all(|(f, v)| unsafe { f.GetCompletedValue() } >= *v)
    }
}

/// Cached local VRAM-pressure probe (refreshed every 250 ms) for idle fallback-heap eviction.
fn vram_pressure_tight() -> bool {
    static LAST_MS: AtomicU64 = AtomicU64::new(0);
    static TIGHT: AtomicBool = AtomicBool::new(false);
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let now_ms = START.get_or_init(Instant::now).elapsed().as_millis() as u64;
    let prev = LAST_MS.load(Ordering::Relaxed);
    if now_ms.saturating_sub(prev) >= 250 && LAST_MS.compare_exchange(prev, now_ms, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
        let tight = super::vram::commit_low()
            || super::real::video_budget().is_some_and(|(local, _)| local.Budget > 0 && local.CurrentUsage.saturating_mul(100) >= local.Budget.saturating_mul(85));
        TIGHT.store(tight, Ordering::Release);
        return tight;
    }
    TIGHT.load(Ordering::Acquire)
}

struct Pool<H> {
    heap: H,
    /// Bump pointer in allocation units.
    next_unit: u32,
    /// Number of blocks currently live or still cooling in this heap (0 => safe to evict).
    pinned_count: u32,
}

struct Slot {
    pool: u32,
    offset: u32,
    users: u32,
    idle_since: Option<Instant>,
}

struct SmallPools<H, G = ()> {
    pools: Vec<Pool<H>>,
    /// Per-size-class (`1..=MAX_UNITS`) reusable blocks `(pool, offset)`.
    free: [Vec<(u32, u32)>; (MAX_UNITS as usize) + 1],
    /// Blocks given back, reusable only after `COOL` AND GPU fence completion (`G`).
    cooling: VecDeque<(Block, Instant, G)>,
    /// Live or idle-kept `(va, units)` → backing block.
    slots: HashMap<SlotKey, Slot>,
    /// Idle slots, oldest first (`sweep_idle`).
    idle: VecDeque<(SlotKey, Instant)>,
    /// Live resource raw pointer → `(va, units)`.
    resources: HashMap<usize, SlotKey>,
    /// Total 4 KiB units currently held by `slots` (live + idle-kept).
    used_units: usize,
}

impl<H: Clone, G> SmallPools<H, G> {
    fn new() -> Self {
        SmallPools {
            pools: Vec::new(),
            free: std::array::from_fn(|_| Vec::new()),
            cooling: VecDeque::new(),
            slots: HashMap::new(),
            idle: VecDeque::new(),
            resources: HashMap::new(),
            used_units: 0,
        }
    }

    fn sweep_idle_with(&mut self, now: Instant, mut capture_gate: impl FnMut() -> G) {
        while self.idle.front().is_some_and(|&(_, since)| now.duration_since(since) >= IDLE) {
            let Some((key, since)) = self.idle.pop_front() else { break };
            if self.slots.get(&key).is_some_and(|s| s.idle_since == Some(since)) {
                if let Some(s) = self.slots.remove(&key) {
                    self.used_units -= key.1 as usize;
                    self.cooling.push_back(((s.pool, s.offset, key.1), now, capture_gate()));
                }
            }
        }
    }

    fn drain_cooling_with(&mut self, now: Instant, mut gpu_ready: impl FnMut(&G) -> bool) {
        while self.cooling.front().is_some_and(|(_, at, gate)| now.duration_since(*at) >= COOL && gpu_ready(gate)) {
            if let Some(((pool, offset, units), _, _)) = self.cooling.pop_front() {
                self.free[units as usize].push((pool, offset));
            }
        }
    }

    fn try_acquire_with(&mut self, key: SlotKey, now: Instant, capture_gate: impl FnMut() -> G, gpu_ready: impl FnMut(&G) -> bool) -> Option<(H, u32)> {
        self.sweep_idle_with(now, capture_gate);
        if let Some(s) = self.slots.get_mut(&key) {
            s.users += 1;
            s.idle_since = None;
            return Some((self.pools[s.pool as usize].heap.clone(), s.offset));
        }
        let units = key.1;
        debug_assert!((1..=MAX_UNITS).contains(&units));
        self.drain_cooling_with(now, gpu_ready);
        let (pool, offset) = if let Some(b) = self.free[units as usize].pop() {
            b
        } else if let Some((idx, p)) = self.pools.iter_mut().enumerate().rev().find(|(_, p)| p.next_unit + units <= POOL_UNITS) {
            let off = p.next_unit;
            p.next_unit += units;
            (idx as u32, off)
        } else {
            return None;
        };
        let heap = self.pools[pool as usize].heap.clone();
        self.slots.insert(key, Slot { pool, offset, users: 1, idle_since: None });
        self.used_units += units as usize;
        Some((heap, offset))
    }

    /// Register a newly created pool heap and acquire `(va, units)`.
    /// Re-checks `try_acquire` first under the lock so two threads racing to create a new 16 MiB
    /// pool do not push a redundant empty heap (the extra `heap` drops cleanly if another thread
    /// already added room or created the same slot).
    fn add_pool_and_acquire_with(&mut self, heap: H, key: SlotKey, now: Instant, mut capture_gate: impl FnMut() -> G, mut gpu_ready: impl FnMut(&G) -> bool) -> (H, u32) {
        if let Some(got) = self.try_acquire_with(key, now, &mut capture_gate, &mut gpu_ready) {
            return got;
        }
        let pool_idx = self.pools.len() as u32;
        self.pools.push(Pool { heap, next_unit: 0, pinned_count: 0 });
        if let Some(got) = self.try_acquire_with(key, now, &mut capture_gate, &mut gpu_ready) {
            return got;
        }
        // `pool_idx` has `next_unit == 0` and `units <= MAX_UNITS < POOL_UNITS`, so `try_acquire`
        // above always succeeds.
        unreachable!("fresh pool {pool_idx} must fit {} units", key.1);
    }

    fn note_resource(&mut self, res: usize, key: SlotKey) {
        self.resources.insert(res, key);
    }

    /// Release one user of `key`. When `rollback` is true (creation failed before the GPU ever saw
    /// the resource), the block returns directly to `free` if this was its only user.
    fn release_key(&mut self, key: SlotKey, now: Instant, rollback: bool) {
        let Some(s) = self.slots.get_mut(&key) else { return };
        s.users = s.users.saturating_sub(1);
        if s.users == 0 {
            if rollback {
                if let Some(s) = self.slots.remove(&key) {
                    self.used_units -= key.1 as usize;
                    self.free[key.1 as usize].push((s.pool, s.offset));
                }
            } else {
                s.idle_since = Some(now);
                self.idle.push_back((key, now));
            }
        }
    }

    fn forget(&mut self, res: usize, now: Instant) -> bool {
        let Some(key) = self.resources.remove(&res) else {
            return false;
        };
        self.release_key(key, now, false);
        true
    }

    fn bytes(&self) -> (usize, usize) {
        (self.used_units * UNIT, self.pools.len() * POOL_UNITS as usize * UNIT)
    }

    #[cfg(test)]
    fn flush_idle_with(&mut self, now: Instant, mut capture_gate: impl FnMut() -> G) {
        while let Some((key, since)) = self.idle.pop_front() {
            if self.slots.get(&key).is_some_and(|s| s.idle_since == Some(since)) {
                if let Some(s) = self.slots.remove(&key) {
                    self.used_units -= key.1 as usize;
                    self.cooling.push_back(((s.pool, s.offset, key.1), now, capture_gate()));
                }
            }
        }
    }
}

impl<H: Clone> SmallPools<H, ()> {
    #[cfg(test)]
    fn try_acquire(&mut self, key: SlotKey, now: Instant) -> Option<(H, u32)> { self.try_acquire_with(key, now, || (), |_| true) }
    #[cfg(test)]
    fn add_pool_and_acquire(&mut self, heap: H, key: SlotKey, now: Instant) -> (H, u32) { self.add_pool_and_acquire_with(heap, key, now, || (), |_| true) }
    #[cfg(test)]
    fn flush_idle(&mut self, now: Instant) { self.flush_idle_with(now, || ()); }
}

impl<H: Clone> SmallPools<H, GpuMarks> {
    fn try_acquire(&mut self, key: SlotKey, now: Instant) -> Option<(H, u32)> { self.try_acquire_with(key, now, GpuMarks::capture, GpuMarks::is_complete) }
    fn add_pool_and_acquire(&mut self, heap: H, key: SlotKey, now: Instant) -> (H, u32) { self.add_pool_and_acquire_with(heap, key, now, GpuMarks::capture, GpuMarks::is_complete) }
    #[cfg(test)]
    fn sweep_idle(&mut self, now: Instant) { self.sweep_idle_with(now, GpuMarks::capture); }
    #[cfg(test)]
    fn flush_idle(&mut self, now: Instant) { self.flush_idle_with(now, GpuMarks::capture); }
}

// D3D12 heaps are free-threaded.
unsafe impl Send for SmallPools<ID3D12Heap, GpuMarks> {}

static SMALL_POOLS: Mutex<BTreeMap<usize, SmallPools<ID3D12Heap, GpuMarks>>> = Mutex::new(BTreeMap::new());

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_VRAM_SMALL").map_or(true, |v| v.trim() != "0"))
}

/// Whether `desc` and `console_bytes` are eligible for 4 KiB small-resource sub-allocation.
fn candidate(desc: &D3D12_RESOURCE_DESC, console_bytes: Option<u64>) -> bool {
    const PAGE: u64 = 0x1_0000;
    desc.Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN
        && desc.SampleDesc.Count <= 1
        && desc.Flags.0 & 0x7 == 0
        && matches!(desc.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE1D | D3D12_RESOURCE_DIMENSION_TEXTURE2D | D3D12_RESOURCE_DIMENSION_TEXTURE3D)
        && console_bytes.is_some_and(|b| b > 0 && b < PAGE)
}

/// Place a sub-64 KiB sampled texture at 4 KiB granularity in a shared default heap.
/// Returns `None` if the texture is not a small resource (the caller falls back to `vram::Pages`
/// or a committed resource).
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn place(
    dev: &ID3D12Device,
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    if !enabled() || !candidate(desc, console_bytes) {
        return None;
    }
    let small_desc = D3D12_RESOURCE_DESC { Alignment: D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT as u64, ..*desc };
    let info = dev.GetResourceAllocationInfo(0, &[small_desc]);
    if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 || info.Alignment != UNIT as u64 || info.SizeInBytes >= 0x1_0000 {
        return None;
    }
    let units = (info.SizeInBytes as usize).div_ceil(UNIT) as u32;
    if !(1..=MAX_UNITS).contains(&units) {
        return None;
    }
    let dev_key = dev.as_raw() as usize;
    let key = (va, units);
    let now = Instant::now();

    let (heap, offset_units) = {
        let mut g = SMALL_POOLS.lock().unwrap();
        let pools = g.entry(dev_key).or_insert_with(SmallPools::new);
        pools.try_acquire(key, now)
    }
    .or_else(|| {
        // Create a new 16 MiB default heap outside `SMALL_POOLS.lock()`.
        let heap_desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_UNITS as u64 * UNIT as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            // 64 KiB heap alignment is valid for 4 KiB-aligned small resources at any 4 KiB offset.
            Alignment: D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut new_heap: Option<ID3D12Heap> = None;
        if let Err(e) = super::title_textures::own_work(|| dev.CreateHeap(&heap_desc, &mut new_heap)) {
            crate::xlog_warn!("D3D12.X: small texture pool CreateHeap: {e}");
            return None;
        }
        let new_heap = new_heap?;
        let mut g = SMALL_POOLS.lock().unwrap();
        let pools = g.entry(dev_key).or_insert_with(SmallPools::new);
        Some(pools.add_pool_and_acquire(new_heap, key, Instant::now()))
    })?;

    let hr = super::real::dev_create_placed_resource(
        this,
        heap.as_raw() as usize,
        offset_units as usize * UNIT,
        &small_desc as *const _ as usize,
        state as usize,
        clear as usize,
        riid,
        ppv,
    );
    if hr < 0 || (*ppv).is_null() {
        let mut g = SMALL_POOLS.lock().unwrap();
        if let Some(pools) = g.get_mut(&dev_key) {
            pools.release_key(key, Instant::now(), true);
        }
        return None;
    }
    let raw = *ppv as usize;
    SMALL_POOLS.lock().unwrap().entry(dev_key).or_insert_with(SmallPools::new).note_resource(raw, key);
    Some(hr)
}

/// Release a small-pool texture on COM release. Returns `true` if `resource` belonged to a small
/// pool (so `vram::forget` can skip locking `PAGES`).
pub(super) fn forget(resource: usize) -> bool {
    let mut g = SMALL_POOLS.lock().unwrap();
    let now = Instant::now();
    for pools in g.values_mut() {
        if pools.forget(resource, now) {
            return true;
        }
    }
    false
}

/// 64 KiB allocation unit (`D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT`) for default-heap fallback resources.
const FALLBACK_UNIT: usize = 0x1_0000;
/// Largest fallback resource pooled here: 4 MiB (`64 * 64 KiB`).
const FALLBACK_MAX_UNITS: u32 = 64;
/// Units per shared 16 MiB default fallback heap (`256 * 64 KiB`).
const FALLBACK_POOL_UNITS: u32 = 256;

/// Sub-allocated 16 MiB `D3D12_HEAP_TYPE_DEFAULT` heaps (`D3D12_HEAP_FLAG_NONE`, ResourceHeapTier 2)
/// replacing per-resource `CreateCommittedResource` calls on the non-reservable fallback path
/// (e.g., `TEXTURE1D`, 3D packed-mip textures, and small non-UMA buffers).
struct DefaultFallbackPools<H, G = ()> {
    pools: Vec<Option<Pool<H>>>,
    /// Per-size-class (`1..=FALLBACK_MAX_UNITS`) reusable blocks `(pool, offset)`.
    free: [Vec<(u32, u32)>; (FALLBACK_MAX_UNITS as usize) + 1],
    /// Released blocks waiting BOTH `COOL` AND GPU fence completion (`G`) before reuse.
    cooling: VecDeque<(Block, Instant, G)>,
    /// Live resource raw pointer -> `(pool, offset_units, size_units)`.
    resources: HashMap<usize, Block>,
    /// Total 64 KiB units currently backing live resources.
    used_units: usize,
}

impl<H: Clone, G> DefaultFallbackPools<H, G> {
    fn new() -> Self {
        DefaultFallbackPools {
            pools: Vec::new(),
            free: std::array::from_fn(|_| Vec::new()),
            cooling: VecDeque::new(),
            resources: HashMap::new(),
            used_units: 0,
        }
    }

    fn drain_cooling_with(&mut self, now: Instant, mut gpu_ready: impl FnMut(&G) -> bool) {
        while self.cooling.front().is_some_and(|(_, at, gate)| now.duration_since(*at) >= COOL && gpu_ready(gate)) {
            if let Some(((pool, offset, units), _, _)) = self.cooling.pop_front() {
                if let Some(p) = self.pools.get_mut(pool as usize).and_then(|o| o.as_mut()) {
                    p.pinned_count = p.pinned_count.saturating_sub(1);
                    self.free[units as usize].push((pool, offset));
                }
            }
        }
    }

    /// Evict completely empty 16 MiB heaps (`pinned_count == 0`, zero live or cooling blocks)
    /// when `evict` is true (e.g. local VRAM > 85% budget). Caller drops returned heaps outside locks.
    fn sweep_idle_heaps(&mut self, evict: bool) -> Vec<H> {
        if !evict {
            return Vec::new();
        }
        let mut dropped = Vec::new();
        for (idx, slot) in self.pools.iter_mut().enumerate() {
            if slot.as_ref().is_some_and(|p| p.pinned_count == 0) {
                if let Some(p) = slot.take() {
                    dropped.push(p.heap);
                    let pidx = idx as u32;
                    for list in &mut self.free {
                        list.retain(|&(b_pool, _)| b_pool != pidx);
                    }
                }
            }
        }
        dropped
    }

    fn try_acquire_with(&mut self, units: u32, now: Instant, gpu_ready: impl FnMut(&G) -> bool) -> Option<(H, Block)> {
        debug_assert!((1..=FALLBACK_MAX_UNITS).contains(&units));
        self.drain_cooling_with(now, gpu_ready);
        let (pool, offset) = loop {
            if let Some((pidx, off)) = self.free[units as usize].pop() {
                if self.pools.get(pidx as usize).and_then(|o| o.as_ref()).is_some() {
                    break (pidx, off);
                }
                continue;
            }
            if let Some((idx, p)) = self
                .pools
                .iter_mut()
                .enumerate()
                .rev()
                .filter_map(|(i, o)| o.as_mut().map(|p| (i, p)))
                .find(|(_, p)| p.next_unit + units <= FALLBACK_POOL_UNITS)
            {
                let off = p.next_unit;
                p.next_unit += units;
                break (idx as u32, off);
            }
            return None;
        };
        let p = self.pools[pool as usize].as_mut().expect("acquired pool must exist");
        p.pinned_count += 1;
        let heap = p.heap.clone();
        self.used_units += units as usize;
        Some((heap, (pool, offset, units)))
    }

    fn add_pool_and_acquire_with(&mut self, heap: H, units: u32, now: Instant, mut gpu_ready: impl FnMut(&G) -> bool) -> (H, Block) {
        if let Some(got) = self.try_acquire_with(units, now, &mut gpu_ready) {
            return got;
        }
        let new_pool = Pool { heap, next_unit: 0, pinned_count: 0 };
        let pool_idx = if let Some((idx, slot)) = self.pools.iter_mut().enumerate().find(|(_, o)| o.is_none()) {
            *slot = Some(new_pool);
            idx as u32
        } else {
            let idx = self.pools.len() as u32;
            self.pools.push(Some(new_pool));
            idx
        };
        if let Some(got) = self.try_acquire_with(units, now, &mut gpu_ready) {
            return got;
        }
        unreachable!("fresh fallback pool {pool_idx} must fit {units} units");
    }

    fn rollback(&mut self, block: Block) {
        self.used_units -= block.2 as usize;
        if let Some(p) = self.pools.get_mut(block.0 as usize).and_then(|o| o.as_mut()) {
            p.pinned_count = p.pinned_count.saturating_sub(1);
            self.free[block.2 as usize].push((block.0, block.1));
        }
    }

    fn note_resource(&mut self, res: usize, block: Block) {
        self.resources.insert(res, block);
    }

    fn forget_with(&mut self, res: usize, now: Instant, capture_gate: impl FnOnce() -> G) -> bool {
        let Some(block) = self.resources.remove(&res) else {
            return false;
        };
        self.used_units -= block.2 as usize;
        self.cooling.push_back((block, now, capture_gate()));
        true
    }

    fn active_pool_count(&self) -> usize {
        self.pools.iter().filter(|o| o.is_some()).count()
    }

    fn bytes(&self) -> (usize, usize) {
        (self.used_units * FALLBACK_UNIT, self.active_pool_count() * FALLBACK_POOL_UNITS as usize * FALLBACK_UNIT)
    }
}

impl<H: Clone> DefaultFallbackPools<H, ()> {
    #[cfg(test)]
    fn try_acquire(&mut self, units: u32, now: Instant) -> Option<(H, Block)> { self.try_acquire_with(units, now, |_| true) }
    #[cfg(test)]
    fn add_pool_and_acquire(&mut self, heap: H, units: u32, now: Instant) -> (H, Block) { self.add_pool_and_acquire_with(heap, units, now, |_| true) }
    #[cfg(test)]
    fn forget(&mut self, res: usize, now: Instant) -> bool { self.forget_with(res, now, || ()) }
}

impl<H: Clone> DefaultFallbackPools<H, GpuMarks> {
    fn try_acquire(&mut self, units: u32, now: Instant) -> Option<(H, Block)> { self.try_acquire_with(units, now, GpuMarks::is_complete) }
    fn add_pool_and_acquire(&mut self, heap: H, units: u32, now: Instant) -> (H, Block) { self.add_pool_and_acquire_with(heap, units, now, GpuMarks::is_complete) }
    fn forget(&mut self, res: usize, now: Instant) -> bool { self.forget_with(res, now, GpuMarks::capture) }
}

unsafe impl Send for DefaultFallbackPools<ID3D12Heap, GpuMarks> {}

static FALLBACK_POOLS: Mutex<BTreeMap<usize, DefaultFallbackPools<ID3D12Heap, GpuMarks>>> = Mutex::new(BTreeMap::new());
static FALLBACK_LIVE: AtomicUsize = AtomicUsize::new(0);
static TIER2_CACHE: Mutex<BTreeMap<usize, bool>> = Mutex::new(BTreeMap::new());

fn fallback_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_VRAM_FALLBACK_POOL").map_or(true, |v| v.trim() != "0"))
}

fn tier2_cached(dev: &ID3D12Device) -> bool {
    let dev_key = dev.as_raw() as usize;
    if let Some(&ok) = TIER2_CACHE.lock().unwrap().get(&dev_key) {
        return ok;
    }
    let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    let ok = unsafe {
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut opts as *mut _ as *mut c_void, std::mem::size_of_val(&opts) as u32).is_ok()
            && opts.ResourceHeapTier.0 >= 2
    };
    TIER2_CACHE.lock().unwrap().insert(dev_key, ok);
    ok
}

fn fallback_candidate(desc: &D3D12_RESOURCE_DESC) -> bool {
    let max_bytes = FALLBACK_MAX_UNITS as u64 * FALLBACK_UNIT as u64;
    if desc.SampleDesc.Count > 1 {
        return false;
    }
    match desc.Dimension {
        D3D12_RESOURCE_DIMENSION_BUFFER => desc.Width > 0 && desc.Width <= max_bytes,
        D3D12_RESOURCE_DIMENSION_TEXTURE1D | D3D12_RESOURCE_DIMENSION_TEXTURE2D | D3D12_RESOURCE_DIMENSION_TEXTURE3D => true,
        _ => false,
    }
}

/// Place a non-reservable fallback resource (`TEXTURE1D`, packed-mip 3D texture, small non-UMA buffer, etc.)
/// into a shared 16 MiB `D3D12_HEAP_TYPE_DEFAULT` heap via `CreatePlacedResource` instead of paying a
/// per-resource `CreateCommittedResource` WDDM allocation. Returns `None` to fall back to `CreateCommittedResource`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn place_fallback(
    this: *mut c_void,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    if !fallback_enabled() || !fallback_candidate(desc) {
        return None;
    }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2_cached(dev) {
        return None;
    }
    let placed_desc = D3D12_RESOURCE_DESC { Alignment: 0, ..*desc };
    let info = dev.GetResourceAllocationInfo(0, &[placed_desc]);
    if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 || info.Alignment == 0 || info.Alignment > FALLBACK_UNIT as u64 {
        return None;
    }
    let units = (info.SizeInBytes as usize).div_ceil(FALLBACK_UNIT) as u32;
    if !(1..=FALLBACK_MAX_UNITS).contains(&units) {
        return None;
    }
    let dev_key = dev.as_raw() as usize;
    let now = Instant::now();

    let (heap, block) = {
        let mut g = FALLBACK_POOLS.lock().unwrap();
        let pools = g.entry(dev_key).or_insert_with(DefaultFallbackPools::new);
        pools.try_acquire(units, now)
    }
    .or_else(|| {
        let heap_desc = D3D12_HEAP_DESC {
            SizeInBytes: FALLBACK_POOL_UNITS as u64 * FALLBACK_UNIT as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64,
            Flags: D3D12_HEAP_FLAG_NONE,
        };
        let mut new_heap: Option<ID3D12Heap> = None;
        if let Err(e) = super::title_textures::own_work(|| dev.CreateHeap(&heap_desc, &mut new_heap)) {
            crate::xlog_warn!("D3D12.X: default fallback pool CreateHeap: {e}");
            return None;
        }
        let new_heap = new_heap?;
        let mut g = FALLBACK_POOLS.lock().unwrap();
        let pools = g.entry(dev_key).or_insert_with(DefaultFallbackPools::new);
        Some(pools.add_pool_and_acquire(new_heap, units, Instant::now()))
    })?;

    let hr = super::real::dev_create_placed_resource(
        this,
        heap.as_raw() as usize,
        block.1 as usize * FALLBACK_UNIT,
        &placed_desc as *const _ as usize,
        state as usize,
        clear as usize,
        riid,
        ppv,
    );
    if hr < 0 || (*ppv).is_null() {
        let mut g = FALLBACK_POOLS.lock().unwrap();
        if let Some(pools) = g.get_mut(&dev_key) {
            pools.rollback(block);
        }
        return None;
    }
    let raw = *ppv as usize;
    FALLBACK_POOLS.lock().unwrap().entry(dev_key).or_insert_with(DefaultFallbackPools::new).note_resource(raw, block);
    FALLBACK_LIVE.fetch_add(1, Ordering::Release);
    Some(hr)
}

/// Release a default-fallback-pool resource on COM release.
pub(super) fn forget_fallback(resource: usize) -> bool {
    if FALLBACK_LIVE.load(Ordering::Acquire) == 0 {
        return false;
    }
    let tight = vram_pressure_tight();
    let (found, dropped_heaps) = {
        let mut g = FALLBACK_POOLS.lock().unwrap();
        let now = Instant::now();
        let mut found = false;
        let mut dropped = Vec::new();
        for pools in g.values_mut() {
            if !found && pools.forget(resource, now) {
                FALLBACK_LIVE.fetch_sub(1, Ordering::Relaxed);
                found = true;
            }
            if tight {
                pools.drain_cooling_with(now, GpuMarks::is_complete);
                dropped.extend(pools.sweep_idle_heaps(true));
            }
        }
        (found, dropped)
    };
    drop(dropped_heaps);
    found
}

/// `(used_bytes, pool_bytes)` across all devices (4 KiB small pools + 64 KiB default fallback pools).
pub(super) fn bytes() -> (usize, usize) {
    let (su, sp) = {
        let g = SMALL_POOLS.lock().unwrap();
        g.values().fold((0, 0), |(used, pools), p| {
            let (u, t) = p.bytes();
            (used + u, pools + t)
        })
    };
    let (fu, fp) = {
        let g = FALLBACK_POOLS.lock().unwrap();
        g.values().fold((0, 0), |(used, pools), p| {
            let (u, t) = p.bytes();
            (used + u, pools + t)
        })
    };
    (su + fu, sp + fp)
}

#[cfg(test)]
pub(super) fn fallback_bytes_for(device: usize) -> (usize, usize) {
    let g = FALLBACK_POOLS.lock().unwrap();
    g.get(&device).map_or((0, 0), DefaultFallbackPools::bytes)
}

#[cfg(test)]
pub(super) fn bytes_for(device: usize) -> (usize, usize) {
    let mut g = SMALL_POOLS.lock().unwrap();
    let Some(p) = g.get_mut(&device) else { return (0, 0) };
    p.sweep_idle(Instant::now());
    p.bytes()
}

#[cfg(test)]
pub(super) fn flush_idle_for(device: usize) {
    let mut g = SMALL_POOLS.lock().unwrap();
    if let Some(p) = g.get_mut(&device) {
        p.flush_idle(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_pool_packs_shares_and_recycles_by_size_class() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut p: SmallPools<u32> = SmallPools::new();
        let alloc = |p: &mut SmallPools<u32>, va: usize, units: u32, res: usize, now: Instant| -> (u32, u32) {
            let got = p.try_acquire((va, units), now).unwrap_or_else(|| {
                let id = p.pools.len() as u32;
                p.add_pool_and_acquire(id, (va, units), now)
            });
            p.note_resource(res, (va, units));
            got
        };

        // Sixteen 4 KiB textures pack into 64 KiB (not 16 * 64 KiB = 1 MiB).
        for i in 0..16usize {
            let (h, off) = alloc(&mut p, 0x10_0000 + i * UNIT, 1, 100 + i, at(0));
            assert_eq!((h, off), (0, i as u32));
        }
        assert_eq!(p.bytes(), (16 * UNIT, POOL_UNITS as usize * UNIT));

        // Re-placing the exact same (va, units) shares the exact same (heap, offset) with zero growth.
        let (h_dup, off_dup) = alloc(&mut p, 0x10_4000, 1, 200, at(10));
        assert_eq!((h_dup, off_dup), (0, 4));
        assert_eq!(p.bytes().0, 16 * UNIT);

        // Releasing one of the two aliases keeps the slot alive; releasing both makes it idle.
        assert!(p.forget(200, at(20)));
        assert_eq!(p.slots[&(0x10_4000, 1)].users, 1);
        assert!(p.forget(104, at(30)));
        assert_eq!(p.slots[&(0x10_4000, 1)].users, 0);

        // Re-acquiring within IDLE reuses the exact same slot.
        let (h_re, off_re) = alloc(&mut p, 0x10_4000, 1, 201, at(1000));
        assert_eq!((h_re, off_re), (0, 4));
        assert!(p.forget(201, at(1100)));

        // After IDLE (2 s) + COOL (2 s), the 4 KiB block is recycled for a new address.
        let (h_wait, off_wait) = alloc(&mut p, 0x20_0000, 1, 202, at(3200));
        assert_eq!((h_wait, off_wait), (0, 16), "still cooling at +2.1s, must not reuse yet");
        let (h_rec, off_rec) = alloc(&mut p, 0x20_1000, 1, 203, at(5300));
        assert_eq!((h_rec, off_rec), (0, 4), "block (0, 4) recycled after IDLE + COOL");

        // Rollback on creation failure immediately returns the block to the free list.
        let (h_fail, off_fail) = p.try_acquire((0x30_0000, 4), at(5400)).unwrap();
        assert_eq!((h_fail, off_fail), (0, 17));
        p.release_key((0x30_0000, 4), at(5400), true);
        let (h_next, off_next) = alloc(&mut p, 0x30_4000, 4, 204, at(5400));
        assert_eq!((h_next, off_next), (0, 17));

        // Concurrent CreateHeap race: if room already exists in pool 0, `add_pool_and_acquire`
        // must reuse existing room instead of pushing a redundant empty 16 MiB pool.
        let pools_before = p.pools.len();
        let (h_raced, _) = p.add_pool_and_acquire(999, (0x40_0000, 1), at(5500));
        assert_eq!(h_raced, 0);
        assert_eq!(p.pools.len(), pools_before, "must not push redundant pool when room already exists");

        // `flush_idle` clears unreferenced idle slots while keeping live (`users > 0`) slots intact.
        let used_before_flush = p.bytes().0;
        let _ = alloc(&mut p, 0x50_0000, 2, 205, at(5600));
        assert!(p.forget(205, at(5610)));
        assert_eq!(p.bytes().0, used_before_flush + 2 * UNIT);
        p.flush_idle(at(5620));
        assert_eq!(p.bytes().0, used_before_flush, "flush_idle must reclaim unreferenced slots immediately");
    }

    #[test]
    fn fallback_pool_blocks_recycle_until_gpu_fence_completes_even_after_cool() {
        use std::sync::Arc;
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut p: DefaultFallbackPools<u32, Arc<AtomicBool>> = DefaultFallbackPools::new();
        let (h0, b0) = p.add_pool_and_acquire_with(10, 4, at(0), |done| done.load(Ordering::Acquire));
        assert_eq!((h0, b0), (10, (0, 0, 4)));
        p.note_resource(0x1001, b0);

        let gpu_done = Arc::new(AtomicBool::new(false));
        assert!(p.forget_with(0x1001, at(10), || gpu_done.clone()));

        let (h1, b1) = p.try_acquire_with(4, at(10_010), |done| done.load(Ordering::Acquire)).unwrap();
        assert_eq!((h1, b1), (10, (0, 4, 4)), "must not alias in-flight GPU block even after COOL");

        gpu_done.store(true, Ordering::Release);
        let (h2, b2) = p.try_acquire_with(4, at(10_020), |done| done.load(Ordering::Acquire)).unwrap();
        assert_eq!((h2, b2), (10, (0, 0, 4)), "completed GPU block must recycle after fence + COOL");
    }

    #[test]
    fn small_pool_blocks_recycle_until_gpu_fence_completes_even_after_cool() {
        use std::sync::Arc;
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let gpu_done = Arc::new(AtomicBool::new(false));
        let mut p: SmallPools<u32, Arc<AtomicBool>> = SmallPools::new();

        let (h0, off0) = p.add_pool_and_acquire_with(1, (0x1000, 2), at(0), || gpu_done.clone(), |d| d.load(Ordering::Acquire));
        assert_eq!((h0, off0), (1, 0));
        p.note_resource(0x2001, (0x1000, 2));
        assert!(p.forget(0x2001, at(10)));

        p.sweep_idle_with(at(2_500), || gpu_done.clone());
        let (h1, off1) = p.try_acquire_with((0x2000, 2), at(12_500), || gpu_done.clone(), |d| d.load(Ordering::Acquire)).unwrap();
        assert_eq!((h1, off1), (1, 2), "must not alias cooling small-pool block before GPU fence completes");

        gpu_done.store(true, Ordering::Release);
        let (h2, off2) = p.try_acquire_with((0x3000, 2), at(12_600), || gpu_done.clone(), |d| d.load(Ordering::Acquire)).unwrap();
        assert_eq!((h2, off2), (1, 0));
    }

    #[test]
    fn fallback_pool_evicts_empty_heaps_under_vram_pressure() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut p: DefaultFallbackPools<u32, bool> = DefaultFallbackPools::new();
        let (_, b0) = p.add_pool_and_acquire_with(42, 8, at(0), |&done| done);
        p.note_resource(0x3001, b0);
        assert_eq!(p.active_pool_count(), 1);
        assert!(p.forget_with(0x3001, at(10), || false));

        p.drain_cooling_with(at(5_000), |&done| done);
        assert!(p.sweep_idle_heaps(true).is_empty(), "heap with cooling block must stay pinned");
        assert_eq!(p.active_pool_count(), 1);

        p.drain_cooling_with(at(5_000), |_| true);
        let evicted = p.sweep_idle_heaps(true);
        assert_eq!(evicted, vec![42]);
        assert_eq!(p.active_pool_count(), 0);
        assert_eq!(p.bytes(), (0, 0));
    }

    #[test]
    fn candidate_accepts_small_texture1d_for_4k_small_pool() {
        let mut d = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE1D,
            Width: 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        };
        assert!(candidate(&d, Some(4096)), "small sampled TEXTURE1D must qualify for 4 KiB SmallPools");
        assert!(!candidate(&d, Some(64 * 1024)), "64 KiB TEXTURE1D stays on default placement");
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        assert!(!candidate(&d, Some(4096)), "UAV TEXTURE1D must not enter SmallPools");
    }

    #[test]
    fn candidate_filters_large_gpu_owned_and_msaa_textures() {
        let mut d = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        };
        assert!(candidate(&d, Some(16 * 1024)));
        assert!(!candidate(&d, Some(64 * 1024)), "64 KiB stays on vram::Pages");
        assert!(!candidate(&d, None));
        assert!(!candidate(&d, Some(0)));
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        assert!(!candidate(&d, Some(16 * 1024)));
        d.Flags = D3D12_RESOURCE_FLAG_NONE;
        d.SampleDesc.Count = 4;
        assert!(!candidate(&d, Some(16 * 1024)));
        d.SampleDesc.Count = 1;
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        assert!(!candidate(&d, Some(16 * 1024)));
    }

    /// On the GPU: 512 small 4 KiB (64x64 BC1) and 16 KiB (64x64 RGBA8) sampled textures pack at
    /// 4 KiB granularity inside a single 16 MiB heap (16x and 4x smaller than 64 KiB tiles),
    /// same-address placements alias their pixels on hardware, adjacent 4 KiB textures have zero
    /// crosstalk, and a small sampled texture over an existing 64 KiB GPU target falls back to
    /// `vram::Pages` so target aliasing is preserved.
    #[test]
    fn gpu_small_textures_pack_at_4k_granularity_and_preserve_pixels() {
        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory6 = CreateDXGIFactory1().expect("DXGI factory");
            let sel = crate::gpu::adapter::Selector::from_env_or("auto");
            let mut dev: Option<ID3D12Device> = None;
            for (adapter, _) in crate::gpu::adapter::candidates(&factory, &sel) {
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).is_ok() && dev.is_some() {
                    break;
                }
            }
            let Some(dev) = dev else {
                crate::gpu::recomp::runner::skip_test("no D3D12 device");
                return;
            };
            let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut opts as *mut _ as *mut c_void, std::mem::size_of_val(&opts) as u32).is_err()
                || opts.ResourceHeapTier.0 < 2
            {
                crate::gpu::recomp::runner::skip_test(&format!("resource heap tier {} < 2", opts.ResourceHeapTier.0));
                return;
            }
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let dev_key = dev.as_raw() as usize;
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;

            let desc_of = |w: u64, h: u32, fmt: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: w,
                Height: h,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: fmt,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: flags,
                ..Default::default()
            };
            let place_sampled = |va: usize, d: &D3D12_RESOURCE_DESC, cbytes: u64| -> ID3D12Resource {
                let mut p = std::ptr::null_mut();
                let hr = super::super::vram::place_pages(dev.as_raw(), va, d, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), Some(cbytes), riid, &mut p);
                assert!(hr.is_some_and(|h| h >= 0) && !p.is_null(), "place_pages at {va:#x} failed: {hr:x?}");
                assert!(super::super::vram::is_placed_texture(p as usize));
                ID3D12Resource::from_raw(p)
            };

            // 1. 512 x 64x64 BC1 (2 KiB on console -> 4 KiB small resource on PC): must pack into
            //    2 MiB (16x smaller than 512 x 64 KiB = 32 MiB!) inside a single 16 MiB heap.
            super::super::vram::flush_idle_for(dev_key);
            let bc1 = desc_of(64, 64, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE);
            let bc1_info = dev.GetResourceAllocationInfo(0, &[D3D12_RESOURCE_DESC { Alignment: D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT as u64, ..bc1 }]);
            if bc1_info.SizeInBytes == u64::MAX || bc1_info.SizeInBytes == 0 || bc1_info.Alignment != UNIT as u64 || bc1_info.SizeInBytes >= 0x1_0000 {
                crate::gpu::recomp::runner::skip_test(&format!(
                    "adapter does not support 4 KiB small-resource placement for 64x64 BC1 (align={}, size={})",
                    bc1_info.Alignment, bc1_info.SizeInBytes
                ));
                return;
            }
            let (small0, _) = bytes_for(dev_key);
            let (pages0, _) = super::super::vram::page_bytes_for(dev_key);
            let bc1_base = 0x60_0000_0000usize;
            let mut bc1_res = Vec::with_capacity(512);
            for k in 0..512usize {
                bc1_res.push(place_sampled(bc1_base + k * 0x1000, &bc1, 2048));
            }
            let (small_after_bc1, small_pool_after_bc1) = bytes_for(dev_key);
            let bc1_small_delta = small_after_bc1 - small0;
            let bc1_total_delta = super::super::vram::page_bytes_for(dev_key).0 - pages0;
            println!("512 x 64x64 BC1: +{bc1_small_delta} B in small pools (pool={small_pool_after_bc1} B), +{bc1_total_delta} B total (vs 33554432 B as 64 KiB tiles)");
            assert_eq!(bc1_small_delta, 512 * UNIT, "each 64x64 BC1 must take 4 KiB in small pool");
            assert_eq!(bc1_total_delta, 512 * UNIT, "zero 64 KiB Pages tiles allocated for 4 KiB BC1 textures");
            drop(bc1_res);

            // 2. Hardware upload + readback: two adjacent 64x64 RGBA8 (16 KiB) textures and a
            //    same-address alias verify zero crosstalk and exact pixel sharing on the GPU.
            super::super::vram::flush_idle_for(dev_key);
            let rgba = desc_of(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
            let rgba_info = dev.GetResourceAllocationInfo(0, &[D3D12_RESOURCE_DESC { Alignment: D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT as u64, ..rgba }]);
            if rgba_info.SizeInBytes != 16384 || rgba_info.Alignment != UNIT as u64 {
                crate::gpu::recomp::runner::skip_test(&format!(
                    "step 2: adapter does not support 4 KiB 16 KiB placement for 64x64 RGBA8 (align={}, size={})",
                    rgba_info.Alignment, rgba_info.SizeInBytes
                ));
                return;
            }
            let va_a = 0x61_0000_1000usize;
            let va_b = 0x61_0000_5000usize;
            let a1 = place_sampled(va_a, &rgba, 16384);
            let (small_after_a1, _) = bytes_for(dev_key);
            let a2 = place_sampled(va_a, &rgba, 16384);
            assert_eq!(bytes_for(dev_key).0, small_after_a1, "same-va small texture must share slot");
            let b = place_sampled(va_b, &rgba, 16384);
            assert_eq!(bytes_for(dev_key).0 - small_after_a1, 16384, "64x64 RGBA8 must take 16 KiB");

            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let fill_a = vec![0x3Cu8; 64 * 64 * 4];
            let fill_b = vec![0xA5u8; 64 * 64 * 4];
            let sub = |data: Vec<u8>| crate::gpu::texture::Subresource { rows: 64, row_bytes: 64 * 4, depth: 1, data };
            let batch = vec![
                (a2.clone(), vec![sub(fill_a.clone())], vec![D3D12_RESOURCE_STATE_COMMON]),
                (b.clone(), vec![sub(fill_b.clone())], vec![D3D12_RESOURCE_STATE_COMMON]),
            ];
            let failed = crate::gpu::texture::upload_batch(&dev, &queue, &batch).unwrap();
            assert!(failed.is_empty(), "upload failed: {failed:?}");

            let readback_tex = |r: &ID3D12Resource| -> Vec<u8> {
                let rd = r.GetDesc();
                let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
                let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
                dev.GetCopyableFootprints(&rd, 0, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut row_bytes), Some(&mut total));
                let mut rb: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb,
                ).unwrap();
                let rb = rb.unwrap();
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                crate::gpu::texture::wait(&dev, &queue).unwrap();
                let mut p = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut p)).unwrap();
                let mut out = Vec::with_capacity(rows as usize * row_bytes as usize);
                for y in 0..rows as usize {
                    out.extend_from_slice(std::slice::from_raw_parts((p as *const u8).add(y * fp.Footprint.RowPitch as usize), row_bytes as usize));
                }
                rb.Unmap(0, None);
                out
            };

            assert_eq!(readback_tex(&a1), fill_a, "same-va alias a1 must see pixels uploaded to a2");
            assert_eq!(readback_tex(&b), fill_b, "adjacent 4 KiB texture b must not crosstalk with a");
            drop((b, a2, a1));

            // 3. A small sampled texture at a 64 KiB-aligned va occupied by an existing GPU-owned
            //    render target must stay on vram::Pages (not vram_small) so it aliases the target.
            if opts.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
                crate::gpu::recomp::runner::skip_test(&format!("step 3: tiled resources tier {} < 2", opts.TiledResourcesTier.0));
                return;
            }
            super::super::vram::flush_idle_for(dev_key);
            let rt_va = 0x62_0000_0000usize;
            let rt_desc = desc_of(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            let mut rt_raw = std::ptr::null_mut();
            let hr = super::super::vram::place(dev.as_raw(), rt_va, &rt_desc, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32, std::ptr::null(), Some(16384), riid, &mut rt_raw);
            assert!(hr.is_some_and(|h| h >= 0) && !rt_raw.is_null());
            let rt = ID3D12Resource::from_raw(rt_raw);
            let small_before_alias = bytes_for(dev_key).0;
            let sampled_view = place_sampled(rt_va, &rgba, 16384);
            assert_eq!(bytes_for(dev_key).0, small_before_alias, "sampled view over live GPU target must alias vram::Pages, not vram_small");
            drop((sampled_view, rt));
        }
    }

    #[test]
    fn fallback_pool_packs_cools_recycles_and_rolls_back() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut p: DefaultFallbackPools<u32> = DefaultFallbackPools::new();
        let alloc = |p: &mut DefaultFallbackPools<u32>, units: u32, res: usize, now: Instant| -> (u32, Block) {
            let (h, b) = p.try_acquire(units, now).unwrap_or_else(|| {
                let id = p.pools.len() as u32;
                p.add_pool_and_acquire(id, units, now)
            });
            p.note_resource(res, b);
            (h, b)
        };

        // 8 x 64 KiB resources pack contiguously into 512 KiB of pool 0.
        for i in 0..8usize {
            let (h, b) = alloc(&mut p, 1, 1000 + i, at(0));
            assert_eq!((h, b), (0, (0, i as u32, 1)));
        }
        assert_eq!(p.bytes(), (8 * FALLBACK_UNIT, FALLBACK_POOL_UNITS as usize * FALLBACK_UNIT));

        // Release resource 1003; must stay in cooling until COOL (2s) elapses.
        assert!(p.forget(1003, at(100)));
        assert_eq!(p.bytes().0, 7 * FALLBACK_UNIT);
        let (_, b_wait) = alloc(&mut p, 1, 1100, at(1500));
        assert_eq!(b_wait, (0, 8, 1), "still cooling at +1.4s, must not reuse yet");
        let (_, b_rec) = alloc(&mut p, 1, 1101, at(2200));
        assert_eq!(b_rec, (0, 3, 1), "block (0, 3, 1) recycled after COOL");

        // Rollback on creation failure returns the block immediately to the free list.
        let (_, b_fail) = p.try_acquire(4, at(2300)).unwrap();
        assert_eq!(b_fail, (0, 9, 4));
        p.rollback(b_fail);
        let (_, b_reused) = alloc(&mut p, 4, 1102, at(2300));
        assert_eq!(b_reused, (0, 9, 4));

        // Concurrent CreateHeap race: add_pool_and_acquire reuses existing room without pushing an empty pool.
        let pools_before = p.pools.len();
        let (h_raced, _) = p.add_pool_and_acquire(999, 2, at(2400));
        assert_eq!(h_raced, 0);
        assert_eq!(p.pools.len(), pools_before, "must not push redundant fallback pool when room exists");
    }

    #[test]
    fn fallback_candidate_filters_msaa_and_oversized_buffers() {
        let mut buf = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 4096,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        assert!(fallback_candidate(&buf));
        buf.Width = (FALLBACK_MAX_UNITS as u64 * FALLBACK_UNIT as u64) + 1;
        assert!(!fallback_candidate(&buf), "buffers > 4 MiB stay committed");
        buf.Width = 0;
        assert!(!fallback_candidate(&buf));

        let mut tex1d = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE1D,
            Width: 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            ..Default::default()
        };
        assert!(fallback_candidate(&tex1d));
        tex1d.SampleDesc.Count = 4;
        assert!(!fallback_candidate(&tex1d), "MSAA requires 4 MiB alignment");
    }

    #[test]
    fn gpu_fallback_pool_places_texture1d_and_small_buffers_cleanly() {
        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory6 = CreateDXGIFactory1().expect("DXGI factory");
            let sel = crate::gpu::adapter::Selector::from_env_or("auto");
            let mut dev: Option<ID3D12Device> = None;
            for (adapter, _) in crate::gpu::adapter::candidates(&factory, &sel) {
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).is_ok() && dev.is_some() {
                    break;
                }
            }
            let Some(dev) = dev else {
                crate::gpu::recomp::runner::skip_test("no D3D12 device");
                return;
            };
            if !tier2_cached(&dev) {
                crate::gpu::recomp::runner::skip_test("resource heap tier < 2");
                return;
            }
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let dev_key = dev.as_raw() as usize;
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;

            let (used0, _) = fallback_bytes_for(dev_key);
            let tex1d_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE1D,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };
            let mut p1 = std::ptr::null_mut();
            let hr1 = place_fallback(dev.as_raw(), &tex1d_desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), riid, &mut p1);
            assert!(hr1.is_some_and(|h| h >= 0) && !p1.is_null(), "TEXTURE1D fallback placement failed: {hr1:x?}");
            assert!(!super::super::vram::is_placed_texture(p1 as usize), "fallback pool resources must preserve non-shared committed semantics");
            let r1 = ID3D12Resource::from_raw(p1);

            let buf_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 4096,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };
            let mut p2 = std::ptr::null_mut();
            let hr2 = place_fallback(dev.as_raw(), &buf_desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), riid, &mut p2);
            assert!(hr2.is_some_and(|h| h >= 0) && !p2.is_null(), "small buffer fallback placement failed: {hr2:x?}");
            assert!(!super::super::vram::is_placed_texture(p2 as usize));
            let r2 = ID3D12Resource::from_raw(p2);

            let (used1, pool1) = fallback_bytes_for(dev_key);
            assert_eq!(used1 - used0, 2 * FALLBACK_UNIT, "TEXTURE1D + 4 KiB buffer must consume 2 x 64 KiB fallback units");
            assert!(pool1 >= FALLBACK_POOL_UNITS as usize * FALLBACK_UNIT);
            drop((r2, r1));
            let (used2, _) = fallback_bytes_for(dev_key);
            assert_eq!(used2, used0, "COM release must return both fallback blocks via forget_fallback");
        }
    }

    /// Offline D3D12 microbenchmark: measures (1) `CreateCommittedResource` vs `DefaultFallbackPools::place_fallback`
    /// (`CreatePlacedResource` into 16 MiB heap) for TEXTURE1D + small fallback buffers, and
    /// (2) double `xg::Texture::new` vs deduplicated single `xg::Texture::new` + `Arc::clone`.
    #[test]
    #[ignore]
    fn bench_fallback_pool_vs_committed_and_xg_dedup() {
        let _vram_guard = super::super::vram::VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory6 = CreateDXGIFactory1().expect("DXGI factory");
            let sel = crate::gpu::adapter::Selector::from_env_or("auto");
            let mut dev: Option<ID3D12Device> = None;
            for (adapter, _) in crate::gpu::adapter::candidates(&factory, &sel) {
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).is_ok() && dev.is_some() {
                    break;
                }
            }
            let dev = dev.expect("D3D12 device");
            assert!(tier2_cached(&dev), "requires ResourceHeapTier >= 2");
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;

            let tex1d_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE1D,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };
            let buf_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 4096,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };
            let heap_props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };

            // Warm up one heap in DefaultFallbackPools so steady-state placement is measured.
            let mut warm = std::ptr::null_mut();
            let _ = place_fallback(dev.as_raw(), &tex1d_desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), riid, &mut warm);
            if !warm.is_null() {
                drop(ID3D12Resource::from_raw(warm));
            }

            const N: usize = 256;
            let t_committed = Instant::now();
            let mut committed_res = Vec::with_capacity(N);
            for i in 0..N {
                let d = if i % 2 == 0 { &tex1d_desc } else { &buf_desc };
                let mut p = std::ptr::null_mut();
                let hr = super::super::real::dev_create_committed_resource(
                    dev.as_raw(),
                    &heap_props as *const _ as usize,
                    0,
                    d as *const _ as usize,
                    D3D12_RESOURCE_STATE_COMMON.0 as usize,
                    0,
                    riid,
                    &mut p,
                );
                assert!(hr >= 0 && !p.is_null(), "CreateCommittedResource failed: {hr:#x}");
                committed_res.push(ID3D12Resource::from_raw(p));
            }
            let committed_us = t_committed.elapsed().as_micros() as f64;
            drop(committed_res);

            let t_pooled = Instant::now();
            let mut pooled_res = Vec::with_capacity(N);
            for i in 0..N {
                let d = if i % 2 == 0 { &tex1d_desc } else { &buf_desc };
                let mut p = std::ptr::null_mut();
                let hr = place_fallback(dev.as_raw(), d, D3D12_RESOURCE_STATE_COMMON.0 as u32, std::ptr::null(), riid, &mut p);
                assert!(hr.is_some_and(|h| h >= 0) && !p.is_null(), "place_fallback failed: {hr:x?}");
                pooled_res.push(ID3D12Resource::from_raw(p));
            }
            let pooled_us = t_pooled.elapsed().as_micros() as f64;
            drop(pooled_res);

            let speedup_alloc = committed_us / pooled_us.max(1.0);
            println!(
                "BENCH Opt B ({N}x mixed TEXTURE1D+Buffer): CreateCommittedResource={:.2} ms ({:.2} us/op) vs place_fallback={:.2} ms ({:.2} us/op) => {:.2}x speedup",
                committed_us / 1000.0, committed_us / N as f64, pooled_us / 1000.0, pooled_us / N as f64, speedup_alloc
            );

            // Opt A microbenchmark: 1000x double xg::Texture::new vs single xg::Texture::new + Arc::clone
            let mut console = [0u8; 0x38];
            console[0..4].copy_from_slice(&3u32.to_le_bytes()); // Texture2D
            console[0x10..0x18].copy_from_slice(&1024u64.to_le_bytes());
            console[0x18..0x1C].copy_from_slice(&1024u32.to_le_bytes());
            console[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
            console[0x1E..0x20].copy_from_slice(&10u16.to_le_bytes()); // 10 mips
            console[0x20..0x24].copy_from_slice(&71u32.to_le_bytes()); // BC1_UNORM
            console[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            const M: usize = 2000;
            let t_double = Instant::now();
            let mut acc = 0u64;
            for _ in 0..M {
                let a = crate::hle::xg::Texture::new(&console).unwrap();
                let b = crate::hle::xg::Texture::new(&console).unwrap();
                acc = acc.wrapping_add(a.size() ^ b.size());
            }
            let double_us = t_double.elapsed().as_micros() as f64;

            let t_dedup = Instant::now();
            for _ in 0..M {
                let a = std::sync::Arc::new(crate::hle::xg::Texture::new(&console).unwrap());
                let b = std::sync::Arc::clone(&a);
                acc = acc.wrapping_add(a.size() ^ b.size());
            }
            let dedup_us = t_dedup.elapsed().as_micros() as f64;
            std::hint::black_box(acc);

            let speedup_xg = double_us / dedup_us.max(1.0);
            println!(
                "BENCH Opt A ({M}x 1024x1024 10-mip BC1 xg::Texture::new): double={:.2} ms ({:.2} us/op) vs dedup={:.2} ms ({:.2} us/op) => {:.2}x speedup",
                double_us / 1000.0, double_us / M as f64, dedup_us / 1000.0, dedup_us / M as f64, speedup_xg
            );
        }
    }
}
