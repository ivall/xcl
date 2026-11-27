//! Console textures from console memory → PC textures. On the console a texture is the
//! memory a T# points at: the title writes it already tiled (GFX10 swizzle, mip chain and
//! slices as addrlib lays them out – its archives store textures that way) and samples it
//! directly. A PC texture has an opaque layout, so every mip level of every slice is detiled
//! (`tiling::Chain`) into a D3D12 subresource and copied in. SW_LINEAR resources use
//! the GFX10 256-byte row, level and slice alignment directly.
//!
//! 2D textures, arrays and cubes, 3D textures (thick and thin modes) and linear ones, the _X
//! modes with the console's pipe configuration (`console_pipes`). What is not here yet: MSAA,
//! and the binding to the console API that creates textures in title memory (and tells when
//! the title rewrote one) – that comes with the game executable.

use super::desc::{self, Image, ImageType};
use super::fence;
use super::tiling::{self, Chain, PipeConfig};
use windows::Win32::Graphics::Direct3D12::*;

/// Pipes / packers of the console's GB_ADDR_CONFIG – only the XOR (_X) modes depend on it.
/// From the console's own layout library: every texel of every mode matches with 32 pipes /
/// 8 packers by default (Series X), 8 pipes / 4 packers with its hardware version 4 (taken
/// to be Series S).
pub const CONSOLE_PIPES: PipeConfig = PipeConfig { pipes_log2: 5, pkrs_log2: 3 };
pub const SERIES_S_PIPES: PipeConfig = PipeConfig { pipes_log2: 3, pkrs_log2: 2 };

/// The pipe configuration of the console XCL reports to the title (`XSystemGetDeviceType`).
pub fn console_pipes() -> PipeConfig {
    if crate::state::config().is_series_s() {
        SERIES_S_PIPES
    } else {
        CONSOLE_PIPES
    }
}

/// Linear data of one subresource: `depth` slices (3D, else 1) of `rows` rows of `row_bytes`
/// (elements: pixels or 4×4 blocks).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Subresource {
    pub rows: usize,
    pub row_bytes: usize,
    pub depth: usize,
    pub data: Vec<u8>,
}

/// Where and how a T#'s texture lies in console memory.
pub struct Placement {
    /// first byte (the T# address rounded down to a block; the rest is the base swizzle)
    pub start: u64,
    pub bytes: usize,
    pub chain: Chain,
    /// array slices (cube faces), or the storage layers of a 3D texture.
    pub slices: usize,
}

pub fn placement(t: &Image, cfg: PipeConfig) -> Result<Placement, String> {
    if !desc::valid_image_shape(t) {
        return Err(format!("invalid {:?} texture dimensions, mip range or array view", t.ty));
    }
    if !desc::valid_cube_view(t) {
        return Err(format!("cube view {}..{} is not a nonempty whole number of faces", t.base_array, t.depth));
    }
    if matches!(t.ty, ImageType::D2Msaa | ImageType::D2MsaaArray) {
        return Err(format!("texture {:?} – not yet (MSAA)", t.ty));
    }
    let mut surface = *t;
    if matches!(surface.ty, ImageType::D1 | ImageType::D1Array) {
        // The T# height field is ignored for 1D resources, just as image_d3d12 ignores it.
        surface.height = 1;
    }
    // linear (SW_MODE 0) included: addrlib's 256-byte rows, levels from the smallest
    let chain = tiling::image_chain(&surface, cfg).ok_or_else(|| format!("SW_MODE {} / format {} without a tiling pattern", t.swizzle_mode, t.format))?;
    let slices = match t.ty {
        ImageType::D1Array | ImageType::D2Array | ImageType::Cube => t.depth.max(1) as usize,
        ImageType::D3 => chain.layers(),
        _ => 1,
    };
    let start = t.address & !(chain.layout.block_bytes as u64 - 1);
    let bytes = chain.slice_bytes.checked_mul(slices).ok_or("tiled texture size overflow")?;
    Ok(Placement { start, bytes, chain, slices })
}

/// The subresources of `t` in D3D12 order (mip + slice × mips) from `mem`, its bytes from
/// `Placement::start` on.
pub fn subresources(t: &Image, mem: &[u8], cfg: PipeConfig) -> Result<Vec<Subresource>, String> {
    let p = placement(t, cfg)?;
    if mem.len() < p.bytes {
        return Err(format!("{} B of memory, but the texture takes {} B", mem.len(), p.bytes));
    }
    let chain = &p.chain;
    let e = 1usize << chain.layout.elem_log2;
    if matches!(t.ty, ImageType::D3) {
        return Ok(chain
            .levels
            .iter()
            .enumerate()
            .map(|(i, l)| Subresource {
                rows: l.height,
                row_bytes: l.width * e,
                depth: l.depth,
                data: chain.detile_volume(mem, i),
            })
            .collect());
    }
    let mut out = Vec::with_capacity(p.slices * chain.levels.len());
    for s in 0..p.slices {
        let slice = &mem[s * chain.slice_bytes..(s + 1) * chain.slice_bytes];
        for (i, l) in chain.levels.iter().enumerate() {
            out.push(Subresource { depth: 1, rows: l.height, row_bytes: l.width * e, data: chain.detile_level(slice, i, s) });
        }
    }
    Ok(out)
}

/// The console memory a T#'s texture occupies, when it lies in the unified memory (UMA).
fn checked_span(start: u64, bytes: usize) -> Result<(usize, usize), String> {
    if bytes == 0 {
        return Err("empty texture memory span".into());
    }
    let start = usize::try_from(start).map_err(|_| "texture address does not fit the host address space")?;
    let end = start.checked_add(bytes).ok_or("texture memory span overflows the host address space")?;
    Ok((start, end))
}

pub fn uma_bytes(t: &Image, cfg: PipeConfig) -> Result<&'static [u8], String> {
    let p = placement(t, cfg)?;
    let u = super::uma::get().ok_or("no UMA")?;
    let (a, end) = checked_span(p.start, p.bytes)?;
    if !u.contains(a) || !u.contains(end - 1) {
        return Err(format!("texture {a:#x}+{:#x} outside UMA", p.bytes));
    }
    // SAFETY: inside the UMA reservation, which lives for the whole process
    Ok(unsafe { std::slice::from_raw_parts(a as *const u8, p.bytes) })
}

fn err(x: windows::core::Error) -> String {
    format!("{x}")
}

/// A PC texture (default heap, left in the shader-resource states) with the console
/// texture's contents: `mem` from `Placement::start` on.
pub unsafe fn create(dev: &ID3D12Device, queue: &ID3D12CommandQueue, t: &Image, mem: &[u8], cfg: PipeConfig) -> Result<ID3D12Resource, String> {
    let subs = subresources(t, mem, cfg)?;
    let (rd, _) = desc::image_d3d12(t).ok_or_else(|| format!("format {} without a DXGI equivalent", t.format))?;
    let mut r: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r)
        .map_err(|x| format!("CreateCommittedResource: {x}"))?;
    let r = r.ok_or("CreateCommittedResource without a resource")?;
    upload(dev, queue, &r, &subs, D3D12_RESOURCE_STATE_COPY_DEST, SHADER_READ)?;
    Ok(r)
}

/// The states `create` leaves a texture in.
pub const SHADER_READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0);

fn validate_subresources(subs: &[Subresource]) -> Result<(), String> {
    if subs.is_empty() {
        return Err("texture upload without subresources".into());
    }
    for (i, s) in subs.iter().enumerate() {
        if s.depth == 0 || s.rows == 0 || s.row_bytes == 0 {
            return Err(format!("texture subresource {i} has a zero dimension"));
        }
        let bytes = s
            .depth
            .checked_mul(s.rows)
            .and_then(|n| n.checked_mul(s.row_bytes))
            .ok_or_else(|| format!("texture subresource {i} size overflow"))?;
        if s.data.len() < bytes {
            return Err(format!("texture subresource {i} has {} B, needs {bytes} B", s.data.len()));
        }
    }
    Ok(())
}

fn validate_subresource_count(rd: &D3D12_RESOURCE_DESC, count: usize) -> Result<(), String> {
    let arrays = match rd.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D | D3D12_RESOURCE_DIMENSION_TEXTURE2D => rd.DepthOrArraySize as usize,
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => 1,
        _ => return Err("texture upload requires a texture resource".into()),
    };
    // Every format accepted by desc::image_d3d12 is single-plane. Add the plane count here if
    // planar formats are introduced later.
    let expected = (rd.MipLevels as usize)
        .checked_mul(arrays)
        .ok_or("texture resource subresource count overflow")?;
    if count != expected {
        return Err(format!("texture upload has {count} subresources, resource requires {expected}"));
    }
    Ok(())
}

/// Copy `subs` (all subresources, D3D12 order) into `res` on `queue` and wait; `res` is in
/// `before` and ends in `after` (e.g. again after the title rewrote the memory).
pub unsafe fn upload(dev: &ID3D12Device, queue: &ID3D12CommandQueue, res: &ID3D12Resource, subs: &[Subresource], before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> Result<(), String> {
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(err)?;
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(err)?;
    let staging = record_upload(dev, &list, res, subs, before, after)?;
    list.Close().map_err(err)?;
    queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).map_err(err)?)]);
    let r = wait(dev, queue);
    drop(staging);
    r
}

/// What `upload_batch` keeps for a queue between batches: its allocator and list, a fence, and
/// one persistently mapped upload buffer the batches' texels go through. Making them for every
/// batch – an upload buffer per texture, an allocator, a list and a fence per batch – cost a
/// retail game most of its frame once it streamed textures: each of those D3D12 calls waits for
/// the video memory manager, 200 – 650 ms apiece then.
struct Uploader {
    dev_key: usize,
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    value: u64,
    /// (buffer, its mapped bytes, capacity)
    staging: Option<(ID3D12Resource, *mut u8, u64)>,
    last_used_ms: u64,
    last_used_frame: u64,
}

// D3D12 objects are free-threaded; the mapped pointer is only used by the holder of the uploader
unsafe impl Send for Uploader {}

/// queue (raw pointer) → its uploader, while no batch uses it
static UPLOADERS: std::sync::Mutex<Option<std::collections::HashMap<usize, Uploader>>> = std::sync::Mutex::new(None);

/// the smallest upload buffer made, and the placement alignment of texture data in it
const STAGING_MIN: u64 = 4 << 20;
const PLACEMENT: u64 = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as u64;

/// Maximum single upload staging buffer cached in the shared power-of-2 pool (32 MiB, matching
/// `title_textures::BATCH_BYTES` so 8-32 MiB streaming bursts stay pooled instead of stalling on
/// unpooled `CreateCommittedResource`).
const STAGING_POOL_MAX_ITEM: u64 = 32 << 20;
/// Maximum idle `(ID3D12CommandAllocator, ID3D12GraphicsCommandList)` pairs cached per device
/// for `submit_batch` (`XCL_UPLOAD_CMD_POOL=0` disables pooling).
const CMD_POOL_MAX_PER_DEV: usize = 16;
/// Default total byte cap for the shared upload staging pool (256 MiB, `XCL_STAGING_POOL_MB`).
const STAGING_POOL_DEFAULT_CAP: u64 = 256 << 20;
/// Idle staging buffers untouched for at least 2 seconds AND 30 frames are released to reclaim
/// Windows commit headroom.
const STAGING_IDLE_MS: u64 = 2_000;
const STAGING_IDLE_FRAMES: u64 = 30;

static CURRENT_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn monotonic_ms() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

fn staging_pool_cap_bytes() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("XCL_STAGING_POOL_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(STAGING_POOL_DEFAULT_CAP, |mb| mb.saturating_mul(1 << 20))
    })
}

/// Power-of-2 bucket size for the shared upload staging pool (`<= 32 MiB`), or `None` if oversized.
fn staging_pool_bucket(bytes: u64) -> Option<u64> {
    let cap = bytes.max(64 << 10).checked_next_power_of_two()?;
    (cap <= STAGING_POOL_MAX_ITEM).then_some(cap)
}

struct PooledStaging<T> {
    item: T,
    last_ms: u64,
    last_frame: u64,
}

/// Power-of-2 upload staging buffer pool (`<= 8 MiB` per entry, bounded total capacity) with
/// idle reclamation after `STAGING_IDLE_MS` and `STAGING_IDLE_FRAMES`.
struct StagingPool<T> {
    buckets: std::collections::BTreeMap<u64, Vec<PooledStaging<T>>>,
    total_bytes: u64,
    cap_bytes: u64,
}

impl<T> StagingPool<T> {
    fn new(cap_bytes: u64) -> Self {
        Self {
            buckets: std::collections::BTreeMap::new(),
            total_bytes: 0,
            cap_bytes,
        }
    }

    fn acquire(&mut self, bytes: u64) -> Option<(u64, T)> {
        let bucket = staging_pool_bucket(bytes)?;
        let list = self.buckets.get_mut(&bucket)?;
        let entry = list.pop()?;
        if list.is_empty() {
            self.buckets.remove(&bucket);
        }
        self.total_bytes = self.total_bytes.saturating_sub(bucket);
        Some((bucket, entry.item))
    }

    /// Return a GPU-idle power-of-2 buffer to the pool, or hand it back to the caller to drop if
    /// pooling is disabled or the buffer exceeds `STAGING_POOL_MAX_ITEM` / `cap_bytes`.
    fn release(&mut self, cap: u64, item: T, now_ms: u64, frame: u64) -> Option<T> {
        if self.cap_bytes == 0 || cap == 0 || !cap.is_power_of_two() || cap > STAGING_POOL_MAX_ITEM || cap > self.cap_bytes {
            return Some(item);
        }
        while self.total_bytes.saturating_add(cap) > self.cap_bytes {
            if !self.evict_oldest() {
                return Some(item);
            }
        }
        self.buckets.entry(cap).or_default().push(PooledStaging { item, last_ms: now_ms, last_frame: frame });
        self.total_bytes = self.total_bytes.saturating_add(cap);
        None
    }

    fn evict_oldest(&mut self) -> bool {
        let mut victim: Option<(u64, usize, u64)> = None;
        for (&cap, list) in &self.buckets {
            for (idx, e) in list.iter().enumerate() {
                if victim.is_none_or(|(_, _, ts)| e.last_ms < ts) {
                    victim = Some((cap, idx, e.last_ms));
                }
            }
        }
        let Some((cap, idx, _)) = victim else { return false };
        if let Some(list) = self.buckets.get_mut(&cap) {
            list.remove(idx);
            if list.is_empty() {
                self.buckets.remove(&cap);
            }
            self.total_bytes = self.total_bytes.saturating_sub(cap);
            return true;
        }
        false
    }

    fn trim_idle(&mut self, now_ms: u64, frame: u64, force_all: bool) -> (usize, u64) {
        let mut freed_count = 0usize;
        let mut freed_bytes = 0u64;
        let mut empty_buckets = Vec::new();
        for (&cap, list) in &mut self.buckets {
            let before = list.len();
            if force_all {
                list.clear();
            } else {
                list.retain(|e| {
                    let ms_idle = now_ms.saturating_sub(e.last_ms) >= STAGING_IDLE_MS;
                    let fr_idle = frame.saturating_sub(e.last_frame) >= STAGING_IDLE_FRAMES;
                    !(ms_idle && fr_idle)
                });
            }
            let dropped = before - list.len();
            if dropped > 0 {
                freed_count += dropped;
                let bytes = cap.saturating_mul(dropped as u64);
                freed_bytes = freed_bytes.saturating_add(bytes);
                self.total_bytes = self.total_bytes.saturating_sub(bytes);
            }
            if list.is_empty() {
                empty_buckets.push(cap);
            }
        }
        for cap in empty_buckets {
            self.buckets.remove(&cap);
        }
        (freed_count, freed_bytes)
    }
}

struct MappedStaging(ID3D12Resource, *mut u8);

// D3D12 resources are free-threaded; mapped pointer is owned exclusively while pooled or held.
unsafe impl Send for MappedStaging {}

static STAGING_POOLS: std::sync::Mutex<Option<std::collections::HashMap<usize, StagingPool<MappedStaging>>>> = std::sync::Mutex::new(None);

fn upload_cmd_pool_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_UPLOAD_CMD_POOL").map_or(true, |v| v.trim() != "0"))
}

/// Bounded LIFO pool of idle `(ID3D12CommandAllocator, ID3D12GraphicsCommandList)` pairs per device.
struct CmdPool<T> {
    items: Vec<T>,
    max_items: usize,
}

impl<T> CmdPool<T> {
    fn new(max_items: usize) -> Self {
        Self { items: Vec::new(), max_items }
    }

    fn acquire(&mut self) -> Option<T> {
        self.items.pop()
    }

    fn release(&mut self, item: T) -> Option<T> {
        if self.max_items == 0 || self.items.len() >= self.max_items {
            return Some(item);
        }
        self.items.push(item);
        None
    }

    fn clear(&mut self) {
        self.items.clear();
    }
}

struct PooledCmd {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
}

// D3D12 command allocators and closed command lists are free-threaded while idle in the pool.
unsafe impl Send for PooledCmd {}

static CMD_POOLS: std::sync::Mutex<Option<std::collections::HashMap<usize, CmdPool<PooledCmd>>>> = std::sync::Mutex::new(None);

struct CmdSlot {
    dev_key: usize,
    cmd: Option<PooledCmd>,
    recyclable: bool,
}

impl Drop for CmdSlot {
    fn drop(&mut self) {
        if !self.recyclable || !upload_cmd_pool_enabled() {
            return;
        }
        if let Some(cmd) = self.cmd.take() {
            let overflow = {
                let mut pools = CMD_POOLS.lock().unwrap();
                let pool = pools
                    .get_or_insert_with(Default::default)
                    .entry(self.dev_key)
                    .or_insert_with(|| CmdPool::new(CMD_POOL_MAX_PER_DEV));
                pool.release(cmd)
            };
            drop(overflow);
        }
    }
}

fn staging_capacity(bytes: u64) -> Option<u64> {
    const LARGE: u64 = 64 << 20;
    const STEP: u64 = 16 << 20;
    if bytes <= LARGE {
        Some(bytes.max(STAGING_MIN).next_power_of_two())
    } else {
        bytes.checked_next_multiple_of(STEP)
    }
}

impl Uploader {
    unsafe fn new(dev: &ID3D12Device) -> Result<Uploader, String> {
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(err)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(err)?;
        list.Close().map_err(err)?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
        Ok(Uploader {
            dev_key: windows::core::Interface::as_raw(dev) as usize,
            alloc,
            list,
            fence,
            value: 0,
            staging: None,
            last_used_ms: monotonic_ms(),
            last_used_frame: CURRENT_FRAME.load(std::sync::atomic::Ordering::Relaxed),
        })
    }

    /// An upload buffer of at least `bytes`, mapped: the one kept, a pooled power-of-2 buffer,
    /// or a freshly allocated one.
    unsafe fn staging(&mut self, dev: &ID3D12Device, bytes: u64, now_ms: u64, frame: u64) -> Result<(ID3D12Resource, *mut u8), String> {
        self.last_used_ms = now_ms;
        self.last_used_frame = frame;
        if let Some((b, p, cap)) = &self.staging {
            if *cap >= bytes {
                return Ok((b.clone(), *p));
            }
        }
        let cap = staging_capacity(bytes).ok_or("texture upload buffer size overflow")?;
        // A cached uploader has passed its last copy fence. Return its old `<= 8 MiB` buffer to
        // the shared power-of-2 pool (or drop an oversized one) before acquiring the replacement
        // so the growth path does not keep two oversized buffers alive simultaneously.
        if let Some((old_b, old_p, old_cap)) = self.staging.take() {
            if old_cap <= STAGING_POOL_MAX_ITEM && staging_pool_cap_bytes() > 0 {
                let mut pools = STAGING_POOLS.lock().unwrap();
                let pool = pools
                    .get_or_insert_with(Default::default)
                    .entry(self.dev_key)
                    .or_insert_with(|| StagingPool::new(staging_pool_cap_bytes()));
                let _ = pool.release(old_cap, MappedStaging(old_b, old_p), now_ms, frame);
            }
        }
        if cap <= STAGING_POOL_MAX_ITEM && staging_pool_cap_bytes() > 0 {
            let mut pools = STAGING_POOLS.lock().unwrap();
            if let Some((pooled_cap, MappedStaging(b, p))) = pools
                .as_mut()
                .and_then(|m| m.get_mut(&self.dev_key))
                .and_then(|pool| pool.acquire(cap))
            {
                self.staging = Some((b.clone(), p, pooled_cap));
                return Ok((b, p));
            }
        }
        let b = buffer(dev, cap)?;
        let mut p = std::ptr::null_mut();
        b.Map(0, None, Some(&mut p)).map_err(err)?;
        self.staging = Some((b.clone(), p as *mut u8, cap));
        Ok((b, p as *mut u8))
    }
}

/// Reclaim idle upload staging buffers (unused for `>= 2 s` and `>= 30` frames, or immediately
/// under low Windows commit headroom) to prevent post-streaming upload buffers from pinning
/// commit memory indefinitely.
pub fn trim_idle_staging(frame: u64, low_commit: bool) {
    CURRENT_FRAME.store(frame, std::sync::atomic::Ordering::Relaxed);
    if !low_commit && !frame.is_multiple_of(15) {
        return;
    }
    let now_ms = monotonic_ms();
    let mut freed_bytes = 0u64;
    let mut freed_count = 0usize;
    {
        let mut pools = STAGING_POOLS.lock().unwrap();
        if let Some(map) = pools.as_mut() {
            for pool in map.values_mut() {
                let (n, b) = pool.trim_idle(now_ms, frame, low_commit);
                freed_count += n;
                freed_bytes = freed_bytes.saturating_add(b);
            }
        }
    }
    if low_commit {
        let mut pools = CMD_POOLS.lock().unwrap();
        if let Some(map) = pools.as_mut() {
            for pool in map.values_mut() {
                pool.clear();
            }
        }
    }
    {
        let mut ups = UPLOADERS.lock().unwrap();
        if let Some(map) = ups.as_mut() {
            for up in map.values_mut() {
                let Some((_, _, cap)) = up.staging else { continue };
                let idle = now_ms.saturating_sub(up.last_used_ms) >= STAGING_IDLE_MS
                    && frame.saturating_sub(up.last_used_frame) >= STAGING_IDLE_FRAMES;
                let shrink_low = low_commit && cap > STAGING_MIN;
                if idle || shrink_low {
                    up.staging = None;
                    freed_count += 1;
                    freed_bytes = freed_bytes.saturating_add(cap);
                }
            }
        }
    }
    if freed_bytes > 0 {
        static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 4 || n.is_power_of_two() {
            crate::xlog_info!(
                "D3D12.X: released {freed_count} idle texture upload staging buffer(s) ({:.1} MiB{})",
                freed_bytes as f64 / (1024.0 * 1024.0),
                if low_commit { ", low commit" } else { "" }
            );
        }
    }
}

/// Several uploads on one list, one wait: (resource, subresources, states) each – the states
/// the texture is in and is left in: one for every subresource, or one per subresource (D3D12
/// order). A failure of one is returned with its index; the others still run. The texels go
/// through the queue's kept upload buffer (`Uploader`).
pub unsafe fn upload_batch(dev: &ID3D12Device, queue: &ID3D12CommandQueue, items: &[(ID3D12Resource, Vec<Subresource>, Vec<D3D12_RESOURCE_STATES>)]) -> Result<Vec<(usize, String)>, String> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let key = windows::core::Interface::as_raw(queue) as usize;
    let taken = UPLOADERS.lock().unwrap().as_mut().and_then(|m| m.remove(&key));
    let mut up = match taken {
        Some(u) => u,
        None => Uploader::new(dev)?,
    };
    // where each texture's texels go in the upload buffer
    let (failed, planned, total) = plan_batch(dev, items);
    if planned.is_empty() {
        put_back(key, up);
        return Ok(failed);
    }
    let now_ms = monotonic_ms();
    let frame = CURRENT_FRAME.load(std::sync::atomic::Ordering::Relaxed);
    let (staging, mapped) = up.staging(dev, total, now_ms, frame)?;
    up.alloc.Reset().map_err(err)?;
    up.list.Reset(&up.alloc, None).map_err(err)?;
    for (k, l, base) in &planned {
        let (res, subs, states) = &items[*k];
        record_into(&up.list, res, subs, l, states, states, &staging, mapped, *base);
    }
    up.list.Close().map_err(err)?;
    queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&up.list).map_err(err)?)]);
    up.value += 1;
    let copy_error = |e| format!("texture copy: {e}");
    queue.Signal(&up.fence, up.value).map_err(|e| copy_error(fence::api_error(dev, "Signal", e)))?;
    if let Err(e) = fence::wait(&up.fence, up.value, dev, None, 10_000) {
        // the GPU may still read the upload buffer and run the list: they stay alive (a fresh
        // uploader next time) – freeing them under the GPU removed the device
        std::mem::forget(up);
        return Err(copy_error(e));
    }
    up.last_used_ms = monotonic_ms();
    up.last_used_frame = CURRENT_FRAME.load(std::sync::atomic::Ordering::Relaxed);
    put_back(key, up);
    Ok(failed)
}

unsafe fn plan_batch(dev: &ID3D12Device, items: &[(ID3D12Resource, Vec<Subresource>, Vec<D3D12_RESOURCE_STATES>)]) -> (Vec<(usize, String)>, Vec<(usize, Layout, u64)>, u64) {
    let mut failed = Vec::new();
    let mut planned = Vec::with_capacity(items.len());
    let mut total = 0u64;
    for (k, (res, subs, states)) in items.iter().enumerate() {
        match layout(dev, res, subs, states, states) {
            Ok(l) => {
                let base = total.div_ceil(PLACEMENT) * PLACEMENT;
                total = base + l.total;
                planned.push((k, l, base));
            }
            Err(e) => failed.push((k, e)),
        }
    }
    (failed, planned, total)
}

struct StagingSlot {
    dev_key: usize,
    cap: u64,
    pooled: Option<MappedStaging>,
    _unpooled: Option<ID3D12Resource>,
}

impl Drop for StagingSlot {
    fn drop(&mut self) {
        if let Some(ms) = self.pooled.take() {
            if self.cap <= STAGING_POOL_MAX_ITEM && staging_pool_cap_bytes() > 0 {
                let now_ms = monotonic_ms();
                let frame = CURRENT_FRAME.load(std::sync::atomic::Ordering::Relaxed);
                let mut pools = STAGING_POOLS.lock().unwrap();
                let pool = pools
                    .get_or_insert_with(Default::default)
                    .entry(self.dev_key)
                    .or_insert_with(|| StagingPool::new(staging_pool_cap_bytes()));
                let _ = pool.release(self.cap, ms, now_ms, frame);
            }
        }
    }
}

/// GPU-side upload resources kept alive until `queue`'s fence passes the submitted batch.
/// Dropping an `InFlight` (strictly after its GPU fence completes) returns pooled `<= 32 MiB`
/// staging buffers to `STAGING_POOLS` and idle `(allocator, list)` pairs to `CMD_POOLS`.
pub struct InFlight {
    _cmd: CmdSlot,
    _staging: StagingSlot,
    pub bytes: u64,
}

/// Acquire an open recording `(ID3D12CommandAllocator, ID3D12GraphicsCommandList)` pair for
/// `dev`, reusing a previously completed pair from `CMD_POOLS` when available.
unsafe fn acquire_upload_cmd(dev: &ID3D12Device, dev_key: usize) -> Result<PooledCmd, String> {
    if upload_cmd_pool_enabled() {
        let cached = CMD_POOLS
            .lock()
            .unwrap()
            .as_mut()
            .and_then(|m| m.get_mut(&dev_key))
            .and_then(CmdPool::acquire);
        if let Some(cmd) = cached {
            if cmd.alloc.Reset().is_ok() && cmd.list.Reset(&cmd.alloc, None).is_ok() {
                return Ok(cmd);
            }
        }
    }
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(err)?;
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(err)?;
    Ok(PooledCmd { alloc, list })
}

/// Record and execute a batch of texture uploads directly on `queue` without waiting on the CPU.
/// The returned `InFlight` must stay alive until `queue` completes this submission.
pub unsafe fn submit_batch(dev: &ID3D12Device, queue: &ID3D12CommandQueue, items: &[(ID3D12Resource, Vec<Subresource>, Vec<D3D12_RESOURCE_STATES>)]) -> Result<(Vec<(usize, String)>, Option<InFlight>), String> {
    let (failed, planned, bytes) = plan_batch(dev, items);
    if planned.is_empty() {
        return Ok((failed, None));
    }
    let dev_key = windows::core::Interface::as_raw(dev) as usize;
    let cmd = acquire_upload_cmd(dev, dev_key)?;
    let slot = if let Some(bucket) = (staging_pool_cap_bytes() > 0).then(|| staging_pool_bucket(bytes)).flatten() {
        let cached = STAGING_POOLS
            .lock()
            .unwrap()
            .as_mut()
            .and_then(|m| m.get_mut(&dev_key))
            .and_then(|pool| pool.acquire(bucket));
        let (cap, ms) = match cached {
            Some(hit) => hit,
            None => {
                let b = buffer(dev, bucket)?;
                let mut mapped = std::ptr::null_mut();
                b.Map(0, None, Some(&mut mapped)).map_err(err)?;
                (bucket, MappedStaging(b, mapped as *mut u8))
            }
        };
        for (k, layout, base) in &planned {
            let (res, subs, states) = &items[*k];
            record_into(&cmd.list, res, subs, layout, states, states, &ms.0, ms.1, *base);
        }
        StagingSlot { dev_key, cap, pooled: Some(ms), _unpooled: None }
    } else {
        let staging = buffer(dev, bytes)?;
        let mut mapped = std::ptr::null_mut();
        staging.Map(0, None, Some(&mut mapped)).map_err(err)?;
        for (k, layout, base) in &planned {
            let (res, subs, states) = &items[*k];
            record_into(&cmd.list, res, subs, layout, states, states, &staging, mapped as *mut u8, *base);
        }
        staging.Unmap(0, None);
        StagingSlot { dev_key, cap: bytes, pooled: None, _unpooled: Some(staging) }
    };
    cmd.list.Close().map_err(err)?;
    queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&cmd.list).map_err(err)?)]);
    let tracked = slot.cap;
    Ok((failed, Some(InFlight { _cmd: CmdSlot { dev_key, cmd: Some(cmd), recyclable: true }, _staging: slot, bytes: tracked })))
}

fn put_back(key: usize, up: Uploader) {
    UPLOADERS.lock().unwrap().get_or_insert_with(Default::default).insert(key, up);
}

/// Record the copy of `subs` into `res` on `list` (barriers from `before` and back to `after`);
/// the staging buffer returned must live until the list has run.
pub unsafe fn record_upload(dev: &ID3D12Device, list: &ID3D12GraphicsCommandList, res: &ID3D12Resource, subs: &[Subresource], before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> Result<ID3D12Resource, String> {
    record_upload_states(dev, list, res, subs, &[before], &[after])
}

/// `record_upload` with the states per subresource: `before` / `after` hold one state for every
/// subresource or one per subresource (D3D12 order; a title can leave its mips in different
/// states).
pub unsafe fn record_upload_states(
    dev: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    res: &ID3D12Resource,
    subs: &[Subresource],
    before: &[D3D12_RESOURCE_STATES],
    after: &[D3D12_RESOURCE_STATES],
) -> Result<ID3D12Resource, String> {
    let l = layout(dev, res, subs, before, after)?;
    let up = buffer(dev, l.total)?;
    let mut p = std::ptr::null_mut();
    up.Map(0, None, Some(&mut p)).map_err(err)?;
    record_into(list, res, subs, &l, before, after, &up, p as *mut u8, 0);
    up.Unmap(0, None);
    Ok(up)
}

/// Where the texels of an upload go in an upload buffer: each subresource's footprint (offsets
/// from 0) and rows, and the bytes of all of them.
struct Layout {
    fps: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    total: u64,
}

/// The `Layout` of uploading `subs` into `res`, checked against the resource.
unsafe fn layout(dev: &ID3D12Device, res: &ID3D12Resource, subs: &[Subresource], before: &[D3D12_RESOURCE_STATES], after: &[D3D12_RESOURCE_STATES]) -> Result<Layout, String> {
    validate_subresources(subs)?;
    let rd = res.GetDesc();
    let n = subs.len();
    validate_subresource_count(&rd, n)?;
    for s in [before, after] {
        if s.len() != 1 && s.len() != n {
            return Err(format!("{} subresource states for {n} subresources", s.len()));
        }
    }
    let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut row_bytes = vec![0u64; n];
    let mut total = 0u64;
    dev.GetCopyableFootprints(&rd, 0, n as u32, 0, Some(fps.as_mut_ptr()), Some(rows.as_mut_ptr()), Some(row_bytes.as_mut_ptr()), Some(&mut total));
    for (i, ((s, fp), (&nrows, &nbytes))) in subs.iter().zip(&fps).zip(rows.iter().zip(&row_bytes)).enumerate() {
        if s.rows != nrows as usize || s.depth != fp.Footprint.Depth as usize || s.row_bytes != nbytes as usize {
            return Err(format!(
                "texture subresource {i} is {}x{}x{} B, resource expects {}x{}x{} B",
                s.row_bytes, s.rows, s.depth, nbytes, nrows, fp.Footprint.Depth
            ));
        }
    }
    Ok(Layout { fps, rows, total })
}

/// Write `subs` into the mapped upload buffer `up` (`mapped`) from `base` (a multiple of
/// D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT) as `l` places them, and record their copies into
/// `res` on `list` (barriers from `before` and back to `after`).
#[allow(clippy::too_many_arguments)]
unsafe fn record_into(list: &ID3D12GraphicsCommandList, res: &ID3D12Resource, subs: &[Subresource], l: &Layout, before: &[D3D12_RESOURCE_STATES], after: &[D3D12_RESOURCE_STATES], up: &ID3D12Resource, mapped: *mut u8, base: u64) {
    let n = subs.len();
    for ((s, fp), &nrows) in subs.iter().zip(&l.fps).zip(&l.rows) {
        let pitch = fp.Footprint.RowPitch as usize;
        let row = s.row_bytes.min(pitch);
        // depth slices are NumRows rows apart (rows of 4×4 blocks for BCn)
        let slice_pitch = pitch * nrows as usize;
        for z in 0..(fp.Footprint.Depth as usize).min(s.depth) {
            for y in 0..(nrows as usize).min(s.rows) {
                let at = base as usize + fp.Offset as usize + z * slice_pitch + y * pitch;
                std::ptr::copy_nonoverlapping(s.data[(z * s.rows + y) * s.row_bytes..].as_ptr(), mapped.add(at), row);
            }
        }
    }
    let to_copy = transitions(res, before, &[D3D12_RESOURCE_STATE_COPY_DEST], n);
    if !to_copy.is_empty() {
        list.ResourceBarrier(&to_copy);
    }
    for (k, fp) in l.fps.iter().enumerate() {
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(res),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 },
        };
        let placed = D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: base + fp.Offset, Footprint: fp.Footprint };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(up), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: placed } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
    }
    let back = transitions(res, &[D3D12_RESOURCE_STATE_COPY_DEST], after, n);
    if !back.is_empty() {
        list.ResourceBarrier(&back);
    }
}

/// The transitions of `r`'s `n` subresources from `from` to `to` (each one state for all, or one
/// per subresource): one for all subresources when both are uniform, else one per subresource
/// that changes.
fn transitions(r: &ID3D12Resource, from: &[D3D12_RESOURCE_STATES], to: &[D3D12_RESOURCE_STATES], n: usize) -> Vec<D3D12_RESOURCE_BARRIER> {
    if from.len() == 1 && to.len() == 1 {
        return if from[0] == to[0] { Vec::new() } else { vec![transition(r, from[0], to[0])] };
    }
    let at = |s: &[D3D12_RESOURCE_STATES], k: usize| if s.len() == 1 { s[0] } else { s[k] };
    (0..n).filter(|&k| at(from, k) != at(to, k)).map(|k| transition_sub(r, k as u32, at(from, k), at(to, k))).collect()
}

/// Wait until `queue` has run everything submitted so far.
pub unsafe fn wait(dev: &ID3D12Device, queue: &ID3D12CommandQueue) -> Result<(), String> {
    let copy_error = |e| format!("texture copy: {e}");
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)
        .map_err(|e| copy_error(fence::api_error(dev, "CreateFence", e)))?;
    queue.Signal(&fence, 1).map_err(|e| copy_error(fence::api_error(dev, "Signal", e)))?;
    fence::wait(&fence, 1, dev, None, 10_000).map_err(copy_error)
}

unsafe fn buffer(dev: &ID3D12Device, size: u64) -> Result<ID3D12Resource, String> {
    let rd = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut r)
        .map_err(|x| format!("texture upload buffer: {x}"))?;
    r.ok_or_else(|| "texture upload buffer".into())
}

fn transition(r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    transition_sub(r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after)
}

fn transition_sub(r: &ID3D12Resource, sub: u32, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: sub,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use windows::Win32::Graphics::Dxgi::Common::*;

    fn image(format: u32, ty: ImageType, w: u32, h: u32, slices: u32, mips: u32, mode: u32, address: u64) -> Image {
        Image {
            address,
            format,
            width: w,
            height: h,
            depth: slices,
            ty,
            base_level: 0,
            last_level: mips - 1,
            max_mip: mips - 1,
            base_array: 0,
            dst_sel: [4, 5, 6, 7],
            swizzle_mode: mode,
            min_lod: 0.0,
        }
    }

    #[test]
    fn texture_memory_spans_reject_empty_truncated_and_wrapping_addresses() {
        assert_eq!(checked_span(0x1000, 0x200).unwrap(), (0x1000, 0x1200));
        assert!(checked_span(0x1000, 0).unwrap_err().contains("empty"));
        assert!(checked_span(u64::MAX, 2).unwrap_err().contains("address"));
        assert!(checked_span((usize::MAX - 1) as u64, 2).unwrap_err().contains("overflows"));
        assert_eq!(checked_span((usize::MAX - 1) as u64, 1).unwrap(), (usize::MAX - 1, usize::MAX));
    }

    #[test]
    fn upload_subresources_are_validated_before_unsafe_copying() {
        assert!(validate_subresources(&[]).is_err());
        let zero = Subresource { depth: 1, rows: 0, row_bytes: 4, data: vec![] };
        assert!(validate_subresources(&[zero]).unwrap_err().contains("zero dimension"));
        let short = Subresource { depth: 2, rows: 3, row_bytes: 4, data: vec![0; 23] };
        assert_eq!(validate_subresources(&[short]).unwrap_err(), "texture subresource 0 has 23 B, needs 24 B");
        let overflow = Subresource { depth: usize::MAX, rows: 2, row_bytes: 1, data: vec![] };
        assert!(validate_subresources(&[overflow]).unwrap_err().contains("size overflow"));
        let complete = Subresource { depth: 2, rows: 3, row_bytes: 4, data: vec![0; 25] };
        assert!(validate_subresources(&[complete]).is_ok());
    }

    #[test]
    fn upload_requires_every_resource_subresource_exactly_once() {
        let mut rd = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            DepthOrArraySize: 3,
            MipLevels: 4,
            ..Default::default()
        };
        assert!(validate_subresource_count(&rd, 12).is_ok());
        assert_eq!(validate_subresource_count(&rd, 11).unwrap_err(), "texture upload has 11 subresources, resource requires 12");
        assert_eq!(validate_subresource_count(&rd, 13).unwrap_err(), "texture upload has 13 subresources, resource requires 12");
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        rd.DepthOrArraySize = 9;
        assert!(validate_subresource_count(&rd, 4).is_ok());
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        assert!(validate_subresource_count(&rd, 4).unwrap_err().contains("requires a texture"));
    }

    /// Random linear subresources for `t`, and the console memory they are, tiled.
    fn console_texture(t: &Image) -> (Vec<Subresource>, Vec<u8>) {
        let p = placement(t, CONSOLE_PIPES).unwrap();
        let mut mem = vec![0u8; p.bytes];
        let mut subs = Vec::new();
        let mut n = 0x1234_5678u32;
        let mut random = |len: usize| -> Vec<u8> {
            (0..len)
                .map(|_| {
                    n ^= n << 13;
                    n ^= n >> 17;
                    n ^= n << 5;
                    n as u8
                })
                .collect()
        };
        let chain = &p.chain;
        let e = 1usize << chain.layout.elem_log2;
        if matches!(t.ty, ImageType::D3) {
            for (i, l) in chain.levels.iter().enumerate() {
                let data = random(l.width * l.height * l.depth * e);
                chain.tile_volume(&mut mem, i, &data);
                subs.push(Subresource { rows: l.height, row_bytes: l.width * e, depth: l.depth, data });
            }
            return (subs, mem);
        }
        for s in 0..p.slices {
            let slice = &mut mem[s * chain.slice_bytes..(s + 1) * chain.slice_bytes];
            for (i, l) in chain.levels.iter().enumerate() {
                let data = random(l.width * l.height * e);
                chain.tile_level(slice, i, s, &data);
                subs.push(Subresource { depth: 1, rows: l.height, row_bytes: l.width * e, data });
            }
        }
        (subs, mem)
    }

    #[test]
    fn subresources_of_a_mipped_array_from_console_memory() {
        let rgba8 = desc::gfx10_format(DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
        let t = image(rgba8, ImageType::D2Array, 96, 40, 3, 6, 5, 0x10_0000 | 0x300);
        let (want, mem) = console_texture(&t);
        let p = placement(&t, CONSOLE_PIPES).unwrap();
        assert_eq!((p.start, p.slices, want.len()), (0x10_0000, 3, 18));
        assert_eq!(subresources(&t, &mem, CONSOLE_PIPES).unwrap(), want);
        assert!(subresources(&t, &mem[..mem.len() - 1], CONSOLE_PIPES).is_err());
    }

    #[test]
    fn one_dimensional_arrays_ignore_tsharp_height_for_linear_and_tiled_layouts() {
        let rgba8 = desc::gfx10_format(DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
        let gpu = device();
        for mode in [0, 1] {
            let t = image(rgba8, ImageType::D1Array, 33, 37, 2, 3, mode, 0x36_0000);
            let mut normalized = t;
            normalized.height = 1;
            assert_eq!(placement(&t, CONSOLE_PIPES).unwrap().bytes, placement(&normalized, CONSOLE_PIPES).unwrap().bytes);
            let (want, mem) = console_texture(&t);
            assert!(want.iter().all(|s| s.rows == 1));
            assert_eq!(subresources(&t, &mem, CONSOLE_PIPES).unwrap(), want);
            if let Some((dev, queue)) = &gpu {
                unsafe {
                    let res = create(dev, queue, &t, &mem, CONSOLE_PIPES).unwrap();
                    assert_eq!(read_back(dev, queue, &res, &want), want);
                }
            }
        }
    }

    #[test]
    fn non_array_2d_ignores_tsharp_depth_for_linear_and_tiled_uploads() {
        let rgba8 = desc::gfx10_format(DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
        let gpu = device();
        for mode in [0, 1] {
            let t = image(rgba8, ImageType::D2, 17, 3, 9, 2, mode, 0x37_0000);
            assert_eq!(desc::image_d3d12(&t).unwrap().0.DepthOrArraySize, 1);
            assert_eq!(placement(&t, CONSOLE_PIPES).unwrap().slices, 1);
            let (want, mem) = console_texture(&t);
            assert_eq!(subresources(&t, &mem, CONSOLE_PIPES).unwrap(), want);
            if let Some((dev, queue)) = &gpu {
                unsafe {
                    let res = create(dev, queue, &t, &mem, CONSOLE_PIPES).unwrap();
                    assert_eq!(read_back(dev, queue, &res, &want), want);
                }
            }
        }
    }

    #[test]
    fn invalid_cube_face_ranges_are_rejected_before_placement() {
        let rgba8 = desc::gfx10_format(DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
        let mut t = image(rgba8, ImageType::Cube, 16, 16, 5, 1, 1, 0x37_8000);
        assert!(placement(&t, CONSOLE_PIPES).err().unwrap().contains("whole number of faces"));
        t.depth = 12;
        t.base_array = 12;
        assert!(placement(&t, CONSOLE_PIPES).err().unwrap().contains("whole number of faces"));
        t.base_array = 6;
        assert!(placement(&t, CONSOLE_PIPES).is_ok());
    }

    #[test]
    fn invalid_mip_counts_are_rejected_before_layout_shifts() {
        let rgba8 = desc::gfx10_format(DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
        let mut t = image(rgba8, ImageType::D2, 8, 4, 1, 4, 0, 0x37_c000);
        assert!(placement(&t, CONSOLE_PIPES).is_ok());
        t.max_mip = 32;
        t.last_level = 32;
        assert!(placement(&t, CONSOLE_PIPES).err().unwrap().contains("invalid"));
        t.max_mip = 4;
        t.last_level = 4;
        assert!(placement(&t, CONSOLE_PIPES).is_err());
        t.max_mip = 3;
        t.last_level = 2;
        t.base_level = 3;
        assert!(placement(&t, CONSOLE_PIPES).is_err());
    }

    #[test]
    fn oversized_array_depth_is_rejected_before_placement_allocation() {
        let rgba8 = desc::gfx10_format(DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
        let t = image(rgba8, ImageType::D2Array, 1, 1, u16::MAX as u32 + 1, 1, 1, 0x37_d000);
        assert!(placement(&t, CONSOLE_PIPES).err().unwrap().contains("invalid"));
        assert!(desc::image_d3d12(&t).is_none());
    }

    fn device() -> Option<(ID3D12Device, ID3D12CommandQueue)> {
        unsafe {
            let dev = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0).ok()?;
            let queue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).ok()?;
            Some((dev, queue))
        }
    }

    /// Every subresource of a PC texture, read back (rows of `row_bytes`).
    unsafe fn read_back(dev: &ID3D12Device, queue: &ID3D12CommandQueue, res: &ID3D12Resource, subs: &[Subresource]) -> Vec<Subresource> {
        let rd = res.GetDesc();
        let n = subs.len();
        let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
        let mut nrows = vec![0u32; n];
        let mut total = 0u64;
        dev.GetCopyableFootprints(&rd, 0, n as u32, 0, Some(fps.as_mut_ptr()), Some(nrows.as_mut_ptr()), None, Some(&mut total));
        let mut rb: Option<ID3D12Resource> = None;
        let bd = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
        let rb = rb.unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        list.ResourceBarrier(&[transition(res, SHADER_READ, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        for (k, fp) in fps.iter().enumerate() {
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(res), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 } };
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        }
        list.ResourceBarrier(&[transition(res, D3D12_RESOURCE_STATE_COPY_SOURCE, SHADER_READ)]);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).unwrap())]);
        wait(dev, queue).unwrap();
        let mut p = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).unwrap();
        let out = subs
            .iter()
            .zip(fps.iter().zip(&nrows))
            .map(|(s, (fp, &nr))| {
                let pitch = fp.Footprint.RowPitch as usize;
                let mut data = Vec::with_capacity(s.depth * s.rows * s.row_bytes);
                for z in 0..s.depth {
                    for y in 0..s.rows {
                        data.extend_from_slice(std::slice::from_raw_parts((p as *const u8).add(fp.Offset as usize + (z * nr as usize + y) * pitch), s.row_bytes));
                    }
                }
                Subresource { rows: s.rows, row_bytes: s.row_bytes, depth: s.depth, data }
            })
            .collect();
        rb.Unmap(0, None);
        out
    }

    /// Tiled console memory → PC texture → read back = the linear data it was tiled from, for
    /// every mip and slice: an RGBA8 array with 4 KB tiles, a BC7 cube with 64 KB tiles, 3D
    /// textures (thick 4 KB / 64 KB, BC1 thick, linear) and a linear 2D texture.
    #[test]
    fn gpu_console_textures_upload_every_mip_and_slice() {
        let Some((dev, queue)) = device() else {
            println!("no D3D12 device – skipping");
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        };
        let rgba8 = desc::gfx10_format(DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
        let bc7 = desc::gfx10_format(DXGI_FORMAT_BC7_UNORM).unwrap();
        let bc1 = desc::gfx10_format(DXGI_FORMAT_BC1_UNORM).unwrap();
        let rgba16f = desc::gfx10_format(DXGI_FORMAT_R16G16B16A16_FLOAT).unwrap();
        for t in [
            image(rgba8, ImageType::D2Array, 200, 72, 3, 8, 5, 0x40_0000),
            image(bc7, ImageType::Cube, 256, 256, 6, 9, 10, 0x80_0000),
            image(rgba8, ImageType::D3, 40, 24, 20, 6, 5, 0x100_0000),
            image(rgba16f, ImageType::D3, 64, 64, 64, 7, 9, 0x200_0000),
            image(bc1, ImageType::D3, 64, 32, 8, 4, 5, 0x300_0000),
            image(rgba8, ImageType::D3, 33, 17, 5, 3, 0, 0x400_0000),
            image(rgba8, ImageType::D2, 100, 30, 1, 7, 0, 0x500_0000),
        ] {
            let (want, mem) = console_texture(&t);
            unsafe {
                let res = create(&dev, &queue, &t, &mem, CONSOLE_PIPES).unwrap();
                let got = read_back(&dev, &queue, &res, &want);
                for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(g == w, "format {} {:?} mode {} subresource {k} ({} × {} rows × {} B) differs", t.format, t.ty, t.swizzle_mode, w.depth, w.rows, w.row_bytes);
                }
            }
        }
    }

    /// Batches through one queue share its upload buffer (`Uploader`): two batches of three
    /// textures (mips, an odd size, BC7) each read back what was uploaded, and the second batch
    /// made no new upload buffer.
    #[test]
    fn gpu_upload_batches_reuse_the_queues_upload_buffer() {
        let Some((dev, queue)) = device() else {
            println!("no D3D12 device – skipping");
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        };
        unsafe {
            let make = |w: u64, h: u32, mips: u16, format| -> ID3D12Resource {
                let rd = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: w,
                    Height: h,
                    DepthOrArraySize: 1,
                    MipLevels: mips,
                    Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    ..Default::default()
                };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &rd, SHADER_READ, None, &mut r).unwrap();
                r.unwrap()
            };
            // every subresource of `r`, filled with a pattern of `seed`
            let subs = |r: &ID3D12Resource, seed: u8| -> Vec<Subresource> {
                let rd = r.GetDesc();
                let n = rd.MipLevels as usize;
                let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
                let (mut rows, mut row_bytes) = (vec![0u32; n], vec![0u64; n]);
                dev.GetCopyableFootprints(&rd, 0, n as u32, 0, Some(fps.as_mut_ptr()), Some(rows.as_mut_ptr()), Some(row_bytes.as_mut_ptr()), None);
                (0..n)
                    .map(|k| {
                        let (r, b, d) = (rows[k] as usize, row_bytes[k] as usize, fps[k].Footprint.Depth as usize);
                        Subresource { rows: r, row_bytes: b, depth: d, data: (0..r * b * d).map(|i| (i as u8).wrapping_mul(31) ^ seed ^ k as u8).collect() }
                    })
                    .collect()
            };
            let textures = [make(64, 64, 7, DXGI_FORMAT_R8G8B8A8_UNORM), make(100, 30, 1, DXGI_FORMAT_R8G8B8A8_UNORM), make(256, 256, 9, DXGI_FORMAT_BC7_UNORM)];
            let key = windows::core::Interface::as_raw(&queue) as usize;
            let kept = || UPLOADERS.lock().unwrap().as_ref().and_then(|m| m.get(&key)).and_then(|u| u.staging.as_ref()).map(|s| (windows::core::Interface::as_raw(&s.0) as usize, s.2));
            let mut buffers = Vec::new();
            for round in 0..2u8 {
                let items: Vec<_> = textures.iter().enumerate().map(|(i, r)| (r.clone(), subs(r, round * 16 + i as u8), vec![SHADER_READ])).collect();
                assert!(upload_batch(&dev, &queue, &items).unwrap().is_empty(), "round {round}: an upload failed");
                for (i, (r, want, _)) in items.iter().enumerate() {
                    let got = read_back(&dev, &queue, r, want);
                    for (k, (g, w)) in got.iter().zip(want).enumerate() {
                        assert!(g == w, "round {round}, texture {i}, subresource {k} differs");
                    }
                }
                buffers.push(kept().expect("the queue's upload buffer is kept"));
            }
            assert_eq!(buffers[0], buffers[1], "the second batch made a new upload buffer");
        }
    }

    #[test]
    fn large_staging_uses_sixteen_mib_growth_steps() {
        let mib = 1 << 20;
        assert_eq!(staging_capacity(1), Some(4 * mib));
        assert_eq!(staging_capacity(33 * mib), Some(64 * mib));
        assert_eq!(staging_capacity(64 * mib + 1), Some(80 * mib));
        assert_eq!(staging_capacity(138_713_408), Some(144 * mib));
        assert_eq!(staging_capacity(256 * mib + 1), Some(272 * mib));
        assert_eq!(staging_capacity(u64::MAX), None);
    }

    #[test]
    fn staging_pool_buckets_caps_reuses_and_reclaims_idle() {
        let mib = 1u64 << 20;
        assert_eq!(staging_pool_bucket(1), Some(64 << 10));
        assert_eq!(staging_pool_bucket(3 * mib), Some(4 * mib));
        assert_eq!(staging_pool_bucket(8 * mib), Some(8 * mib));
        assert_eq!(staging_pool_bucket(12 * mib), Some(16 * mib));
        assert_eq!(staging_pool_bucket(32 * mib), Some(32 * mib));
        assert_eq!(staging_pool_bucket(32 * mib + 1), None);

        // Disabled pool (cap 0) immediately returns released items.
        let mut disabled = StagingPool::<u32>::new(0);
        assert_eq!(disabled.release(4 * mib, 1, 0, 0), Some(1));

        // Power-of-2 pooling with 12 MiB cap: LIFO reuse, oldest eviction on cap pressure,
        // and idle reclamation only once BOTH >= 2 s and >= 30 frames have elapsed.
        let mut pool = StagingPool::<u32>::new(12 * mib);
        assert_eq!(pool.release(4 * mib, 10, 100, 1), None);
        assert_eq!(pool.release(8 * mib, 20, 200, 2), None);
        assert_eq!(pool.total_bytes, 12 * mib);
        // Buffers exceeding pool cap or non-power-of-2 buffers bypass the pool.
        assert_eq!(pool.release(16 * mib, 99, 250, 3), Some(99));
        assert_eq!(pool.release(6 * mib, 98, 250, 3), Some(98));
        // Inserting another 4 MiB evicts the oldest (id 10, ts 100).
        assert_eq!(pool.release(4 * mib, 30, 300, 4), None);
        assert_eq!(pool.total_bytes, 12 * mib);
        assert_eq!(pool.acquire(3 * mib), Some((4 * mib, 30)));
        assert_eq!(pool.total_bytes, 8 * mib);
        // Not idle yet: only time elapsed (frames < 30), or only frames elapsed (time < 2000ms).
        assert_eq!(pool.trim_idle(5_000, 10, false), (0, 0));
        assert_eq!(pool.trim_idle(1_000, 100, false), (0, 0));
        // Both thresholds met -> idle 8 MiB buffer reclaimed.
        assert_eq!(pool.trim_idle(2_200, 32, false), (1, 8 * mib));
        assert_eq!(pool.total_bytes, 0);
        // Low-commit forced trim reclaims immediately regardless of idle age.
        assert_eq!(pool.release(4 * mib, 40, 2_200, 32), None);
        assert_eq!(pool.trim_idle(2_200, 32, true), (1, 4 * mib));
        assert_eq!(pool.total_bytes, 0);
    }

    #[test]
    fn cmd_pool_reuses_lifo_enforces_cap_and_handles_disabled() {
        let mut disabled = CmdPool::<u32>::new(0);
        assert_eq!(disabled.release(1), Some(1));
        assert_eq!(disabled.acquire(), None);

        let mut pool = CmdPool::<u32>::new(2);
        assert_eq!(pool.acquire(), None);
        assert_eq!(pool.release(10), None);
        assert_eq!(pool.release(20), None);
        assert_eq!(pool.release(30), Some(30), "cap exceeded must return overflow item to drop");
        assert_eq!(pool.acquire(), Some(20));
        assert_eq!(pool.acquire(), Some(10));
        assert_eq!(pool.acquire(), None);
        assert_eq!(pool.release(40), None);
        pool.clear();
        assert_eq!(pool.acquire(), None);
    }

    #[test]
    fn gpu_submit_batch_reuses_cmd_list_and_16mib_staging_pixel_exact() {
        use windows::core::Interface;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
        unsafe {
            let selector = crate::gpu::adapter::Selector::from_env_or("auto");
            let (dev, _name) = match large_upload_device(&selector) {
                Ok(d) => d,
                Err(e) => {
                    crate::gpu::recomp::runner::skip_test(&format!("no D3D12 device: {e}"));
                    return;
                }
            };
            let dev_key = dev.as_raw() as usize;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();

            // 2048x1536 RGBA8 = 12 MiB -> lands in 16 MiB staging pool bucket (previously unpooled > 8 MiB).
            let (w, h) = (2048u64, 1536u32);
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: w,
                Height: h,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };
            let mut tex: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &rd,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut tex,
            )
            .unwrap();
            let tex = tex.unwrap();

            let row_bytes = (w as usize) * 4;
            let fill1 = vec![0x5Au8; row_bytes * (h as usize)];
            let fill2 = vec![0xA3u8; row_bytes * (h as usize)];
            let mk_item = |data: Vec<u8>| vec![(tex.clone(), vec![Subresource { rows: h as usize, row_bytes, depth: 1, data }], vec![D3D12_RESOURCE_STATE_COMMON])];

            // Batch 1: creates or acquires cmd + 16 MiB staging; wait on GPU fence before dropping InFlight.
            let (failed1, f1) = submit_batch(&dev, &queue, &mk_item(fill1)).unwrap();
            assert!(failed1.is_empty());
            let f1 = f1.expect("in-flight batch 1");
            assert_eq!(f1.bytes, 16 << 20, "12 MiB upload must use 16 MiB pooled staging bucket");
            let list_ptr1 = f1._cmd.cmd.as_ref().unwrap().list.as_raw() as usize;
            wait(&dev, &queue).unwrap();
            drop(f1);

            // Batch 2: must reuse the exact same pooled command list and 16 MiB staging bucket.
            let (failed2, f2) = submit_batch(&dev, &queue, &mk_item(fill2.clone())).unwrap();
            assert!(failed2.is_empty());
            let f2 = f2.expect("in-flight batch 2");
            assert_eq!(f2.bytes, 16 << 20);
            let list_ptr2 = f2._cmd.cmd.as_ref().unwrap().list.as_raw() as usize;
            assert_eq!(list_ptr2, list_ptr1, "completed upload command list must be reused from CMD_POOLS");
            wait(&dev, &queue).unwrap();
            drop(f2);

            // Read back first row of texture to verify bit-exact upload correctness.
            let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let (mut rows, mut rb_row_bytes, mut total) = (0u32, 0u64, 0u64);
            dev.GetCopyableFootprints(&rd, 0, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut rb_row_bytes), Some(&mut total));
            let mut rb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: total,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut rb,
            )
            .unwrap();
            let rb = rb.unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&rb),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&tex),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            wait(&dev, &queue).unwrap();
            let mut p = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut p)).unwrap();
            let got = std::slice::from_raw_parts(p as *const u8, row_bytes);
            assert!(got.iter().all(|&b| b == 0xA3), "readback row must match batch 2 fill byte 0xA3");
            rb.Unmap(0, None);

            // Clean up pooled entries created for this test device.
            if let Some(map) = CMD_POOLS.lock().unwrap().as_mut() {
                map.remove(&dev_key);
            }
            if let Some(map) = STAGING_POOLS.lock().unwrap().as_mut() {
                map.remove(&dev_key);
            }
        }
    }

    /// Offline D3D12 hardware microbenchmark: compares legacy per-batch `CreateCommandAllocator` +
    /// `CreateCommandList` + unpooled 12 MiB `CreateCommittedResource` staging vs pooled `submit_batch`.
    #[test]
    #[ignore]
    fn bench_submit_batch_pooled_vs_legacy_12mib() {
        use windows::core::Interface;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
        unsafe {
            let selector = crate::gpu::adapter::Selector::from_env_or("auto");
            let (dev, name) = large_upload_device(&selector).expect("D3D12 device");
            let dev_key = dev.as_raw() as usize;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();

            let (w, h) = (2048u64, 1536u32); // 12 MiB RGBA8
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: w,
                Height: h,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };
            let mut tex: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &rd,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut tex,
            )
            .unwrap();
            let tex = tex.unwrap();
            let row_bytes = (w as usize) * 4;
            let items = vec![(tex.clone(), vec![Subresource { rows: h as usize, row_bytes, depth: 1, data: vec![0x42u8; row_bytes * (h as usize)] }], vec![D3D12_RESOURCE_STATE_COMMON])];

            const N: usize = 32;
            // Warm up 1 pooled submit so steady-state reuse is measured.
            let (_, warm) = submit_batch(&dev, &queue, &items).unwrap();
            wait(&dev, &queue).unwrap();
            drop(warm);

            // Legacy path: fresh CreateCommandAllocator + CreateCommandList + unpooled CreateCommittedResource (12 MiB > old 8 MiB cap).
            let t_legacy = std::time::Instant::now();
            for _ in 0..N {
                let (_, planned, bytes) = plan_batch(&dev, &items);
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                let staging = buffer(&dev, bytes).unwrap();
                let mut mapped = std::ptr::null_mut();
                staging.Map(0, None, Some(&mut mapped)).unwrap();
                for (k, layout, base) in &planned {
                    let (res, subs, states) = &items[*k];
                    record_into(&list, res, subs, layout, states, states, &staging, mapped as *mut u8, *base);
                }
                staging.Unmap(0, None);
                list.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                wait(&dev, &queue).unwrap();
            }
            let legacy_ms = t_legacy.elapsed().as_secs_f64() * 1000.0;

            // Pooled path: submit_batch reuses both (alloc, list) from CMD_POOLS and 16 MiB buffer from STAGING_POOLS.
            let t_pooled = std::time::Instant::now();
            for _ in 0..N {
                let (_, f) = submit_batch(&dev, &queue, &items).unwrap();
                wait(&dev, &queue).unwrap();
                drop(f);
            }
            let pooled_ms = t_pooled.elapsed().as_secs_f64() * 1000.0;

            println!(
                "BENCH PR #6 ({name}, {N}x 12 MiB RGBA8 submit_batch): legacy={legacy_ms:.2} ms ({:.2} ms/op) vs pooled={pooled_ms:.2} ms ({:.2} ms/op) => {:.2}x speedup",
                legacy_ms / N as f64,
                pooled_ms / N as f64,
                legacy_ms / pooled_ms.max(0.001),
            );

            if let Some(map) = CMD_POOLS.lock().unwrap().as_mut() {
                map.remove(&dev_key);
            }
            if let Some(map) = STAGING_POOLS.lock().unwrap().as_mut() {
                map.remove(&dev_key);
            }
        }
    }

    fn select_large_upload_device_with<F, D>(
        selector: &crate::gpu::adapter::Selector,
        make_factory: impl FnOnce() -> Result<F, String>,
        warp_device: impl FnOnce(&F) -> Result<D, String>,
        hardware_device: impl FnOnce(&F, &crate::gpu::adapter::Selector) -> Option<(D, String)>,
        default_device: impl FnOnce() -> Result<D, String>,
    ) -> Result<(D, String), String> {
        match make_factory() {
            Ok(factory) if *selector == crate::gpu::adapter::Selector::Warp => {
                warp_device(&factory).map(|dev| (dev, "WARP (software renderer)".to_string()))
            }
            Ok(factory) => hardware_device(&factory, selector).ok_or_else(|| "no GPU".to_string()),
            Err(err) if *selector == crate::gpu::adapter::Selector::Auto => default_device()
                .map(|dev| (dev, "default D3D12 adapter".to_string()))
                .map_err(|dev_err| format!("DXGI factory unavailable ({err}); default D3D12 device unavailable ({dev_err})")),
            Err(err) => Err(format!("DXGI factory unavailable for {selector}: {err}")),
        }
    }

    unsafe fn large_upload_device(selector: &crate::gpu::adapter::Selector) -> Result<(ID3D12Device, String), String> {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory6};
        select_large_upload_device_with(
            selector,
            || CreateDXGIFactory2::<IDXGIFactory6>(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| e.to_string()),
            |factory| {
                let warp: IDXGIAdapter1 = factory.EnumWarpAdapter().map_err(|e| format!("WARP adapter unavailable: {e}"))?;
                crate::gpu::agility::create_device(&warp, D3D_FEATURE_LEVEL_11_0).map_err(|e| format!("WARP device unavailable: {e}"))
            },
            |factory, sel| {
                crate::gpu::adapter::candidates(factory, sel).into_iter().find_map(|(a, desc)| {
                    crate::gpu::agility::create_device(&a, D3D_FEATURE_LEVEL_11_0).ok().map(|dev| (dev, crate::gpu::adapter::name(&desc)))
                })
            },
            || crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0).map_err(|e| e.to_string()),
        )
    }

    #[test]
    fn large_upload_device_selection_handles_factory_fallback_and_selectors() {
        use crate::gpu::adapter::Selector;
        // 1. Auto + DXGI_ERROR_UNSUPPORTED falls back to default device.
        let mut used_default = false;
        let ok = select_large_upload_device_with(&Selector::Auto, || Err::<(), _>("DXGI_ERROR_UNSUPPORTED (0x887A0004)".into()), |_| unreachable!(), |_, _| unreachable!(), || { used_default = true; Ok(42u32) });
        assert!(used_default && ok == Ok((42, "default D3D12 adapter".to_string())));
        // 2. Auto + double failure returns Err without panicking.
        let err = select_large_upload_device_with(&Selector::Auto, || Err::<(), _>("E_FAIL".into()), |_| unreachable!(), |_, _| unreachable!(), || Err::<u32, _>("no D3D12 device".into()));
        assert!(err.as_ref().is_err_and(|e| e.contains("E_FAIL") && e.contains("no D3D12 device")));
        // 3. Explicit XCL_ADAPTER filter + factory error must not ignore filter via default fallback.
        for sel in [Selector::Name("rtx".into()), Selector::Index(1), Selector::Warp] {
            let mut fallback = false;
            let res = select_large_upload_device_with(&sel, || Err::<(), _>("DXGI_ERROR_UNSUPPORTED".into()), |_| unreachable!(), |_, _| unreachable!(), || { fallback = true; Ok(99u32) });
            assert!(!fallback && res.is_err());
        }
        // 4. Headless CI: Auto + factory ok with 0 hardware adapters skips cleanly without fallback.
        let mut fallback = false;
        let none = select_large_upload_device_with(&Selector::Auto, || Ok(()), |_| unreachable!(), |_, _| None::<(u32, String)>, || { fallback = true; Ok(7u32) });
        assert!(!fallback && none == Err("no GPU".to_string()));
        // 5. Selector::Warp + factory ok uses EnumWarpAdapter path.
        let warp = select_large_upload_device_with(&Selector::Warp, || Ok(()), |_| Ok(11u32), |_, _| unreachable!(), || unreachable!());
        assert_eq!(warp, Ok((11, "WARP (software renderer)".to_string())));
    }

    #[test]
    fn gpu_large_mixed_upload_keeps_tight_staging_and_exact_pixels() {
        use windows::core::Interface;

        unsafe {
            if std::env::var_os("XCL_D3D12_DEBUG").is_some() {
                match crate::gpu::agility::configuration_interface::<ID3D12Debug>(&CLSID_D3D12Debug) {
                    Ok(debug) => debug.EnableDebugLayer(),
                    Err(error) => println!("D3D12 debug layer unavailable: {error}"),
                }
            }
            let selector = crate::gpu::adapter::Selector::from_env_or("auto");
            let (dev, adapter_name) = match large_upload_device(&selector) {
                Ok(selected) => selected,
                Err(reason) => {
                    crate::gpu::recomp::runner::skip_test(&reason);
                    return;
                }
            };
            println!("large upload adapter: {adapter_name}");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let make = |width, height, format| {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: width, Height: height,
                    DepthOrArraySize: 1, MipLevels: 1, Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default()
                };
                let mut out = None::<ID3D12Resource>;
                let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
                dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, SHADER_READ, None, &mut out).unwrap();
                out.unwrap()
            };
            let pixels = |rows: usize, row_bytes: usize, seed: u8| {
                let data = (0..rows * row_bytes).map(|i| (i as u8).wrapping_mul(31) ^ (i >> 13) as u8 ^ seed).collect();
                vec![Subresource { depth: 1, rows, row_bytes, data }]
            };
            let rgba = (make(8192, 4096, DXGI_FORMAT_R8G8B8A8_UNORM), pixels(4096, 32768, 27), vec![SHADER_READ]);
            let bc7 = (make(2048, 2048, DXGI_FORMAT_BC7_UNORM), pixels(512, 8192, 91), vec![SHADER_READ]);
            let key = queue.as_raw() as usize;
            let kept = || {
                let guard = UPLOADERS.lock().unwrap();
                let stage = guard.as_ref().unwrap()[&key].staging.as_ref().unwrap();
                (stage.0.as_raw() as usize, stage.2)
            };
            let batch = [rgba, bc7];
            assert!(upload_batch(&dev, &queue, &batch[..1]).unwrap().is_empty());
            let first = kept();
            let before = std::time::Instant::now();
            assert!(upload_batch(&dev, &queue, &batch).unwrap().is_empty());
            let growth = before.elapsed();
            let grown = kept();
            for (resource, bytes, _) in &batch {
                assert!(read_back(&dev, &queue, resource, bytes) == *bytes, "mixed batch pixel mismatch");
            }
            let before = std::time::Instant::now();
            assert!(upload_batch(&dev, &queue, &batch).unwrap().is_empty());
            let repeat = before.elapsed();
            let reused = kept();
            for (resource, bytes, _) in &batch {
                assert!(read_back(&dev, &queue, resource, bytes) == *bytes, "repeat pixel mismatch");
            }
            // Drop the test's large uploader after its last copy fence and readbacks have passed.
            let uploader = UPLOADERS.lock().unwrap().as_mut().unwrap().remove(&key);
            drop(uploader);
            if let Ok(info) = dev.cast::<ID3D12InfoQueue>() {
                let mut errors = 0;
                for index in 0..info.GetNumStoredMessagesAllowedByRetrievalFilter() {
                    let mut size = 0;
                    info.GetMessage(index, None, &mut size).unwrap();
                    let mut buffer = vec![0usize; size.div_ceil(std::mem::size_of::<usize>())];
                    let msg = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    info.GetMessage(index, Some(msg), &mut size).unwrap();
                    if matches!((*msg).Severity, D3D12_MESSAGE_SEVERITY_ERROR | D3D12_MESSAGE_SEVERITY_CORRUPTION) {
                        errors += 1;
                    }
                }
                println!("D3D12 debug errors or corruption messages: {errors}");
                assert_eq!(errors, 0);
            }
            println!("large upload: request={} B first_capacity={} B grown_capacity={} B growth={growth:?} repeat={repeat:?} byte_exact=true",
                (128 + 4) << 20, first.1, grown.1);
            assert_eq!(first.1, 128 << 20);
            assert_eq!(grown, reused, "the repeated batch must reuse the staging buffer");
            assert_eq!(grown.1, 144 << 20, "staging for 132 MiB must fit the next large size class");
        }
    }

    /// Mean |level `k` - level `k - 1` scaled down 2×| per channel (0..255 scale) of a texture
    /// with `ch` channels of `cb` bytes (unorm); None when the sizes do not halve.
    fn mip_error(big: &Subresource, small: &Subresource, ch: usize, cb: usize) -> Option<f64> {
        let e = ch * cb;
        let (bw, bh, sw, sh) = (big.row_bytes / e, big.rows, small.row_bytes / e, small.rows);
        if sw * 2 != bw || sh * 2 != bh {
            return None;
        }
        let get = |d: &[u8], w: usize, x: usize, y: usize, c: usize| -> u32 {
            let o = (y * w + x) * e + c * cb;
            if cb == 1 {
                d[o] as u32
            } else {
                u16::from_le_bytes([d[o], d[o + 1]]) as u32 >> 8
            }
        };
        let mut sum = 0u64;
        for y in 0..sh {
            for x in 0..sw {
                for c in 0..ch {
                    let avg = (get(&big.data, bw, 2 * x, 2 * y, c) + get(&big.data, bw, 2 * x + 1, 2 * y, c) + get(&big.data, bw, 2 * x, 2 * y + 1, c) + get(&big.data, bw, 2 * x + 1, 2 * y + 1, c) + 2) / 4;
                    sum += (avg as i32 - get(&small.data, sw, x, y, c) as i32).unsigned_abs() as u64;
                }
            }
        }
        Some(sum as f64 / (sw * sh * ch) as f64)
    }

    /// The retail TTPF archive (`XCL_SAMPLE_TTPF`): (1) the mip chain model gives the space
    /// every texture takes; (2) on its uncompressed unorm textures each detiled mip level is
    /// close to the level above it scaled down 2× (mip tail levels included), while the same
    /// tail levels read from a wrong place in the tail block (origin 0, 0) are not.
    #[test]
    fn retail_ttpf_textures_detile_with_their_mip_chains() {
        let Ok(path) = std::env::var("XCL_SAMPLE_TTPF") else {
            println!("XCL_SAMPLE_TTPF not set – skipping");
            return;
        };
        let a = Ttpf::open(&path);
        let (mut fit, mut sized, mut unreadable) = (0, 0, 0);
        let rgba8 = desc::gfx10_format(DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
        let (mut checked, mut close, mut worst) = (0, 0, Vec::new());
        let (mut tail_levels, mut tail_err, mut wrong_err) = (0, 0.0f64, 0.0f64);
        let mut volumes = Vec::new();
        for t in &a.textures {
            let Some((e, side, _)) = ttpf_element(t.dxgi) else { continue };
            let gfx10 = match (e, side) {
                (0, 1) => 1,
                (1, 1) => 7,
                (2, 1) => rgba8,
                (3, 1) => 62,
                (3, 4) => 169,
                _ => 171,
            };
            let ty = match t.dim {
                3 => ImageType::D3,
                2 => ImageType::Cube,
                _ if t.slices > 1 => ImageType::D2Array,
                _ => ImageType::D2,
            };
            let img = image(gfx10, ty, t.w, t.h, t.slices, t.mips, t.mode, 0);
            let p = placement(&img, CONSOLE_PIPES).unwrap();
            sized += 1;
            if p.bytes == t.end - t.start || p.bytes.div_ceil(4096) * 4096 == t.end - t.start {
                fit += 1;
            }
            // volumes of 16-bit float texels (colour-grading LUTs): neighbours along x and z
            // differ little read as thick blocks, a lot read as 2D slices
            if t.dim == 3 && matches!(t.dxgi, 10 | 54) {
                if let Some(mem) = a.read(t.start, t.end) {
                    let thick = subresources(&img, &mem, CONSOLE_PIPES).unwrap().remove(0);
                    let flat = image(gfx10, ImageType::D2Array, t.w, t.h, t.slices, t.mips, t.mode, 0);
                    let as_2d: Vec<u8> = subresources(&flat, &mem, CONSOLE_PIPES).unwrap().into_iter().flat_map(|s| s.data).collect();
                    let (w, h, d, e) = (t.w as usize, t.h as usize, t.slices as usize, 1usize << e);
                    volumes.push((t.index, roughness(&thick.data, w, h, d, e), roughness(&as_2d, w, h, d, e)));
                }
            }
            // unorm channels: R8 / A8, RGBA8 / BGRA8, RGBA16
            let (ch, cb) = match t.dxgi {
                61 | 65 => (1, 1),
                27..=29 | 87 | 90 | 91 => (4, 1),
                11 => (4, 2),
                _ => continue,
            };
            if t.mips < 2 || t.slices != 1 {
                continue;
            }
            let Some(mem) = a.read(t.start, t.start + p.bytes) else {
                unreadable += 1;
                continue;
            };
            let subs = subresources(&img, &mem, CONSOLE_PIPES).unwrap();
            let mut bad = 0.0f64;
            for k in 1..subs.len() {
                let Some(err) = mip_error(&subs[k - 1], &subs[k], ch, cb) else { continue };
                bad = bad.max(err);
                // control: a tail level read from the tail block's corner
                let chain = &p.chain;
                let l = chain.levels[k];
                if l.origin != (0, 0) && l.width >= 2 && l.height >= 2 {
                    let mut wrong = chain.clone();
                    wrong.levels[k].origin = (0, 0);
                    let ws = Subresource { depth: 1, rows: l.height, row_bytes: subs[k].row_bytes, data: wrong.detile_level(&mem, k, 0) };
                    tail_levels += 1;
                    tail_err += err;
                    wrong_err += mip_error(&subs[k - 1], &ws, ch, cb).unwrap();
                }
            }
            checked += 1;
            if bad < 8.0 {
                close += 1;
            } else if worst.len() < 8 {
                worst.push((t.index, t.w, t.h, t.mips, t.mode, bad));
            }
        }
        let n = tail_levels.max(1) as f64;
        println!("TTPF: the mip chain model size matches for {fit} of {sized} textures");
        println!("TTPF: textures with mips (R8/RGBA8/RGBA16): {close} of {checked} – every mip close to the previous one downscaled (< 8/255), {unreadable} in unreadable blocks; outliers: {worst:?}");
        println!("TTPF: {tail_levels} levels in the mip tail – error {:.2}/255, at the wrong place in the block {:.2}/255", tail_err / n, wrong_err / n);
        println!("TTPF: volumes (index, neighbour difference as thick blocks, as 2D slices): {volumes:?}");
        assert!(fit * 100 >= sized * 99, "mip chain model: {fit} / {sized}");
        assert!(checked >= 5 && close * 100 >= checked * 90, "mips: {close} / {checked}");
        assert!(tail_levels >= 5 && wrong_err > 3.0 * tail_err, "mip tail: {tail_err} vs {wrong_err}");
        assert!(!volumes.is_empty() && volumes.iter().all(|v| v.2 > 3.0 * v.1), "volumes: {volumes:?}");
    }

    /// Mean |difference| between x and z neighbours of the first channel (half floats, `e`-byte
    /// texels) of a `w` × `h` × `d` volume, relative to the mean |value|.
    fn roughness(v: &[u8], w: usize, h: usize, d: usize, e: usize) -> f64 {
        let half = |b: u16| -> f64 {
            let (s, x, m) = ((b >> 15) as i32, ((b >> 10) & 31) as i32, (b & 1023) as f64);
            let v = if x == 0 { m / 1024.0 * 2f64.powi(-14) } else { (1.0 + m / 1024.0) * 2f64.powi(x - 15) };
            if s == 1 { -v } else { v }
        };
        let at = |x: usize, y: usize, z: usize| half(u16::from_le_bytes([v[((z * h + y) * w + x) * e], v[((z * h + y) * w + x) * e + 1]]));
        let (mut diff, mut mag) = (0.0, 0.0);
        for z in 0..d - 1 {
            for y in 0..h {
                for x in 0..w - 1 {
                    let c = at(x, y, z);
                    diff += (at(x + 1, y, z) - c).abs() + (at(x, y, z + 1) - c).abs();
                    mag += c.abs();
                }
            }
        }
        diff / mag.max(1e-9)
    }

    /// (element log2, pixels per side, uncompressed) of the DXGI formats the archive uses
    fn ttpf_element(dxgi: u32) -> Option<(u32, u32, bool)> {
        Some(match dxgi {
            70..=72 | 79..=81 => (3, 4, false),
            73..=78 | 82..=84 | 94..=99 => (4, 4, false),
            10 | 11 => (3, 1, true),
            27..=29 | 87 | 90 | 91 => (2, 1, true),
            54 => (1, 1, true),
            61 | 65 => (0, 1, true),
            _ => return None,
        })
    }

    struct TtpfTexture {
        index: usize,
        w: u32,
        h: u32,
        dxgi: u32,
        mode: u32,
        mips: u32,
        slices: u32,
        /// dword 1 bits 21-23: 1 2D, 2 cube, 3 volume (slices = depth), 5 2D array
        dim: u32,
        start: usize,
        end: usize,
    }

    /// Just enough of a .ttpf archive: its zlib blocks of one unpacked stream and the texture
    /// table.
    struct Ttpf {
        d: Vec<u8>,
        /// (file offset, position in the stream), by position
        blocks: Vec<(usize, usize)>,
        textures: Vec<TtpfTexture>,
        cache: std::cell::RefCell<HashMap<usize, Option<std::rc::Rc<Vec<u8>>>>>,
    }

    impl Ttpf {
        fn open(path: &str) -> Ttpf {
            let d = std::fs::read(path).unwrap();
            let u32_at = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
            assert_eq!(&d[..4], b"RDS5");
            let total = u32_at(8) as usize | (u32_at(0xc) as usize & 0xff) << 32;
            let at = (u32_at(0x10) as usize) << 8;
            let n = (u32_at(4) as usize).min((at - 0x18) / 8);
            let mut blocks: Vec<(usize, usize)> = (0..n)
                .filter_map(|i| {
                    let v = u64::from_le_bytes(d[0x18 + 8 * i..0x20 + 8 * i].try_into().unwrap());
                    let pos = ((v >> 24) & 0xff_ffff) | ((v >> 48) & 0xf) << 24;
                    (v != 0).then_some((((v & 0xff_ffff) << 8) as usize, (pos << 6) as usize))
                })
                .collect();
            blocks.sort_by_key(|b| b.1);
            let count = u32_at(at + 8) as usize;
            let mut textures: Vec<TtpfTexture> = (0..count)
                .map(|i| {
                    let o = at + 16 + 16 * i;
                    let (a, b, c) = (u32_at(o), u32_at(o + 4), u32_at(o + 8));
                    TtpfTexture {
                        index: i,
                        w: (a & 0x3fff) + 1,
                        h: ((a >> 14) & 0x3fff) + 1,
                        dxgi: (b & 0x3ff) >> 3,
                        // tile mode bits: 0 = 64 KB tiles (SW_64KB_D), else 4 KB (SW_4KB_S)
                        mode: if (b >> 24) & 3 == 0 { 10 } else { 5 },
                        mips: (a >> 28).max(1),
                        slices: ((b >> 10) & 0x7ff) + 1,
                        dim: (b >> 21) & 7,
                        start: (c & 0x1f_ffff) as usize * 4096 + 0x800,
                        end: total,
                    }
                })
                .collect();
            let mut by_start: Vec<usize> = (0..count).collect();
            by_start.sort_by_key(|&i| textures[i].start);
            for k in 0..count.saturating_sub(1) {
                textures[by_start[k]].end = textures[by_start[k + 1]].start;
            }
            Ttpf { d, blocks, textures, cache: Default::default() }
        }

        /// bytes [a, b) of the unpacked stream (None: in a block that is not zlib)
        fn read(&self, a: usize, b: usize) -> Option<Vec<u8>> {
            let mut out = Vec::with_capacity(b - a);
            let mut pos = a;
            while pos < b {
                let i = self.blocks.partition_point(|bl| bl.1 <= pos).checked_sub(1)?;
                let data = self.block(i)?;
                let from = pos - self.blocks[i].1;
                let to = (b - self.blocks[i].1).min(data.len());
                if to <= from {
                    return None;
                }
                out.extend_from_slice(&data[from..to]);
                pos += to - from;
            }
            Some(out)
        }

        fn block(&self, i: usize) -> Option<std::rc::Rc<Vec<u8>>> {
            self.cache
                .borrow_mut()
                .entry(i)
                .or_insert_with(|| {
                    let off = self.blocks[i].0;
                    let next = self.blocks.iter().map(|b| b.0).filter(|&o| o > off).min().unwrap_or(self.d.len());
                    let src = self.d.get(off..next)?;
                    (src.first() == Some(&0x78)).then(|| miniz_oxide::inflate::decompress_to_vec_zlib(src).ok().map(std::rc::Rc::new)).flatten()
                })
                .clone()
        }
    }
}
