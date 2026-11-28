//! Enhanced barriers of console code: `Barrier(NumBarrierGroups, pBarrierGroups)` (the PC's
//! ID3D12GraphicsCommandList7 #80) as legacy `ResourceBarrier` calls.
//!
//! The console list has it at #146 (XboxBvhBuilder.dll calls it with (1, a description) between
//! its passes; a retail game on the newer layout calls it there too, with texture barrier groups).
//! That game creates its resources with barrier layouts (device #127 / #133) and moves them with
//! Barrier only: taken as a UAV barrier, every transition was dropped and its render targets
//! stayed in the layout they were created in (the debug layer: a target drawn to while in
//! NON_PIXEL_SHADER_RESOURCE | PIXEL_SHADER_RESOURCE).
//!
//! Translation:
//! * texture barriers → transitions, layouts → D3D12_RESOURCE_STATES by the same table as at
//!   resource creation (`layout_state`); an UNDEFINED layout before → the state the resource was
//!   last given (`STATES`: at creation with a layout, by legacy and enhanced barriers), COMMON if
//!   not known; a subresource range → its subresources (all of them: one barrier);
//! * global and buffer barriers, a texture staying in UNORDERED_ACCESS, a transition from or to
//!   UNORDERED_ACCESS → one UAV barrier for all resources per call (translated shaders reach
//!   title memory through XCL's window buffers, which a barrier of the title's resource alone does
//!   not order – as `list_x::resource_barrier_x`);
//! * an UNDEFINED layout before on a placed texture (memory reused) → an aliasing barrier for
//!   any resources first.
//!
//! Buffers have no layout: in the legacy model they stay in COMMON (promotion / decay).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// D3D12_BARRIER_TYPE
pub const TYPE_GLOBAL: u32 = 0;
pub const TYPE_TEXTURE: u32 = 1;
pub const TYPE_BUFFER: u32 = 2;

/// D3D12_BARRIER_LAYOUT_UNDEFINED
pub const LAYOUT_UNDEFINED: u32 = u32::MAX;
/// the last D3D12_BARRIER_LAYOUT (DIRECT_QUEUE_GENERIC_READ_COMPUTE_QUEUE_ACCESSIBLE, 0x1F)
#[cfg(test)]
const LAYOUT_LAST: u32 = 31;

/// D3D12_BARRIER_ACCESS_UNORDERED_ACCESS
const ACCESS_UNORDERED_ACCESS: u32 = 0x10;
/// D3D12_BARRIER_ACCESS_NO_ACCESS
const ACCESS_NO_ACCESS: u32 = 0x8000_0000;
/// D3D12_BARRIER_SYNC_NONE
const SYNC_NONE: u32 = 0;

/// sizes of the barrier structures (x64 d3d12.h)
const GROUP_SIZE: usize = 0x10;
const GLOBAL_SIZE: usize = 0x10;
const TEXTURE_SIZE: usize = 0x40;
const BUFFER_SIZE: usize = 0x28;

/// more groups / barriers per group than this: not a Barrier call
const MAX_GROUPS: u32 = 1024;
const MAX_BARRIERS: u32 = 1 << 16;
/// subresource ranges with more subresources than this: the whole resource
const MAX_SUBRESOURCES: usize = 1024;

/// D3D12_RESOURCE_STATES bits (VERTEX_AND_CONSTANT_BUFFER | INDEX_BUFFER | INDIRECT_ARGUMENT)
const STATE_BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x200;
const STATE_UNORDERED_ACCESS: u32 = 0x8;
const ALL_SUBRESOURCES: u32 = 0xFFFF_FFFF;

/// D3D12_RESOURCE_BARRIER (32 bytes) as words: type, flags, then the union at +8
pub type Legacy = [u32; 8];
const LEGACY_TRANSITION: u32 = 0;
const LEGACY_ALIASING: u32 = 1;
const LEGACY_UAV: u32 = 2;

/// The resource state (D3D12_RESOURCE_STATES) of a D3D12_BARRIER_LAYOUT; None: UNDEFINED or
/// not a layout. Textures have no buffer-only states (GENERIC_READ without
/// VERTEX_AND_CONSTANT_BUFFER / INDEX_BUFFER / INDIRECT_ARGUMENT).
pub fn layout_state(layout: u32, texture: bool) -> Option<u32> {
    let generic_read = if texture { 0x8C0 } else { 0xAC3 };
    let compute_generic_read = if texture { 0x840 } else { 0xA41 };
    Some(match layout {
        0 | 18 | 24 | 30 => 0,      // common / present, each queue's common
        1 | 19 | 31 => generic_read, // generic read, direct queue generic read (+ compute-accessible 0x1F)
        25 => compute_generic_read,  // the compute queue's generic read: no pixel shader resource
        2 => 0x4,                   // render target
        3 | 20 | 26 => 0x8,         // unordered access
        4 => 0x10,                  // depth write
        5 => 0x20,                  // depth read
        6 | 21 => 0xC0,             // shader resource (all stages)
        27 => 0x40,                 // the compute queue's shader resource
        7 | 22 | 28 => 0x800,       // copy source
        8 | 23 | 29 => 0x400,       // copy dest
        9 => 0x2000,                // resolve source
        10 => 0x1000,               // resolve dest
        11 => 0x100_0000,           // shading rate source
        12..=17 => 0,               // video: COMMON
        _ => return None,
    })
}

/// layouts past the PC's (any console-only layout > 0x1F) – translated by their accesses (`access_state`)
const LAYOUT_MAX_CONSOLE: u32 = 0xFFFF;

/// The state of a texture in a layout the PC does not have, from the barrier's access to it:
/// none (NO_ACCESS) / COMMON → COMMON.
pub fn access_state(access: u32) -> u32 {
    if access & ACCESS_NO_ACCESS != 0 {
        return 0;
    }
    [(0x8, 0x4), (0x10, 0x8), (0x20, 0x10), (0x40, 0x20), (0x80, 0xC0), (0x400, 0x400), (0x800, 0x800), (0x1000, 0x1000), (0x2000, 0x2000), (0x10000, 0x100_0000)]
        .iter()
        .filter(|(a, _)| access & a != 0)
        .fold(0, |s, (_, st)| s | st)
}

/// A texture's state in `layout` (None: UNDEFINED), by `access` for a console-only layout.
fn texture_state(layout: u32, access: u32) -> Option<u32> {
    if layout == LAYOUT_UNDEFINED {
        return None;
    }
    Some(layout_state(layout, true).unwrap_or_else(|| {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 8 {
            crate::xlog_info!("D3D12.X: Barrier: console layout {layout:#x} (access {access:#x}) taken as state {:#x}", access_state(access));
        }
        access_state(access)
    }))
}

/// The D3D12_RESOURCE_STATES a list of `list_type` may name in a transition (the PC refuses the
/// others there: a compute list moving a render target fails).
pub fn allowed_states(list_type: i32) -> u32 {
    match list_type {
        // compute: vertex/constant buffer, UAV, non-pixel shader resource, indirect argument,
        // copy dest / source, raytracing structure
        2 => 0x1 | 0x8 | 0x40 | 0x200 | 0x400 | 0x800 | 0x40_0000,
        // copy
        3 => 0x400 | 0x800,
        _ => u32::MAX,
    }
}

/// SyncBefore, SyncAfter, AccessBefore, AccessAfter
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Access {
    pub sync_before: u32,
    pub sync_after: u32,
    pub access_before: u32,
    pub access_after: u32,
}

impl Access {
    fn read(w: &[u32]) -> Access {
        Access { sync_before: w[0], sync_after: w[1], access_before: w[2], access_after: w[3] }
    }

    /// waits on earlier work (a barrier with SyncBefore NONE / AccessBefore NO_ACCESS orders
    /// nothing before it)
    fn orders_earlier_work(&self) -> bool {
        self.sync_before != SYNC_NONE && self.access_before != ACCESS_NO_ACCESS
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Texture {
    pub access: Access,
    pub layout_before: u32,
    pub layout_after: u32,
    pub resource: usize,
    /// IndexOrFirstMipLevel, NumMipLevels, FirstArraySlice, NumArraySlices, FirstPlane, NumPlanes
    pub range: [u32; 6],
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    Global(Access),
    Texture(Texture),
    Buffer { access: Access, resource: usize },
}

/// What the translation needs to know of a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexInfo {
    pub buffer: bool,
    pub mips: u32,
    pub arrays: u32,
    pub planes: u32,
    /// a placed texture (its memory may be another resource's before)
    pub placed: bool,
    /// D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS on the PC
    pub uav: bool,
}

/// Read `n` D3D12_BARRIER_GROUPs at `groups` (console memory; `readable` checks each range).
///
/// # Safety
/// `readable(p, len)` must only say yes for memory that can be read.
pub unsafe fn parse(n: u32, groups: usize, readable: &dyn Fn(usize, usize) -> bool) -> Result<Vec<Item>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if n > MAX_GROUPS {
        return Err(format!("{n} groups"));
    }
    if groups == 0 || groups % 8 != 0 || !readable(groups, n as usize * GROUP_SIZE) {
        return Err(format!("groups at {groups:#x}"));
    }
    let mut out = Vec::new();
    for g in 0..n as usize {
        let p = groups + g * GROUP_SIZE;
        let ty = (p as *const u32).read_unaligned();
        let count = ((p + 4) as *const u32).read_unaligned();
        let barriers = ((p + 8) as *const u64).read_unaligned() as usize;
        let size = match ty {
            TYPE_GLOBAL => GLOBAL_SIZE,
            TYPE_TEXTURE => TEXTURE_SIZE,
            TYPE_BUFFER => BUFFER_SIZE,
            _ => return Err(format!("group {g}: type {ty}")),
        };
        if count == 0 {
            continue;
        }
        if count > MAX_BARRIERS || barriers == 0 || barriers % 4 != 0 || !readable(barriers, count as usize * size) {
            return Err(format!("group {g}: {count} barriers at {barriers:#x}"));
        }
        for k in 0..count as usize {
            let b = barriers + k * size;
            let mut w = [0u32; TEXTURE_SIZE / 4];
            std::ptr::copy_nonoverlapping(b as *const u8, w.as_mut_ptr() as *mut u8, size);
            let access = Access::read(&w);
            let ptr = |at: usize| (w[at / 4] as u64 | (w[at / 4 + 1] as u64) << 32) as usize;
            out.push(match ty {
                TYPE_GLOBAL => Item::Global(access),
                TYPE_TEXTURE => {
                    let (layout_before, layout_after) = (w[4], w[5]);
                    let valid = |l: u32| l <= LAYOUT_MAX_CONSOLE || l == LAYOUT_UNDEFINED;
                    if !valid(layout_before) || !valid(layout_after) {
                        return Err(format!("group {g} barrier {k}: layouts {layout_before:#x} → {layout_after:#x}"));
                    }
                    let resource = ptr(0x18);
                    if resource == 0 || !readable(resource, 8) {
                        return Err(format!("group {g} barrier {k}: texture {resource:#x}"));
                    }
                    Item::Texture(Texture { access, layout_before, layout_after, resource, range: [w[8], w[9], w[10], w[11], w[12], w[13]], flags: w[14] })
                }
                _ => Item::Buffer { access, resource: ptr(0x10) },
            });
        }
    }
    Ok(out)
}

/// The subresource indices of an enhanced barrier's range (`[ALL_SUBRESOURCES]`: all of them).
pub fn subresources(r: [u32; 6], info: &TexInfo) -> Vec<u32> {
    if r[0] == u32::MAX {
        return vec![ALL_SUBRESOURCES];
    }
    if r[1] == 0 {
        // NumMipLevels 0: IndexOrFirstMipLevel is a subresource index
        return vec![r[0]];
    }
    let (mips, arrays, planes) = (info.mips.max(1), info.arrays.max(1), info.planes.max(1));
    let (m0, a0, p0) = (r[0], r[2], r[4]);
    if m0 >= mips || a0 >= arrays || p0 >= planes {
        return Vec::new();
    }
    let nm = r[1].min(mips - m0);
    let na = r[3].max(1).min(arrays - a0);
    let np = r[5].max(1).min(planes - p0);
    if m0 == 0 && nm == mips && a0 == 0 && na == arrays && p0 == 0 && np == planes {
        return vec![ALL_SUBRESOURCES];
    }
    if nm as usize * na as usize * np as usize > MAX_SUBRESOURCES {
        return vec![ALL_SUBRESOURCES];
    }
    let mut out = Vec::with_capacity((nm * na * np) as usize);
    for p in p0..p0 + np {
        for a in a0..a0 + na {
            for m in m0..m0 + nm {
                out.push(m + a * mips + p * mips * arrays);
            }
        }
    }
    out
}

/// Legacy barriers for enhanced ones, and the states the resources are left in.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Translation {
    pub barriers: Vec<Legacy>,
    /// (resource, state of all its subresources) after the barriers
    pub states: Vec<(usize, u32)>,
    /// transitions a list of this type cannot make (left out)
    pub refused: usize,
}

fn transition(resource: usize, sub: u32, before: u32, after: u32) -> Legacy {
    [LEGACY_TRANSITION, 0, resource as u32, (resource as u64 >> 32) as u32, sub, before, after, 0]
}

/// Translate parsed barriers. `info`: what is known of a texture (None: not a resource XCL can
/// ask); `tracked`: the state a resource was last given; `allowed`: `allowed_states` of the list.
pub fn translate(items: &[Item], info: &dyn Fn(usize) -> Option<TexInfo>, tracked: &dyn Fn(usize) -> Option<u32>, allowed: u32) -> Translation {
    let mut t = Translation::default();
    let (mut uav, mut alias) = (false, false);
    let mut moves: Vec<Legacy> = Vec::new();
    // states given earlier in this call (a resource moved twice in one call)
    let mut local: HashMap<usize, u32> = HashMap::new();
    for item in items {
        let tex = match *item {
            Item::Global(a) | Item::Buffer { access: a, .. } => {
                uav |= a.orders_earlier_work();
                continue;
            }
            Item::Texture(tex) => tex,
        };
        let Some(ti) = info(tex.resource) else { continue };
        if ti.buffer {
            // a texture barrier naming a buffer: its sync only
            uav |= tex.access.orders_earlier_work();
            continue;
        }
        let Some(after) = texture_state(tex.layout_after, tex.access.access_after) else { continue };
        // the state the resource was left in (by this call, before it, at creation); a texture the
        // PC has without ALLOW_UNORDERED_ACCESS (depth with UAV on the console) is never in it
        let known = || {
            let s = local.get(&tex.resource).copied().or_else(|| tracked(tex.resource)).unwrap_or(0) & !STATE_BUFFER_ONLY;
            if !ti.uav && s == STATE_UNORDERED_ACCESS {
                0
            } else {
                s
            }
        };
        if !ti.uav && after == STATE_UNORDERED_ACCESS {
            // it stays where it is (its UAV views are null on the PC)
            uav |= tex.access.orders_earlier_work();
            continue;
        }
        let before = match texture_state(tex.layout_before, tex.access.access_before) {
            Some(s) if ti.uav || s != STATE_UNORDERED_ACCESS => s,
            Some(_) => known(),
            None => {
                alias |= ti.placed;
                known()
            }
        };
        if before == after {
            // the layout stays: UAV writes before need a UAV barrier
            uav |= after & STATE_UNORDERED_ACCESS != 0 || (tex.access.access_before & ACCESS_UNORDERED_ACCESS != 0 && tex.access.orders_earlier_work());
            continue;
        }
        if (before | after) & !allowed != 0 {
            t.refused += 1;
            continue;
        }
        uav |= (before | after) & STATE_UNORDERED_ACCESS != 0;
        let subs = subresources(tex.range, &ti);
        if subs == [ALL_SUBRESOURCES] {
            local.insert(tex.resource, after);
            t.states.retain(|e| e.0 != tex.resource);
            t.states.push((tex.resource, after));
        }
        moves.extend(subs.into_iter().map(|s| transition(tex.resource, s, before, after)));
    }
    if alias {
        t.barriers.push([LEGACY_ALIASING, 0, 0, 0, 0, 0, 0, 0]);
    }
    t.barriers.extend(moves);
    if uav {
        t.barriers.push([LEGACY_UAV, 0, 0, 0, 0, 0, 0, 0]);
    }
    t
}

// ---------------------------------------------------------------------------------------
// resource states as last given (for UNDEFINED layouts before)
// ---------------------------------------------------------------------------------------

static STATES: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);

/// `resource` (all its subresources) is in `state` from now on (creation, a barrier).
pub fn note_state(resource: usize, state: u32) {
    if resource != 0 {
        STATES.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(resource, state);
    }
}

/// The state `resource` was last given.
pub fn tracked_state(resource: usize) -> Option<u32> {
    STATES.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&resource).copied())
}

/// Drop tracked state when a resource is released so a reused pointer starts fresh.
pub fn forget(resource: usize) {
    if let Some(m) = STATES.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.remove(&resource);
    }
}

/// Note the states legacy barriers leave whole resources in.
pub fn note_legacy(barriers: &[Legacy]) {
    for b in barriers {
        if b[0] == LEGACY_TRANSITION && b[4] == ALL_SUBRESOURCES {
            note_state((b[2] as u64 | (b[3] as u64) << 32) as usize, b[6]);
        }
    }
}

// ---------------------------------------------------------------------------------------
// the list methods
// ---------------------------------------------------------------------------------------

/// planes of a DXGI format on the PC (depth + stencil, video 4:2:0)
fn planes(format: i32) -> u32 {
    match format {
        19..=22 | 44..=47 | 103..=105 => 2,
        _ => 1,
    }
}

/// What a texture barrier needs to know of `resource` (a PC resource the title holds).
unsafe fn resource_info(resource: usize) -> Option<TexInfo> {
    let raw = resource as *mut c_void;
    let r = ID3D12Resource::from_raw_borrowed(&raw)?;
    let d = r.GetDesc();
    let buffer = d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
    let arrays = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { d.DepthOrArraySize as u32 };
    Some(TexInfo { buffer, mips: d.MipLevels as u32, arrays, planes: planes(d.Format.0), placed: super::vram::is_placed_texture(resource), uav: d.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) })
}

type ResourceBarrierFn = unsafe extern "system" fn(*mut c_void, u32, *const Legacy);

static CALLS: AtomicU64 = AtomicU64::new(0);
static FAILED: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);

/// Record `Barrier(n, groups)` on `list` through its PC `ResourceBarrier`; a call that does not
/// read as barrier groups gets a UAV barrier for all resources (what #146 always did).
///
/// # Safety
/// `list` must be a command list and `resource_barrier` its PC ResourceBarrier.
pub unsafe fn record(list: *mut c_void, resource_barrier: ResourceBarrierFn, list_type: i32, n: u32, groups: usize) {
    let readable = |p: usize, len: usize| crate::gpu::shader::readable(p, len);
    let items = match parse(n, groups, &readable) {
        Ok(items) => items,
        Err(e) => {
            if FAILED.fetch_add(1, Ordering::Relaxed) < 8 {
                crate::xlog_warn!("D3D12.X: Barrier({n}, {groups:#x}): not barrier groups ({e}) – a UAV barrier for all resources");
            }
            let uav: Legacy = [LEGACY_UAV, 0, 0, 0, 0, 0, 0, 0];
            resource_barrier(list, 1, &uav);
            return;
        }
    };
    let t = translate(&items, &|r| resource_info(r), &tracked_state, allowed_states(list_type));
    let k = CALLS.fetch_add(1, Ordering::Relaxed);
    if k < 8 {
        let textures = items.iter().filter(|i| matches!(i, Item::Texture(_))).count();
        crate::xlog_info!(
            "D3D12.X: Barrier: {n} groups, {} barriers ({textures} texture) → {} legacy barriers{}",
            items.len(),
            t.barriers.len(),
            if k == 7 { " (no more logged)" } else { "" }
        );
    }
    if t.refused > 0 && REFUSED.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::xlog_warn!("D3D12.X: Barrier: {} transitions a list of type {list_type} cannot make – left out", t.refused);
    }
    if super::present::trace_frame() {
        for i in &items {
            if let Item::Texture(x) = i {
                if super::present::is_watched(x.resource) {
                    static W: AtomicU64 = AtomicU64::new(0);
                    if W.fetch_add(1, Ordering::Relaxed) < 24 {
                        crate::xlog_info!(
                            "frame trace: Barrier on a watched target {:#x}: layout {:#x} → {:#x}, subresources {:x?}, access {:#x} → {:#x}",
                            x.resource,
                            x.layout_before,
                            x.layout_after,
                            x.range,
                            x.access.access_before,
                            x.access.access_after
                        );
                    }
                }
            }
        }
    }
    if !t.barriers.is_empty() {
        if super::title_textures::any() {
            let tt: Vec<(usize, u32, D3D12_RESOURCE_STATES)> = t
                .barriers
                .iter()
                .filter(|b| b[0] == LEGACY_TRANSITION)
                .map(|b| ((b[2] as u64 | (b[3] as u64) << 32) as usize, b[4], D3D12_RESOURCE_STATES(b[6] as i32)))
                .collect();
            if !tt.is_empty() {
                super::title_textures::note_transitions(list as usize, &tt);
            }
        }
        resource_barrier(list, t.barriers.len() as u32, t.barriers.as_ptr());
    }
    for (r, s) in t.states {
        note_state(r, s);
    }
}

const PC_GET_TYPE: usize = 8;
const PC_RESOURCE_BARRIER: usize = 26;

unsafe fn record_on_adopted(this: *mut c_void, n: u32, groups: usize) {
    let barrier: ResourceBarrierFn = std::mem::transmute(super::real::orig_fn(this, PC_RESOURCE_BARRIER));
    let get_type: unsafe extern "system" fn(*mut c_void) -> i32 = std::mem::transmute(super::real::orig_fn(this, PC_GET_TYPE));
    record(this, barrier, get_type(this), n, groups);
}

/// command list #146 Barrier(NumBarrierGroups, pBarrierGroups)
pub(super) unsafe extern "system" fn barrier_x(this: *mut c_void, n: u32, groups: *const c_void) {
    record_on_adopted(this, n, groups as usize);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn structure_layouts_match_d3d12_h() {
        assert_eq!(size_of::<D3D12_BARRIER_GROUP>(), GROUP_SIZE);
        assert_eq!(size_of::<D3D12_GLOBAL_BARRIER>(), GLOBAL_SIZE);
        assert_eq!(size_of::<D3D12_TEXTURE_BARRIER>(), TEXTURE_SIZE);
        assert_eq!(size_of::<D3D12_BUFFER_BARRIER>(), BUFFER_SIZE);
        assert_eq!(offset_of!(D3D12_BARRIER_GROUP, NumBarriers), 4);
        assert_eq!(offset_of!(D3D12_BARRIER_GROUP, Anonymous), 8);
        assert_eq!(offset_of!(D3D12_TEXTURE_BARRIER, LayoutBefore), 0x10);
        assert_eq!(offset_of!(D3D12_TEXTURE_BARRIER, LayoutAfter), 0x14);
        assert_eq!(offset_of!(D3D12_TEXTURE_BARRIER, pResource), 0x18);
        assert_eq!(offset_of!(D3D12_TEXTURE_BARRIER, Subresources), 0x20);
        assert_eq!(offset_of!(D3D12_TEXTURE_BARRIER, Flags), 0x38);
        assert_eq!(offset_of!(D3D12_BUFFER_BARRIER, pResource), 0x10);
        assert_eq!(offset_of!(D3D12_BUFFER_BARRIER, Offset), 0x18);
        assert_eq!(size_of::<D3D12_RESOURCE_BARRIER>(), size_of::<Legacy>());
        assert_eq!(D3D12_BARRIER_LAYOUT_UNDEFINED.0 as u32, LAYOUT_UNDEFINED);
        assert_eq!(D3D12_BARRIER_LAYOUT_VIDEO_QUEUE_COMMON.0 as u32, 30);
        assert_eq!(LAYOUT_LAST, 0x1F);
        assert_eq!(D3D12_BARRIER_ACCESS_NO_ACCESS.0 as u32, ACCESS_NO_ACCESS);
        assert_eq!(D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 as u32, ACCESS_UNORDERED_ACCESS);
    }

    #[test]
    fn layouts_map_to_the_legacy_states() {
        let s = |l: D3D12_BARRIER_LAYOUT| layout_state(l.0 as u32, true);
        assert_eq!(s(D3D12_BARRIER_LAYOUT_COMMON), Some(D3D12_RESOURCE_STATE_COMMON.0 as u32));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_RENDER_TARGET), Some(D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS), Some(D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE), Some(D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ), Some(D3D12_RESOURCE_STATE_DEPTH_READ.0 as u32));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE), Some(D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE.0 as u32));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE), Some(0xC0));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE), Some(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 as u32));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_COPY_SOURCE), Some(D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_DEST), Some(D3D12_RESOURCE_STATE_COPY_DEST.0 as u32));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE), Some(D3D12_RESOURCE_STATE_RESOLVE_SOURCE.0 as u32));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_RESOLVE_DEST), Some(D3D12_RESOURCE_STATE_RESOLVE_DEST.0 as u32));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE), Some(D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE.0 as u32));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON), Some(0));
        assert_eq!(s(D3D12_BARRIER_LAYOUT_UNDEFINED), None);
        // 0x1F = D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ_COMPUTE_QUEUE_ACCESSIBLE
        assert_eq!(layout_state(0x1F, true), Some(0x8C0));
        assert_eq!(layout_state(0x1F, false), Some(D3D12_RESOURCE_STATE_GENERIC_READ.0 as u32));
        assert_eq!(layout_state(32, true), None);
        // GENERIC_READ: the buffer-only states for buffers only
        assert_eq!(layout_state(1, false), Some(D3D12_RESOURCE_STATE_GENERIC_READ.0 as u32));
        assert_eq!(layout_state(1, true), Some(D3D12_RESOURCE_STATE_GENERIC_READ.0 as u32 & !STATE_BUFFER_ONLY));
        // the creation table (device #127 / #133) is the same
        for l in (0..=LAYOUT_LAST).chain([LAYOUT_UNDEFINED]) {
            assert_eq!(unsafe { super::super::device_x::layout_state_for_desc(l, 0) }, layout_state(l, false).unwrap_or(0), "layout {l}");
        }
    }

    // --- parsing ---

    #[repr(C, align(8))]
    struct Group {
        ty: u32,
        n: u32,
        p: usize,
    }

    #[repr(C, align(8))]
    #[derive(Clone, Copy)]
    struct Tex {
        a: [u32; 4],
        before: u32,
        after: u32,
        res: usize,
        range: [u32; 6],
        flags: u32,
        pad: u32,
    }

    #[repr(C, align(8))]
    struct Buf {
        a: [u32; 4],
        res: usize,
        offset: u64,
        size: u64,
    }

    fn tex(res: usize, before: u32, after: u32) -> Tex {
        Tex { a: [0x40, 0x10, 0x8, 0x80], before, after, res, range: [u32::MAX, 0, 0, 0, 0, 0], flags: 0, pad: 0 }
    }

    fn all(_: usize, _: usize) -> bool {
        true
    }

    #[test]
    fn groups_parse_from_console_memory() {
        let res = Box::new(0u64);
        let r = &*res as *const u64 as usize;
        let t = [tex(r, 6, 2), Tex { range: [1, 2, 0, 1, 0, 1], flags: 1, ..tex(r, LAYOUT_UNDEFINED, 3) }];
        let gl = [[0x80u32, 0x80, 0x10, 0x10]];
        let b = [Buf { a: [0x200, 0x2, 0x400, 0x1], res: 0x1234, offset: 0, size: u64::MAX }];
        let groups = [
            Group { ty: TYPE_TEXTURE, n: 2, p: t.as_ptr() as usize },
            Group { ty: TYPE_GLOBAL, n: 1, p: gl.as_ptr() as usize },
            Group { ty: TYPE_BUFFER, n: 1, p: b.as_ptr() as usize },
            Group { ty: TYPE_BUFFER, n: 0, p: 0 },
        ];
        let readable = |p: usize, n: usize| crate::gpu::shader::readable(p, n);
        let items = unsafe { parse(4, groups.as_ptr() as usize, &readable) }.unwrap();
        assert_eq!(items.len(), 4);
        let access = Access { sync_before: 0x40, sync_after: 0x10, access_before: 0x8, access_after: 0x80 };
        assert_eq!(items[0], Item::Texture(Texture { access, layout_before: 6, layout_after: 2, resource: r, range: [u32::MAX, 0, 0, 0, 0, 0], flags: 0 }));
        assert_eq!(items[1], Item::Texture(Texture { access, layout_before: LAYOUT_UNDEFINED, layout_after: 3, resource: r, range: [1, 2, 0, 1, 0, 1], flags: 1 }));
        assert_eq!(items[2], Item::Global(Access { sync_before: 0x80, sync_after: 0x80, access_before: 0x10, access_after: 0x10 }));
        assert_eq!(items[3], Item::Buffer { access: Access { sync_before: 0x200, sync_after: 0x2, access_before: 0x400, access_after: 0x1 }, resource: 0x1234 });
        assert_eq!(unsafe { parse(0, 0, &readable) }.unwrap(), Vec::new(), "no groups: nothing to do");
    }

    #[test]
    fn what_is_not_barrier_groups_is_refused() {
        let res = Box::new(0u64);
        let r = &*res as *const u64 as usize;
        let readable = |p: usize, n: usize| crate::gpu::shader::readable(p, n);
        unsafe {
            assert!(parse(1, 0, &readable).is_err(), "null groups");
            assert!(parse(1, 0x100, &readable).is_err(), "an alignment, not a pointer");
            assert!(parse(MAX_GROUPS + 1, 0x1000, &all).is_err(), "too many groups");
            let bad_type = [Group { ty: 3, n: 1, p: r }];
            assert!(parse(1, bad_type.as_ptr() as usize, &readable).is_err());
            let null_barriers = [Group { ty: TYPE_GLOBAL, n: 1, p: 0 }];
            assert!(parse(1, null_barriers.as_ptr() as usize, &readable).is_err());
            let t = [tex(r, 6, 0x1_0000)];
            let bad_layout = [Group { ty: TYPE_TEXTURE, n: 1, p: t.as_ptr() as usize }];
            assert!(parse(1, bad_layout.as_ptr() as usize, &readable).is_err());
            let t = [tex(0, 6, 2)];
            let null_texture = [Group { ty: TYPE_TEXTURE, n: 1, p: t.as_ptr() as usize }];
            assert!(parse(1, null_texture.as_ptr() as usize, &readable).is_err());
        }
    }

    // --- translation ---

    const RT: usize = 0x1000;
    const UAV_TEX: usize = 0x2000;
    const ARRAY: usize = 0x3000;
    const PLACED: usize = 0x4000;
    const BUFFER: usize = 0x5000;
    const DEPTH: usize = 0x6000;

    fn info(r: usize) -> Option<TexInfo> {
        let t = TexInfo { buffer: false, mips: 1, arrays: 1, planes: 1, placed: false, uav: true };
        match r {
            RT | UAV_TEX => Some(t),
            ARRAY => Some(TexInfo { mips: 3, arrays: 2, ..t }),
            PLACED => Some(TexInfo { placed: true, ..t }),
            BUFFER => Some(TexInfo { buffer: true, ..t }),
            DEPTH => Some(TexInfo { uav: false, ..t }),
            _ => None,
        }
    }

    fn texture(resource: usize, layout_before: u32, layout_after: u32) -> Item {
        Item::Texture(Texture { access: Access { sync_before: 0x40, sync_after: 0x10, access_before: 0x8, access_after: 0x80 }, layout_before, layout_after, resource, range: [u32::MAX, 0, 0, 0, 0, 0], flags: 0 })
    }

    fn none(_: usize) -> Option<u32> {
        None
    }

    const UAV_ALL: Legacy = [LEGACY_UAV, 0, 0, 0, 0, 0, 0, 0];

    #[test]
    fn a_render_target_goes_from_shader_resource_to_render_target_and_back() {
        let t = translate(&[texture(RT, 6, 2)], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![transition(RT, ALL_SUBRESOURCES, 0xC0, 0x4)]);
        assert_eq!(t.states, vec![(RT, 0x4)]);
        let t = translate(&[texture(RT, 2, 6)], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![transition(RT, ALL_SUBRESOURCES, 0x4, 0xC0)]);
        // the present layout (COMMON)
        let t = translate(&[texture(RT, 2, 0)], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![transition(RT, ALL_SUBRESOURCES, 0x4, 0)]);
    }

    #[test]
    fn an_undefined_layout_before_is_the_state_last_given() {
        let tracked = |r: usize| (r == RT).then_some(0xC0);
        let t = translate(&[texture(RT, LAYOUT_UNDEFINED, 2)], &info, &tracked, u32::MAX);
        assert_eq!(t.barriers, vec![transition(RT, ALL_SUBRESOURCES, 0xC0, 0x4)]);
        // not known: COMMON
        let t = translate(&[texture(UAV_TEX, LAYOUT_UNDEFINED, 2)], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![transition(UAV_TEX, ALL_SUBRESOURCES, 0, 0x4)]);
        // already in it: nothing
        let t = translate(&[texture(RT, LAYOUT_UNDEFINED, 6)], &info, &tracked, u32::MAX);
        assert!(t.barriers.is_empty());
        // moved twice in one call: the second move starts where the first one ended
        let t = translate(&[texture(RT, 6, 2), texture(RT, LAYOUT_UNDEFINED, 7)], &info, &tracked, u32::MAX);
        assert_eq!(t.barriers, vec![transition(RT, ALL_SUBRESOURCES, 0xC0, 0x4), transition(RT, ALL_SUBRESOURCES, 0x4, 0x800)]);
        assert_eq!(t.states, vec![(RT, 0x800)]);
        // a buffer-only state noted for a texture is not named
        let generic = |_: usize| Some(0xAC3);
        let t = translate(&[texture(RT, LAYOUT_UNDEFINED, 2)], &info, &generic, u32::MAX);
        assert_eq!(t.barriers, vec![transition(RT, ALL_SUBRESOURCES, 0x8C0, 0x4)]);
    }

    #[test]
    fn a_placed_texture_with_undefined_contents_gets_an_aliasing_barrier_first() {
        let t = translate(&[texture(PLACED, LAYOUT_UNDEFINED, 2)], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![[LEGACY_ALIASING, 0, 0, 0, 0, 0, 0, 0], transition(PLACED, ALL_SUBRESOURCES, 0, 0x4)]);
        let t = translate(&[texture(PLACED, 6, 2)], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![transition(PLACED, ALL_SUBRESOURCES, 0xC0, 0x4)]);
    }

    #[test]
    fn unordered_access_gets_one_uav_barrier_for_all_resources() {
        // staying in UNORDERED_ACCESS (UAV → UAV sync)
        let t = translate(&[texture(UAV_TEX, 3, 3)], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![UAV_ALL]);
        // moving out of it, plus a global barrier: one UAV barrier
        let g = Item::Global(Access { sync_before: 0x80, sync_after: 0x80, access_before: 0x10, access_after: 0x10 });
        let t = translate(&[texture(UAV_TEX, 3, 6), g, g], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![transition(UAV_TEX, ALL_SUBRESOURCES, 0x8, 0xC0), UAV_ALL]);
        // a buffer barrier
        let b = Item::Buffer { access: Access { sync_before: 0x80, sync_after: 0x2, access_before: 0x10, access_after: 0x1 }, resource: BUFFER };
        assert_eq!(translate(&[b], &info, &none, u32::MAX).barriers, vec![UAV_ALL]);
        // orders nothing before it: nothing
        let n = Item::Global(Access { sync_before: SYNC_NONE, sync_after: 0x80, access_before: ACCESS_NO_ACCESS, access_after: 0x10 });
        assert!(translate(&[n], &info, &none, u32::MAX).barriers.is_empty());
        // shader resource → shader resource: nothing
        assert!(translate(&[texture(RT, 6, 21)], &info, &none, u32::MAX).barriers.is_empty());
    }

    #[test]
    fn subresource_ranges_name_their_subresources() {
        let a = info(ARRAY).unwrap(); // 3 mips, 2 slices
        assert_eq!(subresources([u32::MAX, 0, 0, 0, 0, 0], &a), vec![ALL_SUBRESOURCES]);
        assert_eq!(subresources([4, 0, 0, 0, 0, 0], &a), vec![4], "NumMipLevels 0: a subresource index");
        assert_eq!(subresources([1, 1, 0, 1, 0, 1], &a), vec![1]);
        assert_eq!(subresources([1, 2, 1, 1, 0, 1], &a), vec![4, 5]);
        assert_eq!(subresources([0, 1, 0, 2, 0, 1], &a), vec![0, 3]);
        assert_eq!(subresources([0, 3, 0, 2, 0, 1], &a), vec![ALL_SUBRESOURCES], "the whole resource");
        assert_eq!(subresources([0, 9, 0, 9, 0, 9], &a), vec![ALL_SUBRESOURCES], "clamped to it");
        assert_eq!(subresources([3, 1, 0, 1, 0, 1], &a), Vec::<u32>::new(), "past the last mip");
        // planes: depth + stencil
        let ds = TexInfo { buffer: false, mips: 2, arrays: 1, planes: 2, placed: false, uav: false };
        assert_eq!(subresources([0, 2, 0, 1, 1, 1], &ds), vec![2, 3], "the stencil plane");
        // through the translation: per subresource, the resource's own state not noted
        let mut x = texture(ARRAY, 6, 2);
        if let Item::Texture(t) = &mut x {
            t.range = [1, 2, 1, 1, 0, 1];
        }
        let t = translate(&[x], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![transition(ARRAY, 4, 0xC0, 0x4), transition(ARRAY, 5, 0xC0, 0x4)]);
        assert!(t.states.is_empty());
    }

    #[test]
    fn what_a_list_cannot_name_is_left_out() {
        // a compute list: no render target, no pixel shader resource
        let t = translate(&[texture(RT, 6, 2)], &info, &none, allowed_states(2));
        assert!(t.barriers.is_empty());
        assert_eq!(t.refused, 1);
        let t = translate(&[texture(UAV_TEX, 27, 26)], &info, &none, allowed_states(2));
        assert_eq!(t.barriers, vec![transition(UAV_TEX, ALL_SUBRESOURCES, 0x40, 0x8), UAV_ALL]);
        // unknown resources, buffers named by texture barriers, UNDEFINED after
        assert!(translate(&[texture(0x9999, 6, 2)], &info, &none, u32::MAX).barriers.is_empty());
        assert_eq!(translate(&[texture(BUFFER, 6, 2)], &info, &none, u32::MAX).barriers, vec![UAV_ALL], "a buffer: its sync only");
        assert!(translate(&[texture(RT, 6, LAYOUT_UNDEFINED)], &info, &none, u32::MAX).barriers.is_empty());
    }

    /// a depth texture the PC has without ALLOW_UNORDERED_ACCESS (UAV on the console) never goes
    /// to UNORDERED_ACCESS: the PC refuses that state for it
    #[test]
    fn a_texture_without_uav_on_the_pc_stays_out_of_unordered_access() {
        let t = translate(&[texture(DEPTH, 4, 3)], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![UAV_ALL], "stays in DEPTH_WRITE, the sync kept");
        assert!(t.states.is_empty());
        let written = |_: usize| Some(0x10);
        let t = translate(&[texture(DEPTH, 3, 5)], &info, &written, u32::MAX);
        assert_eq!(t.barriers, vec![transition(DEPTH, ALL_SUBRESOURCES, 0x10, 0x20)], "from where it really is");
        let noted_uav = |_: usize| Some(0x8);
        let t = translate(&[texture(DEPTH, LAYOUT_UNDEFINED, 4)], &info, &noted_uav, u32::MAX);
        assert_eq!(t.barriers, vec![transition(DEPTH, ALL_SUBRESOURCES, 0, 0x10)], "a UAV state noted at creation is COMMON");
    }

    /// layout 0x1F (DIRECT_QUEUE_GENERIC_READ_COMPUTE_QUEUE_ACCESSIBLE) maps deterministically to
    /// texture GENERIC_READ (`0x8C0`); any future unknown console-only layout (> 0x1F) falls back
    /// to the barrier's access bits
    #[test]
    fn a_console_layout_is_taken_by_its_access() {
        let with = |before: u32, after: u32, access_before: u32, access_after: u32| {
            Item::Texture(Texture { access: Access { sync_before: 0x40, sync_after: 0x10, access_before, access_after }, layout_before: before, layout_after: after, resource: RT, range: [u32::MAX, 0, 0, 0, 0, 0], flags: 0 })
        };
        // 0x1F is consistent in both directions regardless of access bits (no Before!=After mismatch)
        let t = translate(&[with(2, 0x1F, 0x8, 0x80)], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![transition(RT, ALL_SUBRESOURCES, 0x4, 0x8C0)]);
        let t = translate(&[with(0x1F, 2, 0x800, 0x8)], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![transition(RT, ALL_SUBRESOURCES, 0x8C0, 0x4)]);
        // unknown console layout > 0x1F: taken by access
        let t = translate(&[with(2, 0x20, 0x8, 0x80)], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![transition(RT, ALL_SUBRESOURCES, 0x4, 0xC0)]);
        let t = translate(&[with(0x20, 2, 0x80, 0x8)], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![transition(RT, ALL_SUBRESOURCES, 0xC0, 0x4)]);
        let t = translate(&[with(2, 0x20, 0x8, ACCESS_NO_ACCESS)], &info, &none, u32::MAX);
        assert_eq!(t.barriers, vec![transition(RT, ALL_SUBRESOURCES, 0x4, 0)], "no access: COMMON");
        assert_eq!(access_state(0x400 | 0x800), 0xC00);
        assert_eq!(access_state(0), 0);
    }

    #[test]
    fn states_are_noted_from_legacy_barriers_too() {
        let r = 0xDEAD_0000usize;
        note_legacy(&[transition(r, ALL_SUBRESOURCES, 0, 0x4), transition(r, 3, 0x4, 0x800)]);
        assert_eq!(tracked_state(r), Some(0x4), "a single subresource does not move the whole resource");
        note_state(r, 0xC0);
        assert_eq!(tracked_state(r), Some(0xC0));
        forget(r);
        assert_eq!(tracked_state(r), None);
        assert_eq!(tracked_state(r + 8), None);
    }

    // --- on a PC list ---

    /// A render target created in SHADER_RESOURCE is cleared after an enhanced barrier moves it
    /// to RENDER_TARGET and back: the list closes and the clear reaches the texture.
    #[test]
    fn enhanced_barriers_move_a_render_target_on_the_gpu() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                eprintln!("no D3D12 device – skipped");
                return;
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 64,
                Height: 64,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let mut rt: Option<ID3D12Resource> = None;
            let srv_state = D3D12_RESOURCE_STATES(layout_state(6, true).unwrap() as i32);
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, srv_state, None, &mut rt).unwrap();
            let rt = rt.unwrap();
            let heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).unwrap();
            let rtv = heap.GetCPUDescriptorHandleForHeapStart();
            dev.CreateRenderTargetView(&rt, None, rtv);
            let mut rb: Option<ID3D12Resource> = None;
            let bd = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 256 * 64, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
            let rb = rb.unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let raw = list.as_raw();
            let barrier: ResourceBarrierFn = std::mem::transmute(*(*(raw as *const *const usize)).add(PC_RESOURCE_BARRIER));
            let r = rt.as_raw() as usize;
            note_state(r, srv_state.0 as u32);
            let record_one = |before: u32, after: u32| {
                let t = [tex(r, before, after)];
                let g = [Group { ty: TYPE_TEXTURE, n: 1, p: t.as_ptr() as usize }];
                record(raw, barrier, D3D12_COMMAND_LIST_TYPE_DIRECT.0, 1, g.as_ptr() as usize);
            };
            record_one(LAYOUT_UNDEFINED, 2); // from the state noted at creation
            list.ClearRenderTargetView(rtv, &[0.0, 1.0, 0.0, 1.0], None);
            record_one(2, 0x1F); // DIRECT_QUEUE_GENERIC_READ_COMPUTE_QUEUE_ACCESSIBLE (includes COPY_SOURCE)
            let fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, Width: 64, Height: 64, Depth: 1, RowPitch: 256 } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rt), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            record_one(0x1F, 6); // back to shader resource
            assert_eq!(tracked_state(r), Some(0xC0));
            list.Close().expect("the list closes");
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            let t0 = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 {
                assert!(t0.elapsed().as_secs() < 10, "GPU timeout");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(dev.GetDeviceRemovedReason().is_ok(), "device removed");
            let mut p = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut p)).unwrap();
            let px = std::slice::from_raw_parts(p as *const u32, 64 * 64);
            assert!(px.iter().all(|&v| v == 0xFF00FF00), "cleared green: {:#x}", px[0]);
            rb.Unmap(0, None);
            STATES.lock().unwrap().as_mut().unwrap().remove(&r);
        }
    }
}
