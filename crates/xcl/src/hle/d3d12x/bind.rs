//! Root arguments of console code, bound when it dispatches with a translated pipeline.
//!
//! A translated console shader reads its root arguments the console's way (user SGPRs s4…,
//! the rest from a table at s[0:1]: `console_list::user_data`) and memory through XCL's
//! shader ABI (u0 io, u1 res, u2… unified memory). So on a command list of console code the
//! compute root signature and root arguments are kept here – packed by the root signature's
//! layout – instead of going to the PC list, and each Dispatch with a translated pipeline
//! binds the ABI root signature, writes the user SGPRs (`io`) and the argument table to
//! unified memory and dispatches. That memory stays until the list's allocator is reset.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::console_list::{self, Layout, Param};

#[derive(Default)]
struct State {
    layout: Option<Arc<Layout>>,
    args: Vec<u8>,
    /// the graphics root signature's layout and arguments
    layout_gfx: Option<Arc<Layout>>,
    args_gfx: Vec<u8>,
    /// the title's graphics root signature (a PC object), bound before a draw with an
    /// untranslated pipeline – created with it – since SetGraphicsRootSignature binds nothing
    rs_gfx: usize,
    pso: usize,
    allocator: usize,
    /// unified memory of this recording (io, argument tables)
    uma: Vec<usize>,
    /// where the next draw's io and argument table go (blocks in `uma`)
    arena: crate::gpu::uma::Arena,
    /// Last PC graphics / compute root signatures and PC pipeline bound on this list (`0` = unknown).
    /// Binding XCL's ABI root signature also binds `u2..u5` for the process lifetime; cleared on
    /// `reset`, `clear_state`, `KickoffX`, PC `Reset`, and untranslated root-sig switches.
    bound_rs_gfx: usize,
    bound_rs_compute: usize,
    bound_pso: usize,
}

static STATES: Mutex<Option<HashMap<usize, State>>> = Mutex::new(None);

/// draws / dispatches of console code: with a translated pipeline, skipped, with a PC one
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// draws and dispatches of console code so far (`XCL_ACTIVITY`)
pub fn activity() -> (u64, u64) {
    (COUNTS[DRAWS].load(Ordering::Relaxed), COUNTS[DISPATCHES].load(Ordering::Relaxed))
}
const DRAWS: usize = 0;
const DRAWS_SKIPPED: usize = 1;
const DRAWS_PC: usize = 2;
const DISPATCHES: usize = 3;
const DISPATCHES_SKIPPED: usize = 4;

fn count(k: usize) {
    let n = COUNTS[k].fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() && n >= 64 {
        let c: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let (allocators, blocks) = console_list::pending_blocks();
        let (coherent_used, coherent_total, sparse_used) = crate::gpu::uma::get()
            .map_or((0, 0, 0), |u| (u.used() >> 20, u.size() >> 20, u.sparse_used() >> 20));
        crate::xlog_info!(
            "D3D12.X: console code so far: {} draws with translated pipelines ({} skipped, {} with PC pipelines), {} dispatches ({} skipped); coherent {coherent_used}/{coherent_total} MB, sparse {sparse_used} MB, {blocks} blocks held for {allocators} allocators",
            c[DRAWS], c[DRAWS_SKIPPED], c[DRAWS_PC], c[DISPATCHES], c[DISPATCHES_SKIPPED],
        );
    }
}

fn with<R>(list: *mut c_void, f: impl FnOnce(&mut State) -> R) -> R {
    f(STATES.lock().unwrap().get_or_insert_with(HashMap::new).entry(list as usize).or_default())
}

/// The list was created or reset onto `allocator` with `initial_pso`: the previous
/// recording's memory is the previous allocator's until that one is reset, and pipeline
/// and root signature state reset per D3D12 `CreateCommandList` / `Reset` semantics.
pub(super) fn reset(list: *mut c_void, allocator: usize, initial_pso: usize) {
    let (old, blocks) = with(list, |s| {
        let old = std::mem::replace(&mut s.allocator, allocator);
        s.arena = Default::default();
        s.pso = initial_pso;
        s.layout = None;
        s.args.clear();
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
        s.bound_rs_gfx = 0;
        s.bound_rs_compute = 0;
        s.bound_pso = 0;
        (old, std::mem::take(&mut s.uma))
    });
    console_list::hold_until_reset(old, blocks);
}

/// `ClearState(pso)`: resets pipeline and root-argument state back to creation defaults
/// with `pso` bound, keeping the current recording's allocator and UMA arena intact.
pub(super) fn clear_state(list: *mut c_void, pso: usize) {
    with(list, |s| {
        s.pso = pso;
        s.layout = None;
        s.args.clear();
        s.layout_gfx = None;
        s.args_gfx.clear();
        s.rs_gfx = 0;
        s.bound_rs_gfx = 0;
        s.bound_rs_compute = 0;
        s.bound_pso = 0;
    });
}

/// Carry bound pipeline and root-signature / root-argument state from `from` to `to` across a
/// deferred mid-list split (`to` records into `allocator` with its own fresh UMA arena while
/// `from` retains its previously allocated blocks until `allocator` resets).
pub(super) fn inherit_for_split(from: *mut c_void, to: *mut c_void, allocator: usize) {
    let (layout, args, layout_gfx, args_gfx, rs_gfx, pso) = with(from, |s| {
        (s.layout.clone(), s.args.clone(), s.layout_gfx.clone(), s.args_gfx.clone(), s.rs_gfx, s.pso)
    });
    let (old, blocks) = with(to, |s| {
        let old = std::mem::replace(&mut s.allocator, allocator);
        s.arena = Default::default();
        s.layout = layout;
        s.args = args;
        s.layout_gfx = layout_gfx;
        s.args_gfx = args_gfx;
        s.rs_gfx = rs_gfx;
        s.pso = pso;
        s.bound_rs_gfx = 0;
        s.bound_rs_compute = 0;
        s.bound_pso = 0;
        (old, std::mem::take(&mut s.uma))
    });
    console_list::hold_until_reset(old, blocks);
}

/// Invalidate cached PC root-signature and pipeline bindings after a PC-list reset or bundle execution.
pub(super) fn invalidate_pc_bindings(list: *mut c_void) {
    if let Ok(mut g) = STATES.lock() {
        if let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(list as usize))) {
            s.bound_rs_gfx = 0;
            s.bound_rs_compute = 0;
            s.bound_pso = 0;
        }
    }
}

/// Record a direct PC `SetPipelineState` call (e.g. from depth-bias fallback or `cl_set_pipeline_state`).
pub(super) fn note_pc_pso(list: *mut c_void, pso: usize) {
    if let Ok(mut g) = STATES.lock() {
        if let Some(s) = g.as_mut().and_then(|m| m.get_mut(&(list as usize))) {
            s.bound_pso = pso;
        }
    }
}

/// The list is gone.
pub(super) fn forget(list: *mut c_void) {
    if let Some(s) = STATES.lock().unwrap().as_mut().and_then(|m| m.remove(&(list as usize))) {
        console_list::hold_until_reset(s.allocator, s.uma);
    }
}

#[cfg(test)]
pub(crate) fn seed_reset_state_for_test(list: *mut c_void, allocator: usize, pso: usize, uma: Vec<usize>) {
    with(list, |s| {
        s.allocator = allocator;
        s.pso = pso;
        s.layout = Some(Arc::new(Layout::default()));
        s.args = vec![1, 2, 3, 4];
        s.layout_gfx = Some(Arc::new(Layout::default()));
        s.args_gfx = vec![5, 6, 7, 8];
        s.uma = uma;
    });
}

#[cfg(test)]
pub(crate) fn reset_state_for_test(list: *mut c_void) -> (usize, usize, Vec<u8>, Vec<u8>, Vec<usize>) {
    with(list, |s| (s.allocator, s.pso, s.args.clone(), s.args_gfx.clone(), s.uma.clone()))
}

#[cfg(test)]
pub(crate) fn remove_state_for_test(list: *mut c_void) {
    STATES.lock().unwrap().as_mut().and_then(|m| m.remove(&(list as usize)));
}

/// bytes of parameter `i` at constant `offset` (of the compute or the graphics arguments)
fn put_in(list: *mut c_void, gfx: bool, i: u32, offset: u32, bytes: &[u8]) {
    with(list, |s| {
        let (layout, args) = if gfx { (&s.layout_gfx, &mut s.args_gfx) } else { (&s.layout, &mut s.args) };
        let Some(at) = layout.as_ref().and_then(|l| l.params.get(i as usize)).map(|p| match *p {
            Param::Constants { at, .. } => at as usize + 4 * offset as usize,
            Param::Descriptor { at } | Param::Table { at } => at as usize,
        }) else {
            return;
        };
        if let Some(t) = args.get_mut(at..at + bytes.len()) {
            t.copy_from_slice(bytes);
        }
    })
}

fn put(list: *mut c_void, i: u32, offset: u32, bytes: &[u8]) {
    put_in(list, false, i, offset, bytes)
}

/// #34 SetGraphicsRootSignature (PC #30)
pub(super) unsafe extern "system" fn set_graphics_root_signature(this: *mut c_void, rs: *mut c_void) {
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args_gfx = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout_gfx = layout;
        s.rs_gfx = rs as usize;
    });
}

/// #36 SetGraphicsRootDescriptorTable (PC #32)
pub(super) unsafe extern "system" fn set_graphics_root_table(this: *mut c_void, i: u32, handle: u64) {
    put_in(this, true, i, 0, &(handle as u32).to_le_bytes());
}

/// #38 SetGraphicsRoot32BitConstant (PC #34)
pub(super) unsafe extern "system" fn set_graphics_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put_in(this, true, i, offset, &value.to_le_bytes());
}

/// #40 SetGraphicsRoot32BitConstants (PC #36)
pub(super) unsafe extern "system" fn set_graphics_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put_in(this, true, i, offset, &bytes);
    }
}

/// #42 / #44 / #46 SetGraphicsRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #38 /
/// #40 / #42)
pub(super) unsafe extern "system" fn set_graphics_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    put_in(this, true, i, 0, &va.to_le_bytes());
}

/// `XCL_TRACE_ADDRESSES=1`: root descriptor addresses of console code – in unified memory
/// (translated shaders see them) or not – counted and logged
fn note_address(va: u64) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static IN: AtomicU64 = AtomicU64::new(0);
    static OUT: AtomicU64 = AtomicU64::new(0);
    if !*ON.get_or_init(|| std::env::var_os("XCL_TRACE_ADDRESSES").is_some()) {
        return;
    }
    let inside = crate::gpu::uma::get().is_some_and(|u| u.contains(va as usize));
    let (i, o) = if inside { (IN.fetch_add(1, Ordering::Relaxed) + 1, OUT.load(Ordering::Relaxed)) } else { (IN.load(Ordering::Relaxed), OUT.fetch_add(1, Ordering::Relaxed) + 1) };
    if (i + o).is_power_of_two() {
        crate::xlog_info!("D3D12.X: root descriptor addresses: {i} in unified memory, {o} outside (latest {va:#x})");
    }
}

/// command signatures that change root arguments (they need the title's root signature)
static ROOT_SIGNATURES: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

pub(super) fn note_root_changing_signature(sig: usize) {
    ROOT_SIGNATURES.lock().unwrap().get_or_insert_with(Default::default).insert(sig);
}

/// command signatures with a Dispatch argument (noted at every creation, so a signature made
/// at a freed one's address does not keep its kind)
static DISPATCH_SIGNATURES: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

pub(super) fn note_signature_dispatches(sig: usize, dispatches: bool) {
    let mut s = DISPATCH_SIGNATURES.lock().unwrap();
    let s = s.get_or_insert_with(Default::default);
    if dispatches {
        s.insert(sig);
    } else {
        s.remove(&sig);
    }
}

fn signature_dispatches(sig: usize) -> bool {
    DISPATCH_SIGNATURES.lock().unwrap().as_ref().is_some_and(|s| s.contains(&sig))
}

/// Call the original PC setter. bind_abi manages its own cache when it also writes u2..u5.
unsafe fn set_pc_signature(this: *mut c_void, rs: *mut c_void, gfx: bool) {
    let slot = if gfx { 30 } else { 29 };
    let set: unsafe extern "system" fn(*mut c_void, *mut c_void) = std::mem::transmute(super::real::orig_fn(this, slot));
    set(this, rs);
}

pub(super) unsafe extern "system" fn external_root_signature<const GFX: bool>(this: *mut c_void, rs: *mut c_void) {
    set_pc_signature(this, rs, GFX);
    if let Some(state) = STATES.lock().unwrap().as_mut().and_then(|m| m.get_mut(&(this as usize))) {
        if GFX { state.bound_rs_gfx = 0; } else { state.bound_rs_compute = 0; }
    }
}

unsafe fn bind_title_graphics_signature(list: &ID3D12GraphicsCommandList) {
    let rs = with(list.as_raw(), |s| {
        if s.rs_gfx != 0 {
            s.bound_rs_gfx = 0;
        }
        s.rs_gfx
    });
    if rs != 0 {
        set_pc_signature(list.as_raw(), rs as *mut c_void, true);
    }
}

/// #59 ExecuteIndirect(signature, max count, arguments, offset, count buffer, offset): with a
/// translated pipeline, on XCL's ABI with the kept root arguments (signatures that change root
/// arguments themselves cannot be – skipped, logged)
pub(super) unsafe extern "system" fn execute_indirect(this: *mut c_void, sig: usize, max: u32, args: usize, args_off: u64, count_buf: usize, count_off: u64) {
    type F = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
    let pc: F = std::mem::transmute(super::real::orig_fn(this, 59));
    let pso = super::pso_async::resolve_wait(with(this, |s| s.pso));
    let (gfx, compute) = (super::real::is_translated_graphics(pso), super::real::is_translated_compute(pso));
    let (args_off, count_off) = (args_off + super::real::buffer_delta(args), count_off + super::real::buffer_delta(count_buf));
    if !gfx && !compute {
        // A pipeline still being built (or whose build failed): its stand-in draws nothing, and
        // the list may hold XCL's ABI root signature from a translated draw before it – the PC
        // call with that pair is invalid (debug layer 953, "the currently set Root Signature
        // doesn't match the currently set Pipeline State Object"; a retail title's world then
        // hung the GPU, 0x887a0006). Skipped, like a draw or dispatch of such a pipeline.
        if super::pso_async::is_pending(with(this, |s| s.pso)) {
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a pipeline that is still being built skipped");
            }
            return;
        }
        // a console compute program without a translation (its build failed): nothing to run,
        // as for Dispatch. The PC call would run it with the list's compute root signature, which
        // can still be XCL's ABI one from a translated dispatch before it (debug layer 953).
        // A native PC compute pipeline is not one of these and still goes to the PC call.
        if signature_dispatches(sig) && super::real::is_console_compute(pso) {
            count(DISPATCHES_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect dispatch with a pipeline that has no translated console shader ({pso:#x}) skipped");
            }
            return;
        }
        // a PC pipeline: the title's root signature, as for a draw (`graphics_bound`)
        if let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) {
            bind_title_graphics_signature(list);
        }
        return fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off));
    }
    if ROOT_SIGNATURES.lock().unwrap().as_ref().is_some_and(|s| s.contains(&sig)) {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect that changes root arguments with a translated pipeline skipped");
        }
        return;
    }
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    match bind_abi(this, list, pso, gfx) {
        Ok(()) => {
            count(if gfx { DRAWS } else { DISPATCHES });
            fresh_indirect_arguments(this, || pc(this, sig, max, args, args_off, count_buf, count_off))
        }
        Err(e) => {
            count(if gfx { DRAWS_SKIPPED } else { DISPATCHES_SKIPPED });
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: ExecuteIndirect with a translated pipeline skipped: {e}");
            }
        }
    }
}

/// XCL's own buffer for `fresh_indirect_arguments` (device, buffer): COMMON between its uses
static ARGUMENT_ORDER: Mutex<Option<(usize, ID3D12Resource)>> = Mutex::new(None);

/// `op` (an ExecuteIndirect on `list`) reading the arguments and count the list wrote before it.
///
/// The console orders a dispatch that writes indirect arguments before the ExecuteIndirect that
/// reads them with a barrier that reaches the PC as a UAV barrier (`list_x::resource_barrier_x`).
/// NVIDIA's driver (RTX 5090) fetches an ExecuteIndirect's arguments without waiting for that
/// barrier: it ran the count that was there before the list in all but a few of 100 tries, also
/// when the dispatch wrote them through another resource over the same memory, as translated
/// shaders do through the memory window buffers (`execute_indirect_reads_the_arguments_written_before_it`).
/// A retail title's ExecuteIndirect after the dispatch that writes its group count then runs
/// whatever count the memory held before. A transition to INDIRECT_ARGUMENT makes the driver
/// wait for the writes – of any buffer, as measured there. The title's buffer is in the title's
/// state (its transitions reach the PC list), so a buffer of XCL's own takes it, COMMON ->
/// INDIRECT_ARGUMENT before and back after. Only on a list XCL made (`list_x::on_create`) and
/// not a bundle, which cannot hold barriers; its type comes from XCL's state, not a call to it.
unsafe fn fresh_indirect_arguments(list: *mut c_void, op: impl FnOnce()) {
    let tracked = super::list_x::tracked_list_type(list).is_some_and(|t| t != D3D12_COMMAND_LIST_TYPE_BUNDLE);
    let buffer = if tracked { ID3D12GraphicsCommandList::from_raw_borrowed(&list).and_then(|l| argument_order_buffer(l)) } else { None };
    let Some(buffer) = buffer else { return op() };
    let b = buffer.as_raw() as usize;
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    op();
    super::list_x::transition(list, b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
}

/// The device's buffer for `fresh_indirect_arguments` (made on first use)
unsafe fn argument_order_buffer(list: &ID3D12GraphicsCommandList) -> Option<ID3D12Resource> {
    let mut dev: Option<ID3D12Device> = None;
    list.GetDevice(&mut dev).ok()?;
    let dev = dev?;
    let mut g = ARGUMENT_ORDER.lock().unwrap();
    if let Some((_, b)) = g.as_ref().filter(|(d, _)| *d == dev.as_raw() as usize) {
        return Some(b.clone());
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut b: Option<ID3D12Resource> = None;
    if let Err(e) = dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut b) {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: ExecuteIndirect: no buffer to order its arguments after the writes before it ({e}) – it may read old ones");
        }
        return None;
    }
    let b = b?;
    *g = Some((dev.as_raw() as usize, b.clone()));
    Some(b)
}

/// A DrawInstanced of the console's rect list (three corners of each rectangle, the fourth
/// implied) as two triangles a rectangle; false: not a rect list, or one this cannot expand
/// (the caller draws the three vertices as a triangle – half of each rectangle).
///
/// The PC has no rect lists. The fourth corner is made here, on the CPU, from the title's
/// vertex data: v3 = v0 + v2 − v1 (seen in a retail title's full-screen passes: (−1,1) (−1,−1)
/// (1,−1) with texture coordinates (0,0) (0,1) (1,1) – the missing corner is (1,1), (1,0)).
/// The vertex layout is not looked at: each 4-byte word that is the same in all three vertices
/// is kept (a packed colour), any other is taken as a float. Vertices a shader makes from the
/// vertex index alone (no vertex buffer) are not expanded.
unsafe fn draw_rects(this: *mut c_void, list: &ID3D12GraphicsCommandList, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) -> bool {
    if !cpu_rect_expansion(with(this, |s| s.pso)) {
        return false;
    }
    let Some(vbs) = super::list_x::rect_list_state(this) else { return false };
    let rects = (vertices / 3) as usize;
    if vbs.is_empty() || rects == 0 || rects > 4096 {
        return false;
    }
    let mut views = Vec::new();
    for &(slot, (va, bytes, stride), _) in &vbs {
        let (s, at) = (stride as usize, va as usize + first_vertex as usize * stride as usize);
        let len = rects * 3 * s;
        if va == 0 || s == 0 || s % 4 != 0 || first_vertex as usize * s + len > bytes as usize || !crate::gpu::shader::readable(at, len) {
            return false;
        }
        let p = allocate_upload_memory(this, rects * 6 * s, 16);
        if p.is_null() {
            return false;
        }
        let word = |v: usize, w: usize| (at as *const u8).add(v * s + w * 4).cast::<u32>().read_unaligned();
        for r in 0..rects {
            let out = p.add(r * 6 * s);
            let src = (at as *const u8).add(r * 3 * s);
            // v0 v1 v2, then v0 v2 v3
            std::ptr::copy_nonoverlapping(src, out, 3 * s);
            std::ptr::copy_nonoverlapping(src, out.add(3 * s), s);
            std::ptr::copy_nonoverlapping(src.add(2 * s), out.add(4 * s), s);
            for w in 0..s / 4 {
                let (a, b, c) = (word(r * 3, w), word(r * 3 + 1, w), word(r * 3 + 2, w));
                let d = if a == b && b == c { a } else { (f32::from_bits(a) + f32::from_bits(c) - f32::from_bits(b)).to_bits() };
                out.add(5 * s + w * 4).cast::<u32>().write_unaligned(d);
            }
        }
        views.push((slot, D3D12_VERTEX_BUFFER_VIEW { BufferLocation: super::list_x::pc_va_of(p as u64), SizeInBytes: (rects * 6 * s) as u32, StrideInBytes: stride }));
    }
    for (slot, v) in &views {
        list.IASetVertexBuffers(*slot, Some(&[*v]));
    }
    list.DrawInstanced(rects as u32 * 6, instances, 0, first_instance);
    for &(slot, _, pc) in &vbs {
        if let Some(v) = pc {
            list.IASetVertexBuffers(slot, Some(&[v]));
        }
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: rect lists are drawn as two triangles a rectangle, the fourth corner made on the CPU");
    }
    true
}

/// Whether a rect list drawn with the bound pipeline `pso` is expanded on the CPU (`draw_rects`):
/// not with the rect list GS (`real::has_rect_list_gs`), which completes each rect itself – both
/// would draw every rect twice (each CPU half is three corners of the rect).
fn cpu_rect_expansion(pso: usize) -> bool {
    !super::real::has_rect_list_gs(super::pso_async::resolve(pso))
}

/// #12 DrawInstanced
pub(super) unsafe extern "system" fn draw_instanced(this: *mut c_void, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
    DRAW_ARGS.with(|d| d.set((false, vertices, instances, first_vertex, 0)));
    if let Some(list) = graphics_bound(this) {
        if draw_rects(this, &list, vertices, instances, first_vertex, first_instance) {
            return;
        }
        super::list_x::sync_title_buffers(this, None, first_vertex, vertices, if first_instance != 0 { instances.max(2) } else { instances });
        list.DrawInstanced(vertices, instances, first_vertex, first_instance);
    }
}

thread_local! {
    /// the draw being bound: (indexed, count, instances, first index / vertex, base vertex) – for the frame trace
    static DRAW_ARGS: std::cell::Cell<(bool, u32, u32, u32, i32)> = const { std::cell::Cell::new((false, 0, 0, 0, 0)) };
}

/// #13 DrawIndexedInstanced
pub(super) unsafe extern "system" fn draw_indexed_instanced(this: *mut c_void, indices: u32, instances: u32, first_index: u32, base_vertex: i32, first_instance: u32) {
    if let Some(list) = prepare_indexed_draw(this, indices, instances, first_index, base_vertex) {
        super::list_x::sync_title_buffers(this, Some((first_index, indices, base_vertex)), 0, 0, if first_instance != 0 { instances.max(2) } else { instances });
        emit_prepared_indexed_draw(&list, indices, instances, first_index, base_vertex, first_instance);
    }
}

/// Prepare an indexed draw once, before callers make temporary native state changes.
pub(super) unsafe fn prepare_indexed_draw(
    this: *mut c_void,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
) -> Option<ID3D12GraphicsCommandList> {
    DRAW_ARGS.with(|d| d.set((true, indices, instances, first_index, base_vertex)));
    graphics_bound(this)
}

/// Submit an indexed draw whose translated graphics state has already been prepared.
pub(super) unsafe fn emit_prepared_indexed_draw(
    list: &ID3D12GraphicsCommandList,
    indices: u32,
    instances: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    list.DrawIndexedInstanced(indices, instances, first_index, base_vertex, first_instance);
}

/// Before a draw: with a translated pipeline its ABI root signature and the kept graphics
/// arguments are bound (None: the draw is skipped – a console pipeline without a
/// translation, or no unified memory).
unsafe fn graphics_bound(this: *mut c_void) -> Option<ID3D12GraphicsCommandList> {
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this)?.clone();
    let pso = with(this, |s| s.pso);
    // a pipeline still being built in the background is waited for (its stand-in draws nothing)
    if !super::real::is_translated_graphics(super::pso_async::resolve_wait(pso)) {
        count(DRAWS_PC);
        if super::present::trace_frame() {
            // which pipelines draw without a translation (first 12 distinct)
            static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
            let mut g = SEEN.lock().unwrap();
            if !g.contains(&pso) && g.len() < 12 {
                g.push(pso);
                let real = super::pso_async::resolve_wait(pso);
                crate::xlog_info!(
                    "frame trace: draw with an untranslated pipeline {pso:#x} (resolves to {real:#x}; pending {}, two-pass {}, RT0 {:?})",
                    super::pso_async::is_pending(pso),
                    super::real::two_pass_of(real).is_some(),
                    super::present::rt0_resource(this as usize).map(|r| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0) }))
                );
            }
        }
        // a PC pipeline (also a stand-in or fallback: made with the title's root signature):
        // the title's root signature, which nothing else binds on the PC list – a draw without
        // one is invalid (the debug layer reports it on every GPU; WARP faults on it)
        bind_title_graphics_signature(&list);
        return Some(list);
    }
    if super::present::trace_frame() {
        trace_ui_draw(this, pso);
    }
    match bind_abi(this, &list, pso, true) {
        Ok(()) => {
            count(DRAWS);
            Some(list)
        }
        Err(e) => {
            count(DRAWS_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: draw with a translated pipeline skipped: {e}");
            }
            None
        }
    }
}

/// `XCL_TRACE_FRAME`: draws into a presented buffer (the UI) once the title has presented for a
/// while – their pipeline, root arguments and the resources their descriptor tables start with
/// (format, size, kept in step with title memory or not). The first 24.
/// `XCL_TRACE_FRAME`, after 300 presents: one line per distinct (VS, PS, render target) drawn –
/// the shaders' dump hashes, the target and the textures of the first descriptor table.
unsafe fn trace_draw_kinds(this: *mut c_void, pso: usize) {
    // keyed by the target's shape, not the object: a UI library's transient targets (Renoir's
    // layers) are new objects every frame and would fill the list at once
    static SEEN: Mutex<Vec<(u64, u64, Option<(u64, u32, i32, u32)>)>> = Mutex::new(Vec::new());
    if super::present::activity().0 < 300 {
        return;
    }
    let Some((vs, ps)) = super::real::pso_shaders(super::pso_async::resolve(pso)) else { return };
    let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
    let desc = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.SampleDesc.Count) });
    {
        let mut g = SEEN.lock().unwrap();
        let key = (vs, ps, desc(rt));
        if g.len() >= 160 || g.contains(&key) {
            return;
        }
        g.push(key);
    }
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut tex = Vec::new();
    if let (Some(layout), Some(u)) = (layout, crate::gpu::uma::get()) {
        for p in layout.params.iter() {
            if let Param::Table { at } = *p {
                let t = word(at as usize) as usize;
                for k in 0..4usize {
                    if let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) {
                        tex.push(format!("{:?}{}", desc(r), super::title_textures::address(r).map_or(String::new(), |a| format!(" at {a:#x}"))));
                    }
                }
            }
            if tex.len() >= 4 {
                break;
            }
        }
    }
    crate::xlog_info!(
        "frame trace: draws vs {vs:016x} ps {ps:016x} into {:?}{} – textures {}; {}",
        desc(rt),
        if super::present::is_presented(rt) { " (presented)" } else { "" },
        tex.join(", "),
        super::list_x::state_summary(this)
    );
}

/// `XCL_TRACE_SHADERS=hash,…` (dump hashes, hex): the draws traced with their root arguments
/// are those whose vertex or pixel shader is listed – into any target, at any time (the first
/// 24); without it, draws into a presented target (or any – `XCL_TRACE_UI_ANY`) after 300
/// presents.
fn traced_shaders() -> &'static [u64] {
    static S: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    S.get_or_init(|| std::env::var("XCL_TRACE_SHADERS").unwrap_or_default().split(',').filter_map(|h| u64::from_str_radix(h.trim(), 16).ok()).collect())
}

/// `XCL_TRACE_FRAME` with a watched target (`present::watch_rt`): draws whose descriptor
/// tables name it – that object, or another at its title address (the first 16).
unsafe fn trace_reads_of_watched(this: *mut c_void, pso: usize) {
    let Some(w) = super::present::watched_rt() else { return };
    let w_addr = super::real::title_address(w).or_else(|| super::title_textures::address(w));
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..16usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let addr = super::real::title_address(r).or_else(|| super::title_textures::address(r));
            if super::present::is_watched(r) || (addr.is_some() && addr == w_addr) {
                let rt = super::present::rt0_resource(this as usize).unwrap_or(0);
                // each (source, target, shaders) once
                static SEEN: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());
                {
                    let mut g = SEEN.lock().unwrap();
                    if g.len() >= 32 || g.contains(&(r, rt, pso)) {
                        return;
                    }
                    g.push((r, rt, pso));
                }
                if !super::present::is_presented(rt) {
                    super::present::follow(rt);
                }
                if let Some(dsv) = super::list_x::bound_dsv(this) {
                    super::list_x::watch_dsv(dsv);
                }
                let d = |r: usize| ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| { let d = x.GetDesc(); (d.Width, d.Height, d.Format.0, d.Flags.0) });
                crate::xlog_info!(
                    "frame trace: a draw (shaders {:x?}, {}) reads a watched target {r:#x} {:?} at {addr:x?} (slot {k}), drawing into {rt:#x} {:?}{} – {}",
                    super::real::pso_shaders(super::pso_async::resolve(pso)),
                    super::list_x::state_summary(this),
                    d(r),
                    d(rt),
                    if super::present::is_presented(rt) { " (presented)" } else { "" },
                    root_args_summary(this, 32)
                );
                let args = DRAW_ARGS.with(|d| d.get());
                crate::xlog_info!(
                    "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
                    super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
                    super::list_x::geometry_summary(this, args)
                );
                return;
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: render targets a draw into the watched target samples (a UI library's
/// layers) are dumped with it (`present::follow`).
unsafe fn follow_layer_sources(this: *mut c_void) {
    let (Some(w), Some(rt)) = (super::present::watched_rt(), super::present::rt0_resource(this as usize)) else { return };
    if rt != w {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    for p in layout.params.iter() {
        let Param::Table { at } = *p else { continue };
        let t = word(at as usize) as usize;
        for k in 0..4usize {
            let Some(r) = super::console_heap::resource_at(u.base() + t + 32 * k) else { continue };
            let layer = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).is_some_and(|x| {
                let d = x.GetDesc();
                d.Flags.0 & 1 != 0 && d.Width >= 64 && d.Height >= 32 && d.SampleDesc.Count == 1
            });
            if layer && r != w {
                super::present::follow(r);
            }
        }
    }
}

unsafe fn trace_ui_draw(this: *mut c_void, pso: usize) {
    trace_draw_kinds(this, pso);
    trace_reads_of_watched(this, pso);
    follow_layer_sources(this);
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // `XCL_TRACE_MANY=n`: n draws instead of 24 (a whole frame of a UI)
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let cap = *CAP.get_or_init(|| std::env::var("XCL_TRACE_MANY").ok().and_then(|v| v.parse().ok()).unwrap_or(24));
    if N.load(std::sync::atomic::Ordering::Relaxed) >= cap {
        return;
    }
    let Some(rt) = super::present::rt0_resource(this as usize) else { return };
    let wanted = traced_shaders();
    if wanted.is_empty() {
        // draws into a presented target, or – `XCL_TRACE_UI_ANY` – into any target
        if super::present::activity().0 < 300 || (!super::present::is_presented(rt) && std::env::var_os("XCL_TRACE_UI_ANY").is_none()) {
            return;
        }
    } else if !super::real::pso_shaders(super::pso_async::resolve(pso)).is_some_and(|(v, p)| wanted.contains(&v) || wanted.contains(&p)) {
        return;
    } else {
        super::present::watch_rt(rt);
    }
    N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let sh = super::real::pso_shaders(super::pso_async::resolve(pso)).map(|(v, p)| format!("vs {v:016x} ps {p:016x}"));
    crate::xlog_info!("frame trace: UI draw (pipeline {pso:#x}, {}): {}", sh.unwrap_or_default(), root_args_summary(this, 16));
    if !wanted.is_empty() {
        // the picked shaders' geometry too (vertices dumped at the next present)
        let args = DRAW_ARGS.with(|d| d.get());
        crate::xlog_info!(
            "frame trace: … that draw: {args:?} (indexed, count, instances, first, base vertex); input layout [{}]; {}",
            super::real::pso_layout(super::pso_async::resolve(pso)).unwrap_or_default(),
            super::list_x::geometry_summary(this, args)
        );
    }
}

/// A draw's root arguments for the frame trace: constants, root buffers (their first words, as
/// the CPU sees them), the first slots of descriptor tables (the PC resources they name).
unsafe fn root_args_summary(this: *mut c_void, buffer_words: usize) -> String {
    let Some(u) = crate::gpu::uma::get() else { return String::new() };
    let (layout, args) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone()));
    let Some(layout) = layout else { return String::new() };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).unwrap_or(0);
    let mut what = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => what.push(format!("p{i} constants {:x?}", (0..count.min(8) as usize).map(|k| word(at as usize + 4 * k)).collect::<Vec<_>>())),
            Param::Descriptor { at } => {
                let va = (word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32) as usize;
                let w: Vec<u32> = if va != 0 && crate::gpu::shader::readable(va, 4 * buffer_words) { (0..buffer_words).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect() } else { Vec::new() };
                what.push(format!("p{i} buffer {va:#x} {w:x?}"));
            }
            Param::Table { at } => {
                let t = word(at as usize) as usize;
                let slots: Vec<String> = (0..4usize)
                    .map(|k| {
                        let at = u.base() + t + 32 * k;
                        let w = super::console_heap::words_at((t + 32 * k) as u32);
                        match super::console_heap::resource_at(at) {
                            Some(r) => {
                                if super::title_textures::address(r).is_some() {
                                    super::title_textures::probe_gpu_content(r);
                                }
                                let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
                                format!("[{k}] {r:#x} {:?} title {:?} T# {:x?}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), super::title_textures::address(r).map(|a| format!("{a:#x}")), w.map(|w| w[..4].to_vec()))
                            }
                            None => format!("[{k}] - T# {:x?}", w.map(|w| w[..4].to_vec())),
                        }
                    })
                    .collect();
                what.push(format!("p{i} table {t:#x}: {}", slots.join(", ")));
            }
        }
    }
    what.join("; ")
}

/// #33 SetComputeRootSignature (PC #29)
pub(super) unsafe extern "system" fn set_compute_root_signature(this: *mut c_void, rs: *mut c_void) {
    let layout = console_list::layout_of(rs as usize);
    with(this, |s| {
        s.args = vec![0u8; layout.as_ref().map_or(0, |l| l.size as usize)];
        s.layout = layout;
    });
}

/// #35 SetComputeRootDescriptorTable (PC #31): the table's console address, 32 bits
pub(super) unsafe extern "system" fn set_compute_root_table(this: *mut c_void, i: u32, handle: u64) {
    put(this, i, 0, &(handle as u32).to_le_bytes());
}

/// #37 SetComputeRoot32BitConstant (PC #33)
pub(super) unsafe extern "system" fn set_compute_root_constant(this: *mut c_void, i: u32, value: u32, offset: u32) {
    put(this, i, offset, &value.to_le_bytes());
}

/// #39 SetComputeRoot32BitConstants (PC #35)
pub(super) unsafe extern "system" fn set_compute_root_constants(this: *mut c_void, i: u32, n: u32, data: *const u32, offset: u32) {
    if !data.is_null() {
        let bytes: Vec<u8> = std::slice::from_raw_parts(data, n as usize).iter().flat_map(|w| w.to_le_bytes()).collect();
        put(this, i, offset, &bytes);
    }
}

/// #41 / #43 / #45 SetComputeRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View (PC #37 /
/// #39 / #41): the 64-bit GPU address
pub(super) unsafe extern "system" fn set_compute_root_descriptor(this: *mut c_void, i: u32, va: u64) {
    note_address(va);
    put(this, i, 0, &va.to_le_bytes());
}

/// #25 SetPipelineState: noted, then bound as on the PC
pub(super) unsafe extern "system" fn set_pipeline_state(this: *mut c_void, pso: *mut c_void) {
    let resolved = super::real::bind_pipeline(this, pso);
    with(this, |s| {
        s.pso = pso as usize;
        s.bound_pso = resolved;
    });
}

/// #14 Dispatch: with a translated pipeline, on XCL's ABI with the kept root arguments
pub(super) unsafe extern "system" fn dispatch(this: *mut c_void, x: u32, y: u32, z: u32) {
    let pso = super::pso_async::resolve_wait(with(this, |s| s.pso));
    let Some(list) = ID3D12GraphicsCommandList::from_raw_borrowed(&this) else { return };
    if !super::real::is_translated_compute(pso) {
        // a console pipeline without a translation (including a background build that
        // failed): nothing to run (its root arguments were not given to the PC list either)
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: Dispatch with a pipeline that has no translated console shader ({pso:#x}) skipped");
        }
        count(DISPATCHES_SKIPPED);
        return;
    }
    match translated_dispatch(this, list, pso, x, y, z) {
        Ok(()) => count(DISPATCHES),
        Err(e) => {
            count(DISPATCHES_SKIPPED);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                crate::xlog_warn!("D3D12.X: Dispatch with a translated pipeline skipped: {e}");
            }
        }
    }
}

/// `XCL_TRACE_FRAME`: a full-screen dispatch (enough groups to cover a 4K target in tiles of up
/// to 128 pixels) – what its root arguments say. A bindless title passes the indices of its
/// descriptors in root constants and constant buffers: every small value there is looked up
/// in the dispatch's descriptor tables, and the resources found (the pass's inputs and
/// outputs) are logged.
unsafe fn trace_fullscreen(this: *mut c_void, pso: usize, groups: [u32; 3]) {
    if groups[0] < 30 || groups[1] < 17 {
        return;
    }
    let Some(u) = crate::gpu::uma::get() else { return };
    let (layout, args) = with(this, |s| (s.layout.clone(), s.args.clone()));
    let Some(layout) = layout else { return };
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&pso) || g.len() >= 12 {
            return;
        }
        g.push(pso);
    }
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let mut tables: Vec<u32> = Vec::new();
    let mut values: Vec<u32> = Vec::new();
    let mut what: Vec<String> = Vec::new();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => {
                let w: Vec<u32> = (0..count as usize).filter_map(|k| word(at as usize + 4 * k)).collect();
                what.push(format!("p{i} constants {w:x?}"));
                values.extend(&w);
            }
            Param::Descriptor { at } => {
                let lo = word(at as usize).unwrap_or(0) as u64;
                let hi = word(at as usize + 4).unwrap_or(0) as u64;
                let va = (lo | hi << 32) as usize;
                let n = 64usize;
                if va != 0 && crate::gpu::shader::readable(va, 4 * n) {
                    let w: Vec<u32> = (0..n).map(|k| ((va + 4 * k) as *const u32).read_unaligned()).collect();
                    what.push(format!("p{i} buffer {va:#x}: {w:x?}"));
                    values.extend(&w);
                } else {
                    what.push(format!("p{i} buffer {va:#x} (not readable)"));
                }
            }
            Param::Table { at } => {
                let v = word(at as usize).unwrap_or(0);
                what.push(format!("p{i} table {v:#x}"));
                tables.push(v);
            }
        }
    }
    let mut found: Vec<(u32, usize)> = Vec::new();
    for &v in &values {
        if v >= 1 << 20 {
            continue;
        }
        for &t in &tables {
            let slot = u.base() + t as usize + 32 * v as usize;
            if let Some(r) = super::console_heap::resource_at(slot) {
                if !found.iter().any(|x| x.1 == r) {
                    found.push((v, r));
                }
            }
        }
    }
    let res: Vec<String> = found
        .iter()
        .map(|&(v, r)| {
            let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc());
            format!("[{v}] {r:#x} {:?}{}", d.map(|d| (d.Dimension.0, d.Width, d.Height, d.Format.0)), if super::present::is_presented(r) { " PRESENTED" } else { "" })
        })
        .collect();
    crate::xlog_info!("frame trace: full-screen dispatch {groups:?}, pipeline {pso:#x}: {}", what.join("; "));
    crate::xlog_info!("frame trace: … its indices name: {}", if res.is_empty() { "nothing".to_string() } else { res.join("; ") });
}

/// #148 AllocateUploadMemory(size, alignment) → CPU/GPU pointer in unified memory (used by
/// `XboxBvhBuilder.dll` and other console middleware on default title command lists).
/// Freed automatically when the list's command allocator is reset.
pub(super) unsafe extern "system" fn allocate_upload_memory(this: *mut c_void, size: usize, align: usize) -> *mut u8 {
    let Some(u) = crate::gpu::uma::get() else {
        return std::ptr::null_mut();
    };
    let size = size.max(16);
    let align = align.max(16);
    if size <= crate::gpu::uma::ARENA_BLOCK / 2 {
        if let Some(p) = with(this, |s| s.arena.take(u, size, align, &mut s.uma)) {
            return p;
        }
    }
    if u.has_sparse() {
        if let Some((p, _)) = super::sparse::alloc(size, align) {
            with(this, |s| s.uma.push(p));
            return p as *mut u8;
        }
    }
    match u.alloc(size, align) {
        Some(p) => {
            with(this, |s| s.uma.push(p as usize));
            p
        }
        None => std::ptr::null_mut(),
    }
}

unsafe fn translated_dispatch(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, x: u32, y: u32, z: u32) -> Result<(), String> {
    if super::present::trace_frame() {
        trace_fullscreen(this, pso, [x, y, z]);
    }
    bind_abi(this, list, pso, false)?;
    list.Dispatch(x, y, z);
    Ok(())
}

/// XCL's ABI for a translated pipeline: its root signature, the user SGPRs (`io`, the kept
/// arguments the console's way – compute or graphics) and the argument table in unified
/// memory, u1 res, u2… the memory windows; then the pipeline.
unsafe fn bind_abi(this: *mut c_void, list: &ID3D12GraphicsCommandList, pso: usize, gfx: bool) -> Result<(), String> {
    super::list_x::ensure_view_heap(this);
    let u = crate::gpu::uma::get().ok_or("no unified memory")?;
    let (windows, win_count) = super::real::uma_window_vas_fast();
    if win_count == 0 {
        return Err("unified memory is not open on the GPU".into());
    }
    let first = windows[0];
    // a pipeline still being built in the background is waited for
    let real = super::pso_async::resolve_wait(pso);
    // a compute shader takes no more arguments from registers than it has (`inline_for`)
    let user_sgprs = if gfx { None } else { super::real::pso_user_sgprs(real) };
    let rs_raw = super::real::pso_root_signature(real)
        .or_else(|| super::real::abi_rs_cached().map(|p| p as usize))
        .ok_or("no XCL ABI root signature")?;
    let rs_ptr = rs_raw as *mut c_void;
    if rs_ptr.is_null() { return Err("root signature".into()); }
    let real_ptr = real as *mut c_void;
    let pc = ID3D12PipelineState::from_raw_borrowed(&real_ptr).ok_or("pipeline")?;

    // Single lock for reading root args, bump-allocating UMA arena slices, and updating cached PC bindings.
    let (io_at, res_at, need_rs, need_pso) = with(this, |s| -> Result<(usize, usize, bool, bool), String> {
        let (layout, args) = if gfx { (&s.layout_gfx, &s.args_gfx) } else { (&s.layout, &s.args) };
        let inline = layout.as_ref().map_or(0, |l| if gfx { l.inline_gfx } else { console_list::inline_for(l, user_sgprs) } as usize).min(args.len());
        let table_bytes = args.len() - inline;
        // 32-bit console addresses: window 0; pieces of the list's arena (not a 64 KiB grain each)
        let table_p = s.arena.take(u, table_bytes.max(16), 256, &mut s.uma).ok_or("unified memory is full")?;
        std::ptr::copy_nonoverlapping(args[inline..].as_ptr(), table_p, table_bytes);
        let table_at = table_p as usize;

        let mut io = [0u32; crate::gpu::recomp::KERNEL_SGPRS];
        match layout.as_ref() {
            Some(l) if gfx => console_list::write_user_data_gfx(&mut io, l, args, table_at as u64),
            Some(l) => console_list::write_user_data_for(&mut io, l, user_sgprs, args, table_at as u64),
            None => {
                io[0] = table_at as u32;
                io[1] = (table_at >> 32) as u32;
            }
        }
        io[crate::gpu::recomp::IO_UMA_BASE_HI] = (u.base() >> 32) as u32;
        let io_bytes = std::mem::size_of_val(&io);
        let io_p = s.arena.take(u, io_bytes, 256, &mut s.uma).ok_or("unified memory is full")?;
        std::ptr::copy_nonoverlapping(io.as_ptr() as *const u8, io_p, io_bytes);
        let res_p = s.arena.take(u, 256, 256, &mut s.uma).ok_or("unified memory is full")?;
        std::ptr::write_bytes(res_p, 0, 256);

        let slot_rs = if gfx { &mut s.bound_rs_gfx } else { &mut s.bound_rs_compute };
        let need_rs = *slot_rs != rs_raw;
        *slot_rs = rs_raw;
        let need_pso = s.bound_pso != real;
        s.bound_pso = real;
        Ok((io_p as usize, res_p as usize, need_rs, need_pso))
    })?;

    let io_va = super::real::uma_va_of_in(u, &windows, win_count, io_at).unwrap_or(first);
    let res_va = super::real::uma_va_of_in(u, &windows, win_count, res_at).unwrap_or(first);
    if need_rs {
        set_pc_signature(this, rs_ptr, gfx);
    }
    if gfx {
        list.SetGraphicsRootUnorderedAccessView(0, io_va);
        list.SetGraphicsRootUnorderedAccessView(1, res_va);
        if need_rs {
            for k in 0..4 {
                list.SetGraphicsRootUnorderedAccessView((k + 2) as u32, if k < win_count { windows[k] } else { first });
            }
        }
    } else {
        list.SetComputeRootUnorderedAccessView(0, io_va);
        list.SetComputeRootUnorderedAccessView(1, res_va);
        if need_rs {
            for k in 0..4 {
                list.SetComputeRootUnorderedAccessView((k + 2) as u32, if k < win_count { windows[k] } else { first });
            }
        }
    }
    if need_pso {
        list.SetPipelineState(pc);
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn last_draw_args() -> (bool, u32, u32, u32, i32) {
    DRAW_ARGS.with(|d| d.get())
}

#[cfg(test)]
pub(super) fn counts_snapshot() -> [u64; 5] {
    std::array::from_fn(|k| COUNTS[k].load(Ordering::Relaxed))
}

#[cfg(test)]
pub(super) fn tracked_pso(list: *mut c_void) -> usize {
    STATES
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&(list as usize)))
        .map_or(0, |s| s.pso)
}

#[cfg(test)]
mod tests {
    use super::super::{device_x, list_x, pso_async, real, title_list};
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A pipeline with the rect list GS draws its rect lists whole: draw_rects leaves them alone.
    #[test]
    fn rect_lists_with_the_rect_list_gs_are_not_expanded_on_the_cpu() {
        let (with_gs, without) = (0x7E57_4000usize, 0x7E57_5000usize);
        real::register_rect_list_gs_for_test(with_gs);
        assert!(!cpu_rect_expansion(with_gs));
        assert!(cpu_rect_expansion(without) && cpu_rect_expansion(0));
        real::forget_pso_for_test(with_gs);
        assert!(cpu_rect_expansion(with_gs));
    }

    /// An ExecuteIndirect whose bound pipeline is still being built is skipped (its stand-in
    /// with XCL's ABI root signature on the list is the debug layer's 953 and a hung GPU);
    /// a built or failed one goes to the PC call. The gate `execute_indirect` uses.
    #[test]
    fn execute_indirect_with_a_pipeline_still_building_is_skipped() {
        let (building, built, failed) = (0x7E57_6001usize, 0x7E57_6002usize, 0x7E57_6003usize);
        // lazy submits: no worker starts them, so the states stay under the test's control
        pso_async::submit_lazy(building, false, Box::new(|| 0x1234), 0);
        pso_async::submit_lazy(built, false, Box::new(|| 0x1235), 0);
        pso_async::submit_lazy(failed, false, Box::new(|| 0), 0);
        assert!(pso_async::is_pending(building));
        assert_eq!(pso_async::resolve_wait(built), 0x1235);
        assert_eq!(pso_async::resolve_wait(failed), failed);
        assert!(!pso_async::is_pending(built) && !pso_async::is_pending(failed));
        // never resolved: still pending (the skip case), then forgotten
        assert!(pso_async::is_pending(building));
        pso_async::released(building);
        pso_async::released(built);
        pso_async::released(failed);
        assert!(!pso_async::is_pending(building));
    }

    /// A dispatch writes an ExecuteIndirect's group count through another buffer over the same
    /// memory (as translated shaders write through the memory windows), a UAV barrier (what a
    /// console barrier becomes), then the ExecuteIndirect: it runs the count just written, not
    /// the one the memory held before the list (`fresh_indirect_arguments`; without it an RTX
    /// 5090 ran the old count in nearly every try).
    #[test]
    fn execute_indirect_reads_the_arguments_written_before_it() {
        use crate::gpu::{fence, recomp::runner::skip_test};
        use std::mem::{transmute_copy, ManuallyDrop};
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

        const TRIES: u32 = 40;
        // mode 0: write (x, 1, 1) at a[0..12]; mode 1: count the groups
        let src = r#"
#define RS "RootConstants(num32BitConstants=2, b0), UAV(u0), UAV(u1)"
cbuffer C : register(b0) { uint mode; uint x; };
RWByteAddressBuffer a : register(u0);
RWByteAddressBuffer c : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint t : SV_GroupIndex) {
    if (t != 0u) return;
    if (mode == 0u) a.Store3(0, uint3(x, 1u, 1u)); else c.InterlockedAdd(0, 1u);
}
"#;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                skip_test("no GPU");
                return;
            }
            let dev = dev.unwrap();
            let dxil = crate::gpu::recomp::dxc::compile(src, "main", "cs_6_6", &[]).unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
            let cs = D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: transmute_copy(&rs), CS: cs, ..Default::default() }).unwrap();
            let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() };
            let mut sig: Option<ID3D12CommandSignature> = None;
            dev.CreateCommandSignature(&D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 12, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 }, None, &mut sig).unwrap();
            let sig = sig.unwrap();
            let ty = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: ty, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let desc = |width| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: width,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            // the arguments and the buffer the dispatch writes them through: one memory
            let mut heap: Option<ID3D12Heap> = None;
            let heap_desc = D3D12_HEAP_DESC { SizeInBytes: 1 << 16, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: 0, Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS };
            dev.CreateHeap(&heap_desc, &mut heap).unwrap();
            let heap = heap.unwrap();
            let placed = || {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreatePlacedResource(&heap, 0, &desc(1 << 16), D3D12_RESOURCE_STATE_COMMON, None, &mut r).unwrap();
                r.unwrap()
            };
            let (args, written) = (placed(), placed());
            // the group count, read on the CPU
            let cpu_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
            let mut groups: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&cpu_heap, D3D12_HEAP_FLAG_NONE, &desc(256), D3D12_RESOURCE_STATE_COMMON, None, &mut groups).unwrap();
            let groups = groups.unwrap();
            let mut count = std::ptr::null_mut();
            groups.Map(0, None, Some(&mut count)).unwrap();
            let count = count as *mut u32;
            let uav_barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: ManuallyDrop::new(None) }) },
            };
            let run = |list: &ID3D12GraphicsCommandList, close: &dyn Fn()| {
                close();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let v = done.GetCompletedValue() + 1;
                queue.Signal(&done, v).unwrap();
                fence::wait(&done, v, &dev, None, 10_000).unwrap();
            };
            let (mut old, mut wrong) = (0u32, Vec::new());
            for k in 0..TRIES {
                // each try writes a count the memory does not hold yet
                let new = if k % 2 == 0 { 4096 } else { 0 };
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(ty).unwrap();
                let record = |x: u32| -> ID3D12GraphicsCommandList {
                    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, ty, &alloc, &pso).unwrap();
                    list.SetComputeRootSignature(&rs);
                    list.SetComputeRootUnorderedAccessView(1, written.GetGPUVirtualAddress());
                    list.SetComputeRootUnorderedAccessView(2, groups.GetGPUVirtualAddress());
                    list.SetComputeRoot32BitConstants(0, 2, [0u32, x].as_ptr() as _, 0);
                    list.Dispatch(1, 1, 1);
                    list
                };
                let before = record(old);
                run(&before, &|| before.Close().unwrap());
                alloc.Reset().unwrap();
                *count = 0;
                let list = record(new);
                list.ResourceBarrier(&[uav_barrier.clone()]);
                list.SetComputeRoot32BitConstants(0, 2, [1u32, 0].as_ptr() as _, 0);
                // from here as XCL records a title's ExecuteIndirect: on the adopted list
                let raw = list.as_raw();
                real::adopt(raw, real::Kind::CommandList);
                list_x::on_create(raw, alloc.as_raw() as usize, 0, ty);
                type Ei = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);
                type Close = unsafe extern "system" fn(*mut c_void) -> windows::core::HRESULT;
                let ei: Ei = std::mem::transmute(real::orig_fn(raw, 59));
                fresh_indirect_arguments(raw, || ei(raw, sig.as_raw() as usize, 1, args.as_raw() as usize, 0, 0, 0));
                let close: Close = std::mem::transmute(real::orig_fn(raw, 9));
                run(&list, &|| close(raw).unwrap());
                let ran = std::ptr::read_volatile(count);
                if ran != new {
                    wrong.push((k, ran));
                }
                old = new;
            }
            groups.Unmap(0, None);
            assert!(wrong.is_empty(), "{} of {TRIES} ExecuteIndirects ran another group count than the one written before them (try, groups): {wrong:?}", wrong.len());
        }
    }

    /// Minimal RTS0 v2 root signature blob with one 4-constant parameter.
    fn test_rts0_blob() -> Vec<u8> {
        let words: [u32; 12] = [
            2, 1, 24, 0, 0, 0, // version=2, num_params=1, params_offset=24
            1, 0, 36,          // param 0: type=Constants(1), visibility=0, payload_offset=36
            0, 0, 4,           // shader_reg=0, reg_space=0, num_32bit_values=4
        ];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn reset_and_clear_state_track_initial_pso_and_clear_root_args() {
        let list = 0x7100_0001usize as *mut c_void;
        let rs_compute = 0x7100_0010usize;
        let rs_gfx = 0x7100_0020usize;
        let alloc1 = 0x7100_0100usize;
        let alloc2 = 0x7100_0200usize;
        let pso_init = 0x7100_1000usize;
        let pso_mid = 0x7100_2000usize;
        let pso_cleared = 0x7100_3000usize;
        let pso_reset = 0x7100_4000usize;

        let blob = test_rts0_blob();
        console_list::note_root_signature(rs_compute, &blob);
        console_list::note_root_signature(rs_gfx, &blob);

        // Initial CreateCommandList / reset seeds PSO and allocator with clean root state.
        reset(list, alloc1, pso_init);
        with(list, |s| {
            assert_eq!(s.pso, pso_init);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Populate compute + graphics root signatures, constants, and a different PSO.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0xDEAD_BEEF, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0xCAFE_BABE, 0);
        }
        with(list, |s| {
            s.pso = pso_mid;
            assert!(s.layout.is_some() && !s.args.is_empty());
            assert!(s.layout_gfx.is_some() && !s.args_gfx.is_empty());
        });

        // ClearState(pso_cleared) updates PSO and clears root signatures/args while keeping allocator.
        clear_state(list, pso_cleared);
        with(list, |s| {
            assert_eq!(s.pso, pso_cleared);
            assert_eq!(s.allocator, alloc1);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // ClearState(null) clears PSO to 0.
        clear_state(list, 0);
        assert_eq!(tracked_pso(list), 0);

        // Re-populate root args, then Reset(alloc2, pso_reset) switches allocator/PSO and clears root args.
        unsafe {
            set_compute_root_signature(list, rs_compute as *mut c_void);
            set_compute_root_constant(list, 0, 0x1234_5678, 0);
            set_graphics_root_signature(list, rs_gfx as *mut c_void);
            set_graphics_root_constant(list, 0, 0x8765_4321, 0);
        }
        reset(list, alloc2, pso_reset);
        with(list, |s| {
            assert_eq!(s.pso, pso_reset);
            assert_eq!(s.allocator, alloc2);
            assert!(s.layout.is_none() && s.args.is_empty());
            assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty());
        });

        // Reset with null initial PSO clears stale PSO back to 0.
        reset(list, alloc2, 0);
        assert_eq!(tracked_pso(list), 0);

        forget(list);
    }

    #[repr(C)]
    struct FakeAdoptedList {
        vt: *const usize,
        refs: AtomicUsize,
        dispatch_calls: AtomicUsize,
        set_compute_rs_calls: AtomicUsize,
        set_compute_uav_calls: AtomicUsize,
        set_pso_calls: AtomicUsize,
        execute_indirect_calls: AtomicUsize,
    }

    unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_add(1, Ordering::Relaxed) as u32 + 1
    }

    unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
        (*(this as *const FakeAdoptedList)).refs.fetch_sub(1, Ordering::Relaxed) as u32 - 1
    }

    unsafe extern "system" fn fake_dispatch(this: *mut c_void, _x: u32, _y: u32, _z: u32) {
        (*(this as *const FakeAdoptedList)).dispatch_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_pso(this: *mut c_void, _pso: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_pso_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_rs(this: *mut c_void, _rs: *mut c_void) {
        (*(this as *const FakeAdoptedList)).set_compute_rs_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_set_compute_uav(this: *mut c_void, _slot: u32, _va: u64) {
        (*(this as *const FakeAdoptedList)).set_compute_uav_calls.fetch_add(1, Ordering::Relaxed);
    }

    unsafe extern "system" fn fake_execute_indirect(
        this: *mut c_void,
        _sig: usize,
        _max: u32,
        _args: usize,
        _args_off: u64,
        _count_buf: usize,
        _count_off: u64,
    ) {
        (*(this as *const FakeAdoptedList)).execute_indirect_calls.fetch_add(1, Ordering::Relaxed);
    }

    #[test]
    fn async_compute_pso_success_and_failure_in_dispatch_and_execute_indirect() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;

        // Extended vtable layout expected by `real::orig_fn`: entry -1 holds `orig_vt` pointer,
        // and entries 0..64 hold the COM method pointers used via `ID3D12GraphicsCommandList`.
        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeAdoptedList {
            // SAFETY: `ext_vt` has 65 elements so `.add(1)` points to slot 0 with `orig_vt` at `-1`.
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;

        // 1. Background compute build SUCCEEDS: real PSO R is registered in TRANSLATED_COMPUTE,
        // stand-in S is NOT. ExecuteIndirect resolves S -> R and treats it as translated.
        let standin_ok = 0x7200_0001usize;
        let real_ok = 0x7200_0002usize;
        let sig_root_changing = 0x7200_00F0usize;
        note_root_changing_signature(sig_root_changing);

        pso_async::submit(
            standin_ok,
            true,
            Box::new(move || {
                real::register_translated_compute_for_test(real_ok);
                real_ok
            }),
        );
        assert_eq!(pso_async::resolve_wait(standin_ok), real_ok);
        assert!(!real::is_translated_compute(standin_ok));
        assert!(real::is_translated_compute(real_ok));

        reset(list_ptr, 0x7200_0100, standin_ok);
        unsafe {
            // With `sig_root_changing`, a translated compute PSO is caught by the root-changing
            // guard instead of falling through to `pc` (which happened when `R` wasn't registered).
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            0,
            "ExecuteIndirect must recognize resolved async compute PSO R as translated"
        );

        // 2. Background compute build FAILS (returns 0): stand-in S_fail is NOT in
        // TRANSLATED_COMPUTE. Dispatch must skip cleanly without binding XCL's ABI RS/UAVs
        // onto S_fail, and ExecuteIndirect must not enter bind_abi either.
        let standin_fail = 0x7200_0003usize;
        pso_async::submit(standin_fail, true, Box::new(|| 0));
        assert_eq!(pso_async::resolve_wait(standin_fail), standin_fail);
        assert!(!real::is_translated_compute(standin_fail));

        reset(list_ptr, 0x7200_0100, standin_fail);
        let before = counts_snapshot();
        unsafe {
            dispatch(list_ptr, 1, 1, 1);
        }
        let after = counts_snapshot();
        assert_eq!(after[DISPATCHES], before[DISPATCHES]);
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 0);

        unsafe {
            execute_indirect(list_ptr, sig_root_changing, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            fake.execute_indirect_calls.load(Ordering::Relaxed),
            1,
            "failed async compute stand-in must not enter translated ExecuteIndirect path"
        );

        // 3. The same failed stand-in (made from a console shader) with a Dispatch signature:
        // nothing to run, as for Dispatch – the PC call would pair the stand-in with the list's
        // compute root signature (XCL's ABI one after a translated dispatch: debug layer 953)
        let sig_dispatch = 0x7200_00F8usize;
        real::register_console_compute_for_test(standin_fail);
        note_signature_dispatches(sig_dispatch, true);
        let before = counts_snapshot();
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        let after = counts_snapshot();
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 1, "a dispatch of an untranslated console compute pipeline must not reach the PC ExecuteIndirect");
        assert!(after[DISPATCHES_SKIPPED] > before[DISPATCHES_SKIPPED]);

        // 4. A native PC compute pipeline (not made from a console shader) with the same Dispatch
        // signature still goes to the PC call, as on main
        let native = 0x7200_0004usize;
        reset(list_ptr, 0x7200_0100, native);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 2, "a native PC compute pipeline must still reach the PC ExecuteIndirect");

        // a signature made later at the same address without a Dispatch argument goes to the PC
        reset(list_ptr, 0x7200_0100, standin_fail);
        note_signature_dispatches(sig_dispatch, false);
        unsafe {
            execute_indirect(list_ptr, sig_dispatch, 1, 0, 0, 0, 0);
        }
        assert_eq!(fake.execute_indirect_calls.load(Ordering::Relaxed), 3);

        forget(list_ptr);
        let _ = pso_async::released(standin_ok);
        let _ = pso_async::released(standin_fail);
        real::forget_pso_for_test(real_ok);
        real::forget_pso_for_test(standin_fail);
        assert!(!real::is_console_compute(standin_fail), "a released pipeline's address does not stay a console one");
    }

    #[test]
    fn create_list_reset_and_clear_state_track_initial_pso_on_d3d12() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        if console_list::enabled() {
            return;
        }
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            real::adopt(dev.as_raw(), real::Kind::Device);

            // Build two real PC compute PSOs from the inbox dummy CS bytecode.
            let mut ser = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).expect("serialize RS");
            let ser = ser.unwrap();
            let rs: ID3D12RootSignature = dev
                .CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize()))
                .expect("CreateRootSignature");
            let cs = real::dummy_cs_bytecode();
            let make_pso = || -> ID3D12PipelineState {
                let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as *const c_void, BytecodeLength: cs.len() },
                    ..Default::default()
                };
                let pso = dev.CreateComputePipelineState(&desc).expect("CreateComputePipelineState");
                std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
                pso
            };
            let pso1 = make_pso();
            let pso2 = make_pso();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");

            // Stand-in synthetic key backed by real `pso1` in pso_async: CreateCommandList must
            // resolve it via resolve_for_bind for PC D3D12 while tracking `standin` in `bind`.
            let standin = 0x7300_0001usize;
            let pso1_raw = pso1.as_raw() as usize;
            pso_async::submit(standin, true, Box::new(move || pso1_raw));
            assert_eq!(pso_async::resolve_wait(standin), pso1_raw);

            let mut raw_list: *mut c_void = std::ptr::null_mut();
            let iid = windows_sys::core::GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
            let hr = device_x::create_command_list_pc(
                dev.as_raw(),
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32,
                alloc.as_raw() as usize,
                standin,
                &iid,
                &mut raw_list,
            );
            assert!(hr >= 0 && !raw_list.is_null(), "create_command_list_pc failed: {hr:#x}");
            let backing = title_list::backing(raw_list).expect("wrapped in title_list");
            assert_eq!(tracked_pso(backing), standin, "CreateCommandList must seed bind::State.pso");

            // ClearState(pso2) updates bind::State.pso to pso2 and clears root args.
            let rs_key = 0x7300_0010usize;
            console_list::note_root_signature(rs_key, &test_rts0_blob());
            set_compute_root_signature(backing, rs_key as *mut c_void);
            set_compute_root_constant(backing, 0, 0xABCD, 0);
            let cl = ID3D12GraphicsCommandList::from_raw_borrowed(&backing).unwrap();
            cl.ClearState(&pso2);
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "ClearState must update bind::State.pso");
            with(backing, |s| assert!(s.layout.is_none() && s.args.is_empty()));

            // ClearState(None) clears bind::State.pso to 0.
            cl.ClearState(None);
            assert_eq!(tracked_pso(backing), 0, "ClearState(None) must clear bind::State.pso");

            // Close + reset_x(alloc, pso2) updates bind::State.pso to pso2 and clears root args.
            set_graphics_root_signature(backing, rs_key as *mut c_void);
            set_graphics_root_constant(backing, 0, 0x1111, 0);
            cl.Close().expect("Close");
            let hr = list_x::reset_x(backing, alloc.as_raw() as usize, pso2.as_raw() as usize);
            assert!(hr >= 0, "reset_x failed: {hr:#x}");
            assert_eq!(tracked_pso(backing), pso2.as_raw() as usize, "Reset must update bind::State.pso");
            with(backing, |s| assert!(s.layout_gfx.is_none() && s.args_gfx.is_empty()));

            // Release the wrapped title list cleanly (calls bind::forget + list_x::on_release).
            let release_fn: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(raw_list as *const *const usize)).add(2));
            assert_eq!(release_fn(raw_list), 0);
            let _ = pso_async::released(standin);
        }
    }

    #[test]
    fn binding_cache_flags_invalidate_on_reset_clear_state_and_external_calls() {
        let list = 0x7350_0001usize as *mut c_void;
        reset(list, 0x7350_0100, 0x7350_0200);
        with(list, |s| {
            s.bound_rs_gfx = 0x1111;
            s.bound_rs_compute = 0x2222;
            s.bound_pso = 0x3333;
        });

        note_pc_pso(list, 0x4444);
        with(list, |s| {
            assert_eq!(s.bound_pso, 0x4444);
            assert_eq!(s.bound_rs_gfx, 0x1111);
            assert_eq!(s.bound_rs_compute, 0x2222);
        });

        invalidate_pc_bindings(list);
        with(list, |s| {
            assert_eq!((s.bound_rs_gfx, s.bound_rs_compute, s.bound_pso), (0, 0, 0));
            assert_eq!(s.pso, 0x7350_0200);
            assert_eq!(s.allocator, 0x7350_0100);
        });

        with(list, |s| {
            s.bound_rs_gfx = 0x1111;
            s.bound_rs_compute = 0x2222;
            s.bound_pso = 0x3333;
        });
        clear_state(list, 0x7350_0300);
        with(list, |s| {
            assert_eq!((s.bound_rs_gfx, s.bound_rs_compute, s.bound_pso), (0, 0, 0));
            assert_eq!(s.pso, 0x7350_0300);
        });

        with(list, |s| {
            s.bound_rs_gfx = 0x1111;
            s.bound_rs_compute = 0x2222;
            s.bound_pso = 0x3333;
        });
        reset(list, 0x7350_0101, 0);
        with(list, |s| {
            assert_eq!((s.bound_rs_gfx, s.bound_rs_compute, s.bound_pso), (0, 0, 0));
        });

        forget(list);
    }

    #[test]
    fn translated_draw_rebinds_after_indirect_draw_and_bundle() {
        // Capture native calls, including which root parameters are rewritten before each draw.
        #[derive(Clone, Debug, PartialEq)]
        enum Call { Signature(usize), Uav(u32), Pipeline(usize), Draw, Indirect, Bundle(usize) }
        #[repr(C)]
        struct List { vt: *const usize, calls: Mutex<Vec<Call>> }
        unsafe fn record(this: *mut c_void, call: Call) {
            (*this.cast::<List>()).calls.lock().unwrap().push(call);
        }
        unsafe extern "system" fn refcount(_: *mut c_void) -> u32 { 1 }
        unsafe extern "system" fn signature(this: *mut c_void, rs: *mut c_void) { record(this, Call::Signature(rs as usize)); }
        unsafe extern "system" fn uav(this: *mut c_void, slot: u32, _: u64) { record(this, Call::Uav(slot)); }
        unsafe extern "system" fn pipeline(this: *mut c_void, pso: *mut c_void) { record(this, Call::Pipeline(pso as usize)); }
        unsafe extern "system" fn draw(this: *mut c_void, _: u32, _: u32, _: u32, _: u32) { record(this, Call::Draw); }
        unsafe extern "system" fn indirect(this: *mut c_void, _: usize, _: u32, _: usize, _: u64, _: usize, _: u64) { record(this, Call::Indirect); }
        unsafe extern "system" fn bundle(this: *mut c_void, ptr: *mut c_void) { record(this, Call::Bundle(ptr as usize)); }

        let mut orig = [0usize; 64];
        orig[1] = refcount as usize;
        orig[2] = refcount as usize;
        orig[12] = draw as usize;
        orig[25] = pipeline as usize;
        orig[27] = bundle as usize;
        orig[29] = signature as usize;
        orig[30] = signature as usize;
        orig[41] = uav as usize;
        orig[42] = uav as usize;
        orig[59] = indirect as usize;
        let mut ext = [0usize; 65];
        ext[0] = orig.as_ptr() as usize;
        ext[1..].copy_from_slice(&orig);
        ext[1 + 29] = external_root_signature::<false> as usize;
        ext[1 + 30] = external_root_signature::<true> as usize;
        let list = List { vt: ext[1..].as_ptr(), calls: Mutex::new(Vec::new()) };
        let ptr = &list as *const List as *mut c_void;
        let take = || std::mem::take(&mut *list.calls.lock().unwrap());
        let allocator = 0x7361_0100;
        let gfx = 0x7361_0001;
        let title = 0x7361_0002;
        if real::init().is_err() || real::uma_window_vas_fast().1 == 0 {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        unsafe {
            let Some(abi) = real::abi_rs_raw(real::device_raw()) else {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            };
            use Call::*;
            let full = || vec![Signature(abi as usize), Uav(0), Uav(1), Uav(2), Uav(3), Uav(4), Uav(5)];
            real::register_translated_graphics_for_test(gfx);
            reset(ptr, allocator, gfx);
            let draw_one = || draw_instanced(ptr, 3, 1, 0, 0);
            let mut initial = full();
            initial.extend([Pipeline(gfx), Draw]);
            draw_one();
            assert_eq!(take(), initial);
            set_graphics_root_signature(ptr, title as *mut c_void);
            for indirect in [true, false] {
                with(ptr, |s| s.pso = 0);
                if indirect {
                    execute_indirect(ptr, 1, 1, 0, 0, 0, 0);
                } else {
                    draw_one();
                }
                assert_eq!(take(), [Signature(title), if indirect { Indirect } else { Draw }]);
                with(ptr, |s| s.pso = gfx);
                draw_one();
                let mut expected = full();
                expected.push(Draw);
                assert_eq!(take(), expected);
            }
            // A compute bind changes the shared PSO but preserves graphics root arguments.
            let compute = 0x7361_0003;
            let cl = ID3D12GraphicsCommandList::from_raw_borrowed(&ptr).unwrap();
            bind_abi(ptr, cl, compute, false).unwrap();
            let mut expected = full();
            expected.push(Pipeline(compute));
            assert_eq!(take(), expected);
            draw_one();
            assert_eq!(take(), [Uav(0), Uav(1), Pipeline(gfx), Draw]);

            cl.SetComputeRootSignature(None);
            bind_abi(ptr, cl, compute, false).unwrap();
            let mut expected = vec![Signature(0)];
            expected.extend(full());
            expected.push(Pipeline(compute));
            assert_eq!(take(), expected);
            cl.SetGraphicsRootSignature(None);
            draw_one();
            let mut expected = vec![Signature(0)];
            expected.extend(initial.iter().cloned());
            assert_eq!(take(), expected);

            // Clearing fallback depth bias binds the base PSO directly on the native list.
            let base = 0x7361_0004;
            list_x::on_create(ptr, allocator, base, D3D12_COMMAND_LIST_TYPE_DIRECT);
            list_x::set_depth_bias_x(ptr, -1.0, 0.0, 0.0);
            list_x::set_depth_bias_x(ptr, 0.0, 0.0, 0.0);
            draw_one();
            assert_eq!(take(), [Pipeline(base), Uav(0), Uav(1), Pipeline(gfx), Draw]);
            list_x::on_release(ptr as usize);

            let other = List { vt: ext[1..].as_ptr(), calls: Mutex::new(Vec::new()) };
            let other_ptr = &other as *const List as *mut c_void;
            real::cl_execute_bundle(ptr, other_ptr);
            draw_one();
            let mut expected = vec![Bundle(other_ptr as usize)];
            expected.extend(initial);
            assert_eq!(take(), expected);
            real::forget_pso_for_test(gfx);
        }
        forget(ptr);
        console_list::allocator_reset(allocator as *mut c_void);
    }

    #[test]
    fn bind_abi_caches_root_sig_u2_to_u5_and_pso_across_dispatches() {
        let mut orig_vt = [0usize; 64];
        orig_vt[1] = fake_add_ref as usize;
        orig_vt[2] = fake_release as usize;
        orig_vt[14] = fake_dispatch as usize;
        orig_vt[25] = fake_set_pso as usize;
        orig_vt[29] = fake_set_compute_rs as usize;
        orig_vt[41] = fake_set_compute_uav as usize;
        orig_vt[59] = fake_execute_indirect as usize;

        let mut ext_vt = vec![0usize; 65];
        ext_vt[0] = orig_vt.as_ptr() as usize;
        ext_vt[1..].copy_from_slice(&orig_vt);

        let fake = Box::new(FakeAdoptedList {
            vt: unsafe { ext_vt.as_ptr().add(1) },
            refs: AtomicUsize::new(1),
            dispatch_calls: AtomicUsize::new(0),
            set_compute_rs_calls: AtomicUsize::new(0),
            set_compute_uav_calls: AtomicUsize::new(0),
            set_pso_calls: AtomicUsize::new(0),
            execute_indirect_calls: AtomicUsize::new(0),
        });
        let list_ptr = &*fake as *const FakeAdoptedList as *mut c_void;

        if real::init().is_err() {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        let raw_dev = real::device_raw();
        let (fast_vas, fast_count) = real::uma_window_vas_fast();
        let slow_vas = real::uma_window_vas();
        assert_eq!(fast_count, slow_vas.len());
        for (i, &va) in slow_vas.iter().take(4).enumerate() {
            assert_eq!(fast_vas[i], va);
        }
        let Some(u) = crate::gpu::uma::get() else {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        };
        if fast_count == 0 || unsafe { real::abi_rs_raw(raw_dev) }.is_none() {
            crate::gpu::recomp::runner::skip_test("no GPU");
            return;
        }
        if let Some(base_p) = u.alloc(4096, 256) {
            let p = base_p as usize;
            assert_eq!(
                real::uma_va_of_in(u, &fast_vas, fast_count, p),
                real::uma_va_of(p)
            );
            u.free(p);
        }

        let pso_a = 0x7360_0001usize;
        let pso_b = 0x7360_0002usize;
        real::register_translated_compute_for_test(pso_a);
        real::register_translated_compute_for_test(pso_b);

        reset(list_ptr, 0x7360_0100, pso_a);
        unsafe {
            // 1st dispatch: binds compute RS (1), u0..u5 (6), PSO (1).
            dispatch(list_ptr, 1, 1, 1);
            assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 1);
            assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 1);
            assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 6);
            assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 1);

            // 2nd dispatch with same RS & PSO: skips RS & u2..u5 & SetPipelineState, binds only u0..u1 (+2).
            dispatch(list_ptr, 1, 1, 1);
            assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 2);
            assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 1);
            assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 8);
            assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 1);

            // Switch to pso_b (same ABI RS): SetPipelineState updates (+1), RS & u2..u5 stay cached.
            with(list_ptr, |s| s.pso = pso_b);
            dispatch(list_ptr, 1, 1, 1);
            assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 3);
            assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 1);
            assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 10);
            assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 2);

            // External invalidation (e.g. KickoffX PC_RESET): forces full rebind on next dispatch.
            invalidate_pc_bindings(list_ptr);
            dispatch(list_ptr, 1, 1, 1);
            assert_eq!(fake.dispatch_calls.load(Ordering::Relaxed), 4);
            assert_eq!(fake.set_compute_rs_calls.load(Ordering::Relaxed), 2);
            assert_eq!(fake.set_compute_uav_calls.load(Ordering::Relaxed), 16);
            assert_eq!(fake.set_pso_calls.load(Ordering::Relaxed), 3);
        }

        forget(list_ptr);
        real::forget_pso_for_test(pso_a);
        real::forget_pso_for_test(pso_b);
    }
}
