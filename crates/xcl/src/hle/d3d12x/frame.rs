//! Console frame pacing: the device's frame clock (slots 78–81) and the flip tracking
//! `PresentX` feeds.
//!
//! A console title does not wait on a swap chain: it sets a frame interval
//! (`SetFrameIntervalX`, e.g. 16 667 µs), schedules frame events at an offset into each
//! interval (`ScheduleFrameEventX`: type 0 = the frame origin, 1 and 2 = further points of the
//! frame) and blocks on them (`WaitFrameEventX`, which hands out a frame token at the origin).
//! `PresentX` names the token of the frame it flips, `GetFrameStatisticsX` reports when that
//! frame's origin and flip happened. Semantics as observed on Fortnite (UE) – arguments,
//! return codes, the ancillary handle block `{ u32 count; HANDLE* list @8; i32* signaled @0x10 }`.
//!
//! XCL runs the clock on a thread of its own (high-resolution waitable timer, the last
//! ~0.6 ms spun): at each event it releases the type's internal semaphore (origin) or sets its
//! event, and signals the title's ancillary handles. Not locked to the monitor's vblank.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows_sys::Win32::Foundation::{CloseHandle, DuplicateHandle, DUPLICATE_SAME_ACCESS, HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows_sys::Win32::System::Threading::{
    CreateEventW, CreateSemaphoreW, CreateWaitableTimerExW, GetCurrentProcess, GetCurrentThread, ReleaseSemaphore, SetEvent, SetThreadPriority,
    SetWaitableTimer, WaitForMultipleObjects, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, INFINITE, THREAD_PRIORITY_TIME_CRITICAL, TIMER_ALL_ACCESS,
};

use crate::util::{HRESULT, S_OK};

const S_FALSE: HRESULT = 1;
/// what ScheduleFrameEventX returns when it starts the clock
const STARTED_CLOCK: HRESULT = 0x1000_0000;
const E_FAIL: HRESULT = 0x8000_4005u32 as i32;
const E_INVALIDARG: HRESULT = 0x8007_0057u32 as i32;
/// frame event types: 0 origin, 1 and 2 further points of the frame
const TYPES: usize = 3;
const ORIGIN_SEMAPHORE_MAX: i32 = 4;
/// WaitFrameEventX flags
const WAIT_ALL: u32 = 1;
const TOKEN_ON_TIMEOUT: u32 = 2;
/// ScheduleFrameEventX flags
const UNSCHEDULE: u32 = 1;
const TOKEN_MASK: u64 = 0xFFFF_FFFF_FFFF;
const KNOWN_TOKENS: usize = 64;
const FLIPPED_KEEP: usize = 64;

/// `XCL_FRAME_INTERVAL_US`: optional override for the console frame clock interval (µs).
/// - unset: honor the title's `SetFrameIntervalX` interval (default)
/// - `0` / `uncapped` / `off`: uncap `WaitFrameEventX` (return immediately with a fresh token)
/// - positive µs (e.g. `8333` for 120 Hz, `4167` for 240 Hz): clamp the clock's effective interval
///   to `min(requested, override)` (at least 1000 µs so the clock thread never busy-spins).
pub fn parse_interval_override(raw: Option<&str>) -> Option<u32> {
    let s = raw?.trim();
    if s.is_empty() {
        return None;
    }
    if s.eq_ignore_ascii_case("uncapped") || s.eq_ignore_ascii_case("off") {
        return Some(0);
    }
    s.parse::<u32>().ok()
}

fn interval_override() -> Option<u32> {
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| parse_interval_override(std::env::var("XCL_FRAME_INTERVAL_US").ok().as_deref()))
}

/// Effective clock thread interval (µs): `None` means uncapped (handled in `wait_frame_event_x`).
pub fn effective_clock_interval_us(requested_us: u32, override_us: Option<u32>) -> Option<u32> {
    if requested_us == 0 {
        return None;
    }
    match override_us {
        None => Some(requested_us),
        Some(0) => None,
        Some(ov) => Some(ov.min(requested_us).max(1000)),
    }
}

/// Frame intervals the console accepts (µs, with the rounding a title may use); the 120 Hz
/// ones (8 333, 25 000) only on a display that runs at 120 Hz.
pub fn interval_valid(len_us: u32, high_refresh: bool) -> bool {
    // either side of the nominal length: 1_000_000 / 60 = 16 666 and 16 667 are both 60 Hz
    let near = |base: u32, span: u32| len_us.abs_diff(base) <= span;
    if near(8333, 9) || near(25000, 25) {
        return high_refresh;
    }
    near(16667, 16) || near(33333, 34) || near(50000, 49) || near(66667, 67)
}

/// The first deadline `base + offset + k·interval` after `after`.
pub fn next_deadline(base: i64, interval: i64, offset: i64, after: i64) -> i64 {
    if interval <= 0 {
        return i64::MAX;
    }
    let first = base + offset;
    if after < first {
        return first;
    }
    first + ((after - first) / interval + 1) * interval
}

/// The title's ancillary handles of a frame event, duplicated. Closed when neither the schedule
/// nor the clock thread holds them any more: the thread signals them outside the clock's lock,
/// and a handle closed under it meanwhile could be another object's by then (#103).
struct Handles(Vec<HANDLE>);

impl Drop for Handles {
    fn drop(&mut self) {
        for &h in &self.0 {
            unsafe { CloseHandle(h) };
        }
    }
}

unsafe impl Send for Handles {}
unsafe impl Sync for Handles {}

struct Kind {
    /// semaphore (origin) or auto-reset event the clock signals
    internal: HANDLE,
    offset_us: u32,
    /// the title's ancillary handles, duplicated
    extra: Arc<Handles>,
    scheduled: bool,
}

struct Clock {
    interval_us: u32,
    active: bool,
    kinds: [Kind; TYPES],
    /// bumped on every change: the clock thread re-plans
    generation: u64,
    base_qpc: i64,
}

unsafe impl Send for Clock {}

static CLOCK: Mutex<Option<Clock>> = Mutex::new(None);
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);
/// (token, qpc of the origin it was issued at)
static ISSUED: Mutex<VecDeque<(u64, i64)>> = Mutex::new(VecDeque::new());
/// (token, origin qpc, flip qpc)
static FLIPPED: Mutex<VecDeque<(u64, i64, i64)>> = Mutex::new(VecDeque::new());
static LAST_ORIGIN: AtomicI64 = AtomicI64::new(0);
/// qpc of the origins the clock released and no WaitFrameEventX has consumed yet (the origin
/// semaphore counts up to ORIGIN_SEMAPHORE_MAX): each token gets its own frame's origin, not
/// the latest one (#103 – with 2–3 frames in flight the statistics named a newer frame's)
static ORIGINS: Mutex<VecDeque<i64>> = Mutex::new(VecDeque::new());

/// The clock's origin `now`: the origin semaphore `sem` released, `now` queued in `q` for the
/// wait it wakes (`take_origin`); `after_release` runs in between (a test's pause). The queue
/// stays locked across both, so a woken wait takes its origin only once it is queued (released
/// first, it found the queue empty and took a newer QPC, and the next wait this older origin).
/// false: the semaphore is full (ORIGIN_SEMAPHORE_MAX pending) – an origin lost to the title too.
unsafe fn publish_origin(q: &Mutex<VecDeque<i64>>, sem: HANDLE, now: i64, after_release: impl FnOnce()) -> bool {
    let mut o = q.lock().unwrap();
    if ReleaseSemaphore(sem, 1, std::ptr::null_mut()) == 0 {
        return false;
    }
    after_release();
    o.push_back(now);
    while o.len() > ORIGIN_SEMAPHORE_MAX as usize {
        o.pop_front();
    }
    true
}

/// The origin a wait the origin semaphore woke consumes (`publish_origin`).
fn take_origin(q: &Mutex<VecDeque<i64>>) -> Option<i64> {
    q.lock().unwrap().pop_front()
}
static THREAD: AtomicBool = AtomicBool::new(false);

pub fn qpc() -> i64 {
    let mut v = 0i64;
    unsafe { QueryPerformanceCounter(&mut v) };
    v
}

fn qpf() -> i64 {
    static F: OnceLock<i64> = OnceLock::new();
    *F.get_or_init(|| {
        let mut v = 0i64;
        unsafe { QueryPerformanceFrequency(&mut v) };
        v.max(1)
    })
}

fn ticks(us: i64) -> i64 {
    us * qpf() / 1_000_000
}

/// the monitor runs at 120 Hz or more
fn high_refresh() -> bool {
    use windows::Win32::Graphics::Gdi::{EnumDisplaySettingsW, DEVMODEW, ENUM_CURRENT_SETTINGS};
    static HIGH: OnceLock<bool> = OnceLock::new();
    *HIGH.get_or_init(|| unsafe {
        let mut dm = DEVMODEW { dmSize: std::mem::size_of::<DEVMODEW>() as u16, ..Default::default() };
        EnumDisplaySettingsW(windows::core::PCWSTR::null(), ENUM_CURRENT_SETTINGS, &mut dm).as_bool() && dm.dmDisplayFrequency >= 119
    })
}

fn wake() -> HANDLE {
    static W: OnceLock<usize> = OnceLock::new();
    *W.get_or_init(|| unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) } as usize) as HANDLE
}

fn with<R>(f: impl FnOnce(&mut Clock) -> R) -> R {
    let mut g = CLOCK.lock().unwrap();
    let c = g.get_or_insert_with(|| {
        let kind = |i: usize| Kind {
            internal: unsafe {
                if i == 0 {
                    CreateSemaphoreW(std::ptr::null(), 0, ORIGIN_SEMAPHORE_MAX, std::ptr::null())
                } else {
                    CreateEventW(std::ptr::null(), 0, 0, std::ptr::null())
                }
            },
            offset_us: 0,
            extra: Arc::new(Handles(Vec::new())),
            scheduled: false,
        };
        Clock { interval_us: 0, active: false, kinds: [kind(0), kind(1), kind(2)], generation: 0, base_qpc: 0 }
    });
    f(c)
}

/// device #78 SetFrameIntervalX(output, length µs, periodicity, flags)
pub unsafe extern "system" fn set_frame_interval_x(_this: *mut std::ffi::c_void, _output: *mut std::ffi::c_void, len_us: u32, periodicity: u32, flags: u32) -> HRESULT {
    if len_us != 0 && !interval_valid(len_us, high_refresh()) {
        crate::xlog_warn!("D3D12.X: SetFrameIntervalX({len_us} µs) – not an interval of this display");
        return E_INVALIDARG;
    }
    with(|c| {
        if c.interval_us != len_us {
            c.kinds[0].scheduled = false;
            if c.active && len_us != 0 {
                c.base_qpc = qpc();
            }
        }
        c.interval_us = len_us;
        if len_us == 0 {
            c.active = false;
            for k in c.kinds.iter_mut() {
                k.scheduled = false;
            }
        }
        c.generation += 1;
    });
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| match interval_override() {
        Some(0) => crate::xlog_info!("D3D12.X: frame interval {len_us} µs requested (XCL_FRAME_INTERVAL_US=0 – uncapped, periodicity {periodicity}, flags {flags:#x})"),
        Some(ov) => {
            let eff = effective_clock_interval_us(len_us, Some(ov)).unwrap_or(0);
            crate::xlog_info!("D3D12.X: frame interval {len_us} µs requested → {eff} µs effective (XCL_FRAME_INTERVAL_US={ov}, periodicity {periodicity}, flags {flags:#x})");
        }
        None => crate::xlog_info!("D3D12.X: frame interval {len_us} µs (periodicity {periodicity}, flags {flags:#x})"),
    });
    SetEvent(wake());
    S_FALSE
}

fn is_writable_protect(protect: u32) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_READWRITE, PAGE_WRITECOPY};
    matches!(protect & 0xFF, PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)
}

/// Fast-path working-set check for `[p, end)`: returns `Some(true)` if every page is resident and
/// writable, `Some(false)` if any resident page is non-writable/guarded, or `None` to fall back to `VirtualQuery`.
fn resident_and_writable(p: usize, end: usize) -> Option<bool> {
    use windows_sys::Win32::System::Memory::PAGE_GUARD;
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    const MAX_PAGES: usize = 16;
    let first = p >> 12;
    let pages = ((end - 1) >> 12) - first + 1;
    if pages > MAX_PAGES {
        return None;
    }
    let mut ws: [PSAPI_WORKING_SET_EX_INFORMATION; MAX_PAGES] = unsafe { std::mem::zeroed() };
    for (k, e) in ws[..pages].iter_mut().enumerate() {
        e.VirtualAddress = ((first + k) << 12) as *mut std::ffi::c_void;
    }
    let size = (pages * std::mem::size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32;
    if unsafe { K32QueryWorkingSetEx(GetCurrentProcess(), ws.as_mut_ptr() as *mut std::ffi::c_void, size) } == 0 {
        return None;
    }
    let mut all_resident = true;
    for e in &ws[..pages] {
        let flags = unsafe { e.VirtualAttributes.Flags };
        let protect = ((flags >> 4) & 0x7FF) as u32;
        if flags & 1 != 0 && protect != 0 {
            if (protect & PAGE_GUARD) != 0 || !is_writable_protect(protect) {
                return Some(false);
            }
        } else {
            all_resident = false;
        }
    }
    all_resident.then_some(true)
}

/// Whether `[p, p + len)` is committed, unguarded, and writable without touching any page.
fn writable(p: usize, len: usize) -> bool {
    writable_with_uma(crate::gpu::uma::get(), p, len)
}

fn writable_with_uma(uma: Option<&crate::gpu::uma::Uma>, p: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD};
    if p == 0 || len == 0 || len > (256 << 20) {
        return false;
    }
    let Some(end) = p.checked_add(len) else { return false };
    // A committed coherent UMA window is PAGE_READWRITE as a whole (`uma.rs`, `kernelx.rs`):
    // short-circuit before `K32QueryWorkingSetEx`/`VirtualQuery` just like `gpu::shader::readable`
    // so non-resident pages inside a 4 GiB coherent window never trigger a slow `VirtualQuery` walk.
    if uma.is_some_and(|u| u.in_committed_window(p, len)) {
        return true;
    }
    if let Some(ok) = resident_and_writable(p, end) {
        return ok;
    }
    let mut cur = p;
    while cur < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = unsafe { VirtualQuery(cur as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
        if n == 0 || mbi.State != MEM_COMMIT || (mbi.Protect & PAGE_GUARD) != 0 || !is_writable_protect(mbi.Protect) {
            return false;
        }
        let Some(next) = (mbi.BaseAddress as usize).checked_add(mbi.RegionSize) else { return false };
        if next <= cur {
            return false;
        }
        cur = next;
    }
    true
}

/// Stack-allocated ancillary handles (at most 63, matching `MAXIMUM_WAIT_OBJECTS - 1`).
struct AncillaryHandles {
    handles: [HANDLE; 63],
    len: usize,
}

impl std::ops::Deref for AncillaryHandles {
    type Target = [HANDLE];
    fn deref(&self) -> &[HANDLE] {
        &self.handles[..self.len]
    }
}

impl IntoIterator for AncillaryHandles {
    type Item = HANDLE;
    type IntoIter = std::iter::Take<std::array::IntoIter<HANDLE, 63>>;
    fn into_iter(self) -> Self::IntoIter {
        self.handles.into_iter().take(self.len)
    }
}

/// the handles of an ancillary block `{ u32 count; HANDLE* list @8; i32* signaled @0x10 }`
unsafe fn ancillary(block: *const u8) -> Result<AncillaryHandles, HRESULT> {
    let mut out = AncillaryHandles { handles: [std::ptr::null_mut(); 63], len: 0 };
    if block.is_null() {
        return Ok(out);
    }
    if !crate::gpu::shader::readable(block as usize, 16) {
        return Err(E_INVALIDARG);
    }
    let count = (block as *const u32).read_unaligned() as usize;
    if count == 0 {
        return Ok(out);
    }
    let list = (block.add(8) as *const usize).read_unaligned();
    if count > 63 || !crate::gpu::shader::readable(list, count * 8) {
        return Err(E_INVALIDARG);
    }
    for (i, slot) in out.handles[..count].iter_mut().enumerate() {
        *slot = (list as *const HANDLE).add(i).read_unaligned();
    }
    out.len = count;
    Ok(out)
}

unsafe fn note_signaled(block: *const u8, index: i32) {
    if block.is_null() || !crate::gpu::shader::readable(block as usize, 0x18) {
        return;
    }
    let out = (block.add(0x10) as *const usize).read_unaligned();
    if writable(out, 4) {
        (out as *mut i32).write_unaligned(index);
    }
}

/// device #79 ScheduleFrameEventX(type, offset µs, ancillary, flags)
pub unsafe extern "system" fn schedule_frame_event_x(_this: *mut std::ffi::c_void, ty: u32, offset_us: u32, block: *const u8, flags: u32) -> HRESULT {
    trace_calls(format_args!("ScheduleFrameEventX(type {ty}, offset {offset_us} µs, block {block:p}, flags {flags:#x})"));
    let unschedule = flags & UNSCHEDULE != 0;
    let invalid = with(|c| c.interval_us == 0 || ty as usize >= TYPES || (!c.active && ty != 0) || (unschedule && (ty == 0 || !c.kinds[0].scheduled)));
    if invalid {
        return E_INVALIDARG;
    }
    let handles = match ancillary(block) {
        Ok(h) => h,
        Err(e) => return e,
    };
    let mut dups = Vec::with_capacity(handles.len());
    if !unschedule {
        for h in handles {
            let mut d: HANDLE = std::ptr::null_mut();
            if DuplicateHandle(GetCurrentProcess(), h, GetCurrentProcess(), &mut d, 0, 0, DUPLICATE_SAME_ACCESS) == 0 {
                dups.into_iter().for_each(|d| {
                    CloseHandle(d);
                });
                return E_FAIL;
            }
            dups.push(d);
        }
    }
    let hr = with(|c| {
        let mut hr = S_FALSE;
        if !c.active {
            c.active = true;
            c.base_qpc = qpc();
            hr = STARTED_CLOCK;
        }
        let interval = c.interval_us.max(1);
        let k = &mut c.kinds[ty as usize];
        // the previous handles close with their last holder (the clock thread may be signalling them)
        k.extra = Arc::new(Handles(dups));
        if unschedule {
            k.scheduled = false;
        } else {
            k.offset_us = offset_us % interval;
            k.scheduled = true;
        }
        c.generation += 1;
        hr
    });
    if !THREAD.swap(true, Ordering::SeqCst) {
        let _ = std::thread::Builder::new().name("xcl-frame-clock".into()).spawn(clock_thread);
    }
    SetEvent(wake());
    hr
}

/// `XCL_ACTIVITY`: WaitFrameEventX calls, their timeouts, GetFrameStatisticsX calls, and those
/// that found no flip yet
static ACTIVITY: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

pub fn activity() -> [u64; 4] {
    ACTIVITY.each_ref().map(|a| a.load(Ordering::Relaxed))
}

/// device #80 WaitFrameEventX(type, timeout ms, ancillary, flags, token in/out)
/// `XCL_TRACE_FRAME_EVENTS`: the title's frame-event calls (the first 40)
fn trace_calls(what: std::fmt::Arguments) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static N: AtomicU64 = AtomicU64::new(0);
    if *ON.get_or_init(|| std::env::var_os("XCL_TRACE_FRAME_EVENTS").is_some()) && N.fetch_add(1, Ordering::Relaxed) < 40 {
        crate::xlog_info!("frame events: {what}");
    }
}

pub unsafe extern "system" fn wait_frame_event_x(_this: *mut std::ffi::c_void, ty: u32, timeout_ms: u32, block: *const u8, flags: u32, token: *mut u64) -> HRESULT {
    trace_calls(format_args!("WaitFrameEventX(type {ty}, timeout {timeout_ms} ms, block {block:p}, flags {flags:#x}, token in {:#x})", if token.is_null() { 0 } else { *token }));
    let scheduled = with(|c| {
        ((ty as usize) < TYPES && c.kinds[ty as usize].scheduled)
            .then(|| (c.kinds[ty as usize].internal as usize, (interval_override() == Some(0)).then(|| Arc::clone(&c.kinds[ty as usize].extra))))
    });
    let Some((internal, scheduled_extra)) = scheduled else { return E_INVALIDARG };
    let token_in = if token.is_null() { 0 } else { *token };
    if ty == 0 && token_in != 0 && (!ISSUED.lock().unwrap().iter().any(|e| e.0 == token_in) || flags & TOKEN_ON_TIMEOUT != 0) {
        return E_INVALIDARG;
    }
    let extra = match ancillary(block) {
        Ok(h) => h,
        Err(e) => return e,
    };
    let (hr, index, origin_now) = if let Some(scheduled_extra) = scheduled_extra {
        for &h in &scheduled_extra.0 {
            if SetEvent(h) == 0 {
                ReleaseSemaphore(h, 1, std::ptr::null_mut());
            }
        }
        let (hr, index) = if !extra.is_empty() && (flags & WAIT_ALL) != 0 {
            let r = WaitForMultipleObjects(extra.len() as u32, extra.as_ptr(), 1, timeout_ms);
            ACTIVITY[0].fetch_add(1, Ordering::Relaxed);
            if r == WAIT_TIMEOUT {
                ACTIVITY[1].fetch_add(1, Ordering::Relaxed);
                (S_FALSE, -1)
            } else if ((r - WAIT_OBJECT_0) as usize) < extra.len() {
                (S_OK, -2)
            } else {
                (E_FAIL, -1)
            }
        } else {
            ACTIVITY[0].fetch_add(1, Ordering::Relaxed);
            (S_OK, -2)
        };
        let now = qpc();
        if ty == 0 && hr == S_OK {
            LAST_ORIGIN.store(now, Ordering::Relaxed);
        }
        (hr, index, Some(now))
    } else {
        let mut hs = [std::ptr::null_mut(); 64];
        hs[0] = internal as HANDLE;
        hs[1..1 + extra.len()].copy_from_slice(&extra);
        let count = 1 + extra.len();
        let r = WaitForMultipleObjects(count as u32, hs.as_ptr(), (flags & WAIT_ALL != 0) as i32, timeout_ms);
        ACTIVITY[0].fetch_add(1, Ordering::Relaxed);
        if r == WAIT_TIMEOUT {
            ACTIVITY[1].fetch_add(1, Ordering::Relaxed);
        }
        let (hr, index) = if r == WAIT_TIMEOUT {
            (S_FALSE, -1)
        } else if ((r - WAIT_OBJECT_0) as usize) < count {
            // -2: the frame event itself, else the index of the ancillary handle
            (S_OK, if r == WAIT_OBJECT_0 { -2 } else { (r - WAIT_OBJECT_0) as i32 - 1 })
        } else {
            (E_FAIL, -1)
        };
        (hr, index, None)
    };
    note_signaled(block, index);
    if hr == E_FAIL {
        return hr;
    }
    let issue = index == -2 || (index == -1 && flags & TOKEN_ON_TIMEOUT != 0);
    if ty == 0 && issue {
        // the origin this wait consumed (a timeout with TOKEN_ON_TIMEOUT: now)
        let origin = origin_now
            .or_else(|| if index == -2 { take_origin(&ORIGINS) } else { None })
            .unwrap_or_else(qpc);
        let mut issued = ISSUED.lock().unwrap();
        if token_in != 0 {
            if let Some(e) = issued.iter_mut().find(|e| e.0 == token_in) {
                e.1 = origin;
            }
        } else {
            let mut t = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed) & TOKEN_MASK;
            if t == 0 {
                t = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed) & TOKEN_MASK;
            }
            if issued.len() >= KNOWN_TOKENS {
                issued.pop_front();
            }
            issued.push_back((t, origin));
            if !token.is_null() {
                *token = t;
            }
        }
    }
    hr
}

/// `PresentX` flipped the frame of `token`.
pub fn mark_flipped(token: u64) {
    if token == 0 {
        return;
    }
    // the frame's own origin (a token XCL did not issue: the latest origin)
    let origin = ISSUED.lock().unwrap().iter().rev().find(|e| e.0 == token).map(|e| e.1).unwrap_or_else(|| LAST_ORIGIN.load(Ordering::Relaxed));
    let mut f = FLIPPED.lock().unwrap();
    f.push_back((token, origin, qpc()));
    while f.len() > FLIPPED_KEEP {
        f.pop_front();
    }
}

/// device #81 GetFrameStatisticsX(token, type, count in/out, statistics): one record
/// `{ u32 type; …; i64 origin qpc @0x10; i64 flip qpc @0x18 }` once the frame was flipped.
pub unsafe extern "system" fn frame_statistics_x(_this: *mut std::ffi::c_void, token: u64, ty: u32, count: *mut u32, stats: *mut u8) -> HRESULT {
    if count.is_null() || stats.is_null() {
        return E_INVALIDARG;
    }
    let flipped = FLIPPED.lock().unwrap().iter().rev().find(|f| f.0 == token).map(|f| (f.1, f.2));
    ACTIVITY[2].fetch_add(1, Ordering::Relaxed);
    let Some((origin, flip)) = flipped else {
        ACTIVITY[3].fetch_add(1, Ordering::Relaxed);
        *count = 0;
        return E_FAIL;
    };
    if *count < 1 {
        return E_INVALIDARG;
    }
    *count = 1;
    (stats as *mut u32).write_unaligned(ty);
    (stats.add(0x10) as *mut i64).write_unaligned(if origin != 0 && origin <= flip { origin } else { flip });
    (stats.add(0x18) as *mut i64).write_unaligned(flip);
    S_OK
}

/// sleep until `deadline` (qpc); false: woken by a change of the schedule
unsafe fn sleep_until(timer: HANDLE, deadline: i64, allow_spin: bool) -> bool {
    const SPIN_US: i64 = 600;
    loop {
        let left = deadline - qpc();
        if left <= 0 {
            return true;
        }
        let left_us = left * 1_000_000 / qpf();
        if !allow_spin || left_us > SPIN_US {
            let wait_us = if allow_spin { left_us - SPIN_US } else { left_us.max(1) };
            let due = -(wait_us * 10); // relative, 100 ns units
            SetWaitableTimer(timer, &due, 0, None, std::ptr::null(), 0);
            let hs = [timer, wake()];
            if WaitForMultipleObjects(2, hs.as_ptr(), 0, INFINITE) == WAIT_OBJECT_0 + 1 {
                return false;
            }
        } else {
            std::hint::spin_loop();
        }
    }
}

fn clock_thread() {
    unsafe {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        let mut timer = CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if timer.is_null() {
            timer = CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), 0, TIMER_ALL_ACCESS);
        }
        let mut last = [i64::MIN / 2; TYPES];
        loop {
            // the next events of the current schedule: every type due at the earliest deadline
            // (types at the same point of the frame fire together – a retail game schedules the
            // origin and type 2 both at offset 0 and waits on type 2; picking one, the origin
            // always won and type 2 never came)
            let plan = with(|c| {
                let effective_us = if c.active { effective_clock_interval_us(c.interval_us, interval_override()) } else { None }?;
                let interval = ticks(effective_us as i64);
                let now = qpc();
                let due: Vec<(i64, usize, usize, Arc<Handles>)> = c
                    .kinds
                    .iter()
                    .enumerate()
                    .filter(|(_, k)| k.scheduled)
                    .map(|(i, k)| {
                        let offset_us = k.offset_us % effective_us.max(1);
                        let mut d = next_deadline(c.base_qpc, interval, ticks(offset_us as i64), now - 1);
                        if d - last[i] < interval / 2 {
                            d += interval; // this point of the frame already fired
                        }
                        (d, i, k.internal as usize, Arc::clone(&k.extra))
                    })
                    .collect();
                let first = due.iter().map(|e| e.0).min()?;
                Some((c.generation, first, effective_us > 2000, due.into_iter().filter(|e| e.0 == first).collect::<Vec<_>>()))
            });
            let Some((generation, deadline, allow_spin, due)) = plan else {
                WaitForMultipleObjects(1, &wake(), 0, INFINITE);
                continue;
            };
            if !sleep_until(timer, deadline, allow_spin) || !with(|c| c.generation == generation) {
                continue;
            }
            for (_, ty, internal, extra) in due {
                last[ty] = deadline;
                if ty == 0 {
                    let now = qpc();
                    LAST_ORIGIN.store(now, Ordering::Relaxed);
                    publish_origin(&ORIGINS, internal as HANDLE, now, || {});
                } else {
                    SetEvent(internal as HANDLE);
                }
                for &h in &extra.0 {
                    if SetEvent(h) == 0 {
                        ReleaseSemaphore(h, 1, std::ptr::null_mut());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the clock is one per process: the tests that run it take turns
    static SERIAL: Mutex<()> = Mutex::new(());

    /// A wait the origin semaphore wakes takes the origin that woke it, also when it runs between
    /// the clock's release and the queueing (a pause there, `after_release`): finding the queue
    /// empty, it would take a newer QPC, and the next wait the older origin – origins going
    /// backwards (`statistics_name_each_frames_origin_...`, "each frame its own origin").
    #[test]
    fn a_woken_wait_takes_the_origin_that_woke_it() {
        let q = Mutex::new(VecDeque::new());
        unsafe {
            let sem = CreateSemaphoreW(std::ptr::null(), 0, ORIGIN_SEMAPHORE_MAX, std::ptr::null()) as usize;
            assert_ne!(sem, 0);
            std::thread::scope(|s| {
                let q = &q;
                let waiter = s.spawn(move || {
                    let r = WaitForMultipleObjects(1, &(sem as HANDLE), 0, 5000);
                    (r == WAIT_OBJECT_0, take_origin(q))
                });
                assert!(publish_origin(q, sem as HANDLE, 42, || std::thread::sleep(std::time::Duration::from_millis(100))));
                assert_eq!(waiter.join().unwrap(), (true, Some(42)), "the woken wait's origin");
            });
            // two origins, two waits: in order
            assert!(publish_origin(&q, sem as HANDLE, 43, || {}) && publish_origin(&q, sem as HANDLE, 44, || {}));
            assert_eq!((take_origin(&q), take_origin(&q), take_origin(&q)), (Some(43), Some(44), None));
            CloseHandle(sem as HANDLE);
        }
    }

    #[test]
    fn interval_override_parsing_and_clamping() {
        assert_eq!(parse_interval_override(None), None);
        assert_eq!(parse_interval_override(Some("")), None);
        assert_eq!(parse_interval_override(Some("bad")), None);
        assert_eq!(parse_interval_override(Some("0")), Some(0));
        assert_eq!(parse_interval_override(Some("uncapped")), Some(0));
        assert_eq!(parse_interval_override(Some("OFF")), Some(0));
        assert_eq!(parse_interval_override(Some("8333")), Some(8333));

        assert_eq!(effective_clock_interval_us(0, Some(8333)), None);
        assert_eq!(effective_clock_interval_us(33333, None), Some(33333));
        assert_eq!(effective_clock_interval_us(33333, Some(0)), None);
        assert_eq!(effective_clock_interval_us(33333, Some(8333)), Some(8333));
        assert_eq!(effective_clock_interval_us(16667, Some(33333)), Some(16667));
        assert_eq!(effective_clock_interval_us(33333, Some(200)), Some(1000));
    }

    #[test]
    fn console_intervals_and_deadlines() {
        assert!(interval_valid(16667, false) && interval_valid(16666 + 16, false) && interval_valid(33333, false));
        assert!(!interval_valid(16000, false) && !interval_valid(8333, false) && interval_valid(8333, true));
        assert!(interval_valid(16666, false) && interval_valid(66666, false) && interval_valid(33333 - 34, false));
        assert_eq!(next_deadline(100, 10, 3, 50), 103);
        assert_eq!(next_deadline(100, 10, 3, 103), 113);
        assert_eq!(next_deadline(100, 10, 3, 120), 123);
    }

    /// The origin event of a 60 Hz clock: waits return about one interval apart with fresh
    /// tokens; a flipped token has statistics, an unknown one does not.
    #[test]
    fn origin_events_issue_tokens_and_flips_have_statistics() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let dev = std::ptr::null_mut();
            assert_eq!(set_frame_interval_x(dev, std::ptr::null_mut(), 16667, 1, 0), S_FALSE);
            assert_eq!(wait_frame_event_x(dev, 0, 10, std::ptr::null(), 0, std::ptr::null_mut()), E_INVALIDARG, "not scheduled");
            let hr = schedule_frame_event_x(dev, 0, 0, std::ptr::null(), 0);
            assert!(hr == STARTED_CLOCK || hr == S_FALSE, "{hr:#x}");
            let mut tokens = Vec::new();
            let t0 = std::time::Instant::now();
            for _ in 0..4 {
                let mut t = 0u64;
                assert_eq!(wait_frame_event_x(dev, 0, 200, std::ptr::null(), 0, &mut t), S_OK);
                tokens.push(t);
            }
            let ms = t0.elapsed().as_millis();
            assert!((25..=200).contains(&ms), "4 origins at 60 Hz took {ms} ms");
            assert!(tokens.windows(2).all(|w| w[1] > w[0]), "{tokens:?}");
            mark_flipped(tokens[2]);
            let (mut n, mut st) = (1u32, [0u8; 0x40]);
            assert_eq!(frame_statistics_x(dev, tokens[2], 0, &mut n, st.as_mut_ptr()), S_OK);
            let (origin, flip) = (i64::from_le_bytes(st[0x10..0x18].try_into().unwrap()), i64::from_le_bytes(st[0x18..0x20].try_into().unwrap()));
            assert!(n == 1 && origin > 0 && origin <= flip);
            assert_eq!(frame_statistics_x(dev, 0xDEAD, 0, &mut n, st.as_mut_ptr()), E_FAIL);
            assert_eq!(set_frame_interval_x(dev, std::ptr::null_mut(), 16000, 1, 0), E_INVALIDARG);
            assert_eq!(set_frame_interval_x(dev, std::ptr::null_mut(), 0, 0, 0), S_FALSE, "stops the clock");
        }
    }

    /// Each flipped frame reports its own origin, not the latest one (frames in flight), and the
    /// ancillary handles of a frame event rescheduled while the clock fires stay valid until
    /// the clock thread is done with them (#103).
    #[test]
    fn statistics_name_each_frames_origin_and_rescheduled_handles_outlive_the_clock_tick() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let dev = std::ptr::null_mut();
            assert_eq!(set_frame_interval_x(dev, std::ptr::null_mut(), 16667, 1, 0), S_FALSE);
            let hr = schedule_frame_event_x(dev, 0, 0, std::ptr::null(), 0);
            assert!(hr == STARTED_CLOCK || hr == S_FALSE, "{hr:#x}");
            // three frames in flight, flipped after the third origin
            let mut tokens = [0u64; 3];
            for t in tokens.iter_mut() {
                assert_eq!(wait_frame_event_x(dev, 0, 200, std::ptr::null(), 0, t), S_OK);
            }
            for &t in &tokens {
                mark_flipped(t);
            }
            let mut origins = Vec::new();
            for &t in &tokens {
                let (mut n, mut st) = (1u32, [0u8; 0x40]);
                assert_eq!(frame_statistics_x(dev, t, 0, &mut n, st.as_mut_ptr()), S_OK);
                let origin = i64::from_le_bytes(st[0x10..0x18].try_into().unwrap());
                let flip = i64::from_le_bytes(st[0x18..0x20].try_into().unwrap());
                assert!(origin <= flip);
                origins.push(origin);
            }
            assert!(origins[0] < origins[1] && origins[1] < origins[2], "each frame its own origin: {origins:?}");
            let spread = (origins[2] - origins[0]) * 1_000_000 / qpf();
            assert!((20_000..=100_000).contains(&spread), "two intervals apart: {spread} µs");

            // a type-1 event with an ancillary event, scheduled and unscheduled while the clock
            // fires it (offset 0: at the origin) – the thread must never signal a closed handle
            let ev = CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
            let list = [ev];
            let mut signaled = 0i32;
            let mut block = [0u8; 0x18];
            block[..4].copy_from_slice(&1u32.to_le_bytes());
            block[8..16].copy_from_slice(&(list.as_ptr() as usize).to_le_bytes());
            block[16..24].copy_from_slice(&(&mut signaled as *mut i32 as usize).to_le_bytes());
            let t0 = std::time::Instant::now();
            while t0.elapsed() < std::time::Duration::from_millis(120) {
                assert_eq!(schedule_frame_event_x(dev, 1, 0, block.as_ptr(), 0), S_FALSE);
                std::thread::sleep(std::time::Duration::from_micros(300));
                assert_eq!(schedule_frame_event_x(dev, 1, 0, std::ptr::null(), UNSCHEDULE), S_FALSE);
            }
            CloseHandle(ev);
            // the clock still runs
            let mut t = 0u64;
            assert_eq!(wait_frame_event_x(dev, 0, 200, std::ptr::null(), 0, &mut t), S_OK);
            assert_eq!(set_frame_interval_x(dev, std::ptr::null_mut(), 0, 0, 0), S_FALSE, "stops the clock");
        }
    }

    #[test]
    fn note_signaled_skips_readonly_guard_and_cross_page_targets() {
        use windows_sys::Win32::System::Memory::{
            VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_RELEASE,
            MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        };
        struct VmGuard(*mut std::ffi::c_void);
        impl Drop for VmGuard {
            fn drop(&mut self) {
                if !self.0.is_null() { unsafe { VirtualFree(self.0, 0, MEM_RELEASE) }; }
            }
        }
        let mk_block = |out_ptr: usize| -> [u8; 0x18] {
            let mut b = [0u8; 0x18];
            b[0x10..0x18].copy_from_slice(&out_ptr.to_le_bytes());
            b
        };
        unsafe {
            assert!(!writable(0, 4) && !writable(0x1000, 0) && !writable(usize::MAX - 1, 4));
            let base = VirtualAlloc(std::ptr::null(), 0x5000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            assert!(!base.is_null());
            let _vm = VmGuard(base);
            let res_only = VirtualAlloc(std::ptr::null(), 0x1000, MEM_RESERVE, PAGE_READWRITE);
            assert!(!res_only.is_null());
            let _vm_res = VmGuard(res_only);
            assert!(!writable(res_only as usize, 4));

            let p0 = base as *mut u8;
            let p1 = p0.add(0x1000); // RW -> RW straddle target (untouched & resident)
            let p2 = p0.add(0x2000); // PAGE_READONLY
            let p3 = p0.add(0x3000); // PAGE_READWRITE | PAGE_GUARD
            let p4 = p0.add(0x4000); // PAGE_EXECUTE_READ -> PAGE_NOACCESS
            assert!(writable(p0 as usize, 4) && writable(p0 as usize + 0xFFE, 4));
            (p1.add(0xFFE) as *mut u16).write_unaligned(0xBEEF);
            (p2 as *mut i32).write_unaligned(0x55AA_1234);
            let mut old = 0u32;
            assert_ne!(VirtualProtect(p2 as *const _, 0x1000, PAGE_READONLY, &mut old), 0);
            assert_ne!(VirtualProtect(p3 as *const _, 0x1000, PAGE_READWRITE | PAGE_GUARD, &mut old), 0);
            assert_ne!(VirtualProtect(p4 as *const _, 0x1000, PAGE_EXECUTE_READ, &mut old), 0);

            assert!(!writable(p2 as usize, 4), "PAGE_READONLY rejected");
            let _ = (p2 as *const i32).read_volatile(); // ensure resident in working set
            assert!(!writable(p2 as usize, 4), "resident PAGE_READONLY rejected");
            assert!(!writable(p1 as usize + 0xFFE, 4), "RW -> RO cross-page straddle rejected");
            assert!(!writable(p3 as usize, 4), "PAGE_GUARD rejected");
            assert!(!writable(p4 as usize, 4), "PAGE_EXECUTE_READ rejected");
            assert_ne!(VirtualProtect(p4 as *const _, 0x1000, PAGE_NOACCESS, &mut old), 0);
            assert!(!writable(p4 as usize, 4), "PAGE_NOACCESS rejected");

            // note_signaled writes to writable target and safely skips RO / straddle / GUARD.
            let mut ok_val = -1i32;
            note_signaled(mk_block(&mut ok_val as *mut i32 as usize).as_ptr(), 7);
            assert_eq!(ok_val, 7);
            note_signaled(mk_block(p2 as usize).as_ptr(), 9);
            assert_eq!((p2 as *const i32).read_unaligned(), 0x55AA_1234);
            note_signaled(mk_block(p1 as usize + 0xFFE).as_ptr(), 11);
            assert_eq!((p1.add(0xFFE) as *const u16).read_unaligned(), 0xBEEF);
            note_signaled(mk_block(p3 as usize).as_ptr(), 13);
            let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(VirtualQuery(p3 as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()), 0);
            assert_ne!(mbi.Protect & PAGE_GUARD, 0, "PAGE_GUARD must remain intact");
        }
    }

    #[test]
    fn writable_uses_committed_window_shortcut_and_falls_through_on_boundary() {
        use windows_sys::Win32::System::Memory::{
            VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_READWRITE,
        };
        let u = crate::gpu::uma::Uma::with_windows(1, 0x2_0000).expect("local committed Uma window");
        let (win_base, win_len) = u.window(0);
        assert_eq!(win_len, 0x2_0000);

        // Untouched page in a committed coherent window is non-resident, yet fast-path writable without faulting it in.
        assert_eq!(resident_and_writable(win_base, win_base + 4), None);
        assert!(writable_with_uma(Some(&u), win_base, 4));
        assert_eq!(resident_and_writable(win_base, win_base + 4), None, "fast path must not touch the page");
        assert!(writable_with_uma(Some(&u), win_base + win_len - 4, 4));

        // Guard checks still reject zero/null/oversized/overflowing spans even when UMA is present.
        assert!(!writable_with_uma(Some(&u), 0, 4));
        assert!(!writable_with_uma(Some(&u), win_base, 0));
        assert!(!writable_with_uma(Some(&u), win_base, (256 << 20) + 1));
        assert!(!writable_with_uma(Some(&u), usize::MAX - 1, 4));

        // Cross-window boundary straddle and past-the-end addresses miss `in_committed_window` and fall through.
        assert!(!u.in_committed_window(win_base + win_len - 2, 4));
        assert!(!writable_with_uma(Some(&u), win_base + win_len - 2, 4));
        assert!(!writable_with_uma(Some(&u), win_base + win_len, 4));

        // Prove `Some(&u)` takes the `in_committed_window` shortcut before `resident_and_writable`/`VirtualQuery`:
        // on a temporary `PAGE_GUARD` probe page inside this local `Uma`, `None` rejects via `VirtualQuery`
        // while `Some(&u)` short-circuits without touching the guard page.
        unsafe {
            let probe = win_base + 0x1000;
            let mut old = 0u32;
            assert_ne!(VirtualProtect(probe as *const _, 0x1000, PAGE_READWRITE | PAGE_GUARD, &mut old), 0);
            assert!(!writable_with_uma(None, probe, 4), "fallback path must reject PAGE_GUARD");
            assert!(writable_with_uma(Some(&u), probe, 4), "committed-window fast path must short-circuit");
            let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(VirtualQuery(probe as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()), 0);
            assert_ne!(mbi.Protect & PAGE_GUARD, 0, "shortcut must not touch PAGE_GUARD");
            assert_ne!(VirtualProtect(probe as *const _, 0x1000, old, &mut old), 0);
        }
    }

    #[test]
    fn ancillary_stack_handles_boundary_cases() {
        unsafe {
            assert!(ancillary(std::ptr::null()).unwrap().is_empty());
            let mut block = [0u8; 0x18];
            assert!(ancillary(block.as_ptr()).unwrap().is_empty());

            let list: [HANDLE; 63] = std::array::from_fn(|i| ((i + 1) * 4) as HANDLE);
            block[..4].copy_from_slice(&63u32.to_le_bytes());
            block[8..16].copy_from_slice(&(list.as_ptr() as usize).to_le_bytes());
            let got = ancillary(block.as_ptr()).unwrap();
            assert_eq!(&*got, &list[..]);
            assert_eq!(got.into_iter().collect::<Vec<_>>(), list);

            for bad in [64u32, u32::MAX] {
                block[..4].copy_from_slice(&bad.to_le_bytes());
                assert_eq!(ancillary(block.as_ptr()).err(), Some(E_INVALIDARG));
            }
            block[..4].copy_from_slice(&1u32.to_le_bytes());
            block[8..16].copy_from_slice(&0x1000usize.to_le_bytes());
            assert_eq!(ancillary(block.as_ptr()).err(), Some(E_INVALIDARG));
        }
    }

    #[test]
    fn issued_and_flipped_ring_buffers_evict_oldest_in_order() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let base_tok = 0x7000_0000_0000u64;
        {
            let mut issued = ISSUED.lock().unwrap();
            issued.clear();
            for i in 0..(KNOWN_TOKENS + 10) {
                if issued.len() >= KNOWN_TOKENS { issued.pop_front(); }
                issued.push_back((base_tok + i as u64, (i as i64) + 1));
            }
            assert_eq!(issued.len(), KNOWN_TOKENS);
            assert_eq!(issued.front().unwrap().0, base_tok + 10);
            assert_eq!(issued.back().unwrap().0, base_tok + (KNOWN_TOKENS + 9) as u64);
        }
        FLIPPED.lock().unwrap().clear();
        for i in 0..(FLIPPED_KEEP + 10) {
            mark_flipped(base_tok + i as u64);
        }
        assert_eq!(FLIPPED.lock().unwrap().len(), FLIPPED_KEEP);
        let mut st = [0u8; 0x40];
        unsafe {
            for i in 0..10 {
                let mut n = 1u32;
                assert_eq!(frame_statistics_x(std::ptr::null_mut(), base_tok + i as u64, 0, &mut n, st.as_mut_ptr()), E_FAIL);
            }
            for i in 10..(FLIPPED_KEEP + 10) {
                let mut n = 1u32;
                assert_eq!(frame_statistics_x(std::ptr::null_mut(), base_tok + i as u64, 0, &mut n, st.as_mut_ptr()), S_OK);
            }
        }
    }
}

