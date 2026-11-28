//! `XCL_TRACE_WAITS=1`: which objects the title's threads wait on, and who signals them.
//!
//! The title's event and semaphore functions and its waits (imports of the kernel family,
//! routed to kernelx) go through these wrappers. Each wrapper calls Windows and remembers, per
//! handle, where in the title it was made and when and from where it was last set or released;
//! per thread, what it waits on and since when. A thread dump (`XCL_THREAD_DUMP_AT`, a crash)
//! then lists the title's threads that have waited 2 s or more: a title stalled on its own
//! objects (#505) names the object and the code that was meant to signal it.
//!
//! Off by default: without the variable the wrappers are not in the import table at all.
//! Polls (a timeout of 0) are not recorded as waits.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::Threading as th;

/// waits shorter than this are not listed in a dump
const LONG_WAIT_SECS: f64 = 2.0;
/// handles listed for one wait (WaitForMultipleObjects takes up to 64)
const MAX_LISTED: usize = 8;

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_TRACE_WAITS").is_ok_and(|v| !matches!(v.trim(), "" | "0")))
}

pub fn exports() -> Vec<(&'static str, usize)> {
    crate::xlog_info!("XCL_TRACE_WAITS: the title's events, semaphores and waits are traced (listed in thread dumps)");
    crate::hle_exports![
        CreateEventA,
        CreateEventW,
        CreateEventExW,
        CreateSemaphoreW,
        CreateSemaphoreExW,
        SetEvent,
        ReleaseSemaphore,
        CloseHandle,
        WaitForSingleObject,
        WaitForSingleObjectEx,
        WaitForMultipleObjects,
        WaitForMultipleObjectsEx,
    ]
}

/// What the title made a handle as, and who last signaled it.
#[derive(Clone, Default)]
struct Object {
    /// "event", "manual-reset event", "semaphore"; empty: not made through these imports
    kind: &'static str,
    name: Option<String>,
    made_by: usize,
    made_at: f64,
    signals: u64,
    signaled_by: usize,
    signaled_at: f64,
}

#[derive(Clone)]
struct Wait {
    api: &'static str,
    handles: Vec<usize>,
    count: usize,
    all: bool,
    timeout: u32,
    caller: usize,
    since: f64,
}

fn objects() -> &'static Mutex<HashMap<usize, Object>> {
    static M: OnceLock<Mutex<HashMap<usize, Object>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

/// thread id → its wait
fn waits() -> &'static Mutex<HashMap<u32, Wait>> {
    static M: OnceLock<Mutex<HashMap<u32, Wait>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

/// The first return address on the stack in the title's code (its exe or a game DLL), or the
/// wrapper's caller when none is (tests).
#[inline(never)]
fn caller() -> usize {
    use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
    let mut frames = [std::ptr::null_mut::<c_void>(); 6];
    let n = unsafe { RtlCaptureStackBackTrace(1, frames.len() as u32, frames.as_mut_ptr(), std::ptr::null_mut()) } as usize;
    let f: Vec<usize> = frames[..n].iter().map(|&p| p as usize).collect();
    f.iter().copied().find(|&a| crate::modules::find_mapped_by_addr(a).is_some()).or_else(|| f.get(2).copied()).unwrap_or(0)
}

fn made(h: HANDLE, kind: &'static str, name: Option<String>) {
    if h.is_null() {
        return;
    }
    let o = Object { kind, name, made_by: caller(), made_at: crate::log::uptime(), ..Default::default() };
    if let Ok(mut m) = objects().lock() {
        m.insert(h as usize, o);
    }
}

fn signaled(h: HANDLE) {
    let (by, at) = (caller(), crate::log::uptime());
    if let Ok(mut m) = objects().lock() {
        let o = m.entry(h as usize).or_default();
        o.signals += 1;
        o.signaled_by = by;
        o.signaled_at = at;
    }
}

/// The calling thread's wait, recorded until it is dropped.
struct Waiting(u32);

impl Drop for Waiting {
    fn drop(&mut self) {
        if let Ok(mut w) = waits().lock() {
            w.remove(&self.0);
        }
    }
}

fn waiting(api: &'static str, handles: &[HANDLE], all: bool, timeout: u32) -> Option<Waiting> {
    if timeout == 0 {
        return None;
    }
    let tid = unsafe { th::GetCurrentThreadId() };
    let w = Wait {
        api,
        handles: handles.iter().take(MAX_LISTED).map(|&h| h as usize).collect(),
        count: handles.len(),
        all,
        timeout,
        caller: caller(),
        since: crate::log::uptime(),
    };
    waits().lock().ok()?.insert(tid, w);
    Some(Waiting(tid))
}

unsafe fn wide_name(p: *const u16) -> Option<String> {
    if p.is_null() {
        return None;
    }
    let n = (0..).take_while(|&i| *p.add(i) != 0).count();
    Some(String::from_utf16_lossy(std::slice::from_raw_parts(p, n)))
}

unsafe fn ansi_name(p: *const u8) -> Option<String> {
    (!p.is_null()).then(|| std::ffi::CStr::from_ptr(p as *const _).to_string_lossy().into_owned())
}

fn event_kind(manual: bool) -> &'static str {
    if manual {
        "manual-reset event"
    } else {
        "event"
    }
}

// --- the wrappers: the title's imports ---------------------------------------------------

#[allow(non_snake_case)]
unsafe extern "system" fn CreateEventA(attr: *const c_void, manual: i32, initial: i32, name: *const u8) -> HANDLE {
    let h = th::CreateEventA(attr as _, manual, initial, name);
    made(h, event_kind(manual != 0), ansi_name(name));
    h
}

#[allow(non_snake_case)]
unsafe extern "system" fn CreateEventW(attr: *const c_void, manual: i32, initial: i32, name: *const u16) -> HANDLE {
    let h = th::CreateEventW(attr as _, manual, initial, name);
    made(h, event_kind(manual != 0), wide_name(name));
    h
}

#[allow(non_snake_case)]
unsafe extern "system" fn CreateEventExW(attr: *const c_void, name: *const u16, flags: u32, access: u32) -> HANDLE {
    let h = th::CreateEventExW(attr as _, name, flags, access);
    made(h, event_kind(flags & th::CREATE_EVENT_MANUAL_RESET != 0), wide_name(name));
    h
}

#[allow(non_snake_case)]
unsafe extern "system" fn CreateSemaphoreW(attr: *const c_void, initial: i32, max: i32, name: *const u16) -> HANDLE {
    let h = th::CreateSemaphoreW(attr as _, initial, max, name);
    made(h, "semaphore", wide_name(name));
    h
}

#[allow(non_snake_case)]
unsafe extern "system" fn CreateSemaphoreExW(attr: *const c_void, initial: i32, max: i32, name: *const u16, flags: u32, access: u32) -> HANDLE {
    let h = th::CreateSemaphoreExW(attr as _, initial, max, name, flags, access);
    made(h, "semaphore", wide_name(name));
    h
}

#[allow(non_snake_case)]
unsafe extern "system" fn SetEvent(h: HANDLE) -> i32 {
    signaled(h);
    th::SetEvent(h)
}

#[allow(non_snake_case)]
unsafe extern "system" fn ReleaseSemaphore(h: HANDLE, count: i32, previous: *mut i32) -> i32 {
    signaled(h);
    th::ReleaseSemaphore(h, count, previous)
}

#[allow(non_snake_case)]
unsafe extern "system" fn CloseHandle(h: HANDLE) -> i32 {
    if let Ok(mut m) = objects().lock() {
        m.remove(&(h as usize));
    }
    windows_sys::Win32::Foundation::CloseHandle(h)
}

#[allow(non_snake_case)]
unsafe extern "system" fn WaitForSingleObject(h: HANDLE, ms: u32) -> u32 {
    let _w = waiting("WaitForSingleObject", &[h], false, ms);
    th::WaitForSingleObject(h, ms)
}

#[allow(non_snake_case)]
unsafe extern "system" fn WaitForSingleObjectEx(h: HANDLE, ms: u32, alertable: i32) -> u32 {
    let _w = waiting("WaitForSingleObjectEx", &[h], false, ms);
    th::WaitForSingleObjectEx(h, ms, alertable)
}

#[allow(non_snake_case)]
unsafe extern "system" fn WaitForMultipleObjects(n: u32, hs: *const HANDLE, all: i32, ms: u32) -> u32 {
    let list = if hs.is_null() { &[][..] } else { std::slice::from_raw_parts(hs, n as usize) };
    let _w = waiting("WaitForMultipleObjects", list, all != 0, ms);
    th::WaitForMultipleObjects(n, hs, all, ms)
}

#[allow(non_snake_case)]
unsafe extern "system" fn WaitForMultipleObjectsEx(n: u32, hs: *const HANDLE, all: i32, ms: u32, alertable: i32) -> u32 {
    let list = if hs.is_null() { &[][..] } else { std::slice::from_raw_parts(hs, n as usize) };
    let _w = waiting("WaitForMultipleObjectsEx", list, all != 0, ms);
    th::WaitForMultipleObjectsEx(n, hs, all, ms, alertable)
}

// --- the dump -----------------------------------------------------------------------------

#[link(name = "ntdll", kind = "raw-dylib")]
extern "system" {
    fn NtQueryObject(h: HANDLE, class: u32, info: *mut c_void, len: u32, ret: *mut u32) -> i32;
}

/// The kernel's name for a handle's object type ("Event", "Thread", …).
fn object_type(h: usize) -> Option<String> {
    const OBJECT_TYPE_INFORMATION: u32 = 2;
    #[repr(C, align(8))]
    struct Buf([u8; 1024]);
    let mut buf = Buf([0; 1024]);
    let mut ret = 0u32;
    let st = unsafe { NtQueryObject(h as HANDLE, OBJECT_TYPE_INFORMATION, buf.0.as_mut_ptr() as *mut c_void, buf.0.len() as u32, &mut ret) };
    if st < 0 {
        return None;
    }
    // PUBLIC_OBJECT_TYPE_INFORMATION starts with the type name, a UNICODE_STRING
    let len = u16::from_ne_bytes([buf.0[0], buf.0[1]]) as usize / 2;
    let ptr = usize::from_ne_bytes(buf.0[8..16].try_into().ok()?) as *const u16;
    (!ptr.is_null()).then(|| String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(ptr, len) }))
}

fn describe(h: usize, o: Option<&Object>, kernel_type: impl Fn(usize) -> Option<String>) -> String {
    let at = crate::modules::describe_address;
    let mut s = format!("{h:#x} ");
    match o.filter(|o| !o.kind.is_empty()) {
        Some(o) => {
            s += o.kind;
            if let Some(n) = &o.name {
                s += &format!(" \"{n}\"");
            }
            s += &format!(" made at {:.1} s by {}", o.made_at, at(o.made_by));
        }
        None => s += &format!("{} not made through the traced imports", kernel_type(h).unwrap_or_else(|| "object".into())),
    }
    let verb = if o.is_some_and(|o| o.kind == "semaphore") { "released" } else { "set" };
    match o.filter(|o| o.signals > 0) {
        Some(o) => s += &format!(", last {verb} at {:.1} s by {} ({} times)", o.signaled_at, at(o.signaled_by), o.signals),
        None => s += &format!(", never {verb} by the title"),
    }
    s
}

/// The waits of `min_secs` or more as log lines (thread, how long, on what), oldest first.
fn waits_over(min_secs: f64) -> Vec<String> {
    let now = crate::log::uptime();
    // a dump at a crash: the locks may be held by a thread that will not come back
    let Ok(w) = waits().try_lock() else { return vec!["waits: the wait table is busy".into()] };
    let mut long: Vec<(u32, Wait)> = w.iter().filter(|(_, w)| now - w.since >= min_secs).map(|(&t, w)| (t, w.clone())).collect();
    drop(w);
    long.sort_by(|a, b| a.1.since.total_cmp(&b.1.since));
    let objs = objects().try_lock().ok();
    let mut out = Vec::new();
    for (tid, w) in long {
        let timeout = if w.timeout == u32::MAX { "no timeout".to_string() } else { format!("timeout {} ms", w.timeout) };
        let what = if w.count > 1 { format!(" ({} of {})", if w.all { "all" } else { "any" }, w.count) } else { String::new() };
        out.push(format!("thread {tid}: {:.1} s in {}{what} from {}, {timeout}, on:", now - w.since, w.api, crate::modules::describe_address(w.caller)));
        for &h in &w.handles {
            out.push(format!("    {}", describe(h, objs.as_ref().and_then(|m| m.get(&h)), object_type)));
        }
        if w.count > w.handles.len() {
            out.push(format!("    … and {} more", w.count - w.handles.len()));
        }
    }
    out
}

/// For a thread dump: the size of the tracer's own tables, then the title's threads that have
/// waited `LONG_WAIT_SECS` or more, what on, where the objects were made and who last
/// signaled them. Empty when off.
pub fn long_waits() -> Vec<String> {
    if !enabled() {
        return Vec::new();
    }
    let size = |n: Option<usize>| n.map_or_else(|| "busy".to_string(), |n| n.to_string());
    let handles = objects().try_lock().ok().map(|m| m.len());
    let waiting = waits().try_lock().ok().map(|m| m.len());
    let mut out = vec![format!("tracer tables: {} handles, {} threads waiting", size(handles), size(waiting))];
    out.extend(waits_over(LONG_WAIT_SECS));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// the recorded wait of a thread waiting on `h`
    fn wait_on(h: usize) -> Option<Wait> {
        waits().lock().unwrap().values().find(|w| w.handles.contains(&h)).cloned()
    }

    #[test]
    fn a_thread_waiting_on_an_event_nobody_set_is_listed_until_the_event_is_set() {
        unsafe {
            let ev = CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) as usize;
            assert_ne!(ev, 0);
            let t = std::thread::spawn(move || WaitForSingleObject(ev as HANDLE, 10_000));
            let deadline = Instant::now() + Duration::from_secs(5);
            while wait_on(ev).is_none() {
                assert!(Instant::now() < deadline, "the wait was not recorded");
                std::thread::sleep(Duration::from_millis(5));
            }
            let w = wait_on(ev).unwrap();
            assert_eq!((w.api, w.count, w.timeout), ("WaitForSingleObject", 1, 10_000));
            let lines = waits_over(0.0);
            let i = lines.iter().position(|l| l.contains(&format!("{ev:#x} event made at"))).expect("the event is listed");
            assert!(lines[i].ends_with("never set by the title"), "{}", lines[i]);
            assert!(lines[i - 1].contains("in WaitForSingleObject from"), "{}", lines[i - 1]);

            assert_ne!(SetEvent(ev as HANDLE), 0);
            assert_eq!(t.join().unwrap(), 0, "WAIT_OBJECT_0");
            assert!(wait_on(ev).is_none(), "the wait ends with the call");
            assert_eq!(objects().lock().unwrap()[&ev].signals, 1);
            let o = objects().lock().unwrap()[&ev].clone();
            assert!(describe(ev, Some(&o), |_| None).contains("last set at"));
            assert_ne!(CloseHandle(ev as HANDLE), 0);
            assert!(!objects().lock().unwrap().contains_key(&ev), "a closed handle is forgotten");
        }
    }

    #[test]
    fn a_poll_is_not_a_wait_and_semaphores_count_releases() {
        unsafe {
            let name: Vec<u16> = "xcl-waits-test\0".encode_utf16().collect();
            let s = CreateSemaphoreW(std::ptr::null(), 0, 4, name.as_ptr()) as usize;
            assert_ne!(s, 0);
            assert_eq!(WaitForSingleObject(s as HANDLE, 0), 0x102, "WAIT_TIMEOUT");
            assert!(wait_on(s).is_none(), "a timeout of 0 is a poll");
            assert_ne!(ReleaseSemaphore(s as HANDLE, 2, std::ptr::null_mut()), 0);
            assert_ne!(ReleaseSemaphore(s as HANDLE, 1, std::ptr::null_mut()), 0);
            let o = objects().lock().unwrap()[&s].clone();
            let d = describe(s, Some(&o), |_| None);
            assert!(d.contains("semaphore \"xcl-waits-test\" made at") && d.contains("last released at") && d.ends_with("(2 times)"), "{d}");
            CloseHandle(s as HANDLE);
        }
    }

    #[test]
    fn an_object_made_elsewhere_is_named_by_its_kernel_type() {
        let t = std::thread::spawn(|| ());
        let h = std::os::windows::io::AsRawHandle::as_raw_handle(&t) as usize;
        let d = describe(h, None, object_type);
        t.join().unwrap();
        assert!(d.starts_with(&format!("{h:#x} Thread not made through the traced imports")), "{d}");
        assert!(d.ends_with("never set by the title"), "{d}");
    }
}
