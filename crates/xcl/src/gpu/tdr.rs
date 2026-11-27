//! Windows' GPU timeout (TDR). When the GPU does not preempt within `TdrDelay` seconds (2 when
//! it is not set) of the scheduler asking it to, WDDM resets the GPU (at the default
//! `TdrLevel`) and the title's device is removed with `DEVICE_HUNG` or `DEVICE_RESET`. The
//! removal code alone does not tell that apart from a hang the driver found itself. Windows
//! writes a `Display` event 4101 ("display driver stopped responding and has recovered") to the
//! System log when it was its timeout; this reads the settings and looks for that event, so the
//! log can say whether Windows logged one.

use windows_sys::Win32::Foundation::{GetLastError, ERROR_NO_MORE_ITEMS};
use windows_sys::Win32::System::EventLog::{EvtClose, EvtNext, EvtQuery, EvtQueryChannelPath, EvtQueryReverseDirection};
use windows_sys::Win32::System::Registry::{RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_DWORD};

/// `DXGI_ERROR_DEVICE_HUNG` and `DXGI_ERROR_DEVICE_RESET`: the removals a TDR gives.
pub fn is_hang(code: u32) -> bool {
    matches!(code, 0x887A_0006 | 0x887A_0007)
}

/// The timeout's settings in `HKLM\System\CurrentControlSet\Control\GraphicsDrivers`
/// (None: not set, so Windows' default).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Settings {
    pub delay: Option<u32>,
    pub level: Option<u32>,
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}

fn dword(name: &str) -> Option<u32> {
    let key = wide(r"System\CurrentControlSet\Control\GraphicsDrivers");
    let name = wide(name);
    let mut v = 0u32;
    let mut n = 4u32;
    let r = unsafe { RegGetValueW(HKEY_LOCAL_MACHINE, key.as_ptr(), name.as_ptr(), RRF_RT_REG_DWORD, std::ptr::null_mut(), (&raw mut v).cast(), &mut n) };
    (r == 0).then_some(v)
}

pub fn settings() -> Settings {
    Settings { delay: dword("TdrDelay"), level: dword("TdrLevel") }
}

/// One line on what the timeout is, for the log. `TdrDelay` is how long the GPU may take to
/// preempt when the scheduler asks it to; `TdrLevel` is what Windows does then: 0 nothing,
/// 1 a bug check with no recovery, 3 (the default) a reset. 2 (recover to VGA) is not
/// implemented by Windows. Both are what the registry holds now; Windows reads them at boot, so
/// a change shows here before it applies.
pub fn describe(s: Settings) -> String {
    let delay = s.delay.map_or("2 s (TdrDelay not set)".into(), |d| format!("{d} s (TdrDelay {d})"));
    let what = match s.level {
        Some(0) => "Windows' GPU timeout detection (TDR) is off (TdrLevel 0)".into(),
        Some(1) => format!("Windows bug-checks, with no recovery, when the GPU does not preempt within {delay} (TdrLevel 1)"),
        None | Some(3) => format!("Windows resets the GPU when it does not preempt within {delay}"),
        Some(l) => format!("TdrLevel {l} is not a level Windows implements; the GPU's preemption timeout is {delay}"),
    };
    format!("{what}, per the registry (read by Windows at boot)")
}

/// The System log query for Windows' timeout events of the last `ms`.
fn query(ms: u32) -> Vec<u16> {
    wide(&format!("*[System[Provider[@Name='Display'] and EventID=4101 and TimeCreated[timediff(@SystemTime) <= {ms}]]]"))
}

/// Adds up the batches `next` reads (`Ok`: events in the batch, `Err`: `GetLastError` of a
/// failed read). Only `ERROR_NO_MORE_ITEMS` ends a query; any other failure, before or after
/// some events, leaves the count unknown rather than short (its error code).
fn count_events(mut next: impl FnMut() -> Result<u32, u32>) -> Result<u32, u32> {
    let mut n = 0u32;
    loop {
        match next() {
            Ok(0) | Err(ERROR_NO_MORE_ITEMS) => return Ok(n),
            Ok(got) => n += got,
            Err(e) => return Err(e),
        }
    }
}

/// How many timeout events Windows logged in the last `ms` (None: the System log could not be read).
pub fn recent_resets(ms: u32) -> Option<u32> {
    read_resets(ms).ok()
}

/// `recent_resets`, with the Win32 error of a read that failed.
fn read_resets(ms: u32) -> Result<u32, u32> {
    read_query(&query(ms))
}

/// The number of System log events `q` matches, or the Win32 error of a read that failed.
fn read_query(q: &[u16]) -> Result<u32, u32> {
    let path = wide("System");
    unsafe {
        let h = EvtQuery(0, path.as_ptr(), q.as_ptr(), EvtQueryChannelPath | EvtQueryReverseDirection);
        if h == 0 {
            return Err(GetLastError());
        }
        let mut events = [0isize; 16];
        let n = count_events(|| {
            let mut got = 0u32;
            if EvtNext(h, events.len() as u32, events.as_mut_ptr(), 1000, 0, &mut got) == 0 {
                return Err(GetLastError());
            }
            for &e in &events[..got as usize] {
                EvtClose(e);
            }
            Ok(got)
        });
        EvtClose(h);
        n
    }
}

/// After a removal: whether Windows logged a timeout around it, for the log. Windows can write
/// its event a moment after the title sees the removal, so this looks for up to `wait`.
pub fn removal_note(code: u32, wait: std::time::Duration) -> Option<String> {
    if !is_hang(code) {
        return None;
    }
    note(wait, &describe(settings()), || recent_resets(30_000))
}

/// `removal_note` with the System log read by `resets`. The event names no device, so a recent
/// one is evidence, not proof: another adapter, or an earlier recovered timeout, logs the same.
fn note(wait: std::time::Duration, s: &str, mut resets: impl FnMut() -> Option<u32>) -> Option<String> {
    let start = std::time::Instant::now();
    loop {
        match resets() {
            Some(n) if n > 0 => {
                return Some(format!("{n} Display 4101 event(s) in the System log in the last 30 s, consistent with Windows' GPU timeout (the event does not name the device) – {s}"))
            }
            None => return Some(format!("the System log could not be read to tell whether this was Windows' GPU timeout – {s}")),
            Some(_) if start.elapsed() >= wait => {
                return Some(format!("no Display 4101 event in the System log in the last 30 s: not Windows' GPU timeout, as far as the log shows – {s}"))
            }
            Some(_) => std::thread::sleep(std::time::Duration::from_millis(250)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_hung_and_reset_devices_are_timeouts() {
        assert!(is_hang(0x887A_0006));
        assert!(is_hang(0x887A_0007));
        assert!(!is_hang(0x887A_0005));
        assert!(!is_hang(0x887A_0020));
    }

    #[test]
    fn the_settings_are_described_as_windows_applies_them() {
        let cases = [
            (None, None, "Windows resets the GPU when it does not preempt within 2 s (TdrDelay not set), per the registry (read by Windows at boot)"),
            (None, Some(10), "Windows resets the GPU when it does not preempt within 10 s (TdrDelay 10), per the registry (read by Windows at boot)"),
            (Some(3), Some(10), "Windows resets the GPU when it does not preempt within 10 s (TdrDelay 10), per the registry (read by Windows at boot)"),
            (Some(0), Some(10), "Windows' GPU timeout detection (TDR) is off (TdrLevel 0), per the registry (read by Windows at boot)"),
            // a bug check, not a recovery
            (Some(1), None, "Windows bug-checks, with no recovery, when the GPU does not preempt within 2 s (TdrDelay not set) (TdrLevel 1), per the registry (read by Windows at boot)"),
            (Some(2), Some(10), "TdrLevel 2 is not a level Windows implements; the GPU's preemption timeout is 10 s (TdrDelay 10), per the registry (read by Windows at boot)"),
        ];
        for (level, delay, want) in cases {
            assert_eq!(describe(Settings { delay, level }), want, "TdrLevel {level:?}, TdrDelay {delay:?}");
        }
    }

    #[test]
    fn the_query_asks_for_display_4101_in_a_window() {
        let q = String::from_utf16(&query(30_000)).unwrap();
        assert_eq!(q, "*[System[Provider[@Name='Display'] and EventID=4101 and TimeCreated[timediff(@SystemTime) <= 30000]]]\0");
    }

    #[test]
    fn the_system_log_and_the_settings_can_be_read() {
        let _ = settings();
        // without elevation: the query is valid and the count comes back. A query Windows
        // refuses is ours to fix; a log this machine does not let us read is not.
        match read_resets(1_000) {
            Ok(_) => {}
            Err(windows_sys::Win32::Foundation::ERROR_EVT_INVALID_QUERY) => panic!("Windows refused the System log query"),
            Err(e) => crate::gpu::recomp::runner::skip_test(&format!("the System log is not readable here (error {e})")),
        }
    }

    #[test]
    fn a_query_windows_refuses_is_told_apart() {
        // what `the_system_log_and_the_settings_can_be_read` fails on, rather than skipping; only
        // where a valid query reads the log at all (Wine's wevtapi reads none and gives error 0)
        if let Err(e) = read_resets(1_000) {
            return crate::gpu::recomp::runner::skip_test(&format!("the System log is not readable here (error {e})"));
        }
        assert_eq!(read_query(&wide("*[System[EventID=")), Err(windows_sys::Win32::Foundation::ERROR_EVT_INVALID_QUERY));
    }

    #[test]
    fn other_removals_get_no_note() {
        assert_eq!(removal_note(0x887A_0005, std::time::Duration::ZERO), None);
        let note = removal_note(0x887A_0006, std::time::Duration::ZERO).unwrap();
        assert!(note.contains("TdrDelay") || note.contains("TdrLevel 0"), "{note}");
    }

    /// A reader that returns `reads` in turn, as `EvtNext` would.
    fn reads(mut v: Vec<Result<u32, u32>>) -> impl FnMut() -> Result<u32, u32> {
        v.reverse();
        move || v.pop().expect("read past the end of the query")
    }

    const ERROR_TIMEOUT: u32 = 1460;

    #[test]
    fn only_the_end_of_the_query_gives_a_count() {
        assert_eq!(count_events(reads(vec![Err(ERROR_NO_MORE_ITEMS)])), Ok(0));
        assert_eq!(count_events(reads(vec![Ok(3), Ok(2), Err(ERROR_NO_MORE_ITEMS)])), Ok(5));
        // a first read that times out is not an empty log, and a failure after some events is
        // not a complete count
        assert_eq!(count_events(reads(vec![Err(ERROR_TIMEOUT)])), Err(ERROR_TIMEOUT));
        assert_eq!(count_events(reads(vec![Ok(2), Err(ERROR_TIMEOUT)])), Err(ERROR_TIMEOUT));
    }

    #[test]
    fn a_recent_event_is_evidence_not_proof() {
        let s = "settings";
        let found = note(std::time::Duration::ZERO, s, || Some(1)).unwrap();
        assert!(found.starts_with("1 Display 4101 event(s)") && found.contains("consistent with Windows' GPU timeout"), "{found}");
        assert!(!found.contains("it was"), "{found}");
        let unread = note(std::time::Duration::ZERO, s, || None).unwrap();
        assert!(unread.starts_with("the System log could not be read"), "{unread}");
        let none = note(std::time::Duration::ZERO, s, || Some(0)).unwrap();
        assert!(none.starts_with("no Display 4101 event"), "{none}");
    }

    #[test]
    fn an_event_written_after_the_removal_is_still_found() {
        let mut n = 0;
        let found = note(std::time::Duration::from_secs(3), "settings", || {
            n += 1;
            Some(if n < 3 { 0 } else { 1 })
        });
        assert!(found.unwrap().contains("consistent with"));
        assert_eq!(n, 3);
    }
}
