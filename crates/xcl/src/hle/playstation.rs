//! PlayStation controllers (DualShock 4, DualSense) as the title's gamepad. Windows gives them to
//! XInput only through extra software (Steam Input, DS4Windows), so without it a player with
//! one had no gamepad. They are read here as HID devices: a thread reads the first one found
//! and `gameinput::poll` uses it when no XInput controller answers. Reports read (the id byte
//! first): USB 0x01 for both pads; over Bluetooth the simple 0x01 report a pad sends until a
//! title asks for its calibration (we never do), the DualSense's full 0x31 and the
//! DualShock 4's full 0x11. Rumble and the light bar are
//! not driven.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use windows_sys::core::GUID;

use super::gameinput::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Layout {
    DualShock4,
    DualSense,
}

/// Sony's controllers by USB product id (vendor 054C)
const PADS: [(&str, Layout, &str); 5] = [
    ("05c4", Layout::DualShock4, "DualShock 4"),
    ("09cc", Layout::DualShock4, "DualShock 4"),
    ("0ba0", Layout::DualShock4, "DualShock 4 USB wireless adaptor"),
    ("0ce6", Layout::DualSense, "DualSense"),
    ("0df2", Layout::DualSense, "DualSense Edge"),
];

/// the pad's state now; None while no PlayStation controller is read
static STATE: Mutex<Option<GamepadState>> = Mutex::new(None);

/// A PlayStation controller's state, when one is connected (the reading thread starts on the
/// first call).
pub fn pad() -> Option<GamepadState> {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if !STARTED.swap(true, Ordering::Relaxed) && std::env::var("XCL_PLAYSTATION_PAD").map_or(true, |v| v.trim() != "0") {
        if let Err(e) = std::thread::Builder::new().name("xcl playstation pad".into()).spawn(reader) {
            crate::xlog_warn!("GameInput: no thread for PlayStation controllers: {e}");
        }
    }
    *STATE.lock().unwrap()
}

/// Looks for a controller every second; reads one until it is gone.
fn reader() {
    loop {
        if let Some((path, layout, name)) = find() {
            read_until_gone(&path, layout, name);
            *STATE.lock().unwrap() = None;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

#[link(name = "cfgmgr32")]
extern "system" {
    fn CM_Get_Device_Interface_List_SizeW(len: *mut u32, guid: *const GUID, device: *const u16, flags: u32) -> u32;
    fn CM_Get_Device_Interface_ListW(guid: *const GUID, device: *const u16, buf: *mut u16, len: u32, flags: u32) -> u32;
}

/// GUID_DEVINTERFACE_HID
const HID: GUID = GUID::from_u128(0x4d1e55b2_f16f_11cf_88cb_001111000030);

/// The first present HID interface of a PlayStation controller: (path, layout, name).
fn find() -> Option<(Vec<u16>, Layout, &'static str)> {
    let mut len = 0u32;
    // (CM_GET_DEVICE_INTERFACE_LIST_PRESENT = 0)
    if unsafe { CM_Get_Device_Interface_List_SizeW(&mut len, &HID, std::ptr::null(), 0) } != 0 || len < 2 {
        return None;
    }
    let mut buf = vec![0u16; len as usize];
    if unsafe { CM_Get_Device_Interface_ListW(&HID, std::ptr::null(), buf.as_mut_ptr(), len, 0) } != 0 {
        return None;
    }
    buf.split(|&c| c == 0).filter(|p| !p.is_empty()).find_map(|p| {
        let (layout, name) = known(&String::from_utf16_lossy(p))?;
        Some((p.iter().copied().chain([0]).collect(), layout, name))
    })
}

/// The controller a HID interface path names, if it is one read here.
pub(crate) fn known(path: &str) -> Option<(Layout, &'static str)> {
    let p = path.to_ascii_lowercase();
    // (USB: vid_054c&pid_xxxx; Bluetooth: vid&0002054c_pid&xxxx)
    if !p.contains("vid_054c") && !p.contains("vid&0002054c") {
        return None;
    }
    PADS.iter()
        .find(|(pid, _, _)| p.contains(&format!("pid_{pid}")) || p.contains(&format!("pid&{pid}")))
        .map(|&(_, layout, name)| (layout, name))
}

fn read_until_gone(path: &[u16], layout: Layout, name: &str) {
    use windows_sys::Win32::Foundation::{CloseHandle, GENERIC_READ, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{CreateFileW, ReadFile, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING};
    let h = unsafe { CreateFileW(path.as_ptr(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, std::ptr::null(), OPEN_EXISTING, 0, std::ptr::null_mut()) };
    if h == INVALID_HANDLE_VALUE {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if !ONCE.swap(true, Ordering::Relaxed) {
            crate::xlog_warn!("GameInput: the {name} could not be opened ({}) – it is not the title's gamepad", std::io::Error::last_os_error());
        }
        return;
    }
    crate::xlog_info!("GameInput: a {name} is read as the title's gamepad when no XInput controller answers");
    let mut buf = [0u8; 128];
    let mut odd = false;
    loop {
        let mut n = 0u32;
        if unsafe { ReadFile(h, buf.as_mut_ptr(), buf.len() as u32, &mut n, std::ptr::null_mut()) } == 0 {
            break;
        }
        match parse(layout, &buf[..n as usize]) {
            Some(s) => *STATE.lock().unwrap() = Some(s),
            None if !odd => {
                odd = true;
                crate::xlog_info!("GameInput: the {name} sent a report this does not read (id {:#x}, {n} bytes)", buf[0]);
            }
            None => {}
        }
    }
    unsafe { CloseHandle(h as *mut c_void) };
    crate::xlog_info!("GameInput: the {name} was disconnected");
}

/// An input report as a GameInput gamepad state; None for a report that is no pad state.
pub(crate) fn parse(layout: Layout, r: &[u8]) -> Option<GamepadState> {
    match (layout, *r.first()?) {
        // A DualSense over USB: sticks, triggers, then the buttons from byte 8
        (Layout::DualSense, 0x01) if r.len() == 64 => full(&r[1..]),
        // A DualSense over Bluetooth, full: one header byte after the id
        (Layout::DualSense, 0x31) if r.len() >= 12 => full(&r[2..]),
        // The simple layout: a DualShock 4 over USB or Bluetooth, a DualSense over Bluetooth
        // (the read may be padded to the longest report)
        (_, 0x01) if r.len() >= 10 => simple(&r[1..]),
        // A DualShock 4 over Bluetooth, full: two header bytes after the id
        (Layout::DualShock4, 0x11) if r.len() >= 14 => simple(&r[3..]),
        _ => None,
    }
}

/// The simple layout: sticks, hat and buttons, a system byte, then L2 and R2.
fn simple(d: &[u8]) -> Option<GamepadState> {
    state(d, 4, 7, 8)
}

/// A DualSense's full report: sticks, L2, R2, a counter, then the buttons.
fn full(d: &[u8]) -> Option<GamepadState> {
    state(d, 7, 4, 5)
}

/// `d` starts at the left stick x; `b` is the index of the face buttons and hat (the shoulder
/// buttons follow), `l2` and `r2` of the triggers.
fn state(d: &[u8], b: usize, l2: usize, r2: usize) -> Option<GamepadState> {
    let r = d.get(..b.max(l2).max(r2).max(b + 1) + 1)?;
    const HAT: [u32; 8] = [DPAD_UP, DPAD_UP | DPAD_RIGHT, DPAD_RIGHT, DPAD_DOWN | DPAD_RIGHT, DPAD_DOWN, DPAD_DOWN | DPAD_LEFT, DPAD_LEFT, DPAD_UP | DPAD_LEFT];
    const BITS: [(usize, u8, u32); 10] = [
        (0, 0x10, X),  // square
        (0, 0x20, A),  // cross
        (0, 0x40, B),  // circle
        (0, 0x80, Y),  // triangle
        (1, 0x01, LEFT_SHOULDER),
        (1, 0x02, RIGHT_SHOULDER),
        (1, 0x10, VIEW), // share / create
        (1, 0x20, MENU), // options
        (1, 0x40, LEFT_STICK),
        (1, 0x80, RIGHT_STICK),
    ];
    let hat = HAT.get((r[b] & 0xF) as usize).copied().unwrap_or(0);
    let buttons = BITS.iter().filter(|&&(k, m, _)| r[b + k] & m != 0).fold(hat, |s, &(_, _, g)| s | g);
    let x = |v: u8| ((v as f32 - 128.0) / 127.0).clamp(-1.0, 1.0);
    let y = |v: u8| ((128.0 - v as f32) / 127.0).clamp(-1.0, 1.0);
    Some(GamepadState {
        buttons,
        left_trigger: r[l2] as f32 / 255.0,
        right_trigger: r[r2] as f32 / 255.0,
        left_x: x(r[0]),
        left_y: y(r[1]),
        right_x: x(r[2]),
        right_y: y(r[3]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DualShock 4 report and a DualSense USB report with the same input give the same
    /// gamepad state; idle sticks are centred and no button is pressed.
    #[test]
    fn playstation_reports_read_as_a_gamepad() {
        // idle DualShock 4 (USB, 64 bytes): sticks 128, hat 8 = none
        let mut ds4 = [0u8; 64];
        ds4[..10].copy_from_slice(&[0x01, 128, 128, 128, 128, 0x08, 0, 0, 0, 0]);
        let idle = parse(Layout::DualShock4, &ds4).unwrap();
        assert_eq!(idle.buttons, 0);
        assert_eq!((idle.left_x, idle.left_y, idle.right_x, idle.right_y), (0.0, 0.0, 0.0, 0.0));
        // cross + circle, hat down-left, L1 + options + R3, left stick full left and up,
        // right stick full right and down, L2 full, R2 half
        ds4[..10].copy_from_slice(&[0x01, 0, 0, 255, 255, 0x60 | 5, 0x01 | 0x20 | 0x80, 0, 255, 128]);
        let s = parse(Layout::DualShock4, &ds4).unwrap();
        assert_eq!(s.buttons, A | B | DPAD_DOWN | DPAD_LEFT | LEFT_SHOULDER | MENU | RIGHT_STICK);
        assert_eq!((s.left_x, s.left_y, s.right_x, s.right_y), (-1.0, 1.0, 1.0, -1.0));
        assert_eq!(s.left_trigger, 1.0);
        assert!((s.right_trigger - 0.502).abs() < 0.001);
        // the same on a DualSense over USB: triggers at 5 and 6, buttons from 8
        let mut ds = [0u8; 64];
        ds[..11].copy_from_slice(&[0x01, 0, 0, 255, 255, 255, 128, 0, 0x60 | 5, 0x01 | 0x20 | 0x80, 0]);
        assert_eq!(parse(Layout::DualSense, &ds), Some(s));
        // other reports and short ones are not read
        assert_eq!(parse(Layout::DualShock4, &[0x05; 64]), None);
        assert_eq!(parse(Layout::DualShock4, &[0x01; 9]), None);
        assert_eq!(parse(Layout::DualSense, &[0x31; 11]), None);
    }

    /// The same input, as the Bluetooth reports lay it out, reads as the same gamepad state.
    #[test]
    fn bluetooth_reports_read_as_the_same_gamepad() {
        let usb = {
            let mut r = [0u8; 64];
            r[..11].copy_from_slice(&[0x01, 0, 0, 255, 255, 255, 128, 0, 0x60 | 5, 0x01 | 0x20 | 0x80, 0]);
            parse(Layout::DualSense, &r).unwrap()
        };
        // a DualSense over Bluetooth, full: one header byte after the id
        let mut full = [0u8; 78];
        full[..12].copy_from_slice(&[0x31, 0x01, 0, 0, 255, 255, 255, 128, 0, 0x60 | 5, 0x01 | 0x20 | 0x80, 0]);
        assert_eq!(parse(Layout::DualSense, &full), Some(usb));
        // the simple report of a DualSense over Bluetooth, exact and padded to 78 bytes
        let mut simple = [0u8; 78];
        simple[..10].copy_from_slice(&[0x01, 0, 0, 255, 255, 0x60 | 5, 0x01 | 0x20 | 0x80, 0, 255, 128]);
        assert_eq!(parse(Layout::DualSense, &simple), Some(usb));
        assert_eq!(parse(Layout::DualSense, &simple[..10]), Some(usb));
        // a DualShock 4 over Bluetooth, full: two header bytes after the id
        let mut ds4 = [0u8; 78];
        ds4[..14].copy_from_slice(&[0x11, 0xC0, 0, 0, 0, 255, 255, 0x60 | 5, 0x01 | 0x20 | 0x80, 0, 255, 128, 0, 0]);
        assert_eq!(parse(Layout::DualShock4, &ds4), Some(usb));
        // a DualSense does not send 0x11, and a DualShock 4 not 0x31
        assert_eq!(parse(Layout::DualSense, &ds4), None);
        assert_eq!(parse(Layout::DualShock4, &full), None);
    }

    #[test]
    fn playstation_controllers_are_known_by_their_hid_path() {
        let usb = r"\\?\hid#vid_054c&pid_05c4#6&75b5218&4&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}";
        assert_eq!(known(usb), Some((Layout::DualShock4, "DualShock 4")));
        assert_eq!(known(&usb.to_uppercase()), Some((Layout::DualShock4, "DualShock 4")));
        assert_eq!(known(r"\\?\hid#vid_054c&pid_0ce6#7&1#{4d1e55b2-f16f-11cf-88cb-001111000030}"), Some((Layout::DualSense, "DualSense")));
        let bt = r"\\?\hid#{00001124-0000-1000-8000-00805f9b34fb}_vid&0002054c_pid&";
        assert_eq!(known(&format!("{bt}09cc#9&1#{{4d1e55b2-f16f-11cf-88cb-001111000030}}")).map(|k| k.0), Some(Layout::DualShock4));
        assert_eq!(known(&format!("{bt}0ce6#9&1#{{4d1e55b2-f16f-11cf-88cb-001111000030}}")).map(|k| k.0), Some(Layout::DualSense));
        assert_eq!(known(r"\\?\hid#vid_045e&pid_0b13#1#{4d1e55b2-f16f-11cf-88cb-001111000030}"), None);
    }
}
