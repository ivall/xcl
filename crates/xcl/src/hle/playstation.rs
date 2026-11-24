//! PlayStation controllers (DualShock 4, DualSense) as the title's gamepad. Windows gives them to
//! XInput only through extra software (Steam Input, DS4Windows), so without it a player with
//! one had no gamepad. They are read here as HID devices: a thread reads the first one found
//! (input report 0x01; a DualShock 4 over Bluetooth sends the same first bytes as over USB) and
//! `gameinput::poll` uses it when no XInput controller answers. A DualSense over USB also
//! takes the title's rumble, sent by a thread of its own whenever it changes; sending it also
//! sets the light bar and player LEDs to their defaults. The low and high levels play as
//! compatible vibration, or with `haptics = "auto"` on its haptic actuators through its USB
//! audio channels 3–4 (`dualsense_haptics`) when that output has 4 channels; the trigger levels
//! become adaptive trigger effects (`[dualsense]` in `xcl.toml`). Reads are overlapped, so `shutdown` (at process exit)
//! can stop them, the motors and the triggers.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::Duration;

use windows_sys::core::GUID;
use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_IO_PENDING, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0};
use windows_sys::Win32::Storage::FileSystem::{CreateFileW, ReadFile, WriteFile, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING};
use windows_sys::Win32::System::Threading::{CreateEventW, SetEvent, WaitForMultipleObjects};
use windows_sys::Win32::System::IO::{CancelIoEx, GetOverlappedResult, OVERLAPPED};

use super::dualsense_haptics as haptics;
use super::dualsense_report::{self as report, TriggerEffect};
use super::gameinput::*;
use crate::config::DualSenseConfig;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Layout {
    DualShock4,
    /// over USB (its Bluetooth reports are laid out differently and are not read)
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
    while !SHUTDOWN.load(Ordering::Acquire) {
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
    // Bluetooth HID (service 1124): a DualSense's reports there are not the USB ones
    let bluetooth = p.contains("00001124-0000-1000-8000-00805f9b34fb");
    PADS.iter()
        .find(|(pid, layout, _)| (p.contains(&format!("pid_{pid}")) || p.contains(&format!("pid&{pid}"))) && !(bluetooth && *layout == Layout::DualSense))
        .map(|&(_, layout, name)| (layout, name))
}

/// a manual-reset event set by `shutdown`: wakes the reader
fn stop_event() -> HANDLE {
    static E: OnceLock<usize> = OnceLock::new();
    *E.get_or_init(|| unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) as usize }) as HANDLE
}
static SHUTDOWN: AtomicBool = AtomicBool::new(false);
/// the reading handle, for `shutdown`'s CancelIoEx
static READ_HANDLE: AtomicUsize = AtomicUsize::new(0);

fn open(path: &[u16], access: u32, flags: u32) -> Option<HANDLE> {
    let h = unsafe { CreateFileW(path.as_ptr(), access, FILE_SHARE_READ | FILE_SHARE_WRITE, std::ptr::null(), OPEN_EXISTING, flags, std::ptr::null_mut()) };
    (h != INVALID_HANDLE_VALUE).then_some(h)
}

/// One overlapped read: the report's length, or None when the read failed, `shutdown` stopped
/// it or nothing came for 2 s (a controller that went out of Bluetooth range can leave a read
/// pending without an error).
fn read_report(h: HANDLE, buf: &mut [u8], ev: HANDLE) -> Option<usize> {
    unsafe {
        let mut ov: OVERLAPPED = std::mem::zeroed();
        ov.hEvent = ev;
        let mut n = 0u32;
        if ReadFile(h, buf.as_mut_ptr(), buf.len() as u32, &mut n, &mut ov) == 0 && GetLastError() != ERROR_IO_PENDING {
            return None;
        }
        let w = WaitForMultipleObjects(2, [ev, stop_event()].as_ptr(), 0, 2000);
        if w != WAIT_OBJECT_0 {
            CancelIoEx(h, &ov);
            GetOverlappedResult(h, &ov, &mut n, 1);
            return None;
        }
        (GetOverlappedResult(h, &ov, &mut n, 0) != 0).then_some(n as usize)
    }
}

fn read_until_gone(path: &[u16], layout: Layout, name: &str) {
    let Some(h) = open(path, GENERIC_READ, FILE_FLAG_OVERLAPPED) else {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if !ONCE.swap(true, Ordering::Relaxed) {
            crate::xlog_warn!("GameInput: the {name} could not be opened ({}) – it is not the title's gamepad", std::io::Error::last_os_error());
        }
        return;
    };
    READ_HANDLE.store(h as usize, Ordering::Release);
    // a DualSense over USB takes the rumble on a handle of its own (writes are not overlapped)
    let out = if layout == Layout::DualSense { open(path, GENERIC_WRITE, 0) } else { None };
    crate::xlog_info!(
        "GameInput: a {name} is read as the title's gamepad when no XInput controller answers{}",
        if out.is_some() { ", with rumble" } else { "" }
    );
    with_output(|o| *o = Output { conn: out.map(|h| h as usize), rumble: o.rumble, sent: None });
    let cfg = &crate::state::config().dualsense;
    if out.is_some() && cfg.haptics.eq_ignore_ascii_case("auto") {
        // (finding and starting the audio output can take a moment: not on the reading thread)
        let device = cfg.haptics_device.clone();
        let _ = std::thread::Builder::new().name("xcl dualsense haptics start".into()).spawn(move || {
            if haptics::start(&device) {
                with_output(|o| o.play_haptics());
            }
        });
    }
    let ev = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
    let mut buf = [0u8; 128];
    let mut odd = false;
    while !SHUTDOWN.load(Ordering::Acquire) {
        let Some(n) = read_report(h, &mut buf, ev) else { break };
        match parse(layout, &buf[..n]) {
            Some(s) => *STATE.lock().unwrap() = Some(s),
            None if !odd => {
                odd = true;
                crate::xlog_info!("GameInput: the {name} sent a report this does not read (id {:#x}, {n} bytes)", buf[0]);
            }
            None => {}
        }
    }
    READ_HANDLE.store(0, Ordering::Release);
    haptics::stop();
    if let Some(c) = with_output(|o| o.conn.take()) {
        unsafe { CloseHandle(c as HANDLE) };
    }
    unsafe {
        CloseHandle(ev);
        CloseHandle(h);
    }
    if !SHUTDOWN.load(Ordering::Acquire) {
        crate::xlog_info!("GameInput: the {name} was disconnected");
    }
}

/// What a DualSense is to play, and what was last sent to it.
#[derive(Default)]
struct Output {
    /// the write handle of a connected DualSense
    conn: Option<usize>,
    /// the title's levels: low and high motor, left and right trigger
    rumble: [f32; 4],
    sent: Option<[u8; 48]>,
}

/// A trigger's effect for the title's trigger rumble `level` (0..1).
fn trigger_effect(level: f32, cfg: &DualSenseConfig) -> TriggerEffect {
    let step = |v: f32| (v.clamp(0.0, 1.0) * 8.0).round() as u8;
    let strength = cfg.trigger_strength.clamp(0.0, 1.0);
    let base = step(cfg.trigger_resistance * strength);
    let resting = if base > 0 { TriggerEffect::Feedback { start: 0, strength: base } } else { TriggerEffect::Off };
    match cfg.triggers.to_ascii_lowercase().as_str() {
        "off" => TriggerEffect::Off,
        "resistance" => match step(level * strength).max(base) {
            0 => TriggerEffect::Off,
            s => TriggerEffect::Feedback { start: 0, strength: s },
        },
        // "vibration" (default): a buzz, stronger and a little faster with the level
        _ => match step(level * strength) {
            0 => resting,
            a => TriggerEffect::Vibration { start: 0, amplitude: a, frequency: 30 + (level.clamp(0.0, 1.0) * 30.0) as u8 },
        },
    }
}

/// The motor levels after the settings (0..1, low and high).
fn motor_levels(rumble: [f32; 4], cfg: &DualSenseConfig) -> (f32, f32) {
    if cfg.haptics.eq_ignore_ascii_case("off") {
        return (0.0, 0.0);
    }
    let k = cfg.rumble_strength.clamp(0.0, 1.0);
    (rumble[0].clamp(0.0, 1.0) * k, rumble[1].clamp(0.0, 1.0) * k)
}

/// The output report for the title's levels: the motors as compatible vibration, or left to
/// the audio haptics while those play.
fn levels_report(rumble: [f32; 4], cfg: &DualSenseConfig, haptics: bool) -> [u8; 48] {
    let b = |v: f32| (v * 255.0) as u8;
    let (lo, hi) = motor_levels(rumble, cfg);
    report::Output {
        rumble: (!haptics).then(|| (b(lo), b(hi))),
        left_trigger: trigger_effect(rumble[2], cfg),
        right_trigger: trigger_effect(rumble[3], cfg),
        ..Default::default()
    }
    .usb()
}

impl Output {
    /// The output report to send now: None while it is the one last sent.
    fn next_report(&mut self) -> Option<[u8; 48]> {
        self.next_report_with(&crate::state::config().dualsense, haptics::running())
    }

    fn next_report_with(&mut self, cfg: &DualSenseConfig, haptics: bool) -> Option<[u8; 48]> {
        let r = levels_report(self.rumble, cfg, haptics);
        (self.sent != Some(r)).then(|| {
            self.sent = Some(r);
            r
        })
    }

    /// Hands the motor levels to the audio haptics while those play.
    fn play_haptics(&self) {
        if haptics::running() {
            let (lo, hi) = motor_levels(self.rumble, &crate::state::config().dualsense);
            haptics::set(lo, hi);
        }
    }
}

static OUTPUT: Mutex<Option<Output>> = Mutex::new(None);
static WAKE: Condvar = Condvar::new();

fn with_output<R>(f: impl FnOnce(&mut Output) -> R) -> R {
    let r = f(OUTPUT.lock().unwrap().get_or_insert_with(Output::default));
    WAKE.notify_all();
    r
}

fn write(h: usize, r: &[u8]) -> bool {
    let mut n = 0u32;
    unsafe { WriteFile(h as HANDLE, r.as_ptr(), r.len() as u32, &mut n, std::ptr::null_mut()) != 0 }
}

/// Sends the output report whenever it changes, so the title's thread never waits on HID.
fn writer() {
    let mut g = OUTPUT.lock().unwrap();
    while !SHUTDOWN.load(Ordering::Acquire) {
        let o = g.get_or_insert_with(Output::default);
        if let Some(c) = o.conn {
            if let Some(r) = o.next_report() {
                drop(g);
                if !write(c, &r) {
                    crate::xlog_debug!("GameInput: the DualSense did not take an output report");
                }
                g = OUTPUT.lock().unwrap();
                continue;
            }
        }
        g = WAKE.wait_timeout(g, Duration::from_millis(500)).unwrap().0;
    }
}

/// The title's rumble (0..1: low and high motor, left and right trigger); false when no
/// DualSense takes it.
pub fn rumble(low: f32, high: f32, left_trigger: f32, right_trigger: f32) -> bool {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if !STARTED.swap(true, Ordering::Relaxed) {
        if let Err(e) = std::thread::Builder::new().name("xcl playstation output".into()).spawn(writer) {
            crate::xlog_warn!("GameInput: no thread for the DualSense's rumble: {e}");
        }
    }
    with_output(|o| {
        o.rumble = [low, high, left_trigger, right_trigger];
        o.play_haptics();
        o.conn.is_some()
    })
}

/// At process exit: a pending read is cancelled, the motors stopped and the triggers let go (a
/// DualSense keeps the last rumble and trigger effect it was sent until it is switched off).
pub fn shutdown() {
    if SHUTDOWN.swap(true, Ordering::AcqRel) {
        return;
    }
    haptics::stop();
    unsafe { SetEvent(stop_event()) };
    let h = READ_HANDLE.load(Ordering::Acquire);
    if h != 0 {
        unsafe { CancelIoEx(h as HANDLE, std::ptr::null()) };
    }
    // try_lock: the process may end with another thread holding it
    if let Ok(mut g) = OUTPUT.try_lock() {
        if let Some(o) = g.as_mut() {
            if let (Some(c), Some(_)) = (o.conn, o.sent) {
                write(c, &report::Output::default().usb());
            }
        }
    }
}

/// Input report 0x01 as a GameInput gamepad state (HID sticks: 0 left / up, 128 centre).
pub(crate) fn parse(layout: Layout, r: &[u8]) -> Option<GamepadState> {
    // (face buttons and hat, the byte after it, left trigger, right trigger)
    let (b, l2, r2) = match layout {
        Layout::DualShock4 if r.len() >= 10 => (5, 8, 9),
        Layout::DualSense if r.len() >= 11 => (8, 5, 6),
        _ => return None,
    };
    if r[0] != 0x01 {
        return None;
    }
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
        left_x: x(r[1]),
        left_y: y(r[2]),
        right_x: x(r[3]),
        right_y: y(r[4]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rumble levels become the USB report's motor bytes (right small, left big), and a
    /// report is sent only when it changes.
    #[test]
    fn dualsense_rumble_is_sent_when_it_changes() {
        // (the settings and the haptics state passed in: another test can connect a real
        // DualSense and start its haptics in this process)
        let cfg = DualSenseConfig::default();
        let mut o = Output { rumble: [1.0, 0.5, 0.0, 0.0], ..Output::default() };
        let r = o.next_report_with(&cfg, false).unwrap();
        assert_eq!((r[0], r[1] & 0x03, r[3], r[4]), (0x02, 0x03, 127, 255));
        assert_eq!(o.next_report_with(&cfg, false), None);
        o.rumble = [0.0; 4];
        let r = o.next_report_with(&cfg, false).unwrap();
        assert_eq!((r[3], r[4]), (0, 0));
        assert_eq!(o.next_report_with(&cfg, false), None);
    }

    /// The title's trigger rumble as adaptive trigger effects under the settings.
    #[test]
    fn trigger_rumble_becomes_trigger_effects() {
        let mut cfg = DualSenseConfig::default();
        assert_eq!(trigger_effect(0.0, &cfg), TriggerEffect::Off);
        assert_eq!(trigger_effect(1.0, &cfg), TriggerEffect::Vibration { start: 0, amplitude: 8, frequency: 60 });
        assert_eq!(trigger_effect(0.5, &cfg), TriggerEffect::Vibration { start: 0, amplitude: 4, frequency: 45 });
        cfg.trigger_strength = 0.5;
        assert_eq!(trigger_effect(1.0, &cfg), TriggerEffect::Vibration { start: 0, amplitude: 4, frequency: 60 });
        // a resting resistance between rumbles
        cfg.trigger_resistance = 0.5;
        assert_eq!(trigger_effect(0.0, &cfg), TriggerEffect::Feedback { start: 0, strength: 2 });
        cfg.triggers = "resistance".into();
        cfg.trigger_strength = 1.0;
        assert_eq!(trigger_effect(0.0, &cfg), TriggerEffect::Feedback { start: 0, strength: 4 });
        assert_eq!(trigger_effect(1.0, &cfg), TriggerEffect::Feedback { start: 0, strength: 8 });
        cfg.triggers = "off".into();
        assert_eq!(trigger_effect(1.0, &cfg), TriggerEffect::Off);
    }

    /// Motors as compatible vibration, or left to the haptics while those play; strength and
    /// "off" apply to both.
    #[test]
    fn rumble_levels_go_to_the_motors_or_the_haptics() {
        let mut cfg = DualSenseConfig::default();
        let r = levels_report([1.0, 0.5, 0.0, 1.0], &cfg, false);
        assert_eq!((r[1] & 0x0F, r[3], r[4]), (0x0F, 127, 255));
        assert_eq!(r[11..22], TriggerEffect::Vibration { start: 0, amplitude: 8, frequency: 60 }.block(), "right trigger");
        assert_eq!(r[22], 0x05, "left trigger off");
        let r = levels_report([1.0, 0.5, 0.0, 0.0], &cfg, true);
        assert_eq!((r[1] & 0x0F, r[3], r[4]), (0x0C, 0, 0), "the actuators are the audio haptics'");
        cfg.rumble_strength = 0.5;
        assert_eq!(motor_levels([1.0, 0.5, 0.0, 0.0], &cfg), (0.5, 0.25));
        cfg.haptics = "off".into();
        assert_eq!(motor_levels([1.0, 1.0, 0.0, 0.0], &cfg), (0.0, 0.0));
    }

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
        assert_eq!(parse(Layout::DualShock4, &[0x11; 64]), None);
        assert_eq!(parse(Layout::DualShock4, &[0x01; 9]), None);
    }

    #[test]
    fn playstation_controllers_are_known_by_their_hid_path() {
        let usb = r"\\?\hid#vid_054c&pid_05c4#6&75b5218&4&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}";
        assert_eq!(known(usb), Some((Layout::DualShock4, "DualShock 4")));
        assert_eq!(known(&usb.to_uppercase()), Some((Layout::DualShock4, "DualShock 4")));
        assert_eq!(known(r"\\?\hid#vid_054c&pid_0ce6#7&1#{4d1e55b2-f16f-11cf-88cb-001111000030}"), Some((Layout::DualSense, "DualSense")));
        let bt = r"\\?\hid#{00001124-0000-1000-8000-00805f9b34fb}_vid&0002054c_pid&";
        assert_eq!(known(&format!("{bt}09cc#9&1#{{4d1e55b2-f16f-11cf-88cb-001111000030}}")).map(|k| k.0), Some(Layout::DualShock4));
        assert_eq!(known(&format!("{bt}0ce6#9&1#{{4d1e55b2-f16f-11cf-88cb-001111000030}}")), None, "a DualSense over Bluetooth is not read");
        assert_eq!(known(r"\\?\hid#vid_045e&pid_0b13#1#{4d1e55b2-f16f-11cf-88cb-001111000030}"), None);
    }
}
