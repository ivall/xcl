//! The XCL window in borderless fullscreen and back (Alt+Enter, F11; `fullscreen` in xcl.toml
//! starts in it), and the image kept at its own aspect ratio inside the client area.
//!
//! Borderless, not exclusive: the window becomes a `WS_POPUP` covering the whole monitor it is
//! on (not its work area – the taskbar goes under it), and toggling back restores the style and
//! placement it had. DXGI's own Alt+Enter stays off (`DXGI_MWA_NO_ALT_ENTER`): no display mode
//! changes.
//!
//! When the client area's shape differs from the image's, the area filter draws into the
//! largest centred rectangle of the image's aspect (`letterbox`), the rest of the back buffer
//! black (`present/downscale.rs`).

use std::sync::Mutex;

use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Gdi::{GetMonitorInfoW, MonitorFromWindow, MONITORINFO, MONITOR_DEFAULTTONEAREST};
use windows::Win32::UI::WindowsAndMessaging::*;

const VK_RETURN: usize = 0x0D;
const VK_F11: usize = 0x7A;
/// WM_(SYS)KEYDOWN lParam: Alt is down (context code) / the key was already down (auto-repeat)
const KF_ALT: isize = 1 << 29;
const KF_REPEAT: isize = 1 << 30;

/// What a key message does to the window mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    /// Alt+Enter or F11 pressed: fullscreen on / off
    Toggle,
    /// part of the toggle (auto-repeat, the Alt+Enter character that would beep): eaten
    Swallow,
}

/// The window-mode meaning of a window message (None: not ours, it goes on as before).
pub fn key(msg: u32, wp: usize, lp: isize) -> Option<Key> {
    let press = if lp & KF_REPEAT != 0 { Key::Swallow } else { Key::Toggle };
    match (msg, wp) {
        (WM_SYSKEYDOWN, VK_RETURN) if lp & KF_ALT != 0 => Some(press),
        (WM_KEYDOWN | WM_SYSKEYDOWN, VK_F11) => Some(press),
        (WM_SYSCHAR, 0x0D) if lp & KF_ALT != 0 => Some(Key::Swallow),
        _ => None,
    }
}

/// A rectangle of the back buffer, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Area {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// The largest rectangle of `image`'s aspect ratio centred in `client` (the bars around it are
/// black). A side within a pixel of the client's is the client's: no 1-pixel bar from rounding.
pub fn letterbox(image: (u32, u32), client: (u32, u32)) -> Area {
    let (iw, ih, cw, ch) = (image.0 as u64, image.1 as u64, client.0 as u64, client.1 as u64);
    let full = Area { x: 0, y: 0, w: client.0, h: client.1 };
    if iw == 0 || ih == 0 || cw == 0 || ch == 0 {
        return full;
    }
    let (w, h) = if cw * ih > ch * iw {
        // the client is wider: bars left and right
        (((ch * iw + ih / 2) / ih).clamp(1, cw), ch)
    } else {
        (cw, ((cw * ih + iw / 2) / iw).clamp(1, ch))
    };
    let (w, h) = (if cw - w <= 1 { cw } else { w }, if ch - h <= 1 { ch } else { h });
    Area { x: ((cw - w) / 2) as u32, y: ((ch - h) / 2) as u32, w: w as u32, h: h as u32 }
}

/// The window style in fullscreen: no caption, frame or borders, the rest (visible, …) kept.
pub fn fullscreen_style(style: u32) -> u32 {
    (style & !WS_OVERLAPPEDWINDOW.0) | WS_POPUP.0
}

/// The fullscreen window's position and size (x, y, w, h): the monitor's whole rectangle.
pub fn fullscreen_rect(monitor: RECT) -> (i32, i32, i32, i32) {
    (monitor.left, monitor.top, monitor.right - monitor.left, monitor.bottom - monitor.top)
}

/// The window in fullscreen: its handle, and the style and placement to restore.
struct Saved {
    hwnd: isize,
    style: isize,
    placement: WINDOWPLACEMENT,
}

static SAVED: Mutex<Option<Saved>> = Mutex::new(None);

/// `hwnd` is in borderless fullscreen.
pub fn is_fullscreen(hwnd: HWND) -> bool {
    SAVED.lock().unwrap().as_ref().is_some_and(|s| s.hwnd == hwnd.0 as isize)
}

/// Borderless fullscreen on the monitor `hwnd` is on, or back to the window it was.
///
/// # Safety
/// `hwnd` is a top-level window of this process, called on the thread that owns it.
pub unsafe fn toggle(hwnd: HWND) {
    let mut saved = SAVED.lock().unwrap();
    if let Some(s) = saved.take_if(|s| s.hwnd == hwnd.0 as isize) {
        let mut placement = s.placement;
        if !IsWindowVisible(hwnd).as_bool() {
            placement.showCmd = SW_HIDE.0 as u32; // the toggle does not show a hidden window
        }
        SetWindowLongPtrW(hwnd, GWL_STYLE, s.style);
        let _ = SetWindowPlacement(hwnd, &placement);
        let _ = SetWindowPos(hwnd, None, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        crate::xlog_info!("XCL window: back from fullscreen");
        return;
    }
    let mut placement = WINDOWPLACEMENT { length: std::mem::size_of::<WINDOWPLACEMENT>() as u32, ..Default::default() };
    let mut mi = MONITORINFO { cbSize: std::mem::size_of::<MONITORINFO>() as u32, ..Default::default() };
    if GetWindowPlacement(hwnd, &mut placement).is_err() || !GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mut mi).as_bool() {
        crate::xlog_warn!("XCL window: no placement or monitor – fullscreen not entered");
        return;
    }
    let style = GetWindowLongPtrW(hwnd, GWL_STYLE);
    SetWindowLongPtrW(hwnd, GWL_STYLE, fullscreen_style(style as u32) as isize);
    let (x, y, w, h) = fullscreen_rect(mi.rcMonitor);
    let _ = SetWindowPos(hwnd, Some(HWND_TOP), x, y, w, h, SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    *saved = Some(Saved { hwnd: hwnd.0 as isize, style, placement });
    crate::xlog_info!("XCL window: borderless fullscreen {w}x{h} at {x},{y} (Alt+Enter / F11 to leave)");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hle::d3d12x::present::downscale::tests::{draw, pattern, reference};
    use crate::hle::d3d12x::present::downscale::Scaler;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
    use windows::Win32::Graphics::Dxgi::*;

    fn area(x: u32, y: u32, w: u32, h: u32) -> Area {
        Area { x, y, w, h }
    }

    #[test]
    fn letterbox_keeps_the_image_aspect_centred() {
        let uhd = (3840, 2160);
        // 16:10: bars top and bottom; 21:9 and 32:9: left and right; 4:3: top and bottom
        assert_eq!(letterbox(uhd, (1920, 1200)), area(0, 60, 1920, 1080));
        assert_eq!(letterbox(uhd, (2560, 1600)), area(0, 80, 2560, 1440));
        assert_eq!(letterbox(uhd, (2560, 1080)), area(320, 0, 1920, 1080));
        assert_eq!(letterbox(uhd, (5120, 1440)), area(1280, 0, 2560, 1440));
        assert_eq!(letterbox(uhd, (1024, 768)), area(0, 96, 1024, 576));
        // the same ratio: no bars (2560x1440 fullscreen with a 4K image)
        for client in [(1280, 720), (2560, 1440), (3840, 2160)] {
            assert_eq!(letterbox(uhd, client), area(0, 0, client.0, client.1), "{client:?}");
        }
        // odd sizes: within a pixel of the ratio fills; otherwise rounded and centred
        assert_eq!(letterbox(uhd, (1366, 768)), area(0, 0, 1366, 768));
        assert_eq!(letterbox(uhd, (1281, 721)), area(0, 0, 1281, 721));
        assert_eq!(letterbox(uhd, (1000, 1001)), area(0, 219, 1000, 563));
        assert_eq!(letterbox((1920, 1080), (3001, 1000)), area(611, 0, 1778, 1000));
        // nothing to keep: the client as it is
        assert_eq!(letterbox((0, 0), (800, 600)), area(0, 0, 800, 600));
        assert_eq!(letterbox(uhd, (0, 0)), area(0, 0, 0, 0));
        assert_eq!(letterbox(uhd, (1, 5000)), area(0, 2499, 1, 1));
    }

    #[test]
    fn fullscreen_covers_the_monitor_without_a_frame() {
        assert_eq!(fullscreen_rect(RECT { left: 0, top: 0, right: 2560, bottom: 1440 }), (0, 0, 2560, 1440));
        // a monitor left of / above the primary one
        assert_eq!(fullscreen_rect(RECT { left: -1920, top: -1080, right: 0, bottom: 0 }), (-1920, -1080, 1920, 1080));
        let s = fullscreen_style((WS_OVERLAPPEDWINDOW | WS_VISIBLE | WS_CLIPCHILDREN).0);
        assert_eq!(s, (WS_POPUP | WS_VISIBLE | WS_CLIPCHILDREN).0);
    }

    #[test]
    fn alt_enter_and_f11_toggle_plain_enter_does_not() {
        assert_eq!(key(WM_SYSKEYDOWN, VK_RETURN, KF_ALT | 1), Some(Key::Toggle));
        assert_eq!(key(WM_KEYDOWN, VK_F11, 1), Some(Key::Toggle));
        assert_eq!(key(WM_SYSKEYDOWN, VK_F11, KF_ALT | 1), Some(Key::Toggle));
        // held down: one toggle, the repeats are eaten
        assert_eq!(key(WM_SYSKEYDOWN, VK_RETURN, KF_ALT | KF_REPEAT | 1), Some(Key::Swallow));
        assert_eq!(key(WM_KEYDOWN, VK_F11, KF_REPEAT | 1), Some(Key::Swallow));
        // the character of Alt+Enter (DefWindowProc would beep)
        assert_eq!(key(WM_SYSCHAR, 0x0D, KF_ALT | 1), Some(Key::Swallow));
        // plain Enter, other keys, key-ups: not ours
        assert_eq!(key(WM_KEYDOWN, VK_RETURN, 1), None);
        assert_eq!(key(WM_CHAR, 0x0D, 1), None);
        assert_eq!(key(WM_SYSKEYDOWN, 0x73, KF_ALT | 1), None); // Alt+F4
        assert_eq!(key(WM_KEYUP, VK_F11, 1), None);
        assert_eq!(key(WM_SYSKEYUP, VK_RETURN, KF_ALT | 1), None);
    }

    /// A real (hidden) window: fullscreen covers its monitor without a frame, and the toggle
    /// back restores its style and rectangle.
    #[test]
    fn toggle_enters_and_restores_a_window() {
        unsafe {
            let style = WS_OVERLAPPEDWINDOW;
            let Ok(hwnd) = CreateWindowExW(WINDOW_EX_STYLE(0), windows::core::w!("STATIC"), windows::core::w!("xcl window_mode test"), style, 100, 100, 640, 480, None, None, None, None) else {
                crate::gpu::recomp::runner::skip_test("no desktop to create a window on");
                return;
            };
            let rect = |h| {
                let mut r = RECT::default();
                GetWindowRect(h, &mut r).unwrap();
                r
            };
            let before = rect(hwnd);
            let mut mi = MONITORINFO { cbSize: std::mem::size_of::<MONITORINFO>() as u32, ..Default::default() };
            assert!(GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mut mi).as_bool());

            toggle(hwnd);
            assert!(is_fullscreen(hwnd));
            let s = GetWindowLongPtrW(hwnd, GWL_STYLE) as u32;
            assert!(s & WS_POPUP.0 != 0 && s & (WS_CAPTION.0 | WS_THICKFRAME.0) == 0, "style {s:#x}");
            assert_eq!(rect(hwnd), mi.rcMonitor);
            assert!(!IsWindowVisible(hwnd).as_bool(), "the toggle shows no hidden window");

            toggle(hwnd);
            assert!(!is_fullscreen(hwnd));
            assert_eq!(GetWindowLongPtrW(hwnd, GWL_STYLE) as u32 & style.0, style.0);
            assert_eq!(rect(hwnd), before);
            assert!(!IsWindowVisible(hwnd).as_bool());
            let _ = DestroyWindow(hwnd);
        }
    }

    /// The area filter into the centred rectangle of a client of another shape: the bars are
    /// opaque black (cleared – a new target is all zeros, alpha too), the image area is the area
    /// filter of the whole image into that rectangle.
    #[test]
    fn letterboxed_draw_has_black_bars_and_the_filtered_image() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp = CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>());
                if warp.is_err() || D3D12CreateDevice(&warp.unwrap(), D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            if crate::gpu::recomp::dxc::location().is_err() {
                crate::gpu::recomp::runner::skip_test("no DXC");
                return;
            }
            let scaler = Scaler::new(&dev, DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
            let (sw, sh) = (384u32, 216u32);
            let src = pattern(sw as usize, sh as usize);
            // a taller client (bars top and bottom) and a wider one (left and right, odd offset)
            for (cw, ch, want) in [(256u32, 200u32, area(0, 28, 256, 144)), (320, 120, area(53, 0, 213, 120))] {
                let a = letterbox((sw, sh), (cw, ch));
                assert_eq!(a, want);
                let got = draw(&dev, &scaler, &src, sw, sh, cw, ch, a);
                let filtered = reference(&src, sw as usize, sh as usize, a.w as usize, a.h as usize);
                for y in 0..ch {
                    for x in 0..cw {
                        let px = &got[((y * cw + x) * 4) as usize..][..4];
                        let inside = (a.x..a.x + a.w).contains(&x) && (a.y..a.y + a.h).contains(&y);
                        if !inside {
                            assert_eq!(px, [0, 0, 0, 255], "{cw}x{ch}: bar pixel ({x}, {y})");
                            continue;
                        }
                        let w = filtered[((y - a.y) * a.w + (x - a.x)) as usize];
                        for c in 0..3 {
                            assert!((px[c] as f64 - w[c]).abs() <= 1.5, "{cw}x{ch}: pixel ({x}, {y}) is {px:?}, the area filter gives {w:?}");
                        }
                    }
                }
            }
        }
    }
}
