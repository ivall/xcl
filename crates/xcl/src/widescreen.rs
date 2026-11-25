//! The title rendered at another size than the console's 16:9 output (a 21:9 screen).
//!
//! A console title picks its output from a fixed list of modes (on the console the display is
//! always 16:9). RAGE titles (the newer retail game) keep that list as a table in their
//! read-only data: `{width, height, refresh, 1}` × 5 – 1280x720, 1664x936, 1920x1080,
//! 2560x1440 and 3840x2160, each at 60 Hz – and size their back buffers and render targets from
//! the largest.
//! With `render_resolution` set, three things are patched when the executable is mapped, each
//! found by its contents (not an address: another build of the title is patched too, or left
//! alone with a warning):
//! - the largest mode becomes the render size (`patch`);
//! - a debug break that fires when the render-target arena is more than 2 MB larger than the
//!   4K targets need becomes a nop (`patch_slack_check`);
//! - the loads of the title's output size (3840x2160, the console's HDMI output) read the render
//!   size, so its UI is laid out and drawn for the same screen (`patch_output_size`). The UI then
//!   anchors to the wider screen's edges, unstretched.
//!
//! The render-target arena does not grow: the size must take no more memory than 3840x2160 in
//! 128x128 tiles (`fits`); "screen" picks the largest such size of the monitor's aspect ratio
//! (4224x1792 for 21:9).

/// The table: 5 modes of 4 u32 each.
const MODES: [[u32; 4]; 5] = [[1280, 720, 60, 1], [1664, 936, 60, 1], [1920, 1080, 60, 1], [2560, 1440, 60, 1], [3840, 2160, 60, 1]];

/// The pixels of the title's largest mode in 128x128 tiles (3840x2176), which its render-target
/// memory is sized for.
pub const BUDGET: u64 = 3840 * 2176;

/// `w`x`h` in whole 128x128 tiles fits the title's render-target memory.
pub fn fits(w: u32, h: u32) -> bool {
    (w as u64).div_ceil(128) * 128 * (h as u64).div_ceil(128) * 128 <= BUDGET
}

fn table_bytes() -> Vec<u8> {
    MODES.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
}

/// `render_resolution` (`XCL_RENDER_RES` overrides): "" / "native" (the title's own), "WxH", or
/// "screen" (the monitor's aspect ratio at the title's 2160 lines). None: no change.
pub fn render_size(setting: &str, screen: Option<(u32, u32)>) -> Option<(u32, u32)> {
    let s = setting.trim().to_ascii_lowercase();
    if s.is_empty() || s == "native" || s == "off" {
        return None;
    }
    if s == "screen" {
        let (sw, sh) = screen?;
        if sw == 0 || sh == 0 || sw as u64 * 9 == sh as u64 * 16 {
            return None;
        }
        // The screen's aspect ratio in no more memory than 3840x2160: the title's render-target
        // memory is sized for that (5120x2160 and 4416x1864 overran it). Targets are laid out in
        // 128x128 tiles, so both sides are multiples of 128 (no padding, half and quarter sizes
        // too); the largest such size within 2 % of the screen's aspect ratio.
        let aspect = sw as f64 / sh as f64;
        let mut h = 2176u64;
        while h >= 512 {
            let w = ((h as f64 * aspect / 128.0).round() as u64 * 128).max(128);
            if w * h <= BUDGET && ((w as f64 / h as f64) / aspect - 1.0).abs() < 0.02 {
                return Some((w as u32, h as u32));
            }
            h -= 128;
        }
        return None;
    }
    let (w, h) = s.split_once('x')?;
    let (w, h) = (w.trim().parse::<u32>().ok()?, h.trim().parse::<u32>().ok()?);
    ((256..=16384).contains(&w) && (256..=16384).contains(&h)).then_some((w, h))
}

/// Offsets of the mode table in `data`.
pub fn find_table(data: &[u8]) -> Vec<usize> {
    let t = table_bytes();
    data.windows(t.len()).enumerate().filter(|(_, w)| *w == t.as_slice()).map(|(i, _)| i).collect()
}

/// In `data` (a mapped section, writable): the largest mode becomes `size`. Returns the offsets
/// of the tables changed.
pub fn patch(data: &mut [u8], size: (u32, u32)) -> Vec<usize> {
    let at = find_table(data);
    for &i in &at {
        let e = i + 16 * (MODES.len() - 1);
        data[e..e + 4].copy_from_slice(&size.0.to_le_bytes());
        data[e + 4..e + 8].copy_from_slice(&size.1.to_le_bytes());
    }
    at
}

/// After the title allocates its full-size render targets from a 64 MB (82 MB) arena it checks
/// that less than 2 MB of it is left (an arena larger than needed) with a debug break:
/// `mov rax, [rsp+78h]; sub rax, [rsp+70h]; add rax, [rip+x]; cmp rax, 1FFFFFh; jbe +1; int3`.
/// At another render size more is left over: the break becomes a nop (the code after it is the
/// check's normal path).
const SLACK_CHECK: [&[u8]; 2] = [&[0x48, 0x8B, 0x44, 0x24, 0x78, 0x48, 0x2B, 0x44, 0x24, 0x70, 0x48, 0x03, 0x05], &[0x48, 0x3D, 0xFF, 0xFF, 0x1F, 0x00, 0x76, 0x01, 0xCC]];

/// In `code` (executable, writable): the arena-slack break made a nop. Returns the offsets of
/// the breaks changed.
pub fn patch_slack_check(code: &mut [u8]) -> Vec<usize> {
    let (head, tail) = (SLACK_CHECK[0], SLACK_CHECK[1]);
    let len = head.len() + 4 + tail.len();
    let at: Vec<usize> = (0..code.len().saturating_sub(len - 1)).filter(|&i| code[i..].starts_with(head) && code[i + head.len() + 4..].starts_with(tail)).map(|i| i + len - 1).collect();
    for &i in &at {
        code[i] = 0x90;
    }
    at
}

/// The function that gives the UI its viewport: `sub rsp, 38h; vmovsd xmm1, [rip+O]; vxorps
/// xmm0, xmm0, xmm0; lea rdx, [rsp+20h]; vcvtdq2ps xmm1, xmm1; vmovlhps xmm0, xmm0, xmm1` –
/// (0, 0, output width, output height), O the title's output size (u32 width, height).
const UI_VIEWPORT: [&[u8]; 2] = [&[0x48, 0x83, 0xEC, 0x38, 0xC5, 0xFB, 0x10, 0x0D], &[0xC5, 0xF8, 0x57, 0xC0, 0x48, 0x8D, 0x54, 0x24, 0x20, 0xC5, 0xF8, 0x5B, 0xC9, 0xC5, 0xF8, 0x16, 0xC1]];

/// Loads with a RIP-relative operand (opcode bytes before the disp32) that read the output size.
const LOADS: [&[u8]; 6] = [
    &[0xC5, 0xFB, 0x10, 0x0D], // vmovsd xmm1, [rip+d]
    &[0x48, 0x8D, 0x05],       // lea rax, [rip+d]
    &[0x48, 0x8D, 0x15],       // lea rdx, [rip+d]
    &[0x44, 0x8B, 0x05],       // mov r8d, [rip+d]
    &[0x44, 0x8B, 0x0D],       // mov r9d, [rip+d]
    &[0x8B, 0x05],             // mov eax, [rip+d]
];

fn disp_target(code: &[u8], code_rva: usize, at: usize) -> usize {
    let d = i32::from_le_bytes(code[at..at + 4].try_into().unwrap());
    (code_rva + at + 4).wrapping_add(d as isize as usize)
}

/// The title keeps two sizes: the render size (`SetRenderSize(w, h)`: `mov [rip+R], ecx; mov
/// [rip+R+4], edx; ret`), which follows the mode table, and the output size at R+0x20 – the
/// console's HDMI output, 3840x2160 – that the UI's viewport, an output-sized target and the
/// inverse screen size are read from. With a patched mode the UI was laid out for the render
/// size and drawn as if the screen were 3840x2160 (scaled by render / 4K twice, off the screen).
/// Here those loads read the render size instead (disp32 - 0x20). `code`: the executable section
/// at `code_rva`. Returns the offsets of the loads changed (empty when the signature is missing
/// or the setter does not match).
pub fn patch_output_size(code: &mut [u8], code_rva: usize) -> Vec<usize> {
    let (head, tail) = (UI_VIEWPORT[0], UI_VIEWPORT[1]);
    let sig: Vec<usize> = (0..code.len().saturating_sub(head.len() + 4 + tail.len())).filter(|&i| code[i..].starts_with(head) && code[i + head.len() + 4..].starts_with(tail)).collect();
    let [ui] = sig[..] else { return Vec::new() };
    let output = disp_target(code, code_rva, ui + head.len());
    let render = output.wrapping_sub(0x20);
    // the setter of the render size, 0x20 below
    let setter = (0..code.len().saturating_sub(13)).any(|i| code[i] == 0x89 && code[i + 1] == 0x0D && code[i + 6] == 0x89 && code[i + 7] == 0x15 && code[i + 12] == 0xC3 && disp_target(code, code_rva, i + 2) == render && disp_target(code, code_rva, i + 8) == render + 4);
    if !setter {
        return Vec::new();
    }
    let mut changed: Vec<usize> = Vec::new();
    let mut done_at = usize::MAX;
    for i in 0..code.len().saturating_sub(8) {
        if !matches!(code[i], 0xC5 | 0x48 | 0x44 | 0x8B) {
            continue;
        }
        for op in LOADS {
            if code[i..].starts_with(op) {
                let at = i + op.len();
                // `8B 05` inside `44 8B 05`: the same operand
                if at == done_at {
                    continue;
                }
                let t = disp_target(code, code_rva, at);
                if t == output || t == output + 4 {
                    done_at = at;
                    let d = i32::from_le_bytes(code[at..at + 4].try_into().unwrap()) - 0x20;
                    code[at..at + 4].copy_from_slice(&d.to_le_bytes());
                    changed.push(i);
                }
            }
        }
    }
    changed
}

/// The title's executable at `base` was mapped (still writable): with a render size set, its
/// mode table and arena check are patched. `sections`: (rva, length, executable).
pub unsafe fn patch_image(base: usize, sections: &[(u32, usize, bool)]) {
    let env = std::env::var("XCL_RENDER_RES").ok();
    let setting = env.as_deref().unwrap_or(&crate::state::config().render_resolution);
    let Some(size) = render_size(setting, crate::hle::d3d12x::present::screen_size()) else { return };
    let (mut tables, mut breaks, mut outputs) = (0, 0, 0);
    for &(rva, len, exec) in sections {
        let data = std::slice::from_raw_parts_mut((base + rva as usize) as *mut u8, len);
        if exec {
            for off in patch_slack_check(data) {
                breaks += 1;
                crate::xlog_info!("render size: the title's render-target arena check at +{:#x} no longer breaks", rva as usize + off);
            }
            let loads = patch_output_size(data, rva as usize);
            outputs += loads.len();
            if !loads.is_empty() {
                let at: Vec<String> = loads.iter().map(|o| format!("+{:#x}", rva as usize + o)).collect();
                crate::xlog_info!("render size: the title's UI and output-sized targets use the render size ({} loads: {})", loads.len(), at.join(" "));
            }
            continue;
        }
        for off in patch(data, size) {
            tables += 1;
            crate::xlog_info!("render size: the title's largest display mode (3840x2160, table at +{:#x}) is now {}x{}", rva as usize + off, size.0, size.1);
        }
    }
    if tables == 0 {
        crate::xlog_warn!("render size {}x{}: the title has no display-mode table XCL knows – it renders at its own size", size.0, size.1);
    } else {
        if breaks == 0 {
            crate::xlog_warn!("render size: the title's render-target arena check was not found – a size other than 3840x2160 may stop it");
        }
        if outputs == 0 {
            crate::xlog_warn!("render size: the title's output size was not found – its UI is drawn for 3840x2160 and lands in the wrong place");
        }
        if !fits(size.0, size.1) {
            crate::xlog_warn!("render size {}x{} needs more memory than 3840x2160 (in 128x128 tiles): the title's render-target arena overflows – try render_resolution = \"screen\"", size.0, size.1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_size_settings() {
        assert_eq!(render_size("", None), None);
        assert_eq!(render_size("native", Some((5120, 2160))), None);
        assert_eq!(render_size("5120x2160", None), Some((5120, 2160)));
        assert_eq!(render_size(" 3440 x 1440 ", None), Some((3440, 1440)));
        assert_eq!(render_size("10x10", None), None);
        assert_eq!(render_size("wide", None), None);
        // 21:9 screens: their aspect ratio within 3840x2160's pixels
        assert_eq!(render_size("screen", Some((5120, 2160))), Some((4224, 1792)));
        assert_eq!(render_size("screen", Some((3440, 1440))), Some((4224, 1792)));
        assert_eq!(render_size("screen", Some((2560, 1080))), Some((4224, 1792)));
        // 32:9, 16:10
        assert_eq!(render_size("screen", Some((5120, 1440))), Some((4992, 1408)));
        assert_eq!(render_size("screen", Some((1920, 1200))), Some((3456, 2176)));
        for s in [(5120, 2160), (3440, 1440), (2560, 1080), (5120, 1440), (1920, 1200)] {
            let (w, h) = render_size("screen", Some(s)).unwrap();
            assert!(fits(w, h) && w % 128 == 0 && h % 128 == 0, "{s:?}: {w}x{h}");
        }
        assert!(fits(3840, 2160) && fits(4224, 1792) && !fits(4416, 1864) && !fits(5120, 2160));
        // a 16:9 screen: nothing to change
        assert_eq!(render_size("screen", Some((3840, 2160))), None);
        assert_eq!(render_size("screen", None), None);
    }

    #[test]
    fn patch_rewrites_the_largest_mode_only() {
        let mut data = vec![0xAAu8; 40];
        data.extend(table_bytes());
        data.extend([0xBB; 12]);
        let before = data.clone();
        assert_eq!(patch(&mut data, (5120, 2160)), vec![40]);
        let e = 40 + 64;
        assert_eq!(&data[e..e + 8], [5120u32.to_le_bytes(), 2160u32.to_le_bytes()].concat().as_slice());
        // everything else as it was, the patched table no longer matches
        assert_eq!(data[..e], before[..e]);
        assert_eq!(data[e + 8..], before[e + 8..]);
        assert!(find_table(&data).is_empty());
        let mut other = vec![0u8; 200];
        assert!(patch(&mut other, (5120, 2160)).is_empty());
    }

    #[test]
    fn slack_check_break_becomes_a_nop() {
        let mut code = vec![0xCCu8; 7];
        code.extend(SLACK_CHECK[0]);
        code.extend([0xCB, 0xC0, 0x16, 0x05]);
        code.extend(SLACK_CHECK[1]);
        code.extend([0x48, 0x8D]);
        let before = code.clone();
        let at = patch_slack_check(&mut code);
        assert_eq!(at, vec![7 + 13 + 4 + 8]);
        assert_eq!(code[at[0]], 0x90);
        assert_eq!(code[..at[0]], before[..at[0]]);
        assert_eq!(code[at[0] + 1..], before[at[0] + 1..]);
        // another limit: left alone
        let mut other = before.clone();
        other[7 + 13 + 4 + 4] = 0x2F;
        assert!(patch_slack_check(&mut other).is_empty());
    }
    /// A RIP-relative instruction at `at` in `code` (section at `rva`) whose disp32 (after `op`)
    /// points at `target`.
    fn rip(code: &mut Vec<u8>, rva: usize, op: &[u8], target: usize, tail: &[u8]) -> usize {
        let at = code.len();
        code.extend(op);
        let next = rva + code.len() + 4;
        code.extend(((target as isize - next as isize) as i32).to_le_bytes());
        code.extend(tail);
        at
    }

    #[test]
    fn output_size_loads_read_the_render_size() {
        let rva = 0x1000;
        let (render, output) = (0x9000usize, 0x9020usize);
        let mut code = vec![0xCC; 16];
        // SetRenderSize: mov [rip+R], ecx; mov [rip+R+4], edx; ret
        rip(&mut code, rva, &[0x89, 0x0D], render, &[]);
        rip(&mut code, rva, &[0x89, 0x15], render + 4, &[0xC3]);
        // the UI viewport function
        let ui = rip(&mut code, rva, UI_VIEWPORT[0], output, UI_VIEWPORT[1]);
        let a = rip(&mut code, rva, &[0x44, 0x8B, 0x0D], output + 4, &[]); // mov r9d, [height]
        let b = rip(&mut code, rva, &[0x44, 0x8B, 0x05], output, &[]); // mov r8d, [width]
        let c = rip(&mut code, rva, &[0x48, 0x8D, 0x05], output, &[]); // lea rax, [width]
        let other = rip(&mut code, rva, &[0x44, 0x8B, 0x05], output + 8, &[0xCC, 0xCC]); // not the size
        let before = code.clone();
        let mut got = patch_output_size(&mut code, rva);
        got.sort();
        assert_eq!(got, vec![ui + 4, a, b, c]);
        for (at, len, to) in [(ui + 4, 4, render), (a, 3, render + 4), (b, 3, render), (c, 3, render)] {
            assert_eq!(disp_target(&code, rva, at + len), to, "load at {at}");
        }
        assert_eq!(code[other..], before[other..], "other loads stay");
        // without the setter 0x20 below, nothing changes
        let mut no_setter = before.clone();
        no_setter[16] = 0x90;
        assert!(patch_output_size(&mut no_setter, rva).is_empty());
        assert_eq!(no_setter[17..], before[17..]);
    }

}
