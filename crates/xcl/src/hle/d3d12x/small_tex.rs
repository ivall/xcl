//! Small sampled textures of the title that get no shared pages (`vram::place_pages` refuses them:
//! a 1D texture cannot be a reserved resource) on 4 KiB placements in heaps of XCL's own
//! (`XCL_SMALL_TEXTURES=1`), instead of a committed resource each: a committed resource takes at
//! least 64 KiB of video memory, and a title's thousands of tiny lookup textures (a few hundred
//! bytes each) took ~300 MiB that way.
//!
//! Only what the PC itself allows: no render target, depth or UAV flags, one sample, and
//! `GetResourceAllocationInfo` giving `D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT` (4 KiB) for the
//! description asking for it (its most detailed level fits in 64 KiB). Anything else, and a
//! texture the heaps here cannot take, is made committed as before.
//!
//! A texture's units are free again once its last reference goes (`gone`, where `late_release`
//! lets it go: the queues are past the work submitted before; with `XCL_LATE_RELEASE=0` nothing
//! is placed here). The heaps stay (4 MiB each).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

/// a placement unit: the small-resource alignment
const UNIT: u64 = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT as u64;
/// one heap of XCL's own
const HEAP_BYTES: u64 = 4 << 20;
const UNITS: usize = (HEAP_BYTES / UNIT) as usize;

/// Whether small sampled textures go on 4 KiB placements (`XCL_SMALL_TEXTURES=1`).
pub(super) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SMALL_TEXTURES").is_ok_and(|v| v.trim() == "1"))
}

/// The units of one heap: first fit of a run of `n`.
struct Units(Vec<bool>);

impl Units {
    fn new() -> Self {
        Units(vec![false; UNITS])
    }

    fn take(&mut self, n: usize) -> Option<usize> {
        let mut run = 0;
        for i in 0..self.0.len() {
            run = if self.0[i] { 0 } else { run + 1 };
            if run == n {
                let first = i + 1 - n;
                self.0[first..=i].iter_mut().for_each(|u| *u = true);
                return Some(first);
            }
        }
        None
    }

    fn give(&mut self, first: usize, n: usize) {
        self.0[first..first + n].iter_mut().for_each(|u| *u = false);
    }
}

struct Pool {
    /// the device the heaps are of
    dev: usize,
    heaps: Vec<(ID3D12Heap, Units)>,
    /// live texture → (heap, first unit, units)
    live: HashMap<usize, (usize, usize, usize)>,
    units_live: usize,
}

static POOL: Mutex<Option<Pool>> = Mutex::new(None);

/// The units `desc` takes on a 4 KiB placement, or None when the PC does not place it so.
pub(super) unsafe fn units_for(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> Option<usize> {
    let shut = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || desc.Dimension == D3D12_RESOURCE_DIMENSION_UNKNOWN || (desc.Flags & shut).0 != 0 || desc.SampleDesc.Count != 1 {
        return None;
    }
    let mut small = *desc;
    small.Alignment = UNIT;
    let info = dev.GetResourceAllocationInfo(0, &[small]);
    (info.Alignment == UNIT && info.SizeInBytes != u64::MAX && info.SizeInBytes > 0 && info.SizeInBytes <= D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64)
        .then(|| info.SizeInBytes.div_ceil(UNIT) as usize)
}

/// A sampled texture `desc` on a 4 KiB placement in a heap of XCL's own (device `this`), made as
/// `CreatePlacedResource` (adopted as any resource); None when it is not placed so (the caller makes
/// it committed).
pub(super) unsafe fn place(this: *mut c_void, desc: &D3D12_RESOURCE_DESC, state: u32, clear: *const c_void, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    // (with `XCL_LATE_RELEASE=0` a texture's units would be free at the title's release, while an
    // upload of XCL's into it can still be in flight: committed then)
    if !enabled() {
        return None;
    }
    if !super::late_release::enabled() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::xlog_warn!("D3D12.X: XCL_SMALL_TEXTURES=1 does nothing with XCL_LATE_RELEASE=0: small textures stay committed"));
        return None;
    }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    let n = units_for(dev, desc)?;
    let mut small = *desc;
    small.Alignment = UNIT;
    // the units first (the heap stays while the pool does); the resource is made outside the lock
    let (heap_raw, heap_index, first) = {
        let mut g = POOL.lock().unwrap();
        let pool = g.get_or_insert_with(|| Pool { dev: this as usize, heaps: Vec::new(), live: HashMap::new(), units_live: 0 });
        if pool.dev != this as usize {
            return None;
        }
        let found = pool.heaps.iter_mut().enumerate().find_map(|(k, (h, u))| u.take(n).map(|first| (h.as_raw() as usize, k, first)));
        match found {
            Some(f) => f,
            None => {
                let hd = D3D12_HEAP_DESC {
                    SizeInBytes: HEAP_BYTES,
                    Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    Alignment: 0,
                    Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
                };
                let mut heap: Option<ID3D12Heap> = None;
                if dev.CreateHeap(&hd, &mut heap).is_err() {
                    return None;
                }
                let heap = heap?;
                let mut units = Units::new();
                let first = units.take(n)?;
                let raw = heap.as_raw() as usize;
                pool.heaps.push((heap, units));
                let k = pool.heaps.len();
                if k <= 4 || k.is_power_of_two() {
                    // (XCL asks no residency priority for these heaps or for a committed texture,
                    // with or without XCL_VRAM_HOT_PRIORITY)
                    crate::xlog_info!("D3D12.X: small textures: heap {k} of 4 MiB made, residency priority: default, no override (as for a committed texture)");
                }
                (raw, k - 1, first)
            }
        }
    };
    let hr = super::real::dev_create_placed_resource(this, heap_raw, first * UNIT as usize, &small as *const _ as usize, state as usize, clear as usize, riid, ppv);
    let mut g = POOL.lock().unwrap();
    let pool = g.as_mut().unwrap();
    if hr < 0 || (*ppv).is_null() {
        pool.heaps[heap_index].1.give(first, n);
        static FAILED: AtomicU64 = AtomicU64::new(0);
        if FAILED.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: a small texture's 4 KiB placement failed ({hr:#x}); it is made committed");
        }
        return None;
    }
    pool.live.insert(*ppv as usize, (heap_index, first, n));
    pool.units_live += n;
    static PLACED: AtomicU64 = AtomicU64::new(0);
    let k = PLACED.fetch_add(1, Ordering::Relaxed) + 1;
    if k.is_power_of_two() && k >= 64 {
        crate::xlog_info!(
            "D3D12.X: {k} small textures on 4 KiB placements so far; {} live in {} KiB of {} heaps of 4 MiB (committed: at least {} KiB)",
            pool.live.len(),
            pool.units_live * UNIT as usize >> 10,
            pool.heaps.len(),
            pool.live.len() * 64
        );
    }
    Some(hr)
}

/// The texture `res` is destroyed next: its units are free.
pub(super) fn gone(res: usize) {
    if !enabled() {
        return;
    }
    let mut g = POOL.lock().unwrap();
    if let Some(pool) = g.as_mut() {
        if let Some((k, first, n)) = pool.live.remove(&res) {
            pool.heaps[k].1.give(first, n);
            pool.units_live -= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_SAMPLE_DESC};

    /// Units go first fit and come back: a freed run is taken again, a run longer than any gap is not.
    #[test]
    fn units_are_taken_first_fit_and_given_back() {
        let mut u = Units::new();
        assert_eq!(u.take(1), Some(0));
        assert_eq!(u.take(4), Some(1));
        assert_eq!(u.take(1), Some(5));
        u.give(1, 4);
        assert_eq!(u.take(2), Some(1));
        assert_eq!(u.take(3), Some(6), "the 2-unit gap left at 3 is too small");
        assert_eq!(u.take(2), Some(3));
        let mut full = Units::new();
        assert_eq!(full.take(UNITS), Some(0));
        assert_eq!(full.take(1), None);
        full.give(UNITS - 2, 2);
        assert_eq!(full.take(3), None);
        assert_eq!(full.take(2), Some(UNITS - 2));
    }

    fn tex(dim: D3D12_RESOURCE_DIMENSION, w: u64, h: u32, format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: dim,
            Alignment: 0,
            Width: w,
            Height: h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: flags,
        }
    }

    /// The PC's rules decide: a tiny 1D or 2D sampled texture is placed on 4 KiB, a render target
    /// or a texture over 64 KiB is not, and a buffer is never.
    #[test]
    fn only_textures_the_pc_places_on_4_kib_are_taken() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev) }.is_err() {
            return crate::gpu::recomp::runner::skip_test("no D3D12 device");
        }
        let dev = dev.unwrap();
        unsafe {
            // 256 texels of RGBA8: 1 KiB of texels, on a 4 KiB placement of the driver's size (8 KiB
            // on an RTX 5070 Ti with driver 610.74)
            let n = units_for(&dev, &tex(D3D12_RESOURCE_DIMENSION_TEXTURE1D, 256, 1, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, D3D12_RESOURCE_FLAG_NONE));
            assert!(n.is_some_and(|n| (1..=16).contains(&n)), "{n:?}");
            // 64x64 RGBA8: 16 KiB
            let n = units_for(&dev, &tex(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE));
            assert!(n.is_some_and(|n| (4..=16).contains(&n)), "{n:?}");
            assert_eq!(units_for(&dev, &tex(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)), None);
            assert_eq!(units_for(&dev, &tex(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 512, 512, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE)), None);
            let mut buf = tex(D3D12_RESOURCE_DIMENSION_BUFFER, 4096, 1, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE);
            buf.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            assert_eq!(units_for(&dev, &buf), None);
        }
    }
}
