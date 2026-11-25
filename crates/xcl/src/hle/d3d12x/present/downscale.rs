//! The title's image shrunk to the XCL window's client area by XCL itself.
//!
//! A 3840x2160 frame copied into a swap chain of its own size is shrunk by DWM with a single
//! bilinear tap per window pixel: at 3x (1280x720) or 1.5x it skips texels, and thin strokes and
//! text stair-step. Here the swap chain is the client size instead, and a full-screen triangle's
//! pixel shader averages the whole footprint of each window pixel in the game image (an area /
//! box filter, exact partial-texel weights), in linear light.
//!
//! Linear light: 8- and 10-bit UNORM images are decoded with the sRGB curve before averaging
//! and encoded again after – the swap chain has the default (sRGB) colour space, so that is how
//! the window shows those values; an _SRGB source is decoded by its view. FP16 is scRGB, linear
//! already.

use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// How the title's image reaches the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// copied into a swap chain of the image's size (the client area is exactly as large; and
    /// without the filter, where DWM stretches it)
    Copy,
    /// copied 1:1 into a swap chain of the client size at its place, black around it (a 16:9
    /// image on a 21:9 screen of the same height)
    Place,
    /// drawn into a swap chain of the client size through the area filter, at its place
    Filter,
    /// not shown: the client area is empty (minimized), the swap chain keeps its size
    Skip,
}

/// How the image fills the client area (`scaling` in xcl.toml, `XCL_SCALING` overrides).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scaling {
    /// as large as fits with its aspect ratio kept, centred, black bars around it
    Fit,
    /// the largest whole multiple of the image that fits (1:1 pixels at 1x), else as `Fit`
    Integer,
    /// the whole client area, aspect ratio not kept
    Stretch,
}

impl Scaling {
    pub fn parse(s: &str) -> Option<Scaling> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fit" | "aspect" | "auto" | "" => Some(Scaling::Fit),
            "integer" | "pixel" => Some(Scaling::Integer),
            "stretch" | "fill" => Some(Scaling::Stretch),
            _ => None,
        }
    }
}

/// Where the image goes in the client area: (x, y, width, height).
pub fn place(image: (u32, u32), client: (u32, u32), scaling: Scaling) -> (u32, u32, u32, u32) {
    let ((iw, ih), (cw, ch)) = (image, client);
    if iw == 0 || ih == 0 || cw == 0 || ch == 0 {
        return (0, 0, cw, ch);
    }
    let (w, h) = match scaling {
        Scaling::Stretch => (cw, ch),
        Scaling::Integer if (cw / iw).min(ch / ih) >= 1 => {
            let k = (cw / iw).min(ch / ih);
            (iw * k, ih * k)
        }
        _ => {
            let (iw, ih, cw, ch) = (iw as u64, ih as u64, cw as u64, ch as u64);
            if cw * ih > ch * iw {
                // wider than the image: bars left and right
                (((ch * iw + ih / 2) / ih).clamp(1, cw) as u32, ch as u32)
            } else {
                (cw as u32, ((cw * ih + iw / 2) / iw).clamp(1, ch) as u32)
            }
        }
    };
    ((cw - w) / 2, (ch - h) / 2, w, h)
}

/// `Skip` when the client area is empty (minimized); `Copy` when the image fills it 1:1;
/// `Place` when it goes there 1:1 with bars around it; else `Filter`.
pub fn route(image: (u32, u32), client: (u32, u32), scaling: Scaling) -> Route {
    if client.0 == 0 || client.1 == 0 {
        return Route::Skip;
    }
    let (_, _, w, h) = place(image, client, scaling);
    if (w, h) != image {
        Route::Filter
    } else if client == image {
        Route::Copy
    } else {
        Route::Place
    }
}

const SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), DescriptorTable(SRV(t0), visibility=SHADER_VISIBILITY_PIXEL)"
Texture2D<float4> src : register(t0);
cbuffer C : register(b0) { uint2 dst; uint flags; uint pad; uint2 origin; };

[RootSignature(RS)]
float4 vs(uint id : SV_VertexID) : SV_Position {
    float2 t = float2((id << 1) & 2, id & 2);
    return float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
}

float3 decode(float3 c) { return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4)); }
float3 encode(float3 c) { return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055); }

// the mean of the texels under this window pixel, each weighted by the area it covers
[RootSignature(RS)]
float4 ps(float4 pos : SV_Position) : SV_Target {
    uint sw, sh;
    src.GetDimensions(sw, sh);
    float2 s = float2(sw, sh) / float2(dst);
    float2 lo = (floor(pos.xy) - float2(origin)) * s;
    float2 hi = min(lo + s, float2(sw, sh));
    int2 a = (int2)floor(lo);
    int2 b = (int2)ceil(hi);
    float4 sum = 0;
    float total = 0;
    for (int y = a.y; y < b.y; y++) {
        float wy = min(hi.y, y + 1) - max(lo.y, y);
        for (int x = a.x; x < b.x; x++) {
            float w = wy * (min(hi.x, x + 1) - max(lo.x, x));
            float4 c = src.Load(int3(x, y, 0));
            if (flags & 1) c.rgb = decode(saturate(c.rgb));
            sum += c * w;
            total += w;
        }
    }
    float4 r = sum / max(total, 1e-6);
    if (flags & 2) r.rgb = encode(saturate(r.rgb));
    return r;
}
"#;

/// The area-filter pipeline for one back buffer format, with its two descriptors.
pub struct Scaler {
    pub fmt: DXGI_FORMAT,
    rs: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    srv: ID3D12DescriptorHeap,
    rtv: ID3D12DescriptorHeap,
}

/// The SRV format for a source of `f` (a typeless one is read as UNORM / FLOAT, like the copy).
fn view_format(f: DXGI_FORMAT) -> DXGI_FORMAT {
    match f {
        DXGI_FORMAT_R8G8B8A8_TYPELESS => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_TYPELESS => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_TYPELESS => DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        f => f,
    }
}

impl Scaler {
    /// The pipeline drawing into a `fmt` render target on `dev` (Err: no DXC, or the device
    /// refused it).
    pub unsafe fn new(dev: &ID3D12Device, fmt: DXGI_FORMAT) -> Result<Scaler, String> {
        use crate::gpu::recomp::dxc::compile_cached;
        let (vs, _) = compile_cached(SHADER, "vs", "vs_6_0")?;
        let (ps, _) = compile_cached(SHADER, "ps", "ps_6_0")?;
        let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &ps).map_err(|e| format!("CreateRootSignature: {e}"))?;
        let mut pd = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&rs),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr() as _, BytecodeLength: ps.len() },
            SampleMask: u32::MAX,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pd.RTVFormats[0] = fmt;
        let pso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&pd).map_err(|e| format!("CreateGraphicsPipelineState: {e}"))?;
        let heap = |ty, flags| {
            dev.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: 1, Flags: flags, NodeMask: 0 })
                .map_err(|e| format!("CreateDescriptorHeap: {e}"))
        };
        let srv = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)?;
        let rtv = heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_DESCRIPTOR_HEAP_FLAG_NONE)?;
        Ok(Scaler { fmt, rs, pso, srv, rtv })
    }

    /// Record on `list` the draw of `src` (subresource 0, readable as a pixel shader resource)
    /// over all of `dst` (`w`x`h`, `self.fmt`, in RENDER_TARGET state). The descriptors are
    /// rewritten: the previous recording must have finished on the GPU.
    #[cfg(test)]
    pub unsafe fn record(&self, dev: &ID3D12Device, list: &ID3D12GraphicsCommandList, src: &ID3D12Resource, dst: &ID3D12Resource, w: u32, h: u32) {
        self.record_into(dev, list, src, dst, (0, 0, w, h), false);
    }

    /// The view of `dst` in the scaler's RTV heap (forgotten by `note_rtv` at once).
    unsafe fn target(&self, dev: &ID3D12Device, dst: &ID3D12Resource) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let rtv = self.rtv.GetCPUDescriptorHandleForHeapStart();
        dev.CreateRenderTargetView(dst, None, rtv);
        // `dev` is the title's (adopted) device: its CreateRenderTargetView hook keeps a reference
        // to the view's resource for guessing the title's frame (`note_rtv`). Held on a swap chain
        // buffer, that reference makes every ResizeBuffers fail (DXGI_ERROR_INVALID_CALL): the
        // presenter's own view is forgotten at once.
        super::note_rtv(rtv.ptr, std::ptr::null_mut());
        rtv
    }

    /// Record on `list` a clear of all of `dst` (RENDER_TARGET state) to black: the bars around
    /// a placed image.
    pub unsafe fn clear(&self, dev: &ID3D12Device, list: &ID3D12GraphicsCommandList, dst: &ID3D12Resource) {
        let rtv = self.target(dev, dst);
        list.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
    }

    /// As `record`, into the `rect` (x, y, width, height) of `dst` only; `clear`: the rest of
    /// `dst` is cleared to black first.
    pub unsafe fn record_into(&self, dev: &ID3D12Device, list: &ID3D12GraphicsCommandList, src: &ID3D12Resource, dst: &ID3D12Resource, rect: (u32, u32, u32, u32), clear: bool) {
        let (x, y, w, h) = rect;
        let view = view_format(src.GetDesc().Format);
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: view,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, ..Default::default() } },
        };
        dev.CreateShaderResourceView(src, Some(&srv_desc), self.srv.GetCPUDescriptorHandleForHeapStart());
        let rtv = self.target(dev, dst);
        if clear {
            list.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
        }
        let decode = matches!(view, DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_R10G10B10A2_UNORM);
        let encode = self.fmt != DXGI_FORMAT_R16G16B16A16_FLOAT;
        let constants = [w, h, decode as u32 | (encode as u32) << 1, 0, x, y];
        list.SetGraphicsRootSignature(&self.rs);
        list.SetDescriptorHeaps(&[Some(self.srv.clone())]);
        list.SetGraphicsRoot32BitConstants(0, constants.len() as u32, constants.as_ptr() as _, 0);
        list.SetGraphicsRootDescriptorTable(1, self.srv.GetGPUDescriptorHandleForHeapStart());
        list.SetPipelineState(&self.pso);
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: x as f32, TopLeftY: y as f32, Width: w as f32, Height: h as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: x as i32, top: y as i32, right: (x + w) as i32, bottom: (y + h) as i32 }]);
        list.OMSetRenderTargets(1, Some(&rtv as *const _), false, None);
        list.DrawInstanced(3, 1, 0, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::core::{w, Interface, PCWSTR};
    use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::WindowsAndMessaging::*;

    #[test]
    fn route_filters_only_when_the_window_is_smaller() {
        let uhd = (3840, 2160);
        for client in [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2000), (3000, 2160), (5120, 2880)] {
            assert_eq!(route(uhd, client, Scaling::Fit), Route::Filter, "{client:?}");
        }
        assert_eq!(route(uhd, (3840, 2160), Scaling::Fit), Route::Copy);
        // a 21:9 screen of the image's height: the image 1:1 between bars
        assert_eq!(route(uhd, (5120, 2160), Scaling::Fit), Route::Place);
        assert_eq!(route(uhd, (5120, 2160), Scaling::Stretch), Route::Filter);
        assert_eq!(route(uhd, (7680, 4320), Scaling::Integer), Route::Filter);
        assert_eq!(route(uhd, (4000, 2400), Scaling::Integer), Route::Place);
        // minimized: nothing shown – not a copy into a swap chain resized to the image
        for client in [(0, 0), (1280, 0), (0, 720)] {
            assert_eq!(route(uhd, client, Scaling::Fit), Route::Skip, "{client:?}");
        }
    }

    #[test]
    fn place_keeps_the_aspect_ratio() {
        let uhd = (3840, 2160);
        // 21:9 at 5120x2160: 640-pixel bars left and right
        assert_eq!(place(uhd, (5120, 2160), Scaling::Fit), (640, 0, 3840, 2160));
        assert_eq!(place(uhd, (3440, 1440), Scaling::Fit), (440, 0, 2560, 1440));
        // taller than 16:9 (16:10): bars above and below
        assert_eq!(place(uhd, (1920, 1200), Scaling::Fit), (0, 60, 1920, 1080));
        assert_eq!(place(uhd, (1280, 720), Scaling::Fit), (0, 0, 1280, 720));
        assert_eq!(place(uhd, (5120, 2160), Scaling::Stretch), (0, 0, 5120, 2160));
        // integer: 1080p twice on a 4K screen; a window smaller than the image falls back to fit
        assert_eq!(place((1920, 1080), (5120, 2160), Scaling::Integer), (640, 0, 3840, 2160));
        assert_eq!(place((1920, 1080), (3440, 1440), Scaling::Integer), (760, 180, 1920, 1080));
        assert_eq!(place(uhd, (3440, 1440), Scaling::Integer), (440, 0, 2560, 1440));
        for s in ["fit", "Integer", " stretch "] {
            assert!(Scaling::parse(s).is_some(), "{s}");
        }
        assert_eq!(Scaling::parse("zoom"), None);
    }

    fn srgb_to_linear(c: f64) -> f64 {
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    }

    fn linear_to_srgb(c: f64) -> f64 {
        if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
    }

    /// 1-pixel lines and a 1-pixel checker, the hard case for a single bilinear tap.
    fn pattern(w: usize, h: usize) -> Vec<u8> {
        let mut v = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let t = &mut v[(y * w + x) * 4..][..4];
                t[0] = if (x + y) % 2 == 0 { 255 } else { 0 };
                t[1] = if x % 7 == 0 || y % 5 == 0 { 255 } else { 16 };
                t[2] = ((x * 3 + y * 5) % 256) as u8;
                t[3] = 255;
            }
        }
        v
    }

    /// The area filter on the CPU, in linear light (sRGB curve), as 8-bit sRGB.
    fn reference(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<[f64; 3]> {
        let lut: Vec<f64> = (0..256).map(|v| srgb_to_linear(v as f64 / 255.0)).collect();
        let (sx, sy) = (sw as f64 / dw as f64, sh as f64 / dh as f64);
        let mut out = Vec::with_capacity(dw * dh);
        for oy in 0..dh {
            let (ly, hy) = (oy as f64 * sy, (oy as f64 + 1.0) * sy);
            for ox in 0..dw {
                let (lx, hx) = (ox as f64 * sx, (ox as f64 + 1.0) * sx);
                let (mut sum, mut total) = ([0.0; 3], 0.0);
                for y in ly.floor() as usize..(hy.ceil() as usize).min(sh) {
                    let wy = hy.min(y as f64 + 1.0) - ly.max(y as f64);
                    for x in lx.floor() as usize..(hx.ceil() as usize).min(sw) {
                        let wt = wy * (hx.min(x as f64 + 1.0) - lx.max(x as f64));
                        for c in 0..3 {
                            sum[c] += wt * lut[src[(y * sw + x) * 4 + c] as usize];
                        }
                        total += wt;
                    }
                }
                out.push(sum.map(|s| 255.0 * linear_to_srgb(s / total)));
            }
        }
        out
    }

    fn buffer(dev: &ID3D12Device, size: u64, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        unsafe { dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).unwrap() };
        r.unwrap()
    }

    fn texture(dev: &ID3D12Device, w: u32, h: u32, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: w as u64,
            Height: h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: flags,
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        unsafe { dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).unwrap() };
        r.unwrap()
    }

    fn copy_location(r: &ID3D12Resource, fp: Option<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>) -> D3D12_TEXTURE_COPY_LOCATION {
        unsafe {
            match fp {
                Some(fp) => D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } },
                None => D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } },
            }
        }
    }

    /// A `w`x`h` RGBA8 texture holding `src`, left in `after`: the upload is recorded on `list`
    /// (the upload buffer returned with it must live until the list ran).
    unsafe fn upload(dev: &ID3D12Device, list: &ID3D12GraphicsCommandList, src: &[u8], w: u32, h: u32, after: D3D12_RESOURCE_STATES) -> (ID3D12Resource, ID3D12Resource) {
        let tex = texture(dev, w, h, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        let (mut fp, mut size) = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(), 0u64);
        dev.GetCopyableFootprints(&tex.GetDesc(), 0, 1, 0, Some(&mut fp), None, None, Some(&mut size));
        let up = buffer(dev, size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let mut p = std::ptr::null_mut();
        up.Map(0, None, Some(&mut p)).unwrap();
        for y in 0..h as usize {
            let row = &src[y * w as usize * 4..][..w as usize * 4];
            std::ptr::copy_nonoverlapping(row.as_ptr(), (p as *mut u8).add(fp.Offset as usize + y * fp.Footprint.RowPitch as usize), row.len());
        }
        up.Unmap(0, None);
        list.CopyTextureRegion(&copy_location(&tex, None), 0, 0, 0, &copy_location(&up, Some(fp)), None);
        list.ResourceBarrier(&[super::super::transition(&tex, D3D12_RESOURCE_STATE_COPY_DEST, after)]);
        (tex, up)
    }

    /// `list` closed, run on `queue` and waited for.
    unsafe fn run(dev: &ID3D12Device, queue: &ID3D12CommandQueue, list: &ID3D12GraphicsCommandList) {
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        crate::gpu::fence::wait(&fence, 1, dev, None, 30_000).unwrap();
    }

    /// The `w`x`h` RGBA8 texels of a readback (`readback_copy`) that ran.
    unsafe fn texels(rb: &ID3D12Resource, fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, w: u32, h: u32) -> Vec<u8> {
        let mut out = vec![0u8; (w * h * 4) as usize];
        let mut p = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).unwrap();
        for y in 0..h as usize {
            let row = &mut out[y * w as usize * 4..][..w as usize * 4];
            std::ptr::copy_nonoverlapping((p as *const u8).add(fp.Offset as usize + y * fp.Footprint.RowPitch as usize), row.as_mut_ptr(), row.len());
        }
        rb.Unmap(0, None);
        out
    }

    /// `src` (sw x sh RGBA8) through the scaler into a dw x dh target, read back as RGBA8.
    unsafe fn downscale(dev: &ID3D12Device, scaler: &Scaler, src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        let (tex, _up) = upload(dev, &list, src, sw, sh, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        let rt = texture(dev, dw, dh, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET);
        scaler.record(dev, &list, &tex, &rt, dw, dh);
        list.ResourceBarrier(&[super::super::transition(&rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let (rb, fp) = super::super::readback_copy(&list, &rt, &rt.GetDesc()).unwrap();
        run(dev, &queue, &list);
        texels(&rb, &fp, dw, dh)
    }

    /// `got` (dw pixels a row, RGBA8) is the CPU area filter's `want` within 1.5 of 255.
    fn check(got: &[u8], want: &[[f64; 3]], dw: usize, what: &str) {
        let mut worst = (0.0f64, 0usize);
        for (i, w) in want.iter().enumerate() {
            for c in 0..3 {
                let d = (got[i * 4 + c] as f64 - w[c]).abs();
                if d > worst.0 {
                    worst = (d, i);
                }
            }
        }
        let i = worst.1;
        assert!(worst.0 <= 1.5, "{what}: pixel ({}, {}) is {:?}, the area filter gives {:?}", i % dw, i / dw, &got[i * 4..i * 4 + 3], want[i]);
    }

    /// A device on the GPU, else on WARP.
    fn device() -> Option<ID3D12Device> {
        let mut dev: Option<ID3D12Device> = None;
        unsafe {
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp = CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>()).ok()?;
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).ok()?;
            }
        }
        dev
    }

    /// 3840x2160 → 1280x720 (3x) and 1920x1080 → 1280x720 (1.5x) match the CPU area filter.
    #[test]
    fn downscale_matches_a_cpu_area_filter() {
        unsafe {
            let Some(dev) = device() else {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            };
            if crate::gpu::recomp::dxc::location().is_err() {
                crate::gpu::recomp::runner::skip_test("no DXC");
                return;
            }
            let scaler = Scaler::new(&dev, DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
            for (sw, sh) in [(3840u32, 2160u32), (1920, 1080)] {
                let (dw, dh) = (1280u32, 720u32);
                let src = pattern(sw as usize, sh as usize);
                let got = downscale(&dev, &scaler, &src, sw, sh, dw, dh);
                let want = reference(&src, sw as usize, sh as usize, dw as usize, dh as usize);
                check(&got, &want, dw as usize, &format!("{sw}x{sh} → {dw}x{dh}"));
            }
        }
    }

    /// A 16:9 image on wider screens: 1:1 between black bars (`Place`, 1280x720 in 1680x720)
    /// and through the filter between bars (1280x720 in 1032x432: 768x432 at x 132) – the bars
    /// black, the image as the copy or the CPU filter gives it.
    #[test]
    fn placed_image_between_black_bars() {
        unsafe {
            let Some(dev) = device() else {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            };
            if crate::gpu::recomp::dxc::location().is_err() {
                crate::gpu::recomp::runner::skip_test("no DXC");
                return;
            }
            let scaler = Scaler::new(&dev, DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
            let (iw, ih) = (1280u32, 720u32);
            let src = pattern(iw as usize, ih as usize);
            for (client, want_route, want_rect) in [((1680u32, 720u32), Route::Place, (200, 0, 1280, 720)), ((1032, 432), Route::Filter, (132, 0, 768, 432))] {
                let route = route((iw, ih), client, Scaling::Fit);
                let rect = place((iw, ih), client, Scaling::Fit);
                assert_eq!((route, rect), (want_route, want_rect), "{client:?}");
                let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
                let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                let (tex, _up) = upload(&dev, &list, &src, iw, ih, D3D12_RESOURCE_STATE_COMMON);
                // the back buffer, with something in it the bars must cover
                let fill = vec![200u8; (client.0 * client.1 * 4) as usize];
                let (bb, _up2) = upload(&dev, &list, &fill, client.0, client.1, D3D12_RESOURCE_STATE_COMMON);
                let bb_rt = texture(&dev, client.0, client.1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
                list.ResourceBarrier(&[super::super::transition(&bb, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                list.CopyResource(&bb_rt, &bb);
                list.ResourceBarrier(&[super::super::transition(&bb_rt, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT)]);
                let after = super::super::draw(&dev, &list, Some(&scaler), route, &tex, &bb_rt, D3D12_RESOURCE_STATE_PRESENT, rect, false);
                list.ResourceBarrier(&[super::super::transition(&bb_rt, after, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                let (rb, fp) = super::super::readback_copy(&list, &bb_rt, &bb_rt.GetDesc()).unwrap();
                run(&dev, &queue, &list);
                let got = texels(&rb, &fp, client.0, client.1);
                let (x0, y0, w, h) = rect;
                let reference = (route == Route::Filter).then(|| reference(&src, iw as usize, ih as usize, w as usize, h as usize));
                for y in 0..client.1 {
                    for x in 0..client.0 {
                        let t = &got[((y * client.0 + x) * 4) as usize..][..3];
                        let inside = x >= x0 && x < x0 + w && y >= y0 && y < y0 + h;
                        if !inside {
                            assert_eq!(t, [0, 0, 0], "{client:?}: bar pixel ({x}, {y})");
                        } else if let Some(r) = &reference {
                            let want = r[((y - y0) * w + (x - x0)) as usize];
                            for c in 0..3 {
                                assert!((t[c] as f64 - want[c]).abs() <= 1.5, "{client:?}: ({x}, {y}) is {t:?}, the area filter gives {want:?}");
                            }
                        } else {
                            let s = &src[(((y - y0) * iw + (x - x0)) * 4) as usize..][..3];
                            assert_eq!(t, s, "{client:?}: ({x}, {y}) is not the image's texel");
                        }
                    }
                }
            }
        }
    }

    /// A hidden popup window (no frame: the client area is the whole window) on a thread of its
    /// own that pumps its messages, as the XCL window is.
    struct TestWindow {
        hwnd: HWND,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    unsafe extern "system" fn test_wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        if msg == WM_DESTROY {
            PostQuitMessage(0);
            return LRESULT(0);
        }
        DefWindowProcW(hwnd, msg, wp, lp)
    }

    impl TestWindow {
        fn new() -> TestWindow {
            let (tx, rx) = std::sync::mpsc::channel();
            let thread = std::thread::spawn(move || unsafe {
                let inst = GetModuleHandleW(PCWSTR::null()).unwrap();
                let class = w!("XCLPresentTestWindow");
                let wc = WNDCLASSW {
                    lpfnWndProc: Some(test_wndproc),
                    hInstance: inst.into(),
                    lpszClassName: class,
                    ..Default::default()
                };
                RegisterClassW(&wc);
                let hwnd = CreateWindowExW(WINDOW_EX_STYLE(0), class, w!("XCL present test"), WS_POPUP, 0, 0, 1280, 720, None, None, Some(inst.into()), None).unwrap();
                tx.send(hwnd.0 as isize).unwrap();
                let mut msg = MSG::default();
                while GetMessageW(&mut msg, None, 0, 0).as_bool() {
                    DispatchMessageW(&msg);
                }
            });
            TestWindow {
                hwnd: HWND(rx.recv().unwrap() as *mut std::ffi::c_void),
                thread: Some(thread),
            }
        }

        /// The client area made `w`x`h` (0x0: as when minimized).
        unsafe fn resize(&self, w: u32, h: u32) {
            SetWindowPos(self.hwnd, None, 0, 0, w as i32, h as i32, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE).unwrap();
            assert_eq!(super::super::client_size(self.hwnd), (w, h));
        }
    }

    impl Drop for TestWindow {
        fn drop(&mut self) {
            unsafe {
                let _ = PostMessageW(Some(self.hwnd), WM_CLOSE, WPARAM(0), LPARAM(0));
            }
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }

    /// The swap chain follows the window while a title presents: 1280x720, larger (1920x1080),
    /// smaller (800x450) – each frame shown at the client size, the filter drawing into the new
    /// size as the CPU filter does; minimized (0x0): nothing shown and no resize; restored: shown
    /// again. On an adopted device, as the title's is (its CreateRenderTargetView hook is in the
    /// path).
    #[test]
    fn swap_chain_follows_the_window() {
        use crate::hle::d3d12x::real;
        unsafe {
            let Some(dev) = device() else {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            };
            if crate::gpu::recomp::dxc::location().is_err() {
                crate::gpu::recomp::runner::skip_test("no DXC");
                return;
            }
            real::adopt(dev.as_raw(), real::Kind::Device);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            // inside a present, as `present_now` runs it: these submissions are not the title's frames
            super::super::IN_PRESENT.with(|c| c.set(true));
            // the title's 3840x2160 image, in COMMON as PresentX hands it over
            let (iw, ih) = (3840u32, 2160u32);
            let pixels = pattern(iw as usize, ih as usize);
            let (image, _up) = upload(&dev, &list, &pixels, iw, ih, D3D12_RESOURCE_STATE_COMMON);
            run(&dev, &queue, &list);
            let desc = image.GetDesc();
            let window = TestWindow::new();
            let mut p = super::super::presenter_for(&dev, queue.as_raw() as usize).expect("presenter");
            for (k, (w, h)) in [(1280u32, 720u32), (1920, 1080), (800, 450), (0, 0), (1280, 720)].into_iter().enumerate() {
                window.resize(w, h);
                let before = (p.w, p.h);
                let shown = super::super::present_frame(&mut p, window.hwnd, image.clone(), &desc);
                if w == 0 {
                    assert!(!shown, "minimized: nothing is shown");
                    assert_eq!((p.w, p.h), before, "minimized: the swap chain keeps its size");
                    continue;
                }
                assert!(shown, "{w}x{h}: the frame is shown");
                let sc = p.sc.clone().unwrap();
                let d = sc.GetDesc1().unwrap();
                assert_eq!((p.w, p.h, d.Width, d.Height), (w, h, w, h), "the swap chain follows the {w}x{h} client area");
                assert_eq!(p.route, Some((Route::Filter, (iw, ih), (w, h))));
                if k > 2 {
                    continue; // a size checked already
                }
                // the filter into the new back buffer, read back
                crate::gpu::fence::wait(&p.fence, p.value, &dev, None, 30_000).unwrap();
                let bb: ID3D12Resource = sc.GetBuffer(sc.GetCurrentBackBufferIndex()).unwrap();
                alloc.Reset().unwrap();
                list.Reset(&alloc, None).unwrap();
                list.ResourceBarrier(&[super::super::transition(&bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
                p.scaler.as_ref().unwrap().record(&dev, &list, &image, &bb, w, h);
                list.ResourceBarrier(&[super::super::transition(&bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                let (rb, fp) = super::super::readback_copy(&list, &bb, &bb.GetDesc()).unwrap();
                list.ResourceBarrier(&[super::super::transition(&bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT)]);
                run(&dev, &queue, &list);
                drop(bb);
                let want = reference(&pixels, iw as usize, ih as usize, w as usize, h as usize);
                check(&texels(&rb, &fp, w, h), &want, w as usize, &format!("{iw}x{ih} → {w}x{h} window"));
            }
            // a resize DXGI refuses (a back buffer still referenced): the frame is shown anyway,
            // drawn at the swap chain's size; another size is asked for at once
            let held: ID3D12Resource = p.sc.as_ref().unwrap().GetBuffer(0).unwrap();
            window.resize(1920, 1080);
            assert!(super::super::present_frame(&mut p, window.hwnd, image.clone(), &desc), "a refused resize still shows the frame");
            assert_eq!((p.w, p.h), (1280, 720), "refused: the swap chain keeps its size");
            drop(held);
            window.resize(800, 450);
            assert!(super::super::present_frame(&mut p, window.hwnd, image.clone(), &desc));
            assert_eq!((p.w, p.h), (800, 450), "the next size is asked for at once");
            // the last frame done before the presenter goes (the runtime waits too, `present_now`):
            // its allocator, the filter's pipeline and descriptor heaps and the swap chain released
            // under the GPU hung the device – the one every GPU test of the process shares
            crate::gpu::fence::wait(&p.fence, p.value, &dev, None, 30_000).unwrap();
            drop(p);
            super::super::IN_PRESENT.with(|c| c.set(false));
        }
    }
}
