//! Persistent on-disk pipeline library cache (ID3D12PipelineLibrary1).
//!
//! Eliminates runtime DXIL-to-ISA driver compilation stalls by caching
//! driver-compiled pipeline states across application launches.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

struct Library {
    library: ID3D12PipelineLibrary1,
    /// Backing blob from disk that must outlive the library instance.
    _blob: Vec<u8>,
    added: u64,
}

// SAFETY: D3D12 pipeline library operations are guarded by the internal Mutex.
unsafe impl Send for Library {}

static LIBRARY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
static DIR: OnceLock<PathBuf> = OnceLock::new();

pub static GRAPHICS_LOADED: AtomicU64 = AtomicU64::new(0);
pub static GRAPHICS_STORED: AtomicU64 = AtomicU64::new(0);
pub static COMPUTE_LOADED: AtomicU64 = AtomicU64::new(0);
pub static COMPUTE_STORED: AtomicU64 = AtomicU64::new(0);

const BATCH_SAVE_THRESHOLD: u64 = 16;
const LIBRARY_FILE: &str = "host-pipelines.bin";

/// Folder of the PSO disk cache: XCL_PSO_CACHE, else %LOCALAPPDATA%\xcl\pso.
pub fn cache_dir() -> PathBuf {
    std::env::var("XCL_PSO_CACHE")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("LOCALAPPDATA").map(|l| PathBuf::from(l).join("xcl").join("pso")))
        .unwrap_or_else(|_| PathBuf::from("xcl-pso-cache"))
}

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_PIPELINE_LIBRARY").map_or(true, |v| v.trim() != "0"))
}

/// Initialize the pipeline library for the active D3D12 device.
pub fn init(device: &ID3D12Device) {
    if !enabled() {
        return;
    }
    let dir = cache_dir();
    let _ = DIR.set(dir.clone());
    let _ = LIBRARY.get_or_init(|| {
        let dev1 = device.cast::<ID3D12Device1>().ok()?;
        let blob = std::fs::read(dir.join(LIBRARY_FILE)).unwrap_or_default();
        let lib = open(&dev1, blob)?;
        Some(Mutex::new(lib))
    });
}

fn library() -> Option<&'static Mutex<Library>> {
    LIBRARY.get().and_then(|opt| opt.as_ref())
}

fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // vkd3d-proton requires a null blob pointer when length is zero to construct an empty library;
        // passing an empty slice through windows-rs forwards a dangling non-null pointer (0x1) that fails validation.
        let ptr = if bytes.is_empty() {
            std::ptr::null()
        } else {
            bytes.as_ptr().cast()
        };
        let mut raw = std::ptr::null_mut();
        unsafe {
            (windows::core::Interface::vtable(device).CreatePipelineLibrary)(
                windows::core::Interface::as_raw(device),
                ptr,
                bytes.len(),
                &ID3D12PipelineLibrary1::IID,
                &mut raw,
            )
            .and_then(|| windows::core::Type::from_abi(raw))
        }
    };
    if !blob.is_empty() {
        match make(&blob) {
            Ok(library) => {
                crate::xlog_info!("D3D12.X: loaded host pipeline library ({} B)", blob.len());
                return Some(Library { library, _blob: blob, added: 0 });
            }
            Err(e) => {
                crate::xlog_info!("D3D12.X: pipeline library rejected ({e}), starting fresh library");
            }
        }
    }
    let library = make(&[]).ok()?;
    Some(Library { library, _blob: Vec::new(), added: 0 })
}

static ROOT_SIG_HASHES: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

/// Associates a root signature's interface address with its deterministic content hash.
pub fn note_root_signature(raw: usize, hash: u64) {
    if raw == 0 {
        return;
    }
    let mut g = ROOT_SIG_HASHES.lock().unwrap();
    g.get_or_insert_with(HashMap::new).insert(raw, hash);
}

/// Resolves the deterministic content hash of a root signature.
pub fn root_signature_hash(rs: Option<&ID3D12RootSignature>) -> u64 {
    let Some(rs) = rs else { return 0 };
    let raw = windows::core::Interface::as_raw(rs) as usize;
    ROOT_SIG_HASHES.lock().unwrap().as_ref().and_then(|m| m.get(&raw).copied()).unwrap_or(0)
}

fn hash_blend_desc(desc: &D3D12_BLEND_DESC) -> u64 {
    // Structure padding bytes between RenderTargetWriteMask and the next slot
    // contain undefined stack garbage that invalidates hashes across compilers.
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    h = (h ^ (desc.AlphaToCoverageEnable.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.IndependentBlendEnable.0 as u64)).wrapping_mul(0x100_0000_01b3);
    for rt in &desc.RenderTarget {
        h = (h ^ (rt.BlendEnable.0 as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (rt.LogicOpEnable.0 as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (rt.SrcBlend.0 as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (rt.DestBlend.0 as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (rt.BlendOp.0 as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (rt.SrcBlendAlpha.0 as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (rt.DestBlendAlpha.0 as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (rt.BlendOpAlpha.0 as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (rt.LogicOp.0 as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (rt.RenderTargetWriteMask as u64)).wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn hash_depth_stencil_desc(desc: &D3D12_DEPTH_STENCIL_DESC) -> u64 {
    // StencilWriteMask is followed by alignment padding bytes that must not enter the hash.
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    h = (h ^ (desc.DepthEnable.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.DepthWriteMask.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.DepthFunc.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.StencilEnable.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.StencilReadMask as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.StencilWriteMask as u64)).wrapping_mul(0x100_0000_01b3);

    h = (h ^ (desc.FrontFace.StencilFailOp.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.FrontFace.StencilDepthFailOp.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.FrontFace.StencilPassOp.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.FrontFace.StencilFunc.0 as u64)).wrapping_mul(0x100_0000_01b3);

    h = (h ^ (desc.BackFace.StencilFailOp.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.BackFace.StencilDepthFailOp.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.BackFace.StencilPassOp.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.BackFace.StencilFunc.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h
}

fn hash_rasterizer_desc(desc: &D3D12_RASTERIZER_DESC) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    h = (h ^ (desc.FillMode.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.CullMode.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.FrontCounterClockwise.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.DepthBias as u32 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.DepthBiasClamp.to_bits() as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.SlopeScaledDepthBias.to_bits() as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.DepthClipEnable.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.MultisampleEnable.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.AntialiasedLineEnable.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.ForcedSampleCount as u64)).wrapping_mul(0x100_0000_01b3);
    h = (h ^ (desc.ConservativeRaster.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h
}

fn hash_input_layout(layout: &D3D12_INPUT_LAYOUT_DESC) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    h = (h ^ (layout.NumElements as u64)).wrapping_mul(0x100_0000_01b3);
    if layout.NumElements == 0 || layout.pInputElementDescs.is_null() {
        return h;
    }
    // SAFETY: D3D12 input layout descriptor array spans NumElements valid entries when non-null.
    let elems = unsafe { std::slice::from_raw_parts(layout.pInputElementDescs, layout.NumElements as usize) };
    for e in elems {
        if !e.SemanticName.is_null() {
            // SAFETY: SemanticName is a null-terminated ASCII string in D3D12_INPUT_ELEMENT_DESC.
            let name = unsafe { std::ffi::CStr::from_ptr(e.SemanticName.as_ptr() as *const i8) };
            for &b in name.to_bytes() {
                h = (h ^ (b as u64)).wrapping_mul(0x100_0000_01b3);
            }
        }
        h = (h ^ (e.SemanticIndex as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (e.Format.0 as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (e.InputSlot as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (e.AlignedByteOffset as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (e.InputSlotClass.0 as u64)).wrapping_mul(0x100_0000_01b3);
        h = (h ^ (e.InstanceDataStepRate as u64)).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Computes a unique name for a graphics pipeline based on its state and translated shaders.
pub fn graphics_pso_name(
    desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    vs: &[u8],
    ps: Option<&[u8]>,
    gs: Option<&[u8]>,
) -> String {
    let mut h1 = crate::gpu::shader::fnv64(vs);
    let mut h2 = 0xcbf2_9ce4_8422_2325u64;

    if let Some(p) = ps {
        h1 = (h1 ^ crate::gpu::shader::fnv64(p)).wrapping_mul(0x100_0000_01b3);
    }
    if let Some(g) = gs {
        h2 = (h2 ^ crate::gpu::shader::fnv64(g)).wrapping_mul(0x100_0000_01b3);
    }

    let rs_hash = root_signature_hash(desc.pRootSignature.as_ref());
    h1 = (h1 ^ rs_hash).wrapping_mul(0x100_0000_01b3);

    h2 = (h2 ^ hash_blend_desc(&desc.BlendState)).wrapping_mul(0x100_0000_01b3);
    h1 = (h1 ^ hash_rasterizer_desc(&desc.RasterizerState)).wrapping_mul(0x100_0000_01b3);
    h2 = (h2 ^ hash_depth_stencil_desc(&desc.DepthStencilState)).wrapping_mul(0x100_0000_01b3);
    h1 = (h1 ^ hash_input_layout(&desc.InputLayout)).wrapping_mul(0x100_0000_01b3);

    h1 = (h1 ^ (desc.SampleMask as u64)).wrapping_mul(0x100_0000_01b3);
    h2 = (h2 ^ (desc.PrimitiveTopologyType.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h1 = (h1 ^ (desc.NumRenderTargets as u64)).wrapping_mul(0x100_0000_01b3);
    let num_rt = (desc.NumRenderTargets as usize).min(8);
    for &fmt in &desc.RTVFormats[..num_rt] {
        h2 = (h2 ^ (fmt.0 as u64)).wrapping_mul(0x100_0000_01b3);
    }
    h2 = (h2 ^ (desc.DSVFormat.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h1 = (h1 ^ (desc.SampleDesc.Count as u64)).wrapping_mul(0x100_0000_01b3);
    h1 = (h1 ^ (desc.SampleDesc.Quality as u64)).wrapping_mul(0x100_0000_01b3);
    h2 = (h2 ^ (desc.IBStripCutValue.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h2 = (h2 ^ (desc.Flags.0 as u64)).wrapping_mul(0x100_0000_01b3);
    h1 = (h1 ^ (desc.NodeMask as u64)).wrapping_mul(0x100_0000_01b3);

    format!("gp_{h1:016x}{h2:016x}")
}

/// Computes a unique name for a compute pipeline based on its root signature and compute shader.
pub fn compute_pso_name(desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC, cs: &[u8]) -> String {
    let mut h1 = crate::gpu::shader::fnv64(cs);
    let rs_hash = root_signature_hash(desc.pRootSignature.as_ref());
    h1 = (h1 ^ rs_hash).wrapping_mul(0x100_0000_01b3);
    let h2 = (desc.Flags.0 as u64).wrapping_mul(0x100_0000_01b3) ^ ((desc.NodeMask as u64) << 32) ^ 0x9e3779b97f4a7c15u64;

    format!("cp_{h1:016x}{h2:016x}")
}

/// Attempts to load a graphics pipeline from the persistent library.
pub fn load_graphics(
    name: &str,
    desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
) -> Option<ID3D12PipelineState> {
    let lib_mutex = library()?;
    let l = lib_mutex.lock().unwrap();
    let wide = HSTRING::from(name);
    // SAFETY: D3D12 PipelineLibrary load call with valid pointer and descriptor references.
    let res: windows::core::Result<ID3D12PipelineState> = unsafe {
        l.library.LoadGraphicsPipeline(PCWSTR(wide.as_ptr()), desc)
    };
    match res {
        Ok(pso) => {
            GRAPHICS_LOADED.fetch_add(1, Ordering::Relaxed);
            Some(pso)
        }
        Err(_) => None,
    }
}

static PENDING_ADDED: AtomicU64 = AtomicU64::new(0);
static LAST_SAVE_MS: AtomicU64 = AtomicU64::new(0);
static SAVE_EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
static BG_SAVING: AtomicBool = AtomicBool::new(false);
static SAVE_REQUESTED: AtomicBool = AtomicBool::new(false);

const SAVE_EVERY_MS: u64 = 20_000;

fn current_save_ms() -> u64 {
    SAVE_EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + 1
}

fn due(pending: u64, now: bool, prev_ms: u64, now_ms: u64) -> bool {
    if pending == 0 {
        return false;
    }
    now || pending >= BATCH_SAVE_THRESHOLD || prev_ms == 0 || now_ms.saturating_sub(prev_ms) >= SAVE_EVERY_MS
}

fn save_due(now: bool, now_ms: u64) -> bool {
    if DIR.get().is_none() || library().is_none() {
        return false;
    }
    due(
        PENDING_ADDED.load(Ordering::Acquire),
        now,
        LAST_SAVE_MS.load(Ordering::Relaxed),
        now_ms,
    )
}

fn store_pipeline(name: &str, pso: &ID3D12PipelineState, counter: &AtomicU64) {
    let Some(lib_mutex) = library() else { return };
    let should_save = {
        let mut l = lib_mutex.lock().unwrap();
        let wide = HSTRING::from(name);
        // SAFETY: D3D12 PipelineLibrary store call with valid wide name and pipeline state pointer.
        if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), pso) }.is_ok() {
            l.added += 1;
            PENDING_ADDED.fetch_add(1, Ordering::Release);
            counter.fetch_add(1, Ordering::Relaxed);
            l.added >= BATCH_SAVE_THRESHOLD
        } else {
            false
        }
    };
    if should_save {
        save_async();
    }
}

/// Stores a newly compiled graphics pipeline into the library.
pub fn store_graphics(name: &str, pso: &ID3D12PipelineState) {
    store_pipeline(name, pso, &GRAPHICS_STORED);
}

/// Attempts to load a compute pipeline from the persistent library.
pub fn load_compute(
    name: &str,
    desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC,
) -> Option<ID3D12PipelineState> {
    let lib_mutex = library()?;
    let l = lib_mutex.lock().unwrap();
    let wide = HSTRING::from(name);
    // SAFETY: D3D12 PipelineLibrary load call with valid pointer and descriptor references.
    let res: windows::core::Result<ID3D12PipelineState> = unsafe {
        l.library.LoadComputePipeline(PCWSTR(wide.as_ptr()), desc)
    };
    match res {
        Ok(pso) => {
            COMPUTE_LOADED.fetch_add(1, Ordering::Relaxed);
            Some(pso)
        }
        Err(_) => None,
    }
}

/// Stores a newly compiled compute pipeline into the library.
pub fn store_compute(name: &str, pso: &ID3D12PipelineState) {
    store_pipeline(name, pso, &COMPUTE_STORED);
}

/// Logs telemetry summary of loaded vs stored pipelines.
pub fn log_summary() {
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !enabled() || library().is_none() {
        return;
    }
    if LOGGED.swap(true, Ordering::Relaxed) {
        return;
    }
    let gl = GRAPHICS_LOADED.load(Ordering::Relaxed);
    let cl = COMPUTE_LOADED.load(Ordering::Relaxed);
    let gs = GRAPHICS_STORED.load(Ordering::Relaxed);
    let cs = COMPUTE_STORED.load(Ordering::Relaxed);
    crate::xlog_info!(
        "D3D12.X: pipeline library summary: {gl} graphics loaded, {cl} compute loaded, {gs} graphics stored, {cs} compute stored"
    );
}

/// Requests a non-blocking background save when new pipelines are due, so frame
/// presentation and pipeline build threads never serialize or write to disk.
pub fn save_async() {
    if !save_due(false, current_save_ms()) {
        return;
    }
    SAVE_REQUESTED.store(true, Ordering::SeqCst);
    if BG_SAVING.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        return;
    }
    if std::thread::Builder::new()
        .name("xcl-pso-save".into())
        .spawn(|| {
            struct Guard;
            impl Drop for Guard {
                fn drop(&mut self) {
                    BG_SAVING.store(false, Ordering::SeqCst);
                }
            }
            let _guard = Guard;
            loop {
                SAVE_REQUESTED.store(false, Ordering::SeqCst);
                let wrote = save_inner(false);
                if wrote && SAVE_REQUESTED.load(Ordering::SeqCst) && save_due(false, current_save_ms()) {
                    continue;
                }
                BG_SAVING.store(false, Ordering::SeqCst);
                // Re-check after clearing BG_SAVING so a request arriving right before
                // the flag cleared is never stranded when no frame presents follow.
                if wrote
                    && SAVE_REQUESTED.load(Ordering::SeqCst)
                    && save_due(false, current_save_ms())
                    && BG_SAVING.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok()
                {
                    continue;
                }
                break;
            }
        })
        .is_err()
    {
        BG_SAVING.store(false, Ordering::SeqCst);
    }
}

/// Serializes and writes the library to disk if new pipelines were stored.
pub fn save(now: bool) {
    let _ = save_inner(now);
}

fn save_inner(now: bool) -> bool {
    // Serialize-then-write order must remain monotonic so an earlier snapshot cannot overwrite a newer one.
    static SAVE_IO: Mutex<()> = Mutex::new(());

    if now {
        log_summary();
    }

    let (Some(dir), Some(lib_mutex)) = (DIR.get(), library()) else { return false };
    // Avoid blocking callers on SAVE_IO when nothing is pending, within throttle, or busy.
    let now_ms = current_save_ms();
    if !save_due(now, now_ms) {
        return false;
    }
    let _io_guard = if now {
        SAVE_IO.lock().unwrap()
    } else {
        match SAVE_IO.try_lock() {
            Ok(g) => g,
            Err(_) => return false,
        }
    };

    let blob = {
        let mut l = lib_mutex.lock().unwrap();
        if l.added == 0 {
            return false;
        }
        let batch_ready = l.added >= BATCH_SAVE_THRESHOLD;
        let prev_ms = LAST_SAVE_MS.load(Ordering::Relaxed);
        if !now && !batch_ready && prev_ms != 0 && now_ms.saturating_sub(prev_ms) < SAVE_EVERY_MS {
            return false;
        }
        LAST_SAVE_MS.store(now_ms, Ordering::Relaxed);

        // SAFETY: allocating exact byte buffer queried from GetSerializedSize.
        let bytes = unsafe {
            let size = l.library.GetSerializedSize();
            let mut bytes = vec![0u8; size];
            l.library.Serialize(&mut bytes).map(|()| bytes)
        };
        match bytes {
            Ok(bytes) => {
                l.added = 0;
                PENDING_ADDED.store(0, Ordering::Release);
                bytes
            }
            Err(e) => {
                crate::xlog_warn!("D3D12.X: pipeline library serialization failed: {e}");
                return false;
            }
        }
    };

    let target = dir.join(LIBRARY_FILE);
    if let Err(e) = std::fs::create_dir_all(dir).and_then(|()| write_atomically(&target, &blob)) {
        crate::xlog_warn!("D3D12.X: saving pipeline library to {} failed: {e}", target.display());
        false
    } else {
        crate::xlog_info!(
            "D3D12.X: saved pipeline library ({} B, {} graphics loaded, {} compute loaded)",
            blob.len(),
            GRAPHICS_LOADED.load(Ordering::Relaxed),
            COMPUTE_LOADED.load(Ordering::Relaxed)
        );
        true
    }
}

fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let tmp = path.with_extension(format!("tmp{}_{}", std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed)));
    std::fs::write(&tmp, bytes)?;
    if path.exists() {
        let _ = std::fs::remove_file(path);
    }
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_names_are_deterministic_and_differentiate_states() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;

        let vs = b"dxil_vertex_shader_bytecode_test";
        let ps1 = b"dxil_pixel_shader_bytecode_test_1";
        let ps2 = b"dxil_pixel_shader_bytecode_test_2";

        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;

        let n1 = graphics_pso_name(&desc, vs, Some(ps1), None);
        let n1_dup = graphics_pso_name(&desc, vs, Some(ps1), None);
        let n2 = graphics_pso_name(&desc, vs, Some(ps2), None);

        assert_eq!(n1, n1_dup);
        assert_ne!(n1, n2);

        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        let n3 = graphics_pso_name(&desc, vs, Some(ps1), None);
        assert_ne!(n1, n3);

        // Blend state differentiation
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.BlendState.RenderTarget[0].BlendEnable = true.into();
        let n_blend = graphics_pso_name(&desc, vs, Some(ps1), None);
        assert_ne!(n1, n_blend);

        // Depth/stencil state differentiation
        desc.BlendState.RenderTarget[0].BlendEnable = false.into();
        desc.DepthStencilState.DepthEnable = true.into();
        let n_depth = graphics_pso_name(&desc, vs, Some(ps1), None);
        assert_ne!(n1, n_depth);

        // Render target format differentiation
        desc.DepthStencilState.DepthEnable = false.into();
        desc.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        let n_fmt = graphics_pso_name(&desc, vs, Some(ps1), None);
        assert_ne!(n1, n_fmt);
    }

    #[test]
    fn compute_pipeline_names_differentiate_flags_and_shaders() {
        let cs1 = b"dxil_compute_shader_bytecode_test_1";
        let cs2 = b"dxil_compute_shader_bytecode_test_2";

        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC::default();
        desc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

        let n1 = compute_pso_name(&desc, cs1);
        let n1_dup = compute_pso_name(&desc, cs1);
        let n2 = compute_pso_name(&desc, cs2);

        assert_eq!(n1, n1_dup);
        assert_ne!(n1, n2);

        desc.Flags = D3D12_PIPELINE_STATE_FLAGS(1);
        let n3 = compute_pso_name(&desc, cs1);
        assert_ne!(n1, n3);
    }

    #[test]
    fn corrupted_blob_falls_back_to_empty_library() {
        let Ok(dev) = crate::gpu::agility::create_device(
            None::<&windows::core::IUnknown>,
            windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0,
        ) else {
            return;
        };
        let Ok(dev1) = dev.cast::<ID3D12Device1>() else {
            return;
        };

        let corrupted = vec![0x7f; 1024];
        let lib = open(&dev1, corrupted).expect("failed to create fallback empty library after rejecting corrupted blob");
        assert!(lib._blob.is_empty());
        let size = unsafe { lib.library.GetSerializedSize() };
        assert!(size > 0);
    }

    #[test]
    fn atomic_file_writing_prevents_partial_writes() {
        let dir = std::env::temp_dir().join(format!("xcl-pso-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test-library.bin");

        let data = b"serialized_pipeline_library_mock_payload";
        assert!(write_atomically(&path, data).is_ok());
        assert_eq!(std::fs::read(&path).unwrap(), data);

        let data2 = b"serialized_pipeline_library_second_write";
        assert!(write_atomically(&path, data2).is_ok());
        assert_eq!(std::fs::read(&path).unwrap(), data2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn padding_bytes_do_not_affect_pipeline_hash() {
        let vs = b"dxil_test_vs";
        let mut desc1 = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        desc1.BlendState.RenderTarget[0].RenderTargetWriteMask = 0x0f;
        desc1.DepthStencilState.StencilWriteMask = 0xff;

        let mut raw = vec![0x7fu8; std::mem::size_of::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>()];
        let desc2 = unsafe {
            let p = raw.as_mut_ptr() as *mut D3D12_GRAPHICS_PIPELINE_STATE_DESC;
            std::ptr::write(p, D3D12_GRAPHICS_PIPELINE_STATE_DESC::default());
            (*p).BlendState.RenderTarget[0].RenderTargetWriteMask = 0x0f;
            (*p).DepthStencilState.StencilWriteMask = 0xff;

            let rt_write_mask_ptr = &(*p).BlendState.RenderTarget[0].RenderTargetWriteMask as *const u8;
            let blend_padding = std::slice::from_raw_parts_mut(rt_write_mask_ptr.add(1) as *mut u8, 3);
            blend_padding.fill(0xee);

            let ds_write_mask_ptr = &(*p).DepthStencilState.StencilWriteMask as *const u8;
            let ds_padding = std::slice::from_raw_parts_mut(ds_write_mask_ptr.add(1) as *mut u8, 2);
            ds_padding.fill(0xcc);
            &*p
        };

        let n1 = graphics_pso_name(&desc1, vs, None, None);
        let n2 = graphics_pso_name(desc2, vs, None, None);
        assert_eq!(n1, n2);
    }

    #[test]
    fn telemetry_counters_track_loads_and_stores() {
        let gl_before = GRAPHICS_LOADED.load(Ordering::Relaxed);
        let cl_before = COMPUTE_LOADED.load(Ordering::Relaxed);
        let gs_before = GRAPHICS_STORED.load(Ordering::Relaxed);
        let cs_before = COMPUTE_STORED.load(Ordering::Relaxed);

        GRAPHICS_LOADED.fetch_add(2, Ordering::Relaxed);
        COMPUTE_LOADED.fetch_add(3, Ordering::Relaxed);
        GRAPHICS_STORED.fetch_add(4, Ordering::Relaxed);
        COMPUTE_STORED.fetch_add(5, Ordering::Relaxed);

        assert_eq!(GRAPHICS_LOADED.load(Ordering::Relaxed), gl_before + 2);
        assert_eq!(COMPUTE_LOADED.load(Ordering::Relaxed), cl_before + 3);
        assert_eq!(GRAPHICS_STORED.load(Ordering::Relaxed), gs_before + 4);
        assert_eq!(COMPUTE_STORED.load(Ordering::Relaxed), cs_before + 5);

        log_summary();
    }

    #[test]
    fn input_layout_differentiates_graphics_pipeline_names() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32_FLOAT;
        use windows::core::PCSTR;

        let vs = b"dxil_vs_layout_test";
        let elems_a = [D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(b"POSITION\0".as_ptr()),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        }];
        let elems_b = [D3D12_INPUT_ELEMENT_DESC {
            AlignedByteOffset: 16,
            ..elems_a[0]
        }];

        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        desc.InputLayout = D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: elems_a.as_ptr(),
            NumElements: elems_a.len() as u32,
        };
        let n_a = graphics_pso_name(&desc, vs, None, None);

        desc.InputLayout.pInputElementDescs = elems_b.as_ptr();
        let n_b = graphics_pso_name(&desc, vs, None, None);
        assert_ne!(n_a, n_b);

        desc.InputLayout = D3D12_INPUT_LAYOUT_DESC::default();
        let n_empty = graphics_pso_name(&desc, vs, None, None);
        assert_ne!(n_a, n_empty);
    }

    #[test]
    fn save_due_respects_pending_and_throttle() {
        assert!(!due(0, true, 0, 50_000));
        assert!(!due(0, false, 0, 50_000));
        assert!(due(1, false, 0, 100));
        assert!(!due(1, false, 10_000, 29_999));
        assert!(due(1, false, 10_000, 30_000));
        assert!(due(BATCH_SAVE_THRESHOLD, false, 10_000, 10_001));
        assert!(due(1, true, 10_000, 10_001));
        save_async();
    }

    #[test]
    fn root_signature_hash_is_stable_across_com_pointers() {
        use std::ffi::c_void;
        use std::mem::ManuallyDrop;

        #[repr(C)]
        struct FakeRootSig {
            vt: *const usize,
        }
        unsafe extern "system" fn add_ref(_this: *mut c_void) -> u32 {
            1
        }
        unsafe extern "system" fn release(_this: *mut c_void) -> u32 {
            1
        }
        let mut vt = [0usize; 4];
        vt[1] = add_ref as usize;
        vt[2] = release as usize;

        let mut obj_a = FakeRootSig { vt: vt.as_ptr() };
        let mut obj_b = FakeRootSig { vt: vt.as_ptr() };
        let ptr_a = &mut obj_a as *mut _ as usize;
        let ptr_b = &mut obj_b as *mut _ as usize;
        assert_ne!(ptr_a, ptr_b);

        const HASH_ONE: u64 = 0x1122_3344_5566_7788;
        const HASH_TWO: u64 = 0x8877_6655_4433_2211;
        note_root_signature(ptr_a, HASH_ONE);
        note_root_signature(ptr_b, HASH_ONE);

        unsafe {
            let rs_a = ManuallyDrop::new(Some(ID3D12RootSignature::from_raw(ptr_a as *mut c_void)));
            let rs_b = ManuallyDrop::new(Some(ID3D12RootSignature::from_raw(ptr_b as *mut c_void)));

            let vs = b"dxil_vs_rs_test";
            let ps = b"dxil_ps_rs_test";
            let cs = b"dxil_cs_rs_test";

            let gdesc_a = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::ptr::read(&rs_a),
                ..Default::default()
            };
            let gdesc_b = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::ptr::read(&rs_b),
                ..Default::default()
            };

            let gp_a = graphics_pso_name(&gdesc_a, vs, Some(ps), None);
            let gp_b = graphics_pso_name(&gdesc_b, vs, Some(ps), None);
            assert_eq!(gp_a, gp_b, "same root signature hash at different COM addresses must yield identical graphics PSO names");

            let cdesc_a = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::ptr::read(&rs_a),
                ..Default::default()
            };
            let cdesc_b = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::ptr::read(&rs_b),
                ..Default::default()
            };

            let cp_a = compute_pso_name(&cdesc_a, cs);
            let cp_b = compute_pso_name(&cdesc_b, cs);
            assert_eq!(cp_a, cp_b, "same root signature hash at different COM addresses must yield identical compute PSO names");

            note_root_signature(ptr_b, HASH_TWO);
            let gp_b2 = graphics_pso_name(&gdesc_b, vs, Some(ps), None);
            let cp_b2 = compute_pso_name(&cdesc_b, cs);
            assert_ne!(gp_a, gp_b2, "distinct root signature hashes must differentiate graphics PSO names");
            assert_ne!(cp_a, cp_b2, "distinct root signature hashes must differentiate compute PSO names");
        }

        if let Some(m) = ROOT_SIG_HASHES.lock().unwrap().as_mut() {
            m.remove(&ptr_a);
            m.remove(&ptr_b);
        }
    }
}
