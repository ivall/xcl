//! What the PC has of what a run needs, checked by `xcl run` before the game starts: DXC, the
//! system XGameRuntime and the GPU the game will get (Shader Model 6.6, mesh shaders, wave
//! width). Without one of them the game still starts, but draws nothing or part of the
//! picture, so each missing piece is printed with what to do, and kept in `report.json`.
//! Under Wine it also looks at the D3D12 / DXGI libraries the process got (see `docs/linux.md`).

use serde::{Deserialize, Serialize};

/// How much a missing piece costs the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Ok,
    /// runs, with a known limit
    Note,
    /// part of the game is missing (no geometry, no system services, software rendering)
    Warning,
    /// the game draws nothing
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Check {
    pub level: Level,
    pub what: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
}

/// The GPU the game gets, as D3D12 reports it.
#[derive(Debug, Clone, Default)]
pub struct Gpu {
    pub name: String,
    pub warp: bool,
    /// (6, 6) = Shader Model 6.6
    pub shader_model: (u8, u8),
    /// the message for a missing SM 6.6 (it depends on the Agility SDK state)
    pub no_sm66: String,
    pub mesh: bool,
    pub wave_min: u32,
    pub wave_max: u32,
}

/// Everything the checks look at; `checks` turns it into the list.
#[derive(Debug, Clone)]
pub struct Found {
    pub dxc: Result<String, String>,
    /// the DXC in that folder, for its version
    pub dxc_toolchain: crate::gpu::recomp::dxc::Toolchain,
    pub xgameruntime: bool,
    pub gpu: Result<Gpu, String>,
    /// the D3D12 / DXGI libraries of a Wine prefix; `None` on Windows
    pub wine: Option<WineLibs>,
}

/// What the libraries loaded under Wine are, found by looking into their images. `None`: that
/// library is not loaded, nothing to say.
#[derive(Debug, Clone, Copy, Default)]
pub struct WineLibs {
    /// `d3d12core.dll` holds vkd3d-proton's vtable-identity assertions
    pub vkd3d_asserts_vtables: Option<bool>,
    /// `dxgi.dll` is DXVK's
    pub dxgi_is_dxvk: Option<bool>,
}

/// The assertion that stops vkd3d-proton on the objects XCL adopts (upstream issue #241).
const VKD3D_VTABLE_ASSERT: &str = "iface->lpVtbl == &d3d12_heap_vtbl";

fn check(level: Level, what: impl Into<String>, fix: Option<&str>) -> Check {
    Check { level, what: what.into(), fix: fix.map(str::to_string) }
}

pub fn checks(f: &Found) -> Vec<Check> {
    let mut v = Vec::new();
    match &f.gpu {
        Err(e) => v.push(check(Level::Error, format!("no D3D12 device: {e}"), Some("update the GPU driver"))),
        Ok(g) => {
            if g.warp {
                v.push(check(Level::Warning, "GPU: WARP (software renderer) – no hardware D3D12 device was usable", Some("update the GPU driver; d3d12_adapter in xcl.toml or XCL_ADAPTER picks the GPU")));
            } else {
                v.push(check(Level::Ok, format!("GPU: {}", g.name), None));
            }
            if g.shader_model < (6, 6) {
                v.push(check(Level::Error, format!("{} (has {}.{}): shaders cannot run", g.no_sm66, g.shader_model.0, g.shader_model.1), Some("update the GPU driver; on Windows 10, put the D3D12 Agility SDK's D3D12Core.dll in a D3D12 folder next to xcl.exe")));
            }
            if !g.mesh {
                v.push(check(Level::Warning, "no mesh shaders: the game's geometry and NGG programs are not drawn", Some("a GPU with D3D12 mesh shaders is needed (NVIDIA RTX 20 / GTX 16 or newer, AMD RX 6000 or newer, Intel Arc)")));
            }
            let wave64 = g.wave_min <= 64 && g.wave_max >= 64;
            let wave32 = g.wave_min <= 32 && g.wave_max >= 32;
            if !wave64 && wave32 {
                v.push(check(Level::Note, format!("waves of {}–{} lanes: the console's 64-lane programs run as two 32-lane halves (slower; vertex / pixel shaders that read lanes 32–63 can differ)", g.wave_min, g.wave_max), None));
            } else if !wave64 && !wave32 && g.wave_max > 0 {
                v.push(check(Level::Error, format!("waves of {}–{} lanes: neither 64 nor 32, the console's programs cannot run", g.wave_min, g.wave_max), None));
            }
        }
    }
    match &f.dxc {
        Ok(dir) => {
            use crate::gpu::recomp::dxc;
            let t = &f.dxc_toolchain;
            v.push(check(Level::Ok, format!("DXC {}: {dir}", t.dxcompiler), None));
            if t.outdated() {
                v.push(check(Level::Note, format!("DXC is older than {}.{}: {}", dxc::RECOMMENDED.0, dxc::RECOMMENDED.1, dxc::OUTDATED_COST), Some(dxc::OUTDATED_FIX)));
            }
        }
        Err(_) => v.push(check(Level::Error, "no DXC (dxcompiler.dll + dxil.dll): no shader can be compiled, the game draws nothing", Some("install the Windows SDK, or put dxcompiler.dll and dxil.dll next to xcl.exe (or in %XCL_DXC_DIR%)"))),
    }
    if !f.xgameruntime {
        v.push(check(Level::Warning, "no System32\\xgameruntime.dll: dialogs, text entry, game invites and game streaming are missing (XCL's own task queues, async and networking are used)", Some("install Gaming Services (Microsoft Store)")));
    }
    if let Some(w) = &f.wine {
        if w.vkd3d_asserts_vtables == Some(true) {
            v.push(check(
                Level::Error,
                format!("this vkd3d-proton stops on the objects XCL hands to D3D12 (its d3d12core.dll holds the assertion `{VKD3D_VTABLE_ASSERT}`, upstream issue #241): the game aborts as soon as it uses one of them"),
                Some("use the vkd3d-proton built by scripts/linux/build-vkd3d-proton.sh (patched, assertions off) – see docs/linux.md"),
            ));
        }
        if w.dxgi_is_dxvk == Some(false) {
            v.push(check(
                Level::Warning,
                "dxgi.dll is Wine's, not DXVK's: presenting crashes in wined3d (upstream issue #175)",
                Some("copy DXVK's dxgi.dll, d3d11.dll and d3d10core.dll into the prefix's System32 and run with WINEDLLOVERRIDES=\"d3d12,d3d12core,dxgi,d3d11,d3d10core=n,b\" – see docs/linux.md"),
            ));
        }
    }
    v
}

/// The adapter `setting` (`d3d12_adapter`, overridden by `XCL_ADAPTER`) picks – the same
/// choice and the same fallback to WARP as the D3D12 backend's device.
fn probe_gpu(setting: &str) -> Result<Gpu, String> {
    use crate::gpu::adapter::{self, Selector};
    use windows::Win32::Graphics::Direct3D::*;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::*;
    unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| format!("DXGI: {e}"))?;
        let sel = Selector::from_env_or(setting);
        let mut found: Option<(ID3D12Device, String)> = None;
        if sel != Selector::Warp {
            let mut list = adapter::candidates(&factory, &sel);
            if list.is_empty() {
                list = adapter::candidates(&factory, &Selector::Auto);
            }
            for (a, desc) in list {
                if let Ok(dev) = crate::gpu::agility::create_device(&a, D3D_FEATURE_LEVEL_12_0) {
                    found = Some((dev, format!("{} ({} MiB VRAM)", adapter::name(&desc), desc.DedicatedVideoMemory >> 20)));
                    break;
                }
            }
        }
        let warp = found.is_none();
        let (dev, name) = match found {
            Some(x) => x,
            None => {
                let w: IDXGIAdapter = factory.EnumWarpAdapter().map_err(|e| format!("WARP: {e}"))?;
                let d = crate::gpu::agility::create_device(&w, D3D_FEATURE_LEVEL_11_0).map_err(|e| format!("WARP: {e}"))?;
                (d, "WARP (software renderer)".into())
            }
        };
        // the highest model asked for that the runtime knows; a lower one comes back if that is all
        let mut shader_model = (0, 0);
        for m in [D3D_SHADER_MODEL_6_6, D3D_SHADER_MODEL_6_5, D3D_SHADER_MODEL_6_0] {
            let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: m };
            if dev.CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &mut sm as *mut _ as _, std::mem::size_of_val(&sm) as u32).is_ok() {
                let x = sm.HighestShaderModel.0 as u32;
                shader_model = ((x >> 4) as u8, (x & 15) as u8);
                break;
            }
        }
        let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        let _ = dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &mut o1 as *mut _ as _, std::mem::size_of_val(&o1) as u32);
        let mut o7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        let mesh = dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &mut o7 as *mut _ as _, std::mem::size_of_val(&o7) as u32).is_ok()
            && o7.MeshShaderTier.0 >= D3D12_MESH_SHADER_TIER_1.0;
        let no_sm66 = crate::gpu::agility::format_no_sm66_message(&name, crate::gpu::agility::state());
        Ok(Gpu { name, warp, shader_model, no_sm66, mesh, wave_min: o1.WaveLaneCountMin, wave_max: o1.WaveLaneCountMax })
    }
}

fn xgameruntime_present() -> bool {
    let root = std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into());
    std::path::Path::new(&root).join("System32").join("xgameruntime.dll").is_file()
}

/// Whether the image of the loaded module `name` holds `needle`; `None` when it is not loaded.
/// Only committed, readable pages of the module's own allocation are read.
fn image_contains(name: &str, needle: &[u8]) -> Option<bool> {
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::System::Memory::*;
    let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    // SAFETY: `wide` is NUL-terminated; the regions read are the ones VirtualQuery reports as
    // committed and readable, inside the allocation of the module's base address.
    unsafe {
        let base = GetModuleHandleW(wide.as_ptr()) as *const u8;
        if base.is_null() {
            return None;
        }
        let readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
        let mut at = base;
        loop {
            let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            if VirtualQuery(at as _, &mut mbi, std::mem::size_of_val(&mbi)) == 0 || mbi.AllocationBase as *const u8 != base {
                return Some(false);
            }
            if mbi.State == MEM_COMMIT && mbi.Protect & readable != 0 && mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS) == 0 {
                let region = std::slice::from_raw_parts(mbi.BaseAddress as *const u8, mbi.RegionSize);
                if region.windows(needle.len()).any(|w| w == needle) {
                    return Some(true);
                }
            }
            at = (mbi.BaseAddress as *const u8).add(mbi.RegionSize);
        }
    }
}

/// The libraries the process has by now (after the GPU probe has loaded them), under Wine.
fn wine_libs() -> Option<WineLibs> {
    crate::util::under_wine().then(|| WineLibs {
        vkd3d_asserts_vtables: image_contains("d3d12core.dll", VKD3D_VTABLE_ASSERT.as_bytes()),
        dxgi_is_dxvk: image_contains("dxgi.dll", b"DXVK"),
    })
}

/// Check this PC for a run with the adapter `setting`.
pub fn run(setting: &str) -> Vec<Check> {
    let gpu = probe_gpu(setting);
    let found = Found {
        dxc: crate::gpu::recomp::dxc::location().map(|p| p.display().to_string()),
        dxc_toolchain: crate::gpu::recomp::dxc::toolchain(),
        xgameruntime: xgameruntime_present(),
        gpu,
        wine: wine_libs(),
    };
    checks(&found)
}

/// The checks as console lines: OK ones first as `  label`, the rest as WARNING / ERROR with
/// what to do.
pub fn print(checks: &[Check]) {
    for c in checks {
        let tag = match c.level {
            Level::Ok => "  ",
            Level::Note => "  NOTE: ",
            Level::Warning => "  WARNING: ",
            Level::Error => "  ERROR: ",
        };
        println!("{tag}{}", c.what);
        if let Some(fix) = &c.fix {
            println!("           → {fix}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> Gpu {
        Gpu { name: "Radeon RX 9070 (16304 MiB VRAM)".into(), warp: false, shader_model: (6, 6), no_sm66: "x: no SM 6.6".into(), mesh: true, wave_min: 32, wave_max: 64 }
    }

    fn found(gpu: Gpu) -> Found {
        Found { dxc: Ok(r"C:\dxc".into()), dxc_toolchain: dxc(1, 9), xgameruntime: true, gpu: Ok(gpu), wine: None }
    }

    fn dxc(major: u32, minor: u32) -> crate::gpu::recomp::dxc::Toolchain {
        let dxcompiler = crate::gpu::recomp::dxc::Library { major, minor, ..Default::default() };
        crate::gpu::recomp::dxc::Toolchain { dir: r"C:\dxc".into(), dxcompiler, ..Default::default() }
    }

    fn levels(f: &Found) -> Vec<Level> {
        checks(f).into_iter().map(|c| c.level).collect()
    }

    #[test]
    fn a_complete_pc_has_nothing_to_report() {
        assert_eq!(levels(&found(gpu())), [Level::Ok, Level::Ok]);
    }

    #[test]
    fn each_missing_piece_is_reported_with_a_fix() {
        let mut f = found(Gpu { shader_model: (6, 5), mesh: false, ..gpu() });
        f.dxc = Err("dxcompiler.dll not found".into());
        f.xgameruntime = false;
        let c = checks(&f);
        assert_eq!(c.iter().map(|c| c.level).collect::<Vec<_>>(), [Level::Ok, Level::Error, Level::Warning, Level::Error, Level::Warning]);
        assert!(c[1].what.contains("has 6.5"), "{:?}", c[1]);
        // XCL's own task queues and networking stand in for the PC runtime's: not missing
        let missing = c[4].what.split(" missing").next().unwrap();
        assert!(missing.contains("dialogs") && !missing.contains("task queues") && !missing.contains("networking"), "{:?}", c[4]);
        assert!(c.iter().filter(|c| c.level != Level::Ok).all(|c| c.fix.is_some()), "{c:?}");
    }

    #[test]
    fn wave_width() {
        // NVIDIA: 32 only – the split form, a note
        let c = checks(&found(Gpu { wave_min: 32, wave_max: 32, ..gpu() }));
        assert_eq!(c[1].level, Level::Note);
        assert!(c[1].what.contains("two 32-lane halves"), "{:?}", c[1]);
        // 64 in range: nothing; 8–16 (neither size): an error
        assert_eq!(levels(&found(Gpu { wave_min: 64, wave_max: 64, ..gpu() })), [Level::Ok, Level::Ok]);
        assert_eq!(levels(&found(Gpu { wave_min: 8, wave_max: 16, ..gpu() }))[1], Level::Error);
    }

    #[test]
    fn an_old_dxc_is_a_note_with_a_fix() {
        let c = checks(&Found { dxc_toolchain: dxc(1, 7), ..found(gpu()) });
        assert_eq!(c.iter().map(|c| c.level).collect::<Vec<_>>(), [Level::Ok, Level::Ok, Level::Note]);
        assert_eq!(c[1].what, r"DXC 1.7: C:\dxc");
        assert!(c[2].what.contains("older than 1.8") && c[2].fix.as_deref().is_some_and(|f| f.contains("XCL_DXC_DIR")), "{:?}", c[2]);
        assert_eq!(levels(&Found { dxc_toolchain: dxc(1, 8), ..found(gpu()) }), [Level::Ok, Level::Ok]);
    }

    #[test]
    fn warp_and_no_device() {
        assert_eq!(levels(&found(Gpu { warp: true, ..gpu() }))[0], Level::Warning);
        let f = Found { gpu: Err("DXGI: E_FAIL".into()), ..found(gpu()) };
        assert_eq!(checks(&f)[0].level, Level::Error);
    }

    #[test]
    fn checks_round_trip_through_the_report() {
        let c = checks(&Found { dxc: Err("x".into()), ..found(gpu()) });
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains("\"level\":\"error\""), "{json}");
        assert_eq!(serde_json::from_str::<Vec<Check>>(&json).unwrap(), c);
    }
}
