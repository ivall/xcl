//! xcl.exe – command line front-end and (in host mode) the process the game runs in.

use std::ffi::c_void;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Our own TLS callback: ntdll calls it for every thread start/exit in the process.
/// In host mode it initialises the game's static TLS for the new thread and forwards
/// the notification to the game's TLS callbacks / DllMains.
unsafe extern "system" fn xcl_tls_callback(_h: *mut c_void, reason: u32, _r: *mut c_void) {
    if xcl::host::HOST_MODE.load(Ordering::Acquire) {
        xcl::loader::on_tls_event(reason);
    }
}

#[used]
#[link_section = ".CRT$XLY"]
static XCL_TLS_CALLBACK: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) = xcl_tls_callback;

const HELP: &str = "\
XCL – Xbox Series X|S → Windows compatibility layer

Usage:
  xcl run <game folder | file.exe> [-- game arguments]
      Runs the game. The game folder is only read; writes, game saves and logs go to
      %LOCALAPPDATA%\\XCL\\<title>\\ (or %XCL_DATA_ROOT%\\<title>\\).
  xcl inspect <file.exe|file.dll> [--json]
      Analysis without running: imports and what XCL has for them (HLE / Windows / stub).
  xcl disasm <file> [--offset N] [--length N] [--all] [--hex] [--fssm FILE]
      Disassembler for console GPU code (RDNA2) – e.g. for shaders saved in <data>\\shaders\\.
      Console containers (XDAT/XBCH/FSSM) are unpacked; --fssm saves the unpacked FSSM.
  xcl shaders <file | folder | package.rpf> [--top N] [--json] [--recomp] [--compile N]
      Shader corpus statistics: formats, stages, RDNA2 instruction histogram (priorities
      for the recompiler). RPF9 packages are scanned for shader records.
      --recomp: translates every RDNA2 program → HLSL and shows what is missing;
      --compile N: additionally compiles N complete translations with DXC.
      --precompile [--only STAGE]: caches compiled shaders before a run (e.g. --only cs).
      --prepare-corpus: builds every staged native pipeline recipe for a title folder.
          --pipelines also builds compute pipelines and caches them for the next run.
          --used-only skips compute shaders whose pipelines were not used in earlier runs.
      --export-hlsl DIR [--count N] [--min-kb N] [--record-hash HEX]: writes deterministic
          compute HLSL for investigation; an exact hash bypasses the size threshold.
      --measure-pso HEX [--repeat N]: times cache-disabled driver compilation of one exact
          compute record with XCL's real root signature and static samplers.
      --direct [--sample N] [--seed S] [--validate] [--compare]: direct DXIL translator
      coverage over a seeded sample (default 200 programs, 0 = all); also
      --timeout, --max-bytes, --threads, --json, --hash H, --dump DIR.
  xcl pso-precompile <game folder | shaders.rpf> [--threads N] [--records FILE] [--used]
      Before a run: translates the game's compute programs (its shaders.rpf) and has the
      driver compile their pipelines, the ones an earlier run used first, biggest first.
      Fills the caches the run reads (DXIL, driver blobs); what is cached is skipped.
      --records FILE / --used: only the programs of these record hashes / pipeline keys.
      Run it with the run's XCL_DIRECT_DXIL / XCL_HEAP_WINDOWS: they change the keys.
  xcl rpf <package.rpf> [--top N] [--find NAME] [--extract NAME --out DIR]
      Lists or extracts readable inline records. Archives larger than 1 GB use a
      bounded-memory header probe; --probe requests it explicitly and --full overrides it.
      Use --probe only for header summaries, without --find, --extract, --out or --full.
  xcl --version
";

fn main() {
    // keep the TLS callback from being discarded by the linker
    unsafe { std::ptr::read_volatile(&XCL_TLS_CALLBACK) };

    if let Some(boot) = std::env::var_os("XCL_HOST") {
        std::env::remove_var("XCL_HOST");
        std::process::exit(xcl::host::host_main(&PathBuf::from(boot)));
    }

    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("run") if args.len() >= 2 => {
            let rest: Vec<String> = match args.iter().position(|a| a == "--") {
                Some(i) => args[i + 1..].to_vec(),
                None => args[2..].to_vec(),
            };
            xcl::launcher::run(&PathBuf::from(&args[1]), &rest)
        }
        Some("inspect") if args.len() >= 2 => {
            let json = args.iter().any(|a| a == "--json");
            xcl::inspect::run(&PathBuf::from(&args[1]), json)
        }
        Some("disasm") if args.len() >= 2 => xcl::disasm::run(&PathBuf::from(&args[1]), &args[2..]),
        Some("shaders") if args.len() >= 2 => xcl::shaders::run(&PathBuf::from(&args[1]), &args[2..]),
        Some("pso-precompile") if args.len() >= 2 => xcl::shaders::pso_precompile(&PathBuf::from(&args[1]), &args[2..]),
        Some("rpf") if args.len() >= 2 => xcl::rpf::run(&PathBuf::from(&args[1]), &args[2..]),
        Some("--version") | Some("-V") => {
            println!("xcl {}", env!("CARGO_PKG_VERSION"));
            0
        }
        None if started_by_double_click() => run_picked_game(),
        _ => {
            print!("{HELP}");
            if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
                0
            } else {
                2
            }
        }
    };
    std::process::exit(code);
}

/// xcl.exe opened from Explorer: the console is its own (from a terminal, the shell shares it).
fn started_by_double_click() -> bool {
    let mut ids = [0u32; 4];
    unsafe { windows_sys::Win32::System::Console::GetConsoleProcessList(ids.as_mut_ptr(), ids.len() as u32) == 1 }
}

/// No command line: a small menu (choose a game / run the last one again), then the game runs
/// in this console. On an error the console stays open until Enter, so the message can be read.
fn run_picked_game() -> i32 {
    let last_file = xcl::launcher::data_root().join("last-game.txt");
    let last = std::fs::read_to_string(&last_file).ok().map(|s| PathBuf::from(s.trim())).filter(|p| p.exists());
    let target = {
        let _ui = Ui::enter();
        let picked = match menu(last.as_deref()) {
            Choice::Again => last.clone(),
            Choice::Cancel => None,
            Choice::Choose | Choice::NoMenu => pick_game_file(),
        };
        // MicrosoftGame.config stands for its folder: XCL picks the exe the config names
        picked.map(|file| match file.file_name() {
            Some(n) if n.eq_ignore_ascii_case("MicrosoftGame.config") => file.parent().map(PathBuf::from).unwrap_or(file),
            _ => file,
        })
    };
    let Some(target) = target else { return 0 };
    let _ = std::fs::create_dir_all(xcl::launcher::data_root());
    let _ = std::fs::write(&last_file, target.to_string_lossy().as_bytes());
    println!("xcl run \"{}\"", target.display());
    let code = xcl::launcher::run_from_menu(&target);
    if code != 0 {
        println!("\nXCL ended with code {code}. Press Enter to close.");
        let _ = std::io::stdin().read_line(&mut String::new());
    }
    code
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// `name` of system DLL `dll`, looked up when the menu needs it: xcl.exe does not import the
/// dialog libraries (the game runs in this executable, and must not get them).
unsafe fn system_proc(dll: &str, name: &str) -> Option<usize> {
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    let lib = LoadLibraryW(wide(dll).as_ptr());
    if lib.is_null() {
        return None;
    }
    GetProcAddress(lib, format!("{name}\0").as_ptr()).map(|f| f as usize)
}

const MENU_MANIFEST: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <dependency><dependentAssembly>
    <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"/>
  </dependentAssembly></dependency>
</assembly>
"#;

static MANIFEST_ID: AtomicU64 = AtomicU64::new(0);

/// Exclusively created temporary manifest file removed automatically on drop.
struct TempManifest(PathBuf);

impl TempManifest {
    fn path(&self) -> &Path {
        &self.0
    }

    fn create_in(dir: &Path, content: &[u8]) -> std::io::Result<Self> {
        Self::create_in_with_counter(dir, content, &MANIFEST_ID)
    }

    fn create_in_with_counter(dir: &Path, content: &[u8], counter: &AtomicU64) -> std::io::Result<Self> {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;
        let pid = std::process::id();
        let (path, file) = (0..1024)
            .find_map(|_| {
                let seq = counter.fetch_add(1, Ordering::Relaxed);
                let candidate = dir.join(format!("xcl-menu-{pid}-{seq}.manifest"));
                match std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .share_mode(0)
                    .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
                    .open(&candidate)
                {
                    Ok(file) => Some(Ok((candidate, file))),
                    Err(err)
                        if err.kind() == std::io::ErrorKind::AlreadyExists
                            || (err.kind() == std::io::ErrorKind::PermissionDenied && candidate.symlink_metadata().is_ok()) =>
                    {
                        None
                    }
                    Err(err) => Some(Err(err)),
                }
            })
            .unwrap_or_else(|| Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "exhausted temp manifest names")))?;
        // Declare `guard` before the write scope so `file` always closes before `TempManifest::drop` deletes it.
        let guard = TempManifest(path);
        {
            let mut file = file;
            file.write_all(content)?;
            file.flush()?;
        }
        Ok(guard)
    }
}

impl Drop for TempManifest {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

unsafe fn activate_manifest_in(dir: &Path) -> Option<(windows_sys::Win32::Foundation::HANDLE, usize)> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::System::ApplicationInstallationAndServicing::{ActivateActCtx, CreateActCtxW, ReleaseActCtx, ACTCTXW};
    let manifest = TempManifest::create_in(dir, MENU_MANIFEST.as_bytes()).ok()?;
    let source: Vec<u16> = manifest.path().as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let mut c: ACTCTXW = std::mem::zeroed();
    c.cbSize = std::mem::size_of::<ACTCTXW>() as u32;
    c.lpSource = source.as_ptr();
    let h = CreateActCtxW(&c);
    drop(manifest);
    if h == INVALID_HANDLE_VALUE {
        return None;
    }
    let mut cookie = 0usize;
    if ActivateActCtx(h, &mut cookie) == 0 {
        ReleaseActCtx(h);
        return None;
    }
    Some((h, cookie))
}

/// While the menu is open: COM for the file dialog, and the common controls 6 (the task
/// dialog, the current look) through an activation context of its own – not a manifest of
/// xcl.exe, which would apply to the game too.
struct Ui {
    com_init: bool,
    actctx: Option<(windows_sys::Win32::Foundation::HANDLE, usize)>,
}

impl Ui {
    fn enter() -> Ui {
        Self::enter_in(&std::env::temp_dir())
    }

    fn enter_in(dir: &Path) -> Ui {
        let com_init = unsafe {
            windows_sys::Win32::System::Com::CoInitializeEx(std::ptr::null(), windows_sys::Win32::System::Com::COINIT_APARTMENTTHREADED as u32) >= 0
        };
        let actctx = unsafe { activate_manifest_in(dir) };
        Ui { com_init, actctx }
    }
}

impl Drop for Ui {
    fn drop(&mut self) {
        use windows_sys::Win32::System::ApplicationInstallationAndServicing::{DeactivateActCtx, ReleaseActCtx};
        unsafe {
            if let Some((h, cookie)) = self.actctx {
                DeactivateActCtx(0, cookie);
                ReleaseActCtx(h);
            }
            if self.com_init {
                windows_sys::Win32::System::Com::CoUninitialize();
            }
        }
    }
}

enum Choice {
    Choose,
    Again,
    Cancel,
    /// no task dialog on this system: straight to the file dialog
    NoMenu,
}

/// The menu: a task dialog with "Choose a game…" and, after a first run, "Run again".
fn menu(last: Option<&std::path::Path>) -> Choice {
    use windows_sys::Win32::UI::Controls::*;
    const CHOOSE: i32 = 100;
    const AGAIN: i32 = 101;
    type TaskDialogIndirect = unsafe extern "system" fn(*const TASKDIALOGCONFIG, *mut i32, *mut i32, *mut i32) -> i32;
    let Some(f) = (unsafe { system_proc("comctl32.dll", "TaskDialogIndirect") }) else { return Choice::NoMenu };
    let title = wide("XCL");
    let heading = wide("Run an Xbox Series game");
    let content = wide("XCL runs the game from its unpacked files. Choose the game's exe or its MicrosoftGame.config.");
    let again = last.map(|p| wide(&format!("Run again\n{}", p.display())));
    let choose = wide("Choose a game…\nThe game's exe or MicrosoftGame.config");
    let mut buttons = Vec::new();
    if let Some(a) = &again {
        buttons.push(TASKDIALOG_BUTTON { nButtonID: AGAIN, pszButtonText: a.as_ptr() });
    }
    buttons.push(TASKDIALOG_BUTTON { nButtonID: CHOOSE, pszButtonText: choose.as_ptr() });
    let mut c: TASKDIALOGCONFIG = unsafe { std::mem::zeroed() };
    c.cbSize = std::mem::size_of::<TASKDIALOGCONFIG>() as u32;
    c.hwndParent = unsafe { windows_sys::Win32::System::Console::GetConsoleWindow() };
    c.dwFlags = TDF_USE_COMMAND_LINKS | TDF_ALLOW_DIALOG_CANCELLATION | TDF_POSITION_RELATIVE_TO_WINDOW;
    c.dwCommonButtons = TDCBF_CANCEL_BUTTON;
    c.pszWindowTitle = title.as_ptr();
    c.pszMainInstruction = heading.as_ptr();
    c.pszContent = content.as_ptr();
    c.cButtons = buttons.len() as u32;
    c.pButtons = buttons.as_ptr();
    c.nDefaultButton = if again.is_some() { AGAIN } else { CHOOSE };
    let mut pressed = 0i32;
    let hr = unsafe { std::mem::transmute::<usize, TaskDialogIndirect>(f)(&c, &mut pressed, std::ptr::null_mut(), std::ptr::null_mut()) };
    match (hr, pressed) {
        (h, _) if h < 0 => Choice::NoMenu,
        (_, CHOOSE) => Choice::Choose,
        (_, AGAIN) => Choice::Again,
        _ => Choice::Cancel,
    }
}

/// The standard Windows open dialog (None: cancelled).
fn pick_game_file() -> Option<PathBuf> {
    use windows_sys::Win32::UI::Controls::Dialogs::*;
    type GetOpenFileName = unsafe extern "system" fn(*mut OPENFILENAMEW) -> i32;
    let f = unsafe { system_proc("comdlg32.dll", "GetOpenFileNameW") }?;
    // pairs of (description, pattern), each ending in NUL, the list in one more
    let filter: Vec<u16> = "Xbox game (*.exe, MicrosoftGame.config)\0*.exe;MicrosoftGame.config\0All files\0*.*\0\0".encode_utf16().collect();
    let title = wide("XCL – choose a game");
    let mut file = vec![0u16; 32768];
    let mut ofn: OPENFILENAMEW = unsafe { std::mem::zeroed() };
    ofn.lStructSize = std::mem::size_of::<OPENFILENAMEW>() as u32;
    ofn.hwndOwner = unsafe { windows_sys::Win32::System::Console::GetConsoleWindow() };
    ofn.lpstrFilter = filter.as_ptr();
    ofn.lpstrFile = file.as_mut_ptr();
    ofn.nMaxFile = file.len() as u32;
    ofn.lpstrTitle = title.as_ptr();
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if unsafe { std::mem::transmute::<usize, GetOpenFileName>(f)(&mut ofn) } == 0 {
        return None;
    }
    let n = file.iter().position(|&c| c == 0).unwrap_or(file.len());
    Some(PathBuf::from(String::from_utf16_lossy(&file[..n])))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            static DIR_ID: AtomicU64 = AtomicU64::new(0);
            let n = DIR_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("xcl-test-2272-{label}-{}-{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn temp_manifest_skips_collisions_and_removes_on_drop() {
        let dir = TestDir::new("collisions");
        let pid = std::process::id();
        let fixed = dir.0.join("xcl-menu.manifest");
        let victim = dir.0.join("victim.txt");
        let col0 = dir.0.join(format!("xcl-menu-{pid}-0.manifest"));
        let col1 = dir.0.join(format!("xcl-menu-{pid}-1.manifest"));
        std::fs::write(&fixed, b"sentinel").unwrap();
        std::fs::write(&victim, b"untouched").unwrap();
        std::fs::hard_link(&victim, &col0).unwrap();
        std::fs::create_dir(&col1).unwrap();

        let counter = AtomicU64::new(0);
        let manifest = TempManifest::create_in_with_counter(&dir.0, MENU_MANIFEST.as_bytes(), &counter).unwrap();
        let created_path = manifest.path().to_path_buf();
        assert_eq!(created_path, dir.0.join(format!("xcl-menu-{pid}-2.manifest")));
        assert_eq!(std::fs::read(&created_path).unwrap(), MENU_MANIFEST.as_bytes());
        assert_eq!(std::fs::read(&fixed).unwrap(), b"sentinel");
        assert_eq!(std::fs::read(&victim).unwrap(), b"untouched");
        assert!(col1.is_dir());

        drop(manifest);
        assert!(!created_path.exists(), "TempManifest::drop must delete the manifest file");
        assert!(TempManifest::create_in_with_counter(&dir.0.join("missing"), b"x", &counter).is_err());
    }

    #[test]
    fn live_actctx_activates_after_manifest_removal_and_ui_guards_com() {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::ApplicationInstallationAndServicing::{
            ActivateActCtx, CreateActCtxW, DeactivateActCtx, GetCurrentActCtx, ReleaseActCtx, ACTCTXW,
        };
        use windows_sys::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED};

        let dir = TestDir::new("actctx");
        let fixed = dir.0.join("xcl-menu.manifest");
        std::fs::write(&fixed, b"sentinel").unwrap();

        // 1. Create activation context, delete TempManifest immediately, then activate and verify.
        unsafe {
            let manifest = TempManifest::create_in(&dir.0, MENU_MANIFEST.as_bytes()).unwrap();
            let path = manifest.path().to_path_buf();
            let source: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
            let mut c: ACTCTXW = std::mem::zeroed();
            c.cbSize = std::mem::size_of::<ACTCTXW>() as u32;
            c.lpSource = source.as_ptr();
            let h = CreateActCtxW(&c);
            drop(manifest);
            assert_ne!(h, INVALID_HANDLE_VALUE, "CreateActCtxW failed");
            assert!(!path.exists(), "manifest file must be gone before ActivateActCtx");
            let mut cookie = 0usize;
            assert_ne!(ActivateActCtx(h, &mut cookie), 0, "ActivateActCtx must succeed after file removal");
            let mut current = std::ptr::null_mut();
            assert_ne!(GetCurrentActCtx(&mut current), 0);
            assert_eq!(current, h);
            ReleaseActCtx(current);
            assert_ne!(DeactivateActCtx(0, cookie), 0);
            ReleaseActCtx(h);
        }

        // 2. Ui::enter_in on a fresh STA thread initializes COM, activates, and leaves zero extra files.
        let sta_dir = dir.0.clone();
        std::thread::spawn(move || {
            let ui = Ui::enter_in(&sta_dir);
            assert!(ui.com_init);
            assert!(ui.actctx.is_some());
            let entries: Vec<_> = std::fs::read_dir(&sta_dir).unwrap().map(|e| e.unwrap().file_name()).collect();
            assert_eq!(entries.len(), 1, "no temporary manifest should remain while Ui is active: {entries:?}");
            drop(ui);
        })
        .join()
        .unwrap();

        // 3. Ui::enter_in on an already-MTA thread observes RPC_E_CHANGED_MODE and skips CoUninitialize.
        let mta_dir = dir.0.clone();
        std::thread::spawn(move || unsafe {
            assert!(CoInitializeEx(std::ptr::null(), COINIT_MULTITHREADED as u32) >= 0);
            let ui = Ui::enter_in(&mta_dir);
            assert!(!ui.com_init, "RPC_E_CHANGED_MODE must not mark COM as initialized by Ui");
            assert!(ui.actctx.is_some());
            drop(ui);
            CoUninitialize();
        })
        .join()
        .unwrap();

        assert_eq!(std::fs::read(&fixed).unwrap(), b"sentinel");
        let entries: Vec<_> = std::fs::read_dir(&dir.0).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(entries.len(), 1, "leftover temp manifests remained: {entries:?}");
    }
}
