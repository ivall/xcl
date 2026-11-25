//! What the window knows without a window: XCL's folders, its log lines, a title's `xcl.toml`,
//! the window's own settings and the two languages of its texts.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// XCL's data root, as `xcl` finds it: `%XCL_DATA_ROOT%` or `%LOCALAPPDATA%\XCL`.
pub fn data_root() -> PathBuf {
    if let Ok(p) = std::env::var("XCL_DATA_ROOT") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    let base = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    PathBuf::from(base).join("XCL")
}

/// The window's own folder under the data root: its settings and the sessions it saved.
pub fn gui_dir() -> PathBuf {
    data_root().join("gui")
}

// ---------------------------------------------------------------- log lines

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity {
    Error,
    Warn,
    Info,
    Detail,
}

/// How serious a line is. A line of `xcl.log` carries its level (`[  1.234][ 1234][WARN ] …`);
/// what `xcl` prints to the console does not, so those go by what they say.
pub fn severity(line: &str) -> Severity {
    if let Some(tag) = log_tag(line) {
        return match tag {
            "ERROR" => Severity::Error,
            "WARN" => Severity::Warn,
            "INFO" => Severity::Info,
            _ => Severity::Detail,
        };
    }
    let l = line.to_ascii_lowercase();
    const ERRORS: [&str; 9] =
        ["crash", "unhandled exception", "device removed", "device was removed", "error", "failed", "panicked", "0x887a", "out of memory"];
    const WARNINGS: [&str; 4] = ["warning", "missing on this pc", "unimplemented", "not available"];
    // cargo's and git's own words for nothing wrong
    if l.contains("0 errors") || l.contains("error: 0") {
        return Severity::Info;
    }
    if ERRORS.iter().any(|w| l.contains(w)) {
        Severity::Error
    } else if WARNINGS.iter().any(|w| l.contains(w)) {
        Severity::Warn
    } else {
        Severity::Info
    }
}

/// The level tag of an `xcl.log` line (third bracket), trimmed.
fn log_tag(line: &str) -> Option<&str> {
    let mut rest = line.strip_prefix('[')?;
    for _ in 0..2 {
        rest = &rest[rest.find("][")? + 2..];
    }
    let tag = rest[..rest.find(']')?].trim();
    matches!(tag, "ERROR" | "WARN" | "INFO" | "DEBUG" | "TRACE").then_some(tag)
}

/// The title's data folder from the line `xcl run` prints at the start (`  data:  C:\…`).
pub fn data_dir_line(line: &str) -> Option<PathBuf> {
    let p = line.trim_start().strip_prefix("data:")?.trim();
    (!p.is_empty()).then(|| PathBuf::from(p))
}

/// The graphics adapter from the line `xcl run` prints at the start (`  GPU: name (N MiB VRAM)`).
pub fn gpu_line(line: &str) -> Option<String> {
    let g = line.trim_start().strip_prefix("GPU:")?.trim();
    (!g.is_empty()).then(|| g.to_string())
}

/// The exit code from `xcl run`'s last words (`Game exited with code 0x… (…)`).
pub fn exit_code_line(line: &str) -> Option<u32> {
    let rest = line.trim().strip_prefix("Game exited with code 0x")?;
    u32::from_str_radix(rest.split_whitespace().next()?, 16).ok()
}

/// What an exit code of the game's process means, where it is one of the known ones.
pub fn exit_code_meaning(code: u32, lang: Lang) -> Option<&'static str> {
    let (en, es) = match code {
        0 => ("ended normally", "terminó normalmente"),
        0x887a_0005 | 0x887a_0007 => ("the graphics device was removed", "se perdió el dispositivo gráfico"),
        0x887a_0006 => ("the GPU stopped answering (device hung)", "la GPU dejó de responder (colgada)"),
        0x8007_000e => ("out of memory", "sin memoria"),
        0xc000_0005 => ("access violation (crash)", "violación de acceso (caída)"),
        0xc000_001d => ("illegal instruction: the CPU lacks an extension", "instrucción ilegal: al procesador le falta una extensión"),
        0xc000_0409 => ("stack check failed (crash)", "falló la comprobación de pila (caída)"),
        0x8000_0003 => ("the title's own assert (breakpoint)", "aserción del propio juego (breakpoint)"),
        1 | 2 => ("XCL could not start the game", "XCL no pudo iniciar el juego"),
        _ => return None,
    };
    Some(if lang == Lang::Es { es } else { en })
}

/// Cargo's progress line (`    Building [====>     ] 12/345: xcl, windows`), as it prints it with
/// `CARGO_TERM_PROGRESS_WHEN=always`: the units done, their number and what is being built.
pub fn cargo_progress(line: &str) -> Option<(u32, u32, String)> {
    let rest = line.trim_start().strip_prefix("Building")?.trim_start().strip_prefix('[')?;
    let rest = rest[rest.find(']')? + 1..].trim_start();
    let (counts, names) = rest.split_once(':').unwrap_or((rest, ""));
    let (done, total) = counts.trim().split_once('/')?;
    let (done, total): (u32, u32) = (done.trim().parse().ok()?, total.trim().parse().ok()?);
    (total > 0 && done <= total).then(|| (done, total, names.trim().trim_end_matches("...").trim().to_string()))
}

/// `xcl shaders --precompile` says how far it is on a line of its own, `  22400/26503 (463 s)`:
/// shaders done, their number, and the seconds the pass has run.
pub fn precompile_progress(line: &str) -> Option<(u32, u32, u64)> {
    let (counts, rest) = line.trim().split_once(' ')?;
    let (done, total) = counts.split_once('/')?;
    let secs: f64 = rest.trim().strip_prefix('(')?.strip_suffix("s)")?.trim().parse().ok()?;
    let (done, total): (u32, u32) = (done.parse().ok()?, total.parse().ok()?);
    (total > 0 && done <= total && secs >= 0.0).then_some((done, total, secs as u64))
}

/// `  pipelines: 400 made, 0 refused` of the same pass: the compute pipelines the driver built
/// and the ones it refused.
pub fn pipelines_line(line: &str) -> Option<(u32, u32)> {
    let (made, refused) = line.trim().strip_prefix("pipelines:")?.split_once(',')?;
    Some((made.trim().strip_suffix("made")?.trim().parse().ok()?, refused.trim().strip_suffix("refused")?.trim().parse().ok()?))
}

/// `--used-only: 499 pipelines earlier runs used (used-pipelines.txt); ...` of the same pass:
/// the pipelines it set out to build.
pub fn used_pipelines_line(line: &str) -> Option<u32> {
    line.trim().strip_prefix("--used-only:")?.split_whitespace().next()?.parse().ok()
}

/// `compiled 812, already cached 25691, errors 0, skipped (GS ...) 3, time 512.3 s`, the pass's
/// last word on its shaders: compiled now, found in the cache, failed.
pub fn precompile_result(line: &str) -> Option<(u32, u32, u32)> {
    let mut parts = line.trim().split(',');
    let compiled = parts.next()?.trim().strip_prefix("compiled ")?.trim().parse().ok()?;
    let cached = parts.next()?.trim().strip_prefix("already cached ")?.trim().parse().ok()?;
    let errors = parts.next()?.trim().strip_prefix("errors ")?.trim().parse().ok()?;
    Some((compiled, cached, errors))
}

/// The time a pass still takes, from how fast its last reports came. A pass that goes on from
/// an earlier one reports thousands of shaders done in its first seconds (they are skipped,
/// not built), so the speed is taken between reports and never from zero.
#[derive(Clone, Debug, Default)]
pub struct Eta {
    reports: std::collections::VecDeque<(u32, u64)>,
}

impl Eta {
    /// Reports the speed is taken over: the last ones, as a pass's shaders differ a lot in size.
    const KEEP: usize = 6;

    pub fn report(&mut self, done: u32, secs: u64) {
        if self.reports.back().is_some_and(|&(d, s)| done < d || secs < s) {
            // another pass
            self.reports.clear();
        }
        if self.reports.back() != Some(&(done, secs)) {
            self.reports.push_back((done, secs));
        }
        while self.reports.len() > Self::KEEP {
            self.reports.pop_front();
        }
    }

    /// Seconds left for `total`, once two reports tell a speed.
    pub fn left(&self, total: u32) -> Option<u64> {
        let (&(d0, s0), &(d1, s1)) = (self.reports.front()?, self.reports.back()?);
        (d1 > d0 && s1 > s0).then(|| total.saturating_sub(d1) as u64 * (s1 - s0) / (d1 - d0) as u64)
    }
}

/// `secs` as a person says it: `2 h 05 min`, `12 min`, `less than a minute`.
pub fn duration_text(secs: u64, lang: Lang) -> String {
    let minutes = secs.div_ceil(60);
    match (minutes / 60, minutes % 60) {
        (0, 0 | 1) if secs < 60 => tr(lang, "under_minute").to_string(),
        (0, m) => format!("{m} min"),
        (h, m) => format!("{h} h {m:02} min"),
    }
}

/// The game's shader packages (`*shader*.rpf`), for `xcl shaders --precompile`: in the game's
/// folder and up to two folders below it. `game` is that folder, or the game's exe or config.
pub fn shader_packs(game: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, depth: usize, found: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_lowercase();
            if path.is_dir() {
                if depth > 0 {
                    walk(&path, depth - 1, found);
                }
            } else if name.ends_with(".rpf") && name.contains("shader") {
                found.push(path);
            }
        }
    }
    let dir = if game.is_dir() { game } else { game.parent().unwrap_or(game) };
    let mut found = Vec::new();
    walk(dir, 2, &mut found);
    found.sort();
    found
}

/// A full commit id, as `git rev-parse HEAD` prints it.
pub fn is_commit_id(line: &str) -> bool {
    line.len() == 40 && line.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Where XCL's shader translators are: a change there gives translated shaders another identity
/// (`XCL_TRANSLATOR_SOURCE_ID`), and what was precompiled before is translated and built again.
pub const TRANSLATOR_PATHS: [&str; 2] = ["crates/xcl/src/gpu", "crates/xcl/build.rs"];

// ---------------------------------------------------------------- the window's settings

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    #[default]
    En,
    Es,
}

/// The window's colours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Theme {
    /// The project's own: the night blue, violet and cyan of its pictures.
    Xcl,
    /// Black and greys, for who wants no colour.
    Black,
    Light,
}

impl Theme {
    /// The one the theme button switches to.
    pub fn next(self) -> Theme {
        match self {
            Theme::Xcl => Theme::Black,
            Theme::Black => Theme::Light,
            Theme::Light => Theme::Xcl,
        }
    }

    /// The text key of the theme's name.
    pub fn key(self) -> &'static str {
        match self {
            Theme::Xcl => "theme_xcl",
            Theme::Black => "theme_black",
            Theme::Light => "theme",
        }
    }
}

/// What the window keeps between runs (`gui\settings.toml` under the data root).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub lang: Lang,
    /// The game: its folder, its exe or its `MicrosoftGame.config`.
    pub game: String,
    /// `xcl.exe`; empty = next to this program, then `target\release` of `repo`.
    pub xcl_exe: String,
    /// XCL's source folder, for updating and building; empty = found from this program's place.
    pub repo: String,
    /// The title's data folder of the last run (where its `xcl.toml` and logs are).
    pub last_data_dir: String,
    /// `XCL_NETWORK=0`: the title's network calls fail as offline.
    pub offline: bool,
    /// `XCL_PIPELINE_THREADS`; 0 = XCL's own number.
    pub pipeline_threads: u32,
    /// `XCL_RUN_SECONDS`: end the run after this long; 0 = no limit.
    pub run_seconds: u32,
    /// More variables, one `NAME=value` a line.
    pub extra_env: String,
    /// Arguments passed on to the game.
    pub game_args: String,
    /// Keep what a run printed under `gui\sessions`.
    pub save_sessions: bool,
    /// `XCL_DIRECT_DXIL`: "" (the HLSL path), "1" (compute shaders go the direct way) or "all"
    /// (vertex and pixel shaders too).
    pub direct_dxil: String,
    /// `XCL_DXC_THREADS`; 0 = XCL's own number (the cores less one).
    pub dxc_threads: u32,
    /// Off = `XCL_TRANSLATED_CACHE=0`: translated shaders are not kept on disk.
    pub shader_cache: bool,
    /// Off = `XCL_PSO_DISK_CACHE=0`: the driver's pipelines are not kept on disk.
    pub pipeline_cache: bool,
    /// Off = `XCL_ASYNC_PSO=0`: pipelines are built where the title asks for them.
    pub async_pipelines: bool,
    /// `XCL_DRED=1`: the GPU's last operations are logged when the device is removed.
    pub dred: bool,
    /// The window's colours before it had three themes: dark or light. Still written, so that
    /// an older build of the window reads its own.
    pub dark: bool,
    /// The window's colours: `xcl` (the project's own, the default), `black` or `light`,
    /// whatever Windows uses. Empty = from `dark`.
    pub theme: String,
    /// The graphics adapter `xcl run` named at its last run, for the home page.
    pub last_gpu: String,
    /// The game's shader package for the precompile pass; empty = found in the game's folder.
    pub shader_pack: String,
    /// `--used-only`: precompile only the compute pipelines earlier runs of the title used.
    pub used_only: bool,
    /// The precompile pass compiles the vertex and pixel shaders too (no `--only cs`): without
    /// them the title compiles those itself, on a black screen.
    pub all_stages: bool,
    /// When the last precompile pass that ran to its end did so (UTC), and what it reported:
    /// shaders compiled, already in the cache, errors. Empty = none finished from this window.
    pub precompiled_at: String,
    pub precompiled_result: String,
    /// An update changed XCL's shader translators since the last precompile pass.
    pub precompile_stale: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            lang: Lang::default(),
            game: String::new(),
            xcl_exe: String::new(),
            repo: String::new(),
            last_data_dir: String::new(),
            offline: true,
            pipeline_threads: 0,
            run_seconds: 0,
            extra_env: String::new(),
            game_args: String::new(),
            save_sessions: true,
            direct_dxil: String::new(),
            dxc_threads: 0,
            shader_cache: true,
            pipeline_cache: true,
            async_pipelines: true,
            dred: false,
            dark: true,
            theme: String::new(),
            last_gpu: String::new(),
            shader_pack: String::new(),
            used_only: true,
            all_stages: true,
            precompiled_at: String::new(),
            precompiled_result: String::new(),
            precompile_stale: false,
        }
    }
}

impl Settings {
    pub fn load(path: &Path) -> Settings {
        std::fs::read_to_string(path).ok().and_then(|s| toml::from_str(&s).ok()).unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let body = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, body).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The variables a run gets, in the order shown. Lines of `extra_env` that are not
    /// `NAME=value` are returned apart, to be shown as refused.
    pub fn environment(&self) -> (Vec<(String, String)>, Vec<String>) {
        let mut env = Vec::new();
        let mut refused = Vec::new();
        if self.offline {
            env.push(("XCL_NETWORK".to_string(), "0".to_string()));
        }
        if self.pipeline_threads > 0 {
            env.push(("XCL_PIPELINE_THREADS".to_string(), self.pipeline_threads.to_string()));
        }
        if self.run_seconds > 0 {
            env.push(("XCL_RUN_SECONDS".to_string(), self.run_seconds.to_string()));
        }
        if matches!(self.direct_dxil.trim(), "1" | "all") {
            env.push(("XCL_DIRECT_DXIL".to_string(), self.direct_dxil.trim().to_string()));
        }
        if self.dxc_threads > 0 {
            env.push(("XCL_DXC_THREADS".to_string(), self.dxc_threads.to_string()));
        }
        for (on, name) in [(self.shader_cache, "XCL_TRANSLATED_CACHE"), (self.pipeline_cache, "XCL_PSO_DISK_CACHE"), (self.async_pipelines, "XCL_ASYNC_PSO")] {
            if !on {
                env.push((name.to_string(), "0".to_string()));
            }
        }
        if self.dred {
            env.push(("XCL_DRED".to_string(), "1".to_string()));
        }
        for line in self.extra_env.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            match line.split_once('=') {
                Some((name, value)) if is_env_name(name.trim()) => {
                    let name = name.trim().to_string();
                    env.retain(|(n, _)| !n.eq_ignore_ascii_case(&name));
                    env.push((name, value.trim().to_string()));
                }
                _ => refused.push(line.to_string()),
            }
        }
        (env, refused)
    }

    /// The game's arguments, split at spaces; a part in double quotes stays whole.
    pub fn arguments(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let (mut quoted, mut any) = (false, false);
        for c in self.game_args.chars() {
            match c {
                '"' => {
                    quoted = !quoted;
                    any = true;
                }
                c if c.is_whitespace() && !quoted => {
                    if any {
                        out.push(std::mem::take(&mut cur));
                        any = false;
                    }
                }
                c => {
                    cur.push(c);
                    any = true;
                }
            }
        }
        if any {
            out.push(cur);
        }
        out
    }
}

fn is_env_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !name.starts_with(|c: char| c.is_ascii_digit())
}

// ---------------------------------------------------------------- a title's xcl.toml

/// The settings of a title's `xcl.toml` the window edits. The file's other keys, its comments
/// and its order are kept as they are (`TitleFile`).
#[derive(Clone, Debug, PartialEq)]
pub struct TitleConfig {
    pub device_type: String,
    pub title_memory_mb: u64,
    /// `title_memory_fit`, where the file has it (an XCL that fits the standard budget to the
    /// GPU's video memory writes the key): `None` on an XCL without the setting.
    pub title_memory_fit: Option<bool>,
    pub uma_mb: u64,
    pub raytracing: String,
    pub d3d12_adapter: String,
    pub d3d12_debug: bool,
    pub log_level: String,
    pub window_width: u32,
    pub window_height: u32,
    /// "auto", "center" or "x,y".
    pub window_pos: String,
    /// "auto" (the title's frames in XCL's window) or "off".
    pub present: String,
    pub keyboard_mouse: bool,
    pub mouse_sensitivity: f32,
}

impl Default for TitleConfig {
    /// XCL's defaults, for keys a file does not have.
    fn default() -> Self {
        TitleConfig {
            device_type: "anaconda".into(),
            title_memory_mb: 0,
            title_memory_fit: None,
            uma_mb: 0,
            raytracing: "auto".into(),
            d3d12_adapter: "auto".into(),
            d3d12_debug: false,
            log_level: "info".into(),
            window_width: 1280,
            window_height: 720,
            window_pos: "auto".into(),
            present: "auto".into(),
            keyboard_mouse: true,
            mouse_sensitivity: 1.0,
        }
    }
}

impl TitleConfig {
    pub fn is_series_s(&self) -> bool {
        self.device_type.eq_ignore_ascii_case("lockhart") || self.device_type.eq_ignore_ascii_case("series s")
    }

    /// The budget XCL tells the title with these settings (MiB), as `Config::title_memory_budget_mb`.
    pub fn title_memory_budget_mb(&self) -> u64 {
        match self.title_memory_mb {
            0 if self.is_series_s() => 8_192,
            0 => 13_824,
            v => v,
        }
    }
}

/// A title's `xcl.toml` as text that keeps its shape.
pub struct TitleFile {
    doc: toml_edit::DocumentMut,
}

impl TitleFile {
    pub fn parse(text: &str) -> Result<TitleFile, String> {
        text.parse::<toml_edit::DocumentMut>().map(|doc| TitleFile { doc }).map_err(|e| e.to_string())
    }

    pub fn read(path: &Path) -> Result<TitleFile, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        TitleFile::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn config(&self) -> TitleConfig {
        let d = TitleConfig::default();
        let text = |k: &str, or: &str| self.doc.get(k).and_then(|v| v.as_str()).unwrap_or(or).to_string();
        let int = |k: &str, or: i64| self.doc.get(k).and_then(|v| v.as_integer()).unwrap_or(or).max(0);
        let flag = |k: &str, or: bool| self.doc.get(k).and_then(|v| v.as_bool()).unwrap_or(or);
        let float = self.doc.get("mouse_sensitivity").and_then(|v| v.as_float().or(v.as_integer().map(|i| i as f64)));
        TitleConfig {
            device_type: text("device_type", &d.device_type),
            title_memory_mb: int("title_memory_mb", 0) as u64,
            title_memory_fit: self.doc.get("title_memory_fit").and_then(|v| v.as_bool()),
            uma_mb: int("uma_mb", 0) as u64,
            raytracing: text("raytracing", &d.raytracing),
            d3d12_adapter: text("d3d12_adapter", &d.d3d12_adapter),
            d3d12_debug: flag("d3d12_debug", d.d3d12_debug),
            log_level: text("log_level", &d.log_level),
            window_width: int("window_width", d.window_width as i64) as u32,
            window_height: int("window_height", d.window_height as i64) as u32,
            window_pos: text("window_pos", &d.window_pos),
            present: text("present", &d.present),
            keyboard_mouse: flag("keyboard_mouse", d.keyboard_mouse),
            mouse_sensitivity: float.map_or(d.mouse_sensitivity, |f| f as f32),
        }
    }

    /// Writes the edited settings into the document. Every other key stays, a changed value
    /// keeps the comment on its line, and a key the file does not have is added only when its
    /// value is not XCL's default.
    pub fn set(&mut self, c: &TitleConfig) {
        use toml_edit::Value;
        let d = TitleConfig::default();
        let doc = &mut self.doc;
        let mut put = |key: &str, new: Value, is_default: bool| match doc.get_mut(key).and_then(|i| i.as_value_mut()) {
            Some(old) => {
                let decor = old.decor().clone();
                *old = new;
                *old.decor_mut() = decor;
            }
            None if is_default => {}
            None => doc[key] = toml_edit::Item::Value(new),
        };
        // two decimals: the slider's step, and no 1.2000000476837158 in the file
        let mouse = (c.mouse_sensitivity as f64 * 100.0).round() / 100.0;
        put("device_type", c.device_type.trim().into(), c.device_type.trim() == d.device_type);
        put("title_memory_mb", (c.title_memory_mb as i64).into(), c.title_memory_mb == d.title_memory_mb);
        // never added: only an XCL that knows the key writes it
        if let Some(fit) = c.title_memory_fit {
            put("title_memory_fit", fit.into(), true);
        }
        put("uma_mb", (c.uma_mb as i64).into(), c.uma_mb == d.uma_mb);
        put("raytracing", c.raytracing.trim().into(), c.raytracing.trim() == d.raytracing);
        put("d3d12_adapter", c.d3d12_adapter.trim().into(), c.d3d12_adapter.trim() == d.d3d12_adapter);
        put("d3d12_debug", c.d3d12_debug.into(), c.d3d12_debug == d.d3d12_debug);
        put("log_level", c.log_level.trim().into(), c.log_level.trim() == d.log_level);
        put("window_width", i64::from(c.window_width).into(), c.window_width == d.window_width);
        put("window_height", i64::from(c.window_height).into(), c.window_height == d.window_height);
        put("window_pos", c.window_pos.trim().into(), c.window_pos.trim() == d.window_pos);
        put("present", c.present.trim().into(), c.present.trim() == d.present);
        put("keyboard_mouse", c.keyboard_mouse.into(), c.keyboard_mouse == d.keyboard_mouse);
        put("mouse_sensitivity", mouse.into(), mouse == f64::from(d.mouse_sensitivity));
    }

    pub fn text(&self) -> String {
        self.doc.to_string()
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.text()).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// The titles XCL has run: the folders under the data root with an `xcl.toml`, newest first.
pub fn title_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("xcl.toml").is_file())
        .map(|p| (p.join("xcl.toml").metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH), p))
        .collect();
    dirs.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    dirs.into_iter().map(|(_, p)| p).collect()
}

// ---------------------------------------------------------------- XCL's own files

/// XCL's source folder above `exe` (`<repo>\target\<profile>\xcl-gui.exe`): the nearest folder
/// above it with `crates\xcl\Cargo.toml`.
pub fn repo_above(exe: &Path) -> Option<PathBuf> {
    exe.ancestors().skip(1).take(5).find(|d| d.join("crates").join("xcl").join("Cargo.toml").is_file()).map(Path::to_path_buf)
}

/// `xcl.exe` for these settings: the one named, else the one next to this program, else the
/// release build of the source folder.
pub fn find_xcl(settings: &Settings, me: &Path, repo: Option<&Path>) -> Option<PathBuf> {
    let name = if cfg!(windows) { "xcl.exe" } else { "xcl" };
    let named = settings.xcl_exe.trim();
    if !named.is_empty() {
        return Some(PathBuf::from(named)).filter(|p| p.is_file());
    }
    let beside = me.parent().map(|d| d.join(name));
    let built = repo.map(|r| r.join("target").join("release").join(name));
    [beside, built].into_iter().flatten().find(|p| p.is_file())
}

// ---------------------------------------------------------------- sessions

/// `2026-11-24_15-08-31` for seconds since 1970 (UTC): the names of saved sessions.
pub fn stamp(secs: u64) -> String {
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // civil date from days (Howard Hinnant's algorithm)
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let (d, m) = (doy - (153 * mp + 2) / 5 + 1, if mp < 10 { mp + 3 } else { mp - 9 });
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}_{:02}-{:02}-{:02}", rem / 3_600, rem % 3_600 / 60, rem % 60)
}

/// Removes the oldest files of `dir` beyond the newest `keep` (names sort by time).
pub fn prune(dir: &Path, keep: usize) {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_file()).collect();
    files.sort();
    let extra = files.len().saturating_sub(keep);
    for p in files.into_iter().take(extra) {
        let _ = std::fs::remove_file(p);
    }
}

// ---------------------------------------------------------------- texts

/// The window's texts, English and Spanish.
pub fn tr(lang: Lang, key: &'static str) -> &'static str {
    let (en, es) = match key {
        "game" => ("Game", "Juego"),
        "game_hint" => ("the game's folder, its .exe or its MicrosoftGame.config", "la carpeta del juego, su .exe o su MicrosoftGame.config"),
        "folder" => ("Folder…", "Carpeta…"),
        "file" => ("File…", "Archivo…"),
        "play" => ("Play", "Jugar"),
        "stop" => ("Stop", "Detener"),
        "idle" => ("Not running", "Detenido"),
        "running" => ("Running", "En ejecución"),
        "busy" => ("Working", "Trabajando"),
        "exited" => ("Exited with", "Terminó con"),
        "stopped" => ("Stopped from this window", "Detenido desde esta ventana"),
        "console" => ("Console", "Consola"),
        "log" => ("Log (xcl.log)", "Registro (xcl.log)"),
        "errors" => ("Errors and warnings", "Errores y avisos"),
        "settings" => ("Settings", "Ajustes"),
        "xcl" => ("XCL", "XCL"),
        "filter" => ("Filter", "Filtro"),
        "follow" => ("Follow", "Seguir"),
        "copy" => ("Copy", "Copiar"),
        "clear" => ("Clear", "Limpiar"),
        "lines" => ("lines", "líneas"),
        "open_logs" => ("Open the logs folder", "Abrir carpeta de registros"),
        "open_sessions" => ("Open saved sessions", "Abrir sesiones guardadas"),
        "open_report" => ("Open report.json", "Abrir report.json"),
        "no_xcl" => ("xcl.exe not found – set it, or build XCL, under the XCL tab", "No se encuentra xcl.exe: indícalo o compila XCL en la pestaña XCL"),
        "no_game" => ("Choose the game first", "Primero elige el juego"),
        "title_cfg" => ("This title (xcl.toml)", "Este juego (xcl.toml)"),
        "title_none" => (
            "Greyed until the game has run once: XCL writes a title's xcl.toml at its first run, and these settings are kept in it.",
            "En gris hasta que el juego corra una vez: XCL crea el xcl.toml del juego en su primera ejecución, y estos ajustes se guardan ahí.",
        ),
        "title" => ("Title", "Juego"),
        "console_kind" => ("Console", "Consola"),
        "memory" => ("Title memory (MiB)", "Memoria del juego (MiB)"),
        "memory_hint" => ("0 = the console's budget", "0 = el presupuesto de la consola"),
        "memory_told" => ("the title is told", "al juego se le informa"),
        "memory_fit" => ("Fit the memory to my graphics card (when the figure is 0)", "Ajustar la memoria a la de mi tarjeta gráfica (cuando el valor es 0)"),
        "memory_fit_on" => (
            "at most the console's figure: XCL lowers it to the card's video memory less 1536 MiB, never below 4096 – the log says the figure it used",
            "como máximo el valor de la consola: XCL lo baja a la memoria de video de la tarjeta menos 1536 MiB, nunca bajo 4096; el registro dice el valor que usó",
        ),
        "memory_fit_manual" => ("a figure written by hand is used as it is", "un valor escrito a mano se usa tal cual"),
        "uma" => ("GPU-visible memory (MiB)", "Memoria visible para la GPU (MiB)"),
        "uma_hint" => ("0 = XCL's choice (2048, or 4095 for \"advanced\")", "0 = lo que XCL elija (2048, o 4095 en modo \"advanced\")"),
        "raytracing" => ("Ray tracing", "Trazado de rayos"),
        "adapter" => ("Graphics adapter", "Adaptador gráfico"),
        "adapter_hint" => ("auto, warp, a number, or part of the name", "auto, warp, un número o parte del nombre"),
        "debug_layer" => ("D3D12 debug layer (needs Graphics Tools)", "Capa de depuración de D3D12 (requiere Herramientas de gráficos)"),
        "log_level" => ("Log level", "Nivel de registro"),
        "window" => ("Window", "Ventana"),
        "kbm" => ("Keyboard and mouse as the gamepad", "Teclado y mouse como mando"),
        "mouse" => ("Mouse sensitivity", "Sensibilidad del mouse"),
        "save" => ("Save", "Guardar"),
        "reload" => ("Reload", "Recargar"),
        "saved" => ("Saved", "Guardado"),
        "while_running" => ("Settings are read when the game starts: changes apply to the next run.", "Los ajustes se leen al iniciar el juego: los cambios valen para la próxima ejecución."),
        "run_cfg" => ("Every run", "Cada ejecución"),
        "offline" => ("Offline (XCL_NETWORK=0)", "Sin red (XCL_NETWORK=0)"),
        "threads" => ("Pipeline threads (0 = XCL's own)", "Hilos de pipelines (0 = los de XCL)"),
        "seconds" => ("Stop after seconds (0 = no limit)", "Detener tras segundos (0 = sin límite)"),
        "extra_env" => ("More variables, one NAME=value a line", "Más variables, una NOMBRE=valor por línea"),
        "refused" => ("Not NAME=value, left out", "No es NOMBRE=valor, se omite"),
        "args" => ("Arguments for the game", "Argumentos para el juego"),
        "sessions" => ("Save what each run prints", "Guardar lo que imprime cada ejecución"),
        "xcl_exe" => ("xcl.exe", "xcl.exe"),
        "repo" => ("XCL's source folder", "Carpeta del código de XCL"),
        "auto" => ("found by itself", "se busca solo"),
        "using" => ("using", "se usa"),
        "none" => ("none", "ninguno"),
        "update" => ("Update only (git pull)", "Solo actualizar (git pull)"),
        "build" => ("Build only (cargo build)", "Solo compilar (cargo build)"),
        "version" => ("Show version (git log -1)", "Ver versión (git log -1)"),
        "build_note" => (
            "Output goes to the Console tab. Updating keeps your local changes and puts them back on top; git stops if one conflicts. This builds xcl.exe; the window itself is rebuilt with it closed (cargo build --release).",
            "La salida va a la pestaña Consola. Actualizar conserva tus cambios locales y los vuelve a poner encima; git se detiene si alguno choca. Esto compila xcl.exe; la ventana misma se recompila con ella cerrada (cargo build --release).",
        ),
        "update_build" => ("Update and build", "Actualizar y compilar"),
        "step_git" => ("Downloading changes from GitHub", "Bajando cambios de GitHub"),
        "step_cargo" => ("Building XCL", "Compilando XCL"),
        "starting" => ("starting", "iniciando"),
        "step_head" => ("Noting the version before the update", "Anotando la versión antes de actualizar"),
        "step_diff" => ("Checking whether the shader translators changed", "Revisando si cambiaron los traductores de shaders"),
        "step_precompile" => ("Precompiling shaders", "Precompilando shaders"),
        "under_minute" => ("less than a minute", "menos de un minuto"),
        "eta" => ("about {} left", "faltan cerca de {}"),
        "eta_wait" => ("time left: after the next report", "tiempo restante: tras el próximo aviso"),
        "pipelines" => ("pipelines built", "pipelines compilados"),
        "shaders" => ("Shaders", "Shaders"),
        "xcl_program" => ("XCL itself", "El propio XCL"),
        "shader_pack" => ("Shader package", "Paquete de shaders"),
        "shader_pack_hint" => ("found in the game's folder", "se busca en la carpeta del juego"),
        "all_stages" => (
            "All the shaders of the package: vertex and pixel too (without this, only compute, and the game compiles the rest itself)",
            "Todos los shaders del paquete: también vértices y píxeles (sin esto, solo cómputo, y el juego compila el resto por dentro)",
        ),
        "used_only" => (
            "Compute pipelines: only those the game used in earlier runs (much faster; play once first)",
            "Pipelines de cómputo: solo los que el juego usó en ejecuciones anteriores (mucho más rápido; juega una vez antes)",
        ),
        "precompile" => ("Precompile shaders", "Precompilar shaders"),
        "shaders_where" => ("These are the choices of Precompile shaders, the button under Play on Home.", "Estas son las opciones de Precompilar shaders, el botón bajo Jugar en Inicio."),
        "precompile_note" => (
            "Compiles the game's shaders ahead (vertex, pixel and compute) and builds its compute pipelines, so the game does not compile them itself on a black screen. It can take hours. Stop it whenever you like: the next pass goes on from where this one stopped. The driver still builds the drawing pipelines inside the game, the first time.",
            "Compila por adelantado los shaders del juego (vértices, píxeles y cómputo) y construye sus pipelines de cómputo, para que el juego no los compile por dentro con la pantalla en negro. Puede tardar horas. Detenlo cuando quieras: la próxima pasada sigue desde donde quedó esta. Los pipelines de dibujo los sigue construyendo el driver dentro del juego, la primera vez.",
        ),
        "precompile_warn" => (
            "What is precompiled belongs to this build of XCL, this graphics card and this driver. After an update of XCL that changes its shader translators, or a new graphics driver, the pass starts from zero.",
            "Lo precompilado pertenece a esta compilación de XCL, a esta tarjeta gráfica y a este driver. Tras una actualización de XCL que cambie sus traductores de shaders, o un driver gráfico nuevo, la pasada empieza de cero.",
        ),
        "precompile_stale" => (
            "The last update changed XCL's shader translators: what was precompiled before no longer counts. Precompile again (from zero).",
            "La última actualización cambió los traductores de shaders de XCL: lo precompilado antes ya no sirve. Precompila de nuevo (desde cero).",
        ),
        "copy_selection" => ("Copy selection", "Copiar selección"),
        "selected" => ("Lines selected", "Líneas seleccionadas"),
        "select_hint" => (
            "click a line, Shift+click or drag to another; Ctrl+A takes all, Ctrl+C copies; part of a line can be picked here",
            "clic en una línea, Mayús+clic o arrastra hasta otra; Ctrl+A toma todas, Ctrl+C copia; aquí se puede marcar parte de una línea",
        ),
                "precompile_again" => ("The update changed the shader translators: precompile again", "La actualización cambió los traductores de shaders: precompila de nuevo"),
                "precompile_done" => ("Done: the shaders are precompiled", "Listo: los shaders quedaron precompilados"),
        "precompiled" => ("Shaders precompiled", "Shaders precompilados"),
        "precompiled_never" => ("Not precompiled from this window yet", "Aún no se ha precompilado desde esta ventana"),
        "result_compiled" => ("compiled", "compilados"),
        "result_cached" => ("already in the cache", "ya en la caché"),
        "result_errors" => ("errors", "errores"),
        "no_pack" => ("No shader package (*shader*.rpf) in the game's folder: choose it", "No hay paquete de shaders (*shader*.rpf) en la carpeta del juego: elígelo"),
        "close_game" => ("Not while the game runs", "No mientras el juego corre"),
        "done" => ("Done: XCL is up to date and built", "Listo: XCL quedó actualizado y compilado"),
        "updated" => ("Done: XCL's source is up to date (not built yet)", "Listo: el código de XCL quedó actualizado (falta compilar)"),
        "step_failed" => ("It stopped with an error – see the Console tab", "Se detuvo con un error: revisa la pestaña Consola"),
        "no_git" => ("Git is not installed, or not in the PATH (install Git for Windows)", "Git no está instalado o no está en el PATH (instala Git for Windows)"),
        "no_cargo" => ("Rust (cargo) is not installed, or not in the PATH (https://rustup.rs)", "Rust (cargo) no está instalado o no está en el PATH (https://rustup.rs)"),
        "one_by_one" => ("One step at a time", "Paso a paso"),
        "gfx_run" => ("Graphics and shaders (every run)", "Gráficos y shaders (cada ejecución)"),
        "translator" => ("Shader translator", "Traductor de shaders"),
        "tr_hlsl" => ("HLSL + DXC (XCL's default)", "HLSL + DXC (el de XCL por defecto)"),
        "tr_cs" => ("Direct DXIL: compute shaders (experimental)", "DXIL directo: shaders de cómputo (experimental)"),
        "tr_all" => ("Direct DXIL: all shaders (experimental)", "DXIL directo: todos los shaders (experimental)"),
        "dxc_threads" => ("Shader compiler threads (0 = XCL's own)", "Hilos del compilador de shaders (0 = los de XCL)"),
        "shader_cache" => ("Keep translated shaders on disk", "Guardar en disco los shaders traducidos"),
        "pipeline_cache" => ("Keep the driver's pipelines on disk", "Guardar en disco los pipelines del driver"),
        "async_pso" => ("Build pipelines in the background", "Compilar pipelines en segundo plano"),
        "dred" => ("Log the GPU's last operations when it is lost (XCL_DRED)", "Registrar las últimas operaciones de la GPU si se pierde (XCL_DRED)"),
        "gfx_note" => (
            "These are the graphics switches XCL has. It has no resolution scale, V-Sync, frame limit or filters: the game draws as on the console.",
            "Estos son los ajustes gráficos que XCL tiene. No tiene escala de resolución, V-Sync, límite de FPS ni filtros: el juego dibuja como en la consola.",
        ),
        "window_pos" => ("Window position", "Posición de la ventana"),
        "pos_auto" => ("Windows decides", "La elige Windows"),
        "pos_center" => ("Centre of the main monitor", "Centro del monitor principal"),
        "pos_custom" => ("At x,y", "En x,y"),
        "present" => ("Show the game's image in XCL's window", "Mostrar la imagen del juego en la ventana de XCL"),
        "home" => ("Home", "Inicio"),
        "tagline" => ("Xbox Series titles on Windows", "Juegos de Xbox Series en Windows"),
        "sec_logs" => ("LOGS", "REGISTROS"),
        "sec_setup" => ("SETUP", "CONFIGURACIÓN"),
        "pick_game" => ("Choose a game", "Elige un juego"),
        "this_pc" => ("This PC", "Este PC"),
        "gpu" => ("Graphics", "Gráficos"),
        "gpu_unknown" => ("known after the first run", "se sabrá tras la primera ejecución"),
        "data_dir" => ("Data folder", "Carpeta de datos"),
        "last_run" => ("Last run", "Última ejecución"),
        "not_yet" => ("not yet", "todavía no"),
        "network" => ("Network", "Red"),
        "net_off" => ("off (offline)", "desactivada (sin red)"),
        "net_on" => ("on", "activada"),
        "ready" => ("READY", "LISTO"),
        "state_running" => ("RUNNING", "EN EJECUCIÓN"),
        "state_busy" => ("WORKING", "TRABAJANDO"),
        "state_no_xcl" => ("XCL NOT BUILT", "XCL SIN COMPILAR"),
        "theme" => ("Light theme", "Tema claro"),
        "theme_xcl" => ("XCL theme", "Tema XCL"),
        "theme_black" => ("Black theme", "Tema negro"),
        "language" => ("Español", "English"),
        other => (other, other),
    };
    if lang == Lang::Es {
        es
    } else {
        en
    }
}

impl Settings {
    pub fn theme(&self) -> Theme {
        match self.theme.as_str() {
            "xcl" => Theme::Xcl,
            "black" => Theme::Black,
            "light" => Theme::Light,
            // settings from before the themes had names
            _ if self.dark => Theme::Xcl,
            _ => Theme::Light,
        }
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = match theme {
            Theme::Xcl => "xcl",
            Theme::Black => "black",
            Theme::Light => "light",
        }
        .to_string();
        self.dark = theme != Theme::Light;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_lines_carry_their_level_and_console_lines_are_read() {
        assert_eq!(severity("[    1.234][ 1234][ERROR] D3D12 device removed"), Severity::Error);
        assert_eq!(severity("[   12.000][12345][WARN ] D3D12: DRED requested but unavailable"), Severity::Warn);
        assert_eq!(severity("[    0.001][    1][INFO ] title memory: an error-free line"), Severity::Info);
        assert_eq!(severity("[    0.001][    1][DEBUG] x"), Severity::Detail);
        assert_eq!(severity("CRASH: unhandled exception 0xc0000005 in nvwgf2umx.dll"), Severity::Error);
        assert_eq!(severity("xcl: failed to start the game process: x"), Severity::Error);
        assert_eq!(severity("  WARNING: this CPU lacks avx2"), Severity::Warn);
        assert_eq!(severity("   Compiling xcl v0.1.0"), Severity::Info);
        assert_eq!(severity("test result: ok. 12 passed; 0 errors"), Severity::Info);
        // a bracket that is not a log line's
        assert_eq!(severity("[note] nothing"), Severity::Info);
    }

    #[test]
    fn the_start_and_the_end_of_a_run_are_read() {
        assert_eq!(data_dir_line("  data:  C:\\Users\\a b\\AppData\\Local\\XCL\\Title_1"), Some(PathBuf::from("C:\\Users\\a b\\AppData\\Local\\XCL\\Title_1")));
        assert_eq!(data_dir_line("  exe:   D:\\g\\game.exe"), None);
        assert_eq!(gpu_line("  GPU: Intel(R) Iris(R) Xe Graphics (128 MiB VRAM)").as_deref(), Some("Intel(R) Iris(R) Xe Graphics (128 MiB VRAM)"));
        assert_eq!(gpu_line("xcl: D3D12.X: backend 'real'"), None);
        assert_eq!(exit_code_line("Game exited with code 0x887a0006 (2289696774)"), Some(0x887a_0006));
        assert_eq!(exit_code_line("Game exited"), None);
        assert!(exit_code_meaning(0x887a_0006, Lang::Es).unwrap().contains("GPU"));
        assert_eq!(exit_code_meaning(0x1234_5678, Lang::En), None);
    }

    #[test]
    fn a_run_gets_its_variables_and_arguments() {
        let mut s = Settings { pipeline_threads: 12, extra_env: "XCL_UMA_MB = 4095\n# note\nnot a variable\nxcl_network=1\n1X=2".into(), ..Settings::default() };
        let (env, refused) = s.environment();
        // a line of the box replaces the switch of the same name
        assert_eq!(
            env,
            [("XCL_PIPELINE_THREADS", "12"), ("XCL_UMA_MB", "4095"), ("xcl_network", "1")].map(|(a, b)| (a.to_string(), b.to_string()))
        );
        assert_eq!(refused, ["not a variable", "1X=2"]);
        let g = Settings { offline: false, direct_dxil: "all".into(), dxc_threads: 6, shader_cache: false, dred: true, ..Settings::default() };
        assert_eq!(
            g.environment().0,
            [("XCL_DIRECT_DXIL", "all"), ("XCL_DXC_THREADS", "6"), ("XCL_TRANSLATED_CACHE", "0"), ("XCL_DRED", "1")].map(|(a, b)| (a.to_string(), b.to_string()))
        );
        s.game_args = " -windowed  \"-name=a b\" x".into();
        assert_eq!(s.arguments(), ["-windowed", "-name=a b", "x"]);
        s.game_args = "\"\"".into();
        assert_eq!(s.arguments(), [""]);
    }

    #[test]
    fn settings_round_trip_and_old_files_get_defaults() {
        let dir = std::env::temp_dir().join(format!("xcl_gui_settings_{}", std::process::id()));
        let path = dir.join("gui").join("settings.toml");
        let s = Settings { lang: Lang::Es, game: "D:\\Games\\X".into(), run_seconds: 60, ..Settings::default() };
        s.save(&path).unwrap();
        assert_eq!(Settings::load(&path), s);
        std::fs::write(&path, "game = \"E:\\\\g\"\n").unwrap();
        let old = Settings::load(&path);
        assert_eq!((old.game.as_str(), old.offline, old.save_sessions), ("E:\\g", true, true));
        // an unreadable file is the defaults, not an error
        std::fs::write(&path, "game = [").unwrap();
        assert_eq!(Settings::load(&path), Settings::default());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_titles_file_keeps_what_the_window_does_not_edit() {
        let text = "# XCL – title configuration.\n\ndevice_type = \"anaconda\"\ngamertag = \"Player1\"\ntitle_memory_mb = 0 # the console's\nmouse_sensitivity = 1\n\n[mounts]\n\"G:\" = \"{game}\"\n";
        let mut f = TitleFile::parse(text).unwrap();
        let mut c = f.config();
        assert_eq!((c.title_memory_mb, c.window_width, c.mouse_sensitivity, c.raytracing.as_str()), (0, 1280, 1.0, "auto"));
        assert_eq!(c.title_memory_budget_mb(), 13_824);
        c.device_type = "lockhart".into();
        assert_eq!(c.title_memory_budget_mb(), 8_192);
        c.title_memory_mb = 10_240;
        c.mouse_sensitivity = 1.2;
        f.set(&c);
        let out = f.text();
        assert!(out.starts_with("# XCL – title configuration.\n\ndevice_type = \"lockhart\"\ngamertag = \"Player1\"\ntitle_memory_mb = 10240 # the console's\n"), "{out}");
        assert!(out.ends_with("mouse_sensitivity = 1.2\n\n[mounts]\n\"G:\" = \"{game}\"\n"), "{out}");
        // and reads back as written
        assert_eq!(TitleFile::parse(&out).unwrap().config(), c);
        // a key the file lacks appears once it is not the default
        c.window_width = 1920;
        c.window_pos = "center".into();
        // a key only some XCLs have is not added to a file without it
        assert!(c.title_memory_fit.is_none());
        c.title_memory_fit = Some(false);
        f.set(&c);
        assert!(f.text().contains("window_pos = \"center\"\n") && !f.text().contains("present"), "{}", f.text());
        assert!(!f.text().contains("title_memory_fit"), "{}", f.text());
        let mut fitted = TitleFile::parse("title_memory_mb = 0\ntitle_memory_fit = true # follows the GPU\n").unwrap();
        let mut fc = fitted.config();
        assert_eq!(fc.title_memory_fit, Some(true));
        fc.title_memory_fit = Some(false);
        fitted.set(&fc);
        assert_eq!(fitted.text(), "title_memory_mb = 0\ntitle_memory_fit = false # follows the GPU\n");
        assert!(f.text().contains("window_width = 1920\n") && !f.text().contains("window_height"), "{}", f.text());
        assert!(TitleFile::parse("device_type = ").is_err());
    }

    #[test]
    fn xcl_and_its_titles_are_found() {
        let dir = std::env::temp_dir().join(format!("xcl_gui_find_{}", std::process::id()));
        let repo = dir.join("xcl");
        let release = repo.join("target").join("release");
        std::fs::create_dir_all(repo.join("crates").join("xcl")).unwrap();
        std::fs::create_dir_all(&release).unwrap();
        std::fs::write(repo.join("crates").join("xcl").join("Cargo.toml"), "").unwrap();
        let me = release.join("xcl-gui.exe");
        assert_eq!(repo_above(&me), Some(repo.clone()));
        assert_eq!(repo_above(&dir.join("elsewhere").join("xcl-gui.exe")), None);
        let name = if cfg!(windows) { "xcl.exe" } else { "xcl" };
        let s = Settings::default();
        assert_eq!(find_xcl(&s, &me, Some(&repo)), None);
        std::fs::write(release.join(name), "").unwrap();
        assert_eq!(find_xcl(&s, &me, Some(&repo)), Some(release.join(name)));
        // a named one that is not there is not replaced by another
        let named = Settings { xcl_exe: dir.join("gone.exe").to_string_lossy().into_owned(), ..Settings::default() };
        assert_eq!(find_xcl(&named, &me, Some(&repo)), None);

        let root = dir.join("data");
        for t in ["A", "B", "gui"] {
            std::fs::create_dir_all(root.join(t)).unwrap();
        }
        std::fs::write(root.join("A").join("xcl.toml"), "").unwrap();
        std::fs::write(root.join("B").join("xcl.toml"), "").unwrap();
        let titles = title_dirs(&root);
        assert_eq!(titles.len(), 2);
        assert!(titles.iter().all(|p| p.file_name().is_some_and(|n| n != "gui")));

        let sessions = dir.join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        for i in 0..5 {
            std::fs::write(sessions.join(format!("{}.log", stamp(1_795_500_000 + i))), "").unwrap();
        }
        prune(&sessions, 2);
        let mut left: Vec<String> = std::fs::read_dir(&sessions).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        left.sort();
        assert_eq!(left, [format!("{}.log", stamp(1_795_500_003)), format!("{}.log", stamp(1_795_500_004))]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cargo_s_progress_line_is_read() {
        assert_eq!(cargo_progress("    Building [=======>                 ] 12/345: xcl, windows..."), Some((12, 345, "xcl, windows".to_string())));
        assert_eq!(cargo_progress("    Building [=========================] 345/345"), Some((345, 345, String::new())));
        assert_eq!(cargo_progress("   Compiling xcl v0.1.0"), None);
        assert_eq!(cargo_progress("    Building [==>  ] 9/0: x"), None);
    }

    #[test]
    fn stamps_are_dates() {
        assert_eq!(stamp(0), "1970-01-01_00-00-00");
        assert_eq!(stamp(951_782_400), "2000-02-29_00-00-00");
        assert_eq!(stamp(1_795_543_711), "2026-11-24_18-08-31");
    }

    #[test]
    fn precompile_lines_are_read() {
        assert_eq!(precompile_result("compiled 812, already cached 25691, errors 0, skipped (GS – paired at pipeline creation) 3, time 512.3 s"), Some((812, 25691, 0)));
        assert_eq!(precompile_result("compiled, already cached"), None);
        assert_eq!(precompile_result("  pipelines: 25 made, 0 refused"), None);
        assert_eq!(used_pipelines_line("--used-only: 499 pipelines earlier runs used (used-pipelines.txt); compute programs of other pipelines are passed over"), Some(499));
        assert_eq!(used_pipelines_line("--used-only: no earlier run"), None);
        assert_eq!(used_pipelines_line("precompiling 26503 shaders into C:\\x"), None);
        assert_eq!(precompile_progress("  22400/26503 (463 s)"), Some((22400, 26503, 463)));
        assert_eq!(precompile_progress("26503/26503 (7205 s)"), Some((26503, 26503, 7205)));
        assert_eq!(precompile_progress("  pipelines: 400 made, 0 refused"), None);
        assert_eq!(precompile_progress("  30000/26503 (1 s)"), None, "more done than there are");
        assert_eq!(precompile_progress("DXC: 3/10"), None);
        assert_eq!(precompile_progress("   Building [=====>   ] 12/345: xcl"), None, "cargo's bar is not this pass");
        assert_eq!(pipelines_line("  pipelines: 400 made, 0 refused"), Some((400, 0)));
        assert_eq!(pipelines_line("pipelines: 1225 made, 3 refused"), Some((1225, 3)));
        assert_eq!(pipelines_line("pipelines made on the GPU 824, refused 0"), None, "the pass's last line is another");
    }

    #[test]
    fn time_left_is_taken_between_reports() {
        let mut eta = Eta::default();
        assert_eq!(eta.left(26503), None);
        // the first report: 22 200 shaders an earlier pass left done, skipped in 4 s
        eta.report(22200, 4);
        assert_eq!(eta.left(26503), None, "one report tells no speed");
        eta.report(22400, 463);
        // 200 shaders in 459 s, 4103 to go
        assert_eq!(eta.left(26503), Some(4103 * 459 / 200));
        eta.report(22400, 463);
        assert_eq!(eta.left(26503), Some(4103 * 459 / 200), "the same report twice changes nothing");
        // only the last reports count
        for (done, secs) in [(22600, 701), (22800, 1006), (23000, 1300), (23200, 1600), (23400, 1900), (23600, 2200)] {
            eta.report(done, secs);
        }
        assert_eq!(eta.left(26503), Some((26503 - 23600) * (2200 - 701) / 1000));
        assert_eq!(eta.left(23600), Some(0));
        // another pass starts over
        eta.report(100, 3);
        assert_eq!(eta.left(26503), None);
    }

    #[test]
    fn durations_are_said_in_hours_and_minutes() {
        assert_eq!(duration_text(0, Lang::En), "less than a minute");
        assert_eq!(duration_text(59, Lang::Es), "menos de un minuto");
        assert_eq!(duration_text(60, Lang::En), "1 min");
        assert_eq!(duration_text(61, Lang::En), "2 min");
        assert_eq!(duration_text(3600, Lang::En), "1 h 00 min");
        assert_eq!(duration_text(9417, Lang::Es), "2 h 37 min");
    }

    #[test]
    fn shader_packages_are_found_in_the_games_folder() {
        let dir = std::env::temp_dir().join(format!("xcl_gui_packs_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("data").join("deep").join("deeper")).unwrap();
        for file in ["Shaders.RPF", "game.exe", "data/shaders_extra.rpf", "data/textures.rpf", "data/deep/deeper/shaders_far.rpf"] {
            std::fs::write(dir.join(file), b"x").unwrap();
        }
        let names = |from: &Path| shader_packs(from).iter().map(|p| p.strip_prefix(&dir).unwrap().to_string_lossy().replace('\\', "/")).collect::<Vec<_>>();
        // from the folder, and from the game's exe in it; three folders down is too far
        assert_eq!(names(&dir), ["Shaders.RPF", "data/shaders_extra.rpf"]);
        assert_eq!(names(&dir.join("game.exe")), ["Shaders.RPF", "data/shaders_extra.rpf"]);
        assert!(shader_packs(&dir.join("none").join("game.exe")).is_empty(), "a folder that is not there");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn commit_ids_are_told_from_other_lines() {
        assert!(is_commit_id("4178e888c0b5be50a2887eda45cdb832763b68a4"));
        assert!(!is_commit_id("4178e88"));
        assert!(!is_commit_id("Already up to date. 0000000000000000000000"));
        assert!(!is_commit_id("crates/xcl/src/gpu/recomp/lower.rs"));
    }

    #[test]
    fn settings_from_before_the_precompile_page_load() {
        let old: Settings = toml::from_str("game = 'D:\\Game'\ndark = false\n").unwrap();
        assert!(old.used_only && old.all_stages && !old.precompile_stale && old.shader_pack.is_empty() && !old.dark);
        // the theme is told from the old switch until one is chosen by name
        assert_eq!(old.theme(), Theme::Light);
        assert_eq!(Settings::default().theme(), Theme::Xcl);
        let mut s = Settings::default();
        for _ in 0..3 {
            let next = s.theme().next();
            s.set_theme(next);
            assert_eq!(s.theme(), next);
            assert_eq!(s.dark, next != Theme::Light);
        }
        assert_eq!(s.theme(), Theme::Xcl);
    }

    #[test]
    fn every_text_has_both_languages() {
        for key in ["play", "stop", "errors", "memory_hint", "title_none", "build_note", "precompile", "precompile_warn", "precompile_stale", "eta", "stopped"] {
            assert_ne!(tr(Lang::En, key), key);
            assert_ne!(tr(Lang::Es, key), tr(Lang::En, key), "{key}");
        }
        assert_eq!(tr(Lang::Es, "no such key"), "no such key");
    }
}
