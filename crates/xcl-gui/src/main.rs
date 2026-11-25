//! A window for XCL: pick the game once, run and stop it, read what it prints and its
//! `xcl.log` as they grow (errors and warnings on a tab of their own), edit the title's
//! `xcl.toml`, and update and build XCL – without a command line. It drives `xcl.exe` from
//! outside: nothing of XCL's runtime is changed or linked.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod logic;
mod runner;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use eframe::egui;
use logic::{tr, Lang, Settings, Severity, TitleConfig, TitleFile};
use runner::{Event, Running, Tail};

/// Lines a view keeps; the oldest go first.
const MAX_LINES: usize = 200_000;
/// Files kept under `gui\sessions` (three a run: console, xcl.log, report.json).
const KEEP_SESSION_FILES: usize = 60;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("XCL").with_inner_size([1100.0, 720.0]).with_min_inner_size([760.0, 420.0]),
        ..Default::default()
    };
    eframe::run_native("XCL", options, Box::new(|_cc| Ok(Box::new(App::new()))))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Home,
    Console,
    Log,
    Errors,
    Settings,
    Xcl,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Job {
    Game,
    Tool,
}

/// What the window takes from a step's output, besides showing it.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum Capture {
    #[default]
    Nothing,
    /// The commit the source folder is at (`git rev-parse HEAD`).
    Head,
    /// Files of the shader translators an update changed: each line is one.
    TranslatorChanges,
}

/// One program of a chain run from the XCL tab (update, then build).
struct Step {
    program: String,
    /// `{before}` in an argument is the commit noted before the update.
    args: Vec<String>,
    env: Vec<(String, String)>,
    /// The text key of what it does, for the progress bar.
    label: &'static str,
    capture: Capture,
}

impl Step {
    fn git(args: &[&str]) -> Step {
        Step { program: "git".into(), args: args.iter().map(|s| s.to_string()).collect(), env: Vec::new(), label: "step_git", capture: Capture::Nothing }
    }

    /// Cargo prints its progress bar to a terminal only; these two make it print the line
    /// (`Building [==> ] 12/345: …`) to a pipe too, which the window turns into its own bar.
    fn cargo(args: &[&str]) -> Step {
        let env = [("CARGO_TERM_PROGRESS_WHEN", "always"), ("CARGO_TERM_PROGRESS_WIDTH", "100")].map(|(k, v)| (k.to_string(), v.to_string())).to_vec();
        Step { program: "cargo".into(), args: args.iter().map(|s| s.to_string()).collect(), env, label: "step_cargo", capture: Capture::Nothing }
    }

    /// Updating, with a look at what it changed: the commit before, the pull, and the files of
    /// the shader translators that differ since. A change there makes XCL translate and build
    /// again what a precompile pass did, and the window says so.
    fn update() -> Vec<Step> {
        let mut changed = vec!["diff", "--name-only", "{before}", "HEAD", "--"];
        changed.extend(logic::TRANSLATOR_PATHS);
        vec![
            Step { label: "step_head", capture: Capture::Head, ..Step::git(&["rev-parse", "HEAD"]) },
            // local changes (this crate is one) are set aside and put back on top
            Step::git(&["pull", "--rebase", "--autostash"]),
            Step { label: "step_diff", capture: Capture::TranslatorChanges, ..Step::git(&changed) },
        ]
    }

    fn build() -> Step {
        // `-p xcl` only: this program is running and cannot be written over
        Step::cargo(&["build", "--release", "-p", "xcl"])
    }
}

/// How far the running step of a chain is.
struct Progress {
    label: &'static str,
    /// Which step of how many.
    step: (usize, usize),
    /// Units done and their number, once the program says them.
    units: Option<(u32, u32)>,
    detail: String,
}

/// The window's own colours, over egui's dark and light themes.
struct Palette {
    bg: egui::Color32,
    side: egui::Color32,
    card: egui::Color32,
    line: egui::Color32,
    field: egui::Color32,
    button: egui::Color32,
    button_hover: egui::Color32,
    accent: egui::Color32,
}

fn palette(dark: bool) -> Palette {
    let c = egui::Color32::from_rgb;
    if dark {
        Palette { bg: c(17, 19, 24), side: c(12, 14, 18), card: c(25, 28, 35), line: c(38, 42, 52), field: c(12, 14, 18), button: c(40, 45, 56), button_hover: c(52, 58, 72), accent: c(38, 132, 255) }
    } else {
        Palette { bg: c(243, 245, 248), side: c(232, 235, 240), card: c(255, 255, 255), line: c(218, 222, 230), field: c(243, 245, 248), button: c(226, 230, 237), button_hover: c(210, 216, 226), accent: c(24, 110, 230) }
    }
}

/// Dark or light, with rounded corners, roomier controls and the window's own colours.
fn apply_style(ctx: &egui::Context, dark: bool) {
    let theme = if dark { egui::Theme::Dark } else { egui::Theme::Light };
    ctx.set_theme(theme);
    let p = palette(dark);
    ctx.style_mut_of(theme, |style| {
        style.spacing.item_spacing = egui::vec2(8.0, 7.0);
        style.spacing.button_padding = egui::vec2(11.0, 5.0);
        let v = &mut style.visuals;
        v.panel_fill = p.bg;
        v.window_fill = p.card;
        v.extreme_bg_color = p.field;
        v.selection.bg_fill = p.accent;
        v.selection.stroke.color = egui::Color32::WHITE;
        v.widgets.noninteractive.bg_stroke.color = p.line;
        v.widgets.inactive.weak_bg_fill = p.button;
        v.widgets.inactive.bg_fill = p.button;
        v.widgets.hovered.weak_bg_fill = p.button_hover;
        v.widgets.hovered.bg_fill = p.button_hover;
        for w in [&mut v.widgets.noninteractive, &mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
            w.corner_radius = egui::CornerRadius::same(8);
        }
    });
}

/// A rounded box of the page: the hero, the two tables under it, this PC in the side column.
fn card<R>(ui: &mut egui::Ui, p: &Palette, margin: i8, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::new().fill(p.card).stroke(egui::Stroke::new(1.0, p.line)).corner_radius(12).inner_margin(margin).show(ui, add).inner
}

struct Line {
    text: String,
    severity: Severity,
}

/// One scrolling view of lines.
#[derive(Default)]
struct View {
    lines: Vec<Line>,
    filter: String,
    free: bool,
    /// The lines selected, as the line clicked first and the one reached (either may be the
    /// later one), and whether the mouse is still held down from that click.
    selection: Option<(usize, usize)>,
    selecting: bool,
}

impl View {
    fn push(&mut self, text: String, severity: Severity) {
        if self.lines.len() >= MAX_LINES {
            let gone = MAX_LINES / 10;
            self.lines.drain(..gone);
            // the selection moves up with its lines, and is dropped once one of them is gone
            self.selection = self.selection.and_then(|(a, b)| Some((a.checked_sub(gone)?, b.checked_sub(gone)?)));
        }
        self.lines.push(Line { text, severity });
    }

    /// The selected lines that `shows` lets through, joined as text.
    fn selected_text(&self, shows: impl Fn(&Line) -> bool) -> Option<String> {
        let (a, b) = self.selection?;
        let (from, to) = (a.min(b), a.max(b).min(self.lines.len().checked_sub(1)?));
        let lines: Vec<&str> = self.lines.get(from..=to)?.iter().filter(|l| shows(l)).map(|l| l.text.as_str()).collect();
        (!lines.is_empty()).then(|| lines.join("\n"))
    }
}

struct App {
    settings: Settings,
    settings_path: PathBuf,
    saved: Settings,
    me: PathBuf,
    tab: Tab,
    console: View,
    log: View,
    errors: View,
    tx: Sender<Event>,
    rx: Receiver<Event>,
    running: Option<(Running, Job)>,
    /// Stop was pressed for the running program.
    stopping: bool,
    started: Instant,
    /// How the last program ended: the game's own code where `xcl run` printed it.
    ended: Option<String>,
    game_code: Option<u32>,
    tail: Option<Tail>,
    data_dir: Option<PathBuf>,
    session: Option<(std::fs::File, String)>,
    titles: Vec<PathBuf>,
    title: Option<PathBuf>,
    title_file: Option<TitleFile>,
    title_cfg: TitleConfig,
    title_saved: TitleConfig,
    notice: String,
    /// The steps of a chain still to run, and how many it had.
    pending: std::collections::VecDeque<Step>,
    chain_len: usize,
    progress: Option<Progress>,
    capture: Capture,
    /// The running chain builds XCL too (it does not only update).
    builds: bool,
    /// The commit noted before an update, and whether the update changed the shader translators.
    head_before: Option<String>,
    translators_changed: bool,
    /// The precompile pass: its speed, when it last reported, and the pipelines it built.
    eta: logic::Eta,
    eta_at: Instant,
    pipelines: Option<(u32, u32)>,
}

impl App {
    fn new() -> App {
        let settings_path = logic::gui_dir().join("settings.toml");
        let settings = Settings::load(&settings_path);
        let (tx, rx) = channel();
        let mut app = App {
            saved: settings.clone(),
            settings,
            settings_path,
            me: std::env::current_exe().unwrap_or_default(),
            tab: Tab::Home,
            console: View::default(),
            log: View::default(),
            errors: View::default(),
            tx,
            rx,
            running: None,
            stopping: false,
            started: Instant::now(),
            ended: None,
            game_code: None,
            tail: None,
            data_dir: None,
            session: None,
            titles: Vec::new(),
            title: None,
            title_file: None,
            title_cfg: TitleConfig::default(),
            title_saved: TitleConfig::default(),
            notice: String::new(),
            pending: std::collections::VecDeque::new(),
            chain_len: 0,
            progress: None,
            capture: Capture::Nothing,
            builds: false,
            head_before: None,
            translators_changed: false,
            eta: logic::Eta::default(),
            eta_at: Instant::now(),
            pipelines: None,
        };
        if !app.settings.last_data_dir.is_empty() {
            app.data_dir = Some(PathBuf::from(&app.settings.last_data_dir));
        }
        app.find_titles();
        app
    }

    fn lang(&self) -> Lang {
        self.settings.lang
    }

    fn t(&self, key: &'static str) -> &'static str {
        tr(self.lang(), key)
    }

    fn repo(&self) -> Option<PathBuf> {
        let named = self.settings.repo.trim();
        if named.is_empty() {
            logic::repo_above(&self.me)
        } else {
            Some(PathBuf::from(named))
        }
    }

    fn xcl(&self) -> Option<PathBuf> {
        logic::find_xcl(&self.settings, &self.me, self.repo().as_deref())
    }

    /// The titles XCL has run; the one of the last run is chosen, else the newest.
    fn find_titles(&mut self) {
        self.titles = logic::title_dirs(&logic::data_root());
        let wanted = self.data_dir.clone().filter(|d| self.titles.contains(d)).or_else(|| self.titles.first().cloned());
        if wanted != self.title || self.title_file.is_none() {
            self.load_title(wanted);
        }
    }

    fn load_title(&mut self, dir: Option<PathBuf>) {
        self.title = dir;
        self.title_file = None;
        if let Some(dir) = &self.title {
            match TitleFile::read(&dir.join("xcl.toml")) {
                Ok(f) => {
                    self.title_cfg = f.config();
                    self.title_saved = self.title_cfg.clone();
                    self.title_file = Some(f);
                }
                Err(e) => self.notice = e,
            }
        }
    }

    fn save_title(&mut self) {
        let (Some(dir), Some(file)) = (&self.title, &mut self.title_file) else { return };
        file.set(&self.title_cfg);
        match file.write(&dir.join("xcl.toml")) {
            Ok(()) => {
                self.title_saved = self.title_cfg.clone();
                self.notice = format!("{}: {}", tr(self.settings.lang, "saved"), dir.join("xcl.toml").display());
            }
            Err(e) => self.notice = e,
        }
    }

    fn save_settings(&mut self) {
        if self.settings != self.saved {
            match self.settings.save(&self.settings_path) {
                Ok(()) => self.saved = self.settings.clone(),
                Err(e) => self.notice = e,
            }
        }
    }

    fn say(&mut self, text: impl Into<String>) {
        let text = text.into();
        if let Some((file, _)) = &mut self.session {
            let _ = writeln!(file, "{text}");
        }
        let severity = logic::severity(&text);
        if matches!(severity, Severity::Error | Severity::Warn) {
            self.errors.push(text.clone(), severity);
        }
        self.console.push(text, severity);
    }

    fn start(&mut self, job: Job, program: &Path, args: Vec<String>, env: Vec<(String, String)>, cwd: Option<PathBuf>, ctx: &egui::Context) {
        if self.running.is_some() {
            return;
        }
        self.ended = None;
        self.stopping = false;
        self.game_code = None;
        self.notice.clear();
        if job == Job::Game {
            self.console.lines.clear();
            self.log.lines.clear();
            self.errors.lines.clear();
            self.tail = None;
            self.session = None;
            if self.settings.save_sessions {
                let dir = logic::gui_dir().join("sessions");
                let stamp = logic::stamp(SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()));
                if std::fs::create_dir_all(&dir).is_ok() {
                    if let Ok(f) = std::fs::File::create(dir.join(format!("{stamp}_console.log"))) {
                        self.session = Some((f, stamp));
                    }
                }
            }
        }
        let shown: Vec<String> = std::iter::once(program.to_string_lossy().into_owned()).chain(args.iter().cloned()).collect();
        self.say(format!("> {}", shown.join(" ")));
        for (k, v) in &env {
            self.say(format!("  {k}={v}"));
        }
        let wake = ctx.clone();
        match runner::start(program, &args, &env, cwd.as_deref(), self.tx.clone(), move || wake.request_repaint()) {
            Ok(r) => {
                self.running = Some((r, job));
                self.started = Instant::now();
            }
            Err(e) => {
                self.say(format!("error: {e}"));
                self.session = None;
                self.pending.clear();
                self.progress = None;
                match program.to_str() {
                    Some("git") => self.notice = self.t("no_git").into(),
                    Some("cargo") => self.notice = self.t("no_cargo").into(),
                    _ => {}
                }
            }
        }
    }

    fn play(&mut self, ctx: &egui::Context) {
        let game = self.settings.game.trim().trim_matches('"').to_string();
        if game.is_empty() {
            self.notice = self.t("no_game").into();
            return;
        }
        let Some(xcl) = self.xcl() else {
            self.notice = self.t("no_xcl").into();
            self.tab = Tab::Xcl;
            return;
        };
        self.save_settings();
        let (env, refused) = self.settings.environment();
        let mut args = vec!["run".to_string(), game];
        let game_args = self.settings.arguments();
        if !game_args.is_empty() {
            args.push("--".into());
            args.extend(game_args);
        }
        self.tab = Tab::Console;
        self.start(Job::Game, &xcl, args, env, None, ctx);
        for line in refused {
            self.say(format!("warning: {}: {line}", self.t("refused")));
        }
    }

    /// Runs `steps` one after another in XCL's source folder; a step that fails ends the chain.
    fn chain(&mut self, steps: Vec<Step>, ctx: &egui::Context) {
        if self.running.is_some() {
            return;
        }
        if self.repo().filter(|r| r.is_dir()).is_none() {
            self.notice = format!("{}: {}", self.t("repo"), self.t("none"));
            return;
        }
        self.save_settings();
        self.chain_len = steps.len();
        self.builds = steps.iter().any(|s| s.label == "step_cargo");
        self.pending = steps.into();
        self.head_before = None;
        self.translators_changed = false;
        self.next_step(ctx);
    }

    /// `xcl shaders <package> --precompile --only cs --pipelines`: the game's compute pipelines
    /// built ahead and kept where a run reads them, so the title does not wait for the graphics
    /// driver while its world loads. Its progress lines become the bar and the time left.
    fn precompile(&mut self, ctx: &egui::Context) {
        if self.running.is_some() {
            return;
        }
        let Some(xcl) = self.xcl() else {
            self.notice = self.t("no_xcl").into();
            return;
        };
        let Some(pack) = self.shader_pack() else {
            self.notice = self.t("no_pack").into();
            return;
        };
        self.save_settings();
        let mut args: Vec<String> = ["shaders", &pack.to_string_lossy(), "--precompile", "--only", "cs", "--pipelines"].map(String::from).to_vec();
        if self.settings.used_only {
            args.push("--used-only".into());
        }
        // the adapter and the caches a run would use: what is built here is what it looks for
        let (env, _) = self.settings.environment();
        let env = env.into_iter().filter(|(k, _)| matches!(k.as_str(), "XCL_ADAPTER" | "XCL_DXIL_CACHE" | "XCL_DIRECT_DXIL" | "XCL_DXC_THREADS" | "XCL_PIPELINE_THREADS" | "XCL_DXC_DIR")).collect();
        self.pending.clear();
        self.chain_len = 1;
        self.capture = Capture::Nothing;
        self.eta = logic::Eta::default();
        self.pipelines = None;
        self.progress = Some(Progress { label: "step_precompile", step: (1, 1), units: None, detail: String::new() });
        self.start(Job::Tool, &xcl, args, env, None, ctx);
    }

    /// The shader package the precompile pass reads: the chosen one, else the first found in
    /// the game's folder.
    fn shader_pack(&self) -> Option<PathBuf> {
        let named = self.settings.shader_pack.trim().trim_matches('"');
        if !named.is_empty() {
            return Some(PathBuf::from(named));
        }
        let game = self.settings.game.trim().trim_matches('"');
        (!game.is_empty()).then(|| logic::shader_packs(Path::new(game)).into_iter().next()).flatten()
    }

    fn next_step(&mut self, ctx: &egui::Context) {
        let (Some(step), Some(repo)) = (self.pending.pop_front(), self.repo()) else { return };
        self.progress = Some(Progress { label: step.label, step: (self.chain_len - self.pending.len(), self.chain_len), units: None, detail: String::new() });
        self.capture = step.capture;
        // without a commit noted before, the comparison is of HEAD with itself: no change
        let before = self.head_before.clone().unwrap_or_else(|| "HEAD".into());
        let args = step.args.iter().map(|a| a.replace("{before}", &before)).collect();
        self.start(Job::Tool, Path::new(&step.program), args, step.env, Some(repo), ctx);
    }

    /// Takes what the running program and the followed log have sent.
    fn receive(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.rx.try_recv() {
            match event {
                Event::Console(line) => {
                    if matches!(self.running, Some((_, Job::Tool))) {
                        if let Some((done, total, detail)) = logic::cargo_progress(&line) {
                            // the bar's own line: shown as the bar, not as text
                            if let Some(p) = &mut self.progress {
                                p.units = Some((done, total));
                                p.detail = detail;
                            }
                            continue;
                        }
                        if let Some((done, total, secs)) = logic::precompile_progress(&line) {
                            if let Some(p) = &mut self.progress {
                                p.units = Some((done, total));
                            }
                            self.eta.report(done, secs);
                            self.eta_at = Instant::now();
                        }
                        if let Some(made) = logic::pipelines_line(&line) {
                            self.pipelines = Some(made);
                        }
                        match self.capture {
                            Capture::Head if logic::is_commit_id(line.trim()) => self.head_before = Some(line.trim().to_string()),
                            Capture::TranslatorChanges if line.trim().starts_with("crates/") => self.translators_changed = true,
                            _ => {}
                        }
                    }
                    if let Some(dir) = logic::data_dir_line(&line) {
                        // the title's folder: its log is followed from here on
                        let wake = ctx.clone();
                        self.tail = Some(Tail::follow(dir.join("logs").join("xcl.log"), self.tx.clone(), move || wake.request_repaint()));
                        self.settings.last_data_dir = dir.to_string_lossy().into_owned();
                        self.data_dir = Some(dir);
                    }
                    if let Some(gpu) = logic::gpu_line(&line) {
                        self.settings.last_gpu = gpu;
                    }
                    if let Some(code) = logic::exit_code_line(&line) {
                        self.game_code = Some(code);
                    }
                    self.say(line);
                }
                Event::Log(line) => {
                    let severity = logic::severity(&line);
                    if matches!(severity, Severity::Error | Severity::Warn) {
                        self.errors.push(line.clone(), severity);
                    }
                    self.log.push(line, severity);
                }
                Event::Exited(code) => {
                    let job = self.running.take().map(|(_, job)| job);
                    if let Some(tail) = &self.tail {
                        tail.stop();
                    }
                    let lang = self.lang();
                    let code = self.game_code.or(code.map(|c| c as u32));
                    let text = match code {
                        _ if std::mem::take(&mut self.stopping) => tr(lang, "stopped").to_string(),
                        Some(c) => match logic::exit_code_meaning(c, lang) {
                            Some(meaning) => format!("{c:#x} – {meaning}"),
                            None => format!("{c:#x}"),
                        },
                        None => "?".to_string(),
                    };
                    // "Stopped from this window" is the whole of it; a code is "Exited with <code>"
                    let stopped = text == tr(lang, "stopped");
                    let text = if stopped { text } else { format!("{} {text}", tr(lang, "exited")) };
                    self.say(format!("[{}] {text}", logic::stamp(SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()))));
                    self.ended = Some(text);
                    if job == Some(Job::Tool) {
                        let ok = code == Some(0) && !stopped;
                        let chained = self.chain_len > 1;
                        if ok && !self.pending.is_empty() {
                            self.next_step(ctx);
                        } else {
                            let precompiled = self.progress.as_ref().is_some_and(|p| p.label == "step_precompile");
                            self.pending.clear();
                            self.progress = None;
                            self.capture = Capture::Nothing;
                            if self.translators_changed {
                                // also when a later step failed: the sources are the new ones
                                self.settings.precompile_stale = true;
                                self.save_settings();
                            }
                            if !ok {
                                // a pass stopped half-way is no failure: the next one goes on from it
                                if !(precompiled && stopped) {
                                    self.notice = tr(lang, "step_failed").into();
                                }
                            } else if precompiled {
                                self.settings.precompile_stale = false;
                                self.save_settings();
                                self.notice = tr(lang, "precompile_done").into();
                            } else if self.translators_changed {
                                self.notice = tr(lang, "precompile_again").into();
                            } else if chained {
                                self.notice = tr(lang, if self.builds { "done" } else { "updated" }).into();
                            }
                        }
                    }
                    if job == Some(Job::Game) {
                        self.keep_session();
                        self.find_titles();
                        self.save_settings();
                    }
                }
            }
        }
    }

    /// Copies the run's `xcl.log` and `report.json` next to its console text: XCL itself keeps
    /// the last two runs only.
    fn keep_session(&mut self) {
        let Some((_, stamp)) = self.session.take() else { return };
        let dir = logic::gui_dir().join("sessions");
        if let Some(data) = &self.data_dir {
            let logs = data.join("logs");
            let _ = std::fs::copy(logs.join("xcl.log"), dir.join(format!("{stamp}_xcl.log")));
            let _ = std::fs::copy(logs.join("report.json"), dir.join(format!("{stamp}_report.json")));
        }
        logic::prune(&dir, KEEP_SESSION_FILES);
    }

    // ------------------------------------------------------------ the window's parts

    /// The column on the left: the program's name, the pages, and this PC at the bottom.
    fn side(&mut self, ui: &mut egui::Ui) {
        let lang = self.lang();
        let p = palette(self.settings.dark);
        ui.add_space(14.0);
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(30.0, 30.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 8.0, p.accent);
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "X", egui::FontId::proportional(18.0), egui::Color32::WHITE);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                ui.label(egui::RichText::new("XCL").strong().size(17.0));
                ui.label(egui::RichText::new(tr(lang, "tagline")).weak().size(10.5));
            });
        });
        ui.add_space(16.0);
        let errors = format!("{} ({})", tr(lang, "errors"), self.errors.lines.len());
        let groups: [(&str, Vec<(Tab, String)>); 3] = [
            ("", vec![(Tab::Home, tr(lang, "home").to_string())]),
            ("sec_logs", vec![(Tab::Console, tr(lang, "console").to_string()), (Tab::Log, tr(lang, "log").to_string()), (Tab::Errors, errors)]),
            ("sec_setup", vec![(Tab::Settings, tr(lang, "settings").to_string()), (Tab::Xcl, tr(lang, "xcl").to_string())]),
        ];
        ui.with_layout(egui::Layout::top_down_justified(egui::Align::Min), |ui| {
            ui.spacing_mut().item_spacing.y = 3.0;
            ui.spacing_mut().button_padding = egui::vec2(10.0, 7.0);
            for (title, tabs) in groups {
                if !title.is_empty() {
                    ui.add_space(12.0);
                    ui.label(egui::RichText::new(tr(lang, title)).weak().size(10.5));
                    ui.add_space(2.0);
                }
                for (tab, text) in tabs {
                    ui.selectable_value(&mut self.tab, tab, text);
                }
            }
        });
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                if ui.small_button(tr(lang, "language")).clicked() {
                    self.settings.lang = if self.lang() == Lang::Es { Lang::En } else { Lang::Es };
                }
                // the button names the theme it switches to
                if ui.small_button(tr(lang, if self.settings.dark { "theme" } else { "theme_dark" })).clicked() {
                    self.settings.dark = !self.settings.dark;
                }
            });
            ui.add_space(4.0);
            card(ui, &p, 10, |ui| {
                ui.set_width(ui.available_width());
                // this column is laid out from the bottom up: the last line first
                ui.spacing_mut().item_spacing.y = 2.0;
                let gpu = if self.settings.last_gpu.is_empty() { tr(lang, "gpu_unknown") } else { self.settings.last_gpu.as_str() };
                ui.add(egui::Label::new(egui::RichText::new(gpu).size(11.5)).wrap());
                ui.label(egui::RichText::new(tr(lang, "this_pc").to_uppercase()).weak().size(10.5));
            });
        });
    }

    /// What is going on, above every page: shown while something runs, ended, or needs saying.
    fn status(&mut self, ui: &mut egui::Ui) {
        let lang = self.lang();
        ui.horizontal(|ui| {
            match (&self.running, &self.ended) {
                // the precompile pass says all of this beside its own button
                (Some(_), _) if self.precompiling() => {}
                (Some((_, job)), _) => {
                    let secs = self.started.elapsed().as_secs();
                    let what = if *job == Job::Game { tr(lang, "running") } else { tr(lang, "busy") };
                    ui.colored_label(egui::Color32::from_rgb(90, 190, 110), format!("●  {what}  {:02}:{:02}", secs / 60, secs % 60));
                    if *job == Job::Tool {
                        self.progress_bar(ui, (ui.available_width() - 90.0).clamp(200.0, 620.0));
                    }
                    if ui.button(tr(lang, "stop")).clicked() {
                        self.stop();
                    }
                }
                (None, Some(text)) => {
                    ui.label(text);
                }
                (None, None) => {}
            }
            if !self.notice.is_empty() {
                ui.colored_label(egui::Color32::from_rgb(230, 170, 60), &self.notice);
            }
        });
    }

    /// Whether the pass that precompiles the shaders is the thing running.
    fn precompiling(&self) -> bool {
        self.running.is_some() && self.progress.as_ref().is_some_and(|p| p.label == "step_precompile")
    }

    fn stop(&mut self) {
        if let Some((running, _)) = &self.running {
            self.stopping = true;
            self.pending.clear();
            running.stop();
        }
    }

    /// The first page: the game with its Play button, then what is known of it and of this PC.
    fn home(&mut self, ui: &mut egui::Ui) {
        let lang = self.lang();
        let p = palette(self.settings.dark);
        let busy = self.running.is_some();
        let ctx = ui.ctx().clone();
        egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            card(ui, &p, 22, |ui| {
                ui.set_width(ui.available_width());
                let (state, colour) = match &self.running {
                    Some((_, Job::Game)) => ("state_running", egui::Color32::from_rgb(70, 170, 100)),
                    Some((_, Job::Tool)) => ("state_busy", egui::Color32::from_rgb(200, 150, 50)),
                    None if self.xcl().is_none() => ("state_no_xcl", egui::Color32::from_rgb(200, 90, 90)),
                    None => ("ready", egui::Color32::from_rgb(70, 170, 100)),
                };
                egui::Frame::new().fill(colour).corner_radius(5).inner_margin(egui::Margin::symmetric(7, 2)).show(ui, |ui| {
                    ui.label(egui::RichText::new(tr(lang, state)).size(10.0).strong().color(egui::Color32::WHITE));
                });
                let game = self.settings.game.trim().trim_matches('"').to_string();
                // the folder's or the file's own name, whichever way its path is written
                let name = game.trim_end_matches(['\\', '/']).rsplit(['\\', '/']).next().filter(|n| !n.is_empty()).map(str::to_string);
                ui.label(egui::RichText::new(name.as_deref().unwrap_or(tr(lang, "pick_game"))).size(28.0).strong());
                ui.add_enabled(!busy, egui::TextEdit::singleline(&mut self.settings.game).hint_text(tr(lang, "game_hint")).desired_width(f32::INFINITY));
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    let play = egui::Button::new(egui::RichText::new(tr(lang, "play")).size(16.0).strong().color(egui::Color32::WHITE))
                        .fill(p.accent)
                        .min_size(egui::vec2(140.0, 40.0));
                    if ui.add_enabled(!busy, play).clicked() {
                        self.play(&ctx);
                    }
                    let side = |text: &str| egui::Button::new(text.to_string()).min_size(egui::vec2(0.0, 40.0));
                    if ui.add_enabled(busy && !self.precompiling(), side(tr(lang, "stop"))).clicked() {
                        self.stop();
                    }
                    if ui.add_enabled(!busy, side(tr(lang, "folder"))).clicked() {
                        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                            self.settings.game = dir.to_string_lossy().into_owned();
                        }
                    }
                    if ui.add_enabled(!busy, side(tr(lang, "file"))).clicked() {
                        if let Some(file) = rfd::FileDialog::new().add_filter("exe / MicrosoftGame.config", &["exe", "config"]).pick_file() {
                            self.settings.game = file.to_string_lossy().into_owned();
                        }
                    }
                });
                // under Play: the pass that spares the wait while the world loads, with what it
                // is and when it has to be done again
                ui.add_space(12.0);
                let warn = egui::Color32::from_rgb(230, 170, 60);
                let precompile = egui::Button::new(egui::RichText::new(tr(lang, "precompile")).size(14.0).strong().color(egui::Color32::WHITE)).fill(p.accent).min_size(egui::vec2(0.0, 34.0));
                ui.horizontal(|ui| {
                    if ui.add_enabled(!busy, precompile).clicked() {
                        self.precompile(&ctx);
                    }
                    // the pass's progress and the time left, beside its button
                    if self.precompiling() {
                        if ui.add(egui::Button::new(tr(lang, "stop")).min_size(egui::vec2(0.0, 34.0))).clicked() {
                            self.stop();
                        }
                        self.progress_bar(ui, ui.available_width().min(720.0));
                    }
                });
                if self.settings.precompile_stale {
                    ui.colored_label(warn, tr(lang, "precompile_stale"));
                }
                ui.weak(tr(lang, "precompile_note"));
                ui.colored_label(warn, tr(lang, "precompile_warn"));
            });
            ui.add_space(10.0);
            let (none, not_yet) = (tr(lang, "none").to_string(), tr(lang, "not_yet").to_string());
            let path = |p: Option<PathBuf>| p.map_or(none.clone(), |p| p.to_string_lossy().into_owned());
            let later = |p: Option<PathBuf>| p.map_or(not_yet.clone(), |p| p.to_string_lossy().into_owned());
            let memory = match (self.title_file.is_some(), self.title_cfg.title_memory_mb, self.title_cfg.title_memory_fit) {
                (false, ..) => tr(lang, "not_yet").to_string(),
                (true, 0, Some(true)) => format!("≤ {} MiB", self.title_cfg.title_memory_budget_mb()),
                (true, ..) => format!("{} MiB", self.title_cfg.title_memory_budget_mb()),
            };
            let game_rows = [
                (tr(lang, "title"), later(self.title.as_ref().and_then(|t| t.file_name()).map(PathBuf::from))),
                (tr(lang, "data_dir"), later(self.data_dir.clone())),
                (tr(lang, "last_run"), self.ended.clone().unwrap_or_else(|| tr(lang, "not_yet").to_string())),
                (tr(lang, "errors"), self.errors.lines.len().to_string()),
                (tr(lang, "memory"), memory),
            ];
            let gpu = if self.settings.last_gpu.is_empty() { tr(lang, "gpu_unknown").to_string() } else { self.settings.last_gpu.clone() };
            let translator = match self.settings.direct_dxil.trim() {
                "1" => "tr_cs",
                "all" => "tr_all",
                _ => "tr_hlsl",
            };
            let pc_rows = [
                (tr(lang, "gpu"), gpu),
                (tr(lang, "xcl_exe"), path(self.xcl())),
                (tr(lang, "repo"), path(self.repo())),
                (tr(lang, "translator"), tr(lang, translator).to_string()),
                (tr(lang, "network"), tr(lang, if self.settings.offline { "net_off" } else { "net_on" }).to_string()),
            ];
            ui.columns(2, |cols| {
                for (col, (title, rows)) in cols.iter_mut().zip([(tr(lang, "game"), game_rows), (tr(lang, "this_pc"), pc_rows)]) {
                    card(col, &p, 16, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(egui::RichText::new(title.to_uppercase()).weak().size(10.5));
                        for (key, value) in rows {
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new(key).weak());
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    ui.add(egui::Label::new(value).truncate());
                                });
                            });
                        }
                    });
                }
            });
        });
    }

    /// The bar of the running step: cargo's own count once it says it, else a moving one.
    fn progress_bar(&self, ui: &mut egui::Ui, width: f32) {
        let Some(p) = &self.progress else { return };
        let lang = self.lang();
        let step = if p.step.1 > 1 { format!("{}/{}  ", p.step.0, p.step.1) } else { String::new() };
        let (fraction, text) = match p.units {
            Some((done, total)) => {
                let detail = if p.label == "step_precompile" { format!("  {}", self.precompile_detail(total)) } else if p.detail.is_empty() { String::new() } else { format!("  {}", p.detail) };
                (done as f32 / total as f32, format!("{step}{}  {done}/{total} ({:.0} %){detail}", tr(lang, p.label), done as f32 * 100.0 / total as f32))
            }
            None => (0.0, format!("{step}{} – {}…", tr(lang, p.label), tr(lang, "starting"))),
        };
        ui.add(egui::ProgressBar::new(fraction).desired_width(width).text(text).animate(p.units.is_none()));
    }

    /// The precompile pass in words: the pipelines built so far and the time left, counted down
    /// between the pass's reports (they come minutes apart).
    fn precompile_detail(&self, total: u32) -> String {
        let lang = self.lang();
        let left = match self.eta.left(total) {
            Some(secs) => tr(lang, "eta").replace("{}", &logic::duration_text(secs.saturating_sub(self.eta_at.elapsed().as_secs()), lang)),
            None => tr(lang, "eta_wait").to_string(),
        };
        match self.pipelines {
            Some((made, _)) => format!("· {made} {} · {left}", tr(lang, "pipelines")),
            None => format!("· {left}"),
        }
    }

    fn lines(&mut self, ui: &mut egui::Ui, which: Tab) {
        let lang = self.lang();
        let logs = self.data_dir.as_ref().map(|d| d.join("logs"));
        let view = match which {
            Tab::Log => &mut self.log,
            Tab::Errors => &mut self.errors,
            _ => &mut self.console,
        };
        let needle = view.filter.trim().to_lowercase();
        let shown: Vec<usize> = if needle.is_empty() {
            Vec::new()
        } else {
            view.lines.iter().enumerate().filter(|(_, l)| l.text.to_lowercase().contains(&needle)).map(|(i, _)| i).collect()
        };
        let count = if needle.is_empty() { view.lines.len() } else { shown.len() };
        let at = |row: usize| if needle.is_empty() { row } else { shown[row] };
        let mut copy_selection = false;
        ui.horizontal(|ui| {
            ui.label(tr(lang, "filter"));
            ui.add(egui::TextEdit::singleline(&mut view.filter).desired_width(220.0));
            let mut follow = !view.free;
            ui.checkbox(&mut follow, tr(lang, "follow"));
            view.free = !follow;
            if ui.button(tr(lang, "copy")).clicked() {
                let text: Vec<&str> = (0..count).map(|row| view.lines[at(row)].text.as_str()).collect();
                ui.ctx().copy_text(text.join("\n"));
            }
            if ui.add_enabled(view.selection.is_some(), egui::Button::new(tr(lang, "copy_selection"))).on_hover_text(tr(lang, "select_hint")).clicked() {
                copy_selection = true;
            }
            if ui.button(tr(lang, "clear")).clicked() {
                view.lines.clear();
                view.selection = None;
            }
            ui.label(format!("{count} {}", tr(lang, "lines")));
            ui.separator();
            if let Some(logs) = logs.filter(|l| l.is_dir()) {
                if ui.button(tr(lang, "open_logs")).clicked() {
                    runner::open(&logs);
                }
                if logs.join("report.json").is_file() && ui.button(tr(lang, "open_report")).clicked() {
                    runner::open(&logs.join("report.json"));
                }
            }
            let sessions = logic::gui_dir().join("sessions");
            if sessions.is_dir() && ui.button(tr(lang, "open_sessions")).clicked() {
                runner::open(&sessions);
            }
        });
        ui.separator();
        // a filter that leaves nothing, or a cleared view, has no rows to lay out
        let count = if view.lines.is_empty() { 0 } else { count };
        let row_height = ui.text_style_height(&egui::TextStyle::Monospace);
        // Lines are selected whole: a click takes one, Shift+click or a drag takes all up to
        // another (the view scrolls while the mouse is held past its edge), Ctrl+A takes every
        // line shown and Ctrl+C copies. Only the rows in view exist as widgets, so egui's own
        // text selection could never reach past them; a part of a line is taken in the box
        // under the view, which holds the selected lines as text.
        let shows = |l: &Line| needle.is_empty() || l.text.to_lowercase().contains(&needle);
        let chosen = view.selected_text(shows);
        let typing = ui.ctx().egui_wants_keyboard_input();
        let (pointer, down, shift, copy_key, all_key) = ui.input(|i| {
            let copy = i.events.iter().any(|e| matches!(e, egui::Event::Copy));
            (i.pointer.latest_pos(), i.pointer.primary_down(), i.modifiers.shift, copy, i.modifiers.command && i.key_pressed(egui::Key::A))
        });
        if !down {
            view.selecting = false;
        }
        if all_key && !typing && count > 0 {
            view.selection = Some((at(0), at(count - 1)));
            view.free = true;
        }
        if let Some(text) = chosen.as_ref().filter(|_| copy_selection || (copy_key && !typing)) {
            ui.ctx().copy_text(text.clone());
        }
        // the box of the selected lines keeps its place under the view
        let box_rows = chosen.as_ref().map_or(0, |t| t.lines().count().min(5));
        let box_height = if box_rows == 0 { 0.0 } else { box_rows as f32 * row_height + 66.0 };
        let list_height = (ui.available_height() - box_height).max(row_height * 3.0);
        let highlight = ui.visuals().selection.bg_fill.gamma_multiply(0.45);
        let (mut first_shown, mut last_shown) = (None, None);
        let area = egui::ScrollArea::both().auto_shrink(false).max_height(list_height).stick_to_bottom(!view.free).show_rows(ui, row_height, count, |ui, rows| {
            let width = ui.available_width();
            for row in rows {
                let index = at(row);
                first_shown.get_or_insert(index);
                last_shown = Some(index);
                let line = &view.lines[index];
                let text = egui::RichText::new(&line.text).monospace();
                let text = match line.severity {
                    Severity::Error => text.color(egui::Color32::from_rgb(240, 100, 100)),
                    Severity::Warn => text.color(egui::Color32::from_rgb(230, 180, 70)),
                    Severity::Detail => text.weak(),
                    Severity::Info => text,
                };
                let selected = view.selection.is_some_and(|(a, b)| (a.min(b)..=a.max(b)).contains(&index));
                // the row's place first, so that its highlight lies under its text
                let place = ui.painter().add(egui::Shape::Noop);
                let label = ui.add(egui::Label::new(text).wrap_mode(egui::TextWrapMode::Extend).selectable(false).sense(egui::Sense::click_and_drag()));
                let rect = egui::Rect::from_min_size(label.rect.min, egui::vec2(label.rect.width().max(width), label.rect.height()));
                if selected {
                    ui.painter().set(place, egui::Shape::rect_filled(rect, 0.0, highlight));
                }
                let under_pointer = pointer.is_some_and(|p| p.y >= rect.top() && p.y < rect.bottom() && ui.clip_rect().contains(p));
                if under_pointer && ui.input(|i| i.pointer.primary_pressed()) {
                    view.selection = Some(match view.selection {
                        Some((anchor, _)) if shift => (anchor, index),
                        _ => (index, index),
                    });
                    view.selecting = true;
                    // the view stays where it is while lines are chosen in it
                    view.free = true;
                } else if under_pointer && view.selecting {
                    if let Some((anchor, _)) = view.selection {
                        view.selection = Some((anchor, index));
                    }
                }
            }
        });
        if view.selecting {
            // past an edge with the mouse held: the view follows and the selection with it
            if let (Some(p), Some((anchor, _))) = (pointer, view.selection) {
                let rect = area.inner_rect;
                let step = if p.y < rect.top() { row_height * 2.0 } else if p.y > rect.bottom() { -row_height * 2.0 } else { 0.0 };
                if step != 0.0 {
                    ui.scroll_with_delta(egui::vec2(0.0, step));
                    if let Some(edge) = if step > 0.0 { first_shown } else { last_shown } {
                        view.selection = Some((anchor, edge));
                    }
                    ui.ctx().request_repaint();
                }
            }
        }
        if let Some(text) = chosen {
            ui.separator();
            let n = text.lines().count();
            ui.horizontal(|ui| {
                ui.weak(format!("{}: {n} – {}", tr(lang, "selected"), tr(lang, "select_hint")));
            });
            egui::ScrollArea::both().id_salt("selected").auto_shrink(false).show(ui, |ui| {
                ui.add(egui::TextEdit::multiline(&mut text.as_str()).font(egui::TextStyle::Monospace).desired_width(f32::INFINITY).desired_rows(box_rows));
            });
        }
    }

    fn settings_tab(&mut self, ui: &mut egui::Ui) {
        let lang = self.lang();
        egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            ui.heading(tr(lang, "run_cfg"));
            ui.checkbox(&mut self.settings.offline, tr(lang, "offline"));
            ui.checkbox(&mut self.settings.save_sessions, tr(lang, "sessions"));
            egui::Grid::new("run").num_columns(2).show(ui, |ui| {
                ui.label(tr(lang, "threads"));
                ui.add(egui::DragValue::new(&mut self.settings.pipeline_threads).range(0..=64));
                ui.end_row();
                ui.label(tr(lang, "seconds"));
                ui.add(egui::DragValue::new(&mut self.settings.run_seconds).range(0..=86_400));
                ui.end_row();
                ui.label(tr(lang, "args"));
                ui.add(egui::TextEdit::singleline(&mut self.settings.game_args).desired_width(420.0));
                ui.end_row();
            });
            ui.label(tr(lang, "extra_env"));
            ui.add(egui::TextEdit::multiline(&mut self.settings.extra_env).code_editor().desired_rows(3).desired_width(520.0));
            for line in self.settings.environment().1 {
                ui.colored_label(egui::Color32::from_rgb(230, 170, 60), format!("{}: {line}", tr(lang, "refused")));
            }

            ui.add_space(12.0);
            ui.separator();
            ui.heading(tr(lang, "gfx_run"));
            egui::Grid::new("gfx").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
                ui.label(tr(lang, "translator"));
                let names = [("", "tr_hlsl"), ("1", "tr_cs"), ("all", "tr_all")];
                let now = names.iter().find(|(v, _)| *v == self.settings.direct_dxil.trim()).map_or("tr_hlsl", |(_, k)| *k);
                egui::ComboBox::from_id_salt("translator").selected_text(tr(lang, now)).width(340.0).show_ui(ui, |ui| {
                    for (value, key) in names {
                        ui.selectable_value(&mut self.settings.direct_dxil, value.to_string(), tr(lang, key));
                    }
                });
                ui.end_row();
                ui.label(tr(lang, "dxc_threads"));
                ui.add(egui::DragValue::new(&mut self.settings.dxc_threads).range(0..=64));
                ui.end_row();
            });
            ui.checkbox(&mut self.settings.shader_cache, tr(lang, "shader_cache"));
            ui.checkbox(&mut self.settings.pipeline_cache, tr(lang, "pipeline_cache"));
            ui.checkbox(&mut self.settings.async_pipelines, tr(lang, "async_pso"));
            ui.checkbox(&mut self.settings.dred, tr(lang, "dred"));
            ui.weak(tr(lang, "gfx_note"));

            ui.add_space(12.0);
            ui.separator();
            ui.heading(tr(lang, "title_cfg"));
            if self.titles.is_empty() {
                ui.colored_label(egui::Color32::from_rgb(230, 170, 60), tr(lang, "title_none"));
                if ui.button(tr(lang, "reload")).clicked() {
                    self.find_titles();
                }
                // what the section holds, greyed: there is no file to write them to yet
                let mut preview = TitleConfig::default();
                ui.add_enabled_ui(false, |ui| Self::title_controls(ui, lang, &mut preview));
                return;
            }
            let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let mut chosen = self.title.clone();
            ui.horizontal(|ui| {
                ui.label(tr(lang, "title"));
                egui::ComboBox::from_id_salt("title").selected_text(chosen.as_deref().map(name).unwrap_or_default()).width(320.0).show_ui(ui, |ui| {
                    for dir in &self.titles {
                        ui.selectable_value(&mut chosen, Some(dir.clone()), name(dir));
                    }
                });
            });
            if chosen != self.title {
                self.load_title(chosen);
            }
            if self.title_file.is_none() {
                return;
            }
            Self::title_controls(ui, lang, &mut self.title_cfg);
            ui.add_space(6.0);
            let changed = self.title_cfg != self.title_saved;
            let (mut save, mut reload) = (false, false);
            ui.horizontal(|ui| {
                save = ui.add_enabled(changed, egui::Button::new(tr(lang, "save"))).clicked();
                reload = ui.button(tr(lang, "reload")).clicked();
                if let Some(dir) = &self.title {
                    ui.weak(dir.join("xcl.toml").to_string_lossy());
                }
            });
            if save {
                self.save_title();
            }
            if reload {
                self.title_file = None;
                self.find_titles();
            }
            if self.running.is_some() {
                ui.weak(tr(lang, "while_running"));
            }
        });
    }

    /// The controls of a title's `xcl.toml`.
    fn title_controls(ui: &mut egui::Ui, lang: Lang, c: &mut TitleConfig) {
        egui::Grid::new("title").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
            ui.label(tr(lang, "console_kind"));
            ui.horizontal(|ui| {
                let mut series_s = c.is_series_s();
                ui.radio_value(&mut series_s, false, "Series X (anaconda)");
                ui.radio_value(&mut series_s, true, "Series S (lockhart)");
                if series_s != c.is_series_s() {
                    c.device_type = if series_s { "lockhart" } else { "anaconda" }.into();
                }
            });
            ui.end_row();

            ui.label(tr(lang, "memory"));
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut c.title_memory_mb).range(0..=65_536).speed(64));
                for preset in [0u64, 6_144, 8_192, 10_240, 13_824] {
                    let name = if preset == 0 { "auto".to_string() } else { preset.to_string() };
                    if ui.small_button(name).clicked() {
                        c.title_memory_mb = preset;
                    }
                }
            });
            ui.end_row();
            if let Some(fit) = &mut c.title_memory_fit {
                ui.label("");
                ui.add_enabled(c.title_memory_mb == 0, egui::Checkbox::new(fit, tr(lang, "memory_fit")));
                ui.end_row();
            }
            ui.label("");
            let told = format!("{} – {}: {} MiB", tr(lang, "memory_hint"), tr(lang, "memory_told"), c.title_memory_budget_mb());
            let how = match (c.title_memory_mb, c.title_memory_fit) {
                (0, Some(true)) => format!("{told} ({})", tr(lang, "memory_fit_on")),
                (0, _) | (_, None) => told,
                _ => format!("{told} ({})", tr(lang, "memory_fit_manual")),
            };
            ui.add(egui::Label::new(egui::RichText::new(how).weak()).wrap());
            ui.end_row();

            ui.label(tr(lang, "uma"));
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut c.uma_mb).range(0..=16_380).speed(64));
                ui.weak(tr(lang, "uma_hint"));
            });
            ui.end_row();

            ui.label(tr(lang, "raytracing"));
            egui::ComboBox::from_id_salt("rt").selected_text(c.raytracing.clone()).show_ui(ui, |ui| {
                for v in ["auto", "tier1_1", "tier1_0", "off"] {
                    ui.selectable_value(&mut c.raytracing, v.to_string(), v);
                }
            });
            ui.end_row();

            ui.label(tr(lang, "adapter"));
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut c.d3d12_adapter).desired_width(160.0));
                ui.weak(tr(lang, "adapter_hint"));
            });
            ui.end_row();

            ui.label(tr(lang, "log_level"));
            egui::ComboBox::from_id_salt("level").selected_text(c.log_level.clone()).show_ui(ui, |ui| {
                for v in ["error", "warn", "info", "debug", "trace"] {
                    ui.selectable_value(&mut c.log_level, v.to_string(), v);
                }
            });
            ui.end_row();

            ui.label(tr(lang, "window"));
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut c.window_width).range(320..=7_680));
                ui.label("×");
                ui.add(egui::DragValue::new(&mut c.window_height).range(180..=4_320));
                for (w, h) in [(1280, 720), (1920, 1080), (2560, 1440)] {
                    if ui.small_button(format!("{w}×{h}")).clicked() {
                        (c.window_width, c.window_height) = (w, h);
                    }
                }
            });
            ui.end_row();

            ui.label(tr(lang, "window_pos"));
            ui.horizontal(|ui| {
                let custom = !matches!(c.window_pos.trim(), "auto" | "center");
                if ui.radio(c.window_pos.trim() == "auto", tr(lang, "pos_auto")).clicked() {
                    c.window_pos = "auto".into();
                }
                if ui.radio(c.window_pos.trim() == "center", tr(lang, "pos_center")).clicked() {
                    c.window_pos = "center".into();
                }
                if ui.radio(custom, tr(lang, "pos_custom")).clicked() && !custom {
                    c.window_pos = "0,0".into();
                }
                if custom {
                    ui.add(egui::TextEdit::singleline(&mut c.window_pos).desired_width(90.0));
                }
            });
            ui.end_row();

            ui.label(tr(lang, "mouse"));
            ui.add(egui::Slider::new(&mut c.mouse_sensitivity, 0.1..=5.0).step_by(0.05));
            ui.end_row();
        });
        let mut shown = !c.present.trim().eq_ignore_ascii_case("off");
        if ui.checkbox(&mut shown, tr(lang, "present")).changed() {
            c.present = if shown { "auto" } else { "off" }.into();
        }
        ui.checkbox(&mut c.keyboard_mouse, tr(lang, "kbm"));
        ui.checkbox(&mut c.d3d12_debug, tr(lang, "debug_layer"));
    }

    fn xcl_tab(&mut self, ui: &mut egui::Ui) {
        let lang = self.lang();
        let busy = self.running.is_some();
        let found = |p: Option<PathBuf>| p.map_or_else(|| tr(lang, "none").to_string(), |p| p.to_string_lossy().into_owned());
        egui::Grid::new("xcl").num_columns(3).spacing([12.0, 8.0]).show(ui, |ui| {
            ui.label(tr(lang, "xcl_exe"));
            ui.add(egui::TextEdit::singleline(&mut self.settings.xcl_exe).hint_text(tr(lang, "auto")).desired_width(460.0));
            if ui.button(tr(lang, "file")).clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("xcl", &["exe"]).pick_file() {
                    self.settings.xcl_exe = p.to_string_lossy().into_owned();
                }
            }
            ui.end_row();
            ui.label("");
            ui.weak(format!("{}: {}", tr(lang, "using"), found(self.xcl())));
            ui.end_row();

            ui.label(tr(lang, "repo"));
            ui.add(egui::TextEdit::singleline(&mut self.settings.repo).hint_text(tr(lang, "auto")).desired_width(460.0));
            if ui.button(tr(lang, "folder")).clicked() {
                if let Some(p) = rfd::FileDialog::new().pick_folder() {
                    self.settings.repo = p.to_string_lossy().into_owned();
                }
            }
            ui.end_row();
            ui.label("");
            ui.weak(format!("{}: {}", tr(lang, "using"), found(self.repo())));
            ui.end_row();
        });
        ui.add_space(8.0);
        let ctx = ui.ctx().clone();
        if ui.add_enabled(!busy, egui::Button::new(egui::RichText::new(tr(lang, "update_build")).strong().size(16.0).color(egui::Color32::WHITE)).fill(palette(self.settings.dark).accent).min_size(egui::vec2(0.0, 38.0))).clicked() {
            let mut steps = Step::update();
            steps.push(Step::build());
            self.chain(steps, &ctx);
        }
        if matches!(self.running, Some((_, Job::Tool))) && self.progress.as_ref().is_some_and(|p| p.label != "step_precompile") {
            ui.add_space(4.0);
            self.progress_bar(ui, ui.available_width().min(720.0));
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.weak(tr(lang, "one_by_one"));
            if ui.add_enabled(!busy, egui::Button::new(tr(lang, "version"))).clicked() {
                self.chain(vec![Step::git(&["log", "-1", "--format=%h %ci %s"])], &ctx);
            }
            if ui.add_enabled(!busy, egui::Button::new(tr(lang, "update"))).clicked() {
                self.chain(Step::update(), &ctx);
            }
            if ui.add_enabled(!busy, egui::Button::new(tr(lang, "build"))).clicked() {
                self.chain(vec![Step::build()], &ctx);
            }
        });
        ui.weak(tr(lang, "build_note"));

        ui.add_space(14.0);
        ui.separator();
        ui.add_space(6.0);
        ui.heading(tr(lang, "shaders"));
        let warn = egui::Color32::from_rgb(230, 170, 60);
        if self.settings.precompile_stale {
            ui.colored_label(warn, tr(lang, "precompile_stale"));
        }
        egui::Grid::new("shaders").num_columns(3).spacing([12.0, 8.0]).show(ui, |ui| {
            ui.label(tr(lang, "shader_pack"));
            ui.add(egui::TextEdit::singleline(&mut self.settings.shader_pack).hint_text(tr(lang, "shader_pack_hint")).desired_width(460.0));
            if ui.button(tr(lang, "file")).clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("rpf", &["rpf"]).pick_file() {
                    self.settings.shader_pack = p.to_string_lossy().into_owned();
                }
            }
            ui.end_row();
            ui.label("");
            ui.weak(format!("{}: {}", tr(lang, "using"), found(self.shader_pack())));
            ui.end_row();
        });
        ui.checkbox(&mut self.settings.used_only, tr(lang, "used_only"));
        ui.add_space(4.0);
        let game_runs = matches!(self.running, Some((_, Job::Game)));
        ui.horizontal(|ui| {
            let precompile = ui.add_enabled(!busy, egui::Button::new(egui::RichText::new(tr(lang, "precompile")).strong().size(16.0).color(egui::Color32::WHITE)).fill(palette(self.settings.dark).accent).min_size(egui::vec2(0.0, 38.0)));
            if precompile.clicked() {
                self.precompile(&ctx);
            }
            if self.precompiling() {
                if ui.add(egui::Button::new(tr(lang, "stop")).min_size(egui::vec2(0.0, 38.0))).clicked() {
                    self.stop();
                }
                self.progress_bar(ui, ui.available_width().min(720.0));
            }
        });
        if game_runs {
            ui.weak(tr(lang, "close_game"));
        }
        ui.add_space(4.0);
        ui.weak(tr(lang, "precompile_note"));
        ui.colored_label(warn, tr(lang, "precompile_warn"));
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        apply_style(&ctx, self.settings.dark);
        self.receive(&ctx);
        if self.running.is_some() {
            // the clock in the status line
            ctx.request_repaint_after(Duration::from_millis(500));
        }
        let p = palette(self.settings.dark);
        let side = egui::Frame::new().fill(p.side).inner_margin(egui::Margin::symmetric(12, 0));
        egui::Panel::left("side").resizable(false).exact_size(214.0).frame(side).show(ui, |ui| self.side(ui));
        // nothing to say above the page while the precompile pass runs: it reports beside its button
        let said = if self.running.is_some() { !self.precompiling() } else { self.ended.is_some() };
        if said || !self.notice.is_empty() {
            let strip = egui::Frame::new().fill(p.bg).inner_margin(egui::Margin::symmetric(16, 8));
            egui::Panel::top("status").frame(strip).show(ui, |ui| self.status(ui));
        }
        let page = egui::Frame::new().fill(p.bg).inner_margin(16);
        egui::CentralPanel::default().frame(page).show(ui, |ui| match self.tab {
            Tab::Home => self.home(ui),
            Tab::Settings => self.settings_tab(ui),
            Tab::Xcl => {
                egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| self.xcl_tab(ui));
            }
            tab => self.lines(ui, tab),
        });
        if !ctx.egui_is_using_pointer() && !ctx.egui_wants_keyboard_input() {
            self.save_settings();
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Some((running, _)) = &self.running {
            running.stop();
        }
        let _ = self.settings.save(&self.settings_path);
    }
}
