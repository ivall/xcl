//! The process logger's contract: Level::parse, enabled() ordering, and write()
//! emitting one `[<elapsed>][<tid>][<TAG>] <msg>\n` line per enabled level (#352).
//!
//! The logger keeps process-wide state (a file handle, a level, an echo flag), so
//! this is a single test in its own binary: no other test may observe or set that
//! state, and no two tests in this file may race over it.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use xcl::log::{self, Level};

struct TempLog(PathBuf);

impl TempLog {
    fn new() -> Self {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("xcl-log-contract-{}-{nonce}", std::process::id()));
        // nested: init() is responsible for creating the parent directory
        let path = root.join("nested").join("log.txt");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempLog {
    fn drop(&mut self) {
        if let Some(dir) = self.0.parent().and_then(|p| p.parent()) {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

#[test]
fn the_log_level_filter_and_file_line_format() {
    let log = TempLog::new();

    // Level::parse: known names map to their level (case-insensitively); unknown -> Info
    assert_eq!(Level::parse("error"), Level::Error);
    assert_eq!(Level::parse("warn"), Level::Warn);
    assert_eq!(Level::parse("warning"), Level::Warn);
    assert_eq!(Level::parse("debug"), Level::Debug);
    assert_eq!(Level::parse("trace"), Level::Trace);
    assert_eq!(Level::parse("TRACE"), Level::Trace);
    assert_eq!(Level::parse("Warning"), Level::Warn);
    assert_eq!(Level::parse("bogus"), Level::Info);
    assert_eq!(Level::parse(""), Level::Info);

    log::init(Some(log.path()), Level::Warn, false);

    // init() creates the file's parent directory (and the file itself)
    assert!(log.path().exists(), "init did not create the log file");
    let parent = log.path().parent().expect("log path has a parent");
    assert!(parent.is_dir(), "init did not create the parent directory");

    // enabled(): the level set at init and every level below it
    assert!(log::enabled(Level::Error));
    assert!(log::enabled(Level::Warn));
    assert!(!log::enabled(Level::Info));
    assert!(!log::enabled(Level::Debug));
    assert!(!log::enabled(Level::Trace));

    log::write(Level::Error, format_args!("error-marker"));
    log::write(Level::Warn, format_args!("warn-marker"));
    log::write(Level::Info, format_args!("info-marker"));
    log::write(Level::Debug, format_args!("debug-marker"));
    log::write(Level::Trace, format_args!("trace-marker"));
    log::flush();

    let content = std::fs::read_to_string(log.path()).expect("read the log file");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2, "expected exactly the Error and Warn lines, got:\n{content}");

    // each line is `[<elapsed>][<tid>][<TAG>] <msg>`; every enabled line names its tag and message
    for (line, tag, marker) in [(lines[0], "ERROR", "error-marker"), (lines[1], "WARN ", "warn-marker")] {
        // `[<elapsed>][<tid>][<TAG>] <msg>` – four ']'-separated segments, each parseable
        let parts: Vec<&str> = line.split(']').collect();
        assert_eq!(parts.len(), 4, "line {line:?} does not have four ']'-separated segments");
        assert!(parts[0].starts_with('[') && parts[0][1..].trim().parse::<f64>().is_ok(), "elapsed not numeric: {line:?}");
        assert!(parts[1].starts_with('[') && parts[1][1..].trim().parse::<u32>().is_ok(), "tid not numeric: {line:?}");
        assert!(line.contains(&format!("[{tag}] {marker}")), "line {line:?} is missing tag/message");
    }
    assert!(!content.contains("info-marker"), "an Info line slipped through the Warn filter");
    assert!(!content.contains("debug-marker"), "a Debug line slipped through the Warn filter");
    assert!(!content.contains("trace-marker"), "a Trace line slipped through the Warn filter");
}
