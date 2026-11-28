//! Prevents a change from modifying the checks that evaluate it.
//!
//! A gate is only trustworthy if the code under review cannot alter it. When `--base REF` is
//! passed, the gate inspects everything touched since branching from `REF` (`git diff REF...HEAD`)
//! and rejects:
//!
//! - any modification to a path passed via `--protect` (such as the gate itself or the workflow
//!   that runs it);
//! - any loosened constraint in the limits file: increased thresholds, newly added limit lines,
//!   or removal of the file itself. Limits may only go down.
//!
//! For this check to hold, the gate cannot rely on the version introduced by the change. The CI
//! step therefore restores the gate from `REF` before running it (see the README).

use crate::rules::Violation;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Runs `git -C root args` and returns its output, or the error text.
fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|e| format!("cannot run git: {e}"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        Err(format!("git {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim()))
    }
}

/// `rule = count` lines of a limits file, comments and blank lines ignored.
fn parse_limits(text: &str) -> BTreeMap<String, usize> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .filter_map(|(r, n)| Some((r.trim().to_string(), n.trim().parse().ok()?)))
        .collect()
}

fn found(path: &str, text: String) -> Violation {
    Violation { rule: "gate-tamper", path: path.to_string(), line: 0, text }
}

/// Whether `file` is `protected` itself, or lives under it when `protected` is a directory.
fn is_protected(file: &str, protected: &str) -> bool {
    let dir = protected.trim_end_matches('/');
    file == dir || file.starts_with(&format!("{dir}/"))
}

/// Compares the change against `base`. `limits` is the limits file, relative to the root.
pub fn check(
    root: &Path,
    base: &str,
    protect: &[String],
    limits: Option<&str>,
    out: &mut Vec<Violation>,
) -> Result<(), String> {
    let changed = git(root, &["diff", "--name-only", &format!("{base}...HEAD")])?;
    for file in changed.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if Some(file) == limits {
            check_limits(root, base, file, out);
        } else if protect.iter().any(|p| is_protected(file, p)) {
            out.push(found(file, "this change edits a file that is protected from changes under review".into()));
        }
    }
    Ok(())
}

/// A limit may go down. Anything else is a way to let new findings in.
fn check_limits(root: &Path, base: &str, file: &str, out: &mut Vec<Violation>) {
    let Ok(before) = git(root, &["show", &format!("{base}:{file}")]) else {
        return; // the file did not exist at the base: this change introduces it
    };
    let Ok(after) = fs::read_to_string(root.join(file)) else {
        out.push(found(file, "the limits file was deleted".into()));
        return;
    };
    let (before, after) = (parse_limits(&before), parse_limits(&after));
    for (rule, now) in &after {
        match before.get(rule) {
            Some(was) if now > was => out.push(found(file, format!("`{rule}` raised from {was} to {now}"))),
            None if *now > 0 => out.push(found(file, format!("`{rule}` is a new limit line ({now}); a new rule starts at zero"))),
            _ => {}
        }
    }
}
