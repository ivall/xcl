//! The rules that read the build files, not the Rust source: lock file, profiles, toolchain.

use crate::rules::Violation;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn found(rule: &'static str, path: &str, line: usize, text: &str) -> Violation {
    Violation { rule, path: path.to_string(), line, text: text.trim().to_string() }
}

/// Every `Cargo.toml` under `dir`, with the same exclusions as the source scan.
pub fn manifests(dir: &Path, skip: &[String], out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for p in entries.flatten().map(|e| e.path()) {
        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if p.is_dir() {
            if !crate::SKIP_DIRS.contains(&name.as_str()) && !skip.iter().any(|s| p.ends_with(s)) {
                manifests(&p, skip, out);
            }
        } else if name == "Cargo.toml" {
            out.push(p);
        }
    }
}

/// Flags `overflow-checks` and `debug-assertions` set in the dev and test profiles: they must keep
/// their defaults there.
pub fn profiles(rel: &str, text: &str, out: &mut Vec<Violation>) {
    let mut watched = false;
    for (i, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.starts_with('[') {
            watched = t == "[profile.dev]" || t == "[profile.test]";
        } else if watched && (t.starts_with("overflow-checks") || t.starts_with("debug-assertions")) {
            out.push(found("cargo-profile", rel, i + 1, t));
        }
    }
}

/// Flags a toolchain pinned to nightly or beta.
pub fn toolchain(root: &Path, out: &mut Vec<Violation>) {
    for name in ["rust-toolchain.toml", "rust-toolchain"] {
        let Ok(text) = fs::read_to_string(root.join(name)) else { continue };
        for (i, line) in text.lines().enumerate() {
            let t = line.trim();
            if !t.starts_with('#') && (t.contains("nightly") || t.contains("beta")) {
                out.push(found("toolchain-channel", name, i + 1, t));
            }
        }
    }
}

/// Flags a root `Cargo.toml` with no `Cargo.lock` next to it, or one that git does not track.
pub fn lock_file(root: &Path, out: &mut Vec<Violation>) {
    if !root.join("Cargo.toml").is_file() {
        return;
    }
    if !root.join("Cargo.lock").is_file() {
        out.push(found("cargo-lock", "Cargo.lock", 0, "no Cargo.lock next to the root Cargo.toml"));
        return;
    }
    let tracked = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "--error-unmatch", "Cargo.lock"])
        .output()
        .map(|o| o.status.success());
    // outside a git checkout there is nothing to be tracked by: the file's presence is all there is to check
    let in_git = Command::new("git").arg("-C").arg(root).args(["rev-parse", "--git-dir"]).output().is_ok_and(|o| o.status.success());
    if in_git && tracked.is_ok_and(|t| !t) {
        out.push(found("cargo-lock", "Cargo.lock", 0, "Cargo.lock is not tracked by git"));
    }
}
