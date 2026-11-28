//! Build gate for any Rust repository. It refuses, over every `.rs` and `Cargo.toml` it finds,
//! what the Rust documentation and tooling (std, clippy, Cargo, rustup) name as rules, and it says
//! for each failing rule what is wrong, where the rule is written, and how to fix it.
//!
//! Usage:
//!   xcl-gate [--root DIR] [--limits FILE] [--config-file FILE] [--skip DIR]... [--no-sync] [--all]
//!            [--base REF] [--protect PATH]...
//!
//! `--base` and `--protect` stop a change from editing the gate or raising a limit (see guard.rs).
//!
//! Without `--limits` every rule must be at zero. A limits file (`rule = count` per line) is a
//! baseline for code that already exists: a count above it fails, and it only ever goes down.

mod cargo_rules;
mod guard;
mod rules;
mod sync;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rules::{Violation, RULES};

/// Directories that hold no source of the repository.
pub const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules"];

/// What the caller chose.
pub struct Options {
    /// The one file, relative to the root, where size constants may be written. Off when absent.
    pub config_file: Option<String>,
    /// Directories to leave out.
    pub skip: Vec<String>,
}

fn rust_files(dir: &Path, skip: &[String], out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    let mut items: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    items.sort();
    for p in items {
        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if p.is_dir() {
            if !SKIP_DIRS.contains(&name.as_str()) && !skip.iter().any(|s| p.ends_with(s)) {
                rust_files(&p, skip, out);
            }
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

fn relative(root: &Path, p: &Path) -> String {
    p.strip_prefix(root).unwrap_or(p).to_string_lossy().replace('\\', "/")
}

fn read_limits(path: &Path) -> Result<BTreeMap<String, usize>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut limits = BTreeMap::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let (rule, n) = line.split_once('=').ok_or_else(|| format!("bad limits line `{line}`"))?;
        let n = n.trim().parse::<usize>().map_err(|_| format!("bad limit for {}", rule.trim()))?;
        limits.insert(rule.trim().to_string(), n);
    }
    Ok(limits)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut root = PathBuf::from(".");
    let (mut limits_path, mut sync_first, mut show_all, mut print_limits) = (None::<PathBuf>, true, false, false);
    let mut opts = Options { config_file: None, skip: Vec::new() };
    let (mut base, mut protect) = (None::<String>, Vec::<String>::new());
    while let Some(a) = args.next() {
        match a.as_str() {
            "--base" => base = args.next(),
            "--protect" => protect.extend(args.next()),
            "--root" => root = args.next().map(PathBuf::from).unwrap_or(root),
            "--limits" => limits_path = args.next().map(PathBuf::from),
            "--config-file" => opts.config_file = args.next(),
            "--skip" => opts.skip.extend(args.next()),
            "--no-sync" => sync_first = false,
            "--all" => show_all = true,
            "--print-limits" => print_limits = true,
            other => {
                eprintln!("xcl-gate: unknown argument `{other}`");
                return ExitCode::from(2);
            }
        }
    }

    if sync_first && !print_limits {
        match sync::bring_up_to_date(&root) {
            Ok(sync::Synced::NotARepository) => println!("xcl-gate: not a git checkout, nothing to fetch"),
            Ok(sync::Synced::Current(h)) => println!("xcl-gate: checkout is current with its remote ({h})"),
            Ok(sync::Synced::FastForwarded { commits, head }) => {
                println!("xcl-gate: fast-forwarded {commits} commit(s), now at {head}");
            }
            Err(e) => {
                eprintln!("xcl-gate: FAILED, the checkout is not the repository's current state: {e}");
                return ExitCode::from(2);
            }
        }
    }

    let mut files = Vec::new();
    rust_files(&root, &opts.skip, &mut files);
    let mut found: Vec<Violation> = Vec::new();
    for f in &files {
        match fs::read_to_string(f) {
            Ok(text) => rules::scan(&relative(&root, f), &text, &opts, &mut found),
            Err(e) => found.push(Violation { rule: "unreadable", path: relative(&root, f), line: 0, text: e.to_string() }),
        }
    }
    let mut manifests = Vec::new();
    cargo_rules::manifests(&root, &opts.skip, &mut manifests);
    for m in &manifests {
        if let Ok(text) = fs::read_to_string(m) {
            cargo_rules::profiles(&relative(&root, m), &text, &mut found);
        }
    }
    cargo_rules::toolchain(&root, &mut found);
    cargo_rules::lock_file(&root, &mut found);
    if let Some(base) = &base {
        let limits_rel = limits_path.as_deref().map(|p| relative(&root, p));
        if let Err(e) = guard::check(&root, base, &protect, limits_rel.as_deref(), &mut found) {
            eprintln!("xcl-gate: cannot compare with {base}: {e}");
            return ExitCode::from(2);
        }
    }

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for v in &found {
        *counts.entry(v.rule).or_default() += 1;
    }
    if print_limits {
        counts.iter().for_each(|(r, n)| println!("{r} = {n}"));
        return ExitCode::SUCCESS;
    }
    println!("xcl-gate: {} Rust file(s), {} Cargo.toml, {} finding(s)", files.len(), manifests.len(), found.len());

    let limits = match limits_path.as_deref().map(read_limits) {
        Some(Ok(l)) => l,
        Some(Err(e)) => {
            eprintln!("xcl-gate: {e}");
            return ExitCode::from(2);
        }
        None => BTreeMap::new(),
    };
    let mut failed = false;
    for (rule, have) in &counts {
        let limit = limits.get(*rule).copied().unwrap_or(0);
        if *have <= limit {
            continue;
        }
        failed = true;
        println!("\nFAIL {rule}: {have} found, {limit} allowed");
        if let Some(r) = RULES.iter().find(|r| r.id == *rule) {
            println!("  why:    {}\n  source: {}\n  fix:    {}", r.why, r.source, r.fix);
        }
        let take = if show_all { usize::MAX } else { 5 };
        for v in found.iter().filter(|v| v.rule == *rule).take(take) {
            println!("  {}:{}: {}", v.path, v.line, v.text);
        }
        if !show_all && *have > 5 {
            println!("  ... {} more (--all lists them)", have - 5);
        }
    }
    if failed {
        println!("\nxcl-gate: BUILD STOPPED. Fix the findings above; a limit only goes down, never up.");
        return ExitCode::FAILURE;
    }
    println!("xcl-gate: OK");
    ExitCode::SUCCESS
}
