//! Brings the checkout up to the current state of its remote before the gate reads a line, so a
//! stale clone or a shallow checkout can never pass the gate on a partial view of the repository.

use std::path::Path;
use std::process::Command;

/// Runs `git -C root args`; returns trimmed stdout, or the command's stderr as the error.
fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|e| format!("cannot run git: {e}"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(format!("git {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim()))
    }
}

/// What the sync did, for the gate's report.
pub enum Synced {
    /// `root` is not a git checkout (a fixture or an exported tree): nothing to fetch.
    NotARepository,
    /// The checkout already holds everything its remote holds.
    Current(String),
    /// The checkout was behind its upstream branch and has been fast-forwarded.
    FastForwarded { commits: usize, head: String },
}

/// Fetches the remote and makes the checkout complete and current.
///
/// - a shallow checkout is deepened to the whole history;
/// - a branch behind its upstream is fast-forwarded when the tree is clean;
/// - a branch behind with local changes, or diverged, is an error: the gate would judge code the
///   maintainers no longer have.
///
/// A detached HEAD (a CI checkout of a pull request) has no upstream branch, so only the fetch
/// and the deepening apply.
pub fn bring_up_to_date(root: &Path) -> Result<Synced, String> {
    if git(root, &["rev-parse", "--git-dir"]).is_err() {
        return Ok(Synced::NotARepository);
    }
    git(root, &["fetch", "--quiet", "--prune", "origin"])?;
    if git(root, &["rev-parse", "--is-shallow-repository"])? == "true" {
        git(root, &["fetch", "--quiet", "--unshallow", "origin"])?;
    }

    let head = git(root, &["rev-parse", "--short", "HEAD"])?;
    let Ok(upstream) = git(root, &["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]) else {
        return Ok(Synced::Current(head));
    };

    let count = |range: &str| -> Result<usize, String> {
        git(root, &["rev-list", "--count", range])?.parse().map_err(|_| "unreadable commit count".to_string())
    };
    let behind = count("HEAD..@{upstream}")?;
    let ahead = count("@{upstream}..HEAD")?;
    if behind == 0 {
        return Ok(Synced::Current(head));
    }
    if ahead > 0 {
        return Err(format!(
            "the branch has diverged from {upstream} ({ahead} ahead, {behind} behind): rebase it, then run the gate"
        ));
    }
    if !git(root, &["status", "--porcelain", "--untracked-files=no"])?.is_empty() {
        return Err(format!(
            "the tree is {behind} commit(s) behind {upstream} and has local changes: commit or stash them, pull, then run the gate"
        ));
    }
    git(root, &["merge", "--quiet", "--ff-only", "@{upstream}"])?;
    Ok(Synced::FastForwarded { commits: behind, head: git(root, &["rev-parse", "--short", "HEAD"])? })
}
