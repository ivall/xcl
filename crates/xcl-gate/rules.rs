//! The per-file scan. Each rule carries its reason, its source in the Rust documentation and the
//! way to fix it, so a failure explains itself. It is a text scan, not a compiler: it works line
//! by line and can miscount code inside a string or a macro.

use crate::Options;

/// What a rule is, in words the gate prints when the rule fails.
pub struct Rule {
    pub id: &'static str,
    pub why: &'static str,
    pub source: &'static str,
    pub fix: &'static str,
}

pub const RULES: &[Rule] = &[
    Rule {
        id: "unsafe-impl",
        why: "an `unsafe impl` takes over a contract the compiler cannot check (Send, Sync, an unsafe trait); a wrong one is undefined behaviour",
        source: "Rust Reference, unsafe traits; Rustonomicon, Send and Sync",
        fix: "avoid the impl if the fields already are Send/Sync; otherwise put a `// SAFETY:` line right above it saying why the contract holds",
    },
    Rule {
        id: "unsafe-fn-doc",
        why: "a public `unsafe fn` hands its caller a precondition; without a documented one nobody can call it soundly",
        source: "clippy::missing_safety_doc; Rust API Guidelines, unsafe functions document their preconditions",
        fix: "add a `# Safety` section to its doc comment listing what the caller must guarantee",
    },
    Rule {
        id: "unsafe-block",
        why: "an `unsafe` block moves the proof of memory safety from the compiler to the author, and the proof must be written down",
        source: "clippy::undocumented_unsafe_blocks; Rust standard library developer guide, SAFETY comments",
        fix: "put a `// SAFETY:` comment on the line above the block (or on its own line) saying why every precondition holds",
    },
    Rule {
        id: "panic-source",
        why: "`unwrap` and `expect` abort the thread on None or Err, and a poisoned lock panics every later caller",
        source: "clippy::unwrap_used, clippy::expect_used; std::sync::Mutex, lock poisoning",
        fix: "return the error, or handle it; for a lock use `lock().unwrap_or_else(PoisonError::into_inner)` when the data stays usable",
    },
    Rule {
        id: "mem-leak",
        why: "`mem::forget`, `Box::leak` and `.leak()` give memory away for the life of the process",
        source: "clippy::mem_forget; std::mem::forget and Box::leak documentation",
        fix: "keep the owner alive in a structure that is dropped, or reclaim it with `Box::from_raw` / `drop`",
    },
    Rule {
        id: "mem-uninit",
        why: "`mem::uninitialized` creates invalid values and has been deprecated since Rust 1.39",
        source: "std::mem::uninitialized documentation",
        fix: "use `MaybeUninit` and justify it, or initialise the value",
    },
    Rule {
        id: "unbounded-allocation",
        why: "an allocation whose size comes from outside can ask for more than exists and abort the process",
        source: "std::vec::Vec::try_reserve documentation; std::alloc, allocation failure aborts the process",
        fix: "bound the size (`.min(LIMIT)`, a constant, or the length of data you already hold) or reserve it with `try_reserve` and handle the error",
    },
    Rule {
        id: "hardcoded-memory",
        why: "the project keeps every memory size in one configuration file so it can be tuned and audited",
        source: "project policy, enabled by --config-file",
        fix: "move the number into the configuration file and read it from there",
    },
    Rule {
        id: "cargo-lock",
        why: "without a tracked Cargo.lock the dependency versions, and their checksums, are not fixed",
        source: "The Cargo Book, Cargo.toml vs Cargo.lock",
        fix: "commit Cargo.lock",
    },
    Rule {
        id: "cargo-profile",
        why: "overriding these options in a development profile hides integer overflows and failed assertions",
        source: "The Cargo Book, profiles: overflow-checks and debug-assertions",
        fix: "remove `overflow-checks` and `debug-assertions` from [profile.dev] and [profile.test]",
    },
    Rule {
        id: "toolchain-channel",
        why: "a secure application is built with a stable toolchain, not a nightly or beta one",
        source: "rustup book, release channels (stable, beta, nightly)",
        fix: "pin `channel = \"stable\"` (or a version number) in rust-toolchain.toml",
    },
    Rule {
        id: "gate-tamper",
        why: "a change under review edited the checks that judge it, or raised a limit; the gate would then approve itself",
        source: "repository policy, enabled by --base and --protect",
        fix: "drop the edit from this change; a change to the gate or to a limit goes through its own review by the code owners, and a limit only goes down",
    },
];

/// One finding.
pub struct Violation {
    pub rule: &'static str,
    pub path: String,
    pub line: usize,
    pub text: String,
}

/// The code of a line, without a trailing `// comment`.
fn code(line: &str) -> &str {
    match line.find(" //") {
        Some(i) => &line[..i],
        None => line,
    }
}

fn is_comment(line: &str) -> bool {
    line.trim_start().starts_with("//")
}

/// A `// SAFETY:` mark on the line itself or in the `back` lines above it.
fn has_safety_note(lines: &[&str], at: usize, back: usize) -> bool {
    let from = at.saturating_sub(back);
    lines[from..=at].iter().any(|l| l.contains("SAFETY:"))
}

/// Whether the doc comment above line `at` has a `# Safety` heading.
fn documents_safety(lines: &[&str], at: usize) -> bool {
    lines[..at]
        .iter()
        .rev()
        .take_while(|l| {
            let t = l.trim_start();
            t.starts_with("///") || t.starts_with("#[") || t.starts_with("//!")
        })
        .any(|l| l.trim_start().starts_with("///") && l.contains('#') && l.contains("Safety"))
}

/// Text inside the brackets that follow `open` (`with_capacity(`), parentheses balanced.
fn argument_after(line: &str, open: &str) -> Option<String> {
    let at = line.find(open)?;
    let (mut depth, mut out) = (1usize, String::new());
    for c in line[at + open.len()..].chars() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(out);
                }
            }
            _ => {}
        }
        out.push(c);
    }
    None
}

/// The length of a `vec![value; len]`, when the line has one.
fn vec_length(line: &str) -> Option<String> {
    let inner = argument_after(line, "vec![")?;
    let (mut depth, mut cut) = (0i32, None);
    for (i, c) in inner.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ';' if depth == 0 => cut = Some(i),
            _ => {}
        }
    }
    cut.map(|i| inner[i + 1..].to_string())
}

/// Whether an allocation length is bounded by what the text shows: a literal, a constant, the
/// length of data already held, a `min`, a `size_of`. Anything else is unproven.
fn bounded(arg: &str) -> bool {
    let a = arg.trim();
    if a.is_empty() || a.starts_with(|c: char| c.is_ascii_digit()) {
        return true;
    }
    let mut words = a.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).filter(|w| !w.is_empty());
    let constants = |w: &str| w.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
    if words.all(constants) {
        return true;
    }
    [".len()", ".min(", "size_of"].iter().any(|b| a.contains(b))
}

/// A numeric `const` whose name says it sizes memory: returns `(name, value)`.
fn size_constant(line: &str) -> Option<(&str, &str)> {
    const WORDS: &[&str] = &[
        "BUDGET", "CAP", "KEEP", "LIMIT", "FILL", "STACK", "STAGING", "PAGES", "POOL", "SLOTS", "HEADROOM", "RESERVE",
        "MIB", "GIB", "MB", "GB",
    ];
    let t = line.trim_start();
    let t = ["pub(crate) ", "pub(super) ", "pub "]
        .iter()
        .find_map(|v| t.strip_prefix(v))
        .unwrap_or(t)
        .strip_prefix("const ")?;
    let (name, rest) = t.split_once(':')?;
    let (ty, value) = rest.split_once('=')?;
    if !matches!(ty.trim(), "usize" | "u64" | "u32") {
        return None;
    }
    let value = value.trim().trim_end_matches(';').trim();
    let plain = !value.is_empty() && value.chars().all(|c| c.is_ascii_digit() || " _*+<>()".contains(c));
    let name = name.trim();
    let sized = name.contains("SIZE") || name.contains("BYTES");
    let big = value.replace('_', "").parse::<u64>().map_or(true, |v| v >= 1024);
    (plain && (WORDS.iter().any(|w| name.contains(w)) || (sized && big))).then_some((name, value))
}

/// Applies every per-file rule to one file.
pub fn scan(rel: &str, text: &str, opts: &Options, out: &mut Vec<Violation>) {
    let lines: Vec<&str> = text.lines().collect();
    let mut hit = |rule: &'static str, i: usize| {
        out.push(Violation { rule, path: rel.to_string(), line: i + 1, text: lines[i].trim().to_string() });
    };
    let is_test_file = rel.starts_with("tests/") || rel.contains("/tests/");
    let mut test_pending = false;
    let mut test_depth: usize = 0;

    for (i, line) in lines.iter().enumerate() {
        if is_comment(line) {
            continue;
        }
        let c = code(line);
        let t = c.trim_start();

        if test_depth == 0 && (t.starts_with("#[cfg(test)]") || t.starts_with("#[test]")) {
            test_pending = true;
        }
        let opens = c.chars().filter(|&ch| ch == '{').count();
        let closes = c.chars().filter(|&ch| ch == '}').count();
        let in_test = is_test_file || test_depth > 0 || (test_pending && opens > 0);
        if test_pending && opens > 0 {
            test_depth = opens.saturating_sub(closes);
            test_pending = false;
        } else if test_pending && c.contains(';') {
            test_pending = false;
        } else if test_depth > 0 {
            test_depth = test_depth.saturating_add(opens).saturating_sub(closes);
        }

        if t.contains("unsafe impl ") && !has_safety_note(&lines, i, 1) {
            hit("unsafe-impl", i);
        }
        if t.starts_with("pub") && (t.contains("unsafe fn ") || t.contains("unsafe extern")) && t.contains("fn ") && !documents_safety(&lines, i) {
            hit("unsafe-fn-doc", i);
        }
        if (c.contains("unsafe {") || c.contains("unsafe{")) && !has_safety_note(&lines, i, 3) {
            hit("unsafe-block", i);
        }
        if !in_test && (c.contains(".unwrap()") || c.contains(".expect(")) {
            hit("panic-source", i);
        }
        if c.contains("mem::forget(") || c.contains(".leak()") || c.contains("Box::leak(") {
            hit("mem-leak", i);
        }
        if c.contains("mem::uninitialized") {
            hit("mem-uninit", i);
        }
        let lengths = [argument_after(c, "with_capacity("), vec_length(c)];
        if lengths.iter().flatten().any(|a| !bounded(a)) {
            hit("unbounded-allocation", i);
        }
        if opts.config_file.is_some() && opts.config_file.as_deref() != Some(rel) && size_constant(c).is_some() {
            hit("hardcoded-memory", i);
        }
    }
}
