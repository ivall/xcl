# xcl-gate

A build gate for any Rust repository. It reads every `.rs` file and every `Cargo.toml`, refuses what
the Rust documentation and tooling (std, clippy, Cargo, rustup) call a rule, and says for each failing
rule why it fails, where the rule is written, how to fix it, and which lines break it.

No path of any project is inside it. What a project chooses is passed on the command line:

```
xcl-gate [--root DIR] [--limits FILE] [--config-file FILE] [--skip DIR]... [--no-sync] [--all]
         [--base REF] [--protect PATH]...
```

- no `--limits`: every rule must be at zero;
- `--limits FILE` (`rule = count` per line): a baseline for code that already exists. A count above it
  fails the build, a count at or under it passes. A limit only goes down;
- `--config-file FILE`: turns on `hardcoded-memory` and names the one file where size constants may live;
- `--skip DIR`: leaves a directory out;
- `--base REF` and `--protect PATH`: see "Protecting the gate" below;
- before reading, the gate fetches the remote and fast-forwards a clean checkout, and fails if it cannot,
  so it never judges a stale tree (`--no-sync` for an offline run);
- `--all` lists every finding of a failing rule instead of the first five.

## Rules and their sources

| rule | what it refuses | source |
| --- | --- | --- |
| `unsafe-impl` | `unsafe impl` with no `// SAFETY:` line above | Rust Reference, Rustonomicon |
| `unsafe-fn-doc` | public `unsafe fn` with no `# Safety` section | clippy `missing_safety_doc` |
| `unsafe-block` | `unsafe { }` with no `// SAFETY:` comment | clippy `undocumented_unsafe_blocks` |
| `panic-source` | `unwrap()` and `expect()` | clippy `unwrap_used`, `expect_used` |
| `mem-leak` | `mem::forget`, `Box::leak`, `.leak()` | clippy `mem_forget` |
| `mem-uninit` | `mem::uninitialized` | std documentation |
| `unbounded-allocation` | `vec![0; n]`, `with_capacity(n)` with a size nothing bounds | `Vec::try_reserve` |
| `hardcoded-memory` | a memory size written as a constant outside the config file | project policy |
| `cargo-lock` | `Cargo.lock` missing or not tracked | The Cargo Book |
| `cargo-profile` | `overflow-checks` or `debug-assertions` set in dev or test profiles | The Cargo Book |
| `toolchain-channel` | a toolchain pinned to nightly or beta | rustup book |
| `gate-tamper` | a change that edits the gate, its workflow, or raises a limit | project policy |

## Protecting the gate

A gate that the code under review can edit approves itself. Three layers keep that from happening:

1. **The gate that runs is the target branch's.** On a pull request the CI step first restores
   `crates/xcl-gate` from the target branch, so the copy a contributor changed is never the one executed.
2. **The guard.** With `--base origin/<branch>` the gate lists what the change touched
   (`git diff base...HEAD`) and fails on any edit to a `--protect` path (the gate, the workflow, the
   CODEOWNERS file), and on any limit that goes up, any new limit line, or a deleted limits file.
3. **Code owners.** `.github/CODEOWNERS` assigns those paths to one owner. Turn on "Require review from
   Code Owners" in the branch protection of the default branch. This last step is a GitHub setting: no
   file in the repository can switch it on.

Without layer 3 the first two are a check, not a lock: someone who can edit the workflow and merge
without review gets around them.

## What it does not do

It is a text scan, not a compiler. It reads lines, so it can miscount code inside a string, a comment
block or a macro, and it does not tell test code from other code. Rules that need types are left to
clippy and are not claimed.

## In the CI

`install-in-repo.sh REPO CONFIG_FILE OWNER` copies the gate into `crates/xcl-gate`, writes the baseline
into `scripts/gate-limits.txt`, adds the "Build gate" step before "Build and test" and writes CODEOWNERS.
Put the step before the build and the tests, so a refused finding stops the build first.
