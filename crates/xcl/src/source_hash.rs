//! The hash `build.rs` gives the translators' sources (`XCL_TRANSLATOR_SOURCE_ID`,
//! `XCL_PER_LANE_SOURCE_ID`, `XCL_DIRECT_DXIL_SOURCE_ID`), part of the translated cache's identity
//! (`recomp::pso`). `build.rs` includes this file (`#[path]`), so its tests run with the crate's.

/// FNV-1a 64 of `bytes`, continuing `h`.
pub fn add(h: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *h ^= byte as u64;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

/// `add` of a source file's text with each CRLF taken as LF. rustc reads both the same, but a
/// checkout with `core.autocrlf` has CRLF and one without has LF: hashed as bytes, the same commit
/// got two identities, and a build from a tree where one file was rewritten with LF a third, so
/// none of them found the others' translations.
pub fn add_source(h: &mut u64, text: &[u8]) {
    let mut rest = text;
    while let Some(at) = rest.windows(2).position(|w| w == b"\r\n") {
        add(h, &rest[..at]);
        rest = &rest[at + 1..];
    }
    add(h, rest);
}

use std::path::{Path, PathBuf};

// `pso.rs` also contains cache storage and pipeline setup. Keep it out of the source hash so
// those changes do not discard translations; bump this only when its container parsing,
// translation options, profiles, or post-translation HLSL changes.
const TRANSLATION_GLUE_REVISION: &str = "pso-glue-v2";

const TRANSLATOR_SOURCES: &[&str] = &[
    "src/gpu/dxbc.rs",
    "src/gpu/rdna2.rs",
    "src/gpu/rdna2_spec.rs",
    "src/gpu/rdna2_xbox.rs",
    "src/gpu/recomp/dxc.rs",
    "src/gpu/recomp/emit.rs",
    "src/gpu/recomp/mesh_wave.rs",
    "src/gpu/recomp/mod.rs",
    "src/gpu/recomp/upper_lanes.rs",
    "src/gpu/recomp/x64_masks.rs",
    "src/gpu/shader.rs",
    "src/gpu/xbox.rs",
];

/// The per-lane form's sources (`XCL_PER_LANE`): part of only its own cache entries' identity,
/// so editing them does not invalidate the HLSL path's.
const PER_LANE_SOURCES: &[&str] = &[
    "src/gpu/recomp/lane_masks.rs",
    "src/gpu/recomp/lower.rs",
    "src/gpu/recomp/lower/wave.rs",
    "src/gpu/recomp/lower_dxil.rs",
    "src/gpu/recomp/ssa.rs",
];

/// The direct DXIL translator's sources (`XCL_DIRECT_DXIL`), part of only its own cache entries'
/// identity: every `.rs` file under `dir` but the test files (`*tests.rs`; tests inside the other
/// files are hashed with them), so a new one cannot be forgotten.
fn direct_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("list direct DXIL sources") {
        let path = entry.expect("list direct DXIL sources").path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if path.is_dir() {
            direct_sources(&path, out);
        } else if name.ends_with(".rs") && !name.ends_with("tests.rs") {
            out.push(path);
        }
    }
}

/// The three IDs (translator, per-lane, direct DXIL) of the crate checked out at `root`;
/// `watch` gets each path, relative to `root`, the IDs read.
pub fn source_ids(root: &Path, mut watch: impl FnMut(&str)) -> (u64, u64, u64) {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    add(&mut h, b"xcl-translator-source-v1\0");
    add(&mut h, TRANSLATION_GLUE_REVISION.as_bytes());
    add(&mut h, &[0]);
    for relative in TRANSLATOR_SOURCES {
        watch(relative);
        add(&mut h, relative.as_bytes());
        add(&mut h, &[0]);
        add_source(&mut h, &std::fs::read(root.join(relative)).expect("read shader translator source"));
        add(&mut h, &[0]);
    }
    let mut p = 0xcbf2_9ce4_8422_2325u64;
    for relative in PER_LANE_SOURCES {
        watch(relative);
        add(&mut p, relative.as_bytes());
        add(&mut p, &[0]);
        add_source(&mut p, &std::fs::read(root.join(relative)).expect("read per-lane source"));
        add(&mut p, &[0]);
    }
    // a directory: cargo looks at every file under it
    watch("src/gpu/dxil");
    let mut sources = Vec::new();
    direct_sources(&root.join("src/gpu/dxil"), &mut sources);
    sources.sort();
    let mut d = 0xcbf2_9ce4_8422_2325u64;
    for path in &sources {
        add(&mut d, path.strip_prefix(root).unwrap_or(path).to_string_lossy().replace('\\', "/").as_bytes());
        add(&mut d, &[0]);
        add_source(&mut d, &std::fs::read(path).expect("read direct DXIL source"));
        add(&mut d, &[0]);
    }
    (h, p, d)
}

#[cfg(test)]
mod tests {
    use super::{add, add_source, source_ids};

    /// The IDs built into this binary are the hash of the checkout being tested. They were not
    /// when `build.rs` took its root from `env!("CARGO_MANIFEST_DIR")`: Cargo reuses a compiled
    /// build script in another worktree that shares the target directory, and that script hashed
    /// the first worktree's sources.
    #[test]
    fn built_ids_are_this_checkouts_sources() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let (h, p, d) = source_ids(root, |_| {});
        assert_eq!(format!("{h:016x}"), env!("XCL_TRANSLATOR_SOURCE_ID"));
        assert_eq!(format!("{p:016x}"), env!("XCL_PER_LANE_SOURCE_ID"));
        assert_eq!(format!("{d:016x}"), env!("XCL_DIRECT_DXIL_SOURCE_ID"));
    }

    fn hash(f: fn(&mut u64, &[u8]), text: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        f(&mut h, text);
        h
    }

    #[test]
    fn a_source_hashes_the_same_with_crlf_and_lf_line_endings() {
        let lf = b"fn a() {\n    b();\n}\n";
        let crlf = b"fn a() {\r\n    b();\r\n}\r\n";
        assert_eq!(hash(add_source, crlf), hash(add_source, lf));
        // an LF file keeps the identity it had before
        assert_eq!(hash(add_source, lf), hash(add, lf));
    }

    #[test]
    fn other_carriage_returns_and_other_text_still_count() {
        let lf = b"let s = \"a\";\n";
        assert_ne!(hash(add_source, b"let s = \"a\r\";\n"), hash(add_source, lf));
        assert_ne!(hash(add_source, b"let s = \"b\";\n"), hash(add_source, lf));
        assert_ne!(hash(add_source, b"a\r\r\nb"), hash(add_source, b"a\nb"));
        // only the CR right before an LF goes
        assert_eq!(hash(add_source, b"a\r\r\nb"), hash(add, b"a\r\nb"));
    }
}
