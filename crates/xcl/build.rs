use std::path::PathBuf;

// FNV-1a over the translators' sources, each CRLF taken as LF (`source_hash.rs`, tested with the
// crate)
#[path = "src/source_hash.rs"]
mod source_hash;

fn main() {
    // xcl.exe without high-entropy ASLR: the title's memory then lies under 2^40, which a
    // retail title's BVH builder requires of its GPU addresses (#585, PR #631)
    match std::env::var("CARGO_CFG_TARGET_ENV").as_deref() {
        Ok("msvc") => println!("cargo:rustc-link-arg-bin=xcl=/HIGHENTROPYVA:NO"),
        Ok("gnu") => println!("cargo:rustc-link-arg-bin=xcl=-Wl,--disable-high-entropy-va"),
        _ => {}
    }
    // read when the script runs, not when it was compiled: Cargo reuses a compiled build script
    // in another checkout that shares the target directory (a git worktree), and the sources it
    // hashes must be this checkout's
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let (h, p, d) = source_hash::source_ids(&root, |relative| println!("cargo:rerun-if-changed={relative}"));
    println!("cargo:rustc-env=XCL_TRANSLATOR_SOURCE_ID={h:016x}");
    println!("cargo:rustc-env=XCL_PER_LANE_SOURCE_ID={p:016x}");
    println!("cargo:rustc-env=XCL_DIRECT_DXIL_SOURCE_ID={d:016x}");
}
