//! `xcl::gpu::adapter::Selector` parse/fallback and the `XCL_ADAPTER` env override.
//!
//! Pure logic: no GPU, no DXGI factory, no env-dependent GPU state, so this runs anywhere.
//! The out-of-range numeric-token fallback in `select_indices` is `pub(crate)` and is
//! covered by the module's own inline tests in `src/gpu/adapter_tests.rs`; what this file
//! pins down is the public contract: `parse`, `from_env_or`, `allows`, and `Display`.

use std::env;

use xcl::gpu::adapter::{Selector, ENV};

/// One at a time: `XCL_ADAPTER` is process-global, so every test that touches it holds this.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Restores the previous `XCL_ADAPTER` on drop, so a test failure never leaks it.
struct EnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prev: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn new() -> Self {
        let lock = ENV_LOCK.lock().unwrap();
        EnvGuard { _lock: lock, prev: env::var_os(ENV) }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.prev {
            Some(v) => env::set_var(ENV, v),
            None => env::remove_var(ENV),
        }
    }
}

#[test]
fn parse_empty_blank_and_auto_are_auto() {
    assert_eq!(Selector::parse(""), Selector::Auto);
    assert_eq!(Selector::parse("   "), Selector::Auto);
    assert_eq!(Selector::parse("auto"), Selector::Auto);
    assert_eq!(Selector::parse("AUTO"), Selector::Auto);
    assert_eq!(Selector::parse(" Auto "), Selector::Auto);
}

#[test]
fn parse_warp_is_case_insensitive_and_trimmed() {
    assert_eq!(Selector::parse("warp"), Selector::Warp);
    assert_eq!(Selector::parse("WARP"), Selector::Warp);
    assert_eq!(Selector::parse(" warp "), Selector::Warp);
}

#[test]
fn parse_digits_are_positional_indices() {
    assert_eq!(Selector::parse("0"), Selector::Index(0));
    assert_eq!(Selector::parse("1"), Selector::Index(1));
    assert_eq!(Selector::parse("5070"), Selector::Index(5070));
    assert_eq!(Selector::parse(" 1 "), Selector::Index(1));
}

#[test]
fn parse_leading_zeros_are_never_a_positional_index() {
    assert_eq!(Selector::parse("00"), Selector::Name("00".into()));
    assert_eq!(Selector::parse("007"), Selector::Name("007".into()));
    assert_eq!(Selector::parse("0001"), Selector::Name("0001".into()));
    assert_eq!(Selector::parse(" 00 "), Selector::Name("00".into()));
}

#[test]
fn parse_non_digits_and_overflow_are_names() {
    assert_eq!(Selector::parse("+1"), Selector::Name("+1".into()));
    assert_eq!(Selector::parse("-1"), Selector::Name("-1".into()));
    assert_eq!(Selector::parse("1 0"), Selector::Name("1 0".into()));
    assert_eq!(Selector::parse("1.5"), Selector::Name("1.5".into()));
    // 2^64 does not fit usize: a parse failure degrades to a Name, never an Index.
    assert_eq!(Selector::parse("18446744073709551616"), Selector::Name("18446744073709551616".into()));
}

#[test]
fn parse_name_is_lowercased_and_trimmed() {
    assert_eq!(Selector::parse("radeon"), Selector::Name("radeon".into()));
    assert_eq!(Selector::parse("Radeon"), Selector::Name("radeon".into()));
    assert_eq!(Selector::parse(" RTX 3080 "), Selector::Name("rtx 3080".into()));
    assert_eq!(Selector::parse("RX 7900"), Selector::Name("rx 7900".into()));
}

#[test]
fn from_env_or_prefers_env_when_set_and_non_blank() {
    let _guard = EnvGuard::new();
    env::set_var(ENV, "warp");
    assert_eq!(Selector::from_env_or("auto"), Selector::Warp);
    env::set_var(ENV, "5070");
    assert_eq!(Selector::from_env_or("auto"), Selector::Index(5070));
    env::set_var(ENV, "radeon");
    assert_eq!(Selector::from_env_or("auto"), Selector::Name("radeon".into()));
    // The configured string is ignored entirely while the env var is set.
    env::set_var(ENV, "auto");
    assert_eq!(Selector::from_env_or("warp"), Selector::Auto);
}

#[test]
fn from_env_or_falls_back_to_configured_when_unset_or_blank() {
    let _guard = EnvGuard::new();
    env::remove_var(ENV);
    assert_eq!(Selector::from_env_or("warp"), Selector::Warp);
    assert_eq!(Selector::from_env_or("5070"), Selector::Index(5070));
    assert_eq!(Selector::from_env_or("radeon"), Selector::Name("radeon".into()));
    // A configured value is parsed the same way the env value is: "00" is a name.
    assert_eq!(Selector::from_env_or("00"), Selector::Name("00".into()));
    env::set_var(ENV, "");
    assert_eq!(Selector::from_env_or("warp"), Selector::Warp);
    env::set_var(ENV, "   ");
    assert_eq!(Selector::from_env_or("warp"), Selector::Warp);
}

#[test]
fn allows_matches_by_direct_rule_only() {
    let nv = "NVIDIA GeForce RTX 3080";
    let amd = "AMD Radeon(TM) Graphics";
    assert!(Selector::Auto.allows(0, nv) && Selector::Auto.allows(1, amd));
    assert!(!Selector::Warp.allows(0, nv));
    assert!(!Selector::Warp.allows(1, amd));

    let idx = Selector::Index(2);
    assert!(idx.allows(2, amd));
    assert!(!idx.allows(1, amd));
    // Position identity, not a name match: the same index matches regardless of name.
    assert!(idx.allows(2, nv));

    // Case-insensitive, substring-based name match through the parser.
    assert!(Selector::parse("RTX").allows(0, nv));
    assert!(!Selector::parse("RTX").allows(1, amd));
    assert!(Selector::parse("radeon").allows(1, amd));
    assert!(!Selector::parse("intel").allows(0, nv));
    assert!(!Selector::parse("intel").allows(1, amd));

    // Index and name matching are disjoint rules: an index past the position of the
    // name still does not match by name, and a name does not match by position.
    assert!(Selector::Index(2) != Selector::Name("2".into()));
    assert!(!Selector::Index(3080).allows(0, nv));
}

#[test]
fn display_renders_each_variant() {
    assert_eq!(Selector::Auto.to_string(), "auto");
    assert_eq!(Selector::Warp.to_string(), "warp");
    assert_eq!(Selector::Index(5070).to_string(), "5070");
    assert_eq!(Selector::Index(0).to_string(), "0");
    assert_eq!(Selector::Name("radeon".into()).to_string(), "\"radeon\"");
}
