use super::{chosen_vendor, select_indices, Selector};

#[test]
fn selector_parsing() {
    assert_eq!(Selector::parse(""), Selector::Auto);
    assert_eq!(Selector::parse(" Auto "), Selector::Auto);
    assert_eq!(Selector::parse("WARP"), Selector::Warp);
    assert_eq!(Selector::parse("0"), Selector::Index(0));
    assert_eq!(Selector::parse("1"), Selector::Index(1));
    assert_eq!(Selector::parse("007"), Selector::Name("007".into()));
    assert_eq!(Selector::parse("+1"), Selector::Name("+1".into()));
    assert_eq!(Selector::parse("Radeon"), Selector::Name("radeon".into()));
    assert_eq!(Selector::parse(" RTX 3080 "), Selector::Name("rtx 3080".into()));
}

#[test]
fn selector_matching() {
    let nv = "NVIDIA GeForce RTX 3080";
    let amd = "AMD Radeon(TM) Graphics";
    assert!(Selector::Auto.allows(0, nv) && Selector::Auto.allows(1, amd));
    assert!(!Selector::Warp.allows(0, nv));
    assert!(Selector::Index(1).allows(1, amd) && !Selector::Index(1).allows(0, nv));
    let radeon = Selector::parse("radeon");
    assert!(radeon.allows(1, amd) && !radeon.allows(0, nv));
    assert!(Selector::parse("geforce rtx").allows(0, nv));
}

#[test]
fn numeric_selector_falls_back_to_model_token_when_out_of_range() {
    let gpus = [
        "NVIDIA GeForce RTX 5070",
        "AMD Radeon™ RX 7900XTX",
        "Intel® Arc™ A770 Graphics",
    ];
    // In-range numbers still pick by index.
    assert_eq!(select_indices(&gpus, &Selector::parse("0")), vec![0]);
    assert_eq!(select_indices(&gpus, &Selector::parse("1")), vec![1]);
    assert_eq!(select_indices(&gpus, &Selector::parse("2")), vec![2]);

    // Out-of-range numbers match standalone numeric model tokens (bounded by non-digits).
    assert_eq!(select_indices(&gpus, &Selector::parse("5070")), vec![0]);
    assert_eq!(select_indices(&gpus, &Selector::parse("7900")), vec![1]);
    assert_eq!(select_indices(&gpus, &Selector::parse("770")), vec![2]);

    // Empty adapter list produces no match.
    let empty: [&str; 0] = [];
    assert!(select_indices(&empty, &Selector::parse("0")).is_empty());

    // Partial digit substrings inside a larger model number must NOT match.
    let two = ["NVIDIA GeForce RTX 2080", "AMD Radeon(TM) Graphics"];
    assert!(select_indices(&two, &Selector::parse("2")).is_empty());
    assert!(select_indices(&two, &Selector::parse("208")).is_empty());
    assert!(select_indices(&two, &Selector::parse("80")).is_empty());
    assert_eq!(select_indices(&two, &Selector::parse("2080")), vec![0]);
}

#[test]
fn chosen_vendor_is_the_adapter_the_backend_opens() {
    const AMD: u32 = 0x1002;
    const NV: u32 = 0x10de;
    const INTEL: u32 = 0x8086;
    // hybrid machines, hardware adapters in high-performance order (the dedicated GPU first)
    let nv_dgpu_amd_igpu = [("NVIDIA GeForce RTX 4070", NV), ("AMD Radeon(TM) Graphics", AMD)];
    let amd_dgpu_intel_igpu = [("AMD Radeon RX 7900 XT", AMD), ("Intel(R) UHD Graphics 770", INTEL)];
    assert_eq!(chosen_vendor(&nv_dgpu_amd_igpu, &Selector::Auto), Some(NV), "AMD iGPU must not make an NVIDIA run 'AMD'");
    assert_eq!(chosen_vendor(&amd_dgpu_intel_igpu, &Selector::Auto), Some(AMD), "an AMD dGPU behind an Intel iGPU is still AMD");
    // `d3d12_adapter` / `XCL_ADAPTER` choose the other one
    assert_eq!(chosen_vendor(&nv_dgpu_amd_igpu, &Selector::parse("radeon")), Some(AMD));
    assert_eq!(chosen_vendor(&nv_dgpu_amd_igpu, &Selector::Index(1)), Some(AMD));
    assert_eq!(chosen_vendor(&amd_dgpu_intel_igpu, &Selector::parse("intel")), Some(INTEL));
    // a selector that matches nothing falls back to `auto`, as `real::init` does
    assert_eq!(chosen_vendor(&amd_dgpu_intel_igpu, &Selector::parse("geforce")), Some(AMD));
    // WARP or no hardware adapter: no vendor
    assert_eq!(chosen_vendor(&amd_dgpu_intel_igpu, &Selector::Warp), None);
    assert_eq!(chosen_vendor::<&str>(&[], &Selector::Auto), None);
}
