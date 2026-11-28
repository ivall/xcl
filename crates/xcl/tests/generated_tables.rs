//! The generated call tables XCL dispatches through: the PC D3D12 interfaces with their vtable
//! slots (`hle/d3d12_layout.rs`, from d3d12.h) and the GDK runtime's (class, interface, slot)
//! rows (`hle/xgameruntime/gdk_table.rs`, from xgameruntime.lib). A wrong slot calls the wrong
//! method of a live COM object, which crashes far from the cause, so a regenerated table has to
//! keep the shape COM gives these tables.

use std::collections::{HashMap, HashSet};

use xcl::hle::d3d12_layout::{Iface, IFACES};
use xcl::hle::xgameruntime::gdk_table::API;

fn by_name() -> HashMap<&'static str, &'static Iface> {
    IFACES.iter().map(|i| (i.name, i)).collect()
}

/// Names and IIDs are unique, and every IID is a lower-case registry-format GUID (what the
/// lookups print and parse).
#[test]
fn interface_names_and_iids_are_unique_and_well_formed() {
    assert!(IFACES.len() >= 100, "{} interfaces", IFACES.len());
    let mut names = HashSet::new();
    let mut iids = HashSet::new();
    for i in IFACES {
        assert!(names.insert(i.name), "{} twice", i.name);
        assert!(iids.insert(i.iid), "IID {} twice ({})", i.iid, i.name);
        let groups: Vec<&str> = i.iid.split('-').collect();
        let lens: Vec<usize> = groups.iter().map(|g| g.len()).collect();
        assert_eq!(lens, [8, 4, 4, 4, 12], "{}: {}", i.name, i.iid);
        assert!(i.iid.chars().all(|c| c == '-' || c.is_ascii_digit() || ('a'..='f').contains(&c)), "{}: {}", i.name, i.iid);
    }
}

/// COM layout: every vtable starts with IUnknown's three methods and then repeats its parent's
/// whole vtable in order, so a method keeps its slot in every derived interface. A parent is
/// IUnknown or another row of the table, and no interface derives from itself.
#[test]
fn every_vtable_extends_its_parents() {
    let all = by_name();
    for i in IFACES {
        assert_eq!(i.methods[..3], ["QueryInterface", "AddRef", "Release"], "{}", i.name);
        if i.parent == "IUnknown" {
            continue;
        }
        let p = all.get(i.parent).unwrap_or_else(|| panic!("{}: parent {} is not in the table", i.name, i.parent));
        assert!(i.methods.starts_with(p.methods), "{} does not start with {}'s vtable", i.name, p.name);
        // walk up: the chain ends at IUnknown without a cycle
        let mut seen = HashSet::from([i.name]);
        let mut at: &Iface = p;
        while at.parent != "IUnknown" {
            assert!(seen.insert(at.name), "{}: parent cycle at {}", i.name, at.name);
            at = all[at.parent];
        }
    }
}

/// One name, one slot: `slot()` takes the first match, so a name that appears twice in a
/// vtable would hide the second method.
#[test]
fn method_names_are_unique_within_a_vtable() {
    for i in IFACES {
        let mut seen = HashSet::new();
        for m in i.methods {
            assert!(seen.insert(*m), "{}: {} twice", i.name, m);
        }
    }
}

/// The numbered versions of an interface (ID3D12Device, ID3D12Device1, …) each derive from the
/// one before, so the most derived one's vtable holds every earlier version's.
#[test]
fn numbered_versions_derive_from_the_previous_one() {
    let all = by_name();
    for i in IFACES {
        let base_len = i.name.trim_end_matches(|c: char| c.is_ascii_digit()).len();
        let (base, n) = (&i.name[..base_len], &i.name[base_len..]);
        let Ok(n) = n.parse::<u32>() else { continue };
        let prev = if n == 1 { base.to_string() } else { format!("{base}{}", n - 1) };
        if let Some(p) = all.get(prev.as_str()) {
            assert_eq!(i.parent, p.name, "{} should derive from {}", i.name, p.name);
            assert!(i.methods.len() > p.methods.len(), "{} adds no method to {}", i.name, p.name);
        }
    }
}

/// Slots of the methods D3D12.X forwards most, as d3d12.h's C vtables (`*Vtbl` structs)
/// number them. These are the calls every frame makes.
#[test]
fn hot_path_slots_match_d3d12_h() {
    let all = by_name();
    let slot = |iface: &str, m: &str| all[iface].methods.iter().position(|x| *x == m);
    for (iface, m, want) in [
        ("ID3D12GraphicsCommandList", "Close", 9),
        ("ID3D12GraphicsCommandList", "Reset", 10),
        ("ID3D12GraphicsCommandList", "DrawInstanced", 12),
        ("ID3D12GraphicsCommandList", "DrawIndexedInstanced", 13),
        ("ID3D12GraphicsCommandList", "Dispatch", 14),
        ("ID3D12GraphicsCommandList", "CopyBufferRegion", 15),
        ("ID3D12GraphicsCommandList", "CopyTextureRegion", 16),
        ("ID3D12GraphicsCommandList", "SetPipelineState", 25),
        ("ID3D12GraphicsCommandList", "ResourceBarrier", 26),
        ("ID3D12GraphicsCommandList", "SetDescriptorHeaps", 28),
        ("ID3D12GraphicsCommandList", "ExecuteIndirect", 59),
        ("ID3D12GraphicsCommandList4", "DispatchRays", 76),
        ("ID3D12GraphicsCommandList6", "DispatchMesh", 79),
        ("ID3D12CommandQueue", "Signal", 14),
        ("ID3D12CommandQueue", "Wait", 15),
        ("ID3D12Fence", "GetCompletedValue", 8),
        ("ID3D12Fence", "SetEventOnCompletion", 9),
        ("ID3D12Fence", "Signal", 10),
        ("ID3D12Resource", "Map", 8),
        ("ID3D12Resource", "Unmap", 9),
        ("ID3D12Resource", "GetGPUVirtualAddress", 11),
    ] {
        assert_eq!(slot(iface, m), Some(want), "{iface}::{m}");
    }
}

/// The GDK table: one row per function (the lookup is by name), every slot past IUnknown's
/// three, a nonzero class and interface, and no two functions on one vtable slot.
#[test]
fn gdk_rows_are_unique_and_past_iunknown() {
    assert!(API.len() >= 300, "{} rows", API.len());
    let mut names = HashSet::new();
    let mut slots = HashMap::new();
    for &(name, class, iface, slot) in API {
        assert!(name.starts_with('X') && name.chars().all(|c| c.is_ascii_alphanumeric()), "{name}");
        assert!(names.insert(name), "{name} twice");
        assert!(class != 0 && iface != 0, "{name}: zero class or interface");
        assert!(slot >= 3, "{name} at slot {slot}, an IUnknown method");
        if let Some(other) = slots.insert((class, iface, slot), name) {
            panic!("{name} and {other} on the same slot {slot}");
        }
    }
}
