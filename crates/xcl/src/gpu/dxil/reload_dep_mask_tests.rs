use super::*;

/// The precomputed u128 dependency mask must agree with the per-register `reads` scan for
/// every tracked address shape, including the `SOFFSET = 125` sentinel (no offset register)
/// and the base pair low/high dwords.
#[test]
fn dep_mask_agrees_with_reads_for_all_tracked_address_shapes() {
    let mut keys = Vec::new();
    for base in [0u64, 1, 2, 4, 122, 123] {
        for soff in [0u64, 1, 9, 124, 125] {
            keys.push((base, soff, 0x40u64, 4u64));
        }
    }
    for k in &keys {
        let m = dep_mask(k);
        for r in 0..128u64 {
            let bit = (m >> r) & 1 != 0;
            assert_eq!(bit, reads(k, r), "key {k:?} register {r}");
        }
        if k.1 == 125 {
            assert_ne!(m & (1u128 << 125), 0, "sentinel register bit must be set");
        }
    }
}

/// The mask-based invalidation decision must agree with the per-register scan for every
/// (key, changes-mask) pair the analyzer can produce: `regs & dep_mask != 0` iff
/// `(0..128).any(|r| regs >> r & 1 != 0 && reads(k, r))`. `u128::MAX` is included (MOVREL,
/// unknown operand types), as is `0` (no change).
#[test]
fn mask_intersection_matches_per_register_scan() {
    let keys: Vec<Key> = {
        let mut v = Vec::new();
        for base in [0u64, 1, 2, 3, 122, 123] {
            for soff in [0u64, 9, 124, 125] {
                v.push((base, soff, 0x40u64, 4u64));
            }
        }
        v
    };
    let masks = [
        0u128,
        u128::MAX,
        1u128,
        1u128 << 4,
        1u128 << 5,
        1u128 << 9,
        1u128 << 125,
        (1u128 << 4) | (1u128 << 5),
        (1u128 << 9) | (1u128 << 10),
        (1u128 << 125) ^ (1u128 << 1),
        (1u128 << 118) | (1u128 << 119),
    ];
    for k in &keys {
        let m = dep_mask(k);
        for &regs in &masks {
            let masked = regs & m != 0;
            let scanned = (0..128u64).any(|r| regs >> r & 1 != 0 && reads(k, r));
            assert_eq!(masked, scanned, "key {k:?} regs {regs:#x}");
        }
    }
}
