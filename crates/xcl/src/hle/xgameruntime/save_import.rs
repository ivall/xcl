//! `XCL_IMPORT_SAVE=<folder>[;<folder>…]` (local, one title): make a decrypted PS5 story save of
//! the newer retail game (README) a container its Xbox build reads.
//!
//! Each folder is a decrypted PS5 save folder with the title's packed story save `chives.dat`
//! (`SLAYER1`, the file name, a u64 size, then an ASAV file). That ASAV file is what the Xbox build
//! stores as the slot's `Save Game Data` too: the same header (checksum at +0x160), the same
//! encrypted, LZ4-compressed JSAV and the same 32-byte signature – the title's codec keys are the
//! same on both consoles. What the Xbox build adds is the `Save Game Header` blob (296 bytes):
//! u32 6, u32 data size, u32 hash of the data, u32 hash of the header, u64 time, u32 0, u32 stored
//! size (ASAV +0x18), u32 unpacked size (ASAV +0x1c), the UTF-16 title. A zero stored size tells
//! the title the data is not compressed: it then parses the LZ4 stream as JSAV and fails to load.
//!
//! The container is named after the ASAV slot (`SARC0000`, `SARC0015` …). A slot that exists is
//! left alone (`XCL_IMPORT_SAVE_FORCE=1` replaces it).

use std::path::{Path, PathBuf};
use std::sync::Once;

const ASAV_HEADER: usize = 0x170;

/// signed-byte Jenkins one-at-a-time with the final avalanche, chained through `seed` (the
/// title's save hash: header fields and the ASAV checksum)
pub fn joaat_s(data: &[u8], seed: u32) -> u32 {
    let mut h = seed;
    for &c in data {
        h = h.wrapping_add(c as i8 as i32 as u32);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h.wrapping_add(h << 15)
}

/// ASAV +0x160: the hash of the 0x170-byte header with that field zero
pub fn asav_checksum(asav: &[u8]) -> u32 {
    let mut h = asav[..ASAV_HEADER].to_vec();
    h[0x160..0x164].fill(0);
    joaat_s(&h, 0x4bf6_72ed)
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// The 296-byte `Save Game Header` blob for `data` (an ASAV file), as the title writes it.
pub fn container_header(data: &[u8]) -> Vec<u8> {
    let mut h = vec![0u8; 0x128];
    h[0..4].copy_from_slice(&6u32.to_le_bytes());
    h[4..8].copy_from_slice(&(data.len() as u32).to_le_bytes());
    h[0x10..0x18].copy_from_slice(&data[0x20..0x28]); // time
    h[0x1c..0x20].copy_from_slice(&data[0x18..0x1c]); // stored size (0: not compressed)
    let unpacked = match u32_at(data, 0x1c) {
        0 => data.len() as u32,
        n => n,
    };
    h[0x20..0x24].copy_from_slice(&unpacked.to_le_bytes());
    h[0x24..0x124].copy_from_slice(&data[0x28..0x128]); // title
    let data_hash = joaat_s(data, 0);
    h[8..12].copy_from_slice(&data_hash.to_le_bytes());
    let mut x = joaat_s(&h[0x10..0x18], 0);
    x = joaat_s(&h[0x24..0x124], x);
    for o in [0x18, 0x1c, 0x20, 0x08] {
        x = joaat_s(&h[o..o + 4], x);
    }
    h[12..16].copy_from_slice(&x.to_le_bytes());
    h
}

/// The ASAV file inside `chives.dat`, checked (magic, size at +0x168, checksum), and its slot.
pub fn chives_asav(chives: &[u8]) -> Result<(String, Vec<u8>), String> {
    if chives.get(..8) != Some(b"SLAYER1\0") {
        return Err("not a SLAYER1 file".into());
    }
    let n = chives.get(12..16).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize).ok_or("short file")?;
    let asav = chives.get(24 + n..).ok_or("short file")?;
    if asav.len() < ASAV_HEADER || &asav[..4] != b"ASAV" {
        return Err("no ASAV file inside".into());
    }
    let size = u32_at(asav, 0x168) as usize;
    if size < ASAV_HEADER || size > asav.len() {
        return Err(format!("ASAV size {size} does not fit the file ({} bytes)", asav.len()));
    }
    let asav = &asav[..size];
    if asav_checksum(asav) != u32_at(asav, 0x160) {
        return Err("ASAV header checksum does not match".into());
    }
    let slot: String = asav[0x128..0x154].iter().take_while(|&&c| c != 0).map(|&c| c as char).collect();
    if slot.is_empty() || !slot.bytes().all(|c| c.is_ascii_alphanumeric()) {
        return Err(format!("odd slot name {slot:?}"));
    }
    Ok((slot, asav.to_vec()))
}

fn folders() -> Vec<PathBuf> {
    std::env::var_os("XCL_IMPORT_SAVE")
        .map(|v| v.to_string_lossy().split(';').map(str::trim).filter(|s| !s.is_empty()).map(PathBuf::from).collect())
        .unwrap_or_default()
}

/// Import every `XCL_IMPORT_SAVE` folder into the provider's folder `root`, once per process.
pub fn import_once(root: &Path) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let force = std::env::var_os("XCL_IMPORT_SAVE_FORCE").is_some_and(|v| v != "0");
        for dir in folders() {
            let res = std::fs::read(dir.join("chives.dat")).map_err(|e| format!("chives.dat: {e}")).and_then(|c| chives_asav(&c));
            let (slot, data) = match res {
                Ok(x) => x,
                Err(e) => {
                    crate::xlog_warn!("save import: {}: {e}", dir.display());
                    continue;
                }
            };
            let cdir = root.join(super::gamesave::enc(&slot));
            if cdir.exists() && !force {
                crate::xlog_warn!("save import: {slot} exists – left alone (XCL_IMPORT_SAVE_FORCE=1 replaces it)");
                continue;
            }
            let ops = vec![("Save Game Header".to_string(), Some(container_header(&data))), ("Save Game Data".to_string(), Some(data.clone()))];
            let hr = super::gamesave::apply(&cdir, Some(&slot), &ops);
            crate::xlog_warn!("save import: {} → {slot} ({} bytes) → {hr:#x}", dir.display(), data.len());
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joaat_s_sign_extends_bytes() {
        assert_eq!(joaat_s(b"", 0), 0);
        // ASCII is the classic joaat; a byte with the high bit counts as negative
        assert_eq!(joaat_s(b"a", 0), 0xca2e_9442);
        let unsigned = {
            let mut h: u32 = 0x80;
            h = h.wrapping_add(h << 10);
            h ^= h >> 6;
            h = h.wrapping_add(h << 3);
            h ^= h >> 11;
            h.wrapping_add(h << 15)
        };
        assert_ne!(joaat_s(&[0x80], 0), unsigned);
    }

    fn asav(stored: u32, body: usize) -> Vec<u8> {
        let mut a = vec![0u8; ASAV_HEADER + body];
        a[..4].copy_from_slice(b"ASAV");
        a[0x18..0x1c].copy_from_slice(&stored.to_le_bytes());
        a[0x1c..0x20].copy_from_slice(&0x1000u32.to_le_bytes());
        a[0x20] = 7;
        a[0x128..0x130].copy_from_slice(b"SARC0000");
        let n = a.len() as u32;
        a[0x168..0x16c].copy_from_slice(&n.to_le_bytes());
        let c = asav_checksum(&a);
        a[0x160..0x164].copy_from_slice(&c.to_le_bytes());
        a
    }

    #[test]
    fn header_carries_sizes_and_chained_hashes() {
        let a = asav(0x180, 32);
        let h = container_header(&a);
        assert_eq!(h.len(), 0x128);
        assert_eq!(u32_at(&h, 4), a.len() as u32);
        assert_eq!(u32_at(&h, 8), joaat_s(&a, 0));
        // the stored size is what makes the title decompress
        assert_eq!(u32_at(&h, 0x1c), 0x180);
        assert_eq!(u32_at(&h, 0x20), 0x1000);
        assert_eq!(h[0x10], 7);
        let mut b = a.clone();
        b[ASAV_HEADER] = 1;
        assert_ne!(container_header(&b)[12..16], h[12..16]);
    }

    #[test]
    fn chives_asav_checks_the_packed_copy() {
        let a = asav(0x180, 32);
        let mut c = b"SLAYER1\0".to_vec();
        c.extend(1u32.to_le_bytes());
        c.extend(4u32.to_le_bytes());
        c.extend(b"SARC");
        c.extend((a.len() as u64).to_le_bytes());
        c.extend(&a);
        c.extend([0u8; 16]); // padding after the ASAV file is not part of it
        assert_eq!(chives_asav(&c), Ok(("SARC0000".to_string(), a.clone())));
        let mut bad = c.clone();
        bad[24 + 4 + 0x20] ^= 1; // a header byte: the checksum no longer matches
        assert!(chives_asav(&bad).is_err());
        assert!(chives_asav(b"JSAV").is_err());
    }
}
