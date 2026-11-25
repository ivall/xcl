use super::{is_compute_container, Input, Pipelines};
use crate::gpu::recomp;
use crate::hle::d3d12x::real::shaders_key;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::Ordering;

const HEADER: &str = "# xcl pso manifest v1: ";

fn compute_prefix(reader: impl BufRead, cutoff: u64) -> Result<Vec<u64>, String> {
    let mut lines = reader.lines();
    let header = lines.next().transpose().map_err(|e| e.to_string())?.unwrap_or_default();
    if !header.starts_with(HEADER) {
        return Err("expected an XCL PSO manifest v1 header".into());
    }
    let mut keys = Vec::new();
    let mut seen = HashSet::new();
    for (seq, row) in lines.enumerate() {
        let row = row.map_err(|e| e.to_string())?;
        let fields: Vec<_> = row.split('\t').collect();
        let invalid = || format!("invalid manifest event on line {}", seq + 2);
        if fields.first().and_then(|v| v.parse::<usize>().ok()) != Some(seq)
            || fields.get(1).and_then(|v| v.parse::<u128>().ok()).is_none() {
            return Err(invalid());
        }
        match fields[2..] {
            ["use", key, kind @ ("compute" | "graphics"), "ready" | "cold"] => {
                if key.len() != 16 || !key.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err(invalid());
                }
                let key = u64::from_str_radix(key, 16).map_err(|_| invalid())?;
                if kind == "compute" && seen.insert(key) {
                    keys.push(key);
                }
            }
            ["build", _, _, _, _, _, _] => {}
            ["present", n] => {
                let n = n.parse::<u64>().map_err(|_| invalid())?;
                if n == cutoff {
                    return Ok(keys);
                }
                if n > cutoff {
                    break;
                }
            }
            _ => return Err(invalid()),
        }
    }
    Err(format!("manifest has no recorded present {cutoff}"))
}

fn record_indices(input: &Input, keys: &[u64]) -> Result<Vec<usize>, String> {
    let mut wanted: HashSet<_> = keys.iter().copied().collect();
    let mut found = HashMap::new();
    for index in 0..input.len() {
        if wanted.is_empty() {
            break;
        }
        let (_, Some(blob)) = input.get(index) else { continue };
        let key = shaders_key(std::slice::from_ref(&blob));
        if wanted.contains(&key) && is_compute_container(&blob) {
            found.insert(key, index);
            wanted.remove(&key);
        }
    }
    if !wanted.is_empty() {
        let missing: Vec<_> = keys.iter().filter(|key| wanted.contains(*key)).map(|key| format!("{key:016x}")).collect();
        return Err(format!("unmatched compute keys ({}): {}", missing.len(), missing.join(", ")));
    }
    Ok(keys.iter().map(|key| found[key]).collect())
}

pub(super) fn run(input: &Input, manifest: &Path, cutoff: u64, windows: u32, split: Option<bool>) -> Result<(), String> {
    let file = std::fs::File::open(manifest).map_err(|e| format!("{}: {e}", manifest.display()))?;
    let keys = compute_prefix(BufReader::new(file), cutoff)?;
    let indices = record_indices(input, &keys)?;
    println!("present {cutoff}: {} compute pipelines in first-use order", keys.len());
    if keys.is_empty() {
        return Ok(());
    }
    let pipelines = Pipelines::new()?;
    let split = split.unwrap_or_else(|| recomp::pso::split_wave64_forced() || recomp::runner::wave64_supported().is_ok_and(|w| !w));
    recomp::dxc::set_memory_reserve(2);
    for (key, index) in keys.iter().zip(indices) {
        let (_, blob) = input.get(index);
        let blob = blob.ok_or_else(|| format!("cannot reread record for {key:016x}"))?;
        if shaders_key(std::slice::from_ref(&blob)) != *key {
            return Err(format!("record for {key:016x} changed during precompile"));
        }
        let compiled = recomp::pso::console_to_dxil_in(&blob, recomp::pso::Stage::Compute, 0, split, windows)
            .map_err(|e| format!("compute {key:016x}: {e}"))?;
        pipelines.make(&compiled);
    }
    let made = pipelines.made.load(Ordering::Relaxed);
    let reused = pipelines.reused.load(Ordering::Relaxed);
    let refused = pipelines.refused.load(Ordering::Relaxed);
    let capped = pipelines.capped.load(Ordering::Relaxed);
    println!("present {cutoff}: made {made}, including {reused} from driver blobs; refused {refused}, over cap {capped}");
    if refused + capped > 0 {
        return Err(format!("{} compute pipelines still need a runtime build", refused + capped));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EVENTS: &str = "# xcl pso manifest v1: seq\n0\t0\tuse\t0000000000000009\tcompute\tready\n1\t0\tuse\t0000000000000002\tgraphics\tcold\n2\t1\tbuild\tcompute\t100\t-\t-\t2\t0\n3\t2\tuse\t0000000000000001\tcompute\tcold\n4\t2\tuse\t0000000000000009\tcompute\tcold\n5\t2\tpresent\t1\n6\t3\tuse\t0000000000000003\tcompute\tcold\n7\t3\tpresent\t2\n";

    #[test]
    fn selects_compute_first_uses_in_order_through_exact_present() {
        assert_eq!(compute_prefix(EVENTS.as_bytes(), 1).unwrap(), [9, 1]);
        assert_eq!(compute_prefix(EVENTS.as_bytes(), 2).unwrap(), [9, 1, 3]);
        assert!(compute_prefix(EVENTS.as_bytes(), 4).unwrap_err().contains("no recorded present 4"));
    }

    #[test]
    fn rejects_corrupt_prefix_and_accepts_empty_prefix() {
        let empty = "# xcl pso manifest v1: seq\n0\t0\tpresent\t1\n";
        assert!(compute_prefix(empty.as_bytes(), 1).unwrap().is_empty());
        for bad in ["1\t0\tpresent\t1", "0", "0\t0\tuse\tbad\tcompute\tcold", "0\t0\tunknown"] {
            let manifest = format!("{HEADER}seq\n{bad}\n1\t1\tpresent\t1\n");
            assert!(compute_prefix(manifest.as_bytes(), 1).is_err());
        }
        assert!(compute_prefix(b"unknown\n0\t0\tpresent\t1\n".as_slice(), 1).is_err());
    }

    #[test]
    fn resolves_records_in_manifest_order_when_input_order_is_reversed() {
        let root = std::env::temp_dir().join(format!("xcl-milestone-order-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let first = super::super::tests::compute_sample();
        let mut second = first.clone();
        *second.last_mut().unwrap() ^= 1;
        let keys = [shaders_key(std::slice::from_ref(&first)), shaders_key(std::slice::from_ref(&second))];
        let paths = [root.join("second.bin"), root.join("first.bin")];
        std::fs::write(&paths[0], second).unwrap();
        std::fs::write(&paths[1], first).unwrap();
        let result = record_indices(&Input::Files(paths.to_vec()), &keys);
        for path in paths { std::fs::remove_file(path).unwrap(); }
        std::fs::remove_dir(root).unwrap();
        assert_eq!(result.unwrap(), [1, 0]);
    }

    #[test]
    fn reports_all_unmatched_keys_in_first_use_order() {
        let error = record_indices(&Input::Files(vec![]), &[9, 1]).unwrap_err();
        assert!(error.contains("0000000000000009, 0000000000000001"));
    }
}
