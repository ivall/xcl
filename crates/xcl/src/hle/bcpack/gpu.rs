//! Opt-in GPU BCPack offload backed by `bcpack-gpu-sdk` (`bcpack_gpu::ffi`, DX12 Async Compute).
//!
//! Enabled only when built with `--features gpu-bcpack` AND toggled at runtime
//! with `XCL_GPU_BCPACK=1` (off by default). Uploads `xcl`'s built-in
//! `tables.rs` tables at worker startup via runtime C-ABI setters, routes
//! `BC1`, `BC3`, `BC4`, `BC5`, `BC6H`, and `BC7` (Modes 0-7) to GPU Async
//! Compute (`bcpack-gpu-sdk v0.15.1`), and coalesces concurrent worker jobs
//! via `BcPackDecodeBatch`. Chunks smaller than `XCL_GPU_BCPACK_MIN_KB`
//! (default `32` KiB) or unsupported/invalid sizes fall back transparently to
//! `hle::bcpack::victory::decode`.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::OnceLock;

use bcpack_gpu::ffi::{
    bcpack_gpu_create, bcpack_gpu_destroy, bcpack_gpu_set_bc1_tables,
    bcpack_gpu_set_bc6h_tables, bcpack_gpu_set_bc7_tables, BcPackBatchItem, BcPackContext,
    BcPackDecodeAsync, BcPackDecodeBatch, BCPACK_FORMAT_BC1, BCPACK_FORMAT_BC3,
    BCPACK_FORMAT_BC4, BCPACK_FORMAT_BC5, BCPACK_FORMAT_BC6H, BCPACK_FORMAT_BC7,
    BCPACK_SWIZZLE_LINEAR,
};

use super::tables::{BC1, BC6, BC7, CTRL7, DICTIONARY, RANK1};

const MAX_BATCH_CHUNKS: u32 = 64;
const MAX_BATCH_BLOCKS: u32 = 2_097_152;
const MAX_BATCH_STREAM_BYTES: u32 = 24 * 1024 * 1024;

struct Job {
    bc: u8,
    src: Vec<u8>,
    out_len: usize,
    reply: mpsc::SyncSender<Result<Vec<u8>, &'static str>>,
}

struct FfiDecoder(*mut BcPackContext);

fn bc_format_layout(bc: u8) -> Option<(u32, usize)> {
    match bc {
        1 => Some((BCPACK_FORMAT_BC1, 8usize)),
        3 => Some((BCPACK_FORMAT_BC3, 16usize)),
        4 => Some((BCPACK_FORMAT_BC4, 8usize)),
        5 => Some((BCPACK_FORMAT_BC5, 16usize)),
        6 => Some((BCPACK_FORMAT_BC6H, 16usize)),
        7 => Some((BCPACK_FORMAT_BC7, 16usize)),
        _ => None,
    }
}

impl FfiDecoder {
    fn new() -> Option<Self> {
        // SAFETY: `bcpack_gpu_create` is thread-safe and returns null on adapter failure.
        let ctx = unsafe {
            bcpack_gpu_create(MAX_BATCH_CHUNKS, MAX_BATCH_BLOCKS, MAX_BATCH_STREAM_BYTES)
        };
        if ctx.is_null() {
            return None;
        }
        let dec = Self(ctx);
        if !upload_xcl_tables(dec.0) {
            return None;
        }
        Some(dec)
    }

    fn decode(&mut self, bc: u8, src: &[u8], out_len: usize) -> Result<Vec<u8>, &'static str> {
        let Some((fmt, blk_bytes)) = bc_format_layout(bc) else {
            return Err("unsupported BC format");
        };
        if out_len == 0 || !out_len.is_multiple_of(blk_bytes) {
            return Err("invalid output size");
        }
        let num_blocks = (out_len / blk_bytes) as u32;
        let mut out = vec![0u8; out_len];
        let mut fence = 0u64;
        // SAFETY: `self.0` is live for `&mut self`; `src` and `out` match the passed lengths.
        let rc = unsafe {
            BcPackDecodeAsync(
                self.0,
                src.as_ptr(),
                src.len() as u32,
                num_blocks,
                fmt,
                BCPACK_SWIZZLE_LINEAR,
                0,
                0,
                out.as_mut_ptr(),
                out_len as u32,
                &mut fence,
            )
        };
        if rc == 0 {
            Ok(out)
        } else {
            Err("GPU BCPack decode error")
        }
    }

    fn decode_batch(&mut self, jobs: &[Job]) -> Vec<Result<Vec<u8>, &'static str>> {
        let ctx = self.0;
        decode_batch_with(jobs, |items| {
            let mut kernel_ms = 0.0f64;
            // SAFETY: `ctx` is live for `&mut self`; valid items point into `job.src` and exclusive `out_bufs` entries.
            unsafe {
                BcPackDecodeBatch(
                    ctx,
                    items.as_mut_ptr(),
                    items.len() as u32,
                    &mut kernel_ms,
                )
            }
        })
    }
}

fn decode_batch_with<F>(jobs: &[Job], mut invoke: F) -> Vec<Result<Vec<u8>, &'static str>>
where
    F: FnMut(&mut [BcPackBatchItem]) -> i32,
{
    let mut out_bufs: Vec<Vec<u8>> = Vec::with_capacity(jobs.len());
    let mut items: Vec<BcPackBatchItem> = Vec::with_capacity(jobs.len());

    for job in jobs {
        let valid = bc_format_layout(job.bc).filter(|&(_, blk_bytes)| {
            job.out_len > 0 && job.out_len % blk_bytes == 0 && !job.src.is_empty()
        });
        if let Some((fmt, blk_bytes)) = valid {
            let mut buf = vec![0u8; job.out_len];
            let out_ptr = buf.as_mut_ptr();
            items.push(BcPackBatchItem {
                compressed_data: job.src.as_ptr(),
                compressed_size: job.src.len() as u32,
                num_blocks: (job.out_len / blk_bytes) as u32,
                format: fmt,
                swizzle_mode: BCPACK_SWIZZLE_LINEAR,
                width_blocks: 0,
                height_blocks: 0,
                out_host_buffer: out_ptr,
                out_capacity: job.out_len as u32,
                status: -1,
            });
            out_bufs.push(buf);
        } else {
            items.push(BcPackBatchItem {
                compressed_data: std::ptr::null(),
                compressed_size: 0,
                num_blocks: 0,
                format: BCPACK_FORMAT_BC1,
                swizzle_mode: BCPACK_SWIZZLE_LINEAR,
                width_blocks: 0,
                height_blocks: 0,
                out_host_buffer: std::ptr::null_mut(),
                out_capacity: 0,
                status: -1,
            });
            out_bufs.push(Vec::new());
        }
    }

    let rc = invoke(&mut items);
    if rc < 0 {
        return vec![Err("GPU BCPack batch decode error"); jobs.len()];
    }

    out_bufs
        .into_iter()
        .zip(items)
        .map(|(buf, item)| {
            if item.status == 0 {
                Ok(buf)
            } else {
                Err("GPU BCPack batch decode error")
            }
        })
        .collect()
}

impl Drop for FfiDecoder {
    fn drop(&mut self) {
        // SAFETY: `self.0` was created by `bcpack_gpu_create` and is destroyed once here.
        unsafe { bcpack_gpu_destroy(self.0) };
    }
}

fn worker_loop(rx: Receiver<Job>) {
    let mut dec = FfiDecoder::new();
    while let Ok(first) = rx.recv() {
        let mut batch = Vec::with_capacity(MAX_BATCH_CHUNKS as usize);
        batch.push(first);
        while batch.len() < MAX_BATCH_CHUNKS as usize {
            match rx.try_recv() {
                Ok(job) => batch.push(job),
                Err(_) => break,
            }
        }

        if batch.len() == 1 {
            let job = batch.pop().expect("single job");
            let res = match dec.as_mut() {
                Some(d) => d.decode(job.bc, &job.src, job.out_len),
                None => Err("no DX12 compute adapter"),
            };
            let _ = job.reply.send(res);
        } else {
            let results = match dec.as_mut() {
                Some(d) => d.decode_batch(&batch),
                None => vec![Err("no DX12 compute adapter"); batch.len()],
            };
            for (job, res) in batch.into_iter().zip(results) {
                let _ = job.reply.send(res);
            }
        }
    }
}

fn sender() -> Option<&'static Sender<Job>> {
    static TX: OnceLock<Option<Sender<Job>>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("xcl-gpu-bcpack".into())
            .spawn(move || worker_loop(rx))
            .ok()
            .map(|_| tx)
    })
    .as_ref()
}

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("XCL_GPU_BCPACK")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

fn min_bytes() -> usize {
    static MIN: OnceLock<usize> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("XCL_GPU_BCPACK_MIN_KB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(32)
            * 1024
    })
}

fn upload_xcl_tables(ctx: *mut BcPackContext) -> bool {
    let mut bc1_ep = vec![0u8; DICTIONARY * 6];
    let mut bc1_ix = vec![0u8; DICTIONARY * 16];
    for (k, entry) in BC1.iter().enumerate() {
        bc1_ep[k * 6..(k + 1) * 6].copy_from_slice(&entry.ep[..6]);
        bc1_ix[k * 16..(k + 1) * 16].copy_from_slice(&entry.ix[..16]);
    }

    let mut bc6_ep16 = vec![0u16; DICTIONARY * 12];
    let mut bc6_ix_mode = [0u8; DICTIONARY];
    let mut bc6_ix16 = vec![0i16; DICTIONARY * 16];
    for (k, (ep, ix)) in BC6.iter().enumerate() {
        for c in 0..4 {
            for ch in 0..3 {
                bc6_ep16[k * 12 + c * 3 + ch] = ep.e[c][ch];
            }
        }
        bc6_ix_mode[k] = ix.mode;
        bc6_ix16[k * 16..(k + 1) * 16].copy_from_slice(&ix.ix);
    }

    let mut bc7_subs = [0u8; DICTIONARY];
    let mut bc7_ep8 = vec![0u8; DICTIONARY * 24];
    let mut bc7_ix4 = vec![0u8; DICTIONARY * 32];
    for (k, (ep, ix)) in BC7.iter().enumerate() {
        bc7_subs[k] = ((ep.ctrl & 0xff) >> 6) as u8;
        for r in 0..3 {
            for c in 0..8 {
                bc7_ep8[k * 24 + r * 8 + c] = (ep.rows[r][c] >> 8) as u8;
            }
        }
        for s in 0..2 {
            let bits = ix.bits[s];
            if bits > 0 && bits <= 4 {
                for p in 0..16 {
                    bc7_ix4[k * 32 + s * 16 + p] = ((ix.ix[s][p] << (4 - bits)) & 0xF) as u8;
                }
            }
        }
    }

    // SAFETY: `ctx` is live; all pointers match the exact buffer sizes required by `bcpack_gpu.h`.
    unsafe {
        bcpack_gpu_set_bc1_tables(ctx, RANK1.as_ptr(), bc1_ep.as_ptr(), bc1_ix.as_ptr()) == 0
            && bcpack_gpu_set_bc7_tables(
                ctx,
                CTRL7.as_ptr(),
                bc7_subs.as_ptr(),
                bc7_ep8.as_ptr(),
                bc7_ix4.as_ptr(),
            ) == 0
            && bcpack_gpu_set_bc6h_tables(
                ctx,
                bc6_ep16.as_ptr(),
                bc6_ix_mode.as_ptr(),
                bc6_ix16.as_ptr(),
            ) == 0
    }
}

/// Attempts GPU BCPack decode when `XCL_GPU_BCPACK=1`. Returns `None` to fall
/// back cleanly to CPU `victory::decode` for small chunks or GPU errors.
pub(super) fn try_decode(mode: u8, src: &[u8], out_len: usize) -> Option<Result<Vec<u8>, &'static str>> {
    if !enabled() || out_len < min_bytes() {
        return None;
    }
    submit_gpu_worker(mode, src, out_len)
}

fn submit_with_sender(
    tx: Option<&Sender<Job>>,
    mode: u8,
    src: &[u8],
    out_len: usize,
) -> Option<Result<Vec<u8>, &'static str>> {
    let tx = tx?;
    let blk = super::block_size(mode)?;
    if out_len == 0
        || out_len % blk != 0
        || src.len() + 256 > MAX_BATCH_STREAM_BYTES as usize
        || out_len > (MAX_BATCH_BLOCKS as usize) * 8
    {
        return None;
    }
    let (reply_tx, reply_rx) = mpsc::sync_channel(1);
    let job = Job {
        bc: mode,
        src: src.to_vec(),
        out_len,
        reply: reply_tx,
    };
    if tx.send(job).is_err() {
        return None;
    }
    match reply_rx.recv() {
        Ok(Ok(out)) => Some(Ok(out)),
        Ok(Err(_)) | Err(_) => None,
    }
}

fn submit_gpu_worker(mode: u8, src: &[u8], out_len: usize) -> Option<Result<Vec<u8>, &'static str>> {
    submit_with_sender(sender(), mode, src, out_len)
}

#[cfg(test)]
mod tests {
    use super::super::{block_size, victory};
    use super::*;

    #[test]
    fn gpu_bcpack_matches_xcl_cpu_all_vectors() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/bcpack");
        let mut bins: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("bin"))
            .collect();
        bins.sort();
        assert!(!bins.is_empty(), "expected bcpack .bin test vectors");

        let mut valid_count = 0usize;
        for path in &bins {
            let stem = path.file_stem().unwrap().to_str().unwrap();
            let parts: Vec<&str> = stem.split('-').collect();
            assert_eq!(parts.len(), 3, "unexpected filename: {stem}");
            let mode: u8 = parts[0].strip_prefix("bc").unwrap().parse().unwrap();
            // Vector filenames count 8-byte output units (`tests.rs`).
            let units: usize = parts[1].parse().unwrap();
            let out_len = units * 8;
            let blk = block_size(mode).unwrap();
            let blocks = out_len / blk;
            let src = std::fs::read(path).unwrap();

            let mut cpu_out = vec![0u8; out_len];
            let cpu_res = victory::decode(mode, &src, &mut cpu_out, victory::Tables::builtin());
            if let Err(e) = cpu_res {
                // Some fixtures in `tests/data/bcpack` (e.g. `bc1-1-8`, `bc7-700-2`, `bc7-8192-3`)
                // are lane-only or negative tests that do not form full BC blocks.
                println!("[xcl-gpu-bcpack] {stem} (lane/negative fixture, CPU Err: {e})");
                continue;
            }

            let gpu_out = submit_gpu_worker(mode, &src, out_len)
                .unwrap_or_else(|| panic!("GPU worker returned None for {stem}"))
                .unwrap_or_else(|e| panic!("GPU decode failed for {stem}: {e}"));

            let mismatches = gpu_out
                .iter()
                .zip(cpu_out.iter())
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "[xcl-gpu-bcpack] {stem} (mode BC{mode}, {blocks} blocks, {out_len} B): {mismatches} byte mismatches"
            );
            assert_eq!(gpu_out, cpu_out, "mismatch for vector {stem}");
            valid_count += 1;
        }
        println!(
            "[xcl-gpu-bcpack] verified {} valid vectors (+ 1 expected-err): 0 byte mismatches",
            valid_count
        );
    }

    #[test]
    fn gpu_bcpack_batch_all_valid_vectors_and_isolation() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/bcpack");
        let mut bins: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("bin"))
            .collect();
        bins.sort();

        let mut jobs: Vec<Job> = Vec::new();
        let mut expected: Vec<(String, Vec<u8>)> = Vec::new();
        for path in &bins {
            let stem = path.file_stem().unwrap().to_str().unwrap().to_owned();
            let parts: Vec<&str> = stem.split('-').collect();
            let mode: u8 = parts[0].strip_prefix("bc").unwrap().parse().unwrap();
            let units: usize = parts[1].parse().unwrap();
            let out_len = units * 8;
            let src = std::fs::read(path).unwrap();
            let mut cpu_out = vec![0u8; out_len];
            if victory::decode(mode, &src, &mut cpu_out, victory::Tables::builtin()).is_ok() {
                let (tx, _rx) = mpsc::sync_channel(1);
                jobs.push(Job {
                    bc: mode,
                    src,
                    out_len,
                    reply: tx,
                });
                expected.push((stem, cpu_out));
            }
        }
        let valid_n = jobs.len();
        assert!(valid_n >= 10);

        let (bad_tx, _bad_rx) = mpsc::sync_channel(1);
        jobs.insert(
            valid_n / 2,
            Job {
                bc: 7,
                src: vec![0u8; 32],
                out_len: 4,
                reply: bad_tx,
            },
        );

        let mut dec = FfiDecoder::new().expect("DX12 compute adapter");
        let results = dec.decode_batch(&jobs);
        assert_eq!(results.len(), valid_n + 1);
        assert!(results[valid_n / 2].is_err());

        let mut exp_it = expected.iter();
        for (i, res) in results.into_iter().enumerate() {
            if i == valid_n / 2 {
                continue;
            }
            let (stem, cpu_out) = exp_it.next().unwrap();
            let gpu_out = res.unwrap_or_else(|e| panic!("batch decode failed for {stem}: {e}"));
            assert_eq!(&gpu_out, cpu_out, "batch mismatch for {stem}");
        }
        println!("[xcl-gpu-bcpack] batch decoded {valid_n} valid vectors (+1 isolated err): 0 byte mismatches");
    }

    fn first_valid_vector() -> (u8, Vec<u8>, Vec<u8>) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/bcpack");
        let mut bins: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("bin"))
            .collect();
        bins.sort();
        for path in bins {
            let stem = path.file_stem().unwrap().to_str().unwrap();
            let parts: Vec<&str> = stem.split('-').collect();
            let mode: u8 = parts[0].strip_prefix("bc").unwrap().parse().unwrap();
            let units: usize = parts[1].parse().unwrap();
            let out_len = units * 8;
            let src = std::fs::read(&path).unwrap();
            let mut cpu_out = vec![0u8; out_len];
            if victory::decode(mode, &src, &mut cpu_out, victory::Tables::builtin()).is_ok() {
                return (mode, src, cpu_out);
            }
        }
        panic!("expected at least one valid bcpack test vector");
    }

    #[test]
    fn batch_early_error_return_rejects_items_and_falls_back_to_cpu() {
        let (mode, src, cpu_expected) = first_valid_vector();
        let src_snapshot = src.clone();
        let out_len = cpu_expected.len();
        let (tx, _rx) = mpsc::sync_channel(1);
        let jobs = [Job {
            bc: mode,
            src: src.clone(),
            out_len,
            reply: tx,
        }];

        let early_err_results = decode_batch_with(&jobs, |items| {
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].status, -1, "batch item must default to failure sentinel");
            -99
        });
        assert_eq!(early_err_results.len(), 1);
        assert!(early_err_results[0].is_err());

        let tampered_status_results = decode_batch_with(&jobs, |items| {
            items[0].status = 0;
            -99
        });
        assert_eq!(tampered_status_results.len(), 1);
        assert!(
            tampered_status_results[0].is_err(),
            "negative batch return code must fail even if an item reports status 0"
        );

        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let (worker_tx, worker_rx) = mpsc::channel();
        let recv_handle = std::thread::spawn(move || {
            let job = worker_rx.recv().unwrap();
            let res = decode_batch_with(&[job], |_| -99).pop().unwrap();
            let _ = reply_tx.send(res);
        });
        let worker_job = Job {
            bc: mode,
            src: src.clone(),
            out_len,
            reply: mpsc::sync_channel(1).0,
        };
        worker_tx.send(worker_job).unwrap();
        drop(worker_tx);
        recv_handle.join().unwrap();
        let gpu_outcome: Option<Result<Vec<u8>, &'static str>> = match reply_rx.recv() {
            Ok(Ok(out)) => Some(Ok(out)),
            Ok(Err(_)) | Err(_) => None,
        };
        assert!(gpu_outcome.is_none());
        assert_eq!(src, src_snapshot);

        let mut cpu_fallback = vec![0u8; out_len];
        victory::decode(mode, &src, &mut cpu_fallback, victory::Tables::builtin())
            .expect("CPU fallback decode must succeed after GPU batch failure");
        assert_eq!(cpu_fallback, cpu_expected);
    }

    #[test]
    fn unavailable_worker_falls_back_to_cpu_without_panicking() {
        let (mode, src, cpu_expected) = first_valid_vector();
        let src_snapshot = src.clone();
        let out_len = cpu_expected.len();

        assert!(submit_with_sender(None, mode, &src, out_len).is_none());

        let (dead_tx, dead_rx) = mpsc::channel();
        drop(dead_rx);
        assert!(submit_with_sender(Some(&dead_tx), mode, &src, out_len).is_none());

        assert_eq!(src, src_snapshot);
        let mut cpu_fallback = vec![0u8; out_len];
        victory::decode(mode, &src, &mut cpu_fallback, victory::Tables::builtin())
            .expect("CPU fallback decode must succeed when worker is unavailable");
        assert_eq!(cpu_fallback, cpu_expected);
    }
}
