mod support;

use std::collections::BTreeMap;
use std::hint::black_box;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;
use support::{BatchSize, BenchmarkId, Criterion, Throughput};
use xcl::config::Config;
use xcl::manifest;
use xcl::util;
use xcl::vfs::Vfs;

const MANIFEST: &str = r#"<Game configVersion="1">
  <Identity Name="Synthetic.Benchmark" Publisher="CN=Benchmark" Version="1.0.0.0" />
  <PersistentLocalStorage><SizeMB>4096</SizeMB></PersistentLocalStorage>
  <VirtualMachine>
    <XboxAnacondaTitleMemory>Advanced</XboxAnacondaTitleMemory>
    <XboxLockhartTitleMemory>Standard</XboxLockhartTitleMemory>
  </VirtualMachine>
  <TitleId>12345678</TitleId>
  <Resources>
    <Resource Language="en-US"/><Resource Language="fr-FR"/>
    <Resource Language="de-DE"/><Resource Language="ja-JP"/>
  </Resources>
</Game>"#;

const CONFIG: &str = r#"
device_type = "anaconda"
gamertag = "Player"
user_id = 281474976710655
title_memory = "auto"
pls_mb = 0
uma_mb = 0
log_level = "info"
graphics = "d3d12"
d3d12_adapter = "auto"
present = "auto"
raytracing = "auto"
window_width = 1280
window_height = 720
keyboard_mouse = true
mouse_sensitivity = 1.0

[mounts]
"GAME:" = "{game}"
"G:" = "{game}"
"S:" = "{data}\\pls"

[aliases]
"api-ms-win-*" = "@system"
"kernelx.dll" = "kernelx.dll"

[stub_returns]
"example.dll!Example" = "S_OK"
"#;

const ALLOCATION_JOBS: usize = 128;

struct AllocationJob {
    owner: Arc<[u8; 64]>,
    label: String,
    payload: Vec<u8>,
}

fn allocation_job_batch(seed: usize) -> usize {
    let owner = Arc::new([seed as u8; 64]);
    let mut jobs = Vec::with_capacity(ALLOCATION_JOBS);
    for index in 0..ALLOCATION_JOBS {
        let mut label = String::with_capacity(48);
        label.push_str("pipeline-job-");
        label.extend(std::iter::repeat_n((b'a' + (index % 26) as u8) as char, 16 + index % 16));
        jobs.push(Box::new(AllocationJob {
            owner: owner.clone(),
            label,
            payload: vec![(seed ^ index) as u8; 96 + index % 64],
        }));
    }
    let checksum = jobs.iter().fold(0usize, |checksum, job| {
        checksum ^ job.owner[0] as usize ^ job.label.len() ^ job.payload.len()
    });
    black_box(&jobs);
    checksum
}

#[repr(align(64))]
struct AllocationResult(AtomicUsize);

struct AllocationWorkers {
    start: Arc<Barrier>,
    done: Arc<Barrier>,
    stop: Arc<AtomicBool>,
    epoch: Arc<AtomicUsize>,
    results: Arc<Vec<AllocationResult>>,
    threads: Vec<JoinHandle<()>>,
}

impl AllocationWorkers {
    fn new(count: usize) -> Self {
        let start = Arc::new(Barrier::new(count + 1));
        let done = Arc::new(Barrier::new(count + 1));
        let stop = Arc::new(AtomicBool::new(false));
        let epoch = Arc::new(AtomicUsize::new(0));
        let results: Arc<Vec<AllocationResult>> =
            Arc::new((0..count).map(|_| AllocationResult(AtomicUsize::new(0))).collect());
        let threads = (0..count).map(|worker| {
            let start = start.clone();
            let done = done.clone();
            let stop = stop.clone();
            let epoch = epoch.clone();
            let results = results.clone();
            std::thread::Builder::new().name(format!("alloc-bench-{worker}")).spawn(move || loop {
                start.wait();
                if stop.load(Ordering::Acquire) { break }
                let checksum = allocation_job_batch(epoch.load(Ordering::Acquire).wrapping_add(worker));
                results[worker].0.store(checksum, Ordering::Release);
                done.wait();
            }).expect("allocator benchmark worker")
        }).collect();
        Self { start, done, stop, epoch, results, threads }
    }

    fn run(&self) -> usize {
        self.epoch.fetch_add(1, Ordering::Release);
        self.start.wait();
        self.done.wait();
        self.results.iter().fold(0, |checksum, result| checksum ^ result.0.load(Ordering::Acquire))
    }
}

impl Drop for AllocationWorkers {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.start.wait();
        for thread in self.threads.drain(..) {
            thread.join().expect("allocator benchmark worker panicked");
        }
    }
}

fn allocator_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime/allocator");
    group.throughput(Throughput::Elements(ALLOCATION_JOBS as u64));
    group.bench_function("job_mix_single", |b| {
        let mut seed = 0usize;
        b.iter(|| {
            seed = seed.wrapping_add(1);
            allocation_job_batch(black_box(seed))
        });
    });

    let workers = std::thread::available_parallelism().map_or(4, |count| count.get())
        .saturating_sub(1).clamp(2, 8);
    let pool = AllocationWorkers::new(workers);
    group.throughput(Throughput::Elements((ALLOCATION_JOBS * workers) as u64));
    group.bench_function(&format!("job_mix_parallel_{workers}"), |b| b.iter(|| pool.run()));
    group.finish();
}

fn utility_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime/util");
    for (name, pattern, value) in [
        ("glob_exact", "kernelx.dll", "kernelx.dll"),
        (
            "glob_prefix",
            "api-ms-win-*",
            "api-ms-win-core-file-l2-1-0.dll",
        ),
        ("glob_bounded", "xgame*.dll", "xgameruntime.dll"),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| util::glob_match(black_box(pattern), black_box(value)))
        });
    }
    for value in [
        "kernelx",
        "D:\\Game\\D3D12_XS.DLL",
        "api-ms-win-core-file-l2-1-0",
    ] {
        group.bench_with_input(
            BenchmarkId::new("dll_key", value.len()),
            value,
            |b, value| {
                b.iter(|| util::dll_key(black_box(value)));
            },
        );
    }
    for value in ["S_OK", "E_NOTIMPL", "0x80004005", "42"] {
        group.bench_with_input(
            BenchmarkId::new("parse_return", value),
            value,
            |b, value| {
                b.iter(|| util::parse_ret(black_box(value)));
            },
        );
    }
    for value in [
        "short",
        "D:\\Synthetic Game\\Content\\localization\\en-US\\strings.bin",
    ] {
        group.throughput(Throughput::Bytes(value.len() as u64));
        group.bench_with_input(BenchmarkId::new("utf16", value.len()), value, |b, value| {
            b.iter(|| util::wide(black_box(value)));
        });
    }
    group.finish();
}

fn manifest_and_config_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime/config");
    group.throughput(Throughput::Bytes(MANIFEST.len() as u64));
    group.bench_function("manifest_parse", |b| {
        b.iter(|| manifest::parse(black_box(MANIFEST)))
    });

    group.throughput(Throughput::Bytes(CONFIG.len() as u64));
    group.bench_function("toml_parse", |b| {
        b.iter(|| toml::from_str::<Config>(black_box(CONFIG)).unwrap());
    });

    let parsed_manifest = manifest::parse(MANIFEST);
    group.throughput(Throughput::Elements(1));
    group.bench_function("resolve_defaults", |b| {
        b.iter_batched(
            Config::default,
            |mut config| black_box(config.resolve(black_box(Some(&parsed_manifest)))),
            BatchSize::SmallInput,
        );
    });

    let mut mounts = BTreeMap::new();
    mounts.insert("GAME:".to_owned(), "{game}".to_owned());
    mounts.insert("S:".to_owned(), "{data}\\pls".to_owned());
    let mut resolved = Config {
        mounts,
        ..Config::default()
    };
    resolved.resolve(Some(&parsed_manifest));
    group.throughput(Throughput::Elements(1));
    group.bench_function("resolved_memory_queries", |b| {
        b.iter(|| {
            black_box(resolved.title_memory_budget_mb());
            black_box(resolved.pls_drive());
            black_box(resolved.raytracing_tier_override());
        });
    });
    group.finish();
}

fn vfs_benches(c: &mut Criterion) {
    let vfs = Vfs::new(
        Path::new(r"C:\bench\game"),
        Path::new(r"C:\bench\data"),
        vec![
            ("GAME:".to_owned(), r"C:\bench\game".to_owned()),
            ("G:".to_owned(), r"C:\bench\game".to_owned()),
            ("S:".to_owned(), r"C:\bench\data\pls".to_owned()),
        ],
    );
    let full = |path: &str| Some(path.replace('/', "\\"));
    let cases = [
        ("mounted_game", r"GAME:\data\textures.bin"),
        ("mounted_drive", r"G:\data\textures.bin"),
        ("embedded_mount", r"C:\cwd\GAME:\data\textures.bin"),
        ("direct_game", r"C:\bench\game\data\textures.bin"),
        ("host_path", r"C:\Windows\System32\kernel32.dll"),
        ("unc", r"\\server\share\file.bin"),
    ];

    let mut group = c.benchmark_group("runtime/vfs");
    for (name, path) in cases {
        group.throughput(Throughput::Bytes(path.len() as u64));
        group.bench_with_input(BenchmarkId::new("classify", name), path, |b, path| {
            b.iter(|| vfs.classify_with(black_box(path), black_box(full)));
        });
    }
    group.throughput(Throughput::Bytes(
        cases.iter().map(|(_, path)| path.len() as u64).sum(),
    ));
    group.bench_function("classify_mixed_batch", |b| {
        b.iter(|| {
            for (_, path) in black_box(&cases) {
                black_box(vfs.classify_with(path, full));
            }
        });
    });
    for rel in ["", "data", r"data\textures\terrain.bin"] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("overlay_path", rel.len()),
            rel,
            |b, rel| {
                b.iter(|| vfs.overlay_path(black_box(rel)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("original_path", rel.len()),
            rel,
            |b, rel| {
                b.iter(|| vfs.original_path(black_box(rel)));
            },
        );
    }
    group.finish();
}

fn main() {
    support::run(&[utility_benches, manifest_and_config_benches, vfs_benches, allocator_benches]);
}
