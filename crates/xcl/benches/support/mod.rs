#![allow(dead_code)]

use std::fmt::Display;
use std::hint::black_box;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct Config {
    test: bool,
    filter: Option<String>,
}

pub struct Criterion {
    config: Config,
    ran: usize,
}

pub struct BenchmarkGroup<'a> {
    criterion: &'a mut Criterion,
    name: String,
    throughput: Option<Throughput>,
    sample_size: usize,
    measurement_time: Duration,
}

pub struct Bencher {
    name: String,
    test: bool,
    throughput: Option<Throughput>,
    sample_size: usize,
    measurement_time: Duration,
}

#[derive(Clone, Copy)]
pub enum Throughput {
    Bytes(u64),
    Elements(u64),
}

pub enum BatchSize {
    SmallInput,
}

pub struct BenchmarkId(String);

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn test_helpers() {
    assert_eq!(median(&[1.0, 3.0]), 2.0);
    assert_eq!(median(&[1.0, 4.0, 9.0]), 4.0);
}

impl BenchmarkId {
    pub fn new(name: impl Display, parameter: impl Display) -> Self {
        Self(format!("{name}/{parameter}"))
    }

    pub fn from_parameter(parameter: impl Display) -> Self {
        Self(parameter.to_string())
    }
}

impl Display for BenchmarkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Criterion {
    fn from_args() -> Self {
        let mut test = false;
        let mut filter = None;
        for argument in std::env::args().skip(1) {
            if argument == "--test" {
                test = true;
            } else if !argument.starts_with('-') && filter.is_none() {
                filter = Some(argument);
            }
        }
        Self { config: Config { test, filter }, ran: 0 }
    }

    pub fn benchmark_group<'a>(&'a mut self, name: &str) -> BenchmarkGroup<'a> {
        BenchmarkGroup {
            criterion: self,
            name: name.to_owned(),
            throughput: None,
            sample_size: 30,
            measurement_time: Duration::from_secs(2),
        }
    }

    pub fn bench_function(&mut self, name: &str, routine: impl FnMut(&mut Bencher)) {
        self.run_one(name.to_owned(), None, 30, Duration::from_secs(2), routine);
    }

    fn run_one(
        &mut self,
        name: String,
        throughput: Option<Throughput>,
        sample_size: usize,
        measurement_time: Duration,
        mut routine: impl FnMut(&mut Bencher),
    ) {
        if self.config.filter.as_ref().is_some_and(|filter| !name.contains(filter)) {
            return;
        }
        self.ran += 1;
        routine(&mut Bencher {
            name,
            test: self.config.test,
            throughput,
            sample_size: sample_size.max(3),
            measurement_time,
        });
    }
}

impl BenchmarkGroup<'_> {
    pub fn throughput(&mut self, throughput: Throughput) {
        self.throughput = Some(throughput);
    }

    pub fn sample_size(&mut self, sample_size: usize) {
        self.sample_size = sample_size;
    }

    pub fn measurement_time(&mut self, measurement_time: Duration) {
        self.measurement_time = measurement_time;
    }

    pub fn bench_function(&mut self, name: &str, routine: impl FnMut(&mut Bencher)) {
        let name = format!("{}/{}", self.name, name);
        self.criterion.run_one(
            name, self.throughput, self.sample_size, self.measurement_time, routine,
        );
    }

    pub fn bench_with_input<T: ?Sized>(
        &mut self,
        id: BenchmarkId,
        input: &T,
        mut routine: impl FnMut(&mut Bencher, &T),
    ) {
        let name = format!("{}/{}", self.name, id);
        self.criterion.run_one(
            name,
            self.throughput,
            self.sample_size,
            self.measurement_time,
            |bencher| routine(bencher, input),
        );
    }

    pub fn finish(self) {}
}

impl Bencher {
    pub fn iter<R>(&mut self, mut routine: impl FnMut() -> R) {
        if self.test {
            let _ = black_box(routine());
            println!("test {} ... ok", self.name);
            return;
        }
        self.measure_iterations(|iterations| {
            let started = Instant::now();
            for _ in 0..iterations {
                let _ = black_box(routine());
            }
            started.elapsed()
        });
    }

    pub fn iter_batched<I, R>(
        &mut self,
        mut setup: impl FnMut() -> I,
        mut routine: impl FnMut(I) -> R,
        _batch_size: BatchSize,
    ) {
        if self.test {
            let input = setup();
            let _ = black_box(routine(input));
            println!("test {} ... ok", self.name);
            return;
        }
        self.measure_iterations(|iterations| {
            let mut elapsed = Duration::ZERO;
            let mut remaining = iterations;
            while remaining != 0 {
                let batch = remaining.min(1_024) as usize;
                let inputs: Vec<_> = (0..batch).map(|_| setup()).collect();
                let started = Instant::now();
                for input in inputs {
                    let _ = black_box(routine(input));
                }
                elapsed += started.elapsed();
                remaining -= batch as u64;
            }
            elapsed
        });
    }

    fn measure_iterations(&self, mut run: impl FnMut(u64) -> Duration) {
        let warmup = self.measurement_time.div_f64(10.0).min(Duration::from_millis(100));
        let mut iterations = 1u64;
        let elapsed = loop {
            let elapsed = run(iterations);
            if elapsed >= warmup || iterations >= 1 << 40 {
                break elapsed;
            }
            let scale = (warmup.as_secs_f64() / elapsed.as_secs_f64().max(1e-9)).ceil() as u64;
            iterations = iterations.saturating_mul(scale.clamp(2, 10)).min(1 << 40);
        };

        let seconds_per_iteration = elapsed.as_secs_f64() / iterations as f64;
        let target_sample = self.measurement_time.as_secs_f64() / self.sample_size as f64;
        let sample_iterations = (target_sample / seconds_per_iteration.max(1e-12))
            .ceil()
            .clamp(1.0, (1u64 << 40) as f64) as u64;
        let mut samples = Vec::with_capacity(self.sample_size);
        for _ in 0..self.sample_size {
            samples.push(run(sample_iterations).as_secs_f64() / sample_iterations as f64);
        }
        samples.sort_unstable_by(f64::total_cmp);
        let median = median(&samples);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let minimum = samples[0];
        print!(
            "bench {}: median {:.1} ns/iter, mean {:.1}, min {:.1}",
            self.name,
            median * 1e9,
            mean * 1e9,
            minimum * 1e9,
        );
        if let Some(throughput) = self.throughput {
            let (amount, unit) = match throughput {
                Throughput::Bytes(amount) => (amount, "B/s"),
                Throughput::Elements(amount) => (amount, "elem/s"),
            };
            print!(", {:.1} {unit}", amount as f64 / median.max(1e-12));
        }
        println!();
    }
}

pub fn run(groups: &[fn(&mut Criterion)]) {
    let mut criterion = Criterion::from_args();
    if criterion.config.test {
        test_helpers();
    }
    for group in groups {
        group(&mut criterion);
    }
    if criterion.ran == 0 {
        eprintln!("no benchmark matched the requested filter");
        std::process::exit(2);
    }
}
