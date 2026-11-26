//! Opt-in attribution of time spent in coarse runtime operations.
//!
//! A disabled span is one relaxed atomic load. In particular, it does not read the clock, allocate,
//! lock, or update a counter. Callers should keep spans coarse enough that enabled tracing remains
//! useful diagnostics rather than becoming the workload.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub const HISTOGRAM_BUCKETS: usize = 64;
pub const SLOW_SPAN_CAPACITY: usize = 128;
pub const SLOW_SPAN_NS: u64 = 50_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum StallKind {
    ShaderTranslation,
    DxcQueue,
    DxcCompile,
    PipelineWait,
    DriverCompileGate,
    DriverCompile,
    SubmitPreparation,
    MemorySynchronization,
    Residency,
    CacheIo,
    LockContention,
    GpuWait,
    Present,
    DirectStorage,
}

impl StallKind {
    pub const ALL: [Self; 14] = [
        Self::ShaderTranslation,
        Self::DxcQueue,
        Self::DxcCompile,
        Self::PipelineWait,
        Self::DriverCompileGate,
        Self::DriverCompile,
        Self::SubmitPreparation,
        Self::MemorySynchronization,
        Self::Residency,
        Self::CacheIo,
        Self::LockContention,
        Self::GpuWait,
        Self::Present,
        Self::DirectStorage,
    ];

    const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AggregateSnapshot {
    pub kind: StallKind,
    pub count: u64,
    pub total_ns: u64,
    pub max_ns: u64,
    /// Duration counts in base-2 nanosecond buckets. Bucket zero also contains zero-duration spans.
    pub histogram: [u64; HISTOGRAM_BUCKETS],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SlowSpan {
    pub sequence: u64,
    pub kind: StallKind,
    pub started_ns: u64,
    pub duration_ns: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Snapshot {
    pub aggregates: Vec<AggregateSnapshot>,
    pub slow_spans: Vec<SlowSpan>,
}

struct Aggregate {
    count: AtomicU64,
    total_ns: AtomicU64,
    max_ns: AtomicU64,
    histogram: [AtomicU64; HISTOGRAM_BUCKETS],
}

impl Aggregate {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
            max_ns: AtomicU64::new(0),
            histogram: [const { AtomicU64::new(0) }; HISTOGRAM_BUCKETS],
        }
    }

    fn record(&self, duration_ns: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_ns.fetch_add(duration_ns, Ordering::Relaxed);
        self.max_ns.fetch_max(duration_ns, Ordering::Relaxed);
        self.histogram[histogram_bucket(duration_ns)].fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self, kind: StallKind) -> AggregateSnapshot {
        AggregateSnapshot {
            kind,
            count: self.count.load(Ordering::Relaxed),
            total_ns: self.total_ns.load(Ordering::Relaxed),
            max_ns: self.max_ns.load(Ordering::Relaxed),
            histogram: std::array::from_fn(|i| self.histogram[i].load(Ordering::Relaxed)),
        }
    }
}

const fn histogram_bucket(duration_ns: u64) -> usize {
    if duration_ns == 0 { 0 } else { (u64::BITS - 1 - duration_ns.leading_zeros()) as usize }
}

struct SlowSpans {
    next_sequence: u64,
    spans: VecDeque<SlowSpan>,
}

struct Recorder {
    enabled: AtomicBool,
    aggregates: [Aggregate; StallKind::ALL.len()],
    slow_spans: Mutex<SlowSpans>,
}

trait Clock: Sync {
    fn now_ns(&self) -> u64;
}

impl Recorder {
    const fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            aggregates: [const { Aggregate::new() }; StallKind::ALL.len()],
            slow_spans: Mutex::new(SlowSpans { next_sequence: 0, spans: VecDeque::new() }),
        }
    }

    fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    #[inline]
    fn start_with<'a>(&'a self, kind: StallKind, clock: &'a dyn Clock) -> Option<Span<'a>> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }
        Some(Span { recorder: self, kind, started_ns: clock.now_ns(), clock, finished: false })
    }

    fn record(&self, kind: StallKind, started_ns: u64, duration_ns: u64) {
        self.aggregates[kind.index()].record(duration_ns);
        if duration_ns < SLOW_SPAN_NS {
            return;
        }
        let mut slow = self.slow_spans.lock().unwrap_or_else(|e| e.into_inner());
        let span = SlowSpan {
            sequence: slow.next_sequence,
            kind,
            started_ns,
            duration_ns,
        };
        slow.next_sequence = slow.next_sequence.wrapping_add(1);
        if slow.spans.len() == SLOW_SPAN_CAPACITY {
            slow.spans.pop_front();
        }
        slow.spans.push_back(span);
    }

    fn snapshot(&self) -> Snapshot {
        let aggregates = StallKind::ALL.iter().map(|&kind| self.aggregates[kind.index()].snapshot(kind)).collect();
        let slow_spans = self.slow_spans.lock().unwrap_or_else(|e| e.into_inner()).spans.iter().copied().collect();
        Snapshot { aggregates, slow_spans }
    }
}

pub struct Span<'a> {
    recorder: &'a Recorder,
    kind: StallKind,
    started_ns: u64,
    clock: &'a dyn Clock,
    finished: bool,
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.finished = true;
            self.recorder.record(self.kind, self.started_ns, self.clock.now_ns().saturating_sub(self.started_ns));
        }
    }
}

static RECORDER: Recorder = Recorder::new();
static CLOCK: SystemClock = SystemClock;

pub fn enable() {
    RECORDER.enable();
}

#[inline]
pub fn span(kind: StallKind) -> Option<Span<'static>> {
    RECORDER.start_with(kind, &CLOCK)
}

pub fn snapshot() -> Option<Snapshot> {
    RECORDER.enabled.load(Ordering::Relaxed).then(|| RECORDER.snapshot())
}

struct SystemClock;

impl Clock for SystemClock {
    fn now_ns(&self) -> u64 {
        static ORIGIN: OnceLock<Instant> = OnceLock::new();
        ORIGIN.get_or_init(Instant::now).elapsed().as_nanos().min(u64::MAX as u128) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize};

    struct TestClock {
        now: AtomicU64,
        calls: AtomicUsize,
    }

    impl TestClock {
        fn new(now: u64) -> Self {
            Self { now: AtomicU64::new(now), calls: AtomicUsize::new(0) }
        }
    }

    impl Clock for TestClock {
        fn now_ns(&self) -> u64 {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.now.load(Ordering::Relaxed)
        }
    }

    #[test]
    fn disabled_span_does_not_read_the_clock_or_record() {
        let recorder = Recorder::new();
        let clock = TestClock::new(10);
        assert!(recorder.start_with(StallKind::DxcCompile, &clock).is_none());
        assert_eq!(clock.calls.load(Ordering::Relaxed), 0);
        assert_eq!(recorder.snapshot().aggregates[StallKind::DxcCompile.index()].count, 0);
    }

    #[test]
    fn a_span_updates_its_aggregate_and_log2_bucket() {
        let recorder = Recorder::new();
        recorder.enable();
        let clock = TestClock::new(10);
        let span = recorder.start_with(StallKind::PipelineWait, &clock).unwrap();
        clock.now.store(10 + 1024, Ordering::Relaxed);
        drop(span);

        let aggregate = &recorder.snapshot().aggregates[StallKind::PipelineWait.index()];
        assert_eq!((aggregate.count, aggregate.total_ns, aggregate.max_ns), (1, 1024, 1024));
        assert_eq!(aggregate.histogram[10], 1);
        assert_eq!(aggregate.histogram.iter().sum::<u64>(), 1);
        assert_eq!(clock.calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn only_slow_spans_enter_the_bounded_ring() {
        let recorder = Recorder::new();
        recorder.enable();
        let clock = TestClock::new(0);
        for i in 0..SLOW_SPAN_CAPACITY + 3 {
            clock.now.store(i as u64, Ordering::Relaxed);
            let span = recorder.start_with(StallKind::GpuWait, &clock).unwrap();
            clock.now.store(i as u64 + SLOW_SPAN_NS, Ordering::Relaxed);
            drop(span);
        }
        clock.now.store(1_000_000_000, Ordering::Relaxed);
        let fast = recorder.start_with(StallKind::Present, &clock).unwrap();
        clock.now.store(1_000_000_001, Ordering::Relaxed);
        drop(fast);

        let snapshot = recorder.snapshot();
        assert_eq!(snapshot.slow_spans.len(), SLOW_SPAN_CAPACITY);
        assert_eq!(snapshot.slow_spans[0].sequence, 3);
        assert!(snapshot.slow_spans.iter().all(|span| span.kind == StallKind::GpuWait));
        assert_eq!(snapshot.aggregates[StallKind::Present.index()].count, 1);
    }
}
