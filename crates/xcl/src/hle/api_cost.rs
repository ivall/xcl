//! Diagnostic (`XCL_ACTIVITY`, and `pso_async`'s pipeline line): the time the title's threads
//! spend in some of XCL's calls while it loads, summed over the threads.
//!
//! With `XCL_LOOK_STATS=2` also the calls a frame is made of on the thread that makes the title's
//! submits (its submits, presents and waits), and every call counted on that thread apart: what
//! it does in a frame besides submitting, added up.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub enum Call {
    PlacedTexture,
    PlacedBuffer,
    PhysicalPages,
    Memory,
    Pipeline,
    /// a bind's wait for a pipeline still being built (`pso_async::resolve_wait`)
    PipelineWait,
    /// the title's ExecuteCommandLists, whole (`submit_cost::done`; `XCL_LOOK_STATS=2` only, as
    /// the ones below)
    Submit,
    /// PresentX, or a present of the fallback's at a console method
    Present,
    /// a present's wait for the copy of the frame before
    PresentCopyWait,
    WaitFrameEvent,
    /// the title's blocking waits for a fence (SetEventOnCompletion without an event)
    FenceWait,
    /// copies back landed for the title's look at a fence (`sparse::settle_readbacks_within`)
    FenceSettle,
    /// draw recording on command lists (`DrawInstanced`, `DrawIndexedInstanced`, `DrawIndexedX`,
    /// `DispatchMesh`, non-dispatch `ExecuteIndirect`)
    DrawRecord,
    /// compute dispatch recording (`Dispatch`, `DispatchX`, dispatch `ExecuteIndirect`)
    DispatchRecord,
    /// root signatures/arguments, pipeline state, descriptor heaps, IA buffers, render targets
    BindState,
    /// device view creation and descriptor copies
    ViewDescriptor,
    /// GPU-view tile asks (`view_want::want`; sub-timer, may nest inside draw/dispatch/bind/view/copy)
    ViewWant,
    /// resource barriers, UAV barriers, and console cache/pipeline flushes
    ListBarrier,
    /// copies, fills, clears, resolves, and bottom-of-pipe writes
    ListCopy,
    /// command list Close/Reset and command allocator Reset
    ListLifecycle,
}

pub const CALLS: [(Call, &str); 20] = [
    (Call::PlacedTexture, "placed textures"),
    (Call::PlacedBuffer, "placed buffers"),
    (Call::PhysicalPages, "physical page calls"),
    (Call::Memory, "memory calls"),
    (Call::Pipeline, "pipelines"),
    (Call::PipelineWait, "waits for pipelines being built"),
    (Call::Submit, "ExecuteCommandLists"),
    (Call::Present, "presents"),
    (Call::PresentCopyWait, "waits of a present for the last frame's copy"),
    (Call::WaitFrameEvent, "WaitFrameEventX"),
    (Call::FenceWait, "blocking fence waits"),
    (Call::FenceSettle, "copies back settled at fence looks"),
    (Call::DrawRecord, "draw recording"),
    (Call::DispatchRecord, "dispatch recording"),
    (Call::BindState, "state and root bindings"),
    (Call::ViewDescriptor, "view and descriptor calls"),
    (Call::ViewWant, "GPU-view tile asks"),
    (Call::ListBarrier, "barriers and flushes"),
    (Call::ListCopy, "copies, clears, and memory ops"),
    (Call::ListLifecycle, "list close and reset"),
];

/// Calls that nest inside another timed call and must not be added into the submitting thread's
/// non-overlapping completed-call total.
pub const fn is_sub_call(call: Call) -> bool {
    matches!(call, Call::PresentCopyWait | Call::PipelineWait | Call::ViewWant)
}

/// Coherent single-writer `(count, ns)` counter pair protected by a seqlock generation
/// (odd while updating, even when quiescent). Safe for `ON_SUBMITTER` because `SUBMITTER`
/// latches at most one thread ID via `compare_exchange(0, tid)` for the process lifetime.
struct CounterPair {
    seq: AtomicU64,
    count: AtomicU64,
    ns: AtomicU64,
}

impl CounterPair {
    const fn new() -> Self {
        Self {
            seq: AtomicU64::new(0),
            count: AtomicU64::new(0),
            ns: AtomicU64::new(0),
        }
    }

    fn add_single_writer(&self, ns: u64) {
        let g = self.seq.load(Ordering::Relaxed);
        self.seq.store(g.wrapping_add(1), Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Release);
        self.count.store(self.count.load(Ordering::Relaxed).wrapping_add(1), Ordering::Relaxed);
        self.ns.store(self.ns.load(Ordering::Relaxed).saturating_add(ns), Ordering::Relaxed);
        self.seq.store(g.wrapping_add(2), Ordering::Release);
    }

    fn load_coherent(&self) -> (u64, u64) {
        loop {
            let g1 = self.seq.load(Ordering::Acquire);
            if g1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let count = self.count.load(Ordering::Relaxed);
            let ns = self.ns.load(Ordering::Relaxed);
            std::sync::atomic::fence(Ordering::Acquire);
            let g2 = self.seq.load(Ordering::Relaxed);
            if g1 == g2 {
                return (count, ns / 1000);
            }
            std::hint::spin_loop();
        }
    }
}

/// calls, nanoseconds (sub-µs calls accumulate without per-call truncation; exposed as µs)
static COST: [[AtomicU64; 2]; CALLS.len()] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; CALLS.len()];
/// the same on the submitting thread (`SUBMITTER`), read coherently via `CounterPair`'s seqlock
static ON_SUBMITTER: [CounterPair; CALLS.len()] = [const { CounterPair::new() }; CALLS.len()];
/// the thread that makes the title's submits (0: none seen, or `XCL_LOOK_STATS` is not 2)
static SUBMITTER: AtomicU32 = AtomicU32::new(0);

/// Counts the call it stands in when it goes.
pub struct Timer {
    pub call: Call,
    start: Instant,
}

pub fn time(call: Call) -> Timer {
    Timer { call, start: Instant::now() }
}

/// Times `call` only when `on` is true (zero clock, TLS, or atomic overhead when false).
#[inline]
pub fn time_if(on: bool, call: Call) -> Option<Timer> {
    on.then(|| time(call))
}

impl Drop for Timer {
    fn drop(&mut self) {
        add(self.call, self.start.elapsed());
    }
}

fn this_thread() -> u32 {
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

/// A call that took `d`.
pub fn add(call: Call, d: Duration) {
    let ns = d.as_nanos().min(u64::MAX as u128) as u64;
    let idx = call as usize;
    let c = &COST[idx];
    c[0].fetch_add(1, Ordering::Relaxed);
    c[1].fetch_add(ns, Ordering::Relaxed);
    let s = SUBMITTER.load(Ordering::Acquire);
    if s != 0 && s == this_thread() {
        ON_SUBMITTER[idx].add_single_writer(ns);
    }
}

/// The calling thread made one of the title's submits: the first thread to is the submitting
/// thread from then on (a retail title's world made every submit and present on one thread).
/// Called with `XCL_LOOK_STATS=2` only.
pub fn note_submitter() {
    let _ = SUBMITTER.compare_exchange(0, this_thread(), Ordering::AcqRel, Ordering::Acquire);
}

/// (calls, microseconds) so far, in the order of `CALLS`.
pub fn totals() -> [(u64, u64); CALLS.len()] {
    std::array::from_fn(|k| (COST[k][0].load(Ordering::Relaxed), COST[k][1].load(Ordering::Relaxed) / 1000))
}

/// Coherent `(calls, microseconds)` snapshot of the submitting thread, in the order of `CALLS`.
pub fn submitter_totals() -> [(u64, u64); CALLS.len()] {
    std::array::from_fn(|k| ON_SUBMITTER[k].load_coherent())
}

/// Formats per-category `(calls, ms)` deltas across all threads since `before`, retaining any
/// category with a nonzero count **or** duration delta so multi-writer snapshots never drop
/// completed duration when count and duration are observed on separate ticks.
pub fn all_threads_summary_parts(now: &[(u64, u64)], before: &[(u64, u64)]) -> Vec<String> {
    CALLS
        .iter()
        .zip(now.iter().zip(before))
        .filter_map(|(c, (n, o))| {
            let dc = n.0.saturating_sub(o.0);
            let dus = n.1.saturating_sub(o.1);
            if dc == 0 && dus == 0 {
                return None;
            }
            Some(format!("{dc} {} in {} ms", c.1, dus / 1000))
        })
        .collect()
}

/// Formats durations of submitting-thread calls **completed** since `before` alongside the
/// sample's elapsed `window_ms`. Summing only top-level calls avoids double-counting nested
/// sub-timers (`is_sub_call`). Because a call's full duration is published when it completes, a
/// call that started before the window and finished inside it can make the completed total exceed
/// `window_ms`; accordingly no in-window "untimed" remainder is claimed. Any nonzero duration
/// delta is accounted even if its count delta is zero.
pub fn submitter_summary_line(now: &[(u64, u64)], before: &[(u64, u64)], window_ms: u64) -> Option<String> {
    let mut parts = Vec::new();
    let mut completed_us = 0u64;
    for (c, (n, o)) in CALLS.iter().zip(now.iter().zip(before)) {
        let dc = n.0.saturating_sub(o.0);
        let dus = n.1.saturating_sub(o.1);
        if dc == 0 && dus == 0 {
            continue;
        }
        if !is_sub_call(c.0) {
            completed_us = completed_us.saturating_add(dus);
        }
        parts.push(format!("{dc} {} in {} ms", c.1, dus / 1000));
    }
    if parts.is_empty() {
        return None;
    }
    let completed_ms = completed_us / 1000;
    Some(format!("{}; completed {completed_ms} ms (window {window_ms} ms)", parts.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_call_is_counted_with_its_time() {
        // (in the order of the enum: a name stands for its own count)
        assert!(CALLS.iter().enumerate().all(|(k, c)| c.0 as usize == k));
        let before = totals()[Call::Memory as usize];
        {
            let mut t = time(Call::Pipeline);
            t.call = Call::Memory;
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let after = totals()[Call::Memory as usize];
        assert!(after.0 > before.0 && after.1 >= before.1 + 2000);
    }

    #[test]
    fn the_submitting_thread_has_its_own_count() {
        // (no other test makes a submitter: this thread is it)
        note_submitter();
        let me = this_thread();
        assert_eq!(SUBMITTER.load(Ordering::Relaxed), me);
        std::thread::spawn(note_submitter).join().unwrap();
        assert_eq!(SUBMITTER.load(Ordering::Relaxed), me, "the first thread stays the submitting one");
        let k = Call::FenceSettle as usize;
        let (all, mine) = (totals()[k], submitter_totals()[k]);
        add(Call::FenceSettle, Duration::from_micros(300));
        std::thread::spawn(|| add(Call::FenceSettle, Duration::from_micros(50))).join().unwrap();
        let (all2, mine2) = (totals()[k], submitter_totals()[k]);
        assert_eq!((all2.0 - all.0, all2.1 - all.1), (2, 350));
        assert_eq!((mine2.0 - mine.0, mine2.1 - mine.1), (1, 300));
    }

    #[test]
    fn sub_microsecond_calls_accumulate_without_truncation() {
        let k = Call::ListBarrier as usize;
        let before = totals()[k];
        for _ in 0..1000 {
            add(Call::ListBarrier, Duration::from_nanos(500));
        }
        let after = totals()[k];
        assert_eq!(after.0 - before.0, 1000);
        assert_eq!(after.1 - before.1, 500);
    }

    #[test]
    fn time_if_disabled_adds_zero_counts_and_zero_time() {
        let k = Call::ListLifecycle as usize;
        let before_all = totals()[k];
        let before_sub = submitter_totals()[k];
        {
            let guard = time_if(false, Call::ListLifecycle);
            assert!(guard.is_none());
        }
        assert_eq!(totals()[k], before_all);
        assert_eq!(submitter_totals()[k], before_sub);
    }

    #[test]
    fn submitter_summary_excludes_sub_calls_and_reports_completed_window() {
        let before = [(0u64, 0u64); CALLS.len()];
        assert_eq!(submitter_summary_line(&before, &before, 2000), None);

        let mut now = before;
        // Top-level calls: 100 + 250 + 50 = 400 ms completed
        now[Call::Submit as usize] = (10, 100_000);
        now[Call::DrawRecord as usize] = (500, 250_000);
        now[Call::BindState as usize] = (2000, 50_000);
        // Nested sub-calls: displayed, but excluded from `completed`
        now[Call::PipelineWait as usize] = (3, 40_000);
        now[Call::PresentCopyWait as usize] = (2, 30_000);
        now[Call::ViewWant as usize] = (400, 90_000);

        let line = submitter_summary_line(&now, &before, 2000).unwrap();
        assert!(line.contains("10 ExecuteCommandLists in 100 ms"), "{line}");
        assert!(line.contains("500 draw recording in 250 ms"), "{line}");
        assert!(line.contains("2000 state and root bindings in 50 ms"), "{line}");
        assert!(line.contains("3 waits for pipelines being built in 40 ms"), "{line}");
        assert!(line.contains("2 waits of a present for the last frame's copy in 30 ms"), "{line}");
        assert!(line.contains("400 GPU-view tile asks in 90 ms"), "{line}");
        assert!(line.ends_with("; completed 400 ms (window 2000 ms)"), "{line}");
        assert!(!line.contains("untimed"), "{line}");
        assert!(is_sub_call(Call::PresentCopyWait) && is_sub_call(Call::PipelineWait) && is_sub_call(Call::ViewWant));
        assert!(!is_sub_call(Call::Submit) && !is_sub_call(Call::DrawRecord) && !is_sub_call(Call::FenceWait));
    }

    #[test]
    fn summary_retains_nonzero_duration_delta_when_count_delta_is_zero() {
        // Deterministic torn-snapshot sequence `(0,0) -> (1,0) -> (1,400_000)` on a top-level
        // call and `(0,0) -> (1,0) -> (1,50_000)` on a nested sub-call: the second tick must
        // retain the 400 ms duration in both `all_threads_summary_parts` and `submitter_summary_line`,
        // without double-counting the nested sub-call into `completed`.
        let snap0 = [(0u64, 0u64); CALLS.len()];
        let mut snap1 = snap0;
        snap1[Call::FenceWait as usize] = (1, 0);
        snap1[Call::ViewWant as usize] = (1, 0);
        let mut snap2 = snap1;
        snap2[Call::FenceWait as usize] = (1, 400_000);
        snap2[Call::ViewWant as usize] = (1, 50_000);

        let sub1 = submitter_summary_line(&snap1, &snap0, 2000).unwrap();
        assert!(sub1.contains("1 blocking fence waits in 0 ms"), "{sub1}");
        assert!(sub1.contains("1 GPU-view tile asks in 0 ms"), "{sub1}");
        assert!(sub1.ends_with("; completed 0 ms (window 2000 ms)"), "{sub1}");

        let sub2 = submitter_summary_line(&snap2, &snap1, 2000).unwrap();
        assert!(sub2.contains("0 blocking fence waits in 400 ms"), "{sub2}");
        assert!(sub2.contains("0 GPU-view tile asks in 50 ms"), "{sub2}");
        assert!(sub2.ends_with("; completed 400 ms (window 2000 ms)"), "{sub2}");

        let all2 = all_threads_summary_parts(&snap2, &snap1);
        assert!(all2.iter().any(|s| s == "0 blocking fence waits in 400 ms"), "{all2:?}");
        assert!(all2.iter().any(|s| s == "0 GPU-view tile asks in 50 ms"), "{all2:?}");
    }

    #[test]
    fn submitter_counter_pair_snapshots_are_coherent_under_concurrent_writes() {
        use std::sync::Arc;
        const N: u64 = 50_000;
        const STEP_NS: u64 = 10_000_000; // 10_000 us per call
        let pair = Arc::new(CounterPair::new());
        let writer_pair = Arc::clone(&pair);
        let writer = std::thread::spawn(move || {
            for _ in 0..N {
                writer_pair.add_single_writer(STEP_NS);
            }
        });
        let mut samples = 0u64;
        loop {
            let (count, us) = pair.load_coherent();
            assert_eq!(us, count * (STEP_NS / 1000), "torn snapshot observed: count={count}, us={us}");
            samples += 1;
            if count == N {
                break;
            }
        }
        writer.join().unwrap();
        assert!(samples >= 1);
        assert_eq!(pair.load_coherent(), (N, N * (STEP_NS / 1000)));
    }

    #[test]
    fn submitter_summary_describes_straddling_completed_call_without_untimed_claim() {
        // Synthetic 2.5 s blocking fence wait starting at t=0.0 s and completing at t=2.5 s
        // across two 2.0 s activity ticks: tick 1 sees no completed calls; tick 2 reports the
        // completed 2500 ms call alongside the 2000 ms window without claiming "0 ms untimed".
        let snap0 = [(0u64, 0u64); CALLS.len()];
        let tick1_snap = snap0;
        assert_eq!(submitter_summary_line(&tick1_snap, &snap0, 2000), None);

        let mut tick2_snap = tick1_snap;
        tick2_snap[Call::FenceWait as usize] = (1, 2_500_000);
        let tick2 = submitter_summary_line(&tick2_snap, &tick1_snap, 2000).unwrap();
        assert_eq!(tick2, "1 blocking fence waits in 2500 ms; completed 2500 ms (window 2000 ms)");
        assert!(!tick2.contains("untimed"), "{tick2}");
    }
}
