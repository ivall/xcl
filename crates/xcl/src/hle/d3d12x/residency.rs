//! Host-memory admission and `E_OUTOFMEMORY` back-off for texture upload staging (#3011, C5; #2688).
//!
//! Provenance (GPL-3.0 §5a): The `E_OUTOFMEMORY` 500 ms `RETRY_UPLOAD_AT` cooldown,
//! 32 ms-cached `GlobalMemoryStatusEx` (`ullAvailPageFile`) upload/growth reserve checks
//! (`512 MiB` upload floor, `1024 MiB` fresh-growth floor keeping pooled-buffer reuse open),
//! and thread-local `UPLOAD_DEFERRALS` observation are adapted for XCL from Vice-Tay
//! (`https://github.com/TaynanGT/Vice-Tay`, commit `57deba5c4c752b97a8286bdbd80b96517f0988fc`,
//! `crates/victory-gpu/src/residency.rs`, license GPL-3.0, upstream authors TaynanGT,
//! codec_d (codec-d), avatarmanz/VIctory-7900XTX and VIctory contributors; adapted for XCL
//! on 2026-11-27). Gated behind `XCL_UPLOAD_BACKOFF=1` (off by default).

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

use crate::util::E_OUTOFMEMORY;

const MIB: u64 = 1 << 20;
/// Cooldown after an `E_OUTOFMEMORY` allocation failure.
pub const UPLOAD_RETRY_MS: u64 = 500;
/// Interval between OS commit-headroom (`GlobalMemoryStatusEx`) refreshes.
pub const COMMIT_SAMPLE_MS: u64 = 32;
/// Protective commit floor (`ullAvailPageFile`) to admit texture upload passes (pooled reuse stays open above this).
pub const UPLOAD_RESERVE_BYTES: u64 = 512 * MIB;
/// Protective commit floor required after committing a brand-new upload staging buffer.
pub const GROWTH_RESERVE_BYTES: u64 = 1024 * MIB;

/// `XCL_UPLOAD_BACKOFF=1`: opt-in `E_OUTOFMEMORY` cooldown + commit-headroom upload admission (off by default).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_UPLOAD_BACKOFF").is_ok_and(|v| v.trim() == "1"))
}

pub(super) fn coarse_ms() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64
}

pub(super) fn query_avail_commit() -> Option<u64> {
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status.ullAvailPageFile)
}

/// Aggregate-only telemetry for `XCL_UPLOAD_BACKOFF=1` (zero game data).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub deferrals: u64,
    pub oom_cooldowns: u64,
    pub growth_denials: u64,
    pub free_trims: u64,
    pub free_trim_bytes: u64,
}

struct CommitSample {
    next_sample: u64,
    available: u64,
    /// Bytes reserved by in-flight staging allocations not yet reflected in `GlobalMemoryStatusEx`.
    pending_growth: u64,
}

impl CommitSample {
    fn refresh_locked(&mut self, now: u64, query: impl FnOnce() -> Option<u64>) -> u64 {
        if now >= self.next_sample {
            self.next_sample = now.saturating_add(COMMIT_SAMPLE_MS);
            if let Some(commit) = query() {
                self.available = commit.saturating_sub(self.pending_growth);
            }
        }
        self.available
    }
}

/// Holds an in-flight staging growth reservation until the OS allocation call finishes so a
/// concurrent 32 ms sample refresh cannot reset the debited headroom budget.
pub(super) struct GrowthReservation<'a> {
    res: Option<&'a Residency>,
    bytes: u64,
}

impl Drop for GrowthReservation<'_> {
    fn drop(&mut self) {
        if let Some(res) = self.res {
            let mut g = res.commit.lock().unwrap();
            g.pending_growth = g.pending_growth.saturating_sub(self.bytes);
        }
    }
}

/// Cooldown and cached OS commit-headroom state (isolated instances allow race-free unit/GPU tests).
pub(super) struct Residency {
    commit: Mutex<CommitSample>,
    retry_upload_at: AtomicU64,
    deferrals: AtomicU64,
    oom_cooldowns: AtomicU64,
    growth_denials: AtomicU64,
    free_trims: AtomicU64,
    free_trim_bytes: AtomicU64,
}

impl Residency {
    pub(super) const fn new() -> Self {
        Self {
            commit: Mutex::new(CommitSample { next_sample: 0, available: u64::MAX, pending_growth: 0 }),
            retry_upload_at: AtomicU64::new(0),
            deferrals: AtomicU64::new(0),
            oom_cooldowns: AtomicU64::new(0),
            growth_denials: AtomicU64::new(0),
            free_trims: AtomicU64::new(0),
            free_trim_bytes: AtomicU64::new(0),
        }
    }

    pub(super) fn record_allocation_error(&self, hr: i32, now: u64, active: bool) -> bool {
        if !active || hr != E_OUTOFMEMORY {
            return false;
        }
        self.retry_upload_at.fetch_max(now.saturating_add(UPLOAD_RETRY_MS), Ordering::AcqRel);
        let n = self.oom_cooldowns.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 || n.is_power_of_two() {
            crate::xlog_warn!(
                "D3D12.X: upload backoff: E_OUTOFMEMORY triggered {UPLOAD_RETRY_MS} ms upload cooldown ({n} so far)"
            );
        }
        true
    }

    pub(super) fn in_cooldown_at(&self, now: u64, active: bool) -> bool {
        active && now < self.retry_upload_at.load(Ordering::Acquire)
    }

    pub(super) fn retry_delay_at(&self, now: u64) -> Duration {
        let rem = self.retry_upload_at.load(Ordering::Acquire).saturating_sub(now);
        Duration::from_millis(if rem > 0 { rem } else { UPLOAD_RETRY_MS })
    }

    pub(super) fn sample_commit_with(&self, now: u64, query: impl FnOnce() -> Option<u64>) -> u64 {
        self.commit.lock().unwrap().refresh_locked(now, query)
    }

    pub(super) fn admit_upload_with(&self, now: u64, active: bool, query: impl FnOnce() -> Option<u64>) -> bool {
        if !active {
            return true;
        }
        if now < self.retry_upload_at.load(Ordering::Acquire) {
            return false;
        }
        has_upload_reserve(self.sample_commit_with(now, query))
    }

    pub(super) fn reserve_staging_growth_with(
        &self,
        bytes: u64,
        now: u64,
        active: bool,
        query: impl FnOnce() -> Option<u64>,
    ) -> Option<GrowthReservation<'_>> {
        if !active {
            return Some(GrowthReservation { res: None, bytes: 0 });
        }
        if now < self.retry_upload_at.load(Ordering::Acquire) {
            self.growth_denials.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let mut g = self.commit.lock().unwrap();
        let avail = g.refresh_locked(now, query);
        if !has_growth_reserve(avail, bytes) {
            self.growth_denials.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        g.available = avail.saturating_sub(bytes);
        g.pending_growth = g.pending_growth.saturating_add(bytes);
        Some(GrowthReservation { res: Some(self), bytes })
    }

    pub(super) fn allow_staging_growth_with(
        &self,
        bytes: u64,
        now: u64,
        active: bool,
        query: impl FnOnce() -> Option<u64>,
    ) -> bool {
        self.reserve_staging_growth_with(bytes, now, active, query).is_some()
    }

    pub(super) fn note_deferred(&self) {
        self.deferrals.fetch_add(1, Ordering::Relaxed);
        UPLOAD_DEFERRALS.with(|c| c.set(c.get().wrapping_add(1)));
    }

    pub(super) fn note_free_trim(&self, count: usize, bytes: u64) {
        if count > 0 {
            self.free_trims.fetch_add(count as u64, Ordering::Relaxed);
            self.free_trim_bytes.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    pub(super) fn stats(&self) -> Stats {
        Stats {
            deferrals: self.deferrals.load(Ordering::Relaxed),
            oom_cooldowns: self.oom_cooldowns.load(Ordering::Relaxed),
            growth_denials: self.growth_denials.load(Ordering::Relaxed),
            free_trims: self.free_trims.load(Ordering::Relaxed),
            free_trim_bytes: self.free_trim_bytes.load(Ordering::Relaxed),
        }
    }
}

static GLOBAL: Residency = Residency::new();

thread_local! {
    static UPLOAD_DEFERRALS: Cell<u64> = const { Cell::new(0) };
}

pub(super) fn global() -> &'static Residency {
    &GLOBAL
}

pub fn has_upload_reserve(commit: u64) -> bool {
    commit >= UPLOAD_RESERVE_BYTES
}

pub fn has_growth_reserve(commit: u64, bytes: u64) -> bool {
    commit.saturating_sub(bytes) >= GROWTH_RESERVE_BYTES
}

/// Record a D3D12 allocation error; arms the `500 ms` upload cooldown on `E_OUTOFMEMORY` when enabled.
pub fn allocation_error(hr: i32) -> bool {
    GLOBAL.record_allocation_error(hr, coarse_ms(), enabled())
}

/// True while inside an active `E_OUTOFMEMORY` cooldown window.
pub fn in_cooldown() -> bool {
    GLOBAL.in_cooldown_at(coarse_ms(), enabled())
}

/// Whether texture upload work may proceed (pooled buffer reuse stays open down to `UPLOAD_RESERVE_BYTES`).
pub fn admit_upload() -> bool {
    GLOBAL.admit_upload_with(coarse_ms(), enabled(), query_avail_commit)
}

/// Whether the staging pool may commit a brand-new `bytes`-sized OS upload buffer.
pub fn allow_staging_growth(bytes: u64) -> bool {
    GLOBAL.allow_staging_growth_with(bytes, coarse_ms(), enabled(), query_avail_commit)
}

/// Delay before retrying deferred texture uploads (remaining cooldown, or `UPLOAD_RETRY_MS`).
pub fn retry_delay() -> Duration {
    GLOBAL.retry_delay_at(coarse_ms())
}

/// Increment thread-local and process-wide upload deferral counters.
pub fn note_upload_deferred() {
    GLOBAL.note_deferred();
}

/// Detect whether any upload deferral occurred during `work` without consuming enclosing observations.
pub fn upload_deferred_during<T>(work: impl FnOnce() -> T) -> (T, bool) {
    let before = UPLOAD_DEFERRALS.with(Cell::get);
    let result = work();
    (result, UPLOAD_DEFERRALS.with(Cell::get) != before)
}

/// Aggregate-only counters for `XCL_UPLOAD_BACKOFF=1`.
pub fn stats() -> Stats {
    GLOBAL.stats()
}

/// Log final aggregate-only summary at exit when `XCL_UPLOAD_BACKOFF=1` is enabled.
pub fn log_summary() {
    if !enabled() {
        return;
    }
    let s = stats();
    crate::xlog_info!(
        "D3D12.X: upload backoff summary (XCL_UPLOAD_BACKOFF=1): deferrals={}, oom_cooldowns={}, growth_denials={}, free_trims={} ({} MiB)",
        s.deferrals,
        s.oom_cooldowns,
        s.growth_denials,
        s.free_trims,
        s.free_trim_bytes >> 20
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_mode_is_a_complete_noop() {
        let r = Residency::new();
        assert!(!r.record_allocation_error(E_OUTOFMEMORY, 100, false));
        assert!(!r.in_cooldown_at(100, false));
        assert!(r.admit_upload_with(100, false, || Some(0)));
        assert!(r.allow_staging_growth_with(64 * MIB, 100, false, || Some(0)));
        assert_eq!(r.stats(), Stats::default());
    }

    #[test]
    fn oom_arms_500ms_cooldown_and_expires_cleanly() {
        let r = Residency::new();
        // Non-OOM errors must not arm cooldown.
        assert!(!r.record_allocation_error(crate::util::E_INVALIDARG, 1000, true));
        assert!(!r.in_cooldown_at(1000, true));

        assert!(r.record_allocation_error(E_OUTOFMEMORY, 1000, true));
        assert!(r.in_cooldown_at(1000, true));
        assert!(r.in_cooldown_at(1499, true));
        assert!(!r.admit_upload_with(1200, true, || Some(8192 * MIB)));
        assert!(!r.allow_staging_growth_with(16 * MIB, 1200, true, || Some(8192 * MIB)));
        assert_eq!(r.retry_delay_at(1200), Duration::from_millis(300));
        assert_eq!(r.retry_delay_at(1500), Duration::from_millis(UPLOAD_RETRY_MS));

        // Cooldown expires exactly at t + 500 ms.
        assert!(!r.in_cooldown_at(1500, true));
        assert!(r.admit_upload_with(1500, true, || Some(8192 * MIB)));
        assert!(r.allow_staging_growth_with(64 * MIB, 1500, true, || Some(8192 * MIB)));
        assert_eq!(r.stats().oom_cooldowns, 1);
        assert_eq!(r.stats().growth_denials, 1);
    }

    #[test]
    fn reserve_floors_keep_pooled_reuse_open_while_gating_fresh_growth() {
        for (commit, admit, grow_0, grow_64) in [
            (256 * MIB, false, false, false),
            (511 * MIB, false, false, false),
            (512 * MIB, true, false, false),
            (768 * MIB, true, false, false),
            (1024 * MIB, true, true, false),
            (1088 * MIB, true, true, true),
        ] {
            assert_eq!(has_upload_reserve(commit), admit, "commit={commit}");
            assert_eq!(has_growth_reserve(commit, 0), grow_0, "commit={commit}");
            assert_eq!(has_growth_reserve(commit, 64 * MIB), grow_64, "commit={commit}");
        }
    }

    #[test]
    fn commit_sample_is_cached_for_32ms_and_refreshes_after() {
        let r = Residency::new();
        let mut calls = 0u32;
        let c0 = r.sample_commit_with(100, || {
            calls += 1;
            Some(768 * MIB)
        });
        assert_eq!((c0, calls), (768 * MIB, 1));
        // Within 32 ms window: cached without re-querying OS.
        let c1 = r.sample_commit_with(131, || {
            calls += 1;
            Some(4096 * MIB)
        });
        assert_eq!((c1, calls), (768 * MIB, 1));
        assert!(r.admit_upload_with(131, true, || unreachable!("must use cached sample")));
        assert!(!r.allow_staging_growth_with(16 * MIB, 131, true, || unreachable!("must use cached sample")));
        // At +32 ms: refreshes cleanly.
        let c2 = r.sample_commit_with(132, || {
            calls += 1;
            Some(4096 * MIB)
        });
        assert_eq!((c2, calls), (4096 * MIB, 2));
        assert!(r.allow_staging_growth_with(64 * MIB, 132, true, || unreachable!("must use refreshed sample")));
    }

    #[test]
    fn staging_growth_debits_cached_sample_and_holds_in_flight_reservation() {
        let r = Residency::new();
        // 1,072 MiB leaves 48 MiB above the 1,024 MiB growth floor: the first 32 MiB grant passes
        // (debiting cached headroom to 1,040 MiB), and a second 32 MiB grant within the 32 ms
        // window is denied instead of dropping headroom to 1,008 MiB.
        let g1 = r.reserve_staging_growth_with(32 * MIB, 100, true, || Some(1072 * MIB));
        assert!(g1.is_some());
        assert!(r.reserve_staging_growth_with(32 * MIB, 110, true, || unreachable!("must use debited sample")).is_none());
        assert_eq!(r.stats().growth_denials, 1);

        // Even if the 32 ms window rolls over while the first allocation is still in flight (before
        // the OS reports the commit), a concurrent refresh subtracts the reserved bytes so it
        // cannot reset the budget.
        assert_eq!(r.sample_commit_with(132, || Some(1072 * MIB)), 1040 * MIB);
        assert!(r.reserve_staging_growth_with(32 * MIB, 132, true, || unreachable!("must use debited refresh")).is_none());
        assert_eq!(r.stats().growth_denials, 2);

        // Once the first allocation finishes, its completed bytes remain debited for the rest of
        // the current window; a subsequent window observing restored headroom admits growth again.
        drop(g1);
        assert!(!r.allow_staging_growth_with(32 * MIB, 140, true, || unreachable!("still within debited window")));
        assert_eq!(r.stats().growth_denials, 3);
        assert!(r.allow_staging_growth_with(32 * MIB, 164, true, || Some(1072 * MIB)));
        assert!(!r.allow_staging_growth_with(32 * MIB, 165, true, || unreachable!("debited again in new window")));
        assert_eq!(r.stats().growth_denials, 4);
    }

    #[test]
    fn upload_deferral_reaches_enclosing_observation() {
        let r = Residency::new();
        r.note_deferred();
        assert!(!upload_deferred_during(|| ()).1);
        let (_, deferred) = upload_deferred_during(|| {
            assert!(upload_deferred_during(|| r.note_deferred()).1);
            assert!(!upload_deferred_during(|| ()).1);
        });
        assert!(deferred);
        assert_eq!(r.stats().deferrals, 2);
    }
}
