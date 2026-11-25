//! Diagnostic (`XCL_LOOK_STATS=1`): what the looks for CPU writes find, as averages over 256 of them
//! (logged): how many of a look's pages the looks before it had found too, in how many runs they
//! lie, how many of the pages scanned had changed, and how long before the submit that looked the
//! youngest of its lists had been closed.

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_LOOK_STATS").is_ok_and(|v| v.trim() == "1"))
}

/// the upper ends (µs) of the steps `Stats::ago` counts by; the last step has none
const STEPS: [u64; 6] = [20, 100, 500, 2_000, 10_000, 50_000];

#[derive(Default)]
struct Stats {
    looks: u64,
    us: u64,
    pages: u64,
    /// of them found between the submits (`XCL_LOOK_AHEAD`)
    ahead: u64,
    runs: u64,
    /// pages also in the look before, and in all three before
    once: u64,
    thrice: u64,
    /// pages scanned (the ones of video memory's windows), and those with a changed chunk
    scanned: u64,
    changed: u64,
    ago: [u64; STEPS.len() + 1],
    /// the pages of the three looks before, the last one first
    before: [Vec<usize>; 3],
}

static STATS: Mutex<Option<Stats>> = Mutex::new(None);

/// How many of `a`'s pages every one of `others` has too (all sorted, each page once).
fn in_all(a: &[usize], others: &[Vec<usize>]) -> u64 {
    let mut at = vec![0usize; others.len()];
    a.iter()
        .filter(|&&p| {
            others.iter().zip(at.iter_mut()).all(|(o, k)| {
                while *k < o.len() && o[*k] < p {
                    *k += 1;
                }
                o.get(*k) == Some(&p)
            })
        })
        .count() as u64
}

/// The runs of neighbouring pages of `pages` (sorted, 4 KiB each).
fn runs_of(pages: &[usize]) -> u64 {
    pages.iter().zip(std::iter::once(&usize::MAX).chain(pages)).filter(|(p, q)| **p != q.wrapping_add(0x1000)).count() as u64
}

/// A look found `pages` (sorted), `ahead` of them before its submit; it took `took`. `closed`: how
/// long before its submit the youngest of the submit's lists was closed (None: not known).
pub(super) fn look(pages: &[usize], ahead: usize, took: Duration, closed: Option<u64>) {
    if !on() {
        return;
    }
    let mut g = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let s = g.get_or_insert_with(Stats::default);
    s.looks += 1;
    s.us += took.as_micros() as u64;
    s.pages += pages.len() as u64;
    s.ahead += ahead as u64;
    s.runs += runs_of(pages);
    s.once += in_all(pages, &s.before[..1]);
    s.thrice += in_all(pages, &s.before);
    s.before.rotate_right(1);
    s.before[0].clear();
    s.before[0].extend_from_slice(pages);
    if let Some(us) = closed {
        s.ago[STEPS.partition_point(|&e| e <= us)] += 1;
    }
    if s.looks % 256 == 0 {
        let a = |v: u64| v / 256;
        let share = |v: u64, of: u64| if of == 0 { 0 } else { v * 100 / of };
        crate::xlog_info!(
            "D3D12.X: the last 256 looks for CPU writes: {:.2} ms and {} pages a look ({} found ahead) in {} runs ({} % of the pages also in the look before, {} % in all three before); {} scanned a look, {} % of them changed; youngest list closed before the submit: {} <20 µs, {} <100 µs, {} <500 µs, {} <2 ms, {} <10 ms, {} <50 ms, {} longer",
            s.us as f64 / 256e3,
            a(s.pages),
            a(s.ahead),
            a(s.runs),
            share(s.once, s.pages),
            share(s.thrice, s.pages),
            a(s.scanned),
            share(s.changed, s.scanned),
            s.ago[0],
            s.ago[1],
            s.ago[2],
            s.ago[3],
            s.ago[4],
            s.ago[5],
            s.ago[6]
        );
        *s = Stats { looks: s.looks, before: std::mem::take(&mut s.before), ..Stats::default() };
    }
}

/// A scan went over `pages` pages and found the changed `runs` (address, length; sorted).
pub(super) fn scan(pages: usize, runs: &[(usize, usize)]) {
    if !on() {
        return;
    }
    let mut g = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let s = g.get_or_insert_with(Stats::default);
    s.scanned += pages as u64;
    let mut next = 0usize;
    for &(a, n) in runs {
        let (first, end) = ((a & !0xFFF).max(next), (a + n + 0xFFF) & !0xFFF);
        if end > first {
            s.changed += ((end - first) >> 12) as u64;
            next = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_found_again_and_their_runs() {
        let p = |v: &[usize]| v.iter().map(|k| k * 0x1000).collect::<Vec<_>>();
        let a = p(&[1, 2, 3, 7, 9, 10]);
        assert_eq!(runs_of(&a), 3);
        assert_eq!(runs_of(&[]), 0);
        assert_eq!(runs_of(&p(&[1])), 1);
        assert_eq!(in_all(&a, &[p(&[2, 3, 4, 10])]), 3);
        assert_eq!(in_all(&a, &[p(&[2, 3, 4, 10]), p(&[3, 7, 10]), p(&[1, 3])]), 1);
        assert_eq!(in_all(&a, &[p(&[])]), 0);
    }
}
