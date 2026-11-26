//! Priority queue shared by pipeline-preparation workers.
//!
//! PSO preparation uses worker bands so background preloads cannot occupy the workers reserved
//! for builds that the title will need soon.

#![allow(dead_code)] // Some operations are seams for the following preparation changes.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};

pub(super) type Work = Box<dyn FnOnce() + Send + 'static>;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub(super) enum Priority {
    Speculative,
    Preload,
    BoundSoon,
    Foreground,
}

impl Priority {
    const COUNT: usize = 4;

    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct JobId(u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum JobStatus {
    Queued,
    Running,
    Complete,
    Cancelled,
    Panicked,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct RankedPreloadSnapshot {
    pub(super) submissions: u64,
    pub(super) started: u64,
    pub(super) promoted: u64,
}

struct RankedPreloadCounters {
    submissions: AtomicU64,
    started: AtomicU64,
    promoted: AtomicU64,
}

impl RankedPreloadCounters {
    const fn new() -> Self {
        Self {
            submissions: AtomicU64::new(0),
            started: AtomicU64::new(0),
            promoted: AtomicU64::new(0),
        }
    }

    fn snapshot(&self) -> RankedPreloadSnapshot {
        RankedPreloadSnapshot {
            submissions: self.submissions.load(Ordering::Relaxed),
            started: self.started.load(Ordering::Relaxed),
            promoted: self.promoted.load(Ordering::Relaxed),
        }
    }
}

impl JobStatus {
    fn terminal(self) -> bool {
        matches!(self, Self::Complete | Self::Cancelled | Self::Panicked)
    }
}

struct Completion {
    status: Mutex<JobStatus>,
    changed: Condvar,
}

impl Completion {
    fn new() -> Self {
        Self { status: Mutex::new(JobStatus::Queued), changed: Condvar::new() }
    }

    fn set(&self, status: JobStatus) {
        *self.status.lock().unwrap_or_else(|e| e.into_inner()) = status;
        self.changed.notify_all();
    }

    fn get(&self) -> JobStatus {
        *self.status.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait(&self) -> JobStatus {
        let mut status = self.status.lock().unwrap_or_else(|e| e.into_inner());
        while !status.terminal() {
            status = self.changed.wait(status).unwrap_or_else(|e| e.into_inner());
        }
        *status
    }
}

pub(super) struct JobHandle {
    id: JobId,
    queue: Weak<Inner>,
    completion: Arc<Completion>,
}

impl JobHandle {
    pub(super) fn id(&self) -> JobId {
        self.id
    }

    pub(super) fn status(&self) -> JobStatus {
        self.completion.get()
    }

    pub(super) fn wait(&self) -> JobStatus {
        self.completion.wait()
    }

    /// Raise this queued job to `priority`. Running and completed jobs cannot be promoted.
    pub(super) fn promote(&self, priority: Priority) -> bool {
        self.queue.upgrade().is_some_and(|q| q.promote(self.id, priority))
    }

    /// Cancel this job only if no worker has started it.
    pub(super) fn cancel(&self) -> bool {
        self.queue.upgrade().is_some_and(|q| q.cancel(self.id))
    }
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct Ticket {
    id: JobId,
    generation: u64,
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct PreloadTicket {
    unranked: bool,
    rank: usize,
    sequence: u64,
    ticket: Ticket,
}

struct Job {
    priority: Priority,
    preload_rank: Option<usize>,
    generation: u64,
    work: Work,
    completion: Arc<Completion>,
}

struct State {
    next_id: u64,
    next_sequence: u64,
    stopped: bool,
    jobs: HashMap<JobId, Job>,
    ready: [VecDeque<Ticket>; Priority::COUNT],
    preload_ready: BinaryHeap<Reverse<PreloadTicket>>,
}

impl State {
    fn push(&mut self, priority: Priority, ticket: Ticket, preload_rank: Option<usize>) {
        if priority == Priority::Preload {
            self.next_sequence = self.next_sequence.checked_add(1).expect("build queue sequence overflow");
            self.preload_ready.push(Reverse(PreloadTicket {
                unranked: preload_rank.is_none(),
                rank: preload_rank.unwrap_or(0),
                sequence: self.next_sequence,
                ticket,
            }));
        } else {
            self.ready[priority.index()].push_back(ticket);
        }
    }

    fn pop(&mut self, priority: Priority) -> Option<Ticket> {
        if priority == Priority::Preload {
            self.preload_ready.pop().map(|ticket| ticket.0.ticket)
        } else {
            self.ready[priority.index()].pop_front()
        }
    }
}

struct Inner {
    state: Mutex<State>,
    ranked_preloads: RankedPreloadCounters,
    preload_changed: Condvar,
    foreground_changed: Condvar,
    all_changed: Condvar,
}

impl Inner {
    fn changed(&self, low: Priority, high: Priority) -> &Condvar {
        if high <= Priority::Preload {
            &self.preload_changed
        } else if low >= Priority::BoundSoon {
            &self.foreground_changed
        } else {
            &self.all_changed
        }
    }

    fn wake(&self, priority: Priority) {
        if priority <= Priority::Preload {
            self.preload_changed.notify_one();
        } else {
            self.foreground_changed.notify_one();
        }
        self.all_changed.notify_one();
    }

    fn promote(&self, id: JobId, priority: Priority) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(job) = state.jobs.get_mut(&id) else { return false };
        if priority <= job.priority {
            return true;
        }
        let ranked_preload = job.priority == Priority::Preload && job.preload_rank.is_some();
        job.priority = priority;
        job.generation += 1;
        let ticket = Ticket { id, generation: job.generation };
        let preload_rank = job.preload_rank;
        state.push(priority, ticket, preload_rank);
        if ranked_preload {
            self.ranked_preloads.promoted.fetch_add(1, Ordering::Relaxed);
        }
        self.wake(priority);
        true
    }

    fn cancel(&self, id: JobId) -> bool {
        let job = self.state.lock().unwrap_or_else(|e| e.into_inner()).jobs.remove(&id);
        if let Some(job) = job {
            job.completion.set(JobStatus::Cancelled);
            true
        } else {
            false
        }
    }

    fn take(&self, wait: bool, low: Priority, high: Priority) -> Option<(Work, Arc<Completion>)> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let changed = self.changed(low, high);
        loop {
            for priority in (low.index()..=high.index()).rev() {
                let priority = match priority {
                    0 => Priority::Speculative,
                    1 => Priority::Preload,
                    2 => Priority::BoundSoon,
                    _ => Priority::Foreground,
                };
                while let Some(ticket) = state.pop(priority) {
                    let current = state.jobs.get(&ticket.id).is_some_and(|job| {
                        job.priority == priority && job.generation == ticket.generation
                    });
                    if current {
                        let job = state.jobs.remove(&ticket.id).unwrap();
                        if priority == Priority::Preload && job.preload_rank.is_some() {
                            self.ranked_preloads.started.fetch_add(1, Ordering::Relaxed);
                        }
                        job.completion.set(JobStatus::Running);
                        return Some((job.work, job.completion));
                    }
                }
            }
            if state.stopped || !wait {
                return None;
            }
            state = changed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn shutdown(&self) {
        let jobs = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if state.stopped {
                return;
            }
            state.stopped = true;
            for ready in &mut state.ready {
                ready.clear();
            }
            state.preload_ready.clear();
            std::mem::take(&mut state.jobs)
        };
        for (_, job) in jobs {
            job.completion.set(JobStatus::Cancelled);
        }
        self.preload_changed.notify_all();
        self.foreground_changed.notify_all();
        self.all_changed.notify_all();
    }
}

pub(super) struct BuildQueue {
    inner: Arc<Inner>,
}

impl BuildQueue {
    pub(super) fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    next_id: 0,
                    next_sequence: 0,
                    stopped: false,
                    jobs: HashMap::new(),
                    ready: std::array::from_fn(|_| VecDeque::new()),
                    preload_ready: BinaryHeap::new(),
                }),
                ranked_preloads: RankedPreloadCounters::new(),
                preload_changed: Condvar::new(),
                foreground_changed: Condvar::new(),
                all_changed: Condvar::new(),
            }),
        }
    }

    /// Add work to the queue. A stopped queue refuses it without running it.
    pub(super) fn submit(&self, priority: Priority, work: Work) -> Option<JobHandle> {
        self.submit_ranked(priority, None, work)
    }

    /// Add work, optionally ordered by an earlier-run rank within the Preload band only.
    pub(super) fn submit_ranked(&self, priority: Priority, preload_rank: Option<usize>, work: Work) -> Option<JobHandle> {
        let completion = Arc::new(Completion::new());
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.stopped {
            return None;
        }
        state.next_id = state.next_id.checked_add(1).expect("build queue job ID overflow");
        let id = JobId(state.next_id);
        let generation = 0;
        let preload_rank = if priority == Priority::Preload { preload_rank } else { None };
        state.jobs.insert(id, Job { priority, preload_rank, generation, work, completion: completion.clone() });
        state.push(priority, Ticket { id, generation }, preload_rank);
        if preload_rank.is_some() {
            self.inner.ranked_preloads.submissions.fetch_add(1, Ordering::Relaxed);
        }
        drop(state);
        self.inner.wake(priority);
        Some(JobHandle { id, queue: Arc::downgrade(&self.inner), completion })
    }

    pub(super) fn worker(&self) -> Worker {
        self.worker_between(Priority::Speculative, Priority::Foreground)
    }

    pub(super) fn ranked_preload_stats(&self) -> RankedPreloadSnapshot {
        self.inner.ranked_preloads.snapshot()
    }

    pub(super) fn worker_between(&self, low: Priority, high: Priority) -> Worker {
        Worker { inner: self.inner.clone(), low, high }
    }

    /// Run one ready job on this thread, for callers that help drain a batch.
    pub(super) fn try_run_one(&self) -> bool {
        self.try_run_one_between(Priority::Speculative, Priority::Foreground)
    }

    pub(super) fn try_run_one_between(&self, low: Priority, high: Priority) -> bool {
        self.inner.take(false, low, high).is_some_and(|job| {
            run(job);
            true
        })
    }

    pub(super) fn shutdown(&self) {
        self.inner.shutdown();
    }
}

impl Drop for BuildQueue {
    fn drop(&mut self) {
        self.inner.shutdown();
    }
}

pub(super) struct Worker {
    inner: Arc<Inner>,
    low: Priority,
    high: Priority,
}

impl Worker {
    pub(super) fn run(self) {
        while let Some(job) = self.inner.take(true, self.low, self.high) {
            run(job);
        }
    }
}

fn run((work, completion): (Work, Arc<Completion>)) {
    let status = if catch_unwind(AssertUnwindSafe(work)).is_ok() { JobStatus::Complete } else { JobStatus::Panicked };
    completion.set(status);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn record(out: &Arc<Mutex<Vec<u32>>>, value: u32) -> Work {
        let out = out.clone();
        Box::new(move || out.lock().unwrap().push(value))
    }

    #[test]
    fn highest_priority_runs_first_and_each_priority_is_fifo() {
        let q = BuildQueue::new();
        let out = Arc::new(Mutex::new(Vec::new()));
        q.submit(Priority::Speculative, record(&out, 1)).unwrap();
        q.submit(Priority::BoundSoon, record(&out, 2)).unwrap();
        q.submit(Priority::BoundSoon, record(&out, 3)).unwrap();
        q.submit(Priority::Foreground, record(&out, 4)).unwrap();
        q.submit(Priority::Preload, record(&out, 5)).unwrap();
        while q.try_run_one() {}
        assert_eq!(*out.lock().unwrap(), [4, 2, 3, 5, 1]);
    }

    #[test]
    fn preload_jobs_run_in_earlier_first_use_order() {
        let q = BuildQueue::new();
        let out = Arc::new(Mutex::new(Vec::new()));
        q.submit_ranked(Priority::Preload, None, record(&out, 1)).unwrap();
        q.submit_ranked(Priority::Preload, Some(20), record(&out, 2)).unwrap();
        q.submit_ranked(Priority::Preload, Some(3), record(&out, 3)).unwrap();
        while q.try_run_one_between(Priority::Preload, Priority::Preload) {}
        assert_eq!(*out.lock().unwrap(), [3, 2, 1]);
    }

    #[test]
    fn preload_rank_ties_and_unranked_jobs_are_fifo() {
        let q = BuildQueue::new();
        let out = Arc::new(Mutex::new(Vec::new()));
        q.submit_ranked(Priority::Preload, None, record(&out, 1)).unwrap();
        q.submit_ranked(Priority::Preload, Some(4), record(&out, 2)).unwrap();
        q.submit_ranked(Priority::Preload, Some(4), record(&out, 3)).unwrap();
        q.submit_ranked(Priority::Preload, None, record(&out, 4)).unwrap();
        while q.try_run_one_between(Priority::Preload, Priority::Preload) {}
        assert_eq!(*out.lock().unwrap(), [2, 3, 1, 4]);
        assert_eq!(
            q.ranked_preload_stats(),
            RankedPreloadSnapshot { submissions: 2, started: 2, promoted: 0 },
        );
    }

    #[test]
    fn promoted_and_cancelled_ranked_tickets_go_stale() {
        let q = BuildQueue::new();
        let out = Arc::new(Mutex::new(Vec::new()));
        let promoted = q.submit_ranked(Priority::Preload, Some(0), record(&out, 1)).unwrap();
        let cancelled = q.submit_ranked(Priority::Preload, Some(1), record(&out, 2)).unwrap();
        q.submit_ranked(Priority::Preload, Some(2), record(&out, 3)).unwrap();
        assert!(promoted.promote(Priority::Foreground));
        assert!(cancelled.cancel());
        assert!(q.try_run_one_between(Priority::Preload, Priority::Preload));
        assert!(!q.try_run_one_between(Priority::Preload, Priority::Preload));
        assert!(q.try_run_one_between(Priority::Foreground, Priority::Foreground));
        assert_eq!(*out.lock().unwrap(), [3, 1]);
        assert_eq!(
            q.ranked_preload_stats(),
            RankedPreloadSnapshot { submissions: 3, started: 1, promoted: 1 },
        );
    }

    #[test]
    fn promotion_is_monotonic_and_joins_the_new_priority_fifo() {
        let q = BuildQueue::new();
        let out = Arc::new(Mutex::new(Vec::new()));
        q.submit(Priority::Foreground, record(&out, 1)).unwrap();
        let promoted = q.submit(Priority::Speculative, record(&out, 2)).unwrap();
        assert!(promoted.promote(Priority::Foreground));
        assert!(promoted.promote(Priority::Preload), "a lower request leaves it promoted");
        while q.try_run_one() {}
        assert_eq!(*out.lock().unwrap(), [1, 2]);
    }

    #[test]
    fn foreground_worker_band_is_not_trapped_by_preload_work() {
        let q = BuildQueue::new();
        let normal = q.worker_between(Priority::BoundSoon, Priority::Foreground);
        let preload = q.worker_between(Priority::Speculative, Priority::Preload);
        let normal_thread = std::thread::Builder::new().name("normal-band".into()).spawn(move || normal.run()).unwrap();
        let preload_thread = std::thread::Builder::new().name("preload-band".into()).spawn(move || preload.run()).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        for (priority, label) in [(Priority::Preload, "preload"), (Priority::Foreground, "foreground")] {
            let tx = tx.clone();
            q.submit(priority, Box::new(move || {
                tx.send((label, std::thread::current().name().unwrap().to_string())).unwrap();
            })).unwrap();
        }
        let mut ran = vec![
            rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap(),
            rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap(),
        ];
        ran.sort_unstable();
        assert_eq!(ran, [("foreground", "normal-band".into()), ("preload", "preload-band".into())]);
        q.shutdown();
        normal_thread.join().unwrap();
        preload_thread.join().unwrap();
    }

    #[test]
    fn promoted_preload_moves_to_the_foreground_worker_band() {
        let q = BuildQueue::new();
        let ran = Arc::new(AtomicBool::new(false));
        let mark = ran.clone();
        let job = q.submit(Priority::Preload, Box::new(move || mark.store(true, Ordering::Release))).unwrap();
        assert!(job.promote(Priority::Foreground));
        assert!(!q.try_run_one_between(Priority::Speculative, Priority::Preload));
        assert!(q.try_run_one_between(Priority::BoundSoon, Priority::Foreground));
        assert!(ran.load(Ordering::Acquire));
    }

    #[test]
    fn job_ids_are_stable_and_not_reused() {
        let q = BuildQueue::new();
        let a = q.submit(Priority::Preload, Box::new(|| {})).unwrap();
        let b = q.submit(Priority::Preload, Box::new(|| {})).unwrap();
        assert_ne!(a.id(), b.id());
        let id = a.id();
        assert!(a.cancel());
        assert_eq!(a.id(), id);
    }

    #[test]
    fn only_a_queued_job_can_be_cancelled() {
        let q = BuildQueue::new();
        let cancelled = q.submit(Priority::Preload, Box::new(|| panic!("must not run"))).unwrap();
        assert!(cancelled.cancel());
        assert_eq!(cancelled.wait(), JobStatus::Cancelled);

        let running = Arc::new(AtomicBool::new(false));
        let finish = Arc::new((Mutex::new(false), Condvar::new()));
        let (seen, gate) = (running.clone(), finish.clone());
        let handle = q.submit(Priority::Foreground, Box::new(move || {
            seen.store(true, Ordering::Release);
            let mut open = gate.0.lock().unwrap();
            while !*open {
                open = gate.1.wait(open).unwrap();
            }
        })).unwrap();
        let worker = q.worker();
        let thread = std::thread::spawn(move || worker.run());
        while !running.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        assert_eq!(handle.status(), JobStatus::Running);
        assert!(!handle.cancel());
        *finish.0.lock().unwrap() = true;
        finish.1.notify_all();
        assert_eq!(handle.wait(), JobStatus::Complete);
        q.shutdown();
        thread.join().unwrap();
    }

    #[test]
    fn a_panicking_job_completes_and_the_worker_continues() {
        let q = BuildQueue::new();
        let panicked = q.submit(Priority::Foreground, Box::new(|| panic!("expected"))).unwrap();
        let ran = Arc::new(AtomicBool::new(false));
        let mark = ran.clone();
        let after = q.submit(Priority::BoundSoon, Box::new(move || mark.store(true, Ordering::Release))).unwrap();
        assert!(q.try_run_one());
        assert_eq!(panicked.wait(), JobStatus::Panicked);
        assert!(q.try_run_one());
        assert_eq!(after.wait(), JobStatus::Complete);
        assert!(ran.load(Ordering::Acquire));
    }

    #[test]
    fn shutdown_cancels_queued_work_and_wakes_workers() {
        let q = BuildQueue::new();
        let a = q.submit(Priority::Foreground, Box::new(|| panic!("must not run"))).unwrap();
        let b = q.submit(Priority::Speculative, Box::new(|| panic!("must not run"))).unwrap();
        let worker = q.worker();
        q.shutdown();
        let thread = std::thread::spawn(move || worker.run());
        assert_eq!(a.wait(), JobStatus::Cancelled);
        assert_eq!(b.wait(), JobStatus::Cancelled);
        assert!(q.submit(Priority::Foreground, Box::new(|| {})).is_none());
        thread.join().unwrap();
    }
}
