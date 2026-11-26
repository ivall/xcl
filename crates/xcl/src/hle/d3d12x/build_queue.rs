//! Priority queue shared by pipeline-preparation workers.
//!
//! PSO preparation uses worker bands so background preloads cannot occupy the workers reserved
//! for builds that the title will need soon.

#![allow(dead_code)] // Some operations are seams for the following preparation changes.

use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct JobId(u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum JobStatus {
    Queued,
    Running,
    Complete,
    Cancelled,
    Panicked,
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

#[derive(Clone, Copy)]
struct Ticket {
    id: JobId,
    generation: u64,
}

struct Job {
    priority: Priority,
    generation: u64,
    work: Work,
    completion: Arc<Completion>,
}

struct State {
    next_id: u64,
    stopped: bool,
    jobs: HashMap<JobId, Job>,
    ready: [VecDeque<Ticket>; Priority::COUNT],
}

struct Inner {
    state: Mutex<State>,
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
        job.priority = priority;
        job.generation += 1;
        let ticket = Ticket { id, generation: job.generation };
        state.ready[priority.index()].push_back(ticket);
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
                while let Some(ticket) = state.ready[priority].pop_front() {
                    let current = state.jobs.get(&ticket.id).is_some_and(|job| {
                        job.priority.index() == priority && job.generation == ticket.generation
                    });
                    if current {
                        let job = state.jobs.remove(&ticket.id).unwrap();
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
                    stopped: false,
                    jobs: HashMap::new(),
                    ready: std::array::from_fn(|_| VecDeque::new()),
                }),
                preload_changed: Condvar::new(),
                foreground_changed: Condvar::new(),
                all_changed: Condvar::new(),
            }),
        }
    }

    /// Add work to the queue. A stopped queue refuses it without running it.
    pub(super) fn submit(&self, priority: Priority, work: Work) -> Option<JobHandle> {
        let completion = Arc::new(Completion::new());
        let mut state = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.stopped {
            return None;
        }
        state.next_id = state.next_id.checked_add(1).expect("build queue job ID overflow");
        let id = JobId(state.next_id);
        let generation = 0;
        state.jobs.insert(id, Job { priority, generation, work, completion: completion.clone() });
        state.ready[priority.index()].push_back(Ticket { id, generation });
        drop(state);
        self.inner.wake(priority);
        Some(JobHandle { id, queue: Arc::downgrade(&self.inner), completion })
    }

    pub(super) fn worker(&self) -> Worker {
        self.worker_between(Priority::Speculative, Priority::Foreground)
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
