//! The title's last `Release` of a resource, put off until the GPU is past the work in flight.
//!
//! D3D12 on the PC asks that an object outlives the GPU work that uses it. A console title
//! keeps to that by its own count of finished frames – but XCL's submits are not the title's:
//! texture uploads go out with a later submit, tile mappings on a queue of XCL's own. A retail
//! title's final `Release` of a resource then came while a queue still had work on it (debug
//! layer: "CORRUPTION: … is referenced by GPU operations in-flight", id 921).
//!
//! The last reference is held here instead. Each batch of held resources gets a fence value
//! signalled on every queue with work on it; the batch goes once all of them have passed it.
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// a batch is not closed more often than this (a `Signal` a queue each time)
const SEAL_EVERY: Duration = Duration::from_millis(10);
/// how often the thread here looks for batches that can go
const TICK: Duration = Duration::from_millis(4);

struct Batch {
    dead: Vec<usize>,
    /// (fence, value) a queue: all reached = nothing submitted before the batch closed is left
    marks: Vec<(ID3D12Fence, u64)>,
}

#[derive(Default)]
struct Held {
    /// queue → its fence here and the last value signalled on it
    fences: HashMap<usize, (ID3D12Fence, u64)>,
    open: Vec<usize>,
    sealed: VecDeque<Batch>,
    last_seal: Option<Instant>,
}

// (the interface pointers are only used under the lock)
unsafe impl Send for Held {}

impl Held {
    fn is_empty(&self) -> bool {
        self.open.is_empty() && self.sealed.is_empty()
    }

    /// Close the open batch (at most every `SEAL_EVERY`, or `seal_now`) with a fence value on each of `queues`,
    /// and take out the batches every queue is past: the resources to release now.
    unsafe fn drain(&mut self, queues: &[ID3D12CommandQueue], seal_now: bool) -> Vec<usize> {
        if !self.open.is_empty() && (seal_now || self.last_seal.is_none_or(|t| t.elapsed() >= SEAL_EVERY)) {
            let mut marks = Vec::with_capacity(queues.len());
            let mut ok = true;
            for q in queues {
                let key = q.as_raw() as usize;
                if !self.fences.contains_key(&key) {
                    let mut dev: Option<ID3D12Device> = None;
                    let made = q.GetDevice(&mut dev).ok().and(dev).and_then(|d| d.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok());
                    let Some(f) = made else {
                        ok = false;
                        break;
                    };
                    self.fences.insert(key, (f, 0));
                }
                let (fence, value) = self.fences.get_mut(&key).unwrap();
                if q.Signal(&*fence, *value + 1).is_err() {
                    ok = false;
                    break;
                }
                *value += 1;
                marks.push((fence.clone(), *value));
            }
            // (no fence or no signal: the device is gone – the batch stays open, and held)
            if ok {
                self.last_seal = Some(Instant::now());
                let dead = std::mem::take(&mut self.open);
                self.sealed.push_back(Batch { dead, marks });
            }
        }
        let mut gone = Vec::new();
        while self.sealed.front().is_some_and(|b| b.passed()) {
            gone.extend(self.sealed.pop_front().unwrap().dead);
        }
        gone
    }

    /// Whether a drain has something to do: a sealed batch every queue is past (the front one:
    /// they pass in order), or the open batch to close (`seal_now`, else every `SEAL_EVERY`).
    /// A few `GetCompletedValue`s, no queue taken (each submit asks).
    unsafe fn due(&self, seal_now: bool) -> bool {
        self.sealed.front().is_some_and(|b| b.passed()) || (!self.open.is_empty() && (seal_now || self.last_seal.is_none_or(|t| t.elapsed() >= SEAL_EVERY)))
    }

    /// Work in flight: the queues are not past the newest sealed batch yet. (None sealed: they
    /// were past everything at the last drain.) A fence value signalled now comes after that
    /// batch's on each queue, and is passed after it.
    unsafe fn in_flight(&self) -> bool {
        self.sealed.back().is_some_and(|b| !b.passed())
    }

    /// `res` held; whether nothing is known in flight (its release may wait for the queues to
    /// say so, `hold`).
    unsafe fn admit(&mut self, res: usize) -> bool {
        self.open.push(res);
        !self.in_flight()
    }
}

impl Batch {
    /// Every queue is past its mark (a batch with none: at once).
    unsafe fn passed(&self) -> bool {
        self.marks.iter().all(|(f, v)| f.GetCompletedValue() >= *v)
    }
}

static HELD: Mutex<Option<Held>> = Mutex::new(None);

/// Whether last releases are held (`XCL_LATE_RELEASE=0` turns it off).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_LATE_RELEASE").is_ok_and(|v| v.trim() == "0"))
}

/// how long a last `Release` itself waits for queues with nothing to do to say so
const IDLE_WAIT: Duration = Duration::from_micros(300);

/// `res` (an adopted resource) holds one reference that is now this module's. With no work
/// known in flight it goes before this returns (its batch closed now, and the queues pass the
/// fence value at once): a title that frees the memory under a resource right after its last
/// `Release` finds it unpinned, as before. With work in flight (the queues not past the last
/// batch) it could not go in `IDLE_WAIT` – a value signalled now is passed after that batch's
/// – so there is no wait and no seal (but every `SEAL_EVERY`): the thread here lets it go, as
/// it did after the wait. That signalled every queue (about 10) and spun 300 µs on the title's
/// thread at each last `Release` while the GPU had frames in flight.
pub unsafe fn hold(res: usize) {
    let idle = HELD.lock().unwrap().get_or_insert_with(Held::default).admit(res);
    if idle {
        let start = Instant::now();
        drain_known(true);
        while start.elapsed() < IDLE_WAIT && HELD.lock().unwrap().as_ref().is_some_and(|h| !h.is_empty()) {
            std::hint::spin_loop();
            drain_known(false);
        }
        if HELD.lock().unwrap().as_ref().is_none_or(Held::is_empty) {
            return;
        }
    } else {
        drain_known(false);
    }
    // A thread of its own lets the held ones go: the title may release a resource and then
    // wait for its memory without another submit (an imported heap pins the pages under its
    // resources – unmapping them stood still while one was held "until the next submit").
    static THREAD: std::sync::Once = std::sync::Once::new();
    THREAD.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-late-release".into()).spawn(|| loop {
            std::thread::sleep(TICK);
            unsafe { drain_known(false) };
        });
    });
}

/// Release what every queue XCL knows of is past: the title's, the tile-mapping one, the
/// upload ones. (Each submit of the title's comes here: nothing is taken but `HELD` before a
/// batch is known to be due.)
pub unsafe fn drain_known(seal_now: bool) {
    if !HELD.lock().unwrap().as_ref().is_some_and(|h| h.due(seal_now)) {
        return;
    }
    let mut queues: Vec<ID3D12CommandQueue> = Vec::new();
    for raw in super::present::known_queues() {
        let raw = raw as *mut c_void;
        if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&raw) {
            queues.push(q.clone());
        }
    }
    queues.extend(super::sparse::queue());
    queues.extend(super::title_textures::upload_queues());
    let gone = match HELD.lock().unwrap().as_mut() {
        Some(h) => h.drain(&queues, seal_now),
        None => return,
    };
    // (outside the lock: a release may end other objects, whose releases come back here)
    for res in gone {
        let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(super::real::orig_fn(res as *mut c_void, 2));
        f(res as *mut c_void);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

    /// A held resource stays while a queue has not got past the work submitted before it was
    /// released, and goes once the queue has.
    #[test]
    fn a_held_resource_goes_when_the_queue_is_past_its_batch() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            // the queue stands still, as with work in flight: it waits for a fence nobody has signalled
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Wait(&gate, 1).unwrap();
            let mut res: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 4096,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut res).unwrap();
            let raw = res.unwrap().into_raw() as usize;

            let mut held = Held::default();
            held.open.push(raw);
            let queues = [queue.clone()];
            assert!(held.drain(&queues, false).is_empty(), "the queue has not passed the batch's fence value");
            std::thread::sleep(Duration::from_millis(100));
            assert!(held.drain(&queues, false).is_empty(), "still in flight: still held");
            assert!(!held.is_empty());

            gate.Signal(1).unwrap();
            let start = Instant::now();
            let mut gone = Vec::new();
            while gone.is_empty() && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(5));
                gone = held.drain(&queues, false);
            }
            assert_eq!(gone, vec![raw], "released once the queue is past the batch");
            assert!(held.is_empty());
            drop(ID3D12Resource::from_raw(raw as *mut c_void));
        }
    }

    /// A drain is due only for a sealed batch the queues are past (one with no marks: at once)
    /// or an open batch to close: now when asked, else every `SEAL_EVERY`.
    #[test]
    fn a_drain_is_due_only_when_a_batch_can_go_or_be_closed() {
        unsafe {
            let mut held = Held::default();
            assert!(!held.due(true) && !held.due(false), "nothing held");
            assert!(held.admit(0x1000), "nothing in flight");
            assert!(held.due(false), "never closed: due");
            held.last_seal = Some(Instant::now());
            assert!(!held.due(false), "closed one just now: not before SEAL_EVERY");
            assert!(held.due(true), "now when asked");
            held.last_seal = Instant::now().checked_sub(SEAL_EVERY);
            assert!(held.due(false));
            let mut unmarked = Held::default();
            unmarked.sealed.push_back(Batch { dead: vec![0x2000], marks: Vec::new() });
            assert!(unmarked.due(false) && !unmarked.in_flight(), "a batch with no marks goes at once");
        }
    }

    /// While a queue is not past the newest batch, work is in flight: a resource held then
    /// does not close a batch of its own (but every `SEAL_EVERY`) nor wait; once the queue is
    /// past, the batch is due and the next one is not in flight.
    #[test]
    fn a_resource_held_with_work_in_flight_does_not_close_a_batch_of_its_own() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Wait(&gate, 1).unwrap();
            let queues = [queue.clone()];
            let mut held = Held::default();
            assert!(held.admit(0x1000), "nothing in flight yet");
            assert!(held.drain(&queues, true).is_empty());
            assert!(held.in_flight(), "the queue stands still before the batch's value");
            assert!(!held.admit(0x2000), "held with work in flight");
            assert!(!held.due(false), "not due: no batch passed, one closed just now");
            assert!(held.drain(&queues, false).is_empty());
            assert_eq!(held.sealed.len(), 1, "no batch of its own");
            assert_eq!(held.open, vec![0x2000]);
            gate.Signal(1).unwrap();
            let start = Instant::now();
            while !held.due(false) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(1));
            }
            assert!(held.due(false) && !held.in_flight(), "past the batch: due");
            // the first batch now, the second once the idle queue is past its value
            let mut gone = held.drain(&queues, true);
            assert_eq!(gone.first(), Some(&0x1000));
            let start = Instant::now();
            while gone.len() < 2 && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(1));
                gone.extend(held.drain(&queues, false));
            }
            assert_eq!(gone, vec![0x1000, 0x2000]);
            assert!(held.is_empty() && !held.in_flight());
        }
    }

    /// With no queue at all nothing can be in flight: the next drain lets the batch go.
    #[test]
    fn without_queues_a_batch_goes_at_once() {
        let mut held = Held::default();
        held.open.push(0x1000);
        assert_eq!(unsafe { held.drain(&[], false) }, vec![0x1000]);
        assert!(held.is_empty());
    }
}
