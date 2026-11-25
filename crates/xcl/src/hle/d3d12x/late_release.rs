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
    dead_pc: Vec<ID3D12Resource>,
    /// (fence, value) a queue: all reached = nothing submitted before the batch closed is left
    marks: Vec<(ID3D12Fence, u64)>,
}

#[derive(Default)]
struct Held {
    /// queue → its fence here and the last value signalled on it
    fences: HashMap<usize, (ID3D12Fence, u64)>,
    open: Vec<usize>,
    /// Owned PC resources (such as `list_x` predication buffers) held until every queue passes the batch fence.
    open_pc: Vec<ID3D12Resource>,
    sealed: VecDeque<Batch>,
    last_seal: Option<Instant>,
}

// (the interface pointers are only used under the lock)
unsafe impl Send for Held {}

impl Held {
    fn is_empty(&self) -> bool {
        self.open.is_empty() && self.open_pc.is_empty() && self.sealed.is_empty()
    }

    #[cfg(test)]
    unsafe fn drain(&mut self, queues: &[ID3D12CommandQueue], seal_now: bool) -> Vec<usize> {
        self.drain_all(queues, seal_now).0
    }

    /// Close the open batch (at most every `SEAL_EVERY`, or `seal_now`) with a fence value on each of `queues`,
    /// and take out the batches every queue is past: the raw and owned-PC resources to release now.
    unsafe fn drain_all(&mut self, queues: &[ID3D12CommandQueue], seal_now: bool) -> (Vec<usize>, Vec<ID3D12Resource>) {
        if (!self.open.is_empty() || !self.open_pc.is_empty()) && (seal_now || queues.is_empty() || self.last_seal.is_none_or(|t| t.elapsed() >= SEAL_EVERY)) {
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
                let dead_pc = std::mem::take(&mut self.open_pc);
                self.sealed.push_back(Batch { dead, dead_pc, marks });
            }
        }
        let mut gone = Vec::new();
        let mut gone_pc = Vec::new();
        while self.sealed.front().is_some_and(|b| b.marks.iter().all(|(f, v)| f.GetCompletedValue() >= *v)) {
            let b = self.sealed.pop_front().unwrap();
            gone.extend(b.dead);
            gone_pc.extend(b.dead_pc);
        }
        (gone, gone_pc)
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

fn ensure_thread() {
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

/// Defer COM `Release` of PC-owned resources (such as `list_x` predication buffers) until every
/// queue XCL knows of has passed the batch fence.
pub(super) unsafe fn hold_pc_resources(resources: Vec<ID3D12Resource>) {
    if resources.is_empty() {
        return;
    }
    if !enabled() {
        drop(resources);
        return;
    }
    HELD.lock().unwrap().get_or_insert_with(Held::default).open_pc.extend(resources);
    drain_known(false);
    if HELD.lock().unwrap().as_ref().is_some_and(|h| !h.is_empty()) {
        ensure_thread();
    }
}

/// `res` (an adopted resource) holds one reference that is now this module's. With no work in
/// flight it goes before this returns (the queues pass a fence value at once): a title that
/// frees the memory under a resource right after its last `Release` finds it unpinned, as
/// before. Otherwise the thread here lets it go.
pub unsafe fn hold(res: usize) {
    HELD.lock().unwrap().get_or_insert_with(Held::default).open.push(res);
    let start = Instant::now();
    drain_known(true);
    while start.elapsed() < IDLE_WAIT && HELD.lock().unwrap().as_ref().is_some_and(|h| !h.is_empty()) {
        std::hint::spin_loop();
        drain_known(false);
    }
    if HELD.lock().unwrap().as_ref().is_none_or(Held::is_empty) {
        return;
    }
    ensure_thread();
}

/// Release what every queue XCL knows of is past: the title's, the tile-mapping one, the
/// upload ones.
pub unsafe fn drain_known(seal_now: bool) {
    if HELD.lock().unwrap().as_ref().is_none_or(Held::is_empty) {
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
    let (gone, gone_pc) = match HELD.lock().unwrap().as_mut() {
        Some(h) => h.drain_all(&queues, seal_now),
        None => return,
    };
    // (outside the lock: a release may end other objects, whose releases come back here)
    drop(gone_pc);
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

    /// With no queue at all nothing can be in flight: the next drain lets the batch go.
    #[test]
    fn without_queues_a_batch_goes_at_once() {
        let mut held = Held::default();
        held.open.push(0x1000);
        assert_eq!(unsafe { held.drain(&[], false) }, vec![0x1000]);
        assert!(held.is_empty());
    }

    unsafe fn com_refcount(r: &ID3D12Resource) -> u32 {
        let raw = r.as_raw();
        let vt = *(raw as *const *const usize);
        let add: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(1));
        let rel: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(2));
        add(raw);
        rel(raw)
    }

    fn make_test_resource(dev: &ID3D12Device) -> ID3D12Resource {
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
        unsafe {
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut res,
            )
            .unwrap();
        }
        res.unwrap()
    }

    /// A held PC resource keeps its COM reference while any queue is behind the batch fence, and drops it once passed.
    #[test]
    fn held_pc_resources_wait_until_the_queue_is_past_their_batch() {
        struct SignalOnDrop(ID3D12Fence);
        impl Drop for SignalOnDrop {
            fn drop(&mut self) {
                unsafe {
                    let _ = self.0.Signal(1);
                }
            }
        }
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let signal_guard = SignalOnDrop(gate.clone());
            queue.Wait(&gate, 1).unwrap();

            let res = make_test_resource(&dev);
            let spy = res.clone();

            let mut held = Held::default();
            held.open_pc.push(res);
            let queues = [queue];
            let (gone, gone_pc) = held.drain_all(&queues, false);
            assert!(gone.is_empty() && gone_pc.is_empty(), "the queue has not passed the batch's fence value");
            std::thread::sleep(Duration::from_millis(50));
            let (gone, gone_pc) = held.drain_all(&queues, false);
            assert!(gone.is_empty() && gone_pc.is_empty(), "still in flight: still held");
            assert!(!held.is_empty());
            assert!(com_refcount(&spy) >= 2, "owned PC resource must stay alive while queue is blocked");

            drop(signal_guard);
            let start = Instant::now();
            let mut gone_pc = Vec::new();
            while gone_pc.is_empty() && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(5));
                gone_pc = held.drain_all(&queues, false).1;
            }
            assert_eq!(gone_pc.len(), 1, "drained once the queue is past the batch");
            assert!(held.is_empty());
            drop(gone_pc);
            assert_eq!(com_refcount(&spy), 1, "owned PC resource must drop once released outside lock");
        }
    }

    /// With no queues in flight, owned PC resources drain on the very next pass.
    #[test]
    fn without_queues_held_pc_resources_go_at_once() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let res = make_test_resource(&dev);
            let spy = res.clone();
            let mut held = Held::default();
            held.open_pc.push(res);
            let (gone, gone_pc) = held.drain_all(&[], false);
            assert!(gone.is_empty());
            assert_eq!(gone_pc.len(), 1);
            assert!(held.is_empty());
            drop(gone_pc);
            assert_eq!(com_refcount(&spy), 1);
        }
    }
}
