//! Diagnostic (`XCL_ACTIVITY`): the time the title's threads spend in some of XCL's calls while it
//! loads, summed over the threads.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone, Copy)]
pub enum Call {
    PlacedTexture,
    PlacedBuffer,
    PhysicalPages,
    Memory,
    Pipeline,
}

pub const CALLS: [(Call, &str); 5] = [
    (Call::PlacedTexture, "placed textures"),
    (Call::PlacedBuffer, "placed buffers"),
    (Call::PhysicalPages, "physical page calls"),
    (Call::Memory, "memory calls"),
    (Call::Pipeline, "pipelines"),
];

/// calls, microseconds
static COST: [[AtomicU64; 2]; CALLS.len()] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; CALLS.len()];

/// Counts the call it stands in when it goes.
pub struct Timer {
    pub call: Call,
    start: Instant,
}

pub fn time(call: Call) -> Timer {
    Timer { call, start: Instant::now() }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let c = &COST[self.call as usize];
        c[0].fetch_add(1, Ordering::Relaxed);
        c[1].fetch_add(self.start.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

/// (calls, microseconds) so far, in the order of `CALLS`.
pub fn totals() -> [(u64, u64); CALLS.len()] {
    std::array::from_fn(|k| (COST[k][0].load(Ordering::Relaxed), COST[k][1].load(Ordering::Relaxed)))
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
}
