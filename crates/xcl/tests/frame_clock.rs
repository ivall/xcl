//! Integration tests for the D3D12.X frame clock (`xcl::hle::d3d12x::frame`).

use std::ptr::{null, null_mut};
use std::sync::Mutex;
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::Threading::{CreateEventW, ResetEvent, SetEvent};
use xcl::hle::d3d12x::frame::{
    activity, frame_statistics_x, mark_flipped, qpc, schedule_frame_event_x, set_frame_interval_x,
    wait_frame_event_x,
};
use xcl::util::{HRESULT, S_OK};

const S_FALSE: HRESULT = 1;
const STARTED_CLOCK: HRESULT = 0x1000_0000;
const E_FAIL: HRESULT = 0x8000_4005u32 as i32;
const E_INVALIDARG: HRESULT = 0x8007_0057u32 as i32;

const WAIT_ALL: u32 = 1;
const TOKEN_ON_TIMEOUT: u32 = 2;
const UNSCHEDULE: u32 = 1;

static SERIAL: Mutex<()> = Mutex::new(());

struct ClockGuard;

impl ClockGuard {
    fn acquire() -> (std::sync::MutexGuard<'static, ()>, Self) {
        let g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            set_frame_interval_x(null_mut(), null_mut(), 0, 0, 0);
        }
        (g, Self)
    }
}

impl Drop for ClockGuard {
    fn drop(&mut self) {
        unsafe {
            set_frame_interval_x(null_mut(), null_mut(), 0, 0, 0);
        }
    }
}

struct EventHandle(HANDLE);

impl EventHandle {
    fn new(manual_reset: bool, initial_state: bool) -> Self {
        let h = unsafe { CreateEventW(null(), manual_reset as i32, initial_state as i32, null()) };
        assert!(!h.is_null(), "CreateEventW failed");
        Self(h)
    }
}

impl Drop for EventHandle {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}

/// Ancillary block layout: `{ u32 count; HANDLE* list @8; i32* signaled @0x10 }`.
fn make_ancillary_block(handles: &[HANDLE], signaled: &mut i32) -> [u8; 0x18] {
    let mut block = [0u8; 0x18];
    block[..4].copy_from_slice(&(handles.len() as u32).to_le_bytes());
    block[8..16].copy_from_slice(&(handles.as_ptr() as usize).to_le_bytes());
    block[16..24].copy_from_slice(&(signaled as *mut i32 as usize).to_le_bytes());
    block
}

fn drain_origin() {
    unsafe {
        while wait_frame_event_x(null_mut(), 0, 0, null(), 0, null_mut()) == S_OK {}
    }
}

#[test]
fn co_scheduled_origin_and_type2_at_offset_zero_both_fire() {
    let (_lock, _guard) = ClockGuard::acquire();
    unsafe {
        assert_eq!(set_frame_interval_x(null_mut(), null_mut(), 16667, 1, 0), S_FALSE);
        let hr = schedule_frame_event_x(null_mut(), 0, 0, null(), 0);
        assert!(hr == STARTED_CLOCK || hr == S_FALSE, "{hr:#x}");
        drain_origin();

        let ev2 = EventHandle::new(false, false);
        let list2 = [ev2.0];
        let mut sig2 = 99i32;
        let block2 = make_ancillary_block(&list2, &mut sig2);
        assert_eq!(schedule_frame_event_x(null_mut(), 2, 0, block2.as_ptr(), 0), S_FALSE);

        // Waiting on type 2 must wake cleanly without starving behind type 0 and must not issue a token.
        let mut sentinel_token = 0x1234_5678u64;
        assert_eq!(
            wait_frame_event_x(null_mut(), 2, 250, null(), 0, &mut sentinel_token),
            S_OK
        );
        assert_eq!(sentinel_token, 0x1234_5678u64, "type 2 must not overwrite token");

        // Type 0 also fired at offset 0 and issued an origin.
        let mut token = 0u64;
        assert_eq!(wait_frame_event_x(null_mut(), 0, 250, null(), 0, &mut token), S_OK);
        assert_ne!(token, 0);

        // The ancillary handle registered on ScheduleFrameEventX for type 2 was signaled too.
        let mut sig_check = 99i32;
        let check_block = make_ancillary_block(&list2, &mut sig_check);
        // Drain type 2 first if already set, or check ev2 directly via an unscheduled/other wait.
        assert_eq!(
            wait_frame_event_x(null_mut(), 2, 250, check_block.as_ptr(), 0, null_mut()),
            S_OK
        );
        assert!(sig_check == 0 || sig_check == -2, "unexpected signaled index {sig_check}");
    }
}

#[test]
fn token_on_timeout_and_token_in_origin_refresh() {
    let (_lock, _guard) = ClockGuard::acquire();
    unsafe {
        assert_eq!(set_frame_interval_x(null_mut(), null_mut(), 16667, 1, 0), S_FALSE);
        let hr = schedule_frame_event_x(null_mut(), 0, 0, null(), 0);
        assert!(hr == STARTED_CLOCK || hr == S_FALSE, "{hr:#x}");
        drain_origin();

        // Timeout with TOKEN_ON_TIMEOUT issues a non-zero token and records signaled = -1.
        let ev = EventHandle::new(true, false);
        let list = [ev.0];
        let mut sig = 42i32;
        let block = make_ancillary_block(&list, &mut sig);
        let mut tok = 0u64;
        assert_eq!(
            wait_frame_event_x(null_mut(), 0, 0, block.as_ptr(), TOKEN_ON_TIMEOUT, &mut tok),
            S_FALSE
        );
        assert_eq!(sig, -1);
        assert_ne!(tok, 0, "TOKEN_ON_TIMEOUT must issue a token");

        // Passing a non-zero token_in together with TOKEN_ON_TIMEOUT is invalid.
        assert_eq!(
            wait_frame_event_x(null_mut(), 0, 0, null(), TOKEN_ON_TIMEOUT, &mut tok),
            E_INVALIDARG
        );

        // Passing an unknown non-zero token_in on type 0 is invalid.
        let mut unknown = 0x00DE_AD00_0000_0001u64;
        assert_eq!(
            wait_frame_event_x(null_mut(), 0, 10, null(), 0, &mut unknown),
            E_INVALIDARG
        );

        // Passing a known non-zero token_in without TOKEN_ON_TIMEOUT refreshes its origin on wake.
        mark_flipped(tok);
        let (mut count, mut stats) = (1u32, [0u8; 0x40]);
        assert_eq!(frame_statistics_x(null_mut(), tok, 0, &mut count, stats.as_mut_ptr()), S_OK);
        let old_origin = i64::from_le_bytes(stats[0x10..0x18].try_into().unwrap());

        let saved_tok = tok;
        assert_eq!(
            wait_frame_event_x(null_mut(), 0, 250, null(), 0, &mut tok),
            S_OK
        );
        assert_eq!(tok, saved_tok, "token_in must be preserved when refreshing origin");

        mark_flipped(tok);
        let mut count2 = 1u32;
        let mut stats2 = [0u8; 0x40];
        assert_eq!(frame_statistics_x(null_mut(), tok, 0, &mut count2, stats2.as_mut_ptr()), S_OK);
        let new_origin = i64::from_le_bytes(stats2[0x10..0x18].try_into().unwrap());
        assert!(new_origin >= old_origin && new_origin <= qpc());
    }
}

#[test]
fn ancillary_signaled_indices_and_wait_all() {
    let (_lock, _guard) = ClockGuard::acquire();
    unsafe {
        // the longest accepted interval: no tick can land between `drain_origin()` and the
        // first ancillary wait below, so the frame event can't win the WaitForMultipleObjects
        // over the already-signaled ancillary handle (index 1)
        assert_eq!(set_frame_interval_x(null_mut(), null_mut(), 66667, 1, 0), S_FALSE);
        let hr = schedule_frame_event_x(null_mut(), 0, 0, null(), 0);
        assert!(hr == STARTED_CLOCK || hr == S_FALSE, "{hr:#x}");
        drain_origin();

        let ev0 = EventHandle::new(true, false);
        let ev1 = EventHandle::new(true, true);
        let list = [ev0.0, ev1.0];
        let mut sig = 99i32;
        let block = make_ancillary_block(&list, &mut sig);

        // Ancillary handle at index 1 is already signaled -> returns S_OK with index 1, no token issued.
        let mut tok = 0u64;
        assert_eq!(
            wait_frame_event_x(null_mut(), 0, 200, block.as_ptr(), 0, &mut tok),
            S_OK
        );
        assert_eq!(sig, 1);
        assert_eq!(tok, 0, "waking on ancillary handle must not issue a frame token");

        // Reset ev1 so both ancillary handles are unsignaled; frame event itself wakes -> index -2.
        ResetEvent(ev1.0);
        sig = 99;
        assert_eq!(
            wait_frame_event_x(null_mut(), 0, 250, block.as_ptr(), 0, &mut tok),
            S_OK
        );
        assert_eq!(sig, -2);
        assert_ne!(tok, 0);

        // WAIT_ALL with an unsignaled ancillary handle must time out (index -1) even as frames tick.
        sig = 99;
        let mut tok_wait_all = 0u64;
        assert_eq!(
            wait_frame_event_x(null_mut(), 0, 30, block.as_ptr(), WAIT_ALL, &mut tok_wait_all),
            S_FALSE
        );
        assert_eq!(sig, -1);
        assert_eq!(tok_wait_all, 0);

        // Signal both ancillary handles; WAIT_ALL waits for the frame event too and reports -2.
        SetEvent(ev0.0);
        SetEvent(ev1.0);
        sig = 99;
        assert_eq!(
            wait_frame_event_x(null_mut(), 0, 250, block.as_ptr(), WAIT_ALL, &mut tok_wait_all),
            S_OK
        );
        assert_eq!(sig, -2);
        assert_ne!(tok_wait_all, 0);
    }
}

#[test]
fn schedule_wait_and_statistics_error_contracts_and_activity_counters() {
    let (_lock, _guard) = ClockGuard::acquire();
    unsafe {
        // Schedule errors while stopped (interval_us == 0).
        assert_eq!(schedule_frame_event_x(null_mut(), 0, 0, null(), 0), E_INVALIDARG);
        assert_eq!(schedule_frame_event_x(null_mut(), 3, 0, null(), 0), E_INVALIDARG);

        assert_eq!(set_frame_interval_x(null_mut(), null_mut(), 16667, 1, 0), S_FALSE);
        // Type != 0 while clock is inactive, and UNSCHEDULE on type 0 or before type 0 is scheduled.
        assert_eq!(schedule_frame_event_x(null_mut(), 1, 0, null(), 0), E_INVALIDARG);
        assert_eq!(schedule_frame_event_x(null_mut(), 0, 0, null(), UNSCHEDULE), E_INVALIDARG);
        assert_eq!(schedule_frame_event_x(null_mut(), 1, 0, null(), UNSCHEDULE), E_INVALIDARG);

        // Invalid ancillary pointers and count > 63.
        let mut sig = 0i32;
        let mut bad_count_block = [0u8; 0x18];
        let bad_ptr = bad_count_block.as_ptr() as usize;
        bad_count_block[..4].copy_from_slice(&64u32.to_le_bytes());
        bad_count_block[8..16].copy_from_slice(&bad_ptr.to_le_bytes());
        bad_count_block[16..24].copy_from_slice(&(&mut sig as *mut i32 as usize).to_le_bytes());
        assert_eq!(
            schedule_frame_event_x(null_mut(), 0, 0, bad_count_block.as_ptr(), 0),
            E_INVALIDARG
        );
        assert_eq!(
            schedule_frame_event_x(null_mut(), 0, 0, 0x1000 as *const u8, 0),
            E_INVALIDARG
        );

        // Unscheduled type wait -> E_INVALIDARG.
        assert_eq!(wait_frame_event_x(null_mut(), 0, 10, null(), 0, null_mut()), E_INVALIDARG);
        assert_eq!(wait_frame_event_x(null_mut(), 3, 10, null(), 0, null_mut()), E_INVALIDARG);

        // Start clock and test bad ancillary block in WaitFrameEventX.
        let hr = schedule_frame_event_x(null_mut(), 0, 0, null(), 0);
        assert!(hr == STARTED_CLOCK || hr == S_FALSE, "{hr:#x}");
        drain_origin();
        assert_eq!(
            wait_frame_event_x(null_mut(), 0, 10, 0x1000 as *const u8, 0, null_mut()),
            E_INVALIDARG
        );

        // GetFrameStatisticsX null pointer checks and mark_flipped(0) no-op.
        let mut count = 1u32;
        let mut stats = [0u8; 0x40];
        assert_eq!(frame_statistics_x(null_mut(), 1, 0, null_mut(), stats.as_mut_ptr()), E_INVALIDARG);
        assert_eq!(frame_statistics_x(null_mut(), 1, 0, &mut count, null_mut()), E_INVALIDARG);
        mark_flipped(0);

        // Measure exact activity() deltas across 1 ok wait, 1 timeout wait, 1 ok stats, 1 miss stats.
        let before = activity();

        let mut tok = 0u64;
        assert_eq!(wait_frame_event_x(null_mut(), 0, 250, null(), 0, &mut tok), S_OK);
        assert_ne!(tok, 0);

        assert_eq!(wait_frame_event_x(null_mut(), 0, 0, null(), 0, null_mut()), S_FALSE);

        mark_flipped(tok);
        count = 0;
        assert_eq!(frame_statistics_x(null_mut(), tok, 2, &mut count, stats.as_mut_ptr()), E_INVALIDARG);
        count = 1;
        assert_eq!(frame_statistics_x(null_mut(), tok, 2, &mut count, stats.as_mut_ptr()), S_OK);
        assert_eq!(count, 1);
        assert_eq!(u32::from_le_bytes(stats[0..4].try_into().unwrap()), 2);

        let mut miss_count = 1u32;
        assert_eq!(
            frame_statistics_x(null_mut(), 0xDEAD_BEEF, 0, &mut miss_count, stats.as_mut_ptr()),
            E_FAIL
        );
        assert_eq!(miss_count, 0);

        let after = activity();
        assert_eq!(
            [
                after[0] - before[0],
                after[1] - before[1],
                after[2] - before[2],
                after[3] - before[3],
            ],
            [2, 1, 3, 1],
            "activity() deltas: waits=2, timeouts=1, stats_calls=3 (1 E_INVALIDARG + 1 S_OK + 1 E_FAIL), stats_misses=1"
        );
    }
}
