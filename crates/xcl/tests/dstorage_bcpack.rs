//! Synthetic BC1 streams submitted through the console DirectStorage interface.

use std::ffi::c_void;
use std::ptr::null_mut;
use std::time::{Duration, Instant};
use windows_sys::core::GUID;
use xcl::hle::dstorage::{stats, DStorageGetFactory};

const FACTORY: GUID = GUID::from_u128(0xfb3b54ac_bf82_463a_b777_73195837864d);
const QUEUE: GUID = GUID::from_u128(0xa3f6649a_8c2c_4b3a_8b6e_10186a04b9bc);
const STATUS_ARRAY: GUID = GUID::from_u128(0xcd66d752_2786_44f7_9070_e2a6fdad3f1b);

#[test]
fn queue_decodes_bc1_with_and_without_zlib() {
    const BLOCKS: usize = 8192;
    const OUTPUT_BYTES: usize = BLOCKS * 8;
    const MEMORY_BC1: u64 = (1 << 22) | (1 << 1);
    let packed = include_bytes!("../../../tests/data/bcpack/bc1-8192-3.bin");
    let zipped = miniz_oxide::deflate::compress_to_vec_zlib(packed, 6);
    let mut outputs = [vec![0xCDu8; OUTPUT_BYTES + 64], vec![0xCDu8; OUTPUT_BYTES + 64]];

    let results = unsafe {
        let mut factory = null_mut();
        assert_eq!(DStorageGetFactory(&FACTORY, &mut factory), 0);
        let fv = *(factory as *const *const usize);
        let create_queue: unsafe extern "system" fn(*mut c_void, *const u8, *const GUID, *mut *mut c_void) -> i32 =
            std::mem::transmute(*fv.add(3));
        let create_status: unsafe extern "system" fn(*mut c_void, u32, *const u8, *const GUID, *mut *mut c_void) -> i32 =
            std::mem::transmute(*fv.add(5));
        let mut desc = [0; 0x18];
        desc[0] = 1; // memory source
        desc[8] = 8; // capacity
        desc[10] = 2; // realtime priority, required for memory queues
        let mut queue = null_mut();
        let mut status = null_mut();
        assert_eq!(create_queue(factory, desc.as_ptr(), &QUEUE, &mut queue), 0);
        assert_eq!(create_status(factory, 2, std::ptr::null(), &STATUS_ARRAY, &mut status), 0);

        let qv = *(queue as *const *const usize);
        let sv = *(status as *const *const usize);
        let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*qv.add(2));
        let release_status: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*sv.add(2));
        let enqueue: unsafe extern "system" fn(*mut c_void, *const u8) = std::mem::transmute(*qv.add(3));
        let enqueue_status: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) = std::mem::transmute(*qv.add(4));
        let submit: unsafe extern "system" fn(*mut c_void) = std::mem::transmute(*qv.add(6));
        let close: unsafe extern "system" fn(*mut c_void) = std::mem::transmute(*qv.add(8));
        let complete: unsafe extern "system" fn(*mut c_void, u32) -> bool = std::mem::transmute(*sv.add(3));
        let get_result: unsafe extern "system" fn(*mut c_void, u32) -> i32 = std::mem::transmute(*sv.add(4));

        for (i, ((options, source, intermediate), output)) in [
            (MEMORY_BC1, packed.as_slice(), 0),
            (MEMORY_BC1 | 1, zipped.as_slice(), packed.len() as u32),
        ].into_iter().zip(&mut outputs).enumerate() {
            let mut request = [0; 0x48];
            request[0..8].copy_from_slice(&options.to_le_bytes());
            request[8..16].copy_from_slice(&(output.as_mut_ptr() as u64).to_le_bytes());
            request[0x18..0x1C].copy_from_slice(&(OUTPUT_BYTES as u32).to_le_bytes());
            request[0x20..0x28].copy_from_slice(&(source.as_ptr() as u64).to_le_bytes());
            request[0x30..0x34].copy_from_slice(&(source.len() as u32).to_le_bytes());
            request[0x34..0x38].copy_from_slice(&intermediate.to_le_bytes());
            enqueue(queue, request.as_ptr());
            enqueue_status(queue, status, i as u32);
        }
        submit(queue);
        let start = Instant::now();
        while !complete(status, 1) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(1));
        }
        if !complete(status, 1) {
            // Closing does not join the worker. Keep its buffers alive if it is still running.
            std::mem::forget((zipped, outputs));
            close(queue);
            release(queue);
            release_status(status);
            panic!("DirectStorage did not complete the synthetic BC1 requests");
        }
        let results = [get_result(status, 0), get_result(status, 1)];
        close(queue);
        release(queue);
        release_status(status);
        results
    };

    for (i, output) in outputs.iter().enumerate() {
        assert_eq!(results[i], 0, "request {i} HRESULT");
        for (block, bytes) in output[..OUTPUT_BYTES].chunks_exact(8).enumerate() {
            // The fixture generator changes an endpoint and two indices every 511 blocks.
            // Every 31st block after the first inherits its endpoint from the previous block.
            let changed = block % 511 == 0;
            let inherits_red = block % 31 == 0 && block % 511 == 1;
            let red = if changed || inherits_red { 0x78 } else { 0 };
            let indices = if changed { 0xAF } else { 0xAA };
            assert_eq!(bytes, [0, red, 0, 0, indices, 0xAA, 0xAA, 0xAA], "request {i}, block {block}");
        }
        assert!(output[OUTPUT_BYTES..].iter().all(|&byte| byte == 0xCD), "request {i} wrote past its destination");
    }
    let store = stats();
    let bc1 = &store.bcpack_unpacked["BC1"];
    assert_eq!((bc1.requests, bc1.failed), (2, 0));
}
