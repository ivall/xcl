//! The console decoder (xapu, `hle::acphal`) on Opus packets, through the exports a title calls:
//! XApuConnect, XApuEnqueueCommand with the commands as the retail title builds them (Activate,
//! Process with one raw Opus packet and a float output buffer, Deactivate), XApuDequeueResult.
//! The commands are written as bytes at their offsets, as the title sends them.
//!
//! Needs an Opus decoder (Windows' MSOpusDecoder, or libopus): without one the tests return early.

use std::ffi::c_void;
use xcl::hle::acphal::{XApuCommand, XApuConnect, XApuConnectInput, XApuConnectOutput, XApuDequeueResult, XApuDisconnect, XApuEnqueueCommand, XApuResult};
use xcl::hle::{opus, xaudio2};

const ACTIVATE: u32 = 1;
const DEACTIVATE: u32 = 2;
const PROCESS: u32 = 3;

// libopus test packets (the same as opus.rs's MONO_HYBRID_20MS): a 440 Hz tone, mono, 20 ms each
const OPUS_440HZ_20MS: &[&[u8]] = &[
    &[0x78, 0x81, 0x7b, 0xc6, 0x11, 0x76, 0xf4, 0x75, 0x00, 0x00, 0x07, 0x8e, 0xca, 0x69, 0xdb, 0xbb, 0xcc, 0x48, 0xdc, 0x16, 0xbd, 0x9b, 0xd3, 0x58, 0x27, 0x78, 0x93, 0xbc, 0x37, 0x44, 0x5e, 0xcf, 0x2d, 0x97, 0x00, 0x69, 0x80, 0x11, 0x80, 0x2c, 0x03, 0x38, 0xab, 0x45, 0x61, 0xfd, 0x89, 0xb7, 0x7e, 0x0d, 0x17, 0xbf, 0x20, 0xa5, 0x2b, 0xb5, 0xea, 0x2b, 0x5e, 0xd3, 0xa7, 0x48, 0xd1, 0x18, 0xf2, 0x0b, 0x5c, 0x01, 0x4f, 0x5f, 0xa1, 0x96, 0x21, 0xae, 0xde, 0xe0, 0x53, 0x00, 0xd6, 0x65, 0x65, 0x3f, 0x1f, 0xa6, 0xcb, 0xb9, 0x77, 0xbf, 0x97, 0x62, 0xc1, 0x12, 0xcf, 0x8d, 0xc2, 0x7d, 0xfd, 0x00, 0x80, 0xff, 0xfa, 0xc6, 0x88, 0xb6, 0xde, 0x10, 0xb8, 0x4f, 0xc6, 0x23, 0x6b, 0xea, 0x87, 0xc9, 0x05],
    &[0x78, 0x9e, 0x1e, 0xa1, 0xe7, 0xfc, 0x95, 0x4d, 0x4e, 0xaa, 0x18, 0x0a, 0xca, 0x5d, 0x07, 0x64, 0x44, 0x34, 0x94, 0x45, 0x3e, 0x66, 0x66, 0xca, 0x44, 0xed, 0x1f, 0x0d, 0x44, 0x6b, 0xf3, 0x4f, 0xcc, 0xa5, 0x55, 0x45, 0x41, 0x9e, 0xb4, 0xf3, 0xfc, 0x13, 0x38, 0x65, 0x79, 0xed, 0x99, 0xf0, 0x1b, 0x7f, 0xbd, 0x3c, 0x92, 0x73, 0xd5, 0xee, 0x8f, 0xeb, 0xab, 0x76, 0x32, 0xa5, 0xf4, 0xa1, 0x0c, 0x35, 0xbe, 0xab, 0xbb, 0x61, 0x99, 0xbf, 0xbd],
    &[0x78, 0x99, 0xc2, 0x5f, 0x75, 0x9c, 0xfc, 0x4b, 0x3a, 0x45, 0xab, 0x5a, 0x7e, 0x7d, 0x7b, 0x37, 0x7c, 0x02, 0xb4, 0x4a, 0x61, 0xf0, 0x89, 0x27, 0x87, 0xce, 0x50, 0xfb, 0xe3, 0xc3, 0x04, 0xcb, 0xaf, 0x99, 0x64, 0x7c, 0x17, 0x1a, 0x33, 0x8d, 0xeb, 0xae, 0xba, 0x19, 0x9b, 0xdb, 0xe1, 0xd7, 0x19, 0xb5, 0xc8, 0xa1, 0x79, 0xd6, 0x1c, 0xb3, 0x68, 0xe5, 0x2b, 0x25, 0x98, 0x0c, 0x5c, 0xbd, 0x64, 0x08, 0x11, 0xbb, 0x14, 0xd0, 0xbb, 0x08, 0x0d, 0xf9, 0xff, 0xdb, 0x05],
    &[0x78, 0x99, 0xc2, 0x5f, 0x73, 0x2d, 0xd2, 0x2d, 0xed, 0xd7, 0x7f, 0xc4, 0x72, 0x9e, 0xe4, 0xeb, 0x27, 0x57, 0xa3, 0x16, 0xf3, 0xb3, 0xb6, 0x2c, 0x03, 0xfc, 0x49, 0x19, 0x83, 0x78, 0x64, 0x9d, 0x41, 0x4c, 0x6a, 0x16, 0xfc, 0x80, 0xea, 0x13, 0xc6, 0x03, 0x49, 0xbf, 0xa5, 0x19, 0x46, 0x91, 0x46, 0x8c, 0xc6, 0x1e, 0x3d, 0x86, 0xd9, 0x6e, 0x57, 0xa8, 0x95, 0x55, 0x90, 0x38, 0x9d, 0xf3, 0x03, 0x88, 0x71, 0x90, 0x83, 0x6f, 0x80, 0x8a, 0x96, 0x45, 0xb9, 0x64, 0xc3, 0xb6, 0x4d],
    &[0x78, 0x99, 0xc2, 0x5f, 0x73, 0x2d, 0xd2, 0x31, 0x33, 0xbf, 0x7b, 0x34, 0x8a, 0x71, 0x0f, 0x77, 0xd5, 0xd5, 0x1b, 0xda, 0x78, 0xa4, 0x47, 0xbb, 0x8c, 0x85, 0x53, 0xbe, 0x21, 0xd6, 0x5f, 0x9f, 0x2e, 0xba, 0x38, 0xfa, 0x2d, 0x32, 0xbc, 0xa1, 0xd7, 0xce, 0xb0, 0x1b, 0x08, 0x44, 0x28, 0x21, 0xcb, 0x80, 0x40, 0xaf, 0xc4, 0x08, 0xdc, 0x96, 0xfd, 0xb6, 0x95, 0x96, 0xda, 0xf3, 0xae, 0x6c, 0x2b, 0x04, 0xc4, 0x23, 0xb8, 0x12, 0x45, 0x51, 0xf9, 0x5f, 0x9b, 0xb0, 0xc6, 0x86, 0xed, 0x08, 0x0c],
];

fn put_u32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// an xapu connection with `streams` streams, 960 frames per output; the handle
unsafe fn connect(streams: u32) -> *mut c_void {
    // process, streams, commands per stream, data bytes, frames per output, options
    let mut input = [0u8; 24];
    put_u32(&mut input, 4, streams);
    put_u32(&mut input, 8, 64);
    put_u32(&mut input, 16, 960);
    // the output: two pointers (data, ready), written by XApuConnect
    let mut output = [0u64; 2];
    let mut h = std::ptr::null_mut();
    let hr = XApuConnect(input.as_ptr() as *const XApuConnectInput, output.as_mut_ptr() as *mut XApuConnectOutput, &mut h);
    assert_eq!(hr, 0, "XApuConnect");
    assert!(!h.is_null());
    h
}

/// sends a command (its bytes); the results it queued
unsafe fn send(h: *mut c_void, command: &[u8]) -> Vec<XApuResult> {
    let mut n = 0;
    assert_eq!(XApuEnqueueCommand(h, command.as_ptr() as *const XApuCommand, &mut n), 0, "XApuEnqueueCommand");
    let mut v = Vec::new();
    let mut r: XApuResult = std::mem::zeroed();
    while XApuDequeueResult(h, &mut r) == 0 {
        v.push(r);
    }
    v
}

/// Activate / Deactivate as the title builds them: the 24-byte header, channels at +24, 48000 at +56
unsafe fn control(h: *mut c_void, kind: u32, stream: u32, channels: u32) {
    let mut c = [0u8; 64];
    put_u32(&mut c, 0, kind);
    put_u32(&mut c, 4, stream);
    put_u32(&mut c, 24, channels);
    put_u32(&mut c, 56, 48_000);
    send(h, &c);
}

/// a Process of one packet into `out_bytes` bytes at `out` (any byte address); the frames its result
/// reports
unsafe fn process(h: *mut c_void, stream: u32, packet: &[u8], first_frame: u32, out: *mut u8, out_bytes: u32) -> u32 {
    // header, ramp +24, pitch +28, first_frame +32, frames +36, input_bytes +40, input +44,
    // output_bytes +52, output +56
    let mut c = [0u8; 64];
    put_u32(&mut c, 0, PROCESS);
    put_u32(&mut c, 4, stream);
    put_u32(&mut c, 28, 48_000f32.to_bits());
    put_u32(&mut c, 32, first_frame);
    put_u32(&mut c, 36, 960);
    put_u32(&mut c, 40, packet.len() as u32);
    put_u64(&mut c, 44, packet.as_ptr() as u64);
    put_u32(&mut c, 52, out_bytes);
    put_u64(&mut c, 56, out as u64);
    let r = send(h, &c);
    let r = r.iter().rev().find(|r| {
        let kind = r.id.kind;
        kind == PROCESS
    });
    let frames = r.expect("a Process result").frames;
    frames
}

/// float32 samples at a byte address that may not be 4-byte aligned
unsafe fn read_f32s(p: *const u8, n: usize) -> Vec<f32> {
    (0..n).map(|i| (p.add(i * 4) as *const f32).read_unaligned()).collect()
}

/// the energy at `freq` (Goertzel), as opus.rs's tests measure their tones
fn tone_energy(samples: &[f32], freq: f64) -> f64 {
    let n = samples.len();
    let k = (0.5 + (n as f64 * freq) / 48_000.0).floor();
    let w = 2.0 * std::f64::consts::PI * k / n as f64;
    let c = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0, 0.0);
    for &x in samples {
        let s0 = x as f64 + c * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - c * s1 * s2).sqrt() / n as f64
}

/// The title's sequence – Activate, a Process per packet with the pre-skip on the first, Deactivate,
/// Activate again – decodes the Opus packets into its output buffer.
#[test]
fn xapu_decodes_the_titles_opus_packets_into_its_buffer() {
    if !opus::is_available() {
        eprintln!("skipping: no Opus decoder");
        return;
    }
    unsafe {
        let h = connect(2);
        control(h, ACTIVATE, 0, 1);
        let before = xaudio2::stats();
        let (mut frames, mut all) = (Vec::new(), Vec::new());
        for round in 0..5 {
            for (i, &packet) in OPUS_440HZ_20MS.iter().enumerate() {
                let mut buf = vec![7.0f32; 960];
                let first = if round == 0 && i == 0 { 312 } else { 0 };
                let f = process(h, 0, packet, first, buf.as_mut_ptr() as *mut u8, 960 * 4);
                assert!(buf[f as usize..].iter().all(|&s| s == 0.0), "past the frames written: silence, not what was there");
                frames.push(f);
                all.extend_from_slice(&buf[..f as usize]);
            }
        }
        assert_eq!(frames[0], 960 - 312, "first_frame (the pre-skip) is left out");
        assert!(frames[1..].iter().all(|&f| f == 960), "a 20 ms packet is 960 frames");
        assert!(all.iter().any(|s| s.abs() > 0.05), "the output is not silent");
        let (e440, e880) = (tone_energy(&all[480..], 440.0), tone_energy(&all[480..], 880.0));
        assert!(e440 > e880 * 10.0, "the 440 Hz tone comes out ({e440} vs {e880})");
        // the counters are global: the other test here may add to them at the same time
        let a = xaudio2::stats();
        assert!(a.xapu_packets_decoded - before.xapu_packets_decoded >= 25, "report.json counts the packets");
        assert_eq!(a.xapu_decode_errors - before.xapu_decode_errors, 0);

        // Deactivate: a Process answers silence of the length asked for
        control(h, DEACTIVATE, 0, 0);
        let mut buf = vec![7.0f32; 960];
        assert_eq!(process(h, 0, OPUS_440HZ_20MS[1], 0, buf.as_mut_ptr() as *mut u8, 960 * 4), 960);
        assert!(buf.iter().all(|&s| s == 0.0), "a deactivated stream gives silence");

        // Activate again with the same layout (the decoder is reset, not rebuilt): the stream starts
        // over, pre-skip and all, and gives what the first decoder gave
        control(h, ACTIVATE, 0, 1);
        let mut again = Vec::new();
        for (i, &packet) in OPUS_440HZ_20MS.iter().enumerate() {
            let mut buf = vec![7.0f32; 960];
            let f = process(h, 0, packet, if i == 0 { 312 } else { 0 }, buf.as_mut_ptr() as *mut u8, 960 * 4);
            assert_eq!(f, if i == 0 { 648 } else { 960 }, "packet {i} after the second Activate");
            again.extend_from_slice(&buf[..f as usize]);
        }
        assert_eq!(again, all[..again.len()], "a reset decoder gives what a new one gave");
        assert_eq!(XApuDisconnect(h), 0);
    }
}

/// The title's output buffer need not be 4-byte aligned, and a stream the connection does not have
/// gets silence without touching stream 0.
#[test]
fn xapu_writes_unaligned_buffers_and_ignores_unknown_streams() {
    if !opus::is_available() {
        eprintln!("skipping: no Opus decoder");
        return;
    }
    unsafe {
        let h = connect(2);
        control(h, ACTIVATE, 0, 1);
        // one byte off a 4-byte boundary, with guard bytes on both sides
        let mut raw = vec![0xAAu8; 960 * 4 + 8];
        let out = raw.as_mut_ptr().add(if raw.as_ptr() as usize % 4 == 0 { 1 } else { 0 });
        assert_eq!(process(h, 0, OPUS_440HZ_20MS[0], 0, out, 960 * 4), 960);
        assert!(read_f32s(out, 960).iter().any(|x| x.abs() > 0.01), "decoded into the unaligned buffer");
        let off = out as usize - raw.as_ptr() as usize;
        assert!(raw[..off].iter().all(|&b| b == 0xAA) && raw[off + 960 * 4..].iter().all(|&b| b == 0xAA), "nothing written outside it");

        // stream 7 on a 2-stream connection: Activate / Deactivate are ignored, a Process is silence
        control(h, ACTIVATE, 7, 2);
        let mut buf = vec![7.0f32; 960];
        assert_eq!(process(h, 7, OPUS_440HZ_20MS[1], 0, buf.as_mut_ptr() as *mut u8, 960 * 4), 960, "the length asked, mono like stream 0");
        assert!(buf.iter().all(|&s| s == 0.0), "an unknown stream gives silence");
        control(h, DEACTIVATE, 7, 0);
        // stream 0 keeps its decoder and its state: the next packet decodes in sequence
        let mut buf = vec![0.0f32; 960];
        assert_eq!(process(h, 0, OPUS_440HZ_20MS[1], 0, buf.as_mut_ptr() as *mut u8, 960 * 4), 960);
        assert!(buf.iter().any(|x| x.abs() > 0.01), "stream 0 still decodes");
        assert_eq!(XApuDisconnect(h), 0);
    }
}
