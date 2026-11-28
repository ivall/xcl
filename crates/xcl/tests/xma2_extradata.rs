//! XMA1/XMA2 WAVEFORMATEX extradata (`hle::xma2::StreamConfig`), the layout XAudio2's
//! `xma_stream_config` hands the decoder for every XMA voice. Formats per xma2defs.h and FFmpeg's
//! `wmaprodec.c` `decode_init` (legacy XMA2WAVEFORMAT and XMAWAVEFORMAT offsets).

use std::panic::catch_unwind;
use xcl::hle::xma2::{Decoder, StreamConfig, Version, MAX_CHANNELS};

/// A real XMA2WAVEFORMATEX tail (cbSize 34): NumStreams, ChannelMask, SamplesEncoded,
/// BytesPerBlock, PlayBegin/Length, LoopBegin/Length, LoopCount, EncoderVersion, BlockCount.
fn xma2wfx(num_streams: u16, mask: u32) -> Vec<u8> {
    let mut e = num_streams.to_le_bytes().to_vec();
    for v in [mask, 96_000, 0x8000, 0, 96_000, 0, 0] {
        e.extend(v.to_le_bytes());
    }
    e.extend([255, 4, 1, 0]);
    assert_eq!(e.len(), 34);
    e
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// XMA2WAVEFORMATEX's NumStreams decides the layout; every other field is ignored; 0 streams
/// and empty extradata both give the 2+2+...+1 layout; the rate passes through, 0 becomes 48 kHz.
#[test]
fn xma2waveformatex_layouts() {
    let wfx = |ch, ns, rate| StreamConfig::from_xma2_wfx(ch, rate, &xma2wfx(ns, 0x3F)).unwrap();
    assert_eq!(wfx(1, 1, 48_000).stream_channels, vec![1]);
    assert_eq!(wfx(2, 1, 44_100).stream_channels, vec![2]);
    assert_eq!(wfx(2, 2, 44_100).stream_channels, vec![1, 1]);
    assert_eq!(wfx(6, 3, 48_000).stream_channels, vec![2, 2, 2]);
    assert_eq!(wfx(6, 6, 48_000).stream_channels, vec![1; 6]);
    assert_eq!(wfx(8, 4, 48_000).stream_channels, vec![2; 4]);
    assert_eq!(wfx(16, 8, 48_000).stream_channels, vec![2; 8]);
    assert_eq!(wfx(7, 0, 48_000).stream_channels, vec![2, 2, 2, 1]);
    for rate in [1, 8_000, 11_025, 22_050, 44_100, 96_000, u32::MAX] {
        let cfg = wfx(2, 1, rate);
        assert_eq!((cfg.version, cfg.sample_rate, cfg.channels), (Version::Xma2, rate, 2));
        assert_eq!(Decoder::new(cfg).unwrap().sample_rate(), rate);
    }
    assert_eq!(wfx(2, 1, 0).sample_rate, 48_000);
    for ch in 1..=16u16 {
        let cfg = StreamConfig::from_xma2_wfx(ch, 48_000, &[]).unwrap();
        assert_eq!(cfg.stream_channels.len(), (ch as usize).div_ceil(2));
        assert_eq!(cfg.stream_channels.iter().map(|&c| c as u16).sum::<u16>(), ch);
        assert_eq!(Decoder::new(cfg).unwrap().channels(), ch);
    }
}

/// Channel counts, stream counts and extradata sizes no layout fits are Err, not a panic.
#[test]
fn invalid_extradata_is_err() {
    let bad = [
        (0, vec![]),
        (17, vec![]),
        (u16::MAX, vec![]),
        (0, xma2wfx(1, 4)),
        (2, xma2wfx(9, 3)),
        (2, xma2wfx(u16::MAX, 3)),
        (2, xma2wfx(3, 3)),    // 3 streams need 3+ channels
        (6, xma2wfx(2, 0x3F)), // 2 streams carry at most 4
        (2, vec![4]),
        (2, xma2wfx(1, 3)[..33].to_vec()),
        (2, [xma2wfx(1, 3), vec![0]].concat()),
        (2, vec![4, 1, 0, 0]),
        (2, vec![4, 0, 0, 0]),
    ];
    for (ch, extra) in bad {
        assert!(StreamConfig::from_xma2_wfx(ch, 48_000, &extra).is_err(), "XMA2 ch={ch} {extra:?}");
    }
    // Legacy XMA2WAVEFORMAT (rev 4: 40-byte header + 4 per stream, channels at byte 0 of each).
    let mut legacy = vec![0u8; 44];
    legacy[..2].copy_from_slice(&[4, 1]);
    for (stream_ch, total, ok) in [(2, 2, true), (1, 1, true), (0, 0, false), (3, 3, false), (2, 1, false)] {
        legacy[40] = stream_ch;
        assert_eq!(StreamConfig::from_xma2_wfx(total.max(1), 48_000, &legacy).is_ok(), ok, "legacy {stream_ch}");
    }
    // XMAWAVEFORMAT: NumStreams at 4, then 20-byte stream entries from 8 with Channels at +17.
    let mut xma1 = vec![0u8; 28];
    xma1[4] = 1;
    xma1[25] = 2;
    assert_eq!(StreamConfig::from_xma1_wfx(2, 0, &xma1).unwrap().stream_channels, vec![2]);
    assert!(StreamConfig::from_xma1_wfx(1, 48_000, &xma1).is_err());
    assert!(StreamConfig::from_xma1_wfx(2, 48_000, &xma1[..27]).is_err());
    assert!(StreamConfig::from_xma1_wfx(2, 48_000, &xma1[..5]).is_err());
    xma1[25] = 3;
    assert!(StreamConfig::from_xma1_wfx(3, 48_000, &xma1).is_err());
    xma1[4] = 9;
    assert!(StreamConfig::from_xma1_wfx(2, 48_000, &xma1).is_err());
}

/// Random extradata of every size up to 300 bytes never panics, and any layout it accepts is one
/// the decoder takes: 1..=8 streams of 1 or 2 channels summing to the channel count.
#[test]
fn random_extradata_never_panics() {
    for seed in 0..3000u64 {
        let mut r = Lcg(seed);
        let sizes = [0, 2, 33, 34, 35, 36, 40, 44, 48, 68];
        let len = if seed % 10 == 0 { r.next() as usize % 300 } else { sizes[seed as usize / 10 % 10] };
        let mut extra: Vec<u8> = (0..len).map(|_| (r.next() % 5) as u8).collect();
        if len >= 2 && seed % 3 == 0 {
            extra[0] = [3, 4][seed as usize % 2];
        }
        let ch = (r.next() % 19) as u16;
        let res = catch_unwind(|| {
            let parsed = [StreamConfig::from_xma2_wfx(ch, r.0 as u32, &extra), StreamConfig::from_xma1_wfx(ch, 0, &extra)];
            for cfg in parsed {
                if let Ok(cfg) = cfg {
                    assert!((1..=8).contains(&cfg.stream_channels.len()) && cfg.channels as usize <= MAX_CHANNELS);
                    assert!(cfg.stream_channels.iter().all(|&c| c == 1 || c == 2));
                    assert_eq!(cfg.stream_channels.iter().map(|&c| c as u16).sum::<u16>(), ch);
                    Decoder::new(cfg).unwrap();
                }
            }
        });
        assert!(res.is_ok(), "seed {seed}: ch={ch} extra={extra:?}");
    }
}
