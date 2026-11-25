//! DualSense haptics from the title's rumble. Over USB the controller is also a 4-channel
//! audio output whose channels 3 and 4 drive the left and right voice-coil actuators; a
//! thread plays waveforms made from the two rumble levels there (the big, slow motor as a
//! low tone mostly on the left, the small, fast one as a higher tone mostly on the right).
//! Channels 1 and 2 (the speaker and headset jack) stay silent.
//!
//! It needs the controller's audio output in 4 channels (Windows' sound settings: the
//! controller's speaker configuration "Quadraphonic"); otherwise `start` fails and the
//! caller keeps compatible rumble.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

/// the left (big) motor's tone and the right (small) one's, Hz
const LOW_HZ: f32 = 100.0;
const HIGH_HZ: f32 = 230.0;
/// how much of each motor reaches the other side's actuator
const CROSS: f32 = 0.25;

/// A phase-continuous two-tone generator whose amplitudes glide to their targets (no clicks
/// when the title changes the levels).
#[derive(Default, Clone, Debug)]
pub struct Synth {
    phase_lo: f32,
    phase_hi: f32,
    amp_lo: f32,
    amp_hi: f32,
}

impl Synth {
    /// `frames` frames of `channels` interleaved samples at `rate` Hz into `out`; the actuators
    /// are channels 2 and 3 (0-based), the others get silence.
    pub fn fill(&mut self, out: &mut [f32], channels: usize, rate: f32, low: f32, high: f32) {
        let frames = out.len() / channels.max(1);
        let (low, high) = (low.clamp(0.0, 1.0), high.clamp(0.0, 1.0));
        // glide over about 10 ms
        let k = (1.0 / (0.010 * rate)).min(1.0);
        let tau = std::f32::consts::TAU;
        for f in 0..frames {
            self.amp_lo += (low - self.amp_lo) * k;
            self.amp_hi += (high - self.amp_hi) * k;
            self.phase_lo = (self.phase_lo + tau * LOW_HZ / rate) % tau;
            self.phase_hi = (self.phase_hi + tau * HIGH_HZ / rate) % tau;
            let lo = self.amp_lo * self.phase_lo.sin();
            let hi = self.amp_hi * self.phase_hi.sin();
            let frame = &mut out[f * channels..(f + 1) * channels];
            frame.fill(0.0);
            if channels >= 4 {
                frame[2] = ((1.0 - CROSS) * lo + CROSS * hi).clamp(-1.0, 1.0);
                frame[3] = ((1.0 - CROSS) * hi + CROSS * lo).clamp(-1.0, 1.0);
            }
        }
    }
}

/// the levels the thread plays (f32 bits)
static LOW: AtomicU32 = AtomicU32::new(0);
static HIGH: AtomicU32 = AtomicU32::new(0);
static RUNNING: AtomicBool = AtomicBool::new(false);
static STOP: AtomicBool = AtomicBool::new(false);

/// Sets what the actuators play (0..1).
pub fn set(low: f32, high: f32) {
    LOW.store(low.to_bits(), Ordering::Relaxed);
    HIGH.store(high.to_bits(), Ordering::Relaxed);
}

/// Whether the haptics thread is playing.
pub fn running() -> bool {
    RUNNING.load(Ordering::Acquire)
}

/// Stops the thread (it notices within a buffer).
pub fn stop() {
    set(0.0, 0.0);
    STOP.store(true, Ordering::Release);
}

/// Starts playing on the audio output whose name contains `name`; waits until it plays or
/// fails (true: playing).
pub fn start(name: &str) -> bool {
    if running() {
        return true;
    }
    static STARTING: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
    let _g = STARTING.get_or_init(Default::default).lock().unwrap();
    if running() {
        return true;
    }
    STOP.store(false, Ordering::Release);
    let (tx, rx) = std::sync::mpsc::channel();
    let name = name.to_string();
    let spawned = std::thread::Builder::new().name("xcl-dualsense-haptics".into()).spawn(move || {
        crate::eh::arm_current_thread();
        // SAFETY: COM and WASAPI calls on this thread's own objects
        if let Err(e) = unsafe { play(&name, &tx) } {
            let _ = tx.send(Err(e));
        }
        RUNNING.store(false, Ordering::Release);
    });
    if spawned.is_err() {
        return false;
    }
    match rx.recv_timeout(Duration::from_secs(3)) {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            crate::xlog_info!("GameInput: DualSense haptics not available ({e}); compatible rumble instead");
            false
        }
        Err(_) => false,
    }
}

unsafe fn play(name: &str, ready: &std::sync::mpsc::Sender<Result<(), String>>) -> Result<(), String> {
    use windows::Win32::Media::Audio::*;
    use windows::Win32::System::Com::StructuredStorage::PropVariantToStringAlloc;
    use windows::Win32::System::Com::*;

    let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
    let e: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(|e| e.to_string())?;
    let all = e.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE).map_err(|e| e.to_string())?;
    // PKEY_Device_FriendlyName
    let key = windows::Win32::Foundation::PROPERTYKEY { fmtid: windows::core::GUID::from_u128(0xa45c254e_df1c_4efd_8020_67d146a850e0), pid: 14 };
    let want = name.to_ascii_lowercase();
    let mut found = None;
    for i in 0..all.GetCount().unwrap_or(0) {
        let Ok(d) = all.Item(i) else { continue };
        let Ok(props) = d.OpenPropertyStore(STGM_READ) else { continue };
        let Ok(v) = props.GetValue(&key) else { continue };
        let Ok(s) = PropVariantToStringAlloc(&v) else { continue };
        let n = s.to_string().unwrap_or_default();
        CoTaskMemFree(Some(s.0 as *const _));
        if n.to_ascii_lowercase().contains(&want) {
            found = Some((d, n));
            break;
        }
    }
    let (dev, devname) = found.ok_or_else(|| format!("no audio output named like {name:?}: the controller is not on USB, or its audio is off"))?;
    let client: IAudioClient = dev.Activate(CLSCTX_ALL, None).map_err(|e| e.to_string())?;
    let fmt = client.GetMixFormat().map_err(|e| e.to_string())?;
    let (channels, rate, bits) = ((*fmt).nChannels as usize, (*fmt).nSamplesPerSec as f32, (*fmt).wBitsPerSample);
    // float (WAVE_FORMAT_IEEE_FLOAT, or EXTENSIBLE with 32 bits: the shared-mode mix format is float)
    let float = (*fmt).wFormatTag == 3 || ((*fmt).wFormatTag == 0xFFFE && bits == 32);
    if channels < 4 {
        CoTaskMemFree(Some(fmt as *const _));
        return Err(format!("{devname:?} plays {channels} channels; set its speaker configuration to Quadraphonic in Windows' sound settings"));
    }
    if !float && bits != 16 {
        CoTaskMemFree(Some(fmt as *const _));
        return Err(format!("{devname:?}: a {bits}-bit mix format"));
    }
    // a 40 ms buffer
    let r = client.Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 400_000, 0, fmt, None);
    CoTaskMemFree(Some(fmt as *const _));
    r.map_err(|e| e.to_string())?;
    let size = client.GetBufferSize().map_err(|e| e.to_string())?;
    let render: IAudioRenderClient = client.GetService().map_err(|e| e.to_string())?;
    client.Start().map_err(|e| e.to_string())?;
    RUNNING.store(true, Ordering::Release);
    crate::xlog_info!("GameInput: DualSense haptics play on {devname:?} ({channels} channels, {rate} Hz)");
    let _ = ready.send(Ok(()));
    let mut synth = Synth::default();
    let mut buf = Vec::new();
    while !STOP.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(5));
        let Ok(padding) = client.GetCurrentPadding() else { break };
        let n = size.saturating_sub(padding) as usize;
        if n == 0 {
            continue;
        }
        let Ok(p) = render.GetBuffer(n as u32) else { break };
        buf.resize(n * channels, 0.0f32);
        synth.fill(&mut buf, channels, rate, f32::from_bits(LOW.load(Ordering::Relaxed)), f32::from_bits(HIGH.load(Ordering::Relaxed)));
        if float {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), p as *mut f32, buf.len());
        } else {
            let out = std::slice::from_raw_parts_mut(p as *mut i16, buf.len());
            for (o, s) in out.iter_mut().zip(&buf) {
                *o = (s * 32767.0) as i16;
            }
        }
        if render.ReleaseBuffer(n as u32, 0).is_err() {
            break;
        }
    }
    let _ = client.Stop();
    crate::xlog_info!("GameInput: DualSense haptics stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synth_drives_only_the_actuator_channels() {
        let mut s = Synth::default();
        let mut out = vec![1.0f32; 4 * 4800];
        s.fill(&mut out, 4, 48000.0, 0.0, 0.0);
        assert!(out.iter().all(|&v| v == 0.0), "no rumble: silence");
        s.fill(&mut out, 4, 48000.0, 1.0, 0.0);
        let peak = |c: usize| out.chunks(4).skip(2400).map(|f| f[c].abs()).fold(0.0f32, f32::max);
        assert_eq!((peak(0), peak(1)), (0.0, 0.0), "speaker channels stay silent");
        assert!(peak(2) > 0.7, "the big motor on the left actuator");
        assert!(peak(3) > 0.2 && peak(3) < 0.3, "a little of it on the right");
        // the amplitude glides: the first sample after a jump is small
        let mut s = Synth::default();
        let mut one = vec![0.0f32; 4];
        s.fill(&mut one, 4, 48000.0, 1.0, 1.0);
        assert!(one[2].abs() < 0.01);
        // fewer than 4 channels: nothing to drive
        let mut st = vec![1.0f32; 2 * 100];
        Synth::default().fill(&mut st, 2, 48000.0, 1.0, 1.0);
        assert!(st.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn synth_tones_have_their_frequencies() {
        // zero crossings of the right actuator with only the small motor: 2 × 230 a second
        let mut s = Synth::default();
        let mut out = vec![0.0f32; 4 * 48000];
        s.fill(&mut out, 4, 48000.0, 0.0, 1.0);
        let ch: Vec<f32> = out.chunks(4).map(|f| f[3]).skip(4800).collect();
        let crossings = ch.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        let expect = 2.0 * HIGH_HZ * 0.9;
        assert!((crossings as f32 - expect).abs() < 4.0, "{crossings} vs {expect}");
    }
}
