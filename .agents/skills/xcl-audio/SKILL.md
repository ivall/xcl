---
name: xcl-audio
description: >-
  Implement Xbox audio in XCL: XAudio2 voices, XMA, Opus, ACPHAL, and movie
  audio. Use when a cutscene is silent, a voice format is rejected, or when
  editing hle/xaudio2.rs, hle/xma2/, hle/opus.rs, hle/acphal.rs, or hle/mfplat.rs.
---

# Audio

`hle/xaudio2.rs` is `xaudio2_9.dll` and `xaudio2_9redist.dll`. Create flags outside the host-supported set are dropped. Vtables are patched in place. `CreateSourceVoice` turns XMA and XMA2 into PCM16; Opus `0x704F` is decoded (`hle/opus.rs`) and submitted as PCM. The decoder is Windows `MSOpusDecoder.dll` when that DLL and Media Foundation load, otherwise libopus from `opus.dll` (`%XCL_OPUS_DIR%`, then next to `xcl.exe`: `opus.dll`, `libopus.dll`, `libopus-0.dll`). XCL does not ship libopus. `CreateMasteringVoice` maps a console device id to the default device and retries after `CoInitializeEx` if COM was not started. Unknown exports fall back to system `xaudio2_9.dll`.

Ordinals for Bink (`XAudio2Create` is ordinal 1) are in `stubdb.toml`. Do not renumber them.

`hle/xma2/` is a pure-Rust XMA1/XMA2 decoder. `tables.rs` is data. Inverse-quantizer gains are precomputed. A wrong frame decode should fail the voice test, not play silence and return success. Silence is only for a format the report already classifies as dropped.

`hle/acphal.rs` covers `acphal.dll`, `xapu.dll`, and `chad.dll` (APU memory, the command channel, SHAPE contexts, the decoder queue). Commands are acknowledged and frames tick so the title's audio engine starts; the SHAPE mixer is not run. A reconnect drops messages queued for the old channel. xapu Opus (command kind 3) is decoded into the title's output buffer. XCL does not play that buffer: the title reads it back and plays its own mix. With neither `MSOpusDecoder` nor libopus the decoder writes silence and logs once. `XCL_TRACE_APU=1` logs commands. Unimplemented names return `E_NOTIMPL`.

`hle/mfplat.rs` falls back to system `mfplat.dll` for the rest of Media Foundation.

`report.json` `audio` lists engines, voices, and formats played as silence. A cutscene with `presents > 0` and a rejected format is an audio bug. Movie pictures are the title's codec DLL (Bink 2 is already loaded as a game DLL) plus its display shaders; do not special-case a title's shader names.
