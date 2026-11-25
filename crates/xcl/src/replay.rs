//! `xcl replay <capture folder>`: the shaders of one draw or dispatch captured with
//! `XCL_CAPTURE_DRAW` (`hle::d3d12x::capture`), translated and compiled again with this
//! build's translator. The HLSL goes to `<capture>\replay_<name>\` (`--out <name>`: one plain
//! name); the call is not run yet. Calls with a VS and PS, or a CS, are replayed: the capture
//! also writes a pipeline's GS (the rect-list stand-in), which replay refuses up front.

use std::path::Path;
use std::time::Instant;

use crate::gpu::recomp::pso::{self, Stage, StaticSampler};
use crate::hle::d3d12x::capture::manifest::Manifest;

struct Opts {
    out: String,
}

fn opts(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts { out: String::new() };
    let mut i = 0;
    let value = |i: &mut usize| -> Result<&String, String> {
        *i += 1;
        args.get(*i).ok_or_else(|| format!("{} needs a value", args[*i - 1]))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--out" => o.out = value(&mut i)?.clone(),
            other => return Err(format!("unknown option {other}")),
        }
        i += 1;
    }
    if o.out.is_empty() {
        o.out = format!("{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()));
    }
    // (`replay_<name>` in the capture folder, and nowhere else)
    crate::hle::d3d12x::capture::manifest::plain_name("--out", &o.out)?;
    Ok(o)
}

/// The capture's stages are ones replay runs (VS, PS, CS).
fn replayable(m: &Manifest) -> Result<(), String> {
    match m.shaders.iter().find(|s| !matches!(s.stage.as_str(), "vs" | "ps" | "cs")) {
        Some(s) => Err(format!("the capture has a {} shader: xcl replay runs draws with a VS and PS, or dispatches", s.stage)),
        None => Ok(()),
    }
}

/// `xcl replay <dir> [options]`; the process's exit code (0: done, 1: failed, 2: usage, 3: the
/// call timed out, 4: the device was removed).
pub fn run(dir: &Path, args: &[String]) -> i32 {
    let o = match opts(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("xcl replay: {e} (see xcl --help)");
            return 2;
        }
    };
    match work(dir, &o) {
        Ok(code) => code,
        Err(e) => {
            println!("REPLAY RESULT: failed: {e}");
            1
        }
    }
}

/// A stage of the capture as DXIL.
struct Program {
    stage: Stage,
    name: String,
    dxil: Vec<u8>,
}

/// The capture's shaders translated and compiled; the static samplers of all of them.
fn translate(dir: &Path, out: &Path, m: &Manifest) -> Result<(Vec<Program>, Vec<StaticSampler>), String> {
    let windows = (m.mem_windows & 0xFF).max(1);
    let split = m.split_wave64;
    let mut programs = Vec::new();
    let mut samplers: Vec<StaticSampler> = Vec::new();
    for s in &m.shaders {
        let stage = match s.stage.as_str() {
            "vs" => Stage::Vertex,
            "ps" => Stage::Pixel,
            "cs" => Stage::Compute,
            other => return Err(format!("a console {other} stage: not replayed")),
        };
        let blob = std::fs::read(dir.join(&s.file)).map_err(|e| format!("{}: {e}", s.file))?;
        let t0 = Instant::now();
        let (t, profile) = pso::console_translation(&blob, stage, samplers.len() as u32, windows, split).map_err(|e| format!("{}: {e}", s.stage))?;
        let hlsl = t.hlsl;
        samplers.extend(t.static_samplers);
        std::fs::write(out.join(format!("{}.hlsl", s.stage)), &hlsl).map_err(|e| e.to_string())?;
        let t1 = Instant::now();
        let (dxil, cached) = crate::gpu::recomp::dxc::compile_cached(&hlsl, "main", profile).map_err(|e| format!("{} ({profile}): {e}", s.stage))?;
        println!(
            "REPLAY {} {}: translated in {} ms ({} lines, windows {windows}, split {split}), {profile} DXIL {} bytes in {} ms{}",
            s.stage,
            s.hash,
            (t1 - t0).as_millis(),
            hlsl.lines().count(),
            dxil.len(),
            t1.elapsed().as_millis(),
            if cached { " (cached)" } else { "" }
        );
        if hlsl.contains("SamplerDescriptorHeap") && m.samplers.is_empty() {
            println!("REPLAY {}: WARNING: samples through the sampler heap, and the capture has no samplers (default ones)", s.stage);
        }
        programs.push(Program { stage, name: s.stage.clone(), dxil });
    }
    Ok((programs, samplers))
}

fn work(dir: &Path, o: &Opts) -> Result<i32, String> {
    let m = Manifest::read(dir)?;
    replayable(&m)?;
    let out = dir.join(format!("replay_{}", o.out));
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("REPLAY capture {} call #{}: {:?}; {} memory ranges, {} textures, {} views; output in {}", m.hash, m.call_number, m.call, m.memory.len(), m.textures.len(), m.views.len(), out.display());
    for l in &m.left_out {
        println!("REPLAY note: the capture left out: {l}");
    }
    let (programs, samplers) = translate(dir, &out, &m)?;
    println!("REPLAY RESULT: translated and compiled: {} with {} static sampler(s)", programs.iter().map(|p| format!("{} {:?} {} B", p.name, p.stage, p.dxil.len())).collect::<Vec<_>>().join(", "), samplers.len());
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::xbox::tests::{container, dxil_part, fssm};
    use crate::hle::d3d12x::capture::manifest as c;

    fn code(parts: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
        let mut v: Vec<u8> = parts.iter().flat_map(|(n, e, f, l)| crate::gpu::rdna2::encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}"))).collect();
        v.extend(crate::gpu::rdna2::encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        v
    }

    /// An ISG1 / OSG1 part: (semantic, system value, register, mask) of each float element.
    fn signature(elements: &[(&str, u32, u32, u32)]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&(elements.len() as u32).to_le_bytes());
        v.extend_from_slice(&8u32.to_le_bytes());
        let mut names = Vec::new();
        for (name, sv, register, mask) in elements {
            let at = (8 + 32 * elements.len() + names.len()) as u32;
            for w in [0u32, at, 0, *sv, 3, *register, mask | mask << 8, 0] {
                v.extend_from_slice(&w.to_le_bytes());
            }
            names.extend_from_slice(name.as_bytes());
            names.push(0);
        }
        v.extend(names);
        v
    }

    /// A console container of our own: `program` as the code of hardware stage `hw` (4 VS, 5 PS).
    fn shader(kind: u32, hw: u8, program: &[u8], inputs: &[(&str, u32, u32, u32)], outputs: &[(&str, u32, u32, u32)]) -> Vec<u8> {
        container(&[(b"SFI0", &[0; 8]), (b"ISG1", &signature(inputs)), (b"OSG1", &signature(outputs)), (b"HASH", &[0; 20]), (b"DXIL", &dxil_part(kind, &[fssm(program, &[(hw, 0)])]))])
    }

    const V: u64 = 256;

    /// A capture folder made here, around a VS and a PS of our own (no title's data): the VS
    /// passes POSITION on, the PS writes (s8 as a float, the word at the address in s[10:11], a
    /// count of 3 loop passes, 1) to target 0. A second target and a 3D UAV texture have
    /// contents the call does not touch; a third target is not in the capture.
    pub(super) fn synthetic_capture(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("xcl-replay-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let vs = code(&[
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
            ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ]);
        let ps = code(&[
            ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 5), ("OFFSET", 0)], None),
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 9), ("SSRC0", 128)], None),
            // three passes
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 9), ("SSRC0", 9), ("SSRC1", 129)], None),
            ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 9), ("SSRC1", 131)], None),
            ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 8)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 12)], None),
            ("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 9)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)], None),
            ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
        ]);
        std::fs::write(dir.join("vs.bin"), shader(1, 4, &vs, &[("POSITION", 0, 0, 3)], &[("SV_Position", 1, 0, 0xF)])).unwrap();
        std::fs::write(dir.join("ps.bin"), shader(0, 5, &ps, &[("SV_Position", 1, 0, 0xF)], &[("SV_Target", 64, 0, 0xF)])).unwrap();
        // window 0: a triangle over the whole target at 0x1_0000; window 1 (a reserved buffer in
        // the replay): the word the PS reads, 1 GiB in
        let base = 0x40_0000_0000u64;
        let word_at = base + (1 << 32) + (1 << 30);
        let mut memory: Vec<u8> = [-1.0f32, -1.0, -1.0, 3.0, 3.0, -1.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        memory.resize(0x1000, 0);
        memory.extend_from_slice(&0.25f32.to_le_bytes());
        memory.resize(0x2000, 0);
        std::fs::write(dir.join("memory.bin"), memory).unwrap();
        let mut io = vec![0u32; crate::gpu::recomp::KERNEL_SGPRS];
        io[crate::gpu::recomp::IO_UMA_BASE_HI] = (base >> 32) as u32;
        io[8] = 0.5f32.to_bits();
        (io[10], io[11]) = (word_at as u32, (word_at >> 32) as u32);
        // a target with contents (64 rows of 512 bytes) and a 3D texture (rows of 160 bytes, 256 apart)
        std::fs::write(dir.join("tex_1.bin"), vec![0x3Cu8; 64 * 512]).unwrap();
        std::fs::write(dir.join("tex_2.bin"), vec![1u8; (23 * 192 - 1) * 256 + 160]).unwrap();
        let tex = |id: usize, dimension: i32, width: u64, height: u32, depth: u16, format: i32, flags: i32, file: Option<&str>| c::Texture {
            id,
            dimension,
            width,
            height,
            depth_or_array: depth,
            mips: 1,
            format,
            samples: 1,
            flags,
            file: file.map(String::from),
            subresources: 1,
            ..Default::default()
        };
        // (a pipeline description as the capture keeps it: the structure's bytes; no pointer of it is read)
        use windows::Win32::Graphics::Direct3D12::*;
        let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC { SampleMask: u32::MAX, NumRenderTargets: 3, PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, ..Default::default() };
        d.SampleDesc.Count = 1;
        d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        for k in 0..3 {
            d.RTVFormats[k] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(10);
            d.BlendState.RenderTarget[k].RenderTargetWriteMask = 15;
        }
        let desc = c::hex(unsafe { std::slice::from_raw_parts(&d as *const _ as *const u8, std::mem::size_of_val(&d)) });
        let m = Manifest {
            version: c::VERSION,
            hash: "synthetic".into(),
            call_number: 1,
            call: Some(c::Call::Draw { vertices: 3, instances: 1, first_vertex: 0, first_instance: 0 }),
            graphics: true,
            shaders: vec![c::Shader { stage: "vs".into(), hash: String::new(), file: "vs.bin".into() }, c::Shader { stage: "ps".into(), hash: String::new(), file: "ps.bin".into() }],
            mem_windows: 4,
            split_wave64: true,
            pso: Some(c::Pso { desc, input: vec![c::InputElement { name: "POSITION".into(), index: 0, format: 16, slot: 0, offset: 0, class: 0, step: 0 }], rect_list_gs: false }),
            root: c::Root { io, ..Default::default() },
            uma: c::Uma { base, windows: (0..4).map(|k| (base + (k << 32), (4096u64 << 20) - 0x1_0000)).collect() },
            state: c::State { topology: Some(4), viewports: vec![[0.0, 0.0, 64.0, 64.0, 0.0, 1.0]], scissors: vec![[0, 0, 64, 64]], ..Default::default() },
            targets: vec![c::Target { handle: 1, texture: Some(0), desc: None }, c::Target { handle: 2, texture: Some(1), desc: None }, c::Target { handle: 3, texture: None, desc: None }],
            vertex_buffers: vec![c::VertexBuffer { slot: 0, address: base + 0x1_0000, size: 24, stride: 8 }],
            memory_file: "memory.bin".into(),
            memory: vec![
                c::Range { address: base + 0x1_0000, len: 0x1000, window: 0, offset: 0x1_0000, file_offset: 0, ..Default::default() },
                c::Range { address: word_at, len: 0x1000, window: 1, offset: 1 << 30, file_offset: 0x1000, ..Default::default() },
            ],
            textures: vec![tex(0, 3, 64, 64, 1, 10, 1, None), tex(1, 3, 64, 64, 1, 10, 1, Some("tex_1.bin")), tex(2, 4, 40, 23, 192, 42, 4, Some("tex_2.bin"))],
            views: vec![c::View { index: 20, uav: false, texture: 2, desc: None }, c::View { index: 21, uav: true, texture: 2, desc: None }],
            slots: vec![
                c::Slot { param: 0, offset: 0, words: [20, 0, 0, 0xA000_0000, 0, 0, 0, 0], kind: "image".into() },
                c::Slot { param: 0, offset: 32, words: [30, 0, 0, 0x9000_0000, 0, 0, 0, 0], kind: "image".into() },
            ],
            ..Default::default()
        };
        m.write(&dir).unwrap();
        dir
    }

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    /// The shaders of a capture made here translate and compile, and their HLSL is written.
    #[test]
    fn a_capture_s_shaders_translate_and_compile() {
        if crate::gpu::recomp::dxc::location().is_err() {
            return crate::gpu::recomp::runner::skip_test("no DXC");
        }
        let dir = synthetic_capture("translate");
        assert_eq!(work(&dir, &opts(&args(&["--out", "t"])).unwrap()), Ok(0));
        let out = dir.join("replay_t");
        assert!(std::fs::read_to_string(out.join("vs.hlsl")).unwrap().contains("POSITION"));
        assert!(std::fs::read_to_string(out.join("ps.hlsl")).unwrap().contains("SV_Target0"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn options_parse() {
        assert_eq!(opts(&args(&["--out", "x"])).unwrap().out, "x");
        assert!(!opts(&[]).unwrap().out.is_empty());
        assert!(opts(&args(&["--out"])).is_err());
        assert!(opts(&args(&["--what"])).is_err());
        // the output stays in the capture folder
        for bad in ["x/../../outside", "..", "a\\b", "C:\\x", "/x"] {
            assert!(opts(&args(&["--out", bad])).is_err(), "{bad}");
        }
    }

    /// A capture with a geometry shader is refused before anything is translated or written.
    #[test]
    fn a_capture_with_a_gs_is_refused() {
        let dir = synthetic_capture("gs");
        let mut m = Manifest::read(&dir).unwrap();
        m.shaders.push(crate::hle::d3d12x::capture::manifest::Shader { stage: "gs".into(), hash: "0123456789abcdef".into(), file: "vs.bin".into() });
        m.write(&dir).unwrap();
        let e = work(&dir, &opts(&args(&["--out", "g"])).unwrap()).unwrap_err();
        assert!(e.contains("gs shader"), "{e}");
        assert!(!dir.join("replay_g").exists(), "nothing written");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
