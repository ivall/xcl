//! `xcl disasm <file> [--offset N] [--length N] [--all] [--hex] [--fssm FILE]` – RDNA2
//! disassembler for captured shader blobs (or any binary with console GPU code).
//! Console containers (XDAT/XBCH/FSSM) are unpacked and their layout is printed.

use std::path::Path;

use crate::gpu::{rdna2, shader, xbox};

fn parse_num(s: &str) -> Option<usize> {
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) => usize::from_str_radix(h, 16).ok(),
        None => s.parse().ok(),
    }
}

fn number_option(args: &[String], name: &str) -> Result<Option<usize>, String> {
    let Some(index) = args.iter().position(|arg| arg == name) else { return Ok(None) };
    let value = args.get(index + 1).ok_or_else(|| format!("{name} requires a byte count"))?;
    parse_num(value).map(Some).ok_or_else(|| format!("invalid value for {name}: {value}"))
}

fn print_lines(code: &[u8], base: usize, stop: bool) {
    for l in rdna2::disassemble(code, base as u64, stop) {
        let hex: Vec<String> = l.bytes.iter().map(|b| format!("{b:02x}")).collect();
        let mark = if l.flags & rdna2::F_BRANCH != 0 { "  ; →" } else { "" };
        println!("  {:08x}: {:<52} {}{mark}", l.pc, l.text, hex.join(" "));
    }
}

fn hexdump(data: &[u8], base: usize) {
    for (i, row) in data.chunks(16).enumerate() {
        let hex: Vec<String> = row.iter().map(|b| format!("{b:02x}")).collect();
        let txt: String = row.iter().map(|&b| if (0x20..0x7f).contains(&b) { b as char } else { '.' }).collect();
        println!("    {:06x}: {:<47}  {txt}", base + i * 16, hex.join(" "));
    }
}

fn xdat_label(id: u32) -> &'static str {
    match id {
        xbox::XDAT_HASH => "hash",
        xbox::XDAT_XBCH => "XBCH → FSSM (native code)",
        xbox::XDAT_NESTED_HEADER => "nested DXIL header (NOBC)",
        _ => "?",
    }
}

/// `--fssm FILE`: save `data` there (a later part gets `.N`). `false` when the user asked for a
/// file and it could not be written.
fn save_fssm(name: &str, data: &[u8]) -> bool {
    match std::fs::write(name, data) {
        Ok(()) => {
            println!("  saved FSSM to {name}");
            true
        }
        Err(e) => {
            eprintln!("xcl disasm: {name}: {e}");
            false
        }
    }
}

/// `false` when an FSSM the user asked to save could not be written.
fn print_xbox(sh: &xbox::XboxShader, full_hex: bool, fssm_out: Option<&str>) -> bool {
    println!(
        "console container: stage {} (DXIL), SM {}.{}, XDAT @ +{:#x}",
        sh.stage, sh.shader_model.0, sh.shader_model.1, sh.xdat_offset
    );
    for e in &sh.entries {
        println!("  XDAT {:#04x}  +{:#x}  {} B  {}", e.id, e.offset, e.size, xdat_label(e.id));
    }
    if let Some(h) = sh.hash {
        println!("  hash: {}", h.iter().map(|b| format!("{b:02x}")).collect::<String>());
    }
    for n in &sh.notes {
        println!("  NOTE: {n}");
    }
    let mut saved = true;
    for (i, f) in sh.fssm.iter().enumerate() {
        println!(
            "FSSM #{i}: XBCH v{} @ +{:#x}, zlib {} B → {} B, FSSM v{:#x}",
            f.xbch_version,
            f.xbch_offset,
            f.compressed,
            f.data.len(),
            f.version
        );
        if let Some(e) = &f.error {
            println!("  WARNING: {e}");
        }
        for s in &f.stages {
            println!("  hardware stage: {} – descriptor @ +{:#x}", s.label(), s.offset);
        }
        match f.code {
            Some((o, l)) => println!(
                "  code: +{o:#x}, {l} B ({}), programs: {}",
                if f.code_from_header { "pointer from header +0xC0" } else { "found by scanning" },
                f.programs.len()
            ),
            None => println!("  code: not found"),
        }
        let limit = if full_hex { usize::MAX } else { 0x140 };
        println!("  header – non-zero qwords (without the 0xDEADBEEF filler){}:", if full_hex { "" } else { ", up to +0x140; --hex: all" });
        for (o, q) in xbox::header_qwords(f, limit) {
            println!("    +{o:#05x}: {q:016x}");
        }
        if full_hex {
            if let Some((o, l)) = f.code {
                println!("  data after the code (+{:#x}):", o + l);
                hexdump(&f.data[o + l..], o + l);
            }
        }
        if let Some(out) = fssm_out {
            let name = if i == 0 { out.to_string() } else { format!("{out}.{i}") };
            if !save_fssm(&name, &f.data) {
                saved = false;
            }
        }
    }
    saved
}

pub fn run(path: &Path, args: &[String]) -> i32 {
    let (offset, length) = match (number_option(args, "--offset"), number_option(args, "--length")) {
        (Ok(offset), Ok(length)) => (offset, length),
        (Err(error), _) | (_, Err(error)) => {
            eprintln!("xcl disasm: {error}");
            return 2;
        }
    };
    let fssm_out = match args.iter().position(|a| a == "--fssm") {
        Some(i) => match args.get(i + 1) {
            Some(v) => Some(v.as_str()),
            None => {
                eprintln!("xcl disasm: --fssm needs a file name");
                return 2;
            }
        },
        None => None,
    };
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("xcl disasm: {}: {e}", path.display());
            return 2;
        }
    };
    let all = args.iter().any(|a| a == "--all");
    println!("{} – {} B, decoder: {}", path.display(), data.len(), crate::gpu::RDNA2_ARCH);

    if offset.is_some() || length.is_some() {
        let off = offset.unwrap_or(0);
        if off >= data.len() {
            eprintln!("xcl disasm: offset {off:#x} is past the end of the file");
            return 2;
        }
        let len = length.unwrap_or(data.len() - off).min(data.len() - off);
        print_lines(&data[off..off + len], off, !all);
        return 0;
    }

    let (kind, parts) = shader::classify(&data);
    println!("format: {}", kind.label());
    if !parts.is_empty() {
        println!("container parts:");
        for p in &parts {
            println!("  {}  +{:#x}  {} B", p.fourcc, p.offset, p.size);
        }
    }
    // (a file the user asked to save but could not be: the disassembly is printed, the command
    // still failed)
    let mut code = 0;
    if kind == shader::Kind::XboxNative {
        if let Some(sh) = xbox::parse(&data) {
            if !print_xbox(&sh, args.iter().any(|a| a == "--hex"), fssm_out) {
                code = 2;
            }
        }
    }
    let listing = shader::rdna2_listing(&data);
    if listing.is_empty() {
        println!("no RDNA2 programs found (ending in s_endpgm); use --offset N to force decoding");
        if matches!(kind, shader::Kind::Dxil | shader::Kind::Dxbc) {
            println!("(this is a PC-format shader – no recompilation needed)");
        }
    } else {
        print!("{listing}");
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xcl disasm prog.xbs --fssm out must not report success when it saved nothing: the file
    /// it was told to write (here a directory) is refused, and the exit status says so.
    #[test]
    fn a_failed_fssm_write_exits_2() {
        use crate::gpu::xbox::tests::{fssm, sample, CODE};
        let dir = std::env::temp_dir().join(format!("xcl-disasm-fssm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("prog.xbs");
        std::fs::write(&path, sample(0, &[fssm(&CODE, &[(5, 1)])])).unwrap();

        // the target is a directory: nothing can be written there
        let args = vec!["--fssm".to_string(), dir.to_string_lossy().into_owned()];
        assert_eq!(run(&path, &args), 2, "a failed --fssm write must not exit 0");

        // a writable target: the FSSM is saved and the command succeeds
        let out = dir.join("out.fssm");
        let args = vec!["--fssm".to_string(), out.to_string_lossy().into_owned()];
        assert_eq!(run(&path, &args), 0);
        assert!(std::fs::read(&out).unwrap().starts_with(b"FSSM"));

        // a --fssm without a name is an error, not a silent no-op
        assert_eq!(run(&path, &["--fssm".to_string()]), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}