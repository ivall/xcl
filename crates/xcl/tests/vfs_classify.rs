//! How `Vfs::classify_with` sorts the paths a title opens (design §5.4): console mounts, the
//! game folder on the PC, verbatim prefixes, devices and UNC, and XCL's own folders on a drive
//! a console mount names. Pure path logic only: the absolutiser is injected (`full` below,
//! GetFullPathNameW's rules for the inputs used here), the data folder does not exist (no
//! whiteout file is read), and nothing here opens, stages or deletes a file. Every expected
//! Target is the same whichever drive letters the machine has (only C: is assumed).

use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use xcl::vfs::{Target, Vfs};

const ROOT: &str = r"C:\Games\Title";
/// Never created, so `Vfs::new` finds no whiteout file.
const DATA: &str = r"C:\XclClassifyTest\Data\Title";
/// The title's current directory, inside the game folder.
const CWD: &str = r"C:\Games\Title\bin";

/// GetFullPathNameW for the inputs used here: '/' is a separator, repeated separators collapse,
/// "." and ".." are resolved (never above the drive root), "\x" is on the current drive, "C:x"
/// is relative to the current directory and "Q:x" (another drive) to that drive's root, a
/// relative path to the current directory; a trailing separator is kept.
fn full(p: &str) -> Option<String> {
    let s = p.replace('/', "\\");
    if s.starts_with("\\\\") {
        return Some(s);
    }
    let b = s.as_bytes();
    let (root, base, rest): (String, &str, &str) = if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
        let drive = &s[..2];
        if s[2..].starts_with('\\') {
            (format!("{drive}\\"), "", &s[3..])
        } else if drive.eq_ignore_ascii_case("C:") {
            ("C:\\".into(), &CWD[3..], &s[2..])
        } else {
            (format!("{drive}\\"), "", &s[2..])
        }
    } else if let Some(r) = s.strip_prefix('\\') {
        ("C:\\".into(), "", r)
    } else {
        ("C:\\".into(), &CWD[3..], s.as_str())
    };
    let mut parts: Vec<&str> = base.split('\\').filter(|c| !c.is_empty()).collect();
    for c in rest.split('\\') {
        match c {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            c => parts.push(c),
        }
    }
    let trailing = if s.ends_with('\\') && !parts.is_empty() { "\\" } else { "" };
    Some(format!("{root}{}{trailing}", parts.join("\\")))
}

/// The default mounts (config.rs), written the various ways a config file may spell them, plus a
/// directory-style prefix.
fn console() -> Vfs {
    let mounts = vec![
        ("GAME:".to_string(), ROOT.to_string()),
        ("g:".to_string(), ROOT.to_string()),
        ("D:\\".to_string(), ROOT.to_string()),
        ("T:".to_string(), "C:/XclClassifyTest/Data/Title/temp/".to_string()),
        ("S:".to_string(), format!("{DATA}\\pls")),
        ("G:\\Saves".to_string(), format!("{DATA}\\saves")),
    ];
    Vfs::new(Path::new(ROOT), Path::new(DATA), mounts)
}

fn game(rel: &str) -> Target {
    Target::Game { rel: rel.to_string() }
}

fn tr(p: &str) -> Target {
    Target::Translated(p.to_string())
}

#[track_caller]
fn check(v: &Vfs, cases: &[(&str, Target)]) {
    for (path, want) in cases {
        assert_eq!(&v.classify_with(path, full), want, "classify({path:?})");
    }
}

/// G:, GAME: and the console's D: all name the game folder: with or without a separator after
/// the colon ("G:foo" is "G:\foo"), any case, '/' or doubled separators, a trailing separator,
/// and the mount itself is the root (rel ""). rel keeps the title's own spelling.
#[test]
fn console_drive_mounts_name_the_game_folder() {
    let v = console();
    check(&v, &[
        (r"G:\data\hello.txt", game(r"data\hello.txt")),
        (r"g:/data/hello.txt", game(r"data\hello.txt")),
        (r"G:data\hello.txt", game(r"data\hello.txt")),
        (r"G:\\data\\\hello.txt", game(r"data\hello.txt")),
        (r"GAME:\Data\Level_01.PAK", game(r"Data\Level_01.PAK")),
        (r"game:Data", game("Data")),
        (r"D:\hunter\shaders.bin", game(r"hunter\shaders.bin")),
        (r"d:hunter", game("hunter")),
        (r"G:\data\", game("data")),
        (r"G:\data\.\maps\..\hello.txt", game(r"data\hello.txt")),
        ("G:", game("")),
        (r"G:\", game("")),
        ("GAME:", game("")),
        (r"GAME:\\", game("")),
        ("D:/", game("")),
    ]);
}

/// T: and S: go to their folders under XCL's data, the target spelled with '/' and a trailing
/// separator in the config still comes out as one clean Windows path; none of these is
/// recorded as an unmapped console path.
#[test]
fn writable_mounts_translate_to_their_folders() {
    let v = console();
    let temp = format!("{DATA}\\temp");
    check(&v, &[
        (r"T:\scratch.bin", tr(&format!("{temp}\\scratch.bin"))),
        (r"t:/a/B.tmp", tr(&format!("{temp}\\a\\B.tmp"))),
        ("T:scratch.bin", tr(&format!("{temp}\\scratch.bin"))),
        ("T:", tr(&temp)),
        (r"T:\", tr(&temp)),
        (r"S:\profile\slot0.dat", tr(&format!("{DATA}\\pls\\profile\\slot0.dat"))),
        (r"s:\", tr(&format!("{DATA}\\pls"))),
    ]);
    check(&v, &[(r"G:\x", game("x")), (r"C:\Windows\notepad.exe", Target::Pass)]);
    assert_eq!(v.unmapped_paths(), Vec::<String>::new());
}

/// A directory-style prefix ("G:\Saves") beats the drive mount it starts with, but only on a
/// component boundary: "G:\SavesOld" and "G:\Save" stay in the game folder.
#[test]
fn the_longest_mount_prefix_wins_on_a_component_boundary() {
    let v = console();
    let saves = format!("{DATA}\\saves");
    check(&v, &[
        (r"G:\Saves\slot1.sav", tr(&format!("{saves}\\slot1.sav"))),
        (r"g:/SAVES/slot1.sav", tr(&format!("{saves}\\slot1.sav"))),
        (r"G:\Saves", tr(&saves)),
        (r"G:\Saves\", tr(&saves)),
        (r"G:\SavesOld\slot1.sav", game(r"SavesOld\slot1.sav")),
        (r"G:\Save", game("Save")),
        (r"G:\data\Saves\x", game(r"data\Saves\x")),
    ]);
}

/// "GAME:\x" treated as relative by a runtime arrives with a current directory in front; a
/// component ending in ':' can only be a console mount, wherever it sits, in any case.
#[test]
fn a_mount_after_a_current_directory_is_still_the_mount() {
    let v = console();
    check(&v, &[
        (r"C:\Games\Title\bin\GAME:\data\x.pak", game(r"data\x.pak")),
        ("C:/Users/me/AppData/game:/data/x.pak", game(r"data\x.pak")),
        (r"\??\C:\somewhere\G:\Data\X.pak", game(r"Data\X.pak")),
        (r"C:\Users\me\D:\hunter\x", game(r"hunter\x")),
        (r"C:\cwd\GAME:", game("")),
        (r"C:\cwd\GAME:\", game("")),
        (r"C:\cwd\T:\log.txt", tr(&format!("{DATA}\\temp\\log.txt"))),
        // a component only containing the name is not the mount
        (r"C:\Games\Title\bin\MyGAME:\x", game(r"bin\MyGAME:\x")),
    ]);
}

/// The game folder on the PC: the folder itself is rel "", any case or trailing separator;
/// relative, root-relative and drive-relative paths resolve from the current directory; a
/// sibling whose name starts with the root's name ("Title2", "Title.bak") is not the game.
#[test]
fn pc_paths_inside_and_next_to_the_game_folder() {
    let v = console();
    check(&v, &[
        (ROOT, game("")),
        (r"C:\Games\Title\", game("")),
        (r"c:\GAMES\title", game("")),
        (r"C:\GAMES\TITLE\Data\Level_01.PAK", game(r"Data\Level_01.PAK")),
        ("C:/Games/Title/data/x", game(r"data\x")),
        (r"data\x", game(r"bin\data\x")),
        (r".\x", game(r"bin\x")),
        (r"..\data\x", game(r"data\x")),
        ("..", game("")),
        (r"\Games\Title\x", game("x")),
        ("C:x", game(r"bin\x")),
        (r"C:\Games\Title\bin\..\..\Title\data", game("data")),
        (r"..\..\Title2\x", Target::Pass),
        (r"C:\Games\Title2\x", Target::Pass),
        (r"C:\Games\Title2", Target::Pass),
        (r"C:\Games\Title.bak\x", Target::Pass),
        (r"C:\Games\Titl", Target::Pass),
        (r"C:\Games", Target::Pass),
        (r"C:\Games\Title\..\Title2\x", Target::Pass),
        (r"C:\XclClassifyTest\Data\Title\logs\xcl.log", Target::Pass),
    ]);
}

/// \\?\ and \??\ are taken off before anything else (also written with '/'), so a verbatim
/// game or console path is the same as the plain one; \\?\UNC\ is a share.
#[test]
fn verbatim_prefixes_are_taken_off() {
    let v = console();
    check(&v, &[
        (r"\\?\C:\Games\Title\data\x", game(r"data\x")),
        (r"\??\c:\games\title\Data\X", game(r"Data\X")),
        (r"\\?\C:\Games\Title", game("")),
        (r"\\?\C:\Games\Title\", game("")),
        ("//?/C:/Games/Title/data/x", game(r"data\x")),
        (r"\\?\G:\data\x", game(r"data\x")),
        (r"\??\T:\x", tr(&format!("{DATA}\\temp\\x"))),
        (r"\\?\C:\Games\Title2\x", Target::Pass),
        (r"\\?\UNC\server\share\Games\Title\x", Target::Pass),
    ]);
}

/// Devices (DOS names, \\.\, device interfaces, volumes, GLOBALROOT) and UNC shares pass without
/// being absolutised; names that only look like devices are files.
#[test]
fn devices_and_shares_pass_without_being_absolutised() {
    let v = console();
    let no_full = |p: &str| -> Option<String> { panic!("absolutised {p:?}") };
    for p in [
        "CON", "con", "NUL", "nul:", "PRN", "AUX", "COM1", "com9:", "LPT1", "LPT9", "CONIN$", "conout$", "CLOCK$",
        r"\\.\pipe\xcl", "//./pipe/xcl", r"\\.\COM3", r"\\.\PhysicalDrive0",
        r"\\?\hid#vid_045e&pid_0b13&ig_00#9&2b3c4d&0&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}",
        r"\\?\Volume{0b7a2a3e-1f4c-4d8e-9c1a-2f3e4d5c6b7a}\Games\Title\x",
        r"\??\Volume{0b7a2a3e-1f4c-4d8e-9c1a-2f3e4d5c6b7a}",
        r"\\?\GLOBALROOT\Device\HarddiskVolume3\Games\Title\x",
        r"\\server\share\Games\Title\x", "//server/share/x", r"\\?\UNC\server\share\x",
        "",
    ] {
        assert_eq!(v.classify_with(p, no_full), Target::Pass, "classify({p:?})");
    }
    check(&v, &[
        ("COM10", game(r"bin\COM10")),
        ("LPTX", game(r"bin\LPTX")),
        ("console", game(r"bin\console")),
        ("COM", game(r"bin\COM")),
    ]);
}

/// A drive letter no mount names is the PC's: it passes whether or not the drive exists (the
/// game folder is on C:), even with the game folder's path on it.
#[test]
fn drives_without_a_mount_pass() {
    let v = console();
    check(&v, &[
        (r"Q:\x", Target::Pass),
        ("q:foo", Target::Pass),
        (r"Z:\Games\Title\data\x", Target::Pass),
        (r"\\?\Z:\Games\Title", Target::Pass),
    ]);
}

/// The PC's C: is also a console drive ("C:" mounted on the game folder): paths already in the
/// game folder, XCL's data folder or an `own_folder` stay the PC's, anything else on the drive
/// is the console's. `own_folder` takes '/', a trailing separator or a verbatim path; an empty
/// one is ignored.
#[test]
fn own_folders_on_a_mounted_pc_drive_are_not_translated() {
    let mounts = || vec![("C:".to_string(), ROOT.to_string()), ("GAME:".to_string(), ROOT.to_string())];
    let v = Vfs::new(Path::new(ROOT), Path::new(DATA), mounts())
        .own_folder(Path::new(r"\\?\C:\Xcl Cache\dxil\"))
        .own_folder(Path::new("C:/Other Cache/"));
    check(&v, &[
        (r"C:\Games\Title\xcl_testtitle.exe", game("xcl_testtitle.exe")),
        (r"c:\games\title", game("")),
        (r"C:\XclClassifyTest\Data\Title\logs\xcl.log", Target::Pass),
        (r"c:\xclclassifytest\DATA\title", Target::Pass),
        (r"C:\Xcl Cache\dxil\0123456789abcdef.dxil", Target::Pass),
        (r"\\?\C:\Xcl Cache\dxil", Target::Pass),
        (r"C:\Other Cache\x.bin", Target::Pass),
        // the folders' names are not prefixes of other names
        (r"C:\Xcl Cache\dxil2\x.dxil", game(r"Xcl Cache\dxil2\x.dxil")),
        (r"C:\XclClassifyTest\Data\Title2\x", game(r"XclClassifyTest\Data\Title2\x")),
        (r"C:\Games\Title2\x", game(r"Games\Title2\x")),
        // the rest of the drive, and a mount after a current directory in the game folder
        (r"C:\data\hello.txt", game(r"data\hello.txt")),
        (r"C:\Games\Title\bin\GAME:\data\x", game(r"data\x")),
    ]);
    let v = Vfs::new(Path::new(ROOT), Path::new(DATA), mounts()).own_folder(Path::new(""));
    check(&v, &[(r"C:\data\hello.txt", game(r"data\hello.txt")), (r"C:\Games\Title\x", game("x"))]);
}

/// When the absolutiser fails, nothing is claimed: the path passes unchanged.
#[test]
fn a_failed_absolutiser_passes_the_path() {
    let v = console();
    for p in [r"G:\x", r"T:\x", ROOT, r"C:\Games\Title\x", r"C:\cwd\GAME:\x"] {
        assert_eq!(v.classify_with(p, |_| None), Target::Pass, "classify({p:?})");
    }
}

/// The absolutiser only ever sees a Windows path: '/' turned into '\', no verbatim prefix.
#[test]
fn the_absolutiser_sees_normalised_paths() {
    let v = console();
    let seen = RefCell::new(Vec::new());
    for p in ["g:/a/b", "//?/C:/Games/Title/x", r"\??\T:\x", "C:/x/GAME:/y", "x/y", r"\\?\Q:\x", "C:/Games/Title/"] {
        v.classify_with(p, |q: &str| {
            seen.borrow_mut().push(q.to_string());
            full(q)
        });
    }
    let seen = seen.into_inner();
    assert!(!seen.is_empty());
    for q in &seen {
        assert!(!q.contains('/') && !q.starts_with(r"\\?\") && !q.starts_with(r"\??\"), "absolutised {q:?}");
    }
}

/// `game_root`, `original_path` and `overlay_path` are plain '\' joins ("" = the folder itself),
/// with the root given in any spelling.
#[test]
fn original_and_overlay_paths_join_rel() {
    let v = Vfs::new(Path::new("C:/Games/Title/"), Path::new(DATA), Vec::new());
    assert_eq!(v.game_root(), ROOT);
    assert_eq!(v.original_path(""), ROOT);
    assert_eq!(v.original_path(r"data\Level_01.PAK"), format!("{ROOT}\\data\\Level_01.PAK"));
    assert_eq!(v.overlay_path(""), format!("{DATA}\\overlay"));
    assert_eq!(v.overlay_path(r"data\x"), format!("{DATA}\\overlay\\data\\x"));
    // classify and original_path agree on the root's own spelling
    let Target::Game { rel } = v.classify_with(r"c:\GAMES\title\Data\X", full) else { panic!("not the game") };
    assert_eq!(v.original_path(&rel), format!("{ROOT}\\Data\\X"));
}

/// Random paths built from a known prefix, components, separators and case flips, checked
/// against what the prefix means (an independent model of the cases above); a panic is a
/// failure that names the input.
#[test]
fn random_paths_match_a_model_of_the_mounts() {
    const NAMES: &[&str] = &["data", "Maps", "level_01.pak", "Ünïcode", "a b", "x.y.z", "bin", "Title2", "SAVE"];
    // (prefix, what it names): 0 = game folder, 1 = T:'s folder, 2 = not ours
    const PREFIXES: &[(&str, u8)] = &[
        (r"G:\", 0), ("G:", 0), (r"GAME:\", 0), ("GAME:", 0), (r"D:\", 0), (r"C:\Games\Title\", 0),
        (r"\\?\C:\Games\Title\", 0), (r"\??\G:\", 0), (r"C:\Games\Title\bin\GAME:\", 0), (r"C:\Games\Title\bin\..\", 0),
        (r"T:\", 1), ("T:", 1), (r"C:\x\T:\", 1),
        (r"C:\Games\Title2\", 2), (r"C:\Other\", 2), (r"\\srv\share\", 2), (r"Q:\Games\Title\", 2),
    ];
    let v = console();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = |n: usize| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as usize) % n
    };
    for _ in 0..3000 {
        let (prefix, kind) = PREFIXES[rnd(PREFIXES.len())];
        let mut path: String = prefix.chars().map(|c| match rnd(3) {
            0 => c.to_ascii_lowercase(),
            1 => c.to_ascii_uppercase(),
            _ => c,
        }).collect();
        let mut comps = Vec::new();
        for i in 0..rnd(5) {
            if i > 0 || !(path.ends_with('\\') || path.ends_with(':')) {
                path.push_str(["\\", "/", "\\\\"][rnd(3)]);
            }
            match rnd(6) {
                0 => path.push('.'),
                1 => path.push_str(r"zz\.."),
                _ => {
                    let n = NAMES[rnd(NAMES.len())];
                    comps.push(n);
                    path.push_str(n);
                }
            }
        }
        if rnd(4) == 0 {
            path.push('\\');
        }
        let rel = comps.join("\\");
        let join = |base: &str| if rel.is_empty() { base.to_string() } else { format!("{base}\\{rel}") };
        let want = match kind {
            0 => game(&rel),
            1 => tr(&join(&format!("{DATA}\\temp"))),
            _ => Target::Pass,
        };
        let got = catch_unwind(AssertUnwindSafe(|| v.classify_with(&path, full)))
            .unwrap_or_else(|_| panic!("classify({path:?}) panicked"));
        assert_eq!(got, want, "classify({path:?})");
        if let Target::Game { rel } = &got {
            assert!(!rel.starts_with('\\') && !rel.ends_with('\\') && !rel.contains(r"\\"), "rel {rel:?} of {path:?}");
            assert_eq!(v.original_path(rel), join(ROOT));
        }
    }
}

