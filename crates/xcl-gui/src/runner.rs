//! Running a program for the window: its output line by line, the tail of a log file, and
//! stopping it with what it started.

use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

/// What a running program and a followed log send to the window.
pub enum Event {
    /// A line the program printed (stdout or stderr).
    Console(String),
    /// A line added to the followed log file.
    Log(String),
    /// The program ended: its exit code, if it has one.
    Exited(Option<i32>),
}

/// A program the window started.
pub struct Running {
    pid: u32,
    child: Arc<std::sync::Mutex<Child>>,
}

/// Starts `program` with `args` and `env` in `cwd`, without a console window of its own; every
/// line it prints, and its end, go to `events`. `wake` is called after each of them.
pub fn start(
    program: &Path,
    args: &[String],
    env: &[(String, String)],
    cwd: Option<&Path>,
    events: Sender<Event>,
    wake: impl Fn() + Send + Sync + 'static,
) -> Result<Running, String> {
    let mut cmd = Command::new(program);
    cmd.args(args).envs(env.iter().map(|(k, v)| (k, v))).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: this program has no console, and its children get none either
        cmd.creation_flags(0x0800_0000);
    }
    let mut child = cmd.spawn().map_err(|e| format!("{}: {e}", program.display()))?;
    let pid = child.id();
    let wake = Arc::new(wake);
    let mut readers = Vec::new();
    let streams: [Option<Box<dyn Read + Send>>; 2] =
        [child.stdout.take().map(|s| Box::new(s) as Box<dyn Read + Send>), child.stderr.take().map(|s| Box::new(s) as Box<dyn Read + Send>)];
    for stream in streams.into_iter().flatten() {
        let (events, wake) = (events.clone(), wake.clone());
        readers.push(std::thread::spawn(move || {
            for line in lines(stream) {
                if events.send(Event::Console(line)).is_err() {
                    break;
                }
                wake();
            }
        }));
    }
    let child = Arc::new(std::sync::Mutex::new(child));
    let waited = child.clone();
    std::thread::spawn(move || {
        // the pipes close when the program and what it started have ended
        for r in readers {
            let _ = r.join();
        }
        let code = waited.lock().ok().and_then(|mut c| c.wait().ok()).and_then(|s| s.code());
        let _ = events.send(Event::Exited(code));
        wake();
    });
    Ok(Running { pid, child })
}

impl Running {
    /// Ends the program and every process it started (`xcl run` runs the game in a second
    /// process of its own).
    pub fn stop(&self) {
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            let killed = Command::new("taskkill")
                .args(["/PID", &self.pid.to_string(), "/T", "/F"])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .creation_flags(0x0800_0000)
                .status();
            if killed.is_ok_and(|s| s.success()) {
                return;
            }
        }
        let _ = self.pid;
        if let Ok(mut c) = self.child.lock() {
            let _ = c.kill();
        }
    }
}

/// The lines of a stream that are not empty, whatever their encoding: bytes that are not
/// UTF-8 are replaced, and `\r\n` and a lone `\r` (cargo's progress) end a line too.
fn lines(stream: impl Read) -> impl Iterator<Item = String> {
    let mut reader = BufReader::new(stream);
    std::iter::from_fn(move || {
        let mut buf = Vec::new();
        loop {
            let available = match reader.fill_buf() {
                Ok(b) if !b.is_empty() => b,
                _ => break,
            };
            match available.iter().position(|&b| b == b'\n' || b == b'\r') {
                Some(i) => {
                    buf.extend_from_slice(&available[..i]);
                    reader.consume(i + 1);
                    if buf.is_empty() {
                        // the `\n` of `\r\n`, or an empty line
                        continue;
                    }
                    return Some(String::from_utf8_lossy(&buf).into_owned());
                }
                None => {
                    let n = available.len();
                    buf.extend_from_slice(available);
                    reader.consume(n);
                }
            }
        }
        (!buf.is_empty()).then(|| String::from_utf8_lossy(&buf).into_owned())
    })
}

/// A followed file: what is added to it goes to `events` as `Event::Log` lines until `stop`.
pub struct Tail {
    stop: Arc<AtomicBool>,
}

impl Tail {
    /// Follows `path` from its start, waiting for it to appear (XCL makes a new log each run).
    /// After `stop` what is still unread is sent, then the thread ends.
    pub fn follow(path: PathBuf, events: Sender<Event>, wake: impl Fn() + Send + 'static) -> Tail {
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        std::thread::spawn(move || {
            let mut at = 0u64;
            let mut rest = Vec::new();
            loop {
                let last = stopped.load(Ordering::Relaxed);
                let before = at;
                if let Some(added) = read_from(&path, &mut at) {
                    if at < before + added.len() as u64 {
                        // a new file: the unfinished line of the old one is not its start
                        rest.clear();
                    }
                    rest.extend_from_slice(&added);
                    let mut sent = false;
                    while let Some(i) = rest.iter().position(|&b| b == b'\n') {
                        let line: Vec<u8> = rest.drain(..=i).collect();
                        let line = String::from_utf8_lossy(&line).trim_end_matches(['\r', '\n']).to_string();
                        if events.send(Event::Log(line)).is_err() {
                            return;
                        }
                        sent = true;
                    }
                    if sent {
                        wake();
                    }
                }
                if last {
                    if !rest.is_empty() {
                        let _ = events.send(Event::Log(String::from_utf8_lossy(&rest).into_owned()));
                        wake();
                    }
                    return;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        });
        Tail { stop }
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for Tail {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The bytes of `path` from `*at` on (`None`: none, or no file yet). A file shorter than
/// `*at` is a new one and is read from its start.
fn read_from(path: &Path, at: &mut u64) -> Option<Vec<u8>> {
    let mut f = std::fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    if len < *at {
        *at = 0;
    }
    if len == *at {
        return None;
    }
    f.seek(SeekFrom::Start(*at)).ok()?;
    let mut buf = Vec::new();
    // at most 4 MiB a turn: a log that grows faster is read over the next ones
    f.take(4 << 20).read_to_end(&mut buf).ok()?;
    *at += buf.len() as u64;
    Some(buf)
}

/// Opens a folder or a file with the system's program for it.
pub fn open(path: &Path) {
    #[cfg(windows)]
    let cmd = Command::new("explorer").arg(path).spawn();
    #[cfg(target_os = "macos")]
    let cmd = Command::new("open").arg(path).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let cmd = Command::new("xdg-open").arg(path).spawn();
    let _ = cmd;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn lines_end_at_any_line_end_and_survive_bad_bytes() {
        let data = b"one\r\ntwo\n\nthree\rfo\xffur".to_vec();
        let got: Vec<String> = lines(&data[..]).collect();
        assert_eq!(got, ["one", "two", "three", "fo\u{fffd}ur"]);
        assert_eq!(lines(&b""[..]).count(), 0);
    }

    #[test]
    fn a_tail_follows_a_file_that_appears_grows_and_starts_again() {
        let dir = std::env::temp_dir().join(format!("xcl_gui_tail_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("xcl.log");
        let _ = std::fs::remove_file(&path);
        let (tx, rx) = channel();
        let tail = Tail::follow(path.clone(), tx, || {});
        let next = |rx: &std::sync::mpsc::Receiver<Event>| match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Event::Log(l)) => l,
            _ => panic!("no log line"),
        };
        std::fs::write(&path, "[ 0.1][ 1][INFO ] a\r\n[ 0.2][ 1][WARN ] b\npart").unwrap();
        assert_eq!(next(&rx), "[ 0.1][ 1][INFO ] a");
        assert_eq!(next(&rx), "[ 0.2][ 1][WARN ] b");
        // a new, shorter file is read from its start
        std::fs::write(&path, "new\n").unwrap();
        assert_eq!(next(&rx), "new");
        std::fs::write(&path, "new\nend").unwrap();
        tail.stop();
        // the unfinished last line comes with the stop
        assert_eq!(next(&rx), "end");
        assert!(rx.recv_timeout(Duration::from_secs(5)).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_program_s_output_and_end_arrive() {
        let (tx, rx) = channel();
        let (program, args): (&str, Vec<String>) = if cfg!(windows) {
            ("cmd", vec!["/C".into(), "echo out& echo err 1>&2& exit 3".into()])
        } else {
            ("sh", vec!["-c".into(), "echo out; echo err 1>&2; echo $XCL_GUI_TEST; exit 3".into()])
        };
        let env = [("XCL_GUI_TEST".to_string(), "set".to_string())];
        let _running = start(Path::new(program), &args, &env, None, tx, || {}).unwrap();
        let mut lines = Vec::new();
        let code = loop {
            match rx.recv_timeout(Duration::from_secs(10)).expect("an event") {
                Event::Console(l) => lines.push(l.trim().to_string()),
                Event::Exited(code) => break code,
                Event::Log(_) => unreachable!(),
            }
        };
        assert_eq!(code, Some(3));
        assert!(lines.contains(&"out".to_string()) && lines.contains(&"err".to_string()), "{lines:?}");
        if !cfg!(windows) {
            assert!(lines.contains(&"set".to_string()));
        }
        assert!(start(Path::new("no-such-program-xcl-gui"), &[], &[], None, channel().0, || {}).is_err());
    }
}
