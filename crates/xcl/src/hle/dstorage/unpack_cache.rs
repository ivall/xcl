//! `XCL_DSTORAGE_CACHE=<folder>`: the unpacked bytes of file requests (zlib and / or BCPack),
//! kept on disk and read back the next time the same request comes – in the same run or a later
//! one – instead of being decoded on the CPU again.
//!
//! A request is the same when it names the same file (the host path it is open on, after the VFS,
//! its size and modification time), the same offset and source size, and the same decoding (zlib,
//! BCPack mode, intermediate and destination sizes). Each source file has a store of two files in
//! the folder, named by a hash of that path, size and modification time: `<hash>.data` (the
//! unpacked bytes, appended) and `<hash>.idx`
//! (one record per entry, written after its bytes). A record whose bytes are not all in the data
//! file (a run that ended while writing) is ignored, and an entry whose bytes do not match the
//! Adler-32 in its record is not used (`XCL_DSTORAGE_CACHE_VERIFY=0`: not checked when read). Nothing is ever evicted: new entries stop once the folder
//! holds `XCL_DSTORAGE_CACHE_MB` (default 32,768).
//!
//! New entries are written by a thread of their own, so a request does not wait for the disk;
//! while more than `QUEUE_BYTES` wait to be written, further ones are not kept.
//!
//! A store is used by one process at a time: it holds a lock on its index file while open. Another
//! process (a second XCL run of the same files with the same folder) finds the store taken and
//! decodes that file's requests as without the cache, instead of appending at ends it does not own.

use std::collections::HashMap;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::Read;
use std::os::windows::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

const MAGIC: &[u8; 8] = b"XCLDSC1\0";
const RECORD: usize = 40;
/// the most bytes waiting for the writer thread
const QUEUE_BYTES: u64 = 256 << 20;

/// What makes two requests the same, within one source file.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct Key {
    pub offset: u64,
    pub source_size: u32,
    /// the request's zlib and BCPack option bits
    pub decode: u32,
    pub intermediate_size: u32,
    pub destination_size: u32,
}

#[derive(Clone, Copy)]
struct Entry {
    pos: u64,
    len: u32,
    adler: u32,
}

/// The entries of one source file.
pub(super) struct Store {
    data: File,
    entries: Mutex<HashMap<Key, Entry>>,
    /// (index file, end of the data file, end of the index file), held while appending
    append: Mutex<(File, u64, u64)>,
}

#[derive(Default)]
struct Counters {
    hits: AtomicU64,
    hit_bytes: AtomicU64,
    stored: AtomicU64,
    stored_bytes: AtomicU64,
    not_stored_full: AtomicU64,
    not_stored_busy: AtomicU64,
    bad: AtomicU64,
    /// stores another process holds
    taken: AtomicU64,
}

/// What the cache and its writer thread share.
struct Shared {
    dir: PathBuf,
    max_bytes: u64,
    /// whether an entry's bytes are checked against its Adler-32 when read
    verify: bool,
    /// bytes in the folder, with those queued for writing
    used: AtomicU64,
    counters: Counters,
    /// bytes waiting for the writer thread (`flush` waits for 0)
    queued: Mutex<u64>,
    written: Condvar,
}

/// An entry for the writer thread.
struct Job {
    store: Arc<Store>,
    key: Key,
    bytes: Vec<u8>,
}

pub(super) struct Cache {
    shared: Arc<Shared>,
    stores: Mutex<HashMap<u64, Arc<Store>>>,
    writer: OnceLock<mpsc::Sender<Job>>,
}

/// The cache of this run, if `XCL_DSTORAGE_CACHE` names a folder that can be used.
pub(super) fn get() -> Option<&'static Cache> {
    static CACHE: OnceLock<Option<Cache>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let dir = std::env::var_os("XCL_DSTORAGE_CACHE").filter(|d| !d.is_empty())?;
            let mb = std::env::var("XCL_DSTORAGE_CACHE_MB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(32 << 10);
            let verify = std::env::var("XCL_DSTORAGE_CACHE_VERIFY").map_or(true, |v| v != "0");
            match Cache::open(Path::new(&dir), mb << 20, verify) {
                Ok(c) => {
                    crate::xlog_info!("DirectStorage: unpack cache in {} ({} of {mb} MB used)", c.shared.dir.display(), c.shared.used.load(Ordering::Relaxed) >> 20);
                    Some(c)
                }
                Err(e) => {
                    crate::xlog_warn!("DirectStorage: unpack cache {}: {e}; requests are decoded as without it", Path::new(&dir).display());
                    None
                }
            }
        })
        .as_ref()
}

/// Adler-32 of `b` (RFC 1950), the check of an entry's bytes.
fn adler32(b: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let (mut a, mut s) = (1u32, 0u32);
    // 5552 bytes is the most that cannot overflow `s` before the reduction
    for chunk in b.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            s += a;
        }
        a %= MOD;
        s %= MOD;
    }
    (s << 16) | a
}

fn fnv64(bytes: impl IntoIterator<Item = u8>, mut h: u64) -> u64 {
    for b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    h
}

impl Cache {
    pub(super) fn open(dir: &Path, max_bytes: u64, verify: bool) -> std::io::Result<Cache> {
        std::fs::create_dir_all(dir)?;
        let used = std::fs::read_dir(dir)?.flatten().filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum();
        let shared = Shared { dir: dir.to_path_buf(), max_bytes, verify, used: AtomicU64::new(used), counters: Counters::default(), queued: Mutex::new(0), written: Condvar::new() };
        Ok(Cache { shared: Arc::new(shared), stores: Mutex::new(HashMap::new()), writer: OnceLock::new() })
    }

    /// The store of the file `file` is open on.
    pub(super) fn store(&self, file: &File) -> std::io::Result<Arc<Store>> {
        let meta = file.metadata()?;
        let mtime = meta.modified()?.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
        // the host file, not the name it was opened by: through the VFS one name can be an overlay
        // copy or another install's file
        let h = fnv64(final_path(file)?.iter().flat_map(|c| c.to_le_bytes()), 0xcbf2_9ce4_8422_2325);
        let h = fnv64(meta.len().to_le_bytes().into_iter().chain(mtime.to_le_bytes()), h);
        let mut stores = self.stores.lock().unwrap();
        if let Some(s) = stores.get(&h) {
            return Ok(s.clone());
        }
        let dir = &self.shared.dir;
        let s = Store::open(&dir.join(format!("{h:016x}.data")), &dir.join(format!("{h:016x}.idx")));
        if s.as_ref().is_err_and(|e| e.kind() == std::io::ErrorKind::WouldBlock) && self.shared.counters.taken.fetch_add(1, Ordering::Relaxed) == 0 {
            crate::xlog_info!("DirectStorage: another process uses the unpack cache in {}: requests on its files are decoded as without it", dir.display());
        }
        let s = Arc::new(s?);
        stores.insert(h, s.clone());
        Ok(s)
    }

    /// The unpacked bytes of `key`, if stored and intact.
    pub(super) fn get(&self, store: &Store, key: &Key) -> Option<Vec<u8>> {
        let e = *store.entries.lock().unwrap().get(key)?;
        let mut b = vec![0u8; e.len as usize];
        let ok = store.data.seek_read(&mut b, e.pos).is_ok_and(|n| n == b.len()) && (!self.shared.verify || adler32(&b) == e.adler);
        if !ok {
            store.entries.lock().unwrap().remove(key);
            if self.shared.counters.bad.fetch_add(1, Ordering::Relaxed) == 0 {
                crate::xlog_warn!("DirectStorage: an unpack cache entry in {} does not match its checksum: decoded again", self.shared.dir.display());
            }
            return None;
        }
        self.shared.counters.hits.fetch_add(1, Ordering::Relaxed);
        self.shared.counters.hit_bytes.fetch_add(b.len() as u64, Ordering::Relaxed);
        Some(b)
    }

    /// Keep the unpacked bytes of `key`: queued for the writer thread (nothing when the folder is
    /// full or too much waits to be written).
    pub(super) fn put(&self, store: &Arc<Store>, key: &Key, bytes: &[u8]) {
        let sh = &self.shared;
        if store.entries.lock().unwrap().contains_key(key) {
            return;
        }
        let len = bytes.len() as u64 + RECORD as u64;
        if sh.used.fetch_add(len, Ordering::Relaxed) + len > sh.max_bytes {
            sh.used.fetch_sub(len, Ordering::Relaxed);
            if sh.counters.not_stored_full.fetch_add(1, Ordering::Relaxed) == 0 {
                crate::xlog_info!("DirectStorage: the unpack cache in {} is full: no more entries are added", sh.dir.display());
            }
            return;
        }
        {
            let mut q = sh.queued.lock().unwrap();
            if *q + bytes.len() as u64 > QUEUE_BYTES {
                drop(q);
                sh.used.fetch_sub(len, Ordering::Relaxed);
                sh.counters.not_stored_busy.fetch_add(1, Ordering::Relaxed);
                return;
            }
            *q += bytes.len() as u64;
        }
        let tx = self.writer.get_or_init(|| {
            let (tx, rx) = mpsc::channel::<Job>();
            let sh = self.shared.clone();
            // without the thread the receiver is gone and `send` fails below
            let _ = std::thread::Builder::new().name("xcl-dstorage-cache".into()).spawn(move || {
                for job in rx {
                    write(&sh, job);
                }
            });
            tx
        });
        if let Err(mpsc::SendError(job)) = tx.send(Job { store: store.clone(), key: *key, bytes: bytes.to_vec() }) {
            sh.used.fetch_sub(len, Ordering::Relaxed);
            done_writing(sh, job.bytes.len() as u64);
        }
    }

    /// Wait until every queued entry is written.
    #[cfg(test)]
    pub(super) fn flush(&self) {
        let q = self.shared.queued.lock().unwrap();
        drop(self.shared.written.wait_while(q, |q| *q > 0).unwrap());
    }

    pub(super) fn stats(&self) -> super::UnpackCacheStats {
        let c = &self.shared.counters;
        let l = |a: &AtomicU64| a.load(Ordering::Relaxed);
        super::UnpackCacheStats {
            hits: l(&c.hits),
            hit_bytes: l(&c.hit_bytes),
            stored: l(&c.stored),
            stored_bytes: l(&c.stored_bytes),
            not_stored_full: l(&c.not_stored_full),
            not_stored_busy: l(&c.not_stored_busy),
            bad_entries: l(&c.bad),
        }
    }
}

/// The writer thread's part of `put`.
fn write(sh: &Shared, job: Job) {
    let Job { store, key, bytes } = job;
    let len = bytes.len() as u64 + RECORD as u64;
    let adler = adler32(&bytes);
    let mut a = store.append.lock().unwrap();
    let (pos, at) = (a.1, a.2);
    let rec = record(&key, pos, bytes.len() as u32, adler);
    let new = !store.entries.lock().unwrap().contains_key(&key);
    if new && store.data.seek_write(&bytes, pos).is_ok_and(|n| n == bytes.len()) && a.0.seek_write(&rec, at).is_ok_and(|n| n == RECORD) {
        a.1 = pos + bytes.len() as u64;
        a.2 = at + RECORD as u64;
        store.entries.lock().unwrap().insert(key, Entry { pos, len: bytes.len() as u32, adler });
        sh.counters.stored.fetch_add(1, Ordering::Relaxed);
        sh.counters.stored_bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
    } else {
        sh.used.fetch_sub(len, Ordering::Relaxed);
    }
    drop(a);
    // before `done_writing`: once nothing else holds the store, its lock is free when `flush` returns
    drop(store);
    done_writing(sh, bytes.len() as u64);
}

/// The path of the file `file` is open on (`GetFinalPathNameByHandleW`, not hooked by the VFS).
fn final_path(file: &File) -> std::io::Result<Vec<u16>> {
    use std::os::windows::io::AsRawHandle;
    let mut buf = vec![0u16; 512];
    loop {
        let n = unsafe { windows_sys::Win32::Storage::FileSystem::GetFinalPathNameByHandleW(file.as_raw_handle(), buf.as_mut_ptr(), buf.len() as u32, 0) } as usize;
        if n == 0 {
            return Err(std::io::Error::last_os_error());
        }
        if n < buf.len() {
            buf.truncate(n);
            return Ok(buf);
        }
        // too small: `n` is the length it needs, with the terminating zero
        buf.resize(n, 0);
    }
}

fn done_writing(sh: &Shared, n: u64) {
    *sh.queued.lock().unwrap() -= n;
    sh.written.notify_all();
}

trait AppendRecord {
    fn write_all_at_end(&mut self, b: &[u8]) -> std::io::Result<()>;
}

impl AppendRecord for File {
    fn write_all_at_end(&mut self, b: &[u8]) -> std::io::Result<()> {
        let end = self.metadata()?.len();
        let n = self.seek_write(b, end)?;
        if n == b.len() {
            Ok(())
        } else {
            Err(std::io::ErrorKind::WriteZero.into())
        }
    }
}

fn record(k: &Key, pos: u64, len: u32, adler: u32) -> [u8; RECORD] {
    let mut r = [0u8; RECORD];
    r[0..8].copy_from_slice(&k.offset.to_le_bytes());
    r[8..12].copy_from_slice(&k.source_size.to_le_bytes());
    r[12..16].copy_from_slice(&k.decode.to_le_bytes());
    r[16..20].copy_from_slice(&k.intermediate_size.to_le_bytes());
    r[20..24].copy_from_slice(&k.destination_size.to_le_bytes());
    r[24..32].copy_from_slice(&pos.to_le_bytes());
    r[32..36].copy_from_slice(&len.to_le_bytes());
    r[36..40].copy_from_slice(&adler.to_le_bytes());
    r
}

impl Store {
    /// `ErrorKind::WouldBlock`: another process holds the store.
    fn open(data: &Path, idx: &Path) -> std::io::Result<Store> {
        let mut index = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(idx)?;
        // held until the store is dropped; read through this handle, as the lock is per handle
        index.try_lock().map_err(|e| match e {
            TryLockError::WouldBlock => std::io::Error::new(std::io::ErrorKind::WouldBlock, "the store is in use by another process"),
            TryLockError::Error(e) => e,
        })?;
        let data = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(data)?;
        let data_len = data.metadata()?.len();
        let mut raw = Vec::new();
        index.read_to_end(&mut raw)?;
        let mut entries = HashMap::new();
        let mut end = 0u64;
        let mut idx_end = MAGIC.len() as u64;
        if raw.len() < MAGIC.len() || &raw[..MAGIC.len()] != MAGIC {
            // new, or not ours: start both over
            index.set_len(0)?;
            data.set_len(0)?;
            index.write_all_at_end(MAGIC)?;
        } else {
            for r in raw[MAGIC.len()..].chunks_exact(RECORD) {
                let u32_at = |i: usize| u32::from_le_bytes(r[i..i + 4].try_into().unwrap());
                let (pos, len) = (u64::from_le_bytes(r[24..32].try_into().unwrap()), u32_at(32));
                if pos + len as u64 > data_len {
                    continue;
                }
                let key = Key { offset: u64::from_le_bytes(r[0..8].try_into().unwrap()), source_size: u32_at(8), decode: u32_at(12), intermediate_size: u32_at(16), destination_size: u32_at(20) };
                entries.insert(key, Entry { pos, len, adler: u32_at(36) });
                end = end.max(pos + len as u64);
            }
            // a record cut short at the end: keep whole records only
            let whole = MAGIC.len() + (raw.len() - MAGIC.len()) / RECORD * RECORD;
            if whole != raw.len() {
                index.set_len(whole as u64)?;
            }
            idx_end = whole as u64;
        }
        Ok(Store { data, entries: Mutex::new(entries), append: Mutex::new((index, end.max(data_len), idx_end)) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xcl_dscache_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn source(d: &Path) -> (String, File) {
        std::fs::create_dir_all(d).unwrap();
        let p = d.join("pack.bin");
        std::fs::write(&p, b"compressed source").unwrap();
        (p.to_string_lossy().into_owned(), File::open(&p).unwrap())
    }

    const K: Key = Key { offset: 4096, source_size: 100, decode: 1, intermediate_size: 0, destination_size: 65536 };

    #[test]
    fn adler32_matches_rfc_1950() {
        assert_eq!(adler32(b""), 1);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        // more than one 5552-byte block of 0xff (the case that would overflow without the reduction)
        assert_eq!(adler32(&[0xff; 20000]), 0x9F51_D664);
    }

    #[test]
    fn an_entry_is_found_again_in_a_later_run_and_only_for_the_same_request() {
        let d = dir("again");
        let (_, f) = source(&d.join("src"));
        let bytes: Vec<u8> = (0..65536u32).map(|i| (i * 7) as u8).collect();
        {
            let c = Cache::open(&d.join("cache"), 1 << 30, true).unwrap();
            let s = c.store(&f).unwrap();
            assert!(c.get(&s, &K).is_none());
            c.put(&s, &K, &bytes);
            c.flush();
            assert_eq!(c.get(&s, &K).as_deref(), Some(&bytes[..]));
        }
        let c = Cache::open(&d.join("cache"), 1 << 30, true).unwrap();
        let s = c.store(&f).unwrap();
        assert_eq!(c.get(&s, &K).as_deref(), Some(&bytes[..]));
        assert!(c.get(&s, &Key { offset: 8192, ..K }).is_none());
        assert!(c.get(&s, &Key { decode: 1 | 7 << 1, ..K }).is_none());
        assert_eq!(c.stats().hits, 1, "this run found the entry once");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_changed_source_file_or_damaged_bytes_are_not_used() {
        let d = dir("damaged");
        let (path, f) = source(&d.join("src"));
        let c = Cache::open(&d.join("cache"), 1 << 30, true).unwrap();
        let s = c.store(&f).unwrap();
        c.put(&s, &K, &[5u8; 4096]);
        c.flush();
        // a damaged byte in the store's data: found when checked, returned as it is without
        s.data.seek_write(&[6], 10).unwrap();
        assert!(c.get(&s, &K).is_none());
        assert_eq!(c.stats().bad_entries, 1);
        // (the store is free again once the first cache is gone)
        drop((s, c));
        let unchecked = Cache::open(&d.join("cache"), 1 << 30, false).unwrap();
        let s3 = unchecked.store(&f).unwrap();
        assert_eq!(unchecked.get(&s3, &K).map(|b| b[10]), Some(6));
        // the source file changes size: another store, without the entry
        std::fs::write(&path, b"a longer compressed source").unwrap();
        let c2 = Cache::open(&d.join("cache"), 1 << 30, true).unwrap();
        let s2 = c2.store(&File::open(&path).unwrap()).unwrap();
        assert!(c2.get(&s2, &K).is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_record_without_its_bytes_and_a_cut_record_are_ignored() {
        let d = dir("cut");
        let (_, f) = source(&d.join("src"));
        let (data, idx);
        {
            let c = Cache::open(&d.join("cache"), 1 << 30, true).unwrap();
            let s = c.store(&f).unwrap();
            c.put(&s, &K, &[1u8; 1000]);
            c.put(&s, &Key { offset: 0, ..K }, &[2u8; 1000]);
            c.flush();
            let names: Vec<PathBuf> = std::fs::read_dir(d.join("cache")).unwrap().map(|e| e.unwrap().path()).collect();
            data = names.iter().find(|p| p.extension().is_some_and(|e| e == "data")).unwrap().clone();
            idx = names.iter().find(|p| p.extension().is_some_and(|e| e == "idx")).unwrap().clone();
        }
        // the second entry's bytes are lost, and half a record follows
        OpenOptions::new().write(true).open(&data).unwrap().set_len(1500).unwrap();
        let mut i = OpenOptions::new().write(true).open(&idx).unwrap();
        i.write_all_at_end(&[9u8; 20]).unwrap();
        let c = Cache::open(&d.join("cache"), 1 << 30, true).unwrap();
        let s = c.store(&f).unwrap();
        assert_eq!(c.get(&s, &K).as_deref(), Some(&[1u8; 1000][..]));
        assert!(c.get(&s, &Key { offset: 0, ..K }).is_none());
        // new entries go after the bytes that are there, and are found
        c.put(&s, &Key { offset: 0, ..K }, &[3u8; 700]);
        c.flush();
        assert_eq!(c.get(&s, &Key { offset: 0, ..K }).as_deref(), Some(&[3u8; 700][..]));
        assert_eq!(c.get(&s, &K).as_deref(), Some(&[1u8; 1000][..]));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn entries_put_from_many_threads_are_all_written() {
        let d = dir("threads");
        let (_, f) = source(&d.join("src"));
        let c = Cache::open(&d.join("cache"), 1 << 30, true).unwrap();
        let s = c.store(&f).unwrap();
        std::thread::scope(|t| {
            for w in 0..8u64 {
                let (c, s) = (&c, &s);
                t.spawn(move || {
                    for i in 0..50u64 {
                        c.put(s, &Key { offset: w * 1000 + i, ..K }, &[(w * 50 + i) as u8; 3000]);
                    }
                });
            }
        });
        c.flush();
        drop((s, c));
        let c = Cache::open(&d.join("cache"), 1 << 30, true).unwrap();
        let s = c.store(&f).unwrap();
        for w in 0..8u64 {
            for i in 0..50u64 {
                assert_eq!(c.get(&s, &Key { offset: w * 1000 + i, ..K }).as_deref(), Some(&[(w * 50 + i) as u8; 3000][..]));
            }
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_store_held_elsewhere_is_not_used_and_loses_no_entries() {
        let d = dir("held");
        let (_, f) = source(&d.join("src"));
        let a = Cache::open(&d.join("cache"), 1 << 30, true).unwrap();
        let sa = a.store(&f).unwrap();
        a.put(&sa, &K, &[1u8; 256]);
        a.flush();
        // a second cache on the same folder (another process): the lock is per handle
        let b = Cache::open(&d.join("cache"), 1 << 30, true).unwrap();
        let e = b.store(&f).err().expect("the store is held by the first cache");
        assert_eq!(e.kind(), std::io::ErrorKind::WouldBlock);
        a.put(&sa, &Key { offset: 0, ..K }, &[2u8; 256]);
        a.flush();
        drop((sa, a));
        // once it is free, both entries are there
        let sb = b.store(&f).unwrap();
        assert_eq!(b.get(&sb, &K).as_deref(), Some(&[1u8; 256][..]));
        assert_eq!(b.get(&sb, &Key { offset: 0, ..K }).as_deref(), Some(&[2u8; 256][..]));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_full_cache_adds_nothing() {
        let d = dir("full");
        let (_, f) = source(&d.join("src"));
        let c = Cache::open(&d.join("cache"), 2000, true).unwrap();
        let s = c.store(&f).unwrap();
        c.put(&s, &K, &[1u8; 1000]);
        c.put(&s, &Key { offset: 0, ..K }, &[2u8; 1000]);
        c.flush();
        assert!(c.get(&s, &K).is_some());
        assert!(c.get(&s, &Key { offset: 0, ..K }).is_none());
        assert_eq!(c.stats().not_stored_full, 1);
        let _ = std::fs::remove_dir_all(&d);
    }
}
