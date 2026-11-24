//! Memory-mapped `ID3D12PipelineLibrary` persistence (`XCL_PSO_LIBRARY=0`: off).
//!
//! Backing a driver pipeline library with a read-only Windows file mapping kept open for the
//! lifetime of `ID3D12PipelineLibrary` avoids duplicating hundreds of megabytes to >1 GiB of
//! private heap commit on load, and writing new snapshots through a separate file-backed mapping
//! into a fresh generation file (`gen-NNNNNNNN.plib`) avoids both `Vec<u8>` serialize spikes and
//! in-place replacement of an actively mapped file (which fails on Windows or triggers driver
//! device removal).

use std::ffi::c_void;
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::core::{Interface, HSTRING};
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, GetFileSizeEx, SetEndOfFile, SetFilePointerEx, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
    FILE_BEGIN, FILE_GENERIC_READ, FILE_GENERIC_WRITE, FILE_SHARE_DELETE, FILE_SHARE_READ,
    FILE_SHARE_WRITE, OPEN_EXISTING,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile, UnmapViewOfFile, FILE_MAP_READ, FILE_MAP_WRITE,
    MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READONLY, PAGE_READWRITE,
};

const D3D12_ERROR_ADAPTER_NOT_FOUND_HR: i32 = 0x887E_0001u32 as i32;
const D3D12_ERROR_DRIVER_VERSION_MISMATCH_HR: i32 = 0x887E_0002u32 as i32;

/// Whether the on-disk memory-mapped pipeline library is enabled (`XCL_PSO_LIBRARY=0` disables).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_PSO_LIBRARY").map_or(true, |v| v.trim() != "0"))
}

fn wide_path(path: &Path) -> Vec<u16> {
    path.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
}

/// A read-only Windows file mapping kept open (file + section + mapped view) with zero private
/// heap copy. Dropping unmaps the view and closes both handles in reverse order.
pub struct MappedBlob {
    path: PathBuf,
    file: HANDLE,
    mapping: HANDLE,
    view: *const u8,
    len: usize,
}

// The view is strictly read-only and backed by immutable OS handles owned uniquely by `MappedBlob`.
unsafe impl Send for MappedBlob {}
unsafe impl Sync for MappedBlob {}

impl MappedBlob {
    /// Open `path` read-only with shared read/write/delete access and map its entire contents.
    /// Returns `None` when the file does not exist, is empty, or cannot be mapped.
    pub fn open(path: &Path) -> Option<Self> {
        let wide = wide_path(path);
        unsafe {
            let file = CreateFileW(
                wide.as_ptr(),
                FILE_GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                std::ptr::null(),
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                std::ptr::null_mut(),
            );
            if file == INVALID_HANDLE_VALUE || file.is_null() {
                return None;
            }
            let mut size = 0i64;
            if GetFileSizeEx(file, &mut size) == 0 || size <= 0 {
                CloseHandle(file);
                return None;
            }
            let Ok(len) = usize::try_from(size) else {
                CloseHandle(file);
                return None;
            };
            let mapping = CreateFileMappingW(file, std::ptr::null(), PAGE_READONLY, 0, 0, std::ptr::null());
            if mapping.is_null() {
                CloseHandle(file);
                return None;
            }
            let view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
            if view.Value.is_null() {
                CloseHandle(mapping);
                CloseHandle(file);
                return None;
            }
            Some(Self {
                path: path.to_path_buf(),
                file,
                mapping,
                view: view.Value as *const u8,
                len,
            })
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.view as *const c_void
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.view, self.len) }
    }
}

impl Drop for MappedBlob {
    fn drop(&mut self) {
        unsafe {
            if !self.view.is_null() {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.view as *mut c_void });
            }
            if !self.mapping.is_null() {
                CloseHandle(self.mapping);
            }
            if !self.file.is_null() && self.file != INVALID_HANDLE_VALUE {
                CloseHandle(self.file);
            }
        }
    }
}

/// A read-write Windows file mapping for serializing a pipeline library directly into file-backed
/// pages without allocating a matching heap `Vec<u8>`.
pub struct MappedWriteFile {
    path: PathBuf,
    file: HANDLE,
    mapping: HANDLE,
    view: *mut u8,
    len: usize,
    committed: bool,
}

impl MappedWriteFile {
    /// Create (or truncate) `path` with exact byte length `len` and map it writable.
    pub fn create(path: &Path, len: usize) -> Option<Self> {
        if len == 0 {
            return None;
        }
        let size = i64::try_from(len).ok()?;
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let wide = wide_path(path);
        unsafe {
            let file = CreateFileW(
                wide.as_ptr(),
                FILE_GENERIC_READ | FILE_GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_DELETE,
                std::ptr::null(),
                CREATE_ALWAYS,
                FILE_ATTRIBUTE_NORMAL,
                std::ptr::null_mut(),
            );
            if file == INVALID_HANDLE_VALUE || file.is_null() {
                return None;
            }
            let mut new_pos = 0i64;
            if SetFilePointerEx(file, size, &mut new_pos, FILE_BEGIN) == 0 || SetEndOfFile(file) == 0 {
                CloseHandle(file);
                let _ = std::fs::remove_file(path);
                return None;
            }
            let mapping = CreateFileMappingW(
                file,
                std::ptr::null(),
                PAGE_READWRITE,
                ((len as u64) >> 32) as u32,
                (len as u64 & 0xFFFF_FFFF) as u32,
                std::ptr::null(),
            );
            if mapping.is_null() {
                CloseHandle(file);
                let _ = std::fs::remove_file(path);
                return None;
            }
            let view = MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, len);
            if view.Value.is_null() {
                CloseHandle(mapping);
                CloseHandle(file);
                let _ = std::fs::remove_file(path);
                return None;
            }
            Some(Self {
                path: path.to_path_buf(),
                file,
                mapping,
                view: view.Value as *mut u8,
                len,
                committed: false,
            })
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.view as *mut c_void
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.view, self.len) }
    }

    /// Unmap and close the file cleanly, keeping the written file on disk.
    pub fn finish(mut self) -> PathBuf {
        self.committed = true;
        self.close_handles();
        self.path.clone()
    }

    fn close_handles(&mut self) {
        unsafe {
            if !self.view.is_null() {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.view as *mut c_void });
                self.view = std::ptr::null_mut();
            }
            if !self.mapping.is_null() {
                CloseHandle(self.mapping);
                self.mapping = std::ptr::null_mut();
            }
            if !self.file.is_null() && self.file != INVALID_HANDLE_VALUE {
                CloseHandle(self.file);
                self.file = INVALID_HANDLE_VALUE;
            }
        }
    }
}

impl Drop for MappedWriteFile {
    fn drop(&mut self) {
        self.close_handles();
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Generational `.plib` directory manager: each save writes a new `gen-NNNNNNNN.plib` through a
/// `.tmp` staging file so any earlier generation kept memory-mapped by `ID3D12PipelineLibrary` is
/// never truncated or replaced while open.
pub struct LibraryStore {
    dir: PathBuf,
    next_gen: AtomicU64,
}

impl LibraryStore {
    pub fn new(dir: PathBuf) -> Self {
        let latest = latest_generation(&dir).map_or(0, |(g, _)| g);
        Self { dir, next_gen: AtomicU64::new(latest + 1) }
    }

    pub fn for_adapter(adapter_name: &str) -> Self {
        let key = format!("{:016x}", crate::gpu::shader::fnv64(adapter_name.as_bytes()));
        Self::new(crate::gpu::recomp::dxc::cache_dir().join("pso-library").join(key))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Open the newest generation file as a read-only memory mapping and clean up older unmapped
    /// generations from previous runs.
    pub fn open_latest(&self) -> Option<(u64, MappedBlob)> {
        let (gen, path) = latest_generation(&self.dir)?;
        let blob = MappedBlob::open(&path)?;
        self.next_gen.fetch_max(gen + 1, Ordering::AcqRel);
        cleanup_older_generations(&self.dir, gen);
        Some((gen, blob))
    }

    /// Write a fresh generation of `len` bytes using a file-backed writable mapping and commit it
    /// atomically via rename. Never touches any earlier generation currently kept open.
    pub fn write_generation(&self, len: usize, write_fn: impl FnOnce(&mut [u8]) -> bool) -> Option<(u64, PathBuf)> {
        if len == 0 {
            return None;
        }
        let gen = self.next_gen.fetch_add(1, Ordering::AcqRel);
        let final_path = self.dir.join(generation_name(gen));
        let tmp_path = self.dir.join(format!("gen-{gen:08x}-{}.tmp", std::process::id()));
        let mut mapped = MappedWriteFile::create(&tmp_path, len)?;
        let ok = write_fn(mapped.as_mut_slice());
        if !ok {
            drop(mapped);
            return None;
        }
        let staged = mapped.finish();
        if std::fs::rename(&staged, &final_path).is_err() {
            let _ = std::fs::remove_file(&staged);
            return None;
        }
        Some((gen, final_path))
    }
}

fn generation_name(gen: u64) -> String {
    format!("gen-{gen:08x}.plib")
}

fn parse_generation(file_name: &str) -> Option<u64> {
    let hex = file_name.strip_prefix("gen-")?.strip_suffix(".plib")?;
    if hex.is_empty() {
        return None;
    }
    u64::from_str_radix(hex, 16).ok()
}

fn latest_generation(dir: &Path) -> Option<(u64, PathBuf)> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        let Some(gen) = parse_generation(name) else { continue };
        if best.as_ref().is_none_or(|&(b, _)| gen > b) {
            best = Some((gen, path));
        }
    }
    best
}

fn cleanup_older_generations(dir: &Path, keep_gen: u64) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        if parse_generation(name).is_some_and(|g| g < keep_gen) || name.ends_with(".tmp") {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Process-wide D3D12 pipeline library backed by a kept-open read-only `MappedBlob` and
/// generational zero-heap-copy serialization.
pub struct MmapPipelineLibrary {
    lib: ID3D12PipelineLibrary1,
    /// Kept alive for the lifetime of `lib` so file-backed driver pages stay valid with zero heap copy.
    _backing: Option<MappedBlob>,
    store: LibraryStore,
    dirty: AtomicU64,
    saving: AtomicBool,
    save_lock: Mutex<()>,
}

// `ID3D12PipelineLibrary1` methods are free-threaded per the D3D12 specification; `_backing` is
// read-only and `save_lock` serializes disk snapshots.
unsafe impl Send for MmapPipelineLibrary {}
unsafe impl Sync for MmapPipelineLibrary {}

impl MmapPipelineLibrary {
    /// Open the latest memory-mapped pipeline library on `device`, falling back to a fresh empty
    /// library if none exists or if the driver rejects the cached generation (driver/adapter mismatch).
    pub unsafe fn open_or_create(device: &ID3D12Device, adapter_name: &str) -> Option<Self> {
        if !enabled() {
            return None;
        }
        let dev1: ID3D12Device1 = device.cast().ok()?;
        let store = LibraryStore::for_adapter(adapter_name);
        if let Some((gen, blob)) = store.open_latest() {
            match dev1.CreatePipelineLibrary::<ID3D12PipelineLibrary1>(blob.as_slice()) {
                Ok(lib) => {
                    crate::xlog_info!(
                        "D3D12.X: opened memory-mapped pipeline library gen-{gen:08x} ({} MiB file-backed)",
                        blob.len() >> 20
                    );
                    return Some(Self {
                        lib,
                        _backing: Some(blob),
                        store,
                        dirty: AtomicU64::new(0),
                        saving: AtomicBool::new(false),
                        save_lock: Mutex::new(()),
                    });
                }
                Err(e) => {
                    let stale_path = blob.path().to_path_buf();
                    let code = e.code().0;
                    drop(blob);
                    if code == D3D12_ERROR_DRIVER_VERSION_MISMATCH_HR || code == D3D12_ERROR_ADAPTER_NOT_FOUND_HR {
                        let _ = std::fs::remove_file(&stale_path);
                    }
                    crate::xlog_info!(
                        "D3D12.X: pipeline library gen-{gen:08x} rejected ({code:#x}) – starting fresh library"
                    );
                }
            }
        }
        let lib = dev1.CreatePipelineLibrary::<ID3D12PipelineLibrary1>(&[]).ok()?;
        Some(Self {
            lib,
            _backing: None,
            store,
            dirty: AtomicU64::new(0),
            saving: AtomicBool::new(false),
            save_lock: Mutex::new(()),
        })
    }

    pub unsafe fn load_compute(&self, key: &str, desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC) -> Option<ID3D12PipelineState> {
        let name = HSTRING::from(key);
        self.lib.LoadComputePipeline(&name, desc).ok()
    }

    pub unsafe fn load_graphics(&self, key: &str, desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC) -> Option<ID3D12PipelineState> {
        let name = HSTRING::from(key);
        self.lib.LoadGraphicsPipeline(&name, desc).ok()
    }

    pub unsafe fn store(&self, key: &str, pso: &ID3D12PipelineState) -> bool {
        let name = HSTRING::from(key);
        if self.lib.StorePipeline(&name, pso).is_ok() {
            self.dirty.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    pub fn dirty_count(&self) -> u64 {
        self.dirty.load(Ordering::Relaxed)
    }

    /// Serialize the pipeline library directly into a new memory-mapped generation file if at least
    /// `min_dirty` new pipelines have been stored since the last save. Never replaces or truncates
    /// the open `_backing` file.
    pub unsafe fn save_if_dirty(&self, min_dirty: u64) -> Option<PathBuf> {
        if self.dirty.load(Ordering::Relaxed) < min_dirty {
            return None;
        }
        if self.saving.swap(true, Ordering::AcqRel) {
            return None;
        }
        let _guard = self.save_lock.lock().unwrap();
        let dirty_before = self.dirty.load(Ordering::Acquire);
        let size = self.lib.GetSerializedSize();
        let saved = if size > 0 {
            self.store.write_generation(size, |buf| self.lib.Serialize(buf).is_ok())
        } else {
            None
        };
        if let Some((gen, _)) = saved {
            self.dirty.fetch_sub(dirty_before.min(self.dirty.load(Ordering::Relaxed)), Ordering::Relaxed);
            crate::xlog_info!(
                "D3D12.X: saved pipeline library gen-{gen:08x} ({} MiB, {} new pipelines)",
                size >> 20,
                dirty_before
            );
        }
        self.saving.store(false, Ordering::Release);
        saved.map(|(_, p)| p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            static SEQ: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!("xcl-pso-lib-{tag}-{}-{}", std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed)));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn generation_parsing_and_latest_selection() {
        assert_eq!(parse_generation("gen-00000001.plib"), Some(1));
        assert_eq!(parse_generation("gen-00000010.plib"), Some(0x10));
        assert_eq!(parse_generation("gen-00000002-1234.tmp"), None);
        assert_eq!(parse_generation("gen-.plib"), None);
        assert_eq!(parse_generation("other.plib"), None);

        let dir = TempDir::new("latest");
        std::fs::write(dir.0.join("gen-00000001.plib"), b"a").unwrap();
        std::fs::write(dir.0.join("gen-0000000a.plib"), b"bb").unwrap();
        std::fs::write(dir.0.join("gen-00000002.plib"), b"c").unwrap();
        std::fs::write(dir.0.join("gen-000000ff-99.tmp"), b"tmp").unwrap();
        let (gen, path) = latest_generation(&dir.0).unwrap();
        assert_eq!(gen, 0xA);
        assert!(path.ends_with("gen-0000000a.plib"));
    }

    #[test]
    fn mapped_blob_rejects_empty_and_missing_files() {
        let dir = TempDir::new("empty");
        assert!(MappedBlob::open(&dir.0.join("missing.plib")).is_none());
        let empty = dir.0.join("empty.plib");
        std::fs::write(&empty, b"").unwrap();
        assert!(MappedBlob::open(&empty).is_none());
    }

    #[test]
    fn mapped_blob_reads_and_keeps_file_open_across_generation_save() {
        let dir = TempDir::new("gen-save");
        let store = LibraryStore::new(dir.0.clone());

        let payload1: Vec<u8> = (0..16_384usize).map(|i| (i.wrapping_mul(31) ^ 0x5A) as u8).collect();
        let (gen1, path1) = store
            .write_generation(payload1.len(), |buf| {
                buf.copy_from_slice(&payload1);
                true
            })
            .expect("gen1 write");
        assert_eq!(gen1, 1);

        // Open gen1 memory-mapped and keep its mapping open while saving gen2.
        let (opened_gen, blob1) = store.open_latest().expect("open gen1");
        assert_eq!(opened_gen, 1);
        assert_eq!(blob1.as_slice(), payload1.as_slice());

        let payload2: Vec<u8> = (0..32_768usize).map(|i| (i.wrapping_mul(17) ^ 0xA5) as u8).collect();
        let (gen2, path2) = store
            .write_generation(payload2.len(), |buf| {
                buf.copy_from_slice(&payload2);
                true
            })
            .expect("gen2 write while gen1 mapped");
        assert_eq!(gen2, 2);
        assert_ne!(path1, path2);

        // gen1 mapping MUST remain completely valid and unmodified while open.
        assert_eq!(blob1.as_slice(), payload1.as_slice());
        drop(blob1);

        // Re-opening latest selects gen2 and cleans up stale gen1.
        let (opened_gen2, blob2) = store.open_latest().expect("open gen2");
        assert_eq!(opened_gen2, 2);
        assert_eq!(blob2.as_slice(), payload2.as_slice());
        assert!(!path1.exists(), "older generation should be cleaned up once unmapped");
        drop(blob2);
    }
}
