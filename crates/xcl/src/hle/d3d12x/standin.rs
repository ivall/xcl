//! The stand-ins of pipelines not built yet (`pso_async`) share the driver's pipelines.
//!
//! A title gets an object of its own for each pipeline it makes, and a lazy one may never be
//! built. With a driver pipeline behind each of them a retail title's ~82 000 pipelines left
//! ~90 000 copies of the dummy shaders' programs on the process heap (4.8 GB in all). A stand-in
//! draws nothing whatever its state is, so those that differ only in what cannot matter (`Key`)
//! get one driver pipeline, and the title a small object (`Shell`) that is never given to the
//! driver: `pso_async::resolve` and its kin return the driver pipeline in its place.
//!
//! `XCL_SHARED_STANDINS=0`: off (a driver pipeline for each, as the title's own object).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::core::{Interface, IUnknown, GUID};
use windows::Win32::Graphics::Direct3D::WKPDID_D3DDebugObjectNameW;
use windows::Win32::Graphics::Direct3D12::{ID3D12DeviceChild, ID3D12Object, ID3D12Pageable, ID3D12PipelineState};

use crate::util::{HRESULT, E_INVALIDARG, E_NOINTERFACE, S_OK};

const DXGI_ERROR_NOT_FOUND: HRESULT = 0x887A_0002u32 as HRESULT;
const DXGI_ERROR_MORE_DATA: HRESULT = 0x887A_0003u32 as HRESULT;

pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    // (console-shaped lists give the PC list their pipelines' objects themselves)
    *E.get_or_init(|| std::env::var("XCL_SHARED_STANDINS").map_or(true, |v| v != "0") && !super::console_list::enabled())
}

/// What a driver pipeline must agree on with the pipelines it stands in for: the device, the
/// root signature and what the output merger checks.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Key {
    pub device: usize,
    pub root_signature: usize,
    pub kind: u8,
    pub state: Vec<u8>,
}

/// A driver pipeline and the root signature it was made with, held by the shells it stands for.
struct Shared {
    pipeline: usize,
    root_signature: usize,
}

impl Drop for Shared {
    fn drop(&mut self) {
        unsafe {
            unref(self.pipeline);
            if self.root_signature != 0 {
                unref(self.root_signature);
            }
        }
    }
}

static SHARED: Mutex<Option<HashMap<Key, Weak<Shared>>>> = Mutex::new(None);

unsafe fn method(obj: usize, slot: usize) -> usize {
    *(*(obj as *const *const usize)).add(slot)
}

unsafe fn addref(obj: usize) {
    let f: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(method(obj, 1));
    f(obj);
}

unsafe fn unref(obj: usize) {
    let f: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(method(obj, 2));
    f(obj);
}

/// A title's object for a pipeline, and the driver pipeline that stands in for it. `make` makes
/// that one (its reference is taken over) when no other stand-in of `key` is alive.
///
/// # Safety
/// `key.root_signature` is 0 or a live object; `make` returns a live pipeline or an error.
pub unsafe fn shell(key: Key, make: impl FnOnce() -> Result<usize, HRESULT>) -> Result<(*mut c_void, usize), HRESULT> {
    let found = SHARED.lock().unwrap().as_ref().and_then(|m| m.get(&key)).and_then(Weak::upgrade);
    let shared = match found {
        Some(s) => s,
        None => {
            // Outside the lock: the driver's call; if another thread filled `key` meanwhile,
            // reuse the winner and release the redundant driver pipeline outside the lock.
            let pipeline = make()?;
            let mut g = SHARED.lock().unwrap();
            let m = g.get_or_insert_with(HashMap::new);
            if let Some(existing) = m.get(&key).and_then(Weak::upgrade) {
                drop(g);
                unref(pipeline);
                existing
            } else {
                if key.root_signature != 0 { addref(key.root_signature); }
                let s = Arc::new(Shared { pipeline, root_signature: key.root_signature });
                if m.len() >= 64 && m.len().is_power_of_two() { m.retain(|_, w| w.strong_count() != 0); }
                let grew = m.insert(key, Arc::downgrade(&s)).is_none();
                let n = m.len();
                drop(g);
                if grew && n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: {n} driver pipelines stand in for the pipelines not built yet");
                }
                s
            }
        }
    };
    let pipeline = shared.pipeline;
    let p = Box::into_raw(Box::new(Shell { vtable: vtable() as *const usize, refs: AtomicU32::new(1), shared, data: Mutex::new(Vec::new()) }));
    Ok((p as *mut c_void, pipeline))
}

/// ID3D12PipelineState as the title sees it.
#[repr(C)]
struct Shell {
    vtable: *const usize,
    refs: AtomicU32,
    shared: Arc<Shared>,
    /// private data (an object's own: the driver pipeline is shared)
    data: Mutex<Vec<(u128, Data)>>,
}

enum Data {
    Bytes(Vec<u8>),
    /// an interface, with a reference
    Interface(usize),
}

impl Drop for Data {
    fn drop(&mut self) {
        if let Data::Interface(p) = self {
            unsafe { unref(*p) };
        }
    }
}

fn is_pipeline_iid(iid: &GUID) -> bool {
    [IUnknown::IID, ID3D12Object::IID, ID3D12DeviceChild::IID, ID3D12Pageable::IID, ID3D12PipelineState::IID].contains(iid)
}

unsafe extern "system" fn query_interface(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    if ppv.is_null() { return E_INVALIDARG; }
    *ppv = std::ptr::null_mut();
    if riid.is_null() { return E_INVALIDARG; }
    if !is_pipeline_iid(&*riid) { return E_NOINTERFACE; }
    // `Shell`'s vtable implements the five single-inheritance pipeline IIDs (`IUnknown` through
    // `ID3D12PipelineState`). Real `d3d12.dll` pipeline objects can return a subobject pointer
    // distinct from `pipeline` for base IIDs, so accept any non-null success from the driver,
    // release that driver reference, and return `this`.
    let pipeline = (&*(this as *const Shell)).shared.pipeline;
    let f: unsafe extern "system" fn(usize, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(method(pipeline, 0));
    let mut out = std::ptr::null_mut();
    let hr = f(pipeline, riid, &mut out);
    if hr < 0 { return hr; }
    if out.is_null() { return E_NOINTERFACE; }
    unref(out as usize);
    add_ref(this);
    *ppv = this;
    S_OK
}

unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    (&*(this as *const Shell)).refs.fetch_add(1, Ordering::AcqRel) + 1
}

unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    let n = (&*(this as *const Shell)).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if n == 0 {
        super::real::standin_released(this as usize);
        drop(Box::from_raw(this as *mut Shell));
    }
    n
}

unsafe extern "system" fn get_private_data(this: *mut c_void, guid: *const GUID, size: *mut u32, out: *mut u8) -> HRESULT {
    if guid.is_null() || size.is_null() {
        return E_INVALIDARG;
    }
    let data = (&*(this as *const Shell)).data.lock().unwrap();
    let Some((_, d)) = data.iter().find(|(g, _)| *g == (*guid).to_u128()) else {
        *size = 0;
        return DXGI_ERROR_NOT_FOUND;
    };
    let interface;
    let bytes: &[u8] = match d {
        Data::Bytes(b) => b,
        Data::Interface(p) => {
            interface = p.to_le_bytes();
            &interface
        }
    };
    let room = std::mem::replace(&mut *size, bytes.len() as u32) as usize;
    if out.is_null() {
        return S_OK;
    }
    if room < bytes.len() {
        return DXGI_ERROR_MORE_DATA;
    }
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
    if let Data::Interface(p) = d {
        addref(*p);
    }
    S_OK
}

unsafe fn put(this: *mut c_void, guid: *const GUID, d: Option<Data>) -> HRESULT {
    if guid.is_null() {
        return E_INVALIDARG;
    }
    let id = (*guid).to_u128();
    let old = {
        let mut data = (&*(this as *const Shell)).data.lock().unwrap();
        let old = data.iter().position(|(g, _)| *g == id).map(|i| data.swap_remove(i));
        if let Some(d) = d {
            data.push((id, d));
        }
        old
    };
    // (outside the lock: an interface's Release is the title's code)
    drop(old);
    S_OK
}

unsafe extern "system" fn set_private_data(this: *mut c_void, guid: *const GUID, size: u32, data: *const u8) -> HRESULT {
    if guid.is_null() || (size != 0 && data.is_null()) { return E_INVALIDARG; }
    put(this, guid, (size != 0).then(|| Data::Bytes(std::slice::from_raw_parts(data, size as usize).to_vec())))
}

unsafe extern "system" fn set_private_data_interface(this: *mut c_void, guid: *const GUID, obj: usize) -> HRESULT {
    if guid.is_null() { return E_INVALIDARG; }
    if obj != 0 {
        addref(obj);
    }
    put(this, guid, (obj != 0).then_some(Data::Interface(obj)))
}

unsafe extern "system" fn set_name(this: *mut c_void, name: *const u16) -> HRESULT {
    if name.is_null() { return E_INVALIDARG; }
    let mut len = 0usize;
    while *name.add(len) != 0 { len += 1; }
    let bytes = std::slice::from_raw_parts(name as *const u8, (len + 1) * std::mem::size_of::<u16>());
    put(this, &WKPDID_D3DDebugObjectNameW, Some(Data::Bytes(bytes.to_vec())))
}

unsafe extern "system" fn get_device(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let pipeline = (&*(this as *const Shell)).shared.pipeline;
    let f: unsafe extern "system" fn(usize, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(method(pipeline, 7));
    f(pipeline, riid, ppv)
}

unsafe extern "system" fn get_cached_blob(this: *mut c_void, ppv: *mut *mut c_void) -> HRESULT {
    let pipeline = (&*(this as *const Shell)).shared.pipeline;
    let f: unsafe extern "system" fn(usize, *mut *mut c_void) -> HRESULT = std::mem::transmute(method(pipeline, 8));
    f(pipeline, ppv)
}

const SLOTS: usize = 64;

fn vtable() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        let known: [usize; 9] = [
            query_interface as usize,
            add_ref as usize,
            release as usize,
            get_private_data as usize,
            set_private_data as usize,
            set_private_data_interface as usize,
            set_name as usize,
            get_device as usize,
            get_cached_blob as usize,
        ];
        let v: Vec<usize> = (0..SLOTS)
            .map(|i| known.get(i).copied().unwrap_or_else(|| crate::stubs::make_stub_info("d3d12_xs.dll", &format!("ID3D12PipelineState(stand-in)::#{i}"), 0).0))
            .collect();
        Box::leak(v.into_boxed_slice()).as_ptr() as usize
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A COM object that counts its references and has one interface.
    #[repr(C)]
    struct Counted {
        vtable: *const usize,
        refs: AtomicU32,
    }

    const KNOWN: GUID = GUID::from_u128(0x765a30f3_f624_4c6f_a828_ace948622445);

    unsafe extern "system" fn qi(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
        if *riid != KNOWN {
            *ppv = std::ptr::null_mut();
            return 0x8000_4002u32 as HRESULT;
        }
        up(this);
        *ppv = this;
        S_OK
    }
    unsafe extern "system" fn up(this: *mut c_void) -> u32 {
        (*(this as *const Counted)).refs.fetch_add(1, Ordering::AcqRel) + 1
    }
    unsafe extern "system" fn down(this: *mut c_void) -> u32 {
        (*(this as *const Counted)).refs.fetch_sub(1, Ordering::AcqRel) - 1
    }

    fn counted() -> &'static Counted {
        let v: &'static [usize; 3] = Box::leak(Box::new([qi as usize, up as usize, down as usize]));
        Box::leak(Box::new(Counted { vtable: v.as_ptr(), refs: AtomicU32::new(1) }))
    }

    fn at(c: &Counted) -> usize {
        c as *const _ as usize
    }

    fn refs(c: &Counted) -> u32 {
        c.refs.load(Ordering::Acquire)
    }

    fn key(rs: &Counted, state: u8) -> Key {
        Key { device: 1, root_signature: at(rs), kind: 0, state: vec![state] }
    }

    #[test]
    fn pipelines_of_one_key_share_a_driver_pipeline_while_one_of_them_lives() {
        let (rs, a, b) = (counted(), counted(), counted());
        unsafe {
            let (s1, d1) = shell(key(rs, 1), || Ok(at(a))).unwrap();
            let (s2, d2) = shell(key(rs, 1), || panic!("made again")).unwrap();
            assert_ne!(s1, s2, "the title's objects are its own");
            assert_eq!((d1, d2), (at(a), at(a)));
            let (s3, d3) = shell(key(rs, 2), || Ok(at(b))).unwrap();
            assert_eq!(d3, at(b), "another state, another pipeline");
            assert_eq!(refs(rs), 3, "the root signature is held by each driver pipeline");
            assert_eq!(release(s1), 0);
            assert_eq!(refs(a), 1, "still stands in for the second");
            assert_eq!(release(s2), 0);
            assert_eq!((refs(a), refs(rs)), (0, 2), "gone with the last");
            let c = counted();
            let (s4, d4) = shell(key(rs, 1), || Ok(at(c))).unwrap();
            assert_eq!(d4, at(c), "made again when none of its key is alive");
            release(s3);
            release(s4);
            assert_eq!(refs(rs), 1);
            assert_eq!(shell(key(rs, 9), || Err(E_INVALIDARG)).err(), Some(E_INVALIDARG));
            assert_eq!(refs(rs), 1);
        }
    }

    #[test]
    fn a_shell_has_the_drivers_interfaces_and_its_own_references_and_private_data() {
        let (rs, a, held) = (counted(), counted(), counted());
        unsafe {
            let (s, _) = shell(key(rs, 3), || Ok(at(a))).unwrap();
            let (other, _) = shell(key(rs, 3), || panic!("made again")).unwrap();
            let mut out = std::ptr::null_mut();
            assert_eq!(query_interface(s, &KNOWN, &mut out), S_OK);
            assert_eq!(out, s, "the title keeps its object");
            assert_eq!(refs(a), 1, "the driver's is not handed out");
            assert_eq!(release(s), 1);
            assert!(query_interface(s, &GUID::from_u128(7), &mut out) < 0 && out.is_null());

            let id = GUID::from_u128(0x1234);
            let mut n = 99u32;
            assert_eq!(get_private_data(s, &id, &mut n, std::ptr::null_mut()), DXGI_ERROR_NOT_FOUND);
            assert_eq!(n, 0);
            assert_eq!(set_private_data(s, &id, 3, [1u8, 2, 3].as_ptr()), S_OK);
            let mut buf = [0u8; 8];
            n = 2;
            assert_eq!(get_private_data(s, &id, &mut n, buf.as_mut_ptr()), DXGI_ERROR_MORE_DATA);
            assert_eq!(n, 3);
            assert_eq!(get_private_data(s, &id, &mut n, buf.as_mut_ptr()), S_OK);
            assert_eq!(&buf[..3], &[1, 2, 3]);
            assert_eq!(get_private_data(other, &id, &mut n, buf.as_mut_ptr()), DXGI_ERROR_NOT_FOUND, "another object's data is its own");
            assert_eq!(set_private_data_interface(s, &id, at(held)), S_OK);
            n = 8;
            assert_eq!(get_private_data(s, &id, &mut n, buf.as_mut_ptr()), S_OK);
            assert_eq!((usize::from_le_bytes(buf), refs(held)), (at(held), 3), "handed out with a reference");
            assert_eq!(set_private_data(s, &id, 0, std::ptr::null()), S_OK);
            assert_eq!(refs(held), 2, "removed");
            assert_eq!(set_private_data_interface(s, &id, at(held)), S_OK);
            release(s);
            assert_eq!(refs(held), 2, "let go with the object");
            release(other);
            assert_eq!(refs(a), 0);
        }
    }

    #[test]
    fn concurrent_shell_miss_reuses_winner_and_releases_loser_pipeline() {
        let (rs, p1, p2, barrier) = (counted(), counted(), counted(), Arc::new(std::sync::Barrier::new(2)));
        let k = key(rs, 41);
        let run = |p: usize| {
            let (b, k) = (barrier.clone(), k.clone());
            std::thread::spawn(move || unsafe { let (s, d) = shell(k, || { b.wait(); Ok(p) }).unwrap(); (s as usize, d) })
        };
        let (t1, t2) = (run(at(p1)), run(at(p2)));
        let ((s1, d1), (s2, d2)) = (t1.join().unwrap(), t2.join().unwrap());
        assert!(s1 != s2 && d1 == d2);
        let (winner, loser) = if d1 == at(p1) { (p1, p2) } else { (p2, p1) };
        assert_eq!((refs(winner), refs(loser), refs(rs)), (1, 0, 2));
        unsafe { release(s1 as *mut c_void); release(s2 as *mut c_void); }
        assert_eq!((refs(p1), refs(p2), refs(rs)), (0, 0, 1));
    }

    #[test]
    fn query_interface_rejects_secondary_interfaces_and_zeroes_out_on_null_riid() {
        const SECONDARY: GUID = GUID::from_u128(0xdead_beef_1234_5678_9abc_def0_1122_3344);
        #[repr(C)]
        struct Multi { vtable: *const usize, refs: AtomicU32, tearoff: *const Counted }
        unsafe extern "system" fn multi_qi(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
            let m = &*(this as *const Multi);
            if *riid == IUnknown::IID && !m.tearoff.is_null() { up(m.tearoff as *mut c_void); *ppv = m.tearoff as *mut c_void; return S_OK; }
            if is_pipeline_iid(&*riid) || *riid == SECONDARY { up(this); *ppv = this; return S_OK; }
            *ppv = std::ptr::null_mut();
            E_NOINTERFACE
        }
        let mk = |t: *const Counted| -> &'static Multi {
            let v: &'static [usize; 3] = Box::leak(Box::new([multi_qi as usize, up as usize, down as usize]));
            Box::leak(Box::new(Multi { vtable: v.as_ptr(), refs: AtomicU32::new(1), tearoff: t }))
        };
        let (rs, primary, tearoff) = (counted(), mk(std::ptr::null()), counted());
        let tear_host = mk(tearoff as *const Counted);
        unsafe {
            let (s, _) = shell(key(rs, 42), || Ok(primary as *const Multi as usize)).unwrap();
            let mut out = s;
            assert_eq!((query_interface(s, std::ptr::null(), &mut out), out.is_null()), (E_INVALIDARG, true));
            for iid in [IUnknown::IID, ID3D12Object::IID, ID3D12DeviceChild::IID, ID3D12Pageable::IID, ID3D12PipelineState::IID] {
                out = std::ptr::null_mut();
                assert_eq!((query_interface(s, &iid, &mut out), out, primary.refs.load(Ordering::Acquire)), (S_OK, s, 1));
                assert_eq!(release(s), 1);
            }
            out = s;
            assert_eq!((query_interface(s, &SECONDARY, &mut out), out.is_null(), primary.refs.load(Ordering::Acquire)), (E_NOINTERFACE, true, 1));
            release(s);
            let (s2, _) = shell(key(rs, 43), || Ok(tear_host as *const Multi as usize)).unwrap();
            out = std::ptr::null_mut();
            assert_eq!((query_interface(s2, &IUnknown::IID, &mut out), out, refs(tearoff)), (S_OK, s2, 1));
            assert_eq!(release(s2), 1);
            out = s2;
            assert_eq!((query_interface(s2, &SECONDARY, &mut out), out.is_null(), refs(tearoff)), (E_NOINTERFACE, true, 1));
            release(s2);
        }
    }

    #[test]
    fn query_interface_on_real_d3d12_pipeline_accepts_hierarchy_and_rejects_other() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::*;
        let Ok(dxil) = crate::gpu::recomp::dxc::compile("[RootSignature(\"\")][numthreads(1,1,1)] void main() {}", "main", "cs_6_0", &[]) else {
            return crate::gpu::recomp::runner::skip_test("no DXC");
        };
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(f) = CreateDXGIFactory1::<IDXGIFactory4>() else { return crate::gpu::recomp::runner::skip_test("no GPU or WARP"); };
                let Ok(w) = f.EnumWarpAdapter::<IDXGIAdapter>() else { return crate::gpu::recomp::runner::skip_test("no GPU or WARP"); };
                if D3D12CreateDevice(&w, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() { return crate::gpu::recomp::runner::skip_test("no GPU or WARP"); }
            }
            let dev = dev.unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, &dxil).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).unwrap();
            let _ = std::mem::ManuallyDrop::into_inner(desc.pRootSignature);
            let pso_raw = pso.into_raw() as usize;
            let k = Key { device: dev.as_raw() as usize, root_signature: rs.as_raw() as usize, kind: 1, state: vec![0x51] };
            let (s, _) = shell(k, || Ok(pso_raw)).unwrap();
            for iid in [IUnknown::IID, ID3D12Object::IID, ID3D12DeviceChild::IID, ID3D12Pageable::IID, ID3D12PipelineState::IID] {
                let mut out = std::ptr::null_mut();
                assert_eq!(query_interface(s, &iid, &mut out), S_OK, "real pipeline QI failed for {iid:?}");
                assert_eq!(out, s);
                assert_eq!(release(s), 1);
            }
            let mut out = s;
            assert_eq!(query_interface(s, &ID3D12DebugCommandList::IID, &mut out), E_NOINTERFACE);
            assert!(out.is_null());
            assert_eq!(release(s), 0);
        }
    }

    #[test]
    fn set_private_data_validation_and_per_shell_set_name() {
        let (rs, a, id) = (counted(), counted(), GUID::from_u128(0x5678));
        unsafe {
            let (s1, _) = shell(key(rs, 44), || Ok(at(a))).unwrap();
            let (s2, _) = shell(key(rs, 44), || panic!("shared")).unwrap();
            assert_eq!((set_private_data(s1, &id, 3, [4u8, 5, 6].as_ptr()), set_private_data(s1, &id, 4, std::ptr::null())), (S_OK, E_INVALIDARG));
            let (mut buf, mut n) = ([0u8; 16], 16u32);
            assert_eq!((get_private_data(s1, &id, &mut n, buf.as_mut_ptr()), n, &buf[..3]), (S_OK, 3, &[4, 5, 6][..]));
            assert_eq!(set_private_data(s1, &id, 0, [9u8].as_ptr()), S_OK);
            n = 16; assert_eq!(get_private_data(s1, &id, &mut n, buf.as_mut_ptr()), DXGI_ERROR_NOT_FOUND);
            let name: Vec<u16> = "sky\0".encode_utf16().collect();
            assert_eq!((set_name(s1, std::ptr::null()), set_name(s1, name.as_ptr()), set_name(s1, std::ptr::null())), (E_INVALIDARG, S_OK, E_INVALIDARG));
            n = 16; assert_eq!((get_private_data(s1, &WKPDID_D3DDebugObjectNameW, &mut n, buf.as_mut_ptr()), n, &buf[..8]), (S_OK, 8, &[b's', 0, b'k', 0, b'y', 0, 0, 0][..]));
            n = 16; assert_eq!(get_private_data(s2, &WKPDID_D3DDebugObjectNameW, &mut n, buf.as_mut_ptr()), DXGI_ERROR_NOT_FOUND);
            release(s1); release(s2);
        }
    }
}
