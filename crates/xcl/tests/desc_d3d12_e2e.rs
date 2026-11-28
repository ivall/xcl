//! End-to-end D3D12 debug-layer contract tests for `gpu::desc::{image_d3d12, sampler_d3d12}`.
//!
//! Exercises `CreateCommittedResource`, `CreateShaderResourceView`, and `CreateSampler`
//! under the D3D12 debug layer across all `ImageType` variants, mip/slice subranges,
//! RGBA/BGRA/BC formats and component swizzles, and sampler filter/reduction/comparison modes.

use std::ffi::c_void;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory4};

use xcl::gpu::agility;
use xcl::gpu::desc::{self, Image, ImageType};
use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::real;

const NEG_CONTROL_LUID_ENV: &str = "XCL_DESC_D3D12_NEG_CONTROL_LUID";
const NEG_CONTROL_DONE_MARKER: &str = "desc_d3d12_e2e negative control ok";

static DEBUG_ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

unsafe extern "system" fn collect_debug_message(
    _category: D3D12_MESSAGE_CATEGORY,
    severity: D3D12_MESSAGE_SEVERITY,
    id: D3D12_MESSAGE_ID,
    text: windows::core::PCSTR,
    _context: *mut c_void,
) {
    if severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
        let msg = if text.is_null() {
            String::new()
        } else {
            String::from_utf8_lossy(text.as_bytes()).into_owned()
        };
        DEBUG_ERRORS
            .lock()
            .unwrap()
            .push(format!("severity={} id={} {msg}", severity.0, id.0));
    }
}

fn take_debug_errors(iq: Option<&ID3D12InfoQueue>) -> Vec<String> {
    let mut errs = std::mem::take(&mut *DEBUG_ERRORS.lock().unwrap());
    if let Some(iq) = iq {
        unsafe {
            for i in 0..iq.GetNumStoredMessages() {
                let mut len = 0usize;
                if iq.GetMessage(i, None, &mut len).is_err() || len == 0 { continue; }
                let mut buf = vec![0u64; len.div_ceil(8)];
                let msg_ptr = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
                if iq.GetMessage(i, Some(msg_ptr), &mut len).is_ok() && (*msg_ptr).Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                    let msg = &*msg_ptr;
                    let desc = if msg.pDescription.is_null() {
                        String::new()
                    } else {
                        std::ffi::CStr::from_ptr(msg.pDescription as *const i8).to_string_lossy().into_owned()
                    };
                    errs.push(format!("stored id={} {desc}", msg.ID.0));
                }
            }
            iq.ClearStoredMessages();
        }
    }
    errs
}

fn base_image(ty: ImageType) -> Image {
    Image {
        address: 0x1000_0000,
        format: 56, // R8G8B8A8_UNORM
        width: 32,
        height: if matches!(ty, ImageType::D1 | ImageType::D1Array) { 1 } else { 32 },
        depth: match ty {
            ImageType::D1 | ImageType::D2 | ImageType::D2Msaa => 1,
            ImageType::D1Array | ImageType::D2Array | ImageType::D2MsaaArray => 4,
            ImageType::D3 => 8,
            ImageType::Cube => 6,
        },
        ty,
        base_level: 0,
        last_level: 0,
        max_mip: 0,
        base_array: 0,
        dst_sel: [4, 5, 6, 7],
        swizzle_mode: 0,
        min_lod: 0.0,
    }
}

fn check_srv_case(
    dev: &ID3D12Device,
    srv_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    iq: Option<&ID3D12InfoQueue>,
    label: &str,
    img: &Image,
) {
    let (mut res_desc, srv_desc) = desc::image_d3d12(img)
        .unwrap_or_else(|| panic!("{label}: image_d3d12 returned None for {img:?}"));
    if res_desc.SampleDesc.Count > 1 {
        res_desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        dev.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &res_desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        )
        .unwrap_or_else(|e| panic!("{label}: CreateCommittedResource failed: {e}"));
        let resource = resource.expect("CreateCommittedResource returned None");
        dev.CreateShaderResourceView(&resource, Some(&srv_desc), srv_handle);
    }
    let errs = take_debug_errors(iq);
    assert!(errs.is_empty(), "{label}: D3D12 debug errors: {errs:?}");
}

#[allow(clippy::too_many_arguments)]
fn ssharp_words(
    addr_u: u32,
    addr_v: u32,
    addr_w: u32,
    max_aniso_pow2: u32,
    compare_func: u32,
    reduction: u32,
    min_lod_u4_8: u32,
    max_lod_u4_8: u32,
    lod_bias_s5_8: i32,
    mag: u32,
    min: u32,
    mip: u32,
    border_type: u32,
) -> [u32; 4] {
    let w0 = (addr_u & 7)
        | ((addr_v & 7) << 3)
        | ((addr_w & 7) << 6)
        | ((max_aniso_pow2 & 7) << 9)
        | ((compare_func & 7) << 12)
        | ((reduction & 3) << 29);
    let w1 = (min_lod_u4_8 & 0xFFF) | ((max_lod_u4_8 & 0xFFF) << 12);
    let w2 = ((lod_bias_s5_8 as u32) & 0x3FFF)
        | ((mag & 3) << 20)
        | ((min & 3) << 22)
        | ((mip & 3) << 26);
    let w3 = (border_type & 3) << 30;
    [w0, w1, w2, w3]
}

fn check_sampler_case(
    dev: &ID3D12Device,
    sampler_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    iq: Option<&ID3D12InfoQueue>,
    label: &str,
    words: &[u32; 4],
    comparison: bool,
) {
    let desc = desc::sampler_d3d12(words, comparison);
    unsafe {
        dev.CreateSampler(&desc, sampler_handle);
    }
    let errs = take_debug_errors(iq);
    assert!(errs.is_empty(), "{label}: D3D12 debug errors: {errs:?}");
}

#[test]
fn desc_d3d12_srv_and_sampler_debug_layer() {
    if let Ok(luid_str) = std::env::var(NEG_CONTROL_LUID_ENV) {
        run_negative_control_child(&luid_str);
        return;
    }

    std::env::set_var("XCL_D3D12_DEBUG", "1");

    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real {
        println!("no real D3D12 device – skipping");
        runner::skip_test("no GPU");
        return;
    }

    let raw_dev = real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw_dev) }
        .expect("backend device")
        .clone();

    let iq = dev.cast::<ID3D12InfoQueue>().ok();
    let iq1 = dev.cast::<ID3D12InfoQueue1>().ok();
    let has_debug_layer = iq.is_some() || iq1.is_some();

    let mut cookie = 0u32;
    if let Some(ref q1) = iq1 {
        unsafe {
            let _ = q1.RegisterMessageCallback(
                Some(collect_debug_message),
                D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                std::ptr::null_mut(),
                &mut cookie,
            );
        }
    }
    let _ = take_debug_errors(iq.as_ref());

    let srv_heap: ID3D12DescriptorHeap = unsafe {
        dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 1,
            ..Default::default()
        })
    }
    .expect("CreateDescriptorHeap SRV");
    let sampler_heap: ID3D12DescriptorHeap = unsafe {
        dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            NumDescriptors: 1,
            ..Default::default()
        })
    }
    .expect("CreateDescriptorHeap SAMPLER");

    let srv_handle = unsafe { srv_heap.GetCPUDescriptorHandleForHeapStart() };
    let sampler_handle = unsafe { sampler_heap.GetCPUDescriptorHandleForHeapStart() };

    // 1. All ImageType variants with mip/slice subranges.
    let image_cases = [
        ("D1 mip subrange", Image { max_mip: 3, base_level: 1, last_level: 2, ..base_image(ImageType::D1) }),
        ("D1Array subrange", Image { max_mip: 3, base_level: 1, last_level: 2, base_array: 1, ..base_image(ImageType::D1Array) }),
        ("D2 mip subrange + min_lod", Image { width: 64, height: 32, max_mip: 4, base_level: 1, last_level: 3, min_lod: 1.5, ..base_image(ImageType::D2) }),
        ("D2Array subrange", Image { width: 64, height: 32, depth: 5, base_array: 2, max_mip: 3, last_level: 2, ..base_image(ImageType::D2Array) }),
        ("D3 mip subrange", Image { width: 32, height: 16, depth: 8, max_mip: 3, base_level: 1, last_level: 2, ..base_image(ImageType::D3) }),
        ("Cube single", Image { max_mip: 2, last_level: 2, ..base_image(ImageType::Cube) }),
        ("Cube array subrange", Image { depth: 12, base_array: 6, max_mip: 2, base_level: 1, last_level: 2, ..base_image(ImageType::Cube) }),
    ];
    for (label, img) in &image_cases {
        check_srv_case(&dev, srv_handle, iq.as_ref(), label, img);
    }

    for (samples_pow2, name) in [(1, "2x"), (2, "4x")] {
        let msaa = Image { last_level: samples_pow2, ..base_image(ImageType::D2Msaa) };
        check_srv_case(&dev, srv_handle, iq.as_ref(), &format!("D2Msaa {name}"), &msaa);

        let msaa_arr = Image { last_level: samples_pow2, base_array: 1, ..base_image(ImageType::D2MsaaArray) };
        check_srv_case(&dev, srv_handle, iq.as_ref(), &format!("D2MsaaArray {name}"), &msaa_arr);
    }

    // 2. Formats and component swizzles (RGBA, BGRA packed, BC).
    let format_cases: &[(&str, u32, [u32; 4], u32, u32)] = &[
        ("RGBA8 custom swizzle", 56, [6, 5, 4, 1], 16, 16),
        ("RGBA8_SRGB", 130, [4, 5, 6, 7], 16, 16),
        ("RGBA16F reverse swizzle", 71, [7, 6, 5, 4], 16, 16),
        ("R32F constant alpha", 22, [4, 0, 0, 1], 16, 16),
        ("B5G6R5", 133, [4, 5, 6, 1], 16, 16),
        ("B5G5R5A1", 134, [4, 5, 6, 7], 16, 16),
        ("B4G4R4A4 swizzled", 136, [6, 5, 4, 7], 16, 16),
        ("BC1_UNORM", 169, [4, 5, 6, 7], 16, 16),
        ("BC3_UNORM", 173, [4, 5, 6, 7], 16, 16),
        ("BC5_UNORM", 177, [4, 5, 0, 1], 16, 16),
        ("BC6H_UF16", 179, [4, 5, 6, 1], 16, 16),
        ("BC7_UNORM", 181, [4, 5, 6, 7], 16, 16),
    ];
    for &(label, fmt, dst_sel, w, h) in format_cases {
        let mut img = base_image(ImageType::D2);
        img.format = fmt;
        img.dst_sel = dst_sel;
        img.width = w;
        img.height = h;
        img.max_mip = 2;
        img.last_level = 2;
        check_srv_case(&dev, srv_handle, iq.as_ref(), label, &img);
    }

    // 3. Sampler filter, reduction, comparison, addressing, border, and LOD modes.
    let sampler_cases: &[(&str, [u32; 4], bool)] = &[
        ("point wrap mip0", ssharp_words(0, 0, 0, 0, 0, 0, 0, 12 * 256, 0, 0, 0, 0, 0), false),
        ("linear mirror-clamp-border", ssharp_words(1, 2, 6, 0, 0, 0, 256, 8 * 256, -384, 1, 1, 2, 2), false),
        ("aniso 8x mirror_once opaque black", ssharp_words(3, 3, 3, 3, 0, 0, 0, 13 * 256, 128, 2, 2, 2, 1), false),
        ("reduction min", ssharp_words(2, 2, 2, 0, 0, 1, 0, 10 * 256, 0, 1, 1, 2, 0), false),
        ("reduction max aniso", ssharp_words(2, 2, 2, 2, 0, 2, 0, 10 * 256, 0, 2, 2, 2, 0), false),
        ("comparison less linear", ssharp_words(2, 2, 2, 0, 1, 0, 0, 10 * 256, 0, 1, 1, 2, 0), true),
        ("comparison greater_equal aniso", ssharp_words(2, 2, 2, 3, 6, 0, 0, 10 * 256, 0, 2, 2, 2, 0), true),
    ];
    for &(label, words, cmp) in sampler_cases {
        check_sampler_case(&dev, sampler_handle, iq.as_ref(), label, &words, cmp);
    }

    // 4. Negative control: when the debug layer is active, a mismatched SRV dimension must
    // trigger a debug-layer error, proving that the InfoQueue check above is non-vacuous.
    // D3D12CreateDevice returns the existing per-adapter device singleton in the same process,
    // so run the invalid SRV in a child process to keep its RemoveDevice from poisoning
    // XCL's shared backend device (`real::device_raw()`).
    if has_debug_layer {
        let luid = unsafe { dev.GetAdapterLuid() };
        let exe = std::env::current_exe().expect("current_exe");
        let out = std::process::Command::new(&exe)
            .args([
                "--exact",
                "desc_d3d12_srv_and_sampler_debug_layer",
                "--nocapture",
            ])
            .env(
                NEG_CONTROL_LUID_ENV,
                format!("{}:{}", luid.HighPart, luid.LowPart),
            )
            .output()
            .expect("spawn negative control child");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "negative control child failed ({}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
            out.status
        );
        assert!(
            stdout.contains(NEG_CONTROL_DONE_MARKER),
            "negative control child did not emit completion marker:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    } else {
        println!("D3D12 debug layer unavailable on host – ran driver CreateCommittedResource/CreateShaderResourceView/CreateSampler without InfoQueue");
    }
    assert!(
        unsafe { dev.GetDeviceRemovedReason() }.is_ok(),
        "shared backend device should remain healthy"
    );

    if let Some(ref q1) = iq1 {
        unsafe {
            let _ = q1.UnregisterMessageCallback(cookie);
        }
    }
}

fn parse_luid(s: &str) -> LUID {
    let (high, low) = s
        .split_once(':')
        .unwrap_or_else(|| panic!("invalid {NEG_CONTROL_LUID_ENV} value: {s:?}"));
    LUID {
        HighPart: high
            .parse::<i32>()
            .unwrap_or_else(|e| panic!("invalid LUID HighPart {high:?}: {e}")),
        LowPart: low
            .parse::<u32>()
            .unwrap_or_else(|e| panic!("invalid LUID LowPart {low:?}: {e}")),
    }
}

fn run_negative_control_child(luid_str: &str) {
    let luid = parse_luid(luid_str);
    let dbg: ID3D12Debug = agility::configuration_interface(&CLSID_D3D12Debug)
        .expect("CLSID_D3D12Debug in negative-control child");
    unsafe {
        dbg.EnableDebugLayer();
    }

    let neg_dev = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("CreateDXGIFactory1");
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(luid).expect("EnumAdapterByLuid");
        agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0)
            .expect("create_device for negative control")
    };

    let neg_iq = neg_dev
        .cast::<ID3D12InfoQueue>()
        .expect("ID3D12InfoQueue in negative-control child");
    let neg_iq1 = neg_dev.cast::<ID3D12InfoQueue1>().ok();
    let mut neg_cookie = 0u32;
    if let Some(ref q1) = neg_iq1 {
        unsafe {
            let _ = q1.RegisterMessageCallback(
                Some(collect_debug_message),
                D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                std::ptr::null_mut(),
                &mut neg_cookie,
            );
        }
    }
    let setup_errs = take_debug_errors(Some(&neg_iq));
    assert!(
        setup_errs.is_empty(),
        "negative control setup produced unexpected D3D12 errors: {setup_errs:?}"
    );

    let neg_heap: ID3D12DescriptorHeap = unsafe {
        neg_dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 1,
            ..Default::default()
        })
    }
    .expect("CreateDescriptorHeap negative control");
    let neg_srv_handle = unsafe { neg_heap.GetCPUDescriptorHandleForHeapStart() };

    let (res_desc, _) = desc::image_d3d12(&base_image(ImageType::D2)).unwrap();
    let (_, bad_srv) = desc::image_d3d12(&base_image(ImageType::D3)).unwrap();
    let mut bad_res: Option<ID3D12Resource> = None;
    unsafe {
        neg_dev
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &res_desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut bad_res,
            )
            .expect("CreateCommittedResource negative control");
        let bad_res = bad_res.unwrap();
        neg_dev.CreateShaderResourceView(&bad_res, Some(&bad_srv), neg_srv_handle);
    }
    let neg_errs = take_debug_errors(Some(&neg_iq));
    assert!(
        !neg_errs.is_empty(),
        "negative control: expected D3D12 debug error for 3D SRV on 2D resource"
    );
    if let Some(ref q1) = neg_iq1 {
        unsafe {
            let _ = q1.UnregisterMessageCallback(neg_cookie);
        }
    }
    println!("{NEG_CONTROL_DONE_MARKER}");
}
