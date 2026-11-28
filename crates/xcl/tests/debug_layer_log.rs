//! `XCL_D3D12_DEBUG=1`: the D3D12 debug layer's messages reach XCL's log, also on systems
//! without `ID3D12InfoQueue1` (Windows 10), where the message callback XCL registers does not
//! exist. Its own process: the debug layer has to be on before the first device exists.

use windows::core::{Interface, PCSTR};
use windows::Win32::Graphics::Direct3D12::*;

use xcl::gpu::recomp::runner;
use xcl::hle::d3d12x::real;

/// A message in the device's debug layer queue is in the log after what XCL does at every
/// submission and present (`stop_if_device_removed`).
#[test]
fn debug_layer_messages_reach_the_log() {
    let dir = std::env::temp_dir().join(format!("xcl-debug-layer-log-{}", std::process::id()));
    let path = dir.join("xcl.log");
    xcl::log::init(Some(&path), xcl::log::Level::Info, false);
    std::env::set_var("XCL_D3D12_DEBUG", "1");
    if xcl::hle::d3d12x::backend() != xcl::hle::d3d12x::Backend::Real {
        runner::skip_test("no GPU");
        return;
    }
    let raw = real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }.expect("backend device").clone();
    let Ok(queue) = dev.cast::<ID3D12InfoQueue>() else {
        println!("SKIPPED debug_layer_messages_reach_the_log: no D3D12 debug layer");
        return;
    };
    let text = "xcl test: a debug layer message";
    let c = std::ffi::CString::new(text).unwrap();
    unsafe { queue.AddApplicationMessage(D3D12_MESSAGE_SEVERITY_ERROR, PCSTR(c.as_ptr() as *const u8)) }.expect("AddApplicationMessage");
    real::stop_if_device_removed();
    xcl::log::flush();
    let log = std::fs::read_to_string(&path).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(log.contains("D3D12: debug layer enabled"), "the debug layer did not come up:\n{log}");
    assert!(log.lines().any(|l| l.contains("D3D12 debug layer (error") && l.contains(text)), "the debug layer's message is not in the log:\n{log}");
}
