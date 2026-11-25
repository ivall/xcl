use super::*;
use crate::gpu::recomp::{dxc, pso, runner};
use std::sync::Arc;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;

/// A synthetic guest call site exercises the same origin and layout checks as the title.
unsafe fn invoke_guest_slot(list: *mut c_void, slot: u32, shifted: bool) {
    let _layout = real::QUEUE_LAYOUT_TEST.lock().unwrap();
    let mut code = vec![0x48, 0x83, 0xEC, 0x28, 0x48, 0x8B, 0x01, 0xFF, 0x90];
    code.extend_from_slice(&(slot * 8).to_le_bytes());
    code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x28, 0xC3]);
    let address = crate::stubs::alloc_code(code.len());
    std::ptr::copy_nonoverlapping(code.as_ptr(), address, code.len());
    let base = address as usize;
    crate::modules::register_mapped(Arc::new(crate::modules::MappedModule {
        name: "vrs-test.exe".into(), path: "vrs-test.exe".into(), base, size: code.len(),
        entry_rva: 0, is_dll: false, exports: Default::default(),
        tls_callbacks: Mutex::new(Vec::new()), no_thread_calls: AtomicBool::new(false),
        initialized: AtomicBool::new(true), stack_reserve: 0,
    }));
    let previous = real::QUEUE_SHIFTED.swap(shifted, Ordering::AcqRel);
    let invoke: unsafe extern "system" fn(*mut c_void, D3D12_SHADING_RATE, *const D3D12_SHADING_RATE_COMBINER) =
        std::mem::transmute(address);
    let combiners = [D3D12_SHADING_RATE_COMBINER_PASSTHROUGH; 2];
    invoke(list, D3D12_SHADING_RATE_2X2, combiners.as_ptr());
    real::QUEUE_SHIFTED.store(previous, Ordering::Release);
    crate::modules::unregister_mapped(base);
}

#[test]
fn shifted_vrs_draws_with_requested_rate() {
    const SIZE: u32 = 16;
    const HLSL: &str = "
float4 vs_main(uint id : SV_VertexID) : SV_Position {
    float2 p = id == 0 ? float2(-1, -1) : (id == 1 ? float2(-1, 3) : float2(3, -1));
    return float4(p, 0, 1);
}
float4 ps_main(uint rate : SV_ShadingRate) : SV_Target {
    return float4(float(rate), 0, 0, 1);
}";

    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            runner::skip_test("no D3D12 GPU");
            return;
        }
        let dev = dev.unwrap();
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS6::default();
        if dev.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS6, (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS6).cast(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS6>() as u32,
        ).is_err() || options.VariableShadingRateTier != D3D12_VARIABLE_SHADING_RATE_TIER_2 {
            runner::skip_test("VRS tier 2 required for SV_ShadingRate readback");
            return;
        }
        if dxc::location().is_err() {
            runner::skip_test("no DXC");
            return;
        }
        let vs = dxc::compile(HLSL, "vs_main", "vs_6_4", &[]).unwrap();
        let ps = dxc::compile(HLSL, "ps_main", "ps_6_4", &[]).unwrap();
        let root = pso::abi_root_signature(&dev, true).unwrap();
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: transmute_copy(&root),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr().cast(), BytecodeLength: ps.len() },
            SampleMask: u32::MAX,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        desc.RasterizerState.DepthClipEnable = true.into();
        desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        let pso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&desc).unwrap();
        let draw = runner::Draw { sgprs: Vec::new(), scissor: [0, 0, SIZE as i32, SIZE as i32] };
        let read = |route| {
            let pixels = runner::draw_on_multi_configured(
                &dev, &pso, &root, &[0.0; 3], 4, std::slice::from_ref(&draw), &[], &[], &[], SIZE, [-1.0; 4],
                |list| {
                    if let Some((slot, shifted)) = route {
                        let raw = list.as_raw();
                        real::adopt(raw, real::Kind::CommandList);
                        invoke_guest_slot(raw, slot, shifted);
                    }
                },
            ).unwrap();
            pixels[(SIZE * (SIZE / 2) + SIZE / 2) as usize]
        };
        let ordinary = read(None);
        let older = read(Some((124, false)));
        let shifted = read(Some((126, true)));
        assert_eq!(ordinary, [0.0, 0.0, 0.0, 1.0], "default shading rate");
        assert_eq!(older, [5.0, 0.0, 0.0, 1.0], "older slot must retain 2x2");
        assert_eq!(shifted, older, "shifted slot must forward the same 2x2 rate");
        eprintln!("VRS readback tier={} default={ordinary:?} old124={older:?} new126={shifted:?}", options.VariableShadingRateTier.0);
    }
}
