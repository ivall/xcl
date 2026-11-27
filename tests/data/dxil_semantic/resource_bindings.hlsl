cbuffer Params : register(b2, space3) { float gain; };
Texture2D<float4> InputTex[2] : register(t1, space3);
Texture2D<float4> Unbounded[] : register(t11, space15);
SamplerState InputSampler : register(s4, space3);
RWTexture2D<float4> OutputTex : register(u5, space3);
tbuffer Legacy : register(t2, space12) { float4 legacyValue; };

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    OutputTex[tid.xy] = InputTex[NonUniformResourceIndex(tid.x & 1)].SampleLevel(InputSampler, float2(0.5, 0.5), 0) * gain + legacyValue + Unbounded[NonUniformResourceIndex(tid.x)].Load(int3(0, 0, 0));
}
