RWByteAddressBuffer output : register(u0, space0);

[numthreads(8, 4, 2)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    output.Store(dispatch_id.x * 4, WaveActiveSum(dispatch_id.x));
}
