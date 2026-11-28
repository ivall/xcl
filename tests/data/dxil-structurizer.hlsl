// Synthetic compiler-only case for #611: a window switch with both
// function and loop exits. No game shader or XCL translation is involved.
RWByteAddressBuffer mem0 : register(u0);
RWByteAddressBuffer mem1 : register(u1);
RWByteAddressBuffer mem2 : register(u2);
RWByteAddressBuffer mem3 : register(u3);
RWByteAddressBuffer sink : register(u4);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = sink.Load((id.x & 255u) * 4u);
    [loop] for (uint i = 0; i < 32u; ++i) {
        uint address = (x & 1023u) * 4u;
        [branch] switch ((x >> 16u) & 3u) {
        case 1u:
            x ^= mem1.Load(address);
            if ((x & 7u) == 1u) continue;
            break;
        case 2u:
            x ^= mem2.Load(address);
            if ((x & 7u) == 2u) return;
            break;
        case 3u:
            x ^= mem3.Load(address);
            if ((x & 7u) == 3u) continue;
            break;
        default:
            x ^= mem0.Load(address);
            break;
        }
        if ((x & 255u) == 0u) break;
    }
    sink.Store(id.x * 4u, x);
}
