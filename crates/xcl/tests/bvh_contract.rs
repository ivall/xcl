//! Contract tests for `xcl::gpu::bvh` node decoding and traversal paths not covered
//! by the inline unit tests: box16 (type 4) half-float decoding and `sort=false` order,
//! paired-triangle leaves (type 0 + bit 7 -> type 1) with `@60` barycentric rotation bits,
//! unknown node types, ray `t < 0` / `tmax` culling, and `trace_scene` with missing BLASes.

use xcl::gpu::bvh::{build, build_top, intersect_node, prebuild_size, Bvh, Instance, BOX32, HEADER_BYTES, INVALID, ROOT_AT};

const BOX16: u32 = 4;

fn put_words(buf: &mut [u8], at: usize, words: &[u32]) {
    for (k, w) in words.iter().enumerate() {
        buf[at + 4 * k..at + 4 * k + 4].copy_from_slice(&w.to_le_bytes());
    }
}

fn put_tri(buf: &mut [u8], at: usize, verts: &[[f32; 3]; 3]) {
    let words: Vec<u32> = verts.iter().flatten().map(|x| x.to_bits()).collect();
    put_words(buf, at, &words);
}

fn pack_half_pair(lo: u16, hi: u16) -> u32 {
    (lo as u32) | ((hi as u32) << 16)
}

fn write_box16_child(buf: &mut [u8], node_at: usize, slot: usize, child: u32, mn: [u16; 3], mx: [u16; 3]) {
    put_words(buf, node_at + 4 * slot, &[child]);
    let w0 = pack_half_pair(mn[0], mn[1]);
    let w1 = pack_half_pair(mn[2], mx[0]);
    let w2 = pack_half_pair(mx[1], mx[2]);
    put_words(buf, node_at + 16 + 12 * slot, &[w0, w1, w2]);
}

#[test]
fn box16_half_float_decoding_and_sort_order() {
    // IEEE 754 binary16 constants:
    // -1.0 = 0xBC00, 1.0 = 0x3C00, 2.0 = 0x4000, 4.0 = 0x4400, 5.0 = 0x4500,
    //  6.0 = 0x4600, 8.0 = 0x4800, 10.0 = 0x4900, NaN = 0x7E00, subnormal 2^-24 = 0x0001
    let mut data = vec![0u8; 128];
    let (c0, c1, c2, c3) = (0x20u32, 0x30u32, 0x40u32, 0x50u32);

    // Child 0: z in [8.0, 10.0] (farther)
    write_box16_child(&mut data, 0, 0, c0, [0xBC00, 0xBC00, 0x4800], [0x3C00, 0x3C00, 0x4900]);
    // Child 1: z in [2.0, 4.0] (closer), x/y using subnormal +2^-24..1.0
    write_box16_child(&mut data, 0, 1, c1, [0x0001, 0xBC00, 0x4000], [0x3C00, 0x3C00, 0x4400]);
    // Child 2: NaN-disabled box (culled)
    write_box16_child(&mut data, 0, 2, c2, [0x7E00, 0xBC00, 0x4500], [0x3C00, 0x3C00, 0x4600]);
    // Child 3: inverted box mn.z > mx.z (culled)
    write_box16_child(&mut data, 0, 3, c3, [0xBC00, 0xBC00, 0x4600], [0x3C00, 0x3C00, 0x4500]);

    let o = [0.5, 0.0, 0.0];
    let d = [0.0, 0.0, 1.0];
    let inv = [f32::INFINITY, f32::INFINITY, 1.0];

    // sort = false preserves slot order [c0, c1, INVALID, INVALID]
    let unsorted = intersect_node(&data, BOX16, 100.0, o, d, inv, false);
    assert_eq!(unsorted, [c0, c1, INVALID, INVALID]);

    // sort = true orders closest entry t first: c1 (t=2.0) before c0 (t=8.0)
    let sorted = intersect_node(&data, BOX16, 100.0, o, d, inv, true);
    assert_eq!(sorted, [c1, c0, INVALID, INVALID]);

    // tmax = 5.0 culls c0 (entry t=8.0) while keeping c1 (entry t=2.0)
    let culled = intersect_node(&data, BOX16, 5.0, o, d, inv, true);
    assert_eq!(culled, [c1, INVALID, INVALID, INVALID]);
}

#[test]
fn box16_root_traversal_reaches_triangle_leaf() {
    let mut data = vec![0u8; HEADER_BYTES + 64 + 64];
    let box_at = HEADER_BYTES;
    let tri_at = HEADER_BYTES + 64;
    let root = ((box_at / 8) as u32) | BOX16;
    let tri_node = (tri_at / 8) as u32; // type 0

    put_words(&mut data, ROOT_AT, &[root]);
    write_box16_child(&mut data, box_at, 0, tri_node, [0xBC00, 0xBC00, 0x4000], [0x4000, 0x4000, 0x4400]);
    write_box16_child(&mut data, box_at, 1, INVALID, [0; 3], [0; 3]);
    write_box16_child(&mut data, box_at, 2, INVALID, [0; 3], [0; 3]);
    write_box16_child(&mut data, box_at, 3, INVALID, [0; 3], [0; 3]);

    put_tri(&mut data, tri_at, &[[0.0, 0.0, 3.0], [2.0, 0.0, 3.0], [0.0, 2.0, 3.0]]);
    put_words(&mut data, tri_at + 48, &[1, 42, INVALID, 0x89]);

    let bvh = Bvh { data, root };
    let hit = bvh.trace([0.25, 0.25, 0.0], [0.0, 0.0, 1.0], 10.0);
    assert_eq!(hit, Some((3.0, 42)));
}

#[test]
fn paired_triangle_leaf_traversal_and_disable_flag() {
    // Layout: 256B header + 128B box32 root + 64B paired triangle leaf
    let mut data = vec![0u8; HEADER_BYTES + 128 + 64];
    let box_at = HEADER_BYTES;
    let tri_at = HEADER_BYTES + 128;
    let root = ((box_at / 8) as u32) | BOX32;
    let tri0 = (tri_at / 8) as u32;

    put_words(&mut data, ROOT_AT, &[root]);
    put_words(&mut data, box_at, &[tri0, INVALID, INVALID, INVALID]);
    let bounds = [-2.0f32, -2.0, 1.0, 2.0, 2.0, 6.0].map(f32::to_bits);
    put_words(&mut data, box_at + 16, &bounds);

    // v0=(0,0,5), v1=(2,0,5), v2=(0,2,5), v3=(2,2,2)
    // tri0 = (v0, v1, v2) at z=5 (prim 100)
    // tri1 = (v1, v3, v2) at z=2..5 (prim 200)
    put_tri(&mut data, tri_at, &[[0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [0.0, 2.0, 5.0]]);
    put_words(&mut data, tri_at + 36, &[2.0f32.to_bits(), 2.0f32.to_bits(), 2.0f32.to_bits()]);
    // bit 7 set => tri1 present
    put_words(&mut data, tri_at + 48, &[0x81, 100, 200, 0x8989]);

    let bvh = Bvh { data: data.clone(), root };
    // Ray inside tri0 (0.25, 0.25) hits tri0 at z=5, misses tri1
    assert_eq!(bvh.trace([0.25, 0.25, 0.0], [0.0, 0.0, 1.0], 10.0), Some((5.0, 100)));
    // Ray inside tri1 (1.5, 1.5) hits tri1, misses tri0
    let hit1 = bvh.trace([1.5, 1.5, 0.0], [0.0, 0.0, 1.0], 10.0).expect("tri1 must hit");
    assert_eq!(hit1.1, 200);
    assert!(hit1.0 > 2.0 && hit1.0 < 5.0);

    // Clear bit 7 at @48: tri1 is no longer tested even though v3 and @56 are intact
    let mut single = data;
    put_words(&mut single, tri_at + 48, &[0x01, 100, 200, 0x8989]);
    let bvh_single = Bvh { data: single, root };
    assert_eq!(bvh_single.trace([1.5, 1.5, 0.0], [0.0, 0.0, 1.0], 10.0), None);
}

#[test]
fn triangle_barycentric_rotation_bits_at_60() {
    let mut data = vec![0u8; 64];
    // tri0 = ((0,0,4), (1,0,4), (0,1,4)), tri1 = ((1,0,4), (1,1,4), (0,1,4))
    put_tri(&mut data, 0, &[[0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [0.0, 1.0, 4.0]]);
    put_words(&mut data, 36, &[1.0f32.to_bits(), 1.0f32.to_bits(), 4.0f32.to_bits()]);
    // Byte 0 (tri0): I = v0 (0), J = v1 (1) -> 0x04
    // Byte 1 (tri1): I = v2 (2), J = v0 (0) -> 0x02
    put_words(&mut data, 60, &[0x0204]);

    let o0 = [0.2, 0.3, 0.0];
    let d = [0.0, 0.0, 1.0];
    let inv = [f32::INFINITY, f32::INFINITY, 1.0];
    let r0 = intersect_node(&data, 0, 10.0, o0, d, inv, false).map(f32::from_bits);
    let (t0, det0, i0, j0) = (r0[0] / r0[1], r0[1], r0[2] / r0[1], r0[3] / r0[1]);
    assert!((t0 - 4.0).abs() < 1e-6 && det0 != 0.0);
    // Original barycentrics: u=0.5, v=0.2, w=0.3 -> rotated to (u, v) = (0.5, 0.2)
    assert!((i0 - 0.5).abs() < 1e-5 && (j0 - 0.2).abs() < 1e-5, "got ({i0}, {j0})");

    let o1 = [0.8, 0.7, 0.0];
    let r1 = intersect_node(&data, 1, 10.0, o1, d, inv, false).map(f32::from_bits);
    let (t1, i1, j1) = (r1[0] / r1[1], r1[2] / r1[1], r1[3] / r1[1]);
    assert!((t1 - 4.0).abs() < 1e-6);
    // tri1 = (v1=(1,0), v3=(1,1), v2=(0,1)): at (0.8, 0.7), w=0.2, u=0.3, v=0.5
    // rotated to (w, u) = (0.2, 0.3)
    assert!((i1 - 0.2).abs() < 1e-5 && (j1 - 0.3).abs() < 1e-5, "got ({i1}, {j1})");
}

#[test]
fn unknown_node_types_and_culling_boundaries() {
    let data = vec![0u8; 64];
    let d = [0.0, 0.0, 1.0];
    let inv = [f32::INFINITY, f32::INFINITY, 1.0];

    // Instance (6) and invalid type (7) are not box/triangle nodes for intersect_node
    assert_eq!(intersect_node(&data, 6, 10.0, [0.0; 3], d, inv, false), [INVALID; 4]);
    assert_eq!(intersect_node(&data, 7, 10.0, [0.0; 3], d, inv, true), [INVALID; 4]);

    // Empty BVH / TLAS
    let empty_blas = build(&[], 0);
    assert_eq!(empty_blas.root, INVALID);
    assert_eq!(empty_blas.trace([0.0; 3], d, 100.0), None);
    let empty_tlas = build_top(&[]);
    assert_eq!(empty_tlas.root, INVALID);
    assert_eq!(empty_tlas.trace_scene(&|_| None, [0.0; 3], d, 100.0), None);

    // Non-empty BVH culling: behind ray (t < 0) and exact tmax boundary (t < limit is strict)
    let bvh = build(&[[[0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [0.0, 2.0, 5.0]]], 0);
    assert_eq!(bvh.trace([0.25, 0.25, 6.0], d, 100.0), None);
    assert_eq!(bvh.trace([0.25, 0.25, 0.0], d, 5.0), None);
    assert_eq!(bvh.trace([0.25, 0.25, 0.0], d, 5.01), Some((5.0, 0)));
    assert_eq!(bvh.trace([0.25, 0.25, 5.0], d, 0.01), Some((0.0, 0)));
}

#[test]
fn trace_scene_skips_missing_blas_and_prebuild_size_scales() {
    assert_eq!(prebuild_size(0, false), 1024);
    assert_eq!(prebuild_size(16, false), 4096);
    assert_eq!(prebuild_size(16, true), 5120);

    let tri = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
    let blas = build(&[tri], 0);
    let addr_missing = 0x10_0000u64;
    let addr_present = 0x20_0000u64;

    let inst_missing = Instance {
        transform: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
        id: 11,
        mask: 0xFF,
        contribution: 0,
        flags: 0,
        blas_address: addr_missing,
        blas_root: blas.root,
        blas_bounds: blas.bounds(),
    };
    let inst_present = Instance {
        transform: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 6.0]],
        id: 22,
        mask: 0xFF,
        contribution: 0,
        flags: 0,
        blas_address: addr_present,
        blas_root: blas.root,
        blas_bounds: blas.bounds(),
    };

    let tlas = build_top(&[inst_missing, inst_present]);
    let lookup = |addr: u64| (addr == addr_present).then_some(blas.data.as_slice());

    // Closer instance has no BLAS -> must not panic or abort traversal; returns farther hit.
    let hit = tlas.trace_scene(&lookup, [0.25, 0.25, 0.0], [0.0, 0.0, 1.0], 20.0);
    assert_eq!(hit, Some((6.0, 0, 22)));

    // If all BLASes are missing, returns None cleanly.
    assert_eq!(tlas.trace_scene(&|_| None, [0.25, 0.25, 0.0], [0.0, 0.0, 1.0], 20.0), None);
}
