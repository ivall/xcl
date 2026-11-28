//! Geometry mode (ES+GS) for the direct translator: the recompiler's
//! `Mode::Geometry` (`recomp/emit.rs`), a compute pass that runs the ES for the
//! three vertices of each of 21 triangles per group (pre-fetched by XCL), then
//! the GS, and writes the emitted vertices and strip triangles as one mesh
//! record per group into `res` (drawn by the pass-through mesh shader).
//!
//! Per lane: `emitted` / `strip` / `ntri` counters, the lowest LDS address the
//! GS phase wrote (where its output vertices are), and the strip triangles
//! (`[maxt x i32]` in a local array). The split wave64 form runs the ES in both
//! hardware waves, then the GS in the lower one alone with fresh registers.

use super::lower::{Lw, Partner};
use super::module::{Pred, Value};
use super::ops::WaveOp;
use crate::gpu::rdna2::{self, Decoded};
use crate::gpu::recomp::{ranked, GsInterface, MESH_HEADER};

pub struct GeomCtx {
    pub g: GsInterface,
    pub emitted: u32,
    pub strip: u32,
    pub ntri: u32,
    pub base: u32,
    /// local `[maxt x i32]` of strip triangles (k0 | k1 << 10 | k2 << 20)
    pub tri: Value,
    pub maxt: u32,
    pub rec: Value,
}

impl GeomCtx {
    /// Whether `pc` is in the GS code (after the ES's `s_setpc_b64` target).
    pub fn gs_phase(&self, pc: u64) -> bool {
        self.g.gs_start.is_some_and(|s| pc >= s)
    }
}

impl Lw {
    /// Geometry-mode instructions: `s_sendmsg` GS emit / cut, `s_barrier` (none in the
    /// split form's GS phase, which runs in one hardware wave). `None`: the usual lowering.
    pub fn geom_inst(&mut self, d: &Decoded, name: &str) -> Option<()> {
        let gs_phase = self.geom.as_ref()?.gs_phase(d.pc);
        match name {
            "s_barrier" if self.split && gs_phase => Some(()),
            "s_sendmsg" => {
                // MSG_GS (2): op 1 cut, 2 emit, 3 emit + cut (stream 0); the rest: nothing
                let simm = rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0);
                let (msg, op, stream) = (simm & 0xF, (simm >> 4) & 3, (simm >> 8) & 3);
                if msg == 2 && stream == 0 {
                    self.gs_emit_cut(op);
                }
                Some(())
            }
            _ => None,
        }
    }

    fn gs_emit_cut(&mut self, op: u64) {
        let (ve, vs, vn, tri, maxt, maxv) = {
            let g = self.geom.as_ref().expect("geometry");
            (g.emitted, g.strip, g.ntri, g.tri, g.maxt, g.g.record().max_vertices)
        };
        let act = self.act();
        if op & 2 != 0 {
            // strip triangle j = (j, j+1, j+2), odd j with the first two swapped
            let k = self.rd(ve);
            let strip = self.rd(vs);
            let ntri = self.rd(vn);
            let fit = self.cmpc(Pred::ULt, k, maxv);
            let emit = self.land(act, fit);
            let two = self.cmpc(Pred::UGe, strip, 2);
            let room = self.cmpc(Pred::ULt, ntri, maxt);
            let c = self.land(emit, two);
            let c = self.land(c, room);
            let k1 = self.c(1);
            let k2 = self.c(2);
            let km1 = self.sub(k, k1);
            let km2 = self.sub(k, k2);
            let k20 = self.shlc(k, 20);
            let a = self.shlc(km1, 10);
            let even = self.or(km2, a);
            let even = self.or(even, k20);
            let b = self.shlc(km2, 10);
            let odd = self.or(km1, b);
            let odd = self.or(odd, k20);
            let par = self.andc(strip, 1);
            let ev = self.cmpc(Pred::Eq, par, 0);
            let val = self.sel(ev, even, odd);
            let last = self.c(maxt - 1);
            let idx = self.umin(ntri, last);
            let zero = self.c(0);
            let p = self.s.m.gep(tri, &[zero, idx]);
            let old = self.s.m.load(p, 4);
            let nv = self.sel(c, val, old);
            self.s.m.store(p, nv, 4);
            let n1 = self.addc(ntri, 1);
            let n = self.sel(c, n1, ntri);
            self.wr(vn, n);
            let e1 = self.addc(k, 1);
            let e = self.sel(emit, e1, k);
            self.wr(ve, e);
            let s1 = self.addc(strip, 1);
            let st = self.sel(emit, s1, strip);
            self.wr(vs, st);
        }
        if op & 1 != 0 {
            let strip = self.rd(vs);
            let z = self.c(0);
            let st = self.sel(act, z, strip);
            self.wr(vs, st);
        }
    }

    /// A GS-phase LDS store at byte address `at` (active lanes only): the lowest one is
    /// where the lane's output vertices start.
    pub fn geom_note_store(&mut self, at: Value) {
        let Some(g) = self.geom.as_ref() else { return };
        if !g.gs_phase(self.pc) {
            return;
        }
        let v = g.base;
        let cur = self.rd(v);
        let m = self.umin(cur, at);
        self.wr(v, m);
    }

    /// The split form's GS phase starts like a new program in the lower hardware wave:
    /// no partner half, registers as at entry, EXEC of the wave's lanes.
    pub fn geom_gs_phase_registers(&mut self, vars: &[u32]) {
        self.partner = Partner::Missing;
        self.xn = self.c(0);
        self.exec_cache.clear();
        for &v in vars {
            let x = {
                let Lw { s, init, .. } = self;
                init.load(s, v)
            };
            self.wr(v, x);
        }
        let t = self.b(true);
        let (lo, hi) = self.ballot(t);
        self.wr(126, lo);
        self.wr(127, hi);
        let z = self.c(0);
        self.wr(super::lower::VAR_SCC, z);
        let g = self.geom.as_ref().expect("geometry");
        let (ve, vs, vn, vb) = (g.emitted, g.strip, g.ntri, g.base);
        for v in [ve, vs, vn] {
            self.wr(v, z);
        }
        let m = self.c(u32::MAX);
        self.wr(vb, m);
    }

    /// The record: vertex / triangle counts (lane 0), the emitted vertices from LDS and
    /// the strip triangles, each lane after the lanes before it (wave prefix sums).
    pub fn geom_epilogue(&mut self) {
        let (g, ve, vn, vbase, tri, rec) = {
            let x = self.geom.as_ref().expect("geometry");
            (x.g.clone(), x.emitted, x.ntri, x.base, x.tri, x.rec)
        };
        let res = self.res.expect("res");
        let r = g.record();
        let (vd, vbytes) = (r.vert_dwords() as u32, g.vertex_bytes());
        let (maxv, maxp) = (r.max_vertices, r.max_primitives);
        let z = self.c(0);
        let gs_base = self.rd(vbase);
        let has = self.cmpc(Pred::Ne, gs_base, u32::MAX);
        let emitted = self.rd(ve);
        let emitted = self.sel(has, emitted, z);
        let ntri = self.rd(vn);
        let ntri = self.sel(has, ntri, z);
        let vb = self.s.wave_prefix_op(emitted, WaveOp::Sum, true);
        let tv = self.s.wave_active_op(emitted, WaveOp::Sum, true);
        // vertices: k < emitted (and a GS-phase LDS store happened), vb + k < maxv
        let kmax = self.c(maxv);
        let left = self.sub(kmax, vb);
        let room = self.cmp(Pred::UGt, kmax, vb);
        let left = self.sel(room, left, z);
        let v_written = self.umin(emitted, left);
        // keep only triangles whose 3 local indices fit in v_written (tri[j] >> 20 is monotonic)
        let fits_all = self.cmp(Pred::Eq, v_written, emitted);
        let partial = self.cmp(Pred::ULt, v_written, emitted);
        let has_tri = self.cmpc(Pred::UGe, v_written, 3);
        let straddle = self.land(partial, has_tri);
        let vn_valid = self.tmp_var();
        let init_ntri = self.sel(fits_all, ntri, z);
        self.wr(vn_valid, init_ntri);
        let scan = self.sel(straddle, ntri, z);
        self.counted_loop(scan, &mut |lw, j| {
            let zero = lw.c(0);
            let p = lw.s.m.gep(tri, &[zero, j]);
            let t = lw.s.m.load(p, 4);
            let k2 = lw.lshrc(t, 20);
            let ok = lw.cmp(Pred::ULt, k2, v_written);
            let cur = lw.rd(vn_valid);
            let inc = lw.addc(cur, 1);
            let nxt = lw.sel(ok, inc, cur);
            lw.wr(vn_valid, nxt);
        });
        let valid_ntri = self.rd(vn_valid);
        let tb = self.s.wave_prefix_op(valid_ntri, WaveOp::Sum, true);
        let tt = self.s.wave_active_op(valid_ntri, WaveOp::Sum, true);
        let lane = self.lane;
        let first = self.cmpc(Pred::Eq, lane, 0);
        let _ = self.guarded(first, &mut |lw| {
            let a = lw.c(maxv);
            let nv = lw.umin(tv, a);
            let b = lw.c(maxp);
            let np = lw.umin(tt, b);
            let o0 = lw.shlc(rec, 2);
            lw.s.raw_store(res, o0, &[nv, np]);
            Ok(())
        });
        let pos_reg = g.outputs.iter().find(|e| e.system_value == 1).map(|e| e.register).unwrap_or(0);
        let regs: Vec<u32> = std::iter::once(pos_reg).chain(ranked(&g.outputs, true).iter().map(|e| e.register)).collect();
        let lds = self.lds.expect("lds");
        self.counted_loop(v_written, &mut |lw, k| {
            let vi = lw.add(vb, k);
            let kb = lw.c(vbytes);
            let so = lw.mul(k, kb);
            let src = lw.add(gs_base, so);
            let kd = lw.c(vd);
            let dv = lw.mul(vi, kd);
            let dst = lw.addc(rec, MESH_HEADER as u32);
            let dst = lw.add(dst, dv);
            for (slot, &r) in regs.iter().enumerate() {
                let mut vals = [z; 4];
                for c in 0..4u32 {
                    let a = lw.addc(src, 16 * r + 4 * c);
                    let i = lw.lshrc(a, 2);
                    let i = lw.andc(i, 8191);
                    let i = if lw.split { let top = lw.c(lw.lds_top); lw.umin(i, top) } else { i };
                    vals[c as usize] = lw.s.gs_load(lds, i);
                }
                let o = lw.addc(dst, 4 * slot as u32);
                let o = lw.shlc(o, 2);
                lw.s.raw_store(res, o, &vals);
            }
        });
        // triangles: j < valid_ntri, tb + j < maxp
        let pmax = self.c(maxp);
        let left = self.sub(pmax, tb);
        let room = self.cmp(Pred::UGt, pmax, tb);
        let left = self.sel(room, left, z);
        let n = self.umin(valid_ntri, left);
        let prim_base = (MESH_HEADER + maxv as usize * vd as usize) as u32;
        self.counted_loop(n, &mut |lw, j| {
            let zero = lw.c(0);
            let p = lw.s.m.gep(tri, &[zero, j]);
            let t = lw.s.m.load(p, 4);
            let a = lw.andc(t, 0x3FF);
            let a = lw.add(a, vb);
            let b = lw.lshrc(t, 10);
            let b = lw.andc(b, 0x3FF);
            let b = lw.add(b, vb);
            let b = lw.shlc(b, 10);
            let c = lw.lshrc(t, 20);
            let c = lw.andc(c, 0x3FF);
            let c = lw.add(c, vb);
            let c = lw.shlc(c, 20);
            let v = lw.or(a, b);
            let v = lw.or(v, c);
            let o = lw.addc(rec, prim_base);
            let o = lw.add(o, tb);
            let o = lw.add(o, j);
            let o = lw.shlc(o, 2);
            lw.s.raw_store(res, o, &[v]);
        });
    }

    /// `for (k = 0; k < n; k++) body(k)` with a real loop (the counter in an SSA variable).
    fn counted_loop(&mut self, n: Value, body: &mut dyn FnMut(&mut Lw, Value)) {
        let kv = self.tmp_var();
        let z = self.c(0);
        self.wr(kv, z);
        let head = self.block(None);
        let lp = self.block(Some(1));
        let out = self.block(Some(1));
        self.br(head);
        self.s.m.set_block(head);
        let k = self.rd(kv);
        let more = self.cmp(Pred::ULt, k, n);
        self.cond_br(more, lp, out);
        self.s.m.set_block(lp);
        body(self, k);
        let k1 = self.addc(k, 1);
        self.wr(kv, k1);
        self.br(head);
        self.seal(head);
        self.s.m.set_block(out);
    }
}

#[cfg(test)]
mod tests {
    use super::super::kernel::{translate_geometry, KernelOptions};
    use crate::gpu::dxbc::SigElement;
    use crate::gpu::rdna2::encode;
    use crate::gpu::recomp::{dxc, mesh_passthrough, runner, GsInterface, IO_GS_PRIMS, MESH_HEADER};

    fn enc(n: &str, e: &str, f: &[(&str, u64)], lit: Option<u32>) -> Vec<u8> { encode(n, e, f, lit).unwrap_or_else(|x| panic!("{n}: {x}")) }
    fn gs_iface(max_vertex_count: u32, gs_start: u64) -> GsInterface {
        let pos = SigElement { name: "SV_Position".into(), index: 0, system_value: 1, component_type: 3, register: 0, mask: 0xF, rw_mask: 0xF, stream: 0 };
        GsInterface { es_inputs: vec![], es_outputs: vec![pos.clone()], outputs: vec![pos], max_vertex_count, gs_start: Some(gs_start) }
    }

    #[test]
    fn split_epilogue_clamps_lds_to_lds_top_and_validates() {
        let code = [enc("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x22)], None), enc("S_ENDPGM", "ENC_SOPP", &[], None)].concat();
        let g = gs_iface(3, 0);
        for split in [false, true] {
            let k = translate_geometry(&code, KernelOptions { wave64: true, split, mem_windows: 1, sampler_base: 0 }, &g).unwrap();
            if split && dxc::location().is_ok() {
                let text = crate::gpu::dxil::validate::disassemble(&k.dxil).unwrap();
                assert!(text.lines().any(|l| l.contains("@dx.op.binary.i32(i32 40,") && l.contains(", i32 8184)")), "missing lds_top clamp:\n{text}");
                assert!(text.contains("icmp ult i32 ") && text.contains(", 63"), "missing k < maxv guard:\n{text}");
            }
        }
    }

    #[test]
    fn gpu_epilogue_filters_oob_triangles_and_no_store_emits() {
        if runner::adapter().is_err() || runner::wave64_programs_supported() != Ok(true) || dxc::location().is_err() || runner::mesh_shaders_supported() != Ok(true) {
            return runner::skip_test("no GPU / DXC / mesh shaders");
        }
        const V: u64 = 256;
        let ps = dxc::compile("float4 main(float4 p : SV_Position) : SV_Target0 { return 1; }", "main", "ps_6_6", &[]).unwrap();
        let es = enc("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None);
        let gs_pro = [
            enc("S_BFE_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x80008)),
            enc("S_BFM_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 0), ("SSRC1", 128)], None),
            enc("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None),
            enc("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None),
        ].concat();
        // (a) 15 lanes x 20 verts = 300 > 256: lane 12 straddles (16 verts -> 14 valid tris), lanes 13..14 -> 0
        let mut a = [es.clone(), gs_pro.clone(), enc("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 50), ("SRC0", V + 20), ("SRC1", 255), ("SRC2", 2)], Some(320)), enc("S_MOV_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 128)], None)].concat();
        let top = a.len();
        a.extend([enc("DS_WRITE_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 20)], None), enc("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 50), ("SRC0", 144), ("VSRC1", 50)], None), enc("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x22)], None), enc("S_ADD_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 129)], None), enc("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 148)], None)].concat());
        let back = ((top as i64 - a.len() as i64 - 4) / 4) as i16 as u16 as u64;
        a.extend([enc("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", back)], None), enc("S_ENDPGM", "ENC_SOPP", &[], None)].concat());
        // (b) 2 lanes: lane 0 emits 3x with no LDS store; lane 1 stores 3 verts (0x11110000..2) and emits 3x
        let mut b = [es, gs_pro, enc("V_CMP_NE_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 20)], None), enc("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 106)], None), enc("V_MOV_B32", "ENC_VOP1", &[("VDST", 50), ("SRC0", 2)], None)].concat();
        for k in 0..3u32 { b.extend([enc("V_MOV_B32", "ENC_VOP1", &[("VDST", 51), ("SRC0", 255)], Some(0x1111_0000 + k)), enc("DS_WRITE_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 51), ("OFFSET0", (k * 16) as u64)], None)].concat()); }
        b.extend(enc("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 8)], None));
        for _ in 0..3 { b.extend(enc("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x22)], None)); }
        b.extend(enc("S_ENDPGM", "ENC_SOPP", &[], None));
        let mut forms = vec![true];
        if runner::wave64_supported() == Ok(true) { forms.push(false); }
        let run = |code: &[u8], mvc: u32, prims: u32, split: bool| {
            let g = gs_iface(mvc, 4);
            let r = g.record();
            let (rd, ms) = (r.record_dwords(), dxc::compile(&mesh_passthrough(&r), "main", "ms_6_6", &[]).unwrap());
            let dxil = translate_geometry(code, KernelOptions { wave64: true, split, mem_windows: 1, sampler_base: 0 }, &g).unwrap().dxil;
            let mut io = vec![0u32; 128];
            io[IO_GS_PRIMS] = prims;
            (r, runner::mesh_draw(&dxil, &ms, &ps, 1, rd, &io, &[0u32; 16], 16, [0.0; 4]).unwrap().1)
        };
        for split in forms {
            let (ra, got_a) = run(&a, 20, 15, split);
            assert_eq!((got_a[0], got_a[1]), (256, 230), "split={split} overflow header");
            let pb = MESH_HEADER + ra.max_vertices as usize * ra.vert_dwords();
            for j in 0..230usize { let t = got_a[pb + j]; assert!((t & 0x3FF) < 256 && ((t >> 10) & 0x3FF) < 256 && ((t >> 20) & 0x3FF) < 256, "split={split} tri {j} OOB: {t:#x}"); }
            assert_eq!(got_a[pb + 229], 254 | (253 << 10) | (255 << 20), "split={split} straddle last tri");
            let (rb, got_b) = run(&b, 3, 2, split);
            assert_eq!((got_b[0], got_b[1]), (3, 1), "split={split} no-store emit header");
            for k in 0..3usize { assert_eq!(got_b[MESH_HEADER + k * rb.vert_dwords()], 0x1111_0000 + k as u32, "split={split} vert {k}"); }
            assert_eq!(got_b[MESH_HEADER + rb.max_vertices as usize * rb.vert_dwords()], 1 << 10 | 2 << 20, "split={split} valid tri");
        }
    }
}
