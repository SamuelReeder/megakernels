"""Flat unit-stream GEMV of the megakernel.

out[m, n] = sum_k x[m, k] W[n, k] for M <= 16 rows.

Units: a unit is (group of 16 output rows, superstep of 128 K). K is cut into C
chunks of KSc supersteps; units are ordered (chunk, group, superstep) and a
"piece" p = chunk * G + group is KSc consecutive units. Global wave gw streams
units [gw U / NW, (gw + 1) U / NW): one contiguous range, so every wave has the
same number of units (+-1). x lives in LDS (filled by the stage prologue) for
the <= 2 chunks a CTA touches ("slots").

A wave keeps two accumulators (the piece it is in, and the next one) and at each
piece end stores the piece's partial to slot [p][seg] (seg = gw - first wave of p).
It then adds 1 to the group's counter (device-scope atomic); the wave that makes
the count a multiple of the group's segment total sums all segments in fixed
order (deterministic) and runs the epilogue for the group ("last arriver").

Weights "mx": mk4 tiles (mk/fp4.py), fp8 MFMA, x fp8 with a pow2 scale per
(token, superstep). Weights "bf16": pack_bf16 tiles, bf16 MFMA, x bf16, no scales.
"""
from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from . import dev as D
from . import fp4

H4 = fp4.device_helpers()


@dataclass(frozen=True)
class Gemv:
    name: str
    N: int
    K: int
    M: int
    cus: int
    waves: int = 8
    C: int = 1              # K chunks
    kind: str = "mx"        # "mx" | "bf16"
    P: int = 2              # units per pipeline block
    gpc: int = 1            # groups per counter (last-arriver granularity)

    @property
    def G(self):
        return self.N // 16

    @property
    def KS(self):
        return self.K // 128

    @property
    def KSc(self):
        return self.KS // self.C

    @property
    def Kc(self):
        return self.K // self.C

    @property
    def U(self):
        return self.C * self.G * self.KSc

    @property
    def NW(self):
        return self.cus * self.waves

    @property
    def slots(self):
        """K chunks a CTA touches: 1 when C divides the CTA count (chunk boundaries fall
        on CTA boundaries), else <= 2."""
        return 1 if self.cus % self.C == 0 else 2

    @property
    def MT(self):
        """Lane groups (gl) whose accumulators hold live tokens."""
        return (self.M + 3) // 4

    @property
    def esize(self):
        return 1 if self.kind == "mx" else 2

    @property
    def XR(self):
        """LDS bytes per x row (slots x Kc, padded 16 B)."""
        return self.slots * self.Kc * self.esize + 16

    @property
    def XSR(self):
        """LDS x-scale floats per token row."""
        return self.slots * self.KSc

    @property
    def per_wave(self):
        return self.U / self.NW

    @property
    def SEGMAX(self):
        import math
        return math.ceil(self.KSc / max(math.floor(self.per_wave), 1)) + 1

    def nseg_host(self, p):
        wf = ((p * self.KSc + 1) * self.NW - 1) // self.U
        wl = (((p + 1) * self.KSc) * self.NW - 1) // self.U
        return wl - wf + 1

    def check(self):
        assert self.N % 16 == 0 and self.K % 128 == 0 and self.KS % self.C == 0
        assert self.KSc >= self.P, "a pipeline block may cross at most one piece end"
        assert self.U * self.NW < 2 ** 31
        assert self.G % self.gpc == 0
        assert max(self.nseg_host(p) for p in range(self.C * self.G)) <= self.SEGMAX
        return self

    def part_floats(self):
        return self.C * self.G * self.SEGMAX * 256

    # ------------------------------------------------------------ device
    def cta_range(self, bid):
        return (bid * self.waves * self.U) // self.NW, ((bid + 1) * self.waves * self.U) // self.NW

    def chunk0(self, bid):
        return self.cta_range(bid)[0] // (self.G * self.KSc)

    def load(self, w, u):
        """Unit u's weight data for this lane (clamped to the last unit)."""
        uu = fx.min(u, self.U - 1)
        p = uu // self.KSc
        s = uu % self.KSc
        ti = (p % self.G) * self.KS + (p // self.G) * self.KSc + s
        if const_expr(self.kind == "mx"):
            tile = D.ld(w["rt"], (ti * 64 + w["lane"]) * 4, 4, cm=D.CM_NT)
            sw = D.ld(w["rs"], ti * 16 + w["col"])
            return tile, sw
        # bf16 (LM head, > 2 GiB): a 64-bit tile base per unit (wave-uniform)
        rt = D.rsrc(w["tbase"] + fx.Int64(ti) * 4096)
        return [D.ld(rt, (q * 64 + w["lane"]) * 4, 4, cm=D.CM_NT) for q in range(4)]

    def compute(self, w, u, ops):
        """Unit u's contribution (4 floats: tokens 4 gl + r, column col of its group).
        u is clamped like load(): units past the stream end read valid x (their result
        is discarded with a select; a multiply by 0 would keep NaN/inf)."""
        u = fx.min(u, self.U - 1)
        p = u // self.KSc
        s = u % self.KSc
        slot = p // self.G - w["ch0"]
        arow = w["arow"] + slot * (self.Kc * self.esize) + s * (128 * self.esize)
        z4 = fx.Vector.filled(4, 0.0, fx.Float32)
        if const_expr(self.kind == "mx"):
            tile, sw = ops
            tacc = z4
            for d in range_constexpr(4):
                a = fx.Vector(fx.ptr_load(w["xq"] + (arow + 32 * d) // 4,
                                          result_type=fx.Vector.make_type(2, fx.Int32))).bitcast(fx.Int64)[0]
                t = H4["mfma"](a, H4["dec_dword"](tile[d]), z4)
                tacc = D.fma4(t, fx.Vector.filled(4, H4["e8m0_f32"]((sw >> (8 * d)) & 0xFF), fx.Float32), tacc)
            xs = fx.Vector.from_elements(
                [fx.ptr_load(w["xs"] + ((4 * w["gl"] + r) * self.XSR + slot * self.KSc + s)) for r in range(4)],
                fx.Float32)
            return tacc * xs
        acc = z4
        for q in range_constexpr(4):
            for h in range_constexpr(2):
                sub = 2 * q + h
                a = fx.Vector(fx.ptr_load(w["xq"] + (arow + 32 * sub) // 4,
                                          result_type=fx.Vector.make_type(2, fx.Int32))).bitcast(fx.Int16)
                b = fx.Vector.from_elements([ops[q][2 * h], ops[q][2 * h + 1]], fx.Int32).bitcast(fx.Int16)
                acc = fx.Vector(rocdl.mfma_f32_16x16x16bf16_1k(T.vec(4, T.f32), [a, b, acc, 0, 0, 0]))
        return acc

    def setup(self, c, xq, xs, tiles, scales):
        """Per-lane constants of this GEMV. xq/xs: LDS pointers; tiles/scales: 64-bit
        global addresses of the packed weight."""
        lane, gl, col = c["lane"], c["gl"], c["col"]
        bid = c["bid"]
        w = dict(lane=lane, gl=gl, col=col, xq=xq, xs=xs, tbase=tiles)
        if const_expr(self.kind == "mx"):
            w["rt"], w["rs"] = D.rsrc(tiles), D.rsrc(scales)
        # A operand: token = lane % 16 (clamped to M - 1; its scale row is 0 past M)
        w["arow"] = fx.min(col, self.M - 1) * self.XR + 8 * gl * self.esize // 1
        if const_expr(self.kind == "bf16"):
            w["arow"] = fx.min(col, self.M - 1) * self.XR + 4 * gl * 2
        w["ch0"] = self.chunk0(bid)
        gw = bid * self.waves + c["wave"]
        w["gw"] = gw
        w["u0"] = (gw * self.U) // self.NW
        w["u1"] = ((gw + 1) * self.U) // self.NW
        return w

    def first_loads(self, w):
        return [self.load(w, w["u0"] + p) for p in range(self.P)]

    @D.traced
    def run(self, c, w, buf, part, cnt, epilogue):
        """Stream this wave's units. buf: first_loads(). part/cnt: global addresses of
        the partial slots and the group counters. epilogue(cg, [acc4] * gpc) runs in the
        last arriver of counter group cg (groups cg * gpc ...), acc4 per lane: tokens
        4 gl + r of the group's column `col`."""
        P, KSc = self.P, self.KSc
        u0, u1 = w["u0"], w["u1"]
        z4 = fx.Vector.filled(4, 0.0, fx.Float32)
        rp = D.rsrc(part)

        @D.traced
        def block(ub, ops, acc_a, acc_b):
            bnd = (ub // KSc + 1) * KSc           # first unit of the next piece
            for p in range_constexpr(P):
                u = ub + p
                t = self.compute(w, u, ops[p])
                live = u < u1
                ia = live & (u < bnd)
                ib = live & (u >= bnd)
                acc_a = fx.Vector.from_elements([ia.select(acc_a[r] + t[r], acc_a[r]) for r in range(4)], fx.Float32)
                acc_b = fx.Vector.from_elements([ib.select(acc_b[r] + t[r], acc_b[r]) for r in range(4)], fx.Float32)
            if (bnd <= ub + P) | (u1 <= ub + P):
                self.store_part(w, ub // KSc, acc_a, rp)
                acc_a = acc_b
                acc_b = z4
            return acc_a, acc_b

        D.probe(c, 0)
        nblk = (u1 - u0 + P - 1) // P
        buf_a = buf
        acc_a = z4
        acc_b = z4
        for i in range(0, (nblk + 1) // 2):
            ub = u0 + 2 * i * P
            buf_b = [self.load(w, ub + P + p) for p in range(P)]
            acc_a, acc_b = block(ub, buf_a, acc_a, acc_b)
            buf_a = [self.load(w, ub + 2 * P + p) for p in range(P)]
            if 2 * i + 1 < nblk:
                acc_a, acc_b = block(ub + P, buf_b, acc_a, acc_b)
        D.probe(c, 1)
        last = u1 - 1
        if ((last // KSc) != ((u0 + (nblk - 1) * P) // KSc)) & (nblk > 0):
            self.store_part(w, last // KSc, acc_a, rp)
        # counters after the whole stream: one drain instead of one per piece
        rocdl.s_waitcnt(vmcnt=0)
        for p in range(u0 // KSc, (u1 - 1) // KSc + 1):
            self.arrive(c, w, p, rp, cnt, epilogue)
        D.probe(c, 2)

    def nseg(self, p):
        wf = ((p * self.KSc + 1) * self.NW - 1) // self.U
        wl = (((p + 1) * self.KSc) * self.NW - 1) // self.U
        return wf, wl - wf + 1

    @D.traced
    def store_part(self, w, p, acc, rp):
        """Piece p's partial of this wave -> slot [p][gw - first wave of p] (device scope)."""
        wf, _ = self.nseg(p)
        if w["gl"] < self.MT:
            D.st(acc, rp, ((p * self.SEGMAX + w["gw"] - wf) * 64 + w["lane"]) * 4, D.CM_DEV)

    @D.traced
    def arrive(self, c, w, p, rp, cnt, epilogue):
        """Count piece p's partial (stored and drained) on its group's counter; the
        last arriver reduces and runs the epilogue."""
        lane = w["lane"]
        cg = (p % self.G) // self.gpc
        total = fx.Int32(0)
        for ch in range_constexpr(self.C):
            for j in range_constexpr(self.gpc):
                _, n = self.nseg(ch * self.G + cg * self.gpc + j)
                total = total + n
        old = fx.Int32(0)
        if lane == 0:
            old = D.atomic_add_dev(cnt + cg * 4, 1)
        old = D.uniform(old)
        if (old + 1) % total == 0:
            self.la_reduce(lane, cg, rp, epilogue, c)

    @D.traced
    def la_reduce(self, lane, cg, rp, epilogue, c=None):
        """Sum every segment of the counter group's pieces (fixed order) -> epilogue.
        A batch of groups' segment loads goes out together (dead segments masked)."""
        tots = []
        nb = 4
        for j0 in range_constexpr(0, self.gpc, nb):
            pend = []
            for j in range_constexpr(j0, min(j0 + nb, self.gpc)):
                gg = cg * self.gpc + j
                lj = []
                for ch in range_constexpr(self.C):
                    pp = ch * self.G + gg
                    _, n = self.nseg(pp)
                    for sg in range_constexpr(self.SEGMAX):
                        v = D.ld(rp, ((pp * self.SEGMAX + fx.min(n - 1, sg)) * 64 + lane) * 4, 4,
                                 T.f32, D.CM_DEV)
                        lj.append((v, n > sg))
                pend.append(lj)
            for lj in pend:
                tot = fx.Vector.filled(4, 0.0, fx.Float32)
                for v, ok in lj:
                    tot = fx.Vector.from_elements([ok.select(tot[r] + v[r], tot[r]) for r in range(4)], fx.Float32)
                tots.append(tot)
        epilogue(cg, tots)
