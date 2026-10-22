"""Prototype: flat unit-stream MXFP4 GEMV (fp8 MFMA) on gfx942.

Units = (group of 16 rows, superstep of 128 K) in group-major order, U = G * KS.
Wave gw (global) takes units [gw U / NW, (gw + 1) U / NW): a contiguous byte
stream. x is quantized by every CTA into LDS (e4m3fnuz, one pow2 scale per
(token, superstep)). A wave flushes its accumulator at each group end and at its
range end into partial slot [g][seg], seg = wave - first wave of g. The host sums
the slots (the megakernel's consumer stage does this on device).
Usage: python tests/bench_gemv2.py N K M [P]
"""
import os
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl  # noqa: E402
from flydsl.expr import math as fmath  # noqa: E402
from flydsl.expr.typing import Int64, T  # noqa: E402

from flydsl.compiler.ast_rewriter import ASTRewriter  # noqa: E402

from mk import fp4  # noqa: E402
from mk.vendor import buffer_ops as bo  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5120
M = int(sys.argv[3]) if len(sys.argv) > 3 else 1
P = int(sys.argv[4]) if len(sys.argv) > 4 else 4   # units per pipeline block
G, KS = N // 16, K // 128
U = G * KS
CUS = torch.cuda.get_device_properties(0).multi_processor_count
WAVES = int(os.environ.get("WAVES", "8"))
THREADS = 64 * WAVES
NW = CUS * WAVES
XROW = K + 16           # LDS bytes per x row (padded)
SEGMAX = (KS + U // NW - 1) // max(U // NW, 1) + 2
H = fp4.device_helpers()
MODE = int(os.environ.get("MODE", "0"))  # 0 full, 1 no decode, 2 loads only, 3 no LDS A


def bf_lo(w):
    return (fx.Int32(w) << 16).bitcast(fx.Float32)


def bf_hi(w):
    return (fx.Int32(w) & fx.Int32(-65536)).bitcast(fx.Float32)


def gather(v, src_lane):
    return fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(src_lane) * 4, fx.Int32(v).ir_value()))


def gather_f(v, src_lane):
    return gather(v.bitcast(fx.Int32), src_lane).bitcast(fx.Float32)


def pow2_scale(amax):
    """(qinv, scale) powers of two with amax * qinv in [120, 240]."""
    bits = amax.bitcast(fx.Int32)
    e = (bits >> 23) & 0xFF
    e = e + ((bits & 0x7FFFFF) > 0x700000).select(fx.Int32(1), fx.Int32(0))
    return ((261 - e) << 23).bitcast(fx.Float32), ((e - 7) << 23).bitcast(fx.Float32)


def pow2_scale_ref(amax):
    m, e = torch.frexp(amax)
    e = e - 1 + (m * 2 > 1.875).int()
    return torch.exp2(7.0 - e), torch.exp2(e - 7.0)


@fx.struct
class Lds:
    xq: fx.Array[fx.Int32, M * XROW // 4, 16]
    xs: fx.Array[fx.Float32, 16 * KS, 16]


@flyc.kernel(name=f"gemvf_n{N}_k{K}_m{M}_p{P}", known_block_size=[THREADS, 1, 1])
def kgemv(xp: Int64, wt: Int64, ws: Int64, partp: Int64):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    lane = tid % 64
    wave = tid // 64
    lds = fx.SharedAllocator().allocate(Lds).peek()
    xq, xsl = lds.xq.ptr, lds.xs.ptr
    rx = bo.create_buffer_resource_from_addr(xp)
    # ---- x -> LDS: thread = 8 values; 16 threads = one (token, superstep) block
    nch = M * K // 8
    for i in range_constexpr((nch + THREADS - 1) // THREADS):
        c = fx.min(tid + THREADS * i, nch - 1)
        v = fx.Vector(bo.buffer_load(rx, c * 4, vec_width=4, dtype=T.i32))
        f = []
        for e in range_constexpr(4):
            f += [bf_lo(v[e]), bf_hi(v[e])]
        am = fx.Float32(0.0)
        for e in range_constexpr(8):
            am = fx.max(am, fx.Float32(fmath.absf(f[e])))
        for off in (1, 2, 4, 8):
            am = fx.max(am, gather_f(am, lane ^ off))
        qi, sc = pow2_scale(fx.max(am, fx.Float32(1e-30)))
        lo = H["fp8_pack4"](f[0] * qi, f[1] * qi, f[2] * qi, f[3] * qi)
        hi = H["fp8_pack4"](f[4] * qi, f[5] * qi, f[6] * qi, f[7] * qi)
        t = c // (K // 8)
        kc = c % (K // 8)
        if tid + THREADS * i < nch:
            fx.ptr_store(fx.Vector.from_elements([lo, hi], fx.Int32), xq + (t * XROW + kc * 8) // 4)
            if kc % 16 == 0:
                fx.ptr_store(sc, xsl + (t * KS + kc // 16))
    # tokens >= M: scale 0 (their A rows read row M-1, killed by the zero scale)
    for i in range_constexpr((16 * KS + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if (j < 16 * KS) & (j >= M * KS):
            fx.ptr_store(fx.Float32(0.0), xsl + j)
    gpu.barrier()
    rw = bo.create_buffer_resource_from_addr(wt)
    rs = bo.create_buffer_resource_from_addr(ws)
    rp = bo.create_buffer_resource_from_addr(partp)
    gl = lane // 16
    col = lane % 16
    arow = fx.min(col, M - 1) * XROW + 8 * gl   # A operand: token = lane % 16
    gw = bid * WAVES + wave
    u0 = (gw * U) // NW
    u1 = ((gw + 1) * U) // NW

    def load(u):
        uu = fx.min(u, U - 1)
        tile = fx.Vector(bo.buffer_load(rw, (uu * 64 + lane) * 4, vec_width=4, dtype=T.i32,
                                        cache_modifier=2))
        sw = fx.Int32(bo.buffer_load(rs, uu * 16 + col, vec_width=1, dtype=T.i32))
        return tile, sw

    def compute(u, tile, sw):
        st = u % KS
        tacc = fx.Vector.filled(4, 0.0, fx.Float32)
        if const_expr(MODE == 2):
            v = (tile[0] ^ tile[1] ^ tile[2] ^ tile[3] ^ sw).bitcast(fx.Float32)
            return fx.Vector.from_elements([v, v, v, v], fx.Float32), [fx.Float32(1.0)] * 4
        for d in range_constexpr(4):
            if const_expr(MODE == 3):
                a = fx.Vector.from_elements([tile[d], tile[(d + 1) % 4]], fx.Int32).bitcast(fx.Int64)[0]
            else:
                a = fx.Vector(fx.ptr_load(xq + (arow + st * 128 + 32 * d) // 4,
                                          result_type=fx.Vector.make_type(2, fx.Int32))).bitcast(fx.Int64)[0]
            if const_expr(MODE == 1):
                bw = fx.Vector.from_elements([tile[d], sw], fx.Int32).bitcast(fx.Int64)[0]
            else:
                bw = H["dec_dword"](tile[d])
            t = H["mfma"](a, bw, fx.Vector.filled(4, 0.0, fx.Float32))
            sc_ = H["e8m0_f32"]((sw >> (8 * d)) & 0xFF)
            tacc = fx.Vector.from_elements([fmath.fma(t[r], sc_, tacc[r]) for r in range(4)], fx.Float32)
        if const_expr(MODE == 3):
            xs = [fx.Float32(1.0)] * 4
        else:
            xs = [fx.ptr_load(xsl + ((4 * gl + r) * KS + st)) for r in range(4)]
        return tacc, xs

    zero4 = fx.Vector.filled(4, 0.0, fx.Float32)

    def fma4(a, b, c):
        return fx.Vector.from_elements([fmath.fma(a[r], b[r], c[r]) for r in range(4)], fx.Float32)

    def flush(g, acc):
        wf = ((g * KS + 1) * NW - 1) // U
        if gl * 4 < M:
            bo.buffer_store(acc, rp, ((g * SEGMAX + gw - wf) * 64 + lane) * 4)

    # Block of P units: at most one group boundary inside (P <= KS). Units of the
    # current group go to accA, units after the boundary to accB.
    nblk = (u1 - u0 + P - 1) // P

    @ASTRewriter.transform
    def block(ub, ops, acc_a, acc_b):
        bnd = (ub // KS + 1) * KS          # first unit of the next group
        for p in range_constexpr(P):
            u = ub + p
            tacc, xs = compute(u, ops[p][0], ops[p][1])
            live = u < u1
            in_a = live & (u < bnd)
            in_b = live & (u >= bnd)
            xa_ = fx.Vector.from_elements([in_a.select(xs[r], fx.Float32(0.0)) for r in range(4)], fx.Float32)
            xb_ = fx.Vector.from_elements([in_b.select(xs[r], fx.Float32(0.0)) for r in range(4)], fx.Float32)
            acc_a = fma4(tacc, xa_, acc_a)
            acc_b = fma4(tacc, xb_, acc_b)
        # group of accA ended in (or at the end of) this block, or the range ended
        if (bnd <= ub + P) | (u1 <= ub + P):
            flush(ub // KS, acc_a)
            acc_a = acc_b
            acc_b = zero4
        return acc_a, acc_b

    # ping-pong: buffer A holds block 2i, buffer B block 2i + 1; no copies at the backedge
    buf_a = [load(u0 + p) for p in range(P)]
    acc_a = zero4
    acc_b = zero4
    for i in range(0, (nblk + 1) // 2):
        ub = u0 + 2 * i * P
        buf_b = [load(ub + P + p) for p in range(P)]
        acc_a, acc_b = block(ub, buf_a, acc_a, acc_b)
        buf_a = [load(ub + 2 * P + p) for p in range(P)]
        if 2 * i + 1 < nblk:
            acc_a, acc_b = block(ub + P, buf_b, acc_a, acc_b)
    # a group started in the last block and the range ended there: flush it
    last = u1 - 1
    if ((last // KS) != ((u0 + (nblk - 1) * P) // KS)) & (nblk > 0):
        flush(last // KS, acc_a)


@flyc.jit
def launch(xp: Int64, wt: Int64, ws: Int64, partp: Int64, stream: fx.Stream = fx.Stream(None)):
    kgemv(xp, wt, ws, partp).launch(grid=(CUS,), block=(THREADS,), stream=stream)


def timed(fn, it=50):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            for _ in range(it):
                fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / it


def main():
    torch.manual_seed(int(os.environ.get("SEED", "0")))
    w_u8 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    s_u8 = torch.randint(118, 128, (N, K // 32), dtype=torch.uint8, device="cuda")
    tiles, scales = fp4.repack(w_u8, s_u8)
    from mk.reference import dequant_mxfp4
    Wd = dequant_mxfp4(w_u8, s_u8)
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    xf = x.float().reshape(M, KS, 128)
    qi, sc = pow2_scale_ref(xf.abs().amax(-1, keepdim=True).clamp_min(1e-30))
    xq = (xf * qi).to(torch.float8_e4m3fnuz).float() * sc
    ref = xq.reshape(M, K) @ Wd.float().t()
    part = torch.zeros(G, SEGMAX, 64, 4, device="cuda")
    run = lambda: launch(x.data_ptr(), tiles.data_ptr(), scales.data_ptr(), part.data_ptr(),
                         stream=torch.cuda.current_stream())
    run()
    torch.cuda.synchronize()
    # partial slot: lane (gl, col) reg r = token 4 gl + r, column 16 g + col
    p = part.sum(1).reshape(G, 4, 16, 4).permute(1, 3, 0, 2).reshape(16, N)[:M]
    err = ((p - ref).abs().max() / ref.abs().max()).item()
    print(f"N={N} K={K} M={M} P={P} U/NW={U / NW:.2f} rel max err {err:.3e}")
    assert MODE != 0 or err < 1e-5, "mismatch"
    us = timed(run)
    nbytes = N * K // 2 + N * K // 32
    print(f"time {us:.1f} us  weights {nbytes/1e6:.1f} MB  {nbytes/us/1e6:.2f} TB/s")


if __name__ == "__main__":
    main()
