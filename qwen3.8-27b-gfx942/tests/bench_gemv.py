"""Microbenchmark + correctness test: MXFP4 (mk4 layout) GEMV with bf16 MFMA on gfx942.

out[m, n] = sum_k x[m, k] * W[n, k], x bf16 [M, K], W MXFP4, out fp32.
Grid = CU count. A CTA takes groups bid, bid + CUS, ...; wave w covers K slice w
(KS / 8 supersteps), so its activation operands stay in registers. The 8 slices
are summed through LDS in wave order. The next group's tiles are loaded before
the current group is computed.
Usage: python tests/bench_gemv.py [N K M]
"""
import os
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr import const_expr, range_constexpr, gpu  # noqa: E402
from flydsl.expr.typing import Int64, T  # noqa: E402
from flydsl.expr import math as fmath  # noqa: E402
from flydsl.expr import rocdl  # noqa: E402


def bf_lo(w):
    return (fx.Int32(w) << 16).bitcast(fx.Float32)


def bf_hi(w):
    return (fx.Int32(w) & fx.Int32(-65536)).bitcast(fx.Float32)


def pow2_scale(amax):
    """(qinv, scale) powers of two with amax * qinv in [120, 240] (e4m3fnuz max 240)."""
    bits = amax.bitcast(fx.Int32)
    e = (bits >> 23) & 0xFF
    e = e + ((bits & 0x7FFFFF) > 0x700000).select(fx.Int32(1), fx.Int32(0))
    return ((134 + 127 - e) << 23).bitcast(fx.Float32), ((e - 7) << 23).bitcast(fx.Float32)


def pow2_scale_ref(amax):
    m, e = torch.frexp(amax)          # amax = m 2^e, m in [0.5, 1)
    e = e - 1 + (m * 2 > 1.875).int()  # floor(log2) + 1 if mantissa > 1.875
    return torch.exp2(7.0 - e), torch.exp2(e - 7.0)


def lane_gather_f(v, src_lane):
    i = fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(src_lane) * 4, v.bitcast(fx.Int32).ir_value()))
    return i.bitcast(fx.Float32)

from mk import fp4  # noqa: E402
from mk.vendor import buffer_ops as bo  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5120
M = int(sys.argv[3]) if len(sys.argv) > 3 else 1
G, KS = N // 16, K // 128
CUS = torch.cuda.get_device_properties(0).multi_processor_count
THREADS, WAVES = 512, 8
assert KS % WAVES == 0
KSW = KS // WAVES
MT = (M + 3) // 4  # lane groups (gl) holding live tokens
H = fp4.device_helpers()
MODE = int(os.environ.get("MODE", "0"))  # 0 full, 1 no decode, 2 loads only


@fx.struct
class Lds:
    red: fx.Array[fx.Float32, 2 * WAVES * MT * 64 * 4, 16]


FP8_MAX = 240.0


@flyc.kernel(name=f"gemv8_n{N}_k{K}_m{M}_mode{MODE}", known_block_size=[THREADS, 1, 1])
def kgemv(xp: Int64, wt: Int64, ws: Int64, outp: Int64, dbg: Int64):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    lane = tid % 64
    wave = tid // 64
    red = fx.SharedAllocator().allocate(Lds).peek().red.ptr
    rx = bo.create_buffer_resource_from_addr(xp)
    rw = bo.create_buffer_resource_from_addr(wt)
    rs = bo.create_buffer_resource_from_addr(ws)
    ro = bo.create_buffer_resource_from_addr(outp)
    gl = lane // 16
    tok = lane % 16
    tok_ok = tok < M
    st0 = wave * KSW
    # this wave's K slice of x: lane (tok, gl), MFMA step (j, d): k = 128 (st0 + j) + 32 d + 8 gl
    xrow = fx.min(tok, M - 1) * (K // 2)
    raw = []
    amax = fx.Float32(0.0)
    for j in range_constexpr(KSW):
        for d in range_constexpr(4):
            k = (st0 + j) * 128 + 32 * d + 8 * gl
            v = fx.Vector(bo.buffer_load(rx, xrow + k // 2, vec_width=4, dtype=T.i32))
            raw.append(v)
            for e in range_constexpr(4):
                for f in (bf_lo(v[e]), bf_hi(v[e])):
                    amax = fx.max(amax, fx.Float32(fmath.absf(f)))
    amax = fx.max(amax, lane_gather_f(amax, lane ^ 16))
    amax = fx.max(amax, lane_gather_f(amax, lane ^ 32))
    amax = tok_ok.select(fx.max(amax, fx.Float32(1e-30)), fx.Float32(1.0))
    qinv, xscale = pow2_scale(amax)
    xa = []
    for i in range_constexpr(4 * KSW):
        v = raw[i]
        f = []
        for e in range_constexpr(4):
            f += [bf_lo(v[e]) * qinv, bf_hi(v[e]) * qinv]
        lo = H["fp8_pack4"](f[0], f[1], f[2], f[3])
        hi = H["fp8_pack4"](f[4], f[5], f[6], f[7])
        lo = tok_ok.select(lo, fx.Int32(0))
        hi = tok_ok.select(hi, fx.Int32(0))
        xa.append(fx.Vector.from_elements([lo, hi], fx.Int32).bitcast(fx.Int64)[0])
    rdb = bo.create_buffer_resource_from_addr(dbg)
    if bid == 0:
        for i in range_constexpr(4 * KSW):
            bo.buffer_store(fx.Vector.from_elements([xa[i]], fx.Int64).bitcast(fx.Int32), rdb, ((wave * 4 * KSW + i) * 64 + lane) * 2)
        bo.buffer_store(xscale, rdb, WAVES * 4 * KSW * 64 * 2 + wave * 64 + lane)
    # x scale of the tokens this lane's accumulators hold (4 gl + r)
    xs = [lane_gather_f(xscale, 4 * gl + r)
          for r in range(4)]

    def loads(g):
        gg = fx.min(g, G - 1)
        tl = [fx.Vector(bo.buffer_load(rw, ((gg * KS + st0 + j) * 64 + lane) * 4,
                                       vec_width=4, dtype=T.i32, cache_modifier=2))
              for j in range(KSW)]
        sc = [fx.Int32(bo.buffer_load(rs, (gg * KS + st0 + j) * 16 + tok, vec_width=1,
                                      dtype=T.i32)) for j in range(KSW)]
        return tl, sc

    cur = loads(bid)
    it = fx.Int32(0)
    for g in range(bid, G, CUS):
        nxt = loads(g + CUS)
        tl, sc = cur
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for j in range_constexpr(KSW):
            for d in range_constexpr(4):
                if const_expr(MODE == 2):
                    acc = acc + fx.Vector.filled(4, 0.0, fx.Float32) + (tl[j][d] ^ sc[j]).bitcast(fx.Float32)
                else:
                    if const_expr(MODE == 1):
                        b = fx.Vector.from_elements([tl[j][d], sc[j]], fx.Int32).bitcast(fx.Int64)[0]
                    else:
                        b = H["dec_dword"](tl[j][d])
                    t = H["mfma"](xa[4 * j + d], b, fx.Vector.filled(4, 0.0, fx.Float32))
                    acc = acc + t * H["e8m0_f32"]((sc[j] >> (8 * d)) & 0xFF)
        acc = fx.Vector.from_elements([acc[r] * xs[r] for r in range(4)], fx.Float32)
        buf = (it % 2) * (WAVES * MT * 64)
        if gl < MT:
            fx.ptr_store(acc, red + (buf + (wave * MT + gl) * 64 + tok) * 4)
        gpu.barrier()
        # thread (m, col): sum over the waves in order
        if tid < 16 * M:
            m = tid // 16
            col = tid % 16
            tot = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                tot = tot + fx.ptr_load(red + (buf + (w * MT + m // 4) * 64 + col) * 4 + m % 4)
            bo.buffer_store(tot, ro, m * N + g * 16 + col)
        cur = nxt
        it = it + 1


@flyc.jit
def launch(xp: Int64, wt: Int64, ws: Int64, outp: Int64, dbg: Int64, stream: fx.Stream = fx.Stream(None)):
    kgemv(xp, wt, ws, outp, dbg).launch(grid=(CUS,), block=(THREADS,), stream=stream)


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
    xt = os.environ.get("XTEST")
    if xt == "negzero":
        x[:, 100] = -0.0
        x[:, 3000] = -1e-30
    if xt == "ones":
        x = torch.ones(M, K, device="cuda", dtype=torch.bfloat16)
    elif xt and xt.startswith("slice"):
        w0 = int(xt[5:])
        keep = torch.zeros(K, dtype=torch.bool, device="cuda")
        keep[w0 * K // WAVES:(w0 + 1) * K // WAVES] = True
        x[:, ~keep] = 0
    elif xt == "pow2":
        x = torch.exp2(torch.randint(-3, 3, (M, K), device="cuda").float()).to(torch.bfloat16)
    out = torch.zeros(M, N, device="cuda")
    # emulate the kernel's activation quantization: per (token, wave K slice) scale
    xf = x.float().reshape(M, WAVES, K // WAVES)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    qi, sc_ = pow2_scale_ref(amax)
    xq = (xf * qi).to(torch.float8_e4m3fnuz).float() * sc_
    ref = xq.reshape(M, K) @ Wd.float().t()
    ref_exact = x.float() @ Wd.float().t()
    print("fp8 activation error vs exact: %.3e" % ((ref - ref_exact).abs().max() / ref_exact.abs().max()))
    dbg = torch.zeros(WAVES * 4 * KSW * 64 * 2 + WAVES * 64, dtype=torch.int32, device="cuda")
    run = lambda: launch(x.data_ptr(), tiles.data_ptr(), scales.data_ptr(), out.data_ptr(), dbg.data_ptr(),
                         stream=torch.cuda.current_stream())
    run()
    torch.cuda.synchronize()
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    print(f"N={N} K={K} M={M} rel max err {err:.3e}")
    if os.environ.get("DBG"):
        xa = dbg[: WAVES * 4 * KSW * 64 * 2].view(torch.uint8).reshape(WAVES, KSW, 4, 4, 16, 8)  # w j d gl tok b
        xa = xa.permute(4, 0, 1, 2, 3, 5).reshape(16, K)[:M]
        xsc = dbg[WAVES * 4 * KSW * 64 * 2:].view(torch.float32).reshape(WAVES, 64)[:, :M]
        kq = xa.view(torch.float8_e4m3fnuz).float()
        eq = (xf * qi).to(torch.float8_e4m3fnuz).reshape(M, K)
        diff = (xa != eq.view(torch.uint8)).nonzero()
        print("quantized mismatches", diff.shape[0])
        for t_, k_ in diff[:6].tolist():
            w_ = k_ // (K // WAVES)
            print(" k", k_, "x", x[t_, k_].item(), "kernel byte", hex(xa[t_, k_].item()), "torch", hex(eq.view(torch.uint8)[t_, k_].item()),
                  "amax", amax[t_, w_, 0].item(), "kscale", xsc[w_, t_].item() * FP8_MAX)
        d = (out - ref).abs()
        print("mean abs err", d.mean().item(), "ref mean abs", ref.abs().mean().item())
        print("vs exact", ((out - ref_exact).abs().max() / ref_exact.abs().max()).item())
        i = d.argmax().item(); print("worst", i, out.flatten()[i].item(), ref.flatten()[i].item(), ref_exact.flatten()[i].item())
        print("frac cols err>1e-4", (d > 1e-4 * ref.abs().max()).float().mean().item())
        for nm, gsz in (("row", K), ("slice", K // WAVES), ("superstep", 128), ("block32", 32)):
            xg = x.float().reshape(M, K // gsz, gsz)
            am = xg.abs().amax(-1, keepdim=True).clamp_min(1e-30)
            xq2 = (xg * (FP8_MAX / am)).to(torch.float8_e4m3fnuz).float() * (am / FP8_MAX)
            r2 = xq2.reshape(M, K) @ Wd.float().t()
            print(nm, ((out - r2).abs().max() / r2.abs().max()).item())
    assert MODE != 0 or err < 1e-5, "mismatch"
    us = timed(run)
    nbytes = N * K // 2 + N * K // 32
    print(f"time {us:.1f} us  weights {nbytes/1e6:.1f} MB  {nbytes/us/1e6:.2f} TB/s")


if __name__ == "__main__":
    main()
