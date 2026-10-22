"""Stage prologues (fill the GEMV's LDS x) and last-arriver epilogues.

Activation hand-offs between stages:
  h      fp32 [M][H]       residual stream (double buffered by the caller)
  ssq    fp32 [M][H/16]    sum of squares of h per 16-column group (for RMSNorm)
  xq/xs  e4m3fnuz [M][K] + fp32 pow2 scale [M][K/128]: a quantized GEMV input
  y      fp32 [M][N]       raw GEMV output
All global hand-off traffic uses device scope (CM_DEV).
"""
from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from . import dev as D
from .dims import H

NT = 512   # threads per CTA (set by the kernel builder before tracing)


@D.traced
def zero_xs_tail(c, g, xs):
    """x-scale rows of tokens M..15 = 0: their MFMA rows contribute nothing."""
    n = (16 - g.M) * g.XSR
    for i in range_constexpr((n + NT - 1) // NT):
        j = c["tid"] + NT * i
        if j < n:
            fx.ptr_store(fx.Float32(0.0), xs + (g.M * g.XSR + j))


@D.traced
def rstd_from_ssq(c, M, ssq, red, eps):
    """red[t] = 1 / sqrt(sum(ssq[t, :]) / H + eps) for t < M (wave t, fixed order)."""
    r = D.rsrc(ssq)
    lane, wave = c["lane"], c["wave"]
    ng = H // 16
    if wave < M:
        s = fx.Float32(0.0)
        for i in range_constexpr((ng + 63) // 64):
            j = lane + 64 * i
            v = D.ldf(r, wave * ng + fx.min(j, ng - 1), cm=D.CM_DEV)
            s = s + (j < ng).select(v, fx.Float32(0.0))
        s = D.xor_sum(s, lane, (32, 16, 8, 4, 2, 1))
        if lane == 0:
            fx.ptr_store(D.rsq(s / fx.Float32(float(H)) + fx.Float32(eps)), red + wave)
    gpu.barrier()


@D.traced
def prologue_norm(c, g, xq, xs, red, h, ssq, lnw, eps):
    """x = h * rstd * (1 + w) (zero-centred RMSNorm) for the CTA's K chunk slots,
    quantized per (token, 128 K). The h loads go out before the rstd reduction."""
    M, K = g.M, g.K
    assert K == H and g.kind == "mx"
    tid = c["tid"]
    zero_xs_tail(c, g, xs)
    rh, rw = D.rsrc(h), D.rsrc(lnw)
    ch0 = g.chunk0(c["bid"])
    nch = M * g.Kc // 8
    raw = []
    for sl in range_constexpr(g.slots):
        cb = fx.min(ch0 + sl, g.C - 1) * g.Kc
        for i in range_constexpr((nch + NT - 1) // NT):
            j = fx.min(tid + NT * i, nch - 1)
            t = j // (g.Kc // 8)
            kc = j % (g.Kc // 8) * 8
            k0 = cb + kc
            raw.append((sl, i, t, kc, D.ld(rh, t * K + k0, 4, T.f32, D.CM_DEV),
                        D.ld(rh, t * K + k0 + 4, 4, T.f32, D.CM_DEV), D.ld(rw, k0 // 2, 4)))
    rstd_from_ssq(c, M, ssq, red, eps)
    for sl, i, t, kc, a, b, wv in raw:
        rs = fx.ptr_load(red + t)
        f = []
        for e in range_constexpr(4):
            f += [D.bf_lo(wv[e]), D.bf_hi(wv[e])]
        hv = [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
        x = [hv[e] * rs * (fx.Float32(1.0) + f[e]) for e in range(8)]
        store_quant8(c, g, xq, xs, x, t, sl * g.Kc + kc, tid + NT * i < nch)
    gpu.barrier()


@D.traced
def store_quant8(c, g, xq, xs, x, t, k0, live):
    """8 values of token t at column k0 (16 consecutive lanes = one 128 block) ->
    LDS fp8 + the block's pow2 scale."""
    lane = c["lane"]
    am = fx.Float32(0.0)
    for e in range_constexpr(8):
        am = fx.max(am, D.absf(x[e]))
    am = D.xor_max(am, lane, (1, 2, 4, 8))
    qi, sc = D.pow2_scale(fx.max(am, fx.Float32(1e-30)))
    lo, hi = D.quant8(x, qi)
    if live:
        fx.ptr_store(fx.Vector.from_elements([lo, hi], fx.Int32), xq + (t * g.XR + k0) // 4)
        if k0 % 128 == 0:
            fx.ptr_store(sc, xs + (t * g.XSR + k0 // 128))


@D.traced
def prologue_act(c, g, xq, xs, y):
    """LDS x <- act = silu(gate) * up, quantized, for the CTA's chunk slots. y: fp32
    gate_up output [M][2 FFN] in gu_order (per 128 block: 128 gate, then 128 up).
    All loads go out first."""
    M = g.M
    tid = c["tid"]
    zero_xs_tail(c, g, xs)
    ry = D.rsrc(y)
    ch0 = g.chunk0(c["bid"])
    nch = M * g.Kc // 8
    raw = []
    for sl in range_constexpr(g.slots):
        ch = fx.min(ch0 + sl, g.C - 1)
        for i in range_constexpr((nch + NT - 1) // NT):
            j = fx.min(tid + NT * i, nch - 1)
            t = j // (g.Kc // 8)
            kc = j % (g.Kc // 8) * 8                 # column within the chunk
            k = ch * g.Kc + kc                        # intermediate column
            n = t * 2 * g.K + (k // 128) * 256 + k % 128
            raw.append((sl, i, t, kc, [D.ld(ry, n + o, 4, T.f32, D.CM_DEV) for o in (0, 4, 128, 132)]))
    for sl, i, t, kc, v in raw:
        gv = [v[0][0], v[0][1], v[0][2], v[0][3], v[1][0], v[1][1], v[1][2], v[1][3]]
        uv = [v[2][0], v[2][1], v[2][2], v[2][3], v[3][0], v[3][1], v[3][2], v[3][3]]
        x = [D.silu(gv[e]) * uv[e] for e in range(8)]
        store_quant8(c, g, xq, xs, x, t, sl * g.Kc + kc, tid + NT * i < nch)
    gpu.barrier()


@D.traced
def prologue_copy(c, g, xq, xs, srcq, srcs):
    """LDS x <- fp8 [M][K] + scales [M][K/128] from global, the CTA's chunk slots only."""
    M = g.M
    tid = c["tid"]
    zero_xs_tail(c, g, xs)
    rq, rsc = D.rsrc(srcq), D.rsrc(srcs)
    ch0 = g.chunk0(c["bid"])
    for sl in range_constexpr(g.slots):
        ch = fx.min(ch0 + sl, g.C - 1)
        nd = M * g.Kc // 16                      # dwordx4 per slot
        for i in range_constexpr((nd + NT - 1) // NT):
            j = fx.min(tid + NT * i, nd - 1)
            t = j // (g.Kc // 16)
            o = j % (g.Kc // 16) * 16              # byte offset in the chunk
            v = D.ld(rq, (t * g.K + ch * g.Kc + o) // 4, 4, cm=D.CM_DEV)
            if tid + NT * i < nd:
                fx.ptr_store(v, xq + (t * g.XR + sl * g.Kc + o) // 4)
        ns = M * g.KSc
        if tid < ns:
            t = tid // g.KSc
            s = tid % g.KSc
            fx.ptr_store(D.ldf(rsc, t * g.KS + ch * g.KSc + s, cm=D.CM_DEV), xs + (t * g.XSR + sl * g.KSc + s))
    gpu.barrier()


# ---------------------------------------------------------------- epilogues
@D.traced
def epi_res(c, M, grp, tot, h_in, h_out, ssq_out):
    """h_out[:, 16 grp + col] = h_in + tot; ssq_out[t][grp] = sum over the 16 columns."""
    lane, col = c["lane"], c["col"]
    ri, ro, rq = D.rsrc(h_in), D.rsrc(h_out), D.rsrc(ssq_out)
    ng = H // 16
    for r in range_constexpr(4):
        t = 4 * c["gl"] + r
        live = t < M
        tt = fx.min(t, M - 1)
        v = D.ldf(ri, tt * H + grp * 16 + col, cm=D.CM_DEV) + tot[r]
        s = D.xor_sum(v * v, lane, (1, 2, 4, 8))
        if live:
            D.st(v, ro, tt * H + grp * 16 + col, D.CM_DEV)
            if col == 0:
                D.st(s, rq, tt * ng + grp, D.CM_DEV)


@D.traced
def epi_raw(c, M, N, grp, tot, out):
    """out[:, 16 grp + col] = tot (fp32 [M][N])."""
    ro = D.rsrc(out)
    for r in range_constexpr(4):
        t = 4 * c["gl"] + r
        if t < M:
            D.st(tot[r], ro, t * N + grp * 16 + c["col"], D.CM_DEV)


@D.traced
def epi_act(c, M, blk, tots, act_q, act_s, kact):
    """gate_up block blk: groups 0..7 gate, 8..15 up of intermediate columns
    128 blk + 16 j + col. act = silu(gate) * up -> fp8 + one pow2 scale per (token,
    block) -> act_q [M][kact] bytes, act_s [M][kact / 128]."""
    lane, col = c["lane"], c["col"]
    rq, rs = D.rsrc(act_q), D.rsrc(act_s)
    for r in range_constexpr(4):
        t = 4 * c["gl"] + r
        tt = fx.min(t, M - 1)
        a = [D.silu(tots[j][r]) * tots[8 + j][r] for j in range(8)]
        am = fx.Float32(0.0)
        for j in range_constexpr(8):
            am = fx.max(am, D.absf(a[j]))
        am = D.xor_max(am, lane, (1, 2, 4, 8))
        qi, sc = D.pow2_scale(fx.max(am, fx.Float32(1e-30)))
        for j in range_constexpr(8):
            # lane col % 4 == 0 packs columns col .. col + 3 of sub-block j
            v = [D.gather_f(a[j], lane + e) * qi for e in range(4)]
            w = D.fp8_pack4(v[0], v[1], v[2], v[3])
            if (t < M) & (col % 4 == 0):
                D.st(w, rq, (tt * kact + blk * 128 + 16 * j + col) // 4, D.CM_DEV)
        if (t < M) & (col == 0):
            D.st(sc, rs, tt * (kact // 128) + blk, D.CM_DEV)


@D.traced
def prologue_scaled(c, g, xq, xs, y, ssq, norm_eps, per_head):
    """LDS x <- y (fp32 [M][K]) for the CTA's chunk slots, quantized.
    per_head > 0: x = y * rsqrt(mean_head(o^2) + eps) with the head's 8 chunk sums of
    squares from ssq [M][K / per_head][8] (the GDN gated RMSNorm); else x = y."""
    M = g.M
    tid = c["tid"]
    zero_xs_tail(c, g, xs)
    ry, rq = D.rsrc(y), D.rsrc(ssq)
    ch0 = g.chunk0(c["bid"])
    nch = M * g.Kc // 8
    raw = []
    for sl in range_constexpr(g.slots):
        cb = fx.min(ch0 + sl, g.C - 1) * g.Kc
        for i in range_constexpr((nch + NT - 1) // NT):
            j = fx.min(tid + NT * i, nch - 1)
            t = j // (g.Kc // 8)
            kc = j % (g.Kc // 8) * 8
            k0 = cb + kc
            a = D.ld(ry, t * g.K + k0, 4, T.f32, D.CM_DEV)
            b = D.ld(ry, t * g.K + k0 + 4, 4, T.f32, D.CM_DEV)
            hs = None
            if const_expr(per_head > 0):
                hd = k0 // per_head
                hs = [D.ld(rq, (t * (g.K // per_head) + hd) * 8, 4, T.f32, D.CM_DEV),
                      D.ld(rq, (t * (g.K // per_head) + hd) * 8 + 4, 4, T.f32, D.CM_DEV)]
            raw.append((sl, i, t, kc, a, b, hs))
    for sl, i, t, kc, a, b, hs in raw:
        x = [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
        if const_expr(per_head > 0):
            s8 = hs[0][0] + hs[0][1] + hs[0][2] + hs[0][3] + hs[1][0] + hs[1][1] + hs[1][2] + hs[1][3]
            rs = D.rsq(s8 / fx.Float32(float(per_head)) + fx.Float32(norm_eps))
            x = [v * rs for v in x]
        store_quant8(c, g, xq, xs, x, t, sl * g.Kc + kc, tid + NT * i < nch)
    gpu.barrier()


@D.traced
def prologue_norm_bf16(c, g, xq, red, h, ssq, lnw, eps):
    """LM head: LDS x <- bf16(h * rstd * (1 + w)) for the CTA's chunk slots."""
    M, K = g.M, g.K
    assert g.kind == "bf16"
    tid = c["tid"]
    rh, rw = D.rsrc(h), D.rsrc(lnw)
    ch0 = g.chunk0(c["bid"])
    nch = M * g.Kc // 8
    raw = []
    for sl in range_constexpr(g.slots):
        cb = fx.min(ch0 + sl, g.C - 1) * g.Kc
        for i in range_constexpr((nch + NT - 1) // NT):
            j = fx.min(tid + NT * i, nch - 1)
            t = j // (g.Kc // 8)
            kc = j % (g.Kc // 8) * 8
            k0 = cb + kc
            raw.append((sl, i, t, kc, D.ld(rh, t * K + k0, 4, T.f32, D.CM_DEV),
                        D.ld(rh, t * K + k0 + 4, 4, T.f32, D.CM_DEV), D.ld(rw, k0 // 2, 4)))
    rstd_from_ssq(c, M, ssq, red, eps)
    for sl, i, t, kc, a, b, wv in raw:
        rs = fx.ptr_load(red + t)
        f = []
        for e in range_constexpr(4):
            f += [D.bf_lo(wv[e]), D.bf_hi(wv[e])]
        hv = [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
        x = [hv[e] * rs * (fx.Float32(1.0) + f[e]) for e in range(8)]
        words = [fx.Vector.from_elements([x[2 * q], x[2 * q + 1]], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                 for q in range(4)]
        if tid + NT * i < nch:
            fx.ptr_store(fx.Vector.from_elements(words, fx.Int32), xq + (t * g.XR + (sl * g.Kc + kc) * 2) // 4)
    gpu.barrier()


@D.traced
def embed_stage(c, M, tok_ptr, emb, h_out, ssq_out):
    """h = embed[tok] (bf16 -> fp32) and its 16-column sums of squares. CTA b handles
    groups b, b + CUS, ... (thread: token t = tid // 16, column tid % 16)."""
    tid, bid = c["tid"], c["bid"]
    lane = c["lane"]
    ng = H // 16
    t = tid // 16
    col = tid % 16
    if t < M:
        tok = D.ld(D.rsrc(tok_ptr), t)
        re = D.rsrc(emb + fx.Int64(tok) * (H * 2))     # the table is > 2 GiB: 64-bit row base
        ro, rq = D.rsrc(h_out), D.rsrc(ssq_out)
        for gi in range(bid, ng, c["cus"]):
            k = gi * 16 + col
            w = D.ld(re, k // 2)
            v = (k % 2 == 0).select(D.bf_lo(w), D.bf_hi(w))
            D.st(v, ro, t * H + k, D.CM_DEV)
            s = D.xor_sum(v * v, lane, (1, 2, 4, 8))
            if col == 0:
                D.st(s, rq, t * ng + gi, D.CM_DEV)
