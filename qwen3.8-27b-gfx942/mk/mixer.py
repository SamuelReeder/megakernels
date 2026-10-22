"""Token mixers of the megakernel: the Gated DeltaNet core and full attention.

Both run as wave tasks between two grid barriers (inputs: the raw fp32 output of
the in_proj / qkv GEMV; outputs: an fp32 vector the next GEMV prologue quantizes).

GDN task (row m, value head h, chunk c of 16 value rows), one wave:
  lanes 0..31: conv + silu + l2norm of q_j (j = h // 3), 4 channels a lane;
  lanes 32..63: the same for k_j. q, k -> wave LDS.
  lanes 0..15: conv + silu of value channel 16 c + lane; lanes 16..31: z.
  recurrence: lane l -> value row r = l // 4, k quarter l % 4 (32 fp32 of the state).
  y[m][128 h + 16 c + r] = o * norm_w * silu(z) (not yet divided by the head RMS);
  ssq[m][h][c] = sum of o^2 over the 16 rows (the out_proj prologue finishes the norm).
Conv state: double buffered by step parity (read buffer s % 2, write the other); the
task with h % 3 == 0, c == 0 writes the q_j, k_j channels, every task its 16 v channels.

Attention task (row m, kv head j, chunk ci of CH positions), one wave:
  q (6 heads) and k_new: zero-centred RMSNorm over 256, NeoX RoPE on dims 0..63.
  scores (lane = position) -> partial softmax (max, sum) -> P V (lane = 4 dims).
  The chunk holding position pos writes k_new, v_new to the cache (bf16).
  Partial record [m][j][ci]: max[6], sum[6], o[6][256] fp32. The last arriver of
  (m, j) merges the chunks and writes o * sigmoid(gate) -> og[m][6144].
"""
from __future__ import annotations

import math

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from . import dev as D
from .dims import (A_OFF, B_OFF, CONV_DIM, DK, HD, K_OFF, NKV, NQ, NVH, Q_OFF, ROT, THETA, V_OFF,
                   Z_OFF, INP_N, QKV_N, EPS)

CH = 64                 # attention positions per task
PVB = 16                # P V: positions per load batch
REC = 2 * 8 + 6 * HD    # partial record floats (max[6] sum[6] padded to 16, o[6][256])
GQ = NQ // NKV          # 6 q heads per kv head


# ------------------------------------------------------------------ GDN
@D.traced
def gdn_task(c, M, task, y_in, cw, conv_r, conv_w, a_log, dt_bias, norm_w, ssm, y_out, ssq_out,
             wl):
    """One (m, h, chunk) task. wl: this wave's LDS (256 floats: q[128] k[128])."""
    lane = c["lane"]
    m = task // (NVH * 8)
    h = task % (NVH * 8) // 8
    ch = task % 8
    j = h // 3
    ry, rc, rcr, rcw = D.rsrc(y_in), D.rsrc(cw), D.rsrc(conv_r), D.rsrc(conv_w)
    ybase = m * INP_N
    cbase = m * CONV_DIM * 3
    # ---- q_j / k_j: lane l < 32 -> q channels 4l..4l+3, l >= 32 -> k channels 4(l-32)..
    isk = lane >= 32
    ci = (lane % 32) * 4
    chan = isk.select(K_OFF + j * DK + ci, Q_OFF + j * DK + ci)   # conv channel = in_proj col
    xin = D.ld(ry, ybase + chan, 4, T.f32, D.CM_DEV)
    sv = []
    for e in range_constexpr(4):
        sv.append([D.ldf(rcr, cbase + (chan + e) * 3 + t) for t in range(3)])
    wv = [D.ld(rc, (chan + e) * 2, 2) for e in range(4)]            # 4 bf16 taps = 2 dwords
    yv = []
    for e in range_constexpr(4):
        taps = [D.bf_lo(wv[e][0]), D.bf_hi(wv[e][0]), D.bf_lo(wv[e][1]), D.bf_hi(wv[e][1])]
        acc = sv[e][0] * taps[0] + sv[e][1] * taps[1] + sv[e][2] * taps[2] + xin[e] * taps[3]
        yv.append(D.silu(acc))
    ss = yv[0] * yv[0] + yv[1] * yv[1] + yv[2] * yv[2] + yv[3] * yv[3]
    ss = D.xor_sum(ss, lane, (1, 2, 4, 8, 16))
    rn = D.rsq(ss + fx.Float32(1e-6))
    qs = isk.select(fx.Float32(1.0), fx.Float32(1.0 / math.sqrt(DK)))
    for e in range_constexpr(4):
        fx.ptr_store(yv[e] * rn * qs, wl + (isk.select(DK, 0) + ci + e))
    # conv state of the q_j / k_j channels: written once per k head
    if (h % 3 == 0) & (ch == 0):
        nv = []
        for e in range_constexpr(4):
            nv += [sv[e][1], sv[e][2], xin[e]]
        for q in range_constexpr(3):
            D.st(fx.Vector.from_elements(nv[4 * q:4 * q + 4], fx.Float32), rcw, cbase + chan * 3 + 4 * q)
    # ---- v / z of this chunk: lane < 16 -> v channel, 16..31 -> z
    vr = lane % 16
    vch = V_OFF + h * 128 + ch * 16 + vr
    vx = D.ldf(ry, ybase + vch, cm=D.CM_DEV)
    vs = [D.ldf(rcr, cbase + vch * 3 + t) for t in range(3)]
    vw = D.ld(rc, vch * 2, 2)
    vacc = vs[0] * D.bf_lo(vw[0]) + vs[1] * D.bf_hi(vw[0]) + vs[2] * D.bf_lo(vw[1]) + vx * D.bf_hi(vw[1])
    vval = D.silu(vacc)
    if lane < 16:
        D.st(vs[1], rcw, cbase + vch * 3)
        D.st(vs[2], rcw, cbase + vch * 3 + 1)
        D.st(vx, rcw, cbase + vch * 3 + 2)
    zval = D.ldf(ry, ybase + Z_OFF + h * 128 + ch * 16 + vr, cm=D.CM_DEV)
    bval = D.ldf(ry, ybase + B_OFF + h, cm=D.CM_DEV)
    aval = D.ldf(ry, ybase + A_OFF + h, cm=D.CM_DEV)
    alog = D.bf_lo(D.ld(D.rsrc(a_log), h // 2) >> ((h % 2) * 16))
    dtb = D.bf_lo(D.ld(D.rsrc(dt_bias), h // 2) >> ((h % 2) * 16))
    xa = aval + dtb
    sp = (xa > fx.Float32(20.0)).select(xa, D.logf1p_exp(xa))
    decay = D.expf(-D.expf(alog) * sp)
    beta = D.sigmoid(bval)
    # q/k reach the other lanes through this wave's LDS: a wave's LDS ops run in order
    # ---- recurrence: lane -> row r = lane // 4, k quarter kq = lane % 4
    r = lane // 4
    kq = lane % 4
    rs = D.rsrc(ssm)
    sbase = ((m * NVH + h) * 128 + ch * 16 + r) * DK + kq * 32
    S = [D.ld(rs, sbase + 4 * i, 4, T.f32, D.CM_NT) for i in range(8)]
    kk = [fx.ptr_load(wl + (DK + kq * 32 + e)) for e in range(32)]
    qq = [fx.ptr_load(wl + (kq * 32 + e)) for e in range(32)]
    kv = fx.Float32(0.0)
    for i in range_constexpr(8):
        for e in range_constexpr(4):
            kv = kv + S[i][e] * kk[4 * i + e]
    kv = D.xor_sum(kv * decay, lane, (1, 2))
    vrow = D.gather_f(vval, r)                      # v of row r lives in lane r
    delta = (vrow - kv) * beta
    o = fx.Float32(0.0)
    Sn = []
    for i in range_constexpr(8):
        nv = []
        for e in range_constexpr(4):
            s_ = S[i][e] * decay + kk[4 * i + e] * delta
            nv.append(s_)
            o = o + s_ * qq[4 * i + e]
        Sn.append(fx.Vector.from_elements(nv, fx.Float32))
    for i in range_constexpr(8):
        D.st(Sn[i], rs, sbase + 4 * i, D.CM_NT)
    o = D.xor_sum(o, lane, (1, 2))
    zrow = D.gather_f(zval, 16 + r)
    nwr = D.bf_lo(D.ld(D.rsrc(norm_w), (ch * 16 + r) // 2) >> (((ch * 16 + r) % 2) * 16))
    y = o * nwr * D.silu(zrow)
    s2 = D.xor_sum((kq == 0).select(o * o, fx.Float32(0.0)), lane, (4, 8, 16, 32))
    if kq == 0:
        D.st(y, D.rsrc(y_out), m * NVH * 128 + h * 128 + ch * 16 + r, D.CM_DEV)
    if lane == 0:
        D.st(s2, D.rsrc(ssq_out), (m * NVH + h) * 8 + ch, D.CM_DEV)


# ------------------------------------------------------------------ attention
def rms_rope(lane, vals, w_word, cs, rope):
    """Zero-centred RMSNorm over 256 dims spread 4 a lane (dims 4 lane + e), then NeoX
    RoPE on dims 0..63 (pairs d, d + 32 live in lanes l, l ^ 8). w_word: the lane's 4
    bf16 norm weights (2 dwords); cs: (cos, sin) of the lane's 4 dims (dims < 32 index)."""
    ss = vals[0] * vals[0] + vals[1] * vals[1] + vals[2] * vals[2] + vals[3] * vals[3]
    ss = D.xor_sum(ss, lane, (1, 2, 4, 8, 16, 32))
    rn = D.rsq(ss / fx.Float32(float(HD)) + fx.Float32(EPS))
    wf = [D.bf_lo(w_word[0]), D.bf_hi(w_word[0]), D.bf_lo(w_word[1]), D.bf_hi(w_word[1])]
    x = [vals[e] * rn * (fx.Float32(1.0) + wf[e]) for e in range(4)]
    if not rope:
        return x
    cos, sin = cs
    lo = lane < 8                                   # dims 0..31: x1; lanes 8..15: x2
    inr = lane < 16
    out = []
    for e in range_constexpr(4):
        partner = D.gather_f(x[e], lane ^ 8)
        r1 = x[e] * cos[e] - partner * sin[e]      # d < 32: x1 cos - x2 sin
        r2 = x[e] * cos[e] + partner * sin[e]      # d >= 32: x2 cos + x1 sin
        out.append(inr.select(lo.select(r1, r2), x[e]))
    return out


def bf16r(v):
    """Round f32 to bf16 and back (the KV cache precision)."""
    return fx.Float32(fx.Float32(v).to(fx.BFloat16))


def pack_bf2(a, b):
    return fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]


@D.traced
def attn_task(c, task, ncmax, y_qkv, qn_w, kn_w, rope_cs, kc, vc, pos_ptr, maxlen, rec, cnt, og,
              wl):
    """One (m, j, ci) task; wl: this wave's LDS (GQ * HD + 2 * HD // 2 floats).
    Returns nothing; the last arriver of (m, j) writes og."""
    lane = c["lane"]
    m = task // (NKV * ncmax)
    j = task % (NKV * ncmax) // ncmax
    ci = task % ncmax
    pos = D.ld(D.rsrc(pos_ptr), m)
    p0 = ci * CH
    if p0 <= pos:
        _attn_chunk(c, lane, m, j, ci, pos, p0, ncmax, y_qkv, qn_w, kn_w, rope_cs, kc, vc, maxlen,
                    rec, cnt, og, wl)


@D.traced
def _attn_chunk(c, lane, m, j, ci, pos, p0, ncmax, y_qkv, qn_w, kn_w, rope_cs, kc, vc, maxlen, rec,
                cnt, og, wl):
    ry = D.rsrc(y_qkv)
    ybase = m * QKV_N
    d0 = 4 * lane
    # rope table [maxlen][64] fp32: cos[0..31], sin[0..31] of position pos
    rcs = D.rsrc(rope_cs)
    di = d0 % 32
    cosv = D.ld(rcs, pos * 64 + di, 4, T.f32)
    sinv = D.ld(rcs, pos * 64 + 32 + di, 4, T.f32)
    cs = ([cosv[e] for e in range(4)], [sinv[e] for e in range(4)])
    qw = D.ld(D.rsrc(qn_w), d0 // 2, 2)
    kw = D.ld(D.rsrc(kn_w), d0 // 2, 2)
    # q of the 6 heads -> LDS q[h][256]
    for hh in range_constexpr(GQ):
        h = j * GQ + hh
        qv = D.ld(ry, ybase + h * 2 * HD + d0, 4, T.f32, D.CM_DEV)
        q = rms_rope(lane, [qv[e] for e in range(4)], qw, cs, True)
        fx.ptr_store(fx.Vector.from_elements(q, fx.Float32), wl + (hh * HD + d0))
    # k_new / v_new (bf16-rounded, as the cache holds them) -> LDS kn[256] vn[256] (bf16 pairs)
    kv_ = D.ld(ry, ybase + 2 * NQ * HD + j * HD + d0, 4, T.f32, D.CM_DEV)
    vv_ = D.ld(ry, ybase + 2 * NQ * HD + NKV * HD + j * HD + d0, 4, T.f32, D.CM_DEV)
    kn = rms_rope(lane, [kv_[e] for e in range(4)], kw, cs, True)
    kw2 = fx.Vector.from_elements([pack_bf2(kn[0], kn[1]), pack_bf2(kn[2], kn[3])], fx.Int32)
    vw2 = fx.Vector.from_elements([pack_bf2(vv_[0], vv_[1]), pack_bf2(vv_[2], vv_[3])], fx.Int32)
    _store_i32x2(wl, GQ * HD + d0 // 2, kw2)
    _store_i32x2(wl, GQ * HD + HD // 2 + d0 // 2, vw2)
    rk, rv = D.rsrc(kc), D.rsrc(vc)
    cache = ((m * NKV + j) * maxlen) * HD             # element offset of (m, j, pos 0)
    if (pos >= p0) & (pos < p0 + CH):
        D.st(kw2, rk, (cache + pos * HD + d0) // 2, D.CM_DEV)
        D.st(vw2, rv, (cache + pos * HD + d0) // 2, D.CM_DEV)
    # ---- scores: lane = position p0 + lane
    p = p0 + lane
    live = p <= pos
    isnew = p == pos
    pc = fx.min(p, pos)
    # runtime loops over load batches: unrolled, the compiler issues every batch's loads
    # at once and spills them
    sc = [fx.Float32(0.0)] * GQ
    for b in range(0, HD // 32):                       # 8 batches of 32 dims (4 x 16 B)
        kr = [D.ld(rk, (cache + pc * HD + 32 * b + 8 * u) // 2, 4) for u in range(4)]
        sc = _qk_batch(wl, b, isnew, kr, sc)
    scale = 1.0 / math.sqrt(HD)
    mx, sm, pr = [], [], []
    for hh in range_constexpr(GQ):
        s_ = live.select(sc[hh] * scale, fx.Float32(-3.0e38))
        mh = D.xor_max(s_, lane, (1, 2, 4, 8, 16, 32))
        ph = live.select(D.expf(s_ - mh), fx.Float32(0.0))
        mx.append(mh)
        sm.append(D.xor_sum(ph, lane, (1, 2, 4, 8, 16, 32)))
        pr.append(ph)
    # p[h][64] -> LDS (over the q region: q is no longer needed)
    for hh in range_constexpr(GQ):
        fx.ptr_store(pr[hh], wl + (hh * CH + lane))
    # ---- P V: lane -> dims d0..d0+3
    # positions past pos have p = 0 (their address is clamped to pos); loads go out in
    # batches of PVB so their latencies overlap
    o = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(GQ)]
    vl_ = _load_i32x2(wl, GQ * HD + HD // 2 + d0 // 2)
    for b in range(0, CH // PVB):
        vrs = [D.ld(rv, (cache + fx.min(p0 + b * PVB + k, pos) * HD + d0) // 2, 2) for k in range(PVB)]
        o = _pv_batch(wl, b, p0, pos, vl_, vrs, o)
    # ---- partial record
    rr = D.rsrc(rec)
    rbase = ((m * NKV + j) * ncmax + ci) * REC
    if lane < GQ:
        D.st(_sel6(lane, mx), rr, rbase + lane, D.CM_DEV)
        D.st(_sel6(lane, sm), rr, rbase + 8 + lane, D.CM_DEV)
    for hh in range_constexpr(GQ):
        D.st(o[hh], rr, rbase + 16 + hh * HD + d0, D.CM_DEV)
    # ---- last arriver of (m, j) merges the chunks
    rocm_wait()
    nchunk = pos // CH + 1
    old = fx.Int32(0)
    if lane == 0:
        old = D.atomic_add_dev(cnt + (m * NKV + j) * 4, 1)
    old = D.uniform(old)
    if (old + 1) % nchunk == 0:
        _attn_merge(lane, m, j, nchunk, ncmax, ry, ybase, rr, og)


@D.traced
def _qk_batch(wl, b, isnew, kr, sc):
    """Scores of the lane's position over dims 32 b .. 32 b + 31 (q from LDS)."""
    sc = list(sc)
    for u in range_constexpr(4):
        dd = 32 * b + 8 * u
        kl = _load_i32x4(wl, GQ * HD + dd // 2)
        qs = [[_load_f4(wl, hh * HD + dd), _load_f4(wl, hh * HD + dd + 4)] for hh in range(GQ)]
        for e in range_constexpr(4):
            w = isnew.select(kl[e], kr[u][e])
            k0, k1 = D.bf_lo(w), D.bf_hi(w)
            for hh in range_constexpr(GQ):
                qv = qs[hh][e // 2]
                sc[hh] = sc[hh] + qv[2 * (e % 2)] * k0 + qv[2 * (e % 2) + 1] * k1
    return sc


@D.traced
def _pv_batch(wl, b, p0, pos, vl_, vrs, o):
    """o += p V over positions p0 + PVB b + k (p = 0 past pos)."""
    o = [[o[hh][e] for e in range(4)] for hh in range(GQ)]
    for k in range_constexpr(PVB):
        pp = b * PVB + k
        nw = (p0 + pp) == pos
        w0 = nw.select(vl_[0], vrs[k][0])
        w1 = nw.select(vl_[1], vrs[k][1])
        vf = [D.bf_lo(w0), D.bf_hi(w0), D.bf_lo(w1), D.bf_hi(w1)]
        for hh in range_constexpr(GQ):
            ph = fx.ptr_load(wl + (hh * CH + pp))
            for e in range_constexpr(4):
                o[hh][e] = o[hh][e] + ph * vf[e]
    return [fx.Vector.from_elements(o[hh], fx.Float32) for hh in range(GQ)]


def rocm_wait():
    from flydsl.expr import rocdl
    rocdl.s_waitcnt(vmcnt=0)


def _sel6(lane, vals):
    v = vals[0]
    for i in range(1, len(vals)):
        v = (lane == i).select(vals[i], v)
    return v


def _store_i32x2(base_f32, idx, v):
    """Store 2 i32 at float-index idx of an f32 LDS pointer."""
    fx.ptr_store(v.bitcast(fx.Float32), base_f32 + idx)


def _load_i32x2(base_f32, idx):
    return fx.Vector(fx.ptr_load(base_f32 + idx, result_type=fx.Vector.make_type(2, fx.Float32))).bitcast(fx.Int32)


def _load_f4(base_f32, idx):
    return fx.Vector(fx.ptr_load(base_f32 + idx, result_type=fx.Vector.make_type(4, fx.Float32)))


def _load_i32x4(base_f32, idx):
    return fx.Vector(fx.ptr_load(base_f32 + idx, result_type=fx.Vector.make_type(4, fx.Float32))).bitcast(fx.Int32)


MERGE_B = 2   # chunks per merge batch


@D.traced
def _attn_merge(lane, m, j, nchunk, ncmax, ry, ybase, rr, og):
    d0 = 4 * lane
    base = (m * NKV + j) * ncmax * REC
    gmax = [fx.Float32(-3.0e38)] * GQ
    tot = [fx.Float32(0.0)] * GQ
    acc = [[fx.Float32(0.0)] * 4 for _ in range(GQ)]
    for b0 in range(0, (nchunk + MERGE_B - 1) // MERGE_B):
        recs = []
        for k in range_constexpr(MERGE_B):
            cidx = fx.min(b0 * MERGE_B + k, nchunk - 1)
            ok = (b0 * MERGE_B + k) < nchunk
            rb = base + cidx * REC
            ms = D.ld(rr, rb, 4, T.f32, D.CM_DEV), D.ld(rr, rb + 4, 4, T.f32, D.CM_DEV)
            ls = D.ld(rr, rb + 8, 4, T.f32, D.CM_DEV), D.ld(rr, rb + 12, 4, T.f32, D.CM_DEV)
            os_ = [D.ld(rr, rb + 16 + hh * HD + d0, 4, T.f32, D.CM_DEV) for hh in range(GQ)]
            recs.append((ok, ms, ls, os_))
        for ok, ms, ls, os_ in recs:
            for hh in range_constexpr(GQ):
                mc = ms[hh // 4][hh % 4]
                lc = ls[hh // 4][hh % 4]
                mc = ok.select(mc, fx.Float32(-3.0e38))
                nm = fx.max(gmax[hh], mc)
                a_old = D.expf(gmax[hh] - nm)
                a_new = ok.select(D.expf(mc - nm), fx.Float32(0.0))
                tot[hh] = tot[hh] * a_old + lc * a_new
                for e in range_constexpr(4):
                    acc[hh][e] = acc[hh][e] * a_old + os_[hh][e] * a_new
                gmax[hh] = nm
    ro = D.rsrc(og)
    for hh in range_constexpr(GQ):
        h = j * GQ + hh
        gt = D.ld(ry, ybase + h * 2 * HD + HD + d0, 4, T.f32, D.CM_DEV)
        inv = fx.Float32(1.0) / tot[hh]
        vals = [acc[hh][e] * inv * D.sigmoid(gt[e]) for e in range(4)]
        D.st(fx.Vector.from_elements(vals, fx.Float32), ro, m * NQ * HD + h * HD + d0, D.CM_DEV)
