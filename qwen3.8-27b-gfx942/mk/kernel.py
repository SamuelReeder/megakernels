"""The Qwen3.8-27B decode megakernel: one persistent launch per decode step.

Grid = CU count, 512 threads, all CTAs co-resident. Stages are separated by
grid barriers (dev.grid_arrive / grid_wait). Per layer:

  GDN layer:  in_proj GEMV -> | GDN core tasks -> | out_proj GEMV (+h) -> |
  attn layer: qkv GEMV     -> | attention tasks -> | o_proj GEMV (+h)   -> |
  both:       gate_up GEMV -> | down GEMV (+h) -> |
then the final norm + LM head GEMV (bf16) -> logits.

Residual stream: h0 at a layer start, h1 after the mixer (both fp32 [M][H], with
their 16-column sums of squares ssq0 / ssq1). See docs/DESIGN.md.
"""

import os
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from . import dev as D
from . import mixer as X
from . import stages as S
from .dims import (CONV_DIM, EPS, FFN, H, HD, INP_N, NKV, NL, NQ, NVH, QKV_N, VAL_DIM, VOCAB)
from .gemv import Gemv

NTAB = 16      # pointer-table slots per layer
NGLOB = 16     # global pointer-table slots
WAVES = 8
NT = 64 * WAVES

# global table slots
G_EMBED, G_LM, G_FNORM, G_ROPE, G_HB, G_SSQ, G_Y, G_Y2, G_HSSQ, G_PART, G_CNT, G_FLAGS, G_REC, \
    G_TOK, G_POS, G_LOGITS = range(16)
# per-layer slots
T_MIX_T, T_MIX_S, T_OUT_T, T_OUT_S, T_GU_T, T_GU_S, T_DN_T, T_DN_S, T_IN_LN, T_POST_LN = range(10)
T_ALOG, T_DTB, T_CONVW, T_NORMW, T_CONV, T_SSM = range(10, 16)        # GDN layers
T_QN, T_KN, T_KC, T_VC = range(10, 14)                                # attention layers

CNT_STRIDE = 65536   # i32 counters per counter region


@dataclass(frozen=True)
class Cfg:
    M: int
    cus: int
    maxlen: int
    stop: int = 99      # debug: run only the first `stop` stages of each layer (1..5)

    @property
    def ncmax(self):
        return self.maxlen // X.CH

    def specs(self):
        M, cus = self.M, self.cus
        g = lambda name, n, k, C=1, kind="mx": Gemv(name, n, k, M, cus, WAVES, C, kind).check()  # noqa: E731
        return dict(inp=g("inp", INP_N, H), out=g("out", H, VAL_DIM, C=8), qkv=g("qkv", QKV_N, H),
                    o=g("o", H, NQ * HD, C=8), gu=g("gu", 2 * FFN, H), down=g("down", H, FFN, C=8),
                    lm=g("lm", VOCAB, H, C=8, kind="bf16"))

    def part_floats(self):
        return max(s.part_floats() for s in self.specs().values())


CNT_REGION = dict(inp=0, out=1, qkv=2, o=3, gu=4, down=5, lm=6, attn=7)


STOP = [99]
SKIP = set(os.environ.get("MK_SKIP", "").split(","))   # debug: stage code left out (ISA bisection)
PROFILE = {"addr": None}   # debug: i64 [barriers][CUS][2] = (arrive, wait end) per CTA, 100 MHz


def build(cfg: Cfg):
    M, CUS = cfg.M, cfg.cus
    STOP[0] = cfg.stop
    SP = cfg.specs()
    S.NT = NT
    XQB = max(s.XR for s in SP.values()) * M
    XSF = 16 * max(s.XSR for s in SP.values())
    ATT_W = X.GQ * HD + HD          # floats per wave (q / p, then k_new v_new bf16 pairs)
    GDN_W = 256
    NTASK_G = M * NVH * 8
    NTASK_A = M * NKV * cfg.ncmax
    CONV_TOT = (NL * 3 // 4) * M * CONV_DIM * 3 * 4    # bytes of one conv-state parity buffer

    @fx.struct
    class GemvLds:
        xq: fx.Array[fx.Int32, (XQB + 3) // 4, 16]
        xs: fx.Array[fx.Float32, XSF, 16]

    @fx.struct
    class MixLds:
        aw: fx.Array[fx.Float32, WAVES * ATT_W, 16]

    @fx.union
    class StageLds:
        g: GemvLds
        x: MixLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, 16, 16]

    def tab64(r, i):
        v = D.ld(r, 2 * i, 2)
        return (fx.Int64(D.uniform(v[1])) << 32) | fx.Int64(fx.Uint32(D.uniform(v[0])))

    @flyc.kernel(name=f"qwen38_decode_m{M}_l{cfg.maxlen}", known_block_size=[NT, 1, 1])
    def kdec(tab: Int64, glob: Int64, epoch: Int32, parity: Int32, nlayers: Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = D.uniform(tid // 64)
        c = dict(tid=tid, bid=bid, lane=lane, wave=wave, gl=lane // 16, col=lane % 16, cus=CUS)
        alloc = fx.SharedAllocator()
        red = alloc.allocate(Smem).peek().red.ptr
        st_lds = alloc.allocate(StageLds)
        gl_, xl_ = st_lds.g.peek(), st_lds.x.peek()
        xq, xs = gl_.xq.ptr, gl_.xs.ptr
        wl = xl_.aw.ptr + wave * ATT_W
        rg = D.rsrc(glob)
        rt = D.rsrc(tab)
        G = [tab64(rg, k) for k in range(NGLOB)]
        rf = D.rsrc(G[G_FLAGS])
        hb0, hb1 = G[G_HB], G[G_HB] + fx.Int64(M * H * 4)
        q0, q1 = G[G_SSQ], G[G_SSQ] + fx.Int64(M * (H // 16) * 4)
        cnt = lambda k: G[G_CNT] + fx.Int64(CNT_REGION[k] * CNT_STRIDE * 4)  # noqa: E731
        part = G[G_PART]
        gw = bid * WAVES + wave
        ep = [0]

        def stamp(slot):
            t = D.memrealtime()
            if tid == 0:
                D.st(fx.Vector.from_elements([fx.Int32(t & 0xFFFFFFFF), fx.Int32(t >> 32)], fx.Int32),
                     D.rsrc(fx.Int64(PROFILE["addr"])), slot * 2)

        def barrier_then(spec, tiles, scales, e):
            """Arrive, prefetch the GEMV's first units, wait."""
            w = None
            if const_expr(spec is not None):
                # opaque ids: keep the stage's address math inside the stage
                tid_s = D.opaque_v(tid)
                c.update(tid=tid_s, bid=D.opaque_s(bid), lane=tid_s % 64, wave=D.uniform(tid_s // 64),
                         gl=tid_s % 64 // 16, col=tid_s % 16)
                w = spec.setup(c, xq, xs, tiles, scales)
            if const_expr(PROFILE["addr"] is not None):
                slot = ((e - epoch - 1) * CUS + bid) * 2
                stamp(slot)
            D.grid_arrive(rf, e, tid, bid)
            buf = spec.first_loads(w) if spec is not None else None
            D.grid_wait(rf, e, tid, bid, CUS)
            D.grid_wait(rf, e, tid, bid, CUS)
            if const_expr(PROFILE["addr"] is not None):
                stamp(slot + 1)
            return w, buf

        # ---- embedding (+ zero the attention counters: their totals change with pos)
        S.embed_stage(c, M, G[G_TOK], G[G_EMBED], hb0, q0)
        if bid == 0:
            for i in range_constexpr((M * NKV + NT - 1) // NT):
                if tid + NT * i < M * NKV:
                    D.st(fx.Int32(0), D.rsrc(cnt("attn")), tid + NT * i, D.CM_DEV)

        c0 = dict(c)
        for i in range(0, nlayers):
            # re-derive the thread ids inside the loop through opaque moves, so the
            # compiler cannot hoist every stage's per-lane address math out of the layer
            # loop (keeping it all live at once spills registers)
            tid_l = D.opaque_v(tid)
            bid_l = D.opaque_s(bid)
            lane_l, wave_l = tid_l % 64, D.uniform(tid_l // 64)
            c.update(tid=tid_l, bid=bid_l, lane=lane_l, wave=wave_l, gl=lane_l // 16, col=lane_l % 16)
            gw_l = bid_l * WAVES + wave_l
            e0 = epoch + 1 + i * 5
            ri = i * NTAB
            T = lambda k: tab64(rt, ri + k)  # noqa: E731
            if i % 4 == 3:
                _attn_layer(c, SP, T, G, e0, xq, xs, red, wl, hb0, hb1, q0, q1, part, cnt, barrier_then,
                            gw_l, NTASK_A, cfg)
            else:
                _gdn_layer(c, SP, T, G, e0, xq, xs, red, wl, hb0, hb1, q0, q1, part, cnt, barrier_then,
                           gw_l, NTASK_G, parity, CONV_TOT)
            if const_expr(STOP[0] >= 4):
                _mlp(c, SP, T, e0 + 3, xq, xs, red, hb0, hb1, q0, q1, part, cnt, barrier_then, G)
        c.clear()
        c.update(c0)
        # ---- final norm + LM head
        lm = SP["lm"]
        w, buf = barrier_then(lm, G[G_LM], G[G_LM], epoch + 1 + nlayers * 5)
        S.prologue_norm_bf16(c, lm, xq, red, hb0, q0, G[G_FNORM], EPS)
        if const_expr(lm.name not in SKIP):
            lm.run(c, w, buf, part, cnt("lm"), lambda cg, tots: S.epi_raw(c, M, VOCAB, cg, tots[0], G[G_LOGITS]))
        if const_expr(PROFILE["addr"] is not None):
            stamp(((nlayers * 5 + 1) * CUS + bid) * 2)

    @flyc.jit
    def launch(tab: Int64, glob: Int64, epoch: Int32, parity: Int32, nlayers: Int32,
               stream: fx.Stream = fx.Stream(None)):
        kdec(tab, glob, epoch, parity, nlayers).launch(grid=(CUS,), block=(NT,), stream=stream)

    return launch


def barriers_per_step(nlayers):
    return nlayers * 5 + 1


@D.traced
def _gdn_layer(c, SP, T, G, e0, xq, xs, red, wl, hb0, hb1, q0, q1, part, cnt, barrier_then, gw,
               ntask, parity, conv_tot):
    M = c_M(SP)
    inp, out = SP["inp"], SP["out"]
    y, y2, hssq = G[G_Y], G[G_Y2], G[G_HSSQ]
    # in_proj
    w, buf = barrier_then(inp, T(T_MIX_T), T(T_MIX_S), e0)
    S.prologue_norm(c, inp, xq, xs, red, hb0, q0, T(T_IN_LN), EPS)
    if const_expr(inp.name not in SKIP):
        inp.run(c, w, buf, part, cnt("inp"), lambda cg, tots: S.epi_raw(c, M, INP_N, cg, tots[0], y))
    if const_expr(STOP[0] < 2):
        return
    # GDN core
    barrier_then(None, None, None, e0 + 1)
    conv = T(T_CONV)
    conv_r = conv + fx.Int64(parity) * conv_tot
    conv_w = conv + fx.Int64(1 - parity) * conv_tot
    for t in range(gw, (ntask, 0)["gdn" in SKIP], c["cus"] * WAVES):
        X.gdn_task(c, M, t, y, T(T_CONVW), conv_r, conv_w, T(T_ALOG), T(T_DTB), T(T_NORMW), T(T_SSM),
                   y2, hssq, wl)
    if const_expr(STOP[0] < 3):
        return
    # out_proj (+ residual)
    w, buf = barrier_then(out, T(T_OUT_T), T(T_OUT_S), e0 + 2)
    S.prologue_scaled(c, out, xq, xs, y2, hssq, EPS, 128)
    if const_expr(out.name not in SKIP):
        out.run(c, w, buf, part, cnt("out"), lambda cg, tots: S.epi_res(c, M, cg, tots[0], hb0, hb1, q1))


@D.traced
def _attn_layer(c, SP, T, G, e0, xq, xs, red, wl, hb0, hb1, q0, q1, part, cnt, barrier_then, gw,
                ntask, cfg):
    M = c_M(SP)
    qkv, o = SP["qkv"], SP["o"]
    y, y2 = G[G_Y], G[G_Y2]
    w, buf = barrier_then(qkv, T(T_MIX_T), T(T_MIX_S), e0)
    S.prologue_norm(c, qkv, xq, xs, red, hb0, q0, T(T_IN_LN), EPS)
    if const_expr(qkv.name not in SKIP):
        qkv.run(c, w, buf, part, cnt("qkv"), lambda cg, tots: S.epi_raw(c, M, QKV_N, cg, tots[0], y))
    barrier_then(None, None, None, e0 + 1)
    for t in range(gw, (ntask, 0)["attn" in SKIP], c["cus"] * WAVES):
        X.attn_task(c, t, cfg.ncmax, y, T(T_QN), T(T_KN), G[G_ROPE], T(T_KC), T(T_VC), G[G_POS],
                    cfg.maxlen, G[G_REC], cnt("attn"), y2, wl)
    w, buf = barrier_then(o, T(T_OUT_T), T(T_OUT_S), e0 + 2)
    S.prologue_scaled(c, o, xq, xs, y2, y2, EPS, 0)
    if const_expr(o.name not in SKIP):
        o.run(c, w, buf, part, cnt("o"), lambda cg, tots: S.epi_res(c, M, cg, tots[0], hb0, hb1, q1))


@D.traced
def _mlp(c, SP, T, e0, xq, xs, red, hb0, hb1, q0, q1, part, cnt, barrier_then, G):
    M = c_M(SP)
    gu, dn = SP["gu"], SP["down"]
    y = G[G_Y]
    w, buf = barrier_then(gu, T(T_GU_T), T(T_GU_S), e0)
    S.prologue_norm(c, gu, xq, xs, red, hb1, q1, T(T_POST_LN), EPS)
    if const_expr(gu.name not in SKIP):
        gu.run(c, w, buf, part, cnt("gu"), lambda cg, tots: S.epi_raw(c, M, 2 * FFN, cg, tots[0], y))
    if const_expr(STOP[0] < 5):
        return
    w, buf = barrier_then(dn, T(T_DN_T), T(T_DN_S), e0 + 1)
    S.prologue_act(c, dn, xq, xs, y)
    if const_expr(dn.name not in SKIP):
        dn.run(c, w, buf, part, cnt("down"), lambda cg, tots: S.epi_res(c, M, cg, tots[0], hb1, hb0, q0))


def c_M(SP):
    return SP["inp"].M
