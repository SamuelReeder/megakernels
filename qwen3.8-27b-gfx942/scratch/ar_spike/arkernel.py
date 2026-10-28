"""Spike only (disposable). A persistent FlyDSL kernel: ROUNDS all-reduces of M x 5120
fp32 between TP ranks, a grid barrier after each, as the TP2 megakernel would run them.

Round r (barrier epoch e = epoch + 1 + r, tag e, parity region e & 1):
  CTA b owns the 16-column groups b and b + CUS (< 320), like the residual epilogue's
  last arriver of a group. Thread (group, row m, column pair) loads its partial
  src[r][m][col:col+2] and
  - mode "ar": puts it (value, tag) pairs to every rank's region [src rank][m][col]
    (system scope, DSv4.1 Mailbox), polls its own region for every rank's pair and
    sums in rank order 0 .. TP-1 (DSv4.1 sum_partials, fp32 here);
  - mode "local": sums the partial with itself (no peer traffic: the TP1 baseline).
  It stores the sum to out[r][m][col:col+2]; then the TP1 kernel's barrier
  (arrive, wait). Mode "ar_merged": the put only; after the barrier every CTA polls all
  M x 5120 pairs of every rank (the prologue merge of DESIGN-TP2 section 5).
No rank overwrites unread peer data: a rank writes region parity p in round r + 2 only
after it passed round r + 1, which needs every peer's round r + 1 put, which every peer
issues after its own grid barrier that follows its round r reads.
"""

import sys
import zlib

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

sys.path.insert(0, __file__.rsplit("/scratch/", 1)[0])
from mk import dev as D  # noqa: E402
from mk.vendor import buffer_ops as bo  # noqa: E402

H = 5120
GROUPS = H // 16
NT = 512
CM_SYS = 17  # sc0 | sc1: system scope (DSv4.1 ops.CM_SYS)
EPOCH_WORD_LINE = 8


def half_bytes(tp, mmax=8):
    """One parity's region: [src rank][M][H] (value, tag) pairs."""
    return tp * mmax * H * 8


@D.traced
def _spin(load_all, pending):
    v = load_all()
    while pending(v):
        rocdl.s_sleep(1)
        v = load_all()
    return v


class Mailbox:
    """DSv4.1 sync.Mailbox (put_words, poll) for the PEER space only."""

    def __init__(self, tag):
        self.tag = tag

    def put_words(self, base, i, words):
        vals = []
        for w in words:
            vals += [fx.Int32(w), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(vals, fx.Int32),
            D.rsrc(base),
            i * 2,
            cache_modifier=CM_SYS,
        )

    def poll(self, specs):
        tag = self.tag

        def load_all():
            words = []
            for b, i, n in specs:
                w = fx.Vector(
                    bo.buffer_load(
                        D.rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=CM_SYS,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        v = _spin(load_all, pending)
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs


def build(M, cus, tp, rank, mode, stamps):
    assert mode in ("ar", "ar_bf", "ar_merged", "local") and 1 <= M <= 8
    HALF = half_bytes(tp)
    NG = 2  # groups a CTA can own (320 groups on 304 CTAs)
    name = f"arspike_{zlib.crc32(repr((M, cus, tp, rank, mode, stamps)).encode()):08x}"

    def tab64(r, i):
        v = D.ld(r, 2 * i, 2)
        return (fx.Int64(D.uniform(v[1])) << 32) | fx.Int64(fx.Uint32(D.uniform(v[0])))

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kar(peers: Int64, flags: Int64, src: Int64, out: Int64, st: Int64, slot: Int32, rounds: Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        rf = D.rsrc(flags)
        rs = D.rsrc(src)
        ro = D.rsrc(out)
        rp = D.rsrc(peers)
        bases = [tab64(rp, p) for p in range(tp)]
        epoch = D.uniform(D.ld(rf, (cus + EPOCH_WORD_LINE) * D.LINE, cm=D.CM_DEV))

        def stamp(r, k):
            t = D.memrealtime()
            if tid == 0:
                D.st(
                    fx.Vector.from_elements(
                        [fx.Int32(t & 0xFFFFFFFF), fx.Int32(t >> 32)], fx.Int32
                    ),
                    D.rsrc(st),
                    (((slot * rounds + r) * cus + bid) * 2 + k) * 2,
                )

        for r in range(0, rounds):
            e = epoch + 1 + r
            if const_expr(stamps):
                stamp(r, 0)
            gi = tid // (M * 8)
            m = (tid // 8) % M
            q = tid % 8
            g = bid + gi * cus
            if (tid < NG * M * 8) & (g < GROUPS):
                col = g * 16 + 2 * q
                v = D.ldf(rs, (r * M + m) * H + col, 2)
                v0 = fx.Float32(v[0])
                v1 = fx.Float32(v[1])
                if const_expr(mode == "ar_merged"):
                    # the prologue merge (DESIGN-TP2 section 5): push only; every CTA
                    # polls all columns after the barrier
                    mb = Mailbox(e)
                    par = fx.Int64(e & 1) * fx.Int64(HALF)
                    for p in range_constexpr(tp):
                        mb.put_words(
                            bases[p] + par,
                            (rank * M + m) * H + col,
                            [v0.bitcast(fx.Int32), v1.bitcast(fx.Int32)],
                        )
                elif const_expr(mode == "ar"):
                    mb = Mailbox(e)
                    par = fx.Int64(e & 1) * fx.Int64(HALF)
                    for p in range_constexpr(tp):
                        mb.put_words(
                            bases[p] + par,
                            (rank * M + m) * H + col,
                            [v0.bitcast(fx.Int32), v1.bitcast(fx.Int32)],
                        )
                    ws = mb.poll(
                        [(bases[rank] + par, (s * M + m) * H + col, 2) for s in range(tp)]
                    )
                    a0 = fx.Int32(ws[0][0]).bitcast(fx.Float32)
                    a1 = fx.Int32(ws[0][1]).bitcast(fx.Float32)
                    for s in range_constexpr(1, tp):
                        a0 = a0 + fx.Int32(ws[s][0]).bitcast(fx.Float32)
                        a1 = a1 + fx.Int32(ws[s][1]).bitcast(fx.Float32)
                elif const_expr(mode == "ar_bf"):
                    # DSv4.1's format: a bf16 pair a word, one tag a word, fp32 sum
                    mb = Mailbox(e)
                    par = fx.Int64(e & 1) * fx.Int64(HALF)
                    w = (
                        fx.Vector.from_elements([v0, v1], fx.Float32)
                        .to(fx.BFloat16)
                        .bitcast(fx.Int32)[0]
                    )
                    for p in range_constexpr(tp):
                        mb.put_words(bases[p] + par, (rank * M + m) * (H // 2) + col // 2, [w])
                    ws = mb.poll(
                        [(bases[rank] + par, (s * M + m) * (H // 2) + col // 2, 1) for s in range(tp)]
                    )
                    a0 = D.bf_lo(ws[0][0])
                    a1 = D.bf_hi(ws[0][0])
                    for s in range_constexpr(1, tp):
                        a0 = a0 + D.bf_lo(ws[s][0])
                        a1 = a1 + D.bf_hi(ws[s][0])
                else:
                    a0 = v0 + v0
                    a1 = v1 + v1
                if const_expr(mode != "ar_merged"):
                    D.st(
                        fx.Vector.from_elements([a0, a1], fx.Float32),
                        ro,
                        (r * M + m) * H + col,
                    )
            if const_expr(stamps):
                stamp(r, 1)
            D.grid_arrive(rf, e, tid, bid)
            D.grid_wait(rf, e, tid, bid, cus)
            if const_expr(mode == "ar_merged"):
                # every CTA reads every column pair of every rank (M x H / 2 pairs of
                # pairs a source); CTA 0 stores the sums for the exactness check
                mb = Mailbox(e)
                par = fx.Int64(e & 1) * fx.Int64(HALF)
                NPAIR = M * H // 2
                for i in range_constexpr((NPAIR + NT - 1) // NT):
                    j = fx.min(tid + NT * i, NPAIR - 1)
                    mm = j // (H // 2)
                    cc = j % (H // 2) * 2
                    ws = mb.poll(
                        [(bases[rank] + par, (s * M + mm) * H + cc, 2) for s in range(tp)]
                    )
                    a0 = fx.Int32(ws[0][0]).bitcast(fx.Float32)
                    a1 = fx.Int32(ws[0][1]).bitcast(fx.Float32)
                    for s in range_constexpr(1, tp):
                        a0 = a0 + fx.Int32(ws[s][0]).bitcast(fx.Float32)
                        a1 = a1 + fx.Int32(ws[s][1]).bitcast(fx.Float32)
                    if (bid == 0) & (tid + NT * i < NPAIR):
                        D.st(
                            fx.Vector.from_elements([a0, a1], fx.Float32),
                            ro,
                            (r * M + mm) * H + cc,
                        )
        if (bid == 0) & (tid == 0):
            D.st(epoch + rounds, rf, (cus + EPOCH_WORD_LINE) * D.LINE, D.CM_DEV)

    @flyc.jit
    def launch(peers: Int64, flags: Int64, src: Int64, out: Int64, st: Int64, slot: Int32,
               rounds: Int32, stream: fx.Stream = fx.Stream(None)):
        kar(peers, flags, src, out, st, slot, rounds).launch(grid=(cus,), block=(NT,), stream=stream)

    return launch
