"""Milestone test: persistent kernel running L MLP blocks (post-norm, gate_up, act,
down, residual) on the real checkpoint, against the emulating torch reference.

Usage: python tests/test_mlp.py MODEL_DIR [M] [L]
"""
import os
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr.typing import Int32, Int64  # noqa: E402

from mk import dev as D  # noqa: E402
from mk import reference as R  # noqa: E402
from mk import stages as S  # noqa: E402
from mk.dims import EPS, FFN, H  # noqa: E402
from mk.gemv import Gemv  # noqa: E402
from mk.pack import Packed  # noqa: E402

path = sys.argv[1]
M = int(sys.argv[2]) if len(sys.argv) > 2 else 1
L = int(sys.argv[3]) if len(sys.argv) > 3 else 2
CUS = torch.cuda.get_device_properties(0).multi_processor_count
WV = int(os.environ.get("WAVES", "8"))
PP = int(os.environ.get("P", "2"))
S.NT = 64 * WV
GU = Gemv("gu", 2 * FFN, H, M, CUS, waves=WV, P=PP, C=int(os.environ.get("CGU", "1"))).check()
DN = Gemv("down", H, FFN, M, CUS, C=int(os.environ.get("CDN", "8")), waves=WV, P=PP).check()
NTAB = 8
XQB = max(GU.XR, DN.XR) * M
XSF = 16 * max(GU.XSR, DN.XSR)
PART = max(GU.part_floats(), DN.part_floats())


@fx.struct
class Lds:
    xq: fx.Array[fx.Int32, (XQB + 3) // 4, 16]
    xs: fx.Array[fx.Float32, XSF, 16]
    red: fx.Array[fx.Float32, 16, 16]


def tab64(r, i):
    v = D.ld(r, 2 * i, 2)
    return (fx.Int64(D.uniform(v[1])) << 32) | fx.Int64(fx.Uint32(D.uniform(v[0])))


PROBES = torch.zeros(CUS * 16, 16, dtype=torch.int64, device="cuda")
if os.environ.get("PROBE"):
    D.PROBE["addr"] = PROBES.data_ptr()
    D.PROBE["waves"] = WV


@flyc.kernel(name=f"mlp_m{M}", known_block_size=[64 * WV, 1, 1])
def kmlp(tab: Int64, hb: Int64, ssq: Int64, actq: Int64, acts: Int64, part: Int64,
         cnt: Int64, flags: Int64, epoch: Int32, nlayers: Int32, stamps: Int64):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    lane = tid % 64
    c = dict(tid=tid, bid=bid, lane=lane, wave=tid // 64, gl=lane // 16, col=lane % 16)
    lds = fx.SharedAllocator().allocate(Lds).peek()
    xq, xs, red = lds.xq.ptr, lds.xs.ptr, lds.red.ptr
    rt = D.rsrc(tab)
    rf = D.rsrc(flags)
    rst = D.rsrc(stamps)

    def stamp(i, k):
        if tid == 0:
            t = D.memrealtime()
            D.st(fx.Vector.from_elements([fx.Int32(t & 0xFFFFFFFF), fx.Int32(t >> 32)], fx.Int32),
                 rst, ((i * CUS + bid) * 8 + k) * 2)
    stamp = D.traced(stamp)
    for i in range(0, nlayers):
        stamp(i, 0)
        cur = i % 2
        h_in = hb + fx.Int64(cur * M * H * 4)
        h_out = hb + fx.Int64((1 - cur) * M * H * 4)
        q_in = ssq + fx.Int64(cur * M * (H // 16) * 4)
        q_out = ssq + fx.Int64((1 - cur) * M * (H // 16) * 4)
        gu_t, gu_s = tab64(rt, i * NTAB + 0), tab64(rt, i * NTAB + 1)
        dn_t, dn_s = tab64(rt, i * NTAB + 2), tab64(rt, i * NTAB + 3)
        lnw = tab64(rt, i * NTAB + 4)
        # ---- gate_up
        w = GU.setup(c, xq, xs, gu_t, gu_s)
        D.grid_arrive(rf, epoch + 2 * i + 1, tid, bid)
        buf = GU.first_loads(w)
        D.grid_wait(rf, epoch + 2 * i + 1, tid, bid, CUS)
        stamp(i, 1)
        S.prologue_norm(c, GU, xq, xs, red, h_in, q_in, lnw, EPS)
        stamp(i, 2)
        D.PROBE["on"] = os.environ.get("PROBE") == "gu"
        GU.run(c, w, buf, part, cnt, lambda cg, tots: S.epi_raw(c, M, 2 * FFN, cg, tots[0], actq))
        D.PROBE["on"] = os.environ.get("PROBE") == "down"
        stamp(i, 3)
        # ---- down
        w2 = DN.setup(c, xq, xs, dn_t, dn_s)
        D.grid_arrive(rf, epoch + 2 * i + 2, tid, bid)
        stamp(i, 7)
        buf2 = DN.first_loads(w2)
        D.grid_wait(rf, epoch + 2 * i + 2, tid, bid, CUS)
        stamp(i, 4)
        S.prologue_act(c, DN, xq, xs, actq)
        stamp(i, 5)
        DN.run(c, w2, buf2, part, cnt + fx.Int64(65536 * 4),
               lambda cg, tots: S.epi_res(c, M, cg, tots[0], h_in, h_out, q_out))
        stamp(i, 6)


@flyc.jit
def launch(tab: Int64, hb: Int64, ssq: Int64, actq: Int64, acts: Int64, part: Int64, cnt: Int64,
           flags: Int64, epoch: Int32, nlayers: Int32, stamps: Int64, stream: fx.Stream = fx.Stream(None)):
    kmlp(tab, hb, ssq, actq, acts, part, cnt, flags, epoch, nlayers, stamps).launch(
        grid=(CUS,), block=(64 * WV,), stream=stream)


def main():
    t0 = time.time()
    W = R.Weights(path, "mxfp4", keep_raw=True, layers=L)
    PK = Packed(W, layers=L)
    print(f"load+pack {time.time() - t0:.0f}s; LDS xq {XQB} B xs {XSF * 4} B; part {PART * 4 / 1e6:.1f} MB")
    tab = torch.zeros(L * NTAB, dtype=torch.int64, device="cuda")
    for i in range(L):
        gt, gs = PK.mats[(i, "gu")]
        dt, ds = PK.mats[(i, "down")]
        tab[i * NTAB: i * NTAB + 5] = torch.tensor(
            [gt.data_ptr(), gs.data_ptr(), dt.data_ptr(), ds.data_ptr(),
             PK.small[(i, "post_attention_layernorm.weight")].data_ptr()])
    torch.manual_seed(0)
    h0 = torch.randn(M, H, device="cuda") * 2
    hb = torch.zeros(2, M, H, device="cuda")
    hb[0] = h0
    ssq = torch.zeros(2, M, H // 16, device="cuda")
    ssq[0] = h0.reshape(M, H // 16, 16).pow(2).sum(-1)
    actq = torch.zeros(M, 2 * FFN, device="cuda")      # gate_up output (fp32, gu_order)
    acts = torch.zeros(M, FFN // 128, device="cuda")
    part = torch.zeros(PART, device="cuda")
    cnt = torch.zeros(2 * 65536, dtype=torch.int32, device="cuda")
    flags = torch.zeros((CUS + 8) * 32, dtype=torch.int32, device="cuda")
    args = lambda ep: (tab.data_ptr(), hb.data_ptr(), ssq.data_ptr(), actq.data_ptr(), acts.data_ptr(),
                       part.data_ptr(), cnt.data_ptr(), flags.data_ptr(), ep, L, stamps.data_ptr())
    stamps = torch.zeros(L, CUS, 8, dtype=torch.int64, device="cuda")
    launch(*args(0), stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    out = hb[L % 2].clone()
    if os.environ.get("PROBE"):
        pr = PROBES.double() / 100.0
        base = pr[:CUS * WV, 0].min()
        def st_(k):
            v = pr[:, k] - base
            v = v[pr[:, k] > 0]
            return f"min {v.min():.1f} med {v.median():.1f} max {v.max():.1f}"
        print("run start", st_(0)); print("loop end ", st_(1)); print("run end  ", st_(2))
        le = (pr[:CUS * WV, 1] - base).reshape(CUS, WV)
        print("loop end by XCD (bid%8) median:", [round(le[x::8].median().item(), 1) for x in range(8)])
        print("loop end by wave median:", [round(le[:, w_].median().item(), 1) for w_ in range(WV)])
        print("loop end by bid//38 median:", [round(le[k * 38:(k + 1) * 38].median().item(), 1) for k in range(8)])
        cm = le.max(1).values
        cmean = le.mean(1)
        print(f"per-CTA max: min {cm.min():.1f} med {cm.median():.1f} max {cm.max():.1f}; "
              f"per-CTA mean: min {cmean.min():.1f} med {cmean.median():.1f} max {cmean.max():.1f}")
        for b in ():
            d1 = (pr[:, b + 1] - pr[:, b])[pr[:, b] > 0]
            d2 = (pr[:, b + 2] - pr[:, b + 1])[pr[:, b] > 0]
            print(f"flush atomic us: med {d1.median():.2f} max {d1.max():.2f}; LA us: med {d2.median():.2f} max {d2.max():.2f}")
        m = pr[:, 10] > 0
        if m.any():
            la_start = (pr[:, 10] - pr[:, 5])[m]; la_epi = (pr[:, 11] - pr[:, 10])[m]
            print(f"LA count {int(m.sum())}: loads us med {la_start.median():.2f} max {la_start.max():.2f}; "
                  f"epilogue us med {la_epi.median():.2f} max {la_epi.max():.2f}; end max {(pr[:, 11][m] - base).max():.1f}")
    # reference (emulated numerics)
    R.EMULATE = True
    h = h0.clone()
    for i in range(L):
        x = R.rms0(h, W.L(i, "post_attention_layernorm.weight"))
        gt = R.lin(x, W.L(i, "mlp.gate_proj.weight"))
        up = R.lin(x, W.L(i, "mlp.up_proj.weight"))
        h = h + R.lin(gt * torch.sigmoid(gt) * up, W.L(i, "mlp.down_proj.weight"))
    err = ((out - h).abs().max() / h.abs().max()).item()
    cos = torch.nn.functional.cosine_similarity(out, h, dim=-1).min().item()
    print(f"M={M} L={L}: rel max err {err:.3e} cos {cos:.7f}")
    # timing: graph of 20 launches
    ep = [2 * L + 1]

    def run():
        launch(*args(ep[0] * 0 + 1000 * 2 * L), stream=torch.cuda.current_stream())
    hb[0] = h0
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    eps_ = []
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for k in range(20):
                launch(*args((k + 1) * 2 * L), stream=s)
    torch.cuda.synchronize()
    # epochs must keep growing across replays: use fresh flags per replay
    flags.zero_()
    g.replay()
    torch.cuda.synchronize()
    flags.zero_()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    g.replay()
    e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1e3 / 20
    t = (stamps - stamps[:, :, 0].min()).double() / 100.0   # us (100 MHz)
    for i in range(min(L, 3)):
        row = []
        for k, nm in enumerate(["start", "bar1", "pro1", "gu", "bar2", "pro2", "down", "arr2"]):
            v = t[i, :, k]
            row.append(f"{nm} {v.min():.1f}/{v.median():.1f}/{v.max():.1f}")
        print(f"  layer {i}: " + "  ".join(row))
    nbytes = L * (GU.N * GU.K // 2 * 17 // 16 + DN.N * DN.K // 2 * 17 // 16)
    print(f"time per launch {us:.1f} us ({us / L:.1f} us/layer), {nbytes / us / 1e6:.2f} TB/s weights")
    assert err < 2e-2 and cos > 0.9999, "mismatch"


if __name__ == "__main__":
    main()
