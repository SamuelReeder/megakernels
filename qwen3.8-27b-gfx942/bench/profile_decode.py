"""Per-stage time profile of one megakernel decode step (thread 0 stamps per CTA).

Stage b runs from the release of barrier b to the arrival at barrier b + 1. Reports, per
stage kind, the median over layers of: span (first release -> last arrival), the mean
per-CTA busy time, and the barrier cost (last arrival -> last release).
Usage: python bench/profile_decode.py MODEL_DIR M CTX
"""
import os
import statistics as stt
import sys

import torch

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from mk import dev as D  # noqa: E402
from mk import kernel as KK  # noqa: E402
from mk import reference as R  # noqa: E402
from mk.dims import NL  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine  # noqa: E402

path, M, CTX = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
CUS = torch.cuda.get_device_properties(0).multi_processor_count
NB = KK.barriers_per_step(NL) + 1
prof = torch.zeros(NB, CUS, 2, dtype=torch.int64, device="cuda")
KK.PROFILE["addr"] = prof.data_ptr()
PSTAGE = os.environ.get("PROBE")      # per-wave stamps of one GEMV kind (last layer of that kind)
probes = torch.zeros(CUS * 8, 16, dtype=torch.int64, device="cuda")
if PSTAGE:
    D.PROBE.update(addr=probes.data_ptr(), stage=PSTAGE, waves=8)

W = R.Weights(path, "mxfp4", keep_raw=True, dequant=False)
eng = Engine(Packed(W), M, maxlen=((CTX + 64) // 64 + 1) * 64)
del W
tok = torch.zeros(M, dtype=torch.long, device="cuda")
for t in range(CTX):
    eng.step(tok)
torch.cuda.synchronize()
prof.zero_()
probes.zero_()
eng.step(tok)
torch.cuda.synchronize()
p = prof.double().cpu() / 100.0     # us
t0 = p[0, :, 0].min()
p -= t0
names = []
for i in range(NL):
    names += (["qkv", "attn", "o"] if i % 4 == 3 else ["inp", "gdn", "out"]) + ["gu", "down"]
names += ["lm"]
rows = {}
for b in range(NB - 1):
    rel = p[b, :, 1]
    arr = p[b + 1, :, 0]
    span = (arr.max() - rel.min()).item()
    busy = (arr - rel).mean().item()
    bar = (p[b, :, 1].max() - p[b, :, 0].max()).item()
    rows.setdefault(names[b], []).append((span, busy, bar))
total = p[NB - 1, :, 0].max().item()
print(f"embed (start -> first arrival) {p[0, :, 0].max().item():.1f} us; step total {total:.1f} us")
print(f"{'stage':6} {'n':>3} {'span us':>8} {'busy us':>8} {'barrier us':>10} {'sum span us':>11}")
acc = 0.0
for k, v in rows.items():
    sp = stt.median(r[0] for r in v)
    acc += sum(r[0] + r[2] for r in v)
    print(f"{k:6} {len(v):3d} {sp:8.1f} {stt.median(r[1] for r in v):8.1f} {stt.median(r[2] for r in v):10.2f} "
          f"{sum(r[0] for r in v):11.1f}")
print(f"sum of spans + barriers {acc:.1f} us")
if PSTAGE == "gdn":
    pr = probes.double().cpu() / 100.0 - t0
    act = pr[:, 4] > pr[:, 4].max() - 200      # waves that ran a task in the last GDN layer
    for k, nm in ((4, "task start"), (5, "inputs+conv"), (6, "state loaded"), (7, "task end")):
        v = pr[act, k]
        print(f"gdn {nm:13}: min {v.min():.1f} med {v.median():.1f} max {v.max():.1f} us (n {int(act.sum())})")
elif PSTAGE == "attn":
    pr = probes.double().cpu() / 100.0 - t0
    act = pr[:, 4] > 0
    names = ["start", "q/k prep", "QK", "softmax+PV", "record+atomic", "merge"]
    for k in range(4, 10):
        v = pr[act, k] if k < 9 else pr[pr[:, 9] > 0, 9]
        if len(v):
            print(f"attn {names[k - 4]:14} end: min {v.min():.1f} med {v.median():.1f} max {v.max():.1f} us (n {len(v)})")
elif PSTAGE:
    pr = probes.double().cpu() / 100.0 - t0
    b = max(i for i, n in enumerate(names) if n == PSTAGE)      # the probed (last) instance
    print(f"{PSTAGE} timeline (us from its barrier's last release): ")
    rel = p[b, :, 1].max().item()
    for nm, v in (("first release", p[b, :, 1].min()), ("prologue", pr[:, 5]), ("loads issued", pr[:, 6]),
                  ("rstd done", pr[:, 7]), ("quantized", pr[:, 8]), ("stream start", pr[:, 0]), ("stream end", pr[:, 1]),
                  ("hand-off end", pr[:, 2]), ("next arrival", p[b + 1, :, 0]), ("next release", p[b + 1, :, 1])):
        v = torch.as_tensor(v)
        print(f"  {nm:14} med {v.median().item() - rel:6.1f} max {v.max().item() - rel:6.1f}")
    for k, nm in ((0, "stream start"), (1, "stream end"), (2, "hand-off end")):
        v = pr[:, k]
        print(f"{PSTAGE} {nm:13}: min {v.min():.1f} med {v.median():.1f} max {v.max():.1f} us")
    d = pr[:, 1] - pr[:, 0]
    print(f"{PSTAGE} stream time per wave: min {d.min():.1f} med {d.median():.1f} max {d.max():.1f} us")
    d = pr[:, 2] - pr[:, 1]
    print(f"{PSTAGE} hand-off per wave: med {d.median():.1f} max {d.max():.1f} us")
    d = pr[:, 3] - pr[:, 1]
    print(f"{PSTAGE} drain + atomics per wave: med {d.median():.1f} max {d.max():.1f} us")
    d = pr[:, 2] - pr[:, 3]
    print(f"{PSTAGE} reductions per wave: med {d.median():.1f} max {d.max():.1f} us")
    if os.environ.get("DETAIL"):
        st = (pr[:, 1] - pr[:, 0]).reshape(CUS, 8)
        print("stream by XCD (bid%8) median:", [round(st[x::8].median().item(), 1) for x in range(8)])
        print("stream by XCD max:", [round(st[x::8].max().item(), 1) for x in range(8)])
        print("stream by wave median:", [round(st[:, w].median().item(), 1) for w in range(8)])
        print("start spread by XCD:", [round(pr[:, 0].reshape(CUS, 8)[x::8].max().item() - pr[:, 0].min().item(), 1) for x in range(8)])
        q = torch.quantile(st.flatten(), torch.tensor([0.1, 0.5, 0.9, 0.99], dtype=torch.float64))
        print("stream quantiles 10/50/90/99:", [round(v, 1) for v in q.tolist()])
        slow = (st > q[2]).nonzero()
        print("slowest 10% by bid/38:", torch.bincount(slow[:, 0] // 38, minlength=8).tolist(),
              "by xcd:", torch.bincount(slow[:, 0] % 8, minlength=8).tolist())
