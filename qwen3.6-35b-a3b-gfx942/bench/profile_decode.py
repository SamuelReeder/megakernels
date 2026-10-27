"""Per-stage time profile of one megakernel decode step (thread 0 stamps per CTA).

Stage b runs from the release of barrier b to the arrival at barrier b + 1. Reports, per
stage kind, the median over layers of: span (first release -> last arrival), the mean
per-CTA busy time, and the barrier cost (last arrival -> last release).
PROBE=<stage> (gdn, attn, or a GEMV name: inp, out, qkv, o, router, lm) adds per-wave
stamps of the last layer of that kind.
Usage: [PROBE=...] python bench/profile_decode.py FP8_DIR M CTX [TIMELINE.json]
"""
import json
import os
import statistics as stt
import sys

import torch

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from mk import dev as D  # noqa: E402
from mk import kernel as KK  # noqa: E402
from mk import reference as R  # noqa: E402
from mk.dims import NL, is_attn  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine, default_chunks  # noqa: E402

path, M, CTX = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
CUS = torch.cuda.get_device_properties(0).multi_processor_count
NB = KK.barriers_per_step(NL) + 1
prof = torch.zeros(NB, CUS, 2, dtype=torch.int64, device="cuda")
KK.PROFILE["addr"] = prof.data_ptr()
PSTAGE = os.environ.get("PROBE")
probes = torch.zeros(CUS * 8, 16, dtype=torch.int64, device="cuda")
if PSTAGE:
    D.PROBE.update(addr=probes.data_ptr(), stage=PSTAGE, waves=8)

W = R.Weights(path, keep_raw=True)
PK = Packed(W, default_chunks())
del W
torch.cuda.empty_cache()
eng = Engine(PK, M, maxlen=((CTX + 64) // 64 + 1) * 64)
g = torch.Generator(device="cuda").manual_seed(1234)
for t in range(CTX):  # distinct random prompts per row, then one greedy step
    logits = eng.step(torch.randint(0, 200000, (M,), device="cuda", generator=g))
tok = logits.argmax(-1)
torch.cuda.synchronize()
prof.zero_()
probes.zero_()
ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
ev[0].record()
eng.step(tok)
ev[1].record()
torch.cuda.synchronize()
p = prof.double().cpu() / 100.0  # us (100 MHz clock)
t0 = p[0, :, 0].min()
p -= t0
names = []
for i in range(NL):
    names += (["qkv", "attn", "o"] if is_attn(i) else ["inp", "gdn", "out"]) + ["router", "e1", "e2"]
names += ["lm"]
assert len(names) == NB - 1
rows = {}
for b in range(NB - 1):
    rel, arr = p[b, :, 1], p[b + 1, :, 0]
    span = (arr.max() - rel.min()).item()
    busy = (arr - rel).mean().item()
    bar = (p[b, :, 1].max() - p[b, :, 0].max()).item()
    rows.setdefault(names[b], []).append((span, busy, bar))
if len(sys.argv) > 4:  # (arrive, release) per barrier and CTA for tools/timeline.py
    json.dump({"M": M, "ctx": CTX, "gpu": torch.cuda.get_device_name(0), "cus": CUS, "stages": names,
               "us": [[[round(v, 2) for v in cta] for cta in bar] for bar in p.tolist()]},
              open(sys.argv[4], "w"))
total = p[NB - 1, :, 0].max().item()
print(f"M={M} ctx={CTX}: embed (start -> first arrival) {p[0, :, 0].max().item():.1f} us; "
      f"step total {total:.1f} us (CUDA events around this single launch, launch latency "
      f"included: {ev[0].elapsed_time(ev[1]) * 1000:.1f} us)")
print(f"{'stage':6} {'n':>3} {'span us':>8} {'busy us':>8} {'barrier us':>10} {'sum span us':>11}")
acc = 0.0
for k, v in rows.items():
    acc += sum(r[0] + r[2] for r in v)
    print(f"{k:6} {len(v):3d} {stt.median(r[0] for r in v):8.1f} {stt.median(r[1] for r in v):8.1f} "
          f"{stt.median(r[2] for r in v):10.2f} {sum(r[0] for r in v):11.1f}")
print(f"sum of spans + barriers {acc:.1f} us")
pr = probes.double().cpu() / 100.0 - t0
if PSTAGE == "gdn":
    act = pr[:, 4] > pr[:, 4].max() - 200  # waves that ran a task in the last GDN layer
    for k, nm in ((4, "task start"), (5, "inputs+conv"), (6, "state loaded"), (7, "task end")):
        v = pr[act, k]
        print(f"gdn {nm:13}: min {v.min():.1f} med {v.median():.1f} max {v.max():.1f} us "
              f"(n {int(act.sum())})")
elif PSTAGE == "attn":
    act = pr[:, 4] > 0
    b = max(i for i, n in enumerate(names) if n == "attn")
    rel = p[b, :, 1].max().item()
    print(f"attn (us from the last release of its barrier; n tasks {int(act.sum())})")
    labels = ["start", "q/k prep", "QK", "softmax+PV", "record+atomic", "merge"]
    for k in range(4, 10):
        v = pr[act, k] if k < 9 else pr[pr[:, 9] > 0, 9]
        if len(v):
            print(f"attn {labels[k - 4]:14} end: min {v.min() - rel:.1f} med {v.median() - rel:.1f} "
                  f"max {v.max() - rel:.1f} us (n {len(v)})")
elif PSTAGE:
    b = max(i for i, n in enumerate(names) if n == PSTAGE)  # the probed (last) instance
    rel = p[b, :, 1].max().item()
    print(f"{PSTAGE} timeline (us from its barrier's last release):")
    for nm, v in (("first release", p[b, :, 1].min()), ("prologue", pr[:, 5]),
                  ("loads issued", pr[:, 6]), ("rstd done", pr[:, 7]), ("quantized", pr[:, 8]),
                  ("stream start", pr[:, 0]), ("stream end", pr[:, 1]), ("hand-off end", pr[:, 2]),
                  ("next arrival", p[b + 1, :, 0]), ("next release", p[b + 1, :, 1])):
        v = torch.as_tensor(v)
        print(f"  {nm:14} med {v.median().item() - rel:6.1f} max {v.max().item() - rel:6.1f}")
    for nm, d in (("stream", pr[:, 1] - pr[:, 0]), ("hand-off", pr[:, 2] - pr[:, 1]),
                  ("drain + atomics", pr[:, 3] - pr[:, 1]), ("reductions", pr[:, 2] - pr[:, 3])):
        print(f"{PSTAGE} {nm:15} per wave: med {d.median():.1f} max {d.max():.1f} us")
