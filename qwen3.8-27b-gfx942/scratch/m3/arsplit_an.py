"""Disposable: split the all-reduce cost from scratch/m3/arsplit.py stamps.

Usage: python scratch/m3/arsplit_an.py DIR M   (DIR holds ar_{fp32,bf16}{,_local}_m{M}.r{0,1}.pt)

Clock alignment: each GPU has its own 100 MHz counter. The offset d (rank 1 clock - rank 0
clock) and the one-way latency L come from the all-reduce itself (NTP style): x = min over
groups of (poll end on rank 0 - push on rank 1) = L - d, y = min of (poll end on rank 1 -
push on rank 0) = L + d. Symmetric latency is assumed.

Per all-reduce stage (out / o / down) and rank, with times on rank 0's clock:
  start  first CTA release of the barrier that starts the stage
  ready  the rank's last push (its partial of every group is out)
  done   the rank's last poll end (every group summed)
  arrive last CTA arrival at the next barrier (before the drain)
  drain  last CTA "drained" (after s_waitcnt vmcnt(0): every store acknowledged)
  end    last CTA release of the next barrier
segments: gemv = ready - start, wait = max(ready over ranks) - ready (waiting for the
slower GPU), xfer = done - max(ready) (transport + poll of the last partial), tail =
arrive - done, drain = drain - arrive, bar = end - drain. ar_local runs have no peer, so
wait = 0 and xfer = done - ready.
"""
import statistics as st
import sys

import torch

D, M = sys.argv[1], int(sys.argv[2])
AR_STAGES = ("out", "o", "down")
SEGS = ("gemv", "wait", "xfer", "tail", "drain", "bar")


def load(fmt, local):
    tag = f"ar_{fmt}{'_local' if local else ''}_m{M}"
    return [torch.load(f"{D}/{tag}.r{r}.pt") for r in (0, 1)]


def offset(rk):
    """(d, L) in clock ticks from one stamped step of both ranks [(bar, wave)] * 2."""
    x = y = None
    names = rk[0]["names"]
    for b, nm in enumerate(names):
        if nm not in AR_STAGES:
            continue
        w0, w1 = rk[0]["wave"][b], rk[1]["wave"][b]
        m0, m1 = w0[:, 0] > 0, w1[:, 0] > 0
        g0 = {int(g): (int(p), int(e)) for p, _, e, g in w0[m0].tolist()}
        g1 = {int(g): (int(p), int(e)) for p, _, e, g in w1[m1].tolist()}
        for g in g0.keys() & g1.keys():
            a = g0[g][1] - g1[g][0]
            c = g1[g][1] - g0[g][0]
            x = a if x is None else min(x, a)
            y = c if y is None else min(y, c)
    return (y - x) / 2, (x + y) / 2  # x = L - d, y = L + d


def stage_segs(rk, coupled):
    """{stage: {seg: [us per layer, per rank]}} for one stamped step."""
    d = offset(rk)[0] if coupled else 0.0
    names = rk[0]["names"]
    out = {s: {k: [] for k in SEGS} for s in AR_STAGES}
    per = {s: {k: [] for k in ("push_ps", "ps_pe", "drain_cta_med")} for s in AR_STAGES}
    for b, nm in enumerate(names):
        if nm not in AR_STAGES:
            continue
        t = []
        for r in (0, 1):
            sh = -d if r == 1 else 0.0  # rank 1's clock -> rank 0's
            bar, wv = rk[r]["bar"].double(), rk[r]["wave"].double()
            wm = wv[b][wv[b][:, 0] > 0]
            t.append(dict(
                start=bar[b, :, 2].min().item() + sh,
                ready=wm[:, 0].max().item() + sh,
                done=wm[:, 2].max().item() + sh,
                arrive=bar[b + 1, :, 0].max().item() + sh,
                drain=bar[b + 1, :, 1].max().item() + sh,
                end=bar[b + 1, :, 2].max().item() + sh,
            ))
            per[nm]["push_ps"] += ((wm[:, 1] - wm[:, 0]) / 100).tolist()
            per[nm]["ps_pe"] += ((wm[:, 2] - wm[:, 1]) / 100).tolist()
            per[nm]["drain_cta_med"].append(((bar[b + 1, :, 1] - bar[b + 1, :, 0]).median() / 100).item())
        rmax = max(x["ready"] for x in t)
        for x in t:
            r0 = rmax if coupled else x["ready"]
            seg = dict(gemv=x["ready"] - x["start"], wait=r0 - x["ready"], xfer=x["done"] - r0,
                       tail=x["arrive"] - x["done"], drain=x["drain"] - x["arrive"], bar=x["end"] - x["drain"])
            for k in SEGS:
                out[nm][k].append(seg[k] / 100)
    return out, per


def step_us(rk):
    """Whole stamped step per rank: first release of barrier 0 to last release of the last."""
    return [((x["bar"][-1, :, 2].max() - x["bar"][0, :, 2].min()) / 100).item() for x in rk]


def run(fmt, local):
    ranks = load(fmt, local)
    reps = len(ranks[0]["reps"])
    acc = {s: {k: [] for k in SEGS} for s in AR_STAGES}
    per = {s: {k: [] for k in ("push_ps", "ps_pe", "drain_cta_med")} for s in AR_STAGES}
    steps, offs = [], []
    for i in range(reps):
        rk = [dict(names=x["stages"], **x["reps"][i]) for x in ranks]
        o, p = stage_segs(rk, not local)
        if not local:
            offs.append(offset(rk))
        for s in AR_STAGES:
            for k in SEGS:
                acc[s][k] += o[s][k]
            for k in p[s]:
                per[s][k] += p[s][k]
        steps.append(step_us(rk))
    return acc, per, steps, offs


def med(v):
    return st.median(v) if v else float("nan")


for fmt in ("fp32", "bf16"):
    a, pa, sa, offs = run(fmt, False)
    l, pl, sl, _ = run(fmt, True)
    print(f"== {fmt} M={M}: stamped step us (rank 0 / 1, per rep): with AR {[[round(v) for v in s] for s in sa]}, "
          f"ar_local {[[round(v) for v in s] for s in sl]}")
    print(f"   clock offset rank1-rank0 per rep (ms): {[round(d / 1e5, 3) for d, _ in offs]}; "
          f"one-way latency L (us): {[round(L / 100, 2) for _, L in offs]}")
    tot_a = tot_l = 0.0
    for s in AR_STAGES:
        ra = {k: med(a[s][k]) for k in SEGS}
        rl = {k: med(l[s][k]) for k in SEGS}
        print(f"   {s:5s} " + " ".join(f"{k} {ra[k]:6.2f}/{rl[k]:6.2f}" for k in SEGS)
              + f" | sum {sum(ra.values()):6.2f}/{sum(rl.values()):6.2f} us (AR/local, median over layers x ranks x reps)")
        print(f"         delta " + " ".join(f"{k} {ra[k] - rl[k]:+5.2f}" for k in SEGS)
              + f" | push->poll start {med(pa[s]['push_ps']):.2f}/{med(pl[s]['push_ps']):.2f}"
              + f", poll {med(pa[s]['ps_pe']):.2f} (p90 {sorted(pa[s]['ps_pe'])[int(0.9 * len(pa[s]['ps_pe']))]:.2f})"
              + f", per-CTA drain median {med(pa[s]['drain_cta_med']):.2f}/{med(pl[s]['drain_cta_med']):.2f} us")
        n = len(a[s]["gemv"]) // (2 * len(sa))  # layers of this kind
        tot_a += sum(med(a[s][k]) for k in SEGS) * n
        tot_l += sum(med(l[s][k]) for k in SEGS) * n
    print(f"   all-reduce stages: (sum of medians x layers) AR {tot_a:.0f} us, local {tot_l:.0f} us, "
          f"added {tot_a - tot_l:.0f} us = {(tot_a - tot_l) / 128:.2f} us per all-reduce")
