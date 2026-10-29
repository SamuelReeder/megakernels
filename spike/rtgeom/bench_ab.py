"""Spike rtgeom (disposable): decode TPOT of kernel variants (kernel.Cfg.rt_geom levels) on
one geometry, in one process, interleaved. Method of bench/bench_decode.py: a prompt of CTX
random tokens (seed 1234) fed through the megakernel, then STEPS greedy steps (argmax and
token feedback on the device) -> TPOT; fixed-token TPOT (STEPS back-to-back launches);
kernel-only median of 20 single launches between events. Each variant feeds the prompt
once; its caches (KV, GDN states, positions) are saved and restored before each of the
three timed parts, so every part starts at position CTX. Round r runs the variants in a
rotated order. Checks: every variant's prompt logits and greedy tokens equal variant 0's
bitwise.
Usage (model dir): python ../spike/rtgeom/bench_ab.py MODEL M CTX STEPS ROUNDS MAXLEN VARIANTS [OUT.json]
  VARIANTS: comma list of rt_geom levels, e.g. 0,1,2. Env MK_XPACK=1 MK_BLOCK=784: the
  x-packed KV layout (vLLM ROCM_ATTN) instead of the Engine default (block 208, plain).
"""
import json
import os
import platform
import statistics as st
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from mk import reference as R  # noqa: E402
from mk import tp as TP  # noqa: E402
from mk.runner import Engine  # noqa: E402

path, M, CTX, STEPS, ROUNDS, MAXLEN = sys.argv[1], *(int(a) for a in sys.argv[2:7])
VARIANTS = [int(v) for v in sys.argv[7].split(",")]
OUT = sys.argv[8] if len(sys.argv) > 8 else None
XPACK = os.environ.get("MK_XPACK", "0") == "1"
BLOCK = int(os.environ.get("MK_BLOCK", "208"))
assert CTX + STEPS <= MAXLEN


def ev():
    return torch.cuda.Event(enable_timing=True)


def main():
    t0 = time.time()
    ctx = TP.init()
    assert ctx.tp == 1
    W = R.Weights(path, "mxfp4", keep_raw=True, dequant=False)
    PK = TP.packed(ctx, W)
    del W
    g = torch.Generator(device="cuda").manual_seed(1234)
    prompt = torch.randint(0, 200000, (CTX, M), device="cuda", generator=g)
    engs, snaps, tok0, plog = {}, {}, {}, {}
    for v in VARIANTS:
        e = Engine(PK, M, maxlen=MAXLEN, block=BLOCK, xpack=XPACK, rt_geom=v)
        t1 = time.time()
        for t in range(CTX):
            logits = e.step(prompt[t])
        torch.cuda.synchronize()
        print(f"variant rt_geom={v}: cfg {e.cfg} compiled as {e.cfg.compiled()}; prompt "
              f"{time.time() - t1:.0f}s", flush=True)
        plog[v] = logits.clone()
        tok0[v] = logits.argmax(-1)
        snaps[v] = [x.clone() for x in (e.kv, e.state, e.pos)]
        engs[v] = e
    print(f"load+pack+build+prompts {time.time() - t0:.0f}s", flush=True)
    v0 = VARIANTS[0]
    checks = {v: dict(prompt_logits_equal=torch.equal(plog[v], plog[v0]), tokens_equal=True)
              for v in VARIANTS}
    ref_toks = None

    def restore(v):
        for x, s in zip((engs[v].kv, engs[v].state, engs[v].pos), snaps[v]):
            x.copy_(s)

    res = {v: [] for v in VARIANTS}
    for r in range(ROUNDS):
        order = VARIANTS[r % len(VARIANTS):] + VARIANTS[: r % len(VARIANTS)]
        for v in order:
            e = engs[v]
            restore(v)
            tok = tok0[v].clone()
            toks = []
            torch.cuda.synchronize()
            a, b = ev(), ev()
            a.record()
            for s in range(STEPS):
                tok = e.step(tok).argmax(-1)
                toks.append(tok)
            b.record()
            torch.cuda.synchronize()
            tpot = a.elapsed_time(b) / STEPS
            toks = torch.stack(toks)
            if ref_toks is None:
                ref_toks = toks
            checks[v]["tokens_equal"] &= torch.equal(toks, ref_toks)
            restore(v)
            torch.cuda.synchronize()
            a, b = ev(), ev()
            a.record()
            for s in range(STEPS):
                e.step(tok)
            b.record()
            torch.cuda.synchronize()
            fixed = a.elapsed_time(b) / STEPS
            restore(v)
            ks = []
            for s in range(20):
                a, b = ev(), ev()
                torch.cuda.synchronize()
                a.record()
                e.step(tok)
                b.record()
                torch.cuda.synchronize()
                ks.append(a.elapsed_time(b))
            ks.sort()
            res[v].append(dict(tpot_ms=tpot, tpot_fixed_ms=fixed, kernel_ms_median=ks[10]))
            print(f"round {r} rt_geom={v}: TPOT {tpot:.3f} ms, fixed-token TPOT {fixed:.3f} ms, "
                  f"kernel median {ks[10]:.3f} ms", flush=True)
    summ = {}
    for v in VARIANTS:
        s = {}
        for k in ("tpot_ms", "tpot_fixed_ms", "kernel_ms_median"):
            xs = [x[k] for x in res[v]]
            s[k] = dict(median=st.median(xs), min=min(xs), max=max(xs))
        summ[v] = s
        d = (s["tpot_ms"]["median"] / summ[v0]["tpot_ms"]["median"] - 1) * 100
        print(f"SUMMARY M={M} ctx={CTX} maxlen={MAXLEN} block={BLOCK} xpack={int(XPACK)} "
              f"rt_geom={v}: TPOT median {s['tpot_ms']['median']:.3f} ms "
              f"[{s['tpot_ms']['min']:.3f}, {s['tpot_ms']['max']:.3f}] ({d:+.2f}% vs rt_geom={v0}), "
              f"fixed {s['tpot_fixed_ms']['median']:.3f}, kernel {s['kernel_ms_median']['median']:.3f}; "
              f"prompt logits equal {checks[v]['prompt_logits_equal']}, greedy tokens equal "
              f"{checks[v]['tokens_equal']}", flush=True)
    out = dict(M=M, ctx=CTX, steps=STEPS, rounds=ROUNDS, maxlen=MAXLEN, block=BLOCK, xpack=XPACK,
               node=platform.node(), gpu=torch.cuda.get_device_name(0),
               date=time.strftime("%Y-%m-%d %H:%M:%S"), runs={str(v): res[v] for v in VARIANTS},
               summary={str(v): summ[v] for v in VARIANTS},
               checks={str(v): checks[v] for v in VARIANTS})
    if OUT:
        json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
