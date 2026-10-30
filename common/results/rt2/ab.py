"""rt2 evidence: decode TPOT of kernel.Cfg.rt_geom 0 and 2 on one geometry, in one
process, interleaved (from spike/rtgeom/bench_ab.py on users/sareeder/runtime-geom). Per
variant one Engine; per round, in rotated variant order: reset, a prompt of CTX tokens
through the megakernel, then STEPS greedy steps (argmax and token feedback on the device)
-> TPOT; then the kernel-only median of 20 single launches. Checks: every variant's
prompt logits and greedy tokens equal variant 0's bitwise.
Usage (model dir): python ../common/results/rt2/ab.py WEIGHTS M CTX STEPS ROUNDS [OUT.json]
The variant is selected the model's way: Qwen3.8 MK_RTGEOM, Qwen3.6 and Gemma MK_CFG.
maxlen: bench/bench_decode.py's ((CTX + STEPS + 63) // 64 + 1) * 64.
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

MODEL = os.path.basename(os.getcwd())
path, M, CTX, STEPS, ROUNDS = sys.argv[1], *(int(a) for a in sys.argv[2:6])
OUT = sys.argv[6] if len(sys.argv) > 6 else None
VARIANTS = (0, 2)
MAXLEN = ((CTX + STEPS + 63) // 64 + 1) * 64


def load():
    if MODEL.startswith("qwen3.8"):
        from mk import tp as TP

        ctx = TP.init()
        W = R.Weights(path, "mxfp4", keep_raw=True, dequant=False)
        PK = TP.packed(ctx, W)
        prompt = torch.randint(0, 200000, (CTX, M), device="cuda",
                               generator=torch.Generator(device="cuda").manual_seed(1234))

        def engine(v):
            from mk.runner import Engine

            return Engine(PK, M, maxlen=MAXLEN, rt_geom=v)

        return engine, prompt
    from mk.pack import Packed
    from mk.runner import Engine, default_chunks

    if MODEL.startswith("gemma4"):
        from tests.texts import text_prompts

        W = R.Weights(path, "fp8")
        prompt = text_prompts(path, M, CTX)
    else:
        W = R.Weights(path, keep_raw=True)
        prompt = torch.randint(0, 200000, (CTX, M), device="cuda",
                               generator=torch.Generator(device="cuda").manual_seed(1234))
    PK = Packed(W, default_chunks())

    def engine(v):
        os.environ["MK_CFG"] = f"rt_geom={v}"
        e = Engine(PK, M, maxlen=MAXLEN)
        del os.environ["MK_CFG"]
        return e

    return engine, prompt


def ev():
    return torch.cuda.Event(enable_timing=True)


def run_prompt(e, prompt):
    e.reset()
    for t in range(CTX):
        logits = e.step(prompt[t])
        if t % 256 == 255:  # Gemma: more than about 1000 queued steps crash the host process
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return logits.clone()


def main():
    t0 = time.time()
    engine, prompt = load()
    torch.cuda.empty_cache()
    engs = {v: engine(v) for v in VARIANTS}
    for v, e in engs.items():
        print(f"variant rt_geom={v}: cfg {e.cfg}", flush=True)
    print(f"load+pack+build {time.time() - t0:.0f}s", flush=True)
    v0 = VARIANTS[0]
    checks = {v: dict(prompt_logits_equal=True, tokens_equal=True) for v in VARIANTS}
    ref_logits = ref_toks = None
    res = {v: [] for v in VARIANTS}
    for r in range(ROUNDS):
        order = VARIANTS[r % len(VARIANTS):] + VARIANTS[: r % len(VARIANTS)]
        for v in order:
            e = engs[v]
            logits = run_prompt(e, prompt)
            if ref_logits is None:
                ref_logits = logits
            checks[v]["prompt_logits_equal"] &= torch.equal(logits, ref_logits)
            tok = logits.argmax(-1)
            toks = []
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
            ks = []
            for s in range(20):
                a, b = ev(), ev()
                a.record()
                e.step(tok)
                b.record()
                torch.cuda.synchronize()
                ks.append(a.elapsed_time(b))
            ks.sort()
            res[v].append(dict(tpot_ms=tpot, kernel_ms_median=ks[10]))
            print(f"round {r} rt_geom={v}: TPOT {tpot:.3f} ms, kernel median {ks[10]:.3f} ms",
                  flush=True)
    summ = {}
    for v in VARIANTS:
        s = {}
        for k in ("tpot_ms", "kernel_ms_median"):
            xs = [x[k] for x in res[v]]
            s[k] = dict(median=st.median(xs), min=min(xs), max=max(xs))
        summ[v] = s
        d = (s["tpot_ms"]["median"] / summ[v0]["tpot_ms"]["median"] - 1) * 100
        print(f"SUMMARY {MODEL} M={M} ctx={CTX} maxlen={MAXLEN} rt_geom={v}: TPOT median "
              f"{s['tpot_ms']['median']:.3f} ms [{s['tpot_ms']['min']:.3f}, "
              f"{s['tpot_ms']['max']:.3f}] ({d:+.2f}% vs rt_geom={v0}), kernel "
              f"{s['kernel_ms_median']['median']:.3f}; prompt logits equal "
              f"{checks[v]['prompt_logits_equal']}, greedy tokens equal "
              f"{checks[v]['tokens_equal']}", flush=True)
    out = dict(model=MODEL, M=M, ctx=CTX, steps=STEPS, rounds=ROUNDS, maxlen=MAXLEN,
               node=platform.node(), gpu=torch.cuda.get_device_name(0),
               date=time.strftime("%Y-%m-%d %H:%M:%S"),
               cfgs={str(v): repr(e.cfg) for v, e in engs.items()},
               runs={str(v): res[v] for v in VARIANTS}, summary={str(v): summ[v] for v in VARIANTS},
               checks={str(v): checks[v] for v in VARIANTS})
    if OUT:
        json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
