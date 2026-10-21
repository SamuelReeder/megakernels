"""Decode throughput of the megakernel at batch size 1.

Times steady-state generation (prompt already in the cache) at several context lengths,
and reports tokens/s, ms/token and achieved memory bandwidth vs. the measured roofline.
"""

import argparse
import json
import time

import torch

from qwen_mk.engine import Megakernel

ROOFLINE_GBPS = 780.0  # measured with bench/bw.cu on an RTX 5080


def bench(mk, ctx, n_tokens, steps_per_launch, reps=3):
    best = None
    for _ in range(reps):
        mk.reset()
        prompt = [(i * 7919) % 50000 + 1000 for i in range(ctx)]
        mk.run(prompt, n_generate=1)
        torch.cuda.synchronize()
        last = 1000
        t0 = time.perf_counter()
        done = 0
        while done < n_tokens:
            n = min(steps_per_launch, n_tokens - done)
            ids = mk.run([last], n_generate=n)
            last = ids[-1]
            done += n
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return n_tokens / best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[1, 512, 2048, 4096])
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--steps-per-launch", type=int, default=128)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--pf-kb", type=int, nargs="+", default=None, help="sweep L2 prefetch distance (KB per SM)")
    args = ap.parse_args()

    mk = Megakernel(max_seq=max(args.ctx) + args.tokens + 8)
    if args.pf_kb:
        for kb in args.pf_kb:
            mk.pf_bytes = kb * 1024
            tps = bench(mk, args.ctx[0], args.tokens, args.steps_per_launch)
            print(f"pf {kb:5d} KB/SM: {tps:7.1f} tok/s")
        return
    body, head = mk.weight_bytes
    state = 18 * 2 * 16 * 128 * 128 * 4  # GDN state read + write
    rows = []
    for ctx in args.ctx:
        tps = bench(mk, ctx, args.tokens, args.steps_per_launch)
        kv = 6 * 2 * 2 * 256 * 2 * (ctx + args.tokens // 2)
        gbps = (body + head + state + kv) * tps / 1e9
        rows.append(dict(ctx=ctx, tok_s=tps, ms_per_tok=1e3 / tps, GBps=gbps, pct_roofline=100 * gbps / ROOFLINE_GBPS))
        print(f"ctx {ctx:5d}: {tps:7.1f} tok/s  {1e3 / tps:6.3f} ms/tok  {gbps:6.1f} GB/s  "
              f"({100 * gbps / ROOFLINE_GBPS:4.1f}% of {ROOFLINE_GBPS:.0f} GB/s)")
    print(f"bytes/token: weights {body / 1e6:.0f} MB + LM head {head / 1e6:.0f} MB + GDN state {state / 1e6:.0f} MB")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
