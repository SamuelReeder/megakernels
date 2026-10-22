"""Decode throughput of the megakernel at batch size 1.

Times steady-state generation (prompt already in the cache) at several context lengths,
and reports tokens/s, ms/token and achieved memory bandwidth vs. the measured roofline.
"""

import argparse
import json
import time

import numpy as np
import torch

from qwen_mk.engine import Megakernel

ROOFLINE_GBPS = 915.0  # measured with ../common/bw.cu on an idle RTX 5080 (spec: 960)


def bench(mk, ctx, n_tokens, steps_per_launch, reps=3):
    """Returns (median GPU ms/step, best wall-clock tokens/s) for steady-state decoding.

    The median of per-step GPU timestamps is robust to other processes time-slicing the
    GPU; wall-clock includes launch gaps between chunks of `steps_per_launch` tokens.
    """
    steps, best_wall = [], 0.0
    for _ in range(reps):
        mk.reset()
        prompt = [(i * 7919) % 50000 + 1000 for i in range(ctx)]
        last = mk.prefill(prompt + [1000])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        done = 0
        while done < n_tokens:
            n = min(steps_per_launch, n_tokens - done)
            ids = mk.run([last], n_generate=n, timing=True)
            steps += mk.last_step_ns
            last = ids[-1]
            done += n
        torch.cuda.synchronize()
        best_wall = max(best_wall, n_tokens / (time.perf_counter() - t0))
    return float(np.median(steps)) / 1e6, best_wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[1, 512, 2048, 4096])
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--steps-per-launch", type=int, default=128)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--quant", choices=["bf16", "int8"], default="bf16")
    ap.add_argument("--pf-kb", type=int, nargs="+", default=None, help="sweep L2 prefetch distance (KB per SM)")
    args = ap.parse_args()

    mk = Megakernel(max_seq=max(args.ctx) + args.tokens + 8, quant=args.quant)
    if args.pf_kb:
        for kb in args.pf_kb:
            mk.pf_bytes = kb * 1024
            ms, wall = bench(mk, args.ctx[0], args.tokens, args.steps_per_launch)
            print(f"pf {kb:5d} KB/SM: {1e3 / ms:7.1f} tok/s (median step)  {wall:7.1f} tok/s (wall)")
        return
    body, head = mk.weight_bytes
    state = 18 * 2 * 16 * 128 * 128 * 4  # GDN state read + write
    rows = []
    for ctx in args.ctx:
        ms, wall = bench(mk, ctx, args.tokens, args.steps_per_launch)
        kv = 6 * 2 * 2 * 256 * 2 * (ctx + args.tokens // 2)
        gbps = (body + head + state + kv) / ms / 1e6
        rows.append(dict(ctx=ctx, ms_per_tok=ms, tok_s=1e3 / ms, wall_tok_s=wall, GBps=gbps,
                         pct_roofline=100 * gbps / ROOFLINE_GBPS))
        print(f"ctx {ctx:5d}: {1e3 / ms:7.1f} tok/s  {ms:6.3f} ms/tok  {gbps:6.1f} GB/s "
              f"({100 * gbps / ROOFLINE_GBPS:4.1f}% of {ROOFLINE_GBPS:.0f} GB/s)   wall-clock {wall:6.1f} tok/s")
    print(f"bytes/token: weights {body / 1e6:.0f} MB + LM head {head / 1e6:.0f} MB + GDN state {state / 1e6:.0f} MB")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(quant=args.quant, gpu=torch.cuda.get_device_name(), rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
