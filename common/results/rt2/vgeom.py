"""rt2 evidence, Qwen3.8 TP1: kernel.Cfg.rt_geom 0 vs 2 at vLLM's geometry (KV block 784,
x-packed KV as ROCM_ATTN keeps it, maxlen 4096 and 8192), bit for bit. Per M and variant
one Engine; STEPS steps of random tokens (seed 7) through the megakernel from position 0,
so the attention crosses KV blocks (784 positions, 13 chunks a block, the last one 16
positions long). Per step a checksum of the fp32 logits' bits (int64 sum of bits * index)
on the device, plus a sha256 of the logits every 256 steps and at the end. The Engine has
no lm_head=False path; the LM head is not touched by rt_geom.
Usage (qwen3.8-27b-gfx942/): python ../common/results/rt2/vgeom.py MODEL_DIR [Ms=1,2,4,8]
  [STEPS=1800] [MAXLENS=4096,8192]
Prints per (M, maxlen, variant) one HASH line and per (M, maxlen) one EQUAL line."""
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from mk import reference as R  # noqa: E402
from mk import tp as TP  # noqa: E402
from mk.runner import Engine  # noqa: E402

path = sys.argv[1]
Ms = [int(m) for m in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4,8").split(",")]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1800
LENS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "4096,8192").split(",")]
ctx = TP.init()
W = R.Weights(path, "mxfp4", keep_raw=True, dequant=False)
PK = TP.packed(ctx, W)


def run(M, L, v, toks):
    e = Engine(PK, M, maxlen=L, block=784, xpack=True, rt_geom=v)
    e.reset()
    n = e.logits.numel()
    idx = torch.arange(1, n + 1, device="cuda", dtype=torch.int64)
    sums, shas = [], []
    for t in range(STEPS):
        o = e.step(toks[t])
        sums.append((o.reshape(-1).view(torch.int32).to(torch.int64) * idx).sum())
        if t % 256 == 255 or t == STEPS - 1:
            torch.cuda.synchronize()
            shas.append(hashlib.sha256(o.float().cpu().numpy().tobytes()).hexdigest()[:16])
    torch.cuda.synchronize()
    sums = torch.stack(sums).cpu().tolist()
    h = hashlib.sha256(str(sums).encode()).hexdigest()[:16]
    cfg = e.cfg
    del e
    torch.cuda.empty_cache()
    return h, shas, sums, cfg


for M in Ms:
    toks = torch.randint(0, 200000, (STEPS, M), device="cuda",
                         generator=torch.Generator("cuda").manual_seed(7))
    for L in LENS:
        res = {}
        for v in (0, 2):
            h, shas, sums, cfg = run(M, L, v, toks)
            res[v] = (h, shas, sums)
            print(f"HASH M={M} maxlen={L} rt_geom={v} steps={STEPS} stepsums {h} "
                  f"sha256/256steps {' '.join(shas)} cfg {cfg}", flush=True)
        s0, s2 = res[0][2], res[2][2]
        first = next((k for k, (a, b) in enumerate(zip(s0, s2)) if a != b), None)
        print(f"EQUAL M={M} maxlen={L}: rt0 == rt2 {res[0][:2] == res[2][:2]}"
              + ("" if first is None else f" (first differing step {first})"), flush=True)
