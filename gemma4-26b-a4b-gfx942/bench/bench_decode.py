"""Decode benchmark of the megakernel: TPOT at batch M after a prompt of CTX tokens.
From qwen3.6-35b-a3b-gfx942/bench/bench_decode.py (megakernels e2a6fb8).

The prompt is fed through the megakernel itself (one decode step per token), then
STEPS greedy decode steps run back to back (argmax and token feedback on device).
TPOT = wall time of the decode steps / STEPS (CUDA events), and the kernel-only time
of single launches. Usage: python bench/bench_decode.py FP8_DIR M CTX STEPS [REPEATS]
"""
import json
import platform
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from mk import reference as R  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine, default_chunks  # noqa: E402

path, M, CTX, STEPS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
REPEATS = int(sys.argv[5]) if len(sys.argv) > 5 else 3


def main():
    t0 = time.time()
    W = R.Weights(path, "fp8")
    PK = Packed(W, default_chunks())
    del W
    torch.cuda.empty_cache()
    eng = Engine(PK, M, maxlen=((CTX + STEPS + 63) // 64 + 1) * 64)
    print(f"load+pack+build {time.time() - t0:.0f}s", flush=True)
    g = torch.Generator(device="cuda").manual_seed(1234)
    prompt = torch.randint(0, 262144, (CTX, M), device="cuda", generator=g)
    res = []
    for rep in range(REPEATS):
        eng.reset()
        for t in range(CTX):
            logits = eng.step(prompt[t])
            if t % 256 == 255:  # more than about 1000 queued steps crash the host
                torch.cuda.synchronize()  # process (docs/DESIGN.md "Open questions")
        tok = logits.argmax(-1)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for s in range(STEPS):
            logits = eng.step(tok)
            tok = logits.argmax(-1)
        e1.record()
        torch.cuda.synchronize()
        tpot = e0.elapsed_time(e1) / STEPS
        ks = []
        for s in range(20):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            eng.step(tok)
            b.record()
            torch.cuda.synchronize()
            ks.append(a.elapsed_time(b))
        ks.sort()
        res.append(dict(tpot_ms=tpot, kernel_ms_median=ks[len(ks) // 2], kernel_ms_min=ks[0]))
        print(f"rep {rep}: TPOT {tpot:.3f} ms ({1000 / tpot:.1f} tok/s/user, {M * 1000 / tpot:.1f} "
              f"tok/s total), kernel median {ks[len(ks) // 2]:.3f} ms", flush=True)
    out = dict(M=M, ctx=CTX, steps=STEPS, node=platform.node(), gpu=torch.cuda.get_device_name(0),
               date=time.strftime("%Y-%m-%d %H:%M:%S"), cfg=repr(eng.cfg), runs=res)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
