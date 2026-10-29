"""Interleaved speed A/B of kernel configurations in one process: one Engine per Cfg flag
set (MK_CFG syntax), the same real-text prompt fed to each, then ROUNDS rounds in which
every engine decodes STEPS greedy steps back to back (CUDA events, as
bench/bench_decode.py). Interleaving cancels the drift of the GPU's speed between runs
(measured on a shared node: the same kernel 6.72 ms and 7.73 ms TPOT three hours apart).
Prints the median TPOT per configuration and its ratio to the first.
Usage: python bench/ab_speed.py FP8_DIR M "FLAGS_A" "FLAGS_B" [...]
Env: AB_ROUNDS (default 5), AB_STEPS (default 64), AB_CTX (default 512).
"""
import os
import platform
import statistics as stt
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from mk import reference as R  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine, default_chunks  # noqa: E402
from tests.texts import text_prompts  # noqa: E402

path, M, flags = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
ROUNDS = int(os.environ.get("AB_ROUNDS", "5"))
STEPS = int(os.environ.get("AB_STEPS", "64"))
CTX = int(os.environ.get("AB_CTX", "512"))


def main():
    print(f"node {platform.node()} {time.strftime('%Y-%m-%d %H:%M:%S')} M={M} ctx={CTX} "
          f"rounds={ROUNDS} steps={STEPS}", flush=True)
    PK = Packed(R.Weights(path, "fp8"), default_chunks())
    torch.cuda.empty_cache()
    prompt = text_prompts(path, M, CTX)
    maxlen = ((CTX + ROUNDS * STEPS + 63) // 64 + 1) * 64
    engs, toks = [], []
    for f in flags:
        os.environ["MK_CFG"] = f
        e = Engine(PK, M, maxlen=maxlen)
        for t in range(CTX):
            logits = e.step(prompt[t])
            if t % 256 == 255:
                torch.cuda.synchronize()
        toks.append(logits.argmax(-1))
        engs.append(e)
        print(f"{f or '(defaults)'}: {e.cfg}", flush=True)
    torch.cuda.synchronize()
    res = [[] for _ in flags]
    for r in range(ROUNDS):
        for i, e in enumerate(engs):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            tok = toks[i]
            a.record()
            for s in range(STEPS):
                tok = e.step(tok).argmax(-1)
            b.record()
            torch.cuda.synchronize()
            toks[i] = tok
            res[i].append(a.elapsed_time(b) / STEPS)
    base = stt.median(res[0])
    for f, v in zip(flags, res):
        med = stt.median(v)
        print(f"{f or '(defaults)':60} TPOT median {med:.3f} ms  ratio {med / base:.4f}  "
              f"rounds {' '.join(f'{x:.3f}' for x in v)}")


if __name__ == "__main__":
    main()
