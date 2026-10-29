"""Decode benchmark of the megakernel: TPOT at batch M after a prompt of CTX tokens.

The prompt is fed through the megakernel itself (one decode step per token), then
STEPS greedy decode steps run back to back (argmax and token feedback on device).
TPOT = wall time of the decode steps / STEPS (CUDA events), and the kernel-only time
of single launches, and TPOT with the tokens held fixed (back-to-back launches, no
argmax). Usage: python bench/bench_decode.py MODEL_DIR M CTX STEPS [REPEATS]
TP2: MK_TP=2 MK_AR=fp32|bf16 srun -n2 ... (mk/tp.py): the greedy token is the argmax over
both ranks' vocab slices (one RCCL all-gather of (max, index) per step, TP.next_token).
"""
import json
import platform
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from mk import reference as R  # noqa: E402
from mk import tp as TP  # noqa: E402

path, M, CTX, STEPS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
REPEATS = int(sys.argv[5]) if len(sys.argv) > 5 else 3


def main():
    t0 = time.time()
    ctx = TP.init()
    W = R.Weights(path, "mxfp4", keep_raw=True, dequant=False)
    PK = TP.packed(ctx, W)
    del W
    teng = TP.engine(ctx, PK, M, maxlen=((CTX + STEPS + 63) // 64 + 1) * 64)
    eng = TP.raw(teng)
    print(f"load+pack+build {time.time() - t0:.0f}s tp {ctx.tp} ar {ctx.ar_dtype}"
          f"{' local' if ctx.ar_local else ''} cfg {eng.cfg}", flush=True)

    def sync():
        torch.cuda.synchronize()
        TP.agree(ctx, True)  # a gloo barrier

    g = torch.Generator(device="cuda").manual_seed(1234)
    prompt = torch.randint(0, 200000, (CTX, M), device="cuda", generator=g)
    res = []
    for rep in range(REPEATS):
        eng.reset()
        for t in range(CTX):
            logits = eng.step(prompt[t])
        tok = TP.next_token(ctx, logits)
        sync()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for s in range(STEPS):
            logits = eng.step(tok)
            tok = TP.next_token(ctx, logits)
        e1.record()
        torch.cuda.synchronize()
        tpot = e0.elapsed_time(e1) / STEPS
        # fixed tokens: the launches alone, back to back
        sync()
        f0, f1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        f0.record()
        for s in range(STEPS):
            eng.step(tok)
        f1.record()
        torch.cuda.synchronize()
        tpot_fixed = f0.elapsed_time(f1) / STEPS
        # kernel-only: one launch between events, 20 samples
        ks = []
        for s in range(20):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            sync()
            a.record()
            eng.step(tok)
            b.record()
            torch.cuda.synchronize()
            ks.append(a.elapsed_time(b))
        ks.sort()
        res.append(dict(tpot_ms=tpot, tpot_fixed_ms=tpot_fixed, kernel_ms_median=ks[len(ks) // 2],
                        kernel_ms_min=ks[0]))
        print(f"rep {rep}: TPOT {tpot:.3f} ms ({1000 / tpot:.1f} tok/s/user), fixed-token TPOT "
              f"{tpot_fixed:.3f} ms, kernel median {ks[len(ks)//2]:.3f} ms", flush=True)
    out = dict(M=M, ctx=CTX, steps=STEPS, tp=ctx.tp, rank=ctx.rank, ar=ctx.ar_dtype,
               ar_local=ctx.ar_local, node=platform.node(), gpu=torch.cuda.get_device_name(0),
               date=time.strftime("%Y-%m-%d %H:%M:%S"), runs=res)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
