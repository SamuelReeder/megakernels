"""Disposable: the 1-layer check. One decode step at LAYERS layers (several seeds, M rows):
relative error of the kernel's residual against the emulating reference in TP1's sum order
(split 1) and in TP2's order (each row-parallel GEMV as the ranks' K-slice partials added
in rank order; split = tp). MK_TP=2 srun -n2 python scratch/m3/dbg.py MODEL M LAYERS SEEDS"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/scratch/", 1)[0])
from mk import reference as R  # noqa: E402
from mk import tp as TP  # noqa: E402

path, M, LAYERS, SEEDS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ctx = TP.init()
W = R.Weights(path, "mxfp4", keep_raw=True, layers=LAYERS, dequant=ctx.main)
PK = TP.packed(ctx, W, layers=LAYERS)
for seed in range(SEEDS):
    eng = TP.engine(ctx, PK, M, maxlen=512, layers=LAYERS)
    e = eng.eng if ctx.tp > 1 else eng
    torch.manual_seed(seed)
    tok = torch.randint(0, 248320, (M,), device="cuda")
    logits = eng.step(tok)
    torch.cuda.synchronize()
    h = e.hb[0].clone()
    if ctx.main:
        out = []
        for split in sorted({1, ctx.tp}):
            dump = {}
            ref = R.step(W, tok, R.State.new(M, 512), layers=LAYERS, emulate=True, dump=dump,
                         ar_tp=split, ar_bf16=ctx.tp > 1 and ctx.ar_dtype == "bf16")
            hr = dump[LAYERS - 1]
            rel = ((h - hr).norm(dim=-1) / hr.norm(dim=-1))
            cos = torch.nn.functional.cosine_similarity(logits, ref, dim=-1)
            out.append(f"split {split}: residual rel err max {rel.max().item():.2e} "
                       f"median {rel.median().item():.2e} | logits cos min {cos.min().item():.8f}")
        print(f"tp {ctx.tp} ar {ctx.ar_dtype} seed {seed} M {M} tok {tok.tolist()} | " + " || ".join(out),
              flush=True)
    del eng
TP.agree(ctx, True)
