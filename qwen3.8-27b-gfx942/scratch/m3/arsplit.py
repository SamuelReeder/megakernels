"""Disposable: all-reduce timing split (docs/DESIGN-TP2.md section 11). One TP rank of a
srun -n2 run (mk/tp.py: MK_TP=2 MK_AR=fp32|bf16 [MK_AR_LOCAL=1]): CTX warm-up steps, then
REPS stamped steps with dev.ARPROBE set (per-CTA barrier arrive / drained / released and
per-wave push / poll start / poll end of the all-reduce epilogues). Saves the raw i64
stamps (100 MHz clock) to OUT.r<rank>.pt for scratch/m3/arsplit_an.py.
Usage: python scratch/m3/arsplit.py MODEL_DIR M CTX REPS OUT
"""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/scratch/", 1)[0])
from mk import dev as D  # noqa: E402
from mk import kernel as KK  # noqa: E402
from mk import reference as R  # noqa: E402
from mk import tp as TP  # noqa: E402
from mk.dims import NL  # noqa: E402

path, M, CTX, REPS, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
ctx = TP.init()  # first: it selects this rank's GPU
CUS = torch.cuda.get_device_properties(0).multi_processor_count
NB = KK.barriers_per_step(NL) + 1
bar = torch.zeros(NB, CUS, 4, dtype=torch.int64, device="cuda")
wav = torch.zeros(NB, CUS * 8, 4, dtype=torch.int64, device="cuda")
D.ARPROBE.update(bar=bar.data_ptr(), wave=wav.data_ptr())

W = R.Weights(path, "mxfp4", keep_raw=True, dequant=False)
eng = TP.raw(TP.engine(ctx, TP.packed(ctx, W), M, maxlen=((CTX + REPS + 64) // 64 + 1) * 64))
del W
print(f"tp {ctx.tp} rank {ctx.rank} ar {ctx.ar_dtype}{' local' if ctx.ar_local else ''} cfg {eng.cfg}",
      flush=True)
tok = torch.zeros(M, dtype=torch.long, device="cuda")
for t in range(CTX):
    eng.step(tok)
torch.cuda.synchronize()
res = []
for rep in range(REPS):
    bar.zero_()
    wav.zero_()
    TP.agree(ctx, True)  # both ranks start the stamped step together
    t0 = time.perf_counter()
    eng.step(tok)
    torch.cuda.synchronize()
    res.append(dict(bar=bar.cpu().clone(), wave=wav.cpu().clone(), host_us=(time.perf_counter() - t0) * 1e6))
names = []
for i in range(NL):
    names += (["qkv", "attn", "o"] if i % 4 == 3 else ["inp", "gdn", "out"]) + ["gu", "down"]
names += ["lm"]
torch.save(dict(M=M, ctx=CTX, tp=ctx.tp, rank=ctx.rank, ar=ctx.ar_dtype, local=ctx.ar_local,
                cus=CUS, stages=names, reps=res), f"{out}.r{ctx.rank}.pt")
print(f"saved {REPS} stamped steps; step us (host) {[round(r['host_us']) for r in res]}", flush=True)
TP.agree(ctx, True)
