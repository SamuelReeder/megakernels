"""Bit oracle of the common/ refactor (common/DESIGN.md oracle 3): sha256 of the fp32
logits of STEPS decode steps with fixed tokens (seed 7), per M. Any change of the kernel's
output bits changes the hash. The inputs are those of the compile-key hotfix runs
(common/results/cache/cache_model_q38.py, scenario H), so the M = 1 and 8 hashes compare
with the hashes recorded there. The Engine has no lm_head=False path (vLLM only); the
ISA oracle covers that build.
Usage: python tests/logits_hash.py MODEL_DIR [Ms=1,2,4,8] [STEPS=8] [LAYERS=64]
TP2: MK_TP=2 MK_AR=fp32|bf16 srun -n2 ... (mk/tp.py): every rank prints the hash of its
own vocab slice (before the gather) and of the gathered logits.
"""
import hashlib
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402
from mk import tp as TP  # noqa: E402


def main():
    path = sys.argv[1]
    Ms = [int(m) for m in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4,8").split(",")]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    L = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    ctx = TP.init()
    W = R.Weights(path, "mxfp4", keep_raw=True, dequant=False, layers=L)
    PK = TP.packed(ctx, W, layers=L)
    tag = f"tp{ctx.tp} ar {ctx.ar_dtype} rank {ctx.rank}" if ctx.tp > 1 else "tp1"
    for M in Ms:
        eng = TP.engine(ctx, PK, M, maxlen=128, layers=L)
        toks = torch.randint(0, 200000, (steps, M), device="cuda",
                             generator=torch.Generator("cuda").manual_seed(7))
        hl, hg = hashlib.sha256(), hashlib.sha256()
        fin = True
        for t in range(steps):
            o = eng.step(toks[t]).clone()
            torch.cuda.synchronize()
            hg.update(o.float().cpu().numpy().tobytes())
            hl.update(TP.raw(eng).logits.float().cpu().numpy().tobytes())
            fin &= bool(torch.isfinite(o).all())
        print(f"HASH {tag} M={M} lm=on steps={steps} layers={L} logits {hg.hexdigest()[:16]} "
              f"rank-slice {hl.hexdigest()[:16]} finite {fin}", flush=True)
        del eng
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
