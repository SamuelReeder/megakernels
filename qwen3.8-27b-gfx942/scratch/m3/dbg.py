"""Disposable: one TP step at LAYERS layers; compare the residual with the reference and
print where it goes wrong. MK_TP=2 srun -n2 python scratch/m3/dbg.py MODEL M LAYERS"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/scratch/", 1)[0])
from mk import reference as R  # noqa: E402
from mk import tp as TP  # noqa: E402

path, M, LAYERS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ctx = TP.init()
W = R.Weights(path, "mxfp4", keep_raw=True, layers=LAYERS, dequant=ctx.main)
PK = TP.packed(ctx, W, layers=LAYERS)
eng = TP.engine(ctx, PK, M, maxlen=512, layers=LAYERS)
e = eng.eng if ctx.tp > 1 else eng
torch.manual_seed(0)
tok = torch.randint(0, 200000, (M,), device="cuda")
logits = eng.step(tok)
torch.cuda.synchronize()
h = e.hb[0].clone()
print(f"[r{ctx.rank}] tok {tok.tolist()} h norm {h.norm().item():.4f} nan {h.isnan().any().item()} "
      f"logits(local) norm {e.logits.norm().item():.4f} gathered {logits.norm().item():.4f} shape {tuple(logits.shape)}",
      flush=True)
print(f"[r{ctx.rank}] h1 norm {e.hb[1].norm().item():.4f} ssq {e.ssq.norm().item():.4f} y {e.y.norm().item():.4f} "
      f"y2 {e.y2.norm().item():.4f} peer nonzero {int((e.peer.bytes != 0).sum()) if e.peer else '-'}", flush=True)
if ctx.main:
    st = R.State.new(M, 512)
    dump = {}
    ref = R.step(W, tok, st, layers=LAYERS, emulate=True, dump=dump)
    hr = dump[LAYERS - 1]
    emb = W["model.language_model.embed_tokens.weight"][tok].float()
    print(f"ref h norm {hr.norm().item():.4f} | rel err {((h - hr).norm() / hr.norm()).item():.3e} | "
          f"logits cos {torch.nn.functional.cosine_similarity(logits, ref, dim=-1).tolist()}")
    print(f"embed norm {emb.norm().item():.4f}; kernel h1 (after mixer) vs ref: n/a")
TP.agree(ctx, True)
