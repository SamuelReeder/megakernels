"""Does compile_key stop stale FlyDSL binaries? Qwen3.8 TP1 (common/DESIGN.md D5 hotfix).

Each run is a fresh process: 4 layers of the MXFP4 checkpoint, M = 1, 4 fixed-token steps,
sha256 of the logits (cache_model.run). Scenarios, cache on unless stated:
  S0 cache hit: build twice with the same sources and cfg; the second run loads the binary
  S1 edit mk/dev.py silu (x (1 + 2^-10)): base -> same cache -> cache off
  S2 cfg field: KV block 208 -> 64 (cfg.block and the strides, read as cfg.*)
  S3 GEMV chunks: mk/gemv.py CHUNKS lm=8 -> lm=4 (the packing follows)
  H  full model (64 layers), M = 1 and 8, 8 steps, cache off: the logits hash to compare
     before and after the change
Usage: python cache_model_q38.py SRC MXFP4_DIR WORKDIR [S0,S1,S2,S3,H]
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_model import EDIT, run, verdict  # noqa: E402

CHILD = r'''
import hashlib, os, sys
sys.path.insert(0, os.environ["CM_SRC"])
import torch
from mk import reference as R
from mk.pack import Packed
from mk.runner import Engine
L, M, S = (int(os.environ.get(k, d)) for k, d in (("CM_LAYERS", "4"), ("CM_M", "1"), ("CM_STEPS", "4")))
W = R.Weights(os.environ["CM_FP8"], "mxfp4", keep_raw=True, dequant=False, layers=L)
PK = Packed(W, layers=L)
eng = Engine(PK, M, maxlen=128, layers=L, block=int(os.environ.get("CM_BLOCK", "208")))
toks = torch.randint(0, 200000, (S, M), device="cuda", generator=torch.Generator("cuda").manual_seed(7))
h = hashlib.sha256()
for t in range(S):
    o = eng.step(toks[t]).clone()
    torch.cuda.synchronize()
    h.update(o.float().cpu().numpy().tobytes())
print("CFG", eng.cfg)
print("FINITE", bool(torch.isfinite(o.float()).all()))
print("HASH", h.hexdigest()[:16])
'''

CHUNKS = ("CHUNKS = dict(out=2, o=2, down=4, lm=8)\n", "CHUNKS = dict(out=2, o=2, down=4, lm=4)\n")


def main():
    src0, mx, work = sys.argv[1:4]
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else ["S0", "S1", "S2", "S3", "H"]
    os.makedirs(work, exist_ok=True)
    src = os.path.join(work, "q38")
    off = os.path.join(work, "cache_off")

    def fresh(name):
        shutil.rmtree(src, ignore_errors=True)
        shutil.copytree(src0, src)
        cache = os.path.join(work, "cache_" + name)
        shutil.rmtree(cache, ignore_errors=True)
        return cache

    def r(tag, cache, on, **env):
        return run(tag, src, mx, work, cache, on, child_src=CHILD,
                   env_extra={k: str(v) for k, v in env.items()})

    def edit(rel, old_new):
        p = os.path.join(src, rel)
        s = open(p).read()
        assert s.count(old_new[0]) == 1, rel
        open(p, "w").write(s.replace(*old_new))

    if "S0" in only:
        cache = fresh("s0")
        a = r("S0 first build", cache, True)
        b = r("S0 second build, same cache", cache, True)
        ok = a["hash"] == b["hash"] and a["new_cache_files"] == 1 and b["new_cache_files"] == 0
        print(f"VERDICT S0 cache hit: {'hit, same result' if ok else 'unexpected'} "
              f"({a['hash']} {a['new_cache_files']} / {b['hash']} {b['new_cache_files']})", flush=True)
    if "S1" in only:
        cache = fresh("s1")
        a = r("S1 base", cache, True)
        edit("mk/dev.py", EDIT)
        b = r("S1 edited silu, same cache", cache, True)
        c = r("S1 edited silu, cache off", off, False)
        verdict("S1 dev.py helper edit", a, b, c)
    if "S2" in only:
        cache = fresh("s2")
        a = r("S2 block 208", cache, True)
        b = r("S2 block 64, same cache", cache, True, CM_BLOCK=64)
        c = r("S2 block 64, cache off", off, False, CM_BLOCK=64)
        verdict("S2 cfg.block", a, b, c)
    if "S3" in only:
        cache = fresh("s3")
        a = r("S3 CHUNKS lm=8", cache, True)
        edit("mk/gemv.py", CHUNKS)
        b = r("S3 CHUNKS lm=4, same cache", cache, True)
        c = r("S3 CHUNKS lm=4, cache off", off, False)
        verdict("S3 GEMV chunks", a, b, c)
    if "H" in only:
        fresh("h")
        for m in (1, 8):
            r(f"H full model M={m}", off, False, CM_LAYERS=64, CM_M=m, CM_STEPS=8)


if __name__ == "__main__":
    main()
