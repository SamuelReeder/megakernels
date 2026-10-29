"""Can the FlyDSL disk cache serve a stale Qwen3.6 kernel? (common/DESIGN.md D5)

Runs qwen3.6-35b-a3b-gfx942 at M = 1 on a 4-layer slice of the FP8 checkpoint for 4 steps
of fixed tokens, each run in a fresh process, and prints a sha256 of the outputs.
Scenarios (one cache directory each, cache on unless stated):
  S1 dev.py helper edit: base -> edit mk/dev.py silu (x (1 + 2^-10), changes the bits)
     -> same cache (on) -> cache off (the true result of the edit)
  S2 lm_head: lm_head=True -> lm_head=False same cache -> lm_head=False cache off
  S3 GEMV chunks: default -> MK_CHUNKS=lm=8 (Cfg.c_lm, the packing follows) same cache
     -> cache off
Usage (GPU node, venv with flydsl):
  python cache_model.py MODEL_SRC_DIR FP8_DIR WORKDIR [S1,S2,S3]
A run that does not finish in CM_TIMEOUT seconds (default 600) is killed and reported as
rc "timeout" (a stale persistent kernel can spin forever at a barrier).
"""
import json
import os
import shutil
import subprocess
import sys

CHILD = r'''
import hashlib, os, sys
sys.path.insert(0, os.environ["CM_SRC"])
import torch
from mk import reference as R
from mk.pack import Packed
from mk.runner import Engine, default_chunks
L = 4
W = R.Weights(os.environ["CM_FP8"], keep_raw=True, layers=L)
PK = Packed(W, default_chunks(), layers=L)
eng = Engine(PK, 1, maxlen=128, layers=L, lm_head=os.environ["CM_LM"] == "1")
gen = torch.Generator("cuda").manual_seed(7)
toks = torch.randint(0, 200000, (4, 1), device="cuda", generator=gen)
h = hashlib.sha256()
for t in range(4):
    o = eng.step(toks[t]).clone()
    torch.cuda.synchronize()
    h.update(o.float().cpu().numpy().tobytes())
print("CFG", eng.cfg)
print("FINITE", bool(torch.isfinite(o.float()).all()))
print("HASH", h.hexdigest()[:16])
'''

EDIT = ("    return fx.Float32(x) * sigmoid(x)\n",
        "    return fx.Float32(x) * sigmoid(x) * fx.Float32(1.0009765625)\n")


def entries(cache):
    if not os.path.isdir(cache):
        return 0
    return sum(n.endswith(".pkl") for _, _, ns in os.walk(cache) for n in ns)


def run(tag, src, fp8, work, cache, on, lm="1", chunks="", child_src=CHILD, env_extra=None):
    env = dict(os.environ, CM_SRC=src, CM_FP8=fp8, CM_LM=lm, MK_CHUNKS=chunks,
               FLYDSL_RUNTIME_ENABLE_CACHE="1" if on else "0",
               FLYDSL_RUNTIME_CACHE_DIR=cache, **(env_extra or {}))
    env.pop("PYTHONPYCACHEPREFIX", None)
    shutil.rmtree(os.path.join(src, "mk", "__pycache__"), ignore_errors=True)
    child = os.path.join(work, "child.py")
    with open(child, "w") as f:
        f.write(child_src)
    n0 = entries(cache)
    try:
        p = subprocess.run([sys.executable, "-B", child], env=env, capture_output=True,
                           text=True, timeout=int(os.environ.get("CM_TIMEOUT", "600")))
        rc, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc = "timeout"
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
    out = {k: v for k, _, v in (line.partition(" ") for line in stdout.splitlines())
           if k in ("HASH", "FINITE", "CFG")}
    r = dict(run=tag, cache_on=on, lm_head=lm, chunks=chunks, rc=rc,
             hash=out.get("HASH"), finite=out.get("FINITE"), new_cache_files=entries(cache) - n0)
    if rc or "HASH" not in out:
        r["tail"] = (stdout[-1500:] + stderr[-3000:])
    print(json.dumps(r), flush=True)
    return r


def verdict(name, a, b, c):
    """b (changed, cache on) must equal c (changed, cache off): the kernel is deterministic."""
    if b["rc"] and not c["rc"]:
        v = "STALE: the cached run failed, the uncached run passed"
    elif b["hash"] != c["hash"]:
        v = ("STALE: the cache served the old binary" if b["hash"] == a["hash"]
             else "STALE: the cached run gives a wrong result")
    elif b["new_cache_files"] == 0:
        v = "correct result, but no new cache file (unclear)"
    else:
        v = "key changes (new binary compiled, correct result)"
    print(f"VERDICT {name}: {v} (base {a['hash']}, changed+cache {b['hash']}, "
          f"changed no cache {c['hash']})", flush=True)


def main():
    src0, fp8, work = sys.argv[1:4]
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else ["S1", "S2", "S3"]
    os.makedirs(work, exist_ok=True)
    src = os.path.join(work, "q36")
    off = os.path.join(work, "cache_off")
    shutil.rmtree(src, ignore_errors=True)
    shutil.copytree(src0, src)

    if "S1" in only:  # a dev.py helper edit
        cache = os.path.join(work, "cache_s1")
        shutil.rmtree(cache, ignore_errors=True)
        a = run("S1 base", src, fp8, work, cache, True)
        dev = os.path.join(src, "mk", "dev.py")
        s = open(dev).read()
        assert s.count(EDIT[0]) == 1
        open(dev, "w").write(s.replace(*EDIT))
        b = run("S1 edited silu, same cache", src, fp8, work, cache, True)
        c = run("S1 edited silu, cache off", src, fp8, work, off, False)
        verdict("S1 dev.py helper edit", a, b, c)
        shutil.rmtree(src)
        shutil.copytree(src0, src)

    if "S2" in only:
        cache = os.path.join(work, "cache_s2")
        shutil.rmtree(cache, ignore_errors=True)
        a = run("S2 lm_head=True", src, fp8, work, cache, True, lm="1")
        b = run("S2 lm_head=False, same cache", src, fp8, work, cache, True, lm="0")
        c = run("S2 lm_head=False, cache off", src, fp8, work, off, False, lm="0")
        verdict("S2 lm_head (Cfg field read as cfg.lm_head)", a, b, c)

    if "S3" in only:
        cache = os.path.join(work, "cache_s3")
        shutil.rmtree(cache, ignore_errors=True)
        a = run("S3 default chunks", src, fp8, work, cache, True)
        b = run("S3 MK_CHUNKS=lm=8, same cache", src, fp8, work, cache, True, chunks="lm=8")
        c = run("S3 MK_CHUNKS=lm=8, cache off", src, fp8, work, off, False, chunks="lm=8")
        verdict("S3 GEMV K chunks (Cfg.c_lm)", a, b, c)


if __name__ == "__main__":
    main()
