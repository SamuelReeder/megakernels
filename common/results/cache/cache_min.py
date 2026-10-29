"""Which changes reach the FlyDSL disk-cache key? A minimal kernel (common/DESIGN.md D5).

The kernel stores one int32 per thread: a sum of decimal digits, each from one source:
  A (x1)      helper in another file of the same directory, called as H.attr_helper (model: D.ld)
  O (x10)     helper in another directory, called as O.other_helper (planned: mkcore)
  D (x100)    helper in another file of the same directory, imported by name
  K (x1000)   int closure constant of the kernel
  C (x10000)  field of a frozen dataclass in the closure, read in the kernel (model: cfg.lm_head)
  N           the @flyc.kernel name only (no effect on the value)
  X           the A helper change together with a str closure constant of the launcher
              (the explicit compile key of common/DESIGN.md D5)
variant with the cache on (same dir), then the changed variant with the cache off. The cache
served a stale binary when run 2 prints the base value although run 3 prints another one.
Usage (needs a GPU and flydsl): python cache_min.py WORKDIR [MK_DIR]
MK_DIR: a model directory whose mk/vendor/buffer_ops.py the kernel uses for the store.
"""
import json
import os
import shutil
import subprocess
import sys

KERN = '''
import dataclasses
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int64
from mk.vendor import buffer_ops as bo
import helpers as H
import otherhelp as O
from helpers import direct_helper

_STREAM = fx.Stream(None)


@dataclasses.dataclass(frozen=True)
class Cfg:
    c: int


def build(name, K, cfg, key):
    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def k(out: Int64):
        tid = fx.thread_idx.x
        v = H.attr_helper(tid) + O.other_helper(tid) * 10 + direct_helper(tid) * 100
        v = v + K * 1000 + cfg.c * 10000
        bo.buffer_store(fx.Int32(v), bo.create_buffer_resource_from_addr(out), tid)

    @flyc.jit
    def launch(out: Int64, stream: fx.Stream = _STREAM):
        key  # the explicit compile key: a str closure value of the launcher
        k(out).launch(grid=(1,), block=(64,), stream=stream)

    return launch
'''

CHILD = '''
import json, os, sys
sys.path[:0] = [os.environ["CT_WORK"], os.path.join(os.environ["CT_WORK"], "other"), os.environ["CT_MK"]]
import torch
import kern
v = json.loads(os.environ["CT_VAR"])
out = torch.zeros(64, dtype=torch.int32, device="cuda")
kern.build(v["N"], v["K"], kern.Cfg(v["C"]), v["X"])(out.data_ptr())
torch.cuda.synchronize()
print("VALUE", int(out[0]))
'''

BASE = dict(A=1, O=1, D=1, K=1, C=1, N="ct_kernel_a", X="key1")
CASES = [
    ("A_attr_helper_same_dir", dict(A=2)),
    ("O_attr_helper_other_dir", dict(O=2)),
    ("D_direct_helper_same_dir", dict(D=2)),
    ("K_closure_int", dict(K=2)),
    ("C_closure_dataclass_field", dict(C=2)),
    ("N_kernel_name_only", dict(N="ct_kernel_b")),
    ("X_attr_helper_plus_str_key", dict(A=2, X="key2")),
]


def write_sources(work, var):
    os.makedirs(os.path.join(work, "other"), exist_ok=True)
    with open(os.path.join(work, "kern.py"), "w") as f:
        f.write(KERN)
    with open(os.path.join(work, "helpers.py"), "w") as f:
        f.write(f"def attr_helper(x):\n    return x * 0 + {var['A']}\n\n\n"
                f"def direct_helper(x):\n    return x * 0 + {var['D']}\n")
    with open(os.path.join(work, "other", "otherhelp.py"), "w") as f:
        f.write(f"def other_helper(x):\n    return x * 0 + {var['O']}\n")
    with open(os.path.join(work, "child.py"), "w") as f:
        f.write(CHILD)


def entries(cache):
    if not os.path.isdir(cache):
        return []
    return sorted(os.path.relpath(os.path.join(d, n), cache)
                  for d, _, ns in os.walk(cache) for n in ns if n.endswith(".pkl"))


def run(work, mk, var, cache, on):
    write_sources(work, var)
    env = dict(os.environ, CT_WORK=work, CT_MK=mk, CT_VAR=json.dumps(var),
               FLYDSL_RUNTIME_ENABLE_CACHE="1" if on else "0", FLYDSL_RUNTIME_CACHE_DIR=cache)
    env.pop("PYTHONPYCACHEPREFIX", None)
    shutil.rmtree(os.path.join(work, "__pycache__"), ignore_errors=True)
    shutil.rmtree(os.path.join(work, "other", "__pycache__"), ignore_errors=True)
    p = subprocess.run([sys.executable, "-B", os.path.join(work, "child.py")], env=env,
                       capture_output=True, text=True, timeout=600)
    val = [line.split()[1] for line in p.stdout.splitlines() if line.startswith("VALUE")]
    if p.returncode or not val:
        print(p.stdout[-2000:], p.stderr[-4000:], flush=True)
        raise SystemExit(f"child failed: {var}")
    return int(val[0])


def main():
    work = sys.argv[1]
    mk = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    res = []
    for name, change in CASES:
        cache = os.path.join(work, "cache_" + name)
        shutil.rmtree(cache, ignore_errors=True)
        var = dict(BASE, **change)
        v1 = run(work, mk, BASE, cache, True)
        e1 = entries(cache)
        v2 = run(work, mk, var, cache, True)
        e2 = entries(cache)
        v3 = run(work, mk, var, os.path.join(work, "cache_off"), False)
        new_entry = len(e2) > len(e1)
        if name.startswith("N_"):
            verdict = "key changes" if new_entry else "key unchanged (name not in key)"
        else:
            verdict = ("STALE: cache served the old binary" if v2 == v1 != v3
                       else "key changes" if v2 == v3 and new_entry else "unclear")
        r = dict(case=name, change=change, base=v1, changed_cache_on=v2, changed_cache_off=v3,
                 cache_files_after_base=len(e1), cache_files_after_changed=len(e2),
                 verdict=verdict)
        res.append(r)
        print(json.dumps(r), flush=True)
    print("SUMMARY")
    for r in res:
        print(f"{r['case']:28s} base {r['base']:6d} changed(cache on) {r['changed_cache_on']:6d} "
              f"changed(cache off) {r['changed_cache_off']:6d} new cache file "
              f"{r['cache_files_after_changed'] > r['cache_files_after_base']}: {r['verdict']}")


if __name__ == "__main__":
    main()
