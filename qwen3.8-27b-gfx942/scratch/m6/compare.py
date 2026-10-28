"""Disposable: kernel-on against kernel-off greedy outputs of bench/vllm_e2e.py.
Per prompt: tokens equal, first divergent position, max |logprob difference| over the
common prefix; then the TPOT of both runs. Usage: python scratch/m6/compare.py ON.json OFF.json
"""
import json
import sys

on, off = (json.load(open(p)) for p in sys.argv[1:3])
same = 0
for i, (a, b) in enumerate(zip(on["outputs"], off["outputs"])):
    ta, tb = a["tokens"], b["tokens"]
    n = next((k for k, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    eq = ta == tb
    same += eq
    dl = max((abs(x - y) for x, y in zip(a["logprobs"][:n], b["logprobs"][:n])), default=0.0)
    print(f"prompt {i}: tokens equal {eq}, first divergence {n if not eq else '-'} "
          f"of {len(ta)}, max |dlogprob| over common prefix {dl:.4f}")
print(f"{same}/{len(on['outputs'])} prompts token-identical")
for B in sorted(on["tpot_ms"], key=int):
    t1, t0 = on["tpot_ms"][B], off["tpot_ms"][B]
    print(f"batch {B}: TPOT on {t1:.2f} ms, off {t0:.2f} ms, speedup {t0 / t1:.2f}x")
