"""Slice C step 4: compare two bench/vllm_e2e.py results (greedy tokens, logprobs, TPOT).
Usage: python c4_compare.py A.json B.json [C.json ...]  (A = reference)"""
import json
import sys

runs = [(p, json.load(open(p))) for p in sys.argv[1:]]
ref_p, ref = runs[0]
for p, r in runs[1:]:
    tok_eq = [a["tokens"] == b["tokens"] for a, b in zip(ref["outputs"], r["outputs"])]
    lp_eq = [a["logprobs"] == b["logprobs"] for a, b in zip(ref["outputs"], r["outputs"])]
    first = []
    for a, b in zip(ref["outputs"], r["outputs"]):
        d = next((i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), None)
        first.append(d)
    print(f"{p} vs {ref_p}: prompts {len(tok_eq)} tokens equal {sum(tok_eq)}/{len(tok_eq)} "
          f"logprobs bitwise equal {sum(lp_eq)}/{len(lp_eq)} first token diff per prompt {first}")
print("TPOT ms per batch:")
for p, r in runs:
    print(f"  {p}: " + ", ".join(f"B{b} {v:.2f}" for b, v in r["tpot_ms"].items()))
