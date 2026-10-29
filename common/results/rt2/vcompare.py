"""rt2 evidence: compare two bench/vllm_e2e.py results (greedy tokens, logprobs).
Usage: python common/results/rt2/vcompare.py A.json B.json
Prints per prompt whether tokens are equal (first divergence otherwise), whether the
logprobs are bitwise equal and the max |logprob difference| of the generated tokens, then
one RESULT line over all prompts. Reads Qwen3.8 ("logprobs": the generated tokens'
logprobs) and Qwen3.6 / Gemma results ("top": the top-5 logprobs per position)."""
import json
import sys


def lp_list(o):
    """(the generated tokens' logprobs, the full logprob record) of one output."""
    if "top" in o:
        return [d.get(str(t)) for t, d in zip(o["tokens"], o["top"])], o["top"]
    return o["logprobs"], o["logprobs"]


a, b = (json.load(open(p)) for p in sys.argv[1:3])
oa, ob = a["outputs"], b["outputs"]
assert len(oa) == len(ob), (len(oa), len(ob))
all_tok = all_lp = True
for i, (x, y) in enumerate(zip(oa, ob)):
    tx, ty = x["tokens"], y["tokens"]
    eq = tx == ty
    div = next((k for k, (p, q) in enumerate(zip(tx, ty)) if p != q), None)
    (la, ra), (lb, rb) = lp_list(x), lp_list(y)
    lp_eq = ra == rb
    num = [abs(p - q) for p, q in zip(la, lb) if p is not None and q is not None]
    print(f"prompt {i}: {len(tx)} tokens, equal {eq}" + ("" if eq else f" (first divergence {div})")
          + f", logprobs bitwise {lp_eq}" + (f", max |dlogprob| {max(num):.3g}" if num else ""))
    all_tok &= eq
    all_lp &= lp_eq
print(f"RESULT {sys.argv[1]} vs {sys.argv[2]}: tokens equal {all_tok}, logprobs bitwise equal "
      f"{all_lp}; TPOT A {a.get('tpot_ms')} B {b.get('tpot_ms')}; kernel steps A "
      f"{a.get('kernel_steps')} B {b.get('kernel_steps')}")
