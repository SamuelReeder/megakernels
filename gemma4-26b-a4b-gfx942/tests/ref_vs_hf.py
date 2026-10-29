"""Check mk/reference.py against HF transformers on the BF16 checkpoint.

The prompt is longer than the sliding window (1024), so the window mask of the 25 sliding
layers is exercised. HF runs the whole prompt in one forward (eager attention) in fp32
(default) or bf16; the reference runs one decode step per token (fp32 residual, exact mode).
HF in bf16 rounds every activation to bf16 over 30 layers and 128-expert routing, so its
distance to the reference measures bf16 noise, not reference bugs; the pass rule uses fp32.
Usage: python tests/ref_vs_hf.py <bf16 model dir> [N_TOKENS=1300] [HF_DTYPE=fp32|bf16]
Pass (fp32): per-position logits cosine > 0.999 and top-1 agreement >= 90 %.
"""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402
from tests.texts import long_ids  # noqa: E402

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1300
HF_DTYPE = sys.argv[3] if len(sys.argv) > 3 else "fp32"
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration  # noqa: E402

tok = AutoTokenizer.from_pretrained(path)
ids = torch.tensor([long_ids(tok, N)], device="cuda")
L = ids.shape[1]

t0 = time.time()
hf = Gemma4ForConditionalGeneration.from_pretrained(
    path, dtype=torch.float32 if HF_DTYPE == "fp32" else torch.bfloat16, device_map="cuda",
    attn_implementation="eager",
).eval()
with torch.no_grad():
    out = hf(input_ids=ids, use_cache=False, output_hidden_states=True)
hf_logits = out.logits[0].float()  # [L, V]
PROBE = [0, 1, 2, 500, 1100, L - 1]  # positions whose per-layer h is compared
# hidden_states[i] is the input of layer i (i < 30); [30] is after the final norm
hf_h = torch.stack([hs[0, PROBE].float() for hs in out.hidden_states])  # [31, P, H]
del out
print(f"hf ({HF_DTYPE}) done {time.time() - t0:.0f}s, {L} tokens")
del hf
torch.cuda.empty_cache()

MODE = "fp32" if HF_DTYPE == "fp32" else "exact"
W = R.Weights(path, "bf16")
st = R.State.new(1, L + 8, dtype=torch.float32 if MODE == "fp32" else torch.bfloat16)
ours, dumps = [], {}
t0 = time.time()
for t in range(L):
    d = {} if t in PROBE else None
    ours.append(R.step(W, ids[0, t : t + 1], st, dump=d, mode=MODE)[0])
    if d is not None:
        dumps[t] = d
print(f"reference ({MODE} mode) done {time.time() - t0:.0f}s")
for k, t in enumerate(PROBE):
    rel = [((dumps[t][i]["h"][0] - hf_h[i + 1, k]).norm() / hf_h[i + 1, k].norm()).item()
           for i in range(R.NL - 1)]
    print(f"pos {t}: per-layer |h - h_hf| / |h_hf| " + " ".join(f"{x:.1e}" for x in rel))
ours = torch.stack(ours)
cos = torch.nn.functional.cosine_similarity(ours, hf_logits, dim=-1)
top = (ours.argmax(-1) == hf_logits.argmax(-1)).float()
for lo, hi in ((0, 1024), (1024, L)):
    print(f"positions {lo}..{hi - 1}: cos min {cos[lo:hi].min():.6f} mean {cos[lo:hi].mean():.6f}"
          f"  top1 agree {top[lo:hi].mean():.3f}")
print("cos min %.6f mean %.6f  top1 agree %.3f" % (cos.min(), cos.mean(), top.mean()))
print("hf next:", repr(tok.decode(hf_logits[-1].argmax())), " ours next:",
      repr(tok.decode(ours[-1].argmax())))
if HF_DTYPE == "bf16":
    print("REPORTED (HF bf16 is not gated: its distance is HF's own bf16 rounding)")
    sys.exit(0)
ok = cos.min() > 0.999 and top.mean() >= 0.9
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
