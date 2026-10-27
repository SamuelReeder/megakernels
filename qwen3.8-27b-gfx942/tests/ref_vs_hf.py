"""Check mk/reference.py against HF transformers on the real BF16 checkpoint.

Usage: python tests/ref_vs_hf.py <bf16 model dir>
Pass: per-position logits cosine > 0.999 and top-1 agreement >= 90 %.
"""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402

path = sys.argv[1]
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration  # noqa: E402

tok = AutoTokenizer.from_pretrained(path)
text = "The quick brown fox jumps over the lazy dog. In 1969, the first humans landed on the"
ids = tok(text, return_tensors="pt").input_ids.cuda()
L = ids.shape[1]

t0 = time.time()
hf = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
with torch.no_grad():
    hf_logits = hf(input_ids=ids, use_cache=False).logits[0].float()  # [L, V]
print(f"hf done {time.time()-t0:.0f}s")
del hf
torch.cuda.empty_cache()

W = R.Weights(path, "bf16")
st = R.State.new(1, 256)
ours = []
for t in range(L):
    ours.append(R.step(W, ids[0, t: t + 1], st)[0])
ours = torch.stack(ours)
cos = torch.nn.functional.cosine_similarity(ours, hf_logits, dim=-1)
top = (ours.argmax(-1) == hf_logits.argmax(-1)).float()
print("cos min %.6f mean %.6f  top1 agree %.3f" % (cos.min(), cos.mean(), top.mean()))
print("hf next:", tok.decode(hf_logits[-1].argmax()), " ours next:", tok.decode(ours[-1].argmax()))
ok = cos.min() > 0.999 and top.mean() >= 0.9
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
