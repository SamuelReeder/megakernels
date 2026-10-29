"""HF transformers in bf16 against HF in fp32 on the ref_vs_hf prompt: the size of the bf16
model's own rounding effect, to read tests/ref_vs_hf.py's bf16 numbers against.
Usage: python tests/hf_bf16_vs_fp32.py <bf16 model dir> [N_TOKENS=1300]
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from tests.texts import long_ids  # noqa: E402

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1300
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration  # noqa: E402

ids = torch.tensor([long_ids(AutoTokenizer.from_pretrained(path), N)], device="cuda")
logits = {}
for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
    hf = Gemma4ForConditionalGeneration.from_pretrained(
        path, dtype=dt, device_map="cuda", attn_implementation="eager").eval()
    with torch.no_grad():
        logits[name] = hf(input_ids=ids, use_cache=False).logits[0].float().cpu()
    del hf
    torch.cuda.empty_cache()
a, b = logits["bf16"], logits["fp32"]
cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
top = (a.argmax(-1) == b.argmax(-1)).float()
print(f"HF bf16 vs HF fp32, {ids.shape[1]} tokens: cos min {cos.min():.6f} mean {cos.mean():.6f}"
      f"  top1 agree {top.mean():.3f}")
