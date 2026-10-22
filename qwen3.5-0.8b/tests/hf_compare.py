"""Compare the reference decoder against Hugging Face transformers (fp32)."""
import sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_mk import weights
from qwen_mk.reference import ReferenceModel

path = weights.snapshot_path()
tok = AutoTokenizer.from_pretrained(path)
ids = tok("The quick brown fox jumps over the lazy dog. In 1969, Neil Armstrong", return_tensors="pt").input_ids.cuda()
hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).cuda().eval()
print(type(hf).__name__)
with torch.no_grad():
    hf_logits = hf(ids).logits[0].float()
cfg, w = weights.load()
ref = ReferenceModel(cfg, w, max_seq=256)
errs = []
for t in range(ids.shape[1]):
    lg = ref.step(ids[0, t].item())
    errs.append(((lg - hf_logits[t]).abs().max().item(), lg.argmax().item() == hf_logits[t].argmax().item()))
for t, (e, ok) in enumerate(errs):
    print(t, f"{e:.2e}", ok)
