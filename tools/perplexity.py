"""Perplexity on WikiText-2 (test) for the megakernel (bf16 / int8) and HF transformers.

    python tools/perplexity.py --backends mk-bf16 mk-int8 hf-bf16 --windows 16 --ctx 1024

The text is cut into non-overlapping windows of --ctx tokens; each window starts from a
fresh state and every token after the first is scored.
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from qwen_mk import weights

WIKI = ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet")


def wikitext_ids(tok):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(WIKI[0], WIKI[1], repo_type="dataset")
    text = "".join(pq.read_table(path).column("text").to_pylist())
    return tok(text, add_special_tokens=False).input_ids


def nll_megakernel(quant, windows):
    from qwen_mk.engine import Megakernel

    mk = Megakernel(max_seq=max(len(w) for w in windows) + 8, quant=quant)
    total, n = 0.0, 0
    for w in windows:
        mk.reset()
        for i in range(0, len(w) - 1, 256):  # bounded launches (GPU watchdog) and logits memory
            chunk = w[i: min(i + 256, len(w) - 1)]
            _, lg = mk.run(chunk, all_logits=True)
            tgt = torch.tensor(w[i + 1: i + 1 + len(chunk)], device=lg.device)
            total += F.cross_entropy(lg.float(), tgt, reduction="sum").item()
            n += len(chunk)
    return total, n


@torch.no_grad()
def nll_hf(windows):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(weights.snapshot_path(), dtype=torch.bfloat16).cuda().eval()
    total, n = 0.0, 0
    for w in windows:
        ids = torch.tensor([w], device="cuda")
        lg = m(ids).logits[0, :-1].float()
        total += F.cross_entropy(lg, ids[0, 1:], reduction="sum").item()
        n += len(w) - 1
    return total, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["mk-bf16", "mk-int8", "hf-bf16"])
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    ids = wikitext_ids(AutoTokenizer.from_pretrained(weights.snapshot_path()))
    windows = [ids[i * args.ctx: (i + 1) * args.ctx] for i in range(args.windows)]
    results = {}
    for b in args.backends:
        total, n = nll_hf(windows) if b == "hf-bf16" else nll_megakernel(b.split("-")[1], windows)
        results[b] = dict(ppl=math.exp(total / n), tokens=n)
        print(f"{b:8s}: perplexity {results[b]['ppl']:.4f} over {n} tokens")
        torch.cuda.empty_cache()
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(dict(dataset="wikitext-2-raw-v1 test", ctx=args.ctx, windows=args.windows, results=results), f,
                      indent=1)


if __name__ == "__main__":
    main()
