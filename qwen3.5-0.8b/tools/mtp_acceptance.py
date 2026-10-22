"""How often does the MTP head's draft match what the model actually generates next?

Generates greedy chat replies with the megakernel, recomputes the main model's hidden states
for each full sequence with Hugging Face, runs the MTP head over every position, and counts how
often its guess for x_{i+2} equals the generated token. For greedy speculative decoding with one
draft token, that fraction is the acceptance rate.

`--draft-vocab` also scores drafts restricted to the K most frequent tokens (the kernel's reduced
draft head; the ranking comes from tools/draft_vocab.py).

    python tools/mtp_acceptance.py --tokens 256 --draft-vocab 0 16384 32768 65536
"""

import argparse
import json

import numpy as np
import torch

from qwen_mk import weights
from qwen_mk.engine import DRAFT_VOCAB_FILE, Megakernel
from qwen_mk.mtp import MTP, load_mtp

PROMPTS = [
    "Explain how a CPU cache works to a high-school student.",
    "Write a Python function that checks whether a string is a palindrome, with tests.",
    "What are the main causes of the French Revolution?",
    "Write a short poem about autumn in a city.",
    "Summarize the plot of Romeo and Juliet in five sentences.",
    "How do I reverse a linked list in C? Show the code.",
    "Give me a three-day itinerary for Tokyo.",
    "What is the difference between TCP and UDP?",
    "Translate into French: 'The weather is lovely today, let's go for a walk in the park.'",
    "Write a SQL query that finds the top five customers by total order value.",
    "Explain gradient descent with a simple example.",
    "Write a cover letter opening paragraph for a software engineering internship.",
    "What is 17 * 23? Show your reasoning step by step.",
    "Describe the water cycle.",
    "Write a bash script that renames every .txt file in a folder to .md.",
    "What are some tips for improving sleep quality?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--quant", choices=["bf16", "int8"], default="int8")
    ap.add_argument("--draft-vocab", type=int, nargs="+", default=[0], help="0 = full vocabulary")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = weights.snapshot_path()
    tok = AutoTokenizer.from_pretrained(path)
    stops = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    mk = Megakernel(path, max_seq=4096, quant=args.quant, mtp=False)
    subsets = {k: torch.from_numpy(np.sort(np.load(DRAFT_VOCAB_FILE)[:k])).long().cuda()
               for k in args.draft_vocab if k}
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    mtp = MTP(mk.cfg, load_mtp(path), mk.embed, max_seq=4096)

    hits = {k: 0 for k in args.draft_vocab}
    total = 0
    per_prompt = []
    for prompt in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, add_special_tokens=False).input_ids
        mk.reset()
        gen = list(mk.generate(ids, max_new_tokens=args.tokens, stop_ids=stops))
        seq = ids + gen
        with torch.no_grad():
            out = hf(torch.tensor([seq], device="cuda"), output_hidden_states=True, logits_to_keep=1)
        h = out.hidden_states[-1][0].float()  # post final norm
        # sanity: the last hidden state reproduces HF's logits
        assert torch.allclose((h[-1:] @ mk.embed.float().t()), out.logits[0, -1:].float(), atol=0.5, rtol=0.05)
        # MTP at position i sees (h_i, x_{i+1}) and guesses x_{i+2}
        logits = mtp.forward(h[:-1], torch.tensor(seq[1:], device="cuda"), 0)
        first = len(ids) - 1  # h_{first} predicts the first generated token
        target = torch.tensor(seq[first + 2:], device="cuda")
        rates = []
        for k in args.draft_vocab:
            lg = logits[first: len(seq) - 2]
            guess = subsets[k][lg[:, subsets[k]].argmax(-1)] if k else lg.argmax(-1)
            n_hit = int((guess == target).sum())
            hits[k] += n_hit
            rates.append(n_hit / max(1, len(target)))
        total += len(target)
        per_prompt.append(rates)
        print("  ".join(f"{r:5.2f}" for r in rates) + f"  ({len(gen):3d} tokens)  {prompt[:60]}")
    for k in args.draft_vocab:
        print(f"acceptance (1 draft token, greedy, draft vocab {k or mk.cfg.vocab}): {hits[k] / total:.3f} "
              f"over {total} positions")
    if args.json:
        json.dump(dict(draft_vocab=args.draft_vocab, acceptance=[hits[k] / total for k in args.draft_vocab],
                       positions=total, per_prompt=per_prompt), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
