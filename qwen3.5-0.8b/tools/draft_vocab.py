"""Rank tokens by how often the model itself generates them, for the reduced draft head.

The MTP head's draft is an argmax over the vocabulary, and a full-vocabulary draft head reads as
many bytes as the LM head. Restricting it to the tokens the model actually tends to produce costs
little acceptance and most of that time. The ranking is self-distilled: the model writes a user
request from a generic seed (topic x kind of task x language), then answers it in a fresh chat,
and only the replies are counted. No dataset is needed and the benchmark prompts play no part.
Tokens never generated keep their id order after the counted ones.

    python tools/draft_vocab.py --conversations 600 --out qwen_mk/draft_vocab.npy
"""

import argparse
import collections
import random
import time

import numpy as np

from qwen_mk import weights
from qwen_mk.engine import Megakernel

TOPICS = ["Python", "JavaScript", "C++", "Rust", "SQL databases", "Linux", "web development", "machine learning",
          "statistics", "algebra", "geometry", "calculus", "physics", "chemistry", "biology", "astronomy",
          "medicine", "nutrition", "fitness", "psychology", "economics", "personal finance", "marketing",
          "starting a business", "law", "world history", "ancient civilizations", "geography", "travel", "cooking",
          "gardening", "music", "movies", "literature", "poetry", "philosophy", "religion", "education", "parenting",
          "relationships", "careers and job interviews", "productivity", "climate and the environment", "energy",
          "cars", "sports", "video games", "art and design", "photography", "language learning", "pets",
          "home repair", "politics", "computer hardware", "networking", "cybersecurity", "mobile apps"]
KINDS = ["a factual question", "a coding task", "a request to write something", "a request for an explanation",
         "a request for advice", "a math or logic problem", "a request for a list or a plan", "a request to compare things",
         "a request to summarize or rewrite a short text they include"]
LANGUAGES = ["English"] * 12 + ["Chinese", "Spanish", "French", "German"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=int, default=600)
    ap.add_argument("--user-tokens", type=int, default=96)
    ap.add_argument("--reply-tokens", type=int, default=384)
    ap.add_argument("--keep", type=int, default=65536, help="how many ranked ids to save")
    ap.add_argument("--out", default="qwen_mk/draft_vocab.npy")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    path = weights.snapshot_path()
    tok = AutoTokenizer.from_pretrained(path)
    stops = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    chat = lambda text: enc(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False))
    mk = Megakernel(path, max_seq=args.user_tokens + args.reply_tokens + 160, quant="int8", draft_vocab=0)
    rng = random.Random(0)

    counts = collections.Counter()
    t0 = time.time()
    for i in range(args.conversations):
        lang = rng.choice(LANGUAGES)
        seed = (f"Write {rng.choice(KINDS)} that someone might send to an AI assistant about {rng.choice(TOPICS)}"
                + ("" if lang == "English" else f", written in {lang}")
                + ". Reply with only their message.")
        mk.reset()
        q = [t for t in mk.generate(chat(seed), max_new_tokens=args.user_tokens, temperature=1.0, seed=2 * i,
                                    stop_ids=stops, spec=True) if t not in stops]
        mk.reset()
        a = list(mk.generate(chat(tok.decode(q).strip().strip('"')), max_new_tokens=args.reply_tokens,
                             temperature=0.7, seed=2 * i + 1, stop_ids=stops, spec=True))
        counts.update(a)
        if (i + 1) % 50 == 0:
            n = sum(counts.values())
            print(f"{i + 1} conversations, {n} tokens, {len(counts)} distinct, {n / (time.time() - t0):.0f} tok/s",
                  flush=True)
    vocab = mk.cfg.vocab
    seen = sorted(counts, key=lambda t: (-counts[t], t))
    rest = np.setdiff1d(np.arange(vocab), np.array(seen))
    ranked = np.concatenate([np.array(seen, dtype=np.int64), rest])[: args.keep].astype(np.int32)
    np.save(args.out, ranked)
    n = sum(counts.values())
    for k in (8192, 16384, 32768, 65536):
        print(f"top {k:6d}: covers {sum(counts[t] for t in seen[:k]) / n:.4f} of generated tokens")
    print(f"saved {len(ranked)} ranked ids to {args.out}")


if __name__ == "__main__":
    main()
