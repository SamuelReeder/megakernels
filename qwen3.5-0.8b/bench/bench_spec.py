"""Speculative decoding throughput: the same chat replies decoded plainly and with MTP drafts.

How many drafts get accepted depends on the text, so this decodes real chat replies (greedy, the
16 prompts of tools/mtp_acceptance.py) rather than bench_decode.py's synthetic contexts. Both
modes must produce identical tokens (checked). Throughput is tokens / GPU time, from the kernel's
per-step and per-cycle timestamps; each prompt is timed `reps` times, alternating modes, and the
fastest run kept (other GPU clients only ever add time).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from mtp_acceptance import PROMPTS  # noqa: E402

from qwen_mk import weights  # noqa: E402
from qwen_mk.engine import MAX_STEPS_PER_LAUNCH, Megakernel  # noqa: E402


def decode(mk, ids, n, stops, spec):
    """Greedy reply: (tokens, GPU ns per step or cycle)."""
    mk.reset()
    last = mk.prefill(ids)
    out, ns, draft = [], [], last
    while len(out) < n:
        k = min(MAX_STEPS_PER_LAUNCH, n - len(out))
        if spec:
            got, draft = mk.run_spec(last, draft, k, stop_ids=stops, timing=True)
        else:
            got = mk.run([last], n_generate=k, stop_ids=stops, timing=True)
        out += got
        ns += mk.last_step_ns
        if got[-1] in stops:
            break
        last = got[-1]
    return out, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=["bf16", "int8"], default="int8")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--draft-vocab", type=int, nargs="+", default=[32768, 0], help="0 = full vocabulary")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    path = weights.snapshot_path()
    tok = AutoTokenizer.from_pretrained(path)
    stops = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    prompts = [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False),
                   add_special_tokens=False).input_ids for p in PROMPTS]
    rows = []
    for dv in args.draft_vocab:
        mk = Megakernel(path, max_seq=1024, quant=args.quant, draft_vocab=dv)
        decode(mk, prompts[0], 32, stops, True)  # warm-up
        tok_n = plain_ns = spec_ns = cycles = 0
        steps, cyc_ns = [], []
        for ids in prompts:
            best = {}
            for _ in range(args.reps):
                for spec in (False, True):
                    out, ns = decode(mk, ids, args.tokens, stops, spec)
                    if spec not in best or sum(ns) < sum(best[spec][1]):
                        best[spec] = (out, ns)
            (a, pns), (b, sns) = best[False], best[True]
            assert a == b, "speculative decoding changed the output"
            tok_n += len(a)
            plain_ns += sum(pns)
            spec_ns += sum(sns)
            cycles += len(sns)
            steps += pns
            cyc_ns += sns
        row = dict(draft_vocab=mk.draft_vocab, tokens=tok_n, cycles=cycles, tokens_per_cycle=tok_n / cycles,
                   plain_tok_s=tok_n / plain_ns * 1e9, spec_tok_s=tok_n / spec_ns * 1e9, speedup=plain_ns / spec_ns,
                   median_step_ms=float(np.median(steps)) / 1e6, median_cycle_ms=float(np.median(cyc_ns)) / 1e6)
        rows.append(row)
        print(f"draft vocab {mk.draft_vocab:6d}: plain {row['plain_tok_s']:6.1f} tok/s, spec {row['spec_tok_s']:6.1f} "
              f"tok/s ({row['speedup']:.2f}x); {row['tokens_per_cycle']:.3f} tokens/cycle, cycle "
              f"{row['median_cycle_ms']:.3f} ms vs step {row['median_step_ms']:.3f} ms", flush=True)
        del mk
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(quant=args.quant, gpu=torch.cuda.get_device_name(), prompts=len(PROMPTS),
                           max_tokens=args.tokens, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
