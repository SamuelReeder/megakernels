"""vLLM decode throughput on the same footing as bench_decode.py and llama-bench.

Batch 1, bf16, CUDA graphs on. Prompt of `ctx` tokens, then 256 generated tokens; decode time is
(time to generate 256 tokens) - (time to generate 1), so prefill is excluded. Best of 3.

Run it in a separate environment (vLLM pins its own torch):
    ~/venvs/vllm/bin/python bench/bench_vllm.py --json results/vllm_bf16.json
"""

import argparse
import json
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--ctx", type=int, nargs="+", default=[1, 512, 2048, 4096])
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mem", type=float, default=0.25, help="gpu_memory_utilization")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(args.model, dtype="bfloat16", max_model_len=max(args.ctx) + args.tokens + 16,
              gpu_memory_utilization=args.mem, enable_prefix_caching=False, seed=0,
              max_num_seqs=8)  # batch 1; the default 256 needs more Mamba-state blocks than 25% of the GPU holds
    one = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    many = SamplingParams(max_tokens=args.tokens, temperature=0.0, ignore_eos=True)
    rows = []
    for ctx in args.ctx:
        prompt = {"prompt_token_ids": [(i * 7919) % 50000 + 1000 for i in range(ctx)]}
        llm.generate([prompt], one, use_tqdm=False)  # warm up (graph capture, kernels)
        llm.generate([prompt], many, use_tqdm=False)
        best = 0.0
        for _ in range(args.reps):
            t0 = time.perf_counter()
            llm.generate([prompt], one, use_tqdm=False)
            t1 = time.perf_counter()
            out = llm.generate([prompt], many, use_tqdm=False)
            t2 = time.perf_counter()
            n = len(out[0].outputs[0].token_ids)
            best = max(best, (n - 1) / ((t2 - t1) - (t1 - t0)))
        rows.append(dict(ctx=ctx, tok_s=best))
        print(f"ctx {ctx:5d}: {best:7.1f} tok/s")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(engine="vllm", version=vllm.__version__, quant="bf16", rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
