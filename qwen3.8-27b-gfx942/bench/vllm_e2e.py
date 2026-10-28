"""vLLM end-to-end: Qwen3.8-27B MXFP4 with the specialized decode kernel on or off.

Greedy generation of fixed prompts (tokens and logprobs saved for the comparison), then
decode speed: TPOT = (time(256 tokens) - time(1 token)) / 255 for batches of 1 to 8.
Usage: python bench/vllm_e2e.py MODEL_DIR on|off OUT.json [MAX_TOKENS]
Env: MK_TP (tensor parallel size), MK_ATTN (attention backend), MK_NO_CUSTOM_AR=1.
"""
import json
import os
import sys
import time

from vllm import LLM, SamplingParams

PROMPTS = [
    "The quick brown fox jumps over the lazy dog. Explain why this sentence is famous.",
    "Write a Python function that returns the n-th Fibonacci number, with a docstring.",
    "List the planets of the solar system in order from the Sun, one per line.",
    "In 1969, the first humans landed on the Moon. Describe the mission in detail.",
    "Translate into French: 'The weather is nice today, let us go for a walk.'",
    "What is the derivative of x^3 * sin(x)? Show your steps.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Give three tips for writing fast GPU kernels.",
]


def main():
    model, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    kw = dict(model=model, max_model_len=4096, language_model_only=True, enable_prefix_caching=False,
              gpu_memory_utilization=0.85, max_num_seqs=8, seed=0)
    if mode == "on":
        kw["specialized_decode"] = "monokernel"
    if os.environ.get("MK_TP"):  # tensor parallel size (TP2: both GPUs, one process)
        kw["tensor_parallel_size"] = int(os.environ["MK_TP"])
    if os.environ.get("MK_NO_CUSTOM_AR") == "1":  # diagnosis: RCCL for every all-reduce
        kw["disable_custom_all_reduce"] = True
    if os.environ.get("MK_ATTN"):
        kw["attention_backend"] = os.environ["MK_ATTN"]
    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1)
    gen = llm.generate(PROMPTS, sp)
    res = {"mode": mode, "load_s": load_s, "outputs": []}
    for g in gen:
        o = g.outputs[0]
        res["outputs"].append({"text": o.text, "tokens": list(o.token_ids),
                               "logprobs": [next(iter(lp.values())).logprob for lp in o.logprobs]})
    speed = {}
    for B in (1, 2, 4, 8):
        prompts = PROMPTS[:B]
        t = {}
        for n in (1, 256):
            spn = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
            llm.generate(prompts, spn, use_tqdm=False)           # warm-up
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                llm.generate(prompts, spn, use_tqdm=False)
                ts.append(time.perf_counter() - t0)
            t[n] = sorted(ts)[1]
        speed[B] = (t[256] - t[1]) / 255 * 1000
        print(f"{mode} batch {B}: TPOT {speed[B]:.2f} ms ({1000 / speed[B]:.1f} tok/s/user)", flush=True)
    res["tpot_ms"] = speed
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
