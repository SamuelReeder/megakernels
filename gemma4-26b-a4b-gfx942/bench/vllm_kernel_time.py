"""Time the kernel launch alone inside a vLLM engine (kernel on): after a generate of
B distinct 512-token prompts (the rows of bench/vllm_e2e.py's speed test), relaunch the
kernel of width B 50 times on the last step's inputs (CUDA events) and print the
median. Separates the kernel's time in vLLM's configuration (max_model_len, vLLM's KV
layout) from vLLM's per-step work around it. The relaunches rewrite the K / V of the
last step's slots only.
Usage: python bench/vllm_kernel_time.py MODEL_DIR
"""
import os
import sys

import torch


def time_kernel(model, w):
    k = model.specialized_decode.kernel
    ts = []
    for _ in range(50):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        k.launch(w, k.glob)
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def main():
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vllm_e2e import CTX, PROMPTS

    llm = LLM(model=sys.argv[1], max_model_len=4096, language_model_only=True,
              enable_prefix_caching=False, gpu_memory_utilization=0.85, max_num_seqs=8,
              seed=0, specialized_decode="monokernel")
    ids = llm.get_tokenizer().encode(" ".join(PROMPTS * 40))
    for B in (1, 2, 4, 8):
        prompts = [TokensPrompt(prompt_token_ids=ids[37 * r: 37 * r + CTX]) for r in range(B)]
        llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True),
                     use_tqdm=False)
        ms = llm.apply_model(lambda m, B=B: time_kernel(m, B))[0]
        print(f"width {B}: kernel launch median {ms:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
