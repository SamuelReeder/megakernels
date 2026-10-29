"""GPU kernel time per decode step of vLLM's own path (kernel off, CUDA graphs), by kernel
name: what vLLM's fused MoE and the kernels around it take per layer at batch B, for the
comparison with the megakernel's expert stages (docs/DESIGN.md "Milestone 8").
The 512-token real-text prompts of bench/vllm_e2e.py, 64 greedy tokens per batch, under
rocprofv3's kernel trace (hardware timestamps; the torch profiler was tried and rejected:
its kernel durations sum to 1.44x the measured TPOT at batch 8).
Usage: rocprofv3 --kernel-trace --output-format csv -d OUT -o trace -- \
           python bench/vllm_moe_profile.py run MODEL_DIR [B ...]
       python bench/vllm_moe_profile.py report OUT/.../trace_kernel_trace.csv
"""
import collections
import csv
import os
import statistics as stt
import sys
import time

GEN = 64


def run(model, widths):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vllm_e2e import CTX, PROMPTS

    llm = LLM(model=model, max_model_len=4096, language_model_only=True,
              enable_prefix_caching=False, gpu_memory_utilization=0.85, max_num_seqs=8,
              seed=0)
    ids = llm.get_tokenizer().encode(" ".join(PROMPTS * 40))
    sp = SamplingParams(temperature=0.0, max_tokens=GEN, ignore_eos=True)
    for B in widths:
        prompts = [TokensPrompt(prompt_token_ids=ids[37 * r: 37 * r + CTX]) for r in range(B)]
        time.sleep(3)  # a gap in the trace marks the start of batch B
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        print(f"batch {B}: generate {time.perf_counter() - t0:.3f} s", flush=True)


def report(path, widths):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    rows.sort()
    # sections: split at gaps > 2 s (the sleeps before each batch); the last len(widths)
    secs, cur = [], [rows[0]]
    for a, b in zip(rows, rows[1:]):
        if b[0] - a[1] > 2e9:
            secs.append(cur)
            cur = []
        cur.append(b)
    secs.append(cur)
    for B, sec in zip(widths, secs[-len(widths):]):
        # decode steps: the LM head GEMM ends each step; keep the last GEN - 8 steps
        lm = [i for i, r in enumerate(sec) if "wvSplitK" in r[2] or "MT128x16x128" in r[2]]
        steps = list(zip(lm[-(GEN - 8) - 1:-1], lm[-(GEN - 8):]))
        tot = collections.Counter()
        n = collections.Counter()
        moe = []
        for i0, i1 in steps:
            for s, e, name in sec[i0 + 1: i1 + 1]:
                tot[name] += e - s
                n[name] += 1
                if name.startswith("fused_moe_kernel"):
                    moe.append(e - s)
        ns = len(steps)
        wall = stt.median((sec[i1][1] - sec[i0][1]) / 1e3 for i0, i1 in steps)
        print(f"=== batch {B}: {ns} decode steps; step wall time (LM head end to end) median "
              f"{wall:.1f} us; kernel time {sum(tot.values()) / ns / 1e3:.1f} us per step")
        print(f"{'us/step':>9} {'us/call':>8} {'calls/step':>10}  kernel")
        for name, t in tot.most_common(22):
            print(f"{t / ns / 1e3:9.1f} {t / n[name] / 1e3:8.2f} {n[name] / ns:10.1f}  {name[:100]}")
        if moe:
            w13, w2 = moe[0::2], moe[1::2]
            print(f"fused_moe_kernel per call, median: w13 (gate_up) {stt.median(w13) / 1e3:.2f} "
                  f"us, w2 (down) {stt.median(w2) / 1e3:.2f} us")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], [int(b) for b in sys.argv[3:]] or [4, 8])
    else:
        report(sys.argv[2], [int(b) for b in sys.argv[3:]] or [4, 8])
