"""vLLM end-to-end: Qwen3.6-35B-A3B FP8 with the specialized decode kernel on or off.

run:     greedy generation of 8 fixed prompts (tokens and top-5 logprobs saved for the
         comparison), then decode speed: 512-token prompts, TPOT = (time(256 tokens) -
         time(1 token)) / 255 at batch 1, 2, 4, 8 (median of 3). With the kernel on,
         the kernel's own step counter (the grid barrier epoch on the device) is read
         before and after every generate, which proves the kernel ran each decode step.
compare: per prompt the fraction of identical tokens, the first divergence and the
         logprob gap of the two candidates there, under both runs.
Usage: python bench/vllm_e2e.py run MODEL_DIR on|off OUT.json [MAX_TOKENS]
       python bench/vllm_e2e.py compare OFF.json ON.json
Env: MK_ATTN = TRITON_ATTN | ROCM_ATTN (vLLM's choice when unset).
"""
import json
import os
import sys
import time

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
CTX, GEN = 512, 256


def kernel_steps(model):
    """The kernel's decode steps so far (-1: no specialized decode on this model)."""
    sd = getattr(model, "specialized_decode", None)
    return -1 if sd is None else sd.kernel.steps_run()


def run(model, mode, out, max_tokens):
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # apply_model
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(model=model, max_model_len=4096, language_model_only=True,
              enable_prefix_caching=False, gpu_memory_utilization=0.85, max_num_seqs=8,
              seed=0)
    if mode == "on":
        kw["specialized_decode"] = "monokernel"
    if os.environ.get("MK_ATTN"):
        kw["attention_backend"] = os.environ["MK_ATTN"]
    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    steps = lambda: llm.apply_model(kernel_steps)[0]  # noqa: E731
    res = {"mode": mode, "attn": os.environ.get("MK_ATTN", "default"), "load_s": load_s,
           "outputs": [], "kernel_steps": {}}
    s0 = steps()
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=5)
    gen = llm.generate(PROMPTS, sp)
    res["kernel_steps"]["quality"] = steps() - s0
    for g in gen:
        o = g.outputs[0]
        res["outputs"].append({
            "text": o.text,
            "tokens": list(o.token_ids),
            "top": [{str(t): lp.logprob for t, lp in d.items()} for d in o.logprobs],
        })
    # 512-token prompts: windows of a long text, a different one per row
    tok = llm.get_tokenizer()
    ids = tok.encode(" ".join(PROMPTS * 40))
    assert len(ids) >= CTX + 8 * 37
    speed, used = {}, {}
    for B in (1, 2, 4, 8):
        prompts = [TokensPrompt(prompt_token_ids=ids[37 * r: 37 * r + CTX]) for r in range(B)]
        t = {}
        for n in (1, GEN):
            spn = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
            llm.generate(prompts, spn, use_tqdm=False)  # warm-up
            ts = []
            for _ in range(3):
                s0 = steps()
                t0 = time.perf_counter()
                llm.generate(prompts, spn, use_tqdm=False)
                ts.append(time.perf_counter() - t0)
                used[f"b{B}_n{n}"] = steps() - s0
            t[n] = sorted(ts)[1]
        speed[B] = (t[GEN] - t[1]) / (GEN - 1) * 1000
        print(f"{mode} {res['attn']} batch {B}: TPOT {speed[B]:.2f} ms "
              f"({1000 / speed[B]:.1f} tok/s/user), kernel steps per {GEN}-token "
              f"generate {used[f'b{B}_n{GEN}']}", flush=True)
    res["tpot_ms"] = speed
    res["kernel_steps"].update(used)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


def compare(off_path, on_path):
    off, on = json.load(open(off_path)), json.load(open(on_path))
    print(f"kernel off: {off_path} ({off['attn']}), kernel on: {on_path} ({on['attn']})")
    print(f"kernel steps during the 8-prompt generate: on {on['kernel_steps']['quality']}, "
          f"off {off['kernel_steps']['quality']}")
    tot_same = tot = 0
    for i, (a, b) in enumerate(zip(off["outputs"], on["outputs"])):
        ta, tb = a["tokens"], b["tokens"]
        n = min(len(ta), len(tb))
        same = sum(x == y for x, y in zip(ta, tb))
        first = next((j for j in range(n) if ta[j] != tb[j]), None)
        tot_same += same
        tot += max(len(ta), len(tb))
        line = f"prompt {i}: identical {same}/{max(len(ta), len(tb))}"
        if first is None:
            line += ", no divergence"
        else:
            def gap(run, j):
                top = run["top"][j]
                x, y = top.get(str(ta[j])), top.get(str(tb[j]))
                return "n/a" if x is None or y is None else f"{abs(x - y):.4f}"
            line += (f", first divergence at token {first}: off {ta[first]} on {tb[first]}, "
                     f"logprob gap under off {gap(a, first)}, under on {gap(b, first)}")
        print(line)
        print(f"  off: {a['text'][:160]!r}")
        print(f"  on:  {b['text'][:160]!r}")
    print(f"total identical tokens {tot_same}/{tot} ({100 * tot_same / tot:.1f}%)")
    print("TPOT ms   batch:  " + "  ".join(f"{B:>6}" for B in off["tpot_ms"]))
    for r in (off, on):
        print(f"  kernel {r['mode']:<3}       " + "  ".join(f"{v:6.2f}" for v in r["tpot_ms"].values()))
    print("  speedup         " + "  ".join(f"{off['tpot_ms'][B] / on['tpot_ms'][B]:6.2f}"
                                            for B in off["tpot_ms"]))


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 160)
