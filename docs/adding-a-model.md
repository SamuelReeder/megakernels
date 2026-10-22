# Adding a model

Each model lives in its own top-level directory and is a self-contained project. Copy
`qwen3.5-0.8b/` as a starting point, and rename the Python package and console scripts so that
installing several models side by side does not collide.

## Directory template

```
<model>/
  README.md          headline numbers, quick start, results, how it works
  pyproject.toml     its own package and console scripts
  csrc/              kernel (.cu/.h) and PyTorch binding
  <package>/         weights.py, engine.py, reference.py, session.py, chat.py, server.py
  bench/             decode benchmark and run_all.sh
  tools/             timeline profiler, perplexity, plots
  tests/  results/  docs/
```

Shared code goes in `common/` only once a second model needs it.

## Checklist

1. **Reference decoder first.** Write a plain PyTorch fp32 decoder (`reference.py`) and check it
   against Hugging Face logits before writing any CUDA. Every later bug is then a kernel bug.
2. **Port op by op.** Get each op (norm, projection, attention or recurrence, MLP, LM head)
   matching the reference in isolation before fusing, and keep an `all_logits` mode so whole
   prompts can be compared in one launch.
3. **Determinism.** Reduce in a fixed order and avoid float atomics, so that results are bitwise
   reproducible and any mismatch is a real bug.
4. **Measure against the roofline.** Run `common/bw.cu` to find the bandwidth the card sustains,
   and report weights streamed per second as a fraction of it.
5. **Benchmark honestly.** Compare wall-clock to wall-clock against the current llama.cpp (and
   vLLM where it supports the model), and bracket runs with a bandwidth check so contended
   measurements are redone. See `qwen3.5-0.8b/bench/run_all.sh`.
6. **Quality.** Report WikiText-2 perplexity against Hugging Face for every quantization.
7. **Tests.** Correctness against the reference, quantization round trip, determinism, and chat
   session state reuse.

## Pitfalls

- **Registers bound the design.** One CTA per SM with 17 warps caps threads at 96 registers.
  Check spills with `QWEN_MK_DEFINES="-Xptxas=-v"` before adding per-thread state.
- **WSL2 watchdog.** Windows kills kernels running longer than about 2 s, so cap the steps per
  launch and loop on the host.
- **GPU contention skews benchmarks.** Other processes (including Windows-side ones) steal
  bandwidth; trust only runs bracketed by a clean bandwidth check.
- **Shapes are compile-time constants.** Put them in one header and assert them against the
  checkpoint config at load time.
