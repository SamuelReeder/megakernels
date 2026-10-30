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

A definition moves to `common/` when two or more models use it and one version serves all of
them with an unchanged ISA (R-share in [`common/DESIGN.md`](../common/DESIGN.md)).

## gfx942 FlyDSL models

Start from `common/mkcore`, the shared FlyDSL megakernel package: `from mkcore import dev as D`
(barrier, DPP reductions, buffer loads), `from mkcore.gemv import Gemv` (split-K unit-stream
GEMV), `mkcore.fp4` (MXFP4 tile layout), `mkcore.pack` (host-side unit-order packing),
`mkcore.stagelib` (shared stage helpers such as `store_quant8`), `mkcore.key` (compile key) and
`mkcore.vendor.buffer_ops`. Then copy from the closest existing model
(`qwen3.8-27b-gfx942/`, `qwen3.6-35b-a3b-gfx942/` or `gemma4-26b-a4b-gfx942/`) only what
diverges, into the model's `mk/` package:

- `kernel.py`: `Cfg`, `build`, the stage schedule.
- `stages.py`: `rstd_from_ssq`, `prologue_scaled` and the model's prologues and epilogues.
- `mixer.py` / `moe.py` / `attention.py`: the model's token mixer, experts, attention.
- `dims.py`, `reference.py`, `runner.py`, and `pack.py` (`Packed` and the model packers).

Rules from [`common/DESIGN.md`](../common/DESIGN.md):

- R-dep: `mkcore` imports no model module; a model imports `mkcore` and itself only.
- R-share: code moves to `mkcore` only when two or more models use it and one version keeps
  every model's ISA unchanged.
- R-state: model state never lives in `mkcore` modules.
- D4: the model owns its `Cfg`. D5: the compile key and kernel name come from `mkcore.key`
  and must cover every compile-time setting.

Gates and runs use `common/scripts/`: `gates.sh` (all-model gates, R-dep grep), `mkrun` (ship
the model and `common/` to a held gfx942 allocation and run a command there), `setup_env.sh`.
vLLM export: the model's `scripts/export_vllm.py` built on `common/scripts/export_lib.py`,
checked with `common/scripts/export_check.sh` (export diff and ISA hash), and run against a
vLLM branch with `common/scripts/vrun`.

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
- **Shared-memory bank conflicts hide behind DRAM.** A lane that reads its consecutive GEMV inputs
  as `float4`s 64 bytes from its neighbour's puts four lanes of each quarter-warp on the same
  banks. That costs nothing while the op waits on DRAM, and becomes the limit as soon as each
  weight does more work (two tokens per row in a speculative cycle). Store the input vectors
  swizzled; the profiler's "weight wait" column dropping to zero is the tell.
- **Bitwise identity across instantiations needs explicit FMAs.** NVCC contracts `a * b + c`
  differently in different template instantiations of the same expression, so two code paths
  that must agree to the bit (speculative verify vs plain decode) can differ by an ulp. Build with
  `--fmad=false` and write the FMAs you want with `fmaf`.
