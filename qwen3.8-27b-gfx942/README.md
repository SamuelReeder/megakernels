# qwen38-gfx942-mk

A FlyDSL decode megakernel for Qwen3.8-27B on gfx942 (MI300X): one persistent launch
per decode step for the whole model (64 layers, LM head), MXFP4 weights, batch 1 to 8.
Design, decisions and measurements: [docs/DESIGN.md](docs/DESIGN.md). Stock-engine
baseline: [bench/stock/results.md](bench/stock/results.md).
Tensor parallel 2 (two MI300X over XGMI, in-kernel all-reduce, `MK_TP=2` with
`srun -n2`): [docs/DESIGN-TP2.md](docs/DESIGN-TP2.md).

## Layout

| Path | Content |
|---|---|
| `mk/kernel.py` | the kernel: stage sequence, grid barriers |
| `mk/gemv.py` | the split-K MXFP4 / bf16 GEMV stream |
| `mk/stages.py` | GEMV prologues (LDS input) and epilogues |
| `mk/mixer.py` | Gated DeltaNet and attention tasks |
| `mk/dev.py`, `mk/fp4.py` | device helpers, MXFP4 tile layout |
| `mk/pack.py` | offline weight packing |
| `mk/runner.py` | host engine: buffers, pointer tables, one launch per step |
| `mk/reference.py` | PyTorch reference of the decode step (exact and emulate modes) |
| `tests/` | correctness, quality, determinism; `run_suite.sh` runs all of them |
| `bench/` | decode benchmark, per-stage profiler, reproducibility script |
| `mk/tp.py`, `mk/sync.py`, `mk/peer.py` | TP2 harness; copies of vLLM's DSv4.1 mailbox and `PeerBuffer` |
| `scratch/` | TP2 work: all-reduce spike, run plans, analysis scripts, logs |

## Run on Alola

1. Get a gfx942 node on a MARKHAM partition (shared home), for example:
   `salloc -p miopen -A miopen -J mkdev --constraint=GFX942 --gres=gpu:1 --cpus-per-task=16 --mem=200G -t 12:00:00 --no-shell`.
2. Container: `/cluster/images/hipdnn/hipdnn_20261017_gfx942.sqsh` (the measured
   runs; `hipdnn_latest_gfx942.sqsh` points to the newest weekly build).
3. Create the venv once: `bash scripts/setup_env.sh` (torch 2.14.1+rocm7.2,
   flydsl 0.4.0).
4. Models in `$HOME/mk/models/`: `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (kernel) and
   `Qwen/Qwen3.8-27B` (BF16, for the quality checks).
5. From a workstation: `scripts/mkrun '<command>'` syncs this repo to `~/mk/src` and
   runs the command in the held allocation (job name `mkdev`).

Commands (from the repository root on the node):

```bash
python tests/test_decode.py $HOME/mk/models/Qwen3.8-27B-Quark-AWQ-MXFP4 1 64 16   # kernel vs reference
python bench/bench_decode.py $HOME/mk/models/Qwen3.8-27B-Quark-AWQ-MXFP4 1 512 64 3  # TPOT, M=1, ctx 512
python bench/profile_decode.py $HOME/mk/models/Qwen3.8-27B-Quark-AWQ-MXFP4 1 512    # time per stage
bash tests/run_suite.sh $HOME/mk/models $HOME/mk/final                                # everything
```

## Limits

Decode only (a prompt goes in one token per step), greedy sampling on the host, one
GPU, M <= 16 rows, gfx942 only, no vLLM integration.
