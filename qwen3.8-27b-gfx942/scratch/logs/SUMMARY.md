# Milestone 1 spike: in-kernel TP2 all-reduce latency (2026-10-27)

Node banff-cyxtera-s70-4 (MI300X, 304 CUs), job 68191368, GPUs 6 and 7 (XGMI, weight 15,
`20261027a/env.log`). Container hipdnn_20261024_gfx942, torch 2.14.1+rocm7.2, FlyDSL 0.4.0.
Code: `scratch/ar_spike/` (disposable). Raw logs: here and `/cluster/logs/qwen38-tp2/`.

Kernel: 304 CTAs x 512 threads per GPU, 128 rounds per launch. Each round is a push of
(value, tag) pairs to every rank's region (DSv4.1 `Mailbox`, system scope), a poll, an fp32
sum in rank order, a store, and the TP1 grid barrier. Peer buffers come from DSv4.1
`PeerBuffer` (copied: no aiter in the container). The added cost is the round time with the
all-reduce minus the round time of the barrier-only round (stamps, rank 0, 20 launches x 127
rounds). Every round of the checked launch was bitwise exact and bitwise equal on both ranks.

| M | fp32 pairs: added med / p99 us (run a, run b) | bf16-pair pairs: med / p99 us (run b) |
|---:|---|---|
| 1 | 1.15 / 1.64, 0.93 / 1.82 | 0.81 / 1.11 |
| 2 | -, 1.64 / 2.08 | 0.96 / 1.68 |
| 4 | -, 3.55 / 3.96 | 1.84 / 2.31 |
| 8 | 7.07 / 7.72, 7.02 / 7.83 | 3.57 / 4.44 |

Comparison, one all-reduce in a CUDA graph of 128 in a row, median us (fp32 / bf16):

| M | RCCL (torch 2.14.1) | vLLM custom AR (vLLM 0.32.1rc1 nightly 00b6c4c5, AITER off) | vLLM + AITER AR |
|---:|---|---|---|
| 1 | 9.75 / 10.88 | 4.14 / 4.00 | 5.64 / 3.34 |
| 8 | 16.14 / 15.52 | 7.47 / 5.59 | 9.10 / 5.86 |

Files: `20261027a/coll_torch_r0.json`, `20261027a/coll_vllm_r0.json` (AITER on),
`20261027b/coll_vllm-noaiter_r0.json`. Prediction: `20261027{a,b}/predict_r0.txt`.
