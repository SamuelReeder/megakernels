# Spike rtgeom: results

Node quanta-ccs-aus-j02-45 (AUSTIN, MI325X gfx942, 304 CUs), job 68210914, megakernels 19e217f
(rtrun later got `--container-writable`: the vLLM image is read-only on this node and Quark's JIT
writes into it). All variants ran in one job on one GPU, rt 0/1/2 alternating (bench_ab.py).
Logs: logs/aus_runs/ (console_*.log are the rtrun consoles).

## TPOT, standalone (bench_ab.py, 3 reps, median [min, max] ms)

| ctx / maxlen | M | rt0 | rt1 | rt2 |
|---|---|---|---|---|
| 512 / 640 | 1 | 6.798 [6.778, 6.802] | 6.742 [6.735, 6.760] -0.83 % | 6.747 [6.739, 6.754] -0.76 % |
| 512 / 640 | 2 | 7.279 [7.265, 7.288] | 7.235 [7.227, 7.254] -0.61 % | 7.220 [7.217, 7.228] -0.81 % |
| 512 / 640 | 4 | 7.908 [7.891, 7.913] | 7.932 [7.924, 7.953] +0.31 % | 7.932 [7.922, 7.936] +0.31 % |
| 512 / 640 | 8 | 9.045 [9.035, 9.059] | 9.147 [9.144, 9.173] +1.13 % | 9.102 [9.092, 9.106] +0.63 % |
| 3500 / 4096 | 1 | 7.284 [7.279, 7.290] | 7.298 [7.296, 7.306] +0.20 % | 7.289 [7.286, 7.291] +0.06 % |
| 3500 / 4096 | 2 | 7.821 [7.815, 7.823] | 7.784 [7.782, 7.786] -0.47 % | 7.790 [7.790, 7.791] -0.39 % |
| 3500 / 4096 | 4 | 8.760 [8.759, 8.763] | 8.766 [8.761, 8.767] +0.07 % | 8.783 [8.780, 8.790] +0.26 % |
| 3500 / 4096 | 8 | 10.648 [10.639, 10.649] | 10.763 [10.758, 10.763] +1.08 % | 10.668 [10.659, 10.669] +0.19 % |

Logs: bench/ab_{short,long}_m*.log, console_rtgb.log.

## Profile M=8, ctx 3500 (bench/profile_long_rt*_m8.log)

Step total 10036.5 (rt0), 10048.4 (rt1), 10054.4 (rt2) us. The only stage that moves is attn:
sum of spans 1181.1 -> 1259.5 us (rt1, +78 us, busy 29.2 -> 30.9 us per layer) and 1220.6 us (rt2).
`out`/`o` are 60-80 us faster with rt1, so the net profile delta is small; the +1.1 % bench
delta at M=8 is the attention stage (run-time KV strides / block count in the paged-KV loop).

## vLLM (bench/vllm_e2e.py, TP1, 8 greedy prompts, max 128 tokens)

KV block 784 at every maxlen (6 / 11 / 21 block-table columns for 4096 / 8192 / 16384).
rt1 code objects exported once at maxlen 4096 (co_rt1_manifest.jsonl; M=1/2/4/8 sha256 b4f31055,
e421b8eb, 10279595, 7ed37ca7) and launched via hipModuleLaunchKernel at all three maxlens.

| maxlen | tokens spec == rtco | logprobs bitwise | TPOT B=1/2/4/8 spec ms | rtco ms |
|---|---|---|---|---|
| 4096 | equal | equal | 6.66 / 7.12 / 7.81 / 9.05 | 6.75 / 7.14 / 7.89 / 9.18 |
| 8192 | equal | equal | 6.73 / 7.18 / 7.86 / 9.37 | 6.82 / 7.20 / 7.95 / 9.50 |
| 16384 | equal | equal on rerun (b) | 6.73 / 7.18 / 7.89 / 9.12 | 6.78 / 7.18 / 7.94 / 9.23 |

rt4096 (rt1 compiled in-process at 4096) also equals spec4096 bitwise. At 16384 the first rtco run
differed from spec in logprobs of one request (max 0.145, tokens equal); the rerun rtco16384b
equals spec16384 and spec16384b bitwise, while rtco16384 != rtco16384b, so that was run-to-run
vLLM variation, not the kernel. Logs: vllm/*.log, *.json, console_rtgv2.log, console_rtgr.log.

## Tests (rt1, checks/)

test_decode (64 layers, 16 steps) PASS at M=1 and M=8, plain and x-packed KV block 784
(M=8 kernel~exact worst cos 0.987787). test_determinism M=1 and M=8: 3 runs bitwise equal.

## Verdict

rt1: one binary per model and width covers every maxlen (shown in vLLM at 4096/8192/16384; block is
also a run-time value, not exercised), at about +1.1 % TPOT at M=8 (attention); noise at M<=4.
rt2: also one ELF across maxlens for vLLM's fixed block 784 (logs/isa_local.txt: vm1 rt2 L4096 and
L8192 both c568b1c77c7ec4e9; not run through vLLM's code-object path), and within +0.63 % (M=8
short) / +0.26 % (long) everywhere. Adopt rt2: 4 binaries per model (M=1/2/4/8) per block
size; use rt1 only if the block size must vary at run time.
