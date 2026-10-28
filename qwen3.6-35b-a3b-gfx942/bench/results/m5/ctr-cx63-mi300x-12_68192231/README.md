# ctr-cx63-mi300x-12 (SLURM job 68192231): wrong results, cause outside the kernels

Finding: on 2026-10-27 the GPU of job 68192231 on ctr-cx63-mi300x-12 gave
different results for the same inputs. This is also true for code that does not use a
megakernel: the PyTorch reference (`mk/reference.py`) alone. The configuration is a
supported one, and the shared megakernel code is not the cause (evidence below). No
"refuse" check applies, so none was added. Slurm drained the node at 2026-10-27 21:19 EDT
with `Reason=Kill task failed`. The drain came 50 minutes after this job ended.

## The node

- SLURM (`slurm_node.txt`, read 2026-10-28): features `SUPERMICRO, AMDCPU, EPYC9654,
  MARKHAM, UBUNTU24, ROCM10.1.0, GFX942`; GRES `gpu:gfx942-mi300x:8`; partitions
  `defq, rccl`; state `MIXED+DRAIN`.
- GPU, read inside job 68192231 on 2026-10-27. These values were printed to the session
  and not saved to files, so they are transcribed here:
  - torch: `AMD Instinct MI300X`, `gfx942:sramecc+:xnack-`, 304 CUs.
  - `current_compute_partition` SPX and `current_memory_partition` NPS1 on all 8 GPUs.
  - Driver 7.1.9.31600000, kernel 6.8.0-31-generic. The amdgpu parameters
    `mtype_local=-1 noretry=-1 sched_policy=0 cwsr_enable=1 mcbp=-1` are the same as on
    banff-cyxtera-s70-4.
  - VBIOS 113-M3000108-103 (`env.txt`). ctr-cx66-mi300x-13 has the same VBIOS, and the
    Qwen3.8 kernel passes there.
- Same container image (`hipdnn_latest_gfx942.sqsh`), venv and code as the passing runs.
- The node cannot be read again now. It is drained, and `pam_slurm_adopt` refuses SSH
  without a job. Thus `rocminfo` and `amd-smi` output for this node is not on disk.

This is a supported configuration: MI300X, gfx942, 304 CUs, SPX/NPS1, and the same
software stack as the passing nodes.

## What "wrong" means

All runs: `tests/test_decode.py` with M = 8, 4 layers, 6 steps and seed 0. All six runs
ran in the same SLURM step within 3 minutes, with the same code.

| Run | Kernel vs emulating reference, worst cosine | Result |
|---|---|---|
| `test_decode_m8_4layers_run1.txt` | 0.009116 (step 0) | FAIL |
| `test_decode_m8_4layers_run2.txt`, `_run3.txt` | 0.986872 | PASS, but the per-step values differ between the runs |
| `test_decode_m8_4layers_fence_run2.txt` (`MK_FENCE=1`) | NaN logits at steps 2 to 5 | FAIL |
| `test_decode_m8_4layers_fence_run1.txt` | NaN logits at step 5 | PASS (gate bug, see below) |
| `qwen38_test_decode_m8_8layers.txt` (Qwen3.8 kernel, 8 layers) | NaN logits at steps 1 to 5 | FAIL |

## Why the cause is not the shared code

1. **The PyTorch reference alone is not reproducible on this GPU.** The `emulate~exact`
   column of `test_decode.py` compares two pure PyTorch decoders. It does not use the
   kernel: the inputs are the seed tokens, then the exact reference's own argmax.
   - On ctr-cx63-mi300x-12, the six runs of the same code give six different columns,
     and none of them equals the banff column. For example, step 1 gives 0.989013 or
     0.989679, and step 3 gives 0.995038 to 0.995057.
   - On banff-cyxtera-s70-4, eight runs give one column, bit for bit:
     0.997617 0.989679 0.966664 0.995048 0.989652 0.988025. Six runs had the kernel in the
     process (`../node_investigation/control_*`, `chaos_*`). Two runs had no kernel
     (`../node_investigation/reference_only_*`).
2. **Qwen3.8, the same code on both nodes.** The Qwen3.8 code that ran on this node
   (`~/mk/src/qwen3.8-27b-gfx942` on Alola) was run twice on banff-cyxtera-s70-4
   (`../node_investigation/qwen38_same_code_as_ctr12_banff_x2.txt`).
   - At step 0, the pure-PyTorch `emulate~exact` cosine is 0.999882 on banff in both
     runs, and 0.999866 on this node. Step 0 has identical inputs on both nodes: the seed
     tokens and a fresh state.
   - On banff the kernel passes, with bitwise-equal runs.
3. **Timing does not change the kernels' results.** `Cfg.chaos` (`dev.chaos_sleep`) adds
   random delays of 0 to 7 us before every grid-barrier arrival and every counter
   atomic. This changes the arrival order and the last-arriving wave from run to run,
   and costs +3 ms per step (`../node_investigation/chaos_bench_m1.txt`). On banff,
   the results with chaos are bitwise equal to the results without chaos:
   - three 4-layer runs (`chaos_m8_4layers_run*.txt`, all with the same md5 as the
     control runs);
   - one 40-layer `test_decode.py` run (`chaos_m8_40layers.txt`);
   - `test_determinism.py` (`chaos_determinism_m8.txt`: 3 runs bitwise equal, and
     batch-invariant).

   A race in the barrier or the counter protocol would show up as a difference here. In
   the earlier runs on this node, agent-scope release and acquire fences (`MK_FENCE=1`)
   did not remove the failures.
4. A torch bf16 GEMM (8192^3, 300 times) and a 1 GiB copy (200 times) were bitwise stable
   on this GPU. Thus the fault is intermittent, or it is specific to some operations.

Conclusion: this GPU, or this node, computes wrong results. That includes PyTorch
results. The fault is not specific to the megakernels. The hardware cause (which unit)
is not known: the node cannot be accessed. Report the node to the cluster administrators
before you use it again.

## Gate fixed on the way

The NaN steps in `test_decode_m8_4layers_fence_run1.txt` passed the gate.
`worst = min(cosines)` skips a NaN that is not the first value. Both `tests/test_decode.py`
(Qwen3.6 and Qwen3.8) now count steps with non-finite logits, and they fail when that
count is not 0.
