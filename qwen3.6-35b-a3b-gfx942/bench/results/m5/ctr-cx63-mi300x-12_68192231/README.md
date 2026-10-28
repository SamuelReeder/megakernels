# ctr-cx63-mi300x-12: non-deterministic results (node excluded)

On 2026-10-27 the GPU of SLURM job 68192231 on ctr-cx63-mi300x-12 (MI300X, VBIOS
113-M3000108-103, same image, driver and amdgpu parameters as banff-cyxtera-s70-4) gave
run-to-run different results for the same inputs, often with NaN logits:

- `test_decode_m8_4layers_run{1,2,3}.txt`: the same test three times; run 1 has a
  worst kernel~emulate cosine of 0.0091, runs 2 and 3 give 0.986872 (the value every
  run on banff-cyxtera-s70-4 gives, bitwise reproducible there).
- `test_decode_m8_4layers_fence_run{1,2,3}.txt`: the same with `MK_FENCE=1`
  (`buffer_wbl2 sc1` / `buffer_inv sc1` at every grid barrier and around every
  counter atomic). One of three runs still differs (0.990284). The fences do not remove
  the problem, so the kernel's cross-XCD hand-off protocol is not the cause as far as
  these runs show.
- `qwen38_test_decode_m8_8layers.txt`: the Qwen3.8-27B kernel (unchanged, which passes
  its suite on other MI300X nodes) gives NaN logits at step 5 on the same GPU.
- A torch bf16 GEMM (8192^3, 300 times) and a 1 GiB copy (200 times) on this GPU were
  bitwise stable.

Both kernels fail on this GPU and pass on banff-cyxtera-s70-4 (two GPUs) and, for
Qwen3.8, on ctr-cx66-mi300x-13. The milestone 5 runs therefore use other nodes. The
cause (a GPU fault or a timing-dependent race that only this GPU exposes) is not
proven; it stays an open question in docs/DESIGN.md.
