# Step 8: hipDNN launch backend in the vLLM DecodeSpecs (docs/hipdnn-route.md)

`VLLM_SPECIALIZED_DECODE_LAUNCH=hipdnn` makes each gfx942 megakernel DecodeSpec launch a step
through hipdnn_vllm's custom op `megakernel.<spec name>` (inputs `[tab, glob]`, payload schema
1) in place of the FlyDSL JIT launch. The default `flydsl` is unchanged.

## Commits (github.com/SamuelReeder/vllm)

| Branch | Base | Head | Model |
|---|---|---|---|
| `users/sareeder/qwen38-gfx942-specialized-decode` | `0332fdc96` | `e745c7910` | Qwen3.8 TP1 and TP2 |
| `users/sareeder/qwen38-gfx942-tp2` | `9fc7bc243` | `b9a1943fb` | Qwen3.8 TP1 and TP2 |
| `users/sareeder/qwen36-gfx942-specialized-decode` | `6657a8352` | `81b2b128d` | Qwen3.6 |
| `users/sareeder/gemma4-moe-batch` | `5a25d336a` | `3cd36e151` | Gemma4 |

No commit touches `kernels/`, so `src` (the `kernels/` hash) is the rt2 set's
(`common/results/co/NOTES.md`, set `rt2`).

## Decisions

1. Setting: the environment variable `VLLM_SPECIALIZED_DECODE_LAUNCH` (`flydsl`, `hipdnn`) in
   `vllm/envs.py`. Reason: #61311 and our specs read settings from `vllm.envs`
   (`VLLM_SPECIALIZED_DECODE_CHECK_PATHS`, our commit `244171acc`); no spec reads
   `additional_config`. Only the specs read the variable, so it acts as the D11 spec setting.
2. Shared code: `vllm/models/common/specialized_decode/hipdnn_launch.py`, the same file on all
   four branches. The branches diverge at `0b5a1a0ae`, so no model package is common to all four
   at the same version. The helper is model-agnostic (op id, payload, `[tab, glob]`) and uses the
   library's `tp_agree` and `DecodeUnsupported`. It is a new file, so a rebase on #61311 has no
   conflict. Each model's `runtime/decode.py` builds the payload from its own `Cfg`.
3. Warm-up point: `bind_caches`, not `DecodeWarmable.compile`. `bind_caches` has the `Cfg`,
   `tab` and `glob`, runs before graph capture, and runs on every TP rank at the same point (the
   `tp_agree` per width is a collective). vLLM binds twice (profiling cache, then the real
   cache); the second `custom_op_prepare` is a plan-cache hit (same payload and shapes).
4. Startup error: `refuse_config` adds `refuse_launch()`. With `--specialized-decode` set,
   `select_decode_spec` raises `DecodeUnsupported` that names the fix (install the wheel, or add
   `custom_op` to `HIPDNN_VLLM_OPS`). The worker loads hipdnn_vllm with `_bootstrap.load()`,
   which is private. Ask the hipdnn_vllm owners for a public loader.
5. Qwen3.8 TP2: the hipdnn_vllm load runs inside the existing `tp_agree` with the weight packing,
   so a load failure on one rank does not leave the other rank in the `PeerBuffer` collective.
6. With `hipdnn` the kernel object never calls `KK.build` (no JIT compile, no fallback). It still
   imports FlyDSL and AITER: `kernel.py` holds `Cfg`, the key imports `mkcore`, and TP2 peer
   memory uses AITER. So `refuse_config` keeps the FlyDSL check. The preference "no FlyDSL at
   serve time" needs `Cfg` and `compile_key` moved out of `kernel.py` (not done).
7. `nlayers` in the payload is the model's layer count; `launch` asserts `layers == NL` on the
   hipDNN path (partial-layer launches are a FlyDSL debug feature).
8. Payload bytes: `json.dumps(sort_keys=True, separators=(",", ":"))`, the same bytes every step
   (the plan key holds them).
9. Qwen3.6 and Gemma: their `skip_step` checks the decline first, then calls the Qwen3.8
   `_skip_step` that they imported before. On those two branches, only that branch's model gets
   the backend (their `qwen3_5` package is an older Qwen3.8 without `mkcore/key.py`).
10. KV class at 2 KV heads (Qwen3.6, Qwen3.8 TP2): both TRITON views match form 1 first, as in
    the exporter. The strides are in `Cfg`, so `src` still differs between the views.
11. The payload test uses a stand-in `Cfg` with the real `compile_key` and `kernels/` directory:
    `kernel.py` imports AITER, which does not install on the CPU host. The real-`Cfg` check is
    stage 6 of `e2e_plan.sh` (every logged payload must select one manifest entry).

## CPU checks (`local_checks.sh`, `local/`)

ruff 0.14.0 check and format on the changed files: rc 0 on all four branches. Spec tests plus the
#61311 kit (`VLLM_TARGET_DEVICE=cpu`, `/tmp/vllmtest` with flydsl 0.4.1 added): Qwen3.8 TP1 81
passed, Qwen3.8 TP2 branch 81 passed, Qwen3.6 75 passed, Gemma4 83 passed (hipDNN tests 10, 10,
9, 8). Logs: `local/kit_<branch>.txt`, `local/ruff_<branch>.txt`, `local/summary.txt`.

## GPU plan (`e2e_plan.sh`, not run)

Stages: export the rt2 set with `run_set.sh`; put the lock and the code objects into the
megakernel descriptor folder (DVC-local) and verify them; build the gfx94X wheel; run vLLM with
the kernel on for `flydsl`, `hipdnn`, default (wheel, no opt-in) and without the wheel; build a
wheel without the M = 4 objects and run `hipdnn` (H2); then `check` prints H1, H2, H4, H6, TPOT
at B = 1, 2, 4, 8, and the payload-to-manifest match. Qwen3.8 TP2 runs with
`VLLM_SPECIALIZED_DECODE_CHECK_PATHS=1` (H3). GPU route: pending jobs 68215720, 68215745,
68215802.
