Step 8 GPU e2e (e2e_plan.sh through common/results/pipeline), 2026-10-30, Alola MI300X
(ctr-cx66-mi300x-31, job 68220723), vLLM nightly image 00b6c4c5 (Ubuntu 22.04).
Wheel: hipdnn_vllm gfx94X of rocm-libraries adeef272211 (hipdnn-vllm-custom-op + megakernel-engine
18c6ef46d60), built in the vLLM image (wheels.sha256); descriptor folder = engine lock, manifest
sha256 5b83b31430d7 (q8 TP1 + q6 rt2, q8 TP2 + Gemma rt0), objects from co-export 0cafc63 sets.
verdicts.txt: one PASS/FAIL line per check; runs/<model>/{flydsl,hipdnn,default,nowheel}.json;
runs_<model>.log: the vLLM logs (launch and payload lines); drop4/: H2 runs.

H1  PASS q8, q6, g (hipdnn == flydsl, tokens and logprobs bitwise). q8tp2 FAIL bitwise: 6 of 8
    prompts equal; flydsl repeated (flydsl2) is also 6 of 8 equal to flydsl: vLLM's TP2 path is
    not repeatable (qwen3.8-27b-gfx942/docs/DESIGN-TP2.md), so TP2 bitwise H1 rests on the
    code-object check (co-export h1.log, TP2 deployment, PASS).
H2  PASS q8, q6: drop4 wheel; width 4 declines to vLLM's path, widths 1, 2, 8 launch, no raise.
H3  PASS: q8tp2 runs with VLLM_SPECIALIZED_DECODE_CHECK_PATHS=1 complete.
H4  PASS all: FULL graphs, zero CAPTURE_INVARIANT and custom_op fallback rows.
H6  PASS q8, q6, g (default tokens == no wheel, no custom_op row); q8tp2 tokens differ by the
    same TP2 spread, custom_op rows 0.
Payloads: every logged payload selects exactly one manifest entry (all models).
H5, H7: not in this plan (H5: engine matcher tests in pipeline/out/b_mkeng.out; H7: no fault
    injection exists).
