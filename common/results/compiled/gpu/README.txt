gpu.sh plan (plan.sh) on job 68215802 (ctr-cx66-mi300x-31, 2x MI300X), run by the megakernels
hipdnn-design pipeline (common/results/pipeline, stage a_cc), 2026-10-30. verdicts.txt: one line per check.
All gates new == main PASS (q8, q6, g, q8tp2: hashes and ISA). vLLM greedy new == previous heads:
tokens equal for every TP1 pair at 4096 and 8192; rt2 one binary per width for 4096 and 8192 (ISA).
The one FAIL, q8new_tp2_4096 vs q8prev_tp2_4096 (prompt 3 diverges at token 96), is vLLM's TP2
run-to-run spread (qwen3.8-27b-gfx942/docs/DESIGN-TP2.md): tp2det/ reran both heads on job 68218902;
q8prev rerun vs q8prev run 1 differs in prompts 3 and 4, and q8new rerun vs q8prev rerun is bitwise equal.
