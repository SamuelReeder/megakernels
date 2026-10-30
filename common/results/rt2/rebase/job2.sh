#!/bin/bash
# rt2 after the rebase: the q8 eval_text gate of common/scripts/gates.sh (same command),
# which job1 could not run (the BF16 reference checkpoint reached the node during job1;
# q6 and g ran after it). From the tree root, hipdnn image + venv.
# Output: /work/out/rb1/gates/q8/eval_text_rerun.txt, GATE line in summary_eval_text.txt.
set -u
O=/work/out/rb1/gates
cd qwen3.8-27b-gfx942 && python -B tests/eval_text.py "$M/Qwen3.8-27B-Quark-AWQ-MXFP4" "$M/Qwen3.8-27B" > "$O/q8/eval_text_rerun.txt" 2>&1
rc=$?
echo "GATE q8 eval_text $([ $rc = 0 ] && echo PASS || echo "FAIL rc=$rc")" | tee -a "$O/summary_eval_text.txt"
