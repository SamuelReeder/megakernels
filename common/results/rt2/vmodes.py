"""rt2 evidence: per bench/vllm_e2e.py result (Qwen3.8), a short sha1 of each prompt's
(tokens, logprobs), and prompt 1's logprob at position 1, so runs of different trees can
be grouped by output: equal digits = bitwise equal output for that prompt.
Usage: python common/results/rt2/vmodes.py RESULT.json..."""
import hashlib
import json
import sys

for f in sys.argv[1:]:
    o = json.load(open(f))["outputs"]
    sig = [hashlib.sha1(json.dumps([x["tokens"], x["logprobs"]]).encode()).hexdigest()[:8]
           for x in o]
    print(f"{f}: {' '.join(sig)} | prompt 1 logprob[1] {o[1]['logprobs'][1]:.6f}")
