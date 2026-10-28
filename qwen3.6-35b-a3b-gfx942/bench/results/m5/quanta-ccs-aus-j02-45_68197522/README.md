# quanta-ccs-aus-j02-45, job 68197522: first MI325X attempt (environment failure)

The kernel tests passed: `test_decode` at M = 1, 2, 4, 8, determinism, and perplexity.
`ref_vs_hf.txt` failed in HF transformers' `F.conv1d` with `miopenStatusUnknownError`.
The job ran without a home directory, so MIOpen had no writable user database or cache.
`scripts/suite_austin.sh` now sets `HOME` to the job's `/tmp` directory. The repeat run,
job 68197689 on the same node, passes every gate.
