"""Oracle 5 (common/DESIGN.md "Test strategy"): build a model's kernel from its vLLM
export and run it once, so FlyDSL dumps the ISA of the exported code. The standalone
engine (mk.runner, mk.pack, mk.reference) drives the launch; mk.kernel is replaced by the
exported `kernels.kernel` module, which traces the exported modules only (kernels/*.py,
kernels/mkcore, AITER's buffer_ops). KERNELS_MODULE "mk" builds the standalone kernel in
the same environment. Runs tests/logits_hash.py at M = 1, 1 step, 4 layers, lm_head on.
Usage (vLLM image, the vLLM tree first on PYTHONPATH, FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=..):
  python export_isa.py MODEL_DIR KERNELS_MODULE WEIGHTS
"""
import importlib
import os
import runpy
import sys

model_dir, kmod, weights = sys.argv[1:4]
sys.path.insert(0, model_dir)
import mk  # noqa: E402  (puts common/ on sys.path)

if kmod != "mk":
    kk = importlib.import_module(f"{kmod}.kernel")
    sys.modules["mk.kernel"] = kk
    mk.kernel = kk
    print("kernel module", kk.__file__, flush=True)
args = [weights, "1", "1", "4"] + ([] if os.path.exists(f"{model_dir}/mk/fp4.py") else ["on"])
sys.argv = ["logits_hash.py", *args]
runpy.run_path(os.path.join(model_dir, "tests", "logits_hash.py"), run_name="__main__")
