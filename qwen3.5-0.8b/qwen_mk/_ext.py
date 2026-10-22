"""JIT-build the CUDA extension on first use (cached by torch under ~/.cache/torch_extensions)."""

import functools
import hashlib
import os

CSRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")


@functools.cache
def load(verbose=False):
    from torch.utils.cpp_extension import load as _load

    arch = os.environ.get("QWEN_MK_ARCH", "120a")
    defines = os.environ.get("QWEN_MK_DEFINES", "").split()  # e.g. "-DMK_STAGES=6" for tuning runs
    name = "qwen_mk_ext"
    if defines:
        name += "_" + hashlib.sha1(" ".join(defines).encode()).hexdigest()[:8]
    return _load(
        name=name,
        sources=[os.path.join(CSRC, "binding.cpp"), os.path.join(CSRC, "megakernel.cu")],
        # --fmad=false: every FMA is written out (fmaf), so the compiler can't fuse the one- and two-token
        # instantiations of the same expression differently. Speculative decoding relies on them rounding
        # identically to reproduce plain decoding bit for bit.
        extra_cuda_cflags=["-O3", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}", "--expt-relaxed-constexpr",
                           "--fmad=false", *defines],
        extra_cflags=["-O3"],
        extra_include_paths=[CSRC],
        verbose=verbose,
    )
