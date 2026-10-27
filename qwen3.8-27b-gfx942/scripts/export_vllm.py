"""Copy the kernel modules into a vLLM checkout's Qwen3.8 specialized decode.

The vLLM copy imports AITER's FlyDSL buffer_ops instead of mk/vendor, and drops the
test helper that needs mk/reference.py. Usage: python scripts/export_vllm.py VLLM_DIR
"""
import pathlib
import re
import sys

MODULES = ("dims", "dev", "fp4", "gemv", "stages", "mixer", "kernel")
HEADER = ("# SPDX-License-Identifier: Apache-2.0\n"
          "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n")


def main():
    src = pathlib.Path(__file__).resolve().parent.parent / "mk"
    dst = pathlib.Path(sys.argv[1]) / "vllm/models/qwen3_5/amd/specialized_decode/kernels"
    dst.mkdir(parents=True, exist_ok=True)
    for name in MODULES:
        s = (src / f"{name}.py").read_text()
        s = s.replace("from .vendor import buffer_ops as bo",
                      "from aiter.ops.flydsl.kernels import buffer_ops as bo")
        if name == "fp4":       # unpack_ref is a test helper of the standalone repo
            s = re.sub(r"\ndef unpack_ref\(.*?\n\n\n", "\n", s, flags=re.S)
        assert "from .reference" not in s and "vendor" not in s, name
        (dst / f"{name}.py").write_text(HEADER + s)
    print("exported", ", ".join(MODULES), "to", dst)


if __name__ == "__main__":
    main()
