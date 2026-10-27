"""Copy the kernel modules into a vLLM checkout's Qwen3.8 specialized decode.

The vLLM copy imports AITER's FlyDSL buffer_ops instead of mk/vendor, and drops the
test helper that needs mk/reference.py. Then ruff ($RUFF, or ruff on PATH) sorts the
imports and formats the copy with vLLM's config.
Usage: python scripts/export_vllm.py VLLM_DIR
"""
import os
import pathlib
import re
import shutil
import subprocess
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
    ruff = os.environ.get("RUFF") or shutil.which("ruff")
    assert ruff, "set RUFF to a ruff binary (the version in vLLM's pre-commit config)"
    files = [str(dst / f"{name}.py") for name in MODULES]
    cfg = ["--config", str(pathlib.Path(sys.argv[1]) / "pyproject.toml")]
    # from the vLLM root: ruff's import sections depend on the working directory
    run = lambda *a: subprocess.run([ruff, *a, *files], check=True, cwd=sys.argv[1])  # noqa: E731
    run("check", *cfg, "--fix", "--select", "I")
    run("format", *cfg)
    print("exported", ", ".join(MODULES), "to", dst)


if __name__ == "__main__":
    main()
