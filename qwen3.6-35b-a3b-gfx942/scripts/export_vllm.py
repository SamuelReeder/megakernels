"""Copy the kernel modules into a vLLM checkout's Qwen3.6-35B-A3B specialized decode.

The vLLM copy imports AITER's FlyDSL buffer_ops instead of mk/vendor. Of mk/pack.py it
keeps only the tile layout functions (runtime/weights.py packs vLLM's loaded tensors with
them); the checkpoint loader `Packed` needs mk/reference.py and stays here. Then ruff
($RUFF, or ruff on PATH; the version in vLLM's pre-commit config) drops the imports the
cut left unused, sorts the imports and formats the copy with vLLM's config.
Usage: python scripts/export_vllm.py VLLM_DIR
"""

import ast
import os
import pathlib
import shutil
import subprocess
import sys

MODULES = ("dims", "dev", "gemv", "stages", "mixer", "moe", "kernel", "pack")
PACK_KEEP = {"gu_order", "row_scales", "fp8_tiles", "pack_bf16", "unit_order"}
HEADER = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
)
DST = "vllm/models/qwen3_5_moe/amd/specialized_decode/kernels"


def pack_layouts(s: str) -> str:
    """mk/pack.py with only its docstring, imports and the PACK_KEEP functions."""
    tree = ast.parse(s)
    lines = s.splitlines(keepends=True)
    out, kept = [], set()
    for node in tree.body:
        keep = isinstance(node, (ast.Import, ast.ImportFrom)) or (
            isinstance(node, ast.Expr) and node is tree.body[0]  # the docstring
        )
        if isinstance(node, ast.FunctionDef) and node.name in PACK_KEEP:
            keep = True
            kept.add(node.name)
        if keep:
            out.append("".join(lines[node.lineno - 1 : node.end_lineno]) + "\n\n")
    assert kept == PACK_KEEP, f"mk/pack.py lacks {sorted(PACK_KEEP - kept)}"
    return "".join(out)


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    src = pathlib.Path(__file__).resolve().parent.parent / "mk"
    dst = root / DST
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "__init__.py").write_text(HEADER)
    for name in MODULES:
        s = (src / f"{name}.py").read_text()
        s = s.replace(
            "from .vendor import buffer_ops as bo",
            "from aiter.ops.flydsl.kernels import buffer_ops as bo",
        )
        if name == "pack":
            s = pack_layouts(s)
        for mod in ("reference", "runner", "fp4"):
            assert f"from .{mod}" not in s and f"import {mod}" not in s, (name, mod)
        assert "vendor" not in s, name
        (dst / f"{name}.py").write_text(HEADER + s)
    ruff = os.environ.get("RUFF") or shutil.which("ruff")
    assert ruff, "set RUFF to a ruff binary (the version in vLLM's pre-commit config)"
    files = [str(dst / f"{name}.py") for name in MODULES]
    cfg = ["--config", str(root / "pyproject.toml")]

    # from the vLLM root: ruff's import sections depend on the working directory
    def run(*a, files=files):
        subprocess.run([ruff, *a, *files], check=True, cwd=root)

    run("check", *cfg, "--fix", "--select", "F401", files=[str(dst / "pack.py")])
    run("check", *cfg, "--fix", "--select", "I")
    run("format", *cfg)
    print("exported", ", ".join(MODULES), "to", dst)


if __name__ == "__main__":
    main()
