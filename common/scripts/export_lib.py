"""Shared steps of the models' scripts/export_vllm.py (common/DESIGN.md D6): copy kernel
modules into a vLLM checkout's `.../specialized_decode/kernels/`, ship common/mkcore as
`kernels/mkcore/` (without vendor/: vLLM uses AITER's FlyDSL buffer_ops), rewrite the
imports, and run ruff with vLLM's config from the vLLM root ($RUFF, or ruff on PATH; the
version in vLLM's pre-commit config)."""
import os
import pathlib
import re
import shutil
import subprocess

COMMON = pathlib.Path(__file__).resolve().parent.parent
REPO = COMMON.parent
MKCORE = COMMON / "mkcore"
SPDX = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
)
AITER_BO = "from aiter.ops.flydsl.kernels import buffer_ops as bo"


def rewrite(s: str) -> str:
    """Standalone imports -> the vLLM copy's: mkcore.vendor's buffer_ops -> AITER's,
    `mkcore` -> the package-relative `.mkcore`."""
    s = s.replace("from mkcore.vendor import buffer_ops as bo", AITER_BO)
    s = s.replace("from mkcore import ", "from .mkcore import ")
    s = s.replace("from mkcore.", "from .mkcore.")
    return s


def write_modules(src: pathlib.Path, dst: pathlib.Path, modules, header: str,
                  transform=None, excluded=()) -> list:
    """Copy src/<name>.py for each module to dst with `header`, after rewrite() and
    transform(name, text). Asserts that no excluded module is imported and that no vendor
    import is left. Returns the written paths."""
    dst.mkdir(parents=True, exist_ok=True)
    files = []
    for name in modules:
        s = rewrite((src / f"{name}.py").read_text())
        if transform:
            s = transform(name, s)
        for m in excluded:
            assert not re.search(rf"from \.{m}\b|import {m}\b", s), (name, m)
        assert "vendor" not in s, name
        (dst / f"{name}.py").write_text(header + s)
        files.append(str(dst / f"{name}.py"))
    return files


def copy_mkcore(dst: pathlib.Path, header: str) -> list:
    """common/mkcore/*.py (not vendor/) -> dst/mkcore/. The __init__.py header names the
    megakernels commit and the standalone SRC_HASH. Returns the written paths."""
    out = dst / "mkcore"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rev = REPO / "REVISION"  # shipped trees (common/scripts/mkrun) have no .git
    commit = rev.read_text().strip() if rev.exists() else subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip() or "?"
    dirty = "" if rev.exists() else subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "common"],
        capture_output=True, text=True).stdout.strip()
    ns = {"__file__": str(MKCORE / "__init__.py")}
    exec(compile((MKCORE / "__init__.py").read_text(), ns["__file__"], "exec"), ns)
    files = []
    for p in sorted(MKCORE.glob("*.py")):
        s = p.read_text().replace("from .vendor import buffer_ops as bo", AITER_BO)
        assert "vendor" not in s, p
        # a module that moved into mkcore must not stay behind as a model module
        if p.name != "__init__.py":
            (dst / p.name).unlink(missing_ok=True)
        h = header + "# ruff: noqa: E501\n"
        if p.name == "__init__.py":
            h += (f"# megakernels {commit}{' (dirty)' if dirty else ''}, common/mkcore "
                  f"SRC_HASH {ns['SRC_HASH'][:16]}\n")
        (out / p.name).write_text(h + s)
        files.append(str(out / p.name))
    return files


def ruff(root: pathlib.Path, files, *args):
    """ruff <args> --config <root>/pyproject.toml <files>, from the vLLM root (ruff's
    import sections depend on the working directory)."""
    exe = os.environ.get("RUFF") or shutil.which("ruff")
    assert exe, "set RUFF to a ruff binary (the version in vLLM's pre-commit config)"
    cfg = ["--config", str(root / "pyproject.toml")]
    subprocess.run([exe, args[0], *cfg, *args[1:], *files], check=True, cwd=root)
