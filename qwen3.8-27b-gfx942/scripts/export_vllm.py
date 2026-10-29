"""Copy the kernel modules into a vLLM checkout's Qwen3.8 specialized decode.

The vLLM copy imports AITER's FlyDSL buffer_ops instead of mk/vendor, and drops the
test helper that needs mk/reference.py. The TP all-reduce uses vLLM's DSv4.1 mailbox:
the copy imports `Mailbox` and `preg` from vllm/models/deepseek_v41/amd/mono/common/
sync.py, not from mk/sync.py. mk/sync.py is a copy of that file for the standalone
repository; the script stops when the two differ from ``POLL_MAX`` on. mk/sync.py,
mk/peer.py and mk/tp.py are not exported (vLLM has its own PeerBuffer and TP group).
Then ruff ($RUFF, or ruff on PATH) sorts the imports and formats the copy with vLLM's
config.
Usage: python scripts/export_vllm.py VLLM_DIR
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys

MODULES = ("dims", "dev", "fp4", "gemv", "stages", "mixer", "kernel")
NOT_EXPORTED = ("sync", "peer", "tp", "runner", "pack", "reference", "vendor")
VLLM_SYNC = "vllm/models/deepseek_v41/amd/mono/common/sync.py"
VLLM_SYNC_MODULE = VLLM_SYNC[: -len(".py")].replace("/", ".")
HEADER = ("# SPDX-License-Identifier: Apache-2.0\n"
          "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n")


def mailbox_code(path):
    """The mailbox code of a sync.py, from ``POLL_MAX`` on (the imports differ)."""
    s = path.read_text()
    i = s.find("\nPOLL_MAX = ")
    assert i >= 0, f"{path}: no POLL_MAX"
    return s[i:]


def main():
    root = pathlib.Path(sys.argv[1])
    src = pathlib.Path(__file__).resolve().parent.parent / "mk"
    dst = root / "vllm/models/qwen3_5/amd/specialized_decode/kernels"
    assert mailbox_code(src / "sync.py") == mailbox_code(root / VLLM_SYNC), (
        f"mk/sync.py differs from {VLLM_SYNC} from POLL_MAX on: update the copy"
    )
    dst.mkdir(parents=True, exist_ok=True)
    for name in MODULES:
        s = (src / f"{name}.py").read_text()
        s = s.replace("from .vendor import buffer_ops as bo",
                      "from aiter.ops.flydsl.kernels import buffer_ops as bo")
        s = s.replace("from .sync import ", f"from {VLLM_SYNC_MODULE} import ")
        s = s.replace("from mkcore import ", "from .mkcore import ")
        if name == "fp4":       # unpack_ref is a test helper of the standalone repo
            s = re.sub(r"\ndef unpack_ref\(.*?\n\n\n", "\n", s, flags=re.S)
        for m in NOT_EXPORTED:
            assert not re.search(rf"from \.{m}\b|import {m}\b", s), (name, m)
        assert "vendor" not in s, name
        (dst / f"{name}.py").write_text(HEADER + s)
    ruff = os.environ.get("RUFF") or shutil.which("ruff")
    assert ruff, "set RUFF to a ruff binary (the version in vLLM's pre-commit config)"
    files = [str(dst / f"{name}.py") for name in MODULES]
    # common/mkcore ships as kernels/mkcore (common/DESIGN.md D6); M3 moves this into
    # common/scripts/export_lib.py
    (dst / "mkcore").mkdir(exist_ok=True)
    for p in sorted((src.parent.parent / "common" / "mkcore").glob("*.py")):
        (dst / "mkcore" / p.name).write_text(HEADER + p.read_text())
        files.append(str(dst / "mkcore" / p.name))
    cfg = ["--config", str(root / "pyproject.toml")]
    # from the vLLM root: ruff's import sections depend on the working directory
    run = lambda *a: subprocess.run([ruff, *a, *files], check=True, cwd=root)  # noqa: E731
    run("check", *cfg, "--fix", "--select", "I")
    run("format", *cfg)
    print("exported", ", ".join(MODULES), "to", dst)
    # vLLM's full rule set, as its pre-commit runs it: a finding fails the export
    run("check", *cfg)


if __name__ == "__main__":
    main()
