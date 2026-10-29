"""Copy the kernel modules into a vLLM checkout's Qwen3.8 specialized decode.

common/scripts/export_lib.py does the shared steps: the copy ships common/mkcore as
kernels/mkcore (AITER's FlyDSL buffer_ops instead of mkcore/vendor), and ruff sorts the
imports, formats the copy with vLLM's config and runs vLLM's full rule set. Model rules
here: the copy drops the test helper that needs mk/reference.py. The TP all-reduce uses
vLLM's DSv4.1 mailbox: the copy imports `Mailbox` and `preg` from
vllm/models/deepseek_v41/amd/mono/common/sync.py, not from mk/sync.py. mk/sync.py is a
copy of that file for the standalone repository; the script stops when the two differ
from ``POLL_MAX`` on. mk/sync.py, mk/peer.py and mk/tp.py are not exported (vLLM has its
own PeerBuffer and TP group).
Usage: python scripts/export_vllm.py VLLM_DIR
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common" / "scripts"))
import export_lib as E  # noqa: E402

MODULES = ("dims", "tpdev", "fp4", "gemv", "stages", "mixer", "kernel")
NOT_EXPORTED = ("sync", "peer", "tp", "runner", "pack", "reference")
VLLM_SYNC = "vllm/models/deepseek_v41/amd/mono/common/sync.py"
VLLM_SYNC_MODULE = VLLM_SYNC[: -len(".py")].replace("/", ".")


def mailbox_code(path):
    """The mailbox code of a sync.py, from ``POLL_MAX`` on (the imports differ)."""
    s = path.read_text()
    i = s.find("\nPOLL_MAX = ")
    assert i >= 0, f"{path}: no POLL_MAX"
    return s[i:]


def transform(name, s):
    s = s.replace("from .sync import ", f"from {VLLM_SYNC_MODULE} import ")
    if name == "fp4":  # unpack_ref is a test helper of the standalone repo
        s = re.sub(r"\ndef unpack_ref\(.*?\n\n\n", "\n", s, flags=re.S)
    return s


def main():
    root = pathlib.Path(sys.argv[1])
    src = pathlib.Path(__file__).resolve().parent.parent / "mk"
    dst = root / "vllm/models/qwen3_5/amd/specialized_decode/kernels"
    assert mailbox_code(src / "sync.py") == mailbox_code(root / VLLM_SYNC), (
        f"mk/sync.py differs from {VLLM_SYNC} from POLL_MAX on: update the copy"
    )
    files = E.write_modules(src, dst, MODULES, E.SPDX, transform, NOT_EXPORTED)
    files += E.copy_mkcore(dst, E.SPDX)
    E.ruff(root, files, "check", "--fix", "--select", "I")
    E.ruff(root, files, "format")
    print("exported", ", ".join(MODULES), "and mkcore to", dst)
    # vLLM's full rule set, as its pre-commit runs it: a finding fails the export
    E.ruff(root, files, "check")


if __name__ == "__main__":
    main()
