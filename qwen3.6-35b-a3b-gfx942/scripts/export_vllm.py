"""Copy the kernel modules into a vLLM checkout's Qwen3.6-35B-A3B specialized decode.

common/scripts/export_lib.py does the shared steps: the copy ships common/mkcore as
kernels/mkcore (AITER's FlyDSL buffer_ops instead of mkcore/vendor; runtime/weights.py
packs vLLM's loaded tensors with kernels/mkcore/pack.py), and ruff sorts the imports,
formats the copy with vLLM's config and checks it. mk/pack.py (the checkpoint loader
`Packed`, needs mk/reference.py) is not exported.
Usage: python scripts/export_vllm.py VLLM_DIR
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common" / "scripts"))
import export_lib as E  # noqa: E402

MODULES = ("dims", "stages", "mixer", "moe", "kernel")
DST = "vllm/models/qwen3_5_moe/amd/specialized_decode/kernels"


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    src = pathlib.Path(__file__).resolve().parent.parent / "mk"
    dst = root / DST
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "__init__.py").write_text(E.SPDX)
    files = E.write_modules(src, dst, MODULES, E.SPDX, None,
                            ("reference", "runner", "pack", "fp4"))
    files += E.copy_mkcore(dst, E.SPDX)
    E.ruff(root, files, "check", "--fix", "--select", "I")
    E.ruff(root, files, "format")
    E.ruff(root, files, "check")
    print("exported", ", ".join(MODULES), "and mkcore to", dst)


if __name__ == "__main__":
    main()
