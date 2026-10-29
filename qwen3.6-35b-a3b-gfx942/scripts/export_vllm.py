"""Copy the kernel modules into a vLLM checkout's Qwen3.6-35B-A3B specialized decode.

common/scripts/export_lib.py does the shared steps: the copy ships common/mkcore as
kernels/mkcore (AITER's FlyDSL buffer_ops instead of mkcore/vendor), and ruff sorts the
imports and formats the copy with vLLM's config. Model rules here: of mk/pack.py the copy
keeps only the tile layout functions (runtime/weights.py packs vLLM's loaded tensors with
them); the checkpoint loader `Packed` needs mk/reference.py and stays here; ruff drops
the imports the cut left unused.
Usage: python scripts/export_vllm.py VLLM_DIR
"""

import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common" / "scripts"))
import export_lib as E  # noqa: E402

MODULES = ("dims", "stages", "mixer", "moe", "kernel", "pack")
PACK_KEEP = {"gu_order", "row_scales", "fp8_tiles", "pack_bf16", "unit_order"}
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
    (dst / "__init__.py").write_text(E.SPDX)
    files = E.write_modules(src, dst, MODULES, E.SPDX,
                            lambda name, s: pack_layouts(s) if name == "pack" else s,
                            ("reference", "runner", "fp4"))
    files += E.copy_mkcore(dst, E.SPDX)
    E.ruff(root, [str(dst / "pack.py")], "check", "--fix", "--select", "F401")
    E.ruff(root, files, "check", "--fix", "--select", "I")
    E.ruff(root, files, "format")
    print("exported", ", ".join(MODULES), "and mkcore to", dst)


if __name__ == "__main__":
    main()
