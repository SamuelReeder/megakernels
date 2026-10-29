"""Spike rtgeom (disposable): compile one Qwen3.8 decode Cfg with FlyDSL COMPILE_ONLY and
no GPU (FLYDSL_GPU_ARCH=gfx942), write the code object, its llvm-readelf notes and the
final ISA (FLYDSL_DUMP_IR) with the kernel symbol replaced by KERNEL.
Usage (from qwen3.8-27b-gfx942/): python ../spike/rtgeom/co_build.py OUT M MAXLEN BLOCK XPACK RT [LM]
  XPACK, RT, LM: 0 | 1 (LM default 1). The geometry follows mk.runner.Engine (TP1).
Env MK_KPKG=<vllm>/vllm/models/qwen3_5/amd/specialized_decode/kernels: build that package's
kernel (vLLM's copy) instead of mk.kernel.
Prints one line: CO <tag> name sha16 isa_sha16 sgpr vgpr sgpr_spill vgpr_spill lds."""
import importlib
import glob
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

out = pathlib.Path(sys.argv[1])
M, maxlen, block, xpack, rt = (int(a) for a in sys.argv[2:7])
lm = bool(int(sys.argv[7])) if len(sys.argv) > 7 else True
dump = tempfile.mkdtemp(prefix="rtg_dump_")
os.environ.update(FLYDSL_GPU_ARCH="gfx942", COMPILE_ONLY="1", FLYDSL_RUNTIME_ENABLE_CACHE="0",
                  FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=dump)
sys.path.insert(0, os.getcwd())
from flydsl._mlir import ir  # noqa: E402
from flydsl._mlir.dialects import gpu  # noqa: E402
from flydsl.compiler.jit_function import _create_mlir_context  # noqa: E402

if os.environ.get("MK_KPKG"):
    kp = pathlib.Path(os.environ["MK_KPKG"]).resolve()
    sys.path.insert(0, str(kp.parent))
    KK = importlib.import_module(f"{kp.name}.kernel")
else:
    from mk import kernel as KK  # noqa: E402
from mk.dims import DK, DV, HD, Dims  # noqa: E402

d = Dims(1)
page = 3 * d.conv_dim * 2 + d.nvh * DV * DK * 4
nb = -(-maxlen // block)
kv_s = (2 * block * d.nkv * HD, HD * (block if xpack else 1), d.nkv * HD, block * d.nkv * HD)
cfg = KK.Cfg(M, 304, maxlen, block, nb, *kv_s, bool(xpack), page, page, lm_head=lm,
             **(dict(rt_geom=rt) if rt else {}))
launch = KK.build(cfg)
launch(0, 0, 64)
_, art = launch._last_compiled
objs = []


def visit(op):
    if op.name == "gpu.binary":
        for o in ir.ArrayAttr(op.attributes["objects"]):
            objs.append(gpu.ObjectAttr(o).object)
    return ir.WalkResult.ADVANCE


with _create_mlir_context():
    ir.Module.parse(art._ir_text).operation.walk(visit)
assert len(objs) == 1 and objs[0][:4] == b"\x7fELF"
tag = f"{'v' if os.environ.get('MK_KPKG') else ''}m{M}_L{maxlen}_b{block}_x{xpack}_rt{rt}_lm{int(lm)}"
out.mkdir(parents=True, exist_ok=True)
co = out / f"{tag}.co"
co.write_bytes(objs[0])
rd = shutil.which("llvm-readelf") or "/opt/rocm/llvm/bin/llvm-readelf"
notes = subprocess.run([rd, "--notes", str(co)], capture_output=True, text=True).stdout
(out / f"{tag}_notes.txt").write_text(notes)
isa = glob.glob(f"{dump}/*/22_final_isa.s")
isa_sha = "none"
if isa:
    s = re.sub(r"qwen38_decode_m[0-9]+_[0-9a-f]{8}", "KERNEL", open(isa[0]).read())
    s = "\n".join(x for x in s.splitlines() if not re.match(r"^\s*(;|//)", x))
    (out / f"{tag}.s").write_text(s)
    isa_sha = hashlib.sha256(s.encode()).hexdigest()[:16]
shutil.rmtree(dump, ignore_errors=True)


def note(k):
    m = re.search(rf"\.{k}:\s+(\d+)", notes)
    return m.group(1) if m else "?"


name = re.search(r"\.name:\s+(qwen38_decode\S+)", notes)
print(f"CO {tag} {name.group(1) if name else '?'} {hashlib.sha256(objs[0]).hexdigest()[:16]} "
      f"isa {isa_sha} sgpr {note('sgpr_count')} vgpr {note('vgpr_count')} "
      f"sgpr_spill {note('sgpr_spill_count')} vgpr_spill {note('vgpr_spill_count')} "
      f"lds {note('group_segment_fixed_size')} cfg {cfg}", flush=True)
