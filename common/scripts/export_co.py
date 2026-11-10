"""Export the decode megakernels of a vLLM checkout as code objects plus the sha256 lock
(docs/hipdnn-route.md sections 4 and 6, items 1-2).

The kernels come from the checkout's vendored `kernels/` package, not from this repository:
for each deployment the script starts vLLM on that checkout (dummy weights, the model's
config), lets the DecodeSpec bind the caches, and takes each width's `Cfg` from the
runtime object (`kernel.cfgs`), so every `Cfg` is the one the vLLM runtime builds. It
compiles each `Cfg` with FlyDSL (`COMPILE_ONLY=1`), takes the AMDGPU ELF from the lowered
module (`gpu.binary`), and computes `src` with the checkout's own
`kernels/mkcore/key.compile_key(cfg.compiled(), <kernels dir>)` (`cfg` where the checkout
has no `Cfg.compiled()`: rt_geom 0).

Output (OUT): `<model>_M<M>_<src[:16]>.co` per object and `manifest.json`, a list of
entries with exactly the keys of MANIFEST_KEYS, sorted; two runs on one FlyDSL version
give the same bytes. `export.log`: every Cfg, its compiled() form and the vLLM settings.

Usage (in the vLLM image, the checkout's vllm/ first on PYTHONPATH: common/scripts/vrun):
  python export_co.py --model MODEL --weights DIR --vllm-dir DIR --vllm-commit SHA
         --mk-commit SHA --out OUT --deploy SPEC [--deploy SPEC ...]
  SPEC: kv=ROCM_ATTN,block=784,maxlen=4096,tp=1,M=1:2:4:8,cus=304 (maxlen: one or a
  :-list). kv: ROCM_ATTN (x-packed), TRITON_ATTN:1 ([blocks, nkv, block, 2 hd]) or
  TRITON_ATTN:2 ([blocks, 2, block, nkv hd]) of the model's first attention layer; block:
  that layer's KV block. One vLLM process per (deployment, maxlen).
"""
import argparse
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import pathlib
import re
import struct
import subprocess
import sys

MODELS = ("qwen38-27b-mxfp4-gfx942", "qwen36-35b-a3b-fp8-gfx942", "gemma4-26b-a4b-fp8-gfx942")
MANIFEST_KEYS = ("model", "src", "arch", "cus", "kv", "block", "M", "tp", "rt", "lm_head",
                 "symbol", "block_threads", "lds_bytes", "kernarg_bytes", "sha256", "size",
                 "file", "flydsl_version", "vllm_commit", "megakernels_commit")
# the launch ABI of every megakernel: tab (i64), glob (i64), nlayers (i32), by value
KERNARGS = [(0, 8), (8, 8), (16, 4)]
NT_AMDGPU_METADATA = 32
SELF = pathlib.Path(__file__).resolve()  # absolute: workers and --compile processes


def parse_deploy(spec: str) -> dict:
    d = dict(kv.split("=", 1) for kv in spec.split(","))
    out = dict(kv=d["kv"], block=int(d["block"]), tp=int(d["tp"]), cus=int(d["cus"]),
               M=sorted(int(m) for m in d["M"].split(":")),
               maxlen=[int(x) for x in d["maxlen"].split(":")])
    assert set(d) == {"kv", "block", "maxlen", "tp", "M", "cus"}, f"deployment keys: {spec}"
    assert out["kv"] in ("ROCM_ATTN", "TRITON_ATTN:1", "TRITON_ATTN:2"), out["kv"]
    return out


def llm_kwargs(weights: str, kv: str, maxlen: int, tp: int, dummy: bool) -> dict:
    """The engine arguments of the models' bench/vllm_e2e.py (kernel on)."""
    kw = dict(model=weights, max_model_len=maxlen, tensor_parallel_size=tp,
              attention_backend=kv.split(":")[0], language_model_only=True,
              enable_prefix_caching=False, gpu_memory_utilization=0.85, max_num_seqs=8,
              seed=0, specialized_decode="monokernel")
    if dummy:
        kw["load_format"] = "dummy"
    return kw


# ------------------------------------------------------------------ ELF and FlyDSL
def _msgpack_decode(b: bytes):
    try:
        import msgspec
        return msgspec.msgpack.decode(b)
    except ImportError:
        import msgpack
        return msgpack.unpackb(b, raw=False)


def elf_kernel_meta(elf: bytes) -> dict:
    """The one kernel of an AMDGPU code object, from its NT_AMDGPU_METADATA note."""
    assert elf[:4] == b"\x7fELF" and elf[4] == 2 and elf[5] == 1, "not an ELF64 LE object"
    shoff, = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", elf, 0x3A)
    meta = None
    for i in range(shnum):
        sh = shoff + i * shentsize
        sh_type, = struct.unpack_from("<I", elf, sh + 4)
        if sh_type != 7:  # SHT_NOTE
            continue
        off, size = struct.unpack_from("<QQ", elf, sh + 0x18)
        p, end = off, off + size
        while p + 12 <= end:
            namesz, descsz, ntype = struct.unpack_from("<III", elf, p)
            name = elf[p + 12:p + 12 + namesz].rstrip(b"\0")
            desc0 = p + 12 + ((namesz + 3) & ~3)
            if name == b"AMDGPU" and ntype == NT_AMDGPU_METADATA:
                meta = _msgpack_decode(elf[desc0:desc0 + descsz])
            p = desc0 + ((descsz + 3) & ~3)
    assert meta is not None, "no AMDGPU metadata note"
    ks = meta["amdhsa.kernels"]
    assert len(ks) == 1, [k[".name"] for k in ks]
    k = ks[0]
    args = [(a[".offset"], a[".size"]) for a in k[".args"]]
    kinds = {a[".value_kind"] for a in k[".args"]}
    assert args == KERNARGS and kinds == {"by_value"}, (args, kinds)  # no hidden arguments
    return dict(target=meta["amdhsa.target"], symbol=k[".name"],
                kernarg_bytes=k[".kernarg_segment_size"],
                block_threads=k[".reqd_workgroup_size"][0],
                group_segment=k[".group_segment_fixed_size"])


def elf_of(launcher) -> tuple[bytes, str, int]:
    """(ELF, kernel symbol, dynamic LDS bytes) of a FlyDSL launcher's last compile: the
    gpu.binary object of the lowered module and its gpu.launch_func."""
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import gpu
    from flydsl.compiler.jit_function import _create_mlir_context

    _, art = launcher._last_compiled
    objs, launches = [], []

    def visit(op):
        if op.name == "gpu.binary":
            for o in ir.ArrayAttr(op.attributes["objects"]):
                objs.append(gpu.ObjectAttr(o).object)
        elif op.name == "gpu.launch_func":
            launches.append((str(op), ir.SymbolRefAttr(op.opview.kernel).value))
        return ir.WalkResult.ADVANCE

    with _create_mlir_context():
        ir.Module.parse(art._ir_text).operation.walk(visit)
    assert len(objs) == 1 and len(launches) == 1, (len(objs), len(launches))
    text, kref = launches[0]
    m = re.search(r"dynamic_shared_memory_size\s+%?(\w+)", text)
    assert m is None, f"dynamic LDS in the launch: {text}"  # FlyDSL kernels: static LDS only
    return bytes(objs[0]), str(kref[-1]), 0


def canonical_compile(job: dict) -> None:
    """`--compile` (a fresh process): the code object of one Cfg, independent of the
    process. FlyDSL traces and lowers it (COMPILE_ONLY, CPU only); the lowered module
    before `gpu-module-to-binary` (FlyDSL's IR dump) is printed without locations,
    parsed again, and serialized with FlyDSL's binary pass. FlyDSL 0.4.0's own binary
    varies from process to process for some kernels (register allocation; Qwen3.6 M = 1,
    2) although the printed IR is the same: the in-memory module differs, and a parsed
    module does not (NOTES.md). A fresh process also keeps the kernel name a function of
    the Cfg: tracing sets mkcore.dev.PROBE["on"], which the next build's key holds.
    Writes job['out'] and prints `COMPILED {json}`."""
    import tempfile

    dump = tempfile.mkdtemp(prefix="co_dump_")
    os.environ.update(COMPILE_ONLY="1", FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=dump,
                      FLYDSL_COMPILE_ARCH=job["arch"], FLYDSL_GPU_ARCH=job["arch"],
                      FLYDSL_RUNTIME_ENABLE_CACHE="0")
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import gpu
    from flydsl._mlir.passmanager import PassManager
    from flydsl.compiler.backends import get_backend
    from flydsl.compiler.jit_function import _create_mlir_context

    kmod = importlib.import_module(job["module"])
    cfg = eval(job["cfg"], {"Cfg": kmod.Cfg})  # noqa: S307 (our own repr)
    assert repr(cfg) == job["cfg"], (repr(cfg), job["cfg"])
    launcher = kmod.build(cfg)
    assert launcher(0, 0, job["nlayers"]) is None
    _, sym, lds = elf_of(launcher)
    stages = sorted(pathlib.Path(dump).glob("*/[0-9][0-9]_*.mlir"))
    nbin = [i for i, p in enumerate(stages) if p.name[3:] == "gpu_module_to_binary.mlir"]
    assert len(nbin) == 1 and nbin[0] > 0, [p.name for p in stages]
    lowered = stages[nbin[0] - 1].read_text()
    fragment = get_backend(arch=job["arch"]).external_binary_pipeline_fragments(compile_hints={})[1]
    with _create_mlir_context():
        text = ir.Module.parse(lowered).operation.get_asm(enable_debug_info=False)
    objs = []
    with _create_mlir_context():
        m = ir.Module.parse(text)
        PassManager.parse(f"builtin.module({fragment})").run(m.operation)

        def visit(op):
            if op.name == "gpu.binary":
                objs.extend(bytes(gpu.ObjectAttr(o).object) for o in ir.ArrayAttr(op.attributes["objects"]))
            return ir.WalkResult.ADVANCE

        m.operation.walk(visit)
    assert len(objs) == 1, len(objs)
    pathlib.Path(job["out"]).write_bytes(objs[0])
    for p in sorted(pathlib.Path(dump).rglob("*"), reverse=True):
        p.rmdir() if p.is_dir() else p.unlink()
    os.rmdir(dump)
    print("COMPILED " + json.dumps(dict(symbol=sym, lds_bytes=lds, fragment=fragment)), flush=True)


def compile_fresh(kmod, cfg, nlayers: int, arch: str, out: str) -> tuple[bytes, dict]:
    """canonical_compile of one Cfg in a fresh process: (ELF, its COMPILED fields)."""
    job = dict(module=kmod.__name__, cfg=repr(cfg), nlayers=nlayers, arch=arch, out=out)
    r = subprocess.run([sys.executable, "-B", str(SELF), "--compile",
                        json.dumps(job)], capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("COMPILED ")]
    assert r.returncode == 0 and len(lines) == 1, f"compile {job}: {r.stdout[-3000:]}{r.stderr[-3000:]}"
    return pathlib.Path(out).read_bytes(), json.loads(lines[0][len("COMPILED "):])


# ------------------------------------------------------------------ runtime objects
def runtime(model):
    """(SpecializedDecode, its kernel object, the checkout's kernel and key modules,
    kernels/ dir, spec module, the launch's nlayers) of a loaded vLLM model."""
    sd = getattr(model, "specialized_decode", None)
    assert sd is not None, f"{type(model).__name__}: no specialized decode"
    dec = sd.kernel
    base = type(dec).__module__.rsplit(".runtime.", 1)[0]
    kmod = importlib.import_module(base + ".kernels.kernel")
    keymod = importlib.import_module(base + ".kernels.mkcore.key")
    spec = importlib.import_module(base + ".spec")
    nl = inspect.signature(dec.launch).parameters["layers"].default
    return sd, dec, kmod, keymod, pathlib.Path(kmod.__file__).resolve().parent, spec, nl


def src_key(keymod, kdir, cfg) -> str:
    c = cfg.compiled() if hasattr(cfg, "compiled") else cfg
    return keymod.compile_key(c, kdir)


def kv_class(sd, dec, spec) -> tuple[str, int]:
    """(kv layout class, KV block) of the model's first attention layer."""
    sfc = sd.vllm_config.compilation_config.static_forward_context
    name = dec.attn_names[0]
    layer, kv = sfc[name], sfc[name].kv_cache
    assert kv.dim() == 4, tuple(kv.shape)
    backend = spec.attention_backend(layer)
    if backend.startswith("ROCM_ATTN"):
        return backend, kv.shape[2]
    assert backend == "TRITON_ATTN", backend
    nkv, hd = layer.num_kv_heads, layer.head_size
    if kv.shape[1] == nkv and kv.shape[3] == 2 * hd:  # spec.kv_geometry's order
        return "TRITON_ATTN:1", kv.shape[2]
    assert kv.shape[1] == 2 and kv.shape[3] == nkv * hd, tuple(kv.shape)
    return "TRITON_ATTN:2", kv.shape[2]


def deployment_fields(sd, dec, spec) -> dict:
    import torch

    kv, block = kv_class(sd, dec, spec)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return dict(kv=kv, block=block, arch=props.gcnArchName.split(":")[0],
                cus=props.multi_processor_count, model=sd.spec.name,
                tp=sd.vllm_config.parallel_config.tensor_parallel_size,
                widths=list(sd.widths))


def _export_rank(model, out_dir: str, Ms: list) -> dict:
    """Worker side (llm.apply_model): compile every width's Cfg of this rank."""
    import torch

    sd, dec, kmod, keymod, kdir, spec, nl = runtime(model)
    dep = deployment_fields(sd, dec, spec)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    objs = []
    for w in Ms:
        cfg = dec.cfgs[w]
        src = src_key(keymod, kdir, cfg)
        f = f"{dep['model']}_M{w}_{src[:16]}.co"
        tmp = pathlib.Path(out_dir) / f".{f}.rank{rank}"
        elf, c = compile_fresh(kmod, cfg, nl, dep["arch"], str(tmp))
        tmp.unlink()
        sym, lds = c["symbol"], c["lds_bytes"]
        meta = elf_kernel_meta(elf)
        assert meta["symbol"] == sym, (meta["symbol"], sym)
        assert meta["target"].endswith("-" + dep["arch"]), (meta["target"], dep["arch"])
        sha = hashlib.sha256(elf).hexdigest()
        if rank == 0:
            p = pathlib.Path(out_dir) / f
            if p.exists():
                assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, f"{f}: other bytes"
            else:
                p.write_bytes(elf)
        objs.append(dict(
            M=w, src=src, symbol=sym, block_threads=meta["block_threads"], lds_bytes=lds,
            kernarg_bytes=meta["kernarg_bytes"], sha256=sha, size=len(elf), file=f,
            rt=getattr(cfg, "rt_geom", 0) or 0, lm_head=bool(cfg.lm_head),
            cfg_tp=getattr(cfg, "tp", 1), cfg=repr(cfg),
            compiled=repr(cfg.compiled()) if hasattr(cfg, "compiled") else "(no compiled())",
            group_segment=meta["group_segment"], cus_cfg=cfg.cus))
    return dict(rank=rank, dep=dep, objs=objs, kdir=str(kdir), nlayers=nl,
                flydsl=importlib.metadata.version("flydsl"))


def export_one(job: dict) -> None:
    """One vLLM process: one (deployment, maxlen). Writes job['part'] (JSON)."""
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # apply_model
    import vllm
    from vllm import LLM

    vdir = pathlib.Path(job["vllm_dir"]).resolve()
    assert pathlib.Path(vllm.__file__).resolve().is_relative_to(vdir), (vllm.__file__, vdir)
    kw = llm_kwargs(job["weights"], job["kv"], job["maxlen"], job["tp"], dummy=True)
    llm = LLM(**kw)
    ranks = llm.apply_model(functools.partial(_export_rank, out_dir=job["out"], Ms=job["M"]))
    part = dict(job=job, llm_kwargs=kw, ranks=ranks)
    pathlib.Path(job["part"]).write_text(json.dumps(part, indent=1, sort_keys=True))


def check_rank(job: dict, r: dict) -> None:
    dep = r["dep"]
    assert dep["model"] == job["model"], (dep["model"], job["model"])
    for k in ("kv", "block", "tp", "cus"):
        assert dep[k] == job[k], f"deployment {k}: vLLM gives {dep[k]}, asked {job[k]}"
    assert set(job["M"]) <= set(dep["widths"]), (job["M"], dep["widths"])
    for o in r["objs"]:
        assert o["cfg_tp"] == job["tp"] and o["cus_cfg"] == job["cus"], o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--weights")
    ap.add_argument("--vllm-dir")
    ap.add_argument("--vllm-commit")
    ap.add_argument("--mk-commit")
    ap.add_argument("--out")
    ap.add_argument("--deploy", action="append", default=[])
    ap.add_argument("--one", help=argparse.SUPPRESS)  # internal: one vLLM process
    ap.add_argument("--compile", help=argparse.SUPPRESS)  # internal: canonical_compile
    ap.add_argument("--merge", nargs="+", metavar="DIR", help="merge export dirs into --out")
    a = ap.parse_args()
    if a.compile:
        canonical_compile(json.loads(a.compile))
        return
    if a.one:
        export_one(json.loads(pathlib.Path(a.one).read_text()))
        return
    if a.merge:
        merge(pathlib.Path(a.out), a.merge)
        return
    assert re.fullmatch(r"[0-9a-f]{40}", a.vllm_commit or ""), "--vllm-commit: full sha"
    assert re.fullmatch(r"[0-9a-f]{40}", a.mk_commit or ""), "--mk-commit: full sha"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    parts = out / ".parts"
    parts.mkdir()
    entries, log = {}, []
    deps = [parse_deploy(s) for s in a.deploy]
    assert deps, "no --deploy"
    for i, d in enumerate(deps):
        for maxlen in d["maxlen"]:
            job = dict(d, maxlen=maxlen, model=a.model, weights=a.weights, out=str(out),
                       vllm_dir=a.vllm_dir, part=str(parts / f"{i}_{maxlen}.json"))
            jf = parts / f"{i}_{maxlen}.job.json"
            jf.write_text(json.dumps(job))
            print(f"# export {a.model} {d['kv']} block {d['block']} maxlen {maxlen} tp {d['tp']}"
                  f" M {d['M']}", flush=True)
            subprocess.run([sys.executable, "-B", __file__, "--one", str(jf)], check=True)
            part = json.loads(pathlib.Path(job["part"]).read_text())
            ranks = part["ranks"]
            assert len(ranks) == d["tp"], len(ranks)
            for r in ranks:
                check_rank(job, r)
                # TP ranks run one binary: every rank's objects are the same bytes
                assert [o["sha256"] for o in r["objs"]] == [o["sha256"] for o in ranks[0]["objs"]]
            r0 = ranks[0]
            log.append(f"DEPLOY kv={d['kv']} block={d['block']} maxlen={maxlen} tp={d['tp']} "
                       f"cus={d['cus']} widths={r0['dep']['widths']} nlayers={r0['nlayers']} "
                       f"llm={json.dumps({k: v for k, v in part['llm_kwargs'].items() if k != 'model'}, sort_keys=True)}")
            for o in r0["objs"]:
                e = dict(model=a.model, src=o["src"], arch=r0["dep"]["arch"], cus=d["cus"],
                         kv=d["kv"], block=d["block"], M=o["M"], tp=d["tp"], rt=o["rt"],
                         lm_head=o["lm_head"], symbol=o["symbol"],
                         block_threads=o["block_threads"], lds_bytes=o["lds_bytes"],
                         kernarg_bytes=o["kernarg_bytes"], sha256=o["sha256"], size=o["size"],
                         file=o["file"], flydsl_version=r0["flydsl"],
                         vllm_commit=a.vllm_commit, megakernels_commit=a.mk_commit)
                assert tuple(e) == MANIFEST_KEYS
                k = (a.model, o["src"])
                if k in entries:  # rt_geom 2: one object serves every maxlen
                    assert entries[k] == e, (entries[k], e)
                entries[k] = e
                log.append(f"OBJECT M={o['M']} maxlen={maxlen} file={o['file']} sha256={o['sha256']} "
                           f"size={o['size']} symbol={o['symbol']} lds_static={o['group_segment']} "
                           f"cfg={o['cfg']} compiled={o['compiled']}")
    for p in sorted(parts.iterdir()):
        p.unlink()
    parts.rmdir()
    (out / "export.log").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    write_manifest(out, list(entries.values()))


def write_manifest(out: pathlib.Path, entries: list) -> None:
    """Sorted manifest.json in out; asserts that out holds exactly its .co files and
    each has the listed sha256."""
    man = sorted(entries, key=lambda e: (e["model"], e["tp"], e["kv"], e["block"], e["rt"],
                                         e["M"], e["src"]))
    assert len({(e["model"], e["src"]) for e in man}) == len(man), "duplicate (model, src)"
    files = sorted(p.name for p in out.glob("*.co"))
    assert files == sorted(e["file"] for e in man), (files, [e["file"] for e in man])
    for e in man:
        assert tuple(e) == MANIFEST_KEYS, tuple(e)
        assert hashlib.sha256((out / e["file"]).read_bytes()).hexdigest() == e["sha256"], e["file"]
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(f"MANIFEST {out / 'manifest.json'} {len(man)} objects sha256 "
          f"{hashlib.sha256((out / 'manifest.json').read_bytes()).hexdigest()}")


def merge(out: pathlib.Path, dirs: list) -> None:
    """One set from per-model export directories: their manifests and .co files."""
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    for d in map(pathlib.Path, dirs):
        for e in json.loads((d / "manifest.json").read_text()):
            (out / e["file"]).write_bytes((d / e["file"]).read_bytes())
            entries.append(e)
    write_manifest(out, entries)


if __name__ == "__main__":
    main()
