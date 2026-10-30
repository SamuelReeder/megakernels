"""H1 check of an export_co.py set (docs/hipdnn-route.md H1): for each width of a
deployment, the code object in the manifest launched with hipModuleLaunchKernel gives the
same output bits as the FlyDSL launch of the same kernel source and Cfg.

vLLM runs the checkout with the real weights. In each TP rank (llm.apply_model) the
DecodeSpec's own runtime object supplies the Cfg (`kernel.cfgs[M]`), the tables (`tab`,
`glob`, from `bind_caches`) and the step staging (`kernel.stage`, fed block tables, slot
mappings and state indices of fresh, zeroed cache pages under a forward context). Per
width:
  1. src = compile_key(cfg.compiled(), kernels/) selects the manifest entry; its key
     fields equal the runtime's; the .co file's sha256 equals the manifest.
  2. STEPS steps (seeded tokens, positions 0..STEPS-1) with the FlyDSL launch
     (`kernel.launch`), then with the code object, then FlyDSL again, each from zeroed
     pages; sha256 of every step's output bytes (the normed hidden states: the vLLM
     objects have no LM head).
  3. The ELF that the runtime's FlyDSL launcher compiled has the manifest sha256.
PASS: 1 and 3 hold and the three output hashes are equal, on every rank.

Usage (vLLM image, checkout first on PYTHONPATH: common/scripts/vrun):
  python h1_co.py --model MODEL --weights DIR --vllm-dir DIR --co-dir DIR
         --deploy SPEC [--deploy SPEC ...] [--steps 16]
SPEC as in export_co.py. Prints one H1 line per (rank, width) and an H1-SUMMARY line.
"""
import argparse
import ctypes
import functools
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import types

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_co as X  # noqa: E402

KEY_FIELDS = ("arch", "cus", "kv", "block", "M", "tp", "rt", "lm_head")


def _hip():
    lib = sorted({ln.split()[-1] for ln in open("/proc/self/maps") if "libamdhip64" in ln})
    assert len(lib) == 1, f"one HIP runtime expected (torch's): {lib}"
    hip = ctypes.CDLL(lib[0])
    hip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return hip


def _ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: hip error {rc}")


class Staging:
    """Fresh cache pages for M rows and STEPS positions in every KV cache group of the
    kernel: block tables and slot mappings (attention groups), state indices (GDN
    groups); disjoint page ids from 1 (0: vLLM's null block / padding slot)."""

    def __init__(self, dec, sfc, M, steps):
        import torch

        self.dec, self.M, self.steps = dec, M, steps
        nid = 1
        self.attn = []
        for name in dec.attn_groups:
            kv = sfc[name].kv_cache
            B = kv.shape[2]
            nb = -(-steps // B)
            bt = torch.arange(nid, nid + M * nb, dtype=torch.int32).view(M, nb)
            nid += M * nb
            self.attn.append((name, B, bt.cuda()))
        self.gdn = []
        for name in getattr(dec, "gdn_groups", []):
            self.gdn.append((name, torch.arange(nid, nid + M, dtype=torch.int32).cuda()))
            nid += M
        self.tensors = []
        for name in dec.cache_layers:
            c = sfc[name].kv_cache
            self.tensors += list(c) if isinstance(c, (tuple, list)) else [c]
        assert nid <= min(t.shape[0] for t in self.tensors), (nid, "cache pages")
        self.ids = torch.arange(1, nid, device="cuda")
        g = torch.Generator("cuda")
        g.manual_seed(7)
        self.toks = torch.randint(0, 150000, (steps, M), device="cuda", generator=g)

    def zero(self):
        for t in self.tensors:
            t.index_fill_(0, self.ids, 0)

    def stage(self, t):
        import torch
        from vllm.forward_context import override_forward_context
        from vllm.models.common.specialized_decode.spec import Step

        md = {}
        for name, B, bt in self.attn:
            slot = bt[:, t // B].long() * B + t % B
            md[name] = types.SimpleNamespace(block_table=bt, slot_mapping=slot, max_query_len=1)
        for name, sidx in self.gdn:
            md[name] = types.SimpleNamespace(non_spec_state_indices_tensor=sidx)
        pos = torch.full((self.M,), t, dtype=torch.int64, device="cuda")
        step = Step(width=self.M, kernel_width=self.M, positions=pos)
        with override_forward_context(types.SimpleNamespace(attn_metadata=md)):
            return self.dec.stage(step, self.toks[t], None)

    def run(self, launch):
        """sha256 of the output bytes of STEPS steps from zeroed pages; all finite."""
        import torch

        self.zero()
        h, fin = hashlib.sha256(), True
        for t in range(self.steps):
            glob = self.stage(t)
            launch(glob)
            o = self.dec.out[:self.M].clone()
            torch.cuda.synchronize()
            h.update(o.contiguous().view(torch.uint8).cpu().numpy().tobytes())
            fin &= bool(torch.isfinite(o.float()).all())
        return h.hexdigest()[:16], fin


def _h1_rank(model, co_dir: str, Ms: list, steps: int) -> dict:
    import torch

    sd, dec, kmod, keymod, kdir, spec, nl = X.runtime(model)
    dep = X.deployment_fields(sd, dec, spec)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    co_dir = pathlib.Path(co_dir)
    by = {(e["model"], e["src"]): e for e in json.loads((co_dir / "manifest.json").read_text())}
    sfc = sd.vllm_config.compilation_config.static_forward_context
    hip = _hip()
    stream = lambda: torch.cuda.current_stream().cuda_stream  # noqa: E731
    rows = []
    for w in Ms:
        cfg = dec.cfgs[w]
        src = X.src_key(keymod, kdir, cfg)
        row = dict(rank=rank, M=w, src=src)
        rows.append(row)
        e = by.get((dep["model"], src))
        if e is None:
            row["status"] = "MISSING"
            continue
        row["file"] = e["file"]
        rt = dict(arch=dep["arch"], cus=cfg.cus, kv=dep["kv"], block=dep["block"], M=w,
                  tp=dep["tp"], rt=getattr(cfg, "rt_geom", 0) or 0, lm_head=bool(cfg.lm_head))
        row["fields"] = {k: (e[k], rt[k]) for k in KEY_FIELDS if e[k] != rt[k]}
        elf = (co_dir / e["file"]).read_bytes()
        row["file_sha_ok"] = hashlib.sha256(elf).hexdigest() == e["sha256"]
        st = Staging(dec, sfc, w, steps)
        row["fly"] = st.run(lambda g: dec.launch(w, g))
        relf, _, _ = X.elf_of(dec.launches[w])
        row["runtime_elf_sha_ok"] = hashlib.sha256(relf).hexdigest() == e["sha256"]
        mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
        _ck(hip.hipModuleLoadData(ctypes.byref(mod), elf), "hipModuleLoadData")
        _ck(hip.hipModuleGetFunction(ctypes.byref(fn), mod, e["symbol"].encode()),
            "hipModuleGetFunction")
        keep = {}

        def co_launch(glob):
            a = (ctypes.c_int64(dec.tab.data_ptr()), ctypes.c_int64(glob.data_ptr()),
                 ctypes.c_int32(nl))
            p = (ctypes.c_void_p * 3)(*[ctypes.addressof(x) for x in a])
            keep["a"], keep["p"] = a, p
            _ck(hip.hipModuleLaunchKernel(fn, e["cus"], 1, 1, e["block_threads"], 1, 1,
                                          e["lds_bytes"], stream(), p, None),
                "hipModuleLaunchKernel")

        row["co"] = st.run(co_launch)
        row["fly2"] = st.run(lambda g: dec.launch(w, g))
        hashes = {row["fly"][0], row["co"][0], row["fly2"][0]}
        finite = row["fly"][1] and row["co"][1] and row["fly2"][1]
        ok = (len(hashes) == 1 and finite and not row["fields"] and row["file_sha_ok"]
              and row["runtime_elf_sha_ok"])
        row["status"] = "PASS" if ok else "FAIL"
    return dict(rank=rank, dep=dep, rows=rows, nlayers=nl)


def h1_one(job: dict) -> None:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # apply_model
    # TP workers unpickle _h1_rank's references to export_co by module name
    os.environ["PYTHONPATH"] = f"{HERE}:{os.environ.get('PYTHONPATH', '')}"
    import vllm
    from vllm import LLM

    vdir = pathlib.Path(job["vllm_dir"]).resolve()
    assert pathlib.Path(vllm.__file__).resolve().is_relative_to(vdir), (vllm.__file__, vdir)
    llm = LLM(**X.llm_kwargs(job["weights"], job["kv"], job["maxlen"], job["tp"], dummy=False))
    ranks = llm.apply_model(functools.partial(_h1_rank, co_dir=job["co_dir"], Ms=job["M"],
                                              steps=job["steps"]))
    pathlib.Path(job["part"]).write_text(json.dumps(ranks, indent=1, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=X.MODELS)
    ap.add_argument("--weights")
    ap.add_argument("--vllm-dir")
    ap.add_argument("--co-dir")
    ap.add_argument("--deploy", action="append", default=[])
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--one", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        h1_one(json.loads(pathlib.Path(a.one).read_text()))
        return
    co_dir = pathlib.Path(a.co_dir).resolve()
    man = json.loads((co_dir / "manifest.json").read_text())
    tmp = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f"h1_{os.getpid()}"
    tmp.mkdir()
    seen, npass, nrow = set(), 0, 0
    for i, d in enumerate(X.parse_deploy(s) for s in a.deploy):
        for maxlen in d["maxlen"]:
            job = dict(d, maxlen=maxlen, model=a.model, weights=a.weights, vllm_dir=a.vllm_dir,
                       co_dir=str(co_dir), steps=a.steps, part=str(tmp / f"{i}_{maxlen}.json"))
            jf = tmp / f"{i}_{maxlen}.job.json"
            jf.write_text(json.dumps(job))
            print(f"# H1 {a.model} kv={d['kv']} block={d['block']} maxlen={maxlen} tp={d['tp']} "
                  f"M={d['M']} steps={a.steps}", flush=True)
            subprocess.run([sys.executable, "-B", __file__, "--one", str(jf)], check=True)
            ranks = json.loads(pathlib.Path(job["part"]).read_text())
            assert len(ranks) == d["tp"], len(ranks)
            for r in ranks:
                for row in r["rows"]:
                    nrow += 1
                    npass += row["status"] == "PASS"
                    if row["status"] != "MISSING":
                        seen.add(row["file"])
                    print(f"H1 {row['status']} model={a.model} kv={d['kv']} block={d['block']} "
                          f"maxlen={maxlen} tp={d['tp']} rank={row['rank']} M={row['M']} "
                          f"src={row['src'][:16]} file={row.get('file')} "
                          f"fields_differ={row.get('fields')} file_sha_ok={row.get('file_sha_ok')} "
                          f"runtime_elf_sha_ok={row.get('runtime_elf_sha_ok')} "
                          f"flydsl={row.get('fly')} co={row.get('co')} flydsl_again={row.get('fly2')}",
                          flush=True)
    unchecked = sorted(e["file"] for e in man if e["model"] == a.model and e["file"] not in seen)
    print(f"H1-SUMMARY model={a.model} pass {npass}/{nrow} manifest objects of this model "
          f"not checked: {unchecked or 'none'}")
    sys.exit(0 if npass == nrow and not unchecked else 1)


if __name__ == "__main__":
    main()
