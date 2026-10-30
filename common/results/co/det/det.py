"""Throwaway (CoExporter q6 determinism): compile Qwen3.6 Cfgs with FlyDSL COMPILE_ONLY
in this process, N times each; print the ELF sha256. Usage: det.py MODE KDIR N M...
MODE bare: kernels/ imported as a standalone package; vllm: via the vllm package path."""
import sys, os, importlib, importlib.util, hashlib, pathlib
mode, kdir, n, Ms = sys.argv[1], pathlib.Path(sys.argv[2]).resolve(), int(sys.argv[3]), [int(x) for x in sys.argv[4:]]
sys.path.insert(0, os.environ["CO_SCRIPTS"])
import export_co as X
if mode == "bare":
    spec = importlib.util.spec_from_file_location("kpkg", kdir / "__init__.py", submodule_search_locations=[str(kdir)])
    m = importlib.util.module_from_spec(spec); sys.modules["kpkg"] = m; spec.loader.exec_module(m)
    KK = importlib.import_module("kpkg.kernel")
else:
    import vllm  # noqa
    KK = importlib.import_module("vllm.models.qwen3_5_moe.amd.specialized_decode.kernels.kernel")
CFG = "Cfg(M={M}, cus=304, maxlen=4096, block=1056, maxblocks=4, kv_s0=1081344, kv_s1=512, kv_s2=1024, kv_sv=256, kv_xpack=False, conv_s0=2162688, ssm_s0=2162688, lm_head=False, c_inp=1, c_ba=8, c_out=1, c_qkv=1, c_o=16, c_router=1, c_e1=4, c_e2=1, c_lm=1, debug_route=False, fence=False, chaos=False, cta_red=True, skip_idle=True)"
os.environ["COMPILE_ONLY"] = "1"
for M in Ms:
    cfg = eval(CFG.format(M=M), {"Cfg": KK.Cfg})
    assert repr(cfg) == CFG.format(M=M)
    for i in range(n):
        L = KK.build(cfg)
        L(0, 0, 40)
        elf, sym, _ = X.elf_of(L)
        out = pathlib.Path(os.environ["DET_OUT"]) / f"{mode}_M{M}_{os.getpid()}_{i}.co"
        out.write_bytes(elf)
        print(f"DET mode={mode} pid={os.getpid()} M={M} i={i} {sym} sha={hashlib.sha256(elf).hexdigest()[:16]}", flush=True)
