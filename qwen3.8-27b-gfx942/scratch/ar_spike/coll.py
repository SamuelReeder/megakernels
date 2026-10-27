"""Spike only (disposable). Latency of one M x 5120 all-reduce through RCCL and, when
vLLM is importable (the vLLM image), through vLLM's TP all-reduce paths.

Usage: srun -n2 python scratch/ar_spike/coll.py OUT_DIR TAG
Per variant: eager (events around each call, 1000 calls) and a CUDA graph of 128 calls in
a row (replayed 50 times; time / 128), median and p99 per all-reduce.
"""

import json
import os
import socket
import sys
import time
import traceback

import torch
import torch.distributed as dist

RANK = int(os.environ["SLURM_PROCID"])
WS = int(os.environ["SLURM_NTASKS"])
N_EAGER, N_GRAPH, CHAIN = 1000, 50, 128


def log(*a):
    print(f"[r{RANK} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def bench(fn, x, capture_ctx=None, group=None):
    s = torch.cuda.current_stream()
    for _ in range(20):
        fn(x)
    torch.cuda.synchronize()
    dist.barrier(group)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(N_EAGER + 1)]
    ev[0].record(s)
    for i in range(N_EAGER):
        fn(x)
        ev[i + 1].record(s)
    torch.cuda.synchronize()
    eager = [ev[i].elapsed_time(ev[i + 1]) * 1000 for i in range(N_EAGER)]
    res = {"eager_med_us": q(eager, 0.5), "eager_p99_us": q(eager, 0.99)}
    g = torch.cuda.CUDAGraph()
    if capture_ctx is not None:
        with capture_ctx() as st:
            with torch.cuda.graph(g, stream=st):
                for _ in range(CHAIN):
                    fn(x)
    else:
        with torch.cuda.graph(g):
            for _ in range(CHAIN):
                fn(x)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier(group)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(N_GRAPH + 1)]
    ev[0].record(s)
    for i in range(N_GRAPH):
        g.replay()
        ev[i + 1].record(s)
    torch.cuda.synchronize()
    gr = [ev[i].elapsed_time(ev[i + 1]) * 1000 / CHAIN for i in range(N_GRAPH)]
    res.update(graph_med_us=q(gr, 0.5), graph_p99_us=q(gr, 0.99))
    return res


def main():
    outdir, tag = sys.argv[1], sys.argv[2]
    torch.cuda.set_device(RANK)
    dev = torch.device("cuda", RANK)
    env = {k: v for k, v in os.environ.items() if k.startswith(("NCCL", "RCCL", "HSA", "HIP", "ROCR", "VLLM"))}
    log(socket.gethostname(), torch.__version__, torch.version.hip, "env", env)
    shm = os.statvfs("/dev/shm")
    log("/dev/shm free MB", shm.f_bavail * shm.f_frsize >> 20)
    port = int(os.environ.get("SPIKE_PORT", "29534"))
    init = f"tcp://127.0.0.1:{port}"
    out = {"host": socket.gethostname(), "torch": torch.__version__, "tag": tag, "runs": []}
    vllm_ok = False
    try:
        import vllm  # noqa: F401
        from vllm.distributed import parallel_state as ps
        from vllm.config import VllmConfig, set_current_vllm_config

        out["vllm"] = vllm.__version__
        ctxcfg = set_current_vllm_config(VllmConfig())
        ctxcfg.__enter__()
        ps.init_distributed_environment(WS, RANK, init, RANK, "nccl")
        ps.initialize_model_parallel(tensor_model_parallel_size=WS)
        vllm_ok = True
        log("vllm", vllm.__version__, "initialized")
    except Exception:
        log("no vllm:", traceback.format_exc().splitlines()[-1])
        dist.init_process_group("nccl", init_method=init, rank=RANK, world_size=WS, device_id=dev)
    log("nccl init done")
    for dtype in (torch.float32, torch.bfloat16):
        for M in (1, 8):
            x = torch.randn(M, 5120, dtype=dtype, device=dev)
            variants = [("rccl", lambda t: dist.all_reduce(t), None, None)]
            if vllm_ok:
                tp = ps.get_tp_group()
                comm = tp.device_communicator

                def gc():
                    import contextlib

                    @contextlib.contextmanager
                    def c():
                        with ps.graph_capture(dev) as g:
                            yield g.stream
                    return c()

                variants.append(("vllm_tp_all_reduce", lambda t: tp.all_reduce(t), gc, None))
                for name, meth in (("ca_comm", "custom_all_reduce"), ("aiter_ar_comm", "custom_all_reduce"),
                                   ("qr_comm", "quick_all_reduce")):
                    o = getattr(comm, name, None)
                    if o is not None and not getattr(o, "disabled", False):
                        variants.append((f"vllm_{name}", (lambda o, meth: lambda t: getattr(o, meth)(t))(o, meth), gc, None))
                    else:
                        log(f"{name}: {'absent' if o is None else 'disabled'}")
            for name, fn, cap, _ in variants:
                try:
                    r = bench(fn, x, cap)
                except Exception:
                    r = {"error": traceback.format_exc().splitlines()[-1]}
                r.update(variant=name, M=M, dtype=str(dtype))
                out["runs"].append(r)
                log(json.dumps(r))
    json.dump(out, open(os.path.join(outdir, f"coll_{tag}_r{RANK}.json"), "w"), indent=1)
    if vllm_ok:
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
