"""Disposable diagnosis (sitecustomize, active when RANKCHECK_DIR is set): a rank-equality
check of the kernel's output inside vLLM at TP2 (DESIGN-TP2 invariant T2), also during
CUDA-graph replays. It patches the decode runtime when a process imports it (vLLM spawns
its workers, so the patch must come from every interpreter's start).

After every kernel launch, the patched run all-gathers the kernel's hidden states over
the TP group, counts the steps whose rank copies differ in any bit, and copies
(mismatching steps, steps, epoch word) to pinned host memory (captured into the graph).
A worker thread writes them to RANKCHECK_DIR/rankcheck_r<rank>.txt every second.
Usage: PYTHONPATH=scratch/m6/rankcheck:$PYTHONPATH RANKCHECK_DIR=DIR python bench/vllm_e2e.py ...
"""
import importlib.abc
import importlib.util
import os
import sys
import threading
import time

TARGET = "vllm.models.qwen3_5.amd.specialized_decode.runtime.decode"
OUT = os.environ.get("RANKCHECK_DIR")


def _patch(RD):
    import torch

    orig_bind, orig_run = RD.Qwen35Gfx942Decode.bind_caches, RD.Qwen35Gfx942Decode.run

    def bind_caches(self, caches):
        orig_bind(self, caches)
        if self.tp == 1 or hasattr(self, "_rc"):
            return
        self._rc = torch.zeros(3, dtype=torch.int64, device=self.device)
        self._rc_host = torch.zeros(3, dtype=torch.int64).pin_memory()
        path = os.path.join(OUT, f"rankcheck_r{self.rank}.txt")

        def writer():
            while True:
                v = self._rc_host.tolist()
                with open(path, "w") as f:
                    f.write(f"rank {self.rank} mismatching_steps {v[0]} steps {v[1]} epoch {v[2]}\n")
                time.sleep(1)

        threading.Thread(target=writer, daemon=True).start()


    def run(self, step, input_ids, inputs_embeds, **kwargs):
        out = orig_run(self, step, input_ids, inputs_embeds, **kwargs)
        if self.tp > 1 and hasattr(self, "_rc"):
            from vllm.distributed import get_tp_group

            W = out.shape[0]
            g = get_tp_group().all_gather(out.contiguous(), dim=0)
            bits = g.view(torch.int16).view(self.tp, W, -1)
            self._rc[0] += (bits != bits[:1]).any().to(torch.int64)
            self._rc[1] += 1
            ep = self.flags[(self.cus + RD.KK.EPOCH_WORD_LINE) * 32].to(torch.int64)
            self._rc[2] = ep
            self._rc_host.copy_(self._rc, non_blocking=True)
        return out

    RD.Qwen35Gfx942Decode.bind_caches = bind_caches
    RD.Qwen35Gfx942Decode.run = run


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        orig = spec.loader.exec_module

        def exec_module(mod):
            orig(mod)
            _patch(mod)

        spec.loader.exec_module = exec_module
        return spec


if OUT:
    sys.meta_path.insert(0, _Finder())
