"""Disposable diagnosis: bench/vllm_e2e.py with a rank-equality check of the kernel's
output inside vLLM at TP2 (DESIGN-TP2 invariant T2), also during CUDA-graph replays.

The parent patches the decode runtime before vLLM forks its workers
(VLLM_WORKER_MULTIPROC_METHOD=fork). After every kernel launch, the patched run
all-gathers the kernel's hidden states over the TP group, counts the steps whose rank
copies differ in any bit, and copies (mismatching steps, steps, epoch word) to pinned
host memory (captured into the graph). A worker thread writes those numbers to
RANKCHECK_DIR/rankcheck_r<rank>.txt every second.
Usage: RANKCHECK_DIR=DIR python scratch/m6/rankcheck_e2e.py <vllm_e2e.py arguments>
"""
import os
import runpy
import sys
import threading
import time

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
import torch  # noqa: E402

from vllm.models.qwen3_5.amd.specialized_decode.runtime import decode as RD  # noqa: E402

OUT = os.environ["RANKCHECK_DIR"]
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
sys.argv = [os.path.join(os.path.dirname(__file__), "../../bench/vllm_e2e.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
