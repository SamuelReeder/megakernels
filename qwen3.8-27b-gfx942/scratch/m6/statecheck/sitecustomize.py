"""Disposable diagnosis (sitecustomize, active when STATECHECK_DIR is set): why two vLLM TP2
runs of bench/vllm_e2e.py differ (DESIGN-TP2 section 11, milestone 6 repeatability).

Every Python call of the Qwen3.5 model forward that is not a CUDA-graph capture (prefill
and mixed steps; FULL-graph decode replays do not call Python) writes one line to
STATECHECK_DIR/state_r<rank>.txt: call index, token count, bit sums of input_ids,
positions and the returned hidden states, and per layer the bit sums of every cache
tensor (GDN: conv and SSM state; attention: KV cache) before and after the call. The state
after a prefill is the input of the first decode step (kernel or vLLM's graph).
Kernel on: every eager or captured Qwen35Gfx942Decode.run also writes its input_ids sum
(token, the only per-step input not in the caches) with a "launch" tag.
A bit sum is the int64 sum of the same-size integer view: equal tensors give equal sums; a differing sum
proves a differing tensor. Two runs are compared line by line (scratch/m6/statecmp.py).
Usage: PYTHONPATH=scratch/m6/statecheck:$PYTHONPATH STATECHECK_DIR=DIR python bench/vllm_e2e.py ...
"""
import importlib.abc
import importlib.util
import os
import sys

OUT = os.environ.get("STATECHECK_DIR")
TARGETS = {
    "vllm.model_executor.models.qwen3_5",
    "vllm.models.qwen3_5.amd.specialized_decode.runtime.decode",
}
_state = {"n": 0, "depth": 0, "layers": None, "file": None}


def _bits(t):
    import torch

    if t.numel() == 0:
        return torch.zeros((), dtype=torch.int64, device=t.device)
    # a same-size integer view: no copy, any strides (a copy of a cache would not fit)
    it = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    # sum(dtype=int64) converts its whole input first: sum slices of at most 2**26 elements
    t = t.view(it)
    if t.dim() == 0:
        return t.to(torch.int64)
    if t[0].numel() > (1 << 26):
        return torch.stack([_bits(u) for u in t]).sum()
    step = max(1, (1 << 26) // max(1, t[0].numel()))
    return torch.stack([t[i:i + step].sum(dtype=torch.int64) for i in range(0, t.shape[0], step)]).sum()


def _tensors(x):
    import torch

    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _tensors(v)]
    return []


def _file():
    if _state["file"] is None:
        import torch.distributed as dist

        r = dist.get_rank() if dist.is_initialized() else 0
        _state["file"] = open(os.path.join(OUT, f"state_r{r}.txt"), "w", buffering=1)
    return _state["file"]


def _cache_sums(model):
    import torch

    # GDN layers get kv_cache only when vLLM binds the caches (after the profile run), so
    # the list is rebuilt until every cache layer has one; its names go to the file once
    layers = [(n, m) for n, m in model.named_modules() if hasattr(m, "get_kv_cache_spec")]
    if _state["layers"] is None or len(_state["layers"]) != sum(hasattr(m, "kv_cache") for _, m in layers):
        _state["layers"] = [(n, m) for n, m in layers if hasattr(m, "kv_cache")]
        _file().write("layers " + " ".join(f"{n}:{type(m).__name__}" for n, m in _state["layers"]) + "\n")
    out = []
    for n, m in _state["layers"]:
        ts = _tensors(m.kv_cache)
        out.append(torch.stack([_bits(t) for t in ts]) if ts else None)
    return out


def _fmt(sums):
    return ";".join("-" if s is None else ",".join(str(v) for v in s.tolist()) for s in sums)


def _patch_model(mod):
    import torch

    for cls_name in ("Qwen3_5ForCausalLMBase", "Qwen3_5ForConditionalGeneration"):
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        orig = cls.forward

        def forward(self, input_ids=None, positions=None, *a, _orig=orig, **k):
            if _state["depth"] or torch.cuda.is_current_stream_capturing():
                return _orig(self, input_ids, positions, *a, **k)
            _state["depth"] += 1
            try:
                pre = _cache_sums(self)
                out = _orig(self, input_ids, positions, *a, **k)
                post = _cache_sums(self)
                h = out[0] if isinstance(out, tuple) else out
                ids = _bits(input_ids) if input_ids is not None else None
                emb = k.get("inputs_embeds")
                emb = None if emb is None else _bits(emb).item()
                n = 0 if positions is None else positions.shape[-1]
                _state["n"] += 1
                _file().write(
                    f"fwd {_state['n']} tokens {n} ids {None if ids is None else ids.item()} "
                    f"pos {None if positions is None else _bits(positions).item()} "
                    f"emb {emb} hidden {_bits(h).item()} | pre {_fmt(pre)} | post {_fmt(post)}\n"
                )
                return out
            finally:
                _state["depth"] -= 1

        cls.forward = forward


def _patch_decode(RD):
    import torch

    orig = RD.Qwen35Gfx942Decode.run

    def run(self, step, input_ids, inputs_embeds, **kwargs):
        if not torch.cuda.is_current_stream_capturing() and input_ids is not None:
            _file().write(f"launch eager ids {_bits(input_ids).item()} rows {input_ids.shape[0]}\n")
        return orig(self, step, input_ids, inputs_embeds, **kwargs)

    RD.Qwen35Gfx942Decode.run = run


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        orig = spec.loader.exec_module

        def exec_module(mod):
            orig(mod)
            (_patch_model if name.endswith("qwen3_5") else _patch_decode)(mod)

        spec.loader.exec_module = exec_module
        return spec


if OUT:
    sys.meta_path.insert(0, _Finder())
