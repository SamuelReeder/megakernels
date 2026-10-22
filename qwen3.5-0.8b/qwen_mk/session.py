"""Chat sessions on top of the megakernel: chat template, state reuse, fast prefill.

The Gated DeltaNet state can't be rolled back, so a session remembers exactly which tokens
have been fed. It also snapshots the recurrent state (conv history + DeltaNet state, ~20 MB)
just before each `<|im_start|>assistant\\n` header. A new request whose prompt extends the
fed tokens continues directly; otherwise the longest matching snapshot is restored, which
covers the usual chat case where the template re-renders earlier assistant turns slightly
differently. The KV cache needs no snapshot: positions before the restore point are never
overwritten.

Long prefills from scratch run through Hugging Face transformers in one batched forward
pass and the resulting conv / recurrent / KV state is copied into the kernel's buffers;
short ones go through the megakernel one token per step. Either way the MTP head's cache is
filled for the prompt, and replies are decoded speculatively with it.
"""

import threading
import time

import torch

from . import weights
from .engine import Megakernel

HF_PREFILL_MIN = 256  # below this many new tokens, the megakernel's own prefill is faster
MAX_SNAPSHOTS = 8


class HFPrefill:
    def __init__(self, path, device):
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()

    @torch.no_grad()
    def __call__(self, mk, tokens):
        """Reset `mk` and leave it in the state after feeding `tokens`."""
        n = len(tokens)
        ids = torch.tensor([tokens], device=mk.device)
        final = {}  # the final norm's output: the hidden states the LM head (and MTP head) see
        hook = self.model.model.norm.register_forward_hook(lambda m, i, o: final.update(h=o))
        try:
            cache = self.model(ids, use_cache=True, logits_to_keep=1).past_key_values
        finally:
            hook.remove()
        mk.reset()
        g = a = 0
        for i, kind in enumerate(mk.cfg.layer_types):
            layer = cache.layers[i]
            if kind == "linear_attention":
                conv = layer.conv_states[0][0].float()  # [channels, 4] = inputs at positions n-4 .. n-1
                for j in range(conv.shape[1]):
                    p = n - conv.shape[1] + j
                    if p >= 0:
                        mk.conv_hist[g][:, p % mk.ext.CONV_RING] = conv[:, j]
                mk.state[g].copy_(layer.recurrent_states[0][0].transpose(-1, -2))
                g += 1
            else:
                mk.k_cache[a][:, :n] = layer.keys[0]
                mk.v_cache[a][:, :n] = layer.values[0]
                a += 1
        mk.pos = n
        mk.mtp_fed(0, final["h"][0].float(), tokens)


class ChatSession:
    def __init__(self, model=weights.DEFAULT_MODEL, max_seq=32768, hf_prefill=True, device="cuda", quant="int8",
                 spec=True):
        from transformers import AutoTokenizer

        self.path = weights.snapshot_path(model)
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.spec = spec  # speculative decoding with the MTP head (identical output, faster)
        self.mk = Megakernel(self.path, max_seq=max_seq, device=device, quant=quant, mtp=spec)
        self.stop_ids = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")]
        self.header = self.tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self._hf = None
        self.hf_prefill = hf_prefill
        self.fed = []        # tokens whose effect is in the kernel state
        self.snapshots = []  # (token prefix, conv_hist copies, state copies), most recent last
        self.lock = threading.Lock()
        self.stats = {}

    def render(self, messages, enable_thinking=False, tools=None):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=enable_thinking, tools=tools)

    def encode(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    # ---- state management
    def _snapshot(self):
        mk = self.mk  # called only while self.fed is known
        self.snapshots = [s for s in self.snapshots if s[0] != self.fed][-(MAX_SNAPSHOTS - 1):]
        self.snapshots.append((list(self.fed), [h.clone() for h in mk.conv_hist], [s.clone() for s in mk.state],
                               mk.mtp_h))  # (pos, h) is never modified in place, so no clone

    def _restore(self, snap):
        prefix, hist, state, mtp_h = snap
        mk = self.mk
        for dst, src in zip(mk.conv_hist + mk.state, hist + state):
            dst.copy_(src)
        mk.pos = len(prefix)
        mk.mtp_h = mtp_h
        self.fed = list(prefix)

    def _feed(self, tokens):
        if not tokens:
            return
        if self.hf_prefill and not self.fed and len(tokens) >= HF_PREFILL_MIN:
            if self._hf is None:
                self._hf = HFPrefill(self.path, self.mk.device)
            self._hf(self.mk, tokens)
        else:
            self.mk.prefill(tokens + [0])
        self.fed += tokens

    def _last_header(self, ids, start):
        """Index just before the last assistant header at or after `start`, or None."""
        h = self.header
        for i in range(len(ids) - len(h), start - 1, -1):
            if ids[i: i + len(h)] == h:
                return i
        return None

    def _sync_state(self, prompt):
        """Bring the kernel state to `prompt[:-1]`, reusing fed tokens or a snapshot."""
        target = prompt[:-1]
        n = len(self.fed) if self.fed is not None else -1
        if not (0 <= n <= len(target) and target[:n] == self.fed):
            best = max((s for s in self.snapshots if len(s[0]) <= len(target) and target[:len(s[0])] == s[0]),
                       key=lambda s: len(s[0]), default=None)
            if best is not None:
                self._restore(best)
            else:
                self.mk.reset()
                self.fed = []
        start = len(self.fed)
        cut = self._last_header(target, start)
        if cut is not None:
            self._feed(target[start:cut])
            self._snapshot()
            self._feed(target[cut:])
        else:
            self._feed(target[start:])
        return len(target) - start

    # ---- generation
    def stream(self, prompt_ids, max_new_tokens=512, temperature=0.7, seed=None):
        """Yield generated token ids for a tokenized prompt (ends with a stop id if one is hit)."""
        if seed is None:
            seed = time.time_ns() & 0x7FFFFFFF
        with self.lock:
            t0 = time.perf_counter()
            n_prefill = self._sync_state(prompt_ids)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out = []
            self.fed = None  # unknown until generation finishes cleanly
            for t in self.mk.generate([prompt_ids[-1]], max_new_tokens=max_new_tokens, temperature=temperature,
                                      seed=seed, stop_ids=self.stop_ids, chunk=16, spec=self.spec):
                out.append(t)
                yield t
            # Everything but the final prediction has been fed (a stop id, or the token
            # predicted when max_new_tokens ran out). If the caller stops early, `fed` stays
            # None and the next request restores a snapshot instead.
            self.fed = list(prompt_ids) + out[:-1]
            t2 = time.perf_counter()
            self.stats = dict(prefill_tokens=n_prefill, prefill_s=t1 - t0, decode_tokens=len(out),
                              decode_s=t2 - t1, decode_tok_s=len(out) / max(t2 - t1, 1e-9))

    def chat(self, messages, enable_thinking=False, tools=None, **kw):
        """Yield generated ids for a chat (stop ids excluded)."""
        ids = self.encode(self.render(messages, enable_thinking=enable_thinking, tools=tools))
        for t in self.stream(ids, **kw):  # run `stream` to completion so it records the fed state
            if t not in self.stop_ids:
                yield t
