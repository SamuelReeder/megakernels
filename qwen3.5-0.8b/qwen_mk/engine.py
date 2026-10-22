"""Python driver for the decode megakernel: packs weights, owns state, launches steps."""

import os

import numpy as np
import torch

from . import _ext, weights
from .reference import rmsnorm, rope_tables

LAYER_FIELDS = 18
# Steps per launch. Windows' GPU watchdog (TDR) resets kernels that run ~2 s; at ~2 ms/step
# this keeps each launch well under it.
MAX_STEPS_PER_LAUNCH = 256
# How far (bytes per SM) the producer prefetches weights into L2 ahead of its smem ring.
PF_BYTES = 32 * 1024
Q8_GROUP = 32
# Token ids ranked by how often the model generates them (tools/draft_vocab.py). The draft head
# only scores the top DRAFT_VOCAB of them: a full-vocabulary argmax would read as many bytes as
# the LM head every cycle.
DRAFT_VOCAB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "draft_vocab.npy")
DRAFT_VOCAB = 32768


def quantize_q8(w, group=Q8_GROUP):
    """[R, K] weights -> [R, K + 2K/group] bytes: (q + 128) as uint8, then bf16 absmax scales.

    Same scheme as llama.cpp's Q8_0 (symmetric int8, one scale per 32 weights), laid out so
    each row stays one contiguous span the kernel can bulk-copy.
    """
    R, K = w.shape
    out = torch.empty(R, K + 2 * K // group, dtype=torch.uint8, device=w.device)
    for r in range(0, R, 16384):  # in slices: the LM head would need ~3 GB of fp32 temporaries
        g = w[r: r + 16384].float().reshape(-1, K // group, group)
        scale = (g.abs().amax(-1) / 127).clamp_min(1e-12).to(torch.bfloat16)
        q = torch.round(g / scale.float()[..., None]).clamp(-128, 127) + 128
        out[r: r + 16384, :K] = q.reshape(-1, K).to(torch.uint8)
        out[r: r + 16384, K:] = scale.view(torch.uint8).reshape(-1, 2 * K // group)
    return out


def dequantize_q8(packed, K, group=Q8_GROUP):
    """Inverse of quantize_q8, in fp32 (used to check the kernel against the reference)."""
    q = packed[:, :K].to(torch.int16) - 128
    scale = packed[:, K:].contiguous().view(torch.bfloat16).float()
    return (q.float().reshape(len(packed), -1, group) * scale[..., None]).reshape(-1, K)


class Megakernel:
    def __init__(self, model=weights.DEFAULT_MODEL, max_seq=8192, device="cuda", pf_bytes=PF_BYTES, quant="bf16",
                 mtp=True, draft_vocab=DRAFT_VOCAB):
        """`mtp` loads the multi-token-prediction head for speculative decoding (`run_spec`), whose
        drafts come from the `draft_vocab` most frequent tokens (0: the whole vocabulary)."""
        assert quant in ("bf16", "int8"), quant
        self.quant = quant
        self.pf_bytes = pf_bytes
        device = torch.device(device)
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        torch.cuda.set_device(device)
        self.ext = _ext.load()
        self.cfg, w = weights.load(model, device)
        cfg = self.cfg
        self.max_seq = max_seq
        self.device = device
        self.grid = torch.cuda.get_device_properties(device).multi_processor_count
        self._keep = []  # every tensor whose pointer the kernel holds

        def keep(t):
            t = t.contiguous()
            self._keep.append(t)
            return t.data_ptr()

        def mat(t):  # a GEMV weight in the kernel's format
            return keep(quantize_q8(t) if quant == "int8" else t)

        f32 = dict(device=device, dtype=torch.float32)
        self.conv_hist, self.state, self.state_bak, self.k_cache, self.v_cache = [], [], [], [], []
        dummy = keep(torch.zeros(16, **f32))

        def attn_layer(w, m):  # pointer fields of a gated full-attention decoder layer
            p = m + "self_attn."
            w_in = torch.cat([w[p + "q_proj.weight"], w[p + "k_proj.weight"], w[p + "v_proj.weight"]])
            kc = torch.zeros(cfg.n_kv_heads, max_seq, cfg.head_dim, device=device, dtype=torch.bfloat16)
            vc = torch.zeros_like(kc)
            gu = torch.stack([w[m + "mlp.gate_proj.weight"], w[m + "mlp.up_proj.weight"]], dim=1).reshape(-1, cfg.hidden)
            ptrs = [mat(w_in), mat(w[p + "o_proj.weight"]), mat(gu), mat(w[m + "mlp.down_proj.weight"]),
                    keep(w[m + "input_layernorm.weight"]), keep(w[m + "post_attention_layernorm.weight"]),
                    dummy, dummy, dummy, dummy, dummy, dummy,
                    keep(w[p + "q_norm.weight"]), keep(w[p + "k_norm.weight"]), keep(kc), keep(vc), 1, dummy]
            return ptrs, kc, vc

        layer_ptrs = []
        for i, kind in enumerate(cfg.layer_types):
            m = f"layers.{i}."
            if kind == "linear_attention":
                p = m + "linear_attn."
                w_in = torch.cat([w[p + "in_proj_qkv.weight"], w[p + "in_proj_z.weight"],
                                  w[p + "in_proj_b.weight"], w[p + "in_proj_a.weight"]])
                gu = torch.stack([w[m + "mlp.gate_proj.weight"], w[m + "mlp.up_proj.weight"]], dim=1).reshape(-1, cfg.hidden)
                hist = torch.zeros(w[p + "conv1d.weight"].shape[0], self.ext.CONV_RING, **f32)
                st = torch.zeros(cfg.lin_heads, cfg.lin_v_dim, cfg.lin_k_dim, **f32)
                bak = torch.zeros_like(st)
                self.conv_hist.append(hist)
                self.state.append(st)
                self.state_bak.append(bak)
                layer_ptrs += [mat(w_in), mat(w[p + "out_proj.weight"]), mat(gu), mat(w[m + "mlp.down_proj.weight"]),
                               keep(w[m + "input_layernorm.weight"]), keep(w[m + "post_attention_layernorm.weight"]),
                               keep(w[p + "conv1d.weight"].reshape(-1, 4)), keep(w[p + "A_log"].float()),
                               keep(w[p + "dt_bias"]), keep(w[p + "norm.weight"].float()), keep(hist), keep(st),
                               dummy, dummy, dummy, dummy, 0, keep(bak)]
            else:
                ptrs, kc, vc = attn_layer(w, m)
                self.k_cache.append(kc)
                self.v_cache.append(vc)
                layer_ptrs += ptrs
        assert len(layer_ptrs) == LAYER_FIELDS * cfg.n_layers

        cos, sin = rope_tables(cfg, max_seq, device)
        i64 = dict(device=device, dtype=torch.int64)
        T = self.ext.SPEC_T  # every activation buffer holds a speculative cycle's two tokens
        # LL activation buffers (see csrc/megakernel.cu): (epoch << 32) | fp32 bits
        self.ll = [torch.zeros(n, **i64) for n in (T * cfg.hidden, T * 8224, T * 2048, T * 2048, T * cfg.intermediate,
                                                   (2 * T + 2) * self.grid, T * cfg.hidden)]
        self.attn_cnt = torch.zeros(T * cfg.n_kv_heads, device=device, dtype=torch.int32)
        part = torch.zeros(T * cfg.n_kv_heads * self.ext.MAX_SPLITS * 4 * self.ext.PART_STRIDE, **f32)
        lm_head = mat(w["embed_tokens.weight"])
        self.embed = w["embed_tokens.weight"]
        self.final_norm = w["norm.weight"]
        self.mtp = self.draft_ids = None
        draft_head, draft_ids = lm_head, 0
        if mtp:
            from .mtp import MTP, load_mtp

            wm = load_mtp(model, device)
            mtp_ptrs, mkc, mvc = attn_layer(wm, "layers.0.")
            mtp_g = [mat(wm["fc.weight"]), keep(wm["pre_fc_norm_embedding.weight"]),
                     keep(wm["pre_fc_norm_hidden.weight"]), keep(wm["norm.weight"])]
            self.mtp = MTP(cfg, wm, self.embed, max_seq, device)
            self.mtp.k_cache, self.mtp.v_cache = mkc, mvc  # host-side prefill writes the kernel's cache
            if 0 < draft_vocab < cfg.vocab:
                ranked = torch.from_numpy(np.load(DRAFT_VOCAB_FILE)[:draft_vocab])
                assert len(ranked) == draft_vocab, f"{DRAFT_VOCAB_FILE} ranks only {len(ranked)} tokens"
                # ascending ids, so an argmax tie still goes to the smallest token id
                self.draft_ids = ranked.sort().values.to(device=device, dtype=torch.int32)
                draft_head, draft_ids = mat(self.embed[self.draft_ids.long()]), keep(self.draft_ids)
        else:
            mtp_ptrs, mtp_g = [0] * LAYER_FIELDS, [0] * 4
        g = [keep(self.embed), keep(self.final_norm), keep(cos), keep(sin), *[keep(t) for t in self.ll[:6]],
             keep(part), keep(self.attn_cnt), lm_head, keep(self.ll[6]), *mtp_g, draft_head, draft_ids]
        del w
        self.draft_vocab = len(self.draft_ids) if self.draft_ids is not None else cfg.vocab
        self.ext.set_params(layer_ptrs, mtp_ptrs, g, max_seq, self.draft_vocab)
        self.weight_bytes = self._weight_bytes()
        self.reset()

    def _weight_bytes(self):
        """Bytes of weights streamed per decode step: (layers, LM head)."""
        c = self.cfg
        per_layer = {
            "linear_attention": 8224 * 1024 + 1024 * 2048,
            "full_attention": 5120 * 1024 + 1024 * 2048,
        }
        mlp = 3 * c.intermediate * c.hidden
        per_weight = 1 + 2 / Q8_GROUP if self.quant == "int8" else 2
        body = sum(per_layer[t] + mlp for t in c.layer_types) * per_weight
        return int(body), int(c.vocab * c.hidden * per_weight)

    def reset(self):
        """Start a new sequence (the KV cache does not need clearing: positions are rewritten)."""
        for t in self.conv_hist + self.state + self.ll:
            t.zero_()
        self.attn_cnt.zero_()
        self.epoch = 1  # LL buffers are zeroed, so epoch 0 never looks ready
        self.pos = 0
        self.mtp_h = None  # (position, hidden state) whose MTP cache entry still awaits its next token

    # ---- MTP cache for tokens fed outside speculative cycles
    # The MTP entry at position i is built from (h_i, x_{i+1}), so it can only be written once
    # the token after i is known. Speculative cycles maintain it in the kernel; for prompt
    # tokens (fed one per step, or by the HF prefill) it is computed here in one batch.
    def mtp_fed(self, start, hidden, tokens):
        """Positions start .. start+n-1 were fed `tokens` and produced final hidden states `hidden`."""
        if self.mtp is None or not len(tokens):
            return
        tok = torch.as_tensor(tokens, device=self.device)
        self._mtp_close(start, tok[0])
        if len(tok) > 1:
            self.mtp.fill_kv(hidden[:-1], tok[1:], start)
        self.mtp_h = (start + len(tok) - 1, hidden[-1:])

    def _mtp_close(self, pos, token):
        """Token `token` is going to position `pos`: finish the entry for pos - 1 if it is pending."""
        if self.mtp_h is not None and self.mtp_h[0] == pos - 1:
            self.mtp.fill_kv(self.mtp_h[1], torch.as_tensor([token], device=self.device), pos - 1)
        self.mtp_h = None

    @torch.no_grad()
    def run(self, tokens, n_generate=1, temperature=0.0, seed=0, stop_ids=(), return_logits=False, prof=None,
            prof_step=0, timing=False, all_logits=False, track_mtp=False):
        """Feed `tokens`, then let the model continue for up to `n_generate` predicted tokens.

        One kernel launch: the prompt tokens are fed one per step (no LM head except on the
        last), and each predicted token is fed back in on the GPU. Returns the predicted ids
        (ending early with a stop id, which is *not* fed into the state), plus their logits if
        `return_logits`. With `all_logits`, logits for every fed token are returned instead
        (for scoring text). With `timing`, `self.last_step_ns` holds GPU time per step.
        `track_mtp` keeps the MTP head's cache in step with the fed tokens (see `mtp_fed`).
        """
        tokens = list(tokens)
        assert tokens, "need at least one input token"
        assert n_generate >= 1
        n_forced = len(tokens)
        n_steps = n_forced + n_generate - 1
        if self.pos + n_steps > self.max_seq:
            raise ValueError(f"sequence would exceed max_seq={self.max_seq}")
        dev = self.device
        forced = torch.tensor(tokens, dtype=torch.int32, device=dev)
        stops = torch.tensor(list(stop_ids) or [-1], dtype=torch.int32, device=dev)
        out = torch.zeros(n_steps, dtype=torch.int64, device=dev)
        info = torch.zeros(8, dtype=torch.int32, device=dev)
        ts = torch.zeros(n_steps + 1, dtype=torch.int64, device=dev) if timing else None
        lm_from = 0 if all_logits else n_forced - 1
        logits = torch.empty(n_steps, self.cfg.vocab, device=dev) if return_logits or all_logits else None
        track_mtp = track_mtp and self.mtp is not None
        resid = torch.empty(n_steps, self.cfg.hidden, device=dev) if track_mtp else None
        ptr = lambda t: t.data_ptr() if t is not None else 0
        self.ext.launch(forced.data_ptr(), n_forced, n_steps, self.pos, self.epoch, out.data_ptr(), ptr(logits),
                        ptr(prof), prof_step, 1.0 / temperature if temperature > 0 else 0.0, seed, self.grid,
                        self.pf_bytes, stops.data_ptr(), len(stop_ids), info.data_ptr(), ptr(ts),
                        int(self.quant == "int8"), lm_from, ptr(resid), 0, 0, 0, 0, 0, 0)
        done, epoch = info[:2].tolist()
        if track_mtp and done:
            fed = (list(tokens) + (0xFFFFFFFF - (out[n_forced - 1: done - 1] & 0xFFFFFFFF)).tolist())[:done]
            self.mtp_fed(self.pos, rmsnorm(resid[:done], self.final_norm, self.cfg.eps), fed)
        elif done:
            self.mtp_h = None
        self.pos += done
        self.epoch = epoch & 0xFFFFFFFF
        if self.epoch > 0xFFFF0000:
            raise RuntimeError("epoch counter exhausted; call reset()")
        if timing:
            t = ts[: done + 1].cpu()
            self.last_step_ns = (t[1:] - t[:-1]).tolist()
        ids = (0xFFFFFFFF - (out[n_forced - 1: done] & 0xFFFFFFFF)).tolist()
        if all_logits:
            return ids, logits[:done]
        if return_logits:
            return ids, logits[n_forced - 1: done]
        return ids

    @torch.no_grad()
    def run_spec(self, token, draft, n_generate, temperature=0.0, seed=0, stop_ids=(), timing=False,
                 return_logits=False, prof=None, prof_step=0):
        """Speculative decoding: `token` is pending at self.pos and `draft` guesses the one after.

        One launch of up to `n_generate` cycles (each verifies the pending token plus the draft in
        a single pass over the weights). Returns (ids, next_draft): the emitted ids, at most
        `n_generate` and ending early with a stop id, exactly as `run([token], n_generate)` would
        produce them; the state has absorbed all but the last id, which is the next pending token
        and `next_draft` the MTP head's guess for the token after it. Any `draft` is correct: a
        wrong one only costs its cycle. `self.last_cycles` lists (tokens emitted, draft) per cycle.
        """
        assert self.mtp is not None, "construct with mtp=True"
        n_cycles = min(n_generate, MAX_STEPS_PER_LAUNCH)
        if self.pos + 2 * n_cycles + 1 > self.max_seq:
            raise ValueError(f"sequence would exceed max_seq={self.max_seq}")
        self._mtp_close(self.pos, token)
        dev = self.device
        stops = torch.tensor(list(stop_ids) or [-1], dtype=torch.int32, device=dev)
        out = torch.zeros(n_generate, dtype=torch.int32, device=dev)
        cycles = torch.zeros(2 * n_cycles, dtype=torch.int32, device=dev)
        info = torch.zeros(8, dtype=torch.int32, device=dev)
        ts = torch.zeros(n_cycles + 1, dtype=torch.int64, device=dev) if timing else None
        logits = torch.empty(n_cycles, 2, self.cfg.vocab, device=dev) if return_logits else None
        ptr = lambda t: t.data_ptr() if t is not None else 0
        self.ext.launch(0, 0, n_cycles, self.pos, self.epoch, 0, ptr(logits), ptr(prof), prof_step,
                        1.0 / temperature if temperature > 0 else 0.0, seed, self.grid, self.pf_bytes,
                        stops.data_ptr(), len(stop_ids), info.data_ptr(), ptr(ts), int(self.quant == "int8"), 0,
                        0, 1, int(token), int(draft), n_generate, out.data_ptr(), cycles.data_ptr())
        emitted, epoch, pos, pending, next_draft, n_run = info[:6].tolist()
        self.pos = pos
        self.epoch = epoch & 0xFFFFFFFF
        if self.epoch > 0xFFFF0000:
            raise RuntimeError("epoch counter exhausted; call reset()")
        self.last_cycles = cycles[: 2 * n_run].view(-1, 2).tolist()
        if timing:
            t = ts[: n_run + 1].cpu()
            self.last_step_ns = (t[1:] - t[:-1]).tolist()
        ids = out[:emitted].tolist()
        assert ids[-1] == pending
        if return_logits:
            self.last_logits = logits[:n_run]
        return ids, next_draft

    def prefill(self, tokens, chunk=512):
        """Feed all but the last token (in launches short enough for the GPU watchdog)."""
        tokens = list(tokens)
        for i in range(0, len(tokens) - 1, chunk):
            self.run(tokens[i: min(i + chunk, len(tokens) - 1)], track_mtp=True)
        return tokens[-1]

    def prof_buffer(self):
        n = self.grid * self.ext.PROF_MAX_OPS * self.ext.PROF_FIELDS + self.grid * self.ext.PROF_FIELDS
        return torch.zeros(n, dtype=torch.int64, device=self.device)

    def generate(self, prompt_ids, max_new_tokens=256, temperature=0.0, seed=0, stop_ids=(), chunk=16, spec=False):
        """Yield generated ids. The state ends exactly after the last fed token, so a
        conversation can continue by passing the next prompt (starting with the stop id).
        With `spec`, decode speculatively with the MTP head (same ids, fewer passes)."""
        last = self.prefill(prompt_ids)
        produced = 0
        draft = last  # any guess is valid for the first cycle; the MTP head drafts every later one
        while produced < max_new_tokens:
            n = min(chunk, max_new_tokens - produced, MAX_STEPS_PER_LAUNCH)
            if spec:
                ids, draft = self.run_spec(last, draft, n, temperature=temperature, seed=seed, stop_ids=stop_ids)
            else:
                ids = self.run([last], n_generate=n, temperature=temperature, seed=seed, stop_ids=stop_ids)
            for t in ids:
                yield t
                produced += 1
            if ids and ids[-1] in stop_ids:
                return
            last = ids[-1]
        # max_new_tokens reached: the final predicted id has not been fed yet
        self.pending = last
