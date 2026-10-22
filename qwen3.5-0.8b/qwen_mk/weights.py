"""Load the text-only weights of Qwen3.5-0.8B from a Hugging Face snapshot."""

import json
import os
from dataclasses import dataclass

import torch
from safetensors import safe_open

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


@dataclass
class Config:
    hidden: int
    n_layers: int
    layer_types: list
    intermediate: int
    vocab: int
    eps: float
    # full attention
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    # gated deltanet
    lin_heads: int
    lin_k_dim: int
    lin_v_dim: int
    conv_k: int
    eos_token_id: int

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            c = json.load(f)
        t = c["text_config"]
        rope = t["rope_parameters"]
        assert t["linear_num_key_heads"] == t["linear_num_value_heads"]
        return cls(
            hidden=t["hidden_size"],
            n_layers=t["num_hidden_layers"],
            layer_types=t["layer_types"],
            intermediate=t["intermediate_size"],
            vocab=t["vocab_size"],
            eps=t["rms_norm_eps"],
            n_heads=t["num_attention_heads"],
            n_kv_heads=t["num_key_value_heads"],
            head_dim=t["head_dim"],
            rotary_dim=int(t["head_dim"] * rope.get("partial_rotary_factor", 1.0)),
            rope_theta=rope["rope_theta"],
            lin_heads=t["linear_num_value_heads"],
            lin_k_dim=t["linear_key_head_dim"],
            lin_v_dim=t["linear_value_head_dim"],
            conv_k=t["linear_conv_kernel_dim"],
            eos_token_id=t["eos_token_id"],
        )


def snapshot_path(model=DEFAULT_MODEL):
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])


def load(model=DEFAULT_MODEL, device="cuda", dtype=None):
    """Return (config, weights) where weights maps short names to tensors.

    Names drop the `model.language_model.` prefix, e.g. `layers.3.self_attn.q_proj.weight`.
    Tensors keep their checkpoint dtype unless `dtype` is given.
    """
    path = snapshot_path(model)
    cfg = Config.from_json(os.path.join(path, "config.json"))
    prefix = "model.language_model."
    weights = {}
    for name in sorted(os.listdir(path)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(path, name), "pt", device="cpu") as f:
            for k in f.keys():
                if not k.startswith(prefix):
                    continue
                t = f.get_tensor(k)
                if dtype is not None:
                    t = t.to(dtype)
                weights[k[len(prefix):]] = t.to(device)
    return cfg, weights
