"""Qwen3.6-35B-A3B text-model dimensions (config.json text_config).

`check(cfg)` asserts them against a checkpoint's config.
"""

H = 2048  # hidden
VOCAB = 248320
NL = 40  # layers; layer i is full attention iff i % 4 == 3
EPS = 1e-6

# Gated DeltaNet (linear attention) layers
NKH, NVH, DK, DV = 16, 32, 128, 128  # V head h uses K head h // (NVH // NKH)
KEY_DIM, VAL_DIM = NKH * DK, NVH * DV  # 2048, 4096
CONV_DIM = 2 * KEY_DIM + VAL_DIM  # 8192 (q | k | v)
CONV_K = 4

# Full-attention layers
NQ, NKV, HD = 16, 2, 256
ROT = 64  # partial_rotary_factor 0.25 x 256
THETA = 1e7

# Mixture of experts
NE, TOPK, FI = 256, 8, 512  # experts, experts per token, expert intermediate
SFI = 512  # shared expert intermediate

N_GDN = NL * 3 // 4  # 30
N_ATT = NL // 4  # 10


def is_attn(i: int) -> bool:
    return i % 4 == 3


def check(cfg: dict) -> None:
    """Assert the constants above against a config.json (top level or text_config)."""
    t = cfg.get("text_config", cfg)
    want = dict(
        hidden_size=H,
        vocab_size=VOCAB,
        num_hidden_layers=NL,
        rms_norm_eps=EPS,
        full_attention_interval=4,
        linear_num_key_heads=NKH,
        linear_num_value_heads=NVH,
        linear_key_head_dim=DK,
        linear_value_head_dim=DV,
        linear_conv_kernel_dim=CONV_K,
        num_attention_heads=NQ,
        num_key_value_heads=NKV,
        head_dim=HD,
        num_experts=NE,
        num_experts_per_tok=TOPK,
        moe_intermediate_size=FI,
        shared_expert_intermediate_size=SFI,
        attn_output_gate=True,
        tie_word_embeddings=False,
    )
    for k, v in want.items():
        assert t.get(k, cfg.get(k)) == v, (k, t.get(k), v)
    rp = t["rope_parameters"]
    assert rp["rope_theta"] == THETA and rp["partial_rotary_factor"] * HD == ROT, rp
    assert t["layer_types"] == [
        "full_attention" if is_attn(i) else "linear_attention" for i in range(NL)
    ]
