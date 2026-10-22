"""Debug: run layer 0 (GDN) stage by stage and compare each hand-off buffer with the
emulating reference. Usage: python tests/debug_layer.py MODEL_DIR [M] [LAYER_TYPE gdn|attn]"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from mk import reference as R  # noqa: E402
from mk.dims import FFN, H, INP_N, NVH, QKV_N, VAL_DIM  # noqa: E402
from mk.pack import Packed  # noqa: E402
from mk.runner import Engine  # noqa: E402

path = sys.argv[1]
M = int(sys.argv[2]) if len(sys.argv) > 2 else 1
KIND = sys.argv[3] if len(sys.argv) > 3 else "gdn"
NLAY = 1 if KIND == "gdn" else 4


def cmp(name, a, b):
    a, b = a.float(), b.float()
    err = ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
    nan = torch.isnan(a).any().item()
    print(f"  {name:10s} rel max err {err:.3e} nan {nan} |ref| {b.abs().max().item():.3e}")


W = R.Weights(path, "mxfp4", keep_raw=True, layers=NLAY)
PK = Packed(W, layers=NLAY)
torch.manual_seed(0)
tok = torch.randint(0, 200000, (M,), device="cuda")
# reference intermediates for the last layer of NLAY, step 0
R.EMULATE = True
st = R.State.new(M, 512)
i = NLAY - 1
h = W["model.language_model.embed_tokens.weight"][tok].float()
for l in range(NLAY - 1):          # attn case: run 3 GDN layers first with the full reference
    x = R.rms0(h, W.L(l, "input_layernorm.weight"))
    h = h + R.gdn(W, l, x, st, l)
    x = R.rms0(h, W.L(l, "post_attention_layernorm.weight"))
    gt, up = R.lin(x, W.L(l, "mlp.gate_proj.weight")), R.lin(x, W.L(l, "mlp.up_proj.weight"))
    h = h + R.lin(gt * torch.sigmoid(gt) * up, W.L(l, "mlp.down_proj.weight"))
x = R.rms0(h, W.L(i, "input_layernorm.weight"))
if KIND == "gdn":
    pre = "linear_attn."
    y_inp = torch.cat([R.lin(x, W.L(i, pre + n + ".weight")) for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")], 1)
    mix = R.gdn(W, i, x, st, 0)
else:
    pre = "self_attn."
    y_inp = torch.cat([R.lin(x, W.L(i, pre + n + ".weight")) for n in ("q_proj", "k_proj", "v_proj")], 1)
    mix = R.attn(W, i, x, st, 0)
h1 = h + mix
x2 = R.rms0(h1, W.L(i, "post_attention_layernorm.weight"))
gt, up = R.lin(x2, W.L(i, "mlp.gate_proj.weight")), R.lin(x2, W.L(i, "mlp.up_proj.weight"))
h2 = h1 + R.lin(gt * torch.sigmoid(gt) * up, W.L(i, "mlp.down_proj.weight"))
N1 = INP_N if KIND == "gdn" else QKV_N
for stop in (1, 3, 4, 5):
    eng = Engine(PK, M, maxlen=512, layers=NLAY, stop=stop)
    eng.step(tok)
    torch.cuda.synchronize()
    print(f"stop={stop}")
    if stop == 1:
        cmp("mix_gemv", eng.y[:, :N1], y_inp)
    if stop == 3:
        cmp("h1", eng.hb[1], h1)
    if stop == 4:
        from mk.pack import gu_order
        cmp("gate_up", eng.y[:, :2 * FFN], gu_order(torch.cat([gt, up], 1).t()).t())
    if stop == 5:
        cmp("h2", eng.hb[0], h2)
