#!/bin/bash
B=$1; export HIP_VISIBLE_DEVICES= PYTHONPYCACHEPREFIX=/tmp/pyc_det
C='Cfg(M={M}, cus=304, maxlen=4096, block=1056, maxblocks=4, kv_s0=1081344, kv_s1=512, kv_s2=1024, kv_sv=256, kv_xpack=False, conv_s0=2162688, ssm_s0=2162688, lm_head=False, c_inp=1, c_ba=8, c_out=1, c_qkv=1, c_o=16, c_router=1, c_e1=4, c_e2=1, c_lm=1, debug_route=False, fence=False, chaos=False, cta_red=True, skip_idle=True)'
for M in 1 2; do for r in 1 2 3; do
  j=$(python3 -c "import json,sys; print(json.dumps(dict(module='vllm.models.qwen3_5_moe.amd.specialized_decode.kernels.kernel', cfg=sys.argv[1].replace('{M}','$M'), nlayers=40, arch='gfx942', out='$B/canon_M${M}_$r.co')))" "$C")
  python -B ~/coexport/det/export_co.py --compile "$j" 2>&1 | grep -E "^COMPILED|Error" | cut -c1-120
  echo "M=$M r=$r $(sha256sum $B/canon_M${M}_$r.co | cut -c1-16)"
done; done
