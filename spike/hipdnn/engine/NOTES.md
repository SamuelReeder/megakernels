# Spike B: hipDNN custom op -> hip-kernel-provider ingestor -> kpack kernel (disposable)

rocm-libraries base `470e78bcfb0` (origin/develop, 2026-10-29). Spike commit `665e9800a8f`, local branch
`users/sareeder/megakernel-spike` (no fork remote, not pushed). Full diff: `rocm-libraries-spike.diff`.
Build: CPU-only job on quanta-ccs-aus-f01-19 (AUSTIN, MARKHAM gfx942 nodes full). GPU: banff-cyxtera-s70-4,
MI300X, overlap on job 68209737. Image `hipdnn_latest_gfx942.sqsh` (2026-10-24). Logs in `logs/`.

## Build recipe

```
cmake --fresh --preset hip-kernel-provider -S src -B build -G Ninja \
  -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hip-kernel-provider;hipdnn-python" \
  -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DGPU_TARGETS=gfx942 \
  -DENABLE_CLANG_TIDY=OFF -DENABLE_CLANG_FORMAT=OFF \
  -DHIPDNN_SKIP_TESTS=ON -DHIPKERNELPROVIDER_ENABLE_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=/work/install [-DENABLE_MEGAKERNEL_ENGINE=ON]
cmake --build build -j 64 && cmake --install build
python3 src/projects/hipdnn/python/frontend_wheel_package/pack_frontend_wheel.py --build-dir build --wheel-dir wheel
# wheel step fails (no `build` module in the image); the staged wheel/hipdnn_frontend/ package works:
PYTHONPATH=wheel LD_LIBRARY_PATH=install/lib:/opt/rocm/lib python mk_custom_op.py
```

Times (64 cores): baseline configure 35 s, build 38 s; with the option, configure 2 s, incremental build 29 s.
`rocm_kpack` comes from `/opt/rocm-kpack/python` in the image; zstandard + msgpack present.

## Interface

- Op: `custom_op_id` starts with `megakernel.`; exactly 1 node; 2 inputs `[tab, glob]` (INT64 tensors,
  any dims); 0 outputs allowed; `compute_data_type` must be set (any value, FLOAT used).
- Payload (UTF-8 JSON in `data`): `{"model": str, "M": int, "nlayers": int,
  "geometry": {"grid": [x,y,z], "block": [x,y,z], "lds": int}}`.
- Graph match `megakernel.match_custom_op` (UED `graph_match`) binds tab/glob uids and payload fields.
  KERNEL-scope UMD `megakernel.kernel_matches`: KMD `model` == payload model and KMD `M` == payload M.
- Dispatch `megakernel.dispatch`: `kernel(int64 tab_addr, int64 glob_addr, int32 nlayers)`, grid/block/LDS
  from the payload, `handle.getStream()`, workspace 0. Expected kernarg signature
  by_value 8, by_value 8, by_value 4; the packer records it from the code object and prepare() checks it.

## How slice C swaps in the real kernel

`descriptors/megakernel/qwen38_standin/megakernel_flydsl.kdp.json` already ships the slice A code
objects as `kernel_source {"kind": "hsaco", "file": "qwen38_decode_m{M}.co", "symbol":
"qwen38_decode_m{M}_<hash>"}`, `arch: ["gfx942"]`, metadata `{"model": "qwen3.8-27b-flydsl", "M": M}`.
hkp_pack packs the bytes unchanged (sha256 0c7400e5... for M=1, same as slice A) into
`gfx942/kpack/hip_kernel_provider_gfx942.kpack`. build_plans with that model loads the module and
resolves the symbol (logs/g2.log). Slice C: fill tab/glob as `mk.runner.Engine` does, pass their device
addresses as the two inputs, payload model `qwen3.8-27b-flydsl`, grid [304], block [512], lds 0
(the 57408 B LDS is static), nlayers 64. A new code object = new .co file + one UKD entry.
