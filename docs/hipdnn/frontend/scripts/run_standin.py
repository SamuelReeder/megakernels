# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Launch the engine's IntegrationGpuMegakernel stand-in via the Python API."""

import hashlib
import struct
import sys

import hipdnn_frontend as hipdnn


def check(result):
    assert result.is_good(), result.get_message()


hipdnn.set_engine_plugin_paths([sys.argv[1]], hipdnn.PluginLoadingMode.ABSOLUTE)
handle = hipdnn.create_handle()
graph = hipdnn.Graph().set_compute_data_type(hipdnn.DataType.FLOAT)
graph.set_preferred_engine_id_ext("hipkernel:megakernel_standin")
cus = 304
threads = cus * 256
nlayers = 64
magic = 0x0123456789ABCDEF
tab_words = 16
tab = hipdnn.Tensor.create([tab_words], hipdnn.DataType.INT64).set_uid(1)
glob = hipdnn.Tensor.create([threads + 1], hipdnn.DataType.INT64).set_uid(2)
attributes = (
    hipdnn.MegakernelDecodeAttributes()
    .set_model("megakernel-standin-gfx942")
    .set_src("0123456789abcdef")
    .set_arch("gfx942")
    .set_cus(cus)
    .set_kv("ROCM_ATTN")
    .set_block(784)
    .set_m(1)
    .set_tp(1)
    .set_rt(0)
    .set_lm_head(False)
    .set_nlayers(nlayers)
)
graph.megakernel_decode(tab, glob, attributes)
check(graph.validate())
check(graph.build_operation_graph(handle))
check(graph.create_execution_plans())
check(graph.check_support())
check(graph.build_plans())
assert graph.get_workspace_size() == 0

tab_device = hipdnn.DeviceBuffer(tab_words * 8)
glob_device = hipdnn.DeviceBuffer((threads + 1) * 8)
tab_device.copy_from_host(struct.pack("<Q", magic) + bytes((tab_words - 1) * 8))
glob_device.zeros()
check(graph.execute(handle, {1: tab_device, 2: glob_device}))
hipdnn.hip_device_synchronize()
actual = glob_device.copy_to_host()

# IntegrationGpuMegakernel compares each thread word and the tab word separately.
expected = b"".join(
    struct.pack("<Q", (0x4D4B << 48) | (nlayers << 32) | i) for i in range(threads)
) + struct.pack("<Q", magic)
print(f"stand-in bytes: {len(actual)}, one word per {threads} threads and one tab word")
print(f"expected SHA-256: {hashlib.sha256(expected).hexdigest()}")
print(f"actual   SHA-256: {hashlib.sha256(actual).hexdigest()}")
assert actual == expected, "stand-in output differs from the integration oracle"
