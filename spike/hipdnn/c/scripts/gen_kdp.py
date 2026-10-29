"""Slice C: write megakernel_vllm.kdp.json (one UKD per exported code object) from the
manifest.jsonl that vLLM's VLLM_MK_HIPDNN_EXPORT writes, and copy the .co files next to it.
Usage: python gen_kdp.py EXPORT_DIR DESCRIPTOR_DIR [MODEL]
"""
import json
import pathlib
import shutil
import sys
import uuid

ENGINE = "1ea37de2-144e-5c63-8e45-2d933f10e747"
DISPATCH = "5b6947c5-91dc-563e-b6a3-a292361ddf2e"
MATCHER = "f94e96df-ef3a-540e-b6a6-ff22077e5943"
KMD = "011b6c3f-087e-546f-975e-16419971006b"


def uid(name):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"megakernel-spike/{name}"))


def main():
    exp, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    model = sys.argv[3] if len(sys.argv) > 3 else "qwen3.8-27b-flydsl-vllm"
    seen = {}
    for line in (exp / "manifest.jsonl").read_text().splitlines():
        e = json.loads(line)
        seen[(e["M"], e["geom"])] = e
    kds = []
    for (m, geom), e in sorted(seen.items()):
        shutil.copy(exp / e["file"], dst / e["file"])
        name = f"megakernel.qwen3_8_27b.vllm_m{m}_{geom}"
        kds.append({
            "version": "1.0", "id": uid(name), "name": name, "arch": ["gfx942"],
            "kernel_source": {"kind": "hsaco", "file": e["file"], "symbol": e["symbol"]},
            "metadata": {"model": model, "M": m, "geom": geom},
            "priority": 0,
            "provenance": {"specialization_contract": {"schema_version": 1, "consumers": [{
                "engine_id": ENGINE, "kmd_id": KMD, "metadata_fields": [],
                "matcher_only_fields": ["model", "M", "geom"], "bindings": {}, "vocabulary": {}}]}},
        })
    kdp = {"version": "1.0", "id": uid("hipkernel:megakernel_qwen38_vllm"),
           "name": "hipkernel:megakernel_qwen38_vllm", "arch": ["gfx942"], "matchers": [MATCHER],
           "engine": ENGINE, "dispatch": DISPATCH, "kernelDescriptors": kds}
    (dst / "megakernel_vllm.kdp.json").write_text(json.dumps(kdp, indent=2) + "\n")
    print(f"{len(kds)} UKDs: {[k['name'] for k in kds]}")


if __name__ == "__main__":
    main()
