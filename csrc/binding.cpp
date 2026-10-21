// Thin PyTorch binding. Python owns every tensor and passes raw device pointers,
// so the C++ side only assembles the constant-memory parameter block and launches.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "megakernel.h"

namespace mk {
void set_params(const Params& p);
cudaError_t launch(const Args& a, int grid, cudaStream_t stream);
}  // namespace mk

constexpr int LAYER_FIELDS = 17;

template <class T>
static T ptr(int64_t v) { return reinterpret_cast<T>(static_cast<uintptr_t>(v)); }

void set_params(std::vector<int64_t> layer, std::vector<int64_t> g, int64_t max_seq) {
    TORCH_CHECK(layer.size() == (size_t)mk::N_LAYERS * LAYER_FIELDS, "bad layer pointer count");
    TORCH_CHECK(g.size() == 12, "bad global pointer count");
    mk::Params p{};
    for (int l = 0; l < mk::N_LAYERS; l++) {
        const int64_t* f = &layer[l * LAYER_FIELDS];
        mk::LayerPtrs& L = p.layer[l];
        L.w_in = ptr<const mk::bf16*>(f[0]);
        L.w_out = ptr<const mk::bf16*>(f[1]);
        L.w_gu = ptr<const mk::bf16*>(f[2]);
        L.w_down = ptr<const mk::bf16*>(f[3]);
        L.in_norm = ptr<const mk::bf16*>(f[4]);
        L.post_norm = ptr<const mk::bf16*>(f[5]);
        L.conv_w = ptr<const mk::bf16*>(f[6]);
        L.A_log = ptr<const float*>(f[7]);
        L.dt_bias = ptr<const mk::bf16*>(f[8]);
        L.gdn_norm = ptr<const float*>(f[9]);
        L.conv_hist = ptr<float*>(f[10]);
        L.state = ptr<float*>(f[11]);
        L.q_norm = ptr<const mk::bf16*>(f[12]);
        L.k_norm = ptr<const mk::bf16*>(f[13]);
        L.k_cache = ptr<mk::bf16*>(f[14]);
        L.v_cache = ptr<mk::bf16*>(f[15]);
        L.is_attn = f[16];
    }
    p.embed = ptr<const mk::bf16*>(g[0]);
    p.final_norm = ptr<const mk::bf16*>(g[1]);
    p.rope_cos = ptr<const float*>(g[2]);
    p.rope_sin = ptr<const float*>(g[3]);
    p.xl = ptr<unsigned long long*>(g[4]);
    p.projl = ptr<unsigned long long*>(g[5]);
    p.corel = ptr<unsigned long long*>(g[6]);
    p.attnl = ptr<unsigned long long*>(g[7]);
    p.actl = ptr<unsigned long long*>(g[8]);
    p.lml = ptr<unsigned long long*>(g[9]);
    p.attn_part = ptr<float*>(g[10]);
    p.attn_cnt = ptr<unsigned*>(g[11]);
    p.max_seq = max_seq;
    mk::set_params(p);
    C10_CUDA_CHECK(cudaGetLastError());
}

void launch(int64_t forced, int64_t n_forced, int64_t n_steps, int64_t start_pos, int64_t epoch_base,
            int64_t tokens_out, int64_t logits_out, int64_t prof, int64_t prof_step, double inv_temperature,
            int64_t seed, int64_t grid, int64_t pf_bytes) {
    mk::Args a{};
    a.forced = ptr<const int*>(forced);
    a.n_forced = (int)n_forced;
    a.n_steps = (int)n_steps;
    a.start_pos = (int)start_pos;
    a.prof_step = (int)prof_step;
    a.epoch_base = (unsigned)epoch_base;
    a.tokens_out = ptr<unsigned long long*>(tokens_out);
    a.logits_out = ptr<float*>(logits_out);
    a.prof = ptr<unsigned long long*>(prof);
    a.inv_temperature = (float)inv_temperature;
    a.seed = (unsigned)seed;
    a.pf_bytes = pf_bytes;
    C10_CUDA_CHECK(mk::launch(a, (int)grid, c10::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("set_params", &set_params);
    m.def("launch", &launch);
    m.attr("SMEM_BYTES") = mk::SMEM_BYTES;
    m.attr("N_THREADS") = mk::N_THREADS;
    m.attr("OPS_PER_LAYER") = mk::OPS_PER_LAYER;
    m.attr("PROF_MAX_OPS") = mk::PROF_MAX_OPS;
    m.attr("PROF_FIELDS") = mk::PROF_FIELDS;
    m.attr("MAX_SPLITS") = mk::MAX_SPLITS;
    m.attr("PART_STRIDE") = mk::PART_STRIDE;
}
