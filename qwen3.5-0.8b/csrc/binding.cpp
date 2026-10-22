// Thin PyTorch binding. Python owns every tensor and passes raw device pointers,
// so the C++ side only assembles the constant-memory parameter block and launches.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "megakernel.h"

namespace mk {
void set_params(const Params& p);
cudaError_t launch(const Args& a, int grid, cudaStream_t stream);
int watchdog_fired();
}  // namespace mk

constexpr int LAYER_FIELDS = 18;

template <class T>
static T ptr(int64_t v) { return reinterpret_cast<T>(static_cast<uintptr_t>(v)); }

static void layer_ptrs(mk::LayerPtrs& L, const int64_t* f) {
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
    L.state_bak = ptr<float*>(f[17]);
}

void set_params(std::vector<int64_t> layer, std::vector<int64_t> mtp, std::vector<int64_t> g, int64_t max_seq,
                int64_t draft_vocab) {
    TORCH_CHECK(layer.size() == (size_t)mk::N_LAYERS * LAYER_FIELDS, "bad layer pointer count");
    TORCH_CHECK(mtp.size() == (size_t)LAYER_FIELDS, "bad MTP layer pointer count");
    TORCH_CHECK(g.size() == 20, "bad global pointer count");
    mk::Params p{};
    for (int l = 0; l < mk::N_LAYERS; l++) layer_ptrs(p.layer[l], &layer[l * LAYER_FIELDS]);
    layer_ptrs(p.mtp, mtp.data());
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
    p.lm_head = ptr<const void*>(g[12]);
    p.xml = ptr<unsigned long long*>(g[13]);
    p.mtp_fc = ptr<const void*>(g[14]);
    p.mtp_norm_e = ptr<const mk::bf16*>(g[15]);
    p.mtp_norm_h = ptr<const mk::bf16*>(g[16]);
    p.mtp_norm = ptr<const mk::bf16*>(g[17]);
    p.draft_head = ptr<const void*>(g[18]);
    p.draft_ids = ptr<const int*>(g[19]);
    p.draft_vocab = draft_vocab;
    p.max_seq = max_seq;
    mk::set_params(p);
    C10_CUDA_CHECK(cudaGetLastError());
}

void launch(int64_t forced, int64_t n_forced, int64_t n_steps, int64_t start_pos, int64_t epoch_base,
            int64_t tokens_out, int64_t logits_out, int64_t prof, int64_t prof_step, double inv_temperature,
            int64_t seed, int64_t grid, int64_t pf_bytes, int64_t stop_ids, int64_t n_stop, int64_t out_info,
            int64_t step_ts, int64_t q8, int64_t lm_from, int64_t resid_out, int64_t spec, int64_t first_tok,
            int64_t first_draft, int64_t limit, int64_t spec_out, int64_t cycle_out) {
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
    a.stop_ids = ptr<const int*>(stop_ids);
    a.n_stop = (int)n_stop;
    a.out_info = ptr<int*>(out_info);
    a.step_ts = ptr<unsigned long long*>(step_ts);
    a.q8 = (int)q8;
    a.lm_from = (int)lm_from;
    a.resid_out = ptr<float*>(resid_out);
    a.spec = (int)spec;
    a.first_tok = (int)first_tok;
    a.first_draft = (int)first_draft;
    a.limit = (int)limit;
    a.spec_out = ptr<int*>(spec_out);
    a.cycle_out = ptr<int*>(cycle_out);
    C10_CUDA_CHECK(mk::launch(a, (int)grid, c10::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("set_params", &set_params);
    m.def("launch", &launch);
    m.def("watchdog_fired", &mk::watchdog_fired);
    m.attr("SMEM_BYTES") = mk::SMEM_BYTES;
    m.attr("SPEC_SMEM_BYTES") = mk::SPEC_SMEM_BYTES;
    m.attr("SPEC_T") = mk::SPEC_T;
    m.attr("CONV_RING") = mk::CONV_RING;
    m.attr("N_THREADS") = mk::N_THREADS;
    m.attr("OPS_PER_LAYER") = mk::OPS_PER_LAYER;
    m.attr("PROF_MAX_OPS") = mk::PROF_MAX_OPS;
    m.attr("PROF_FIELDS") = mk::PROF_FIELDS;
    m.attr("MAX_SPLITS") = mk::MAX_SPLITS;
    m.attr("PART_STRIDE") = mk::PART_STRIDE;
}
