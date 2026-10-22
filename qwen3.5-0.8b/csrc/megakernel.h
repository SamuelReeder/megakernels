// Shared definitions for the Qwen3.5-0.8B decode megakernel.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace mk {

using bf16 = __nv_bfloat16;

// ---- model shape (Qwen3.5-0.8B text decoder) ----
constexpr int HID = 1024;
constexpr int N_LAYERS = 24;
constexpr int INTER = 3584;
constexpr int VOCAB = 248320;
constexpr float EPS = 1e-6f;

// Gated DeltaNet ("linear_attention") layers
constexpr int LIN_H = 16, LIN_DK = 128, LIN_DV = 128;
constexpr int LIN_QK = LIN_H * LIN_DK;                  // 2048
constexpr int LIN_V = LIN_H * LIN_DV;                   // 2048
constexpr int CONV_DIM = 2 * LIN_QK + LIN_V;            // 6144
constexpr int CONV_K = 4;
constexpr int CONV_RING = 8;                            // conv history slots (pos % 8); >= 5 so two
                                                        // tokens per step never overwrite a slot still read
constexpr int GDN_IN_ROWS = CONV_DIM + LIN_V + 2 * LIN_H;  // qkv | z | b | a = 8224
constexpr int GDN_Z_OFF = CONV_DIM;                     // 6144
constexpr int GDN_B_OFF = CONV_DIM + LIN_V;             // 8192
constexpr int GDN_A_OFF = GDN_B_OFF + LIN_H;            // 8208
constexpr int GDN_SLICE = 32;                           // value columns per recurrence task

// Gated full-attention layers
constexpr int NQ = 8, NKV = 2, HD = 256, ROT = 64;
constexpr int Q_PER_KV = NQ / NKV;                      // 4
constexpr int ATT_K_OFF = NQ * 2 * HD;                  // 4096 (q_proj interleaves [q | gate] per head)
constexpr int ATT_V_OFF = ATT_K_OFF + NKV * HD;         // 4608
constexpr int ATT_IN_ROWS = ATT_V_OFF + NKV * HD;       // 5120
constexpr int MAX_SPLITS = 42;                          // split-K chunks per KV head
constexpr int PART_STRIDE = HD + 2;                     // acc[256], m, l

constexpr int PROJ_MAX = GDN_IN_ROWS;                   // projl stride per token

// ---- weight formats ----
// bf16: plain row-major. int8: each row is K bytes of (q + 128) followed by K/32 bf16 scales,
// so w = (byte - 128) * scale[k / 32] and every row is still one contiguous TMA-able span.
constexpr int Q8_GROUP = 32;
__host__ __device__ constexpr int row_bytes(int K, bool q8) { return q8 ? K + K / Q8_GROUP * 2 : 2 * K; }

// ---- execution shape ----
constexpr int N_CONSUMER_WARPS = 16;
constexpr int N_CONSUMERS = N_CONSUMER_WARPS * 32;
constexpr int N_THREADS = N_CONSUMERS + 32;             // + one producer warp
// Tunables (overridable at build time with QWEN_MK_DEFINES="-DMK_STAGES=6 ..." for A/B runs)
#ifndef MK_STAGES
#define MK_STAGES 5
#endif
#ifndef MK_CHUNK
#define MK_CHUNK 15232  // bytes per ring slot: exactly 14 / 7 / 4 int8 rows of K = 1024 / 2048 / 3584
#endif
constexpr int STAGES = MK_STAGES;
constexpr int CHUNK = MK_CHUNK;
constexpr int VEC_BYTES = 16384;                        // GEMV input vector / op scratch
constexpr int RED_FLOATS = 80;
constexpr int XS_STRIDE = 16;                           // residual rows a CTA owns (<= 13), per token
constexpr int SMEM_BYTES = STAGES * CHUNK + VEC_BYTES + RED_FLOATS * 4 + 2 * STAGES * 8;

// Speculative decoding: each cycle verifies SPEC_T = 2 tokens (the last accepted one and the
// MTP head's draft) in one pass over the weights. Two tokens' MLP activations need 28 KB of
// input vector, so the ring gives up a stage to stay under the 99 KB shared-memory limit.
constexpr int SPEC_T = 2;
constexpr int SPEC_STAGES = STAGES - 1;
constexpr int SPEC_VEC_BYTES = SPEC_T * INTER * 4;
constexpr int SPEC_SMEM_BYTES = SPEC_STAGES * CHUNK + SPEC_VEC_BYTES + RED_FLOATS * 4 + 2 * SPEC_STAGES * 8;

// Ops per decode step: 5 per layer (+1 for the LM head when its output is needed).
// Every op stamps its outputs with the next epoch.
constexpr int OPS_PER_LAYER = 5;
constexpr int PROF_FIELDS = 4;                           // t_ready, t_done, full_wait_ns, (step start)
// A speculative cycle adds the MTP head: fc, one decoder layer, the draft head.
constexpr int PROF_MAX_OPS = N_LAYERS * OPS_PER_LAYER + 1 + 1 + OPS_PER_LAYER + 1;

struct LayerPtrs {
    const bf16* w_in;      // GDN: [8224,1024] = qkv|z|b|a   attn: [5120,1024] = q(+gate)|k|v
    const bf16* w_out;     // [1024,2048]
    const bf16* w_gu;      // [7168,1024], rows interleaved gate0, up0, gate1, up1, ...
    const bf16* w_down;    // [1024,3584]
    const bf16* in_norm;   // [1024]
    const bf16* post_norm; // [1024]
    // GDN only
    const bf16* conv_w;    // [6144,4]
    const float* A_log;    // [16]
    const bf16* dt_bias;   // [16]
    const float* gdn_norm; // [128]
    float* conv_hist;      // [6144,8] ring buffer indexed by pos % 8
    float* state;          // [16][128 v][128 k]
    float* state_bak;      // same; speculative cycles keep the state after their first token here
    // attention only
    const bf16* q_norm;    // [256]
    const bf16* k_norm;    // [256]
    bf16* k_cache;         // [2][max_seq][256]
    bf16* v_cache;
    long long is_attn;
};

struct Params {
    LayerPtrs layer[N_LAYERS];
    const bf16* embed;      // [VOCAB,1024] bf16 token embeddings (input lookup)
    const void* lm_head;    // tied LM head in the kernel's weight format (== embed for bf16)
    LayerPtrs mtp;          // MTP head's decoder layer (gated full attention + MLP)
    const void* mtp_fc;     // [1024, 2048]: [norm(embed(next token)) | norm(hidden)] -> 1024
    const bf16* mtp_norm_e; // pre_fc_norm_embedding [1024]
    const bf16* mtp_norm_h; // pre_fc_norm_hidden [1024]
    const bf16* mtp_norm;   // final norm of the MTP layer [1024]
    const void* draft_head; // LM head rows the draft chooses from (kernel weight format)
    const int* draft_ids;   // token id of each draft_head row (nullptr: row == token id)
    long long draft_vocab;
    const bf16* final_norm; // [1024]
    const float* rope_cos;  // [max_seq,32]
    const float* rope_sin;
    // LL activation buffers: each word is (epoch << 32) | fp32 bits
    // (all [SPEC_T][...]: token j of a speculative cycle at offset j * width)
    unsigned long long* xl;     // residual stream [1024]
    unsigned long long* projl;  // input-projection output [8224]
    unsigned long long* corel;  // GDN recurrence output [2048]
    unsigned long long* attnl;  // attention output [2048]
    unsigned long long* actl;   // MLP activation [3584]
    unsigned long long* lml;    // per-CTA LM-head winners [grid][SPEC_T][2] (key hi, key lo), then draft [grid][2]
    unsigned long long* xml;    // MTP residual stream [1024]
    float* attn_part;           // [SPEC_T][NKV][MAX_SPLITS][4][PART_STRIDE]
    unsigned* attn_cnt;         // [SPEC_T][NKV]
    long long max_seq;
};

struct Args {
    const int* forced;          // tokens fed at steps [0, n_forced)
    int n_forced;
    int lm_from;                // first step that runs the LM head (n_forced - 1, or 0 for all logits)
    int n_steps;
    int start_pos;
    int prof_step;              // which step to record into prof (if prof != nullptr)
    unsigned epoch_base;        // epoch of the launch's first op; advanced by the host
    unsigned long long* tokens_out;  // [n_steps] packed (score, ~index); zeroed by host
    float* logits_out;          // optional [n_steps][VOCAB]
    unsigned long long* prof;   // optional [grid][PROF_MAX_OPS][PROF_FIELDS] + producer stall
    float inv_temperature;      // 0 => greedy
    long long pf_bytes;         // L2 prefetch distance ahead of the smem ring, per CTA
    int q8;                     // weights are int8 (selects the kernel instantiation)
    const int* stop_ids;        // stop when one of these is predicted (it is not fed)
    int n_stop;
    int* out_info;              // [steps (tokens if spec), epoch after the last op, then if spec:
                                //  final pos, pending token, its draft, cycles run]
    unsigned long long* step_ts;  // optional [n_steps + 1] globaltimer at each step boundary
    unsigned seed;
    float* resid_out;           // optional [n_steps][1024]: residual stream after each step's last layer
    // speculative decoding (n_steps counts cycles; a cycle emits 1 or 2 tokens)
    int spec;
    int first_tok, first_draft; // pending token at start_pos and the draft for start_pos + 1
    int limit;                  // emit at most this many tokens
    int* spec_out;              // [limit] emitted token ids
    int* cycle_out;             // optional [n_steps][2]: tokens emitted, draft that was verified
};

}  // namespace mk
