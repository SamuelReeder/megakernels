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
constexpr int GDN_IN_ROWS = CONV_DIM + LIN_V + 2 * LIN_H;  // qkv | z | b | a = 8224
constexpr int GDN_Z_OFF = CONV_DIM;                     // 6144
constexpr int GDN_B_OFF = CONV_DIM + LIN_V;             // 8192
constexpr int GDN_A_OFF = GDN_B_OFF + LIN_H;            // 8208
constexpr int GDN_SLICE = 16;                           // value columns per recurrence task

// Gated full-attention layers
constexpr int NQ = 8, NKV = 2, HD = 256, ROT = 64;
constexpr int Q_PER_KV = NQ / NKV;                      // 4
constexpr int ATT_K_OFF = NQ * 2 * HD;                  // 4096 (q_proj interleaves [q | gate] per head)
constexpr int ATT_V_OFF = ATT_K_OFF + NKV * HD;         // 4608
constexpr int ATT_IN_ROWS = ATT_V_OFF + NKV * HD;       // 5120
constexpr int MAX_SPLITS = 42;                          // split-K chunks per KV head
constexpr int PART_STRIDE = HD + 2;                     // acc[256], m, l

constexpr int PROJ_MAX = GDN_IN_ROWS;

// ---- execution shape ----
constexpr int N_CONSUMER_WARPS = 16;
constexpr int N_CONSUMERS = N_CONSUMER_WARPS * 32;
constexpr int N_THREADS = N_CONSUMERS + 32;             // + one producer warp
constexpr int STAGES = 5;
constexpr int CHUNK = 14336;                            // bytes per ring slot (2 rows of K=3584)
constexpr int VEC_BYTES = 16384;                        // GEMV input vector / op scratch
constexpr int RED_FLOATS = 64;
constexpr int SMEM_BYTES = STAGES * CHUNK + VEC_BYTES + RED_FLOATS * 4 + 2 * STAGES * 8;

// Ops per decode step: 5 per layer (+1 for the LM head when its output is needed).
constexpr int OPS_PER_LAYER = 5;
constexpr int PROF_FIELDS = 4;                           // t_arrive, t_release, full_wait_ns, unused
constexpr int PROF_MAX_OPS = N_LAYERS * OPS_PER_LAYER + 1;

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
    float* conv_hist;      // [6144,4] ring buffer indexed by pos % 4
    float* state;          // [16][128 v][128 k]
    // attention only
    const bf16* q_norm;    // [256]
    const bf16* k_norm;    // [256]
    bf16* k_cache;         // [2][max_seq][256]
    bf16* v_cache;
    long long is_attn;
};

struct Params {
    LayerPtrs layer[N_LAYERS];
    const bf16* embed;      // [VOCAB,1024], tied LM head
    const bf16* final_norm; // [1024]
    const float* rope_cos;  // [max_seq,32]
    const float* rope_sin;
    float* x;               // residual stream [1024]
    float* proj;            // input-projection output [8224]
    float* core;            // GDN recurrence output [2048]
    float* attn;            // attention output [2048]
    float* act;             // MLP activation [3584]
    float* attn_part;       // [NKV][MAX_SPLITS][4][PART_STRIDE]
    unsigned* attn_cnt;     // [NKV]
    unsigned long long* bar;
    long long max_seq;
};

struct Args {
    const int* forced;          // tokens fed at steps [0, n_forced)
    int n_forced;
    int n_steps;
    int start_pos;
    int prof_step;              // which step to record into prof (if prof != nullptr)
    unsigned long long bar_base;
    unsigned long long* tokens_out;  // [n_steps] packed (score, ~index); zeroed by host
    float* logits_out;          // optional [n_steps][VOCAB]
    unsigned long long* prof;   // optional [grid][PROF_MAX_OPS][PROF_FIELDS] + producer stall
    float inv_temperature;      // 0 => greedy
    unsigned seed;
};

}  // namespace mk
