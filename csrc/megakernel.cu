// Qwen3.5-0.8B batch-1 decode megakernel for sm_120 (RTX 50-series).
//
// One persistent CTA per SM. Each CTA has a producer warp and 16 consumer warps.
// The producer walks this SM's slice of every weight matrix in execution order and
// streams it into a shared-memory ring with TMA bulk copies. It never waits on the
// grid barriers, so weights for the next op are already in flight while consumers
// wait for the previous op's activations. Consumers run the ops: GEMVs with fused
// prologues (RMSNorm, gated norm, attention gate) and epilogues (residual add, SwiGLU,
// argmax), the Gated DeltaNet recurrence, and split-K attention.
#include "megakernel.h"

namespace mk {

__constant__ Params P;

// ------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* b, uint32_t parity) {
    uint32_t ok;
    asm volatile(
        "{\n .reg .pred p;\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        " selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
    return ok;
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
    while (!mbar_try_wait(b, parity)) {}
}
__device__ __forceinline__ void tma_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t policy) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
        ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy) : "memory");
}
__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned long long ld_acquire(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
// Named barrier over the consumer warps only (the producer warp never joins).
__device__ __forceinline__ void consumer_sync() {
    asm volatile("bar.sync 1, %0;" ::"n"(N_CONSUMERS) : "memory");
}

// ------------------------------------------------------------------ math helpers
__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
    return v;
}
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.f ? x : log1pf(__expf(x)); }

__device__ __forceinline__ void unpack8(uint4 q, float* f) {
    f[0] = bf_lo(q.x); f[1] = bf_hi(q.x); f[2] = bf_lo(q.y); f[3] = bf_hi(q.y);
    f[4] = bf_lo(q.z); f[5] = bf_hi(q.z); f[6] = bf_lo(q.w); f[7] = bf_hi(q.w);
}

// Order-preserving float -> uint so an integer atomicMax picks the largest score.
// Low 32 bits hold ~index so ties resolve to the smallest index (like torch.argmax).
__device__ __forceinline__ unsigned long long pack_score(float s, int idx) {
    uint32_t b = __float_as_uint(s);
    b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((unsigned long long)b << 32) | (0xFFFFFFFFu - (uint32_t)idx);
}
__device__ __forceinline__ int unpack_index(unsigned long long k) { return (int)(0xFFFFFFFFu - (uint32_t)k); }

__device__ __forceinline__ uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
    h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    return h;
}

// ------------------------------------------------------------------ work partition
// Every GEMV is split into contiguous row ranges, one per CTA. A "unit" is G rows
// (G=2 for the interleaved gate/up pairs). Producer and consumers derive the same
// ranges and chunking from these helpers, which keeps the ring in lockstep.
__device__ __forceinline__ void block_slice(int units, int& u0, int& u1) {
    u0 = (int)((long long)units * blockIdx.x / gridDim.x);
    u1 = (int)((long long)units * (blockIdx.x + 1) / gridDim.x);
}

// ------------------------------------------------------------------ producer
__device__ void stream_weights(const bf16* W, int rows, int K, int G, int& c, char* ring, uint64_t* full,
                               uint64_t* empty, uint64_t policy, unsigned long long& stall) {
    int u0, u1;
    block_slice(rows / G, u0, u1);
    const int ub = G * K * 2;
    const int upc = CHUNK / ub;
    for (int u = u0; u < u1; u += upc, c++) {
        const int n = min(upc, u1 - u);
        const int slot = c % STAGES;
        if (c >= STAGES) {
            const uint32_t par = ((c / STAGES) - 1) & 1;
            if (!mbar_try_wait(&empty[slot], par)) {
                unsigned long long t0 = globaltimer();
                mbar_wait(&empty[slot], par);
                stall += globaltimer() - t0;
            }
        }
        mbar_expect_tx(&full[slot], n * ub);
        tma_load(ring + slot * CHUNK, (const char*)W + (size_t)u * ub, n * ub, &full[slot], policy);
    }
}

__device__ void producer(const Args& A, char* ring, uint64_t* full, uint64_t* empty) {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    int c = 0;
    unsigned long long stall = 0;
    for (int s = 0; s < A.n_steps; s++) {
        for (int l = 0; l < N_LAYERS; l++) {
            const LayerPtrs& L = P.layer[l];
            stream_weights(L.w_in, L.is_attn ? ATT_IN_ROWS : GDN_IN_ROWS, HID, 1, c, ring, full, empty, policy, stall);
            stream_weights(L.w_out, HID, 2048, 1, c, ring, full, empty, policy, stall);
            stream_weights(L.w_gu, 2 * INTER, HID, 2, c, ring, full, empty, policy, stall);
            stream_weights(L.w_down, HID, INTER, 1, c, ring, full, empty, policy, stall);
        }
        if (s >= A.n_forced - 1) stream_weights(P.embed, VOCAB, HID, 1, c, ring, full, empty, policy, stall);
        if (A.prof && s == A.prof_step) {
            A.prof[((size_t)gridDim.x * PROF_MAX_OPS + blockIdx.x) * PROF_FIELDS] = stall;
        }
        stall = 0;
    }
}

// ------------------------------------------------------------------ consumer GEMV
template <int K>
__device__ __forceinline__ float dot_row(const char* row, const float* vec, int lane) {
    const uint4* w = (const uint4*)row;
    constexpr int NV = K / 8;
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < (NV + 31) / 32; i++) {
        const int v = lane + 32 * i;
        if (NV % 32 == 0 || v < NV) {
            const uint4 q = w[v];
            const float4 x0 = *(const float4*)(vec + v * 8);
            const float4 x1 = *(const float4*)(vec + v * 8 + 4);
            a = fmaf(bf_lo(q.x), x0.x, a); a = fmaf(bf_hi(q.x), x0.y, a);
            a = fmaf(bf_lo(q.y), x0.z, a); a = fmaf(bf_hi(q.y), x0.w, a);
            a = fmaf(bf_lo(q.z), x1.x, a); a = fmaf(bf_hi(q.z), x1.y, a);
            a = fmaf(bf_lo(q.w), x1.z, a); a = fmaf(bf_hi(q.w), x1.w, a);
        }
    }
    return warp_sum(a);
}

struct Ring {
    char* buf;
    uint64_t* full;
    uint64_t* empty;
    int c;                          // chunk counter, mirrors the producer
    unsigned long long full_wait;   // ns warp 0 spent waiting for weights
};

// Consume this CTA's slice of a [rows, K] GEMV from the ring. Units are dealt to
// warps round-robin; warps run independently and only meet at the op boundary.
// epi(unit, acc[G]) runs on every lane of the owning warp.
template <int K, int G, class Epi>
__device__ __forceinline__ void gemv(int rows, Ring& R, const float* vec, bool timed, Epi epi) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int u0, u1;
    block_slice(rows / G, u0, u1);
    constexpr int UB = G * K * 2;
    constexpr int UPC = CHUNK / UB;
    for (int u = u0; u < u1; u += UPC, R.c++) {
        const int slot = R.c % STAGES;
        const uint32_t par = (R.c / STAGES) & 1;
        if (timed && warp == 0 && !mbar_try_wait(&R.full[slot], par)) {
            unsigned long long t0 = globaltimer();
            mbar_wait(&R.full[slot], par);
            R.full_wait += globaltimer() - t0;
        } else {
            mbar_wait(&R.full[slot], par);
        }
        const int n = min(UPC, u1 - u);
        const char* tile = R.buf + slot * CHUNK;
        for (int j = (warp - (u - u0) % N_CONSUMER_WARPS + N_CONSUMER_WARPS) % N_CONSUMER_WARPS; j < n;
             j += N_CONSUMER_WARPS) {
            float acc[G];
#pragma unroll
            for (int g = 0; g < G; g++) acc[g] = dot_row<K>(tile + (j * G + g) * K * 2, vec, lane);
            epi(u + j, acc);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&R.empty[slot]);
    }
}

// ------------------------------------------------------------------ GEMV prologues
// RMSNorm of the residual stream (or of an embedding row) into the smem input vector.
__device__ void prologue_rmsnorm(float* vec, float* red, const float* x, const bf16* emb, const bf16* w) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    float v0, v1;
    if (emb) { v0 = bf2f(emb[t]); v1 = bf2f(emb[t + 512]); }
    else { v0 = __ldcg(x + t); v1 = __ldcg(x + t + 512); }
    const float ss = warp_sum(v0 * v0 + v1 * v1);
    if (lane == 0) red[warp] = ss;
    consumer_sync();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < N_CONSUMER_WARPS; i++) tot += red[i];
    const float r = rsqrtf(tot / HID + EPS);
    vec[t] = v0 * r * (1.f + bf2f(w[t]));
    vec[t + 512] = v1 * r * (1.f + bf2f(w[t + 512]));
    consumer_sync();
}

// GDN output: per-head RMSNorm of the recurrence output, gated by silu(z). One warp per head.
__device__ void prologue_gated_norm(float* vec, const LayerPtrs& L) {
    const int h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float o[4], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        o[j] = __ldcg(P.core + h * LIN_DV + lane + 32 * j);
        ss += o[j] * o[j];
    }
    const float r = rsqrtf(warp_sum(ss) / LIN_DV + EPS);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int d = lane + 32 * j;
        const float z = __ldcg(P.proj + GDN_Z_OFF + h * LIN_DV + d);
        vec[h * LIN_DV + d] = o[j] * r * L.gdn_norm[d] * silu(z);
    }
    consumer_sync();
}

// Attention output gated by sigmoid(gate); gate rows sit after each head's query rows.
__device__ void prologue_attn_gate(float* vec) {
    for (int i = threadIdx.x; i < NQ * HD; i += N_CONSUMERS) {
        const int h = i / HD, d = i % HD;
        vec[i] = __ldcg(P.attn + i) * sigmoid(__ldcg(P.proj + h * 2 * HD + HD + d));
    }
    consumer_sync();
}

__device__ void prologue_copy(float* vec, const float* src, int n) {
    for (int i = threadIdx.x * 4; i < n; i += N_CONSUMERS * 4) *(float4*)(vec + i) = __ldcg((const float4*)(src + i));
    consumer_sync();
}

// ------------------------------------------------------------------ Gated DeltaNet recurrence
// Task = (head, slice of 16 value columns). Each task recomputes the causal conv + L2 norm
// for its head's q/k (256 channels, cheap) so tasks stay independent. Warp w owns value
// column (slice*16 + w) of the 128x128 fp32 state, stored [head][v][k] so each column is
// one coalesced 512 B row.
__device__ void gdn_op(const LayerPtrs& L, int pos, float* sm, float* red) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    float* sqk = sm;            // [256] conv'd q then k
    float* sv = sm + 256;       // [16]
    float* sc = sm + 272;       // beta, decay, rq, rk
    const int slot = pos & 3, s1 = (pos - 1) & 3, s2 = (pos - 2) & 3, s3 = (pos - 3) & 3;
    constexpr int SLICES = LIN_DV / GDN_SLICE;
    for (int task = blockIdx.x; task < LIN_H * SLICES; task += gridDim.x) {
        const int h = task / SLICES, sl = task % SLICES;
        // causal conv (window of 4 incl. current) + SiLU
        if (t < 256 + GDN_SLICE) {
            const int ch = t < 128 ? h * LIN_DK + t
                         : t < 256 ? LIN_QK + h * LIN_DK + (t - 128)
                                   : 2 * LIN_QK + h * LIN_DV + sl * GDN_SLICE + (t - 256);
            const float cur = __ldcg(P.proj + ch);
            float* hist = L.conv_hist + ch * CONV_K;
            const uint2 wq = *(const uint2*)(L.conv_w + ch * CONV_K);
            const float acc = bf_lo(wq.x) * __ldcg(hist + s3) + bf_hi(wq.x) * __ldcg(hist + s2) +
                              bf_lo(wq.y) * __ldcg(hist + s1) + bf_hi(wq.y) * cur;
            if (t >= 256 || sl == 0) __stcg(hist + slot, cur);
            if (t < 256) sqk[t] = silu(acc); else sv[t - 256] = silu(acc);
        } else if (t == 288) {
            const float beta = sigmoid(__ldcg(P.proj + GDN_B_OFF + h));
            const float g = -__expf(L.A_log[h]) * softplus(__ldcg(P.proj + GDN_A_OFF + h) + bf2f(L.dt_bias[h]));
            sc[0] = beta;
            sc[1] = __expf(g);
        }
        consumer_sync();
        if (warp < 2) {  // L2 norms of q (warp 0) and k (warp 1)
            float ss = 0.f;
#pragma unroll
            for (int j = 0; j < 4; j++) { const float v = sqk[warp * 128 + lane + 32 * j]; ss += v * v; }
            ss = warp_sum(ss);
            if (lane == 0) sc[2 + warp] = rsqrtf(ss + 1e-6f) * (warp == 0 ? rsqrtf((float)LIN_DK) : 1.f);
        }
        consumer_sync();
        const float beta = sc[0], decay = sc[1], rq = sc[2], rk = sc[3];
        const int col = sl * GDN_SLICE + warp;
        float* S = L.state + ((size_t)h * LIN_DV + col) * LIN_DK;
        float s[4], kv[4], qv[4], km = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int k = lane + 32 * j;
            kv[j] = sqk[128 + k] * rk;
            qv[j] = sqk[k] * rq;
            s[j] = __ldcg(S + k) * decay;
            km += s[j] * kv[j];
        }
        km = warp_sum(km);
        const float delta = (sv[warp] - km) * beta;
        float o = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            s[j] = fmaf(kv[j], delta, s[j]);
            __stcg(S + lane + 32 * j, s[j]);
            o += s[j] * qv[j];
        }
        o = warp_sum(o);
        if (lane == 0) __stcg(P.core + h * LIN_DV + col, o);
        consumer_sync();
    }
}

// ------------------------------------------------------------------ gated full attention
// Split-K decode attention. Task = (kv head, chunk of positions); it handles the 4 query
// heads sharing that KV head. Partials are merged by whichever task finishes last.
__device__ void attn_op(const LayerPtrs& L, int pos, float* sm, float* red) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int n = pos + 1;
    int splits = min(MAX_SPLITS, (n + 63) / 64);
    const int chunk = (n + splits - 1) / splits;
    splits = (n + chunk - 1) / chunk;
    float* sq = sm;              // [4][256]
    float* sacc = sm + 1024;     // [4][256]
    float* sml = sm + 2048;      // [16 warps][4 heads][m,l]
    float* skv = sm + 2176;      // [k 256 | v 256] for the new token
    int* sflag = (int*)(red + 32);
    const size_t ms = (size_t)P.max_seq;
    for (int task = blockIdx.x; task < NKV * splits; task += gridDim.x) {
        const int g = task / splits, sp = task % splits;
        const int t0 = sp * chunk, t1 = min(n, t0 + chunk);
        const bool has_new = (t1 == n);
        // q (and the new k) get per-head RMSNorm, written to smem before RoPE
        if (warp < Q_PER_KV + 1) {
            const bool isq = warp < Q_PER_KV;
            if (isq || has_new) {
                const float* src = isq ? P.proj + (g * Q_PER_KV + warp) * 2 * HD : P.proj + ATT_K_OFF + g * HD;
                const bf16* nw = isq ? L.q_norm : L.k_norm;
                float v[8], ss = 0.f;
#pragma unroll
                for (int j = 0; j < 8; j++) { v[j] = __ldcg(src + lane * 8 + j); ss += v[j] * v[j]; }
                const float r = rsqrtf(warp_sum(ss) / HD + EPS);
                float* dst = isq ? sq + warp * HD : skv;
#pragma unroll
                for (int j = 0; j < 8; j++) dst[lane * 8 + j] = v[j] * r * (1.f + bf2f(nw[lane * 8 + j]));
            }
        } else if (warp == Q_PER_KV + 1 && has_new) {
#pragma unroll
            for (int j = 0; j < 8; j++) skv[HD + lane * 8 + j] = __ldcg(P.proj + ATT_V_OFF + g * HD + lane * 8 + j);
        }
        for (int i = t; i < Q_PER_KV * HD; i += N_CONSUMERS) sacc[i] = 0.f;
        consumer_sync();
        if (t < (Q_PER_KV + 1) * 32) {  // RoPE on the first 64 dims (half-split pairs i, i+32)
            const int hh = t >> 5, i = t & 31;
            if (hh < Q_PER_KV || has_new) {
                float* b = hh < Q_PER_KV ? sq + hh * HD : skv;
                const float c = P.rope_cos[pos * (ROT / 2) + i], s = P.rope_sin[pos * (ROT / 2) + i];
                const float x1 = b[i], x2 = b[i + 32];
                b[i] = x1 * c - x2 * s;
                b[i + 32] = x2 * c + x1 * s;
            }
        }
        consumer_sync();
        if (has_new) {
            const int d = t & (HD - 1);
            bf16* cache = (t < HD) ? L.k_cache : L.v_cache;
            cache[((size_t)g * ms + pos) * HD + d] = __float2bfloat16(skv[t]);
            consumer_sync();
        }
        // each warp scans positions t0+warp, t0+warp+16, ... with an online softmax
        float q[Q_PER_KV][8], m[Q_PER_KV], l[Q_PER_KV], acc[Q_PER_KV][8];
#pragma unroll
        for (int h = 0; h < Q_PER_KV; h++) {
            m[h] = -1e30f;
            l[h] = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) { q[h][j] = sq[h * HD + lane * 8 + j] * 0.0625f; acc[h][j] = 0.f; }
        }
        const bf16* Kc = L.k_cache + (size_t)g * ms * HD + lane * 8;
        const bf16* Vc = L.v_cache + (size_t)g * ms * HD + lane * 8;
        for (int tt = t0 + warp; tt < t1; tt += N_CONSUMER_WARPS) {
            float kf[8], vf[8], sc[Q_PER_KV];
            unpack8(__ldcg((const uint4*)(Kc + (size_t)tt * HD)), kf);
            unpack8(__ldcg((const uint4*)(Vc + (size_t)tt * HD)), vf);
#pragma unroll
            for (int h = 0; h < Q_PER_KV; h++) {
                float d = 0.f;
#pragma unroll
                for (int j = 0; j < 8; j++) d = fmaf(q[h][j], kf[j], d);
                sc[h] = d;
            }
#pragma unroll
            for (int o = 16; o; o >>= 1)
#pragma unroll
                for (int h = 0; h < Q_PER_KV; h++) sc[h] += __shfl_xor_sync(~0u, sc[h], o);
#pragma unroll
            for (int h = 0; h < Q_PER_KV; h++) {
                const float mn = fmaxf(m[h], sc[h]);
                const float corr = __expf(m[h] - mn), p = __expf(sc[h] - mn);
                l[h] = l[h] * corr + p;
#pragma unroll
                for (int j = 0; j < 8; j++) acc[h][j] = fmaf(p, vf[j], acc[h][j] * corr);
                m[h] = mn;
            }
        }
        if (lane == 0)
#pragma unroll
            for (int h = 0; h < Q_PER_KV; h++) { sml[(warp * Q_PER_KV + h) * 2] = m[h]; sml[(warp * Q_PER_KV + h) * 2 + 1] = l[h]; }
        consumer_sync();
#pragma unroll
        for (int h = 0; h < Q_PER_KV; h++) {
            float M = -1e30f;
            for (int w = 0; w < N_CONSUMER_WARPS; w++) M = fmaxf(M, sml[(w * Q_PER_KV + h) * 2]);
            const float sc = __expf(m[h] - M);
#pragma unroll
            for (int j = 0; j < 8; j++) atomicAdd(&sacc[h * HD + lane * 8 + j], acc[h][j] * sc);
        }
        consumer_sync();
        float* part = P.attn_part + ((size_t)g * MAX_SPLITS + sp) * Q_PER_KV * PART_STRIDE;
        for (int i = t; i < Q_PER_KV * HD; i += N_CONSUMERS) __stcg(part + (i / HD) * PART_STRIDE + i % HD, sacc[i]);
        if (t < Q_PER_KV) {
            float M = -1e30f, Ls = 0.f;
            for (int w = 0; w < N_CONSUMER_WARPS; w++) M = fmaxf(M, sml[(w * Q_PER_KV + t) * 2]);
            for (int w = 0; w < N_CONSUMER_WARPS; w++)
                Ls += sml[(w * Q_PER_KV + t) * 2 + 1] * __expf(sml[(w * Q_PER_KV + t) * 2] - M);
            __stcg(part + t * PART_STRIDE + HD, M);
            __stcg(part + t * PART_STRIDE + HD + 1, Ls);
        }
        consumer_sync();
        if (t == 0) {
            __threadfence();
            *sflag = atomicAdd(&P.attn_cnt[g], 1u) == (unsigned)(splits - 1);
        }
        consumer_sync();
        if (*sflag) {  // last task for this KV head merges all partials
            __threadfence();
            const float* pg = P.attn_part + (size_t)g * MAX_SPLITS * Q_PER_KV * PART_STRIDE;
            for (int i = t; i < Q_PER_KV * HD; i += N_CONSUMERS) {
                const int h = i / HD, d = i % HD;
                float M = -1e30f;
                for (int p = 0; p < splits; p++) M = fmaxf(M, __ldcg(pg + (p * Q_PER_KV + h) * PART_STRIDE + HD));
                float num = 0.f, den = 0.f;
                for (int p = 0; p < splits; p++) {
                    const float* pp = pg + (p * Q_PER_KV + h) * PART_STRIDE;
                    const float wgt = __expf(__ldcg(pp + HD) - M);
                    num += wgt * __ldcg(pp + d);
                    den += wgt * __ldcg(pp + HD + 1);
                }
                __stcg(P.attn + (g * Q_PER_KV + h) * HD + d, num / den);
            }
            if (t == 0) P.attn_cnt[g] = 0;
        }
        consumer_sync();
    }
}

// ------------------------------------------------------------------ kernel
__global__ void __launch_bounds__(N_THREADS, 1) decode_kernel(Args A) {
    extern __shared__ __align__(1024) char smem[];
    char* ring = smem;
    float* vec = (float*)(smem + STAGES * CHUNK);
    float* red = (float*)(smem + STAGES * CHUNK + VEC_BYTES);
    uint64_t* full = (uint64_t*)(red + RED_FLOATS);
    uint64_t* empty = full + STAGES;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;

    if (t == 0) {
        for (int s = 0; s < STAGES; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], N_CONSUMER_WARPS); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp == N_CONSUMER_WARPS) {
        if (lane == 0) producer(A, ring, full, empty);
        return;
    }

    Ring R{ring, full, empty, 0, 0};
    unsigned long long target = A.bar_base;
    int op = 0;
    bool prof = false;
    unsigned long long* prow = nullptr;
    auto sync_grid = [&]() {
        consumer_sync();
        if (t == 0) {
            unsigned long long ta = prof ? globaltimer() : 0;
            __threadfence();
            atomicAdd(P.bar, 1ull);
            target += gridDim.x;
            while (ld_acquire(P.bar) < target) {}
            __threadfence();
            if (prof) {
                prow[op * PROF_FIELDS + 0] = ta;
                prow[op * PROF_FIELDS + 1] = globaltimer();
            }
        }
        if (prof && t == 0) prow[op * PROF_FIELDS + 2] = R.full_wait;
        R.full_wait = 0;
        op++;
        consumer_sync();
    };

    for (int s = 0; s < A.n_steps; s++) {
        const int pos = A.start_pos + s;
        prof = A.prof && s == A.prof_step;
        if (prof) {
            prow = A.prof + (size_t)blockIdx.x * PROF_MAX_OPS * PROF_FIELDS;
            if (t == 0) prow[(PROF_MAX_OPS - 1) * PROF_FIELDS + 3] = globaltimer();  // step start
        }
        op = 0;
        const int token = s < A.n_forced ? A.forced[s] : unpack_index(__ldcg(A.tokens_out + s - 1));
        const bf16* emb = P.embed + (size_t)token * HID;

        for (int l = 0; l < N_LAYERS; l++) {
            const LayerPtrs& L = P.layer[l];
            float* proj = P.proj;
            float* x = P.x;
            // token mixer input projection (RMSNorm fused)
            prologue_rmsnorm(vec, red, l == 0 ? nullptr : x, l == 0 ? emb : nullptr, L.in_norm);
            auto store = [&](int r, const float* a) { if (lane == 0) __stcg(proj + r, a[0]); };
            if (L.is_attn) gemv<HID, 1>(ATT_IN_ROWS, R, vec, prof, store);
            else gemv<HID, 1>(GDN_IN_ROWS, R, vec, prof, store);
            sync_grid();
            // token mixer core
            if (L.is_attn) attn_op(L, pos, vec, red);
            else gdn_op(L, pos, vec, red);
            sync_grid();
            // output projection + residual
            if (L.is_attn) prologue_attn_gate(vec);
            else prologue_gated_norm(vec, L);
            gemv<2048, 1>(HID, R, vec, prof, [&](int r, const float* a) {
                if (lane == 0) __stcg(x + r, (l == 0 ? bf2f(emb[r]) : __ldcg(x + r)) + a[0]);
            });
            sync_grid();
            // MLP gate/up with SwiGLU epilogue
            prologue_rmsnorm(vec, red, x, nullptr, L.post_norm);
            gemv<HID, 2>(2 * INTER, R, vec, prof, [&](int u, const float* a) {
                if (lane == 0) __stcg(P.act + u, silu(a[0]) * a[1]);
            });
            sync_grid();
            // MLP down + residual
            prologue_copy(vec, P.act, INTER);
            gemv<INTER, 1>(HID, R, vec, prof, [&](int r, const float* a) {
                if (lane == 0) __stcg(x + r, __ldcg(x + r) + a[0]);
            });
            sync_grid();
        }

        if (s >= A.n_forced - 1) {  // LM head + greedy / Gumbel-max sampling
            prologue_rmsnorm(vec, red, P.x, nullptr, P.final_norm);
            unsigned long long best = 0;
            float* lg = A.logits_out ? A.logits_out + (size_t)s * VOCAB : nullptr;
            gemv<HID, 1>(VOCAB, R, vec, prof, [&](int r, const float* a) {
                float score = a[0];
                if (lg && lane == 0) lg[r] = score;
                if (A.inv_temperature > 0.f) {
                    const float u = ((hash3(A.seed, (uint32_t)pos, (uint32_t)r) >> 8) + 0.5f) * (1.f / 16777216.f);
                    score = score * A.inv_temperature - __logf(-__logf(u));
                }
                const unsigned long long k = pack_score(score, r);
                best = k > best ? k : best;
            });
            if (lane == 0 && best) atomicMax(A.tokens_out + s, best);
            sync_grid();
        }
    }
}

// ------------------------------------------------------------------ host API
void set_params(const Params& p) {
    cudaMemcpyToSymbol(P, &p, sizeof(Params));
}

cudaError_t launch(const Args& a, int grid, cudaStream_t stream) {
    static bool attr = false;
    if (!attr) {
        cudaError_t e = cudaFuncSetAttribute(decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        if (e != cudaSuccess) return e;
        attr = true;
    }
    Args args = a;
    void* kargs[] = {&args};
    return cudaLaunchCooperativeKernel((void*)decode_kernel, dim3(grid), dim3(N_THREADS), kargs, SMEM_BYTES, stream);
}

}  // namespace mk
