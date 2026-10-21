// Qwen3.5-0.8B batch-1 decode megakernel for sm_120 (RTX 50-series).
//
// One persistent CTA per SM. Each CTA has a producer warp and 16 consumer warps.
// The producer walks this SM's slice of every weight matrix in execution order and
// streams it into a shared-memory ring with TMA bulk copies (plus L2 prefetches further
// ahead). It never waits on activations, so weights for the next op are already in flight
// while consumers wait for the previous op's outputs. Consumers run the ops: GEMVs with
// fused prologues (RMSNorm, gated norm, attention gate) and epilogues (residual add,
// SwiGLU, argmax), the Gated DeltaNet recurrence, and split-K attention. Activations move
// between CTAs as flag-in-data (LL) words instead of through grid barriers.
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
// Walks this CTA's weight stream: for every step, the slices of each layer's four GEMVs
// in execution order, then the LM head slice when that step's logits are needed.
template <bool Q8>
struct Cursor {
    int s, l, op;          // op 0..3 = in, out, gate/up, down; l == N_LAYERS is the LM head
    int u, u1, ub, upc;
    const char* W;

    __device__ void setup() {
        int rows, K, G = 1;
        if (l == N_LAYERS) { W = (const char*)P.lm_head; rows = VOCAB; K = HID; }
        else {
            const LayerPtrs& L = P.layer[l];
            switch (op) {
                case 0: W = (const char*)L.w_in; rows = L.is_attn ? ATT_IN_ROWS : GDN_IN_ROWS; K = HID; break;
                case 1: W = (const char*)L.w_out; rows = HID; K = 2048; break;
                case 2: W = (const char*)L.w_gu; rows = 2 * INTER; K = HID; G = 2; break;
                default: W = (const char*)L.w_down; rows = HID; K = INTER; break;
            }
        }
        block_slice(rows / G, u, u1);
        ub = G * row_bytes(K, Q8);
        upc = CHUNK / ub;
    }
    __device__ void next_op(const Args& A) {
        if (l < N_LAYERS && ++op == 4) { op = 0; l++; }
        else if (l == N_LAYERS) { l = 0; op = 0; s++; }
        if (l == N_LAYERS && s < A.lm_from) { l = 0; op = 0; s++; }
        if (s < A.n_steps) setup();
    }
    __device__ void skip_empty(const Args& A) { while (s < A.n_steps && u >= u1) next_op(A); }
    __device__ void init(const Args& A) { s = 0; l = 0; op = 0; setup(); skip_empty(A); }
    __device__ bool valid(const Args& A) const { return s < A.n_steps; }
    __device__ int bytes() const { return min(upc, u1 - u) * ub; }
    __device__ const char* ptr() const { return W + (size_t)u * ub; }
    __device__ void advance(const Args& A) { u += upc; skip_empty(A); }
};

__device__ __forceinline__ void prefetch_l2(const void* src, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

// Two cursors over the same stream: `pf` issues L2 prefetches up to A.pf_bytes ahead of
// `ld`, which fills the shared-memory ring. The ring is only ~70 KB, but L2 is 64 MB, so
// DRAM keeps streaming through grid barriers and the non-GEMV ops.
template <bool Q8>
__device__ void producer(const Args& A, char* ring, uint64_t* full, uint64_t* empty, volatile int* stop,
                         volatile int* issued) {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    Cursor<Q8> ld, pf;
    ld.init(A);
    pf.init(A);
    long long ahead = 0;
    auto run_prefetch = [&]() {
        while (pf.valid(A) && ahead < A.pf_bytes) {
            prefetch_l2(pf.ptr(), pf.bytes());
            ahead += pf.bytes();
            pf.advance(A);
        }
    };
    run_prefetch();
    unsigned long long stall = 0;
    int cur_step = 0;
    int c = 0;
    for (; ld.valid(A); c++) {
        if (ld.s != cur_step) {
            if (A.prof && cur_step == A.prof_step)
                A.prof[((size_t)gridDim.x * PROF_MAX_OPS + blockIdx.x) * PROF_FIELDS] = stall;
            stall = 0;
            cur_step = ld.s;
        }
        const int slot = c % STAGES;
        if (c >= STAGES) {
            const uint32_t par = ((c / STAGES) - 1) & 1;
            if (!mbar_try_wait(&empty[slot], par)) {
                unsigned long long t0 = globaltimer();
                while (!mbar_try_wait(&empty[slot], par)) {
                    if (*stop) {  // consumers hit a stop token: report what is in flight and quit
                        *issued = c;
                        return;
                    }
                }
                stall += globaltimer() - t0;
            }
        }
        const int n = ld.bytes();
        mbar_expect_tx(&full[slot], n);
        tma_load(ring + slot * CHUNK, ld.ptr(), n, &full[slot], policy);
        ahead -= n;
        ld.advance(A);
        run_prefetch();
    }
    *issued = c;
}

// ------------------------------------------------------------------ consumer GEMV
template <int K>
__device__ __forceinline__ float dot_row_bf16(const char* row, const float* vec, int lane) {
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

// Four (q + 128) bytes times four inputs. PRMT builds the float 2^23 + byte, and one FADD
// of (2^23 + 128) recovers q exactly, which is cheaper than an int->float convert.
__device__ __forceinline__ float dot4_u8(uint32_t w, float4 x) {
    constexpr float BIAS = 8388608.f + 128.f;
    float a = (__uint_as_float(__byte_perm(w, 0x4B000000u, 0x7440)) - BIAS) * x.x;
    a = fmaf(__uint_as_float(__byte_perm(w, 0x4B000000u, 0x7441)) - BIAS, x.y, a);
    a = fmaf(__uint_as_float(__byte_perm(w, 0x4B000000u, 0x7442)) - BIAS, x.z, a);
    return fmaf(__uint_as_float(__byte_perm(w, 0x4B000000u, 0x7443)) - BIAS, x.w, a);
}

template <int K>
__device__ __forceinline__ float dot_row_q8(const char* row, const float* vec, int lane) {
    const uint4* w = (const uint4*)row;
    const bf16* scale = (const bf16*)(row + K);
    constexpr int NV = K / 16;  // 16 weights per 16-byte vector, two vectors per scale group
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < (NV + 31) / 32; i++) {
        const int v = lane + 32 * i;
        if (NV % 32 == 0 || v < NV) {
            const uint4 q = w[v];
            const float4* x = (const float4*)(vec + v * 16);
            const float p = dot4_u8(q.x, x[0]) + dot4_u8(q.y, x[1]) + dot4_u8(q.z, x[2]) + dot4_u8(q.w, x[3]);
            a = fmaf(p, bf2f(scale[v >> 1]), a);
        }
    }
    return warp_sum(a);
}

template <int K, bool Q8>
__device__ __forceinline__ float dot_row(const char* row, const float* vec, int lane) {
    if constexpr (Q8) return dot_row_q8<K>(row, vec, lane);
    else return dot_row_bf16<K>(row, vec, lane);
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
template <int K, int G, bool Q8, class Epi>
__device__ __forceinline__ void gemv(int rows, Ring& R, const float* vec, bool timed, Epi epi) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int u0, u1;
    block_slice(rows / G, u0, u1);
    constexpr int RB = row_bytes(K, Q8);
    constexpr int UPC = CHUNK / (G * RB);
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
            for (int g = 0; g < G; g++) acc[g] = dot_row<K, Q8>(tile + (j * G + g) * RB, vec, lane);
            epi(u + j, acc);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&R.empty[slot]);
    }
}

// ------------------------------------------------------------------ LL (flag-in-data) exchange
// Activations cross CTAs as 64-bit words: the high 32 bits hold the epoch of the op that
// produced the value, the low 32 bits the fp32 value. A reader spins until the epoch it
// expects appears, so one store carries both the data and its "ready" signal: no fences,
// no grid barriers, and each consumer waits only for the elements it actually reads.
// Every op needs all of the previous op's output, so no CTA can get far enough ahead to
// overwrite a buffer another CTA is still reading; single-buffered arrays are enough.
__device__ __forceinline__ unsigned long long ld_ll(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_ll_raw(unsigned long long* p, uint32_t v, uint32_t e) {
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(((unsigned long long)e << 32) | v) : "memory");
}
__device__ __forceinline__ void st_ll(unsigned long long* p, float v, uint32_t e) { st_ll_raw(p, __float_as_uint(v), e); }
// Finish an LL read whose first load `v` was already issued (lets callers batch loads).
__device__ __forceinline__ uint32_t ll_wait_raw(unsigned long long v, const unsigned long long* p, uint32_t e) {
    while ((uint32_t)(v >> 32) != e) {
        __nanosleep(32);
        v = ld_ll(p);
    }
    return (uint32_t)v;
}
__device__ __forceinline__ float ll_wait(unsigned long long v, const unsigned long long* p, uint32_t e) {
    return __uint_as_float(ll_wait_raw(v, p, e));
}
__device__ __forceinline__ float ll_get(const unsigned long long* p, uint32_t e) { return ll_wait(ld_ll(p), p, e); }

// ------------------------------------------------------------------ GEMV prologues
// RMSNorm of the residual stream (or of an embedding row) into the smem input vector.
__device__ void prologue_rmsnorm(float* vec, float* red, uint32_t ex, const bf16* emb, const bf16* w) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    float v0, v1;
    if (emb) { v0 = bf2f(emb[t]); v1 = bf2f(emb[t + 512]); }
    else {
        const unsigned long long a = ld_ll(P.xl + t), b = ld_ll(P.xl + t + 512);
        v0 = ll_wait(a, P.xl + t, ex);
        v1 = ll_wait(b, P.xl + t + 512, ex);
    }
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
// The recurrence output comes from the previous op (epoch e-1), z from the one before (e-2).
__device__ void prologue_gated_norm(float* vec, const LayerPtrs& L, uint32_t e) {
    const int h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned long long ro[4], rz[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        ro[j] = ld_ll(P.corel + h * LIN_DV + lane + 32 * j);
        rz[j] = ld_ll(P.projl + GDN_Z_OFF + h * LIN_DV + lane + 32 * j);
    }
    float o[4], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        o[j] = ll_wait(ro[j], P.corel + h * LIN_DV + lane + 32 * j, e - 1);
        ss += o[j] * o[j];
    }
    const float r = rsqrtf(warp_sum(ss) / LIN_DV + EPS);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int d = lane + 32 * j;
        const float z = ll_wait(rz[j], P.projl + GDN_Z_OFF + h * LIN_DV + d, e - 2);
        vec[h * LIN_DV + d] = o[j] * r * L.gdn_norm[d] * silu(z);
    }
    consumer_sync();
}

// Attention output gated by sigmoid(gate); gate rows sit after each head's query rows.
__device__ void prologue_attn_gate(float* vec, uint32_t e) {
    constexpr int PER = NQ * HD / N_CONSUMERS;
    unsigned long long ra[PER], rg[PER];
#pragma unroll
    for (int k = 0; k < PER; k++) {
        const int i = threadIdx.x + k * N_CONSUMERS, h = i / HD, d = i % HD;
        ra[k] = ld_ll(P.attnl + i);
        rg[k] = ld_ll(P.projl + h * 2 * HD + HD + d);
    }
#pragma unroll
    for (int k = 0; k < PER; k++) {
        const int i = threadIdx.x + k * N_CONSUMERS, h = i / HD, d = i % HD;
        vec[i] = ll_wait(ra[k], P.attnl + i, e - 1) * sigmoid(ll_wait(rg[k], P.projl + h * 2 * HD + HD + d, e - 2));
    }
    consumer_sync();
}

__device__ void prologue_act(float* vec, uint32_t e) {
    constexpr int PER = INTER / N_CONSUMERS;  // 7
    unsigned long long r[PER];
#pragma unroll
    for (int k = 0; k < PER; k++) r[k] = ld_ll(P.actl + threadIdx.x + k * N_CONSUMERS);
#pragma unroll
    for (int k = 0; k < PER; k++) {
        const int i = threadIdx.x + k * N_CONSUMERS;
        vec[i] = ll_wait(r[k], P.actl + i, e - 1);
    }
    consumer_sync();
}

// Combine every CTA's LM-head candidate (published as two LL words) into the winning key.
__device__ unsigned long long reduce_lm_slots(uint32_t e, unsigned long long* sk) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    unsigned long long key = 0;
    if (t < gridDim.x) {
        const unsigned long long a = ld_ll(P.lml + 2 * t), b = ld_ll(P.lml + 2 * t + 1);
        key = ((unsigned long long)ll_wait_raw(a, P.lml + 2 * t, e) << 32) | ll_wait_raw(b, P.lml + 2 * t + 1, e);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        const unsigned long long other = __shfl_xor_sync(~0u, key, o);
        key = other > key ? other : key;
    }
    if (lane == 0) sk[warp] = key;
    consumer_sync();
    key = 0;
#pragma unroll
    for (int i = 0; i < N_CONSUMER_WARPS; i++) key = sk[i] > key ? sk[i] : key;
    consumer_sync();
    return key;
}

// ------------------------------------------------------------------ Gated DeltaNet recurrence
// Task = (head, slice of 32 value columns); 64 tasks, so each CTA runs at most one and the
// same CTA owns the same slice of the fp32 state every step. Each task recomputes the causal
// conv + L2 norm for its head's q/k (256 channels, cheap) so tasks stay independent. Warp w
// owns value columns w and w+16 of the slice; the state is stored [head][v][k] so each
// column is one coalesced 512 B row.
__device__ void gdn_op(const LayerPtrs& L, int pos, uint32_t e, float* sm, unsigned long long* ready_ts) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    float* sqk = sm;            // [256] conv'd q then k
    float* sv = sm + 256;       // [32]
    float* sc = sm + 288;       // beta, decay, rq, rk
    const int slot = pos & 3, s1 = (pos - 1) & 3, s2 = (pos - 2) & 3, s3 = (pos - 3) & 3;
    constexpr int SLICES = LIN_DV / GDN_SLICE;
    for (int task = blockIdx.x; task < LIN_H * SLICES; task += gridDim.x) {
        const int h = task / SLICES, sl = task % SLICES;
        if (t < 256 + GDN_SLICE) {  // causal conv (window of 4 incl. current) + SiLU
            const int ch = t < 128 ? h * LIN_DK + t
                         : t < 256 ? LIN_QK + h * LIN_DK + (t - 128)
                                   : 2 * LIN_QK + h * LIN_DV + sl * GDN_SLICE + (t - 256);
            float* hist = L.conv_hist + ch * CONV_K;
            const float h3 = __ldcg(hist + s3), h2 = __ldcg(hist + s2), h1 = __ldcg(hist + s1);
            const uint2 wq = *(const uint2*)(L.conv_w + ch * CONV_K);
            const float cur = ll_get(P.projl + ch, e - 1);
            const float acc = bf_lo(wq.x) * h3 + bf_hi(wq.x) * h2 + bf_lo(wq.y) * h1 + bf_hi(wq.y) * cur;
            if (t >= 256 || sl == 0) __stcg(hist + slot, cur);
            if (t < 256) sqk[t] = silu(acc); else sv[t - 256] = silu(acc);
        } else if (t == 256 + GDN_SLICE) {
            const unsigned long long rb = ld_ll(P.projl + GDN_B_OFF + h), ra = ld_ll(P.projl + GDN_A_OFF + h);
            const float beta = sigmoid(ll_wait(rb, P.projl + GDN_B_OFF + h, e - 1));
            const float a = ll_wait(ra, P.projl + GDN_A_OFF + h, e - 1);
            sc[0] = beta;
            sc[1] = __expf(-__expf(L.A_log[h]) * softplus(a + bf2f(L.dt_bias[h])));
        }
        consumer_sync();
        if (ready_ts && t == 0) *ready_ts = globaltimer();
        if (warp < 2) {  // L2 norms of q (warp 0) and k (warp 1)
            float ss = 0.f;
#pragma unroll
            for (int j = 0; j < 4; j++) { const float v = sqk[warp * 128 + lane + 32 * j]; ss += v * v; }
            ss = warp_sum(ss);
            if (lane == 0) sc[2 + warp] = rsqrtf(ss + 1e-6f) * (warp == 0 ? rsqrtf((float)LIN_DK) : 1.f);
        }
        consumer_sync();
        const float beta = sc[0], decay = sc[1], rq = sc[2], rk = sc[3];
        float kv[4], qv[4], s[2][4], km[2] = {0.f, 0.f};
        float* S[2];
#pragma unroll
        for (int c = 0; c < 2; c++) S[c] = L.state + ((size_t)h * LIN_DV + sl * GDN_SLICE + warp + 16 * c) * LIN_DK;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int k = lane + 32 * j;
            kv[j] = sqk[128 + k] * rk;
            qv[j] = sqk[k] * rq;
            s[0][j] = __ldcg(S[0] + k);
            s[1][j] = __ldcg(S[1] + k);
        }
#pragma unroll
        for (int c = 0; c < 2; c++) {
#pragma unroll
            for (int j = 0; j < 4; j++) { s[c][j] *= decay; km[c] += s[c][j] * kv[j]; }
            km[c] = warp_sum(km[c]);
        }
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const float delta = (sv[warp + 16 * c] - km[c]) * beta;
            float o = 0.f;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                s[c][j] = fmaf(kv[j], delta, s[c][j]);
                __stcg(S[c] + lane + 32 * j, s[c][j]);
                o += s[c][j] * qv[j];
            }
            o = warp_sum(o);
            if (lane == 0) st_ll(P.corel + h * LIN_DV + sl * GDN_SLICE + warp + 16 * c, o, e);
        }
        consumer_sync();
    }
}

// ------------------------------------------------------------------ gated full attention
// Split-K decode attention. Task = (kv head, chunk of positions); it handles the 4 query
// heads sharing that KV head. Partials are merged by whichever task finishes last.
__device__ void attn_op(const LayerPtrs& L, int pos, uint32_t e, float* sm, float* red, unsigned long long* ready_ts) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int n = pos + 1;
    int splits = min(MAX_SPLITS, (n + 31) / 32);
    const int chunk = (n + splits - 1) / splits;
    splits = (n + chunk - 1) / chunk;
    float* sq = sm;              // [4][256]
    float* sacc = sm + 1024;     // [4][256]; reused by the merge for per-split weights
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
                const unsigned long long* src = isq ? P.projl + (g * Q_PER_KV + warp) * 2 * HD : P.projl + ATT_K_OFF + g * HD;
                const bf16* nw = isq ? L.q_norm : L.k_norm;
                unsigned long long raw[8];
#pragma unroll
                for (int j = 0; j < 8; j++) raw[j] = ld_ll(src + lane * 8 + j);
                float v[8], ss = 0.f;
#pragma unroll
                for (int j = 0; j < 8; j++) { v[j] = ll_wait(raw[j], src + lane * 8 + j, e - 1); ss += v[j] * v[j]; }
                const float r = rsqrtf(warp_sum(ss) / HD + EPS);
                float* dst = isq ? sq + warp * HD : skv;
#pragma unroll
                for (int j = 0; j < 8; j++) dst[lane * 8 + j] = v[j] * r * (1.f + bf2f(nw[lane * 8 + j]));
            }
        } else if (warp == Q_PER_KV + 1 && has_new) {
            unsigned long long raw[8];
#pragma unroll
            for (int j = 0; j < 8; j++) raw[j] = ld_ll(P.projl + ATT_V_OFF + g * HD + lane * 8 + j);
#pragma unroll
            for (int j = 0; j < 8; j++) skv[HD + lane * 8 + j] = ll_wait(raw[j], P.projl + ATT_V_OFF + g * HD + lane * 8 + j, e - 1);
        }
        consumer_sync();
        if (ready_ts && t == 0) *ready_ts = globaltimer();
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
        float sc[Q_PER_KV];
#pragma unroll
        for (int h = 0; h < Q_PER_KV; h++) {
            float M = -1e30f;
            for (int w = 0; w < N_CONSUMER_WARPS; w++) M = fmaxf(M, sml[(w * Q_PER_KV + h) * 2]);
            sc[h] = __expf(m[h] - M);
        }
        float* part = P.attn_part + ((size_t)g * MAX_SPLITS + sp) * Q_PER_KV * PART_STRIDE;
        if (t < Q_PER_KV) {
            float M = -1e30f, Ls = 0.f;
            for (int w = 0; w < N_CONSUMER_WARPS; w++) M = fmaxf(M, sml[(w * Q_PER_KV + t) * 2]);
            for (int w = 0; w < N_CONSUMER_WARPS; w++)
                Ls += sml[(w * Q_PER_KV + t) * 2 + 1] * __expf(sml[(w * Q_PER_KV + t) * 2] - M);
            __stcg(part + t * PART_STRIDE + HD, M);
            __stcg(part + t * PART_STRIDE + HD + 1, Ls);
        }
        consumer_sync();
        // Sum the warps' outputs in a fixed order (atomics would make results run-to-run
        // nondeterministic). q / k / v scratch is dead now, so all of `sm` is the buffer.
        float* wbuf = sm;  // [16 warps][256]
#pragma unroll
        for (int h = 0; h < Q_PER_KV; h++) {
#pragma unroll
            for (int j = 0; j < 8; j++) wbuf[warp * HD + lane * 8 + j] = acc[h][j] * sc[h];
            consumer_sync();
            if (t < HD) {
                float a = 0.f;
                for (int w = 0; w < N_CONSUMER_WARPS; w++) a += wbuf[w * HD + t];
                __stcg(part + h * PART_STRIDE + t, a);
            }
            consumer_sync();
        }
        if (t == 0) {
            __threadfence();
            *sflag = atomicAdd(&P.attn_cnt[g], 1u) == (unsigned)(splits - 1);
        }
        consumer_sync();
        if (*sflag) {  // last task for this KV head merges all partials
            __threadfence();
            const float* pg = P.attn_part + (size_t)g * MAX_SPLITS * Q_PER_KV * PART_STRIDE;
            float* sw = sacc;  // [splits][4] merge weights
            for (int i = t; i < splits * Q_PER_KV; i += N_CONSUMERS) {
                sw[i] = __ldcg(pg + i * PART_STRIDE + HD);
                sw[MAX_SPLITS * Q_PER_KV + i] = __ldcg(pg + i * PART_STRIDE + HD + 1);
            }
            consumer_sync();
            if (t < Q_PER_KV) {
                float M = -1e30f, den = 0.f;
                for (int p = 0; p < splits; p++) M = fmaxf(M, sw[p * Q_PER_KV + t]);
                for (int p = 0; p < splits; p++) den += __expf(sw[p * Q_PER_KV + t] - M) * sw[MAX_SPLITS * Q_PER_KV + p * Q_PER_KV + t];
                for (int p = 0; p < splits; p++) sw[p * Q_PER_KV + t] = __expf(sw[p * Q_PER_KV + t] - M) / den;
            }
            consumer_sync();
            for (int i = t; i < Q_PER_KV * HD; i += N_CONSUMERS) {
                const int h = i / HD, d = i % HD;
                float num = 0.f;
#pragma unroll 8
                for (int p = 0; p < splits; p++) num += sw[p * Q_PER_KV + h] * __ldcg(pg + (p * Q_PER_KV + h) * PART_STRIDE + d);
                st_ll(P.attnl + (g * Q_PER_KV + h) * HD + d, num, e);
            }
            if (t == 0) atomicExch(&P.attn_cnt[g], 0u);
        }
        consumer_sync();
    }
}

// ------------------------------------------------------------------ kernel
template <bool Q8>
__global__ void __launch_bounds__(N_THREADS, 1) decode_kernel(Args A) {
    extern __shared__ __align__(1024) char smem[];
    char* ring = smem;
    float* vec = (float*)(smem + STAGES * CHUNK);
    float* red = (float*)(smem + STAGES * CHUNK + VEC_BYTES);
    uint64_t* full = (uint64_t*)(red + RED_FLOATS);
    uint64_t* empty = full + STAGES;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;

    volatile int* stop = (volatile int*)(red + 36);    // consumers -> producer
    volatile int* issued = (volatile int*)(red + 37);  // producer -> consumers: chunks issued
    if (t == 0) {
        for (int s = 0; s < STAGES; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], N_CONSUMER_WARPS); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        *stop = 0;
        *issued = -1;
    }
    __syncthreads();

    if (warp == N_CONSUMER_WARPS) {
        if (lane == 0) producer<Q8>(A, ring, full, empty, stop, issued);
        return;
    }

    // This CTA owns rows [xr0, xr1) of the residual stream (the out/down GEMV rows); it keeps
    // the fp32 master copy in smem and publishes LL copies for everyone else.
    int xr0, xr1;
    block_slice(HID, xr0, xr1);
    float* xs = red + 48;
    unsigned long long* sk = (unsigned long long*)(vec + 2048);

    Ring R{ring, full, empty, 0, 0};
    uint32_t e = A.epoch_base;  // epoch stamped on the outputs of the op about to run
    int op = 0;
    bool prof = false;
    unsigned long long* prow = nullptr;
    auto op_ready = [&]() {
        if (prof && t == 0) prow[op * PROF_FIELDS + 0] = globaltimer();
    };
    auto op_end = [&]() {
        consumer_sync();
        if (prof && t == 0) {
            prow[op * PROF_FIELDS + 1] = globaltimer();
            prow[op * PROF_FIELDS + 2] = R.full_wait;
        }
        R.full_wait = 0;
        e++;
        op++;
    };

    int last_lm = -1;
    int steps_done = A.n_steps;
    for (int s = 0; s < A.n_steps; s++) {
        const int pos = A.start_pos + s;
        prof = A.prof && s == A.prof_step;
        if (prof) {
            prow = A.prof + (size_t)blockIdx.x * PROF_MAX_OPS * PROF_FIELDS;
            if (t == 0) prow[(PROF_MAX_OPS - 1) * PROF_FIELDS + 3] = globaltimer();  // step start
        }
        op = 0;
        if (A.step_ts && blockIdx.x == 0 && t == 0) A.step_ts[s] = globaltimer();
        int token;
        if (s < A.n_forced) token = A.forced[s];
        else {
            const unsigned long long key = reduce_lm_slots(e - 1, sk);
            if (blockIdx.x == 0 && t == 0) A.tokens_out[s - 1] = key;
            token = unpack_index(key);
            bool is_stop = false;
            for (int i = 0; i < A.n_stop; i++) is_stop |= (token == A.stop_ids[i]);
            if (is_stop) {  // every CTA sees the same token, so all stop at this step
                if (t == 0) {
                    *stop = 1;
                    while (*issued < 0) {}
                    for (int c = R.c; c < *issued; c++) mbar_wait(&full[c % STAGES], (c / STAGES) & 1);
                }
                consumer_sync();
                steps_done = s;
                break;
            }
        }
        const bf16* emb = P.embed + (size_t)token * HID;

        for (int l = 0; l < N_LAYERS; l++) {
            const LayerPtrs& L = P.layer[l];
            // token mixer input projection (RMSNorm fused)
            prologue_rmsnorm(vec, red, e - 1, l == 0 ? emb : nullptr, L.in_norm);
            op_ready();
            auto store = [&](int r, const float* a) { if (lane == 0) st_ll(P.projl + r, a[0], e); };
            if (L.is_attn) gemv<HID, 1, Q8>(ATT_IN_ROWS, R, vec, prof, store);
            else gemv<HID, 1, Q8>(GDN_IN_ROWS, R, vec, prof, store);
            op_end();
            // token mixer core
            unsigned long long* rts = (prof && t == 0) ? prow + op * PROF_FIELDS : nullptr;
            if (rts) *rts = globaltimer();  // overwritten once a task's inputs are ready
            if (L.is_attn) attn_op(L, pos, e, vec, red, rts);
            else gdn_op(L, pos, e, vec, rts);
            op_end();
            // output projection + residual
            if (L.is_attn) prologue_attn_gate(vec, e);
            else prologue_gated_norm(vec, L, e);
            op_ready();
            gemv<2048, 1, Q8>(HID, R, vec, prof, [&](int r, const float* a) {
                if (lane == 0) {
                    const float xn = (l == 0 ? bf2f(emb[r]) : xs[r - xr0]) + a[0];
                    xs[r - xr0] = xn;
                    st_ll(P.xl + r, xn, e);
                }
            });
            op_end();
            // MLP gate/up with SwiGLU epilogue
            prologue_rmsnorm(vec, red, e - 1, nullptr, L.post_norm);
            op_ready();
            gemv<HID, 2, Q8>(2 * INTER, R, vec, prof, [&](int u, const float* a) {
                if (lane == 0) st_ll(P.actl + u, silu(a[0]) * a[1], e);
            });
            op_end();
            // MLP down + residual
            prologue_act(vec, e);
            op_ready();
            gemv<INTER, 1, Q8>(HID, R, vec, prof, [&](int r, const float* a) {
                if (lane == 0) {
                    const float xn = xs[r - xr0] + a[0];
                    xs[r - xr0] = xn;
                    st_ll(P.xl + r, xn, e);
                }
            });
            op_end();
        }
        // Publish this step's side data (conv history, KV) before anything of the next step.
        __threadfence();

        if (s >= A.lm_from) {  // LM head + greedy / Gumbel-max sampling
            prologue_rmsnorm(vec, red, e - 1, nullptr, P.final_norm);
            op_ready();
            unsigned long long best = 0;
            float* lg = A.logits_out ? A.logits_out + (size_t)s * VOCAB : nullptr;
            gemv<HID, 1, Q8>(VOCAB, R, vec, prof, [&](int r, const float* a) {
                float score = a[0];
                if (lg && lane == 0) lg[r] = score;
                if (A.inv_temperature > 0.f) {
                    const float u = ((hash3(A.seed, (uint32_t)pos, (uint32_t)r) >> 8) + 0.5f) * (1.f / 16777216.f);
                    score = score * A.inv_temperature - __logf(-__logf(u));
                }
                const unsigned long long k = pack_score(score, r);
                best = k > best ? k : best;
            });
            if (lane == 0) sk[warp] = best;
            consumer_sync();
            if (t == 0) {
                for (int i = 0; i < N_CONSUMER_WARPS; i++) best = sk[i] > best ? sk[i] : best;
                st_ll_raw(P.lml + 2 * blockIdx.x, (uint32_t)(best >> 32), e);
                st_ll_raw(P.lml + 2 * blockIdx.x + 1, (uint32_t)best, e);
            }
            op_end();
            last_lm = s;
        }
    }
    if (blockIdx.x == 0 && steps_done == A.n_steps && last_lm == A.n_steps - 1) {
        const unsigned long long key = reduce_lm_slots(e - 1, sk);
        if (t == 0) A.tokens_out[A.n_steps - 1] = key;
    }
    if (blockIdx.x == 0 && t == 0) {
        if (A.step_ts) A.step_ts[steps_done] = globaltimer();
        A.out_info[0] = steps_done;
        A.out_info[1] = (int)e;
    }
}

// ------------------------------------------------------------------ host API
void set_params(const Params& p) {
    cudaMemcpyToSymbol(P, &p, sizeof(Params));
}

cudaError_t launch(const Args& a, int grid, cudaStream_t stream) {
    static bool attr = false;
    if (!attr) {
        for (const void* f : {(const void*)decode_kernel<false>, (const void*)decode_kernel<true>}) {
            cudaError_t e = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
            if (e != cudaSuccess) return e;
        }
        attr = true;
    }
    Args args = a;
    void* kargs[] = {&args};
    const void* f = a.q8 ? (const void*)decode_kernel<true> : (const void*)decode_kernel<false>;
    return cudaLaunchCooperativeKernel(f, dim3(grid), dim3(N_THREADS), kargs, SMEM_BYTES, stream);
}

}  // namespace mk
