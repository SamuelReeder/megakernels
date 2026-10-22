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
//
// Speculative decoding (the T = 2 instantiation) runs the same ops on two tokens per pass
// over the weights: the last accepted token and the MTP head's guess for the one after it.
// Per token the arithmetic is exactly that of a one-token step, so speculative output is
// bitwise identical to plain decoding. The DeltaNet state after the first token is kept on
// the side, so a rejected guess costs nothing to undo. The MTP head then runs in the same
// kernel and drafts the next cycle's guess.
#include "megakernel.h"

namespace mk {

__constant__ Params P;

// Debug builds (-DMK_WATCHDOG) give up on any wait that lasts 100 ms, so a dataflow bug ends the
// kernel with garbage results instead of hanging the GPU until the OS watchdog resets it.
__device__ int mk_watchdog;
#ifdef MK_WATCHDOG
__device__ __forceinline__ unsigned long long globaltimer_();
__device__ __noinline__ bool spin_expired(unsigned long long& t0) {
    if (*(volatile int*)&mk_watchdog) return true;
    const unsigned long long now = globaltimer_();
    if (!t0) t0 = now;
    else if (now - t0 > 100000000ull) { atomicExch(&mk_watchdog, 1); return true; }
    return false;
}
#define SPIN_GUARD(t0) if (spin_expired(t0)) break
#else
#define SPIN_GUARD(t0)
#endif

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
    unsigned long long t0 = 0;
    while (!mbar_try_wait(b, parity)) { SPIN_GUARD(t0); }
    (void)t0;
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
__device__ __forceinline__ unsigned long long globaltimer_() { return globaltimer(); }
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
// The four GEMVs of a decoder layer, in execution order.
__device__ __forceinline__ void layer_gemv(const LayerPtrs& L, int op, const char*& W, int& rows, int& K, int& G) {
    switch (op) {
        case 0: W = (const char*)L.w_in; rows = L.is_attn ? ATT_IN_ROWS : GDN_IN_ROWS; K = HID; break;
        case 1: W = (const char*)L.w_out; rows = HID; K = 2048; break;
        case 2: W = (const char*)L.w_gu; rows = 2 * INTER; K = HID; G = 2; break;
        default: W = (const char*)L.w_down; rows = HID; K = INTER; break;
    }
}

// Walks this CTA's weight stream: for every step, the slices of each layer's four GEMVs
// in execution order, then the LM head slice when that step's logits are needed. A
// speculative cycle continues with the MTP head: fc, its layer's four GEMVs, the draft head.
template <bool Q8>
struct Cursor {
    int s, l, op;          // l < N_LAYERS: layer GEMV op 0..3; l == N_LAYERS: LM head; l == N_LAYERS + 1: MTP op 0..5
    int u, u1, ub, upc;
    const char* W;

    __device__ void setup() {
        int rows, K, G = 1;
        if (l < N_LAYERS) layer_gemv(P.layer[l], op, W, rows, K, G);
        else if (l == N_LAYERS) { W = (const char*)P.lm_head; rows = VOCAB; K = HID; }
        else if (op == 0) { W = (const char*)P.mtp_fc; rows = HID; K = 2048; }
        else if (op < 5) layer_gemv(P.mtp, op - 1, W, rows, K, G);
        else { W = (const char*)P.draft_head; rows = (int)P.draft_vocab; K = HID; }
        block_slice(rows / G, u, u1);
        ub = G * row_bytes(K, Q8);
        upc = CHUNK / ub;
    }
    __device__ void next_op(const Args& A) {
        if (l < N_LAYERS) { if (++op == 4) { op = 0; l++; } }
        else if (l == N_LAYERS && A.spec) { l++; op = 0; }
        else if (l == N_LAYERS || ++op == 6) { l = 0; op = 0; s++; }
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
template <bool Q8, int NS>
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
        const int slot = c % NS;
        if (c >= NS) {
            const uint32_t par = ((c / NS) - 1) & 1;
            if (!mbar_try_wait(&empty[slot], par)) {
                unsigned long long t0 = globaltimer(), wd = 0;
                while (!mbar_try_wait(&empty[slot], par)) {
                    if (*stop) {  // consumers hit a stop token: report what is in flight and quit
                        *issued = c;
                        return;
                    }
                    SPIN_GUARD(wd);
                }
                (void)wd;
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
// Each row is dotted with T input vectors (token j's at vec + j * K), so a speculative cycle
// reads every weight once for both of its tokens. Per token the operations and their order
// are the same for every T.
//
// The input vectors are stored swizzled: float i lives at swz(i), its 16-byte unit XORed within
// its 64-byte group. A lane reads its 16 (int8) or 8 (bf16) consecutive inputs as float4s, 64 or
// 32 bytes from its neighbour's, which puts 4 (2) lanes of every quarter-warp on the same banks;
// swizzled, each quarter-warp's loads hit 8 distinct bank groups. swz(a + b) = a + swz(b) for a
// multiple of 128, so a lane's offsets are constants and every j * K (K % 128 == 0) passes through.
__device__ __forceinline__ int swz(int i) { return i ^ ((i >> 3) & 12); }

template <int K, int T>
__device__ __forceinline__ void dot_row_bf16(const char* row, const float* vec, int lane, float* out) {
    static_assert(K % 128 == 0, "swizzled input vectors");
    const uint4* w = (const uint4*)row;
    constexpr int NV = K / 8;
    const int o0 = swz(8 * lane), o1 = swz(8 * lane + 4);  // lane's inputs of vector v = lane + 32 i
    float a[T];
#pragma unroll
    for (int j = 0; j < T; j++) a[j] = 0.f;
#pragma unroll
    for (int i = 0; i < (NV + 31) / 32; i++) {
        const int v = lane + 32 * i;
        if (NV % 32 == 0 || v < NV) {
            const uint4 q = w[v];
#pragma unroll
            for (int j = 0; j < T; j++) {
                const float* x = vec + j * K + 256 * i;
                const float4 x0 = *(const float4*)(x + o0);
                const float4 x1 = *(const float4*)(x + o1);
                a[j] = fmaf(bf_lo(q.x), x0.x, a[j]); a[j] = fmaf(bf_hi(q.x), x0.y, a[j]);
                a[j] = fmaf(bf_lo(q.y), x0.z, a[j]); a[j] = fmaf(bf_hi(q.y), x0.w, a[j]);
                a[j] = fmaf(bf_lo(q.z), x1.x, a[j]); a[j] = fmaf(bf_hi(q.z), x1.y, a[j]);
                a[j] = fmaf(bf_lo(q.w), x1.z, a[j]); a[j] = fmaf(bf_hi(q.w), x1.w, a[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < T; j++) out[j] = warp_sum(a[j]);
}

// Four (q + 128) bytes -> four exact floats q. PRMT builds the float 2^23 + byte, and one FADD
// of (2^23 + 128) recovers q exactly, which is cheaper than an int->float convert.
__device__ __forceinline__ void deq4(uint32_t w, float* f) {
    constexpr float BIAS = 8388608.f + 128.f;
    f[0] = __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7440)) - BIAS;
    f[1] = __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7441)) - BIAS;
    f[2] = __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7442)) - BIAS;
    f[3] = __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7443)) - BIAS;
}
__device__ __forceinline__ float dot4(const float* f, float4 x) {
    float a = f[0] * x.x;
    a = fmaf(f[1], x.y, a);
    a = fmaf(f[2], x.z, a);
    return fmaf(f[3], x.w, a);
}

template <int K, int T>
__device__ __forceinline__ void dot_row_q8(const char* row, const float* vec, int lane, float* out) {
    static_assert(K % 128 == 0, "swizzled input vectors");
    const uint4* w = (const uint4*)row;
    const bf16* scale = (const bf16*)(row + K);
    constexpr int NV = K / 16;  // 16 weights per 16-byte vector, two vectors per scale group
    int o[4];                   // lane's inputs of vector v = lane + 32 i, a float4 at a time
#pragma unroll
    for (int c = 0; c < 4; c++) o[c] = swz(16 * lane + 4 * c);
    float a[T];
#pragma unroll
    for (int j = 0; j < T; j++) a[j] = 0.f;
#pragma unroll
    for (int i = 0; i < (NV + 31) / 32; i++) {
        const int v = lane + 32 * i;
        if (NV % 32 == 0 || v < NV) {
            const uint4 q = w[v];
            float f[16];
            deq4(q.x, f); deq4(q.y, f + 4); deq4(q.z, f + 8); deq4(q.w, f + 12);
            const float sc = bf2f(scale[v >> 1]);
#pragma unroll
            for (int j = 0; j < T; j++) {
                const float* x = vec + j * K + 512 * i;
                const float p = dot4(f, *(const float4*)(x + o[0])) + dot4(f + 4, *(const float4*)(x + o[1])) +
                                dot4(f + 8, *(const float4*)(x + o[2])) + dot4(f + 12, *(const float4*)(x + o[3]));
                a[j] = fmaf(p, sc, a[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < T; j++) out[j] = warp_sum(a[j]);
}

template <int K, bool Q8, int T>
__device__ __forceinline__ void dot_row(const char* row, const float* vec, int lane, float* out) {
    if constexpr (Q8) dot_row_q8<K, T>(row, vec, lane, out);
    else dot_row_bf16<K, T>(row, vec, lane, out);
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
// epi(unit, acc[T][G]) runs on every lane of the owning warp.
template <int K, int G, bool Q8, int T, int NS, class Epi>
__device__ __forceinline__ void gemv(int rows, Ring& R, const float* vec, bool timed, Epi epi) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int u0, u1;
    block_slice(rows / G, u0, u1);
    constexpr int RB = row_bytes(K, Q8);
    constexpr int UPC = CHUNK / (G * RB);
    for (int u = u0; u < u1; u += UPC, R.c++) {
        const int slot = R.c % NS;
        const uint32_t par = (R.c / NS) & 1;
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
            float acc[T][G];
#pragma unroll
            for (int g = 0; g < G; g++) {
                float o[T];
                dot_row<K, Q8, T>(tile + (j * G + g) * RB, vec, lane, o);
#pragma unroll
                for (int k = 0; k < T; k++) acc[k][g] = o[k];
            }
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
    unsigned long long t0 = 0;
    while ((uint32_t)(v >> 32) != e) {
        __nanosleep(32);
        v = ld_ll(p);
        SPIN_GUARD(t0);
    }
    (void)t0;
    return (uint32_t)v;
}
__device__ __forceinline__ float ll_wait(unsigned long long v, const unsigned long long* p, uint32_t e) {
    return __uint_as_float(ll_wait_raw(v, p, e));
}
__device__ __forceinline__ float ll_get(const unsigned long long* p, uint32_t e) { return ll_wait(ld_ll(p), p, e); }

// ------------------------------------------------------------------ GEMV prologues
// Every prologue writes token j's input vector at vec + j * K, swizzled (see swz).

// 1 / RMS of T 1024-vectors held as two values per thread.
template <int T>
__device__ __forceinline__ void rms_scale(float* red, const float* v0, const float* v1, float* r) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int j = 0; j < T; j++) {
        const float ss = warp_sum(fmaf(v1[j], v1[j], v0[j] * v0[j]));
        if (lane == 0) red[warp * T + j] = ss;
    }
    consumer_sync();
#pragma unroll
    for (int j = 0; j < T; j++) {
        float tot = 0.f;
#pragma unroll
        for (int i = 0; i < N_CONSUMER_WARPS; i++) tot += red[i * T + j];
        r[j] = rsqrtf(tot / HID + EPS);
    }
}

// RMSNorm of a residual stream (LL words, token j at src + j * HID) or of embedding rows.
template <int T>
__device__ void prologue_rmsnorm(float* vec, float* red, uint32_t ex, const bf16* const* emb, const bf16* w,
                                 const unsigned long long* src) {
    const int t = threadIdx.x;
    float v0[T], v1[T], r[T];
    if (emb) {
#pragma unroll
        for (int j = 0; j < T; j++) { v0[j] = bf2f(emb[j][t]); v1[j] = bf2f(emb[j][t + 512]); }
    } else {
        unsigned long long a[T], b[T];
#pragma unroll
        for (int j = 0; j < T; j++) { a[j] = ld_ll(src + j * HID + t); b[j] = ld_ll(src + j * HID + t + 512); }
#pragma unroll
        for (int j = 0; j < T; j++) {
            v0[j] = ll_wait(a[j], src + j * HID + t, ex);
            v1[j] = ll_wait(b[j], src + j * HID + t + 512, ex);
        }
    }
    rms_scale<T>(red, v0, v1, r);
#pragma unroll
    for (int j = 0; j < T; j++) {
        vec[swz(j * HID + t)] = v0[j] * r[j] * (1.f + bf2f(w[t]));
        vec[swz(j * HID + t + 512)] = v1[j] * r[j] * (1.f + bf2f(w[t + 512]));
    }
    consumer_sync();
}

// GDN output: per-head RMSNorm of the recurrence output, gated by silu(z). One warp per head.
// The recurrence output comes from the previous op (epoch e-1), z from the one before (e-2).
template <int T>
__device__ void prologue_gated_norm(float* vec, const LayerPtrs& L, uint32_t e) {
    const int h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned long long ro[T][4], rz[T][4];
#pragma unroll
    for (int j = 0; j < T; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) {
            ro[j][k] = ld_ll(P.corel + j * LIN_V + h * LIN_DV + lane + 32 * k);
            rz[j][k] = ld_ll(P.projl + j * PROJ_MAX + GDN_Z_OFF + h * LIN_DV + lane + 32 * k);
        }
#pragma unroll
    for (int j = 0; j < T; j++) {
        float o[4], ss = 0.f;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            o[k] = ll_wait(ro[j][k], P.corel + j * LIN_V + h * LIN_DV + lane + 32 * k, e - 1);
            ss = fmaf(o[k], o[k], ss);
        }
        const float r = rsqrtf(warp_sum(ss) / LIN_DV + EPS);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int d = lane + 32 * k;
            const float z = ll_wait(rz[j][k], P.projl + j * PROJ_MAX + GDN_Z_OFF + h * LIN_DV + d, e - 2);
            vec[swz(j * LIN_V + h * LIN_DV + d)] = o[k] * r * L.gdn_norm[d] * silu(z);
        }
    }
    consumer_sync();
}

// Attention output gated by sigmoid(gate); gate rows sit after each head's query rows.
template <int T>
__device__ void prologue_attn_gate(float* vec, uint32_t e) {
    constexpr int PER = NQ * HD / N_CONSUMERS;
    unsigned long long ra[T][PER], rg[T][PER];
#pragma unroll
    for (int j = 0; j < T; j++)
#pragma unroll
        for (int k = 0; k < PER; k++) {
            const int i = threadIdx.x + k * N_CONSUMERS, h = i / HD, d = i % HD;
            ra[j][k] = ld_ll(P.attnl + j * NQ * HD + i);
            rg[j][k] = ld_ll(P.projl + j * PROJ_MAX + h * 2 * HD + HD + d);
        }
#pragma unroll
    for (int j = 0; j < T; j++)
#pragma unroll
        for (int k = 0; k < PER; k++) {
            const int i = threadIdx.x + k * N_CONSUMERS, h = i / HD, d = i % HD;
            vec[swz(j * NQ * HD + i)] = ll_wait(ra[j][k], P.attnl + j * NQ * HD + i, e - 1) *
                                   sigmoid(ll_wait(rg[j][k], P.projl + j * PROJ_MAX + h * 2 * HD + HD + d, e - 2));
        }
    consumer_sync();
}

template <int T>
__device__ void prologue_act(float* vec, uint32_t e) {
    constexpr int PER = INTER / N_CONSUMERS;  // 7
    unsigned long long r[T][PER];
#pragma unroll
    for (int j = 0; j < T; j++)
#pragma unroll
        for (int k = 0; k < PER; k++) r[j][k] = ld_ll(P.actl + j * INTER + threadIdx.x + k * N_CONSUMERS);
#pragma unroll
    for (int j = 0; j < T; j++)
#pragma unroll
        for (int k = 0; k < PER; k++) {
            const int i = threadIdx.x + k * N_CONSUMERS;
            vec[swz(j * INTER + i)] = ll_wait(r[j][k], P.actl + j * INTER + i, e - 1);
        }
    consumer_sync();
}

// MTP fc input for both tokens of a cycle: [rmsnorm(embed(next_j)) | rmsnorm(h_j)], where h_j is
// the main model's final hidden state (the residual stream after the final norm the LM head saw).
__device__ void prologue_mtp_in(float* vec, float* red, uint32_t ex, const int* next) {
    const int t = threadIdx.x;
    float v0[SPEC_T], v1[SPEC_T], r[SPEC_T];
    unsigned long long a[SPEC_T], b[SPEC_T];
#pragma unroll
    for (int j = 0; j < SPEC_T; j++) { a[j] = ld_ll(P.xl + j * HID + t); b[j] = ld_ll(P.xl + j * HID + t + 512); }
#pragma unroll
    for (int j = 0; j < SPEC_T; j++) {
        v0[j] = ll_wait(a[j], P.xl + j * HID + t, ex);
        v1[j] = ll_wait(b[j], P.xl + j * HID + t + 512, ex);
    }
    rms_scale<SPEC_T>(red, v0, v1, r);
#pragma unroll
    for (int j = 0; j < SPEC_T; j++) {
        v0[j] *= r[j] * (1.f + bf2f(P.final_norm[t]));
        v1[j] *= r[j] * (1.f + bf2f(P.final_norm[t + 512]));
    }
    consumer_sync();  // red is reused
    rms_scale<SPEC_T>(red, v0, v1, r);
#pragma unroll
    for (int j = 0; j < SPEC_T; j++) {
        vec[swz(j * 2048 + HID + t)] = v0[j] * r[j] * (1.f + bf2f(P.mtp_norm_h[t]));
        vec[swz(j * 2048 + HID + t + 512)] = v1[j] * r[j] * (1.f + bf2f(P.mtp_norm_h[t + 512]));
    }
    consumer_sync();
#pragma unroll
    for (int j = 0; j < SPEC_T; j++) {
        v0[j] = bf2f(P.embed[(size_t)next[j] * HID + t]);
        v1[j] = bf2f(P.embed[(size_t)next[j] * HID + t + 512]);
    }
    rms_scale<SPEC_T>(red, v0, v1, r);
#pragma unroll
    for (int j = 0; j < SPEC_T; j++) {
        vec[swz(j * 2048 + t)] = v0[j] * r[j] * (1.f + bf2f(P.mtp_norm_e[t]));
        vec[swz(j * 2048 + t + 512)] = v1[j] * r[j] * (1.f + bf2f(P.mtp_norm_e[t + 512]));
    }
    consumer_sync();
}

// Combine every CTA's argmax candidate (two LL words at slots + c * stride) into the winning key.
__device__ unsigned long long reduce_lm(const unsigned long long* slots, int stride, uint32_t e, unsigned long long* sk) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    unsigned long long key = 0;
    if (t < gridDim.x) {
        const unsigned long long* p = slots + (size_t)t * stride;
        const unsigned long long a = ld_ll(p), b = ld_ll(p + 1);
        key = ((unsigned long long)ll_wait_raw(a, p, e) << 32) | ll_wait_raw(b, p + 1, e);
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
//
// With T = 2 the two tokens' updates run back to back on the state held in registers. The
// state after the first token goes to state_bak and the one after the second to state, so
// the next cycle starts from whichever matches the tokens it accepted (`from_bak`).
template <int T>
__device__ void gdn_op(const LayerPtrs& L, int pos, uint32_t e, float* sm, unsigned long long* ready_ts,
                       bool from_bak) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    float* sqk = sm;             // [T][256] conv'd q then k
    float* sv = sm + T * 256;    // [T][32]
    float* sc = sm + T * 288;    // [T][4]: beta, decay, rq, rk
    constexpr int SLICES = LIN_DV / GDN_SLICE;
    constexpr int RM = CONV_RING - 1;
    for (int task = blockIdx.x; task < LIN_H * SLICES; task += gridDim.x) {
        const int h = task / SLICES, sl = task % SLICES;
        if (t < 256 + GDN_SLICE) {  // causal conv (window of 4 incl. current) + SiLU
            const int ch = t < 128 ? h * LIN_DK + t
                         : t < 256 ? LIN_QK + h * LIN_DK + (t - 128)
                                   : 2 * LIN_QK + h * LIN_DV + sl * GDN_SLICE + (t - 256);
            float* hist = L.conv_hist + ch * CONV_RING;
            float win[CONV_K - 1 + T];  // inputs at pos-3 .. pos+T-1
            win[0] = __ldcg(hist + ((pos - 3) & RM));
            win[1] = __ldcg(hist + ((pos - 2) & RM));
            win[2] = __ldcg(hist + ((pos - 1) & RM));
            const uint2 wq = *(const uint2*)(L.conv_w + ch * CONV_K);
            unsigned long long raw[T];
#pragma unroll
            for (int j = 0; j < T; j++) raw[j] = ld_ll(P.projl + j * PROJ_MAX + ch);
#pragma unroll
            for (int j = 0; j < T; j++) win[3 + j] = ll_wait(raw[j], P.projl + j * PROJ_MAX + ch, e - 1);
#pragma unroll
            for (int j = 0; j < T; j++) {
                const float acc = fmaf(bf_hi(wq.y), win[j + 3], fmaf(bf_lo(wq.y), win[j + 2],
                                       fmaf(bf_hi(wq.x), win[j + 1], bf_lo(wq.x) * win[j])));
                if (t >= 256 || sl == 0) __stcg(hist + ((pos + j) & RM), win[3 + j]);
                if (t < 256) sqk[j * 256 + t] = silu(acc); else sv[j * 32 + t - 256] = silu(acc);
            }
        } else if (t < 256 + GDN_SLICE + T) {  // one thread per token: beta and decay
            const int j = t - 256 - GDN_SLICE;
            const unsigned long long* pb = P.projl + j * PROJ_MAX + GDN_B_OFF + h;
            const unsigned long long* pa = P.projl + j * PROJ_MAX + GDN_A_OFF + h;
            const unsigned long long rb = ld_ll(pb), ra = ld_ll(pa);
            const float beta = sigmoid(ll_wait(rb, pb, e - 1));
            const float a = ll_wait(ra, pa, e - 1);
            sc[j * 4] = beta;
            sc[j * 4 + 1] = __expf(-__expf(L.A_log[h]) * softplus(a + bf2f(L.dt_bias[h])));
        }
        consumer_sync();
        if (ready_ts && t == 0) *ready_ts = globaltimer();
        if (warp < 2 * T) {  // L2 norms: warp 2j handles token j's q, warp 2j+1 its k
            const int j = warp >> 1, qk = warp & 1;
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) { const float v = sqk[j * 256 + qk * 128 + lane + 32 * i]; ss = fmaf(v, v, ss); }
            ss = warp_sum(ss);
            if (lane == 0) sc[j * 4 + 2 + qk] = rsqrtf(ss + 1e-6f) * (qk == 0 ? rsqrtf((float)LIN_DK) : 1.f);
        }
        consumer_sync();
        float s[2][4];
        float *S[2], *B[2];
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const size_t off = ((size_t)h * LIN_DV + sl * GDN_SLICE + warp + 16 * c) * LIN_DK;
            S[c] = L.state + off;
            B[c] = L.state_bak + off;
        }
#pragma unroll
        for (int c = 0; c < 2; c++)
#pragma unroll
            for (int i = 0; i < 4; i++) s[c][i] = __ldcg((from_bak ? B[c] : S[c]) + lane + 32 * i);
#pragma unroll
        for (int j = 0; j < T; j++) {
            const float beta = sc[j * 4], decay = sc[j * 4 + 1], rq = sc[j * 4 + 2], rk = sc[j * 4 + 3];
            float kv[4], qv[4], km[2] = {0.f, 0.f};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int k = lane + 32 * i;
                kv[i] = sqk[j * 256 + 128 + k] * rk;
                qv[i] = sqk[j * 256 + k] * rq;
            }
#pragma unroll
            for (int c = 0; c < 2; c++) {
#pragma unroll
                for (int i = 0; i < 4; i++) { s[c][i] *= decay; km[c] = fmaf(s[c][i], kv[i], km[c]); }
                km[c] = warp_sum(km[c]);
            }
#pragma unroll
            for (int c = 0; c < 2; c++) {
                float* dst = j == T - 1 ? S[c] : B[c];
                const float delta = (sv[j * 32 + warp + 16 * c] - km[c]) * beta;
                float o = 0.f;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    s[c][i] = fmaf(kv[i], delta, s[c][i]);
                    __stcg(dst + lane + 32 * i, s[c][i]);
                    o = fmaf(s[c][i], qv[i], o);
                }
                o = warp_sum(o);
                if (lane == 0) st_ll(P.corel + j * LIN_V + h * LIN_DV + sl * GDN_SLICE + warp + 16 * c, o, e);
            }
        }
        consumer_sync();
    }
}

// Copy the state after a cycle's first token over the main state (at the end of a speculative
// launch whose last guess was rejected), so the host always finds the current state in `state`.
__device__ void gdn_restore_state() {
    constexpr int SLICES = LIN_DV / GDN_SLICE;
    for (int l = 0; l < N_LAYERS; l++) {
        const LayerPtrs& L = P.layer[l];
        if (L.is_attn) continue;
        for (int task = blockIdx.x; task < LIN_H * SLICES; task += gridDim.x) {
            const size_t base = (size_t)task * GDN_SLICE * LIN_DK;  // head h, columns of slice sl: contiguous
            for (int i = threadIdx.x; i < GDN_SLICE * LIN_DK / 4; i += N_CONSUMERS)
                __stcg((float4*)(L.state + base) + i, __ldcg((const float4*)(L.state_bak + base) + i));
        }
    }
}

// ------------------------------------------------------------------ gated full attention
// Split-K decode attention. Task = (kv head, chunk of positions); it handles the 4 query
// heads sharing that KV head. Partials are merged by whichever task finishes last.
// Cached positions each warp loads per memory round trip in attention. 1 is fastest: larger
// batches need registers the kernel doesn't have (17 warps/SM cap it at 96 per thread) and spill.
#ifndef MK_KV_BATCH
#define MK_KV_BATCH 1
#endif
constexpr int KV_BATCH = MK_KV_BATCH;

// Split-K over cached positions: task = kv_head * splits + split, at most one per CTA.
__device__ __forceinline__ void attn_split(int n, int& splits, int& chunk) {
    splits = min(MAX_SPLITS, (n + 31) / 32);
    chunk = (n + splits - 1) / splits;
    splits = (n + chunk - 1) / chunk;
}

// With T tokens, query token j (at pos + j) gets exactly the split layout a one-token step at
// that position would use; its tasks are numbered after token j-1's.
template <int T>
__device__ __forceinline__ bool attn_task(int task, int pos, int& j, int& g, int& sp, int& splits, int& chunk) {
#pragma unroll
    for (j = 0; j < T; j++) {
        attn_split(pos + j + 1, splits, chunk);
        if (task < NKV * splits) {
            g = task / splits;
            sp = task % splits;
            return true;
        }
        task -= NKV * splits;
    }
    return false;
}

// Pull this CTA's slice of the KV cache into L2 while the QKV projection runs, so the
// attention scan reads L2 instead of waiting on DRAM one position at a time.
template <int T>
__device__ __forceinline__ void prefetch_kv(const LayerPtrs& L, int pos) {
    int j, g, sp, splits, chunk;
    if (!attn_task<T>(blockIdx.x, pos, j, g, sp, splits, chunk)) return;
    const int t0 = sp * chunk, t1 = min(pos, t0 + chunk);  // positions >= pos are new this step
    if (t1 <= t0) return;
    const size_t off = ((size_t)g * P.max_seq + t0) * HD;
    prefetch_l2(L.k_cache + off, (uint32_t)(t1 - t0) * HD * 2);
    prefetch_l2(L.v_cache + off, (uint32_t)(t1 - t0) * HD * 2);
}

template <int T>
__device__ void attn_op(const LayerPtrs& L, int pos, uint32_t e, float* sm, float* red, unsigned long long* ready_ts) {
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    float* sq = sm;              // [4][256]
    float* sacc = sm + 1024;     // [4][256]; reused by the merge for per-split weights
    float* sml = sm + 2048;      // [16 warps][4 heads][m,l]
    float* skv = sm + 2176;      // [T new positions][k 256 | v 256]
    int* sflag = (int*)(red + 32);
    const size_t ms = (size_t)P.max_seq;
    int jt, g, sp, splits, chunk;
    for (int task = blockIdx.x; attn_task<T>(task, pos, jt, g, sp, splits, chunk); task += gridDim.x) {
        const int n = pos + jt + 1, qpos = pos + jt;
        const int t0 = sp * chunk, t1 = min(n, t0 + chunk);
        // Positions >= pos belong to this step's tokens: their K/V comes from the projection, not
        // the cache. Every task covering one computes it (identically) and writes it to the cache.
        const int nb = max(t0, pos), nn = max(0, t1 - nb);
        const bf16* Kc = L.k_cache + (size_t)g * ms * HD + lane * 8;
        const bf16* Vc = L.v_cache + (size_t)g * ms * HD + lane * 8;
        uint4 kq[KV_BATCH], vq[KV_BATCH];
#pragma unroll
        for (int b = 0; b < KV_BATCH; b++) {  // first batch of cached positions, before q is ready
            const int tt = t0 + warp + b * N_CONSUMER_WARPS;
            if (tt < min(t1, pos)) {
                kq[b] = __ldcg((const uint4*)(Kc + (size_t)tt * HD));
                vq[b] = __ldcg((const uint4*)(Vc + (size_t)tt * HD));
            }
        }
        // q and the new positions' k get per-head RMSNorm, written to smem before RoPE
        if (warp < Q_PER_KV + nn) {
            const bool isq = warp < Q_PER_KV;
            const int i = warp - Q_PER_KV;
            const unsigned long long* src = isq ? P.projl + jt * PROJ_MAX + (g * Q_PER_KV + warp) * 2 * HD
                                                : P.projl + (nb + i - pos) * PROJ_MAX + ATT_K_OFF + g * HD;
            const bf16* nw = isq ? L.q_norm : L.k_norm;
            unsigned long long raw[8];
#pragma unroll
            for (int j = 0; j < 8; j++) raw[j] = ld_ll(src + lane * 8 + j);
            float v[8], ss = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) { v[j] = ll_wait(raw[j], src + lane * 8 + j, e - 1); ss = fmaf(v[j], v[j], ss); }
            const float r = rsqrtf(warp_sum(ss) / HD + EPS);
            float* dst = isq ? sq + warp * HD : skv + i * 2 * HD;
#pragma unroll
            for (int j = 0; j < 8; j++) dst[lane * 8 + j] = v[j] * r * (1.f + bf2f(nw[lane * 8 + j]));
        } else if (warp >= Q_PER_KV + T && warp < Q_PER_KV + T + nn) {
            const int i = warp - Q_PER_KV - T;
            const unsigned long long* src = P.projl + (nb + i - pos) * PROJ_MAX + ATT_V_OFF + g * HD;
            unsigned long long raw[8];
#pragma unroll
            for (int j = 0; j < 8; j++) raw[j] = ld_ll(src + lane * 8 + j);
#pragma unroll
            for (int j = 0; j < 8; j++) skv[i * 2 * HD + HD + lane * 8 + j] = ll_wait(raw[j], src + lane * 8 + j, e - 1);
        }
        consumer_sync();
        if (ready_ts && t == 0) *ready_ts = globaltimer();
        if (t < (Q_PER_KV + nn) * 32) {  // RoPE on the first 64 dims (half-split pairs i, i+32)
            const int hh = t >> 5, i = t & 31;
            float* b = hh < Q_PER_KV ? sq + hh * HD : skv + (hh - Q_PER_KV) * 2 * HD;
            const int rp = hh < Q_PER_KV ? qpos : nb + hh - Q_PER_KV;
            const float c = P.rope_cos[rp * (ROT / 2) + i], s = P.rope_sin[rp * (ROT / 2) + i];
            const float x1 = b[i], x2 = b[i + 32];
            b[i] = fmaf(x1, c, -(x2 * s));
            b[i + 32] = fmaf(x2, c, x1 * s);
        }
        consumer_sync();
        if (nn > 0) {
            for (int i = 0; i < nn; i++) {
                const int d = t & (HD - 1);
                bf16* cache = (t < HD) ? L.k_cache : L.v_cache;
                cache[((size_t)g * ms + nb + i) * HD + d] = __float2bfloat16(skv[i * 2 * HD + t]);
            }
            consumer_sync();
        }
        // q stays in smem (registers are tight: 17 warps/SM leave ~96 per thread)
        float m[Q_PER_KV], l[Q_PER_KV], acc[Q_PER_KV][8];
#pragma unroll
        for (int h = 0; h < Q_PER_KV; h++) {
            m[h] = -1e30f;
            l[h] = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) acc[h][j] = 0.f;
        }
        // Each warp scans positions t0+warp, t0+warp+16, ... with an online softmax, loading
        // KV_BATCH positions per memory round trip (DRAM is saturated by the weight stream, so
        // each trip on this critical path costs ~1 us). The first batch was issued before
        // waiting on q; the new tokens' K/V (written above) is re-read.
        for (int base = t0 + warp, b0 = 0; base < t1; base += KV_BATCH * N_CONSUMER_WARPS, b0++) {
            if (b0 > 0) {
#pragma unroll
                for (int b = 0; b < KV_BATCH; b++) {
                    const int tt = base + b * N_CONSUMER_WARPS;
                    if (tt < t1) {
                        kq[b] = __ldcg((const uint4*)(Kc + (size_t)tt * HD));
                        vq[b] = __ldcg((const uint4*)(Vc + (size_t)tt * HD));
                    }
                }
            }
#pragma unroll
            for (int b = 0; b < KV_BATCH; b++) {
                const int tt = base + b * N_CONSUMER_WARPS;
                if (tt >= t1) break;
                if (tt >= pos) {
                    kq[b] = __ldcg((const uint4*)(Kc + (size_t)tt * HD));
                    vq[b] = __ldcg((const uint4*)(Vc + (size_t)tt * HD));
                }
                float kf[8], vf[8], sc[Q_PER_KV];
                unpack8(kq[b], kf);
                unpack8(vq[b], vf);
#pragma unroll
                for (int h = 0; h < Q_PER_KV; h++) {
                    const float4 q0 = *(const float4*)(sq + h * HD + lane * 8);
                    const float4 q1 = *(const float4*)(sq + h * HD + lane * 8 + 4);
                    float d = q0.x * kf[0];
                    d = fmaf(q0.y, kf[1], d); d = fmaf(q0.z, kf[2], d); d = fmaf(q0.w, kf[3], d);
                    d = fmaf(q1.x, kf[4], d); d = fmaf(q1.y, kf[5], d); d = fmaf(q1.z, kf[6], d); d = fmaf(q1.w, kf[7], d);
                    sc[h] = d;
                }
#pragma unroll
                for (int o = 16; o; o >>= 1)
#pragma unroll
                    for (int h = 0; h < Q_PER_KV; h++) sc[h] += __shfl_xor_sync(~0u, sc[h], o);
#pragma unroll
                for (int h = 0; h < Q_PER_KV; h++) sc[h] *= 0.0625f;  // 1 / sqrt(head_dim)
#pragma unroll
                for (int h = 0; h < Q_PER_KV; h++) {
                    const float mn = fmaxf(m[h], sc[h]);
                    const float corr = __expf(m[h] - mn), p = __expf(sc[h] - mn);
                    l[h] = fmaf(l[h], corr, p);
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[h][j] = fmaf(p, vf[j], acc[h][j] * corr);
                    m[h] = mn;
                }
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
        float* const pg = P.attn_part + ((size_t)jt * NKV + g) * MAX_SPLITS * Q_PER_KV * PART_STRIDE;
        float* part = pg + (size_t)sp * Q_PER_KV * PART_STRIDE;
        if (t < Q_PER_KV) {
            float M = -1e30f, Ls = 0.f;
            for (int w = 0; w < N_CONSUMER_WARPS; w++) M = fmaxf(M, sml[(w * Q_PER_KV + t) * 2]);
            for (int w = 0; w < N_CONSUMER_WARPS; w++)
                Ls = fmaf(sml[(w * Q_PER_KV + t) * 2 + 1], __expf(sml[(w * Q_PER_KV + t) * 2] - M), Ls);
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
        unsigned* cnt = P.attn_cnt + jt * NKV + g;
        if (t == 0) {
            __threadfence();
            *sflag = atomicAdd(cnt, 1u) == (unsigned)(splits - 1);
        }
        consumer_sync();
        if (*sflag) {  // last task for this (token, KV head) merges all partials
            __threadfence();
            float* sw = sacc;  // [splits][4] merge weights
            for (int i = t; i < splits * Q_PER_KV; i += N_CONSUMERS) {
                sw[i] = __ldcg(pg + i * PART_STRIDE + HD);
                sw[MAX_SPLITS * Q_PER_KV + i] = __ldcg(pg + i * PART_STRIDE + HD + 1);
            }
            consumer_sync();
            if (t < Q_PER_KV) {
                float M = -1e30f, den = 0.f;
                for (int p = 0; p < splits; p++) M = fmaxf(M, sw[p * Q_PER_KV + t]);
                for (int p = 0; p < splits; p++)
                    den = fmaf(__expf(sw[p * Q_PER_KV + t] - M), sw[MAX_SPLITS * Q_PER_KV + p * Q_PER_KV + t], den);
                for (int p = 0; p < splits; p++) sw[p * Q_PER_KV + t] = __expf(sw[p * Q_PER_KV + t] - M) / den;
            }
            consumer_sync();
            // Issue every split's load before summing: one L2 round trip per output, not one
            // per split (the merge is on the critical path of the whole layer).
            for (int i = t; i < Q_PER_KV * HD; i += N_CONSUMERS) {
                const int h = i / HD, d = i % HD;
                float v[MAX_SPLITS];
#pragma unroll
                for (int p = 0; p < MAX_SPLITS; p++)
                    v[p] = p < splits ? __ldcg(pg + (p * Q_PER_KV + h) * PART_STRIDE + d) : 0.f;
                float num = 0.f;
#pragma unroll
                for (int p = 0; p < MAX_SPLITS; p++) num = fmaf(p < splits ? sw[p * Q_PER_KV + h] : 0.f, v[p], num);
                st_ll(P.attnl + jt * NQ * HD + (g * Q_PER_KV + h) * HD + d, num, e);
            }
            if (t == 0) atomicExch(cnt, 0u);
        }
        consumer_sync();
    }
}

// ------------------------------------------------------------------ kernel
// T = 1: plain decode, one token per step (prompt tokens are forced, the rest sampled).
// T = 2: speculative cycles. Cycle at position p feeds [x_p, d] (pending token and draft),
// samples y0 at p and y1 at p + 1 exactly as two plain steps would, and accepts the draft if
// y0 == d (the speculative-sampling rule for a deterministic draft: y0 is a sample from the
// model, so accepting exactly when it equals the draft leaves the output distribution intact).
// It emits y0, plus y1 when the draft was accepted, then runs the MTP head on
// (h_p, y0) and (h_p+1, y1) and drafts the next cycle's guess from the last accepted position.
template <bool Q8, int T>
__global__ void __launch_bounds__(N_THREADS, 1) decode_kernel(Args A) {
    constexpr int NS = T == 1 ? STAGES : SPEC_STAGES;
    constexpr int VB = T == 1 ? VEC_BYTES : SPEC_VEC_BYTES;
    extern __shared__ __align__(1024) char smem[];
    char* ring = smem;
    float* vec = (float*)(smem + NS * CHUNK);
    float* red = (float*)(smem + NS * CHUNK + VB);
    uint64_t* full = (uint64_t*)(red + RED_FLOATS);
    uint64_t* empty = full + NS;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;

    volatile int* stop = (volatile int*)(red + 36);    // consumers -> producer
    volatile int* issued = (volatile int*)(red + 37);  // producer -> consumers: chunks issued
    if (t == 0) {
        for (int s = 0; s < NS; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], N_CONSUMER_WARPS); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        *stop = 0;
        *issued = -1;
    }
    __syncthreads();

    if (warp == N_CONSUMER_WARPS) {
        if (lane == 0) producer<Q8, NS>(A, ring, full, empty, stop, issued);
        return;
    }

    // This CTA owns rows [xr0, xr1) of the residual stream (the out/down GEMV rows); it keeps
    // the fp32 master copy in smem (token j at xs + j * XS_STRIDE) and publishes LL copies.
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
    auto drain_and_stop = [&]() {  // leave early: let the producer's in-flight copies land first
        if (t == 0) {
            *stop = 1;
            unsigned long long wd = 0;
            while (*issued < 0) { SPIN_GUARD(wd); }
            (void)wd;
            for (int c = R.c; c < *issued; c++) mbar_wait(&full[c % NS], (c / NS) & 1);
        }
        consumer_sync();
    };

    // One decoder layer on T tokens. `emb` (layer 0 only) supplies the residual stream's start;
    // otherwise it continues from xs / `resid`.
    auto layer = [&](const LayerPtrs& L, int pos, const bf16* const* emb, unsigned long long* resid, bool from_bak) {
        // token mixer input projection (RMSNorm fused)
        if (L.is_attn && t == 0) prefetch_kv<T>(L, pos);
        prologue_rmsnorm<T>(vec, red, e - 1, emb, L.in_norm, resid);
        op_ready();
        auto store = [&](int r, const auto& a) {
            if (lane == 0)
#pragma unroll
                for (int j = 0; j < T; j++) st_ll(P.projl + j * PROJ_MAX + r, a[j][0], e);
        };
        if (L.is_attn) gemv<HID, 1, Q8, T, NS>(ATT_IN_ROWS, R, vec, prof, store);
        else gemv<HID, 1, Q8, T, NS>(GDN_IN_ROWS, R, vec, prof, store);
        op_end();
        // token mixer core
        unsigned long long* rts = (prof && t == 0) ? prow + op * PROF_FIELDS : nullptr;
        if (rts) *rts = globaltimer();  // overwritten once a task's inputs are ready
        if (L.is_attn) attn_op<T>(L, pos, e, vec, red, rts);
        else gdn_op<T>(L, pos, e, vec, rts, from_bak);
        op_end();
        // output projection + residual
        if (L.is_attn) prologue_attn_gate<T>(vec, e);
        else prologue_gated_norm<T>(vec, L, e);
        op_ready();
        gemv<2048, 1, Q8, T, NS>(HID, R, vec, prof, [&](int r, const auto& a) {
            if (lane == 0)
#pragma unroll
                for (int j = 0; j < T; j++) {
                    float* x = xs + j * XS_STRIDE + r - xr0;
                    const float xn = (emb ? bf2f(emb[j][r]) : *x) + a[j][0];
                    *x = xn;
                    st_ll(resid + j * HID + r, xn, e);
                }
        });
        op_end();
        // MLP gate/up with SwiGLU epilogue
        prologue_rmsnorm<T>(vec, red, e - 1, nullptr, L.post_norm, resid);
        op_ready();
        gemv<HID, 2, Q8, T, NS>(2 * INTER, R, vec, prof, [&](int u, const auto& a) {
            if (lane == 0)
#pragma unroll
                for (int j = 0; j < T; j++) st_ll(P.actl + j * INTER + u, silu(a[j][0]) * a[j][1], e);
        });
        op_end();
        // MLP down + residual
        prologue_act<T>(vec, e);
        op_ready();
        gemv<INTER, 1, Q8, T, NS>(HID, R, vec, prof, [&](int r, const auto& a) {
            if (lane == 0)
#pragma unroll
                for (int j = 0; j < T; j++) {
                    float* x = xs + j * XS_STRIDE + r - xr0;
                    const float xn = *x + a[j][0];
                    *x = xn;
                    st_ll(resid + j * HID + r, xn, e);
                }
        });
        op_end();
    };

    // LM head + greedy / Gumbel-max sampling for T tokens at positions pos .. pos+T-1. Each CTA
    // publishes its best (score, index) per token; logits row = s * T + j.
    auto lm_head = [&](int s, int pos) {
        prologue_rmsnorm<T>(vec, red, e - 1, nullptr, P.final_norm, P.xl);
        op_ready();
        unsigned long long best[T];
#pragma unroll
        for (int j = 0; j < T; j++) best[j] = 0;
        gemv<HID, 1, Q8, T, NS>(VOCAB, R, vec, prof, [&](int r, const auto& a) {
#pragma unroll
            for (int j = 0; j < T; j++) {
                float score = a[j][0];
                if (A.logits_out && lane == 0) A.logits_out[((size_t)s * T + j) * VOCAB + r] = score;
                if (A.inv_temperature > 0.f) {
                    const uint32_t h = hash3(A.seed, (uint32_t)(pos + j), (uint32_t)r);
                    const float u = ((h >> 8) + 0.5f) * (1.f / 16777216.f);
                    score = fmaf(score, A.inv_temperature, -__logf(-__logf(u)));
                }
                const unsigned long long k = pack_score(score, r);
                best[j] = k > best[j] ? k : best[j];
            }
        });
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < T; j++) sk[warp * T + j] = best[j];
        consumer_sync();
        if (t < T) {
            unsigned long long b = 0;
            for (int i = 0; i < N_CONSUMER_WARPS; i++) b = sk[i * T + t] > b ? sk[i * T + t] : b;
            st_ll_raw(P.lml + (blockIdx.x * T + t) * 2, (uint32_t)(b >> 32), e);
            st_ll_raw(P.lml + (blockIdx.x * T + t) * 2 + 1, (uint32_t)b, e);
        }
        op_end();
    };

    auto is_stop = [&](int token) {
        bool r = false;
        for (int i = 0; i < A.n_stop; i++) r |= (token == A.stop_ids[i]);
        return r;
    };

    if constexpr (T == 1) {
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
                const unsigned long long key = reduce_lm(P.lml, 2, e - 1, sk);
                if (blockIdx.x == 0 && t == 0) A.tokens_out[s - 1] = key;
                token = unpack_index(key);
                if (is_stop(token)) {  // every CTA sees the same token, so all stop at this step
                    drain_and_stop();
                    steps_done = s;
                    break;
                }
            }
            const bf16* emb[1] = {P.embed + (size_t)token * HID};
            for (int l = 0; l < N_LAYERS; l++) layer(P.layer[l], pos, l == 0 ? emb : nullptr, P.xl, false);
            if (A.resid_out && t < xr1 - xr0) A.resid_out[(size_t)s * HID + xr0 + t] = xs[t];
            // Publish this step's side data (conv history, KV) before anything of the next step.
            __threadfence();
            if (s >= A.lm_from) {
                lm_head(s, pos);
                last_lm = s;
            }
        }
        if (blockIdx.x == 0 && steps_done == A.n_steps && last_lm == A.n_steps - 1) {
            const unsigned long long key = reduce_lm(P.lml, 2, e - 1, sk);
            if (t == 0) A.tokens_out[A.n_steps - 1] = key;
        }
        if (blockIdx.x == 0 && t == 0) {
            if (A.step_ts) A.step_ts[steps_done] = globaltimer();
            A.out_info[0] = steps_done;
            A.out_info[1] = (int)e;
        }
    } else {
        int pos = A.start_pos, emitted = 0, cyc = 0;
        int next_tok = A.first_tok, draft = A.first_draft;
        bool from_bak = false, done = false;
        for (;; cyc++) {
            prof = A.prof && cyc == A.prof_step;
            if (prof) {
                prow = A.prof + (size_t)blockIdx.x * PROF_MAX_OPS * PROF_FIELDS;
                if (t == 0) prow[(PROF_MAX_OPS - 1) * PROF_FIELDS + 3] = globaltimer();  // cycle start
            }
            op = 0;
            if (cyc > 0) {  // the previous cycle's draft
                const int row = unpack_index(reduce_lm(P.lml + SPEC_T * 2 * gridDim.x, 2, e - 1, sk));
                draft = P.draft_ids ? P.draft_ids[row] : row;
                if (done || cyc == A.n_steps) break;
            }
            if (A.step_ts && blockIdx.x == 0 && t == 0) A.step_ts[cyc] = globaltimer();
            const int tok[SPEC_T] = {next_tok, draft};
            const bf16* emb[SPEC_T] = {P.embed + (size_t)tok[0] * HID, P.embed + (size_t)tok[1] * HID};
            for (int l = 0; l < N_LAYERS; l++) layer(P.layer[l], pos, l == 0 ? emb : nullptr, P.xl, from_bak);
            __threadfence();
            lm_head(cyc, pos);

            // verify
            const int y[SPEC_T] = {unpack_index(reduce_lm(P.lml, SPEC_T * 2, e - 1, sk)),
                                   unpack_index(reduce_lm(P.lml + 2, SPEC_T * 2, e - 1, sk))};
            const bool stop0 = is_stop(y[0]);
            const int n_new = (y[0] == tok[1] && !stop0 && emitted + 2 <= A.limit) ? 2 : 1;
            if (blockIdx.x == 0 && t == 0) {
                A.spec_out[emitted] = y[0];
                if (n_new == 2) A.spec_out[emitted + 1] = y[1];
                if (A.cycle_out) { A.cycle_out[2 * cyc] = n_new; A.cycle_out[2 * cyc + 1] = tok[1]; }
            }
            emitted += n_new;
            done = stop0 || (n_new == 2 && is_stop(y[1])) || emitted >= A.limit;
            next_tok = y[n_new - 1];
            from_bak = n_new == 1;

            // MTP head: fc, then its decoder layer on positions pos, pos + 1
            prologue_mtp_in(vec, red, e - 2, y);
            op_ready();
            gemv<2048, 1, Q8, SPEC_T, NS>(HID, R, vec, prof, [&](int r, const auto& a) {
                if (lane == 0)
#pragma unroll
                    for (int j = 0; j < SPEC_T; j++) {
                        xs[j * XS_STRIDE + r - xr0] = a[j][0];
                        st_ll(P.xml + j * HID + r, a[j][0], e);
                    }
            });
            op_end();
            layer(P.mtp, pos, nullptr, P.xml, false);
            // draft head (greedy) on the MTP output at the last accepted position
            prologue_rmsnorm<1>(vec, red, e - 1, nullptr, P.mtp_norm, P.xml + (n_new - 1) * HID);
            op_ready();
            unsigned long long best = 0;
            gemv<HID, 1, Q8, 1, NS>((int)P.draft_vocab, R, vec, prof, [&](int r, const auto& a) {
                const unsigned long long k = pack_score(a[0][0], r);
                best = k > best ? k : best;
            });
            if (lane == 0) sk[warp] = best;
            consumer_sync();
            if (t == 0) {
                for (int i = 0; i < N_CONSUMER_WARPS; i++) best = sk[i] > best ? sk[i] : best;
                st_ll_raw(P.lml + SPEC_T * 2 * gridDim.x + 2 * blockIdx.x, (uint32_t)(best >> 32), e);
                st_ll_raw(P.lml + SPEC_T * 2 * gridDim.x + 2 * blockIdx.x + 1, (uint32_t)best, e);
            }
            op_end();
            pos += n_new;
        }
        if (cyc < A.n_steps) drain_and_stop();
        if (from_bak) gdn_restore_state();
        if (blockIdx.x == 0 && t == 0) {
            if (A.step_ts) A.step_ts[cyc] = globaltimer();
            A.out_info[0] = emitted;
            A.out_info[1] = (int)e;
            A.out_info[2] = pos;
            A.out_info[3] = next_tok;
            A.out_info[4] = draft;
            A.out_info[5] = cyc;
        }
    }
}

// ------------------------------------------------------------------ host API
void set_params(const Params& p) {
    cudaMemcpyToSymbol(P, &p, sizeof(Params));
}

int watchdog_fired() {  // and clear it (always 0 unless built with -DMK_WATCHDOG)
    int v = 0, zero = 0;
    cudaMemcpyFromSymbol(&v, mk_watchdog, sizeof(int));
    cudaMemcpyToSymbol(mk_watchdog, &zero, sizeof(int));
    return v;
}

cudaError_t launch(const Args& a, int grid, cudaStream_t stream) {
    static bool attr = false;
    const void* kernels[2][2] = {{(const void*)decode_kernel<false, 1>, (const void*)decode_kernel<true, 1>},
                                 {(const void*)decode_kernel<false, SPEC_T>, (const void*)decode_kernel<true, SPEC_T>}};
    if (!attr) {
        for (int s = 0; s < 2; s++)
            for (int q = 0; q < 2; q++) {
                cudaError_t e = cudaFuncSetAttribute(kernels[s][q], cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     s ? SPEC_SMEM_BYTES : SMEM_BYTES);
                if (e != cudaSuccess) return e;
            }
        attr = true;
    }
    Args args = a;
    void* kargs[] = {&args};
    const int s = a.spec ? 1 : 0;
    return cudaLaunchCooperativeKernel(kernels[s][a.q8 ? 1 : 0], dim3(grid), dim3(N_THREADS), kargs,
                                       s ? SPEC_SMEM_BYTES : SMEM_BYTES, stream);
}

}  // namespace mk
