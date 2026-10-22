// Memory-bandwidth roofline for this GPU: plain vectorized loads vs. a
// persistent TMA bulk-copy ring (the streaming pattern the megakernel uses).
// Build: nvcc -O3 -arch=sm_120a -o build/bw common/bw.cu
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void read_vec(const uint4* __restrict__ p, size_t n, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678) out[0] = acc;
}

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
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t policy) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
        ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy) : "memory");
}

template <int STAGES, int CHUNK>
__global__ void __launch_bounds__(544, 1) read_tma(const char* __restrict__ p, size_t bytes, unsigned* out) {
    extern __shared__ __align__(128) char smem[];
    uint64_t* full = (uint64_t*)(smem + STAGES * CHUNK);
    uint64_t* empty = full + STAGES;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n_consumer_warps = blockDim.x / 32 - 1;
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], n_consumer_warps); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    size_t per_block = bytes / gridDim.x / CHUNK * CHUNK;
    const char* base = p + per_block * blockIdx.x;
    int n_chunks = per_block / CHUNK;
    if (warp == n_consumer_warps) {
        if (lane == 0) {
            uint64_t policy;
            asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
            for (int c = 0; c < n_chunks; c++) {
                int s = c % STAGES;
                if (c >= STAGES) mbar_wait(&empty[s], ((c / STAGES) - 1) & 1);
                mbar_expect_tx(&full[s], CHUNK);
                tma_load(smem + s * CHUNK, base + (size_t)c * CHUNK, CHUNK, &full[s], policy);
            }
        }
        return;
    }
    unsigned acc = 0;
    for (int c = 0; c < n_chunks; c++) {
        int s = c % STAGES;
        mbar_wait(&full[s], (c / STAGES) & 1);
        const uint4* q = (const uint4*)(smem + s * CHUNK);
        for (int i = threadIdx.x; i < CHUNK / 16; i += n_consumer_warps * 32) {
            uint4 v = q[i];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s]);
    }
    if (acc == 0x12345678) out[0] = acc;
}

template <typename F>
float time_ms(F f, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    f();
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) f();
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / iters;
}

template <int STAGES, int CHUNK>
void run_tma(const char* d, size_t bytes, unsigned* out, int sms) {
    int smem = STAGES * CHUNK + 2 * STAGES * 8;
    CK(cudaFuncSetAttribute(read_tma<STAGES, CHUNK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    size_t used = bytes / sms / CHUNK * CHUNK * sms;
    float ms = time_ms([&] { read_tma<STAGES, CHUNK><<<sms, 544, smem>>>(d, bytes, out); }, 20);
    CK(cudaGetLastError());
    printf("tma ring  stages=%d chunk=%5dB smem=%6dB : %7.1f GB/s\n", STAGES, CHUNK, smem, used / ms / 1e6);
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("%s  SMs=%d  L2=%d MB  smem/block optin=%zu KB  smem/SM=%zu KB\n", prop.name, prop.multiProcessorCount,
           prop.l2CacheSize >> 20, prop.sharedMemPerBlockOptin >> 10, prop.sharedMemPerMultiprocessor >> 10);
    int clk, busw;
    cudaDeviceGetAttribute(&clk, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&busw, cudaDevAttrGlobalMemoryBusWidth, 0);
    printf("memory clock %d MHz, bus %d bit\n", clk / 1000, busw);
    size_t bytes = 1536ull << 20;  // ~ one decode step of weights
    char* d;
    unsigned* out;
    CK(cudaMalloc(&d, bytes));
    CK(cudaMalloc(&out, 4));
    CK(cudaMemset(d, 1, bytes));
    int sms = prop.multiProcessorCount;
    for (int mult : {1, 2, 4, 8, 16}) {
        for (int threads : {256, 512, 1024}) {
            float ms = time_ms([&] { read_vec<<<sms * mult, threads>>>((const uint4*)d, bytes / 16, out); }, 20);
            printf("ld.global grid=%4d x %4d : %7.1f GB/s\n", sms * mult, threads, bytes / ms / 1e6);
        }
    }
    run_tma<4, 16384>(d, bytes, out, sms);
    run_tma<6, 16384>(d, bytes, out, sms);
    run_tma<3, 32768>(d, bytes, out, sms);
    run_tma<8, 8192>(d, bytes, out, sms);
    run_tma<12, 8192>(d, bytes, out, sms);
    run_tma<16, 4096>(d, bytes, out, sms);
    run_tma<24, 4096>(d, bytes, out, sms);
    return 0;
}
