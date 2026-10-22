// Grid-barrier latency for a persistent kernel (one CTA per SM, 544 threads).
// Build: nvcc -O3 -arch=sm_120a -o build/barrier bench/barrier.cu
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ unsigned long long ld_acquire(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned long long ld_relaxed(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void red_release(unsigned long long* p) {
    asm volatile("red.release.gpu.global.add.u64 [%0], 1;" ::"l"(p) : "memory");
}
__device__ __forceinline__ unsigned long long atom_add_acqrel(unsigned long long* p) {
    unsigned long long v;
    asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;" : "=l"(v) : "l"(p) : "memory");
    return v;
}

template <int MODE>
__global__ void __launch_bounds__(544, 1) bar_kernel(unsigned long long* bar, int iters, float* sink) {
    unsigned long long target = 0;
    float acc = 0.f;
    for (int i = 0; i < iters; i++) {
        acc += threadIdx.x * 1e-9f;
        __syncthreads();
        if (threadIdx.x == 0) {
            target += gridDim.x;
            if (MODE == 0) {  // threadfence + atomicAdd + acquire spin + threadfence
                __threadfence();
                atomicAdd(bar, 1ull);
                while (ld_acquire(bar) < target) {}
                __threadfence();
            } else if (MODE == 1) {  // release reduction + acquire spin
                red_release(bar);
                while (ld_acquire(bar) < target) {}
            } else if (MODE == 2) {  // release reduction + relaxed spin + acquire fence
                red_release(bar);
                while (ld_relaxed(bar) < target) {}
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
            } else if (MODE == 3) {  // acq_rel atomic returns old value; skip spin if last
                unsigned long long old = atom_add_acqrel(bar);
                if (old + 1 < target) while (ld_acquire(bar) < target) {}
            } else if (MODE == 4) {  // as 1 with backoff
                red_release(bar);
                while (ld_acquire(bar) < target) __nanosleep(20);
            }
        }
        __syncthreads();
    }
    if (acc == 1234.f) sink[0] = acc;
}

template <int MODE>
void run(const char* name, int sms) {
    unsigned long long* bar;
    float* sink;
    CK(cudaMalloc(&bar, 8));
    CK(cudaMalloc(&sink, 4));
    const int iters = 20000;
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9;
    for (int r = 0; r < 5; r++) {
        CK(cudaMemset(bar, 0, 8));
        void* args[] = {&bar, (void*)&iters, &sink};
        cudaEventRecord(a);
        CK(cudaLaunchCooperativeKernel((void*)bar_kernel<MODE>, dim3(sms), dim3(544), args, 0, 0));
        cudaEventRecord(b);
        CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b);
        best = ms < best ? ms : best;
    }
    printf("%-48s %6.3f us / barrier\n", name, best * 1e3 / iters);
    cudaFree(bar); cudaFree(sink);
}

int main() {
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    run<0>("threadfence + atomicAdd + ld.acquire spin", sms);
    run<1>("red.release + ld.acquire spin", sms);
    run<2>("red.release + ld.relaxed spin + fence", sms);
    run<3>("atom.acq_rel (last arriver skips spin)", sms);
    run<4>("red.release + ld.acquire spin w/ nanosleep", sms);
    return 0;
}
