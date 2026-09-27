// Runner for the Boehm SGEMM kernels in boehm_kernels/.
// Checks each kernel against cuBLAS, then times it and reports GFLOP/s.
//
// Build:
//   nvcc -O3 -arch=sm_86 -o boehm_runner boehm_runner.cu -lcublas
//   (add -DUPTO=3 to only compile kernels 1..3 while later ones are in progress)
//
// Usage:
//   ./boehm_runner                  all kernels, default sizes
//   ./boehm_runner <kernel>         one kernel (0 = cuBLAS), default sizes
//   ./boehm_runner <kernel> <size>  one kernel, one square size (multiple of 128)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Catches launch errors and errors raised while the kernel ran.
#define CUDA_CHECK_LAST()                                                    \
    do {                                                                     \
        CUDA_CHECK(cudaGetLastError());                                      \
        CUDA_CHECK(cudaDeviceSynchronize());                                 \
    } while (0)

#ifndef UPTO
#define UPTO 6
#endif

#include "boehm_kernels/1_naive_gemm.cuh"
#include "boehm_kernels/2_gmem_coalesce.cuh"
#include "boehm_kernels/3_smem_cache.cuh"
#if UPTO >= 4
#include "boehm_kernels/4_1D_blocktiling.cuh"
#endif
#if UPTO >= 5
#include "boehm_kernels/5_2D_blocktiling.cuh"
#endif
#if UPTO >= 6
#include "boehm_kernels/6_vectorized.cuh"
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#endif

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t st__ = (call);                                        \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    (int)st__);                                              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

static const char *kernelNames[] = {
    "cuBLAS",
    "naive",
    "gmem coalesce",
    "smem cache",
    "1D blocktiling",
    "2D blocktiling",
    "vectorized",
};

// ---------------------------------------------------------------------------
// Launchers: one per kernel, all row-major C = alpha * A @ B + beta * C.
// Grid orientation follows each kernel's own blockIdx.x / blockIdx.y usage.
// ---------------------------------------------------------------------------

void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    // cuBLAS is column-major: computing C^T = B^T @ A^T gives row-major C.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

void run_naive(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    naive_gemm<<<grid, block>>>(M, N, K, A, B, alpha, beta, C);
}

void run_gmem_coalesce(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    const uint BLOCKSIZE = 32;
    dim3 block(BLOCKSIZE * BLOCKSIZE);
    dim3 grid(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
    sgemm_global_mem_coalesce<BLOCKSIZE><<<grid, block>>>(M, N, K, alpha, beta, A, B, C);
}

void run_smem_cache(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    const uint BLOCKSIZE = 32;
    dim3 block(BLOCKSIZE * BLOCKSIZE);
    dim3 grid(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
    sgemm_smem_cache<BLOCKSIZE><<<grid, block>>>(M, N, K, alpha, beta, A, B, C);
}

#if UPTO >= 4
void run_1D_blocktiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    const int BM = 64, BN = 64, BK = 8, TM = 8;
    dim3 block((BM * BN) / TM);
    dim3 grid(CEIL_DIV(M, BM), CEIL_DIV(N, BN)); // kernel uses blockIdx.x as row
    sgemm_1DBlocktiling<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
#endif

#if UPTO >= 5
void run_2D_blocktiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); // kernel uses blockIdx.y as row
    sgemm_2DBlocktiling<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
#endif

#if UPTO >= 6
void run_vectorized(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); // kernel uses blockIdx.y as row
    sgemm_vectorize<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
#endif

bool run_kernel(int id, cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    switch(id){
        case 0: run_cublas(handle, M, N, K, alpha, A, B, beta, C); return true;
        case 1: run_naive(M, N, K, alpha, A, B, beta, C); return true;
        case 2: run_gmem_coalesce(M, N, K, alpha, A, B, beta, C); return true;
        case 3: run_smem_cache(M, N, K, alpha, A, B, beta, C); return true;
#if UPTO >= 4
        case 4: run_1D_blocktiling(M, N, K, alpha, A, B, beta, C); return true;
#endif
#if UPTO >= 5
        case 5: run_2D_blocktiling(M, N, K, alpha, A, B, beta, C); return true;
#endif
#if UPTO >= 6
        case 6: run_vectorized(M, N, K, alpha, A, B, beta, C); return true;
#endif
        default: return false;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void randomize(std::vector<float> &v){
    for(auto &x : v){
        x = (float)(rand() % 5) + 0.01f * (rand() % 5);
        if(rand() % 2) x = -x;
    }
}

bool verify(const std::vector<float> &ref, const std::vector<float> &out){
    for(size_t i = 0; i < ref.size(); i++){
        float diff = std::fabs(ref[i] - out[i]);
        if(std::isnan(out[i]) || diff > 0.01f * std::fmax(1.0f, std::fabs(ref[i]))){
            printf("    mismatch at %zu: expected %.4f, got %.4f\n", i, ref[i], out[i]);
            return false;
        }
    }
    return true;
}

void benchmark(int id, cublasHandle_t handle, int size, int repeats){
    const int M = size, N = size, K = size;
    const float alpha = 0.5f, beta = 3.0f;
    const size_t bytesA = sizeof(float) * M * K;
    const size_t bytesB = sizeof(float) * K * N;
    const size_t bytesC = sizeof(float) * M * N;

    std::vector<float> A(M * K), B(K * N), C(M * N), Cref(M * N), Cout(M * N);
    randomize(A); randomize(B); randomize(C);

    float *dA, *dB, *dC, *dCref;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMalloc(&dCref, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C.data(), bytesC, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCref, C.data(), bytesC, cudaMemcpyHostToDevice));

    // Correctness: one run of the kernel vs one run of cuBLAS from the same C.
    bool ok = true;
    if(id != 0){
        run_cublas(handle, M, N, K, alpha, dA, dB, beta, dCref);
        run_kernel(id, handle, M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(Cref.data(), dCref, bytesC, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Cout.data(), dC, bytesC, cudaMemcpyDeviceToHost));
        ok = verify(Cref, Cout);
    }

    // Timing (C keeps accumulating here, which doesn't matter for speed).
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    run_kernel(id, handle, M, N, K, alpha, dA, dB, beta, dC); // warmup
    CUDA_CHECK(cudaEventRecord(start));
    for(int r = 0; r < repeats; r++){
        run_kernel(id, handle, M, N, K, alpha, dA, dB, beta, dC);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK_LAST();

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    const double avgMs = ms / repeats;
    const double gflops = (2.0 * M * N * K + 3.0 * M * N) * 1e-9 / (avgMs * 1e-3);

    printf("  size %5d | %9.3f ms | %8.1f GFLOP/s | %s\n", size, avgMs, gflops, id == 0 ? "reference" : (ok ? "PASS" : "FAIL"));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dCref));
}

int main(int argc, char **argv){
    const int numKernels = UPTO + 1; // 0 (cuBLAS) .. UPTO
    std::vector<int> kernels;
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    const int repeats = 10;

    if(argc >= 2){
        int id = atoi(argv[1]);
        if(id < 0 || id >= numKernels){
            fprintf(stderr, "kernel must be in [0, %d] (0 = cuBLAS)\n", numKernels - 1);
            return EXIT_FAILURE;
        }
        kernels.push_back(id);
    } else {
        for(int i = 0; i < numKernels; i++) kernels.push_back(i);
    }

    if(argc >= 3){
        int size = atoi(argv[2]);
        // Kernels 3+ have no bounds checks, so sizes must tile evenly.
        if(size <= 0 || size % 128 != 0){
            fprintf(stderr, "size must be a positive multiple of 128\n");
            return EXIT_FAILURE;
        }
        sizes = {size};
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    srand(42);
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    for(int id : kernels){
        printf("\n[%d] %s\n", id, kernelNames[id]);
        for(int size : sizes){
            benchmark(id, handle, size, repeats);
        }
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
