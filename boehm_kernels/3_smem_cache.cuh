#pragma once

#include <algorithm>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const uint BLOCKSIZE>
__global__ void sgemm_smem_cache(int M, int N, int K, float alpha, float beta, float *A, float *B, float *C){
    const uint crow = blockIdx.x;
    const uint ccol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const int threadrow = threadIdx.x / BLOCKSIZE;
    const int threadcol = threadIdx.x % BLOCKSIZE;

    A += crow * BLOCKSIZE * K;
    B += ccol * BLOCKSIZE;
    C += crow * BLOCKSIZE * N + ccol * BLOCKSIZE;

    float temp = 0.0f;
    for(int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE){
        As[threadrow * BLOCKSIZE + threadcol] = A[threadrow * K + threadcol];
        Bs[threadrow * BLOCKSIZE + threadcol] = B[threadrow * N + threadcol];

        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for(int dotIdx = 0; dotIdx < BLOCKSIZE; dotIdx++){
            temp += As[threadrow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadcol];
        }

        __syncthreads();
    }
    C[threadrow * N + threadcol] = alpha * temp + beta * C[threadrow * N + threadcol];

}