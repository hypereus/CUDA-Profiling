#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1DBlocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C){
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const uint threadCol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN;

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innercolA = threadIdx.x % BK;
    const uint innerrowA = threadIdx.x / BK;

    const uint innercolB = threadIdx.x % BN;
    const uint innerrowB = threadIdx.x / BN;

    float threadresults[TM] = {0.0f};

    for(uint bkIdx = 0; bkIdx < K; bkIdx += BK){
        As[innerrowA * BK + innercolA] = A[innerrowA * K + innercolA];
        Bs[innerrowB * BN + innercolB] = B[innerrowB * N + innercolB];
        __syncthreads();

        A += BK;
        B += BK * N;

        for(uint dotIdx = 0; dotIdx < BK; ++dotIdx){
            float tmpB = Bs[dotIdx * BN + threadCol];
            for(uint resIdx = 0; resIdx < TM; ++resIdx){
                threadresults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for(uint resIdx = 0; resIdx < TM; ++resIdx){
        C[(threadRow * TM + resIdx) * N + threadCol] = alpha * threadresults[resIdx] + beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }

}