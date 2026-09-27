#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_2DBlocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C){
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const uint resultsBlock = BM * BN;
    const uint threadsBlock = resultsBlock / (TM * TN);

    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint strideA = threadsBlock / BK;

    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;
    const uint strideB = threadsBlock / BN;

    float threadresults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for(uint bkIdx = 0; bkIdx < K; bkIdx += BK){

        for(uint loadOffset = 0; loadOffset < BM; loadOffset += strideA){
            As[(innerRowA + loadOffset)* BK + innerColA] = A[(innerRowA + loadOffset)* K + innerColA];
        }
        for(uint loadOffset = 0; loadOffset < BK; loadOffset += strideB){
            Bs[(innerRowB + loadOffset)* BN + innerColB] = B[(innerRowB + loadOffset)* N + innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        //Storing all the values in the register array and then computes all the results in one go
        for(uint dotIdx = 0; dotIdx < BK; ++dotIdx){
            for(uint i = 0; i < TM; ++i){
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for(uint i = 0; i < TN; ++i){
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for(uint resIdxM = 0; resIdxM < TM; ++resIdxM){
                for(uint resIdxN = 0; resIdxN < TN; ++resIdxN){
                    threadresults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    for(uint resIdxM = 0; resIdxM < TM; ++resIdxM){
        for(uint resIdxN = 0; resIdxN < TN; ++resIdxN){
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = alpha * threadresults[resIdxM * TN + resIdxN] + beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }
}