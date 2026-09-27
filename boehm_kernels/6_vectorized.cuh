#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorize(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C){
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

    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint strideA = threadsBlock / BK;

    const uint innerColB = threadIdx.x % (BN / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint strideB = threadsBlock / BN;

    float threadresults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for(uint bkIdx = 0; bkIdx < K; bkIdx += BK){

        float4 tmp = reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] = reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();

        A += BK;
        B += BK * N;

        //Storing all the values in the register array and then computes all the results in one go
        for(uint dotIdx = 0; dotIdx < BK; ++dotIdx){
            for(uint i = 0; i < TM; ++i){
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
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
        for(uint resIdxN = 0; resIdxN < TN; resIdxN += 4){
            float4 tmp = reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];

            tmp.x = alpha * threadresults[resIdxM * TN + resIdxN] + beta * tmp.x;
            tmp.y = alpha * threadresults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
            tmp.z = alpha * threadresults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
            tmp.w = alpha * threadresults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;

            reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =  tmp;
        }
    }
}