#pragma once

#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>

#include "cuda_check.cuh"

void printDeviceInfo(){
    int nDevices;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    for(int i = 0; i < nDevices; i++){
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        int memClockRate, memBusWidth;
        CUDA_CHECK(cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, i));
        CUDA_CHECK(cudaDeviceGetAttribute(&memBusWidth, cudaDevAttrGlobalMemoryBusWidth, i));

        int smCount, ccMajor, ccMinor, sharedMemPerBlock, maxThreadsPerSM;
        CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, i));
        CUDA_CHECK(cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, i));
        CUDA_CHECK(cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, i));
        CUDA_CHECK(cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, i));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, i));

        assert(prop.multiProcessorCount == smCount);
        assert(prop.major == ccMajor);
        assert(prop.minor == ccMinor);
        assert((int)prop.sharedMemPerBlock == sharedMemPerBlock);
        assert(prop.maxThreadsPerMultiProcessor == maxThreadsPerSM);

        printf("Device Number: %d\n", i);
        printf("Device Name: %s\n", prop.name);
        printf("SM Count: %d\n", prop.multiProcessorCount);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Shared Mem per Block (bytes): %zu\n", prop.sharedMemPerBlock);
        printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Memory Clock Rate (KHz): %d\n", memClockRate);
        printf("Memory Bus Width (bits): %d\n", memBusWidth);
        printf("Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*memClockRate*(memBusWidth/8)/1.0e6);
    }
}
