#include <stdio.h>
#include <cuda_runtime.h>

int main(){
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for(int i = 0; i < nDevices; i++){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        int memClockRate, memBusWidth;
        cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, i);
        cudaDeviceGetAttribute(&memBusWidth, cudaDevAttrGlobalMemoryBusWidth, i);

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
