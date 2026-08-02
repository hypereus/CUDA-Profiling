#include "config.cuh"
#include "cuda_check.cuh"

int main(){
    printDeviceInfo();

    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, (size_t)-1)); // deliberately absurd allocation
}
