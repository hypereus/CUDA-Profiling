# Learning CUDA

Profiling CUDA kernels to check whether their measured performance matches what I predicted.

The first set of kernels are my own implementations of the SGEMM optimizations from Simon Boehm's
[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM).
Each kernel is checked for correctness and benchmarked against cuBLAS.

## Kernels

All kernels compute `C = alpha * A @ B + beta * C` on row-major FP32 matrices.

| # | File | Idea |
|---|---|---|
| 1 | [`1_naive_gemm.cuh`](boehm_kernels/1_naive_gemm.cuh) | One thread per output element, straight from global memory |
| 2 | [`2_gmem_coalesce.cuh`](boehm_kernels/2_gmem_coalesce.cuh) | Remap threads so a warp's global memory accesses are coalesced |
| 3 | [`3_smem_cache.cuh`](boehm_kernels/3_smem_cache.cuh) | Stage 32×32 tiles of A and B in shared memory |
| 4 | [`4_1D_blocktiling.cuh`](boehm_kernels/4_1D_blocktiling.cuh) | Each thread computes a column of `TM` outputs |
| 5 | [`5_2D_blocktiling.cuh`](boehm_kernels/5_2D_blocktiling.cuh) | Each thread computes a `TM × TN` block of outputs from register caches |
| 6 | [`6_vectorized.cuh`](boehm_kernels/6_vectorized.cuh) | `float4` global loads/stores and a transposed A tile in shared memory |

## Building and running

Requires the CUDA toolkit (with cuBLAS) and an NVIDIA GPU. The command below targets an RTX 30-series GPU (`sm_86`). Change `-arch` for other GPUs.

```bash
nvcc -O3 -arch=sm_86 -o boehm_runner boehm_runner.cu -lcublas

./boehm_runner                  # all kernels, sizes 128..4096
./boehm_runner <kernel>         # one kernel (0 = cuBLAS)
./boehm_runner <kernel> <size>  # one kernel, one square size
```

- Sizes must be multiples of 128, because kernels 3–6 don't check bounds.
- Add `-DUPTO=N` to compile only kernels 1..N, which is useful while a later kernel is still unfinished.

For every kernel, [`boehm_runner.cu`](boehm_runner.cu):
1. runs the kernel once and compares the result with cuBLAS `Sgemm`, starting from the same C (`alpha = 0.5`, `beta = 3.0`), and prints PASS/FAIL;
2. does one warmup run, then times 10 runs with CUDA events and reports the average time and GFLOP/s.

## Results

### Test machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (Ampere, compute capability 8.6) |
| SMs | 16 (1536 threads, 64K registers per SM) |
| GPU clock | 1500 MHz base, 2100 MHz max; 1942–1972 MHz sustained under load (60 W power limit) |
| FP32 throughput | 8.0 TFLOP/s theoretical at 1950 MHz (16 SMs × 128 lanes × 2 FLOP); 7.3 TFLOP/s measured with an FMA microbenchmark |
| Memory | 4 GB GDDR6, 128-bit bus at 12 Gbps: 192 GB/s theoretical, 171 GB/s measured device-to-device copy |
| L2 cache | 1.5 MB |
| Shared memory | 100 KB per SM, 48 KB per block (99 KB opt-in) |
| CPU | AMD Ryzen 7 5800H (8 cores / 16 threads) |
| RAM | 16 GB DDR4-3200 (2 × 8 GB) |
| OS | Linux, kernel 7.0.0-34-generic |
| Driver / CUDA | NVIDIA driver 610.57.04, CUDA 13.4 (nvcc V13.4.92) |

Each value is the mean of 5 separate runs of `./boehm_runner`, and each run times 10 launches after a warmup. GFLOP/s is computed from the mean time.

### 4096 × 4096

| Kernel | GFLOP/s | % of cuBLAS |
|---|---:|---:|
| 0. cuBLAS | 3573.3 | 100.0% |
| 1. naive | 57.3 | 1.6% |
| 2. gmem coalesce | 314.4 | 8.8% |
| 3. smem cache | 522.0 | 14.6% |
| 4. 1D blocktiling | 1394.1 | 39.0% |
| 5. 2D blocktiling | 2867.5 | 80.2% |
| 6. vectorized | 3434.3 | 96.1% |

<details>
<summary>All sizes</summary>

#### 128 × 128

| Kernel | GFLOP/s | % of cuBLAS |
|---|---:|---:|
| 0. cuBLAS | 545.3 | 100.0% |
| 1. naive | 48.1 | 8.8% |
| 2. gmem coalesce | 390.2 | 71.6% |
| 3. smem cache | 459.4 | 84.3% |
| 4. 1D blocktiling | 318.3 | 58.4% |
| 5. 2D blocktiling | 127.2 | 23.3% |
| 6. vectorized | 168.5 | 30.9% |

#### 256 × 256

| Kernel | GFLOP/s | % of cuBLAS |
|---|---:|---:|
| 0. cuBLAS | 1885.6 | 100.0% |
| 1. naive | 49.7 | 2.6% |
| 2. gmem coalesce | 445.8 | 23.6% |
| 3. smem cache | 574.6 | 30.5% |
| 4. 1D blocktiling | 1391.9 | 73.8% |
| 5. 2D blocktiling | 628.5 | 33.3% |
| 6. vectorized | 756.0 | 40.1% |

#### 512 × 512

| Kernel | GFLOP/s | % of cuBLAS |
|---|---:|---:|
| 0. cuBLAS | 3401.2 | 100.0% |
| 1. naive | 56.5 | 1.7% |
| 2. gmem coalesce | 318.9 | 9.4% |
| 3. smem cache | 506.3 | 14.9% |
| 4. 1D blocktiling | 1601.7 | 47.1% |
| 5. 2D blocktiling | 2307.4 | 67.8% |
| 6. vectorized | 2621.8 | 77.1% |

#### 1024 × 1024

| Kernel | GFLOP/s | % of cuBLAS |
|---|---:|---:|
| 0. cuBLAS | 4058.7 | 100.0% |
| 1. naive | 57.2 | 1.4% |
| 2. gmem coalesce | 303.6 | 7.5% |
| 3. smem cache | 535.5 | 13.2% |
| 4. 1D blocktiling | 1583.9 | 39.0% |
| 5. 2D blocktiling | 3125.1 | 77.0% |
| 6. vectorized | 3927.8 | 96.8% |

#### 2048 × 2048

| Kernel | GFLOP/s | % of cuBLAS |
|---|---:|---:|
| 0. cuBLAS | 4204.9 | 100.0% |
| 1. naive | 57.4 | 1.4% |
| 2. gmem coalesce | 303.6 | 7.2% |
| 3. smem cache | 521.3 | 12.4% |
| 4. 1D blocktiling | 1415.2 | 33.7% |
| 5. 2D blocktiling | 3023.1 | 71.9% |
| 6. vectorized | 3921.8 | 93.3% |

</details>

**Caveats**
- At 128 and 256 the kernels finish in microseconds, and launch overhead is a large part of the time. At 128 there are also too few blocks to fill the GPU, which is why kernels 5 and 6 (128×128 tiles, a single block at that size) come last there.
- Run-to-run variation is under 2% at 4096 but larger at smaller sizes. It is worst for cuBLAS at 1024, where one standard deviation is about 29% of the mean, so treat "% of cuBLAS" at 128–2048 as approximate.

## Project layout

```
boehm_kernels/     SGEMM kernels, one per optimization step
boehm_runner.cu    correctness check + benchmark against cuBLAS
docs/              notes (empty for now)
results/           benchmark output (empty for now)
```

## TODO

- Kernel 7: autotuned block/thread tile sizes
- Save benchmark output to `results/` and plot it
- Profile each kernel with Nsight Compute and compare against the predictions
