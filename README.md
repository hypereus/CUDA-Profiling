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
| 7 | [`7_autotuned.cuh`](boehm_kernels/7_autotuned.cuh) | *Not yet written* |

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

NVIDIA GeForce RTX 3050 Laptop GPU (sm_86), CUDA 13.4.

Each value is the mean of 5 separate runs of `./boehm_runner`, and each run times 10 launches after a warmup. GFLOP/s is computed from the mean time.

### 4096 × 4096

| Kernel | GFLOP/s | Time Taken (s) | % of cuBLAS |
|---|---:|---:|---:|
| 0. cuBLAS | 3573.3 | 3.848e-02 | 100.0% |
| 1. naive | 57.3 | 2.399e+00 | 1.6% |
| 2. gmem coalesce | 314.4 | 4.373e-01 | 8.8% |
| 3. smem cache | 522.0 | 2.634e-01 | 14.6% |
| 4. 1D blocktiling | 1394.1 | 9.862e-02 | 39.0% |
| 5. 2D blocktiling | 2867.5 | 4.795e-02 | 80.2% |
| 6. vectorized | 3434.3 | 4.003e-02 | 96.1% |

<details>
<summary>All sizes</summary>

#### 128 × 128

| Kernel | GFLOP/s | Time Taken (s) | % of cuBLAS |
|---|---:|---:|---:|
| 0. cuBLAS | 545.3 | 7.782e-06 | 100.0% |
| 1. naive | 48.1 | 8.830e-05 | 8.8% |
| 2. gmem coalesce | 390.2 | 1.088e-05 | 71.6% |
| 3. smem cache | 459.4 | 9.237e-06 | 84.3% |
| 4. 1D blocktiling | 318.3 | 1.333e-05 | 58.4% |
| 5. 2D blocktiling | 127.2 | 3.337e-05 | 23.3% |
| 6. vectorized | 168.5 | 2.519e-05 | 30.9% |

#### 256 × 256

| Kernel | GFLOP/s | Time Taken (s) | % of cuBLAS |
|---|---:|---:|---:|
| 0. cuBLAS | 1885.6 | 1.790e-05 | 100.0% |
| 1. naive | 49.7 | 6.792e-04 | 2.6% |
| 2. gmem coalesce | 445.8 | 7.572e-05 | 23.6% |
| 3. smem cache | 574.6 | 5.874e-05 | 30.5% |
| 4. 1D blocktiling | 1391.9 | 2.425e-05 | 73.8% |
| 5. 2D blocktiling | 628.5 | 5.370e-05 | 33.3% |
| 6. vectorized | 756.0 | 4.465e-05 | 40.1% |

#### 512 × 512

| Kernel | GFLOP/s | Time Taken (s) | % of cuBLAS |
|---|---:|---:|---:|
| 0. cuBLAS | 3401.2 | 7.916e-05 | 100.0% |
| 1. naive | 56.5 | 4.764e-03 | 1.7% |
| 2. gmem coalesce | 318.9 | 8.443e-04 | 9.4% |
| 3. smem cache | 506.3 | 5.318e-04 | 14.9% |
| 4. 1D blocktiling | 1601.7 | 1.681e-04 | 47.1% |
| 5. 2D blocktiling | 2307.4 | 1.167e-04 | 67.8% |
| 6. vectorized | 2621.8 | 1.027e-04 | 77.1% |

#### 1024 × 1024

| Kernel | GFLOP/s | Time Taken (s) | % of cuBLAS |
|---|---:|---:|---:|
| 0. cuBLAS | 4058.7 | 5.299e-04 | 100.0% |
| 1. naive | 57.2 | 3.760e-02 | 1.4% |
| 2. gmem coalesce | 303.6 | 7.085e-03 | 7.5% |
| 3. smem cache | 535.5 | 4.016e-03 | 13.2% |
| 4. 1D blocktiling | 1583.9 | 1.358e-03 | 39.0% |
| 5. 2D blocktiling | 3125.1 | 6.882e-04 | 77.0% |
| 6. vectorized | 3927.8 | 5.475e-04 | 96.8% |

#### 2048 × 2048

| Kernel | GFLOP/s | Time Taken (s) | % of cuBLAS |
|---|---:|---:|---:|
| 0. cuBLAS | 4204.9 | 4.089e-03 | 100.0% |
| 1. naive | 57.4 | 2.998e-01 | 1.4% |
| 2. gmem coalesce | 303.6 | 5.663e-02 | 7.2% |
| 3. smem cache | 521.3 | 3.298e-02 | 12.4% |
| 4. 1D blocktiling | 1415.2 | 1.215e-02 | 33.7% |
| 5. 2D blocktiling | 3023.1 | 5.687e-03 | 71.9% |
| 6. vectorized | 3921.8 | 4.384e-03 | 93.3% |

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
