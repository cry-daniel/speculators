// Minimal build/runtime probe for native SM120 ordered-metadata sparse MMA.
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

__global__ void sm120_sparse_f8f6_probe(std::uint32_t *output) {
  std::uint32_t d0 = 0;
  std::uint32_t d1 = 0;
  const std::uint32_t a0 = 0;
  const std::uint32_t a1 = 0;
  const std::uint32_t a2 = 0;
  const std::uint32_t a3 = 0;
  const std::uint32_t b0 = 0;
  const std::uint32_t b1 = 0;
  const std::uint32_t b2 = 0;
  const std::uint32_t b3 = 0;
  const std::uint32_t c0 = 0;
  const std::uint32_t c1 = 0;
  const std::uint32_t metadata = 0;

  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.sp::ordered_metadata."
      "m16n8k64.row.col.f16.e5m2.e3m2.f16 "
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, "
      "{%10, %11}, %12, 0x0;\n"
      : "=r"(d0), "=r"(d1)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "r"(b2), "r"(b3), "r"(c0), "r"(c1), "r"(metadata));

  const int lane = static_cast<int>(threadIdx.x) & 31;
  output[2 * lane] = d0;
  output[2 * lane + 1] = d1;
}

int main() {
  std::uint32_t *output = nullptr;
  cudaError_t status = cudaMalloc(&output, 64 * sizeof(std::uint32_t));
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
    return 1;
  }
  sm120_sparse_f8f6_probe<<<1, 32>>>(output);
  status = cudaDeviceSynchronize();
  cudaFree(output);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(status));
    return 2;
  }
  std::puts("SM120 sparse F8/F6 MMA probe passed");
  return 0;
}
