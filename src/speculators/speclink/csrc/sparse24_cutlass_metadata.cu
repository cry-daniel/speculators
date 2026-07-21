#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ uint16_t load_element_e(
    uint8_t const* metadata,
    int64_t row,
    int64_t source_col,
    int64_t input_stride) {
  int64_t const offset = row * input_stride + source_col * 2;
  return uint16_t(metadata[offset]) |
         (uint16_t(metadata[offset + 1]) << 8);
}

__global__ void pack_and_reorder_metadata_kernel(
    uint8_t const* metadata,
    uint16_t* reordered,
    int64_t n,
    int64_t source_columns,
    int64_t input_stride) {
  int64_t const linear =
      int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (linear >= n * source_columns) {
    return;
  }
  int64_t const source_row = linear / source_columns;
  int64_t const source_col = linear - source_row * source_columns;

  // CUTLASS host_reorder.h row interweave and 2x2 Z-to-N swizzle.
  int64_t dest_row =
      (source_row / 32) * 32 + (source_row % 8) * 4 +
      (source_row % 32) / 8;
  int64_t dest_col = source_col;
  if (((dest_row & 1) == 0) && ((dest_col & 1) == 1)) {
    ++dest_row;
    --dest_col;
  } else if (((dest_row & 1) == 1) && ((dest_col & 1) == 0)) {
    --dest_row;
    ++dest_col;
  }

  // ColumnMajorInterleaved<2>::packed({n, source_columns}).
  int64_t const dest_offset =
      (dest_col / 2) * (n * 2) + dest_row * 2 + (dest_col & 1);
  reordered[dest_offset] = load_element_e(
      metadata, source_row, source_col, input_stride);
}

}  // namespace

torch::Tensor sparse24_cutlass_reorder_metadata_cuda(
    torch::Tensor metadata,
    int64_t logical_k) {
  c10::cuda::CUDAGuard guard(metadata.device());
  int64_t const n = metadata.size(0);
  int64_t const columns = logical_k / 16;
  auto reordered = torch::empty(
      {n, columns}, metadata.options().dtype(torch::kInt16));
  constexpr int threads = 256;
  int64_t const elements = n * columns;
  int const blocks = int((elements + threads - 1) / threads);
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(metadata.get_device()).stream();
  pack_and_reorder_metadata_kernel<<<blocks, threads, 0, stream>>>(
      metadata.data_ptr<uint8_t>(),
      reinterpret_cast<uint16_t*>(reordered.data_ptr<int16_t>()),
      n,
      columns,
      metadata.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return reordered;
}
