#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "csrc/SpMM_API.cu"

namespace {

void check_cpu_weight(const torch::Tensor& weight) {
  TORCH_CHECK(weight.device().is_cpu(), "weight must be on CPU during compression");
  TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be BF16");
  TORCH_CHECK(weight.dim() == 2 && weight.is_contiguous(),
              "weight must be contiguous [N,K]");
  TORCH_CHECK(weight.size(0) % 64 == 0, "SpInfer N must be divisible by 64");
  TORCH_CHECK(weight.size(1) % 64 == 0, "SpInfer K must be divisible by 64");
}

void check_tokens(int64_t tokens) {
  TORCH_CHECK(tokens == 8 || tokens == 16 || tokens == 32 || tokens == 64 ||
                  tokens == 128 || (tokens > 0 && tokens % 128 == 0),
              "SpInfer token rows must be 8/16/32/64/128 or a multiple of 128");
}

}  // namespace

std::vector<torch::Tensor> spinfer_prepare_cpu(torch::Tensor weight) {
  check_cpu_weight(weight);
  __nv_bfloat16* values_raw = nullptr;
  int* local_offsets_raw = nullptr;
  int* median_offsets_raw = nullptr;
  int* global_offsets_raw = nullptr;
  uint64_t* bitmap_raw = nullptr;
  int max_nnz = 0;
  const int n = static_cast<int>(weight.size(0));
  const int k = static_cast<int>(weight.size(1));
  int num_global_tiles = InitSparseMatrixA_bitmap(
      reinterpret_cast<__nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      n, k, 8, 16, 64, 8, 64, 64, &values_raw, &local_offsets_raw,
      &median_offsets_raw, &global_offsets_raw, &bitmap_raw, max_nnz);
  TORCH_CHECK(num_global_tiles > 0 && values_raw != nullptr &&
                  median_offsets_raw != nullptr && global_offsets_raw != nullptr &&
                  bitmap_raw != nullptr,
              "SpInfer bitmap compression failed");

  const int64_t value_count = global_offsets_raw[num_global_tiles];
  const int64_t median_count = static_cast<int64_t>(num_global_tiles) * 4;
  const int64_t bitmap_count = static_cast<int64_t>(num_global_tiles) * 64;
  auto values =
      torch::empty({value_count}, torch::TensorOptions().dtype(torch::kBFloat16));
  auto global_offsets =
      torch::empty({num_global_tiles + 1},
                   torch::TensorOptions().dtype(torch::kInt32));
  auto median_offsets =
      torch::empty({median_count}, torch::TensorOptions().dtype(torch::kInt32));
  auto bitmap =
      torch::empty({bitmap_count}, torch::TensorOptions().dtype(torch::kInt64));
  auto max_nnz_tensor =
      torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32));

  std::memcpy(values.data_ptr<at::BFloat16>(), values_raw,
              value_count * sizeof(__nv_bfloat16));
  std::memcpy(global_offsets.data_ptr<int32_t>(), global_offsets_raw,
              (num_global_tiles + 1) * sizeof(int));
  std::memcpy(median_offsets.data_ptr<int32_t>(), median_offsets_raw,
              median_count * sizeof(int));
  std::memcpy(bitmap.data_ptr<int64_t>(), bitmap_raw,
              bitmap_count * sizeof(uint64_t));
  // The kernel copies vectorized BF16 chunks; preserve the artifact's
  // round-up convention and cap it at the full 64x64 tile.
  max_nnz = std::min(4096, ((max_nnz + 63) / 64) * 64);
  max_nnz_tensor.data_ptr<int32_t>()[0] = max_nnz;

  std::free(values_raw);
  std::free(local_offsets_raw);
  std::free(median_offsets_raw);
  std::free(global_offsets_raw);
  std::free(bitmap_raw);
  return {values, global_offsets, median_offsets, bitmap, max_nnz_tensor};
}

torch::Tensor spinfer_forward_impl(torch::Tensor input,
                                  torch::Tensor values,
                                  torch::Tensor global_offsets,
                                  torch::Tensor median_offsets,
                                  torch::Tensor bitmap,
                                  torch::Tensor max_nnz,
                                  int64_t n,
                                  int64_t split_k,
                                  c10::optional<torch::Tensor> output_arg) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16 &&
                  input.dim() == 2 && input.is_contiguous(),
              "input must be contiguous CUDA BF16 [M,K]");
  TORCH_CHECK(values.is_cuda() && values.scalar_type() == torch::kBFloat16 &&
                  values.is_contiguous(),
              "values must be contiguous CUDA BF16");
  TORCH_CHECK(global_offsets.is_cuda() &&
                  global_offsets.scalar_type() == torch::kInt32 &&
                  global_offsets.is_contiguous(),
              "global_offsets must be contiguous CUDA int32");
  TORCH_CHECK(median_offsets.is_cuda() &&
                  median_offsets.scalar_type() == torch::kInt32 &&
                  median_offsets.is_contiguous(),
              "median_offsets must be contiguous CUDA int32");
  TORCH_CHECK(bitmap.is_cuda() && bitmap.scalar_type() == torch::kInt64 &&
                  bitmap.is_contiguous(),
              "bitmap must be contiguous CUDA int64");
  TORCH_CHECK(max_nnz.is_cuda() && max_nnz.scalar_type() == torch::kInt32 &&
                  max_nnz.numel() == 1,
              "max_nnz must be one CUDA int32");
  TORCH_CHECK(input.device() == values.device() &&
                  input.device() == global_offsets.device() &&
                  input.device() == median_offsets.device() &&
                  input.device() == bitmap.device() &&
                  input.device() == max_nnz.device(),
              "all tensors must be on one CUDA device");
  TORCH_CHECK(n > 0 && n % 64 == 0, "N must be positive and divisible by 64");
  TORCH_CHECK(input.size(1) % 64 == 0, "K must be divisible by 64");
  TORCH_CHECK(split_k >= 1 && split_k <= input.size(1) / 64,
              "invalid split_k");
  check_tokens(input.size(0));

  c10::cuda::CUDAGuard guard(input.device());
  const bool accumulate = output_arg.has_value();
  auto output = accumulate
                    ? output_arg.value()
                    : torch::empty({input.size(0), n}, input.options());
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kBFloat16 &&
                  output.is_contiguous() &&
                  output.sizes() == torch::IntArrayRef({input.size(0), n}) &&
                  output.device() == input.device(),
              "output must be contiguous CUDA BF16 [M,N] on the input device");
  auto workspace =
      split_k == 1
          ? torch::empty({1}, input.options())
          : torch::empty({split_k, input.size(0), n}, input.options());
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  cudaError_t error = SpMM_SplitK_API_bitmap_v3(
      stream,
      reinterpret_cast<const __nv_bfloat16*>(values.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(values.data_ptr<at::BFloat16>()),
      global_offsets.data_ptr<int32_t>(), median_offsets.data_ptr<int32_t>(),
      reinterpret_cast<const uint64_t*>(bitmap.data_ptr<int64_t>()),
      max_nnz.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      static_cast<int>(n), static_cast<int>(input.size(0)),
      static_cast<int>(input.size(1)),
      reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr<at::BFloat16>()),
      static_cast<int>(split_k), accumulate);
  TORCH_CHECK(error == cudaSuccess, "SpInfer launch failed: ",
              cudaGetErrorString(error));
  return output;
}

torch::Tensor spinfer_forward(torch::Tensor input,
                             torch::Tensor values,
                             torch::Tensor global_offsets,
                             torch::Tensor median_offsets,
                             torch::Tensor bitmap,
                             torch::Tensor max_nnz,
                             int64_t n,
                             int64_t split_k) {
  return spinfer_forward_impl(input, values, global_offsets, median_offsets,
                              bitmap, max_nnz, n, split_k, c10::nullopt);
}

torch::Tensor spinfer_forward_add(torch::Tensor input,
                                 torch::Tensor values,
                                 torch::Tensor global_offsets,
                                 torch::Tensor median_offsets,
                                 torch::Tensor bitmap,
                                 torch::Tensor max_nnz,
                                 torch::Tensor output,
                                 int64_t split_k) {
  return spinfer_forward_impl(input, values, global_offsets, median_offsets,
                              bitmap, max_nnz, output.size(1), split_k, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("prepare_cpu", &spinfer_prepare_cpu, "SpInfer BF16 compression");
  module.def("forward", &spinfer_forward, "SpInfer BF16 SpMM");
  module.def("forward_add", &spinfer_forward_add,
             "SpInfer BF16 SpMM accumulated into an existing output");
}
