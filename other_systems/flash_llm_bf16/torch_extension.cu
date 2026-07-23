#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

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
  TORCH_CHECK(weight.size(0) % 128 == 0, "Flash-LLM N must be divisible by 128");
  TORCH_CHECK(weight.size(1) % 64 == 0, "Flash-LLM K must be divisible by 64");
}

void check_tokens(int64_t tokens) {
  TORCH_CHECK(tokens == 8 || tokens == 16 || tokens == 32 || tokens == 64 ||
                  tokens == 128 || (tokens > 0 && tokens % 128 == 0),
              "Flash-LLM token rows must be 8/16/32/64/128 or a multiple of 128");
}

}  // namespace

std::vector<torch::Tensor> flash_prepare_cpu(torch::Tensor weight) {
  check_cpu_weight(weight);
  uint32_t* compressed_raw = nullptr;
  int* offsets_raw = nullptr;
  const int n = static_cast<int>(weight.size(0));
  const int k = static_cast<int>(weight.size(1));
  int num_offsets = InitSparseMatrixA_API(
      reinterpret_cast<__nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      n, 0, k, &compressed_raw, &offsets_raw);
  TORCH_CHECK(num_offsets > 0 && compressed_raw != nullptr && offsets_raw != nullptr,
              "Flash-LLM compression failed");
  // Tile offsets count uint4 vectors while the host compressor stores one
  // uint32 record per sparse value (four records per vector).
  const int64_t packed_words =
      static_cast<int64_t>(offsets_raw[num_offsets - 1]) * 4;
  auto compressed =
      torch::empty({packed_words}, torch::TensorOptions().dtype(torch::kInt32));
  auto offsets =
      torch::empty({num_offsets}, torch::TensorOptions().dtype(torch::kInt32));
  std::memcpy(compressed.data_ptr<int32_t>(), compressed_raw,
              packed_words * sizeof(uint32_t));
  std::memcpy(offsets.data_ptr<int32_t>(), offsets_raw,
              num_offsets * sizeof(int));
  std::free(compressed_raw);
  std::free(offsets_raw);
  return {compressed, offsets};
}

torch::Tensor flash_forward(torch::Tensor input,
                            torch::Tensor compressed,
                            torch::Tensor offsets,
                            int64_t n,
                            int64_t split_k) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16,
              "input must be CUDA BF16");
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous(),
              "input must be contiguous [M,K]");
  TORCH_CHECK(compressed.is_cuda() && compressed.scalar_type() == torch::kInt32 &&
                  compressed.is_contiguous(),
              "compressed values must be contiguous CUDA int32");
  TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type() == torch::kInt32 &&
                  offsets.is_contiguous(),
              "offsets must be contiguous CUDA int32");
  TORCH_CHECK(input.device() == compressed.device() &&
                  input.device() == offsets.device(),
              "all tensors must be on one CUDA device");
  TORCH_CHECK(n > 0 && n % 128 == 0, "N must be positive and divisible by 128");
  TORCH_CHECK(input.size(1) % 64 == 0, "K must be divisible by 64");
  TORCH_CHECK(split_k >= 1 && split_k <= input.size(1) / 64,
              "invalid split_k");
  check_tokens(input.size(0));

  c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty({input.size(0), n}, input.options());
  auto workspace =
      split_k == 1
          ? torch::empty({1}, input.options())
          : torch::empty({split_k, input.size(0), n}, input.options());
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  // The upstream ABI retains A although the active Tiled-CSL kernel reads only
  // Compressed_A.  Passing input avoids retaining a second dense weight.
  cudaError_t error = SpMM_SplitK_API(
      stream,
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint4*>(compressed.data_ptr<int32_t>()),
      offsets.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      static_cast<int>(n), static_cast<int>(input.size(0)),
      static_cast<int>(input.size(1)),
      reinterpret_cast<__nv_bfloat16*>(workspace.data_ptr<at::BFloat16>()),
      static_cast<int>(split_k));
  TORCH_CHECK(error == cudaSuccess, "Flash-LLM launch failed: ",
              cudaGetErrorString(error));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("prepare_cpu", &flash_prepare_cpu, "Flash-LLM BF16 compression");
  module.def("forward", &flash_forward, "Flash-LLM BF16 SpMM");
}
