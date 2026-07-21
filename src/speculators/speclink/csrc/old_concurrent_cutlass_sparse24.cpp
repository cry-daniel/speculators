#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

torch::Tensor old_concurrent_branch_forward_out_cuda(
    torch::Tensor x,
    torch::Tensor dense_indices,
    torch::Tensor sparse_indices,
    torch::Tensor dense_weight,
    torch::Tensor reordered_metadata,
    torch::Tensor output,
    int64_t branch,
    int64_t persistent_blocks);

std::vector<int64_t> old_concurrent_kernel_attributes_cuda(int64_t branch);

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, char const* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_inputs(
    torch::Tensor const& x,
    torch::Tensor const& dense_indices,
    torch::Tensor const& sparse_indices,
    torch::Tensor const& dense_weight,
    torch::Tensor const& reordered_metadata,
    torch::Tensor const& output,
    int64_t branch,
    int64_t persistent_blocks) {
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(dense_indices, "dense_indices");
  check_cuda_contiguous(sparse_indices, "sparse_indices");
  check_cuda_contiguous(dense_weight, "dense_weight");
  check_cuda_contiguous(reordered_metadata, "reordered_metadata");
  check_cuda_contiguous(output, "output");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
  TORCH_CHECK(dense_weight.scalar_type() == torch::kBFloat16,
              "dense_weight must be BF16");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
              "output must be BF16");
  TORCH_CHECK(reordered_metadata.scalar_type() == torch::kInt16,
              "reordered_metadata must be int16");
  TORCH_CHECK(dense_indices.scalar_type() == torch::kInt64 &&
                  sparse_indices.scalar_type() == torch::kInt64,
              "route indices must be int64");
  TORCH_CHECK(x.dim() == 2 && dense_weight.dim() == 2 && output.dim() == 2,
              "x, dense_weight, and output must be 2D");
  TORCH_CHECK(dense_indices.dim() == 1 && sparse_indices.dim() == 1,
              "route indices must be 1D");
  TORCH_CHECK(reordered_metadata.dim() == 2, "metadata must be 2D");
  TORCH_CHECK(x.size(0) == 2048, "old concurrent kernel requires M=2048");
  TORCH_CHECK(dense_indices.numel() == 256,
              "old concurrent kernel requires 256 dense rows");
  TORCH_CHECK(sparse_indices.numel() == 1792,
              "old concurrent kernel requires 1792 sparse rows");
  const int64_t n = dense_weight.size(0);
  const int64_t k = dense_weight.size(1);
  TORCH_CHECK(n > 0 && n % 64 == 0 && k > 0 && k % 64 == 0,
              "N and K must be positive multiples of 64");
  TORCH_CHECK(x.size(1) == k, "x and dense_weight must share K");
  TORCH_CHECK(reordered_metadata.sizes() ==
                  torch::IntArrayRef({n, k / 16}),
              "reordered_metadata must have shape [N,K/16]");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef({x.size(0), n}),
              "output must have shape [M,N]");
  TORCH_CHECK(x.device() == dense_indices.device() &&
                  x.device() == sparse_indices.device() &&
                  x.device() == dense_weight.device() &&
                  x.device() == reordered_metadata.device() &&
                  x.device() == output.device(),
              "all tensors must share one CUDA device");
  TORCH_CHECK(branch == 0 || branch == 1,
              "branch must be 0 (dense) or 1 (sparse)");
  TORCH_CHECK(persistent_blocks > 0,
              "persistent_blocks must be positive");
}

torch::Tensor branch_forward_out(
    torch::Tensor x,
    torch::Tensor dense_indices,
    torch::Tensor sparse_indices,
    torch::Tensor dense_weight,
    torch::Tensor reordered_metadata,
    torch::Tensor output,
    int64_t branch,
    int64_t persistent_blocks) {
  check_inputs(x, dense_indices, sparse_indices, dense_weight,
               reordered_metadata, output, branch, persistent_blocks);
  return old_concurrent_branch_forward_out_cuda(
      std::move(x), std::move(dense_indices), std::move(sparse_indices),
      std::move(dense_weight), std::move(reordered_metadata),
      std::move(output), branch, persistent_blocks);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "branch_forward_out", &branch_forward_out,
      "Run one frozen old-concurrent pure-role branch",
      py::arg("x"), py::arg("dense_indices"), py::arg("sparse_indices"),
      py::arg("dense_weight"), py::arg("reordered_metadata"),
      py::arg("output"), py::arg("branch"),
      py::arg("persistent_blocks"));
  m.def(
      "kernel_attributes", &old_concurrent_kernel_attributes_cuda,
      "Return frozen old-concurrent branch kernel attributes",
      py::arg("branch"));
}
