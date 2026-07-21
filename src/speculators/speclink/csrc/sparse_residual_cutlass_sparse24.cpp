#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

torch::Tensor cusparselt_sparse_residual_complement_sparse_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual);

torch::Tensor cusparselt_sparse_residual_fused_base_complement_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant);

std::vector<int64_t>
cusparselt_sparse_residual_complement_sparse_kernel_attributes_cuda(
    int64_t token_rows,
    int64_t output_features);

std::vector<int64_t>
cusparselt_sparse_residual_fused_base_complement_kernel_attributes_cuda(
    int64_t token_rows,
    int64_t output_features,
    int64_t variant);

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, char const* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_inputs(
    torch::Tensor const& x,
    torch::Tensor const& cusparselt_packed,
    torch::Tensor const& residual) {
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(cusparselt_packed, "cusparselt_packed");
  check_cuda_contiguous(residual, "residual");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
  TORCH_CHECK(cusparselt_packed.scalar_type() == torch::kBFloat16,
              "cusparselt_packed must be BF16");
  TORCH_CHECK(residual.scalar_type() == torch::kBFloat16,
              "residual must be BF16");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0, "x must have shape [M,K], M>0");
  TORCH_CHECK(cusparselt_packed.dim() == 2,
              "cusparselt_packed must be 2D");
  TORCH_CHECK(residual.dim() == 2, "residual must be 2D");
  int64_t const n = cusparselt_packed.size(0);
  int64_t const k = x.size(1);
  TORCH_CHECK(n > 0 && n % 128 == 0,
              "direct cuSPARSELt metadata decoding requires N % 128 == 0");
  TORCH_CHECK(k > 0 && k % 128 == 0,
              "direct cuSPARSELt metadata decoding requires K % 128 == 0");
  TORCH_CHECK(cusparselt_packed.numel() == int64_t(9) * n * k / 16,
              "packed allocation must contain values followed by metadata");
  TORCH_CHECK(residual.size(0) == n && residual.size(1) == k / 2,
              "residual must have shape [N,K/2]");
  TORCH_CHECK(x.device() == cusparselt_packed.device() &&
                  x.device() == residual.device(),
              "all tensors must share one CUDA device");
}

torch::Tensor complement_sparse_forward(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual) {
  check_inputs(x, cusparselt_packed, residual);
  return cusparselt_sparse_residual_complement_sparse_forward_cuda(
      std::move(x), std::move(cusparselt_packed), std::move(residual));
}

torch::Tensor fused_base_complement_forward(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant) {
  check_inputs(x, cusparselt_packed, residual);
  TORCH_CHECK(x.size(0) % 32 == 0,
              "fused path requires token rows divisible by 32");
  TORCH_CHECK(variant >= 0 && variant <= 2,
              "variant must be 0 (N64S3), 1 (N32S4), or 2 (N128S3)");
  return cusparselt_sparse_residual_fused_base_complement_forward_cuda(
      std::move(x), std::move(cusparselt_packed), std::move(residual), variant);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "cusparselt_complement_sparse_forward", &complement_sparse_forward,
      "HMMA.SP complement GEMM from cuSPARSELt metadata",
      py::arg("x"), py::arg("cusparselt_packed"), py::arg("residual"));
  m.def(
      "cusparselt_complement_sparse_kernel_attributes",
      &cusparselt_sparse_residual_complement_sparse_kernel_attributes_cuda,
      "Return complement HMMA.SP kernel attributes",
      py::arg("token_rows"), py::arg("output_features"));
  m.def(
      "cusparselt_fused_base_complement_forward",
      &fused_base_complement_forward,
      "Fused base+complement HMMA.SP with shared B/E and one epilogue",
      py::arg("x"), py::arg("cusparselt_packed"), py::arg("residual"),
      py::arg("variant") = 0);
  m.def(
      "cusparselt_fused_base_complement_kernel_attributes",
      &cusparselt_sparse_residual_fused_base_complement_kernel_attributes_cuda,
      "Return fused base+complement HMMA.SP kernel attributes",
      py::arg("token_rows"), py::arg("output_features"),
      py::arg("variant") = 0);
}
