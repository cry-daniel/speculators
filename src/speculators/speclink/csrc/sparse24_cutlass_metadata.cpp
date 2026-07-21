#include <torch/extension.h>

#include <cstdint>

namespace py = pybind11;

torch::Tensor sparse24_cutlass_reorder_metadata_cuda(
    torch::Tensor metadata,
    int64_t logical_k);

namespace {

torch::Tensor reorder_metadata(torch::Tensor metadata, int64_t logical_k) {
  TORCH_CHECK(metadata.defined() && metadata.is_cuda() && metadata.is_contiguous(),
              "metadata must be a contiguous CUDA tensor");
  TORCH_CHECK(metadata.scalar_type() == torch::kUInt8,
              "metadata must have dtype torch.uint8");
  TORCH_CHECK(metadata.dim() == 2, "metadata must be 2D");
  TORCH_CHECK(logical_k > 0 && logical_k % 32 == 0,
              "logical K must be a positive multiple of 32");
  TORCH_CHECK(metadata.size(0) > 0 && metadata.size(0) % 32 == 0,
              "logical N must be a positive multiple of 32");
  TORCH_CHECK(metadata.size(1) == logical_k / 8,
              "metadata must have shape [N,K/8]");
  return sparse24_cutlass_reorder_metadata_cuda(
      std::move(metadata), logical_k);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reorder_metadata", &reorder_metadata,
        "Reorder packed canonical selectors for CUTLASS SparseGemm",
        py::arg("metadata"), py::arg("logical_k"));
}
