#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

torch::Tensor cusparselt_sparse_residual_complement_sparse_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant);

torch::Tensor cusparselt_sparse_residual_complement_sparse_splitk2_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant);

torch::Tensor cusparselt_sparse_residual_complement_sparse_splitk4_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant);

torch::Tensor
cusparselt_sparse_residual_complement_sparse_splitk2_persistent_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant,
    int64_t persistent_m_blocks);

torch::Tensor
cusparselt_sparse_residual_complement_sparse_splitk2_chunked_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant,
    int64_t chunk_m_blocks);

torch::Tensor
cusparselt_sparse_residual_complement_sparse_splitk2_indexed_forward_cuda(
    torch::Tensor x,
    torch::Tensor indices,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant);

torch::Tensor cusparselt_sparse_residual_fused_base_complement_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant);

std::vector<int64_t>
cusparselt_sparse_residual_complement_sparse_kernel_attributes_cuda(
    int64_t token_rows,
    int64_t output_features,
    int64_t variant);

std::vector<int64_t>
cusparselt_sparse_residual_fused_base_complement_kernel_attributes_cuda(
    int64_t token_rows,
    int64_t output_features,
    int64_t variant);

torch::Tensor cusparselt_sparse_residual_indexed_add_inplace_cuda(
    torch::Tensor base,
    torch::Tensor correction,
    torch::Tensor indices);

torch::Tensor cusparselt_sparse_residual_indexed_copy_inplace_cuda(
    torch::Tensor destination,
    torch::Tensor source,
    torch::Tensor indices);

torch::Tensor cusparselt_sparse_residual_splitk2_indexed_add_inplace_cuda(
    torch::Tensor base,
    torch::Tensor partials,
    torch::Tensor indices);

torch::Tensor cusparselt_sparse_residual_splitk4_indexed_add_inplace_cuda(
    torch::Tensor base,
    torch::Tensor partials,
    torch::Tensor indices);

torch::Tensor cusparselt_sparse_residual_indexed_gather_cuda(
    torch::Tensor source,
    torch::Tensor indices);

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
    torch::Tensor residual,
    int64_t variant) {
  check_inputs(x, cusparselt_packed, residual);
  TORCH_CHECK(
      (variant >= -1 && variant <= 8) || (variant >= 11 && variant <= 19),
              "complement variant must be -1 (auto), 0 (feature128_token64), "
              "1 (feature64_token64), 2 (feature128_token32), or "
              "3 (feature64_token32), 4 (feature64_token64_s2), or "
              "5 (feature64_token32_s2), 6 (feature128_token64_single_smem), "
              "7 (feature256_token64_s3), "
              "8 (activation_stationary_token32), "
              "11 (b_resident_feature128_token32), or "
              "12 (b_resident_feature64_token32), "
              "13 (b_resident_feature128_token32_a2), or "
              "14 (b_resident_feature64_token32_a2), "
              "15 (b_resident_feature64_token64_b2a1_p40), or "
              "16 (feature64_token64_s4_p40), "
              "17 (b_resident_feature128_token64_b2a1), "
              "18 (b_resident_feature128_token64_b2a1_p192), or "
              "19 (b_resident_feature128_token64_b2a1_p224)");
  return cusparselt_sparse_residual_complement_sparse_forward_cuda(
      std::move(x), std::move(cusparselt_packed), std::move(residual), variant);
}

torch::Tensor fused_base_complement_forward(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant) {
  check_inputs(x, cusparselt_packed, residual);
  TORCH_CHECK(variant >= 0 && variant <= 4,
              "fused variant must be in [0,4]");
  return cusparselt_sparse_residual_fused_base_complement_forward_cuda(
      std::move(x), std::move(cusparselt_packed), std::move(residual), variant);
}

torch::Tensor complement_sparse_splitk2_forward(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant) {
  check_inputs(x, cusparselt_packed, residual);
  TORCH_CHECK(variant >= 0 && variant <= 18,
              "Split-K=2 variant must be 0 (B2/A1 F64), 1 (S4 F64), "
              "2 (S4 F128), 3 (B2/A1 F64 P40), 4 (B2/A1 F128), "
              "5 (B2/A1 F128 P192), 6 (B2/A1 F128 P224), or a "
              "Token128 gate_up variant in [7,18]");
  if (variant == 3) {
    TORCH_CHECK(cusparselt_packed.size(0) == 5120,
                "Split-K=2 P40 complement requires N=5120");
  }
  if (variant == 5) {
    TORCH_CHECK(cusparselt_packed.size(0) == 24576,
                "Split-K=2 P192 complement requires N=24576");
  }
  if (variant == 6) {
    TORCH_CHECK(cusparselt_packed.size(0) == 28672,
                "Split-K=2 P224 complement requires N=28672");
  }
  if (variant == 8) {
    TORCH_CHECK(cusparselt_packed.size(0) == 24576,
                "Token128 P192 complement requires N=24576");
  }
  if (variant == 9) {
    TORCH_CHECK(cusparselt_packed.size(0) == 28672,
                "Token128 P224 complement requires N=28672");
  }
  if (variant == 11) {
    TORCH_CHECK(cusparselt_packed.size(0) == 24576,
                "Feature256 P192 complement requires N=24576");
  }
  if (variant == 12) {
    TORCH_CHECK(cusparselt_packed.size(0) == 28672,
                "Feature256 P224 complement requires N=28672");
  }
  if (variant == 14 || variant == 17) {
    TORCH_CHECK(cusparselt_packed.size(0) == 24576,
                "A2 P192 complement requires N=24576");
  }
  if (variant == 15 || variant == 18) {
    TORCH_CHECK(cusparselt_packed.size(0) == 28672,
                "A2 P224 complement requires N=28672");
  }
  return cusparselt_sparse_residual_complement_sparse_splitk2_forward_cuda(
      std::move(x), std::move(cusparselt_packed), std::move(residual), variant);
}

torch::Tensor complement_sparse_splitk4_forward(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant) {
  check_inputs(x, cusparselt_packed, residual);
  TORCH_CHECK(variant >= 7 && variant <= 12,
              "Split-K=4 requires a Token128 gate_up variant in [7,12]");
  if (variant == 8 || variant == 11) {
    TORCH_CHECK(cusparselt_packed.size(0) == 24576,
                "Split-K=4 P192 complement requires N=24576");
  }
  if (variant == 9 || variant == 12) {
    TORCH_CHECK(cusparselt_packed.size(0) == 28672,
                "Split-K=4 P224 complement requires N=28672");
  }
  return cusparselt_sparse_residual_complement_sparse_splitk4_forward_cuda(
      std::move(x), std::move(cusparselt_packed), std::move(residual), variant);
}

torch::Tensor complement_sparse_splitk2_persistent_forward(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant,
    int64_t persistent_m_blocks) {
  check_inputs(x, cusparselt_packed, residual);
  TORCH_CHECK(variant >= 7 && variant <= 9,
              "persistent Split-K2 requires variant 7, 8, or 9");
  TORCH_CHECK(persistent_m_blocks > 0,
              "persistent_m_blocks must be positive");
  if (variant == 8) {
    TORCH_CHECK(cusparselt_packed.size(0) == 24576,
                "persistent P192 complement requires N=24576");
  }
  if (variant == 9) {
    TORCH_CHECK(cusparselt_packed.size(0) == 28672,
                "persistent P224 complement requires N=28672");
  }
  return
      cusparselt_sparse_residual_complement_sparse_splitk2_persistent_forward_cuda(
          std::move(x), std::move(cusparselt_packed), std::move(residual),
          variant, persistent_m_blocks);
}

torch::Tensor complement_sparse_splitk2_chunked_forward(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant,
    int64_t chunk_m_blocks) {
  check_inputs(x, cusparselt_packed, residual);
  TORCH_CHECK(variant >= 7 && variant <= 9,
              "chunked Split-K2 requires variant 7, 8, or 9");
  TORCH_CHECK(chunk_m_blocks > 0,
              "chunk_m_blocks must be positive");
  if (variant == 8) {
    TORCH_CHECK(cusparselt_packed.size(0) == 24576,
                "chunked P192 complement requires N=24576");
  }
  if (variant == 9) {
    TORCH_CHECK(cusparselt_packed.size(0) == 28672,
                "chunked P224 complement requires N=28672");
  }
  return
      cusparselt_sparse_residual_complement_sparse_splitk2_chunked_forward_cuda(
          std::move(x), std::move(cusparselt_packed), std::move(residual),
          variant, chunk_m_blocks);
}

torch::Tensor complement_sparse_splitk2_indexed_forward(
    torch::Tensor x,
    torch::Tensor indices,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant) {
  check_inputs(x, cusparselt_packed, residual);
  check_cuda_contiguous(indices, "indices");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32,
              "indexed Split-K=2 indices must be int32");
  TORCH_CHECK(indices.dim() == 1 && indices.numel() > 0,
              "indices must be non-empty 1D");
  TORCH_CHECK(indices.device() == x.device(),
              "indices and x must share one CUDA device");
  TORCH_CHECK(variant >= 0 && variant <= 6,
              "indexed Split-K=2 variant must be in [0,6]");
  if (variant == 3) {
    TORCH_CHECK(cusparselt_packed.size(0) == 5120,
                "Split-K=2 P40 complement requires N=5120");
  }
  if (variant == 5) {
    TORCH_CHECK(cusparselt_packed.size(0) == 24576,
                "Split-K=2 P192 complement requires N=24576");
  }
  if (variant == 6) {
    TORCH_CHECK(cusparselt_packed.size(0) == 28672,
                "Split-K=2 P224 complement requires N=28672");
  }
  return cusparselt_sparse_residual_complement_sparse_splitk2_indexed_forward_cuda(
      std::move(x), std::move(indices), std::move(cusparselt_packed),
      std::move(residual), variant);
}

torch::Tensor indexed_add_inplace(
    torch::Tensor base,
    torch::Tensor correction,
    torch::Tensor indices) {
  check_cuda_contiguous(base, "base");
  check_cuda_contiguous(correction, "correction");
  check_cuda_contiguous(indices, "indices");
  TORCH_CHECK(base.scalar_type() == torch::kBFloat16,
              "base must be BF16");
  TORCH_CHECK(correction.scalar_type() == torch::kBFloat16,
              "correction must be BF16");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32 ||
                  indices.scalar_type() == torch::kInt64,
              "indices must be int32 or int64");
  TORCH_CHECK(base.dim() == 2 && correction.dim() == 2,
              "base and correction must be 2D");
  TORCH_CHECK(indices.dim() == 1,
              "indices must be 1D");
  TORCH_CHECK(indices.numel() > 0,
              "indices must be non-empty");
  TORCH_CHECK(correction.size(0) == indices.size(0) &&
                  correction.size(1) == base.size(1),
              "correction must have shape [indices.numel(), base.size(1)]");
  TORCH_CHECK(base.size(1) % 8 == 0,
              "128-bit indexed add requires columns divisible by 8");
  TORCH_CHECK(base.device() == correction.device() &&
                  base.device() == indices.device(),
              "base, correction, and indices must share one CUDA device");
  return cusparselt_sparse_residual_indexed_add_inplace_cuda(
      std::move(base), std::move(correction), std::move(indices));
}

torch::Tensor indexed_copy_inplace(
    torch::Tensor destination,
    torch::Tensor source,
    torch::Tensor indices) {
  check_cuda_contiguous(destination, "destination");
  check_cuda_contiguous(source, "source");
  check_cuda_contiguous(indices, "indices");
  TORCH_CHECK(destination.scalar_type() == torch::kBFloat16 &&
                  source.scalar_type() == torch::kBFloat16,
              "destination and source must be BF16");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32 ||
                  indices.scalar_type() == torch::kInt64,
              "indices must be int32 or int64");
  TORCH_CHECK(destination.dim() == 2 && source.dim() == 2,
              "destination and source must be 2D");
  TORCH_CHECK(indices.dim() == 1 && indices.numel() > 0,
              "indices must be non-empty 1D");
  TORCH_CHECK(source.size(0) == indices.size(0) &&
                  source.size(1) == destination.size(1),
              "source must have shape [indices.numel(), destination.size(1)]");
  TORCH_CHECK(destination.size(1) % 8 == 0,
              "128-bit indexed copy requires columns divisible by 8");
  TORCH_CHECK(destination.device() == source.device() &&
                  destination.device() == indices.device(),
              "destination, source, and indices must share one CUDA device");
  return cusparselt_sparse_residual_indexed_copy_inplace_cuda(
      std::move(destination), std::move(source), std::move(indices));
}

torch::Tensor splitk2_indexed_add_inplace(
    torch::Tensor base,
    torch::Tensor partials,
    torch::Tensor indices) {
  check_cuda_contiguous(base, "base");
  check_cuda_contiguous(partials, "partials");
  check_cuda_contiguous(indices, "indices");
  TORCH_CHECK(base.scalar_type() == torch::kBFloat16,
              "base must be BF16");
  TORCH_CHECK(partials.scalar_type() == torch::kBFloat16,
              "partials must be BF16");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32 ||
                  indices.scalar_type() == torch::kInt64,
              "indices must be int32 or int64");
  TORCH_CHECK(base.dim() == 2 && partials.dim() == 3,
              "base must be 2D and partials must be 3D");
  TORCH_CHECK(partials.size(0) == 2,
              "partials must have exactly two split-K slices");
  TORCH_CHECK(indices.dim() == 1 && indices.numel() > 0,
              "indices must be non-empty 1D");
  TORCH_CHECK(partials.size(1) == indices.size(0) &&
                  partials.size(2) == base.size(1),
              "partials must have shape [2, indices.numel(), base.size(1)]");
  TORCH_CHECK(base.size(1) % 8 == 0,
              "128-bit indexed reduction requires columns divisible by 8");
  TORCH_CHECK(base.device() == partials.device() &&
                  base.device() == indices.device(),
              "base, partials, and indices must share one CUDA device");
  return cusparselt_sparse_residual_splitk2_indexed_add_inplace_cuda(
      std::move(base), std::move(partials), std::move(indices));
}

torch::Tensor splitk4_indexed_add_inplace(
    torch::Tensor base,
    torch::Tensor partials,
    torch::Tensor indices) {
  check_cuda_contiguous(base, "base");
  check_cuda_contiguous(partials, "partials");
  check_cuda_contiguous(indices, "indices");
  TORCH_CHECK(base.scalar_type() == torch::kBFloat16,
              "base must be BF16");
  TORCH_CHECK(partials.scalar_type() == torch::kBFloat16,
              "partials must be BF16");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32 ||
                  indices.scalar_type() == torch::kInt64,
              "indices must be int32 or int64");
  TORCH_CHECK(base.dim() == 2 && partials.dim() == 3,
              "base must be 2D and partials must be 3D");
  TORCH_CHECK(partials.size(0) == 4,
              "partials must have exactly four split-K slices");
  TORCH_CHECK(indices.dim() == 1 && indices.numel() > 0,
              "indices must be non-empty 1D");
  TORCH_CHECK(partials.size(1) == indices.size(0) &&
                  partials.size(2) == base.size(1),
              "partials must have shape [4, indices.numel(), base.size(1)]");
  TORCH_CHECK(base.size(1) % 8 == 0,
              "128-bit indexed reduction requires columns divisible by 8");
  TORCH_CHECK(base.device() == partials.device() &&
                  base.device() == indices.device(),
              "base, partials, and indices must share one CUDA device");
  return cusparselt_sparse_residual_splitk4_indexed_add_inplace_cuda(
      std::move(base), std::move(partials), std::move(indices));
}

torch::Tensor indexed_gather(
    torch::Tensor source,
    torch::Tensor indices) {
  check_cuda_contiguous(source, "source");
  check_cuda_contiguous(indices, "indices");
  TORCH_CHECK(source.scalar_type() == torch::kBFloat16,
              "source must be BF16");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32 ||
                  indices.scalar_type() == torch::kInt64,
              "indices must be int32 or int64");
  TORCH_CHECK(source.dim() == 2,
              "source must be 2D");
  TORCH_CHECK(indices.dim() == 1 && indices.numel() > 0,
              "indices must be non-empty 1D");
  TORCH_CHECK(source.size(1) % 8 == 0,
              "128-bit indexed gather requires columns divisible by 8");
  TORCH_CHECK(source.device() == indices.device(),
              "source and indices must share one CUDA device");
  return cusparselt_sparse_residual_indexed_gather_cuda(
      std::move(source), std::move(indices));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "cusparselt_complement_sparse_forward", &complement_sparse_forward,
      "HMMA.SP complement GEMM from cuSPARSELt metadata",
      py::arg("x"), py::arg("cusparselt_packed"), py::arg("residual"),
      py::arg("variant") = -1);
  m.def(
      "cusparselt_complement_sparse_splitk2_forward",
      &complement_sparse_splitk2_forward,
      "Split-K=2 HMMA.SP complement GEMM from cuSPARSELt metadata",
      py::arg("x"), py::arg("cusparselt_packed"), py::arg("residual"),
      py::arg("variant") = 0);
  m.def(
      "cusparselt_complement_sparse_splitk4_forward",
      &complement_sparse_splitk4_forward,
      "Split-K=4 HMMA.SP complement GEMM from cuSPARSELt metadata",
      py::arg("x"), py::arg("cusparselt_packed"), py::arg("residual"),
      py::arg("variant") = 7);
  m.def(
      "cusparselt_complement_sparse_splitk2_persistent_forward",
      &complement_sparse_splitk2_persistent_forward,
      "Quota-limited persistent Split-K=2 gate_up complement GEMM",
      py::arg("x"), py::arg("cusparselt_packed"), py::arg("residual"),
      py::arg("variant"), py::arg("persistent_m_blocks"));
  m.def(
      "cusparselt_complement_sparse_splitk2_chunked_forward",
      &complement_sparse_splitk2_chunked_forward,
      "Wave-limited chunked Split-K=2 gate_up complement GEMM",
      py::arg("x"), py::arg("cusparselt_packed"), py::arg("residual"),
      py::arg("variant"), py::arg("chunk_m_blocks"));
  m.def(
      "cusparselt_complement_sparse_splitk2_indexed_forward",
      &complement_sparse_splitk2_indexed_forward,
      "Split-K=2 HMMA.SP complement GEMM with mainloop indexed activation",
      py::arg("x"), py::arg("indices"), py::arg("cusparselt_packed"),
      py::arg("residual"), py::arg("variant") = 0);
  m.def(
      "cusparselt_complement_sparse_kernel_attributes",
      &cusparselt_sparse_residual_complement_sparse_kernel_attributes_cuda,
      "Return complement HMMA.SP kernel attributes",
      py::arg("token_rows"), py::arg("output_features"),
      py::arg("variant") = -1);
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
  m.def(
      "cusparselt_indexed_add_inplace", &indexed_add_inplace,
      "Add compact correction rows into base output in place",
      py::arg("base"), py::arg("correction"), py::arg("indices"));
  m.def(
      "cusparselt_indexed_copy_inplace", &indexed_copy_inplace,
      "Copy compact BF16 rows into indexed destination rows",
      py::arg("destination"), py::arg("source"), py::arg("indices"));
  m.def(
      "cusparselt_splitk2_indexed_add_inplace",
      &splitk2_indexed_add_inplace,
      "Reduce two compact BF16 partials and add indexed rows in place",
      py::arg("base"), py::arg("partials"), py::arg("indices"));
  m.def(
      "cusparselt_splitk4_indexed_add_inplace",
      &splitk4_indexed_add_inplace,
      "Reduce four compact BF16 partials and add indexed rows in place",
      py::arg("base"), py::arg("partials"), py::arg("indices"));
  m.def(
      "cusparselt_indexed_gather", &indexed_gather,
      "Gather BF16 rows with aligned 128-bit loads and stores",
      py::arg("source"), py::arg("indices"));
}
