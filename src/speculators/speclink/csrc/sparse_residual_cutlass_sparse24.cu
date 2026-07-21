#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <type_traits>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_sparse_mma.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/gemm/threadblock/mma_sparse_multistage.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/sparse_gemm_with_visitor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

#include "cutlass_dual_sparse_gemm_with_visitor.h"
#include "cutlass_dual_sparse_mma_multistage.h"
#include "cutlass_transpose_epilogue_visitor.h"

namespace {

using Bf16 = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementE = uint16_t;
using GmemLayoutE = cutlass::layout::ColumnMajorInterleaved<2>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using SparseInstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int kEpilogueStages = 1;

// cuSPARSELt and CUTLASS use the same four-selector hardware encoding.  The
// compact residual contains the two positions not retained by the base 2:4
// weight, so its selector is the exact complement of the base selector.
CUTLASS_DEVICE uint16_t complement_selector_word(uint16_t word) {
  uint16_t result = 0;
  CUTLASS_PRAGMA_UNROLL
  for (int index = 0; index < 4; ++index) {
    uint16_t const selector = (word >> (4 * index)) & 0xfu;
    // Four selectors are simply xor 5.  The two edge-adjacent selectors use
    // xor A, which is xor 5 followed by xor F.  This branchless expression is
    // substantially smaller than a six-way switch in every warp-MMA call.
    uint16_t const edge =
        (selector == 0x4 || selector == 0xe) ? 0xfu : 0u;
    uint16_t const complement = selector ^ 0x5u ^ edge;
    result |= complement << (4 * index);
  }
  return result;
}

// Keep CUTLASS's sparse warp iterator and HMMA.SP schedule intact, but replace
// the E fragment immediately before the instruction with its complement.  The
// conversion is register-only and therefore needs no residual metadata tensor.
template <typename BaseWarpMma_>
struct ComplementSparseWarpMma : BaseWarpMma_ {
  using Base = BaseWarpMma_;
  using FragmentC = typename Base::FragmentC;
  using TransformedFragmentA = typename Base::TransformedFragmentA;
  using TransformedFragmentB = typename Base::TransformedFragmentB;
  using FragmentE = typename Base::FragmentE;

  CUTLASS_DEVICE
  void operator()(
      FragmentC& destination,
      TransformedFragmentA const& operand_a,
      TransformedFragmentB const& operand_b,
      FragmentC const& source,
      FragmentE const& metadata) const {
    FragmentE complement = metadata;
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < FragmentE::kElements; ++index) {
      complement[index] = complement_selector_word(
          static_cast<uint16_t>(complement[index]));
    }
    Base::operator()(
        destination, operand_a, operand_b, source, complement);
  }
};

// CUTLASS addresses E in its normal ColumnMajorInterleaved<2> order.
// cuSPARSELt 0.8 appends those same words after its values but macro-swizzles
// 256-word blocks.  This iterator delegates all predicate/tile bookkeeping to
// CUTLASS and changes only the returned global address.  An aligned 128-bit E
// access is eight uint16 words and never crosses a 256-word block boundary.
template <typename StockIteratorE_>
class CusparseLtMetadataIteratorE {
 public:
  using StockIterator = StockIteratorE_;
  using Element = typename StockIterator::Element;
  using Layout = typename StockIterator::Layout;
  using ThreadMap = typename StockIterator::ThreadMap;
  using AccessType = typename StockIterator::AccessType;
  using TensorRef = typename StockIterator::TensorRef;
  using TensorCoord = typename StockIterator::TensorCoord;
  using Params = typename StockIterator::Params;
  static int const kAccessesPerVector = StockIterator::kAccessesPerVector;

 private:
  StockIterator iterator_;
  Element* base_ = nullptr;
  int panels_ = 0;

 public:
  CUTLASS_DEVICE
  CusparseLtMetadataIteratorE(
      Params const& params,
      Element* pointer,
      TensorCoord extent,
      int thread_id,
      TensorCoord threadblock_offset)
      : iterator_(params, pointer, extent, thread_id, threadblock_offset),
        base_(pointer),
        panels_(extent.row() / 128) {}

  CUTLASS_HOST_DEVICE void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);
  }

  CUTLASS_DEVICE void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset(tile_offset);
  }

  CUTLASS_HOST_DEVICE void clear_mask(bool clear = true) {
    iterator_.clear_mask(clear);
  }

  CUTLASS_HOST_DEVICE bool valid() { return iterator_.valid(); }

  CUTLASS_HOST_DEVICE CusparseLtMetadataIteratorE& operator++() {
    ++iterator_;
    return *this;
  }

  CUTLASS_DEVICE AccessType* get() const {
    Element* logical = reinterpret_cast<Element*>(iterator_.get());
    int64_t const logical_offset = logical - base_;
    int64_t const cutlass_block = logical_offset / 256;
    int64_t const word_in_block = logical_offset % 256;
    int64_t const group_blocks = int64_t(2) * panels_;
    int64_t const group_base =
        cutlass_block / group_blocks * group_blocks;
    int64_t const within_group = cutlass_block - group_base;
    int64_t const cusparselt_block = group_base +
        int64_t(2) * (within_group % panels_) + within_group / panels_;
    return reinterpret_cast<AccessType*>(
        base_ + cusparselt_block * 256 + word_in_block);
  }
};

// Residual-only sparse GEMM.  A is already stored as the two compact residual
// BF16 values per K4, E is read from the sole cuSPARSELt allocation through
// CusparseLtMetadataIteratorE, and ComplementSparseWarpMma flips E in registers
// immediately before HMMA.SP.  No dense reconstruction or metadata workspace
// exists on this path.
template <int ThreadblockN, int WarpN, int MainloopStages>
struct ComplementSparseConfiguration {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, ThreadblockN, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, WarpN, 64>;
  using DefaultMma = cutlass::gemm::threadblock::DefaultSparseMma<
      Bf16,
      LayoutA,
      8,
      Bf16,
      LayoutB,
      8,
      float,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      SparseInstructionShape,
      MainloopStages,
      cutlass::arch::OpMultiplyAdd>;
  using MmaCore = typename DefaultMma::MmaCore;
  using IteratorA = typename DefaultMma::IteratorA;
  using IteratorB = typename DefaultMma::IteratorB;
  using IteratorE = CusparseLtMetadataIteratorE<
      typename DefaultMma::IteratorE>;
  using ComplementWarpMma = ::ComplementSparseWarpMma<
      typename MmaCore::MmaPolicy::Operator>;
  using MmaPolicy = cutlass::gemm::threadblock::SparseMmaPolicy<
      ComplementWarpMma,
      typename MmaCore::MmaPolicy::SmemPaddingA,
      typename MmaCore::MmaPolicy::SmemPaddingB,
      typename MmaCore::MmaPolicy::SmemPaddingE,
      MmaCore::MmaPolicy::kPartitionsK>;
  using ThreadblockMma = cutlass::gemm::threadblock::SparseMmaMultistage<
      typename MmaCore::Shape,
      IteratorA,
      typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA,
      IteratorB,
      typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB,
      float,
      cutlass::layout::RowMajor,
      IteratorE,
      typename MmaCore::SmemIteratorE,
      MmaCore::kCacheOpE,
      MmaPolicy,
      MainloopStages>;
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      Bf16, 8, float, float>;
  using BaseEpilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape,
          typename ThreadblockMma::Operator,
          1,
          OutputOp,
          OutputOp::kCount>::Epilogue;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape,
          WarpShape,
          Bf16,
          8,
          kEpilogueStages>;
  using OutputStore = speculators::speclink::VisitorTransposeAuxStore<
      OutputThreadMap,
      Bf16,
      cutlass::FloatRoundStyle::round_to_nearest,
      ThreadblockShape::kM,
      ThreadblockShape::kN,
      WarpShape::kM>;
  using FusionCallbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore,
      cutlass::epilogue::threadblock::VisitorAccFetch>;
  using Epilogue =
      cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<
          BaseEpilogue,
          FusionCallbacks,
          kEpilogueStages>;
  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using Kernel = cutlass::gemm::kernel::SparseGemmWithEpilogueVisitor<
      ThreadblockMma, Epilogue, Swizzle>;
};

// Dense-routed rows need both complementary 2:4 products.  This mainloop
// stages base A and complement A separately, but stages activation B and the
// sole cuSPARSELt metadata E only once.  Both warp MMAs update one FP32
// accumulator fragment and therefore share one epilogue/output write.
template <int ThreadblockN, int WarpN, int MainloopStages>
struct FusedBaseComplementSparseConfiguration {
  using ThreadblockShape = cutlass::gemm::GemmShape<128, ThreadblockN, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, WarpN, 64>;
  using DefaultMma = cutlass::gemm::threadblock::DefaultSparseMma<
      Bf16,
      LayoutA,
      8,
      Bf16,
      LayoutB,
      8,
      float,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      SparseInstructionShape,
      MainloopStages,
      cutlass::arch::OpMultiplyAdd>;
  using MmaCore = typename DefaultMma::MmaCore;
  using IteratorA = typename DefaultMma::IteratorA;
  using IteratorB = typename DefaultMma::IteratorB;
  using IteratorE = CusparseLtMetadataIteratorE<
      typename DefaultMma::IteratorE>;
  using ComplementWarpMma = ::ComplementSparseWarpMma<
      typename MmaCore::MmaPolicy::Operator>;
  using ThreadblockMma =
      cutlass::gemm::threadblock::DualSparseMmaMultistage<
          typename MmaCore::Shape,
          IteratorA,
          typename MmaCore::SmemIteratorA,
          MmaCore::kCacheOpA,
          IteratorB,
          typename MmaCore::SmemIteratorB,
          MmaCore::kCacheOpB,
          float,
          cutlass::layout::RowMajor,
          IteratorE,
          typename MmaCore::SmemIteratorE,
          MmaCore::kCacheOpE,
          typename MmaCore::MmaPolicy,
          ComplementWarpMma,
          MainloopStages>;
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      Bf16, 8, float, float>;
  using BaseEpilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape,
          typename ThreadblockMma::Operator,
          1,
          OutputOp,
          OutputOp::kCount>::Epilogue;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape,
          WarpShape,
          Bf16,
          8,
          kEpilogueStages>;
  using OutputStore = speculators::speclink::VisitorTransposeAuxStore<
      OutputThreadMap,
      Bf16,
      cutlass::FloatRoundStyle::round_to_nearest,
      ThreadblockShape::kM,
      ThreadblockShape::kN,
      WarpShape::kM>;
  using FusionCallbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore,
      cutlass::epilogue::threadblock::VisitorAccFetch>;
  using Epilogue =
      cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<
          BaseEpilogue,
          FusionCallbacks,
          kEpilogueStages>;
  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using Kernel = cutlass::gemm::kernel::DualSparseGemmWithEpilogueVisitor<
      ThreadblockMma, Epilogue, Swizzle>;
};
using NarrowCusparseLtComplementSparse =
    ComplementSparseConfiguration<64, 32, 4>;
using WideCusparseLtComplementSparse =
    ComplementSparseConfiguration<128, 64, 3>;
using FusedBaseComplementN64S3 =
    FusedBaseComplementSparseConfiguration<64, 32, 3>;
using FusedBaseComplementN32S4 =
    FusedBaseComplementSparseConfiguration<32, 32, 4>;
using FusedBaseComplementN128S3 =
    FusedBaseComplementSparseConfiguration<128, 64, 3>;
// Sparse M=256 needs the 64-column tile to expose enough CTAs.  Reusing the
// dense-reconstruction large-N heuristic would select a 128-column sparse tile
// with 254 registers/thread on this build and cut parallelism in half.
bool use_wide_sparse_configuration(int token_rows) {
  return token_rows > 256;
}

template <typename Config>
void set_dynamic_smem_attribute() {
  using Kernel = typename Config::Kernel;
  int const shared_bytes = int(sizeof(typename Kernel::SharedStorage));
  static std::once_flag once;
  std::call_once(once, [shared_bytes]() {
    if (shared_bytes >= (48 << 10)) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          cutlass::Kernel<Kernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          shared_bytes));
    }
  });
}
template <typename Config>
void launch_cusparselt_complement_sparse(
    torch::Tensor const& x,
    torch::Tensor const& cusparselt_packed,
    torch::Tensor const& residual,
    torch::Tensor& output,
    cudaStream_t stream) {
  using Kernel = typename Config::Kernel;
  using IteratorA = typename Config::IteratorA;
  using IteratorB = typename Config::IteratorB;
  using IteratorE = typename Config::IteratorE;
  using OutputStore = typename Config::OutputStore;
  using FusionCallbacks = typename Config::FusionCallbacks;
  using Swizzle = typename Config::Swizzle;
  int const n = int(cusparselt_packed.size(0));
  int const k = int(x.size(1));
  int const m = int(x.size(0));
  cutlass::gemm::GemmCoord const problem{n, m, k};
  Swizzle swizzle;
  cutlass::gemm::GemmCoord const tiled = swizzle.get_tiled_shape(
      problem,
      {Config::ThreadblockShape::kM,
       Config::ThreadblockShape::kN,
       Config::ThreadblockShape::kK},
      1);
  Bf16* residual_ptr = reinterpret_cast<Bf16*>(residual.data_ptr());
  Bf16* x_ptr = reinterpret_cast<Bf16*>(x.data_ptr());
  Bf16* packed_base =
      reinterpret_cast<Bf16*>(cusparselt_packed.data_ptr());
  ElementE* metadata = reinterpret_cast<ElementE*>(
      packed_base + int64_t(n) * k / 2);
  typename IteratorA::TensorRef ref_a(residual_ptr, LayoutA(k / 2));
  typename IteratorB::TensorRef ref_b(x_ptr, LayoutB(k));
  typename IteratorE::TensorRef ref_e(
      metadata, GmemLayoutE::packed({n, k / 16}));
  typename OutputStore::Arguments output_arguments{
      reinterpret_cast<Bf16*>(output.data_ptr())};
  typename FusionCallbacks::Arguments callbacks{{}, output_arguments};
  typename Kernel::Params params(
      problem, tiled, ref_a, ref_b, ref_e, callbacks);
  set_dynamic_smem_attribute<Config>();
  dim3 const grid = swizzle.get_grid_shape(tiled);
  dim3 const block(Kernel::kThreadCount, 1, 1);
  int const shared_bytes = int(sizeof(typename Kernel::SharedStorage));
  cutlass::Kernel<Kernel><<<grid, block, shared_bytes, stream>>>(params);
}

template <typename Config>
void launch_cusparselt_base_complement_fused(
    torch::Tensor const& x,
    torch::Tensor const& cusparselt_packed,
    torch::Tensor const& residual,
    torch::Tensor& output,
    cudaStream_t stream) {
  using Kernel = typename Config::Kernel;
  using IteratorA = typename Config::IteratorA;
  using IteratorB = typename Config::IteratorB;
  using IteratorE = typename Config::IteratorE;
  using OutputStore = typename Config::OutputStore;
  using FusionCallbacks = typename Config::FusionCallbacks;
  using Swizzle = typename Config::Swizzle;
  int const n = int(cusparselt_packed.size(0));
  int const k = int(x.size(1));
  int const m = int(x.size(0));
  cutlass::gemm::GemmCoord const problem{n, m, k};
  Swizzle swizzle;
  cutlass::gemm::GemmCoord const tiled = swizzle.get_tiled_shape(
      problem,
      {Config::ThreadblockShape::kM,
       Config::ThreadblockShape::kN,
       Config::ThreadblockShape::kK},
      1);
  Bf16* packed_base =
      reinterpret_cast<Bf16*>(cusparselt_packed.data_ptr());
  Bf16* residual_ptr = reinterpret_cast<Bf16*>(residual.data_ptr());
  Bf16* x_ptr = reinterpret_cast<Bf16*>(x.data_ptr());
  ElementE* metadata = reinterpret_cast<ElementE*>(
      packed_base + int64_t(n) * k / 2);
  typename IteratorA::TensorRef ref_a_base(packed_base, LayoutA(k / 2));
  typename IteratorA::TensorRef ref_a_complement(
      residual_ptr, LayoutA(k / 2));
  typename IteratorB::TensorRef ref_b(x_ptr, LayoutB(k));
  typename IteratorE::TensorRef ref_e(
      metadata, GmemLayoutE::packed({n, k / 16}));
  typename OutputStore::Arguments output_arguments{
      reinterpret_cast<Bf16*>(output.data_ptr())};
  typename FusionCallbacks::Arguments callbacks{{}, output_arguments};
  typename Kernel::Params params(
      problem,
      tiled,
      ref_a_base,
      ref_a_complement,
      ref_b,
      ref_e,
      callbacks);
  set_dynamic_smem_attribute<Config>();
  dim3 const grid = swizzle.get_grid_shape(tiled);
  dim3 const block(Kernel::kThreadCount, 1, 1);
  int const shared_bytes = int(sizeof(typename Kernel::SharedStorage));
  cutlass::Kernel<Kernel><<<grid, block, shared_bytes, stream>>>(params);
}
template <typename Config>
std::vector<int64_t> configuration_attributes() {
  using Kernel = typename Config::Kernel;
  set_dynamic_smem_attribute<Config>();
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(
      &attributes,
      reinterpret_cast<void const*>(cutlass::Kernel<Kernel>)));
  int const shared_bytes = int(sizeof(typename Kernel::SharedStorage));
  int active_blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm,
      cutlass::Kernel<Kernel>,
      Kernel::kThreadCount,
      shared_bytes));
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  int64_t const occupancy_pct = properties.maxThreadsPerMultiProcessor > 0
      ? (int64_t(active_blocks_per_sm) * Kernel::kThreadCount * 100) /
          properties.maxThreadsPerMultiProcessor
      : 0;
  return {
      int64_t(attributes.numRegs),
      int64_t(shared_bytes),
      int64_t(attributes.localSizeBytes),
      int64_t(attributes.maxThreadsPerBlock),
      int64_t(active_blocks_per_sm),
      occupancy_pct,
      int64_t(Kernel::kThreadCount),
  };
}

}  // namespace
torch::Tensor cusparselt_sparse_residual_complement_sparse_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual) {
  c10::cuda::CUDAGuard const guard(x.device());
  int const n = int(cusparselt_packed.size(0));
  int const m = int(x.size(0));
  auto output = torch::empty({m, n}, x.options());
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  if (use_wide_sparse_configuration(m)) {
    launch_cusparselt_complement_sparse<WideCusparseLtComplementSparse>(
        x, cusparselt_packed, residual, output, stream);
  } else {
    launch_cusparselt_complement_sparse<NarrowCusparseLtComplementSparse>(
        x, cusparselt_packed, residual, output, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor cusparselt_sparse_residual_fused_base_complement_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant) {
  c10::cuda::CUDAGuard const guard(x.device());
  int const n = int(cusparselt_packed.size(0));
  int const m = int(x.size(0));
  auto output = torch::empty({m, n}, x.options());
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  switch (variant) {
    case 1:
      launch_cusparselt_base_complement_fused<FusedBaseComplementN32S4>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 2:
      launch_cusparselt_base_complement_fused<FusedBaseComplementN128S3>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 0:
      launch_cusparselt_base_complement_fused<FusedBaseComplementN64S3>(
          x, cusparselt_packed, residual, output, stream);
      break;
    default:
      TORCH_CHECK(false, "unknown fused base-complement variant: ", variant);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
std::vector<int64_t>
cusparselt_sparse_residual_complement_sparse_kernel_attributes_cuda(
    int64_t token_rows,
    int64_t output_features) {
  TORCH_CHECK(token_rows > 0, "token_rows must be positive");
  TORCH_CHECK(output_features > 0, "output_features must be positive");
  if (use_wide_sparse_configuration(int(token_rows))) {
    return configuration_attributes<WideCusparseLtComplementSparse>();
  }
  return configuration_attributes<NarrowCusparseLtComplementSparse>();
}

std::vector<int64_t>
cusparselt_sparse_residual_fused_base_complement_kernel_attributes_cuda(
    int64_t token_rows,
    int64_t output_features,
    int64_t variant) {
  TORCH_CHECK(token_rows > 0, "token_rows must be positive");
  TORCH_CHECK(output_features > 0, "output_features must be positive");
  switch (variant) {
    case 1:
      return configuration_attributes<FusedBaseComplementN32S4>();
    case 2:
      return configuration_attributes<FusedBaseComplementN128S3>();
    case 0:
      return configuration_attributes<FusedBaseComplementN64S3>();
    default:
      TORCH_CHECK(false, "unknown fused base-complement variant: ", variant);
  }
}
