#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
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
#include "cutlass_sparse_mma_activation_stationary.h"
#include "cutlass_sparse_mma_single_smem.h"
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
  // Four selectors are xor 5.  The edge-adjacent pair 4 <-> E additionally
  // needs xor F.  Evaluate all four nibbles at once: b0..b3 contain the
  // corresponding bit of every nibble in the low bit of each 4-bit lane.
  // This removes the unrolled per-nibble compares from every sparse warp MMA.
  uint16_t constexpr lane_lsb = 0x1111u;
  uint16_t const b0 = word & lane_lsb;
  uint16_t const b2 = (word >> 2) & lane_lsb;
  // 4 (0100) and E (1110) are exactly the valid selectors satisfying
  // b2=1, b0=0, and b3=b1.
  uint16_t const b3_differs_b1 =
      static_cast<uint16_t>((word >> 3) ^ (word >> 1)) & lane_lsb;
  uint16_t const edge =
      static_cast<uint16_t>(b2 & ~b0 & ~b3_differs_b1) & lane_lsb;
  uint16_t const edge_mask =
      static_cast<uint16_t>(edge * 0xfu);
  return word ^ 0x5555u ^ edge_mask;
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
template <typename StockIteratorE_, int StaticPanels = 0>
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
    int64_t cusparselt_block;
    if constexpr (StaticPanels > 0) {
      // Exact-N variants make both divisors compile-time constants.  The
      // compiler replaces the repeated 64-bit integer divides in every E
      // access with multiply/shift sequences.
      int64_t constexpr panels = StaticPanels;
      int64_t constexpr group_blocks = int64_t(2) * panels;
      int64_t const group_base =
          cutlass_block / group_blocks * group_blocks;
      int64_t const within_group = cutlass_block - group_base;
      int64_t const upper_half = within_group >= panels;
      int64_t const panel = within_group - upper_half * panels;
      cusparselt_block = group_base +
          int64_t(2) * panel + upper_half;
    } else {
      int64_t const group_blocks = int64_t(2) * panels_;
      int64_t const group_base =
          cutlass_block / group_blocks * group_blocks;
      int64_t const within_group = cutlass_block - group_base;
      int64_t const upper_half = within_group >= panels_;
      int64_t const panel = within_group - upper_half * panels_;
      cusparselt_block = group_base +
          int64_t(2) * panel + upper_half;
    }
    return reinterpret_cast<AccessType*>(
        base_ + cusparselt_block * 256 + word_in_block);
  }
};

// Indexed verifier rows are invariant for the complete K reduction. CUTLASS's
// generic Gather iterator reloads the same route entry in every mainloop stage,
// placing an integer-load dependency in front of each activation cp.async.
// This production TokenN=64 iterator has one lane load per routed row owned by
// its warp, broadcasts the absolute row pointers once, and keeps them in named
// registers for every later K64 stage.  F64 uses eight per-thread accesses;
// F128 has twice as many producer threads and therefore uses four.
template <typename ThreadMap_, typename ThreadblockShape_>
class CachedIndexedInputIteratorB {
 public:
  using Element = Bf16;
  using Layout = LayoutB;
  using ThreadMap = ThreadMap_;
  using ThreadblockShape = ThreadblockShape_;
  using AccessType = cutlass::Array<Element, 8>;
  using TensorCoord = cutlass::MatrixCoord;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  static int const kAccessesPerVector = 1;

  struct Params {
    int64_t stride = 0;
    CUTLASS_HOST_DEVICE Params() = default;
    CUTLASS_HOST_DEVICE explicit Params(Layout const& layout)
        : stride(layout.stride(0)) {}
  };

 private:
  cutlass::PitchLinearCoord initial_{};
  int extent_k_ = 0;
  int extent_rows_ = 0;
  int tile_k_ = 0;
  int tile_column_ = 0;
  int iteration_index_ = 0;
  bool mask_enabled_ = true;
  uint64_t cached_row_pointer_0_ = 0;
  uint64_t cached_row_pointer_1_ = 0;
  uint64_t cached_row_pointer_2_ = 0;
  uint64_t cached_row_pointer_3_ = 0;
  uint64_t cached_row_pointer_4_ = 0;
  uint64_t cached_row_pointer_5_ = 0;
  uint64_t cached_row_pointer_6_ = 0;
  uint64_t cached_row_pointer_7_ = 0;
  uint32_t cached_route_valid_mask_ = 0;

 public:
  CUTLASS_DEVICE
  CachedIndexedInputIteratorB(
      Params const& params,
      Element* pointer,
      TensorCoord extent,
      int thread_idx,
      TensorCoord const& offset,
      int const* indices)
      : initial_(ThreadMap::initial_offset(thread_idx)),
        extent_k_(extent.row()),
        extent_rows_(extent.column()),
        tile_k_(offset.row()),
        tile_column_(offset.column()) {
    static_assert(
        (ThreadMap::Iterations::kContiguous *
                 ThreadMap::Iterations::kStrided ==
             8 ||
         ThreadMap::Iterations::kContiguous *
                 ThreadMap::Iterations::kStrided ==
             4) &&
            (ThreadMap::Iterations::kStrided == 4 ||
             ThreadMap::Iterations::kStrided == 8),
        "cached indexed B iterator requires TokenN=64 four/eight-access map");
    static_assert(
        ThreadMap::Detail::WarpThreadArrangement::kContiguous == 8 &&
            ThreadMap::Detail::WarpThreadArrangement::kStrided == 4,
        "cached indexed B iterator owner mapping changed");
    int const owned_iteration = (thread_idx % 32) & 7;
    bool const owns_route =
        owned_iteration < ThreadMap::Iterations::kStrided;
    int const source_row = tile_column_ + initial_.strided() +
                           owned_iteration * ThreadMap::Delta::kStrided;
    bool const owned_valid = owns_route && source_row < extent_rows_;
    int const physical_row = owned_valid ? indices[source_row] : 0;
    uint64_t const owned_row_base = reinterpret_cast<uint64_t>(
        pointer + int64_t(physical_row) * params.stride);

    int const row_group = initial_.strided() & 3;
    int const owner_lane = row_group * 8;
    uint64_t const row_base_0 =
        __shfl_sync(0xffffffffu, owned_row_base, owner_lane + 0);
    uint64_t const row_base_1 =
        __shfl_sync(0xffffffffu, owned_row_base, owner_lane + 1);
    uint64_t const row_base_2 =
        __shfl_sync(0xffffffffu, owned_row_base, owner_lane + 2);
    uint64_t const row_base_3 =
        __shfl_sync(0xffffffffu, owned_row_base, owner_lane + 3);
    uint64_t const row_base_4 =
        __shfl_sync(0xffffffffu, owned_row_base, owner_lane + 4);
    uint64_t const row_base_5 =
        __shfl_sync(0xffffffffu, owned_row_base, owner_lane + 5);
    uint64_t const row_base_6 =
        __shfl_sync(0xffffffffu, owned_row_base, owner_lane + 6);
    uint64_t const row_base_7 =
        __shfl_sync(0xffffffffu, owned_row_base, owner_lane + 7);
    int const lane_k = initial_.contiguous();
    cached_row_pointer_0_ = reinterpret_cast<uint64_t>(
        reinterpret_cast<Element*>(row_base_0) + lane_k);
    cached_row_pointer_1_ = reinterpret_cast<uint64_t>(
        reinterpret_cast<Element*>(row_base_1) + lane_k);
    cached_row_pointer_2_ = reinterpret_cast<uint64_t>(
        reinterpret_cast<Element*>(row_base_2) + lane_k);
    cached_row_pointer_3_ = reinterpret_cast<uint64_t>(
        reinterpret_cast<Element*>(row_base_3) + lane_k);
    cached_row_pointer_4_ = reinterpret_cast<uint64_t>(
        reinterpret_cast<Element*>(row_base_4) + lane_k);
    cached_row_pointer_5_ = reinterpret_cast<uint64_t>(
        reinterpret_cast<Element*>(row_base_5) + lane_k);
    cached_row_pointer_6_ = reinterpret_cast<uint64_t>(
        reinterpret_cast<Element*>(row_base_6) + lane_k);
    cached_row_pointer_7_ = reinterpret_cast<uint64_t>(
        reinterpret_cast<Element*>(row_base_7) + lane_k);
    uint32_t const owned_valid_u32 = owned_valid ? 1u : 0u;
    cached_route_valid_mask_ =
        (__shfl_sync(0xffffffffu, owned_valid_u32, owner_lane + 0) << 0) |
        (__shfl_sync(0xffffffffu, owned_valid_u32, owner_lane + 1) << 1) |
        (__shfl_sync(0xffffffffu, owned_valid_u32, owner_lane + 2) << 2) |
        (__shfl_sync(0xffffffffu, owned_valid_u32, owner_lane + 3) << 3) |
        (__shfl_sync(0xffffffffu, owned_valid_u32, owner_lane + 4) << 4) |
        (__shfl_sync(0xffffffffu, owned_valid_u32, owner_lane + 5) << 5) |
        (__shfl_sync(0xffffffffu, owned_valid_u32, owner_lane + 6) << 6) |
        (__shfl_sync(0xffffffffu, owned_valid_u32, owner_lane + 7) << 7);
  }

  CUTLASS_HOST_DEVICE void set_iteration_index(int index) {
    iteration_index_ = index;
  }
  CUTLASS_HOST_DEVICE CachedIndexedInputIteratorB& operator++() {
    ++iteration_index_;
    return *this;
  }
  CUTLASS_DEVICE void add_tile_offset(TensorCoord const& offset) {
    tile_k_ += offset.row() * ThreadblockShape::kK;
    tile_column_ += offset.column() * ThreadblockShape::kN;
  }
  CUTLASS_HOST_DEVICE void clear_mask(bool clear = true) {
    if (clear) mask_enabled_ = false;
  }
  CUTLASS_HOST_DEVICE bool valid() const {
    int const contiguous =
        iteration_index_ % ThreadMap::Iterations::kContiguous;
    int const strided =
        iteration_index_ / ThreadMap::Iterations::kContiguous;
    int const dense_k = tile_k_ + initial_.contiguous() +
                        contiguous * ThreadMap::Delta::kContiguous;
    return mask_enabled_ &&
           ((cached_route_valid_mask_ >> strided) & 1u) &&
           dense_k >= 0 && dense_k + AccessType::kElements <= extent_k_;
  }
  CUTLASS_DEVICE AccessType* get() const {
    int const contiguous =
        iteration_index_ % ThreadMap::Iterations::kContiguous;
    int const strided =
        iteration_index_ / ThreadMap::Iterations::kContiguous;
    uint64_t const row_pointer =
        strided == 0 ? cached_row_pointer_0_ :
        strided == 1 ? cached_row_pointer_1_ :
        strided == 2 ? cached_row_pointer_2_ :
        strided == 3 ? cached_row_pointer_3_ :
        strided == 4 ? cached_row_pointer_4_ :
        strided == 5 ? cached_row_pointer_5_ :
        strided == 6 ? cached_row_pointer_6_ :
                       cached_row_pointer_7_;
    return reinterpret_cast<AccessType*>(
        reinterpret_cast<Element*>(row_pointer) + tile_k_ +
        contiguous * ThreadMap::Delta::kContiguous);
  }
};

// CUTLASS's generic split-K sparse wrapper forwards the K-tile coordinate to
// the epilogue.  This project's transpose visitor intentionally exposes a
// rank-3 logical output with batch extent one, so a K coordinate of one is
// outside that logical tensor and its guarded stores are suppressed.  This
// wrapper makes each z partition an explicit local GEMM with K/2 and passes
// batch zero to the epilogue; the visitor uses blockIdx.z only for the physical
// partial-output offset.  Advancing A/B/E pointers explicitly also makes the
// cuSPARSELt physical split auditable.  Its metadata macro-swizzle restarts
// every K64 group, so the two K/2 payloads are physically contiguous and have
// identical local layout for the validated N=K=5120 shape.
template <
    typename Mma_,
    typename Epilogue_,
    typename ThreadblockSwizzle_,
    int SplitKSlices = 2,
    bool PersistentM = false>
struct LocalSplitK2SparseGemmWithEpilogueVisitor {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Base = cutlass::gemm::kernel::SparseGemmWithEpilogueVisitor<
      Mma, Epilogue, ThreadblockSwizzle>;
  struct Params : Base::Params {
    using BaseParams = typename Base::Params;
    int const* gather_indices = nullptr;
    int tile_m_offset = 0;

    CUTLASS_HOST_DEVICE
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(
        cutlass::gemm::GemmCoord const& problem_size,
        cutlass::gemm::GemmCoord const& grid_tiled_shape,
        typename Mma::IteratorA::TensorRef ref_a,
        typename Mma::IteratorB::TensorRef ref_b,
        typename Mma::IteratorE::TensorRef ref_e,
        typename Base::FusionCallbacks::Arguments output_op,
        int const* indices = nullptr,
        int feature_tile_offset = 0)
        : BaseParams(problem_size, grid_tiled_shape, ref_a, ref_b, ref_e,
                     output_op),
          gather_indices(indices),
          tile_m_offset(feature_tile_offset) {}
  };
  using SharedStorage = typename Base::SharedStorage;
  static int const kThreadCount = Base::kThreadCount;
  static int const kSparse = Base::kSparse;
  static int const kElementsPerElementE = Base::kElementsPerElementE;

  CUTLASS_DEVICE void operator()(Params const& params,
                                 SharedStorage& shared_storage) {
    ThreadblockSwizzle swizzle;
    cutlass::gemm::GemmCoord const block_tile =
        swizzle.get_tile_offset(params.swizzle_log_tile);
    if (params.grid_tiled_shape.n() <= block_tile.n() ||
        block_tile.k() >= SplitKSlices) {
      return;
    }

    int const local_k = params.problem_size.k() / SplitKSlices;
    int const split = block_tile.k();
    int const thread_idx = threadIdx.x;
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int const lane_idx = threadIdx.x % 32;
    int const tile_m_stride =
        PersistentM ? int(gridDim.x) : params.grid_tiled_shape.m();
    for (int tile_m = block_tile.m() + params.tile_m_offset;
         tile_m < params.grid_tiled_shape.m();
         tile_m += tile_m_stride) {
    cutlass::gemm::GemmCoord const tile{
        tile_m, block_tile.n(), block_tile.k()};

    using ElementA = typename Mma::IteratorA::Element;
    using ElementB = typename Mma::IteratorB::Element;
    using ElementE = typename Mma::IteratorE::Element;
    ElementA* ptr_a = params.ref_A.data() + int64_t(split) * local_k / kSparse;
    ElementB* ptr_b = params.ref_B.data() + int64_t(split) * local_k;
    int64_t const metadata_words_per_split =
        int64_t(params.problem_size.m()) * local_k /
        kSparse / kElementsPerElementE;
    ElementE* ptr_e = params.ref_E.data() +
        int64_t(split) * metadata_words_per_split;

    typename Mma::IteratorA iterator_a(
        params.params_A,
        ptr_a,
        {params.problem_size.m(), local_k / kSparse},
        thread_idx,
        {tile.m() * Mma::Shape::kM, 0});
    typename Mma::IteratorB iterator_b(
        params.params_B,
        ptr_b,
        {local_k, params.problem_size.n()},
        thread_idx,
        {0, tile.n() * Mma::Shape::kN},
        params.gather_indices);
    typename Mma::IteratorE iterator_e(
        params.params_E,
        ptr_e,
        {params.problem_size.m(),
         local_k / kSparse / kElementsPerElementE},
        thread_idx,
        {tile.m() * Mma::Shape::kM, 0});

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC accumulators;
    accumulators.clear();
    int const gemm_k_iterations =
        (local_k + Mma::Shape::kK - 1) / Mma::Shape::kK;
    mma(gemm_k_iterations, accumulators, iterator_a, iterator_b, iterator_e,
        accumulators);

    Epilogue epilogue(
        params.output_op,
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx);
    // Output is a rank-3 logical tensor with batch extent one.  Keep the
    // epilogue coordinate in batch zero; the store visitor uses blockIdx.z to
    // select the physical partial slice.
    cutlass::gemm::GemmCoord const output_tile{tile.m(), tile.n(), 0};
    epilogue(accumulators, output_tile, params.problem_shape, thread_idx);
    // The next persistent feature task reuses the same mainloop/epilogue
    // shared storage, so all producer and consumer warps must retire first.
    __syncthreads();
    }
  }
};

// Residual-only sparse GEMM.  A is already stored as the two compact residual
// BF16 values per K4, E is read from the sole cuSPARSELt allocation through
// CusparseLtMetadataIteratorE, and ComplementSparseWarpMma flips E in registers
// immediately before HMMA.SP.  No dense reconstruction or metadata workspace
// exists on this path.
template <
    int ThreadblockM,
    int ThreadblockN,
    int WarpM,
    int WarpN,
    int MainloopStages,
    bool SingleSmem = false,
    int ThreadblockK = 64,
    int WarpK = 64,
    int StaticPanels = 0,
    bool GatherB = false>
struct ComplementSparseConfiguration {
  using ThreadblockShape =
      cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
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
  using GatherIteratorB = CachedIndexedInputIteratorB<
      typename DefaultMma::ThreadMapB, ThreadblockShape>;
  using IteratorB = std::conditional_t<
      GatherB, GatherIteratorB, typename DefaultMma::IteratorB>;
  using IteratorE = CusparseLtMetadataIteratorE<
      typename DefaultMma::IteratorE, StaticPanels>;
  using ComplementWarpMma = ::ComplementSparseWarpMma<
      typename MmaCore::MmaPolicy::Operator>;
  using MmaPolicy = cutlass::gemm::threadblock::SparseMmaPolicy<
      ComplementWarpMma,
      typename MmaCore::MmaPolicy::SmemPaddingA,
      typename MmaCore::MmaPolicy::SmemPaddingB,
      typename MmaCore::MmaPolicy::SmemPaddingE,
      MmaCore::MmaPolicy::kPartitionsK>;
  using MultistageThreadblockMma =
      cutlass::gemm::threadblock::SparseMmaMultistage<
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
  using SingleSmemThreadblockMma =
      cutlass::gemm::threadblock::SparseMmaSingleSmem<
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
          MmaPolicy>;
  using ThreadblockMma = std::conditional_t<
      SingleSmem,
      SingleSmemThreadblockMma,
      MultistageThreadblockMma>;
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

// CTA-local activation-stationary complement.  CUTLASS's physical tile and
// epilogue remain 64xTokenN, but a logical CTA covers two adjacent 64-row
// feature panels.  One warp retains each B fragment and applies it to both
// panels, halving activation global/shared loads relative to two independent
// F64 CTAs.  T16 keeps enough logical CTAs for small dense-token counts.
template <int FeatureM, int TokenN>
struct ActivationStationaryComplementConfiguration {
  using LogicalThreadblockShape =
      cutlass::gemm::GemmShape<2 * FeatureM, TokenN, 64>;
  using PhysicalThreadblockShape =
      cutlass::gemm::GemmShape<FeatureM, TokenN, 64>;
  using WarpShape = cutlass::gemm::GemmShape<FeatureM, TokenN, 64>;
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
      PhysicalThreadblockShape,
      WarpShape,
      SparseInstructionShape,
      2,
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
  using ThreadblockMma =
      cutlass::gemm::threadblock::SparseMmaActivationStationary2<
          LogicalThreadblockShape,
          PhysicalThreadblockShape,
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
          MmaPolicy>;
  using ThreadblockShape = LogicalThreadblockShape;
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      Bf16, 8, float, float>;
  using BaseEpilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          PhysicalThreadblockShape,
          typename ThreadblockMma::Operator,
          1,
          OutputOp,
          OutputOp::kCount>::Epilogue;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          PhysicalThreadblockShape,
          WarpShape,
          Bf16,
          8,
          kEpilogueStages>;
  using OutputStore = speculators::speclink::VisitorTransposeAuxStore<
      OutputThreadMap,
      Bf16,
      cutlass::FloatRoundStyle::round_to_nearest,
      PhysicalThreadblockShape::kM,
      PhysicalThreadblockShape::kN,
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
  using Kernel =
      cutlass::gemm::kernel::ActivationStationarySparseGemmWithEpilogueVisitor<
          ThreadblockMma, Epilogue, Swizzle>;
};

template <
    int FeatureM,
    int TokenN,
    int WarpM,
    int WarpN,
    int BStages = 4,
    int AStages = 1,
    int StaticPanels = 0,
    bool GatherB = false>
struct BResidentComplementConfiguration {
  using ThreadblockShape =
      cutlass::gemm::GemmShape<FeatureM, TokenN, 64>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, 64>;
  using DefaultMma = cutlass::gemm::threadblock::DefaultSparseMma<
      Bf16, LayoutA, 8, Bf16, LayoutB, 8, float,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      SparseInstructionShape,
      4,
      cutlass::arch::OpMultiplyAdd>;
  using MmaCore = typename DefaultMma::MmaCore;
  using IteratorA = typename DefaultMma::IteratorA;
  using GatherIteratorB = CachedIndexedInputIteratorB<
      typename DefaultMma::ThreadMapB, ThreadblockShape>;
  using IteratorB = std::conditional_t<
      GatherB, GatherIteratorB, typename DefaultMma::IteratorB>;
  using IteratorE = CusparseLtMetadataIteratorE<
      typename DefaultMma::IteratorE, StaticPanels>;
  using ComplementWarpMma = ::ComplementSparseWarpMma<
      typename MmaCore::MmaPolicy::Operator>;
  using MmaPolicy = cutlass::gemm::threadblock::SparseMmaPolicy<
      ComplementWarpMma,
      typename MmaCore::MmaPolicy::SmemPaddingA,
      typename MmaCore::MmaPolicy::SmemPaddingB,
      typename MmaCore::MmaPolicy::SmemPaddingE,
      MmaCore::MmaPolicy::kPartitionsK>;
  using ThreadblockMma =
      cutlass::gemm::threadblock::SparseMmaBResidentAStreamed<
          ThreadblockShape,
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
          BStages,
          AStages>;
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
          BaseEpilogue, FusionCallbacks, kEpilogueStages>;
  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using Kernel = cutlass::gemm::kernel::SparseGemmWithEpilogueVisitor<
      ThreadblockMma, Epilogue, Swizzle>;
};

// Dense-routed rows need both complementary 2:4 products.  This mainloop
// stages base A and complement A separately, but stages activation B and the
// sole cuSPARSELt metadata E only once.  Both warp MMAs update one FP32
// accumulator fragment and therefore share one epilogue/output write.
template <
    int ThreadblockN,
    int WarpN,
    int MainloopStages,
    int ThreadblockM = 128,
    int WarpM = 64>
struct FusedBaseComplementSparseConfiguration {
  using ThreadblockShape =
      cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, 64>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, 64>;
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
// Keep the K tile and four-stage mainloop frozen while changing only the
// logical output tile.  In the transposed GEMM used here, ThreadblockM is the
// output-feature panel and ThreadblockN is the dense-token panel.
using ComplementFeature128Token64S4 =
    ComplementSparseConfiguration<128, 64, 64, 32, 4>;
using ComplementFeature64Token64S4 =
    ComplementSparseConfiguration<64, 64, 64, 32, 4>;
using ComplementFeature128Token32S4 =
    ComplementSparseConfiguration<128, 32, 64, 32, 4>;
using ComplementFeature64Token32S4 =
    ComplementSparseConfiguration<64, 32, 64, 32, 4>;
// Low-SMEM screening candidates.  These intentionally trade mainloop depth
// for the ability to reside beside a 76,800-byte cuSPARSELt CTA.  They remain
// separate kernels and are never selected by the production "auto" path.
using ComplementFeature64Token64S2 =
    ComplementSparseConfiguration<64, 64, 64, 32, 2>;
using ComplementFeature64Token32S2 =
    ComplementSparseConfiguration<64, 32, 64, 32, 2>;
using ComplementFeature128Token64SingleSmem =
    ComplementSparseConfiguration<128, 64, 64, 32, 2, true>;
// Wave-aware JIT screening candidate.  A 256-feature CTA keeps cuSPARSELt's
// 128-row value/metadata panels aligned while halving the complement grid.
// Three stages keep its dynamic shared memory within the SM120 block limit.
using ComplementFeature256Token64S3 =
    ComplementSparseConfiguration<256, 64, 64, 64, 3>;
using ComplementActivationStationaryToken32 =
    ActivationStationaryComplementConfiguration<64, 32>;
using ComplementBResidentFeature128Token32 =
    BResidentComplementConfiguration<128, 32, 64, 32>;
using ComplementBResidentFeature64Token32 =
    BResidentComplementConfiguration<64, 32, 64, 32>;
using ComplementBResidentFeature128Token32A2 =
    BResidentComplementConfiguration<128, 32, 64, 32, 4, 2>;
using ComplementBResidentFeature64Token32A2 =
    BResidentComplementConfiguration<64, 32, 64, 32, 4, 2>;
// Exact-N=5120 finalists.  The first is the HBM-cold winner: two activation
// stages, one streamed residual/metadata stage, and two token warps in a
// 20,992-byte CTA that can reside beside the 76,800-byte cuSPARSELt base.  The
// second is the steady-state winner and retains the stock four-stage pipeline.
using ComplementBResidentFeature64Token64B2A1P40 =
    BResidentComplementConfiguration<64, 64, 64, 32, 2, 1, 40>;
using ComplementBResidentFeature64Token64B2A1 =
    BResidentComplementConfiguration<64, 64, 64, 32, 2, 1>;
// gate_up specialization: the long output-feature dimension still exposes
// hundreds of CTAs at FeatureM=128.  Compared with the F64 production kernel,
// each CTA reuses its activation tile across twice as many output features
// while keeping the low-SMEM B2/A1 schedule needed to co-reside with a
// cuSPARSELt base CTA.
using ComplementBResidentFeature128Token64B2A1 =
    BResidentComplementConfiguration<128, 64, 64, 32, 2, 1>;
using ComplementBResidentFeature128Token64B2A1P192 =
    BResidentComplementConfiguration<128, 64, 64, 32, 2, 1, 192>;
using ComplementBResidentFeature128Token64B2A1P224 =
    BResidentComplementConfiguration<128, 64, 64, 32, 2, 1, 224>;
// D=128 specialization.  One Token128 CTA consumes the entire routed dense
// group, so each Feature128 weight/metadata panel is streamed once instead of
// once per Token64 half.  The B-resident mainloop requires two B stages.
using ComplementBResidentFeature128Token128B2A1 =
    BResidentComplementConfiguration<128, 128, 64, 64, 2, 1>;
using ComplementBResidentFeature128Token128B2A1P192 =
    BResidentComplementConfiguration<128, 128, 64, 64, 2, 1, 192>;
using ComplementBResidentFeature128Token128B2A1P224 =
    BResidentComplementConfiguration<128, 128, 64, 64, 2, 1, 224>;
// With Token128 removing duplicate weight streams, Feature256 reuses each
// activation tile across twice as many gate_up outputs.  Split-K=2 still
// exposes 192 (Qwen) or 224 (Llama) CTAs, enough to cover all 170 SMs.
using ComplementBResidentFeature256Token128B2A1 =
    BResidentComplementConfiguration<256, 128, 64, 64, 2, 1>;
using ComplementBResidentFeature256Token128B2A1P192 =
    BResidentComplementConfiguration<256, 128, 64, 64, 2, 1, 192>;
using ComplementBResidentFeature256Token128B2A1P224 =
    BResidentComplementConfiguration<256, 128, 64, 64, 2, 1, 224>;
// Token128 does not co-reside with the 76.8KB cuSPARSELt CTA, so use the
// otherwise-idle shared-memory budget to double-buffer streamed A/E.  B3 also
// screens whether an additional activation stage hides HBM latency.
using ComplementBResidentFeature128Token128B2A2 =
    BResidentComplementConfiguration<128, 128, 64, 64, 2, 2>;
using ComplementBResidentFeature128Token128B2A2P192 =
    BResidentComplementConfiguration<128, 128, 64, 64, 2, 2, 192>;
using ComplementBResidentFeature128Token128B2A2P224 =
    BResidentComplementConfiguration<128, 128, 64, 64, 2, 2, 224>;
using ComplementBResidentFeature128Token128B3A2 =
    BResidentComplementConfiguration<128, 128, 64, 64, 3, 2>;
using ComplementBResidentFeature128Token128B3A2P192 =
    BResidentComplementConfiguration<128, 128, 64, 64, 3, 2, 192>;
using ComplementBResidentFeature128Token128B3A2P224 =
    BResidentComplementConfiguration<128, 128, 64, 64, 3, 2, 224>;
using ComplementFeature64Token64S4P40 =
    ComplementSparseConfiguration<64, 64, 64, 32, 4, false, 64, 64, 40>;
// Production indexed variants consume verifier rows directly from the
// unpermuted activation. CUTLASS's gather iterator redirects each mainloop B
// load through the compact int32 route, removing the standalone MxK gather
// kernel and its temporary tensor.
using IndexedComplementBResidentFeature64Token64B2A1 =
    BResidentComplementConfiguration<64, 64, 64, 32, 2, 1, 0, true>;
using IndexedComplementBResidentFeature128Token64B2A1 =
    BResidentComplementConfiguration<128, 64, 64, 32, 2, 1, 0, true>;
using IndexedComplementBResidentFeature128Token64B2A1P192 =
    BResidentComplementConfiguration<128, 64, 64, 32, 2, 1, 192, true>;
using IndexedComplementBResidentFeature128Token64B2A1P224 =
    BResidentComplementConfiguration<128, 64, 64, 32, 2, 1, 224, true>;
using IndexedComplementBResidentFeature64Token64B2A1P40 =
    BResidentComplementConfiguration<64, 64, 64, 32, 2, 1, 40, true>;
using WideCusparseLtComplementSparse =
    ComplementSparseConfiguration<128, 128, 64, 64, 3>;
using FusedBaseComplementN64S3 =
    FusedBaseComplementSparseConfiguration<64, 32, 3>;
using FusedBaseComplementN32S4 =
    FusedBaseComplementSparseConfiguration<32, 32, 4>;
using FusedBaseComplementN128S3 =
    FusedBaseComplementSparseConfiguration<128, 64, 3>;
using FusedBaseComplementF64N128S3 =
    FusedBaseComplementSparseConfiguration<128, 32, 3, 64, 64>;
using FusedBaseComplementF64N128S4 =
    FusedBaseComplementSparseConfiguration<128, 32, 4, 64, 64>;
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

template <
    typename Config,
    int kSplitKSlices = 2,
    bool kPersistentM = false>
void launch_cusparselt_complement_sparse_splitk2(
    torch::Tensor const& x,
    torch::Tensor const& cusparselt_packed,
    torch::Tensor const& residual,
    torch::Tensor& partials,
    cudaStream_t stream,
    int const* gather_indices = nullptr,
    int gathered_rows = -1,
    int persistent_m_blocks = 0,
    int launch_m_blocks = 0,
    int tile_m_offset = 0) {
  using Kernel = LocalSplitK2SparseGemmWithEpilogueVisitor<
      typename Config::ThreadblockMma,
      typename Config::Epilogue,
      typename Config::Swizzle,
      kSplitKSlices,
      kPersistentM>;
  using IteratorA = typename Config::IteratorA;
  using IteratorB = typename Config::IteratorB;
  using IteratorE = typename Config::IteratorE;
  using OutputStore = typename Config::OutputStore;
  using FusionCallbacks = typename Config::FusionCallbacks;
  using Swizzle = typename Config::Swizzle;
  int const n = int(cusparselt_packed.size(0));
  int const k = int(x.size(1));
  int const m = gathered_rows >= 0 ? gathered_rows : int(x.size(0));
  cutlass::gemm::GemmCoord const problem{n, m, k};
  Swizzle swizzle;
  cutlass::gemm::GemmCoord const tiled = swizzle.get_tiled_shape(
      problem,
      {Config::ThreadblockShape::kM,
       Config::ThreadblockShape::kN,
       Config::ThreadblockShape::kK},
      kSplitKSlices);
  Bf16* residual_ptr = reinterpret_cast<Bf16*>(residual.data_ptr());
  Bf16* x_ptr = reinterpret_cast<Bf16*>(x.data_ptr());
  Bf16* packed_base = reinterpret_cast<Bf16*>(cusparselt_packed.data_ptr());
  ElementE* metadata = reinterpret_cast<ElementE*>(
      packed_base + int64_t(n) * k / 2);
  typename IteratorA::TensorRef ref_a(residual_ptr, LayoutA(k / 2));
  typename IteratorB::TensorRef ref_b(x_ptr, LayoutB(k));
  typename IteratorE::TensorRef ref_e(
      metadata, GmemLayoutE::packed({n, k / 16}));
  typename OutputStore::Arguments output_arguments{
      reinterpret_cast<Bf16*>(partials.data_ptr()), int64_t(m) * n};
  typename FusionCallbacks::Arguments callbacks{{}, output_arguments};
  typename Kernel::Params params(
      problem, tiled, ref_a, ref_b, ref_e, callbacks, gather_indices,
      tile_m_offset);
  int const shared_bytes = int(sizeof(typename Kernel::SharedStorage));
  if (shared_bytes >= (48 << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes));
  }
  dim3 grid = swizzle.get_grid_shape(tiled);
  if constexpr (kPersistentM) {
    TORCH_CHECK(persistent_m_blocks > 0,
                "persistent feature-block quota must be positive");
    grid.x = grid.x < unsigned(persistent_m_blocks)
        ? grid.x
        : unsigned(persistent_m_blocks);
  } else if (launch_m_blocks > 0) {
    grid.x = grid.x < unsigned(launch_m_blocks)
        ? grid.x
        : unsigned(launch_m_blocks);
  }
  dim3 const block(Kernel::kThreadCount, 1, 1);
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

template <typename Index>
__global__ void indexed_gather_bf16x8_kernel(
    uint4* destination,
    uint4 const* source,
    Index const* indices,
    int rows,
    int vectors_per_row) {
  int const row = int(blockIdx.y);
  if (row >= rows) return;
  for (int vector = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
       vector < vectors_per_row;
       vector += int(blockDim.x) * int(gridDim.x)) {
    int const linear = row * vectors_per_row + vector;
    int64_t const source_row = indices[row];
    destination[linear] = source[source_row * int64_t(vectors_per_row) + vector];
  }
}

template <typename Index>
__global__ void indexed_copy_bf16x8_inplace_kernel(
    uint4* destination,
    uint4 const* source,
    Index const* indices,
    int rows,
    int vectors_per_row) {
  int const row = int(blockIdx.y);
  if (row >= rows) return;
  for (int vector = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
       vector < vectors_per_row;
       vector += int(blockDim.x) * int(gridDim.x)) {
    int64_t const destination_row = indices[row];
    destination[destination_row * int64_t(vectors_per_row) + vector] =
        source[row * int64_t(vectors_per_row) + vector];
  }
}

template <typename Index>
__global__ void indexed_add_bf16x8_inplace_kernel(
    __nv_bfloat16* base,
    __nv_bfloat16 const* correction,
    Index const* indices,
    int rows,
    int columns) {
  int const vectors_per_row = columns / 8;
  int const row = int(blockIdx.y);
  if (row >= rows) return;
  for (int vector = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
       vector < vectors_per_row;
       vector += int(blockDim.x) * int(gridDim.x)) {
    int64_t const destination_row = indices[row];
    auto* destination = reinterpret_cast<__nv_bfloat162*>(
        base + destination_row * int64_t(columns)) + vector * 4;
    auto const* source = reinterpret_cast<__nv_bfloat162 const*>(
        correction + int64_t(row) * columns) + vector * 4;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      float2 const lhs = __bfloat1622float2(destination[pair]);
      float2 const rhs = __bfloat1622float2(source[pair]);
      destination[pair] =
          __floats2bfloat162_rn(lhs.x + rhs.x, lhs.y + rhs.y);
    }
  }
}

template <int SplitKSlices, typename Index>
__global__ void splitk2_indexed_add_bf16x8_inplace_kernel(
    __nv_bfloat16* base,
    __nv_bfloat16 const* partials,
    Index const* indices,
    int rows,
    int columns) {
  int const vectors_per_row = columns / 8;
  int64_t const partial_stride = int64_t(rows) * columns;
  int const row = int(blockIdx.y);
  if (row >= rows) return;
  for (int vector = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
       vector < vectors_per_row;
       vector += int(blockDim.x) * int(gridDim.x)) {
    int64_t const destination_row = indices[row];
    auto* destination = reinterpret_cast<__nv_bfloat162*>(
        base + destination_row * int64_t(columns)) + vector * 4;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      float2 const lhs = __bfloat1622float2(destination[pair]);
      float sum_x = lhs.x;
      float sum_y = lhs.y;
#pragma unroll
      for (int split = 0; split < SplitKSlices; ++split) {
        auto const* partial = reinterpret_cast<__nv_bfloat162 const*>(
            partials + int64_t(split) * partial_stride +
            int64_t(row) * columns) + vector * 4;
        float2 const rhs = __bfloat1622float2(partial[pair]);
        sum_x += rhs.x;
        sum_y += rhs.y;
      }
      destination[pair] = __floats2bfloat162_rn(sum_x, sum_y);
    }
  }
}

torch::Tensor cusparselt_sparse_residual_indexed_gather_cuda(
    torch::Tensor source,
    torch::Tensor indices) {
  c10::cuda::CUDAGuard const guard(source.device());
  int const rows = int(indices.numel());
  int const columns = int(source.size(1));
  int const vectors_per_row = columns / 8;
  auto destination = torch::empty({rows, columns}, source.options());
  int const threads = 256;
  dim3 const blocks((vectors_per_row + threads - 1) / threads, rows, 1);
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(source.get_device()).stream();
  if (indices.scalar_type() == torch::kInt32) {
    indexed_gather_bf16x8_kernel<int32_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint4*>(destination.data_ptr()),
        reinterpret_cast<uint4 const*>(source.data_ptr()),
        indices.data_ptr<int32_t>(), rows, vectors_per_row);
  } else {
    indexed_gather_bf16x8_kernel<int64_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint4*>(destination.data_ptr()),
        reinterpret_cast<uint4 const*>(source.data_ptr()),
        indices.data_ptr<int64_t>(), rows, vectors_per_row);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return destination;
}

torch::Tensor cusparselt_sparse_residual_indexed_add_inplace_cuda(
    torch::Tensor base,
    torch::Tensor correction,
    torch::Tensor indices) {
  c10::cuda::CUDAGuard const guard(base.device());
  int const rows = int(correction.size(0));
  int const columns = int(base.size(1));
  int const threads = 256;
  int const vectors_per_row = columns / 8;
  dim3 const blocks((vectors_per_row + threads - 1) / threads, rows, 1);
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(base.get_device()).stream();
  if (indices.scalar_type() == torch::kInt32) {
    indexed_add_bf16x8_inplace_kernel<int32_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(base.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(correction.data_ptr()),
        indices.data_ptr<int32_t>(), rows, columns);
  } else {
    indexed_add_bf16x8_inplace_kernel<int64_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(base.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(correction.data_ptr()),
        indices.data_ptr<int64_t>(), rows, columns);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return base;
}

torch::Tensor cusparselt_sparse_residual_indexed_copy_inplace_cuda(
    torch::Tensor destination,
    torch::Tensor source,
    torch::Tensor indices) {
  c10::cuda::CUDAGuard const guard(destination.device());
  int const rows = int(source.size(0));
  int const vectors_per_row = int(destination.size(1)) / 8;
  int const threads = 256;
  dim3 const blocks((vectors_per_row + threads - 1) / threads, rows, 1);
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(destination.get_device()).stream();
  if (indices.scalar_type() == torch::kInt32) {
    indexed_copy_bf16x8_inplace_kernel<int32_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint4*>(destination.data_ptr()),
        reinterpret_cast<uint4 const*>(source.data_ptr()),
        indices.data_ptr<int32_t>(), rows, vectors_per_row);
  } else {
    indexed_copy_bf16x8_inplace_kernel<int64_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint4*>(destination.data_ptr()),
        reinterpret_cast<uint4 const*>(source.data_ptr()),
        indices.data_ptr<int64_t>(), rows, vectors_per_row);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return destination;
}

torch::Tensor cusparselt_sparse_residual_splitk2_indexed_add_inplace_cuda(
    torch::Tensor base,
    torch::Tensor partials,
    torch::Tensor indices) {
  c10::cuda::CUDAGuard const guard(base.device());
  int const rows = int(partials.size(1));
  int const columns = int(base.size(1));
  int const threads = 256;
  int const vectors_per_row = columns / 8;
  dim3 const blocks((vectors_per_row + threads - 1) / threads, rows, 1);
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(base.get_device()).stream();
  if (indices.scalar_type() == torch::kInt32) {
    splitk2_indexed_add_bf16x8_inplace_kernel<2, int32_t>
        <<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(base.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(partials.data_ptr()),
            indices.data_ptr<int32_t>(), rows, columns);
  } else {
    splitk2_indexed_add_bf16x8_inplace_kernel<2, int64_t>
        <<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(base.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(partials.data_ptr()),
            indices.data_ptr<int64_t>(), rows, columns);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return base;
}

torch::Tensor cusparselt_sparse_residual_splitk4_indexed_add_inplace_cuda(
    torch::Tensor base,
    torch::Tensor partials,
    torch::Tensor indices) {
  c10::cuda::CUDAGuard const guard(base.device());
  int const rows = int(partials.size(1));
  int const columns = int(base.size(1));
  int const threads = 256;
  int const vectors_per_row = columns / 8;
  dim3 const blocks((vectors_per_row + threads - 1) / threads, rows, 1);
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(base.get_device()).stream();
  if (indices.scalar_type() == torch::kInt32) {
    splitk2_indexed_add_bf16x8_inplace_kernel<4, int32_t>
        <<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(base.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(partials.data_ptr()),
            indices.data_ptr<int32_t>(), rows, columns);
  } else {
    splitk2_indexed_add_bf16x8_inplace_kernel<4, int64_t>
        <<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(base.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(partials.data_ptr()),
            indices.data_ptr<int64_t>(), rows, columns);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return base;
}

torch::Tensor
cusparselt_sparse_residual_complement_sparse_splitk2_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant) {
  c10::cuda::CUDAGuard const guard(x.device());
  int const n = int(cusparselt_packed.size(0));
  int const m = int(x.size(0));
  auto partials = torch::empty({2, m, n}, x.options());
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  switch (variant) {
    case 0:
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature64Token64B2A1>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 1:
      launch_cusparselt_complement_sparse_splitk2<
          ComplementFeature64Token64S4>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 2:
      launch_cusparselt_complement_sparse_splitk2<
          ComplementFeature128Token64S4>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 3:
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature64Token64B2A1P40>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 4:
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token64B2A1>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 5:
      TORCH_CHECK(n == 24576, "P192 complement variant requires N=24576");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token64B2A1P192>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 6:
      TORCH_CHECK(n == 28672, "P224 complement variant requires N=28672");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token64B2A1P224>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 7:
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B2A1>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 8:
      TORCH_CHECK(n == 24576, "Token128 P192 requires N=24576");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B2A1P192>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 9:
      TORCH_CHECK(n == 28672, "Token128 P224 requires N=28672");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B2A1P224>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 10:
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature256Token128B2A1>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 11:
      TORCH_CHECK(n == 24576, "Feature256 P192 requires N=24576");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature256Token128B2A1P192>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 12:
      TORCH_CHECK(n == 28672, "Feature256 P224 requires N=28672");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature256Token128B2A1P224>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 13:
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B2A2>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 14:
      TORCH_CHECK(n == 24576, "A2 P192 requires N=24576");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B2A2P192>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 15:
      TORCH_CHECK(n == 28672, "A2 P224 requires N=28672");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B2A2P224>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 16:
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B3A2>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 17:
      TORCH_CHECK(n == 24576, "B3/A2 P192 requires N=24576");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B3A2P192>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 18:
      TORCH_CHECK(n == 28672, "B3/A2 P224 requires N=28672");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B3A2P224>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    default:
      TORCH_CHECK(false, "unknown Split-K=2 complement variant: ", variant);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return partials;
}

torch::Tensor
cusparselt_sparse_residual_complement_sparse_splitk4_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant) {
  c10::cuda::CUDAGuard const guard(x.device());
  int const n = int(cusparselt_packed.size(0));
  int const m = int(x.size(0));
  auto partials = torch::empty({4, m, n}, x.options());
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  switch (variant) {
    case 7:
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B2A1, 4>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 8:
      TORCH_CHECK(n == 24576, "Token128 P192 requires N=24576");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B2A1P192, 4>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 9:
      TORCH_CHECK(n == 28672, "Token128 P224 requires N=28672");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B2A1P224, 4>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 10:
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature256Token128B2A1, 4>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 11:
      TORCH_CHECK(n == 24576, "Feature256 P192 requires N=24576");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature256Token128B2A1P192, 4>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    case 12:
      TORCH_CHECK(n == 28672, "Feature256 P224 requires N=28672");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature256Token128B2A1P224, 4>(
          x, cusparselt_packed, residual, partials, stream);
      break;
    default:
      TORCH_CHECK(false, "Split-K=4 requires a Token128 gate_up variant");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return partials;
}

torch::Tensor
cusparselt_sparse_residual_complement_sparse_splitk2_persistent_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant,
    int64_t persistent_m_blocks) {
  c10::cuda::CUDAGuard const guard(x.device());
  int const n = int(cusparselt_packed.size(0));
  int const m = int(x.size(0));
  auto partials = torch::empty({2, m, n}, x.options());
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  switch (variant) {
    case 7:
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B2A1, 2, true>(
          x, cusparselt_packed, residual, partials, stream,
          nullptr, -1, int(persistent_m_blocks));
      break;
    case 8:
      TORCH_CHECK(n == 24576, "Token128 P192 requires N=24576");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B2A1P192, 2, true>(
          x, cusparselt_packed, residual, partials, stream,
          nullptr, -1, int(persistent_m_blocks));
      break;
    case 9:
      TORCH_CHECK(n == 28672, "Token128 P224 requires N=28672");
      launch_cusparselt_complement_sparse_splitk2<
          ComplementBResidentFeature128Token128B2A1P224, 2, true>(
          x, cusparselt_packed, residual, partials, stream,
          nullptr, -1, int(persistent_m_blocks));
      break;
    default:
      TORCH_CHECK(false,
                  "persistent Split-K2 requires a Feature128/Token128 variant");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return partials;
}

torch::Tensor
cusparselt_sparse_residual_complement_sparse_splitk2_chunked_forward_cuda(
    torch::Tensor x,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant,
    int64_t chunk_m_blocks) {
  c10::cuda::CUDAGuard const guard(x.device());
  int const n = int(cusparselt_packed.size(0));
  int const m = int(x.size(0));
  int const feature_panels = (n + 127) / 128;
  auto partials = torch::empty({2, m, n}, x.options());
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  for (int offset = 0; offset < feature_panels;
       offset += int(chunk_m_blocks)) {
    int const blocks = (feature_panels - offset) < int(chunk_m_blocks)
        ? (feature_panels - offset)
        : int(chunk_m_blocks);
    switch (variant) {
      case 7:
        launch_cusparselt_complement_sparse_splitk2<
            ComplementBResidentFeature128Token128B2A1>(
            x, cusparselt_packed, residual, partials, stream,
            nullptr, -1, 0, blocks, offset);
        break;
      case 8:
        TORCH_CHECK(n == 24576, "Token128 P192 requires N=24576");
        launch_cusparselt_complement_sparse_splitk2<
            ComplementBResidentFeature128Token128B2A1P192>(
            x, cusparselt_packed, residual, partials, stream,
            nullptr, -1, 0, blocks, offset);
        break;
      case 9:
        TORCH_CHECK(n == 28672, "Token128 P224 requires N=28672");
        launch_cusparselt_complement_sparse_splitk2<
            ComplementBResidentFeature128Token128B2A1P224>(
            x, cusparselt_packed, residual, partials, stream,
            nullptr, -1, 0, blocks, offset);
        break;
      default:
        TORCH_CHECK(false,
                    "chunked Split-K2 requires Feature128/Token128");
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return partials;
}

torch::Tensor
cusparselt_sparse_residual_complement_sparse_splitk2_indexed_forward_cuda(
    torch::Tensor x,
    torch::Tensor indices,
    torch::Tensor cusparselt_packed,
    torch::Tensor residual,
    int64_t variant) {
  c10::cuda::CUDAGuard const guard(x.device());
  int const n = int(cusparselt_packed.size(0));
  int const gathered_rows = int(indices.numel());
  auto partials = torch::empty({2, gathered_rows, n}, x.options());
  cudaStream_t const stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  int const* gather_indices = indices.data_ptr<int>();
  switch (variant) {
    case 0:
      launch_cusparselt_complement_sparse_splitk2<
          IndexedComplementBResidentFeature64Token64B2A1>(
          x, cusparselt_packed, residual, partials, stream,
          gather_indices, gathered_rows);
      break;
    case 1:
    case 2:
      TORCH_CHECK(false,
                  "indexed Split-K=2 supports production B-resident variants "
                  "0 and 3 only");
    case 3:
      launch_cusparselt_complement_sparse_splitk2<
          IndexedComplementBResidentFeature64Token64B2A1P40>(
          x, cusparselt_packed, residual, partials, stream,
          gather_indices, gathered_rows);
      break;
    case 4:
      launch_cusparselt_complement_sparse_splitk2<
          IndexedComplementBResidentFeature128Token64B2A1>(
          x, cusparselt_packed, residual, partials, stream,
          gather_indices, gathered_rows);
      break;
    case 5:
      TORCH_CHECK(n == 24576, "P192 complement variant requires N=24576");
      launch_cusparselt_complement_sparse_splitk2<
          IndexedComplementBResidentFeature128Token64B2A1P192>(
          x, cusparselt_packed, residual, partials, stream,
          gather_indices, gathered_rows);
      break;
    case 6:
      TORCH_CHECK(n == 28672, "P224 complement variant requires N=28672");
      launch_cusparselt_complement_sparse_splitk2<
          IndexedComplementBResidentFeature128Token64B2A1P224>(
          x, cusparselt_packed, residual, partials, stream,
          gather_indices, gathered_rows);
      break;
    default:
      TORCH_CHECK(false, "unknown indexed Split-K=2 complement variant: ",
                  variant);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return partials;
}

torch::Tensor cusparselt_sparse_residual_complement_sparse_forward_cuda(
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
    case 0:
      launch_cusparselt_complement_sparse<ComplementFeature128Token64S4>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 1:
      launch_cusparselt_complement_sparse<ComplementFeature64Token64S4>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 2:
      launch_cusparselt_complement_sparse<ComplementFeature128Token32S4>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 3:
      launch_cusparselt_complement_sparse<ComplementFeature64Token32S4>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 4:
      launch_cusparselt_complement_sparse<ComplementFeature64Token64S2>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 5:
      launch_cusparselt_complement_sparse<ComplementFeature64Token32S2>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 6:
      launch_cusparselt_complement_sparse<
          ComplementFeature128Token64SingleSmem>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 7:
      launch_cusparselt_complement_sparse<ComplementFeature256Token64S3>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 8:
      launch_cusparselt_complement_sparse<
          ComplementActivationStationaryToken32>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 11:
      launch_cusparselt_complement_sparse<
          ComplementBResidentFeature128Token32>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 12:
      launch_cusparselt_complement_sparse<
          ComplementBResidentFeature64Token32>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 13:
      launch_cusparselt_complement_sparse<
          ComplementBResidentFeature128Token32A2>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 14:
      launch_cusparselt_complement_sparse<
          ComplementBResidentFeature64Token32A2>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 15:
      TORCH_CHECK(n == 5120, "P40 complement variant requires N=5120");
      launch_cusparselt_complement_sparse<
          ComplementBResidentFeature64Token64B2A1P40>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 16:
      TORCH_CHECK(n == 5120, "P40 complement variant requires N=5120");
      launch_cusparselt_complement_sparse<ComplementFeature64Token64S4P40>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 17:
      launch_cusparselt_complement_sparse<
          ComplementBResidentFeature128Token64B2A1>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 18:
      TORCH_CHECK(n == 24576, "P192 complement variant requires N=24576");
      launch_cusparselt_complement_sparse<
          ComplementBResidentFeature128Token64B2A1P192>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 19:
      TORCH_CHECK(n == 28672, "P224 complement variant requires N=28672");
      launch_cusparselt_complement_sparse<
          ComplementBResidentFeature128Token64B2A1P224>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case -1:
      if (use_wide_sparse_configuration(m)) {
        launch_cusparselt_complement_sparse<WideCusparseLtComplementSparse>(
            x, cusparselt_packed, residual, output, stream);
      } else {
        launch_cusparselt_complement_sparse<ComplementFeature128Token64S4>(
            x, cusparselt_packed, residual, output, stream);
      }
      break;
    default:
      TORCH_CHECK(false, "unknown complement variant: ", variant);
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
    case 3:
      launch_cusparselt_base_complement_fused<
          FusedBaseComplementF64N128S3>(
          x, cusparselt_packed, residual, output, stream);
      break;
    case 4:
      launch_cusparselt_base_complement_fused<
          FusedBaseComplementF64N128S4>(
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
    int64_t output_features,
    int64_t variant) {
  TORCH_CHECK(token_rows > 0, "token_rows must be positive");
  TORCH_CHECK(output_features > 0, "output_features must be positive");
  switch (variant) {
    case 0:
      return configuration_attributes<ComplementFeature128Token64S4>();
    case 1:
      return configuration_attributes<ComplementFeature64Token64S4>();
    case 2:
      return configuration_attributes<ComplementFeature128Token32S4>();
    case 3:
      return configuration_attributes<ComplementFeature64Token32S4>();
    case 4:
      return configuration_attributes<ComplementFeature64Token64S2>();
    case 5:
      return configuration_attributes<ComplementFeature64Token32S2>();
    case 6:
      return configuration_attributes<ComplementFeature128Token64SingleSmem>();
    case 7:
      return configuration_attributes<ComplementFeature256Token64S3>();
    case 8:
      return configuration_attributes<ComplementActivationStationaryToken32>();
    case 11:
      return configuration_attributes<
          ComplementBResidentFeature128Token32>();
    case 12:
      return configuration_attributes<
          ComplementBResidentFeature64Token32>();
    case 13:
      return configuration_attributes<
          ComplementBResidentFeature128Token32A2>();
    case 14:
      return configuration_attributes<
          ComplementBResidentFeature64Token32A2>();
    case 15:
      return configuration_attributes<
          ComplementBResidentFeature64Token64B2A1P40>();
    case 16:
      return configuration_attributes<ComplementFeature64Token64S4P40>();
    case 17:
      return configuration_attributes<
          ComplementBResidentFeature128Token64B2A1>();
    case 18:
      return configuration_attributes<
          ComplementBResidentFeature128Token64B2A1P192>();
    case 19:
      return configuration_attributes<
          ComplementBResidentFeature128Token64B2A1P224>();
    case -1:
      if (use_wide_sparse_configuration(int(token_rows))) {
        return configuration_attributes<WideCusparseLtComplementSparse>();
      }
      return configuration_attributes<ComplementFeature128Token64S4>();
    default:
      TORCH_CHECK(false, "unknown complement variant: ", variant);
  }
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
    case 3:
      return configuration_attributes<FusedBaseComplementF64N128S3>();
    case 4:
      return configuration_attributes<FusedBaseComplementF64N128S4>();
    case 0:
      return configuration_attributes<FusedBaseComplementN64S3>();
    default:
      TORCH_CHECK(false, "unknown fused base-complement variant: ", variant);
  }
}
