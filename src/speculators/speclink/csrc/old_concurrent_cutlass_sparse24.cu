#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/sparse_gemm_with_visitor.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_mma_core_sparse_sm80.h"
#include "cutlass/gemm/threadblock/mma_sparse_multistage.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

#include "cutlass_transpose_epilogue_visitor.h"
#include "old_concurrent_sidecar_mma.h"
namespace {

using Bf16 = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using SparseInstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
using DenseInstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
constexpr int kStages = 3;
constexpr int kFusedStages = 2;
constexpr int kEpilogueStages = 1;
constexpr int kSidecarRoleTimingFields = 13;

template <typename ThreadMap_, typename ElementE_, typename LayoutE_,
          typename ThreadblockShape_, bool AssumeFullTiles_ = false>
class DenseCanonicalIteratorA {
 public:
  using Element = Bf16;
  using Layout = LayoutA;
  using ThreadMap = ThreadMap_;
  using AccessType = cutlass::Array<Element, 8>;
  using TensorRef = cutlass::TensorRef<Element const, Layout>;
  using TensorCoord = cutlass::MatrixCoord;
  using ElementE = ElementE_;
  using LayoutE = LayoutE_;
  using ThreadblockShape = ThreadblockShape_;
  static bool const kAssumeFullTiles = AssumeFullTiles_;
  static int const kAccessesPerVector = 1;

  struct Params {
    int64_t dense_stride = 0;

    CUTLASS_HOST_DEVICE Params() = default;
    CUTLASS_HOST_DEVICE explicit Params(Layout const& layout)
        : dense_stride(layout.stride(0)) {}
  };

 private:
  Params params_;
  Element const* dense_ = nullptr;
  cutlass::PitchLinearCoord initial_{};
  int extent_rows_ = 0;
  int extent_dense_k_ = 0;
  int tile_row_ = 0;
  int tile_dense_k_ = 0;
  int iteration_index_ = 0;
  bool mask_enabled_ = true;

  CUTLASS_HOST_DEVICE void coordinate(int& row, int& dense_k) const {
    int contiguous = iteration_index_ % ThreadMap::Iterations::kContiguous;
    int strided = iteration_index_ / ThreadMap::Iterations::kContiguous;
    dense_k = tile_dense_k_ + initial_.contiguous() +
        contiguous * ThreadMap::Delta::kContiguous;
    row = tile_row_ + initial_.strided() +
        strided * ThreadMap::Delta::kStrided;
  }

 public:
  CUTLASS_DEVICE
  DenseCanonicalIteratorA(
      Params const& params, Element const* dense, TensorCoord extent_compressed,
      int thread_idx, TensorCoord offset_compressed)
      : params_(params),
        dense_(dense),
        initial_(ThreadMap::initial_offset(thread_idx)),
        extent_rows_(extent_compressed.row()),
        extent_dense_k_(extent_compressed.column() * 2),
        tile_row_(offset_compressed.row()),
        tile_dense_k_(offset_compressed.column() * 2) {}

  CUTLASS_HOST_DEVICE void set_iteration_index(int index) {
    iteration_index_ = index;
  }
  CUTLASS_HOST_DEVICE DenseCanonicalIteratorA& operator++() {
    ++iteration_index_;
    return *this;
  }
  CUTLASS_DEVICE void add_tile_offset(TensorCoord const& offset) {
    tile_row_ += offset.row() * ThreadblockShape::kM;
    tile_dense_k_ += offset.column() * ThreadblockShape::kK;
  }
  CUTLASS_HOST_DEVICE void clear_mask(bool clear = true) {
    if (clear) mask_enabled_ = false;
  }
  CUTLASS_HOST_DEVICE bool valid() const {
    if constexpr (kAssumeFullTiles) {
      return mask_enabled_;
    }
    int row = 0;
    int dense_k = 0;
    coordinate(row, dense_k);
    return mask_enabled_ && row < extent_rows_ &&
        dense_k + AccessType::kElements <= extent_dense_k_;
  }
  CUTLASS_DEVICE AccessType const* get() const {
    int row = 0;
    int dense_k = 0;
    coordinate(row, dense_k);
    return reinterpret_cast<AccessType const*>(
        dense_ + int64_t(row) * params_.dense_stride + dense_k);
  }

};

// Logical B is a K x CTA-N matrix partitioned into a compile-time number of
// dense-token and sparse-token columns.  Successive logical tiles advance the
// two physical route queues by their own capacities.  The default remains the
// original equal 64:64 pairing; persistent sidecar kernels instantiate the
// same iterator for 32:96, 64:64, and 96:32 warp ratios.
template <typename ThreadMap_, typename ThreadblockShape_,
          int DenseBranchColumns_ = ThreadblockShape_::kN / 2,
          bool CacheRouteRows_ = false,
          bool CacheRouteRowBases_ = false,
          bool PrebroadcastRouteRowPointers_ = false,
          bool AssumeFullTiles_ = false>
class PairedInputIteratorB {
 public:
  using Element = Bf16;
  using Layout = LayoutB;
  using ThreadMap = ThreadMap_;
  using ThreadblockShape = ThreadblockShape_;
  using AccessType = cutlass::Array<Element, 8>;
  using TensorCoord = cutlass::MatrixCoord;
  static int const kAccessesPerVector = 1;
  static int const kDenseBranchColumns = DenseBranchColumns_;
  static int const kSparseBranchColumns =
      ThreadblockShape::kN - DenseBranchColumns_;
  static bool const kCacheRouteRows = CacheRouteRows_;
  static bool const kCacheRouteRowBases = CacheRouteRowBases_;
  static bool const kPrebroadcastRouteRowPointers =
      PrebroadcastRouteRowPointers_;
  static bool const kAssumeFullTiles = AssumeFullTiles_;
  static_assert(!(kCacheRouteRows && kCacheRouteRowBases),
                "route row-id and row-base caches are separate ablations");
  static_assert(!kPrebroadcastRouteRowPointers || kCacheRouteRowBases,
                "prebroadcast pointers extend the row-base cache");
  static_assert(kDenseBranchColumns >= 0 && kSparseBranchColumns >= 0 &&
                    kDenseBranchColumns + kSparseBranchColumns ==
                        ThreadblockShape::kN,
                "paired input branches must partition CTA-N");

 private:
  Element const* dense_ = nullptr;
  Element const* sparse_ = nullptr;
  int64_t const* dense_indices_ = nullptr;
  int64_t const* sparse_indices_ = nullptr;
  cutlass::PitchLinearCoord initial_{};
  int dense_rows_ = 0;
  int sparse_rows_ = 0;
  int k_ = 0;
  int tile_k_ = 0;
  int tile_column_ = 0;
  int iteration_index_ = 0;
  bool mask_enabled_ = true;
  // A routed GEMM keeps the token permutation fixed for the entire K
  // reduction.  The legacy iterator reloaded the same int64 route entry in
  // every K64 stage, placing a long-scoreboard dependency directly in front of
  // each activation cp.async.  The opt-in persistent variant distributes the
  // unique token rows owned by a warp across its lanes: BM32 uses all 32 owner
  // lanes for eight strided iterations, while BM64 uses 16 owner lanes for
  // four.  SHFL broadcasts each cached value to the K-vector lanes that need
  // it.  Two separately compiled ablations retain either the
  // physical row id or its absolute row-base address.  The latter also hoists
  // source choice and ``physical_row * k`` out of every K64 stage.  Both avoid
  // an eight-entry int64 cache in every thread; the all-false specialization
  // preserves the original iterator for matched A/B comparison.
  int64_t cached_physical_row_ = 0;
  uint64_t cached_row_base_ = 0;
  // BM64 has four routed B rows per thread.  The independent prebroadcast
  // specialization resolves all four owner-lane shuffles once in the task
  // constructor instead of once per K64 stage.  Named scalar members keep a
  // runtime-indexed array from becoming local memory; the unrolled iterator
  // loop constant-folds the selector below.
  uint64_t cached_row_pointer_0_ = 0;
  uint64_t cached_row_pointer_1_ = 0;
  uint64_t cached_row_pointer_2_ = 0;
  uint64_t cached_row_pointer_3_ = 0;
  uint32_t cached_route_valid_mask_ = 0;

  CUTLASS_HOST_DEVICE void coordinate(
      int& source_row, int& dense_k, bool& dense_branch) const {
    int contiguous = iteration_index_ % ThreadMap::Iterations::kContiguous;
    int strided = iteration_index_ / ThreadMap::Iterations::kContiguous;
    dense_k = tile_k_ + initial_.contiguous() +
              contiguous * ThreadMap::Delta::kContiguous;
    int logical_column = tile_column_ + initial_.strided() +
                         strided * ThreadMap::Delta::kStrided;
    int pair_tile = logical_column / ThreadblockShape::kN;
    int local_column = logical_column - pair_tile * ThreadblockShape::kN;
    if constexpr (kDenseBranchColumns == 0) {
      dense_branch = false;
    } else if constexpr (kSparseBranchColumns == 0) {
      dense_branch = true;
    } else {
      dense_branch = local_column < kDenseBranchColumns;
    }
    int branch_columns = dense_branch ? kDenseBranchColumns
                                      : kSparseBranchColumns;
    source_row = pair_tile * branch_columns +
                 (dense_branch ? local_column
                               : local_column - kDenseBranchColumns);
  }

 public:
  CUTLASS_DEVICE
  PairedInputIteratorB(
      Element const* dense, Element const* sparse, int dense_rows,
      int sparse_rows, int k, int thread_idx, TensorCoord offset,
      int64_t const* dense_indices = nullptr,
      int64_t const* sparse_indices = nullptr)
      : dense_(dense),
        sparse_(sparse),
        dense_indices_(dense_indices),
        sparse_indices_(sparse_indices),
        initial_(ThreadMap::initial_offset(thread_idx)),
        dense_rows_(dense_rows),
        sparse_rows_(sparse_rows),
        k_(k),
        tile_k_(offset.row()),
        tile_column_(offset.column()) {
    if constexpr (kCacheRouteRows || kCacheRouteRowBases) {
      static_assert(ThreadMap::Iterations::kContiguous == 1 &&
                        (ThreadMap::Iterations::kStrided == 4 ||
                         ThreadMap::Iterations::kStrided == 8),
                    "route-row cache requires the BM32/BM64 persistent B "
                    "thread map");
      static_assert(
          ThreadMap::Detail::WarpThreadArrangement::kContiguous == 8 &&
              ThreadMap::Detail::WarpThreadArrangement::kStrided == 4 &&
              ThreadMap::Delta::kStrided == 4,
          "route-row owner mapping changed");
      int saved_iteration_index = iteration_index_;
      // Within a warp, lane = row_group * 8 + owned_iteration.  Iteration i
      // for any consumer lane in row_group obtains its route row from owner
      // lane row_group * 8 + i.  BM64 has only four iterations, so lanes with
      // owned_iteration 4..7 deliberately perform no route load.
      int owned_iteration = (thread_idx % 32) & 7;
      bool owns_route = owned_iteration < ThreadMap::Iterations::kStrided;
      iteration_index_ = owns_route ? owned_iteration : 0;
      int source_row = 0;
      int dense_k = 0;
      bool dense_branch = false;
      coordinate(source_row, dense_k, dense_branch);
      int64_t const* indices =
          dense_branch ? dense_indices_ : sparse_indices_;
      int rows = dense_branch ? dense_rows_ : sparse_rows_;
      int64_t physical_row = 0;
      bool owned_route_valid =
          owns_route && (kAssumeFullTiles || source_row < rows);
      if (owned_route_valid) {
        physical_row = indices ? indices[source_row] : int64_t(source_row);
      }
      if constexpr (kCacheRouteRows) {
        cached_physical_row_ = physical_row;
      } else {
        Element const* source = dense_branch ? dense_ : sparse_;
        cached_row_base_ = reinterpret_cast<uint64_t>(
            source + physical_row * int64_t(k_));
      }
      iteration_index_ = saved_iteration_index;

      if constexpr (kPrebroadcastRouteRowPointers) {
        static_assert(ThreadMap::Iterations::kStrided == 4,
                      "pointer prebroadcast is the BM64 four-row ablation");
        int row_group = initial_.strided() & 3;
        int owner_lane_0 = row_group * 8;
        int owner_lane_1 = owner_lane_0 + 1;
        int owner_lane_2 = owner_lane_0 + 2;
        int owner_lane_3 = owner_lane_0 + 3;
        uint64_t row_base_0 =
            __shfl_sync(0xffffffffu, cached_row_base_, owner_lane_0);
        uint64_t row_base_1 =
            __shfl_sync(0xffffffffu, cached_row_base_, owner_lane_1);
        uint64_t row_base_2 =
            __shfl_sync(0xffffffffu, cached_row_base_, owner_lane_2);
        uint64_t row_base_3 =
            __shfl_sync(0xffffffffu, cached_row_base_, owner_lane_3);
        int lane_k = initial_.contiguous();
        cached_row_pointer_0_ = reinterpret_cast<uint64_t>(
            reinterpret_cast<Element const*>(row_base_0) + lane_k);
        cached_row_pointer_1_ = reinterpret_cast<uint64_t>(
            reinterpret_cast<Element const*>(row_base_1) + lane_k);
        cached_row_pointer_2_ = reinterpret_cast<uint64_t>(
            reinterpret_cast<Element const*>(row_base_2) + lane_k);
        cached_row_pointer_3_ = reinterpret_cast<uint64_t>(
            reinterpret_cast<Element const*>(row_base_3) + lane_k);
        if constexpr (kAssumeFullTiles) {
          cached_route_valid_mask_ = 0xfu;
        } else {
          uint32_t owned_valid = owned_route_valid ? 1u : 0u;
          cached_route_valid_mask_ =
              (__shfl_sync(0xffffffffu, owned_valid, owner_lane_0) << 0) |
              (__shfl_sync(0xffffffffu, owned_valid, owner_lane_1) << 1) |
              (__shfl_sync(0xffffffffu, owned_valid, owner_lane_2) << 2) |
              (__shfl_sync(0xffffffffu, owned_valid, owner_lane_3) << 3);
        }
      }
    }
  }

  CUTLASS_HOST_DEVICE void set_iteration_index(int index) {
    iteration_index_ = index;
  }
  CUTLASS_HOST_DEVICE PairedInputIteratorB& operator++() {
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
    if constexpr (kAssumeFullTiles) {
      return mask_enabled_;
    }
    if constexpr (kPrebroadcastRouteRowPointers) {
      int dense_k = tile_k_ + initial_.contiguous();
      return mask_enabled_ &&
          ((cached_route_valid_mask_ >> iteration_index_) & 1u) &&
          dense_k >= 0 && dense_k + AccessType::kElements <= k_;
    }
    int source_row = 0;
    int dense_k = 0;
    bool dense_branch = false;
    coordinate(source_row, dense_k, dense_branch);
    int rows = dense_branch ? dense_rows_ : sparse_rows_;
    return mask_enabled_ && source_row < rows && dense_k >= 0 &&
           dense_k + AccessType::kElements <= k_;
  }
  CUTLASS_DEVICE AccessType const* get() const {
    if constexpr (kPrebroadcastRouteRowPointers) {
      uint64_t row_pointer =
          iteration_index_ == 0 ? cached_row_pointer_0_ :
          iteration_index_ == 1 ? cached_row_pointer_1_ :
          iteration_index_ == 2 ? cached_row_pointer_2_ :
                                  cached_row_pointer_3_;
      return reinterpret_cast<AccessType const*>(
          reinterpret_cast<Element const*>(row_pointer) + tile_k_);
    }
    if constexpr (kCacheRouteRowBases) {
      // All eight iterator accesses differ only in their routed token row;
      // they share the same K-vector offset.  Avoid reconstructing the branch
      // and logical source row after the constructor cached an absolute base.
      int dense_k = tile_k_ + initial_.contiguous();
      int row_group = initial_.strided() & 3;
      int owner_lane = row_group * 8 + iteration_index_;
      uint64_t row_base =
          __shfl_sync(0xffffffffu, cached_row_base_, owner_lane);
      return reinterpret_cast<AccessType const*>(
          reinterpret_cast<Element const*>(row_base) + dense_k);
    }

    int source_row = 0;
    int dense_k = 0;
    bool dense_branch = false;
    coordinate(source_row, dense_k, dense_branch);
    Element const* source = dense_branch ? dense_ : sparse_;
    int64_t physical_row = 0;
    if constexpr (kCacheRouteRows) {
      int row_group = initial_.strided() & 3;
      int owner_lane = row_group * 8 + iteration_index_;
      physical_row = __shfl_sync(
          0xffffffffu, cached_physical_row_, owner_lane);
    } else {
      int64_t const* indices =
          dense_branch ? dense_indices_ : sparse_indices_;
      int rows = dense_branch ? dense_rows_ : sparse_rows_;
      physical_row =
          (indices && source_row < rows) ? indices[source_row]
                                        : int64_t(source_row);
    }
    return reinterpret_cast<AccessType const*>(
        source + physical_row * k_ + dense_k);
  }
};

template <typename ThreadblockShape, typename WarpShape,
          typename ThreadblockMma>
struct VisitorEpilogue {
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      Bf16, 8, float, float>;
  using BaseEpilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape,
          typename ThreadblockMma::Operator,
          ThreadblockMma::WarpCount::kK,
          OutputOp,
          OutputOp::kCount>::Epilogue;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape, WarpShape, Bf16, 8, kEpilogueStages>;
  using OutputStore = speculators::speclink::VisitorTransposeAuxStore<
      OutputThreadMap,
      Bf16,
      cutlass::FloatRoundStyle::round_to_nearest,
      ThreadblockShape::kM,
      ThreadblockShape::kN,
      WarpShape::kM>;
  using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore,
      cutlass::epilogue::threadblock::VisitorAccFetch>;
  using Epilogue =
      cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<
          BaseEpilogue, Callbacks, kEpilogueStages>;
};

template <typename ThreadblockShape, typename WarpShape,
          typename ThreadblockMma,
          int DenseBranchTokens = ThreadblockShape::kN / 2,
          bool VectorizeCallbackSharedStore = false,
          bool PartitionedEpilogueReleaseBarrier = true>
struct PairedVisitorEpilogue {
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      Bf16, 8, float, float>;
  using BaseEpilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape,
          typename ThreadblockMma::Operator,
          ThreadblockMma::WarpCount::kK,
          OutputOp,
          OutputOp::kCount>::Epilogue;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape, WarpShape, Bf16, 8, kEpilogueStages>;
  using OutputStore = speculators::speclink::VisitorPairedTransposeAuxStore<
      OutputThreadMap,
      Bf16,
      cutlass::FloatRoundStyle::round_to_nearest,
      ThreadblockShape::kM,
      ThreadblockShape::kN,
      WarpShape::kM,
      DenseBranchTokens,
      VectorizeCallbackSharedStore,
      PartitionedEpilogueReleaseBarrier>;
  using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore,
      cutlass::epilogue::threadblock::VisitorAccFetch>;
  using Epilogue =
      cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<
          BaseEpilogue, Callbacks, kEpilogueStages>;
};

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_>
struct PairedDenseSparseGemmWithVisitor {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using FusionCallbacks = typename Epilogue::FusionCallbacks;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;
  static int const kSparse = Mma::kSparse;
  static int const kElementsPerElementE = Mma::kElementsPerElementE;

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile = 0;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    Bf16 const* dense_x = nullptr;
    Bf16 const* sparse_x = nullptr;
    int64_t const* dense_indices = nullptr;
    int64_t const* sparse_indices = nullptr;
    int dense_rows = 0;
    int sparse_rows = 0;
    int activation_k = 0;
    typename Mma::IteratorE::Params params_E;
    typename Mma::IteratorE::TensorRef ref_E;
    typename FusionCallbacks::Params output_op;
    cute::Shape<int32_t, int32_t, int32_t> problem_shape;

    CUTLASS_HOST_DEVICE Params() {}

    CUTLASS_HOST_DEVICE
    Params(
        cutlass::gemm::GemmCoord const& problem_size,
        cutlass::gemm::GemmCoord const& grid_tiled_shape,
        typename Mma::IteratorA::TensorRef ref_A,
        Bf16 const* dense_x,
        int dense_rows,
        Bf16 const* sparse_x,
        int sparse_rows,
        typename Mma::IteratorE::TensorRef ref_E,
        typename FusionCallbacks::Arguments output_op,
        int64_t const* dense_indices = nullptr,
        int64_t const* sparse_indices = nullptr)
        : problem_size(problem_size),
          grid_tiled_shape(grid_tiled_shape),
          swizzle_log_tile(ThreadblockSwizzle::get_log_tile(grid_tiled_shape)),
          params_A(ref_A.layout()),
          ref_A(ref_A),
          dense_x(dense_x),
          sparse_x(sparse_x),
          dense_indices(dense_indices),
          sparse_indices(sparse_indices),
          dense_rows(dense_rows),
          sparse_rows(sparse_rows),
          activation_k(problem_size.k()),
          params_E(ref_E.layout()),
          ref_E(ref_E),
          output_op(FusionCallbacks::to_underlying_arguments(
              problem_size, output_op, nullptr)),
          problem_shape(problem_size.m(), problem_size.n(), 1) {}
  };

  CUTLASS_DEVICE void operator()(
      Params const& params, SharedStorage& shared_storage) {
    ThreadblockSwizzle swizzle;
    cutlass::gemm::GemmCoord tile_offset =
        swizzle.get_tile_offset(params.swizzle_log_tile);
    if (params.grid_tiled_shape.m() <= tile_offset.m() ||
        params.grid_tiled_shape.n() <= tile_offset.n()) {
      return;
    }

    cutlass::MatrixCoord offset_A{
        tile_offset.m() * Mma::Shape::kM, 0};
    cutlass::MatrixCoord offset_B{
        0, tile_offset.n() * Mma::Shape::kN};
    cutlass::MatrixCoord offset_E{
        tile_offset.m() * Mma::Shape::kM, 0};
    int thread_idx = int(threadIdx.x);

    typename Mma::IteratorA iterator_A(
        params.params_A,
        params.ref_A.data(),
        {params.problem_size.m(), params.problem_size.k() / kSparse},
        thread_idx,
        offset_A);
    typename Mma::IteratorB iterator_B(
        params.dense_x,
        params.sparse_x,
        params.dense_rows,
        params.sparse_rows,
        params.activation_k,
        thread_idx,
        offset_B,
        params.dense_indices,
        params.sparse_indices);
    typename Mma::IteratorE iterator_E(
        params.params_E,
        params.ref_E.data(),
        {params.problem_size.m(),
         params.problem_size.k() / kSparse / kElementsPerElementE},
        thread_idx,
        offset_E);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % 32;
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC accumulators;
    accumulators.clear();
    int gemm_k_iterations =
        (params.problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
    constexpr int kBranchTokens = Mma::Shape::kN / 2;
    int branch_tile = tile_offset.n();
    bool dense_branch_valid =
        branch_tile * kBranchTokens < params.dense_rows;
    bool sparse_branch_valid =
        branch_tile * kBranchTokens < params.sparse_rows;
    mma(
        gemm_k_iterations,
        accumulators,
        iterator_A,
        iterator_B,
        iterator_E,
        accumulators,
        dense_branch_valid,
        sparse_branch_valid);

    Epilogue epilogue(
        params.output_op,
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx);
    epilogue(accumulators, tile_offset, params.problem_shape, thread_idx);
  }
};

// Persistent counterpart used by the ratio-specialized sidecar kernels.  A
// fixed, hardware-sized CTA pool grid-strides over the active
// (output-channel tile, token-route wave) tasks.  The launched block count is
// therefore independent of M and of the online dense:sparse split, while the
// first wave can still expose more CTAs than the output-channel tile count.
template <typename BaseParams_>
struct PersistentRoleTimingParams {
  using BaseParams = BaseParams_;
  BaseParams base;
  uint64_t* timing_output = nullptr;

  CUTLASS_HOST_DEVICE PersistentRoleTimingParams() = default;
  CUTLASS_HOST_DEVICE PersistentRoleTimingParams(
      BaseParams const& base, uint64_t* timing_output)
      : base(base), timing_output(timing_output) {}
};

// One diagnostic-only shared accumulator per physical warp.  The first ten
// uint64 values are updated by the mainloop; the persistent outer loop owns
// epilogue, complete-task, and task-count accumulation.  Keeping this payload
// in shared memory avoids a 13x64-bit per-thread register frame while
// preserving the existing thirteen-field global ABI.
struct PersistentSidecarWarpTiming {
  speculators::speclink::SidecarRoleTiming role;
  uint64_t epilogue_ns;
  uint64_t task_total_ns;
  uint64_t task_count;

  CUTLASS_DEVICE void clear() {
    role.clear();
    epilogue_ns = 0;
    task_total_ns = 0;
    task_count = 0;
  }
};
static_assert(sizeof(PersistentSidecarWarpTiming) ==
                  kSidecarRoleTimingFields * sizeof(uint64_t),
              "persistent role timing must preserve the 13-field ABI");

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_,
          int DenseWarpCount, bool EnableRoleTiming = false,
          bool OutputTileMajorTasks = false>
struct PersistentRatioDenseSparseGemmWithVisitor {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using OneTileKernel =
      PairedDenseSparseGemmWithVisitor<Mma, Epilogue, ThreadblockSwizzle>;
  using BaseParams = typename OneTileKernel::Params;
  using Params = std::conditional_t<
      EnableRoleTiming,
      PersistentRoleTimingParams<BaseParams>,
      BaseParams>;
  using FusionCallbacks = typename Epilogue::FusionCallbacks;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;
  static int const kSparse = Mma::kSparse;
  static int const kElementsPerElementE = Mma::kElementsPerElementE;
  static int const kWarpTokens = Mma::Shape::kN / WarpCount::kN;
  static int const kDenseBranchTokens = DenseWarpCount * kWarpTokens;
  static int const kSparseBranchTokens =
      Mma::Shape::kN - kDenseBranchTokens;

  static_assert(WarpCount::kM >= 1 && WarpCount::kK == 1,
                "persistent ratio kernel requires one warp-K partition");
  static_assert(DenseWarpCount >= 0 && DenseWarpCount <= WarpCount::kN,
                "persistent ratio kernel role count must fit warp-N");

  union ProductionSharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };
  struct RoleTimingSharedStorage {
    union {
      typename Mma::SharedStorage main_loop;
      typename Epilogue::SharedStorage epilogue;
    };
    PersistentSidecarWarpTiming warp_timing[WarpCount::kCount];
  };
  using SharedStorage = std::conditional_t<
      EnableRoleTiming, RoleTimingSharedStorage, ProductionSharedStorage>;
  static_assert(
      EnableRoleTiming ||
          sizeof(SharedStorage) == sizeof(ProductionSharedStorage),
      "production timing-disabled shared storage must remain unchanged");
  static_assert(
      !EnableRoleTiming ||
          sizeof(SharedStorage) >=
              sizeof(ProductionSharedStorage) +
                  WarpCount::kCount * sizeof(PersistentSidecarWarpTiming),
      "diagnostic shared storage must hold one timing payload per warp");

  CUTLASS_DEVICE void operator()(
      Params const& params, SharedStorage& shared_storage) {
    BaseParams const* base_params_pointer = nullptr;
    if constexpr (EnableRoleTiming) {
      base_params_pointer = &params.base;
    } else {
      base_params_pointer = &params;
    }
    BaseParams const& base_params = *base_params_pointer;
    int thread_idx = int(threadIdx.x);
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % 32;
    // CUTLASS linearizes warp-M first, followed by warp-N and warp-K.  BM32
    // has one warp-M group, so the old modulo expression happened to be
    // equivalent.  Use the full mapping so independent BM64 candidates assign
    // both output-feature warp groups to the same dense/sparse token role.
    int warp_idx_n = (warp_idx / WarpCount::kM) % WarpCount::kN;
    bool is_dense_warp = false;
    if constexpr (DenseWarpCount == WarpCount::kN) {
      is_dense_warp = true;
    } else if constexpr (DenseWarpCount > 0) {
      is_dense_warp = warp_idx_n < DenseWarpCount;
    }
    int branch_warp_rank = is_dense_warp
        ? warp_idx_n
        : warp_idx_n - DenseWarpCount;
    PersistentSidecarWarpTiming* warp_timing = nullptr;
    if constexpr (EnableRoleTiming) {
      warp_timing = &shared_storage.warp_timing[warp_idx];
      if (lane_idx == 0) {
        warp_timing->clear();
      }
    }

    // The legacy queue is route-wave-major and grid-strided.  For weights that
    // exceed L2, that traversal lets a complete pass over output features
    // evict a weight tile before the next route wave reaches it.  The optional
    // cache-blocked traversal gives each persistent CTA a balanced contiguous
    // interval of the output-tile-major queue.  Consecutive tasks therefore
    // keep the same small output-feature weight/metadata tile hot while route
    // waves advance.  This first weight-stationary ablation deliberately does
    // not change the mainloop or its HMMA/pack work; it isolates task-order
    // locality before adding multi-accumulator sidecar reuse.
    int output_tiles = base_params.grid_tiled_shape.m();
    int route_waves = base_params.grid_tiled_shape.n();
    int total_tasks = output_tiles * route_waves;
    int first_task = int(blockIdx.x);
    int tasks_for_block =
        first_task < total_tasks
        ? (total_tasks - 1 - first_task) / int(gridDim.x) + 1
        : 0;
    int task_stride = int(gridDim.x);
    if constexpr (OutputTileMajorTasks) {
      int quotient = total_tasks / int(gridDim.x);
      int remainder = total_tasks - quotient * int(gridDim.x);
      tasks_for_block = quotient + (int(blockIdx.x) < remainder ? 1 : 0);
      first_task = int(blockIdx.x) * quotient +
                   (int(blockIdx.x) < remainder
                        ? int(blockIdx.x)
                        : remainder);
      task_stride = 1;
    }
    for (int local_task = 0; local_task < tasks_for_block; ++local_task) {
        int task_idx = first_task + local_task * task_stride;
        uint64_t task_start = 0;
        if constexpr (EnableRoleTiming) {
          if (lane_idx == 0) {
            task_start =
                speculators::speclink::sidecar_globaltimer_ns();
          }
        }
        int tile_m = 0;
        int tile_n = 0;
        if constexpr (OutputTileMajorTasks) {
          tile_m = task_idx / route_waves;
          tile_n = task_idx - tile_m * route_waves;
        } else {
          tile_m = task_idx % output_tiles;
          tile_n = task_idx / output_tiles;
        }
        cutlass::gemm::GemmCoord tile_offset{tile_m, tile_n, 0};
        cutlass::MatrixCoord offset_A{tile_m * Mma::Shape::kM, 0};
        cutlass::MatrixCoord offset_B{0, tile_n * Mma::Shape::kN};
        cutlass::MatrixCoord offset_E{tile_m * Mma::Shape::kM, 0};

        typename Mma::IteratorA iterator_A(
            base_params.params_A,
            base_params.ref_A.data(),
            {base_params.problem_size.m(),
             base_params.problem_size.k() / kSparse},
            thread_idx,
            offset_A);
        typename Mma::IteratorB iterator_B(
            base_params.dense_x,
            base_params.sparse_x,
            base_params.dense_rows,
            base_params.sparse_rows,
            base_params.activation_k,
            thread_idx,
            offset_B,
            base_params.dense_indices,
            base_params.sparse_indices);
        typename Mma::IteratorE iterator_E(
            base_params.params_E,
            base_params.ref_E.data(),
            {base_params.problem_size.m(),
             base_params.problem_size.k() / kSparse /
                 kElementsPerElementE},
            thread_idx,
            offset_E);

        int branch_capacity = is_dense_warp
            ? kDenseBranchTokens
            : kSparseBranchTokens;
        int branch_rows = is_dense_warp
            ? base_params.dense_rows
            : base_params.sparse_rows;
        int branch_row = tile_n * branch_capacity +
                         branch_warp_rank * kWarpTokens;
        bool warp_branch_valid = branch_row < branch_rows;
        bool sparse_branch_any_valid =
            tile_n * kSparseBranchTokens < base_params.sparse_rows;

        Mma mma(shared_storage.main_loop,
                thread_idx, warp_idx, lane_idx);
        typename Mma::FragmentC accumulators;
        accumulators.clear();
        int gemm_k_iterations =
            (base_params.problem_size.k() + Mma::Shape::kK - 1) /
            Mma::Shape::kK;
        mma(
            gemm_k_iterations,
            accumulators,
            iterator_A,
            iterator_B,
            iterator_E,
            accumulators,
            warp_branch_valid,
            sparse_branch_any_valid,
            base_params.grid_tiled_shape.k(),
            EnableRoleTiming ? &warp_timing->role : nullptr);

        uint64_t epilogue_start = 0;
        if constexpr (EnableRoleTiming) {
          if (lane_idx == 0) {
            epilogue_start =
                speculators::speclink::sidecar_globaltimer_ns();
          }
        }
        Epilogue epilogue(
            base_params.output_op,
            shared_storage.epilogue,
            thread_idx,
            warp_idx,
            lane_idx);
        epilogue(
            accumulators,
            tile_offset,
            base_params.problem_shape,
            thread_idx);
        if constexpr (EnableRoleTiming) {
          if (lane_idx == 0) {
            warp_timing->epilogue_ns +=
                speculators::speclink::sidecar_globaltimer_ns() -
                epilogue_start;
          }
        }

        // Epilogue and mainloop alias the same union.  All threads must finish
        // the current routed tile before the persistent loop reuses it.
        __syncthreads();
        if constexpr (EnableRoleTiming) {
          if (lane_idx == 0) {
            warp_timing->task_total_ns +=
                speculators::speclink::sidecar_globaltimer_ns() - task_start;
            ++warp_timing->task_count;
          }
        }
    }

    if constexpr (EnableRoleTiming) {
      if (lane_idx == 0 && params.timing_output != nullptr) {
        uint64_t* out = params.timing_output +
            (uint64_t(blockIdx.x) * WarpCount::kCount + warp_idx) *
                kSidecarRoleTimingFields;
        out[0] = warp_timing->role.mainloop_ns;
        out[1] = warp_timing->role.dense_mma_ns;
        out[2] = warp_timing->role.pack_ns;
        out[3] = warp_timing->role.sparse_mma_ns;
        out[4] = warp_timing->role.async_wait_ns;
        out[5] = warp_timing->role.role_barrier_ns;
        out[6] = warp_timing->role.cta_barrier_ns;
        out[7] = warp_timing->epilogue_ns;
        out[8] = warp_timing->task_total_ns;
        out[9] = warp_timing->task_count;
        out[10] = warp_timing->role.stage_issue_ns;
        out[11] = warp_timing->role.activation_route_copy_issue_ns;
        out[12] = warp_timing->role.nonproducer_pack_dispatch_ns;
      }
    }
  }
};

// Independent priority-1 kernel: one CTA completes two adjacent D1:S3 route
// waves while loading each canonical weight/metadata K64 tile and constructing
// its sparse sidecar only once.  This is deliberately a separate kernel type;
// the legacy, output-tile-major-only, and auto-selected binaries above retain
template <int DenseWarpCount, bool ParallelConsumers = true,
          bool EnableRoleTiming = false, bool CacheRouteRows = false,
          bool CacheRouteRowBases = false, int TotalWarpCount = 4,
          bool CompactSelectorLoaders = false,
          bool VectorizeCallbackSharedStore = false,
          int ThreadblockM = 32,
          bool OutputTileMajorTasks = false,
          bool ExplicitPackProducerBranch = true,
          bool PartitionedEpilogueReleaseBarrier = true,
          bool PrebroadcastRouteRowPointers = false,
          bool AssumeFullTiles = false>
struct PersistentSidecarConfiguration {
  static_assert(TotalWarpCount >= 2,
                "persistent sidecar requires at least two warps");
  static_assert(DenseWarpCount >= 0 && DenseWarpCount <= TotalWarpCount,
                "persistent sidecar role count must fit the CTA");
  static_assert(ThreadblockM >= 32 && !(ThreadblockM % 32),
                "persistent sidecar BM must be a positive multiple of WarpM");
  static bool const kEnableRoleTiming = EnableRoleTiming;
  static bool const kParallelConsumers = ParallelConsumers;
  static bool const kCacheRouteRows = CacheRouteRows;
  static bool const kCacheRouteRowBases = CacheRouteRowBases;
  static bool const kCompactSelectorLoaders = CompactSelectorLoaders;
  static bool const kVectorizeCallbackSharedStore =
      VectorizeCallbackSharedStore;
  static bool const kOutputTileMajorTasks = OutputTileMajorTasks;
  static bool const kExplicitPackProducerBranch =
      ExplicitPackProducerBranch;
  static bool const kPartitionedEpilogueReleaseBarrier =
      PartitionedEpilogueReleaseBarrier;
  static bool const kPrebroadcastRouteRowPointers =
      PrebroadcastRouteRowPointers;
  static bool const kAssumeFullTiles = AssumeFullTiles;
  static int const kDenseWarpCount = DenseWarpCount;
  static int const kTotalWarpCount = TotalWarpCount;
  static int const kThreadblockM = ThreadblockM;
  static int const kPersistentBlocksPerSm = TotalWarpCount == 4 ? 2 : 1;
  using ThreadblockShape =
      cutlass::gemm::GemmShape<ThreadblockM, TotalWarpCount * 32, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  static int const kDenseBranchTokens = DenseWarpCount * WarpShape::kN;
  static int const kSparseBranchTokens =
      ThreadblockShape::kN - kDenseBranchTokens;
  // CUTLASS's default sparse core assigns its tiny compressed-A loader over
  // every CTA thread.  At eight warps the 32x32 compressed tile has fewer
  // vector accesses than threads and the stock thread map becomes invalid.
  // Only the warp operator/policy and A/E layouts are needed from this core;
  // keep its N extent at four warp groups, while matching the real BM so its
  // metadata iterator covers every output-feature row.  The actual wide B
  // loader below still comes from the dense core, and the mainloop derives its
  // real WarpCount from ThreadblockShape and the sparse warp operator.
  using SparsePolicyShape =
      cutlass::gemm::GemmShape<ThreadblockM, 128, 64>;
  using SparseCore = cutlass::gemm::threadblock::DefaultSparseMmaCore<
      SparsePolicyShape,
      WarpShape,
      SparseInstructionShape,
      Bf16,
      LayoutA,
      Bf16,
      LayoutB,
      float,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassTensorOp,
      kFusedStages,
      cutlass::arch::OpMultiplyAdd,
      false,
      cutlass::arch::CacheOperation::Global,
      cutlass::arch::CacheOperation::Global>;
  using DenseCore = cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape,
      WarpShape,
      DenseInstructionShape,
      Bf16,
      LayoutA,
      Bf16,
      LayoutB,
      float,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassTensorOp,
      kFusedStages,
      cutlass::arch::OpMultiplyAdd,
      false,
      cutlass::arch::CacheOperation::Global,
      cutlass::arch::CacheOperation::Global>;
  using ElementE = typename SparseCore::ElementE;
  using GmemLayoutE = typename SparseCore::GmemLayoutE;
  using IteratorA = DenseCanonicalIteratorA<
      typename DenseCore::IteratorThreadMapA,
      ElementE,
      GmemLayoutE,
      ThreadblockShape,
      AssumeFullTiles>;
  using DenseSmemIteratorA =
      cutlass::transform::threadblock::RegularTileAccessIterator<
          cutlass::MatrixShape<
              ThreadblockShape::kM, ThreadblockShape::kK>,
          Bf16,
          typename DenseCore::SmemLayoutA,
          0,
          typename DenseCore::IteratorThreadMapA>;
  using SharedSmemIteratorB =
      cutlass::transform::threadblock::RegularTileAccessIterator<
          cutlass::MatrixShape<
              ThreadblockShape::kK, ThreadblockShape::kN>,
          Bf16,
          typename SparseCore::SmemLayoutB,
          1,
          typename DenseCore::IteratorThreadMapB>;
  using IteratorB = PairedInputIteratorB<
      typename DenseCore::IteratorThreadMapB,
      ThreadblockShape,
      kDenseBranchTokens,
      CacheRouteRows,
      CacheRouteRowBases,
      PrebroadcastRouteRowPointers,
      AssumeFullTiles>;
  using IteratorE =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<
              ThreadblockShape::kM,
              ThreadblockShape::kK / SparseCore::kSparse /
                  SparseCore::kElementsPerElementE>,
          ElementE,
          GmemLayoutE,
          1,
          typename SparseCore::IteratorThreadMapE,
          cutlass::Array<
              ElementE, 128 / cutlass::sizeof_bits<ElementE>::value>>;
  using ThreadblockMma =
      speculators::speclink::DenseBaseFusedDenseSparseSidecarMma<
          ThreadblockShape,
          IteratorA,
          DenseSmemIteratorA,
          cutlass::arch::CacheOperation::Global,
          IteratorB,
          SharedSmemIteratorB,
          cutlass::arch::CacheOperation::Global,
          float,
          cutlass::layout::RowMajor,
          IteratorE,
          typename SparseCore::SmemIteratorE,
          cutlass::arch::CacheOperation::Global,
          typename SparseCore::MmaPolicy,
          typename DenseCore::MmaPolicy,
          typename DenseCore::SmemLayoutA,
          kFusedStages,
          DenseWarpCount,
          true,
          ParallelConsumers,
          EnableRoleTiming,
          CompactSelectorLoaders,
          ExplicitPackProducerBranch>;
  using Visitor = PairedVisitorEpilogue<
      ThreadblockShape,
      WarpShape,
      ThreadblockMma,
      kDenseBranchTokens,
      VectorizeCallbackSharedStore,
      PartitionedEpilogueReleaseBarrier>;
  using OutputStore = typename Visitor::OutputStore;
  using Callbacks = typename Visitor::Callbacks;
  using Epilogue = typename Visitor::Epilogue;
  using Swizzle =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  using Kernel = PersistentRatioDenseSparseGemmWithVisitor<
      ThreadblockMma, Epilogue, Swizzle, DenseWarpCount,
      EnableRoleTiming, OutputTileMajorTasks>;

  static_assert(
      Kernel::kThreadCount ==
          (ThreadblockM / WarpShape::kM) * TotalWarpCount * 32,
                "persistent sidecar thread count must match its warp count");
  static_assert(
      cutlass::platform::is_same<
          typename DenseCore::SmemLayoutB,
          typename SparseCore::SmemLayoutB>::value,
      "persistent dense and sparse warps must share B layout");
};

// Formal separate descendants.  Keep two independent pure-role kernels and
// the output-major task order, but apply the same task-local activation-pointer
// prebroadcast and exact M2048 full-tile predicate specialization selected by
// the fused kernel.  No packed sparse weight or global intermediate is added.
using PersistentSidecarWidePureD8S0Bm64PrebroadcastFullTilesOutputMajor =
    PersistentSidecarConfiguration<
        8, true, false, false, true, 8, false, false, 64,
        true, true, true, true, true>;
using PersistentSidecarWidePureD0S8Bm64PrebroadcastFullTilesOutputMajor =
    PersistentSidecarConfiguration<
        0, true, false, false, true, 8, false, false, 64,
        true, true, true, true, true>;
template <typename Config>
void set_dynamic_smem_attribute() {
  using Kernel = typename Config::Kernel;
  int shared_bytes = int(sizeof(typename Kernel::SharedStorage));
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
enum class PersistentSidecarBranch : int64_t {
  kDense = 0,
  kSparse = 1,
};
template <typename Config>
void launch_persistent_sidecar_kernel_config(
    torch::Tensor const& x,
    int dense_rows,
    int sparse_rows,
    torch::Tensor const& dense_weight,
    torch::Tensor const& reordered_metadata,
    Bf16* routed_output,
    int64_t const* dense_indices,
    int64_t const* sparse_indices,
    cudaStream_t stream,
    bool enable_dense = true,
    bool enable_sparse = true,
    uint64_t* timing_output = nullptr,
    int persistent_blocks_override = 0) {
  using Kernel = typename Config::Kernel;
  using ThreadblockShape = typename Config::ThreadblockShape;
  using IteratorA = typename Config::IteratorA;
  using IteratorE = typename Config::IteratorE;
  using ElementE = typename Config::ElementE;
  using GmemLayoutE = typename Config::GmemLayoutE;
  using OutputStore = typename Config::OutputStore;
  using Callbacks = typename Config::Callbacks;
  using Swizzle = typename Config::Swizzle;

  int n = int(dense_weight.size(0));
  int k = int(x.size(1));
  int active_dense_rows = enable_dense ? dense_rows : 0;
  int active_sparse_rows = enable_sparse ? sparse_rows : 0;
  int dense_tiles = 0;
  int sparse_tiles = 0;
  if constexpr (Config::kDenseBranchTokens > 0) {
    dense_tiles =
        (active_dense_rows + Config::kDenseBranchTokens - 1) /
        Config::kDenseBranchTokens;
  }
  if constexpr (Config::kSparseBranchTokens > 0) {
    sparse_tiles =
        (active_sparse_rows + Config::kSparseBranchTokens - 1) /
        Config::kSparseBranchTokens;
  }
  int pair_tiles = std::max(dense_tiles, sparse_tiles);
  cutlass::gemm::GemmCoord problem_size{
      n, pair_tiles * ThreadblockShape::kN, k};
  Swizzle swizzle;
  auto tiled_shape = swizzle.get_tiled_shape(
      problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN,
       ThreadblockShape::kK},
      1);
  typename IteratorA::TensorRef ref_a(
      reinterpret_cast<Bf16*>(dense_weight.data_ptr()), LayoutA(k));
  typename IteratorE::TensorRef ref_e(
      reinterpret_cast<ElementE*>(reordered_metadata.data_ptr()),
      GmemLayoutE::packed({n, k / 16}));
  typename OutputStore::Arguments output_args{
      nullptr,
      nullptr,
      active_dense_rows,
      active_sparse_rows,
      routed_output,
      dense_indices,
      sparse_indices};
  typename Callbacks::Arguments callback_args{{}, output_args};
  typename Kernel::BaseParams base_params(
      problem_size,
      tiled_shape,
      ref_a,
      reinterpret_cast<Bf16 const*>(x.data_ptr()),
      active_dense_rows,
      reinterpret_cast<Bf16 const*>(x.data_ptr()),
      active_sparse_rows,
      ref_e,
      callback_args,
      dense_indices,
      sparse_indices);

  set_dynamic_smem_attribute<Config>();
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  // Keep the launch geometry fixed across M and every online dense:sparse
  // split.  Narrow four-warp kernels use two resident CTAs per SM.  The wide
  // D4:S4 candidate uses one eight-warp CTA per SM: it retains the same eight
  // resident warps while fitting its larger B stage in shared memory.
  int persistent_blocks = std::max(
      1, prop.multiProcessorCount * Config::kPersistentBlocksPerSm);
  if (persistent_blocks_override > 0) {
    persistent_blocks = persistent_blocks_override;
  }
  dim3 grid(unsigned(persistent_blocks), 1, 1);
  dim3 block(Kernel::kThreadCount, 1, 1);
  int shared_bytes = int(sizeof(typename Kernel::SharedStorage));
  if constexpr (Config::kEnableRoleTiming) {
    typename Kernel::Params params(base_params, timing_output);
    cutlass::Kernel<Kernel><<<grid, block, shared_bytes, stream>>>(params);
  } else {
    cutlass::Kernel<Kernel><<<grid, block, shared_bytes, stream>>>(base_params);
  }
}
void launch_persistent_sidecar_wide_pure_bm64_prebroadcast_full_tiles_branch_kernel(
    torch::Tensor const& x,
    int dense_rows,
    int sparse_rows,
    torch::Tensor const& dense_weight,
    torch::Tensor const& reordered_metadata,
    Bf16* routed_output,
    int64_t const* dense_indices,
    int64_t const* sparse_indices,
    cudaStream_t stream,
    PersistentSidecarBranch branch,
    int persistent_blocks) {
  TORCH_CHECK(persistent_blocks > 0,
              "pure full-tile BM64 branch blocks must be positive");
  if (branch == PersistentSidecarBranch::kDense) {
    launch_persistent_sidecar_kernel_config<
        PersistentSidecarWidePureD8S0Bm64PrebroadcastFullTilesOutputMajor>(
        x, dense_rows, sparse_rows, dense_weight, reordered_metadata,
        routed_output, dense_indices, sparse_indices, stream, true, false,
        nullptr, persistent_blocks);
    return;
  }
  launch_persistent_sidecar_kernel_config<
      PersistentSidecarWidePureD0S8Bm64PrebroadcastFullTilesOutputMajor>(
      x, dense_rows, sparse_rows, dense_weight, reordered_metadata,
      routed_output, dense_indices, sparse_indices, stream, false, true,
      nullptr, persistent_blocks);
}
template <typename Config>
std::vector<int64_t> attributes() {
  using Kernel = typename Config::Kernel;
  set_dynamic_smem_attribute<Config>();
  cudaFuncAttributes attr{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(
      &attr, reinterpret_cast<void const*>(cutlass::Kernel<Kernel>)));
  int shared_bytes = int(sizeof(typename Kernel::SharedStorage));
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, cutlass::Kernel<Kernel>, Kernel::kThreadCount, shared_bytes));
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int64_t occupancy = prop.maxThreadsPerMultiProcessor > 0
      ? int64_t(active) * Kernel::kThreadCount * 100 /
          prop.maxThreadsPerMultiProcessor
      : 0;
  return {
      int64_t(attr.numRegs),
      int64_t(shared_bytes),
      int64_t(attr.localSizeBytes),
      int64_t(attr.maxThreadsPerBlock),
      int64_t(active),
      occupancy,
  };
}
torch::Tensor
persistent_sidecar_wide_pure_bm64_prebroadcast_full_tiles_branch_routed_forward_out_impl(
    torch::Tensor x,
    torch::Tensor dense_indices,
    torch::Tensor sparse_indices,
    torch::Tensor dense_weight,
    torch::Tensor reordered_metadata,
    torch::Tensor output,
    PersistentSidecarBranch branch,
    int persistent_blocks) {
  c10::cuda::CUDAGuard guard(x.device());
  int device = x.get_device();
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  TORCH_CHECK(
      persistent_blocks <= prop.multiProcessorCount,
      "pure full-tile BM64 branch blocks must not exceed the SM count");
  int dense_rows = int(dense_indices.numel());
  int sparse_rows = int(sparse_indices.numel());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  launch_persistent_sidecar_wide_pure_bm64_prebroadcast_full_tiles_branch_kernel(
      x,
      dense_rows,
      sparse_rows,
      dense_weight,
      reordered_metadata,
      reinterpret_cast<Bf16*>(output.data_ptr()),
      dense_indices.data_ptr<int64_t>(),
      sparse_indices.data_ptr<int64_t>(),
      stream,
      branch,
      persistent_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
}  // namespace

torch::Tensor old_concurrent_branch_forward_out_cuda(
    torch::Tensor x,
    torch::Tensor dense_indices,
    torch::Tensor sparse_indices,
    torch::Tensor dense_weight,
    torch::Tensor reordered_metadata,
    torch::Tensor output,
    int64_t branch,
    int64_t persistent_blocks) {
  TORCH_CHECK(branch == 0 || branch == 1,
              "branch must be 0 (dense) or 1 (sparse)");
  TORCH_CHECK(persistent_blocks > 0 &&
                  persistent_blocks <= std::numeric_limits<int>::max(),
              "persistent_blocks must be a positive int");
  return persistent_sidecar_wide_pure_bm64_prebroadcast_full_tiles_branch_routed_forward_out_impl(
      std::move(x), std::move(dense_indices), std::move(sparse_indices),
      std::move(dense_weight), std::move(reordered_metadata),
      std::move(output), static_cast<PersistentSidecarBranch>(branch),
      int(persistent_blocks));
}

std::vector<int64_t> old_concurrent_kernel_attributes_cuda(int64_t branch) {
  TORCH_CHECK(branch == 0 || branch == 1,
              "branch must be 0 (dense) or 1 (sparse)");
  return branch == 0
      ? attributes<PersistentSidecarWidePureD8S0Bm64PrebroadcastFullTilesOutputMajor>()
      : attributes<PersistentSidecarWidePureD0S8Bm64PrebroadcastFullTilesOutputMajor>();
}
