/***************************************************************************************************
 * CTA-local dense-to-2:4 sidecar for the fused dense/sparse CUTLASS mainloop.
 *
 * Global memory contains only the canonical dense BF16 A matrix and CUTLASS's
 * reordered NK/8 metadata.  Once a dense-A stage and its metadata have arrived
 * in shared memory, sparse producer warps cooperatively pack disjoint K32
 * subtiles of the M32 x WarpK slice.  Every selector and value is still
 * produced exactly once.  The packed values live in a transient, per-stage
 * shared-memory sidecar using the stock sparse-A crosswise layout.  All sparse
 * warp-N consumers then use SparseMmaTensorOp::IteratorA and ldmatrix.
 *
 * ParallelConsumers selects the stage schedule.  The default overlaps dense
 * MMA with the unique sidecar producer and uses a sparse-warp-only named
 * barrier to release the sparse consumers as soon as packing finishes.  The
 * serial-consumer ablation uses a CTA barrier so sparse MMA starts only after
 * all dense MMA and packing have completed.
 **************************************************************************************************/
#pragma once

#include <cstdint>

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

namespace speculators {
namespace speclink {

// Diagnostic-only timing payload.  Persistent timing kernels place one of
// these objects per physical warp in shared memory; the mainloop's lane 0
// accumulates directly into that object instead of carrying ten uint64 values
// in every thread's register frame.  Production instantiations leave
// EnableRoleTiming=false, so ptxas removes the pointer and every timer read.
// %globaltimer is shared across SMs and reports nanoseconds on the supported
// CUDA targets, which lets the persistent kernel aggregate multiple routed
// tasks without trying to turn profiler stall percentages into elapsed time.
struct SidecarRoleTiming {
  uint64_t mainloop_ns;
  uint64_t dense_mma_ns;
  uint64_t pack_ns;
  uint64_t sparse_mma_ns;
  uint64_t async_wait_ns;
  uint64_t role_barrier_ns;
  uint64_t cta_barrier_ns;
  // Diagnostic-only fine split of the old mainloop residual.  The activation
  // interval is nested inside stage_issue_ns and must never be added to it.
  uint64_t stage_issue_ns;
  uint64_t activation_route_copy_issue_ns;
  uint64_t nonproducer_pack_dispatch_ns;

  CUTLASS_DEVICE void clear() {
    mainloop_ns = 0;
    dense_mma_ns = 0;
    pack_ns = 0;
    sparse_mma_ns = 0;
    async_wait_ns = 0;
    role_barrier_ns = 0;
    cta_barrier_ns = 0;
    stage_issue_ns = 0;
    activation_route_copy_issue_ns = 0;
    nonproducer_pack_dispatch_ns = 0;
  }
};

CUTLASS_DEVICE uint64_t sidecar_globaltimer_ns() {
  uint64_t value;
  asm volatile(
      "mov.u64 %0, %%globaltimer;" : "=l"(value) : : "memory");
  return value;
}

/***************************************************************************************************
 * Packs the sparse-A warp tile owned by one sparse warp.
 *
 * The mapping is specialized to the BF16 HMMA.SP m16n8k32 path used by the
 * current fine fused kernel:
 *
 *   lane             -> one logical A row in the M32 warp tile
 *   owned K32 group  -> two K16 selector words
 *   each K16 word    -> 16 dense BF16 values -> 8 compressed BF16 values
 *
 * Dense LDS.128 and sparse STS.128 use CUTLASS's crosswise layout functions,
 * rather than a hand-written physical address, so the sidecar is directly
 * consumable by SparseMmaTensorOp::IteratorA.
 **************************************************************************************************/
template <typename SparseOperator_, typename DenseSmemLayoutA_,
          typename ThreadblockShape_, typename WarpCount_, int Stages,
          bool CompactSelectorLoaders = false>
class WarpOwnedSparse24SidecarPacker {
 public:
  using Operator = SparseOperator_;
  using Element = typename Operator::ElementA;
  using ElementE = typename Operator::ElementE;
  using DenseSmemLayoutA = DenseSmemLayoutA_;
  using SparseSmemLayoutA = typename Operator::LayoutA;
  using SmemLayoutE = typename Operator::LayoutE;
  using ThreadblockShape = ThreadblockShape_;
  using WarpCount = WarpCount_;
  using DenseTensorRef = cutlass::TensorRef<Element, DenseSmemLayoutA>;
  using SparseTensorRef = cutlass::TensorRef<Element, SparseSmemLayoutA>;
  using MetadataTensorRef = cutlass::TensorRef<ElementE, SmemLayoutE>;

  static int const kSparse = Operator::kSparse;
  static int const kInstructionK = Operator::Policy::MmaShape::kK;
  static int const kWarpKIterations =
      Operator::Shape::kK / kInstructionK;
  static int const kK32GroupsPerStage =
      ThreadblockShape::kK / kInstructionK;

 private:
  Element const* dense_pointer_ = nullptr;
  Element* sparse_pointer_ = nullptr;
  ElementE const* metadata_pointer_ = nullptr;
  DenseSmemLayoutA dense_layout_{};
  SparseSmemLayoutA sparse_layout_{};
  SmemLayoutE metadata_layout_{};
  int lane_ = 0;

  struct SelectorPair {
    uint16_t first;
    uint16_t second;
  };

  // A selector nibble contains the two two-bit dense positions retained from
  // one K4 group.  PRMT copies the two selected BF16 values into the packed
  // 32-bit representation expected by sparse IteratorA/HMMA.SP.
  CUTLASS_DEVICE static uint32_t pack_selected_bf16(
      uint32_t packed01, uint32_t packed23, uint8_t selector) {
    uint32_t byte0 = uint32_t(selector & 0x3u) << 1;
    uint32_t byte2 = uint32_t((selector >> 2) & 0x3u) << 1;
    uint32_t permute =
        byte0 | ((byte0 + 1) << 4) | (byte2 << 8) | ((byte2 + 1) << 12);
    return __byte_perm(packed01, packed23, permute);
  }

  // CUTLASS's reordered E tile places the two K16 words for rows r and r+8
  // in four consecutive uint16_t slots.  The legacy mapping assigns those 16
  // aligned LDS.64 requests to lanes {0..7, 16..23}.  Although their physical
  // addresses cover every shared-memory bank exactly once, SM120 reports two
  // wavefronts because the active lanes are split across both half-warps.
  //
  // CompactSelectorLoaders keeps the exact same 16 physical addresses but
  // assigns them to contiguous lanes 0..15.  The consumer shuffle below maps
  // every logical row back to its original selector pair, so this is an
  // isolated metadata-load scheduling ablation, not a new weight format or a
  // change to the sparse sidecar layout.
  CUTLASS_DEVICE SelectorPair load_selector_pair(
      int warp_row_offset, int owned_k32, int read_stage) const {
    uint2 local{0u, 0u};

    bool is_loader = CompactSelectorLoaders ? lane_ < 16
                                            : (lane_ & 8) == 0;
    if (is_loader) {
      int loader_pair = CompactSelectorLoaders
          ? lane_
          : (lane_ & 7) + 8 * (lane_ >> 4);
      int physical_row = 2 * warp_row_offset +
          8 * (loader_pair & 7) + 4 * (loader_pair >> 3);
      int physical_col = read_stage * kK32GroupsPerStage + owned_k32;
      auto offset = metadata_layout_(
          cutlass::MatrixCoord(physical_row, physical_col));
      cutlass::arch::shared_load<8>(
          &local,
          cutlass::arch::cutlass_get_smem_pointer(
              metadata_pointer_ + offset));
    }

    int owner_lane = CompactSelectorLoaders
        ? (lane_ & 7) + 8 * (lane_ >> 4)
        : lane_ & ~8;
    uint32_t words_first =
        __shfl_sync(0xffffffffu, local.x, owner_lane);
    uint32_t words_second =
        __shfl_sync(0xffffffffu, local.y, owner_lane);
    int shift = (lane_ & 8) ? 16 : 0;
    return SelectorPair{
        uint16_t(words_first >> shift),
        uint16_t(words_second >> shift)};
  }

  CUTLASS_DEVICE void pack_k16(
      int logical_row, int dense_k, int compressed_k,
      uint16_t selector_word) const {
    uint4 dense_lo{0u, 0u, 0u, 0u};
    uint4 dense_hi{0u, 0u, 0u, 0u};

    auto dense_lo_offset = dense_layout_(
        cutlass::MatrixCoord(logical_row, dense_k));
    auto dense_hi_offset = dense_layout_(
        cutlass::MatrixCoord(logical_row, dense_k + 8));
    cutlass::arch::shared_load<16>(
        &dense_lo,
        cutlass::arch::cutlass_get_smem_pointer(
            dense_pointer_ + dense_lo_offset));
    cutlass::arch::shared_load<16>(
        &dense_hi,
        cutlass::arch::cutlass_get_smem_pointer(
            dense_pointer_ + dense_hi_offset));

    uint4 packed;
    packed.x = pack_selected_bf16(
        dense_lo.x, dense_lo.y, uint8_t(selector_word & 0xfu));
    packed.y = pack_selected_bf16(
        dense_lo.z, dense_lo.w, uint8_t((selector_word >> 4) & 0xfu));
    packed.z = pack_selected_bf16(
        dense_hi.x, dense_hi.y, uint8_t((selector_word >> 8) & 0xfu));
    packed.w = pack_selected_bf16(
        dense_hi.z, dense_hi.w, uint8_t((selector_word >> 12) & 0xfu));

    auto sparse_offset = sparse_layout_(
        cutlass::MatrixCoord(logical_row, compressed_k));
    cutlass::arch::shared_store<16>(
        cutlass::arch::cutlass_get_smem_pointer(
            sparse_pointer_ + sparse_offset),
        &packed);
  }

 public:
  static_assert(Stages >= 2,
                "sidecar pipeline requires at least double buffering");
  static_assert(kSparse == 2, "sidecar supports structured 2:4 only");
  static_assert(sizeof(Element) == 2,
                "sidecar vector mapping requires 16-bit A elements");
  static_assert(sizeof(ElementE) == 2,
                "sidecar metadata mapping requires uint16 metadata words");
  static_assert(Operator::kElementsPerElementE == 8,
                "sidecar expects one uint16 word per row and dense K16");
  static_assert(Operator::kInterleaved == 2,
                "sidecar expects CUTLASS's 2x interleaved metadata reorder");
  static_assert(
      cutlass::platform::is_same<
          SmemLayoutE, cutlass::layout::ColumnMajor>::value,
      "sidecar metadata physical mapping requires column-major shared E");
  static_assert(Operator::Shape::kM == 32,
                "one producer warp must own exactly 32 A rows");
  static_assert(Operator::MmaIterations::kRow == 2,
                "sidecar metadata mapping requires two M16 instructions");
  static_assert(kInstructionK == 32,
                "sidecar mapping requires HMMA.SP K32 instructions");
  static_assert(!(Operator::Shape::kK % kInstructionK),
                "warp K must be divisible by the sparse instruction K");
  static_assert(!(ThreadblockShape::kK % kInstructionK),
                "threadblock K must be divisible by 32");
  static_assert(WarpCount::kN >= 2,
                "sidecar requires at least one dense and one sparse warp-N group");

  CUTLASS_DEVICE
  WarpOwnedSparse24SidecarPacker(
      DenseTensorRef dense_ref, SparseTensorRef sparse_ref,
      MetadataTensorRef metadata_ref, int lane)
      : dense_pointer_(dense_ref.data()),
        sparse_pointer_(sparse_ref.data()),
        metadata_pointer_(metadata_ref.data()),
        dense_layout_(dense_ref.layout()),
        sparse_layout_(sparse_ref.layout()),
        metadata_layout_(metadata_ref.layout()),
        lane_(lane) {}

  // ProducerRank among ProducerCount sparse warps owns a disjoint subset of
  // K32 instruction tiles.  With WarpK=64, two sparse producer warps each pack
  // one half; a sole sparse warp packs both.  No selector work is duplicated.
  template <int ProducerCount>
  CUTLASS_DEVICE void pack_stage(
      int warp_idx_m, int warp_idx_k, int read_stage,
      int producer_rank) const {
    static_assert(ProducerCount >= 1,
                  "sidecar requires at least one producer warp");
    int warp_row_offset = warp_idx_m * Operator::Shape::kM;
    int logical_row = warp_row_offset + lane_;

    CUTLASS_PRAGMA_UNROLL
    for (int k_group = 0; k_group < kWarpKIterations; ++k_group) {
      if (k_group % ProducerCount != producer_rank) {
        continue;
      }
      int owned_k32 = warp_idx_k * kWarpKIterations + k_group;
      SelectorPair selectors =
          load_selector_pair(warp_row_offset, owned_k32, read_stage);

      int dense_k_base =
          read_stage * ThreadblockShape::kK + owned_k32 * kInstructionK;
      int compressed_k_base =
          read_stage * (ThreadblockShape::kK / kSparse) +
          owned_k32 * (kInstructionK / kSparse);
      pack_k16(
          logical_row, dense_k_base, compressed_k_base, selectors.first);
      pack_k16(
          logical_row, dense_k_base + 16, compressed_k_base + 8,
          selectors.second);
    }

    // The caller immediately publishes this stage with either the sparse-only
    // named barrier or the serial-ablation CTA barrier.  A second producer-warp
    // barrier here would be redundant.
  }
};

/***************************************************************************************************
 * Mixed dense/sparse CTA mainloop with a transient compressed-A sidecar.
 *
 * DenseWarpCount is the number of leading warp-N groups assigned to dense
 * tokens.  The remaining warp-N groups are sparse.  For a ratio geometry such
 * as TB32x128x64 / Warp32x32x64 this permits dense:sparse warp ratios 1:3,
 * 2:2, and 3:1 without recompiling the packer mapping.
 *
 * Sparse warp-N groups cooperatively pack disjoint K32 subtiles for each
 * (warp-M, warp-K) owner.  No selector work is repeated across token warps.
 * ParallelConsumers=true overlaps dense MMA with the producer and releases
 * sparse consumers through a sparse-warp-only named barrier.  The false
 * ablation uses a CTA barrier so sparse MMA begins after dense MMA and packing.
 **************************************************************************************************/
template <typename Shape_, typename IteratorA_, typename SmemIteratorA_,
          cutlass::arch::CacheOperation::Kind CacheOpA,
          typename IteratorB_, typename SmemIteratorB_,
          cutlass::arch::CacheOperation::Kind CacheOpB,
          typename ElementC_, typename LayoutC_, typename IteratorE_,
          typename SmemIteratorE_,
          cutlass::arch::CacheOperation::Kind CacheOpE,
          typename SparsePolicy_, typename DensePolicy_,
          typename DenseSmemLayoutA_, int Stages, int DenseWarpCount,
          bool GuardEmptyBranches, bool ParallelConsumers = true,
          bool EnableRoleTiming = false,
          bool CompactSelectorLoaders = false,
          bool ExplicitPackProducerBranch = true>
class DenseBaseFusedDenseSparseSidecarMma {
 public:
  using Shape = Shape_;
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using IteratorE = IteratorE_;
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  using SmemIteratorE = SmemIteratorE_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = SparsePolicy_;
  using DensePolicy = DensePolicy_;
  using Operator = typename Policy::Operator;
  using DenseOperator = typename DensePolicy::Operator;
  using DenseSmemLayoutA = DenseSmemLayoutA_;
  using SparseSmemLayoutA = typename Operator::LayoutA;
  using FragmentC = typename Operator::FragmentC;
  using ElementE = typename IteratorE::Element;
  using LayoutE = typename IteratorE::Layout;
  using ArchTag = cutlass::arch::Sm80;

  static int const kSparse = Operator::kSparse;
  static int const kMetaSizeInBits = Operator::kMetaSizeInBits;
  static int const kMaxID2 = Operator::kMaxID2;
  static int const kElementsPerElementE = Operator::kElementsPerElementE;
  static cutlass::ComplexTransform const kTransformA = Operator::kTransformA;
  static cutlass::ComplexTransform const kTransformB = Operator::kTransformB;
  static int const kStages = Stages;

  using WarpGemm = typename Operator::Shape;
  using WarpCount = cutlass::gemm::GemmShape<
      Shape::kM / WarpGemm::kM,
      Shape::kN / WarpGemm::kN,
      Shape::kK / WarpGemm::kK>;

  static_assert(Stages >= 2,
                "fused sidecar mainloop requires double buffering");
  static int const kDenseWarpCount = DenseWarpCount;
  static int const kSparseWarpCount = WarpCount::kN - DenseWarpCount;
  // DenseWarpCount and kSparseWarpCount count warp-N roles.  A wider output
  // feature tile may replicate every role over multiple warp-M groups.  The
  // sidecar producers are still selected independently inside each warp-M/K
  // owner, but the named barrier must wait for every physical sparse warp.
  static int const kSparseBarrierWarpCount =
      WarpCount::kM * kSparseWarpCount * WarpCount::kK;
  static bool const kParallelConsumers = ParallelConsumers;
  static bool const kEnableRoleTiming = EnableRoleTiming;
  static bool const kCompactSelectorLoaders = CompactSelectorLoaders;

  static_assert(WarpCount::kN >= 2,
                "fused CTA requires at least two warp-N groups");
  static_assert(DenseWarpCount >= 0 && DenseWarpCount <= WarpCount::kN,
                "DenseWarpCount must be a valid warp-N role count");
  static_assert(WarpCount::kK >= 1,
                "fused CTA requires at least one warp-K partition");
  static_assert(
      cutlass::platform::is_same<
          FragmentC, typename DenseOperator::FragmentC>::value,
      "dense and sparse warps must expose the same accumulator fragment");
  static_assert(
      cutlass::platform::is_same<
          typename Operator::LayoutB,
          typename DenseOperator::LayoutB>::value,
      "dense and sparse warps must share one B shared-memory layout");

  class SharedStorage {
   public:
    using ShapeDenseA =
        cutlass::MatrixShape<Shape::kM, Shape::kK * Stages>;
    using ShapeSparseA =
        cutlass::MatrixShape<Shape::kM, Shape::kK / kSparse * Stages>;
    using ShapeB =
        cutlass::MatrixShape<Shape::kK * Stages, Shape::kN>;
    using ShapeE = cutlass::MatrixShape<
        Shape::kM * 2,
        Shape::kK / kSparse / kElementsPerElementE / 2 * Stages>;
    using DenseTensorRefA =
        cutlass::TensorRef<typename Operator::ElementA, DenseSmemLayoutA>;
    using SparseTensorRefA =
        cutlass::TensorRef<typename Operator::ElementA, SparseSmemLayoutA>;
    using TensorRefB = cutlass::TensorRef<
        typename Operator::ElementB, typename Operator::LayoutB>;
    using TensorRefE =
        cutlass::TensorRef<ElementE, typename Operator::LayoutE>;

    cutlass::AlignedBuffer<
        typename Operator::ElementA, ShapeDenseA::kCount> operand_A;
    cutlass::AlignedBuffer<
        typename Operator::ElementA, ShapeSparseA::kCount> operand_sparse_A;
    cutlass::AlignedBuffer<
        typename Operator::ElementB, ShapeB::kCount> operand_B;
    cutlass::AlignedBuffer<ElementE, ShapeE::kCount> operand_E;

    CUTLASS_HOST_DEVICE DenseTensorRefA operand_A_ref() {
      return {
          operand_A.data(),
          DenseSmemLayoutA::packed(
              {ShapeDenseA::kRow, ShapeDenseA::kColumn})};
    }
    CUTLASS_HOST_DEVICE SparseTensorRefA operand_sparse_A_ref() {
      return {
          operand_sparse_A.data(),
          SparseSmemLayoutA::packed(
              {ShapeSparseA::kRow, ShapeSparseA::kColumn})};
    }
    CUTLASS_HOST_DEVICE TensorRefB operand_B_ref() {
      return {
          operand_B.data(),
          Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn})};
    }

    // The two-route-wave ablation reinterprets the existing two B pipeline
    // stages as two independently addressed KxN route-wave slots.  No extra
    // allocation is introduced: each ref owns exactly one half of operand_B,
    // and both the producer and warp iterator use the same packed KxN layout.
    // The legacy mainloop continues to use operand_B_ref() above.
    template <int Wave>
    CUTLASS_HOST_DEVICE TensorRefB operand_B_wave_ref() {
      static_assert(Stages == 2,
                    "two-wave B slots require exactly two stages");
      static_assert(Wave == 0 || Wave == 1,
                    "two-wave B slot must be wave 0 or wave 1");
      using ShapeBWave =
          cutlass::MatrixShape<Shape::kK, Shape::kN>;
      return {
          operand_B.data() + Wave * ShapeBWave::kCount,
          Operator::LayoutB::packed(
              {ShapeBWave::kRow, ShapeBWave::kColumn})};
    }
    CUTLASS_HOST_DEVICE TensorRefE operand_E_ref() {
      return {
          operand_E.data(),
          Operator::LayoutE::packed({ShapeE::kRow, ShapeE::kColumn})};
    }
  };

 private:
  using SparseWarpIteratorA = typename Operator::IteratorA;
  using SidecarPacker = WarpOwnedSparse24SidecarPacker<
      Operator, DenseSmemLayoutA, Shape, WarpCount, Stages,
      CompactSelectorLoaders>;
  static int const kSidecarProducerCount =
      kSparseWarpCount < SidecarPacker::kWarpKIterations
      ? kSparseWarpCount
      : SidecarPacker::kWarpKIterations;
  static int const kDenseWarpGemmIterations =
      DenseOperator::Shape::kK / DenseOperator::Policy::MmaShape::kK;
  static int const kSparseWarpGemmIterations =
      Operator::Shape::kK / Operator::Policy::MmaShape::kK;

  SmemIteratorA smem_iterator_A_;
  SmemIteratorB smem_iterator_B_;
  SmemIteratorE smem_iterator_E_;
  SharedStorage& shared_;
  int thread_idx_;
  int warp_idx_;
  int lane_idx_;
  bool is_metadata_thread_;

 public:
  CUTLASS_DEVICE
  DenseBaseFusedDenseSparseSidecarMma(
      SharedStorage& shared, int thread_idx, int warp_idx, int lane_idx)
      : smem_iterator_A_(shared.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared.operand_B_ref(), thread_idx),
        smem_iterator_E_(shared.operand_E_ref(), thread_idx),
        shared_(shared),
        thread_idx_(thread_idx),
        warp_idx_(warp_idx),
        lane_idx_(lane_idx),
        is_metadata_thread_(
            thread_idx < IteratorE::ThreadMap::kThreads) {}

 private:
  // Two-route-wave load split.  The canonical dense weight and metadata keep
  // their normal two-stage A/E pipeline, while each B route wave has one
  // fixed shared-memory slot.  These helpers are used only by
  // operator_two_route_waves(); the legacy copy_stage_async() remains intact.
  CUTLASS_DEVICE void copy_weight_stage_async(
      IteratorA& iterator_A, IteratorE& iterator_E,
      bool sparse_branch_any_valid) {
    iterator_A.set_iteration_index(0);
    smem_iterator_A_.set_iteration_index(0);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < IteratorA::ThreadMap::Iterations::kCount; ++j) {
      auto* dst = reinterpret_cast<typename IteratorA::AccessType*>(
          smem_iterator_A_.get());
      constexpr int kBytes =
          cutlass::sizeof_bits<typename IteratorA::Element>::value *
          IteratorA::ThreadMap::kElementsPerAccess /
          IteratorA::kAccessesPerVector / 8;
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
        cutlass::arch::cp_async_zfill<kBytes, CacheOpA>(
            dst + v, iterator_A.get(), iterator_A.valid());
        ++iterator_A;
      }
      ++smem_iterator_A_;
    }

    if (is_metadata_thread_ && sparse_branch_any_valid) {
      iterator_E.set_iteration_index(0);
      smem_iterator_E_.set_iteration_index(0);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < IteratorE::ThreadMap::Iterations::kCount; ++j) {
        auto* dst = reinterpret_cast<typename IteratorE::AccessType*>(
            smem_iterator_E_.get());
        constexpr int kBytes =
            cutlass::sizeof_bits<typename IteratorE::Element>::value *
            IteratorE::ThreadMap::kElementsPerAccess / 8;
        cutlass::arch::cp_async_zfill<kBytes, CacheOpE>(
            dst, iterator_E.get(), iterator_E.valid());
        ++iterator_E;
        ++smem_iterator_E_;
      }
    }
  }

  CUTLASS_DEVICE void copy_b_wave_async_impl(
      IteratorB& iterator_B, SmemIteratorB& smem_iterator_B) {
    iterator_B.set_iteration_index(0);
    smem_iterator_B.set_iteration_index(0);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < IteratorB::ThreadMap::Iterations::kCount; ++j) {
      auto* dst = reinterpret_cast<typename IteratorB::AccessType*>(
          smem_iterator_B.get());
      constexpr int kBytes =
          cutlass::sizeof_bits<typename IteratorB::Element>::value *
          IteratorB::ThreadMap::kElementsPerAccess /
          IteratorB::kAccessesPerVector / 8;
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
        cutlass::arch::cp_async_zfill<kBytes, CacheOpB>(
            dst + v, iterator_B.get(), iterator_B.valid());
        ++iterator_B;
      }
      ++smem_iterator_B;
    }
  }

  template <int Wave>
  CUTLASS_DEVICE void copy_b_wave_async(IteratorB& iterator_B) {
    static_assert(Wave == 0 || Wave == 1,
                  "two-wave B copy requires wave 0 or wave 1");
    // Reconstruct the fixed-slot iterator for every K64 copy.  Merely resetting
    // an iterator's logical iteration index is not a sufficient ownership
    // contract for every CUTLASS layout specialization and could leave a
    // future iterator implementation advanced past its wave slot.
    if constexpr (Wave == 0) {
      SmemIteratorB smem_iterator_B_wave0(
          shared_.template operand_B_wave_ref<0>(), thread_idx_);
      copy_b_wave_async_impl(iterator_B, smem_iterator_B_wave0);
    } else {
      SmemIteratorB smem_iterator_B_wave1(
          shared_.template operand_B_wave_ref<1>(), thread_idx_);
      copy_b_wave_async_impl(iterator_B, smem_iterator_B_wave1);
    }
  }

  CUTLASS_DEVICE void copy_stage_async(
      IteratorA& iterator_A, IteratorB& iterator_B, IteratorE& iterator_E,
      bool sparse_branch_any_valid,
      SidecarRoleTiming* role_timing = nullptr) {
    iterator_A.set_iteration_index(0);
    smem_iterator_A_.set_iteration_index(0);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < IteratorA::ThreadMap::Iterations::kCount; ++j) {
      auto* dst = reinterpret_cast<typename IteratorA::AccessType*>(
          smem_iterator_A_.get());
      constexpr int kBytes =
          cutlass::sizeof_bits<typename IteratorA::Element>::value *
          IteratorA::ThreadMap::kElementsPerAccess /
          IteratorA::kAccessesPerVector / 8;
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
        cutlass::arch::cp_async_zfill<kBytes, CacheOpA>(
            dst + v, iterator_A.get(), iterator_A.valid());
        ++iterator_A;
      }
      ++smem_iterator_A_;
    }

    uint64_t activation_route_copy_start = 0;
    if constexpr (EnableRoleTiming) {
      if (lane_idx_ == 0) {
        activation_route_copy_start = sidecar_globaltimer_ns();
      }
    }
    iterator_B.set_iteration_index(0);
    smem_iterator_B_.set_iteration_index(0);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < IteratorB::ThreadMap::Iterations::kCount; ++j) {
      auto* dst = reinterpret_cast<typename IteratorB::AccessType*>(
          smem_iterator_B_.get());
      constexpr int kBytes =
          cutlass::sizeof_bits<typename IteratorB::Element>::value *
          IteratorB::ThreadMap::kElementsPerAccess /
          IteratorB::kAccessesPerVector / 8;
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
        cutlass::arch::cp_async_zfill<kBytes, CacheOpB>(
            dst + v, iterator_B.get(), iterator_B.valid());
        ++iterator_B;
      }
      ++smem_iterator_B_;
    }
    if constexpr (EnableRoleTiming) {
      if (lane_idx_ == 0 && role_timing != nullptr) {
        role_timing->activation_route_copy_issue_ns +=
            sidecar_globaltimer_ns() - activation_route_copy_start;
      }
    }

    if (is_metadata_thread_ && sparse_branch_any_valid) {
      iterator_E.set_iteration_index(0);
      smem_iterator_E_.set_iteration_index(0);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < IteratorE::ThreadMap::Iterations::kCount; ++j) {
        auto* dst = reinterpret_cast<typename IteratorE::AccessType*>(
            smem_iterator_E_.get());
        constexpr int kBytes =
            cutlass::sizeof_bits<typename IteratorE::Element>::value *
            IteratorE::ThreadMap::kElementsPerAccess / 8;
        cutlass::arch::cp_async_zfill<kBytes, CacheOpE>(
            dst, iterator_E.get(), iterator_E.valid());
        ++iterator_E;
        ++smem_iterator_E_;
      }
    }
  }

  CUTLASS_DEVICE void advance_write_stage(int& write_stage) {
    smem_iterator_A_.add_tile_offset({0, 1});
    smem_iterator_B_.add_tile_offset({1, 0});
    if (is_metadata_thread_) {
      smem_iterator_E_.add_tile_offset({0, 1});
    }
    if (write_stage == Stages - 1) {
      smem_iterator_A_.add_tile_offset({0, -Stages});
      smem_iterator_B_.add_tile_offset({-Stages, 0});
      if (is_metadata_thread_) {
        smem_iterator_E_.add_tile_offset({0, -Stages});
      }
      write_stage = 0;
    } else {
      ++write_stage;
    }
  }

  CUTLASS_DEVICE void advance_global_stage(
      IteratorA& iterator_A, IteratorB& iterator_B, IteratorE& iterator_E) {
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});
    if (is_metadata_thread_) {
      iterator_E.add_tile_offset({0, 1});
    }
  }

  CUTLASS_DEVICE void advance_weight_write_stage(int& write_stage) {
    smem_iterator_A_.add_tile_offset({0, 1});
    if (is_metadata_thread_) {
      smem_iterator_E_.add_tile_offset({0, 1});
    }
    if (write_stage == Stages - 1) {
      smem_iterator_A_.add_tile_offset({0, -Stages});
      if (is_metadata_thread_) {
        smem_iterator_E_.add_tile_offset({0, -Stages});
      }
      write_stage = 0;
    } else {
      ++write_stage;
    }
  }

  CUTLASS_DEVICE void advance_weight_global_stage(
      IteratorA& iterator_A, IteratorE& iterator_E) {
    iterator_A.add_tile_offset({0, 1});
    if (is_metadata_thread_) {
      iterator_E.add_tile_offset({0, 1});
    }
  }

  CUTLASS_DEVICE void advance_b_global_stage(IteratorB& iterator_B) {
    iterator_B.add_tile_offset({1, 0});
  }

  CUTLASS_DEVICE void dense_mma(
      FragmentC& accum, int warp_idx_m, int warp_idx_n, int warp_idx_k,
      int read_stage) {
    typename DenseOperator::IteratorA warp_iterator_A(
        shared_.operand_A_ref(), lane_idx_);
    typename DenseOperator::IteratorB warp_iterator_B(
        shared_.operand_B_ref(), lane_idx_);
    warp_iterator_A.add_tile_offset(
        {warp_idx_m,
         (read_stage * WarpCount::kK + warp_idx_k) *
             kDenseWarpGemmIterations});
    warp_iterator_B.add_tile_offset(
        {(read_stage * WarpCount::kK + warp_idx_k) *
             kDenseWarpGemmIterations,
         warp_idx_n});

    DenseOperator warp_mma;
    static_assert(
        cutlass::platform::is_same<
            typename DenseOperator::FragmentA,
            typename DenseOperator::TransformedFragmentA>::value,
        "BF16 dense A fragments must be representation-identical");
    static_assert(
        cutlass::platform::is_same<
            typename DenseOperator::FragmentB,
            typename DenseOperator::TransformedFragmentB>::value,
        "BF16 dense B fragments must be representation-identical");
    CUTLASS_PRAGMA_UNROLL
    for (int k_group = 0; k_group < kDenseWarpGemmIterations; ++k_group) {
      typename DenseOperator::TransformedFragmentA fragment_A;
      typename DenseOperator::TransformedFragmentB fragment_B;
      warp_iterator_A.set_kgroup_index(k_group);
      warp_iterator_B.set_kgroup_index(k_group);
      warp_iterator_A.load(fragment_A);
      warp_iterator_B.load(fragment_B);
      warp_mma(accum, fragment_A, fragment_B, accum);
      ++warp_iterator_A;
      ++warp_iterator_B;
    }
  }

  CUTLASS_DEVICE void pack_sparse_stage(
      int warp_idx_m, int warp_idx_k, int read_stage) {
    int sparse_warp_rank =
        (warp_idx_ / WarpCount::kM) % WarpCount::kN - DenseWarpCount;
    // Only the first min(sparse-warps, K32-groups) warps own pack work.  The
    // optimized kernels take a warp-uniform branch before constructing the
    // packer.  ExplicitPackProducerBranch=false freezes the pre-optimization
    // legacy behavior for the one-weight naive baseline: consumer-only
    // sparse warps still enter SidecarPacker::pack_stage(), whose owner-rank
    // loop executes zero iterations for them.  Every sparse warp reaches the
    // same named barrier in either mode.
    if constexpr (ExplicitPackProducerBranch) {
      if (sparse_warp_rank >= kSidecarProducerCount) {
        return;
      }
    }
    SidecarPacker packer(
        shared_.operand_A_ref(), shared_.operand_sparse_A_ref(),
        shared_.operand_E_ref(), lane_idx_);
    packer.template pack_stage<kSidecarProducerCount>(
        warp_idx_m, warp_idx_k, read_stage, sparse_warp_rank);
  }

  CUTLASS_DEVICE void sparse_mma(
      FragmentC& accum, int warp_idx_m, int warp_idx_n, int warp_idx_k,
      int read_stage) {
    SparseWarpIteratorA warp_iterator_A(
        shared_.operand_sparse_A_ref(), lane_idx_);
    typename Operator::IteratorB warp_iterator_B(
        shared_.operand_B_ref(), lane_idx_);
    typename Operator::IteratorE warp_iterator_E(
        shared_.operand_E_ref(), lane_idx_);
    warp_iterator_A.add_tile_offset(
        {warp_idx_m,
         (read_stage * WarpCount::kK + warp_idx_k) *
             kSparseWarpGemmIterations});
    warp_iterator_B.add_tile_offset(
        {(read_stage * WarpCount::kK + warp_idx_k) *
             kSparseWarpGemmIterations,
         warp_idx_n});
    warp_iterator_E.add_tile_offset(
        {warp_idx_m,
         (read_stage * WarpCount::kK + warp_idx_k) *
             kSparseWarpGemmIterations});

    static_assert(
        cutlass::platform::is_same<
            typename Operator::FragmentA,
            typename Operator::TransformedFragmentA>::value,
        "BF16 sparse A fragments must be representation-identical");
    static_assert(
        cutlass::platform::is_same<
            typename Operator::FragmentB,
            typename Operator::TransformedFragmentB>::value,
        "BF16 sparse B fragments must be representation-identical");
    Operator warp_mma;
    CUTLASS_PRAGMA_UNROLL
    for (int k_group = 0; k_group < kSparseWarpGemmIterations; ++k_group) {
      typename Operator::FragmentA fragment_A;
      typename Operator::FragmentB fragment_B;
      typename Operator::FragmentE fragment_E;
      warp_iterator_A.set_kgroup_index(k_group);
      warp_iterator_B.set_kgroup_index(k_group);
      warp_iterator_E.set_kgroup_index(k_group);
      warp_iterator_A.load(fragment_A);
      warp_iterator_B.load(fragment_B);
      warp_iterator_E.load(fragment_E);
      warp_mma(accum, fragment_A, fragment_B, accum, fragment_E);
      ++warp_iterator_A;
      ++warp_iterator_B;
      ++warp_iterator_E;
    }
  }

  template <int Wave>
  CUTLASS_DEVICE void dense_mma_wave(
      FragmentC& accum, int warp_idx_m, int warp_idx_n,
      int read_stage) {
    static_assert(Wave == 0 || Wave == 1,
                  "dense two-wave MMA requires wave 0 or wave 1");
    typename DenseOperator::IteratorA warp_iterator_A(
        shared_.operand_A_ref(), lane_idx_);
    typename DenseOperator::IteratorB warp_iterator_B(
        shared_.template operand_B_wave_ref<Wave>(), lane_idx_);
    warp_iterator_A.add_tile_offset(
        {warp_idx_m, read_stage * kDenseWarpGemmIterations});
    // A route-wave ref already selects the B slot and the CTA has one warp-K
    // partition, so B advances only by this warp's token role.
    warp_iterator_B.add_tile_offset({0, warp_idx_n});

    DenseOperator warp_mma;
    CUTLASS_PRAGMA_UNROLL
    for (int k_group = 0; k_group < kDenseWarpGemmIterations; ++k_group) {
      typename DenseOperator::TransformedFragmentA fragment_A;
      typename DenseOperator::TransformedFragmentB fragment_B;
      warp_iterator_A.set_kgroup_index(k_group);
      warp_iterator_B.set_kgroup_index(k_group);
      warp_iterator_A.load(fragment_A);
      warp_iterator_B.load(fragment_B);
      warp_mma(accum, fragment_A, fragment_B, accum);
      ++warp_iterator_A;
      ++warp_iterator_B;
    }
  }

  template <int Wave>
  CUTLASS_DEVICE void sparse_mma_wave(
      FragmentC& accum, int warp_idx_m, int warp_idx_n,
      int read_stage) {
    static_assert(Wave == 0 || Wave == 1,
                  "sparse two-wave MMA requires wave 0 or wave 1");
    SparseWarpIteratorA warp_iterator_A(
        shared_.operand_sparse_A_ref(), lane_idx_);
    typename Operator::IteratorB warp_iterator_B(
        shared_.template operand_B_wave_ref<Wave>(), lane_idx_);
    typename Operator::IteratorE warp_iterator_E(
        shared_.operand_E_ref(), lane_idx_);
    warp_iterator_A.add_tile_offset(
        {warp_idx_m, read_stage * kSparseWarpGemmIterations});
    warp_iterator_B.add_tile_offset({0, warp_idx_n});
    warp_iterator_E.add_tile_offset(
        {warp_idx_m, read_stage * kSparseWarpGemmIterations});

    Operator warp_mma;
    CUTLASS_PRAGMA_UNROLL
    for (int k_group = 0; k_group < kSparseWarpGemmIterations; ++k_group) {
      typename Operator::FragmentA fragment_A;
      typename Operator::FragmentB fragment_B;
      typename Operator::FragmentE fragment_E;
      warp_iterator_A.set_kgroup_index(k_group);
      warp_iterator_B.set_kgroup_index(k_group);
      warp_iterator_E.set_kgroup_index(k_group);
      warp_iterator_A.load(fragment_A);
      warp_iterator_B.load(fragment_B);
      warp_iterator_E.load(fragment_E);
      warp_mma(accum, fragment_A, fragment_B, accum, fragment_E);
      ++warp_iterator_A;
      ++warp_iterator_B;
      ++warp_iterator_E;
    }
  }

 public:
  CUTLASS_DEVICE void operator()(
      int gemm_k_iterations, FragmentC& accum, IteratorA iterator_A,
      IteratorB iterator_B, IteratorE iterator_E,
      FragmentC const& src_accum, bool warp_branch_valid,
      bool sparse_branch_any_valid, int runtime_role_loop_extent,
      SidecarRoleTiming* role_timing = nullptr) {
    // warp_branch_valid is uniform only within the current warp and controls
    // its HMMA tail.  sparse_branch_any_valid must be CTA-uniform: metadata
    // production and the one sidecar producer remain active whenever any
    // sparse warp-N group has work, even if the producer's own N tile is empty.
    uint64_t mainloop_start = 0;
    if constexpr (EnableRoleTiming) {
      if (lane_idx_ == 0) {
        mainloop_start = sidecar_globaltimer_ns();
      }
    }
    if constexpr (!GuardEmptyBranches) {
      warp_branch_valid = true;
      sparse_branch_any_valid = true;
    }
    accum = src_accum;
    int warp_idx_m = warp_idx_ % WarpCount::kM;
    int warp_idx_n = (warp_idx_ / WarpCount::kM) % WarpCount::kN;
    int warp_idx_k = warp_idx_ / (WarpCount::kM * WarpCount::kN);

    int const total_iterations = gemm_k_iterations;
    int loaded_iterations = 0;
    int write_stage = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Stages - 1; ++stage) {
      uint64_t stage_issue_start = 0;
      if constexpr (EnableRoleTiming) {
        if (lane_idx_ == 0) {
          stage_issue_start = sidecar_globaltimer_ns();
        }
      }
      bool invalid = loaded_iterations >= total_iterations;
      iterator_A.clear_mask(invalid);
      iterator_B.clear_mask(invalid);
      iterator_E.clear_mask(invalid);
      copy_stage_async(
          iterator_A, iterator_B, iterator_E, sparse_branch_any_valid,
          EnableRoleTiming ? role_timing : nullptr);
      cutlass::arch::cp_async_fence();
      advance_global_stage(iterator_A, iterator_B, iterator_E);
      advance_write_stage(write_stage);
      ++loaded_iterations;
      if constexpr (EnableRoleTiming) {
        if (lane_idx_ == 0) {
          role_timing->stage_issue_ns +=
              sidecar_globaltimer_ns() - stage_issue_start;
        }
      }
    }

    uint64_t phase_start = 0;
    if constexpr (EnableRoleTiming) {
      if (lane_idx_ == 0) {
        phase_start = sidecar_globaltimer_ns();
      }
    }
    cutlass::arch::cp_async_wait<Stages - 2>();
    if constexpr (EnableRoleTiming) {
      if (lane_idx_ == 0) {
        role_timing->async_wait_ns +=
            sidecar_globaltimer_ns() - phase_start;
        phase_start = sidecar_globaltimer_ns();
      }
    }
    __syncthreads();
    if constexpr (EnableRoleTiming) {
      if (lane_idx_ == 0) {
        role_timing->cta_barrier_ns +=
            sidecar_globaltimer_ns() - phase_start;
      }
    }

    bool is_dense_warp = false;
    if constexpr (DenseWarpCount == WarpCount::kN) {
      is_dense_warp = true;
    } else if constexpr (DenseWarpCount > 0) {
      is_dense_warp = warp_idx_n < DenseWarpCount;
    }
    bool is_sparse_warp = warp_idx_n >= DenseWarpCount;
    int dense_role_runs =
        (is_dense_warp && warp_branch_valid)
        ? runtime_role_loop_extent
        : 0;
    int sparse_role_runs =
        (is_sparse_warp && warp_branch_valid)
        ? runtime_role_loop_extent
        : 0;

    int read_stage = 0;
    CUTLASS_GEMM_LOOP
    for (int iteration = 0; iteration < total_iterations; ++iteration) {
      uint64_t stage_issue_start = 0;
      if constexpr (EnableRoleTiming) {
        if (lane_idx_ == 0) {
          stage_issue_start = sidecar_globaltimer_ns();
        }
      }
      bool invalid = loaded_iterations >= total_iterations;
      iterator_A.clear_mask(invalid);
      iterator_B.clear_mask(invalid);
      iterator_E.clear_mask(invalid);
      copy_stage_async(
          iterator_A, iterator_B, iterator_E, sparse_branch_any_valid,
          EnableRoleTiming ? role_timing : nullptr);
      cutlass::arch::cp_async_fence();
      advance_global_stage(iterator_A, iterator_B, iterator_E);
      advance_write_stage(write_stage);
      ++loaded_iterations;
      if constexpr (EnableRoleTiming) {
        if (lane_idx_ == 0) {
          role_timing->stage_issue_ns +=
              sidecar_globaltimer_ns() - stage_issue_start;
        }
      }

      if constexpr (ParallelConsumers) {
        // Dense consumers and the cooperative packers start together.  Only
        // sparse warps participate in this named barrier, so consumers are
        // released as soon as the sidecar is visible without waiting for the
        // dense HMMA phase.  All roles remain complete warps.
        // A plain role ``if`` is if-converted by ptxas on SM120: sparse warps
        // still issue an all-lanes-predicate-off copy of every dense HMMA.
        // Keep the split-K extent runtime-visible and use it as a 0/1 loop
        // bound.  This forces a real warp-uniform branch before the HMMA
        // basic block while remaining exactly one iteration for this kernel's
        // split_k_slices=1 launch contract.
        if constexpr (kSparseWarpCount > 0) {
          if (is_sparse_warp) {
          if (sparse_branch_any_valid) {
            if constexpr (EnableRoleTiming) {
              if (lane_idx_ == 0) {
                phase_start = sidecar_globaltimer_ns();
              }
            }
            pack_sparse_stage(warp_idx_m, warp_idx_k, read_stage);
            if constexpr (EnableRoleTiming) {
              int sparse_warp_rank = warp_idx_n - DenseWarpCount;
              if (lane_idx_ == 0) {
                uint64_t pack_end = sidecar_globaltimer_ns();
                if (sparse_warp_rank < kSidecarProducerCount) {
                  role_timing->pack_ns += pack_end - phase_start;
                } else {
                  role_timing->nonproducer_pack_dispatch_ns +=
                      pack_end - phase_start;
                }
              }
            }
          }
          if constexpr (EnableRoleTiming) {
            if (lane_idx_ == 0) {
              phase_start = sidecar_globaltimer_ns();
            }
          }
          cutlass::arch::NamedBarrier::sync(
              kSparseBarrierWarpCount * 32, 0);
          if constexpr (EnableRoleTiming) {
            if (lane_idx_ == 0) {
              role_timing->role_barrier_ns +=
                  sidecar_globaltimer_ns() - phase_start;
            }
          }
          CUTLASS_PRAGMA_NO_UNROLL
          for (int role_run = 0;
               role_run < sparse_role_runs;
               ++role_run) {
            if constexpr (EnableRoleTiming) {
              if (lane_idx_ == 0) {
                phase_start = sidecar_globaltimer_ns();
              }
            }
            sparse_mma(
                accum, warp_idx_m, warp_idx_n, warp_idx_k, read_stage);
            if constexpr (EnableRoleTiming) {
              if (lane_idx_ == 0) {
                role_timing->sparse_mma_ns +=
                    sidecar_globaltimer_ns() - phase_start;
              }
            }
          }
          } else {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int role_run = 0;
               role_run < dense_role_runs;
               ++role_run) {
            if constexpr (EnableRoleTiming) {
              if (lane_idx_ == 0) {
                phase_start = sidecar_globaltimer_ns();
              }
            }
            dense_mma(
                accum, warp_idx_m, warp_idx_n, warp_idx_k, read_stage);
            if constexpr (EnableRoleTiming) {
              if (lane_idx_ == 0) {
                role_timing->dense_mma_ns +=
                    sidecar_globaltimer_ns() - phase_start;
              }
            }
          }
          }
        } else {
          // D4:S0: the sparse role and its zero-sized producer/barrier are
          // removed at compile time.  Every physical warp executes dense
          // HMMA for its own 32-token column group.
          CUTLASS_PRAGMA_NO_UNROLL
          for (int role_run = 0;
               role_run < dense_role_runs;
               ++role_run) {
            dense_mma(
                accum, warp_idx_m, warp_idx_n, warp_idx_k, read_stage);
          }
        }
      } else {
        // Serial-consumer ablation: dense MMA and cooperative packers begin
        // together, then a CTA barrier waits for both before sparse MMA.
        if (is_sparse_warp) {
          if (sparse_branch_any_valid) {
            if constexpr (EnableRoleTiming) {
              if (lane_idx_ == 0) {
                phase_start = sidecar_globaltimer_ns();
              }
            }
            pack_sparse_stage(warp_idx_m, warp_idx_k, read_stage);
            if constexpr (EnableRoleTiming) {
              int sparse_warp_rank = warp_idx_n - DenseWarpCount;
              if (lane_idx_ == 0) {
                uint64_t pack_end = sidecar_globaltimer_ns();
                if (sparse_warp_rank < kSidecarProducerCount) {
                  role_timing->pack_ns += pack_end - phase_start;
                } else {
                  role_timing->nonproducer_pack_dispatch_ns +=
                      pack_end - phase_start;
                }
              }
            }
          }
        } else {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int role_run = 0;
               role_run < dense_role_runs;
               ++role_run) {
            if constexpr (EnableRoleTiming) {
              if (lane_idx_ == 0) {
                phase_start = sidecar_globaltimer_ns();
              }
            }
            dense_mma(
                accum, warp_idx_m, warp_idx_n, warp_idx_k, read_stage);
            if constexpr (EnableRoleTiming) {
              if (lane_idx_ == 0) {
                role_timing->dense_mma_ns +=
                    sidecar_globaltimer_ns() - phase_start;
              }
            }
          }
        }

        if constexpr (EnableRoleTiming) {
          if (lane_idx_ == 0) {
            phase_start = sidecar_globaltimer_ns();
          }
        }
        __syncthreads();
        if constexpr (EnableRoleTiming) {
          if (lane_idx_ == 0) {
            role_timing->role_barrier_ns +=
                sidecar_globaltimer_ns() - phase_start;
          }
        }

        if (is_sparse_warp) {
          CUTLASS_PRAGMA_NO_UNROLL
          for (int role_run = 0;
               role_run < sparse_role_runs;
               ++role_run) {
            if constexpr (EnableRoleTiming) {
              if (lane_idx_ == 0) {
                phase_start = sidecar_globaltimer_ns();
              }
            }
            sparse_mma(
                accum, warp_idx_m, warp_idx_n, warp_idx_k, read_stage);
            if constexpr (EnableRoleTiming) {
              if (lane_idx_ == 0) {
                role_timing->sparse_mma_ns +=
                    sidecar_globaltimer_ns() - phase_start;
              }
            }
          }
        }
      }

      if constexpr (EnableRoleTiming) {
        if (lane_idx_ == 0) {
          phase_start = sidecar_globaltimer_ns();
        }
      }
      cutlass::arch::cp_async_wait<Stages - 2>();
      if constexpr (EnableRoleTiming) {
        if (lane_idx_ == 0) {
          role_timing->async_wait_ns +=
              sidecar_globaltimer_ns() - phase_start;
          phase_start = sidecar_globaltimer_ns();
        }
      }
      __syncthreads();
      if constexpr (EnableRoleTiming) {
        if (lane_idx_ == 0) {
          role_timing->cta_barrier_ns +=
              sidecar_globaltimer_ns() - phase_start;
        }
      }
      read_stage = (read_stage + 1 == Stages) ? 0 : read_stage + 1;
    }

    uint64_t final_fence_start = 0;
    if constexpr (EnableRoleTiming) {
      if (lane_idx_ == 0) {
        final_fence_start = sidecar_globaltimer_ns();
      }
    }
    cutlass::arch::cp_async_fence();
    if constexpr (EnableRoleTiming) {
      if (lane_idx_ == 0) {
        role_timing->stage_issue_ns +=
            sidecar_globaltimer_ns() - final_fence_start;
        phase_start = sidecar_globaltimer_ns();
      }
    }
    cutlass::arch::cp_async_wait<0>();
    if constexpr (EnableRoleTiming) {
      if (lane_idx_ == 0) {
        uint64_t wait_end = sidecar_globaltimer_ns();
        role_timing->async_wait_ns += wait_end - phase_start;
        phase_start = wait_end;
      }
    }
    __syncthreads();
    if constexpr (EnableRoleTiming) {
      if (lane_idx_ == 0) {
        role_timing->cta_barrier_ns +=
            sidecar_globaltimer_ns() - phase_start;
      }
    }
    if constexpr (EnableRoleTiming) {
      if (lane_idx_ == 0) {
        uint64_t mainloop_end = sidecar_globaltimer_ns();
        if (role_timing != nullptr) {
          role_timing->mainloop_ns += mainloop_end - mainloop_start;
        }
      }
    }
  }

  // Priority-1 true load/pack-reuse ablation.  Two adjacent route waves own
  // independent B iterators and accumulator fragments, but share every K64
  // canonical-weight/metadata copy and the one online 2:4 sidecar pack.  The
  // original two B pipeline stages are fixed wave-0/wave-1 slots, keeping the
  // shared allocation unchanged.  This first correctness-oriented version
  // intentionally uses CTA barriers before either B slot is overwritten.
  template <bool InterWaveBarriers, bool ElideTailPrefetch = false>
  CUTLASS_DEVICE void operator_two_route_waves(
      int gemm_k_iterations,
      FragmentC& accum_wave0, FragmentC& accum_wave1,
      IteratorA iterator_A,
      IteratorB iterator_B_wave0, IteratorB iterator_B_wave1,
      IteratorE iterator_E,
      FragmentC const& src_accum_wave0,
      FragmentC const& src_accum_wave1,
      bool warp_branch_valid_wave0,
      bool warp_branch_valid_wave1,
      bool sparse_branch_any_valid_wave0,
      bool sparse_branch_any_valid_wave1,
      int runtime_role_loop_extent) {
    static_assert(Stages == 2,
                  "two-route-wave mainloop requires exactly two stages");
    static_assert(WarpCount::kM == 1 && WarpCount::kN == 4 &&
                      WarpCount::kK == 1,
                  "two-route-wave mainloop is the narrow M1:N4:K1 ablation");
    static_assert(
        DenseWarpCount == 0 || DenseWarpCount == 1 ||
            DenseWarpCount == WarpCount::kN,
        "two-route-wave mainloop supports D0:S4, D1:S3, or D4:S0");
    using BThreadMap = typename IteratorB::ThreadMap;
    static_assert(
        BThreadMap::Detail::kWarpsContiguous == 1 &&
            BThreadMap::Detail::kWarpsStrided == WarpCount::kN,
        "warp-owned B publication requires one loader warp per token role");
    static_assert(
        BThreadMap::Detail::WarpThreadArrangement::kStrided *
                BThreadMap::Iterations::kStrided ==
            Shape::kN / WarpCount::kN,
        "each loader warp must own exactly its consumer's B columns");
    static_assert(
        BThreadMap::Detail::WarpThreadArrangement::kContiguous *
                BThreadMap::Iterations::kContiguous *
                BThreadMap::kElementsPerAccess ==
            Shape::kK,
        "each loader warp must cover the complete K64 B extent");

    accum_wave0 = src_accum_wave0;
    accum_wave1 = src_accum_wave1;
    int warp_idx_m = warp_idx_ % WarpCount::kM;
    int warp_idx_n = (warp_idx_ / WarpCount::kM) % WarpCount::kN;
    bool is_dense_warp = false;
    if constexpr (DenseWarpCount == WarpCount::kN) {
      is_dense_warp = true;
    } else if constexpr (DenseWarpCount > 0) {
      is_dense_warp = warp_idx_n < DenseWarpCount;
    }
    bool is_sparse_warp = !is_dense_warp;
    bool sparse_branch_any_valid =
        sparse_branch_any_valid_wave0 ||
        sparse_branch_any_valid_wave1;

    int dense_role_runs_wave0 =
        (is_dense_warp && warp_branch_valid_wave0)
        ? runtime_role_loop_extent
        : 0;
    int dense_role_runs_wave1 =
        (is_dense_warp && warp_branch_valid_wave1)
        ? runtime_role_loop_extent
        : 0;
    int sparse_role_runs_wave0 =
        (is_sparse_warp && warp_branch_valid_wave0)
        ? runtime_role_loop_extent
        : 0;
    int sparse_role_runs_wave1 =
        (is_sparse_warp && warp_branch_valid_wave1)
        ? runtime_role_loop_extent
        : 0;

    // Preload K0.  A/E are copied once for both route waves, while their B
    // operands land in the two explicitly named fixed slots.
    int loaded_iterations = 0;
    int weight_write_stage = 0;
    bool preload_invalid = loaded_iterations >= gemm_k_iterations;
    iterator_A.clear_mask(preload_invalid);
    iterator_E.clear_mask(preload_invalid);
    iterator_B_wave0.clear_mask(preload_invalid);
    iterator_B_wave1.clear_mask(preload_invalid);
    copy_weight_stage_async(
        iterator_A, iterator_E, sparse_branch_any_valid);
    copy_b_wave_async<0>(iterator_B_wave0);
    copy_b_wave_async<1>(iterator_B_wave1);
    cutlass::arch::cp_async_fence();
    advance_weight_global_stage(iterator_A, iterator_E);
    advance_b_global_stage(iterator_B_wave0);
    advance_b_global_stage(iterator_B_wave1);
    advance_weight_write_stage(weight_write_stage);
    ++loaded_iterations;
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    int read_stage = 0;
    CUTLASS_GEMM_LOOP
    for (int iteration = 0; iteration < gemm_k_iterations; ++iteration) {
      // The sparse-only specialization benefits from omitting the final
      // zero-fill publication.  Keep the original publication/drain schedule
      // available for mixed and dense-only binaries: formal A/B showed that
      // its extra async group preserves their preferred instruction schedule.
      bool has_next_k_tile = loaded_iterations < gemm_k_iterations;
      bool next_invalid = !has_next_k_tile;

      // Start the next canonical weight/metadata stage before consuming the
      // current one.  Its async group overlaps current pack and wave-0 MMA.
      if constexpr (!ElideTailPrefetch) {
        iterator_A.clear_mask(next_invalid);
        iterator_E.clear_mask(next_invalid);
        copy_weight_stage_async(
            iterator_A, iterator_E, sparse_branch_any_valid);
        cutlass::arch::cp_async_fence();
        advance_weight_global_stage(iterator_A, iterator_E);
        advance_weight_write_stage(weight_write_stage);
      } else if (has_next_k_tile) {
        copy_weight_stage_async(
            iterator_A, iterator_E, sparse_branch_any_valid);
        cutlass::arch::cp_async_fence();
        advance_weight_global_stage(iterator_A, iterator_E);
        advance_weight_write_stage(weight_write_stage);
      }

      if constexpr (kSparseWarpCount > 0) {
        if (is_sparse_warp) {
        if (sparse_branch_any_valid) {
          pack_sparse_stage(warp_idx_m, 0, read_stage);
        }
        cutlass::arch::NamedBarrier::sync(
            kSparseBarrierWarpCount * 32, 0);
        CUTLASS_PRAGMA_NO_UNROLL
        for (int role_run = 0;
             role_run < sparse_role_runs_wave0;
             ++role_run) {
          sparse_mma_wave<0>(
              accum_wave0, warp_idx_m, warp_idx_n, read_stage);
        }
        } else {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int role_run = 0;
             role_run < dense_role_runs_wave0;
             ++role_run) {
          dense_mma_wave<0>(
              accum_wave0, warp_idx_m, warp_idx_n, read_stage);
        }
        }
      } else {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int role_run = 0;
             role_run < dense_role_runs_wave0;
             ++role_run) {
          dense_mma_wave<0>(
              accum_wave0, warp_idx_m, warp_idx_n, read_stage);
        }
      }

      // The conservative ablation waits for the full CTA.  The optimized
      // instantiation relies on the compile-time-proven B thread map above:
      // warp n loads exactly the 32 logical columns consumed only by warp n,
      // so it may overwrite its own slot immediately after its own MMA.
      if constexpr (InterWaveBarriers) {
        __syncthreads();
      }
      if constexpr (!ElideTailPrefetch) {
        iterator_B_wave0.clear_mask(next_invalid);
        copy_b_wave_async<0>(iterator_B_wave0);
        cutlass::arch::cp_async_fence();
        advance_b_global_stage(iterator_B_wave0);
      } else if (has_next_k_tile) {
        copy_b_wave_async<0>(iterator_B_wave0);
        cutlass::arch::cp_async_fence();
        advance_b_global_stage(iterator_B_wave0);
      }

      if constexpr (kSparseWarpCount > 0) {
        if (is_sparse_warp) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int role_run = 0;
             role_run < sparse_role_runs_wave1;
             ++role_run) {
          sparse_mma_wave<1>(
              accum_wave1, warp_idx_m, warp_idx_n, read_stage);
        }
        } else {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int role_run = 0;
             role_run < dense_role_runs_wave1;
             ++role_run) {
          dense_mma_wave<1>(
              accum_wave1, warp_idx_m, warp_idx_n, read_stage);
        }
        }
      } else {
        CUTLASS_PRAGMA_NO_UNROLL
        for (int role_run = 0;
             role_run < dense_role_runs_wave1;
             ++role_run) {
          dense_mma_wave<1>(
              accum_wave1, warp_idx_m, warp_idx_n, read_stage);
        }
      }

      if constexpr (InterWaveBarriers) {
        __syncthreads();
      }
      if constexpr (!ElideTailPrefetch) {
        iterator_B_wave1.clear_mask(next_invalid);
        copy_b_wave_async<1>(iterator_B_wave1);
        cutlass::arch::cp_async_fence();
        advance_b_global_stage(iterator_B_wave1);
        ++loaded_iterations;
      } else if (has_next_k_tile) {
        copy_b_wave_async<1>(iterator_B_wave1);
        cutlass::arch::cp_async_fence();
        advance_b_global_stage(iterator_B_wave1);
        ++loaded_iterations;
      }

      cutlass::arch::cp_async_wait<0>();
      __syncthreads();
      read_stage = (read_stage == 0) ? 1 : 0;
    }

    if constexpr (!ElideTailPrefetch) {
      cutlass::arch::cp_async_fence();
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();
    }
  }
};

}  // namespace speclink
}  // namespace speculators
