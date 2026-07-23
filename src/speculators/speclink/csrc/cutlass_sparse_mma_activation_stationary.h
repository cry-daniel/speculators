/***************************************************************************************************
 * Copyright (c) 2026.
 *
 * Project-local activation-stationary sparse mainloop.  One physical warp tile
 * computes two adjacent output-feature panels.  For every K tile it loads the
 * activation (B) fragment once, then reuses that register fragment for two
 * residual-weight (A/E) panels and two independent accumulator fragments.
 *
 * This is intentionally different from a wider stock CUTLASS CTA: a stock
 * 128xN CTA assigns the two feature panels to different warps, so both warps
 * reload the same B fragment from shared memory.  Here one warp owns both
 * panels and reuses B in registers.  A and E stream through one shared tile.
 **************************************************************************************************/

#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/sparse_gemm_with_visitor.h"
#include "cutlass/gemm/threadblock/mma_sparse_base.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

template <
    typename LogicalShape_,
    typename PhysicalShape_,
    typename IteratorA_,
    typename SmemIteratorA_,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    typename IteratorB_,
    typename SmemIteratorB_,
    cutlass::arch::CacheOperation::Kind CacheOpB,
    typename ElementC_,
    typename LayoutC_,
    typename IteratorE_,
    typename SmemIteratorE_,
    cutlass::arch::CacheOperation::Kind CacheOpE,
    typename Policy_>
class SparseMmaActivationStationary2 :
    public SparseMmaBase<PhysicalShape_, Policy_, 1> {
 public:
  using Base = SparseMmaBase<PhysicalShape_, Policy_, 1>;
  using Shape = LogicalShape_;
  using PhysicalShape = PhysicalShape_;
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using IteratorE = IteratorE_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = Policy_;
  using Operator = typename Policy::Operator;
  using FragmentC = typename Operator::FragmentC;
  using ElementE = typename IteratorE::Element;
  using LayoutE = typename IteratorE::Layout;
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  using SmemIteratorE = SmemIteratorE_;
  using SharedStorage = typename Base::SharedStorage;
  using WarpCount = typename Base::WarpCount;
  using ArchTag = arch::Sm80;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
  static cutlass::arch::CacheOperation::Kind const kCacheOpE = CacheOpE;
  static int const kSparse = Operator::kSparse;
  static int const kMetaSizeInBits = Operator::kMetaSizeInBits;
  static int const kMaxID2 = Operator::kMaxID2;
  static int const kElementsPerElementE = Operator::kElementsPerElementE;
  static ComplexTransform const kTransformA = Operator::kTransformA;
  static ComplexTransform const kTransformB = Operator::kTransformB;

  static_assert(
      LogicalShape_::kM == 2 * PhysicalShape_::kM,
      "activation-stationary schedule owns exactly two feature panels");
  static_assert(
      LogicalShape_::kN == PhysicalShape_::kN &&
          LogicalShape_::kK == PhysicalShape_::kK,
      "only the output-feature dimension is grouped");
  static_assert(
      Base::WarpCount::kM == 1,
      "each warp must own the two serial feature panels");
  static_assert(
      Base::kWarpGemmIterations == 2,
      "activation-stationary schedule requires two warp-K groups");

  struct Detail {
    static int const TBLoadIterationsA =
        IteratorA::ThreadMap::Iterations::kCount;
    static int const TBLoadIterationsB =
        IteratorB::ThreadMap::Iterations::kCount;
    static int const TBLoadIterationsE =
        IteratorE::ThreadMap::Iterations::kCount;
    static int const kValidWarps = IteratorE::ThreadMap::kThreads / 32;
  };

 private:
  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;
  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;
  using WarpFragmentE = typename Operator::FragmentE;

  SmemIteratorA smem_iterator_A_;
  SmemIteratorB smem_iterator_B_;
  SmemIteratorE smem_iterator_E_;
  bool is_warp_valid_;

 public:
  CUTLASS_DEVICE
  SparseMmaActivationStationary2(
      SharedStorage& shared_storage,
      int thread_idx,
      int warp_idx,
      int lane_idx)
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
        smem_iterator_E_(shared_storage.operand_E_ref(), thread_idx),
        is_warp_valid_(warp_idx < Detail::kValidWarps) {
    int const warp_idx_n = warp_idx % Base::WarpCount::kN;
    int const warp_idx_k = warp_idx / Base::WarpCount::kN;
    this->warp_tile_iterator_A_.add_tile_offset(
        {0, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    this->warp_tile_iterator_E_.add_tile_offset(
        {0, Base::kWarpGemmIterations * warp_idx_k});
  }

 private:
  CUTLASS_DEVICE
  void copy_a(
      IteratorA& iterator_A) {
    iterator_A.set_iteration_index(0);
    smem_iterator_A_.set_iteration_index(0);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::TBLoadIterationsA; ++j) {
      auto* dst_ptr = reinterpret_cast<typename IteratorA::AccessType*>(
          smem_iterator_A_.get());
      int const kSrcBytes =
          sizeof_bits<typename IteratorA::Element>::value *
          IteratorA::ThreadMap::kElementsPerAccess /
          IteratorA::kAccessesPerVector / 8;
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
            dst_ptr + v, iterator_A.get(), iterator_A.valid());
        ++iterator_A;
      }
      ++smem_iterator_A_;
    }
    iterator_A.add_tile_offset({0, 1});
  }

  CUTLASS_DEVICE
  void copy_b(
      IteratorB& iterator_B) {
    iterator_B.set_iteration_index(0);
    smem_iterator_B_.set_iteration_index(0);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::TBLoadIterationsB; ++j) {
      auto* dst_ptr = reinterpret_cast<typename IteratorB::AccessType*>(
          smem_iterator_B_.get());
      int const kSrcBytes =
          sizeof_bits<typename IteratorB::Element>::value *
          IteratorB::ThreadMap::kElementsPerAccess /
          IteratorB::kAccessesPerVector / 8;
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
            dst_ptr + v, iterator_B.get(), iterator_B.valid());
        ++iterator_B;
      }
      ++smem_iterator_B_;
    }
    iterator_B.add_tile_offset({1, 0});
  }

  CUTLASS_DEVICE
  void copy_e(
      IteratorE& iterator_E) {
    iterator_E.set_iteration_index(0);
    smem_iterator_E_.set_iteration_index(0);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::TBLoadIterationsE; ++j) {
      auto* dst_ptr = reinterpret_cast<typename IteratorE::AccessType*>(
          smem_iterator_E_.get());
      int const kSrcBytes =
          sizeof_bits<typename IteratorE::Element>::value *
          IteratorE::ThreadMap::kElementsPerAccess / 8;
      if (is_warp_valid_) {
        cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpE>(
            dst_ptr, iterator_E.get(), iterator_E.valid());
      }
      ++iterator_E;
      ++smem_iterator_E_;
    }
    iterator_E.add_tile_offset({0, 1});
  }

  CUTLASS_DEVICE
  void copy_primary_and_advance(
      IteratorA& iterator_A,
      IteratorB& iterator_B,
      IteratorE& iterator_E) {
    copy_a(iterator_A);
    copy_b(iterator_B);
    copy_e(iterator_E);
    cutlass::arch::cp_async_fence();
  }

  CUTLASS_DEVICE
  void copy_secondary_and_advance(
      IteratorA& iterator_A,
      IteratorE& iterator_E) {
    copy_a(iterator_A);
    copy_e(iterator_E);
    cutlass::arch::cp_async_fence();
  }

 public:
  CUTLASS_DEVICE
  void operator()(
      int gemm_k_iterations,
      FragmentC& accum0,
      FragmentC& accum1,
      IteratorA iterator_A0,
      IteratorA iterator_A1,
      IteratorB iterator_B,
      IteratorE iterator_E0,
      IteratorE iterator_E1) {
    if (gemm_k_iterations <= 0) {
      return;
    }

    copy_primary_and_advance(iterator_A0, iterator_B, iterator_E0);
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    Operator warp_mma;
    WarpLoadedFragmentA loaded_A[2];
    WarpLoadedFragmentB loaded_B[2];
    WarpTransformedFragmentA transformed_A[2];
    WarpTransformedFragmentB transformed_B[2];
    WarpFragmentE fragment_E[2];

    CUTLASS_GEMM_LOOP
    for (int k_tile = 0; k_tile < gemm_k_iterations; ++k_tile) {
      // Primary panel.  B is retained in registers after these loads.
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < 2; ++group) {
        this->warp_tile_iterator_A_.set_kgroup_index(group);
        this->warp_tile_iterator_B_.set_kgroup_index(group);
        this->warp_tile_iterator_E_.set_kgroup_index(group);
        this->warp_tile_iterator_A_.load(loaded_A[group]);
        this->warp_tile_iterator_B_.load(loaded_B[group]);
        this->warp_tile_iterator_E_.load(fragment_E[group]);
        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;
        ++this->warp_tile_iterator_E_;
        warp_mma.transform(
            transformed_A[group], transformed_B[group],
            loaded_A[group], loaded_B[group]);
      }
      this->warp_tile_iterator_A_.add_tile_offset({0, -2});
      this->warp_tile_iterator_B_.add_tile_offset({-2, 0});
      this->warp_tile_iterator_E_.add_tile_offset({0, -2});

      // A/E are no longer read from shared memory.  Stream the second weight
      // panel while HMMA.SP consumes the first panel and resident B fragment.
      __syncthreads();
      copy_secondary_and_advance(iterator_A1, iterator_E1);
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < 2; ++group) {
        warp_mma(
            accum0, transformed_A[group], transformed_B[group],
            accum0, fragment_E[group]);
      }
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();

      // Secondary panel: reload only A/E.  The same loaded_B registers feed
      // its two HMMA.SP groups.
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < 2; ++group) {
        this->warp_tile_iterator_A_.set_kgroup_index(group);
        this->warp_tile_iterator_E_.set_kgroup_index(group);
        this->warp_tile_iterator_A_.load(loaded_A[group]);
        this->warp_tile_iterator_E_.load(fragment_E[group]);
        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_E_;
        warp_mma.transform(
            transformed_A[group], transformed_B[group],
            loaded_A[group], loaded_B[group]);
      }
      this->warp_tile_iterator_A_.add_tile_offset({0, -2});
      this->warp_tile_iterator_E_.add_tile_offset({0, -2});

      bool const has_next = (k_tile + 1 < gemm_k_iterations);
      if (has_next) {
        __syncthreads();
        copy_primary_and_advance(iterator_A0, iterator_B, iterator_E0);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < 2; ++group) {
        warp_mma(
            accum1, transformed_A[group], transformed_B[group],
            accum1, fragment_E[group]);
      }
      if (has_next) {
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();
      }
    }
  }
};

// Asymmetric pipeline for the practical separate-kernel path.  Four B
// (activation) stages stay resident, while A/E use one streamed shared tile.
// Once the current A/E/B fragments are in registers, the freed A/E tile and B
// circular-buffer slot are filled concurrently with HMMA.SP.  Compared with
// stock SparseMmaMultistage<4>, this removes three A/E shared-memory stages
// without shortening the activation lookahead.
template <
    typename Shape_,
    typename IteratorA_,
    typename SmemIteratorA_,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    typename IteratorB_,
    typename SmemIteratorB_,
    cutlass::arch::CacheOperation::Kind CacheOpB,
    typename ElementC_,
    typename LayoutC_,
    typename IteratorE_,
    typename SmemIteratorE_,
    cutlass::arch::CacheOperation::Kind CacheOpE,
    typename Policy_,
    int BStages = 4,
    int AStages = 1>
class SparseMmaBResidentAStreamed {
 public:
  using Shape = Shape_;
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using IteratorE = IteratorE_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = Policy_;
  using Operator = typename Policy::Operator;
  using FragmentC = typename Operator::FragmentC;
  using ElementE = typename IteratorE::Element;
  using LayoutE = typename IteratorE::Layout;
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  using SmemIteratorE = SmemIteratorE_;
  using WarpGemm = typename Operator::Shape;
  using WarpCount = GemmShape<
      Shape::kM / WarpGemm::kM,
      Shape::kN / WarpGemm::kN,
      Shape::kK / WarpGemm::kK>;
  using ArchTag = arch::Sm80;

  static int const kWarpGemmIterations =
      WarpGemm::kK / Operator::Policy::MmaShape::kK;
  static int const kSparse = Operator::kSparse;
  static int const kMetaSizeInBits = Operator::kMetaSizeInBits;
  static int const kMaxID2 = Operator::kMaxID2;
  static int const kElementsPerElementE = Operator::kElementsPerElementE;
  static ComplexTransform const kTransformA = Operator::kTransformA;
  static ComplexTransform const kTransformB = Operator::kTransformB;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
  static cutlass::arch::CacheOperation::Kind const kCacheOpE = CacheOpE;

  static_assert(BStages >= 2, "B-resident pipeline needs at least two B stages");
  static_assert(AStages == 1 || AStages == 2,
                "streamed A/E pipeline supports one or two stages");
  static_assert(kWarpGemmIterations == 2,
                "B-resident pipeline requires two warp-K groups");

  using TensorRefA = TensorRef<typename Operator::ElementA,
                               typename Operator::LayoutA>;
  using TensorRefB = TensorRef<typename Operator::ElementB,
                               typename Operator::LayoutB>;
  using TensorRefE = TensorRef<typename Operator::ElementE,
                               typename Operator::LayoutE>;

  class SharedStorage {
   public:
    using ShapeA = MatrixShape<
        Shape::kM + Policy::SmemPaddingA::kRow,
        Shape::kK / kSparse * AStages + Policy::SmemPaddingA::kColumn>;
    using ShapeB = MatrixShape<
        Shape::kK * BStages + Policy::SmemPaddingB::kRow,
        Shape::kN + Policy::SmemPaddingB::kColumn>;
    using ShapeE = MatrixShape<
        Shape::kM * 2 + Policy::SmemPaddingE::kRow,
        Shape::kK / kSparse / kElementsPerElementE / 2 * AStages +
            Policy::SmemPaddingE::kColumn>;

    AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;
    AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;
    AlignedBuffer<typename Operator::ElementE, ShapeE::kCount> operand_E;

    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutA LayoutA() {
      return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
    }
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutB LayoutB() {
      return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
    }
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutE LayoutE() {
      return Operator::LayoutE::packed({ShapeE::kRow, ShapeE::kColumn});
    }
    CUTLASS_HOST_DEVICE TensorRefA operand_A_ref() {
      return TensorRefA{operand_A.data(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE TensorRefB operand_B_ref() {
      return TensorRefB{operand_B.data(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE TensorRefE operand_E_ref() {
      return TensorRefE{operand_E.data(), LayoutE()};
    }
  };

  struct Detail {
    static int const TBLoadIterationsA =
        IteratorA::ThreadMap::Iterations::kCount;
    static int const TBLoadIterationsB =
        IteratorB::ThreadMap::Iterations::kCount;
    static int const TBLoadIterationsE =
        IteratorE::ThreadMap::Iterations::kCount;
    static int const kValidWarps = IteratorE::ThreadMap::kThreads / 32;
  };

 private:
  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;
  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;
  using WarpFragmentE = typename Operator::FragmentE;

  typename Operator::IteratorA warp_tile_iterator_A_;
  typename Operator::IteratorB warp_tile_iterator_B_;
  typename Operator::IteratorE warp_tile_iterator_E_;
  SmemIteratorA smem_iterator_A_;
  SmemIteratorB smem_iterator_B_;
  SmemIteratorE smem_iterator_E_;
  bool is_warp_valid_;

 public:
  CUTLASS_DEVICE
  SparseMmaBResidentAStreamed(
      SharedStorage& shared,
      int thread_idx,
      int warp_idx,
      int lane_idx)
      : warp_tile_iterator_A_(shared.operand_A_ref(), lane_idx),
        warp_tile_iterator_B_(shared.operand_B_ref(), lane_idx),
        warp_tile_iterator_E_(shared.operand_E_ref(), lane_idx),
        smem_iterator_A_(shared.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared.operand_B_ref(), thread_idx),
        smem_iterator_E_(shared.operand_E_ref(), thread_idx),
        is_warp_valid_(warp_idx < Detail::kValidWarps) {
    int const warp_idx_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int const warp_idx_k = warp_idx / (WarpCount::kM * WarpCount::kN);
    int const warp_idx_m = warp_idx_mn % WarpCount::kM;
    int const warp_idx_n = warp_idx_mn / WarpCount::kM;
    warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, kWarpGemmIterations * warp_idx_k});
    warp_tile_iterator_B_.add_tile_offset(
        {kWarpGemmIterations * warp_idx_k, warp_idx_n});
    warp_tile_iterator_E_.add_tile_offset(
        {warp_idx_m, kWarpGemmIterations * warp_idx_k});
  }

 private:
  CUTLASS_DEVICE void copy_a(IteratorA& iterator) {
    iterator.set_iteration_index(0);
    smem_iterator_A_.set_iteration_index(0);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::TBLoadIterationsA; ++j) {
      auto* destination = reinterpret_cast<typename IteratorA::AccessType*>(
          smem_iterator_A_.get());
      int const bytes = sizeof_bits<typename IteratorA::Element>::value *
          IteratorA::ThreadMap::kElementsPerAccess /
          IteratorA::kAccessesPerVector / 8;
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
        cutlass::arch::cp_async_zfill<bytes, kCacheOpA>(
            destination + v, iterator.get(), iterator.valid());
        ++iterator;
      }
      ++smem_iterator_A_;
    }
    iterator.add_tile_offset({0, 1});
  }

  CUTLASS_DEVICE void copy_b(IteratorB& iterator) {
    iterator.set_iteration_index(0);
    smem_iterator_B_.set_iteration_index(0);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::TBLoadIterationsB; ++j) {
      auto* destination = reinterpret_cast<typename IteratorB::AccessType*>(
          smem_iterator_B_.get());
      int const bytes = sizeof_bits<typename IteratorB::Element>::value *
          IteratorB::ThreadMap::kElementsPerAccess /
          IteratorB::kAccessesPerVector / 8;
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
        cutlass::arch::cp_async_zfill<bytes, kCacheOpB>(
            destination + v, iterator.get(), iterator.valid());
        ++iterator;
      }
      ++smem_iterator_B_;
    }
    iterator.add_tile_offset({1, 0});
  }

  CUTLASS_DEVICE void copy_e(IteratorE& iterator) {
    iterator.set_iteration_index(0);
    smem_iterator_E_.set_iteration_index(0);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::TBLoadIterationsE; ++j) {
      auto* destination = reinterpret_cast<typename IteratorE::AccessType*>(
          smem_iterator_E_.get());
      int const bytes = sizeof_bits<typename IteratorE::Element>::value *
          IteratorE::ThreadMap::kElementsPerAccess / 8;
      if (is_warp_valid_) {
        cutlass::arch::cp_async_zfill<bytes, kCacheOpE>(
            destination, iterator.get(), iterator.valid());
      }
      ++iterator;
      ++smem_iterator_E_;
    }
    iterator.add_tile_offset({0, 1});
  }

 public:
  CUTLASS_DEVICE
  void operator()(
      int gemm_k_iterations,
      FragmentC& accum,
      IteratorA iterator_A,
      IteratorB iterator_B,
      IteratorE iterator_E,
      FragmentC const& src_accum) {
    accum = src_accum;
    if (gemm_k_iterations <= 0) {
      return;
    }

    if constexpr (AStages == 2) {
      // Deep activation lookahead plus a two-stage residual stream.  At most
      // one newly issued cp.async group remains outstanding; it always writes
      // the stage after the one consumed by the next iteration.
      CUTLASS_PRAGMA_UNROLL
      for (int stage = 0; stage < BStages; ++stage) {
        iterator_B.clear_mask(stage >= gemm_k_iterations);
        copy_b(iterator_B);
        smem_iterator_B_.add_tile_offset({1, 0});
        cutlass::arch::cp_async_fence();
      }
      smem_iterator_B_.add_tile_offset({-BStages, 0});

      CUTLASS_PRAGMA_UNROLL
      for (int stage = 0; stage < AStages; ++stage) {
        iterator_A.clear_mask(stage >= gemm_k_iterations);
        iterator_E.clear_mask(stage >= gemm_k_iterations);
        copy_a(iterator_A);
        copy_e(iterator_E);
        smem_iterator_A_.add_tile_offset({0, 1});
        smem_iterator_E_.add_tile_offset({0, 1});
        cutlass::arch::cp_async_fence();
      }
      smem_iterator_A_.add_tile_offset({0, -AStages});
      smem_iterator_E_.add_tile_offset({0, -AStages});
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();

      Operator warp_mma;
      WarpLoadedFragmentA loaded_A[2];
      WarpLoadedFragmentB loaded_B[2];
      WarpTransformedFragmentA transformed_A[2];
      WarpTransformedFragmentB transformed_B[2];
      WarpFragmentE fragment_E[2];
      int a_read_stage = 0;
      int b_read_stage = 0;
      int a_write_stage = 0;
      int b_write_stage = 0;

      CUTLASS_GEMM_LOOP
      for (int k_tile = 0; k_tile < gemm_k_iterations; ++k_tile) {
        CUTLASS_PRAGMA_UNROLL
        for (int group = 0; group < 2; ++group) {
          warp_tile_iterator_A_.set_kgroup_index(group);
          warp_tile_iterator_B_.set_kgroup_index(group);
          warp_tile_iterator_E_.set_kgroup_index(group);
          warp_tile_iterator_A_.load(loaded_A[group]);
          warp_tile_iterator_B_.load(loaded_B[group]);
          warp_tile_iterator_E_.load(fragment_E[group]);
          ++warp_tile_iterator_A_;
          ++warp_tile_iterator_B_;
          ++warp_tile_iterator_E_;
          warp_mma.transform(
              transformed_A[group], transformed_B[group],
              loaded_A[group], loaded_B[group]);
        }

        ++a_read_stage;
        if (a_read_stage == AStages) {
          warp_tile_iterator_A_.add_tile_offset(
              {0, -AStages * kWarpGemmIterations});
          warp_tile_iterator_E_.add_tile_offset(
              {0, -AStages * kWarpGemmIterations});
          a_read_stage = 0;
        }
        ++b_read_stage;
        if (b_read_stage == BStages) {
          warp_tile_iterator_B_.add_tile_offset(
              {-BStages * kWarpGemmIterations, 0});
          b_read_stage = 0;
        }

        bool const has_next = (k_tile + 1 < gemm_k_iterations);
        if (has_next) {
          __syncthreads();
          if (k_tile + AStages < gemm_k_iterations) {
            copy_a(iterator_A);
            copy_e(iterator_E);
            smem_iterator_A_.add_tile_offset({0, 1});
            smem_iterator_E_.add_tile_offset({0, 1});
            ++a_write_stage;
            if (a_write_stage == AStages) {
              smem_iterator_A_.add_tile_offset({0, -AStages});
              smem_iterator_E_.add_tile_offset({0, -AStages});
              a_write_stage = 0;
            }
          }
          if (k_tile + BStages < gemm_k_iterations) {
            copy_b(iterator_B);
            smem_iterator_B_.add_tile_offset({1, 0});
            ++b_write_stage;
            if (b_write_stage == BStages) {
              smem_iterator_B_.add_tile_offset({-BStages, 0});
              b_write_stage = 0;
            }
          }
          cutlass::arch::cp_async_fence();
        }

        CUTLASS_PRAGMA_UNROLL
        for (int group = 0; group < 2; ++group) {
          warp_mma(
              accum, transformed_A[group], transformed_B[group],
              accum, fragment_E[group]);
        }
        if (has_next) {
          cutlass::arch::cp_async_wait<1>();
          __syncthreads();
        }
      }
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();
      return;
    }

    // Activation prologue: retain BStages K tiles in its circular buffer.
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < BStages; ++stage) {
      iterator_B.clear_mask(stage >= gemm_k_iterations);
      copy_b(iterator_B);
      smem_iterator_B_.add_tile_offset({1, 0});
      cutlass::arch::cp_async_fence();
    }
    smem_iterator_B_.add_tile_offset({-BStages, 0});

    // Only the current residual weight and metadata tile is materialized.
    copy_a(iterator_A);
    copy_e(iterator_E);
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    Operator warp_mma;
    WarpLoadedFragmentA loaded_A[2];
    WarpLoadedFragmentB loaded_B[2];
    WarpTransformedFragmentA transformed_A[2];
    WarpTransformedFragmentB transformed_B[2];
    WarpFragmentE fragment_E[2];
    int b_read_stage = 0;

    CUTLASS_GEMM_LOOP
    for (int k_tile = 0; k_tile < gemm_k_iterations; ++k_tile) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < 2; ++group) {
        warp_tile_iterator_A_.set_kgroup_index(group);
        warp_tile_iterator_B_.set_kgroup_index(group);
        warp_tile_iterator_E_.set_kgroup_index(group);
        warp_tile_iterator_A_.load(loaded_A[group]);
        warp_tile_iterator_B_.load(loaded_B[group]);
        warp_tile_iterator_E_.load(fragment_E[group]);
        ++warp_tile_iterator_A_;
        ++warp_tile_iterator_B_;
        ++warp_tile_iterator_E_;
        warp_mma.transform(
            transformed_A[group], transformed_B[group],
            loaded_A[group], loaded_B[group]);
      }
      warp_tile_iterator_A_.add_tile_offset({0, -2});
      warp_tile_iterator_E_.add_tile_offset({0, -2});

      bool const has_next = (k_tile + 1 < gemm_k_iterations);
      if (has_next) {
        // All warps have captured the current operands.  Refill the streamed
        // A/E tile and the B slot that has just become free.
        __syncthreads();
        copy_a(iterator_A);
        copy_e(iterator_E);
        if (k_tile + BStages < gemm_k_iterations) {
          copy_b(iterator_B);
        }
        smem_iterator_B_.add_tile_offset({1, 0});
        if ((k_tile % BStages) == BStages - 1) {
          smem_iterator_B_.add_tile_offset({-BStages, 0});
        }
        cutlass::arch::cp_async_fence();
      }

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < 2; ++group) {
        warp_mma(
            accum, transformed_A[group], transformed_B[group],
            accum, fragment_E[group]);
      }

      ++b_read_stage;
      if (b_read_stage == BStages) {
        warp_tile_iterator_B_.add_tile_offset(
            {-BStages * kWarpGemmIterations, 0});
        b_read_stage = 0;
      }
      if (has_next) {
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();
      }
    }
  }
};

}  // namespace threadblock

namespace kernel {

// Visitor kernel counterpart for the two-accumulator mainloop above.  The
// logical grid advances by 128 feature rows, while each epilogue invocation
// writes one physical 64-row panel.
template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_>
struct ActivationStationarySparseGemmWithEpilogueVisitor :
    public SparseGemmWithEpilogueVisitor<
        Mma_, Epilogue_, ThreadblockSwizzle_> {
  using Base = SparseGemmWithEpilogueVisitor<
      Mma_, Epilogue_, ThreadblockSwizzle_>;
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Params = typename Base::Params;
  using SharedStorage = typename Base::SharedStorage;
  using FusionCallbacks = typename Epilogue::FusionCallbacks;

  static int const kSparse = Base::kSparse;
  static int const kElementsPerElementE = Base::kElementsPerElementE;

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage) {
    ThreadblockSwizzle swizzle;
    cutlass::gemm::GemmCoord tile =
        swizzle.get_tile_offset(params.swizzle_log_tile);
    if (params.grid_tiled_shape.m() <= tile.m() ||
        params.grid_tiled_shape.n() <= tile.n()) {
      return;
    }

    cutlass::MatrixCoord offset_A0{
        tile.m() * Mma::Shape::kM,
        tile.k() * params.gemm_k_size / kSparse};
    cutlass::MatrixCoord offset_A1{
        offset_A0.row() + Mma::PhysicalShape::kM,
        offset_A0.column()};
    cutlass::MatrixCoord offset_B{
        tile.k() * params.gemm_k_size,
        tile.n() * Mma::Shape::kN};
    cutlass::MatrixCoord offset_E0 = offset_A0;
    cutlass::MatrixCoord offset_E1 = offset_A1;

    int const problem_k = min(
        params.problem_size.k(),
        (tile.k() + 1) * params.gemm_k_size);
    int const k_iterations =
        (problem_k - offset_B.row() + Mma::Shape::kK - 1) /
        Mma::Shape::kK;
    int const thread_idx = threadIdx.x;

    typename Mma::IteratorA iterator_A0(
        params.params_A, params.ref_A.data(),
        {params.problem_size.m(), problem_k / kSparse},
        thread_idx, offset_A0);
    typename Mma::IteratorA iterator_A1(
        params.params_A, params.ref_A.data(),
        {params.problem_size.m(), problem_k / kSparse},
        thread_idx, offset_A1);
    typename Mma::IteratorB iterator_B(
        params.params_B, params.ref_B.data(),
        {problem_k, params.problem_size.n()},
        thread_idx, offset_B);
    typename Mma::IteratorE iterator_E0(
        params.params_E, params.ref_E.data(),
        {params.problem_size.m(),
         problem_k / kSparse / kElementsPerElementE},
        thread_idx, offset_E0);
    typename Mma::IteratorE iterator_E1(
        params.params_E, params.ref_E.data(),
        {params.problem_size.m(),
         problem_k / kSparse / kElementsPerElementE},
        thread_idx, offset_E1);

    int const warp_idx = canonical_warp_idx_sync();
    int const lane_idx = threadIdx.x % 32;
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC accum0;
    typename Mma::FragmentC accum1;
    accum0.clear();
    accum1.clear();
    if (k_iterations > 0) {
      mma(
          k_iterations, accum0, accum1,
          iterator_A0, iterator_A1, iterator_B,
          iterator_E0, iterator_E1);
    }

    // Epilogue is configured for the physical 64-row panel.  Convert the
    // logical 128-row grid coordinate into its two physical coordinates.
    cutlass::gemm::GemmCoord epilogue_tile0{
        tile.m() * 2, tile.n(), tile.k()};
    cutlass::gemm::GemmCoord epilogue_tile1{
        tile.m() * 2 + 1, tile.n(), tile.k()};
    Epilogue epilogue0(
        params.output_op, shared_storage.epilogue,
        thread_idx, warp_idx, lane_idx);
    epilogue0(
        accum0, epilogue_tile0, params.problem_shape, thread_idx);
    __syncthreads();
    Epilogue epilogue1(
        params.output_op, shared_storage.epilogue,
        thread_idx, warp_idx, lane_idx);
    epilogue1(
        accum1, epilogue_tile1, params.problem_shape, thread_idx);
  }
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
