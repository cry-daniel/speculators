/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * This file is a small, project-local specialization of CUTLASS's SM80 sparse
 * multistage mainloop.  It retains one shared-memory tile and preloads both
 * warp-K fragments into registers before asynchronously overwriting that tile.
 **************************************************************************************************/

#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/mma_sparse_base.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

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
    typename Policy_>
class SparseMmaSingleSmem :
    public SparseMmaBase<Shape_, Policy_, 1> {
 public:
  using Base = SparseMmaBase<Shape_, Policy_, 1>;
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
  using SharedStorage = typename Base::SharedStorage;
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
  SparseMmaSingleSmem(
      SharedStorage& shared_storage,
      int thread_idx,
      int warp_idx,
      int lane_idx)
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
        smem_iterator_E_(shared_storage.operand_E_ref(), thread_idx),
        is_warp_valid_(warp_idx < Detail::kValidWarps) {
    int warp_idx_mn = warp_idx %
        (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx /
        (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    this->warp_tile_iterator_E_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
  }

 private:
  CUTLASS_DEVICE
  void copy_tile_and_advance(
      IteratorA& iterator_A,
      IteratorB& iterator_B,
      IteratorE& iterator_E) {
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

    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});
    iterator_E.add_tile_offset({0, 1});
    cutlass::arch::cp_async_fence();
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
    static_assert(
        Base::kWarpGemmIterations == 2,
        "single-SMEM schedule currently requires two warp-K groups");

    accum = src_accum;
    if (gemm_k_iterations <= 0) {
      return;
    }

    // Prologue: materialize the first K tile.
    copy_tile_and_advance(iterator_A, iterator_B, iterator_E);
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    Operator warp_mma;
    WarpLoadedFragmentA warp_loaded_A[2];
    WarpLoadedFragmentB warp_loaded_B[2];
    WarpTransformedFragmentA warp_transformed_A[2];
    WarpTransformedFragmentB warp_transformed_B[2];
    WarpFragmentE warp_E[2];

    CUTLASS_GEMM_LOOP
    for (int k_tile = 0; k_tile < gemm_k_iterations; ++k_tile) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < 2; ++group) {
        this->warp_tile_iterator_A_.set_kgroup_index(group);
        this->warp_tile_iterator_B_.set_kgroup_index(group);
        this->warp_tile_iterator_E_.set_kgroup_index(group);
        this->warp_tile_iterator_A_.load(warp_loaded_A[group]);
        this->warp_tile_iterator_B_.load(warp_loaded_B[group]);
        this->warp_tile_iterator_E_.load(warp_E[group]);
        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;
        ++this->warp_tile_iterator_E_;
        warp_mma.transform(
            warp_transformed_A[group],
            warp_transformed_B[group],
            warp_loaded_A[group],
            warp_loaded_B[group]);
      }

      // Every warp has consumed shared memory.  The next tile may now replace
      // it while HMMA.SP consumes the register-resident fragments.
      bool const has_next = (k_tile + 1 < gemm_k_iterations);
      if (has_next) {
        __syncthreads();
        copy_tile_and_advance(iterator_A, iterator_B, iterator_E);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < 2; ++group) {
        warp_mma(
            accum,
            warp_transformed_A[group],
            warp_transformed_B[group],
            accum,
            warp_E[group]);
      }

      this->warp_tile_iterator_A_.add_tile_offset({0, -2});
      this->warp_tile_iterator_B_.add_tile_offset({-2, 0});
      this->warp_tile_iterator_E_.add_tile_offset({0, -2});

      if (has_next) {
        cutlass::arch::cp_async_wait<0>();
        __syncthreads();
      }
    }
  }
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
