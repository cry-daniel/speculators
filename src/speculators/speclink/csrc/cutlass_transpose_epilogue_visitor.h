/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// A CUTLASS 2.x Sm80 EVT root store adapted from VisitorAuxStore.
//
// SparseGemmWithVisitor computes logical D[N, M].  This visitor transposes each
// TensorOp epilogue step through callback shared memory and writes directly to
// contiguous row-major Y[M, N].  It therefore avoids both an intermediate
// D[N, M] allocation and a second transpose kernel.

#pragma once

#include <cstddef>
#include <cstdint>

// Include the device wrapper before visitors.hpp.  CUTLASS's 2.x visitor
// headers depend on GEMM and epilogue thread-map declarations being visible.
#include "cutlass/arch/memory.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

namespace speculators::speclink {

template <class ThreadMap, class Element, cutlass::FloatRoundStyle RoundStyle,
          int CtaM, int CtaN, int WarpM>
struct VisitorTransposeAuxStore {
  static constexpr int kRowsPerStep = 8;
  static constexpr int kWarpRowGroups = CtaM / WarpM;
  static constexpr int kThreads = ThreadMap::Base::kThreads;
  static constexpr int kElementsPerThread =
      (kRowsPerStep * CtaN) / kThreads;
  static constexpr int kPad = 8;

  static_assert(CtaM % WarpM == 0,
                "CTA M must be an integer number of warp M tiles");
  static_assert(WarpM % kRowsPerStep == 0,
                "warp M must be a multiple of a TensorOp epilogue step");
  static_assert((kRowsPerStep * CtaN) % kThreads == 0,
                "transpose tile must divide evenly over epilogue threads");
  static_assert(kElementsPerThread == 4 || kElementsPerThread == 8,
                "visitor supports one 64-bit or 128-bit store per thread");

  struct Arguments {
    Element* ptr_output = nullptr;
    // Optional distance, in elements, between parallel split-K partial
    // outputs.  A zero stride preserves the normal single-slice ABI.
    int64_t split_k_stride = 0;
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  // A step contains one independent 8-row band for every warp row in the CTA.
  // The +8 column padding avoids repeated bank mappings when CTA_N=64 and two
  // threads gather different output-channel halves for the same token.
  struct SharedStorage {
    alignas(16) Element
        tile[kWarpRowGroups * kRowsPerStep][CtaN + kPad];
  };

  CUTLASS_HOST_DEVICE VisitorTransposeAuxStore() {}

  CUTLASS_HOST_DEVICE
  VisitorTransposeAuxStore(Params const& params,
                           SharedStorage const& shared_storage)
      : params_ptr(&params),
        shared_tile(const_cast<Element*>(&shared_storage.tile[0][0])) {}

  Params const* params_ptr = nullptr;
  Element* shared_tile = nullptr;

  template <class RTensor, class CTensor, class ProblemShape>
  struct Callbacks
      : cutlass::epilogue::threadblock::detail::EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(RTensor&& tC_r_output, CTensor&& tC_c_output,
              ProblemShape problem_shape, Params const* params_ptr,
              Element* shared_tile,
              cutlass::gemm::GemmCoord threadblock_tile_offset,
              int thread_idx)
        : tC_r_output(cute::forward<RTensor>(tC_r_output)),
          tC_c_output(cute::forward<CTensor>(tC_c_output)),
          problem_shape(problem_shape),
          params_ptr(params_ptr),
          shared_tile(shared_tile),
          threadblock_tile_offset(threadblock_tile_offset),
          thread_idx(thread_idx) {}

    RTensor tC_r_output;
    CTensor tC_c_output;
    ProblemShape problem_shape;
    Params const* params_ptr;
    Element* shared_tile;
    cutlass::gemm::GemmCoord threadblock_tile_offset;
    int thread_idx;

    CUTLASS_DEVICE void begin_step(int) { cute::clear(tC_r_output); }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto visit(
        int, int, int, int fragment_idx,
        cutlass::Array<ElementAccumulator, FragmentSize> const&,
        cutlass::Array<ElementInput, FragmentSize> const& fragment_input) {
      using Convert = cutlass::NumericArrayConverter<
          Element, ElementInput, FragmentSize, RoundStyle>;
      Convert convert{};
      auto register_frag = cute::recast<cutlass::Array<Element, FragmentSize>>(
          cute::coalesce(tC_r_output));
      register_frag(fragment_idx) = convert(fragment_input);
      return fragment_input;
    }

    CUTLASS_DEVICE void end_step(int step_idx) {
      // CUTLASS's output thread map is authoritative here.  Mapping its exact
      // logical coordinates into a canonical shared tile is robust to its lane,
      // warp-row, group, and cluster decomposition.
      auto src = cute::filter(tC_r_output);
      auto coord = cute::filter(tC_c_output(cute::_, cute::_, cute::_,
                                            step_idx));
      int tile_m = threadblock_tile_offset.m() * CtaM;
      int tile_n = threadblock_tile_offset.n() * CtaN;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(src); ++i) {
        auto logical_coord = coord(i);
        bool guard = cute::elem_less(logical_coord, problem_shape);
        int logical_m = int(cute::get<0>(logical_coord));
        int logical_n = int(cute::get<1>(logical_coord));
        int tile_local_m = logical_m - tile_m;
        int warp_row_group = tile_local_m / WarpM;
        int local_row = tile_local_m - warp_row_group * WarpM -
                        step_idx * kRowsPerStep;
        int local_col = logical_n - tile_n;
        if (guard && warp_row_group >= 0 &&
            warp_row_group < kWarpRowGroups && local_row >= 0 &&
            local_row < kRowsPerStep && local_col >= 0 &&
            local_col < CtaN) {
          shared_tile[(warp_row_group * kRowsPerStep + local_row) *
                          (CtaN + kPad) +
                      local_col] = src(i);
        }
      }

      __syncthreads();

      // Regroup the shared step so every global store is contiguous in output
      // channels for one token.  Depending on the CTA and warp geometry this
      // produces one 64-bit or 128-bit store per thread and warp-row group.
      constexpr int kStoreBits =
          kElementsPerThread * cutlass::sizeof_bits<Element>::value;
      using StoreType = cute::uint_bit_t<kStoreBits>;
      cutlass::Array<Element, kElementsPerThread> output_fragment;
      int linear = thread_idx * kElementsPerThread;
      int local_token = linear / kRowsPerStep;
      int output_in_step = linear % kRowsPerStep;
      int token = tile_n + local_token;
      int output_channels = int(cute::get<0>(problem_shape));
      int tokens = int(cute::get<1>(problem_shape));
      int64_t split_k_offset =
          int64_t(blockIdx.z) * params_ptr->split_k_stride;

      CUTLASS_PRAGMA_UNROLL
      for (int warp_row_group = 0; warp_row_group < kWarpRowGroups;
           ++warp_row_group) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerThread; ++i) {
          output_fragment[i] =
              shared_tile[(warp_row_group * kRowsPerStep + output_in_step +
                           i) *
                              (CtaN + kPad) +
                          local_token];
        }

        int output_channel = tile_m + warp_row_group * WarpM +
                             step_idx * kRowsPerStep + output_in_step;
        bool full_guard =
            token < tokens &&
            output_channel + kElementsPerThread <= output_channels;
        StoreType packed = reinterpret_cast<StoreType const&>(output_fragment);
        cutlass::arch::global_store<StoreType, sizeof(StoreType)>(
            packed,
            static_cast<void*>(params_ptr->ptr_output +
                               split_k_offset +
                               int64_t(token) * output_channels +
                               output_channel),
            full_guard);

        // Aligned LLM linear dimensions use only the vector path.  This keeps
        // the visitor correct for a partial final CTA as well.
        if (token < tokens && !full_guard) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kElementsPerThread; ++i) {
            if (output_channel + i < output_channels) {
              params_ptr->ptr_output[split_k_offset +
                                     int64_t(token) * output_channels +
                                     output_channel + i] = output_fragment[i];
            }
          }
        }
      }

      // The next epilogue step reuses this callback shared-memory tile.
      __syncthreads();
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(
      cutlass::gemm::GemmCoord threadblock_tile_offset, int thread_idx,
      ProblemShape problem_shape) {
    using namespace cute;

    // This logical row-major D tensor is used only to instantiate CUTLASS's
    // output partition and coordinate map.  Writes use ptr_output as Y[M,N]
    // explicitly in end_step().
    Tensor logical_output = make_tensor(
        make_gmem_ptr(params_ptr->ptr_output), problem_shape,
        make_stride(int64_t(get<1>(problem_shape)), _1{}, int64_t(0)));
    Tensor partitioned = group_modes<3, 6>(ThreadMap::partition(
        logical_output, thread_idx, threadblock_tile_offset));
    Tensor registers = make_tensor_like(take<0, 3>(partitioned));

    Tensor identity = make_identity_tensor(logical_output.shape());
    Tensor coordinates = group_modes<3, 6>(
        ThreadMap::partition(identity, thread_idx, threadblock_tile_offset));

    return Callbacks<decltype(registers), decltype(coordinates), ProblemShape>(
        cute::move(registers), cute::move(coordinates), problem_shape,
        params_ptr, shared_tile, threadblock_tile_offset, thread_idx);
  }
};

// Transpose/store visitor for a paired dense+sparse CTA.  Logical token
// columns [0, CtaN/2) are written to the dense branch output and columns
// [CtaN/2, CtaN) to the sparse branch output.  Successive CTA-N tiles advance
// both branch row indices by CtaN/2, so the two outputs remain independently
// contiguous and require no post-kernel deinterleave copy.
template <class ThreadMap, class Element, cutlass::FloatRoundStyle RoundStyle,
          int CtaM, int CtaN, int WarpM,
          int DenseBranchTokens = CtaN / 2,
          bool VectorizeCallbackSharedStore = false,
          bool PartitionedEpilogueReleaseBarrier = true>
struct VisitorPairedTransposeAuxStore {
  static constexpr int kRowsPerStep = 8;
  static constexpr int kDenseBranchTokens = DenseBranchTokens;
  static constexpr int kSparseBranchTokens = CtaN - DenseBranchTokens;
  static constexpr int kWarpRowGroups = CtaM / WarpM;
  static constexpr int kThreads = ThreadMap::Base::kThreads;
  static constexpr int kElementsPerThread =
      (kRowsPerStep * CtaN) / kThreads;
  static constexpr int kPad = 8;

  static_assert(kDenseBranchTokens >= 0 && kSparseBranchTokens >= 0 &&
                    kDenseBranchTokens + kSparseBranchTokens == CtaN,
                "paired visitor branch regions must partition CTA-N");
  static_assert(CtaM % WarpM == 0,
                "CTA M must be an integer number of warp M tiles");
  static_assert(WarpM % kRowsPerStep == 0,
                "warp M must be a multiple of a TensorOp epilogue step");
  static_assert((kRowsPerStep * CtaN) % kThreads == 0,
                "transpose tile must divide evenly over epilogue threads");
  static_assert(kElementsPerThread == 4 || kElementsPerThread == 8,
                "visitor supports one 64-bit or 128-bit store per thread");

  struct Arguments {
    Element* ptr_dense = nullptr;
    Element* ptr_sparse = nullptr;
    int dense_rows = 0;
    int sparse_rows = 0;
    // Routed mode writes directly to one [M,N] output.  The two optional
    // index arrays map branch-local rows back to their original token rows.
    Element* ptr_output = nullptr;
    int64_t const* dense_indices = nullptr;
    int64_t const* sparse_indices = nullptr;
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  struct SharedStorage {
    alignas(16) Element
        tile[kWarpRowGroups * kRowsPerStep][CtaN + kPad];
  };

  // The vector callback path is restricted to the two wide N=256 geometries
  // whose CUTLASS output maps give each physical warp one contiguous
  // eight-BF16 source vector per thread.  BM64 additionally partitions the
  // transpose/gather by warp-M group so it uses two independent 256-thread
  // named barriers instead of two CTA-wide barriers per epilogue step.
  static_assert(
      !VectorizeCallbackSharedStore ||
          ((CtaM == 32 || CtaM == 64) && CtaN == 256 && WarpM == 32),
      "vector callback store is valid only for wide 32/64x256 tiles");
  static_assert(
      !VectorizeCallbackSharedStore || sizeof(Element) == 2,
      "vector callback store requires 16-bit output elements");
  static_assert(
      !VectorizeCallbackSharedStore ||
          ThreadMap::Base::kElementsPerAccess == 8,
      "vector callback store requires eight elements per thread-map access");
  static_assert(
      !VectorizeCallbackSharedStore ||
          (ThreadMap::Base::Shape::kColumn == 256 &&
           ThreadMap::Base::Shape::kRow == 8),
      "vector callback store requires the proven 256x8 output shape");
  static_assert(
      !VectorizeCallbackSharedStore ||
          (ThreadMap::Base::Detail::kAccessWidth == 32 &&
           ThreadMap::Base::Detail::kAccessRows == 1),
      "vector callback store requires the proven one-dimensional lane map");
  static_assert(
      !VectorizeCallbackSharedStore ||
          (ThreadMap::Base::Iterations::kColumn == 1 &&
           ThreadMap::Base::Iterations::kRow == 1),
      "vector callback store requires one column and row iteration");
  static_assert(
      !VectorizeCallbackSharedStore ||
          ((CtaM == 32 && kThreads == 256 && kElementsPerThread == 8) ||
           (CtaM == 64 && kThreads == 512 && kElementsPerThread == 4)),
      "vector callback store requires the proven BM32 or BM64 thread map");
  static_assert(
      !VectorizeCallbackSharedStore ||
          (CtaN % 8 == 0 && kPad % 8 == 0 &&
           ((CtaN + kPad) * sizeof(Element)) % 16 == 0 &&
           alignof(SharedStorage) >= 16),
      "vector callback shared destinations must remain 16-byte aligned");

  CUTLASS_HOST_DEVICE VisitorPairedTransposeAuxStore() {}

  CUTLASS_HOST_DEVICE
  VisitorPairedTransposeAuxStore(
      Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params),
        shared_tile(const_cast<Element*>(&shared_storage.tile[0][0])) {}

  Params const* params_ptr = nullptr;
  Element* shared_tile = nullptr;

  template <class RTensor, class CTensor, class ProblemShape>
  struct Callbacks
      : cutlass::epilogue::threadblock::detail::EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(RTensor&& tC_r_output, CTensor&& tC_c_output,
              ProblemShape problem_shape, Params const* params_ptr,
              Element* shared_tile,
              cutlass::gemm::GemmCoord threadblock_tile_offset,
              int thread_idx)
        : tC_r_output(cute::forward<RTensor>(tC_r_output)),
          tC_c_output(cute::forward<CTensor>(tC_c_output)),
          problem_shape(problem_shape),
          params_ptr(params_ptr),
          shared_tile(shared_tile),
          threadblock_tile_offset(threadblock_tile_offset),
          thread_idx(thread_idx) {}

    RTensor tC_r_output;
    CTensor tC_c_output;
    ProblemShape problem_shape;
    Params const* params_ptr;
    Element* shared_tile;
    cutlass::gemm::GemmCoord threadblock_tile_offset;
    int thread_idx;

    CUTLASS_DEVICE void begin_step(int) { cute::clear(tC_r_output); }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto visit(
        int, int, int, int fragment_idx,
        cutlass::Array<ElementAccumulator, FragmentSize> const&,
        cutlass::Array<ElementInput, FragmentSize> const& fragment_input) {
      using Convert = cutlass::NumericArrayConverter<
          Element, ElementInput, FragmentSize, RoundStyle>;
      Convert convert{};
      auto register_frag = cute::recast<
          cutlass::Array<Element, FragmentSize>>(cute::coalesce(tC_r_output));
      register_frag(fragment_idx) = convert(fragment_input);
      return fragment_input;
    }

    CUTLASS_DEVICE void end_step(int step_idx) {
      auto src = cute::filter(tC_r_output);
      auto coord = cute::filter(
          tC_c_output(cute::_, cute::_, cute::_, step_idx));
      int tile_m = threadblock_tile_offset.m() * CtaM;
      int tile_n = threadblock_tile_offset.n() * CtaN;

      bool vector_stored = false;
      if constexpr (VectorizeCallbackSharedStore) {
        static_assert(
            decltype(cute::size(src))::value == 8,
            "vector callback store requires one contiguous eight-BF16 src");

        // OutputTileThreadLayout maps src[0:8] to one logical output row and
        // eight consecutive logical columns.  Checking both endpoints keeps a
        // complete scalar fallback for partial problem/tile tails instead of
        // mixing a predicated vector store with scalar cleanup.
        auto first_coord = coord(0);
        auto last_coord = coord(7);
        bool first_guard = cute::elem_less(first_coord, problem_shape);
        bool last_guard = cute::elem_less(last_coord, problem_shape);
        int first_m = int(cute::get<0>(first_coord));
        int first_n = int(cute::get<1>(first_coord));
        int last_m = int(cute::get<0>(last_coord));
        int last_n = int(cute::get<1>(last_coord));
        int tile_local_m = first_m - tile_m;
        int warp_row_group = tile_local_m / WarpM;
        int local_row = tile_local_m - warp_row_group * WarpM -
                        step_idx * kRowsPerStep;
        int local_col = first_n - tile_n;
        bool vector_guard =
            first_guard && last_guard && first_m == last_m &&
            last_n == first_n + 7 && warp_row_group >= 0 &&
            warp_row_group < kWarpRowGroups && local_row >= 0 &&
            local_row < kRowsPerStep && local_col >= 0 &&
            local_col + 8 <= CtaN;
        if (vector_guard) {
          cutlass::AlignedArray<Element, 8, 16> vector_fragment;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < 8; ++i) {
            vector_fragment[i] = src(i);
          }
          Element* vector_destination =
              shared_tile +
              (warp_row_group * kRowsPerStep + local_row) *
                  (CtaN + kPad) +
              local_col;
          cutlass::arch::shared_store<16>(
              cutlass::arch::cutlass_get_smem_pointer(vector_destination),
              &vector_fragment);
          vector_stored = true;
        }
      }

      if (!vector_stored) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(src); ++i) {
          auto logical_coord = coord(i);
          bool guard = cute::elem_less(logical_coord, problem_shape);
          int logical_m = int(cute::get<0>(logical_coord));
          int logical_n = int(cute::get<1>(logical_coord));
          int tile_local_m = logical_m - tile_m;
          int warp_row_group = tile_local_m / WarpM;
          int local_row = tile_local_m - warp_row_group * WarpM -
                          step_idx * kRowsPerStep;
          int local_col = logical_n - tile_n;
          if (guard && warp_row_group >= 0 &&
              warp_row_group < kWarpRowGroups && local_row >= 0 &&
              local_row < kRowsPerStep && local_col >= 0 &&
              local_col < CtaN) {
            shared_tile[(warp_row_group * kRowsPerStep + local_row) *
                            (CtaN + kPad) +
                        local_col] = src(i);
          }
        }
      }

      if constexpr (VectorizeCallbackSharedStore && CtaM == 64) {
        // The output thread map linearizes its eight token/warp-row groups
        // before the two warp-M groups: physical warps [0,8) own M group 0
        // and [8,16) own M group 1.  The two M groups own disjoint shared
        // rows and output-channel bands, so each can transpose independently
        // with a 256-thread named barrier.
        int physical_warp = thread_idx / 32;
        int lane = thread_idx % 32;
        constexpr int kWarpsPerMGroup =
            kThreads / (32 * kWarpRowGroups);
        static_assert(kWarpsPerMGroup * kWarpRowGroups * 32 == kThreads,
                      "BM64 epilogue warps must partition evenly by M group");
        static_assert(
            kWarpsPerMGroup ==
                ThreadMap::Base::Detail::kWarpsRemainingForRows,
            "BM64 partition assumes CUTLASS linearizes warp rows before groups");
        int warp_row_group = physical_warp / kWarpsPerMGroup;
        int warp_in_group = physical_warp % kWarpsPerMGroup;
        int group_thread = warp_in_group * 32 + lane;
        cutlass::arch::NamedBarrier::sync(
            (kThreads / kWarpRowGroups), 1 + warp_row_group);

        constexpr int kPartitionedElements = 8;
        using PartitionedStore =
            cutlass::AlignedArray<Element, kPartitionedElements, 16>;
        static_assert(sizeof(PartitionedStore) == 16,
                      "BM64 epilogue store must remain 128 bits");
        PartitionedStore output_fragment;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPartitionedElements; ++i) {
          output_fragment[i] =
              shared_tile[(warp_row_group * kRowsPerStep + i) *
                              (CtaN + kPad) +
                          group_thread];
        }

        int local_token = group_thread;
        int pair_tile = threadblock_tile_offset.n();
        bool dense_branch = false;
        if constexpr (kDenseBranchTokens == CtaN) {
          dense_branch = true;
        } else if constexpr (kDenseBranchTokens > 0) {
          dense_branch = local_token < kDenseBranchTokens;
        }
        int branch_capacity = dense_branch ? kDenseBranchTokens
                                           : kSparseBranchTokens;
        int branch_token = pair_tile * branch_capacity +
                           (dense_branch ? local_token
                                         : local_token - kDenseBranchTokens);
        int branch_rows = dense_branch ? params_ptr->dense_rows
                                       : params_ptr->sparse_rows;
        bool valid_branch_token = branch_token < branch_rows;
        int64_t output_token = branch_token;
        Element* branch_output = dense_branch ? params_ptr->ptr_dense
                                              : params_ptr->ptr_sparse;
        if (params_ptr->ptr_output) {
          int64_t const* branch_indices = dense_branch
              ? params_ptr->dense_indices
              : params_ptr->sparse_indices;
          if (valid_branch_token) {
            output_token = branch_indices[branch_token];
          }
          branch_output = params_ptr->ptr_output;
        }
        int output_channels = int(cute::get<0>(problem_shape));
        int output_channel = tile_m + warp_row_group * WarpM +
                             step_idx * kRowsPerStep;
        bool full_guard = valid_branch_token &&
            output_channel + kPartitionedElements <= output_channels;
        cutlass::arch::global_store<PartitionedStore, sizeof(PartitionedStore)>(
            output_fragment,
            static_cast<void*>(branch_output +
                               output_token * output_channels +
                               output_channel),
            full_guard);
        if (valid_branch_token && !full_guard) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kPartitionedElements; ++i) {
            if (output_channel + i < output_channels) {
              branch_output[output_token * output_channels +
                            output_channel + i] = output_fragment[i];
            }
          }
        }
        if constexpr (PartitionedEpilogueReleaseBarrier) {
          cutlass::arch::NamedBarrier::sync(
              (kThreads / kWarpRowGroups), 1 + warp_row_group);
        }
      } else {
        __syncthreads();

        constexpr int kStoreBits =
            kElementsPerThread * cutlass::sizeof_bits<Element>::value;
        using StoreType = cute::uint_bit_t<kStoreBits>;
        cutlass::Array<Element, kElementsPerThread> output_fragment;
        int linear = thread_idx * kElementsPerThread;
        int local_token = linear / kRowsPerStep;
        int output_in_step = linear % kRowsPerStep;
        int pair_tile = threadblock_tile_offset.n();
        bool dense_branch = false;
        if constexpr (kDenseBranchTokens == CtaN) {
          dense_branch = true;
        } else if constexpr (kDenseBranchTokens > 0) {
          dense_branch = local_token < kDenseBranchTokens;
        }
        int branch_capacity = dense_branch ? kDenseBranchTokens
                                           : kSparseBranchTokens;
        int branch_token = pair_tile * branch_capacity +
                           (dense_branch ? local_token
                                         : local_token - kDenseBranchTokens);
        int branch_rows = dense_branch ? params_ptr->dense_rows
                                       : params_ptr->sparse_rows;
        bool valid_branch_token = branch_token < branch_rows;
        int64_t output_token = branch_token;
        Element* branch_output = dense_branch ? params_ptr->ptr_dense
                                              : params_ptr->ptr_sparse;
        if (params_ptr->ptr_output) {
          int64_t const* branch_indices = dense_branch
              ? params_ptr->dense_indices
              : params_ptr->sparse_indices;
          if (valid_branch_token) {
            output_token = branch_indices[branch_token];
          }
          branch_output = params_ptr->ptr_output;
        }
        int output_channels = int(cute::get<0>(problem_shape));

        CUTLASS_PRAGMA_UNROLL
        for (int warp_row_group = 0; warp_row_group < kWarpRowGroups;
             ++warp_row_group) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kElementsPerThread; ++i) {
            output_fragment[i] =
                shared_tile[(warp_row_group * kRowsPerStep + output_in_step +
                             i) *
                                (CtaN + kPad) +
                            local_token];
          }

          int output_channel = tile_m + warp_row_group * WarpM +
                               step_idx * kRowsPerStep + output_in_step;
          bool full_guard =
              valid_branch_token &&
              output_channel + kElementsPerThread <= output_channels;
          StoreType packed = reinterpret_cast<StoreType const&>(output_fragment);
          cutlass::arch::global_store<StoreType, sizeof(StoreType)>(
              packed,
              static_cast<void*>(branch_output +
                                 output_token * output_channels +
                                 output_channel),
              full_guard);

          if (valid_branch_token && !full_guard) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kElementsPerThread; ++i) {
              if (output_channel + i < output_channels) {
                branch_output[output_token * output_channels +
                              output_channel + i] = output_fragment[i];
              }
            }
          }
        }
        __syncthreads();
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(
      cutlass::gemm::GemmCoord threadblock_tile_offset, int thread_idx,
      ProblemShape problem_shape) {
    using namespace cute;
    // The tensor is a coordinate scaffold only; all physical writes are
    // redirected to the two branch pointers in end_step().
    Tensor logical_output = make_tensor(
        make_gmem_ptr(params_ptr->ptr_output
                          ? params_ptr->ptr_output
                          : params_ptr->ptr_dense),
        problem_shape,
        make_stride(int64_t(get<1>(problem_shape)), _1{}, int64_t(0)));
    Tensor partitioned = group_modes<3, 6>(ThreadMap::partition(
        logical_output, thread_idx, threadblock_tile_offset));
    Tensor registers = make_tensor_like(take<0, 3>(partitioned));

    Tensor identity = make_identity_tensor(logical_output.shape());
    Tensor coordinates = group_modes<3, 6>(
        ThreadMap::partition(identity, thread_idx, threadblock_tile_offset));

    return Callbacks<decltype(registers), decltype(coordinates), ProblemShape>(
        cute::move(registers), cute::move(coordinates), problem_shape,
        params_ptr, shared_tile, threadblock_tile_offset, thread_idx);
  }
};

}  // namespace speculators::speclink
