// Python-callable CUTLASS SparseGemm backend for fp16 2:4 linear layers.
//
// Public operation: Y[M, N] = X[M, K] @ W24[K, N].
// CUTLASS sees W24.T as sparse A[N, K] and X.T as dense B[K, M], producing
// an internal row-major C[N, M].  The fastest serving path can return that
// storage as a non-contiguous [M, N] view.  Contiguous output uses one tiled
// transpose when the caller needs standard row-major layout.

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/std/functional>

#include <cstdlib>
#include <cstdint>
#include <cstring>

#include "cutlass/arch/mma.h"
#include "cutlass/arch/mma_sparse_sm89.h"
#include "cutlass/device_kernel.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_sparse.h"
#include "cutlass/gemm/device/gemm_sparse_with_visitor.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/gemm/threadblock/mma_sparse_multistage.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

using Element = cutlass::half_t;

// CUTLASS ships the SM80 sparse visitor kernel but omits the equivalent SM89
// specialization. The mainloop and epilogue construction are otherwise the
// same as DefaultSparseGemm's SM89 specialization.
namespace cutlass::gemm::kernel {
template <typename ElementA, typename LayoutA, int kAlignmentA,
          typename ElementB, typename LayoutB, int kAlignmentB,
          typename ElementC, typename LayoutC, typename ElementAccumulator,
          typename ThreadblockShape, typename WarpShape,
          typename InstructionShape, typename FusionCallbacks,
          typename ThreadblockSwizzle, int Stages, typename Operator,
          int EpilogueStages>
struct DefaultSparseGemmWithVisitor<
    ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC,
    LayoutC, ElementAccumulator, arch::OpClassTensorOp, arch::Sm89,
    ThreadblockShape, WarpShape, InstructionShape, FusionCallbacks,
    ThreadblockSwizzle, Stages, Operator, EpilogueStages> {
  using Mma = typename cutlass::gemm::threadblock::DefaultSparseMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, arch::Sm89,
      ThreadblockShape, WarpShape, InstructionShape, Stages,
      Operator>::ThreadblockMma;

  static constexpr int kAlignmentC = 128 / sizeof_bits<ElementC>::value;
  using ElementEpilogue = ElementAccumulator;
  static constexpr int kPartitionsK =
      ThreadblockShape::kK / WarpShape::kK;
  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      ElementC, kAlignmentC, ElementAccumulator, ElementEpilogue>;
  using BaseEpilogue =
      typename epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK,
          EpilogueOutputOp, EpilogueOutputOp::kCount>::Epilogue;
  using Epilogue = epilogue::threadblock::EpilogueWithVisitorCallbacks<
      BaseEpilogue, FusionCallbacks, EpilogueStages>;
  using GemmKernel =
      kernel::SparseGemmWithEpilogueVisitor<Mma, Epilogue, ThreadblockSwizzle>;
};
} // namespace cutlass::gemm::kernel

using DeviceThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
using DeviceThreadblockShape64x32x64 = cutlass::gemm::GemmShape<64, 32, 64>;
using DeviceThreadblockShape64x128x64 = cutlass::gemm::GemmShape<64, 128, 64>;
using DeviceThreadblockShape128x32x64 = cutlass::gemm::GemmShape<128, 32, 64>;
using DeviceThreadblockShape128x64x64 = cutlass::gemm::GemmShape<128, 64, 64>;
using DeviceThreadblockShape128x128x64 = cutlass::gemm::GemmShape<128, 128, 64>;
using DeviceThreadblockShape256x32x64 = cutlass::gemm::GemmShape<256, 32, 64>;
using DeviceThreadblockShape256x64x64 = cutlass::gemm::GemmShape<256, 64, 64>;
using DeviceThreadblockShape256x128x64 =
    cutlass::gemm::GemmShape<256, 128, 64>;
using DeviceWarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
using DeviceWarpShape16x32x64 = cutlass::gemm::GemmShape<16, 32, 64>;
using DeviceWarpShape32x64x64 = cutlass::gemm::GemmShape<32, 64, 64>;
using DeviceWarpShape64x32x64 = cutlass::gemm::GemmShape<64, 32, 64>;
using DeviceWarpShape64x64x64 = cutlass::gemm::GemmShape<64, 64, 64>;
using DeviceInstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
using DeviceArchTag = cutlass::arch::Sm89;
using DeviceLayoutC = cutlass::layout::RowMajor;
using DeviceEpilogueOpVec8 = cutlass::epilogue::thread::LinearCombination<
    Element, 8, float, float>;
using DeviceEpilogueOpVec8F16Accum =
    cutlass::epilogue::thread::LinearCombination<Element, 8, Element, Element>;

template <typename ThreadMap, typename ElementOutput>
struct Sparse24TransposeStore {
  struct Arguments {
    ElementOutput *output = nullptr;
    int output_columns = 0;
    int logical_rows = 0;
  };

  using Params = Arguments;
  struct SharedStorage {};

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const &, Arguments const &args, void *) {
    return args;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const &, Arguments const &) {
    return 0;
  }

  CUTLASS_HOST_DEVICE
  Sparse24TransposeStore() = default;

  CUTLASS_HOST_DEVICE
  Sparse24TransposeStore(Params const &params,
                         SharedStorage const &)
      : params_ptr(&params) {}

  static constexpr int kVectorBits =
      ThreadMap::kElementsPerAccess * cutlass::sizeof_bits<ElementOutput>::value;
  using Vector = cute::uint_bit_t<cute::min(128, kVectorBits)>;
  static constexpr int kVectorElements = sizeof(Vector) / sizeof(ElementOutput);

  template <class RegisterTensor, class CoordTensor, class ProblemShape>
  struct Callbacks
      : cutlass::epilogue::threadblock::detail::EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(RegisterTensor &&register_output, CoordTensor &&coordinates,
              ProblemShape problem_shape, Params const *params_ptr)
        : register_output(cute::forward<RegisterTensor>(register_output)),
          coordinates(cute::forward<CoordTensor>(coordinates)),
          problem_shape(problem_shape), params_ptr(params_ptr) {}

    RegisterTensor register_output;
    CoordTensor coordinates;
    ProblemShape problem_shape;
    Params const *params_ptr;

    CUTLASS_DEVICE void begin_step(int) { cute::clear(register_output); }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto visit(
        int, int, int, int fragment_index,
        cutlass::Array<ElementAccumulator, FragmentSize> const &,
        cutlass::Array<ElementInput, FragmentSize> const &fragment) {
      using Converter = cutlass::NumericArrayConverter<
          ElementOutput, ElementInput, FragmentSize,
          cutlass::FloatRoundStyle::round_to_nearest>;
      auto register_fragments =
          cute::recast<cutlass::Array<ElementOutput, FragmentSize>>(
              cute::coalesce(register_output));
      register_fragments(fragment_index) = Converter{}(fragment);
      return fragment;
    }

    CUTLASS_DEVICE void end_step(int step_index) {
      auto source_vectors = cute::filter(register_output);
      auto vector_coordinates =
          cute::filter(coordinates(cute::_, cute::_, cute::_, step_index));
      CUTLASS_PRAGMA_UNROLL
      for (int vector_index = 0; vector_index < cute::size(source_vectors);
           ++vector_index) {
        auto coordinate = vector_coordinates(vector_index);
        int output_column = int(cute::get<0>(coordinate));
        int output_row = int(cute::get<1>(coordinate));
        Vector packed = source_vectors(vector_index);
        ElementOutput const *elements =
            reinterpret_cast<ElementOutput const *>(&packed);
        CUTLASS_PRAGMA_UNROLL
        for (int lane = 0; lane < kVectorElements; ++lane) {
          int row = output_row + lane;
          if (output_column < params_ptr->output_columns &&
              row < params_ptr->logical_rows) {
            params_ptr->output[
                static_cast<int64_t>(row) * params_ptr->output_columns +
                output_column] = elements[lane];
          }
        }
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(
      cutlass::gemm::GemmCoord threadblock_tile_offset, int thread_index,
      ProblemShape problem_shape) {
    auto fake_stride = cute::make_stride(
        int64_t(params_ptr->logical_rows), cute::_1{},
        int64_t(params_ptr->logical_rows) * params_ptr->output_columns);
    auto output = cute::make_tensor(cute::make_gmem_ptr(params_ptr->output),
                                    problem_shape, fake_stride);
    auto partitioned = cute::group_modes<3, 6>(
        ThreadMap::partition(output, thread_index, threadblock_tile_offset));
    auto vector_partition = cute::recast<Vector>(partitioned);
    auto register_output = cute::make_tensor_like(
        cute::take<0, 3>(vector_partition));

    auto identity = cute::make_identity_tensor(output.shape());
    auto coordinates = cute::outer_partition(
        cute::group_modes<3, 6>(ThreadMap::partition(
            identity, thread_index, threadblock_tile_offset)),
        cute::Shape<cute::Int<kVectorElements>>{}, cute::_0{});
    return Callbacks<decltype(register_output), decltype(coordinates),
                     ProblemShape>(cute::move(register_output),
                                   cute::move(coordinates), problem_shape,
                                   params_ptr);
  }

  Params const *params_ptr;
};

template <typename ThreadMap, typename ElementOutput, int TileColumns,
          int TileRows, int Threads, bool IndexedRows = false,
          bool RoutedRows = false, bool AddToOutput = false,
          bool AddRoutedResidual = false,
          bool IndexedCorrection = false>
struct Sparse24VectorTransposeStore {
  struct Arguments {
    ElementOutput *output = nullptr;
    ElementOutput *dense_base = nullptr;
    ElementOutput const *routed_residual = nullptr;
    int output_columns = 0;
    int logical_rows = 0;
    int output_rows = 0;
    int const *row_indices = nullptr;
    int const *dense_slot_by_row = nullptr;
    int dense_rows = 0;
  };

  using Params = Arguments;
  struct SharedStorage {
    alignas(16) ElementOutput tile[TileRows * TileColumns];
    alignas(16) int row_mapping[(IndexedRows || RoutedRows) ? TileRows : 1];
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const &, Arguments const &args, void *) {
    return args;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const &, Arguments const &) {
    return 0;
  }

  CUTLASS_HOST_DEVICE
  Sparse24VectorTransposeStore() {}

  CUTLASS_HOST_DEVICE
  Sparse24VectorTransposeStore(Params const &params,
                               SharedStorage const &shared)
      : params_ptr(&params),
        shared_tile(const_cast<ElementOutput *>(shared.tile)),
        shared_dense_slots(const_cast<int *>(shared.row_mapping)) {}

  static_assert(!(IndexedRows && RoutedRows),
                "indexed and routed transpose stores are mutually exclusive");
  static_assert(!AddToOutput || IndexedRows,
                "in-place output accumulation requires indexed rows");
  static_assert(!AddRoutedResidual || RoutedRows,
                "routed residual accumulation requires routed rows");
  static_assert(!(AddToOutput && AddRoutedResidual),
                "indexed and routed accumulation are mutually exclusive");
  static_assert(!IndexedCorrection || IndexedRows,
                "compact correction requires indexed output rows");
  static_assert(!IndexedCorrection ||
                    (!RoutedRows && !AddToOutput && !AddRoutedResidual),
                "compact correction is a separate indexed store mode");

  static constexpr int kVectorBits =
      ThreadMap::kElementsPerAccess * cutlass::sizeof_bits<ElementOutput>::value;
  using Vector = cute::uint_bit_t<cute::min(128, kVectorBits)>;
  static constexpr int kVectorElements = sizeof(Vector) / sizeof(ElementOutput);
  static constexpr int kVectorsPerRow = TileColumns / kVectorElements;
  static constexpr int kTotalVectors = TileRows * kVectorsPerRow;
  static constexpr int kThreads = Threads;
  static_assert(kVectorElements == 8,
                "vector transpose store expects half8 fragments");
  static_assert(kTotalVectors % kThreads == 0,
                "vector transpose tile must divide evenly over CTA threads");

  template <class RegisterTensor, class CoordTensor, class ProblemShape>
  struct Callbacks
      : cutlass::epilogue::threadblock::detail::EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(RegisterTensor &&register_output, CoordTensor &&coordinates,
              ProblemShape problem_shape, Params const *params_ptr,
              ElementOutput *shared_tile, int *shared_dense_slots,
              int thread_index,
              int tile_output_base, int tile_row_base)
        : register_output(cute::forward<RegisterTensor>(register_output)),
          coordinates(cute::forward<CoordTensor>(coordinates)),
          problem_shape(problem_shape), params_ptr(params_ptr),
          shared_tile(shared_tile), shared_dense_slots(shared_dense_slots),
          thread_index(thread_index),
          tile_output_base(tile_output_base), tile_row_base(tile_row_base) {}

    RegisterTensor register_output;
    CoordTensor coordinates;
    ProblemShape problem_shape;
    Params const *params_ptr;
    ElementOutput *shared_tile;
    int *shared_dense_slots;
    int thread_index;
    int tile_output_base;
    int tile_row_base;

    CUTLASS_DEVICE void begin_step(int) { cute::clear(register_output); }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto visit(
        int, int, int, int fragment_index,
        cutlass::Array<ElementAccumulator, FragmentSize> const &,
        cutlass::Array<ElementInput, FragmentSize> const &fragment) {
      using Converter = cutlass::NumericArrayConverter<
          ElementOutput, ElementInput, FragmentSize,
          cutlass::FloatRoundStyle::round_to_nearest>;
      auto register_fragments =
          cute::recast<cutlass::Array<ElementOutput, FragmentSize>>(
              cute::coalesce(register_output));
      register_fragments(fragment_index) = Converter{}(fragment);
      return fragment;
    }

    CUTLASS_DEVICE void end_step(int step_index) {
      auto source_vectors = cute::filter(register_output);
      auto vector_coordinates =
          cute::filter(coordinates(cute::_, cute::_, cute::_, step_index));
      CUTLASS_PRAGMA_UNROLL
      for (int vector_index = 0; vector_index < cute::size(source_vectors);
           ++vector_index) {
        auto coordinate = vector_coordinates(vector_index);
        int output_column = int(cute::get<0>(coordinate));
        int output_row = int(cute::get<1>(coordinate));
        int local_column = output_column - tile_output_base;
        int local_row = output_row - tile_row_base;
        Vector packed = source_vectors(vector_index);
        ElementOutput const *elements =
            reinterpret_cast<ElementOutput const *>(&packed);
        CUTLASS_PRAGMA_UNROLL
        for (int lane = 0; lane < kVectorElements; ++lane) {
          int row = local_row + lane;
          if (local_column >= 0 && local_column < TileColumns && row >= 0 &&
              row < TileRows) {
            shared_tile[row * TileColumns + local_column] =
                elements[lane];
          }
        }
      }
    }

    CUTLASS_DEVICE void end_epilogue() {
      if constexpr (RoutedRows) {
        if (thread_index < TileRows) {
          int output_row = tile_row_base + thread_index;
          shared_dense_slots[thread_index] =
              output_row < params_ptr->logical_rows
                  ? params_ptr->dense_slot_by_row[output_row]
                  : -2;
        }
      } else if constexpr (IndexedRows) {
        if (thread_index < TileRows) {
          int output_row = tile_row_base + thread_index;
          shared_dense_slots[thread_index] =
              output_row < params_ptr->logical_rows
                  ? params_ptr->row_indices[output_row]
                  : -2;
        }
      }
      __syncthreads();
      CUTLASS_PRAGMA_UNROLL
      for (int vector_index = thread_index; vector_index < kTotalVectors;
           vector_index += kThreads) {
        int local_output_row = vector_index / kVectorsPerRow;
        int vector_column = vector_index % kVectorsPerRow;
        int output_row = tile_row_base + local_output_row;
        int output_column =
            tile_output_base + vector_column * kVectorElements;
        if (output_row < params_ptr->logical_rows) {
          int destination_row = output_row;
          int dense_slot = -1;
          if constexpr (IndexedRows) {
            destination_row = shared_dense_slots[local_output_row];
            if (destination_row < 0 ||
                destination_row >= params_ptr->output_rows) {
              continue;
            }
          }
          ElementOutput *destination = nullptr;
          if constexpr (IndexedCorrection) {
            if (params_ptr->routed_residual == nullptr) {
              if (params_ptr->dense_base == nullptr) {
                continue;
              }
              destination =
                  params_ptr->dense_base +
                  static_cast<int64_t>(output_row) *
                      params_ptr->output_columns +
                  output_column;
            } else {
              destination =
                  params_ptr->output +
                  static_cast<int64_t>(destination_row) *
                      params_ptr->output_columns +
                  output_column;
            }
          } else if constexpr (RoutedRows) {
            dense_slot = shared_dense_slots[local_output_row];
            if (dense_slot >= 0) {
              if (dense_slot >= params_ptr->dense_rows) {
                continue;
              }
              if constexpr (AddRoutedResidual) {
                destination =
                    params_ptr->output +
                    static_cast<int64_t>(destination_row) *
                        params_ptr->output_columns +
                    output_column;
              } else {
                // A null dense_base means another worker group owns exact
                // dense-row stores. Skip the approximate W24 value so the two
                // epilogues write disjoint output rows without a grid barrier.
                if (params_ptr->dense_base == nullptr) {
                  continue;
                }
                destination =
                    params_ptr->dense_base +
                    static_cast<int64_t>(dense_slot) *
                        params_ptr->output_columns +
                    output_column;
              }
            } else {
              destination =
                  params_ptr->output +
                  static_cast<int64_t>(destination_row) *
                      params_ptr->output_columns +
                  output_column;
            }
          } else {
            destination =
                params_ptr->output +
                static_cast<int64_t>(destination_row) *
                    params_ptr->output_columns +
                output_column;
          }
          ElementOutput *source =
              shared_tile + local_output_row * TileColumns +
              vector_column * kVectorElements;
          if (output_column + kVectorElements <= params_ptr->output_columns) {
            if constexpr (IndexedCorrection) {
              if (params_ptr->routed_residual == nullptr) {
                *reinterpret_cast<Vector *>(destination) =
                    *reinterpret_cast<Vector *>(source);
              } else {
                constexpr int kHalf2PerVector = kVectorElements / 2;
                auto *destination_half2 =
                    reinterpret_cast<__half2 *>(destination);
                auto const *source_half2 =
                    reinterpret_cast<__half2 const *>(source);
                auto const *base_half2 =
                    reinterpret_cast<__half2 const *>(
                        params_ptr->routed_residual +
                        static_cast<int64_t>(output_row) *
                            params_ptr->output_columns +
                        output_column);
                CUTLASS_PRAGMA_UNROLL
                for (int pair = 0; pair < kHalf2PerVector; ++pair) {
                  destination_half2[pair] =
                      __hadd2(base_half2[pair], source_half2[pair]);
                }
              }
            } else if constexpr (AddRoutedResidual) {
              if (dense_slot >= 0) {
                constexpr int kHalf2PerVector = kVectorElements / 2;
                auto *destination_half2 =
                    reinterpret_cast<__half2 *>(destination);
                auto const *source_half2 =
                    reinterpret_cast<__half2 const *>(source);
                auto const *residual_half2 =
                    reinterpret_cast<__half2 const *>(
                        params_ptr->routed_residual +
                        static_cast<int64_t>(dense_slot) *
                            params_ptr->output_columns +
                        output_column);
                CUTLASS_PRAGMA_UNROLL
                for (int pair = 0; pair < kHalf2PerVector; ++pair) {
                  destination_half2[pair] =
                      __hadd2(source_half2[pair], residual_half2[pair]);
                }
              } else {
                *reinterpret_cast<Vector *>(destination) =
                    *reinterpret_cast<Vector *>(source);
              }
            } else if constexpr (AddToOutput) {
              constexpr int kHalf2PerVector = kVectorElements / 2;
              auto *destination_half2 =
                  reinterpret_cast<__half2 *>(destination);
              auto const *source_half2 =
                  reinterpret_cast<__half2 const *>(source);
              CUTLASS_PRAGMA_UNROLL
              for (int pair = 0; pair < kHalf2PerVector; ++pair) {
                destination_half2[pair] =
                    __hadd2(destination_half2[pair], source_half2[pair]);
              }
            } else {
              *reinterpret_cast<Vector *>(destination) =
                  *reinterpret_cast<Vector *>(source);
            }
          } else {
            CUTLASS_PRAGMA_UNROLL
            for (int column = 0; column < kVectorElements; ++column) {
              if (output_column + column < params_ptr->output_columns) {
                if constexpr (IndexedCorrection) {
                  if (params_ptr->routed_residual == nullptr) {
                    destination[column] = source[column];
                  } else {
                    destination[column] = ElementOutput(
                        static_cast<float>(source[column]) +
                        static_cast<float>(params_ptr->routed_residual[
                            static_cast<int64_t>(output_row) *
                                params_ptr->output_columns +
                            output_column + column]));
                  }
                } else if constexpr (AddRoutedResidual) {
                  if (dense_slot >= 0) {
                    destination[column] = ElementOutput(
                        static_cast<float>(source[column]) +
                        static_cast<float>(params_ptr->routed_residual[
                            static_cast<int64_t>(dense_slot) *
                                params_ptr->output_columns +
                            output_column + column]));
                  } else {
                    destination[column] = source[column];
                  }
                } else if constexpr (AddToOutput) {
                  reinterpret_cast<__half *>(destination)[column] = __hadd(
                      reinterpret_cast<__half *>(destination)[column],
                      reinterpret_cast<__half const *>(source)[column]);
                } else {
                  destination[column] = source[column];
                }
              }
            }
          }
        }
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(
      cutlass::gemm::GemmCoord threadblock_tile_offset, int thread_index,
      ProblemShape problem_shape) {
    auto fake_stride = cute::make_stride(
        int64_t(params_ptr->logical_rows), cute::_1{},
        int64_t(params_ptr->logical_rows) * params_ptr->output_columns);
    auto output = cute::make_tensor(cute::make_gmem_ptr(params_ptr->output),
                                    problem_shape, fake_stride);
    auto partitioned = cute::group_modes<3, 6>(
        ThreadMap::partition(output, thread_index, threadblock_tile_offset));
    auto vector_partition = cute::recast<Vector>(partitioned);
    auto register_output = cute::make_tensor_like(
        cute::take<0, 3>(vector_partition));

    auto identity = cute::make_identity_tensor(output.shape());
    auto coordinates = cute::outer_partition(
        cute::group_modes<3, 6>(ThreadMap::partition(
            identity, thread_index, threadblock_tile_offset)),
        cute::Shape<cute::Int<kVectorElements>>{}, cute::_0{});
    return Callbacks<decltype(register_output), decltype(coordinates),
                     ProblemShape>(
        cute::move(register_output), cute::move(coordinates), problem_shape,
        params_ptr, shared_tile, shared_dense_slots, thread_index,
        threadblock_tile_offset.m() * TileColumns,
        threadblock_tile_offset.n() * TileRows);
  }

  Params const *params_ptr;
  ElementOutput *shared_tile;
  int *shared_dense_slots;
};

template <typename ThreadMap, typename ElementOutput, int TileColumns,
          int TileRows, int Threads, bool OutputTransposed,
          bool PairAdd = false, bool IndexedRows = false,
          bool RoutedRows = false, bool ResidualCorrection = false,
          bool WriteRoutedApprox = false,
          bool ResidualDeltaOnly = false, bool FastSilu = false>
struct Sparse24SwiGLUStore {
  struct Arguments {
    ElementOutput *output = nullptr;
    ElementOutput *dense_base = nullptr;
    const ElementOutput *correction_base = nullptr;
    ElementOutput *compact_output = nullptr;
    int hidden_size = 0;
    int logical_rows = 0;
    int output_rows = 0;
    int const *row_indices = nullptr;
    int const *dense_slot_by_row = nullptr;
    int dense_rows = 0;
  };

  using Params = Arguments;
  struct SharedStorage {
    alignas(16) ElementOutput tile[TileRows * TileColumns];
    alignas(16) int row_mapping[(IndexedRows || RoutedRows) ? TileRows : 1];
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const &, Arguments const &args, void *) {
    return args;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const &, Arguments const &) {
    return 0;
  }

  CUTLASS_HOST_DEVICE
  Sparse24SwiGLUStore() {}

  CUTLASS_HOST_DEVICE
  Sparse24SwiGLUStore(Params const &params, SharedStorage const &shared)
      : params_ptr(&params),
        shared_tile(const_cast<ElementOutput *>(shared.tile)),
        shared_dense_slots(const_cast<int *>(shared.row_mapping)) {}

  static constexpr int kVectorBits =
      ThreadMap::kElementsPerAccess * cutlass::sizeof_bits<ElementOutput>::value;
  using Vector = cute::uint_bit_t<cute::min(128, kVectorBits)>;
  static constexpr int kVectorElements = sizeof(Vector) / sizeof(ElementOutput);
  static constexpr int kOutputColumns = TileColumns / 2;
  static constexpr int kVectorsPerRow = kOutputColumns / kVectorElements;
  static constexpr int kTotalVectors = TileRows * kVectorsPerRow;
  static constexpr int kThreads = Threads;
  static_assert(TileColumns % (2 * kVectorElements) == 0,
                "SwiGLU tile must contain aligned gate/up halves");
  static_assert(TileColumns == 128 || TileColumns == 256,
                "SwiGLU weights require 64- or 128-channel gate/up pairs");
  static_assert(kVectorElements == 8,
                "SwiGLU store expects half8 epilogue fragments");
  static_assert(kTotalVectors % kThreads == 0,
                "SwiGLU tile must divide evenly over CTA threads");
  static_assert(!RoutedRows || (!PairAdd && !IndexedRows),
                "routed SwiGLU cannot combine pair-add or indexed modes");
  static_assert(!ResidualCorrection ||
                    (!OutputTransposed && !PairAdd && !RoutedRows &&
                     (IndexedRows || ResidualDeltaOnly)),
                "residual correction requires row-major SwiGLU");
  static_assert(!WriteRoutedApprox || RoutedRows,
                "approximate routed output requires routed rows");
  static_assert(!ResidualDeltaOnly || ResidualCorrection,
                "delta-only output requires residual correction");

  template <class RegisterTensor, class CoordTensor, class ProblemShape>
  struct Callbacks
      : cutlass::epilogue::threadblock::detail::EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(RegisterTensor &&register_output, CoordTensor &&coordinates,
              ProblemShape problem_shape, Params const *params_ptr,
              ElementOutput *shared_tile, int *shared_dense_slots,
              int thread_index,
              int tile_output_base, int tile_row_base)
        : register_output(cute::forward<RegisterTensor>(register_output)),
          coordinates(cute::forward<CoordTensor>(coordinates)),
          problem_shape(problem_shape), params_ptr(params_ptr),
          shared_tile(shared_tile), shared_dense_slots(shared_dense_slots),
          thread_index(thread_index),
          tile_output_base(tile_output_base), tile_row_base(tile_row_base) {}

    RegisterTensor register_output;
    CoordTensor coordinates;
    ProblemShape problem_shape;
    Params const *params_ptr;
    ElementOutput *shared_tile;
    int *shared_dense_slots;
    int thread_index;
    int tile_output_base;
    int tile_row_base;

    CUTLASS_DEVICE static float silu(float value) {
      if constexpr (FastSilu) {
        return value / (1.0f + __expf(-value));
      }
      return value / (1.0f + expf(-value));
    }

    CUTLASS_DEVICE void begin_step(int) { cute::clear(register_output); }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto visit(
        int, int, int, int fragment_index,
        cutlass::Array<ElementAccumulator, FragmentSize> const &,
        cutlass::Array<ElementInput, FragmentSize> const &fragment) {
      using Converter = cutlass::NumericArrayConverter<
          ElementOutput, ElementInput, FragmentSize,
          cutlass::FloatRoundStyle::round_to_nearest>;
      auto register_fragments =
          cute::recast<cutlass::Array<ElementOutput, FragmentSize>>(
              cute::coalesce(register_output));
      register_fragments(fragment_index) = Converter{}(fragment);
      return fragment;
    }

    CUTLASS_DEVICE void end_step(int step_index) {
      auto source_vectors = cute::filter(register_output);
      auto vector_coordinates =
          cute::filter(coordinates(cute::_, cute::_, cute::_, step_index));
      CUTLASS_PRAGMA_UNROLL
      for (int vector_index = 0; vector_index < cute::size(source_vectors);
           ++vector_index) {
        auto coordinate = vector_coordinates(vector_index);
        int output_column = int(cute::get<0>(coordinate));
        int output_row = int(cute::get<1>(coordinate));
        int local_column = output_column - tile_output_base;
        int local_row = output_row - tile_row_base;
        Vector packed = source_vectors(vector_index);
        ElementOutput const *elements =
            reinterpret_cast<ElementOutput const *>(&packed);
        CUTLASS_PRAGMA_UNROLL
        for (int lane = 0; lane < kVectorElements; ++lane) {
          int row = local_row + lane;
          if (local_column >= 0 && local_column < TileColumns && row >= 0 &&
              row < TileRows) {
            shared_tile[row * TileColumns + local_column] = elements[lane];
          }
        }
      }
    }

    CUTLASS_DEVICE void end_epilogue() {
      if constexpr (RoutedRows) {
        for (int local_row = thread_index; local_row < TileRows;
             local_row += kThreads) {
          int output_row = tile_row_base + local_row;
          shared_dense_slots[local_row] =
              output_row < params_ptr->logical_rows
                  ? params_ptr->dense_slot_by_row[output_row]
                  : -2;
        }
      } else if constexpr (IndexedRows) {
        for (int local_row = thread_index; local_row < TileRows;
             local_row += kThreads) {
          int output_row = tile_row_base + local_row;
          shared_dense_slots[local_row] =
              output_row < params_ptr->logical_rows
                  ? params_ptr->row_indices[output_row]
                  : -2;
        }
      }
      __syncthreads();
      CUTLASS_PRAGMA_UNROLL
      for (int vector_index = thread_index; vector_index < kTotalVectors;
           vector_index += kThreads) {
        if constexpr (OutputTransposed) {
          constexpr int kRowVectors = TileRows / kVectorElements;
          int local_output_column = vector_index / kRowVectors;
          int row_vector = vector_index % kRowVectors;
          int local_output_row = row_vector * kVectorElements;
          int output_row = tile_row_base + local_output_row;
          int output_column = tile_output_base / 2 + local_output_column;
          if (output_row + kVectorElements > params_ptr->logical_rows ||
              output_column >= params_ptr->hidden_size) {
            continue;
          }
          if constexpr (RoutedRows) {
            Vector packed_output;
            ElementOutput *result =
                reinterpret_cast<ElementOutput *>(&packed_output);
            bool has_dense_row = false;
            CUTLASS_PRAGMA_UNROLL
            for (int lane = 0; lane < kVectorElements; ++lane) {
              ElementOutput *gate =
                  shared_tile + (local_output_row + lane) * TileColumns +
                  local_output_column;
              float gate_value = static_cast<float>(*gate);
              float up_value = static_cast<float>(gate[kOutputColumns]);
              ElementOutput rounded_silu = ElementOutput(silu(gate_value));
              result[lane] = ElementOutput(
                  static_cast<float>(rounded_silu) * up_value);
              int dense_slot = shared_dense_slots[local_output_row + lane];
              has_dense_row = has_dense_row || dense_slot >= 0;
              if (dense_slot >= 0 && dense_slot < params_ptr->dense_rows &&
                  params_ptr->dense_base != nullptr) {
                params_ptr->dense_base[
                    static_cast<int64_t>(output_column) *
                        params_ptr->dense_rows +
                    dense_slot] = *gate;
                params_ptr->dense_base[
                    static_cast<int64_t>(params_ptr->hidden_size +
                                         output_column) *
                            params_ptr->dense_rows +
                        dense_slot] = gate[kOutputColumns];
              }
            }
            ElementOutput *destination =
                params_ptr->output +
                static_cast<int64_t>(output_column) *
                    params_ptr->logical_rows +
                output_row;
            if (has_dense_row && !WriteRoutedApprox) {
              CUTLASS_PRAGMA_UNROLL
              for (int lane = 0; lane < kVectorElements; ++lane) {
                if (shared_dense_slots[local_output_row + lane] < 0) {
                  destination[lane] = result[lane];
                }
              }
            } else {
              *reinterpret_cast<Vector *>(destination) = packed_output;
            }
          } else {
            Vector packed_output;
            ElementOutput *result =
                reinterpret_cast<ElementOutput *>(&packed_output);
            CUTLASS_PRAGMA_UNROLL
            for (int lane = 0; lane < kVectorElements; ++lane) {
              ElementOutput *gate =
                  shared_tile + (local_output_row + lane) * TileColumns +
                  local_output_column;
              float gate_value = static_cast<float>(*gate);
              float up_value = static_cast<float>(gate[kOutputColumns]);
              ElementOutput rounded_silu = ElementOutput(silu(gate_value));
              result[lane] = ElementOutput(
                  static_cast<float>(rounded_silu) * up_value);
            }
            ElementOutput *destination =
                params_ptr->output +
                static_cast<int64_t>(output_column) *
                    params_ptr->logical_rows +
                output_row;
            *reinterpret_cast<Vector *>(destination) = packed_output;
          }
        } else {
          int local_output_row = vector_index / kVectorsPerRow;
          int vector_column = vector_index % kVectorsPerRow;
          int output_row = tile_row_base + local_output_row;
          int output_column =
              tile_output_base / 2 + vector_column * kVectorElements;
          if (output_row >= params_ptr->logical_rows ||
              output_column + kVectorElements > params_ptr->hidden_size) {
            continue;
          }
          if constexpr (ResidualDeltaOnly) {
            if (output_row >= params_ptr->dense_rows) {
              Vector zero{};
              ElementOutput *compact_destination =
                  params_ptr->compact_output +
                  static_cast<int64_t>(output_row) *
                      params_ptr->hidden_size +
                  output_column;
              *reinterpret_cast<Vector *>(compact_destination) = zero;
              continue;
            }
          }
          ElementOutput *gate =
              shared_tile + local_output_row * TileColumns +
              vector_column * kVectorElements;
          ElementOutput *up = gate + kOutputColumns;
          if constexpr (RoutedRows) {
            int dense_slot = shared_dense_slots[local_output_row];
            if (dense_slot >= 0) {
              if (dense_slot < params_ptr->dense_rows &&
                  params_ptr->dense_base != nullptr) {
                ElementOutput *dense_gate =
                    params_ptr->dense_base +
                    static_cast<int64_t>(dense_slot) *
                        params_ptr->hidden_size * 2 +
                    output_column;
                ElementOutput *dense_up =
                    dense_gate + params_ptr->hidden_size;
                *reinterpret_cast<Vector *>(dense_gate) =
                    *reinterpret_cast<Vector *>(gate);
                *reinterpret_cast<Vector *>(dense_up) =
                    *reinterpret_cast<Vector *>(up);
              }
              if constexpr (!WriteRoutedApprox) {
                continue;
              }
            }
          }
          if constexpr (ResidualCorrection) {
            if (params_ptr->correction_base == nullptr &&
                params_ptr->dense_base != nullptr) {
              ElementOutput *base_gate =
                  params_ptr->dense_base +
                  static_cast<int64_t>(output_row) *
                      params_ptr->hidden_size * 2 +
                  output_column;
              ElementOutput *base_up =
                  base_gate + params_ptr->hidden_size;
              *reinterpret_cast<Vector *>(base_gate) =
                  *reinterpret_cast<Vector *>(gate);
              *reinterpret_cast<Vector *>(base_up) =
                  *reinterpret_cast<Vector *>(up);
              continue;
            }
          }
          Vector packed_output;
          ElementOutput *result =
              reinterpret_cast<ElementOutput *>(&packed_output);
          CUTLASS_PRAGMA_UNROLL
          for (int lane = 0; lane < kVectorElements; ++lane) {
            float first;
            float second;
            float base_first = 0.0f;
            float base_second = 0.0f;
            if constexpr (ResidualCorrection) {
              const ElementOutput *base_gate =
                  params_ptr->correction_base +
                  static_cast<int64_t>(output_row) *
                      params_ptr->hidden_size * 2 +
                  output_column;
              const ElementOutput *base_up =
                  base_gate + params_ptr->hidden_size;
              __half gate_sum = __hadd(
                  reinterpret_cast<const __half *>(gate)[lane],
                  reinterpret_cast<const __half *>(base_gate)[lane]);
              __half up_sum = __hadd(
                  reinterpret_cast<const __half *>(up)[lane],
                  reinterpret_cast<const __half *>(base_up)[lane]);
              first = __half2float(gate_sum);
              second = __half2float(up_sum);
              if constexpr (ResidualDeltaOnly) {
                base_first = static_cast<float>(base_gate[lane]);
                base_second = static_cast<float>(base_up[lane]);
              }
            } else {
              first = static_cast<float>(gate[lane]);
              second = static_cast<float>(up[lane]);
            }
            if constexpr (PairAdd) {
              result[lane] = ElementOutput(first + second);
            } else {
              ElementOutput rounded_silu = ElementOutput(silu(first));
              result[lane] = ElementOutput(
                  static_cast<float>(rounded_silu) * second);
              if constexpr (ResidualDeltaOnly) {
                ElementOutput rounded_base_silu =
                    ElementOutput(silu(base_first));
                ElementOutput base_hidden = ElementOutput(
                    static_cast<float>(rounded_base_silu) * base_second);
                result[lane] = ElementOutput(
                    static_cast<float>(result[lane]) -
                    static_cast<float>(base_hidden));
              }
            }
          }
          int destination_row = output_row;
          if constexpr (IndexedRows) {
            destination_row = shared_dense_slots[local_output_row];
            if (destination_row < 0 ||
                destination_row >= params_ptr->output_rows) {
              continue;
            }
          }
          ElementOutput *destination =
              params_ptr->output +
              static_cast<int64_t>(destination_row) * params_ptr->hidden_size +
              output_column;
          if constexpr (!ResidualDeltaOnly) {
            *reinterpret_cast<Vector *>(destination) = packed_output;
          }
          if constexpr (ResidualCorrection) {
            if (params_ptr->compact_output != nullptr) {
              ElementOutput *compact_destination =
                  params_ptr->compact_output +
                  static_cast<int64_t>(output_row) *
                      params_ptr->hidden_size +
                  output_column;
              *reinterpret_cast<Vector *>(compact_destination) = packed_output;
            }
          }
        }
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(
      cutlass::gemm::GemmCoord threadblock_tile_offset, int thread_index,
      ProblemShape problem_shape) {
    auto fake_stride = cute::make_stride(
        int64_t(params_ptr->logical_rows), cute::_1{},
        int64_t(params_ptr->logical_rows) * params_ptr->hidden_size * 2);
    auto output = cute::make_tensor(cute::make_gmem_ptr(params_ptr->output),
                                    problem_shape, fake_stride);
    auto partitioned = cute::group_modes<3, 6>(
        ThreadMap::partition(output, thread_index, threadblock_tile_offset));
    auto vector_partition = cute::recast<Vector>(partitioned);
    auto register_output = cute::make_tensor_like(
        cute::take<0, 3>(vector_partition));

    auto identity = cute::make_identity_tensor(output.shape());
    auto coordinates = cute::outer_partition(
        cute::group_modes<3, 6>(ThreadMap::partition(
            identity, thread_index, threadblock_tile_offset)),
        cute::Shape<cute::Int<kVectorElements>>{}, cute::_0{});
    return Callbacks<decltype(register_output), decltype(coordinates),
                     ProblemShape>(
        cute::move(register_output), cute::move(coordinates), problem_shape,
        params_ptr, shared_tile, shared_dense_slots, thread_index,
        threadblock_tile_offset.m() * TileColumns,
        threadblock_tile_offset.n() * TileRows);
  }

  Params const *params_ptr;
  ElementOutput *shared_tile;
  int *shared_dense_slots;
};

template <typename ThreadMap, typename ElementOutput, int TileColumns,
          int TileRows, int Threads, bool RoutedRows = false,
          bool ResidualCorrection = false>
struct Sparse24QKVPostOpStore {
  struct Arguments {
    ElementOutput *output = nullptr;
    ElementOutput *dense_base = nullptr;
    const ElementOutput *correction_base = nullptr;
    const ElementOutput *q_weight = nullptr;
    const ElementOutput *k_weight = nullptr;
    const ElementOutput *cos_sin_cache = nullptr;
    const int64_t *position_ids = nullptr;
    const int *row_indices = nullptr;
    const int *dense_slot_by_row = nullptr;
    int q_size = 0;
    int kv_size = 0;
    int logical_rows = 0;
    int output_rows = 0;
    int dense_rows = 0;
    int rotary_dim = 0;
    float epsilon = 0.0f;
    bool is_neox = true;
    bool normalize_qk = false;
  };

  using Params = Arguments;
  struct SharedStorage {
    alignas(16) ElementOutput tile[TileRows * TileColumns];
    alignas(16) int row_mapping[
        (RoutedRows || ResidualCorrection) ? TileRows : 1];
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const &, Arguments const &args, void *) {
    return args;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const &, Arguments const &) {
    return 0;
  }

  CUTLASS_HOST_DEVICE
  Sparse24QKVPostOpStore() {}

  CUTLASS_HOST_DEVICE
  Sparse24QKVPostOpStore(Params const &params, SharedStorage const &shared)
      : params_ptr(&params),
        shared_tile(const_cast<ElementOutput *>(shared.tile)),
        shared_row_mapping(const_cast<int *>(shared.row_mapping)) {}

  static constexpr int kHeadDim = 128;
  static constexpr int kHeadsPerTile = TileColumns / kHeadDim;
  static constexpr int kWarps = Threads / 32;
  static constexpr int kVectorBits =
      ThreadMap::kElementsPerAccess * cutlass::sizeof_bits<ElementOutput>::value;
  using Vector = cute::uint_bit_t<cute::min(128, kVectorBits)>;
  static constexpr int kVectorElements = sizeof(Vector) / sizeof(ElementOutput);
  static_assert(TileColumns % kHeadDim == 0,
                "QKV tile must contain complete 128-dimensional heads");
  static_assert(Threads % 32 == 0, "QKV store requires complete warps");
  static_assert(kVectorElements == 8,
                "QKV store expects half8 epilogue fragments");
  static_assert(!(RoutedRows && ResidualCorrection),
                "QKV routed and residual epilogues are separate modes");
  static constexpr int kVectorsPerRow = TileColumns / kVectorElements;
  static constexpr int kTotalVectors = TileRows * kVectorsPerRow;

  template <class RegisterTensor, class CoordTensor, class ProblemShape>
  struct Callbacks
      : cutlass::epilogue::threadblock::detail::EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(RegisterTensor &&register_output, CoordTensor &&coordinates,
              ProblemShape problem_shape, Params const *params_ptr,
              ElementOutput *shared_tile, int *shared_row_mapping,
              int thread_index,
              int tile_output_base, int tile_row_base)
        : register_output(cute::forward<RegisterTensor>(register_output)),
          coordinates(cute::forward<CoordTensor>(coordinates)),
          problem_shape(problem_shape), params_ptr(params_ptr),
          shared_tile(shared_tile),
          shared_row_mapping(shared_row_mapping),
          thread_index(thread_index),
          tile_output_base(tile_output_base), tile_row_base(tile_row_base) {}

    RegisterTensor register_output;
    CoordTensor coordinates;
    ProblemShape problem_shape;
    Params const *params_ptr;
    ElementOutput *shared_tile;
    int *shared_row_mapping;
    int thread_index;
    int tile_output_base;
    int tile_row_base;

    CUTLASS_DEVICE void begin_step(int) { cute::clear(register_output); }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto visit(
        int, int, int, int fragment_index,
        cutlass::Array<ElementAccumulator, FragmentSize> const &,
        cutlass::Array<ElementInput, FragmentSize> const &fragment) {
      using Converter = cutlass::NumericArrayConverter<
          ElementOutput, ElementInput, FragmentSize,
          cutlass::FloatRoundStyle::round_to_nearest>;
      auto register_fragments =
          cute::recast<cutlass::Array<ElementOutput, FragmentSize>>(
              cute::coalesce(register_output));
      register_fragments(fragment_index) = Converter{}(fragment);
      return fragment;
    }

    CUTLASS_DEVICE void end_step(int step_index) {
      auto source_vectors = cute::filter(register_output);
      auto vector_coordinates =
          cute::filter(coordinates(cute::_, cute::_, cute::_, step_index));
      CUTLASS_PRAGMA_UNROLL
      for (int vector_index = 0; vector_index < cute::size(source_vectors);
           ++vector_index) {
        auto coordinate = vector_coordinates(vector_index);
        int output_column = int(cute::get<0>(coordinate));
        int output_row = int(cute::get<1>(coordinate));
        int local_column = output_column - tile_output_base;
        int local_row = output_row - tile_row_base;
        Vector packed = source_vectors(vector_index);
        ElementOutput const *elements =
            reinterpret_cast<ElementOutput const *>(&packed);
        CUTLASS_PRAGMA_UNROLL
        for (int lane = 0; lane < kVectorElements; ++lane) {
          int row = local_row + lane;
          if (local_column >= 0 && local_column < TileColumns && row >= 0 &&
              row < TileRows) {
            shared_tile[row * TileColumns + local_column] = elements[lane];
          }
        }
      }
    }

    CUTLASS_DEVICE void end_epilogue() {
      __syncthreads();
      int lane = thread_index & 31;
      int warp = thread_index >> 5;
      int output_size = params_ptr->q_size + 2 * params_ptr->kv_size;
      int qk_size = params_ptr->q_size + params_ptr->kv_size;

      // Qwen3/Llama use Neox RoPE over all 128 head dimensions.  Give each
      // lane four adjacent values so shared-memory reads, scale loads, and
      // final stores are 64-bit transactions.  The paired Neox half lives in
      // lane^16 at the same component, avoiding dynamic shared-memory reads
      // for every rotated value.
      if (params_ptr->is_neox && params_ptr->rotary_dim == kHeadDim) {
        constexpr int kValuesPerLane = 4;
        struct alignas(8) PackedHalf4 {
          ElementOutput values[kValuesPerLane];
        };
        CUTLASS_PRAGMA_UNROLL
        for (int task = warp; task < TileRows * kHeadsPerTile;
             task += kWarps) {
          int local_row = task / kHeadsPerTile;
          int local_head = task - local_row * kHeadsPerTile;
          int problem_row = tile_row_base + local_row;
          if (problem_row >= params_ptr->logical_rows) {
            continue;
          }
          int output_row = problem_row;
          if constexpr (RoutedRows) {
            output_row = problem_row;
          } else if constexpr (ResidualCorrection) {
            output_row = params_ptr->row_indices[problem_row];
            if (output_row < 0 || output_row >= params_ptr->output_rows) {
              continue;
            }
          }
          int head_offset = tile_output_base + local_head * kHeadDim;
          bool q_or_k = head_offset < qk_size;
          bool normalize = params_ptr->normalize_qk && q_or_k;
          ElementOutput const *weight =
              head_offset < params_ptr->q_size ? params_ptr->q_weight
                                               : params_ptr->k_weight;
          ElementOutput *head =
              shared_tile + local_row * TileColumns +
              local_head * kHeadDim;
          int lane_dim = lane * kValuesPerLane;
          PackedHalf4 packed = *reinterpret_cast<PackedHalf4 const *>(
              head + lane_dim);
          if constexpr (RoutedRows) {
            int dense_slot = params_ptr->dense_slot_by_row[problem_row];
            if (dense_slot >= 0) {
              if (dense_slot < params_ptr->dense_rows) {
                *reinterpret_cast<PackedHalf4 *>(
                    params_ptr->dense_base +
                    static_cast<int64_t>(dense_slot) * output_size +
                    head_offset + lane_dim) = packed;
              }
              continue;
            }
          } else if constexpr (ResidualCorrection) {
            PackedHalf4 base = *reinterpret_cast<PackedHalf4 const *>(
                params_ptr->correction_base +
                static_cast<int64_t>(problem_row) * output_size +
                head_offset + lane_dim);
            auto *packed_half2 = reinterpret_cast<__half2 *>(&packed);
            auto const *base_half2 =
                reinterpret_cast<__half2 const *>(&base);
            CUTLASS_PRAGMA_UNROLL
            for (int pair = 0; pair < kValuesPerLane / 2; ++pair) {
              packed_half2[pair] = __hadd2(packed_half2[pair],
                                           base_half2[pair]);
            }
          }
          float values[kValuesPerLane];
          float square_sum = 0.0f;
          CUTLASS_PRAGMA_UNROLL
          for (int component = 0; component < kValuesPerLane; ++component) {
            values[component] = static_cast<float>(packed.values[component]);
            if (normalize) {
              square_sum += values[component] * values[component];
            }
          }
          if (normalize) {
            CUTLASS_PRAGMA_UNROLL
            for (int delta = 16; delta > 0; delta >>= 1) {
              square_sum +=
                  __shfl_down_sync(0xffffffff, square_sum, delta);
            }
            float inverse_rms = lane == 0
                                    ? rsqrtf(
                                          square_sum /
                                              static_cast<float>(kHeadDim) +
                                          params_ptr->epsilon)
                                    : 0.0f;
            inverse_rms =
                __shfl_sync(0xffffffff, inverse_rms, 0);
            PackedHalf4 packed_weight =
                *reinterpret_cast<PackedHalf4 const *>(weight + lane_dim);
            CUTLASS_PRAGMA_UNROLL
            for (int component = 0; component < kValuesPerLane;
                 ++component) {
              values[component] *=
                  inverse_rms *
                  static_cast<float>(packed_weight.values[component]);
            }
          }

          if (q_or_k) {
            int64_t position = params_ptr->position_ids[output_row];
            int cache_offset = static_cast<int>(position) * kHeadDim;
            int pair_lane = lane ^ 16;
            bool subtract_pair = lane < 16;
            CUTLASS_PRAGMA_UNROLL
            for (int component = 0; component < kValuesPerLane;
                 ++component) {
              int dim = lane_dim + component;
              int cache_dim = dim & 63;
              float pair = __shfl_sync(
                  0xffffffff, values[component], pair_lane);
              float cosine = static_cast<float>(
                  params_ptr->cos_sin_cache[cache_offset + cache_dim]);
              float sine = static_cast<float>(
                  params_ptr->cos_sin_cache[cache_offset + 64 + cache_dim]);
              float value = subtract_pair
                                ? values[component] * cosine - pair * sine
                                : values[component] * cosine + pair * sine;
              packed.values[component] = ElementOutput(value);
            }
          }
          *reinterpret_cast<PackedHalf4 *>(
              params_ptr->output +
              static_cast<int64_t>(output_row) * output_size + head_offset +
              lane_dim) = packed;
        }
        return;
      }

      if constexpr (ResidualCorrection) {
        for (int vector_index = thread_index; vector_index < kTotalVectors;
             vector_index += Threads) {
          int local_row = vector_index / kVectorsPerRow;
          int vector_column = vector_index % kVectorsPerRow;
          int problem_row = tile_row_base + local_row;
          int output_column =
              tile_output_base + vector_column * kVectorElements;
          if (problem_row >= params_ptr->logical_rows ||
              output_column + kVectorElements > output_size) {
            continue;
          }
          ElementOutput *source =
              shared_tile + local_row * TileColumns +
              vector_column * kVectorElements;
          ElementOutput const *base =
              params_ptr->correction_base +
              static_cast<int64_t>(problem_row) * output_size +
              output_column;
          auto *source_half2 = reinterpret_cast<__half2 *>(source);
          auto const *base_half2 =
              reinterpret_cast<__half2 const *>(base);
          CUTLASS_PRAGMA_UNROLL
          for (int pair = 0; pair < kVectorElements / 2; ++pair) {
            source_half2[pair] =
                __hadd2(source_half2[pair], base_half2[pair]);
          }
        }
        __syncthreads();
      }

      for (int task = warp; task < TileRows * kHeadsPerTile;
           task += kWarps) {
        int local_row = task / kHeadsPerTile;
        int local_head = task - local_row * kHeadsPerTile;
        int problem_row = tile_row_base + local_row;
        if (problem_row >= params_ptr->logical_rows) {
          continue;
        }
        int output_row = problem_row;
        if constexpr (RoutedRows) {
          output_row = problem_row;
        } else if constexpr (ResidualCorrection) {
          output_row = params_ptr->row_indices[problem_row];
          if (output_row < 0 || output_row >= params_ptr->output_rows) {
            continue;
          }
        }
        int head_offset = tile_output_base + local_head * kHeadDim;
        bool q_or_k = head_offset < qk_size;
        bool normalize = params_ptr->normalize_qk && q_or_k;
        ElementOutput const *weight =
            head_offset < params_ptr->q_size ? params_ptr->q_weight
                                             : params_ptr->k_weight;
        ElementOutput *head =
            shared_tile + local_row * TileColumns + local_head * kHeadDim;

        if constexpr (RoutedRows) {
          int dense_slot = params_ptr->dense_slot_by_row[problem_row];
          if (dense_slot >= 0) {
            if (dense_slot < params_ptr->dense_rows) {
              CUTLASS_PRAGMA_UNROLL
              for (int dim = lane; dim < kHeadDim; dim += 32) {
                params_ptr->dense_base[
                    static_cast<int64_t>(dense_slot) * output_size +
                    head_offset + dim] = head[dim];
              }
            }
            continue;
          }
        }

        float square_sum = 0.0f;
        if (normalize) {
          CUTLASS_PRAGMA_UNROLL
          for (int dim = lane; dim < kHeadDim; dim += 32) {
            float value = static_cast<float>(head[dim]);
            square_sum += value * value;
          }
          CUTLASS_PRAGMA_UNROLL
          for (int delta = 16; delta > 0; delta >>= 1) {
            square_sum += __shfl_down_sync(0xffffffff, square_sum, delta);
          }
          square_sum = __shfl_sync(0xffffffff, square_sum, 0);
        }
        float inverse_rms =
            normalize
                ? rsqrtf(square_sum / static_cast<float>(kHeadDim) +
                         params_ptr->epsilon)
                : 1.0f;
        int64_t position = params_ptr->position_ids[output_row];
        int cache_offset = static_cast<int>(position) * params_ptr->rotary_dim;

        CUTLASS_PRAGMA_UNROLL
        for (int dim = lane; dim < kHeadDim; dim += 32) {
          float value = static_cast<float>(head[dim]);
          if (normalize) {
            value *= inverse_rms * static_cast<float>(weight[dim]);
          }
          if (q_or_k && dim < params_ptr->rotary_dim) {
            int pair_dim;
            int cache_dim;
            bool subtract_pair;
            if (params_ptr->is_neox) {
              int half_rotary = params_ptr->rotary_dim / 2;
              pair_dim = dim < half_rotary ? dim + half_rotary
                                           : dim - half_rotary;
              cache_dim = dim % half_rotary;
              subtract_pair = dim < half_rotary;
            } else {
              pair_dim = dim ^ 1;
              cache_dim = dim / 2;
              subtract_pair = (dim & 1) == 0;
            }
            float pair = static_cast<float>(head[pair_dim]);
            if (normalize) {
              pair *= inverse_rms * static_cast<float>(weight[pair_dim]);
            }
            float cos_value = static_cast<float>(
                params_ptr->cos_sin_cache[cache_offset + cache_dim]);
            float sin_value = static_cast<float>(
                params_ptr->cos_sin_cache[
                    cache_offset + params_ptr->rotary_dim / 2 + cache_dim]);
            value = subtract_pair ? value * cos_value - pair * sin_value
                                  : value * cos_value + pair * sin_value;
          }
          params_ptr->output[
              static_cast<int64_t>(output_row) * output_size + head_offset +
              dim] = ElementOutput(value);
        }
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(
      cutlass::gemm::GemmCoord threadblock_tile_offset, int thread_index,
      ProblemShape problem_shape) {
    auto fake_stride = cute::make_stride(
        int64_t(params_ptr->logical_rows), cute::_1{},
        int64_t(params_ptr->logical_rows) *
            (params_ptr->q_size + 2 * params_ptr->kv_size));
    auto output = cute::make_tensor(cute::make_gmem_ptr(params_ptr->output),
                                    problem_shape, fake_stride);
    auto partitioned = cute::group_modes<3, 6>(
        ThreadMap::partition(output, thread_index, threadblock_tile_offset));
    auto vector_partition = cute::recast<Vector>(partitioned);
    auto register_output = cute::make_tensor_like(
        cute::take<0, 3>(vector_partition));

    auto identity = cute::make_identity_tensor(output.shape());
    auto coordinates = cute::outer_partition(
        cute::group_modes<3, 6>(ThreadMap::partition(
            identity, thread_index, threadblock_tile_offset)),
        cute::Shape<cute::Int<kVectorElements>>{}, cute::_0{});
    return Callbacks<decltype(register_output), decltype(coordinates),
                     ProblemShape>(
        cute::move(register_output), cute::move(coordinates), problem_shape,
        params_ptr, shared_tile, shared_row_mapping, thread_index,
        threadblock_tile_offset.m() * TileColumns,
        threadblock_tile_offset.n() * TileRows);
  }

  Params const *params_ptr;
  ElementOutput *shared_tile;
  int *shared_row_mapping;
};

template <typename ThreadblockShape_, typename WarpShape_, int Stages_ = 3,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
using DeviceSparseGemmVec8Variant = cutlass::gemm::device::SparseGemm<
    Element, cutlass::layout::RowMajor, Element, cutlass::layout::ColumnMajor,
    Element, DeviceLayoutC, float, cutlass::arch::OpClassTensorOp,
    DeviceArchTag, ThreadblockShape_, WarpShape_,
    DeviceInstructionShape, DeviceEpilogueOpVec8,
    ThreadblockSwizzle_, Stages_, 8, 8, false,
    cutlass::arch::OpMultiplyAdd>;

template <typename ThreadblockShape_, typename WarpShape_, int Stages_ = 3,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
using DeviceSparseGemmF16AccumVariant = cutlass::gemm::device::SparseGemm<
    Element, cutlass::layout::RowMajor, Element, cutlass::layout::ColumnMajor,
    Element, DeviceLayoutC, Element, cutlass::arch::OpClassTensorOp,
    DeviceArchTag, ThreadblockShape_, WarpShape_, DeviceInstructionShape,
    DeviceEpilogueOpVec8F16Accum, ThreadblockSwizzle_, Stages_, 8, 8, false,
    cutlass::arch::OpMultiplyAdd>;

template <typename LayoutB_, typename ThreadblockShape_, typename WarpShape_,
          int Stages_ = 3,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
using DeviceSparseGemmVec8LayoutBVariant = cutlass::gemm::device::SparseGemm<
    Element, cutlass::layout::RowMajor, Element, LayoutB_, Element,
    DeviceLayoutC, float, cutlass::arch::OpClassTensorOp, DeviceArchTag,
    ThreadblockShape_, WarpShape_, DeviceInstructionShape,
    DeviceEpilogueOpVec8, ThreadblockSwizzle_, Stages_, 8, 8, false,
    cutlass::arch::OpMultiplyAdd>;

using DeviceSparseGemmVec8 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape, DeviceWarpShape, 3>;
using DeviceSparseGemmVec8M64N32K64S3 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape64x32x64, DeviceWarpShape, 3>;
using DeviceSparseGemmVec8M64N32K64S2 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape64x32x64, DeviceWarpShape, 2>;
using DeviceSparseGemmVec8M64N32K64S4 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape64x32x64, DeviceWarpShape, 4>;
using DeviceSparseGemmVec8M64N32K64S5 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape64x32x64, DeviceWarpShape, 5>;
using DeviceSparseGemmVec8M64N128K64S3 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape64x128x64,
                                DeviceWarpShape32x64x64, 3>;
using DeviceSparseGemmVec8M64N128K64S2 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape64x128x64,
                                DeviceWarpShape32x64x64, 2>;
using DeviceSparseGemmVec8M64N128K64S4 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape64x128x64,
                                DeviceWarpShape32x64x64, 4>;
using DeviceSparseGemmVec8M64N64K64S2 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape, DeviceWarpShape, 2>;
using DeviceSparseGemmVec8M64N64K64S4 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape, DeviceWarpShape, 4>;
using DeviceSparseGemmVec8M64N64K64S5 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape, DeviceWarpShape, 5>;
using DeviceSparseGemmVec8M64N64K64S6 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape, DeviceWarpShape, 6>;
using DeviceSparseGemmVec8M64N64K64S7 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape, DeviceWarpShape, 7>;
using DeviceSparseGemmVec8M128N32K64S4 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape128x32x64, DeviceWarpShape, 4>;
using DeviceSparseGemmVec8M128N32K64S4Sw2 =
    DeviceSparseGemmVec8Variant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 4,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>>;
using DeviceSparseGemmVec8M128N32K64S4Sw4 =
    DeviceSparseGemmVec8Variant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 4,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;
using DeviceSparseGemmVec8M128N64K64S2 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape128x64x64, DeviceWarpShape, 2>;
using DeviceSparseGemmVec8M128N64K64S3 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape128x64x64, DeviceWarpShape, 3>;
using DeviceSparseGemmVec8M128N64K64S4 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape128x64x64, DeviceWarpShape, 4>;
using DeviceSparseGemmVec8M128N64K64S5 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape128x64x64, DeviceWarpShape, 5>;
using DeviceSparseGemmVec8M128N128K64S2 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape128x128x64,
                                DeviceWarpShape32x64x64, 2>;
using DeviceSparseGemmVec8M128N128K64S3 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape128x128x64,
                                DeviceWarpShape32x64x64, 3>;
using DeviceSparseGemmVec8M128N128K64S3Sw2 =
    DeviceSparseGemmVec8Variant<
        DeviceThreadblockShape128x128x64, DeviceWarpShape32x64x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>>;
using DeviceSparseGemmVec8M128N128K64S3Sw4 =
    DeviceSparseGemmVec8Variant<
        DeviceThreadblockShape128x128x64, DeviceWarpShape32x64x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;
using DeviceSparseGemmVec8M256N64K64S2 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape256x64x64,
                                DeviceWarpShape64x32x64, 2>;
using DeviceSparseGemmVec8M256N64K64S3 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape256x64x64,
                                DeviceWarpShape64x32x64, 3>;
using DeviceSparseGemmVec8M256N64K64S3Sw2 =
    DeviceSparseGemmVec8Variant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>>;
using DeviceSparseGemmVec8M256N64K64S3Sw4 =
    DeviceSparseGemmVec8Variant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;
using DeviceSparseGemmVec8M256N128K64S2 =
    DeviceSparseGemmVec8Variant<DeviceThreadblockShape256x128x64,
                                DeviceWarpShape64x64x64, 2>;
using DeviceSparseGemmVec8M256N32K64S3Sw4 =
    DeviceSparseGemmVec8Variant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;
using DeviceSparseGemmF16AccumM64N32K64S3 =
    DeviceSparseGemmF16AccumVariant<DeviceThreadblockShape64x32x64,
                                    DeviceWarpShape, 3>;
using DeviceSparseGemmF16AccumM64N32K64S4 =
    DeviceSparseGemmF16AccumVariant<DeviceThreadblockShape64x32x64,
                                    DeviceWarpShape, 4>;
using DeviceSparseGemmF16AccumM128N64K64S4 =
    DeviceSparseGemmF16AccumVariant<DeviceThreadblockShape128x64x64,
                                    DeviceWarpShape, 4>;
using DeviceSparseGemmF16AccumM128N64K64S5 =
    DeviceSparseGemmF16AccumVariant<DeviceThreadblockShape128x64x64,
                                    DeviceWarpShape, 5>;
using DeviceSparseGemmF16AccumM256N32K64S3Sw4 =
    DeviceSparseGemmF16AccumVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;
using DeviceSparseGemmF16AccumM256N64K64S3 =
    DeviceSparseGemmF16AccumVariant<DeviceThreadblockShape256x64x64,
                                    DeviceWarpShape64x32x64, 3>;
using DeviceSparseGemmF16AccumM256N64K64S3Sw4 =
    DeviceSparseGemmF16AccumVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;

// CUTLASS 2.x does not expose GatherB on SparseGemm even though its global
// tile iterator supports it. Rebuild only the mainloop iterator so compact
// residual GEMMs can read confidence-selected rows directly from X.
template <typename BaseDeviceGemm>
struct Sparse24GatherGemmTraits {
  using BaseKernel = typename BaseDeviceGemm::GemmKernel;
  using BaseMma = typename BaseKernel::Mma;
  using BaseIteratorB = typename BaseMma::IteratorB;
  using GatherIteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          typename BaseIteratorB::Shape, typename BaseIteratorB::Element,
          typename BaseIteratorB::Layout, BaseIteratorB::kAdvanceRank,
          typename BaseIteratorB::ThreadMap,
          typename BaseIteratorB::AccessType, true>;
  using Mma = cutlass::gemm::threadblock::SparseMmaMultistage<
      typename BaseMma::Shape, typename BaseMma::IteratorA,
      typename BaseMma::SmemIteratorA, BaseMma::kCacheOpA, GatherIteratorB,
      typename BaseMma::SmemIteratorB, BaseMma::kCacheOpB,
      typename BaseMma::ElementC, typename BaseMma::LayoutC,
      typename BaseMma::IteratorE, typename BaseMma::SmemIteratorE,
      BaseMma::kCacheOpE, typename BaseMma::Policy, BaseMma::Base::kStages>;
  using Epilogue = typename BaseKernel::Epilogue;
  using OutputOp = typename BaseKernel::OutputOp;
  using ThreadblockShape = typename Mma::Shape;
  using ElementE = typename Mma::ElementE;
  static constexpr int kThreadCount = BaseKernel::kThreadCount;
  static constexpr int kSparse = Mma::kSparse;
  static constexpr int kElementsPerElementE = Mma::kElementsPerElementE;
};

template <typename BaseDeviceGemm>
struct Sparse24GatherGemmKernel {
  using Traits = Sparse24GatherGemmTraits<BaseDeviceGemm>;
  using Mma = typename Traits::Mma;
  using Epilogue = typename Traits::Epilogue;
  using OutputOp = typename Traits::OutputOp;
  using ThreadblockShape = typename Traits::ThreadblockShape;
  using ElementE = typename Traits::ElementE;
  static constexpr int kThreadCount = Traits::kThreadCount;

  struct Params {
    const Element *x = nullptr;
    const Element *values = nullptr;
    ElementE *metadata = nullptr;
    const int *row_indices = nullptr;
    Element *output = nullptr;
    int rows = 0;
    int output_leading_rows = 0;
    int K = 0;
    int N = 0;
    typename OutputOp::Params output_op{};
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  CUTLASS_DEVICE void operator()(Params const &params,
                                 SharedStorage &shared_storage) {
    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx & 31;
    int row_tiles =
        (params.rows + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
    int feature_tile = blockIdx.x / row_tiles;
    int row_tile = blockIdx.x - feature_tile * row_tiles;
    cutlass::gemm::GemmCoord problem_size(params.N, params.rows, params.K);
    cutlass::gemm::GemmCoord threadblock_offset(
        feature_tile * ThreadblockShape::kM,
        row_tile * ThreadblockShape::kN, 0);
    int sparse_k = params.K / Traits::kSparse;
    int columns_e = sparse_k / Traits::kElementsPerElementE;

    using LayoutA = typename Mma::IteratorA::Layout;
    using LayoutB = typename Mma::IteratorB::Layout;
    using LayoutE = typename Mma::IteratorE::Layout;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;
    typename Mma::IteratorA::Params params_A{LayoutA(sparse_k)};
    typename Mma::IteratorB::Params params_B{LayoutB(params.K)};
    LayoutE layout_e = LayoutE::packed({params.N, columns_e});
    typename Mma::IteratorE::Params params_E{layout_e};

    typename Mma::IteratorA iterator_A(
        params_A, const_cast<Element *>(params.values),
        {params.N, sparse_k}, thread_idx, {threadblock_offset.m(), 0});
    typename Mma::IteratorB iterator_B(
        params_B, const_cast<Element *>(params.x), {params.K, params.rows},
        thread_idx, {0, threadblock_offset.n()}, params.row_indices);
    typename Mma::IteratorE iterator_E(
        params_E, params.metadata, {params.N, columns_e}, thread_idx,
        {threadblock_offset.m(), 0});

    typename Mma::FragmentC accumulators;
    accumulators.clear();
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    int gemm_k_iterations =
        (params.K + ThreadblockShape::kK - 1) / ThreadblockShape::kK;
    __syncthreads();
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
        accumulators);

    LayoutC layout_c(params.output_leading_rows);
    typename Epilogue::OutputTileIterator::Params params_c(layout_c);
    typename Epilogue::OutputTileIterator iterator_C(
        params_c, params.output, problem_size.mn(), thread_idx,
        threadblock_offset.mn());
    typename Epilogue::OutputTileIterator iterator_D(
        params_c, params.output, problem_size.mn(), thread_idx,
        threadblock_offset.mn());
    OutputOp output_op(params.output_op);
    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, accumulators, iterator_C);
  }
};

// Dense routed rows are the sum of complementary 2:4 weights. Accumulate both
// sparse GEMMs inside one CTA so the exact gate/up result reaches SwiGLU and
// the indexed output store without an intermediate correction tensor.
template <typename BaseDeviceGemm>
struct Sparse24DualSwiGLUKernel {
  using BaseKernel = typename BaseDeviceGemm::GemmKernel;
  using Mma = typename BaseKernel::Mma;
  using Epilogue = typename BaseKernel::Epilogue;
  using FusionCallbacks = typename BaseKernel::FusionCallbacks;
  using ThreadblockShape = typename Mma::Shape;
  using ElementE = typename Mma::ElementE;
  static constexpr int kThreadCount = BaseKernel::kThreadCount;
  static constexpr int kSparse = Mma::kSparse;
  static constexpr int kElementsPerElementE = Mma::kElementsPerElementE;

  struct Params {
    const Element *x = nullptr;
    const Element *values[2] = {nullptr, nullptr};
    ElementE *metadata[2] = {nullptr, nullptr};
    int rows = 0;
    int K = 0;
    int N = 0;
    typename FusionCallbacks::Params output_op{};
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  CUTLASS_DEVICE void run_mainloop(
      Params const &params, int operand,
      cutlass::gemm::GemmCoord threadblock_offset, int thread_idx,
      int warp_idx, int lane_idx, typename Mma::FragmentC &accumulators,
      SharedStorage &shared_storage) {
    int sparse_k = params.K / kSparse;
    int columns_e = sparse_k / kElementsPerElementE;
    using LayoutA = typename Mma::IteratorA::Layout;
    using LayoutB = typename Mma::IteratorB::Layout;
    using LayoutE = typename Mma::IteratorE::Layout;
    typename Mma::IteratorA::Params params_A{LayoutA(sparse_k)};
    typename Mma::IteratorB::Params params_B{LayoutB(params.K)};
    LayoutE layout_e = LayoutE::packed({params.N, columns_e});
    typename Mma::IteratorE::Params params_E{layout_e};

    typename Mma::IteratorA iterator_A(
        params_A, const_cast<Element *>(params.values[operand]),
        {params.N, sparse_k}, thread_idx, {threadblock_offset.m(), 0});
    typename Mma::IteratorB iterator_B(
        params_B, const_cast<Element *>(params.x), {params.K, params.rows},
        thread_idx, {0, threadblock_offset.n()});
    typename Mma::IteratorE iterator_E(
        params_E, params.metadata[operand], {params.N, columns_e},
        thread_idx, {threadblock_offset.m(), 0});

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    int gemm_k_iterations =
        (params.K + ThreadblockShape::kK - 1) / ThreadblockShape::kK;
    __syncthreads();
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
        accumulators);
  }

  CUTLASS_DEVICE void operator()(Params const &params,
                                 SharedStorage &shared_storage) {
    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx & 31;
    int row_tiles =
        (params.rows + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
    int feature_tile = blockIdx.x / row_tiles;
    int row_tile = blockIdx.x - feature_tile * row_tiles;
    cutlass::gemm::GemmCoord threadblock_offset(
        feature_tile * ThreadblockShape::kM,
        row_tile * ThreadblockShape::kN, 0);

    typename Mma::FragmentC accumulators;
    accumulators.clear();
    run_mainloop(params, 0, threadblock_offset, thread_idx, warp_idx,
                 lane_idx, accumulators, shared_storage);
    run_mainloop(params, 1, threadblock_offset, thread_idx, warp_idx,
                 lane_idx, accumulators, shared_storage);

    cutlass::gemm::GemmCoord threadblock_tile_index(feature_tile, row_tile, 0);
    auto problem_shape = cute::make_shape(
        int32_t(params.N), int32_t(params.rows), int32_t(1));
    Epilogue epilogue(params.output_op, shared_storage.epilogue, thread_idx,
                      warp_idx, lane_idx);
    epilogue(accumulators, threadblock_tile_index, problem_shape, thread_idx);
  }
};

template <typename DeviceGemm>
union Sparse24PersistentProblemSharedStorage {
  using BaseKernel = typename DeviceGemm::GemmKernel;
  typename BaseKernel::Mma::SharedStorage main_loop;
  typename BaseKernel::Epilogue::SharedStorage epilogue;
};

// Rebuild only the sparse mainloop's B iterator when a compact logical row
// space must gather directly from the original verifier activation.
template <typename BaseMma>
struct Sparse24GatherMma {
  using BaseIteratorB = typename BaseMma::IteratorB;
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<BaseMma::Shape::kK, BaseMma::Shape::kN>,
          typename BaseIteratorB::Element, typename BaseIteratorB::Layout, 0,
          typename BaseIteratorB::ThreadMap,
          typename BaseIteratorB::AccessType, true>;
  using Type = cutlass::gemm::threadblock::SparseMmaMultistage<
      typename BaseMma::Shape, typename BaseMma::IteratorA,
      typename BaseMma::SmemIteratorA, BaseMma::kCacheOpA, IteratorB,
      typename BaseMma::SmemIteratorB, BaseMma::kCacheOpB,
      typename BaseMma::ElementC, typename BaseMma::LayoutC,
      typename BaseMma::IteratorE, typename BaseMma::SmemIteratorE,
      BaseMma::kCacheOpE, typename BaseMma::Policy,
      BaseMma::Detail::kStages>;
};

template <typename BaseMma, bool GatherRows>
struct Sparse24MmaSelector {
  using Type = BaseMma;
};

template <typename BaseMma>
struct Sparse24MmaSelector<BaseMma, true> {
  using Type = typename Sparse24GatherMma<BaseMma>::Type;
};

template <typename FullDeviceGemm, typename ResidualDeviceGemm>
struct Sparse24PairedPersistentKernel {
  using FullBaseKernel = typename FullDeviceGemm::GemmKernel;
  using ResidualBaseKernel = typename ResidualDeviceGemm::GemmKernel;
  using FullMma = typename FullBaseKernel::Mma;
  using ResidualMma = typename ResidualBaseKernel::Mma;
  using FullThreadblockShape = typename FullMma::Shape;
  using ResidualThreadblockShape = typename ResidualMma::Shape;
  using ElementE = typename FullMma::ElementE;
  static constexpr int kThreadCount = FullBaseKernel::kThreadCount;
  static constexpr int kSparse = FullBaseKernel::kSparse;

  static_assert(kThreadCount == ResidualBaseKernel::kThreadCount,
                "paired persistent GEMMs require equal CTA sizes");
  static_assert(kSparse == ResidualBaseKernel::kSparse,
                "paired persistent GEMMs require one sparse format");
  static_assert(
      cutlass::platform::is_same<ElementE,
                                 typename ResidualMma::ElementE>::value,
      "paired persistent GEMMs require one metadata element type");

  struct Problem {
    const Element *x = nullptr;
    const Element *values = nullptr;
    ElementE *metadata = nullptr;
    Element *output = nullptr;
    int rows = 0;
  };

  struct Params {
    Problem problems[2];
    int K = 0;
    int N = 0;
    int first_problem_tiles = 0;
    int total_tiles = 0;
    int full_worker_blocks = 0;
    int residual_worker_blocks = 0;
    int interleaved_schedule = 0;
    typename FullBaseKernel::OutputOp::Params full_output_op{};
    typename ResidualBaseKernel::OutputOp::Params residual_output_op{};
  };

  union SharedStorage {
    Sparse24PersistentProblemSharedStorage<FullDeviceGemm> full;
    Sparse24PersistentProblemSharedStorage<ResidualDeviceGemm> residual;
  };

  template <typename DeviceGemm>
  CUTLASS_DEVICE void run_tile(
      Problem const &problem, int local_tile, int K, int N,
      typename DeviceGemm::GemmKernel::OutputOp::Params const &output_op_params,
      Sparse24PersistentProblemSharedStorage<DeviceGemm> &shared_storage) {
    using BaseKernel = typename DeviceGemm::GemmKernel;
    using Mma = typename BaseKernel::Mma;
    using Epilogue = typename BaseKernel::Epilogue;
    using OutputOp = typename BaseKernel::OutputOp;
    using ThreadblockShape = typename Mma::Shape;
    static constexpr int kElementsPerElementE =
        BaseKernel::kElementsPerElementE;

    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % 32;
    int feature_tiles =
        (N + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
    int row_tiles =
        (problem.rows + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
    int feature_tile = local_tile / row_tiles;
    int row_tile = local_tile - feature_tile * row_tiles;
    if (feature_tile >= feature_tiles) {
      return;
    }

    cutlass::gemm::GemmCoord problem_size(N, problem.rows, K);
    cutlass::gemm::GemmCoord threadblock_offset(
        feature_tile * ThreadblockShape::kM,
        row_tile * ThreadblockShape::kN, 0);
    int sparse_k = K / kSparse;
    int columns_e = K / kSparse / kElementsPerElementE;

    using LayoutA = typename Mma::IteratorA::Layout;
    using LayoutB = typename Mma::IteratorB::Layout;
    using LayoutE = typename Mma::IteratorE::Layout;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;
    typename Mma::IteratorA::Params params_A{LayoutA(sparse_k)};
    typename Mma::IteratorB::Params params_B{LayoutB(K)};
    LayoutE layout_e = LayoutE::packed({N, columns_e});
    typename Mma::IteratorE::Params params_E{layout_e};

    typename Mma::IteratorA iterator_A(
        params_A, const_cast<Element *>(problem.values), {N, sparse_k},
        thread_idx, {threadblock_offset.m(), 0});
    typename Mma::IteratorB iterator_B(
        params_B, const_cast<Element *>(problem.x), {K, problem.rows},
        thread_idx, {0, threadblock_offset.n()});
    typename Mma::IteratorE iterator_E(
        params_E, problem.metadata, {N, columns_e}, thread_idx,
        {threadblock_offset.m(), 0});

    typename Mma::FragmentC accumulators;
    accumulators.clear();
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    int gemm_k_iterations =
        (K + ThreadblockShape::kK - 1) / ThreadblockShape::kK;
    __syncthreads();
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
        accumulators);

    LayoutC layout_c(problem.rows);
    typename Epilogue::OutputTileIterator::Params params_c(layout_c);
    typename Epilogue::OutputTileIterator iterator_C(
        params_c, problem.output, problem_size.mn(), thread_idx,
        threadblock_offset.mn());
    typename Epilogue::OutputTileIterator iterator_D(
        params_c, problem.output, problem_size.mn(), thread_idx,
        threadblock_offset.mn());
    OutputOp output_op(output_op_params);
    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, accumulators, iterator_C);
    __syncthreads();
  }

  CUTLASS_DEVICE void operator()(Params const &params,
                                 SharedStorage &shared_storage) {
    if (params.interleaved_schedule) {
      int residual_tiles = params.total_tiles - params.first_problem_tiles;
      for (int work = blockIdx.x; work < params.total_tiles;
           work += gridDim.x) {
        int residual_before = int(
            (int64_t(work) * residual_tiles) / params.total_tiles);
        int residual_after = int(
            (int64_t(work + 1) * residual_tiles) / params.total_tiles);
        if (residual_after > residual_before) {
          run_tile<ResidualDeviceGemm>(
              params.problems[1], residual_after - 1, params.K, params.N,
              params.residual_output_op, shared_storage.residual);
        } else {
          run_tile<FullDeviceGemm>(
              params.problems[0], work - residual_before, params.K, params.N,
              params.full_output_op, shared_storage.full);
        }
      }
      return;
    }

    if (blockIdx.x < params.full_worker_blocks) {
      for (int tile = blockIdx.x; tile < params.first_problem_tiles;
           tile += params.full_worker_blocks) {
        run_tile<FullDeviceGemm>(
            params.problems[0], tile, params.K, params.N,
            params.full_output_op, shared_storage.full);
      }
      return;
    }

    int worker = blockIdx.x - params.full_worker_blocks;
    int residual_tiles = params.total_tiles - params.first_problem_tiles;
    for (int tile = worker; tile < residual_tiles;
         tile += params.residual_worker_blocks) {
        run_tile<ResidualDeviceGemm>(
            params.problems[1], tile, params.K, params.N,
            params.residual_output_op,
            shared_storage.residual);
    }
  }
};

template <typename FullDeviceGemm, typename ResidualDeviceGemm,
          bool GatherResidualRows = false,
          bool InplaceResidualAdd = false,
          bool FusedQkvPostop = false,
          bool FinalizeResidualAdd = false,
          bool SelfContainedResidualCorrection = false,
          bool LastFullTileResidual = false,
          bool FinalizeQkvPostop = false>
struct Sparse24PairedPersistentVisitorKernel {
  using FullBaseKernel = typename FullDeviceGemm::GemmKernel;
  using ResidualBaseKernel = typename ResidualDeviceGemm::GemmKernel;
  using FullMma = typename FullBaseKernel::Mma;
  using ResidualBaseMma = typename ResidualBaseKernel::Mma;
  using ResidualMma =
      typename Sparse24MmaSelector<ResidualBaseMma,
                                   GatherResidualRows>::Type;
  using FullThreadblockShape = typename FullMma::Shape;
  using ResidualThreadblockShape = typename ResidualMma::Shape;
  using ElementE = typename FullMma::ElementE;
  static constexpr int kThreadCount = FullBaseKernel::kThreadCount;
  static constexpr int kSparse = FullBaseKernel::kSparse;

  static_assert(kThreadCount == ResidualBaseKernel::kThreadCount,
                "paired visitor GEMMs require equal CTA sizes");
  static_assert(kSparse == ResidualBaseKernel::kSparse,
                "paired visitor GEMMs require one sparse format");
  static_assert(
      cutlass::platform::is_same<ElementE,
                                 typename ResidualMma::ElementE>::value,
      "paired visitor GEMMs require one metadata element type");
  static_assert(
      cutlass::platform::is_same<typename ResidualMma::SharedStorage,
                                 typename ResidualBaseMma::SharedStorage>::value,
      "gathering residual rows must preserve visitor MMA shared storage");
  static_assert(
      !InplaceResidualAdd ||
          (FullThreadblockShape::kM == ResidualThreadblockShape::kM),
      "in-place residual accumulation requires matching feature tiles");
  static_assert(!FusedQkvPostop || GatherResidualRows,
                "fused QKV post-op requires gathered residual rows");
  static_assert(!FusedQkvPostop || !InplaceResidualAdd,
                "fused QKV post-op consumes separate paired outputs");
  static_assert(!FinalizeResidualAdd || GatherResidualRows,
                "residual finalizer requires gathered residual rows");
  static_assert(!FinalizeResidualAdd || !InplaceResidualAdd,
                "residual finalizer consumes separate paired outputs");
  static_assert(!FinalizeResidualAdd || !FusedQkvPostop,
                "residual finalizer and QKV post-op are separate modes");
  static_assert(!SelfContainedResidualCorrection || GatherResidualRows,
                "self-contained correction requires gathered dense rows");
  static_assert(!SelfContainedResidualCorrection || !InplaceResidualAdd,
                "self-contained correction does not use feature counters");
  static_assert(!SelfContainedResidualCorrection || !FusedQkvPostop,
                "self-contained correction and QKV post-op are separate");
  static_assert(!SelfContainedResidualCorrection || !FinalizeResidualAdd,
                "self-contained correction and finalizer are separate");
  static_assert(!LastFullTileResidual || InplaceResidualAdd,
                "last-full-tile ownership requires in-place residual add");
  static_assert(!LastFullTileResidual || GatherResidualRows,
                "last-full-tile ownership requires gathered residual rows");
  static_assert(!LastFullTileResidual || !FusedQkvPostop,
                "last-full-tile ownership and QKV post-op are separate");
  static_assert(!LastFullTileResidual || !FinalizeResidualAdd,
                "last-full-tile ownership and finalizer are separate");
  static_assert(!LastFullTileResidual ||
                    !SelfContainedResidualCorrection,
                "last-full-tile ownership and self-contained correction are separate");
  static_assert(!FinalizeQkvPostop || FinalizeResidualAdd,
                "feature-local QKV post-op requires residual finalization");
  static_assert(!FinalizeQkvPostop || GatherResidualRows,
                "feature-local QKV post-op requires gathered residual rows");
  static_assert(!FinalizeQkvPostop || !FusedQkvPostop,
                "feature-local and grid-wide QKV post-ops are separate modes");
  static_assert(
      !FinalizeResidualAdd ||
          (FullThreadblockShape::kM == ResidualThreadblockShape::kM),
      "residual finalizer requires matching feature tiles");
  static_assert(
      !FinalizeQkvPostop ||
          (FullThreadblockShape::kM == 256 &&
           ResidualThreadblockShape::kM == 256 &&
           (kThreadCount == 128 || kThreadCount == 256)),
      "feature-local QKV post-op requires matching 256-feature CTAs with "
      "four or eight warps");
  static_assert(
      !FusedQkvPostop ||
          (FullThreadblockShape::kM == 256 &&
           ResidualThreadblockShape::kM == 256 && kThreadCount == 256),
      "fused QKV post-op requires matching 256-feature, 256-thread CTAs");

  struct Problem {
    const Element *x = nullptr;
    const Element *values = nullptr;
    ElementE *metadata = nullptr;
    const int *row_indices = nullptr;
    int rows = 0;
  };

  struct Params {
    Problem problems[2];
    int K = 0;
    int N = 0;
    int first_problem_tiles = 0;
    int total_tiles = 0;
    int full_worker_blocks = 0;
    int residual_worker_blocks = 0;
    int interleaved_schedule = 0;
    int full_row_tiles = 0;
    int residual_row_tiles = 0;
    int *feature_counters = nullptr;
    const int *dense_slot_by_row = nullptr;
    const Element *q_weight = nullptr;
    const Element *k_weight = nullptr;
    const Element *cos_sin_cache = nullptr;
    const int64_t *position_ids = nullptr;
    int q_size = 0;
    int kv_size = 0;
    int rotary_dim = 0;
    float epsilon = 0.0f;
    int is_neox = 0;
    int normalize_qk = 0;
    int *grid_barrier = nullptr;
    typename FullBaseKernel::FusionCallbacks::Params full_output_op{};
    typename ResidualBaseKernel::FusionCallbacks::Params
        residual_base_output_op{};
    typename ResidualBaseKernel::FusionCallbacks::Params residual_output_op{};
  };

  union GemmSharedStorage {
    Sparse24PersistentProblemSharedStorage<FullDeviceGemm> full;
    Sparse24PersistentProblemSharedStorage<ResidualDeviceGemm> residual;
  };

  struct SharedStorage {
    GemmSharedStorage gemm;
    alignas(16) int route_rows[ResidualThreadblockShape::kN];
  };

  template <typename DeviceGemm, bool GatherRows, bool ResidualProblem>
  CUTLASS_DEVICE void run_tile(
      Params const &params, Problem const &problem, int local_tile, int K, int N,
      typename DeviceGemm::GemmKernel::FusionCallbacks::Params const
          &output_op_params,
      Sparse24PersistentProblemSharedStorage<DeviceGemm> &shared_storage,
      int *route_rows) {
    using BaseKernel = typename DeviceGemm::GemmKernel;
    using BaseMma = typename BaseKernel::Mma;
    using Mma = typename Sparse24MmaSelector<BaseMma, GatherRows>::Type;
    using Epilogue = typename BaseKernel::Epilogue;
    using ThreadblockShape = typename Mma::Shape;
    static constexpr int kElementsPerElementE =
        BaseKernel::kElementsPerElementE;

    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % 32;
    int feature_tiles =
        (N + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
    int row_tiles =
        (problem.rows + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
    int feature_tile = local_tile / row_tiles;
    int row_tile = local_tile - feature_tile * row_tiles;
    if (feature_tile >= feature_tiles) {
      return;
    }

    cutlass::gemm::GemmCoord threadblock_offset(
        feature_tile * ThreadblockShape::kM,
        row_tile * ThreadblockShape::kN, 0);
    int tile_row_base = threadblock_offset.n();
    int tile_rows = problem.rows - tile_row_base;
    tile_rows = tile_rows < ThreadblockShape::kN
                    ? tile_rows
                    : ThreadblockShape::kN;
    if constexpr (GatherRows) {
      for (int local_row = thread_idx; local_row < tile_rows;
           local_row += kThreadCount) {
        route_rows[local_row] =
            problem.row_indices[tile_row_base + local_row];
      }
    }
    int sparse_k = K / kSparse;
    int columns_e = K / kSparse / kElementsPerElementE;

    using LayoutA = typename Mma::IteratorA::Layout;
    using LayoutB = typename Mma::IteratorB::Layout;
    using LayoutE = typename Mma::IteratorE::Layout;
    typename Mma::IteratorA::Params params_A{LayoutA(sparse_k)};
    typename Mma::IteratorB::Params params_B{LayoutB(K)};
    LayoutE layout_e = LayoutE::packed({N, columns_e});
    typename Mma::IteratorE::Params params_E{layout_e};

    typename Mma::IteratorA iterator_A(
        params_A, const_cast<Element *>(problem.values), {N, sparse_k},
        thread_idx, {threadblock_offset.m(), 0});
    typename Mma::IteratorE iterator_E(
        params_E, problem.metadata, {N, columns_e}, thread_idx,
        {threadblock_offset.m(), 0});

    typename Mma::FragmentC accumulators;
    accumulators.clear();
    int gemm_k_iterations =
        (K + ThreadblockShape::kK - 1) / ThreadblockShape::kK;
    if constexpr (GatherRows) {
      typename Mma::IteratorB iterator_B(
          params_B, const_cast<Element *>(problem.x), {K, tile_rows},
          thread_idx, {0, 0}, route_rows);
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
      __syncthreads();
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
          accumulators);
    } else {
      typename Mma::IteratorB iterator_B(
          params_B, const_cast<Element *>(problem.x), {K, problem.rows},
          thread_idx, {0, threadblock_offset.n()});
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
      __syncthreads();
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
          accumulators);
    }

    if constexpr (InplaceResidualAdd && ResidualProblem &&
                  !LastFullTileResidual) {
      if (thread_idx == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> counter(
            params.feature_counters[feature_tile]);
        while (counter.load(cuda::memory_order_acquire) <
               params.full_row_tiles) {
          __nanosleep(64);
        }
      }
      __syncthreads();
    }

    cutlass::gemm::GemmCoord threadblock_tile_index(feature_tile, row_tile, 0);
    auto problem_shape = cute::make_shape(int32_t(N), int32_t(problem.rows),
                                          int32_t(1));
    Epilogue epilogue(output_op_params, shared_storage.epilogue, thread_idx,
                      warp_idx, lane_idx);
    epilogue(accumulators, threadblock_tile_index, problem_shape, thread_idx);
    __syncthreads();
    if constexpr (InplaceResidualAdd) {
      if (thread_idx == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> counter(
            params.feature_counters[feature_tile]);
        int previous;
        if constexpr (LastFullTileResidual && !ResidualProblem) {
          previous = counter.fetch_add(1, cuda::memory_order_acq_rel);
          route_rows[0] = previous + 1 == params.full_row_tiles;
        } else {
          previous = counter.fetch_add(1, cuda::memory_order_release);
        }
        if constexpr (ResidualProblem) {
          int completed =
              params.full_row_tiles + params.residual_row_tiles;
          if (previous + 1 == completed) {
            counter.store(0, cuda::memory_order_release);
          }
        }
      }
      __syncthreads();
    }
    if constexpr (FinalizeResidualAdd && FinalizeQkvPostop) {
      cuda::atomic_ref<int, cuda::thread_scope_device> counter(
          params.feature_counters[feature_tile]);
      int completed_tiles =
          params.full_row_tiles + params.residual_row_tiles;
      if constexpr (ResidualProblem) {
        if (thread_idx == 0) {
          while (counter.load(cuda::memory_order_acquire) <
                 params.full_row_tiles) {
            __nanosleep(64);
          }
        }
        __syncthreads();

        constexpr int kHalf2PerFeatureTile =
            FullThreadblockShape::kM / 2;
        int feature_base = feature_tile * FullThreadblockShape::kM;
        __half2 *full_output = reinterpret_cast<__half2 *>(
            params.full_output_op.op_1.output);
        const __half2 *residual_output = reinterpret_cast<const __half2 *>(
            params.residual_output_op.op_1.output);
        int output_half2_columns = params.N / 2;
        int tile_pairs = tile_rows * kHalf2PerFeatureTile;
        for (int index = thread_idx; index < tile_pairs;
             index += kThreadCount) {
          int local_dense_slot = index / kHalf2PerFeatureTile;
          int local_pair =
              index - local_dense_slot * kHalf2PerFeatureTile;
          int dense_slot = tile_row_base + local_dense_slot;
          int output_row = params.problems[1].row_indices[dense_slot];
          int output_offset =
              output_row * output_half2_columns + feature_base / 2 +
              local_pair;
          int residual_offset =
              dense_slot * output_half2_columns + feature_base / 2 +
              local_pair;
          full_output[output_offset] = __hadd2(
              full_output[output_offset], residual_output[residual_offset]);
        }
        __syncthreads();
        if (thread_idx == 0) {
          counter.fetch_add(1, cuda::memory_order_release);
        }
      } else {
        if (thread_idx == 0) {
          counter.fetch_add(1, cuda::memory_order_release);
          while (counter.load(cuda::memory_order_acquire) < completed_tiles) {
            __nanosleep(64);
          }
        }
        __syncthreads();

        constexpr int kRowsPerWork = kThreadCount / 32;
        int full_rows = params.problems[0].rows;
        int row_tiles = (full_rows + kRowsPerWork - 1) / kRowsPerWork;
        int first_work_row = tile_row_base / kRowsPerWork;
        int last_work_row =
            (tile_row_base + tile_rows + kRowsPerWork - 1) /
            kRowsPerWork;
        for (int work_row = first_work_row; work_row < last_work_row;
             ++work_row) {
          run_fused_qkv_postop<false>(
              params, feature_tile * row_tiles + work_row);
        }
        __syncthreads();
        if (thread_idx == 0) {
          int previous = counter.fetch_add(1, cuda::memory_order_acq_rel);
          if (previous + 1 == completed_tiles + params.full_row_tiles) {
            counter.store(0, cuda::memory_order_release);
          }
        }
      }
      __syncthreads();
    } else if constexpr (FinalizeResidualAdd) {
      if (thread_idx == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> counter(
            params.feature_counters[feature_tile]);
        int previous = counter.fetch_add(1, cuda::memory_order_acq_rel);
        route_rows[0] =
            previous + 1 == params.full_row_tiles + params.residual_row_tiles;
      }
      __syncthreads();
      if (route_rows[0]) {
        constexpr int kHalf2PerFeatureTile =
            FullThreadblockShape::kM / 2;
        int feature_base = feature_tile * FullThreadblockShape::kM;
        int dense_count = params.problems[1].rows;
        __half2 *full_output = reinterpret_cast<__half2 *>(
            params.full_output_op.op_1.output);
        const __half2 *residual_output = reinterpret_cast<const __half2 *>(
            params.residual_output_op.op_1.output);
        int output_half2_columns = params.N / 2;
        int total_pairs = dense_count * kHalf2PerFeatureTile;
        for (int index = thread_idx; index < total_pairs;
             index += kThreadCount) {
          int dense_slot = index / kHalf2PerFeatureTile;
          int local_pair = index - dense_slot * kHalf2PerFeatureTile;
          int output_row = params.problems[1].row_indices[dense_slot];
          int output_offset =
              output_row * output_half2_columns + feature_base / 2 +
              local_pair;
          int residual_offset =
              dense_slot * output_half2_columns + feature_base / 2 +
              local_pair;
          full_output[output_offset] = __hadd2(
              full_output[output_offset], residual_output[residual_offset]);
        }
        if (thread_idx == 0) {
          cuda::atomic_ref<int, cuda::thread_scope_device> counter(
              params.feature_counters[feature_tile]);
          counter.store(0, cuda::memory_order_release);
        }
      }
      __syncthreads();
    }
  }

  CUTLASS_DEVICE void fused_qkv_grid_barrier(Params const &params) {
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      cuda::atomic_ref<int, cuda::thread_scope_device> sense(
          params.grid_barrier[1]);
      int target = sense.load(cuda::memory_order_acquire) ^ 1;
      cuda::atomic_ref<int, cuda::thread_scope_device> arrivals(
          params.grid_barrier[0]);
      int previous = arrivals.fetch_add(1, cuda::memory_order_acq_rel);
      if (previous + 1 == int(gridDim.x)) {
        arrivals.store(0, cuda::memory_order_relaxed);
        sense.store(target, cuda::memory_order_release);
      } else {
        while (sense.load(cuda::memory_order_acquire) != target) {
          __nanosleep(64);
        }
      }
    }
    __syncthreads();
  }

  template <bool AddResidual = true>
  CUTLASS_DEVICE void run_fused_qkv_postop(
      Params const &params, int work) {
    constexpr int kHeadDim = 128;
    constexpr int kFeatureTile = 256;
    constexpr int kRowsPerWork = kThreadCount / 32;
    int full_rows = params.problems[0].rows;
    int dense_count = params.problems[1].rows;
    int row_tiles = (full_rows + kRowsPerWork - 1) / kRowsPerWork;
    int feature_tile = work / row_tiles;
    int row_tile = work - feature_tile * row_tiles;
    int row = row_tile * kRowsPerWork + (threadIdx.x >> 5);
    if (row >= full_rows) {
      return;
    }

    __half *qkv = reinterpret_cast<__half *>(
        params.full_output_op.op_1.output);
    const __half *residual = reinterpret_cast<const __half *>(
        params.residual_output_op.op_1.output);
    const __half *q_weight =
        reinterpret_cast<const __half *>(params.q_weight);
    const __half *k_weight =
        reinterpret_cast<const __half *>(params.k_weight);
    const __half *rope_cache =
        reinterpret_cast<const __half *>(params.cos_sin_cache);
    int lane = threadIdx.x & 31;
    int output_size = params.q_size + 2 * params.kv_size;
    int q_heads = params.q_size / kHeadDim;
    int kv_heads = params.kv_size / kHeadDim;
    int normalized_heads = q_heads + kv_heads;
    int feature_base = feature_tile * kFeatureTile;
    int first_head = feature_base / kHeadDim;
    int dense_slot = -1;
    bool routed_dense = false;
    if constexpr (AddResidual) {
      dense_slot = params.dense_slot_by_row[row];
      routed_dense = dense_slot >= 0 && dense_slot < dense_count;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int local_head = 0; local_head < 2; ++local_head) {
      int head = first_head + local_head;
      int head_offset = feature_base + local_head * kHeadDim;
      if (head_offset >= output_size) {
        continue;
      }
      bool q_or_k = head < normalized_heads;
      if (!q_or_k) {
        if constexpr (AddResidual) {
          if (routed_dense) {
            CUTLASS_PRAGMA_UNROLL
            for (int chunk = 0; chunk < 4; ++chunk) {
              int dim = lane + chunk * 32;
              int feature = head_offset + dim;
              int output_offset = row * output_size + feature;
              int residual_offset = dense_slot * output_size + feature;
              qkv[output_offset] =
                  __hadd(qkv[output_offset], residual[residual_offset]);
            }
          }
        }
        continue;
      }

      const __half *scale = head < q_heads ? q_weight : k_weight;
      float values[4];
      float sum = 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int chunk = 0; chunk < 4; ++chunk) {
        int dim = lane + chunk * 32;
        int feature = head_offset + dim;
        int output_offset = row * output_size + feature;
        __half value = qkv[output_offset];
        if constexpr (AddResidual) {
          if (routed_dense) {
            int residual_offset = dense_slot * output_size + feature;
            value = __hadd(value, residual[residual_offset]);
          }
        }
        values[chunk] = __half2float(value);
        if (params.normalize_qk) {
          sum += values[chunk] * values[chunk];
        }
      }
      if (params.normalize_qk) {
        for (int delta = 16; delta > 0; delta >>= 1) {
          sum += __shfl_down_sync(0xffffffff, sum, delta);
        }
        float inverse_rms = lane == 0
                                ? rsqrtf(sum / float(kHeadDim) +
                                         params.epsilon)
                                : 0.0f;
        inverse_rms = __shfl_sync(0xffffffff, inverse_rms, 0);
        CUTLASS_PRAGMA_UNROLL
        for (int chunk = 0; chunk < 4; ++chunk) {
          int dim = lane + chunk * 32;
          values[chunk] *= inverse_rms * __half2float(scale[dim]);
        }
      }

      int64_t position = params.position_ids[row];
      int cache_offset = static_cast<int>(position) * params.rotary_dim;
      CUTLASS_PRAGMA_UNROLL
      for (int chunk = 0; chunk < 4; ++chunk) {
        int dim = lane + chunk * 32;
        float value = values[chunk];
        if (dim < params.rotary_dim) {
          int pair_dim;
          int cache_dim;
          bool subtract_pair;
          if (params.is_neox) {
            int half_rotary = params.rotary_dim / 2;
            pair_dim = dim < half_rotary ? dim + half_rotary
                                         : dim - half_rotary;
            cache_dim = dim % half_rotary;
            subtract_pair = dim < half_rotary;
          } else {
            pair_dim = dim ^ 1;
            cache_dim = dim / 2;
            subtract_pair = (dim & 1) == 0;
          }
          int pair_chunk = pair_dim / 32;
          int pair_lane = pair_dim & 31;
          float pair = __shfl_sync(
              0xffffffff, values[pair_chunk], pair_lane);
          float cosine = __half2float(
              rope_cache[cache_offset + cache_dim]);
          float sine = __half2float(
              rope_cache[cache_offset + params.rotary_dim / 2 + cache_dim]);
          value = subtract_pair ? value * cosine - pair * sine
                                : value * cosine + pair * sine;
        }
        int output_offset = row * output_size + head_offset + dim;
        qkv[output_offset] = __float2half_rn(value);
      }
    }
  }

  CUTLASS_DEVICE void run_residual_work(
      Params const &params, int tile, SharedStorage &shared_storage) {
    if constexpr (SelfContainedResidualCorrection) {
      Problem base_problem = params.problems[1];
      base_problem.values = params.problems[0].values;
      base_problem.metadata = params.problems[0].metadata;
      run_tile<ResidualDeviceGemm, true, true>(
          params, base_problem, tile, params.K, params.N,
          params.residual_base_output_op, shared_storage.gemm.residual,
          shared_storage.route_rows);
    }
    run_tile<ResidualDeviceGemm, GatherResidualRows, true>(
        params, params.problems[1], tile, params.K, params.N,
        params.residual_output_op, shared_storage.gemm.residual,
        shared_storage.route_rows);
  }

  CUTLASS_DEVICE void operator()(Params const &params,
                                 SharedStorage &shared_storage) {
    if constexpr (LastFullTileResidual) {
      int residual_tiles = params.total_tiles - params.first_problem_tiles;
      for (int tile = blockIdx.x; tile < params.first_problem_tiles;
           tile += gridDim.x) {
        run_tile<FullDeviceGemm, false, false>(
            params, params.problems[0], tile, params.K, params.N,
            params.full_output_op, shared_storage.gemm.full,
            shared_storage.route_rows);
        if (shared_storage.route_rows[0]) {
          int feature_tile = tile / params.full_row_tiles;
          for (int residual_row_tile = 0;
               residual_row_tile < params.residual_row_tiles;
               ++residual_row_tile) {
            int residual_tile =
                feature_tile * params.residual_row_tiles + residual_row_tile;
            if (residual_tile < residual_tiles) {
              run_residual_work(params, residual_tile, shared_storage);
            }
          }
        }
      }
    } else if (params.interleaved_schedule == 2) {
      for (int tile = blockIdx.x; tile < params.first_problem_tiles;
           tile += gridDim.x) {
        run_tile<FullDeviceGemm, false, false>(
            params, params.problems[0], tile, params.K, params.N,
            params.full_output_op, shared_storage.gemm.full,
            shared_storage.route_rows);
      }
      int residual_tiles = params.total_tiles - params.first_problem_tiles;
      for (int tile = blockIdx.x; tile < residual_tiles;
           tile += gridDim.x) {
        run_residual_work(params, tile, shared_storage);
      }
    } else if (params.interleaved_schedule == 1) {
      int residual_tiles = params.total_tiles - params.first_problem_tiles;
      for (int work = blockIdx.x; work < params.total_tiles;
           work += gridDim.x) {
        int residual_before = int(
            (int64_t(work) * residual_tiles) / params.total_tiles);
        int residual_after = int(
            (int64_t(work + 1) * residual_tiles) / params.total_tiles);
        if (residual_after > residual_before) {
          run_residual_work(params, residual_after - 1, shared_storage);
        } else {
          run_tile<FullDeviceGemm, false, false>(
              params, params.problems[0], work - residual_before, params.K, params.N,
              params.full_output_op, shared_storage.gemm.full,
              shared_storage.route_rows);
        }
      }
    } else if (blockIdx.x < params.full_worker_blocks) {
      for (int tile = blockIdx.x; tile < params.first_problem_tiles;
           tile += params.full_worker_blocks) {
        run_tile<FullDeviceGemm, false, false>(
            params, params.problems[0], tile, params.K, params.N,
            params.full_output_op, shared_storage.gemm.full,
            shared_storage.route_rows);
      }
    } else {
      int worker = blockIdx.x - params.full_worker_blocks;
      int residual_tiles = params.total_tiles - params.first_problem_tiles;
      for (int tile = worker; tile < residual_tiles;
           tile += params.residual_worker_blocks) {
        run_residual_work(params, tile, shared_storage);
      }
    }

    if constexpr (FusedQkvPostop) {
      fused_qkv_grid_barrier(params);
      constexpr int kRowsPerWork = kThreadCount / 32;
      int row_tiles =
          (params.problems[0].rows + kRowsPerWork - 1) / kRowsPerWork;
      int feature_tiles = (params.N + 255) / 256;
      int postop_work = feature_tiles * row_tiles;
      for (int work = blockIdx.x; work < postop_work; work += gridDim.x) {
        run_fused_qkv_postop(params, work);
      }
    }
  }
};

// Execute exact mixed-row Gate/Up and Down projections in one resident grid.
// Both stages reuse the paired W24/R24 visitor. The grid-wide barrier is safe
// only because the launch is capped to the composite kernel's resident CTA
// capacity on the host.
template <typename GateFullDeviceGemm, typename GateResidualDeviceGemm,
          typename DownFullDeviceGemm, typename DownResidualDeviceGemm>
struct Sparse24FusedMixedMlpKernel {
  using GateKernel = Sparse24PairedPersistentVisitorKernel<
      GateFullDeviceGemm, GateResidualDeviceGemm, true, true>;
  using DownKernel = Sparse24PairedPersistentVisitorKernel<
      DownFullDeviceGemm, DownResidualDeviceGemm, true, true>;
  static constexpr int kThreadCount = GateKernel::kThreadCount;

  static_assert(kThreadCount == DownKernel::kThreadCount,
                "fused mixed MLP stages require equal CTA sizes");

  struct Params {
    typename GateKernel::Params gate;
    typename DownKernel::Params down;
    int *grid_barrier = nullptr;
  };

  union SharedStorage {
    typename GateKernel::SharedStorage gate;
    typename DownKernel::SharedStorage down;
  };

  CUTLASS_DEVICE void stage_barrier(Params const &params) {
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      cuda::atomic_ref<int, cuda::thread_scope_device> sense(
          params.grid_barrier[1]);
      int target = sense.load(cuda::memory_order_acquire) ^ 1;
      cuda::atomic_ref<int, cuda::thread_scope_device> arrivals(
          params.grid_barrier[0]);
      int previous = arrivals.fetch_add(1, cuda::memory_order_acq_rel);
      if (previous + 1 == int(gridDim.x)) {
        arrivals.store(0, cuda::memory_order_relaxed);
        sense.store(target, cuda::memory_order_release);
      } else {
        while (sense.load(cuda::memory_order_acquire) != target) {
          __nanosleep(64);
        }
      }
    }
    __syncthreads();
  }

  CUTLASS_DEVICE void operator()(Params const &params,
                                 SharedStorage &shared_storage) {
    GateKernel{}(params.gate, shared_storage.gate);
    stage_barrier(params);
    DownKernel{}(params.down, shared_storage.down);
  }
};

// Pipeline a sparse Gate/SwiGLU producer and dense Down consumer in one
// resident grid. Every resident CTA first contributes Gate tiles, then moves
// directly to Down tiles. This keeps the full grid productive during Gate
// instead of parking a fixed set of Down-only CTAs on row counters. Down work
// still waits only on its matching row, so fast Gate workers can overlap the
// producer tail without requiring a grid-wide barrier.
template <typename GateDeviceGemm, typename DownDeviceGemm,
          bool SparseDown = false, bool DynamicDownOwners = false,
          bool GlobalStageBarrier = false>
struct Sparse24GateDenseDownPersistentKernel {
  using GateBaseKernel = typename GateDeviceGemm::GemmKernel;
  using DownBaseKernel = typename DownDeviceGemm::GemmKernel;
  using GateMma = typename GateBaseKernel::Mma;
  using DownMma = typename DownBaseKernel::Mma;
  using GateShape = typename GateMma::Shape;
  using DownShape = typename DownMma::Shape;
  using ElementE = typename GateMma::ElementE;
  static constexpr int kThreadCount = GateBaseKernel::kThreadCount;
  static constexpr int kSparse = GateBaseKernel::kSparse;

  static_assert(kThreadCount == DownBaseKernel::kThreadCount,
                "Gate and Down pipeline require equal CTA sizes");
  static_assert(DownShape::kN % GateShape::kN == 0,
                "Down row tiles must group complete Gate row tiles");
  static constexpr int kGateRowsPerDown =
      DownShape::kN / GateShape::kN;
  static_assert(!DynamicDownOwners || kGateRowsPerDown == 1,
                "dynamic Down ownership requires matching Gate/Down row tiles");
  static_assert(!DynamicDownOwners || !GlobalStageBarrier,
                "dynamic ownership and global stage barriers are separate modes");

  struct Params {
    const Element *x = nullptr;
    const Element *gate_values = nullptr;
    ElementE *gate_metadata = nullptr;
    Element *hidden = nullptr;
    const Element *down_weight = nullptr;
    ElementE *down_metadata = nullptr;
    Element *output = nullptr;
    int rows = 0;
    int model_width = 0;
    int intermediate_size = 0;
    int gate_output_size = 0;
    int gate_feature_tiles = 0;
    int down_feature_tiles = 0;
    int gate_row_tiles = 0;
    int down_row_tiles = 0;
    int gate_tiles = 0;
    int down_tiles = 0;
    int gate_workers = 0;
    int down_workers = 0;
    int stage_mode = 0;
    int *row_counters = nullptr;
    typename GateBaseKernel::FusionCallbacks::Params gate_output_op{};
    typename DownBaseKernel::FusionCallbacks::Params down_output_op{};
  };

  union GemmSharedStorage {
    Sparse24PersistentProblemSharedStorage<GateDeviceGemm> gate;
    Sparse24PersistentProblemSharedStorage<DownDeviceGemm> down;
  };

  struct SharedStorage {
    GemmSharedStorage gemm;
    int down_owner;
  };

  CUTLASS_DEVICE int gate_tiles_for_down_row(
      Params const &params, int down_row_tile) {
    int first_gate_row = down_row_tile * kGateRowsPerDown;
    int gate_rows = params.gate_row_tiles - first_gate_row;
    gate_rows = gate_rows < kGateRowsPerDown ? gate_rows
                                             : kGateRowsPerDown;
    return gate_rows * params.gate_feature_tiles;
  }

  CUTLASS_DEVICE void stage_barrier(Params const &params) {
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      cuda::atomic_ref<int, cuda::thread_scope_device> sense(
          params.row_counters[1]);
      int target = sense.load(cuda::memory_order_acquire) ^ 1;
      cuda::atomic_ref<int, cuda::thread_scope_device> arrivals(
          params.row_counters[0]);
      int previous = arrivals.fetch_add(1, cuda::memory_order_acq_rel);
      if (previous + 1 == int(gridDim.x)) {
        arrivals.store(0, cuda::memory_order_relaxed);
        sense.store(target, cuda::memory_order_release);
      } else {
        while (sense.load(cuda::memory_order_acquire) != target) {
          __nanosleep(64);
        }
      }
    }
    __syncthreads();
  }

  CUTLASS_DEVICE int run_gate_tile(
      Params const &params, int local_tile,
      Sparse24PersistentProblemSharedStorage<GateDeviceGemm> &storage,
      bool publish_row, int *down_owner) {
    using Epilogue = typename GateBaseKernel::Epilogue;
    static constexpr int kElementsPerElementE =
        GateBaseKernel::kElementsPerElementE;
    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx & 31;
    int row_tile = local_tile / params.gate_feature_tiles;
    int feature_tile = local_tile - row_tile * params.gate_feature_tiles;
    if (row_tile >= params.gate_row_tiles) {
      return -1;
    }

    cutlass::gemm::GemmCoord offset(feature_tile * GateShape::kM,
                                    row_tile * GateShape::kN, 0);
    int sparse_k = params.model_width / kSparse;
    int columns_e = sparse_k / kElementsPerElementE;
    using LayoutA = typename GateMma::IteratorA::Layout;
    using LayoutB = typename GateMma::IteratorB::Layout;
    using LayoutE = typename GateMma::IteratorE::Layout;
    typename GateMma::IteratorA::Params params_A{LayoutA(sparse_k)};
    typename GateMma::IteratorB::Params params_B{LayoutB(params.model_width)};
    LayoutE layout_e = LayoutE::packed(
        {params.gate_output_size, columns_e});
    typename GateMma::IteratorE::Params params_E{layout_e};
    typename GateMma::IteratorA iterator_A(
        params_A, const_cast<Element *>(params.gate_values),
        {params.gate_output_size, sparse_k}, thread_idx,
        {offset.m(), 0});
    typename GateMma::IteratorB iterator_B(
        params_B, const_cast<Element *>(params.x),
        {params.model_width, params.rows}, thread_idx,
        {0, offset.n()});
    typename GateMma::IteratorE iterator_E(
        params_E, params.gate_metadata,
        {params.gate_output_size, columns_e}, thread_idx,
        {offset.m(), 0});
    typename GateMma::FragmentC accumulators;
    accumulators.clear();
    GateMma mma(storage.main_loop, thread_idx, warp_idx, lane_idx);
    int gemm_k_iterations =
        (params.model_width + GateShape::kK - 1) / GateShape::kK;
    __syncthreads();
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
        accumulators);

    cutlass::gemm::GemmCoord tile_index(feature_tile, row_tile, 0);
    auto problem_shape = cute::make_shape(
        int32_t(params.gate_output_size), int32_t(params.rows), int32_t(1));
    Epilogue epilogue(params.gate_output_op, storage.epilogue, thread_idx,
                      warp_idx, lane_idx);
    epilogue(accumulators, tile_index, problem_shape, thread_idx);
    __syncthreads();
    if (thread_idx == 0) {
      *down_owner = -1;
      if (publish_row) {
        int down_row_tile = row_tile / kGateRowsPerDown;
        cuda::atomic_ref<int, cuda::thread_scope_device> counter(
            params.row_counters[down_row_tile]);
        counter.fetch_add(1, cuda::memory_order_release);
        if constexpr (DynamicDownOwners) {
          int first_owner_feature =
              params.gate_feature_tiles - params.down_feature_tiles;
          if (feature_tile >= first_owner_feature) {
            *down_owner =
                down_row_tile * params.down_feature_tiles +
                feature_tile - first_owner_feature;
          }
        }
      }
    }
    __syncthreads();
    return *down_owner;
  }

  CUTLASS_DEVICE void run_down_tile(
      Params const &params, int local_tile,
      Sparse24PersistentProblemSharedStorage<DownDeviceGemm> &storage,
      bool wait_for_gate, bool publish_row) {
    using Epilogue = typename DownBaseKernel::Epilogue;
    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx & 31;
    int row_tile = local_tile / params.down_feature_tiles;
    int feature_tile = local_tile - row_tile * params.down_feature_tiles;
    if (row_tile >= params.down_row_tiles) {
      return;
    }
    int expected_gate_tiles =
        wait_for_gate ? gate_tiles_for_down_row(params, row_tile) : 0;
    if (wait_for_gate && thread_idx == 0) {
      cuda::atomic_ref<int, cuda::thread_scope_device> counter(
          params.row_counters[row_tile]);
      while (counter.load(cuda::memory_order_acquire) <
             expected_gate_tiles) {
        __nanosleep(64);
      }
    }
    __syncthreads();

    cutlass::gemm::GemmCoord offset(feature_tile * DownShape::kM,
                                    row_tile * DownShape::kN, 0);
    using LayoutB = typename DownMma::IteratorB::Layout;
    typename DownMma::IteratorB::Params params_B{
        LayoutB(params.intermediate_size)};
    typename DownMma::IteratorB iterator_B(
        params_B, params.hidden,
        {params.intermediate_size, params.rows}, thread_idx,
        {0, offset.n()});
    typename DownMma::FragmentC accumulators;
    accumulators.clear();
    DownMma mma(storage.main_loop, thread_idx, warp_idx, lane_idx);
    int gemm_k_iterations =
        (params.intermediate_size + DownShape::kK - 1) / DownShape::kK;
    if constexpr (SparseDown) {
      static constexpr int kElementsPerElementE =
          DownBaseKernel::kElementsPerElementE;
      int sparse_k = params.intermediate_size / kSparse;
      int columns_e = sparse_k / kElementsPerElementE;
      using LayoutA = typename DownMma::IteratorA::Layout;
      using LayoutE = typename DownMma::IteratorE::Layout;
      typename DownMma::IteratorA::Params params_A{LayoutA(sparse_k)};
      LayoutE layout_e = LayoutE::packed(
          {params.model_width, columns_e});
      typename DownMma::IteratorE::Params params_E{layout_e};
      typename DownMma::IteratorA iterator_A(
          params_A, const_cast<Element *>(params.down_weight),
          {params.model_width, sparse_k}, thread_idx, {offset.m(), 0});
      typename DownMma::IteratorE iterator_E(
          params_E, params.down_metadata,
          {params.model_width, columns_e}, thread_idx, {offset.m(), 0});
      __syncthreads();
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B,
          iterator_E, accumulators);
    } else {
      using LayoutA = typename DownMma::IteratorA::Layout;
      typename DownMma::IteratorA::Params params_A{
          LayoutA(params.intermediate_size)};
      typename DownMma::IteratorA iterator_A(
          params_A, const_cast<Element *>(params.down_weight),
          {params.model_width, params.intermediate_size}, thread_idx,
          {offset.m(), 0});
      __syncthreads();
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B,
          accumulators);
    }

    cutlass::gemm::GemmCoord tile_index(feature_tile, row_tile, 0);
    auto problem_shape = cute::make_shape(
        int32_t(params.model_width), int32_t(params.rows), int32_t(1));
    Epilogue epilogue(params.down_output_op, storage.epilogue, thread_idx,
                      warp_idx, lane_idx);
    epilogue(accumulators, tile_index, problem_shape, thread_idx);
    __syncthreads();
    if (publish_row && thread_idx == 0) {
      cuda::atomic_ref<int, cuda::thread_scope_device> counter(
          params.row_counters[row_tile]);
      int previous = counter.fetch_add(1, cuda::memory_order_release);
      if (previous + 1 ==
          expected_gate_tiles + params.down_feature_tiles) {
        counter.store(0, cuda::memory_order_release);
      }
    }
    __syncthreads();
  }

  CUTLASS_DEVICE void operator()(Params const &params,
                                 SharedStorage &storage) {
    if constexpr (GlobalStageBarrier) {
      if (params.stage_mode == 0) {
        for (int tile = blockIdx.x; tile < params.gate_tiles;
             tile += gridDim.x) {
          run_gate_tile(params, tile, storage.gemm.gate, false,
                        &storage.down_owner);
        }
        stage_barrier(params);
        for (int tile = blockIdx.x; tile < params.down_tiles;
             tile += gridDim.x) {
          run_down_tile(params, tile, storage.gemm.down, false, false);
        }
        stage_barrier(params);
        return;
      }
    }
    if constexpr (DynamicDownOwners) {
      if (params.stage_mode == 0) {
        for (int tile = blockIdx.x; tile < params.gate_tiles;
             tile += gridDim.x) {
          int down_tile = run_gate_tile(
              params, tile, storage.gemm.gate, true, &storage.down_owner);
          if (down_tile >= 0) {
            run_down_tile(
                params, down_tile, storage.gemm.down, true, true);
          }
        }
        return;
      }
    }
    if (params.stage_mode != 2) {
      bool publish_rows = params.stage_mode == 0;
      for (int tile = blockIdx.x; tile < params.gate_tiles;
           tile += gridDim.x) {
        run_gate_tile(params, tile, storage.gemm.gate, publish_rows,
                      &storage.down_owner);
      }
    }
    if (params.stage_mode != 1) {
      bool wait_for_gate = params.stage_mode == 0;
      for (int tile = blockIdx.x; tile < params.down_tiles;
           tile += gridDim.x) {
        run_down_tile(params, tile, storage.gemm.down, wait_for_gate,
                      wait_for_gate);
      }
    }
  }
};

// CUTLASS 3.9's sparse serial split-K kernel advances packed metadata in
// sparse-value units instead of metadata-element units. Keep its mature
// semaphore/epilogue path, but fix that one coordinate locally.
template <typename BaseKernel>
struct Sparse24FixedSplitKSparseGemmKernel {
  using Mma = typename BaseKernel::Mma;
  using Epilogue = typename BaseKernel::Epilogue;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = typename BaseKernel::ThreadblockSwizzle;
  using Params = typename BaseKernel::Params;
  using SharedStorage = typename BaseKernel::SharedStorage;
  static constexpr int kSparse = BaseKernel::kSparse;
  static constexpr int kElementsPerElementE =
      BaseKernel::kElementsPerElementE;
  static constexpr int kThreadCount = BaseKernel::kThreadCount;

  CUTLASS_DEVICE void operator()(Params const &params,
                                 SharedStorage &shared_storage) {
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }

    cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.k() * params.gemm_k_size / kSparse};
    cutlass::MatrixCoord tb_offset_B{
        threadblock_tile_offset.k() * params.gemm_k_size,
        threadblock_tile_offset.n() * Mma::Shape::kN};
    cutlass::MatrixCoord tb_offset_E{
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.k() * params.gemm_k_size / kSparse /
            kElementsPerElementE};

    int problem_size_k = min(
        params.problem_size.k(),
        (threadblock_tile_offset.k() + 1) * params.gemm_k_size);
    int gemm_k_iterations =
        (problem_size_k - tb_offset_B.row() + Mma::Shape::kK - 1) /
        Mma::Shape::kK;
    int thread_idx = int(threadIdx.x);
    typename Mma::IteratorA iterator_A(
        params.params_A, params.ref_A.data(),
        {params.problem_size.m(), problem_size_k / kSparse}, thread_idx,
        tb_offset_A);
    typename Mma::IteratorB iterator_B(
        params.params_B, params.ref_B.data(),
        {problem_size_k, params.problem_size.n()}, thread_idx, tb_offset_B);
    typename Mma::IteratorE iterator_E(
        params.params_E, params.ref_E.data(),
        {params.problem_size.m(),
         problem_size_k / kSparse / kElementsPerElementE},
        thread_idx, tb_offset_E);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % 32;
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC accumulators;
    accumulators.clear();
    if (gemm_k_iterations > 0) {
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
          accumulators);
    }

    OutputOp output_op(params.output_op);
    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
    cutlass::MatrixCoord threadblock_offset(
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.n() * Mma::Shape::kN);
    int block_idx = threadblock_tile_offset.m() +
                    threadblock_tile_offset.n() *
                        params.grid_tiled_shape.m();
    cutlass::Semaphore semaphore(params.semaphore + block_idx, thread_idx);
    semaphore.fetch();
    output_op.set_k_partition(threadblock_tile_offset.k(),
                              params.grid_tiled_shape.k());

    typename Epilogue::OutputTileIterator iterator_C(
        params.params_C, params.ref_C.data(), params.problem_size.mn(),
        thread_idx, threadblock_offset);
    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D, params.ref_D.data(), params.problem_size.mn(),
        thread_idx, threadblock_offset);
    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    if (threadblock_tile_offset.k()) {
      iterator_C = iterator_D;
    }
    semaphore.wait(threadblock_tile_offset.k());
    __threadfence();
    epilogue(output_op, iterator_D, accumulators, iterator_C);

    int lock =
        params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1
            ? 0
            : threadblock_tile_offset.k() + 1;
    __threadfence();
    semaphore.release(lock);
  }
};

// Preserve serial split-K accumulation, then fold the completed dense-row
// residual directly into a contiguous full Down output. The full GEMM signals
// ready_state[0] from its stream; the last residual CTA resets both state
// words so the same buffers are safe for CUDA Graph replay.
template <typename BaseKernel>
struct Sparse24FixedSplitKIndexedAddKernel {
  using Mma = typename BaseKernel::Mma;
  using Epilogue = typename BaseKernel::Epilogue;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = typename BaseKernel::ThreadblockSwizzle;
  using BaseParams = typename BaseKernel::Params;
  using SharedStorage = typename BaseKernel::SharedStorage;
  static constexpr int kSparse = BaseKernel::kSparse;
  static constexpr int kElementsPerElementE =
      BaseKernel::kElementsPerElementE;
  static constexpr int kThreadCount = BaseKernel::kThreadCount;

  struct Params {
    BaseParams gemm;
    Element *full_output;
    int const *dense_rows;
    int *ready_state;
    int dense_count;
    int full_rows;
    int output_columns;
    int final_ctas;
  };

  CUTLASS_DEVICE void operator()(Params const &params,
                                 SharedStorage &shared_storage) {
    BaseParams const &gemm = params.gemm;
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(gemm.swizzle_log_tile);
    if (gemm.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        gemm.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }

    cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.k() * gemm.gemm_k_size / kSparse};
    cutlass::MatrixCoord tb_offset_B{
        threadblock_tile_offset.k() * gemm.gemm_k_size,
        threadblock_tile_offset.n() * Mma::Shape::kN};
    cutlass::MatrixCoord tb_offset_E{
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.k() * gemm.gemm_k_size / kSparse /
            kElementsPerElementE};

    int problem_size_k = min(
        gemm.problem_size.k(),
        (threadblock_tile_offset.k() + 1) * gemm.gemm_k_size);
    int gemm_k_iterations =
        (problem_size_k - tb_offset_B.row() + Mma::Shape::kK - 1) /
        Mma::Shape::kK;
    int thread_idx = int(threadIdx.x);
    typename Mma::IteratorA iterator_A(
        gemm.params_A, gemm.ref_A.data(),
        {gemm.problem_size.m(), problem_size_k / kSparse}, thread_idx,
        tb_offset_A);
    typename Mma::IteratorB iterator_B(
        gemm.params_B, gemm.ref_B.data(),
        {problem_size_k, gemm.problem_size.n()}, thread_idx, tb_offset_B);
    typename Mma::IteratorE iterator_E(
        gemm.params_E, gemm.ref_E.data(),
        {gemm.problem_size.m(),
         problem_size_k / kSparse / kElementsPerElementE},
        thread_idx, tb_offset_E);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % 32;
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC accumulators;
    accumulators.clear();
    if (gemm_k_iterations > 0) {
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
          accumulators);
    }

    OutputOp output_op(gemm.output_op);
    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(gemm.swizzle_log_tile);
    cutlass::MatrixCoord threadblock_offset(
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.n() * Mma::Shape::kN);
    int block_idx = threadblock_tile_offset.m() +
                    threadblock_tile_offset.n() *
                        gemm.grid_tiled_shape.m();
    cutlass::Semaphore semaphore(gemm.semaphore + block_idx, thread_idx);
    semaphore.fetch();
    output_op.set_k_partition(threadblock_tile_offset.k(),
                              gemm.grid_tiled_shape.k());

    typename Epilogue::OutputTileIterator iterator_C(
        gemm.params_C, gemm.ref_C.data(), gemm.problem_size.mn(), thread_idx,
        threadblock_offset);
    typename Epilogue::OutputTileIterator iterator_D(
        gemm.params_D, gemm.ref_D.data(), gemm.problem_size.mn(), thread_idx,
        threadblock_offset);
    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    if (threadblock_tile_offset.k()) {
      iterator_C = iterator_D;
    }
    semaphore.wait(threadblock_tile_offset.k());
    __threadfence();
    epilogue(output_op, iterator_D, accumulators, iterator_C);

    bool final_partition =
        gemm.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1;
    if (final_partition) {
      __syncthreads();
      if (thread_idx == 0) {
        while (atomicAdd(params.ready_state, 0) == 0) {
          __nanosleep(64);
        }
      }
      __syncthreads();

      int feature_base = threadblock_tile_offset.m() * Mma::Shape::kM;
      int dense_base = threadblock_tile_offset.n() * Mma::Shape::kN;
      int tile_elements = Mma::Shape::kM * Mma::Shape::kN;
      auto *residual = reinterpret_cast<__half *>(gemm.ref_D.data());
      auto *full = reinterpret_cast<__half *>(params.full_output);
      for (int linear = thread_idx; linear < tile_elements;
           linear += kThreadCount) {
        int feature = feature_base + linear / Mma::Shape::kN;
        int dense_slot = dense_base + linear % Mma::Shape::kN;
        if (feature < params.output_columns &&
            dense_slot < params.dense_count) {
          int output_row = params.dense_rows[dense_slot];
          __half value = residual[
              static_cast<int64_t>(feature) * gemm.problem_size.n() +
              dense_slot];
          __half *destination =
              full + static_cast<int64_t>(output_row) *
                         params.output_columns +
              feature;
          *destination = __hadd(*destination, value);
        }
      }
      __syncthreads();
      if (thread_idx == 0) {
        __threadfence();
        int completed = atomicAdd(params.ready_state + 1, 1) + 1;
        if (completed == params.final_ctas) {
          atomicExch(params.ready_state + 1, 0);
          __threadfence();
          atomicExch(params.ready_state, 0);
        }
      }
    }

    int lock = final_partition ? 0 : threadblock_tile_offset.k() + 1;
    __threadfence();
    semaphore.release(lock);
  }
};

__global__ void sparse24_cutlass_signal_ready_kernel(int *ready_state) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __threadfence();
    atomicExch(ready_state, 1);
  }
}

// Execute the common 2:4 weight over every verifier row exactly once, while a
// second worker group applies the complementary 2:4 weight only to routed
// dense rows.  The residual mainloop gathers those rows directly from the
// original activation, so the launch does not materialize a compact input.
// Outputs remain separate CUTLASS-transposed tensors; a following fused QKV
// post-op can add the compact residual while materializing Q/K/V, RMSNorm, and
// RoPE.  This keeps the cold-weight traffic at W24 + R24 instead of rereading
// W24 for the dense route.
template <typename FullDeviceGemm, typename ResidualDeviceGemm>
struct Sparse24PairedGatherResidualKernel {
  using FullBaseKernel = typename FullDeviceGemm::GemmKernel;
  using ResidualBaseKernel = typename ResidualDeviceGemm::GemmKernel;
  using FullMma = typename FullBaseKernel::Mma;
  using ResidualBaseMma = typename ResidualBaseKernel::Mma;
  using ResidualMma = typename Sparse24GatherMma<ResidualBaseMma>::Type;
  using FullThreadblockShape = typename FullMma::Shape;
  using ResidualThreadblockShape = typename ResidualMma::Shape;
  using ElementE = typename FullMma::ElementE;
  static constexpr int kThreadCount = FullBaseKernel::kThreadCount;
  static constexpr int kSparse = FullBaseKernel::kSparse;

  static_assert(kThreadCount == ResidualBaseKernel::kThreadCount,
                "paired gather GEMMs require equal CTA sizes");
  static_assert(kSparse == ResidualBaseKernel::kSparse,
                "paired gather GEMMs require one sparse format");
  static_assert(
      cutlass::platform::is_same<ElementE,
                                 typename ResidualMma::ElementE>::value,
      "paired gather GEMMs require one metadata element type");
  static_assert(
      cutlass::platform::is_same<typename ResidualMma::SharedStorage,
                                 typename ResidualBaseMma::SharedStorage>::value,
      "gathering residual rows must not change MMA shared storage");

  struct Params {
    const Element *x = nullptr;
    const Element *full_values = nullptr;
    ElementE *full_metadata = nullptr;
    Element *full_output = nullptr;
    const Element *residual_values = nullptr;
    ElementE *residual_metadata = nullptr;
    Element *residual_output = nullptr;
    const int *dense_rows = nullptr;
    int full_rows = 0;
    int dense_count = 0;
    int K = 0;
    int N = 0;
    int full_tiles = 0;
    int residual_tiles = 0;
    int full_worker_blocks = 0;
    int residual_worker_blocks = 0;
    int interleaved_schedule = 0;
    typename FullBaseKernel::OutputOp::Params full_output_op{};
    typename ResidualBaseKernel::OutputOp::Params residual_output_op{};
  };

  union GemmSharedStorage {
    Sparse24PersistentProblemSharedStorage<FullDeviceGemm> full;
    Sparse24PersistentProblemSharedStorage<ResidualDeviceGemm> residual;
  };

  struct SharedStorage {
    GemmSharedStorage gemm;
    alignas(16) int route_rows[ResidualThreadblockShape::kN];
  };

  CUTLASS_DEVICE void run_full_tile(
      Params const &params, int local_tile,
      Sparse24PersistentProblemSharedStorage<FullDeviceGemm> &storage) {
    using Epilogue = typename FullBaseKernel::Epilogue;
    using OutputOp = typename FullBaseKernel::OutputOp;
    static constexpr int kElementsPerElementE =
        FullBaseKernel::kElementsPerElementE;
    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx & 31;
    int row_tiles =
        (params.full_rows + FullThreadblockShape::kN - 1) /
        FullThreadblockShape::kN;
    int feature_tile = local_tile / row_tiles;
    int row_tile = local_tile - feature_tile * row_tiles;
    int feature_tiles =
        (params.N + FullThreadblockShape::kM - 1) /
        FullThreadblockShape::kM;
    if (feature_tile >= feature_tiles) {
      return;
    }

    cutlass::gemm::GemmCoord problem_size(params.N, params.full_rows,
                                           params.K);
    cutlass::gemm::GemmCoord offset(
        feature_tile * FullThreadblockShape::kM,
        row_tile * FullThreadblockShape::kN, 0);
    int sparse_k = params.K / kSparse;
    int columns_e = sparse_k / kElementsPerElementE;
    using LayoutA = typename FullMma::IteratorA::Layout;
    using LayoutB = typename FullMma::IteratorB::Layout;
    using LayoutE = typename FullMma::IteratorE::Layout;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;
    typename FullMma::IteratorA::Params params_A{LayoutA(sparse_k)};
    typename FullMma::IteratorB::Params params_B{LayoutB(params.K)};
    LayoutE layout_e = LayoutE::packed({params.N, columns_e});
    typename FullMma::IteratorE::Params params_E{layout_e};
    typename FullMma::IteratorA iterator_A(
        params_A, const_cast<Element *>(params.full_values),
        {params.N, sparse_k}, thread_idx, {offset.m(), 0});
    typename FullMma::IteratorB iterator_B(
        params_B, const_cast<Element *>(params.x),
        {params.K, params.full_rows}, thread_idx, {0, offset.n()});
    typename FullMma::IteratorE iterator_E(
        params_E, params.full_metadata, {params.N, columns_e}, thread_idx,
        {offset.m(), 0});

    typename FullMma::FragmentC accumulators;
    accumulators.clear();
    FullMma mma(storage.main_loop, thread_idx, warp_idx, lane_idx);
    int gemm_k_iterations =
        (params.K + FullThreadblockShape::kK - 1) /
        FullThreadblockShape::kK;
    __syncthreads();
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
        accumulators);

    LayoutC layout_c(params.full_rows);
    typename Epilogue::OutputTileIterator::Params params_c(layout_c);
    typename Epilogue::OutputTileIterator iterator_C(
        params_c, params.full_output, problem_size.mn(), thread_idx,
        offset.mn());
    typename Epilogue::OutputTileIterator iterator_D(
        params_c, params.full_output, problem_size.mn(), thread_idx,
        offset.mn());
    OutputOp output_op(params.full_output_op);
    Epilogue epilogue(storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, accumulators, iterator_C);
    __syncthreads();
  }

  CUTLASS_DEVICE void run_residual_tile(
      Params const &params, int local_tile,
      Sparse24PersistentProblemSharedStorage<ResidualDeviceGemm> &storage,
      int *route_rows) {
    using Epilogue = typename ResidualBaseKernel::Epilogue;
    using OutputOp = typename ResidualBaseKernel::OutputOp;
    static constexpr int kElementsPerElementE =
        ResidualBaseKernel::kElementsPerElementE;
    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx & 31;
    int row_tiles =
        (params.dense_count + ResidualThreadblockShape::kN - 1) /
        ResidualThreadblockShape::kN;
    int feature_tile = local_tile / row_tiles;
    int row_tile = local_tile - feature_tile * row_tiles;
    int feature_tiles =
        (params.N + ResidualThreadblockShape::kM - 1) /
        ResidualThreadblockShape::kM;
    if (feature_tile >= feature_tiles) {
      return;
    }

    cutlass::gemm::GemmCoord problem_size(params.N, params.dense_count,
                                           params.K);
    cutlass::gemm::GemmCoord offset(
        feature_tile * ResidualThreadblockShape::kM,
        row_tile * ResidualThreadblockShape::kN, 0);
    int tile_row_base = offset.n();
    int tile_rows = params.dense_count - tile_row_base;
    tile_rows = tile_rows < ResidualThreadblockShape::kN
                    ? tile_rows
                    : ResidualThreadblockShape::kN;
    for (int local_row = thread_idx; local_row < tile_rows;
         local_row += kThreadCount) {
      route_rows[local_row] =
          params.dense_rows[tile_row_base + local_row];
    }

    int sparse_k = params.K / kSparse;
    int columns_e = sparse_k / kElementsPerElementE;
    using LayoutA = typename ResidualMma::IteratorA::Layout;
    using LayoutB = typename ResidualMma::IteratorB::Layout;
    using LayoutE = typename ResidualMma::IteratorE::Layout;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;
    typename ResidualMma::IteratorA::Params params_A{LayoutA(sparse_k)};
    typename ResidualMma::IteratorB::Params params_B{LayoutB(params.K)};
    LayoutE layout_e = LayoutE::packed({params.N, columns_e});
    typename ResidualMma::IteratorE::Params params_E{layout_e};
    typename ResidualMma::IteratorA iterator_A(
        params_A, const_cast<Element *>(params.residual_values),
        {params.N, sparse_k}, thread_idx, {offset.m(), 0});
    typename ResidualMma::IteratorB iterator_B(
        params_B, const_cast<Element *>(params.x), {params.K, tile_rows},
        thread_idx, {0, 0}, route_rows);
    typename ResidualMma::IteratorE iterator_E(
        params_E, params.residual_metadata, {params.N, columns_e}, thread_idx,
        {offset.m(), 0});

    typename ResidualMma::FragmentC accumulators;
    accumulators.clear();
    ResidualMma mma(storage.main_loop, thread_idx, warp_idx, lane_idx);
    int gemm_k_iterations =
        (params.K + ResidualThreadblockShape::kK - 1) /
        ResidualThreadblockShape::kK;
    __syncthreads();
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
        accumulators);

    LayoutC layout_c(params.dense_count);
    typename Epilogue::OutputTileIterator::Params params_c(layout_c);
    typename Epilogue::OutputTileIterator iterator_C(
        params_c, params.residual_output, problem_size.mn(), thread_idx,
        offset.mn());
    typename Epilogue::OutputTileIterator iterator_D(
        params_c, params.residual_output, problem_size.mn(), thread_idx,
        offset.mn());
    OutputOp output_op(params.residual_output_op);
    Epilogue epilogue(storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, accumulators, iterator_C);
    __syncthreads();
  }

  CUTLASS_DEVICE void operator()(Params const &params,
                                 SharedStorage &storage) {
    if (params.interleaved_schedule) {
      int total_tiles = params.full_tiles + params.residual_tiles;
      for (int work = blockIdx.x; work < total_tiles; work += gridDim.x) {
        int residual_before =
            int((int64_t(work) * params.residual_tiles) / total_tiles);
        int residual_after =
            int((int64_t(work + 1) * params.residual_tiles) / total_tiles);
        if (residual_after > residual_before) {
          run_residual_tile(params, residual_after - 1,
                            storage.gemm.residual, storage.route_rows);
        } else {
          run_full_tile(params, work - residual_before, storage.gemm.full);
        }
      }
      return;
    }

    if (blockIdx.x < params.full_worker_blocks) {
      for (int tile = blockIdx.x; tile < params.full_tiles;
           tile += params.full_worker_blocks) {
        run_full_tile(params, tile, storage.gemm.full);
      }
      return;
    }
    int worker = blockIdx.x - params.full_worker_blocks;
    for (int tile = worker; tile < params.residual_tiles;
         tile += params.residual_worker_blocks) {
      run_residual_tile(params, tile, storage.gemm.residual,
                        storage.route_rows);
    }
  }
};

// Dense counterpart of Sparse24GatherMma. The logical B column is a compact
// route slot, while the gather iterator reads the corresponding source row
// directly from the original verifier activation.
template <typename BaseMma>
struct DenseGatherMma {
  using BaseIteratorB = typename BaseMma::IteratorB;
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<BaseMma::Shape::kK, BaseMma::Shape::kN>,
          typename BaseIteratorB::Element, typename BaseIteratorB::Layout, 0,
          typename BaseIteratorB::ThreadMap,
          typename BaseIteratorB::AccessType, true>;
  using Type = cutlass::gemm::threadblock::MmaMultistage<
      typename BaseMma::Shape, typename BaseMma::IteratorA,
      typename BaseMma::SmemIteratorA, BaseMma::kCacheOpA, IteratorB,
      typename BaseMma::SmemIteratorB, BaseMma::kCacheOpB,
      typename BaseMma::ElementC, typename BaseMma::LayoutC,
      typename BaseMma::Policy, BaseMma::Detail::kStages>;
};

// SwiGLU stores consume 128 gate channels followed by their matching 128 up
// channels. Gather A maps the original [all gate, all up] dense weight into
// that logical order, while Gather B independently selects routed token rows.
template <typename BaseMma>
struct DenseGatherABMma {
  using BaseIteratorA = typename BaseMma::IteratorA;
  using BaseIteratorB = typename BaseMma::IteratorB;
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          typename BaseIteratorA::Shape, typename BaseIteratorA::Element,
          typename BaseIteratorA::Layout, BaseIteratorA::kAdvanceRank,
          typename BaseIteratorA::ThreadMap,
          typename BaseIteratorA::AccessType, true>;
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          typename BaseIteratorB::Shape, typename BaseIteratorB::Element,
          typename BaseIteratorB::Layout, BaseIteratorB::kAdvanceRank,
          typename BaseIteratorB::ThreadMap,
          typename BaseIteratorB::AccessType, true>;
  using Type = cutlass::gemm::threadblock::MmaMultistage<
      typename BaseMma::Shape, IteratorA, typename BaseMma::SmemIteratorA,
      BaseMma::kCacheOpA, IteratorB, typename BaseMma::SmemIteratorB,
      BaseMma::kCacheOpB, typename BaseMma::ElementC,
      typename BaseMma::LayoutC, typename BaseMma::Policy,
      BaseMma::Detail::kStages>;
};

template <typename BaseMma, bool GatherA>
struct DenseRoutedMmaSelector {
  using Type = typename DenseGatherMma<BaseMma>::Type;
};

template <typename BaseMma>
struct DenseRoutedMmaSelector<BaseMma, true> {
  using Type = typename DenseGatherABMma<BaseMma>::Type;
};

// Exact row routing without materializing compact activations. Sparse rows
// execute W24 once. Dense rows execute W24 and its complementary 2:4 residual
// in the same CTA and accumulate both before the indexed epilogue. The two
// worker groups write disjoint output rows, so no cross-CTA synchronization is
// needed.
template <typename SparseDeviceGemm, typename DenseDeviceGemm>
struct Sparse24RoutedExactVisitorKernel {
  using SparseBaseKernel = typename SparseDeviceGemm::GemmKernel;
  using DenseBaseKernel = typename DenseDeviceGemm::GemmKernel;
  using SparseBaseMma = typename SparseBaseKernel::Mma;
  using DenseBaseMma = typename DenseBaseKernel::Mma;
  using SparseMma = typename Sparse24GatherMma<SparseBaseMma>::Type;
  using DenseMma = typename Sparse24GatherMma<DenseBaseMma>::Type;
  using ElementE = typename SparseBaseMma::ElementE;
  static constexpr int kThreadCount = SparseBaseKernel::kThreadCount;
  static constexpr int kSparse = SparseBaseKernel::kSparse;

  static_assert(kThreadCount == DenseBaseKernel::kThreadCount,
                "routed exact GEMMs require equal CTA sizes");
  static_assert(kSparse == DenseBaseKernel::kSparse,
                "routed exact GEMMs require one sparse format");
  static_assert(
      cutlass::platform::is_same<typename SparseMma::SharedStorage,
                                 typename SparseBaseMma::SharedStorage>::value,
      "gathering B must not change sparse MMA shared storage");
  static_assert(
      cutlass::platform::is_same<typename DenseMma::SharedStorage,
                                 typename DenseBaseMma::SharedStorage>::value,
      "gathering B must not change dense-route MMA shared storage");

  struct Problem {
    const int *row_indices = nullptr;
    int rows = 0;
  };

  struct Params {
    const Element *x = nullptr;
    const Element *full_values = nullptr;
    ElementE *full_metadata = nullptr;
    const Element *residual_values = nullptr;
    ElementE *residual_metadata = nullptr;
    Problem sparse_problem;
    Problem dense_problem;
    int K = 0;
    int N = 0;
    int sparse_tiles = 0;
    int dense_tiles = 0;
    int sparse_workers = 0;
    int dense_workers = 0;
    typename SparseBaseKernel::FusionCallbacks::Params sparse_output_op{};
    typename DenseBaseKernel::FusionCallbacks::Params dense_output_op{};
  };

  static constexpr int kMaxRouteRows =
      SparseMma::Shape::kN > DenseMma::Shape::kN
          ? SparseMma::Shape::kN
          : DenseMma::Shape::kN;

  struct SharedStorage {
    union {
      Sparse24PersistentProblemSharedStorage<SparseDeviceGemm> sparse;
      Sparse24PersistentProblemSharedStorage<DenseDeviceGemm> dense;
    } gemm;
    alignas(16) int route_rows[kMaxRouteRows];
  };

  template <typename DeviceGemm, bool AddResidual>
  CUTLASS_DEVICE void run_tile(
      Params const &params, Problem const &problem, int local_tile,
      typename DeviceGemm::GemmKernel::FusionCallbacks::Params const
          &output_op_params,
      Sparse24PersistentProblemSharedStorage<DeviceGemm> &shared_storage,
      int *route_rows) {
    using BaseKernel = typename DeviceGemm::GemmKernel;
    using BaseMma = typename BaseKernel::Mma;
    using Mma = typename Sparse24GatherMma<BaseMma>::Type;
    using Epilogue = typename BaseKernel::Epilogue;
    using ThreadblockShape = typename Mma::Shape;
    static constexpr int kElementsPerElementE =
        BaseKernel::kElementsPerElementE;

    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % 32;
    int feature_tiles =
        (params.N + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
    int row_tiles =
        (problem.rows + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
    int feature_tile = local_tile / row_tiles;
    int row_tile = local_tile - feature_tile * row_tiles;
    if (feature_tile >= feature_tiles) {
      return;
    }

    cutlass::gemm::GemmCoord threadblock_offset(
        feature_tile * ThreadblockShape::kM,
        row_tile * ThreadblockShape::kN, 0);
    int tile_row_base = threadblock_offset.n();
    int tile_rows = problem.rows - tile_row_base;
    tile_rows = tile_rows < ThreadblockShape::kN
                    ? tile_rows
                    : ThreadblockShape::kN;
    for (int local_row = thread_idx; local_row < tile_rows;
         local_row += kThreadCount) {
      route_rows[local_row] = problem.row_indices[tile_row_base + local_row];
    }
    int sparse_k = params.K / kSparse;
    int columns_e = params.K / kSparse / kElementsPerElementE;

    using LayoutA = typename Mma::IteratorA::Layout;
    using LayoutB = typename Mma::IteratorB::Layout;
    using LayoutE = typename Mma::IteratorE::Layout;
    typename Mma::IteratorA::Params params_A{LayoutA(sparse_k)};
    typename Mma::IteratorB::Params params_B{LayoutB(params.K)};
    LayoutE layout_e = LayoutE::packed({params.N, columns_e});
    typename Mma::IteratorE::Params params_E{layout_e};

    typename Mma::FragmentC accumulators;
    accumulators.clear();
    int gemm_k_iterations =
        (params.K + ThreadblockShape::kK - 1) / ThreadblockShape::kK;

    auto run_operand = [&](const Element *values, ElementE *metadata) {
      typename Mma::IteratorA iterator_A(
          params_A, const_cast<Element *>(values), {params.N, sparse_k},
          thread_idx, {threadblock_offset.m(), 0});
      typename Mma::IteratorE iterator_E(
          params_E, metadata, {params.N, columns_e}, thread_idx,
          {threadblock_offset.m(), 0});
      typename Mma::IteratorB iterator_B(
          params_B, const_cast<Element *>(params.x), {params.K, tile_rows},
          thread_idx, {0, 0}, route_rows);
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
      __syncthreads();
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
          accumulators);
    };

    run_operand(params.full_values, params.full_metadata);
    if constexpr (AddResidual) {
      run_operand(params.residual_values, params.residual_metadata);
    }

    cutlass::gemm::GemmCoord threadblock_tile_index(feature_tile, row_tile, 0);
    auto problem_shape = cute::make_shape(
        int32_t(params.N), int32_t(problem.rows), int32_t(1));
    Epilogue epilogue(output_op_params, shared_storage.epilogue, thread_idx,
                      warp_idx, lane_idx);
    epilogue(accumulators, threadblock_tile_index, problem_shape, thread_idx);
    __syncthreads();
  }

  CUTLASS_DEVICE void operator()(Params const &params,
                                 SharedStorage &shared_storage) {
    if (blockIdx.x < params.sparse_workers) {
      for (int tile = blockIdx.x; tile < params.sparse_tiles;
           tile += params.sparse_workers) {
        run_tile<SparseDeviceGemm, false>(
            params, params.sparse_problem, tile, params.sparse_output_op,
            shared_storage.gemm.sparse, shared_storage.route_rows);
      }
      return;
    }

    int worker = blockIdx.x - params.sparse_workers;
    for (int tile = worker; tile < params.dense_tiles;
         tile += params.dense_workers) {
      run_tile<DenseDeviceGemm, true>(
          params, params.dense_problem, tile, params.dense_output_op,
          shared_storage.gemm.dense, shared_storage.route_rows);
    }
  }
};

// Heterogeneous exact routing in one launch. Sparse rows execute one 2:4
// mainloop, while confidence-selected dense rows execute one conventional
// dense mainloop over the original weight. Both paths gather activation rows
// in their B iterators and scatter results through the indexed epilogue, so no
// compact activation or output tensors are materialized.
template <typename SparseDeviceGemm, typename DenseDeviceGemm,
          bool GatherDenseA = false, bool GatherSparseRows = true>
struct Sparse24HeterogeneousRoutedVisitorKernel {
  using SparseBaseKernel = typename SparseDeviceGemm::GemmKernel;
  using DenseBaseKernel = typename DenseDeviceGemm::GemmKernel;
  using SparseBaseMma = typename SparseBaseKernel::Mma;
  using DenseBaseMma = typename DenseBaseKernel::Mma;
  using SparseMma =
      typename Sparse24MmaSelector<SparseBaseMma, GatherSparseRows>::Type;
  using DenseMma =
      typename DenseRoutedMmaSelector<DenseBaseMma, GatherDenseA>::Type;
  using ElementE = typename SparseBaseMma::ElementE;
  static constexpr int kThreadCount = SparseBaseKernel::kThreadCount;
  static constexpr int kSparse = SparseBaseKernel::kSparse;

  static_assert(kThreadCount == DenseBaseKernel::kThreadCount,
                "heterogeneous routed GEMMs require equal CTA sizes");
  static_assert(
      cutlass::platform::is_same<typename SparseMma::SharedStorage,
                                 typename SparseBaseMma::SharedStorage>::value,
      "selecting sparse B rows must not change MMA shared storage");
  static_assert(
      cutlass::platform::is_same<typename DenseMma::SharedStorage,
                                 typename DenseBaseMma::SharedStorage>::value,
      "gathering B must not change dense MMA shared storage");

  struct Problem {
    const int *row_indices = nullptr;
    int rows = 0;
  };

  struct Params {
    const Element *x = nullptr;
    const Element *sparse_values = nullptr;
    ElementE *sparse_metadata = nullptr;
    const Element *dense_weight = nullptr;
    const int *dense_weight_rows = nullptr;
    Problem sparse_problem;
    Problem dense_problem;
    int K = 0;
    int N = 0;
    int sparse_tiles = 0;
    int dense_tiles = 0;
    int sparse_workers = 0;
    int dense_workers = 0;
    typename SparseBaseKernel::FusionCallbacks::Params sparse_output_op{};
    typename DenseBaseKernel::FusionCallbacks::Params dense_output_op{};
  };

  static constexpr int kMaxRouteRows =
      SparseMma::Shape::kN > DenseMma::Shape::kN
          ? SparseMma::Shape::kN
          : DenseMma::Shape::kN;

  struct SharedStorage {
    union {
      Sparse24PersistentProblemSharedStorage<SparseDeviceGemm> sparse;
      Sparse24PersistentProblemSharedStorage<DenseDeviceGemm> dense;
    } gemm;
    alignas(16) int route_rows[kMaxRouteRows];
  };

  CUTLASS_DEVICE void load_route_rows(Problem const &problem, int tile_row_base,
                                      int tile_rows, int *route_rows) {
    for (int local_row = threadIdx.x; local_row < tile_rows;
         local_row += kThreadCount) {
      route_rows[local_row] =
          problem.row_indices[tile_row_base + local_row];
    }
  }

  CUTLASS_DEVICE void run_sparse_tile(
      Params const &params, int local_tile,
      Sparse24PersistentProblemSharedStorage<SparseDeviceGemm> &storage,
      int *route_rows) {
    using Epilogue = typename SparseBaseKernel::Epilogue;
    using Shape = typename SparseMma::Shape;
    static constexpr int kElementsPerElementE =
        SparseBaseKernel::kElementsPerElementE;
    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx & 31;
    int row_tiles =
        (params.sparse_problem.rows + Shape::kN - 1) / Shape::kN;
    int feature_tile = local_tile / row_tiles;
    int row_tile = local_tile - feature_tile * row_tiles;
    int feature_tiles = (params.N + Shape::kM - 1) / Shape::kM;
    if (feature_tile >= feature_tiles) {
      return;
    }

    cutlass::gemm::GemmCoord offset(feature_tile * Shape::kM,
                                    row_tile * Shape::kN, 0);
    int tile_row_base = offset.n();
    int tile_rows = params.sparse_problem.rows - tile_row_base;
    tile_rows = tile_rows < Shape::kN ? tile_rows : Shape::kN;
    if constexpr (GatherSparseRows) {
      load_route_rows(params.sparse_problem, tile_row_base, tile_rows,
                      route_rows);
    }

    int sparse_k = params.K / kSparse;
    int columns_e = sparse_k / kElementsPerElementE;
    using LayoutA = typename SparseMma::IteratorA::Layout;
    using LayoutB = typename SparseMma::IteratorB::Layout;
    using LayoutE = typename SparseMma::IteratorE::Layout;
    typename SparseMma::IteratorA::Params params_A{LayoutA(sparse_k)};
    typename SparseMma::IteratorB::Params params_B{LayoutB(params.K)};
    LayoutE layout_e = LayoutE::packed({params.N, columns_e});
    typename SparseMma::IteratorE::Params params_E{layout_e};
    typename SparseMma::IteratorA iterator_A(
        params_A, const_cast<Element *>(params.sparse_values),
        {params.N, sparse_k}, thread_idx, {offset.m(), 0});
    typename SparseMma::IteratorE iterator_E(
        params_E, params.sparse_metadata, {params.N, columns_e}, thread_idx,
        {offset.m(), 0});
    typename SparseMma::FragmentC accumulators;
    accumulators.clear();
    int gemm_k_iterations = (params.K + Shape::kK - 1) / Shape::kK;
    if constexpr (GatherSparseRows) {
      typename SparseMma::IteratorB iterator_B(
          params_B, const_cast<Element *>(params.x), {params.K, tile_rows},
          thread_idx, {0, 0}, route_rows);
      SparseMma mma(storage.main_loop, thread_idx, warp_idx, lane_idx);
      __syncthreads();
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
          accumulators);
    } else {
      typename SparseMma::IteratorB iterator_B(
          params_B, const_cast<Element *>(params.x),
          {params.K, params.sparse_problem.rows}, thread_idx,
          {0, offset.n()});
      SparseMma mma(storage.main_loop, thread_idx, warp_idx, lane_idx);
      __syncthreads();
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
          accumulators);
    }

    cutlass::gemm::GemmCoord tile_index(feature_tile, row_tile, 0);
    auto problem_shape = cute::make_shape(
        int32_t(params.N), int32_t(params.sparse_problem.rows), int32_t(1));
    Epilogue epilogue(params.sparse_output_op, storage.epilogue, thread_idx,
                      warp_idx, lane_idx);
    epilogue(accumulators, tile_index, problem_shape, thread_idx);
    __syncthreads();
  }

  CUTLASS_DEVICE void run_dense_tile(
      Params const &params, int local_tile,
      Sparse24PersistentProblemSharedStorage<DenseDeviceGemm> &storage,
      int *route_rows) {
    using Epilogue = typename DenseBaseKernel::Epilogue;
    using Shape = typename DenseMma::Shape;
    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx & 31;
    int row_tiles =
        (params.dense_problem.rows + Shape::kN - 1) / Shape::kN;
    int feature_tile = local_tile / row_tiles;
    int row_tile = local_tile - feature_tile * row_tiles;
    int feature_tiles = (params.N + Shape::kM - 1) / Shape::kM;
    if (feature_tile >= feature_tiles) {
      return;
    }

    cutlass::gemm::GemmCoord offset(feature_tile * Shape::kM,
                                    row_tile * Shape::kN, 0);
    int tile_row_base = offset.n();
    int tile_rows = params.dense_problem.rows - tile_row_base;
    tile_rows = tile_rows < Shape::kN ? tile_rows : Shape::kN;
    load_route_rows(params.dense_problem, tile_row_base, tile_rows,
                    route_rows);

    using LayoutA = typename DenseMma::IteratorA::Layout;
    using LayoutB = typename DenseMma::IteratorB::Layout;
    typename DenseMma::IteratorA::Params params_A{LayoutA(params.K)};
    typename DenseMma::IteratorB::Params params_B{LayoutB(params.K)};
    typename DenseMma::IteratorA iterator_A(
        params_A, const_cast<Element *>(params.dense_weight),
        {params.N, params.K}, thread_idx, {offset.m(), 0},
        params.dense_weight_rows);
    typename DenseMma::IteratorB iterator_B(
        params_B, const_cast<Element *>(params.x), {params.K, tile_rows},
        thread_idx, {0, 0}, route_rows);
    typename DenseMma::FragmentC accumulators;
    accumulators.clear();
    DenseMma mma(storage.main_loop, thread_idx, warp_idx, lane_idx);
    int gemm_k_iterations = (params.K + Shape::kK - 1) / Shape::kK;
    __syncthreads();
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B,
        accumulators);

    cutlass::gemm::GemmCoord tile_index(feature_tile, row_tile, 0);
    auto problem_shape = cute::make_shape(
        int32_t(params.N), int32_t(params.dense_problem.rows), int32_t(1));
    Epilogue epilogue(params.dense_output_op, storage.epilogue, thread_idx,
                      warp_idx, lane_idx);
    epilogue(accumulators, tile_index, problem_shape, thread_idx);
    __syncthreads();
  }

  CUTLASS_DEVICE void operator()(Params const &params,
                                 SharedStorage &storage) {
    if (blockIdx.x < params.sparse_workers) {
      for (int tile = blockIdx.x; tile < params.sparse_tiles;
           tile += params.sparse_workers) {
        run_sparse_tile(params, tile, storage.gemm.sparse,
                        storage.route_rows);
      }
      return;
    }
    int worker = blockIdx.x - params.sparse_workers;
    for (int tile = worker; tile < params.dense_tiles;
         tile += params.dense_workers) {
      run_dense_tile(params, tile, storage.gemm.dense, storage.route_rows);
    }
  }
};

// Keep the full 2:4 GEMM on contiguous verifier rows and assign a small group
// of adjacent row tiles to each CTA.  The owner CTA finishes those full tiles
// before gathering the confidence-selected dense rows from the same group and
// applying the complementary 2:4 residual.  Feature and row-group ownership
// makes the indexed residual epilogue race-free without a grid-wide barrier.
template <typename FullDeviceGemm, typename ResidualDeviceGemm,
          bool OffsetResidualCorrectionBase>
struct Sparse24GroupedOwnerVisitorKernel {
  using FullBaseKernel = typename FullDeviceGemm::GemmKernel;
  using ResidualBaseKernel = typename ResidualDeviceGemm::GemmKernel;
  using FullMma = typename FullBaseKernel::Mma;
  using ResidualBaseMma = typename ResidualBaseKernel::Mma;
  using ResidualMma = typename Sparse24GatherMma<ResidualBaseMma>::Type;
  using ElementE = typename FullMma::ElementE;
  using ThreadblockShape = typename FullMma::Shape;
  static constexpr int kThreadCount = FullBaseKernel::kThreadCount;
  static constexpr int kSparse = FullBaseKernel::kSparse;

  static_assert(kThreadCount == ResidualBaseKernel::kThreadCount,
                "grouped owner GEMMs require equal CTA sizes");
  static_assert(kSparse == ResidualBaseKernel::kSparse,
                "grouped owner GEMMs require one sparse format");
  static_assert(FullMma::Shape::kM == ResidualMma::Shape::kM &&
                    FullMma::Shape::kN == ResidualMma::Shape::kN &&
                    FullMma::Shape::kK == ResidualMma::Shape::kK,
                "grouped owner GEMMs require matching tile shapes");

  struct Params {
    const Element *x = nullptr;
    const Element *full_values = nullptr;
    ElementE *full_metadata = nullptr;
    const Element *residual_values = nullptr;
    ElementE *residual_metadata = nullptr;
    const int *dense_rows = nullptr;
    int dense_count = 0;
    int full_rows = 0;
    int K = 0;
    int N = 0;
    int group_tiles = 0;
    int owner_groups = 0;
    typename FullBaseKernel::FusionCallbacks::Params full_output_op{};
    typename ResidualBaseKernel::FusionCallbacks::Params residual_output_op{};
  };

  struct SharedStorage {
    union {
      Sparse24PersistentProblemSharedStorage<FullDeviceGemm> full;
      Sparse24PersistentProblemSharedStorage<ResidualDeviceGemm> residual;
    } gemm;
    alignas(16) int route_rows[ResidualMma::Shape::kN];
    int dense_bounds[2];
  };

  CUTLASS_DEVICE int lower_bound(const int *rows, int count,
                                 int target) const {
    int first = 0;
    int length = count;
    while (length > 0) {
      int step = length / 2;
      int middle = first + step;
      if (rows[middle] < target) {
        first = middle + 1;
        length -= step + 1;
      } else {
        length = step;
      }
    }
    return first;
  }

  CUTLASS_DEVICE void run_full_tile(
      Params const &params, int feature_tile, int row_tile,
      Sparse24PersistentProblemSharedStorage<FullDeviceGemm> &storage) {
    using Epilogue = typename FullBaseKernel::Epilogue;
    static constexpr int kElementsPerElementE =
        FullBaseKernel::kElementsPerElementE;
    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % 32;
    cutlass::gemm::GemmCoord threadblock_offset(
        feature_tile * ThreadblockShape::kM,
        row_tile * ThreadblockShape::kN, 0);
    int sparse_k = params.K / kSparse;
    int columns_e = sparse_k / kElementsPerElementE;

    using LayoutA = typename FullMma::IteratorA::Layout;
    using LayoutB = typename FullMma::IteratorB::Layout;
    using LayoutE = typename FullMma::IteratorE::Layout;
    typename FullMma::IteratorA::Params params_A{LayoutA(sparse_k)};
    typename FullMma::IteratorB::Params params_B{LayoutB(params.K)};
    LayoutE layout_e = LayoutE::packed({params.N, columns_e});
    typename FullMma::IteratorE::Params params_E{layout_e};
    typename FullMma::IteratorA iterator_A(
        params_A, const_cast<Element *>(params.full_values),
        {params.N, sparse_k}, thread_idx, {threadblock_offset.m(), 0});
    typename FullMma::IteratorB iterator_B(
        params_B, const_cast<Element *>(params.x),
        {params.K, params.full_rows}, thread_idx,
        {0, threadblock_offset.n()});
    typename FullMma::IteratorE iterator_E(
        params_E, params.full_metadata, {params.N, columns_e}, thread_idx,
        {threadblock_offset.m(), 0});

    typename FullMma::FragmentC accumulators;
    accumulators.clear();
    FullMma mma(storage.main_loop, thread_idx, warp_idx, lane_idx);
    int gemm_k_iterations =
        (params.K + ThreadblockShape::kK - 1) / ThreadblockShape::kK;
    __syncthreads();
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
        accumulators);

    cutlass::gemm::GemmCoord tile_index(feature_tile, row_tile, 0);
    auto problem_shape = cute::make_shape(
        int32_t(params.N), int32_t(params.full_rows), int32_t(1));
    Epilogue epilogue(params.full_output_op, storage.epilogue, thread_idx,
                      warp_idx, lane_idx);
    epilogue(accumulators, tile_index, problem_shape, thread_idx);
    __syncthreads();
  }

  CUTLASS_DEVICE void run_residual_tile(
      Params const &params, int feature_tile, int dense_begin,
      int group_dense_rows, int residual_row_tile,
      Sparse24PersistentProblemSharedStorage<ResidualDeviceGemm> &storage,
      int *route_rows) {
    using Epilogue = typename ResidualBaseKernel::Epilogue;
    static constexpr int kElementsPerElementE =
        ResidualBaseKernel::kElementsPerElementE;
    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = thread_idx % 32;
    int local_row_base = residual_row_tile * ThreadblockShape::kN;
    int tile_rows = group_dense_rows - local_row_base;
    tile_rows = tile_rows < ThreadblockShape::kN
                    ? tile_rows
                    : ThreadblockShape::kN;
    for (int local_row = thread_idx; local_row < tile_rows;
         local_row += kThreadCount) {
      route_rows[local_row] =
          params.dense_rows[dense_begin + local_row_base + local_row];
    }

    cutlass::gemm::GemmCoord threadblock_offset(
        feature_tile * ThreadblockShape::kM, local_row_base, 0);
    int sparse_k = params.K / kSparse;
    int columns_e = sparse_k / kElementsPerElementE;
    using LayoutA = typename ResidualMma::IteratorA::Layout;
    using LayoutB = typename ResidualMma::IteratorB::Layout;
    using LayoutE = typename ResidualMma::IteratorE::Layout;
    typename ResidualMma::IteratorA::Params params_A{LayoutA(sparse_k)};
    typename ResidualMma::IteratorB::Params params_B{LayoutB(params.K)};
    LayoutE layout_e = LayoutE::packed({params.N, columns_e});
    typename ResidualMma::IteratorE::Params params_E{layout_e};
    typename ResidualMma::IteratorA iterator_A(
        params_A, const_cast<Element *>(params.residual_values),
        {params.N, sparse_k}, thread_idx, {threadblock_offset.m(), 0});
    typename ResidualMma::IteratorB iterator_B(
        params_B, const_cast<Element *>(params.x), {params.K, tile_rows},
        thread_idx, {0, 0}, route_rows);
    typename ResidualMma::IteratorE iterator_E(
        params_E, params.residual_metadata, {params.N, columns_e}, thread_idx,
        {threadblock_offset.m(), 0});

    typename ResidualMma::FragmentC accumulators;
    accumulators.clear();
    ResidualMma mma(storage.main_loop, thread_idx, warp_idx, lane_idx);
    int gemm_k_iterations =
        (params.K + ThreadblockShape::kK - 1) / ThreadblockShape::kK;
    __syncthreads();
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_E,
        accumulators);

    auto output_op = params.residual_output_op;
    output_op.op_1.logical_rows = group_dense_rows;
    output_op.op_1.row_indices = params.dense_rows + dense_begin;
    if constexpr (OffsetResidualCorrectionBase) {
      output_op.op_1.correction_base +=
          static_cast<int64_t>(dense_begin) * params.N;
    }
    cutlass::gemm::GemmCoord tile_index(feature_tile, residual_row_tile, 0);
    auto problem_shape = cute::make_shape(
        int32_t(params.N), int32_t(group_dense_rows), int32_t(1));
    Epilogue epilogue(output_op, storage.epilogue, thread_idx, warp_idx,
                      lane_idx);
    epilogue(accumulators, tile_index, problem_shape, thread_idx);
    __syncthreads();
  }

  CUTLASS_DEVICE void operator()(Params const &params,
                                 SharedStorage &storage) {
    int feature_tile = blockIdx.x / params.owner_groups;
    int owner_group = blockIdx.x - feature_tile * params.owner_groups;
    int first_row_tile = owner_group * params.group_tiles;
    int full_row_tiles =
        (params.full_rows + ThreadblockShape::kN - 1) /
        ThreadblockShape::kN;
    int end_row_tile = first_row_tile + params.group_tiles;
    end_row_tile = end_row_tile < full_row_tiles ? end_row_tile
                                                 : full_row_tiles;
    for (int row_tile = first_row_tile; row_tile < end_row_tile; ++row_tile) {
      run_full_tile(params, feature_tile, row_tile, storage.gemm.full);
    }

    int group_row_begin = first_row_tile * ThreadblockShape::kN;
    int group_row_end = end_row_tile * ThreadblockShape::kN;
    group_row_end = group_row_end < params.full_rows ? group_row_end
                                                     : params.full_rows;
    if (threadIdx.x == 0) {
      storage.dense_bounds[0] =
          lower_bound(params.dense_rows, params.dense_count, group_row_begin);
      storage.dense_bounds[1] =
          lower_bound(params.dense_rows, params.dense_count, group_row_end);
    }
    __syncthreads();
    int dense_begin = storage.dense_bounds[0];
    int group_dense_rows = storage.dense_bounds[1] - dense_begin;
    int residual_tiles =
        (group_dense_rows + ThreadblockShape::kN - 1) /
        ThreadblockShape::kN;
    for (int row_tile = 0; row_tile < residual_tiles; ++row_tile) {
      run_residual_tile(params, feature_tile, dense_begin, group_dense_rows,
                        row_tile, storage.gemm.residual,
                        storage.route_rows);
    }
  }
};

template <typename ThreadblockShape_, typename WarpShape_, int Stages_,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct DeviceSparseGemmInlineTransposeVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8, kEpilogueStages>;
  using OutputStore = Sparse24TransposeStore<OutputThreadMap, Element>;
  using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Type = cutlass::gemm::device::SparseGemmWithVisitor<
      Element, cutlass::layout::RowMajor, Element,
      cutlass::layout::ColumnMajor, Element, DeviceLayoutC, float,
      cutlass::arch::OpClassTensorOp, DeviceArchTag, ThreadblockShape_,
      WarpShape_, DeviceInstructionShape, Callbacks, ThreadblockSwizzle_,
      Stages_, 8, 8, cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
};

template <typename ThreadblockShape_, typename WarpShape_, int Stages_,
          bool IndexedRows_ = false,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
          typename LayoutB_ = cutlass::layout::ColumnMajor,
          typename ElementAccumulator_ = float,
          bool AddToOutput_ = false,
          bool IndexedCorrection_ = false>
struct DeviceSparseGemmInlineVectorTransposeVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8, kEpilogueStages>;
  using OutputStore = Sparse24VectorTransposeStore<
      OutputThreadMap, Element, ThreadblockShape_::kM,
      ThreadblockShape_::kN,
      (ThreadblockShape_::kM / WarpShape_::kM) *
          (ThreadblockShape_::kN / WarpShape_::kN) * 32,
      IndexedRows_, false, AddToOutput_, false, IndexedCorrection_>;
  using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Type = cutlass::gemm::device::SparseGemmWithVisitor<
      Element, cutlass::layout::RowMajor, Element, LayoutB_, Element,
      DeviceLayoutC, ElementAccumulator_,
      cutlass::arch::OpClassTensorOp, DeviceArchTag, ThreadblockShape_,
      WarpShape_, DeviceInstructionShape, Callbacks, ThreadblockSwizzle_,
      Stages_, 8, 8, cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
};

// Dense visitor kernel with the same indexed transpose epilogue used by the
// sparse route. The heterogeneous launcher replaces this kernel's B iterator
// with DenseGatherMma, while retaining its MMA policy and epilogue callbacks.
template <typename ThreadblockShape_, typename WarpShape_, int Stages_,
          typename ElementAccumulator_ = float>
struct DeviceDenseGemmInlineVectorTransposeVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8, kEpilogueStages>;
  using OutputStore = Sparse24VectorTransposeStore<
      OutputThreadMap, Element, ThreadblockShape_::kM,
      ThreadblockShape_::kN,
      (ThreadblockShape_::kM / WarpShape_::kM) *
          (ThreadblockShape_::kN / WarpShape_::kN) * 32>;
  using FusionCallbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Builder = cutlass::gemm::kernel::DefaultGemmWithVisitor<
      Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
      Element, cutlass::layout::ColumnMajor,
      cutlass::ComplexTransform::kNone, 8, Element, DeviceLayoutC, 8,
      ElementAccumulator_, ElementAccumulator_,
      cutlass::arch::OpClassTensorOp, DeviceArchTag, ThreadblockShape_,
      WarpShape_, cutlass::gemm::GemmShape<16, 8, 16>, FusionCallbacks,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages_,
      cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
  using GemmKernel = typename Builder::GemmKernel;
};

template <typename ThreadblockShape_, typename WarpShape_, int Stages_>
struct DeviceDenseGemmInlineIndexedTransposeVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8,
          kEpilogueStages>;
  using OutputStore = Sparse24VectorTransposeStore<
      OutputThreadMap, Element, ThreadblockShape_::kM,
      ThreadblockShape_::kN,
      (ThreadblockShape_::kM / WarpShape_::kM) *
          (ThreadblockShape_::kN / WarpShape_::kN) * 32,
      true>;
  using FusionCallbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Builder = cutlass::gemm::kernel::DefaultGemmWithVisitor<
      Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
      Element, cutlass::layout::ColumnMajor,
      cutlass::ComplexTransform::kNone, 8, Element, DeviceLayoutC,
      8, float, float, cutlass::arch::OpClassTensorOp,
      DeviceArchTag,
      ThreadblockShape_, WarpShape_,
      cutlass::gemm::GemmShape<16, 8, 16>, FusionCallbacks,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages_,
      cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
  using GemmKernel = typename Builder::GemmKernel;
};

// Dense gate/up counterpart for heterogeneous SwiGLU. The visitor scatters
// compact routed rows and consumes the paired logical feature order supplied
// by DenseGatherABMma's Gather-A index.
template <typename ThreadblockShape_, typename WarpShape_, int Stages_>
struct DeviceDenseGemmInlineIndexedSwiGLUVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8, kEpilogueStages>;
  using OutputStore = Sparse24SwiGLUStore<
      OutputThreadMap, Element, ThreadblockShape_::kM,
      ThreadblockShape_::kN,
      (ThreadblockShape_::kM / WarpShape_::kM) *
          (ThreadblockShape_::kN / WarpShape_::kN) * 32,
      false, false, true>;
  using FusionCallbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Builder = cutlass::gemm::kernel::DefaultGemmWithVisitor<
      Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
      Element, cutlass::layout::ColumnMajor,
      cutlass::ComplexTransform::kNone, 8, Element, DeviceLayoutC, 8, float,
      float, cutlass::arch::OpClassTensorOp, DeviceArchTag,
      ThreadblockShape_, WarpShape_,
      cutlass::gemm::GemmShape<16, 8, 16>, FusionCallbacks,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages_,
      cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
  using GemmKernel = typename Builder::GemmKernel;
};

template <typename ThreadblockShape_, typename WarpShape_, int Stages_,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
          typename ElementAccumulator_ = float,
          bool AddRoutedResidual_ = false>
struct DeviceSparseGemmInlineRoutedTransposeVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8, kEpilogueStages>;
  using OutputStore = Sparse24VectorTransposeStore<
      OutputThreadMap, Element, ThreadblockShape_::kM,
      ThreadblockShape_::kN,
      (ThreadblockShape_::kM / WarpShape_::kM) *
          (ThreadblockShape_::kN / WarpShape_::kN) * 32,
      false, true, false, AddRoutedResidual_>;
  using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Type = cutlass::gemm::device::SparseGemmWithVisitor<
      Element, cutlass::layout::RowMajor, Element,
      cutlass::layout::ColumnMajor, Element, DeviceLayoutC,
      ElementAccumulator_,
      cutlass::arch::OpClassTensorOp, DeviceArchTag, ThreadblockShape_,
      WarpShape_, DeviceInstructionShape, Callbacks, ThreadblockSwizzle_,
      Stages_, 8, 8, cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
};

template <typename ThreadblockShape_, typename WarpShape_, int Stages_,
          bool OutputTransposed_ = false,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
          typename ElementAccumulator_ = float>
struct DeviceSparseGemmInlineSwiGLUVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8, kEpilogueStages>;
  using OutputStore = Sparse24SwiGLUStore<
      OutputThreadMap, Element, ThreadblockShape_::kM,
      ThreadblockShape_::kN,
      (ThreadblockShape_::kM / WarpShape_::kM) *
          (ThreadblockShape_::kN / WarpShape_::kN) * 32,
      OutputTransposed_>;
  using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Type = cutlass::gemm::device::SparseGemmWithVisitor<
      Element, cutlass::layout::RowMajor, Element,
      cutlass::layout::ColumnMajor, Element, DeviceLayoutC,
      ElementAccumulator_,
      cutlass::arch::OpClassTensorOp, DeviceArchTag, ThreadblockShape_,
      WarpShape_, DeviceInstructionShape, Callbacks, ThreadblockSwizzle_,
      Stages_, 8, 8, cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
};

template <typename ThreadblockShape_, typename WarpShape_, int Stages_,
          bool OutputTransposed_ = false,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
          typename ElementAccumulator_ = float,
          bool WriteRoutedApprox_ = false, bool FastSilu_ = false>
struct DeviceSparseGemmInlineRoutedSwiGLUVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8, kEpilogueStages>;
  using OutputStore = Sparse24SwiGLUStore<
      OutputThreadMap, Element, ThreadblockShape_::kM,
      ThreadblockShape_::kN,
      (ThreadblockShape_::kM / WarpShape_::kM) *
          (ThreadblockShape_::kN / WarpShape_::kN) * 32,
      OutputTransposed_, false, false, true, false,
      WriteRoutedApprox_, false, FastSilu_>;
  using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Type = cutlass::gemm::device::SparseGemmWithVisitor<
      Element, cutlass::layout::RowMajor, Element,
      cutlass::layout::ColumnMajor, Element, DeviceLayoutC,
      ElementAccumulator_,
      cutlass::arch::OpClassTensorOp, DeviceArchTag, ThreadblockShape_,
      WarpShape_, DeviceInstructionShape, Callbacks, ThreadblockSwizzle_,
      Stages_, 8, 8, cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
};

template <typename ThreadblockShape_, typename WarpShape_, int Stages_,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
          typename ElementAccumulator_ = float, bool FastSilu_ = false>
struct DeviceSparseGemmResidualCorrectionSwiGLUVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8, kEpilogueStages>;
  using OutputStore = Sparse24SwiGLUStore<
      OutputThreadMap, Element, ThreadblockShape_::kM,
      ThreadblockShape_::kN,
      (ThreadblockShape_::kM / WarpShape_::kM) *
          (ThreadblockShape_::kN / WarpShape_::kN) * 32,
      false, false, true, false, true, false, false, FastSilu_>;
  using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Type = cutlass::gemm::device::SparseGemmWithVisitor<
      Element, cutlass::layout::RowMajor, Element,
      cutlass::layout::ColumnMajor, Element, DeviceLayoutC,
      ElementAccumulator_, cutlass::arch::OpClassTensorOp, DeviceArchTag,
      ThreadblockShape_, WarpShape_, DeviceInstructionShape, Callbacks,
      ThreadblockSwizzle_, Stages_, 8, 8,
      cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
};

template <typename ThreadblockShape_, typename WarpShape_, int Stages_,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
          typename ElementAccumulator_ = float>
struct DeviceSparseGemmResidualDeltaSwiGLUVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8, kEpilogueStages>;
  using OutputStore = Sparse24SwiGLUStore<
      OutputThreadMap, Element, ThreadblockShape_::kM,
      ThreadblockShape_::kN,
      (ThreadblockShape_::kM / WarpShape_::kM) *
          (ThreadblockShape_::kN / WarpShape_::kN) * 32,
      false, false, false, false, true, false, true>;
  using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Type = cutlass::gemm::device::SparseGemmWithVisitor<
      Element, cutlass::layout::RowMajor, Element,
      cutlass::layout::ColumnMajor, Element, DeviceLayoutC,
      ElementAccumulator_, cutlass::arch::OpClassTensorOp, DeviceArchTag,
      ThreadblockShape_, WarpShape_, DeviceInstructionShape, Callbacks,
      ThreadblockSwizzle_, Stages_, 8, 8,
      cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
};

template <typename ThreadblockShape_, typename WarpShape_, int Stages_,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
          typename ElementAccumulator_ = float>
struct DeviceSparseGemmIndexedSwiGLUVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8, kEpilogueStages>;
  using OutputStore = Sparse24SwiGLUStore<
      OutputThreadMap, Element, ThreadblockShape_::kM,
      ThreadblockShape_::kN,
      (ThreadblockShape_::kM / WarpShape_::kM) *
          (ThreadblockShape_::kN / WarpShape_::kN) * 32,
      false, false, true>;
  using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Type = cutlass::gemm::device::SparseGemmWithVisitor<
      Element, cutlass::layout::RowMajor, Element,
      cutlass::layout::ColumnMajor, Element, DeviceLayoutC,
      ElementAccumulator_, cutlass::arch::OpClassTensorOp, DeviceArchTag,
      ThreadblockShape_, WarpShape_, DeviceInstructionShape, Callbacks,
      ThreadblockSwizzle_, Stages_, 8, 8,
      cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
};

template <typename ThreadblockShape_, typename WarpShape_, int Stages_,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct DeviceSparseGemmInlinePairAddVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8, kEpilogueStages>;
  using OutputStore = Sparse24SwiGLUStore<
      OutputThreadMap, Element, ThreadblockShape_::kM,
      ThreadblockShape_::kN,
      (ThreadblockShape_::kM / WarpShape_::kM) *
          (ThreadblockShape_::kN / WarpShape_::kN) * 32,
      false, true, true>;
  using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Type = cutlass::gemm::device::SparseGemmWithVisitor<
      Element, cutlass::layout::RowMajor, Element,
      cutlass::layout::ColumnMajor, Element, DeviceLayoutC, float,
      cutlass::arch::OpClassTensorOp, DeviceArchTag, ThreadblockShape_,
      WarpShape_, DeviceInstructionShape, Callbacks, ThreadblockSwizzle_,
      Stages_, 8, 8, cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
};

template <typename ThreadblockShape_, typename WarpShape_, int Stages_,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct DeviceSparseGemmInlineQKVPostOpVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8, kEpilogueStages>;
  using OutputStore = Sparse24QKVPostOpStore<
      OutputThreadMap, Element, ThreadblockShape_::kM,
      ThreadblockShape_::kN,
      (ThreadblockShape_::kM / WarpShape_::kM) *
          (ThreadblockShape_::kN / WarpShape_::kN) * 32>;
  using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Type = cutlass::gemm::device::SparseGemmWithVisitor<
      Element, cutlass::layout::RowMajor, Element,
      cutlass::layout::ColumnMajor, Element, DeviceLayoutC, float,
      cutlass::arch::OpClassTensorOp, DeviceArchTag, ThreadblockShape_,
      WarpShape_, DeviceInstructionShape, Callbacks, ThreadblockSwizzle_,
      Stages_, 8, 8, cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
};

template <typename ThreadblockShape_, typename WarpShape_, int Stages_,
          bool RoutedRows_, bool ResidualCorrection_,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct DeviceSparseGemmRoutedQKVPostOpVariant {
  static constexpr int kEpilogueStages = 1;
  using AccumulatorFetch =
      cutlass::epilogue::threadblock::VisitorAccFetch;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape_, WarpShape_, Element, 8, kEpilogueStages>;
  using OutputStore = Sparse24QKVPostOpStore<
      OutputThreadMap, Element, ThreadblockShape_::kM,
      ThreadblockShape_::kN,
      (ThreadblockShape_::kM / WarpShape_::kM) *
          (ThreadblockShape_::kN / WarpShape_::kN) * 32,
      RoutedRows_, ResidualCorrection_>;
  using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<
      OutputStore, AccumulatorFetch>;
  using Type = cutlass::gemm::device::SparseGemmWithVisitor<
      Element, cutlass::layout::RowMajor, Element,
      cutlass::layout::ColumnMajor, Element, DeviceLayoutC, float,
      cutlass::arch::OpClassTensorOp, DeviceArchTag, ThreadblockShape_,
      WarpShape_, DeviceInstructionShape, Callbacks, ThreadblockSwizzle_,
      Stages_, 8, 8, cutlass::arch::OpMultiplyAdd, kEpilogueStages>;
};

using DeviceSparseGemmInlineTransposeM256N64K64S3 =
    typename DeviceSparseGemmInlineTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3>::Type;
using DeviceSparseGemmInlineTransposeM256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineVectorTransposeM256N64K64S3 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3>::Type;
using DeviceSparseGemmInlineVectorTransposeM256N64K64S2Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 2,
        false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineVectorTransposeM256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineVectorTransposeM256N64K64S3W64x64 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x64x64, 3,
        false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineVectorTransposeM64N32K64S3 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape64x32x64, DeviceWarpShape, 3>::Type;
using DeviceSparseGemmInlineVectorTransposeM64N64K64S5 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape, DeviceWarpShape, 5>::Type;
using DeviceSparseGemmInlineVectorTransposeM64N64K64S5W32x64 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape, DeviceWarpShape32x64x64, 5>::Type;
using DeviceSparseGemmInlineVectorTransposeM64N64K64S6 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape, DeviceWarpShape, 6>::Type;
using DeviceSparseGemmInlineVectorTransposeM128N64K64S5 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape128x64x64, DeviceWarpShape, 5>::Type;
using DeviceSparseGemmInlineVectorTransposeM128N32K64S4Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 4, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineVectorTransposeM256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineVectorTransposeM256N32K64S3W32x32 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape, 3>::Type;
using DeviceSparseGemmInlineVectorTransposeF16M64N32K64S3 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape64x32x64, DeviceWarpShape, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::ColumnMajor, Element>::Type;
using DeviceSparseGemmInlineVectorTransposeF16M64N64K64S5 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape, DeviceWarpShape, 5, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::ColumnMajor, Element>::Type;
using DeviceSparseGemmInlineVectorTransposeF16M128N64K64S3Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape128x64x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::ColumnMajor, Element>::Type;
using DeviceSparseGemmInlineVectorTransposeF16M256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::ColumnMajor, Element>::Type;
using DeviceSparseGemmInlineVectorTransposeF16M256N32K64S3W32x32Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::ColumnMajor, Element>::Type;
using DeviceSparseGemmInlineVectorTransposeF16M256N64K64S3 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::ColumnMajor, Element>::Type;
using DeviceSparseGemmInlineVectorTransposeF16M256N64K64S2Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 2, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::ColumnMajor, Element>::Type;
using DeviceSparseGemmInlineVectorTransposeF16M256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::ColumnMajor, Element>::Type;
using DeviceSparseGemmInlineRoutedTransposeM64N64K64S6 =
    typename DeviceSparseGemmInlineRoutedTransposeVariant<
        DeviceThreadblockShape, DeviceWarpShape, 6>::Type;
using DeviceSparseGemmInlineRoutedTransposeM128N32K64S4Sw4 =
    typename DeviceSparseGemmInlineRoutedTransposeVariant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 4,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineRoutedTransposeM128N64K64S5 =
    typename DeviceSparseGemmInlineRoutedTransposeVariant<
        DeviceThreadblockShape128x64x64, DeviceWarpShape, 5>::Type;
using DeviceSparseGemmInlineRoutedTransposeM256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineRoutedTransposeM256N64K64S3 =
    typename DeviceSparseGemmInlineRoutedTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3>::Type;
using DeviceSparseGemmInlineRoutedTransposeM256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineRoutedTransposeF16M256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmInlineRoutedTransposeF16M256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmInlineRoutedResidualTransposeM128N32K64S4Sw4 =
    typename DeviceSparseGemmInlineRoutedTransposeVariant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 4,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, float,
        true>::Type;
using DeviceSparseGemmInlineRoutedResidualTransposeM128N64K64S5 =
    typename DeviceSparseGemmInlineRoutedTransposeVariant<
        DeviceThreadblockShape128x64x64, DeviceWarpShape, 5,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, float,
        true>::Type;
using DeviceSparseGemmInlineRoutedResidualTransposeM256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, float,
        true>::Type;
using DeviceSparseGemmInlineRoutedResidualTransposeM256N64K64S3 =
    typename DeviceSparseGemmInlineRoutedTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, float,
        true>::Type;
using DeviceSparseGemmInlineRoutedResidualTransposeM256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, float,
        true>::Type;
using DeviceSparseGemmInlineIndexedTransposeM64N64K64S6 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape, DeviceWarpShape, 6, true>::Type;
using DeviceSparseGemmInlineIndexedTransposeM128N32K64S4Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 4, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineIndexedTransposeM256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineIndexedTransposeF16M256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::ColumnMajor, Element>::Type;
// Match the 256-thread N64/N128 sparse-route CTAs while retaining an N32 tile
// for the small confidence-selected residual route.
using DeviceSparseGemmInlineIndexedTransposeF16M256N32K64S3W32x32 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::ColumnMajor, Element>::Type;
using DeviceSparseGemmInlineIndexedAddTransposeM64N32K64S3 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape64x32x64, DeviceWarpShape, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::ColumnMajor, float, true>::Type;
using DeviceSparseGemmInlineIndexedAddTransposeM128N32K64S4Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 4, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::ColumnMajor, float, true>::Type;
using DeviceSparseGemmInlineIndexedAddTransposeM128N64K64S5 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape128x64x64, DeviceWarpShape, 5, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::ColumnMajor, float, true>::Type;
using DeviceSparseGemmInlineIndexedAddTransposeM256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::ColumnMajor, float, true>::Type;
using DeviceSparseGemmInlineIndexedAddTransposeF16M256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::ColumnMajor, Element, true>::Type;
using DeviceSparseGemmInlineIndexedAddTransposeF16M256N32K64S3W32x32Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::ColumnMajor, Element, true>::Type;
using DeviceSparseGemmInlineIndexedCorrectionTransposeM256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::ColumnMajor, float, false, true>::Type;
// Keep 256 threads in the narrow residual CTA so it can share one persistent
// visitor grid with the 256x64 full route.
using DeviceSparseGemmInlineIndexedAddTransposeM256N32K64S3W32x32 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::ColumnMajor, float, true>::Type;
using DeviceSparseGemmInlineIndexedTransposeM128N64K64S5 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape128x64x64, DeviceWarpShape, 5, true>::Type;
using DeviceSparseGemmInlineIndexedTransposeM256N64K64S3 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        true>::Type;
using DeviceSparseGemmInlineIndexedTransposeF16M256N64K64S3 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::ColumnMajor, Element>::Type;
using DeviceSparseGemmInlineIndexedTransposeM256N128K64S2 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x128x64, DeviceWarpShape64x64x64, 2,
        true>::Type;
using DeviceSparseGemmInlineIndexedTransposeF16M256N128K64S2 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x128x64, DeviceWarpShape64x64x64, 2, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::ColumnMajor, Element>::Type;
using DeviceSparseGemmInlineIndexedTransposeM256N64K64S2Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 2, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineIndexedTransposeM256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceDenseGemmInlineIndexedTransposeM128N32K64S3 =
    DeviceDenseGemmInlineIndexedTransposeVariant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 3>;
using DeviceDenseGemmInlineVectorTransposeF16M64N64K64S3W16x32 =
    DeviceDenseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape, DeviceWarpShape16x32x64, 3, Element>;
using DeviceDenseGemmInlineVectorTransposeF16M64N128K64S3 =
    DeviceDenseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape64x128x64, DeviceWarpShape, 3, Element>;
using DeviceDenseGemmInlineVectorTransposeF16M64N128K64S3W32x64 =
    DeviceDenseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape64x128x64, DeviceWarpShape32x64x64, 3,
        Element>;
using DeviceDenseGemmInlineVectorTransposeF16M128N128K64S3 =
    DeviceDenseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape128x128x64, DeviceWarpShape32x64x64, 3,
        Element>;
using DeviceDenseGemmInlineIndexedTransposeM128N64K64S3 =
    DeviceDenseGemmInlineIndexedTransposeVariant<
        DeviceThreadblockShape128x64x64, DeviceWarpShape, 3>;
// Match the 256-thread sparse M256xN64 CTA while retaining a narrow dense-row
// tile. This lets the heterogeneous scheduler cover bs16/K10 and bs32/K6 in
// one resident wave instead of forcing both routes to use N64 row tiles.
using DeviceDenseGemmInlineIndexedTransposeM128N32K64S3W16x32 =
    DeviceDenseGemmInlineIndexedTransposeVariant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape16x32x64, 3>;
using DeviceSparseGemmInlineIndexedTransposeBRowM64N64K64S6 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape, DeviceWarpShape, 6, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::RowMajor>::Type;
using DeviceSparseGemmInlineIndexedTransposeBRowM64N64K64S7 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape, DeviceWarpShape, 7, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::RowMajor>::Type;
using DeviceSparseGemmInlineIndexedTransposeBRowM128N32K64S4Sw4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 4, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        cutlass::layout::RowMajor>::Type;
using DeviceSparseGemmInlineIndexedTransposeBRowM128N64K64S4 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape128x64x64, DeviceWarpShape, 4, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::RowMajor>::Type;
using DeviceSparseGemmInlineIndexedTransposeBRowM128N64K64S3 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape128x64x64, DeviceWarpShape, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::RowMajor>::Type;
using DeviceSparseGemmInlineIndexedTransposeBRowM128N64K64S5 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape128x64x64, DeviceWarpShape, 5, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::RowMajor>::Type;
using DeviceSparseGemmInlineIndexedTransposeBRowM256N64K64S3 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::RowMajor>::Type;
using DeviceSparseGemmInlineIndexedTransposeBRowM128N128K64S3 =
    typename DeviceSparseGemmInlineVectorTransposeVariant<
        DeviceThreadblockShape128x128x64, DeviceWarpShape32x64x64, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        cutlass::layout::RowMajor>::Type;
using DeviceSparseGemmInlineSwiGLUM256N64K64S3 =
    typename DeviceSparseGemmInlineSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3>::Type;
using DeviceSparseGemmInlineSwiGLUM256N32K64S3 =
    typename DeviceSparseGemmInlineSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3>::Type;
using DeviceSparseGemmInlineSwiGLUM256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineSwiGLUM256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineSwiGLUF16M256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmInlineSwiGLUF16M256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmInlineSwiGLUF16M128N64K64S3Sw4 =
    typename DeviceSparseGemmInlineSwiGLUVariant<
        DeviceThreadblockShape128x64x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmInlineSwiGLUTransposedM256N64K64S3 =
    typename DeviceSparseGemmInlineSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        true>::Type;
using DeviceSparseGemmInlineSwiGLUTransposedM256N32K64S3 =
    typename DeviceSparseGemmInlineSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        true>::Type;
using DeviceSparseGemmInlineSwiGLUTransposedM256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineSwiGLUTransposedM256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineRoutedSwiGLUM256N64K64S3 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3>::Type;
using DeviceSparseGemmInlineRoutedSwiGLUM256N64K64S2Sw4 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 2,
        false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineRoutedSwiGLUM256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineRoutedSwiGLUM256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineRoutedApproxSwiGLUM256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, float,
        true>::Type;
using DeviceSparseGemmInlineRoutedApproxSwiGLUM256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, float,
        true>::Type;
using DeviceSparseGemmInlineRoutedApproxSwiGLUF16M256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element, true>::Type;
using DeviceSparseGemmInlineRoutedApproxSwiGLUF16M256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element, true>::Type;
using DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Element>::Type;
using DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S2Sw4 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 2, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmInlineRoutedSwiGLUF16M256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmInlineRoutedFastSwiGLUF16M256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, false,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element, false, true>::Type;
using DeviceSparseGemmInlineRoutedSwiGLUTransposedM256N64K64S3 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        true>::Type;
using DeviceSparseGemmInlineRoutedSwiGLUTransposedM256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineRoutedSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3, true,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmResidualCorrectionSwiGLUM256N32K64S3Sw4 =
    typename DeviceSparseGemmResidualCorrectionSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmResidualCorrectionSwiGLUM256N64K64S3Sw4 =
    typename DeviceSparseGemmResidualCorrectionSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmResidualCorrectionSwiGLUF16M256N64K64S3 =
    typename DeviceSparseGemmResidualCorrectionSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Element>::Type;
using DeviceSparseGemmResidualCorrectionSwiGLUF16M256N32K64S3Sw4 =
    typename DeviceSparseGemmResidualCorrectionSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmResidualCorrectionSwiGLUF16M256N32K64S3W32x32Sw4 =
    typename DeviceSparseGemmResidualCorrectionSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmResidualCorrectionSwiGLUF16M256N64K64S3Sw4 =
    typename DeviceSparseGemmResidualCorrectionSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmResidualCorrectionFastSwiGLUF16M256N64K64S3Sw4 =
    typename DeviceSparseGemmResidualCorrectionSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element, true>::Type;
using DeviceSparseGemmResidualCorrectionSwiGLUF16M256N64K64S2Sw4 =
    typename DeviceSparseGemmResidualCorrectionSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 2,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmResidualDeltaSwiGLUM256N32K64S3Sw4 =
    typename DeviceSparseGemmResidualDeltaSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmResidualDeltaSwiGLUM256N64K64S3Sw4 =
    typename DeviceSparseGemmResidualDeltaSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmResidualDeltaSwiGLUF16M256N32K64S3Sw4 =
    typename DeviceSparseGemmResidualDeltaSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmResidualDeltaSwiGLUF16M256N64K64S3Sw4 =
    typename DeviceSparseGemmResidualDeltaSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmIndexedSwiGLUM256N32K64S3Sw4 =
    typename DeviceSparseGemmIndexedSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmIndexedSwiGLUM256N64K64S3Sw4 =
    typename DeviceSparseGemmIndexedSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmIndexedSwiGLUF16M256N32K64S3Sw4 =
    typename DeviceSparseGemmIndexedSwiGLUVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceSparseGemmIndexedSwiGLUF16M256N64K64S3Sw4 =
    typename DeviceSparseGemmIndexedSwiGLUVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
        Element>::Type;
using DeviceDenseGemmInlineIndexedSwiGLUM128N32K64S3 =
    DeviceDenseGemmInlineIndexedSwiGLUVariant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 3>;
using DeviceDenseGemmInlineIndexedSwiGLUM128N64K64S3 =
    DeviceDenseGemmInlineIndexedSwiGLUVariant<
        DeviceThreadblockShape128x64x64, DeviceWarpShape, 3>;
using DeviceSparseGemmInlinePairAddM256N32K64S3Sw4 =
    typename DeviceSparseGemmInlinePairAddVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlinePairAddM256N64K64S3 =
    typename DeviceSparseGemmInlinePairAddVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3>::Type;
using DeviceSparseGemmInlinePairAddM256N64K64S3Sw4 =
    typename DeviceSparseGemmInlinePairAddVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineQKVPostOpM256N64K64S3 =
    typename DeviceSparseGemmInlineQKVPostOpVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3>::Type;
using DeviceSparseGemmRoutedQKVPostOpM256N64K64S3 =
    typename DeviceSparseGemmRoutedQKVPostOpVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        true, false>::Type;
using DeviceSparseGemmResidualQKVPostOpM256N64K64S3 =
    typename DeviceSparseGemmRoutedQKVPostOpVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        false, true>::Type;
using DeviceSparseGemmResidualQKVPostOpM256N32K64S3 =
    typename DeviceSparseGemmRoutedQKVPostOpVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape, 3,
        false, true>::Type;
using DeviceSparseGemmRoutedQKVPostOpM256N32K64S3W64x32 =
    typename DeviceSparseGemmRoutedQKVPostOpVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        true, false>::Type;
using DeviceSparseGemmResidualQKVPostOpM256N32K64S3W64x32 =
    typename DeviceSparseGemmRoutedQKVPostOpVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        false, true>::Type;
using DeviceSparseGemmInlineQKVPostOpM256N64K64S3Sw4 =
    typename DeviceSparseGemmInlineQKVPostOpVariant<
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineQKVPostOpM256N32K64S3Sw4 =
    typename DeviceSparseGemmInlineQKVPostOpVariant<
        DeviceThreadblockShape256x32x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceSparseGemmInlineQKVPostOpM128N64K64S5 =
    typename DeviceSparseGemmInlineQKVPostOpVariant<
        DeviceThreadblockShape128x64x64, DeviceWarpShape, 5>::Type;
using DeviceSparseGemmInlineQKVPostOpM128N32K64S4 =
    typename DeviceSparseGemmInlineQKVPostOpVariant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 4>::Type;
using DeviceSparseGemmInlineQKVPostOpM128N32K64S4Sw2 =
    typename DeviceSparseGemmInlineQKVPostOpVariant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 4,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>>::Type;
using DeviceSparseGemmInlineQKVPostOpM128N32K64S4Sw4 =
    typename DeviceSparseGemmInlineQKVPostOpVariant<
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 4,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>::Type;
using DeviceElementE = typename DeviceSparseGemmVec8::ElementE;
using DenseInstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using DenseEpilogueOpVec8 = cutlass::epilogue::thread::LinearCombination<
    Element, 8, float, float>;
using DenseEpilogueOpVec8F16 =
    cutlass::epilogue::thread::LinearCombination<Element, 8, Element, Element>;

template <typename ThreadblockShape_, typename WarpShape_, int Stages_ = 3,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
using DeviceDenseGemmVec8Variant = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor, Element, cutlass::layout::RowMajor,
    Element, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    DeviceArchTag, ThreadblockShape_, WarpShape_, DenseInstructionShape,
    DenseEpilogueOpVec8, ThreadblockSwizzle_, Stages_, 8, 8, false,
    cutlass::arch::OpMultiplyAdd>;

template <typename ThreadblockShape_, typename WarpShape_, int Stages_ = 3,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
using DeviceDenseGemmBColVec8Variant = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor, Element, cutlass::layout::ColumnMajor,
    Element, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    DeviceArchTag, ThreadblockShape_, WarpShape_, DenseInstructionShape,
    DenseEpilogueOpVec8, ThreadblockSwizzle_, Stages_, 8, 8, false,
    cutlass::arch::OpMultiplyAdd>;

template <typename ThreadblockShape_, typename WarpShape_, int Stages_ = 3,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
using DeviceDenseGemmVec8F16Variant = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor, Element, cutlass::layout::RowMajor,
    Element, cutlass::layout::RowMajor, Element,
    cutlass::arch::OpClassTensorOp, DeviceArchTag, ThreadblockShape_, WarpShape_,
    DenseInstructionShape, DenseEpilogueOpVec8F16, ThreadblockSwizzle_, Stages_,
    8, 8, false, cutlass::arch::OpMultiplyAdd>;

template <typename ThreadblockShape_, typename WarpShape_, int Stages_ = 3,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
using DeviceDenseGemmBColVec8F16Variant = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor, Element, cutlass::layout::ColumnMajor,
    Element, cutlass::layout::RowMajor, Element,
    cutlass::arch::OpClassTensorOp, DeviceArchTag, ThreadblockShape_, WarpShape_,
    DenseInstructionShape, DenseEpilogueOpVec8F16, ThreadblockSwizzle_, Stages_,
    8, 8, false, cutlass::arch::OpMultiplyAdd>;

using DeviceDenseGemmVec8M64N64K64S3 =
    DeviceDenseGemmVec8Variant<DeviceThreadblockShape, DeviceWarpShape, 3>;
using DeviceDenseGemmVec8M64N64K64S4 =
    DeviceDenseGemmVec8Variant<DeviceThreadblockShape, DeviceWarpShape, 4>;
using DeviceDenseGemmVec8M64N128K64S3 =
    DeviceDenseGemmVec8Variant<DeviceThreadblockShape64x128x64,
                               DeviceWarpShape32x64x64, 3>;
using DeviceDenseGemmVec8M128N64K64S3 =
    DeviceDenseGemmVec8Variant<DeviceThreadblockShape128x64x64, DeviceWarpShape, 3>;
using DeviceDenseGemmVec8M128N128K64S3 =
    DeviceDenseGemmVec8Variant<DeviceThreadblockShape128x128x64,
                               DeviceWarpShape32x64x64, 3>;
using DeviceDenseGemmBColVec8M64N64K64S3 =
    DeviceDenseGemmBColVec8Variant<DeviceThreadblockShape, DeviceWarpShape, 3>;
using DeviceDenseGemmBColVec8M64N64K64S4 =
    DeviceDenseGemmBColVec8Variant<DeviceThreadblockShape, DeviceWarpShape, 4>;
using DeviceDenseGemmBColVec8M64N128K64S3 =
    DeviceDenseGemmBColVec8Variant<DeviceThreadblockShape64x128x64,
                                   DeviceWarpShape32x64x64, 3>;
using DeviceDenseGemmBColVec8M128N64K64S3 =
    DeviceDenseGemmBColVec8Variant<DeviceThreadblockShape128x64x64,
                                   DeviceWarpShape, 3>;
using DeviceDenseGemmBColVec8M128N128K64S3 =
    DeviceDenseGemmBColVec8Variant<DeviceThreadblockShape128x128x64,
                                   DeviceWarpShape32x64x64, 3>;
using DeviceDenseGemmVec8F16M64N64K64S3 =
    DeviceDenseGemmVec8F16Variant<DeviceThreadblockShape, DeviceWarpShape, 3>;
using DeviceDenseGemmVec8F16M64N64K64S4 =
    DeviceDenseGemmVec8F16Variant<DeviceThreadblockShape, DeviceWarpShape, 4>;
using DeviceDenseGemmVec8F16M64N128K64S3 =
    DeviceDenseGemmVec8F16Variant<DeviceThreadblockShape64x128x64,
                                  DeviceWarpShape32x64x64, 3>;
using DeviceDenseGemmVec8F16M128N64K64S3 =
    DeviceDenseGemmVec8F16Variant<DeviceThreadblockShape128x64x64,
                                  DeviceWarpShape, 3>;
using DeviceDenseGemmVec8F16M128N128K64S3 =
    DeviceDenseGemmVec8F16Variant<DeviceThreadblockShape128x128x64,
                                  DeviceWarpShape32x64x64, 3>;
using DeviceDenseGemmBColVec8F16M64N64K64S3 =
    DeviceDenseGemmBColVec8F16Variant<DeviceThreadblockShape, DeviceWarpShape,
                                      3>;
using DeviceDenseGemmBColVec8F16M64N64K64S4 =
    DeviceDenseGemmBColVec8F16Variant<DeviceThreadblockShape, DeviceWarpShape,
                                      4>;
using DeviceDenseGemmBColVec8F16M64N128K64S3 =
    DeviceDenseGemmBColVec8F16Variant<DeviceThreadblockShape64x128x64,
                                      DeviceWarpShape32x64x64, 3>;
using DeviceDenseGemmBColVec8F16M128N64K64S3 =
    DeviceDenseGemmBColVec8F16Variant<DeviceThreadblockShape128x64x64,
                                      DeviceWarpShape, 3>;
using DeviceDenseGemmBColVec8F16M128N128K64S3 =
    DeviceDenseGemmBColVec8F16Variant<DeviceThreadblockShape128x128x64,
                                      DeviceWarpShape32x64x64, 3>;

using DenseSimtInstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
using DenseSimtEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    Element, 1, float, float>;
using DenseSimtThreadblockShape64x64x8 =
    cutlass::gemm::GemmShape<64, 64, 8>;
using DenseSimtThreadblockShape128x64x8 =
    cutlass::gemm::GemmShape<128, 64, 8>;
using DenseSimtWarpShape32x32x8 = cutlass::gemm::GemmShape<32, 32, 8>;

template <typename ThreadblockShape_, typename WarpShape_, int Stages_ = 2>
using DeviceDenseGemmSimtBColVariant = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor, Element, cutlass::layout::ColumnMajor,
    Element, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt,
    DeviceArchTag, ThreadblockShape_, WarpShape_, DenseSimtInstructionShape,
    DenseSimtEpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages_, 1,
    1, false, cutlass::arch::OpMultiplyAdd>;

using DeviceDenseGemmSimtBColM64N64K8S2 =
    DeviceDenseGemmSimtBColVariant<DenseSimtThreadblockShape64x64x8,
                                   DenseSimtWarpShape32x32x8, 2>;
using DeviceDenseGemmSimtBColM128N64K8S2 =
    DeviceDenseGemmSimtBColVariant<DenseSimtThreadblockShape128x64x8,
                                   DenseSimtWarpShape32x32x8, 2>;

static_assert(sizeof(DeviceElementE) == 2,
              "device SparseGemm metadata packing expects uint16 E elements");
static_assert(sizeof(typename DeviceSparseGemmVec8M64N64K64S2::ElementE) ==
                  sizeof(DeviceElementE),
              "64x64x64_s2 SparseGemm metadata packing differs");
static_assert(sizeof(typename DeviceSparseGemmVec8M64N64K64S4::ElementE) ==
                  sizeof(DeviceElementE),
              "64x64x64_s4 SparseGemm metadata packing differs");
static_assert(sizeof(typename DeviceSparseGemmVec8M64N64K64S5::ElementE) ==
                  sizeof(DeviceElementE),
              "64x64x64_s5 SparseGemm metadata packing differs");
static_assert(sizeof(typename DeviceSparseGemmVec8M128N64K64S2::ElementE) ==
                  sizeof(DeviceElementE),
              "128x64x64_s2 SparseGemm metadata packing differs");
static_assert(sizeof(typename DeviceSparseGemmVec8M128N64K64S4::ElementE) ==
                  sizeof(DeviceElementE),
              "128x64x64_s4 SparseGemm metadata packing differs");

__global__ void sparse24_cutlass_device_transpose_tiled_kernel(
    const Element *c_tmp, Element *y, int M, int N) {
  constexpr int kTile = 32;
  constexpr int kBlockRows = 8;
  __shared__ Element tile[kTile][kTile + 1];

  int m_in = blockIdx.x * kTile + threadIdx.x;
  int n_in = blockIdx.y * kTile + threadIdx.y;

  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int n = n_in + j;
    if (m_in < M && n < N) {
      tile[threadIdx.y + j][threadIdx.x] = c_tmp[n * M + m_in];
    }
  }
  __syncthreads();

  int n_out = blockIdx.y * kTile + threadIdx.x;
  int m_out = blockIdx.x * kTile + threadIdx.y;

  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int m = m_out + j;
    if (m < M && n_out < N) {
      y[m * N + n_out] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

__global__ void sparse24_cutlass_transpose_add_routed_residual_kernel(
    const Element *full_transposed, const Element *residual_transposed,
    const int *dense_slot_by_row, Element *output, int M, int N,
    int full_leading_dim, int residual_leading_dim, int dense_count) {
  constexpr int kTile = 32;
  constexpr int kBlockRows = 8;
  __shared__ __half tile[kTile][kTile + 1];
  __shared__ int dense_slots[kTile];
  const __half *full = reinterpret_cast<const __half *>(full_transposed);
  const __half *residual =
      reinterpret_cast<const __half *>(residual_transposed);
  __half *out = reinterpret_cast<__half *>(output);

  int row_in = blockIdx.x * kTile + threadIdx.x;
  if (threadIdx.y == 0) {
    dense_slots[threadIdx.x] =
        row_in < M ? dense_slot_by_row[row_in] : -1;
  }
  __syncthreads();

  int column_in = blockIdx.y * kTile + threadIdx.y;
  int dense_slot = dense_slots[threadIdx.x];
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int column = column_in + j;
    if (row_in < M && column < N) {
      __half value = full[column * full_leading_dim + row_in];
      if (dense_slot >= 0 && dense_slot < dense_count) {
        value = __hadd(
            value,
            residual[column * residual_leading_dim + dense_slot]);
      }
      tile[threadIdx.y + j][threadIdx.x] = value;
    }
  }
  __syncthreads();

  int column_out = blockIdx.y * kTile + threadIdx.x;
  int row_out = blockIdx.x * kTile + threadIdx.y;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int row = row_out + j;
    if (row < M && column_out < N) {
      out[row * N + column_out] =
          tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

__global__ void
sparse24_cutlass_transpose_add_routed_residual_to_residual_kernel(
    const Element *full_transposed, const Element *routed_residual_transposed,
    const int *dense_slot_by_row, Element *model_residual, int M, int N,
    int full_leading_dim, int routed_residual_leading_dim, int dense_count) {
  constexpr int kTile = 32;
  constexpr int kBlockRows = 8;
  __shared__ __half tile[kTile][kTile + 1];
  __shared__ int dense_slots[kTile];
  const __half *full = reinterpret_cast<const __half *>(full_transposed);
  const __half *routed_residual =
      reinterpret_cast<const __half *>(routed_residual_transposed);
  __half *residual = reinterpret_cast<__half *>(model_residual);

  int row_in = blockIdx.x * kTile + threadIdx.x;
  if (threadIdx.y == 0) {
    dense_slots[threadIdx.x] =
        row_in < M ? dense_slot_by_row[row_in] : -1;
  }
  __syncthreads();

  int column_in = blockIdx.y * kTile + threadIdx.y;
  int dense_slot = dense_slots[threadIdx.x];
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int column = column_in + j;
    if (row_in < M && column < N) {
      __half value = full[column * full_leading_dim + row_in];
      if (dense_slot >= 0 && dense_slot < dense_count) {
        value = __hadd(
            value,
            routed_residual[
                column * routed_residual_leading_dim + dense_slot]);
      }
      tile[threadIdx.y + j][threadIdx.x] = value;
    }
  }
  __syncthreads();

  int column_out = blockIdx.y * kTile + threadIdx.x;
  int row_out = blockIdx.x * kTile + threadIdx.y;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int row = row_out + j;
    if (row < M && column_out < N) {
      int offset = row * N + column_out;
      residual[offset] =
          __hadd(tile[threadIdx.x][threadIdx.y + j], residual[offset]);
    }
  }
}

struct alignas(16) Sparse24Half8 {
  __half values[8];
};

__global__ void
sparse24_cutlass_transpose_add_routed_residual_partials_kernel(
    const Element *full_transposed, const Element *routed_residual_transposed,
    const int *dense_slot_by_row, Element *model_residual,
    float *square_partials, int M, int N, int full_leading_dim,
    int routed_residual_leading_dim, int dense_count) {
  constexpr int kTile = 32;
  constexpr int kBlockRows = 8;
  __shared__ __half tile[kTile][kTile + 1];
  __shared__ int dense_slots[kTile];
  const __half *full = reinterpret_cast<const __half *>(full_transposed);
  const __half *routed_residual =
      reinterpret_cast<const __half *>(routed_residual_transposed);
  __half *residual = reinterpret_cast<__half *>(model_residual);

  int row_in = blockIdx.x * kTile + threadIdx.x;
  if (threadIdx.y == 0) {
    dense_slots[threadIdx.x] =
        row_in < M ? dense_slot_by_row[row_in] : -1;
  }
  __syncthreads();

  int column_in = blockIdx.y * kTile + threadIdx.y;
  int dense_slot = dense_slots[threadIdx.x];
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int column = column_in + j;
    if (row_in < M && column < N) {
      __half value = full[column * full_leading_dim + row_in];
      if (dense_slot >= 0 && dense_slot < dense_count) {
        value = __hadd(
            value,
            routed_residual[
                column * routed_residual_leading_dim + dense_slot]);
      }
      tile[threadIdx.y + j][threadIdx.x] = value;
    }
  }
  __syncthreads();

  int column_out = blockIdx.y * kTile + threadIdx.x;
  int row_out = blockIdx.x * kTile + threadIdx.y;
  int partial_count = (N + kTile - 1) / kTile;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int row = row_out + j;
    float square = 0.0f;
    if (row < M && column_out < N) {
      int offset = row * N + column_out;
      __half sum =
          __hadd(tile[threadIdx.x][threadIdx.y + j], residual[offset]);
      residual[offset] = sum;
      float value = __half2float(sum);
      square = value * value;
    }
    for (int delta = 16; delta > 0; delta >>= 1) {
      square += __shfl_down_sync(0xffffffff, square, delta);
    }
    if (threadIdx.x == 0 && row < M) {
      square_partials[row * partial_count + blockIdx.y] = square;
    }
  }
}

__global__ void sparse24_cutlass_rmsnorm_from_partials_kernel(
    const Element *model_residual, Element *normalized,
    const Element *weight, const float *square_partials, int M, int N,
    int partial_count, float epsilon) {
  __shared__ float warp_sums[8];
  __shared__ float inverse_rms;
  int row = blockIdx.x;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  float square_sum = 0.0f;
  for (int partial = threadIdx.x; partial < partial_count;
       partial += blockDim.x) {
    square_sum += square_partials[row * partial_count + partial];
  }
  for (int delta = 16; delta > 0; delta >>= 1) {
    square_sum += __shfl_down_sync(0xffffffff, square_sum, delta);
  }
  if (lane == 0) {
    warp_sums[warp] = square_sum;
  }
  __syncthreads();
  if (warp == 0) {
    square_sum = lane < 8 ? warp_sums[lane] : 0.0f;
    for (int delta = 16; delta > 0; delta >>= 1) {
      square_sum += __shfl_down_sync(0xffffffff, square_sum, delta);
    }
    if (lane == 0) {
      inverse_rms =
          rsqrtf(square_sum / static_cast<float>(N) + epsilon);
    }
  }
  __syncthreads();

  const Sparse24Half8 *residual_vectors =
      reinterpret_cast<const Sparse24Half8 *>(model_residual + row * N);
  const Sparse24Half8 *weight_vectors =
      reinterpret_cast<const Sparse24Half8 *>(weight);
  Sparse24Half8 *output_vectors =
      reinterpret_cast<Sparse24Half8 *>(normalized + row * N);
  int vector_count = N / 8;
  for (int vector = threadIdx.x; vector < vector_count;
       vector += blockDim.x) {
    Sparse24Half8 residual_value = residual_vectors[vector];
    Sparse24Half8 weight_value = weight_vectors[vector];
    Sparse24Half8 output_value;
    CUTLASS_PRAGMA_UNROLL
    for (int item = 0; item < 8; ++item) {
      output_value.values[item] = __float2half_rn(
          __half2float(residual_value.values[item]) * inverse_rms *
          __half2float(weight_value.values[item]));
    }
    output_vectors[vector] = output_value;
  }
}

__global__ void sparse24_cutlass_transpose_add_routed_splitk_residual_kernel(
    const Element *full_transposed, const Element *residual_partials,
    const int *dense_slot_by_row, Element *output, int M, int N,
    int full_leading_dim, int residual_leading_dim, int dense_count,
    int split_k_slices) {
  constexpr int kTile = 32;
  constexpr int kBlockRows = 8;
  __shared__ __half tile[kTile][kTile + 1];
  __shared__ int dense_slots[kTile];
  const __half *full = reinterpret_cast<const __half *>(full_transposed);
  const __half *partials =
      reinterpret_cast<const __half *>(residual_partials);
  __half *out = reinterpret_cast<__half *>(output);

  int row_in = blockIdx.x * kTile + threadIdx.x;
  if (threadIdx.y == 0) {
    dense_slots[threadIdx.x] =
        row_in < M ? dense_slot_by_row[row_in] : -1;
  }
  __syncthreads();

  int column_in = blockIdx.y * kTile + threadIdx.y;
  int dense_slot = dense_slots[threadIdx.x];
  int64_t partial_stride = static_cast<int64_t>(N) * residual_leading_dim;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int column = column_in + j;
    if (row_in < M && column < N) {
      float value = __half2float(full[column * full_leading_dim + row_in]);
      if (dense_slot >= 0 && dense_slot < dense_count) {
        int64_t residual_offset =
            static_cast<int64_t>(column) * residual_leading_dim + dense_slot;
        for (int split = 0; split < split_k_slices; ++split) {
          value += __half2float(
              partials[split * partial_stride + residual_offset]);
        }
      }
      tile[threadIdx.y + j][threadIdx.x] = __float2half_rn(value);
    }
  }
  __syncthreads();

  int column_out = blockIdx.y * kTile + threadIdx.x;
  int row_out = blockIdx.x * kTile + threadIdx.y;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int row = row_out + j;
    if (row < M && column_out < N) {
      out[row * N + column_out] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

__global__ void sparse24_cutlass_transpose_add_residual_kernel(
    const Element *transposed, Element *residual, int M, int N,
    int leading_dim) {
  constexpr int kTile = 32;
  constexpr int kBlockRows = 8;
  __shared__ __half tile[kTile][kTile + 1];
  const __half *input = reinterpret_cast<const __half *>(transposed);
  __half *output = reinterpret_cast<__half *>(residual);

  int m_in = blockIdx.x * kTile + threadIdx.x;
  int n_in = blockIdx.y * kTile + threadIdx.y;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int n = n_in + j;
    if (m_in < M && n < N) {
      tile[threadIdx.y + j][threadIdx.x] =
          input[n * leading_dim + m_in];
    }
  }
  __syncthreads();

  int n_out = blockIdx.y * kTile + threadIdx.x;
  int m_out = blockIdx.x * kTile + threadIdx.y;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int m = m_out + j;
    if (m < M && n_out < N) {
      int offset = m * N + n_out;
      output[offset] = __hadd(
          output[offset],
          tile[threadIdx.x][threadIdx.y + j]);
    }
  }
}

template <int kRows>
__global__ void sparse24_cutlass_transpose_add_rmsnorm_kernel(
    const Element *transposed, Element *residual, Element *normalized,
    const Element *weight, int M, int N, int leading_dim, float epsilon) {
  constexpr int kTile = 32;
  constexpr int kBlockRows = 8;
  constexpr int kRowsPerWarp = (kRows + kBlockRows - 1) / kBlockRows;
  static_assert(kRows == 2 || kRows == 4 || kRows == 8 || kRows == 16 ||
                kRows == 32);
  __shared__ __half tile[kTile][kTile + 1];
  __shared__ float inverse_rms[kRows];
  const __half *input = reinterpret_cast<const __half *>(transposed);
  __half *residual_out = reinterpret_cast<__half *>(residual);
  __half *normalized_out = reinterpret_cast<__half *>(normalized);
  const __half *scale = reinterpret_cast<const __half *>(weight);
  int lane = threadIdx.x;
  int warp = threadIdx.y;
  int row_base = blockIdx.x * kRows;
  float row_sums[kRowsPerWarp] = {};

  for (int feature_base = 0; feature_base < N; feature_base += kTile) {
    int input_row = row_base + lane;
    int feature_in = feature_base + warp;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kTile; j += kBlockRows) {
      int feature = feature_in + j;
      if (lane < kRows && input_row < M && feature < N) {
        tile[warp + j][lane] = input[feature * leading_dim + input_row];
      }
    }
    __syncthreads();

    int feature = feature_base + lane;
    int slot = 0;
    for (int local_row = warp; local_row < kRows;
         local_row += kBlockRows, ++slot) {
      int row = row_base + local_row;
      if (row < M && feature < N) {
        int offset = row * N + feature;
        __half sum = __hadd(residual_out[offset], tile[lane][local_row]);
        residual_out[offset] = sum;
        float sum_f = __half2float(sum);
        row_sums[slot] += sum_f * sum_f;
      }
    }
    __syncthreads();
  }

  int slot = 0;
  for (int local_row = warp; local_row < kRows;
       local_row += kBlockRows, ++slot) {
    float sum = row_sums[slot];
    for (int delta = 16; delta > 0; delta >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, delta);
    }
    if (lane == 0) {
      inverse_rms[local_row] =
          rsqrtf(sum / static_cast<float>(N) + epsilon);
    }
  }
  __syncthreads();

  for (int feature_base = 0; feature_base < N; feature_base += kTile) {
    int feature = feature_base + lane;
    for (int local_row = warp; local_row < kRows;
         local_row += kBlockRows) {
      int row = row_base + local_row;
      if (row < M && feature < N) {
        int offset = row * N + feature;
        float value = __half2float(residual_out[offset]);
        normalized_out[offset] = __float2half_rn(
            value * inverse_rms[local_row] * __half2float(scale[feature]));
      }
    }
  }
}

// Qwen3 uses head_dim=128 Q/K RMSNorm immediately after QKV projection.  The
// sparse GEMM stores C as row-major [N, M], so this kernel loads coalesced over
// rows, reduces one head at a time, and writes coalesced contiguous [M, N].
template <int kRows, int kReductionLanes, bool kAddRoutedResidual = false,
          bool kInputRowMajor = false, bool kFuseValueResidual = false>
__global__ void sparse24_cutlass_qkv_transpose_rmsnorm_kernel(
    const Element *qkv_transposed, Element *qkv, const Element *q_weight,
    const Element *k_weight, const Element *cos_sin_cache,
    const int64_t *position_ids, int M, int q_size, int kv_size,
    int leading_dim, int rotary_dim, float epsilon, bool is_neox,
    bool normalize_qk, bool apply_rope,
    const Element *residual_transposed = nullptr,
    const int *dense_slot_by_row = nullptr, int residual_leading_dim = 0,
    int dense_count = 0, bool residual_row_major = false) {
  constexpr int kHeadDim = 128;
  static_assert(kRows == 16 || kRows == 32 || kRows == 64);
  static_assert(kReductionLanes == 4 || kReductionLanes == 8);
  static_assert(!kFuseValueResidual ||
                (kAddRoutedResidual && kInputRowMajor));
  __shared__ __half tile[kHeadDim][kRows + 1];
  __shared__ float partial[kReductionLanes][kRows];
  __shared__ float inverse_rms[kRows];

  const __half *input = reinterpret_cast<const __half *>(qkv_transposed);
  const __half *residual =
      reinterpret_cast<const __half *>(residual_transposed);
  __half *output = reinterpret_cast<__half *>(qkv);
  const __half *q_scale = reinterpret_cast<const __half *>(q_weight);
  const __half *k_scale = reinterpret_cast<const __half *>(k_weight);
  const __half *rope_cache =
      reinterpret_cast<const __half *>(cos_sin_cache);
  int row_lane = threadIdx.x;
  int reduction_lane = threadIdx.y;
  int head = blockIdx.y;
  int q_heads = q_size / kHeadDim;
  int kv_heads = kv_size / kHeadDim;
  int normalized_heads = q_heads + kv_heads;
  bool q_or_k = head < normalized_heads;
  bool normalize = normalize_qk && q_or_k;
  int head_offset = head * kHeadDim;

  for (int local_row = row_lane; local_row < kRows; local_row += 32) {
    int row = blockIdx.x * kRows + local_row;
    int dense_slot = -1;
    if constexpr (kAddRoutedResidual) {
      dense_slot = row < M ? dense_slot_by_row[row] : -1;
    }
    float sum = 0.0f;
    for (int dim = reduction_lane; dim < kHeadDim;
         dim += kReductionLanes) {
      __half value = __float2half(0.0f);
      if (row < M) {
        if constexpr (kInputRowMajor) {
          value = input[row * (q_size + 2 * kv_size) + head_offset + dim];
        } else {
          value = input[(head_offset + dim) * leading_dim + row];
        }
        if constexpr (kAddRoutedResidual) {
          if (dense_slot >= 0 && dense_slot < dense_count) {
            int residual_offset =
                residual_row_major
                    ? dense_slot * (q_size + 2 * kv_size) + head_offset + dim
                    : (head_offset + dim) * residual_leading_dim + dense_slot;
            value = __hadd(value, residual[residual_offset]);
          }
        }
      }
      tile[dim][local_row] = value;
      if (normalize) {
        float value_f = __half2float(value);
        sum += value_f * value_f;
      }
    }
    partial[reduction_lane][local_row] = sum;
  }
  __syncthreads();

  if (reduction_lane == 0) {
    for (int local_row = row_lane; local_row < kRows; local_row += 32) {
      float total = 0.0f;
      for (int lane = 0; lane < kReductionLanes; ++lane) {
        total += partial[lane][local_row];
      }
      inverse_rms[local_row] =
          normalize ? rsqrtf(total / static_cast<float>(kHeadDim) + epsilon)
                    : 1.0f;
    }
  }
  __syncthreads();

  int output_size = q_size + 2 * kv_size;
  const __half *weight = head < q_heads ? q_scale : k_scale;
  for (int output_row_lane = reduction_lane; output_row_lane < kRows;
       output_row_lane += kReductionLanes) {
    int output_row = blockIdx.x * kRows + output_row_lane;
    if (output_row >= M) {
      continue;
    }
    float row_inverse_rms = inverse_rms[output_row_lane];
    for (int dim = row_lane; dim < kHeadDim; dim += 32) {
      __half value = tile[dim][output_row_lane];
      float value_f = __half2float(value);
      if (normalize) {
        value_f *= row_inverse_rms * __half2float(weight[dim]);
      }
      if (apply_rope && q_or_k && dim < rotary_dim) {
        int pair_dim;
        int cache_dim;
        bool subtract_pair;
        if (is_neox) {
          int half_rotary = rotary_dim / 2;
          pair_dim = dim < half_rotary ? dim + half_rotary : dim - half_rotary;
          cache_dim = dim % half_rotary;
          subtract_pair = dim < half_rotary;
        } else {
          pair_dim = dim ^ 1;
          cache_dim = dim / 2;
          subtract_pair = (dim & 1) == 0;
        }
        float pair_f = __half2float(tile[pair_dim][output_row_lane]);
        if (normalize) {
          pair_f *= row_inverse_rms * __half2float(weight[pair_dim]);
        }
        int64_t position = position_ids[output_row];
        int cache_offset = static_cast<int>(position) * rotary_dim;
        float cos_value = __half2float(rope_cache[cache_offset + cache_dim]);
        float sin_value = __half2float(
            rope_cache[cache_offset + rotary_dim / 2 + cache_dim]);
        value_f = subtract_pair ? value_f * cos_value - pair_f * sin_value
                                : value_f * cos_value + pair_f * sin_value;
      }
      output[output_row * output_size + head_offset + dim] =
          __float2half_rn(value_f);
      if constexpr (kFuseValueResidual) {
        // Only Q/K require normalization and RoPE.  Let the first kv_heads Q
        // blocks also update one V head for routed dense rows, avoiding a full
        // V-head grid and leaving sparse-row V values untouched in-place.
        if (head < kv_heads) {
          int dense_slot = dense_slot_by_row[output_row];
          if (dense_slot >= 0 && dense_slot < dense_count) {
            int value_feature =
                q_size + kv_size + head * kHeadDim + dim;
            int value_offset = output_row * output_size + value_feature;
            int residual_offset =
                dense_slot * output_size + value_feature;
            output[value_offset] =
                __hadd(output[value_offset], residual[residual_offset]);
          }
        }
      }
    }
  }
}

// Row-major paired outputs do not need the shared-memory transpose above.
// One warp owns one row/head and keeps all 128 values in registers while it
// performs routed residual addition, optional RMSNorm, and RoPE.  A CTA covers
// two adjacent heads and a fixed row range; V is touched only for dense rows.
template <int kRowsPerBlock>
__global__ void sparse24_cutlass_qkv_rowmajor_routed_postop_kernel(
    Element *qkv_ptr, const Element *residual_ptr,
    const int *dense_slot_by_row, const Element *q_weight_ptr,
    const Element *k_weight_ptr, const Element *cos_sin_cache_ptr,
    const int64_t *position_ids, int M, int dense_count, int q_size,
    int kv_size, int rotary_dim, float epsilon, bool is_neox,
    bool normalize_qk) {
  constexpr int kHeadDim = 128;
  constexpr int kFeatureTile = 256;
  constexpr int kWarps = 8;
  static_assert(kRowsPerBlock == 8 || kRowsPerBlock == 16 ||
                kRowsPerBlock == 32 || kRowsPerBlock == 64 ||
                kRowsPerBlock == 128 || kRowsPerBlock == 256);

  __half *qkv = reinterpret_cast<__half *>(qkv_ptr);
  const __half *residual = reinterpret_cast<const __half *>(residual_ptr);
  const __half *q_weight = reinterpret_cast<const __half *>(q_weight_ptr);
  const __half *k_weight = reinterpret_cast<const __half *>(k_weight_ptr);
  const __half *rope_cache =
      reinterpret_cast<const __half *>(cos_sin_cache_ptr);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int output_size = q_size + 2 * kv_size;
  int q_heads = q_size / kHeadDim;
  int normalized_heads = (q_size + kv_size) / kHeadDim;
  int feature_base = blockIdx.y * kFeatureTile;
  int first_head = feature_base / kHeadDim;
  int row_base = blockIdx.x * kRowsPerBlock;

  CUTLASS_PRAGMA_UNROLL
  for (int local_head = 0; local_head < 2; ++local_head) {
    int head = first_head + local_head;
    int head_offset = feature_base + local_head * kHeadDim;
    if (head_offset >= output_size) {
      continue;
    }
    bool q_or_k = head < normalized_heads;
    const __half *scale = head < q_heads ? q_weight : k_weight;
    for (int local_row = warp; local_row < kRowsPerBlock;
         local_row += kWarps) {
      int row = row_base + local_row;
      if (row >= M) {
        continue;
      }
      int dense_slot = dense_slot_by_row[row];
      bool routed_dense = dense_slot >= 0 && dense_slot < dense_count;
      if (!q_or_k) {
        if (routed_dense) {
          CUTLASS_PRAGMA_UNROLL
          for (int chunk = 0; chunk < 4; ++chunk) {
            int dim = lane + chunk * 32;
            int feature = head_offset + dim;
            int output_offset = row * output_size + feature;
            int residual_offset = dense_slot * output_size + feature;
            qkv[output_offset] =
                __hadd(qkv[output_offset], residual[residual_offset]);
          }
        }
        continue;
      }

      float values[4];
      float sum = 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int chunk = 0; chunk < 4; ++chunk) {
        int dim = lane + chunk * 32;
        int feature = head_offset + dim;
        int output_offset = row * output_size + feature;
        __half value = qkv[output_offset];
        if (routed_dense) {
          int residual_offset = dense_slot * output_size + feature;
          value = __hadd(value, residual[residual_offset]);
        }
        values[chunk] = __half2float(value);
        if (normalize_qk) {
          sum += values[chunk] * values[chunk];
        }
      }
      if (normalize_qk) {
        for (int delta = 16; delta > 0; delta >>= 1) {
          sum += __shfl_down_sync(0xffffffff, sum, delta);
        }
        float inverse_rms = lane == 0
                                ? rsqrtf(sum / float(kHeadDim) + epsilon)
                                : 0.0f;
        inverse_rms =
            __shfl_sync(0xffffffff, inverse_rms, 0);
        CUTLASS_PRAGMA_UNROLL
        for (int chunk = 0; chunk < 4; ++chunk) {
          int dim = lane + chunk * 32;
          values[chunk] *= inverse_rms * __half2float(scale[dim]);
        }
      }

      int64_t position = position_ids[row];
      int cache_offset = static_cast<int>(position) * rotary_dim;
      CUTLASS_PRAGMA_UNROLL
      for (int chunk = 0; chunk < 4; ++chunk) {
        int dim = lane + chunk * 32;
        float value = values[chunk];
        if (dim < rotary_dim) {
          int pair_dim;
          int cache_dim;
          bool subtract_pair;
          if (is_neox) {
            int half_rotary = rotary_dim / 2;
            pair_dim = dim < half_rotary ? dim + half_rotary
                                         : dim - half_rotary;
            cache_dim = dim % half_rotary;
            subtract_pair = dim < half_rotary;
          } else {
            pair_dim = dim ^ 1;
            cache_dim = dim / 2;
            subtract_pair = (dim & 1) == 0;
          }
          int pair_chunk = pair_dim / 32;
          int pair_lane = pair_dim & 31;
          float pair = __shfl_sync(
              0xffffffff, values[pair_chunk], pair_lane);
          float cosine =
              __half2float(rope_cache[cache_offset + cache_dim]);
          float sine = __half2float(
              rope_cache[cache_offset + rotary_dim / 2 + cache_dim]);
          value = subtract_pair ? value * cosine - pair * sine
                                : value * cosine + pair * sine;
        }
        int output_offset =
            row * output_size + head_offset + dim;
        qkv[output_offset] = __float2half_rn(value);
      }
    }
  }
}

struct alignas(8) Sparse24Half4 {
  __half values[4];
};

// Vectorized Qwen/Llama head_dim=128 epilogue.  Each lane owns four adjacent
// values, so QKV, residual, scale, and output accesses are 64-bit contiguous
// transactions.  For Neox RoPE, paired values live in lane^16 at the same
// local component; this avoids the dynamic register-array indexing used by
// the generic warp kernel.
template <int kRowsPerBlock>
__global__ void sparse24_cutlass_qkv_rowmajor_routed_postop_vec4_kernel(
    Element *qkv_ptr, const Element *residual_ptr,
    const int *dense_slot_by_row, const Element *q_weight_ptr,
    const Element *k_weight_ptr, const Element *cos_sin_cache_ptr,
    const int64_t *position_ids, int M, int dense_count, int q_size,
    int kv_size, float epsilon, bool normalize_qk) {
  constexpr int kHeadDim = 128;
  constexpr int kValuesPerLane = 4;
  constexpr int kFeatureTile = 256;
  constexpr int kWarps = 8;
  static_assert(kRowsPerBlock == 8 || kRowsPerBlock == 16 ||
                kRowsPerBlock == 32 || kRowsPerBlock == 64 ||
                kRowsPerBlock == 128 || kRowsPerBlock == 256);

  __half *qkv = reinterpret_cast<__half *>(qkv_ptr);
  const __half *residual = reinterpret_cast<const __half *>(residual_ptr);
  const __half *q_weight = reinterpret_cast<const __half *>(q_weight_ptr);
  const __half *k_weight = reinterpret_cast<const __half *>(k_weight_ptr);
  const __half *rope_cache =
      reinterpret_cast<const __half *>(cos_sin_cache_ptr);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int output_size = q_size + 2 * kv_size;
  int q_heads = q_size / kHeadDim;
  int kv_heads = kv_size / kHeadDim;
  int normalized_heads = q_heads + kv_heads;
  int feature_base = blockIdx.y * kFeatureTile;
  int first_head = feature_base / kHeadDim;
  int row_base = blockIdx.x * kRowsPerBlock;

  CUTLASS_PRAGMA_UNROLL
  for (int local_head = 0; local_head < 2; ++local_head) {
    int head = first_head + local_head;
    int head_offset = feature_base + local_head * kHeadDim;
    if (head_offset >= output_size) {
      continue;
    }
    bool q_or_k = head < normalized_heads;
    const __half *scale = head < q_heads ? q_weight : k_weight;
    for (int local_row = warp; local_row < kRowsPerBlock;
         local_row += kWarps) {
      int row = row_base + local_row;
      if (row >= M) {
        continue;
      }
      int dense_slot = dense_slot_by_row[row];
      bool routed_dense = dense_slot >= 0 && dense_slot < dense_count;
      int lane_feature = head_offset + lane * kValuesPerLane;
      int output_offset = row * output_size + lane_feature;
      Sparse24Half4 packed = *reinterpret_cast<const Sparse24Half4 *>(
          qkv + output_offset);
      if (routed_dense) {
        int residual_offset =
            dense_slot * output_size + lane_feature;
        Sparse24Half4 correction =
            *reinterpret_cast<const Sparse24Half4 *>(
                residual + residual_offset);
        CUTLASS_PRAGMA_UNROLL
        for (int component = 0; component < kValuesPerLane; ++component) {
          packed.values[component] = __hadd(
              packed.values[component], correction.values[component]);
        }
      }
      if (!q_or_k) {
        if (routed_dense) {
          *reinterpret_cast<Sparse24Half4 *>(qkv + output_offset) = packed;
        }
        continue;
      }

      float values[kValuesPerLane];
      float sum = 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int component = 0; component < kValuesPerLane; ++component) {
        values[component] = __half2float(packed.values[component]);
        if (normalize_qk) {
          sum += values[component] * values[component];
        }
      }
      if (normalize_qk) {
        CUTLASS_PRAGMA_UNROLL
        for (int delta = 16; delta > 0; delta >>= 1) {
          sum += __shfl_down_sync(0xffffffff, sum, delta);
        }
        float inverse_rms = lane == 0
                                ? rsqrtf(sum / float(kHeadDim) + epsilon)
                                : 0.0f;
        inverse_rms = __shfl_sync(0xffffffff, inverse_rms, 0);
        Sparse24Half4 packed_scale =
            *reinterpret_cast<const Sparse24Half4 *>(
                scale + lane * kValuesPerLane);
        CUTLASS_PRAGMA_UNROLL
        for (int component = 0; component < kValuesPerLane; ++component) {
          values[component] *=
              inverse_rms * __half2float(packed_scale.values[component]);
        }
      }

      int64_t position = position_ids[row];
      int cache_offset = static_cast<int>(position) * kHeadDim;
      int pair_lane = lane ^ 16;
      bool subtract_pair = lane < 16;
      CUTLASS_PRAGMA_UNROLL
      for (int component = 0; component < kValuesPerLane; ++component) {
        int dim = lane * kValuesPerLane + component;
        int cache_dim = dim & 63;
        float pair =
            __shfl_sync(0xffffffff, values[component], pair_lane);
        float cosine =
            __half2float(rope_cache[cache_offset + cache_dim]);
        float sine = __half2float(
            rope_cache[cache_offset + 64 + cache_dim]);
        float value = subtract_pair
                          ? values[component] * cosine - pair * sine
                          : values[component] * cosine + pair * sine;
        packed.values[component] = __float2half_rn(value);
      }
      *reinterpret_cast<Sparse24Half4 *>(qkv + output_offset) = packed;
    }
  }
}

template <int kRowsPerBlock>
__global__ void
sparse24_cutlass_qkv_rowmajor_routed_postop_cache_vec4_kernel(
    Element *qkv_ptr, const Element *residual_ptr,
    const int *dense_slot_by_row, const Element *q_weight_ptr,
    const Element *k_weight_ptr, const Element *cos_sin_cache_ptr,
    const int64_t *position_ids, const int64_t *slot_mapping,
    Element *key_cache_ptr, Element *value_cache_ptr, int M, int dense_count,
    int cache_token_count, int q_size, int kv_size, int block_size,
    int64_t cache_block_stride, int64_t cache_page_stride,
    int64_t cache_head_stride, float epsilon, bool normalize_qk) {
  constexpr int kHeadDim = 128;
  constexpr int kValuesPerLane = 4;
  constexpr int kFeatureTile = 256;
  constexpr int kWarps = 8;
  static_assert(kRowsPerBlock == 8 || kRowsPerBlock == 16 ||
                kRowsPerBlock == 32 || kRowsPerBlock == 64);

  __half *qkv = reinterpret_cast<__half *>(qkv_ptr);
  const __half *residual = reinterpret_cast<const __half *>(residual_ptr);
  const __half *q_weight = reinterpret_cast<const __half *>(q_weight_ptr);
  const __half *k_weight = reinterpret_cast<const __half *>(k_weight_ptr);
  const __half *rope_cache =
      reinterpret_cast<const __half *>(cos_sin_cache_ptr);
  __half *key_cache = reinterpret_cast<__half *>(key_cache_ptr);
  __half *value_cache = reinterpret_cast<__half *>(value_cache_ptr);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int output_size = q_size + 2 * kv_size;
  int q_heads = q_size / kHeadDim;
  int kv_heads = kv_size / kHeadDim;
  int normalized_heads = q_heads + kv_heads;
  int feature_base = blockIdx.y * kFeatureTile;
  int first_head = feature_base / kHeadDim;
  int row_base = blockIdx.x * kRowsPerBlock;

  CUTLASS_PRAGMA_UNROLL
  for (int local_head = 0; local_head < 2; ++local_head) {
    int head = first_head + local_head;
    int head_offset = feature_base + local_head * kHeadDim;
    if (head_offset >= output_size) {
      continue;
    }
    bool q_or_k = head < normalized_heads;
    bool query_head = head < q_heads;
    const __half *scale = query_head ? q_weight : k_weight;
    for (int local_row = warp; local_row < kRowsPerBlock;
         local_row += kWarps) {
      int row = row_base + local_row;
      if (row >= M) {
        continue;
      }
      int dense_slot = dense_slot_by_row[row];
      bool routed_dense = dense_slot >= 0 && dense_slot < dense_count;
      int lane_feature = head_offset + lane * kValuesPerLane;
      int output_offset = row * output_size + lane_feature;
      Sparse24Half4 packed = *reinterpret_cast<const Sparse24Half4 *>(
          qkv + output_offset);
      if (routed_dense) {
        int residual_offset = dense_slot * output_size + lane_feature;
        Sparse24Half4 correction =
            *reinterpret_cast<const Sparse24Half4 *>(
                residual + residual_offset);
        CUTLASS_PRAGMA_UNROLL
        for (int component = 0; component < kValuesPerLane; ++component) {
          packed.values[component] = __hadd(
              packed.values[component], correction.values[component]);
        }
      }

      if (q_or_k) {
        float values[kValuesPerLane];
        float sum = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int component = 0; component < kValuesPerLane; ++component) {
          values[component] = __half2float(packed.values[component]);
          if (normalize_qk) {
            sum += values[component] * values[component];
          }
        }
        if (normalize_qk) {
          CUTLASS_PRAGMA_UNROLL
          for (int delta = 16; delta > 0; delta >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, delta);
          }
          float inverse_rms = lane == 0
                                  ? rsqrtf(sum / float(kHeadDim) + epsilon)
                                  : 0.0f;
          inverse_rms = __shfl_sync(0xffffffff, inverse_rms, 0);
          Sparse24Half4 packed_scale =
              *reinterpret_cast<const Sparse24Half4 *>(
                  scale + lane * kValuesPerLane);
          CUTLASS_PRAGMA_UNROLL
          for (int component = 0; component < kValuesPerLane; ++component) {
            values[component] *=
                inverse_rms * __half2float(packed_scale.values[component]);
          }
        }

        int64_t position = position_ids[row];
        int rope_offset = static_cast<int>(position) * kHeadDim;
        int pair_lane = lane ^ 16;
        bool subtract_pair = lane < 16;
        CUTLASS_PRAGMA_UNROLL
        for (int component = 0; component < kValuesPerLane; ++component) {
          int dim = lane * kValuesPerLane + component;
          int cache_dim = dim & 63;
          float pair =
              __shfl_sync(0xffffffff, values[component], pair_lane);
          float cosine =
              __half2float(rope_cache[rope_offset + cache_dim]);
          float sine =
              __half2float(rope_cache[rope_offset + 64 + cache_dim]);
          float value = subtract_pair
                            ? values[component] * cosine - pair * sine
                            : values[component] * cosine + pair * sine;
          packed.values[component] = __float2half_rn(value);
        }
      }

      if (query_head) {
        *reinterpret_cast<Sparse24Half4 *>(qkv + output_offset) = packed;
        continue;
      }
      if (row >= cache_token_count) {
        continue;
      }
      int64_t slot = slot_mapping[row];
      if (slot < 0) {
        continue;
      }
      int64_t block = slot / block_size;
      int64_t page = slot - block * block_size;
      int cache_head =
          q_or_k ? head - q_heads : head - normalized_heads;
      int64_t cache_offset = block * cache_block_stride +
                             page * cache_page_stride +
                             cache_head * cache_head_stride +
                             lane * kValuesPerLane;
      __half *cache = q_or_k ? key_cache : value_cache;
      *reinterpret_cast<Sparse24Half4 *>(cache + cache_offset) = packed;
    }
  }
}

__global__ void sparse24_cutlass_qkv_rmsnorm_inplace_kernel(
    Element *qkv, const Element *q_weight, const Element *k_weight, int M,
    int q_size, int kv_size, float epsilon) {
  constexpr int kHeadDim = 128;
  __shared__ float warp_sums[4];
  __shared__ float inverse_rms;

  __half *output = reinterpret_cast<__half *>(qkv);
  const __half *q_scale = reinterpret_cast<const __half *>(q_weight);
  const __half *k_scale = reinterpret_cast<const __half *>(k_weight);
  int row = blockIdx.x;
  int head = blockIdx.y;
  int dim = threadIdx.x;
  int q_heads = q_size / kHeadDim;
  int head_offset = head * kHeadDim;
  int output_size = q_size + 2 * kv_size;
  int offset = row * output_size + head_offset + dim;
  float value = __half2float(output[offset]);
  float sum = value * value;

  for (int delta = 16; delta > 0; delta >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, delta);
  }
  int lane = dim & 31;
  int warp = dim >> 5;
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < 4 ? warp_sums[lane] : 0.0f;
    for (int delta = 16; delta > 0; delta >>= 1) {
      block_sum += __shfl_down_sync(0xffffffff, block_sum, delta);
    }
    if (lane == 0) {
      inverse_rms =
          rsqrtf(block_sum / static_cast<float>(kHeadDim) + epsilon);
    }
  }
  __syncthreads();

  const __half *weight = head < q_heads ? q_scale : k_scale;
  output[offset] =
      __float2half_rn(value * inverse_rms * __half2float(weight[dim]));
}

__global__ void sparse24_cutlass_silu_and_mul_transposed_kernel(
    const Element *gate_up, Element *output, int M, int hidden_size,
    int leading_dim) {
  int row_pairs = (M + 1) / 2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = row_pairs * hidden_size;
  if (idx >= total_pairs) {
    return;
  }
  int hidden = idx / row_pairs;
  int row_pair = idx - hidden * row_pairs;
  int leading_pairs = leading_dim / 2;
  int offset = hidden * leading_pairs + row_pair;
  const __half2 *gate_up_packed =
      reinterpret_cast<const __half2 *>(gate_up);
  __half2 *output_packed = reinterpret_cast<__half2 *>(output);
  __half2 gate = gate_up_packed[offset];
  float2 gate_float = __half22float2(gate);
  gate_float.x = gate_float.x / (1.0f + expf(-gate_float.x));
  gate_float.y = gate_float.y / (1.0f + expf(-gate_float.y));
  __half2 silu = __floats2half2_rn(gate_float.x, gate_float.y);
  __half2 up = gate_up_packed[
      (hidden_size + hidden) * leading_pairs + row_pair];
  output_packed[offset] = __hmul2(silu, up);
}

__global__ void sparse24_cutlass_silu_and_mul_transposed_to_contiguous_kernel(
    const Element *gate_up, Element *output, int M, int hidden_size,
    int leading_dim) {
  constexpr int kTile = 32;
  constexpr int kBlockRows = 8;
  __shared__ Element tile[kTile][kTile + 1];
  const __half *gate_up_half = reinterpret_cast<const __half *>(gate_up);
  __half *output_half = reinterpret_cast<__half *>(output);

  int row_in = blockIdx.x * kTile + threadIdx.x;
  int hidden_in = blockIdx.y * kTile + threadIdx.y;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int hidden = hidden_in + j;
    if (row_in < M && hidden < hidden_size) {
      __half gate = gate_up_half[hidden * leading_dim + row_in];
      __half up =
          gate_up_half[(hidden_size + hidden) * leading_dim + row_in];
      float gate_float = __half2float(gate);
      float silu = gate_float / (1.0f + expf(-gate_float));
      reinterpret_cast<__half *>(tile)[
          (threadIdx.y + j) * (kTile + 1) + threadIdx.x] =
          __hmul(__float2half_rn(silu), up);
    }
  }
  __syncthreads();

  int hidden_out = blockIdx.y * kTile + threadIdx.x;
  int row_out = blockIdx.x * kTile + threadIdx.y;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int row = row_out + j;
    if (row < M && hidden_out < hidden_size) {
      output_half[row * hidden_size + hidden_out] =
          reinterpret_cast<__half *>(tile)[
              threadIdx.x * (kTile + 1) + threadIdx.y + j];
    }
  }
}

__global__ void sparse24_cutlass_routed_swiglu_correction_kernel(
    const Element *dense_base, const Element *dense_residual,
    const int *dense_rows, Element *output, int dense_count,
    int output_rows, int hidden_size) {
  int hidden_pairs = hidden_size / 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dense_count * hidden_pairs;
  if (index >= total) {
    return;
  }
  int dense_row = index / hidden_pairs;
  int hidden_pair = index - dense_row * hidden_pairs;
  int hidden = hidden_pair * 2;
  int output_row = dense_rows[dense_row];
  if (output_row < 0 || output_row >= output_rows) {
    return;
  }

  int64_t gate_offset =
      static_cast<int64_t>(dense_row) * hidden_size * 2 + hidden;
  int64_t up_offset = gate_offset + hidden_size;
  const __half2 *base = reinterpret_cast<const __half2 *>(dense_base);
  const __half2 *residual =
      reinterpret_cast<const __half2 *>(dense_residual);
  __half2 gate = __hadd2(base[gate_offset / 2], residual[gate_offset / 2]);
  __half2 up = __hadd2(base[up_offset / 2], residual[up_offset / 2]);
  float2 gate_float = __half22float2(gate);
  gate_float.x = gate_float.x / (1.0f + expf(-gate_float.x));
  gate_float.y = gate_float.y / (1.0f + expf(-gate_float.y));
  __half2 silu = __floats2half2_rn(gate_float.x, gate_float.y);
  __half2 *output_half2 = reinterpret_cast<__half2 *>(output);
  int64_t output_offset =
      static_cast<int64_t>(output_row) * hidden_size + hidden;
  output_half2[output_offset / 2] = __hmul2(silu, up);
}

__global__ void sparse24_cutlass_routed_swiglu_correction_gather_kernel(
    const Element *dense_base, const Element *dense_residual,
    const int *dense_rows, Element *output, Element *dense_hidden,
    int dense_count, int dense_run, int output_rows, int hidden_size) {
  int hidden_pairs = hidden_size / 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dense_run * hidden_pairs;
  if (index >= total) {
    return;
  }
  int dense_row = index / hidden_pairs;
  int hidden_pair = index - dense_row * hidden_pairs;
  __half2 *compact_half2 = reinterpret_cast<__half2 *>(dense_hidden);
  int64_t compact_offset =
      static_cast<int64_t>(dense_row) * hidden_pairs + hidden_pair;
  if (dense_row >= dense_count) {
    compact_half2[compact_offset] = __float2half2_rn(0.0f);
    return;
  }

  int hidden = hidden_pair * 2;
  int output_row = dense_rows[dense_row];
  if (output_row < 0 || output_row >= output_rows) {
    compact_half2[compact_offset] = __float2half2_rn(0.0f);
    return;
  }
  int64_t gate_offset =
      static_cast<int64_t>(dense_row) * hidden_size * 2 + hidden;
  int64_t up_offset = gate_offset + hidden_size;
  const __half2 *base = reinterpret_cast<const __half2 *>(dense_base);
  const __half2 *residual =
      reinterpret_cast<const __half2 *>(dense_residual);
  __half2 gate = __hadd2(base[gate_offset / 2], residual[gate_offset / 2]);
  __half2 up = __hadd2(base[up_offset / 2], residual[up_offset / 2]);
  float2 gate_float = __half22float2(gate);
  gate_float.x = gate_float.x / (1.0f + expf(-gate_float.x));
  gate_float.y = gate_float.y / (1.0f + expf(-gate_float.y));
  __half2 silu = __floats2half2_rn(gate_float.x, gate_float.y);
  __half2 hidden_value = __hmul2(silu, up);
  compact_half2[compact_offset] = hidden_value;
  __half2 *output_half2 = reinterpret_cast<__half2 *>(output);
  int64_t output_offset =
      static_cast<int64_t>(output_row) * hidden_pairs + hidden_pair;
  output_half2[output_offset] = hidden_value;
}

// Produce the compact hidden-state correction for a pipelined MLP.  The full
// Gate/Up epilogue has already materialized SwiGLU(W24) for every row, so the
// dense-row Down correction consumes SwiGLU(W) - SwiGLU(W24).
__global__ void sparse24_cutlass_routed_swiglu_delta_kernel(
    const Element *dense_base, const Element *dense_residual,
    Element *dense_delta, int dense_count, int dense_run, int hidden_size) {
  int hidden_pairs = hidden_size / 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dense_run * hidden_pairs;
  if (index >= total) {
    return;
  }
  int dense_row = index / hidden_pairs;
  int hidden_pair = index - dense_row * hidden_pairs;
  __half2 *delta = reinterpret_cast<__half2 *>(dense_delta);
  int64_t compact_offset =
      static_cast<int64_t>(dense_row) * hidden_pairs + hidden_pair;
  if (dense_row >= dense_count) {
    delta[compact_offset] = __float2half2_rn(0.0f);
    return;
  }

  int64_t gate_offset =
      static_cast<int64_t>(dense_row) * hidden_size + hidden_pair;
  int64_t up_offset = gate_offset + hidden_size / 2;
  const __half2 *base = reinterpret_cast<const __half2 *>(dense_base);
  const __half2 *residual =
      reinterpret_cast<const __half2 *>(dense_residual);
  __half2 base_gate = base[gate_offset];
  __half2 base_up = base[up_offset];
  __half2 exact_gate = __hadd2(base_gate, residual[gate_offset]);
  __half2 exact_up = __hadd2(base_up, residual[up_offset]);

  float2 base_gate_f = __half22float2(base_gate);
  base_gate_f.x = base_gate_f.x / (1.0f + expf(-base_gate_f.x));
  base_gate_f.y = base_gate_f.y / (1.0f + expf(-base_gate_f.y));
  __half2 base_hidden =
      __hmul2(__floats2half2_rn(base_gate_f.x, base_gate_f.y), base_up);
  float2 exact_gate_f = __half22float2(exact_gate);
  exact_gate_f.x = exact_gate_f.x / (1.0f + expf(-exact_gate_f.x));
  exact_gate_f.y = exact_gate_f.y / (1.0f + expf(-exact_gate_f.y));
  __half2 exact_hidden =
      __hmul2(__floats2half2_rn(exact_gate_f.x, exact_gate_f.y), exact_up);
  delta[compact_offset] = __hsub2(exact_hidden, base_hidden);
}

__global__ void sparse24_cutlass_routed_linear_correction_kernel(
    const Element *dense_base, const Element *dense_residual,
    const int *dense_rows, Element *output, int dense_count,
    int output_rows, int output_columns) {
  int column_pairs = output_columns / 2;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dense_count * column_pairs;
  if (index >= total) {
    return;
  }
  int dense_row = index / column_pairs;
  int column_pair = index - dense_row * column_pairs;
  int output_row = dense_rows[dense_row];
  if (output_row < 0 || output_row >= output_rows) {
    return;
  }
  const __half2 *base = reinterpret_cast<const __half2 *>(dense_base);
  const __half2 *residual =
      reinterpret_cast<const __half2 *>(dense_residual);
  __half2 *output_half2 = reinterpret_cast<__half2 *>(output);
  int64_t dense_offset =
      static_cast<int64_t>(dense_row) * column_pairs + column_pair;
  int64_t output_offset =
      static_cast<int64_t>(output_row) * column_pairs + column_pair;
  output_half2[output_offset] =
      __hadd2(base[dense_offset], residual[dense_offset]);
}

__global__ void sparse24_cutlass_routed_swiglu_correction_transposed_kernel(
    const Element *dense_base, const Element *dense_residual,
    const int *dense_rows, Element *output, int dense_count,
    int output_rows, int hidden_size, int base_ld, int residual_ld,
    int output_ld) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dense_count * hidden_size;
  if (index >= total) {
    return;
  }
  int hidden = index / dense_count;
  int dense_row = index - hidden * dense_count;
  int output_row = dense_rows[dense_row];
  if (output_row < 0 || output_row >= output_rows) {
    return;
  }
  const __half *base = reinterpret_cast<const __half *>(dense_base);
  const __half *residual =
      reinterpret_cast<const __half *>(dense_residual);
  __half *out = reinterpret_cast<__half *>(output);
  int64_t gate_base =
      static_cast<int64_t>(hidden) * base_ld + dense_row;
  int64_t gate_residual =
      static_cast<int64_t>(hidden) * residual_ld + dense_row;
  int64_t up_base =
      static_cast<int64_t>(hidden_size + hidden) * base_ld + dense_row;
  int64_t up_residual =
      static_cast<int64_t>(hidden_size + hidden) * residual_ld + dense_row;
  float gate = __half2float(__hadd(base[gate_base], residual[gate_residual]));
  float up = __half2float(__hadd(base[up_base], residual[up_residual]));
  float silu = gate / (1.0f + __expf(-gate));
  out[static_cast<int64_t>(hidden) * output_ld + output_row] =
      __float2half_rn(silu * up);
}

__global__ void
sparse24_cutlass_routed_swiglu_correction_transpose_tiled_kernel(
    const Element *sparse_hidden, const Element *dense_base,
    const Element *dense_residual, const int *dense_slot_by_row,
    Element *output, int output_rows, int dense_count, int hidden_size) {
  constexpr int kTile = 32;
  constexpr int kBlockRows = 8;
  __shared__ __half tile[kTile][kTile + 1];
  __shared__ int dense_slots[kTile];
  const __half *sparse = reinterpret_cast<const __half *>(sparse_hidden);
  const __half *base = reinterpret_cast<const __half *>(dense_base);
  const __half *residual = reinterpret_cast<const __half *>(dense_residual);
  __half *out = reinterpret_cast<__half *>(output);

  int tile_row_base = blockIdx.y * kTile;
  if (threadIdx.y == 0) {
    int row = tile_row_base + threadIdx.x;
    dense_slots[threadIdx.x] =
        row < output_rows ? dense_slot_by_row[row] : -1;
  }
  __syncthreads();

  int hidden_in = blockIdx.x * kTile + threadIdx.x;
  int local_row = threadIdx.y;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int row = tile_row_base + local_row + j;
    if (row < output_rows && hidden_in < hidden_size) {
      int dense_slot = dense_slots[local_row + j];
      __half value;
      if (dense_slot >= 0 && dense_slot < dense_count) {
        int64_t gate_offset =
            static_cast<int64_t>(dense_slot) * hidden_size * 2 + hidden_in;
        int64_t up_offset = gate_offset + hidden_size;
        __half gate = __hadd(base[gate_offset], residual[gate_offset]);
        __half up = __hadd(base[up_offset], residual[up_offset]);
        float gate_float = __half2float(gate);
        float silu = gate_float / (1.0f + expf(-gate_float));
        value = __hmul(__float2half_rn(silu), up);
      } else {
        value = sparse[static_cast<int64_t>(row) * hidden_size + hidden_in];
      }
      tile[local_row + j][threadIdx.x] = value;
    }
  }
  __syncthreads();

  int output_row = tile_row_base + threadIdx.x;
  int hidden_out = blockIdx.x * kTile + threadIdx.y;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int hidden = hidden_out + j;
    if (output_row < output_rows && hidden < hidden_size) {
      out[static_cast<int64_t>(hidden) * output_rows + output_row] =
          tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

__global__ void sparse24_cutlass_add_prefix_strided_kernel(
    Element *full_out, const Element *prefix_add, int dense_rows, int full_m,
    int prefix_m, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dense_rows * N;
  if (idx >= total) {
    return;
  }
  int m = idx % dense_rows;
  int n = idx / dense_rows;
  int full_offset = n * full_m + m;
  int prefix_offset = n * prefix_m + m;
  full_out[full_offset] = full_out[full_offset] + prefix_add[prefix_offset];
}

__global__ void sparse24_cutlass_add_indexed_rows_strided_kernel(
    Element *full_out, const Element *row_add, const int *row_indices,
    int dense_rows, int full_m, int row_m, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dense_rows * N;
  if (idx >= total) {
    return;
  }
  int row = idx % dense_rows;
  int n = idx / dense_rows;
  int m = row_indices[row];
  if (m < 0 || m >= full_m) {
    return;
  }
  int full_offset = n * full_m + m;
  int row_offset = n * row_m + row;
  full_out[full_offset] = full_out[full_offset] + row_add[row_offset];
}

__global__ void sparse24_cutlass_add_indexed_rows_contiguous_kernel(
    Element *full_out, const Element *row_add, const int *row_indices,
    int dense_rows, int full_m, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dense_rows * N;
  if (idx >= total) {
    return;
  }
  int row = idx / N;
  int n = idx % N;
  int m = row_indices[row];
  if (m < 0 || m >= full_m) {
    return;
  }
  int full_offset = m * N + n;
  int row_offset = row * N + n;
  full_out[full_offset] = full_out[full_offset] + row_add[row_offset];
}

__global__ void
sparse24_cutlass_add_indexed_rows_transposed_to_contiguous_kernel(
    Element *full_out, const Element *row_add, const int *row_indices,
    int dense_rows, int full_m, int row_m, int N) {
  constexpr int kTile = 32;
  constexpr int kBlockRows = 8;
  __shared__ Element tile[kTile][kTile + 1];

  int input_row = blockIdx.x * kTile + threadIdx.x;
  int input_n = blockIdx.y * kTile + threadIdx.y;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int n = input_n + j;
    if (input_row < dense_rows && n < N) {
      tile[threadIdx.y + j][threadIdx.x] =
          row_add[static_cast<int64_t>(n) * row_m + input_row];
    }
  }
  __syncthreads();

  int output_n = blockIdx.y * kTile + threadIdx.x;
  int output_row = blockIdx.x * kTile + threadIdx.y;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kTile; j += kBlockRows) {
    int row = output_row + j;
    if (row < dense_rows && output_n < N) {
      int m = row_indices[row];
      if (m >= 0 && m < full_m) {
        int64_t output_offset = static_cast<int64_t>(m) * N + output_n;
        full_out[output_offset] =
            full_out[output_offset] + tile[threadIdx.x][threadIdx.y + j];
      }
    }
  }
}

__global__ void sparse24_cutlass_sub_indexed_rows_contiguous_kernel(
    Element *full_out, const Element *row_sub, const int *row_indices,
    int sparse_rows, int full_m, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = sparse_rows * N;
  if (idx >= total) {
    return;
  }
  int row = idx / N;
  int n = idx % N;
  int m = row_indices[row];
  if (m < 0 || m >= full_m) {
    return;
  }
  int full_offset = m * N + n;
  int row_offset = row * N + n;
  full_out[full_offset] = full_out[full_offset] - row_sub[row_offset];
}

__global__ void sparse24_cutlass_copy_indexed_rows_strided_kernel(
    Element *full_out, const Element *row_values, const int *row_indices,
    int dense_rows, int full_m, int row_m, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dense_rows * N;
  if (idx >= total) {
    return;
  }
  int row = idx % dense_rows;
  int n = idx / dense_rows;
  int m = row_indices[row];
  if (m < 0 || m >= full_m) {
    return;
  }
  int full_offset = n * full_m + m;
  int row_offset = n * row_m + row;
  full_out[full_offset] = row_values[row_offset];
}

__global__ void sparse24_cutlass_copy_indexed_rows_contiguous_kernel(
    Element *full_out, const Element *row_values, const int *row_indices,
    int dense_rows, int full_m, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dense_rows * N;
  if (idx >= total) {
    return;
  }
  int row = idx / N;
  int n = idx - row * N;
  int m = row_indices[row];
  if (m < 0 || m >= full_m) {
    return;
  }
  full_out[n * full_m + m] = row_values[row * N + n];
}

__global__ void sparse24_cutlass_copy_indexed_rows_rowmajor_kernel(
    Element *full_out, const Element *row_values, const int *row_indices,
    int dense_rows, int full_m, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dense_rows * N;
  if (idx >= total) {
    return;
  }
  int row = idx / N;
  int n = idx - row * N;
  int m = row_indices[row];
  if (m < 0 || m >= full_m) {
    return;
  }
  full_out[m * N + n] = row_values[row * N + n];
}

__global__ void sparse24_cutlass_gather_rows_f16x8_kernel(
    const void *x_ptr, void *out_ptr, const int *row_indices, int dense_rows,
    int K_vectors) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dense_rows * K_vectors;
  if (idx >= total) {
    return;
  }
  int row = idx / K_vectors;
  int k_vec = idx - row * K_vectors;
  int src_row = row_indices[row];
  const uint4 *x = reinterpret_cast<const uint4 *>(x_ptr);
  uint4 *out = reinterpret_cast<uint4 *>(out_ptr);
  out[row * K_vectors + k_vec] = x[src_row * K_vectors + k_vec];
}

__global__ void sparse24_cutlass_gather_rows_strided_kernel(
    const Element *x, Element *out, const int *row_indices, int dense_rows,
    int full_m, int out_m, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dense_rows * K;
  if (idx >= total) {
    return;
  }
  int row = idx % dense_rows;
  int k = idx / dense_rows;
  int src_row = row_indices[row];
  if (src_row < 0 || src_row >= full_m) {
    return;
  }
  out[k * out_m + row] = x[k * full_m + src_row];
}

__global__ void sparse24_cutlass_partition_rows_f16x8_kernel(
    const void *x_ptr, void *dense_out_ptr, void *sparse_out_ptr,
    const int *dense_indices, const int *sparse_indices, int dense_rows,
    int sparse_rows, int K_vectors) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (dense_rows + sparse_rows) * K_vectors;
  if (idx >= total) {
    return;
  }
  int packed_row = idx / K_vectors;
  int k_vec = idx - packed_row * K_vectors;
  const uint4 *x = reinterpret_cast<const uint4 *>(x_ptr);
  if (packed_row < dense_rows) {
    int src_row = dense_indices[packed_row];
    uint4 *dense_out = reinterpret_cast<uint4 *>(dense_out_ptr);
    dense_out[packed_row * K_vectors + k_vec] =
        x[src_row * K_vectors + k_vec];
    return;
  }
  int sparse_row = packed_row - dense_rows;
  int src_row = sparse_indices[sparse_row];
  uint4 *sparse_out = reinterpret_cast<uint4 *>(sparse_out_ptr);
  sparse_out[sparse_row * K_vectors + k_vec] =
      x[src_row * K_vectors + k_vec];
}

__global__ void sparse24_cutlass_merge_rows_f16x8_kernel(
    void *out_ptr, const void *dense_values_ptr,
    const void *sparse_values_ptr, const int *dense_indices,
    const int *sparse_indices, int dense_rows, int sparse_rows,
    int N_vectors) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (dense_rows + sparse_rows) * N_vectors;
  if (idx >= total) {
    return;
  }
  int packed_row = idx / N_vectors;
  int n_vec = idx - packed_row * N_vectors;
  uint4 *out = reinterpret_cast<uint4 *>(out_ptr);
  if (packed_row < dense_rows) {
    const uint4 *dense_values =
        reinterpret_cast<const uint4 *>(dense_values_ptr);
    int dst_row = dense_indices[packed_row];
    out[dst_row * N_vectors + n_vec] =
        dense_values[packed_row * N_vectors + n_vec];
    return;
  }
  int sparse_row = packed_row - dense_rows;
  const uint4 *sparse_values =
      reinterpret_cast<const uint4 *>(sparse_values_ptr);
  int dst_row = sparse_indices[sparse_row];
  out[dst_row * N_vectors + n_vec] =
      sparse_values[sparse_row * N_vectors + n_vec];
}

template <typename Gemm>
int sparse24_cutlass_device_gemm_run(const Element *x, const Element *a_values,
                                     DeviceElementE *a_meta_e, Element *c_tmp,
                                     int M, int K, int N, cudaStream_t stream) {
  int sparse_k = K / Gemm::kSparse;
  int columns_e = K / Gemm::kSparse / Gemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);
  DeviceLayoutC layout_c(M);

  typename Gemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, cutlass::layout::ColumnMajor(K)},
      {c_tmp, layout_c},
      {c_tmp, layout_c},
      {reinterpret_cast<typename Gemm::ElementE *>(a_meta_e),
       Gemm::LayoutE::packed({N, columns_e})},
      typename Gemm::EpilogueOutputOp::Params(),
      1};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename FullDeviceGemm, typename ResidualDeviceGemm>
int sparse24_cutlass_paired_persistent_gemm_run(
    const Element *full_x, const Element *full_values,
    DeviceElementE *full_meta, Element *full_output, int full_rows,
    const Element *residual_x, const Element *residual_values,
    DeviceElementE *residual_meta, Element *residual_output,
    int residual_rows, int K, int N, int interleaved_schedule,
    cudaStream_t stream) {
  using Kernel =
      Sparse24PairedPersistentKernel<FullDeviceGemm, ResidualDeviceGemm>;
  typename Kernel::Params params;
  params.problems[0] = {
      full_x, full_values,
      reinterpret_cast<typename Kernel::ElementE *>(full_meta), full_output,
      full_rows};
  params.problems[1] = {
      residual_x, residual_values,
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta),
      residual_output, residual_rows};
  params.K = K;
  params.N = N;
  params.interleaved_schedule = interleaved_schedule;
  int full_feature_tiles =
      (N + Kernel::FullThreadblockShape::kM - 1) /
      Kernel::FullThreadblockShape::kM;
  int full_row_tiles =
      (full_rows + Kernel::FullThreadblockShape::kN - 1) /
      Kernel::FullThreadblockShape::kN;
  int residual_feature_tiles =
      (N + Kernel::ResidualThreadblockShape::kM - 1) /
      Kernel::ResidualThreadblockShape::kM;
  int residual_row_tiles =
      (residual_rows + Kernel::ResidualThreadblockShape::kN - 1) /
      Kernel::ResidualThreadblockShape::kN;
  params.first_problem_tiles = full_feature_tiles * full_row_tiles;
  params.total_tiles = params.first_problem_tiles +
                       residual_feature_tiles * residual_row_tiles;

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount, smem_size);
  if (active_blocks < 1) {
    active_blocks = 1;
  }
  int persistent_blocks = sm_count * active_blocks;
  int grid = params.total_tiles < persistent_blocks ? params.total_tiles
                                                     : persistent_blocks;
  int residual_tiles = params.total_tiles - params.first_problem_tiles;
  int residual_workers =
      (grid * residual_tiles + params.total_tiles - 1) / params.total_tiles;
  residual_workers = residual_workers < 1 ? 1 : residual_workers;
  residual_workers = residual_workers > residual_tiles ? residual_tiles
                                                        : residual_workers;
  int full_workers = grid - residual_workers;
  if (full_workers < 1) {
    full_workers = 1;
    residual_workers = grid - 1;
  }
  if (full_workers > params.first_problem_tiles) {
    full_workers = params.first_problem_tiles;
    residual_workers = grid - full_workers;
  }
  params.full_worker_blocks = full_workers;
  params.residual_worker_blocks = residual_workers;
  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename FullDeviceGemm, typename ResidualDeviceGemm>
int sparse24_cutlass_paired_gather_residual_gemm_run(
    const Element *x, const Element *full_values,
    DeviceElementE *full_meta, Element *full_output, int full_rows,
    const Element *residual_values, DeviceElementE *residual_meta,
    Element *residual_output, const int *dense_rows, int dense_count, int K,
    int N, int interleaved_schedule, int worker_blocks,
    cudaStream_t stream) {
  using Kernel =
      Sparse24PairedGatherResidualKernel<FullDeviceGemm, ResidualDeviceGemm>;
  typename Kernel::Params params;
  params.x = x;
  params.full_values = full_values;
  params.full_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(full_meta);
  params.full_output = full_output;
  params.residual_values = residual_values;
  params.residual_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta);
  params.residual_output = residual_output;
  params.dense_rows = dense_rows;
  params.full_rows = full_rows;
  params.dense_count = dense_count;
  params.K = K;
  params.N = N;
  params.interleaved_schedule = interleaved_schedule;

  int full_feature_tiles =
      (N + Kernel::FullThreadblockShape::kM - 1) /
      Kernel::FullThreadblockShape::kM;
  int full_row_tiles =
      (full_rows + Kernel::FullThreadblockShape::kN - 1) /
      Kernel::FullThreadblockShape::kN;
  int residual_feature_tiles =
      (N + Kernel::ResidualThreadblockShape::kM - 1) /
      Kernel::ResidualThreadblockShape::kM;
  int residual_row_tiles =
      (dense_count + Kernel::ResidualThreadblockShape::kN - 1) /
      Kernel::ResidualThreadblockShape::kN;
  params.full_tiles = full_feature_tiles * full_row_tiles;
  params.residual_tiles = residual_feature_tiles * residual_row_tiles;
  int total_tiles = params.full_tiles + params.residual_tiles;

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount,
      smem_size);
  if (active_blocks < 1) {
    active_blocks = 1;
  }
  int persistent_blocks = sm_count * active_blocks;
  int grid = total_tiles < persistent_blocks ? total_tiles
                                              : persistent_blocks;
  if (worker_blocks > 0 && worker_blocks < grid) {
    grid = worker_blocks;
  }
  if (grid < 2) {
    return -8;
  }

  int residual_workers =
      (grid * params.residual_tiles + total_tiles - 1) / total_tiles;
  residual_workers = residual_workers < 1 ? 1 : residual_workers;
  residual_workers = residual_workers > params.residual_tiles
                         ? params.residual_tiles
                         : residual_workers;
  int full_workers = grid - residual_workers;
  if (full_workers < 1) {
    full_workers = 1;
    residual_workers = grid - 1;
  }
  if (full_workers > params.full_tiles) {
    full_workers = params.full_tiles;
    residual_workers = grid - full_workers;
  }
  if (residual_workers < 1) {
    return -9;
  }
  params.full_worker_blocks = full_workers;
  params.residual_worker_blocks = residual_workers;

  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename FullDeviceGemm, typename ResidualDeviceGemm>
int sparse24_cutlass_paired_gather_residual_visitor_gemm_run(
    const Element *x, const Element *full_values,
    DeviceElementE *full_meta, Element *full_output, int full_rows,
    const Element *residual_values, DeviceElementE *residual_meta,
    Element *residual_output, const int *dense_rows, int dense_count, int K,
    int N, int interleaved_schedule, int worker_blocks,
    cudaStream_t stream) {
  using Kernel = Sparse24PairedPersistentVisitorKernel<
      FullDeviceGemm, ResidualDeviceGemm, true>;
  using FullCallbacks = typename Kernel::FullBaseKernel::FusionCallbacks;
  using ResidualCallbacks =
      typename Kernel::ResidualBaseKernel::FusionCallbacks;
  typename Kernel::Params params;
  params.problems[0] = {
      x, full_values, reinterpret_cast<typename Kernel::ElementE *>(full_meta),
      nullptr, full_rows};
  params.problems[1] = {
      x, residual_values,
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta),
      dense_rows, dense_count};
  params.K = K;
  params.N = N;
  params.interleaved_schedule = interleaved_schedule;

  typename FullCallbacks::Arguments full_callback_args{};
  full_callback_args.op_1.output = full_output;
  full_callback_args.op_1.output_columns = N;
  full_callback_args.op_1.logical_rows = full_rows;
  cutlass::gemm::GemmCoord full_problem_size(N, full_rows, K);
  params.full_output_op = FullCallbacks::to_underlying_arguments(
      full_problem_size, full_callback_args, nullptr);

  typename ResidualCallbacks::Arguments residual_callback_args{};
  residual_callback_args.op_1.output = residual_output;
  residual_callback_args.op_1.output_columns = N;
  residual_callback_args.op_1.logical_rows = dense_count;
  cutlass::gemm::GemmCoord residual_problem_size(N, dense_count, K);
  params.residual_output_op = ResidualCallbacks::to_underlying_arguments(
      residual_problem_size, residual_callback_args, nullptr);

  int full_feature_tiles =
      (N + Kernel::FullThreadblockShape::kM - 1) /
      Kernel::FullThreadblockShape::kM;
  int full_row_tiles =
      (full_rows + Kernel::FullThreadblockShape::kN - 1) /
      Kernel::FullThreadblockShape::kN;
  int residual_feature_tiles =
      (N + Kernel::ResidualThreadblockShape::kM - 1) /
      Kernel::ResidualThreadblockShape::kM;
  int residual_row_tiles =
      (dense_count + Kernel::ResidualThreadblockShape::kN - 1) /
      Kernel::ResidualThreadblockShape::kN;
  params.first_problem_tiles = full_feature_tiles * full_row_tiles;
  int residual_tiles = residual_feature_tiles * residual_row_tiles;
  params.total_tiles = params.first_problem_tiles + residual_tiles;

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount,
      smem_size);
  if (active_blocks < 1) {
    active_blocks = 1;
  }
  int persistent_blocks = sm_count * active_blocks;
  int grid = params.total_tiles < persistent_blocks ? params.total_tiles
                                                     : persistent_blocks;
  if (worker_blocks > 0 && worker_blocks < grid) {
    grid = worker_blocks;
  }
  if (grid < 2) {
    return -8;
  }

  int residual_workers =
      (grid * residual_tiles + params.total_tiles - 1) / params.total_tiles;
  residual_workers = residual_workers < 1 ? 1 : residual_workers;
  residual_workers = residual_workers > residual_tiles ? residual_tiles
                                                        : residual_workers;
  int full_workers = grid - residual_workers;
  if (full_workers < 1) {
    full_workers = 1;
    residual_workers = grid - 1;
  }
  if (full_workers > params.first_problem_tiles) {
    full_workers = params.first_problem_tiles;
    residual_workers = grid - full_workers;
  }
  if (residual_workers < 1) {
    return -9;
  }
  params.full_worker_blocks = full_workers;
  params.residual_worker_blocks = residual_workers;

  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename FullDeviceGemm, typename ResidualDeviceGemm>
int sparse24_cutlass_paired_gather_residual_qkv_visitor_gemm_run(
    const Element *x, const Element *full_values,
    DeviceElementE *full_meta, Element *full_output, int full_rows,
    const Element *residual_values, DeviceElementE *residual_meta,
    Element *residual_output, const int *dense_rows,
    const int *dense_slot_by_row, int dense_count, int K, int N,
    const Element *q_weight, const Element *k_weight,
    const Element *cos_sin_cache, const int64_t *position_ids,
    int q_size, int kv_size, int rotary_dim, float epsilon, int is_neox,
    int normalize_qk, int *grid_barrier, int interleaved_schedule,
    int worker_blocks, cudaStream_t stream) {
  using Kernel = Sparse24PairedPersistentVisitorKernel<
      FullDeviceGemm, ResidualDeviceGemm, true, false, true>;
  using FullCallbacks = typename Kernel::FullBaseKernel::FusionCallbacks;
  using ResidualCallbacks =
      typename Kernel::ResidualBaseKernel::FusionCallbacks;
  typename Kernel::Params params;
  params.problems[0] = {
      x, full_values, reinterpret_cast<typename Kernel::ElementE *>(full_meta),
      nullptr, full_rows};
  params.problems[1] = {
      x, residual_values,
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta), dense_rows,
      dense_count};
  params.K = K;
  params.N = N;
  params.interleaved_schedule = interleaved_schedule;
  params.dense_slot_by_row = dense_slot_by_row;
  params.q_weight = q_weight;
  params.k_weight = k_weight;
  params.cos_sin_cache = cos_sin_cache;
  params.position_ids = position_ids;
  params.q_size = q_size;
  params.kv_size = kv_size;
  params.rotary_dim = rotary_dim;
  params.epsilon = epsilon;
  params.is_neox = is_neox;
  params.normalize_qk = normalize_qk;
  params.grid_barrier = grid_barrier;

  typename FullCallbacks::Arguments full_callback_args{};
  full_callback_args.op_1.output = full_output;
  full_callback_args.op_1.output_columns = N;
  full_callback_args.op_1.logical_rows = full_rows;
  cutlass::gemm::GemmCoord full_problem_size(N, full_rows, K);
  params.full_output_op = FullCallbacks::to_underlying_arguments(
      full_problem_size, full_callback_args, nullptr);

  typename ResidualCallbacks::Arguments residual_callback_args{};
  residual_callback_args.op_1.output = residual_output;
  residual_callback_args.op_1.output_columns = N;
  residual_callback_args.op_1.logical_rows = dense_count;
  cutlass::gemm::GemmCoord residual_problem_size(N, dense_count, K);
  params.residual_output_op = ResidualCallbacks::to_underlying_arguments(
      residual_problem_size, residual_callback_args, nullptr);

  int full_feature_tiles =
      (N + Kernel::FullThreadblockShape::kM - 1) /
      Kernel::FullThreadblockShape::kM;
  params.full_row_tiles =
      (full_rows + Kernel::FullThreadblockShape::kN - 1) /
      Kernel::FullThreadblockShape::kN;
  int residual_feature_tiles =
      (N + Kernel::ResidualThreadblockShape::kM - 1) /
      Kernel::ResidualThreadblockShape::kM;
  params.residual_row_tiles =
      (dense_count + Kernel::ResidualThreadblockShape::kN - 1) /
      Kernel::ResidualThreadblockShape::kN;
  params.first_problem_tiles = full_feature_tiles * params.full_row_tiles;
  int residual_tiles = residual_feature_tiles * params.residual_row_tiles;
  params.total_tiles = params.first_problem_tiles + residual_tiles;

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount,
      smem_size);
  if (active_blocks < 1) {
    active_blocks = 1;
  }
  int persistent_blocks = sm_count * active_blocks;
  int grid = params.total_tiles < persistent_blocks ? params.total_tiles
                                                     : persistent_blocks;
  if (worker_blocks > 0 && worker_blocks < grid) {
    grid = worker_blocks;
  }
  if (grid < 2) {
    return -8;
  }

  int residual_workers =
      (grid * residual_tiles + params.total_tiles - 1) / params.total_tiles;
  residual_workers = residual_workers < 1 ? 1 : residual_workers;
  residual_workers = residual_workers > residual_tiles ? residual_tiles
                                                        : residual_workers;
  int full_workers = grid - residual_workers;
  if (full_workers < 1) {
    full_workers = 1;
    residual_workers = grid - 1;
  }
  if (full_workers > params.first_problem_tiles) {
    full_workers = params.first_problem_tiles;
    residual_workers = grid - full_workers;
  }
  if (residual_workers < 1) {
    return -9;
  }
  params.full_worker_blocks = full_workers;
  params.residual_worker_blocks = residual_workers;

  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

// Fuse mixed-row QKV verification without a grid-wide barrier. W24 epilogues
// finalize sparse rows and stage routed dense bases. R24 epilogues wait only
// for the matching 256-feature tile, add the staged base, and finalize/scatter
// dense rows. The fixed worker partition keeps full producers resident while
// residual consumers wait on their per-feature counter.
template <typename FullDeviceGemm, typename ResidualDeviceGemm>
int sparse24_cutlass_paired_fused_routed_qkv_epilogue_gemm_run(
    const Element *x, const Element *full_values,
    DeviceElementE *full_meta, const Element *residual_values,
    DeviceElementE *residual_meta, const int *dense_rows,
    const int *dense_slot_by_row, int dense_count, Element *output,
    Element *dense_base, int full_rows, int K, int N,
    const Element *q_weight, const Element *k_weight,
    const Element *cos_sin_cache, const int64_t *position_ids,
    int q_size, int kv_size, int rotary_dim, float epsilon, int is_neox,
    int normalize_qk, int *feature_counters, int worker_blocks,
    int requested_residual_workers,
    cudaStream_t stream) {
  using Kernel = Sparse24PairedPersistentVisitorKernel<
      FullDeviceGemm, ResidualDeviceGemm, true, true>;
  using FullCallbacks = typename Kernel::FullBaseKernel::FusionCallbacks;
  using ResidualCallbacks =
      typename Kernel::ResidualBaseKernel::FusionCallbacks;
  typename Kernel::Params params;
  params.problems[0] = {
      x, full_values, reinterpret_cast<typename Kernel::ElementE *>(full_meta),
      nullptr, full_rows};
  params.problems[1] = {
      x, residual_values,
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta), dense_rows,
      dense_count};
  params.K = K;
  params.N = N;
  params.interleaved_schedule = 0;
  params.feature_counters = feature_counters;

  typename FullCallbacks::Arguments full_callback_args{};
  full_callback_args.op_1.output = output;
  full_callback_args.op_1.dense_base = dense_base;
  full_callback_args.op_1.q_weight = q_weight;
  full_callback_args.op_1.k_weight = k_weight;
  full_callback_args.op_1.cos_sin_cache = cos_sin_cache;
  full_callback_args.op_1.position_ids = position_ids;
  full_callback_args.op_1.dense_slot_by_row = dense_slot_by_row;
  full_callback_args.op_1.q_size = q_size;
  full_callback_args.op_1.kv_size = kv_size;
  full_callback_args.op_1.logical_rows = full_rows;
  full_callback_args.op_1.output_rows = full_rows;
  full_callback_args.op_1.dense_rows = dense_count;
  full_callback_args.op_1.rotary_dim = rotary_dim;
  full_callback_args.op_1.epsilon = epsilon;
  full_callback_args.op_1.is_neox = is_neox != 0;
  full_callback_args.op_1.normalize_qk = normalize_qk != 0;
  cutlass::gemm::GemmCoord full_problem_size(N, full_rows, K);
  params.full_output_op = FullCallbacks::to_underlying_arguments(
      full_problem_size, full_callback_args, nullptr);

  typename ResidualCallbacks::Arguments residual_callback_args{};
  residual_callback_args.op_1.output = output;
  residual_callback_args.op_1.correction_base = dense_base;
  residual_callback_args.op_1.q_weight = q_weight;
  residual_callback_args.op_1.k_weight = k_weight;
  residual_callback_args.op_1.cos_sin_cache = cos_sin_cache;
  residual_callback_args.op_1.position_ids = position_ids;
  residual_callback_args.op_1.row_indices = dense_rows;
  residual_callback_args.op_1.q_size = q_size;
  residual_callback_args.op_1.kv_size = kv_size;
  residual_callback_args.op_1.logical_rows = dense_count;
  residual_callback_args.op_1.output_rows = full_rows;
  residual_callback_args.op_1.dense_rows = dense_count;
  residual_callback_args.op_1.rotary_dim = rotary_dim;
  residual_callback_args.op_1.epsilon = epsilon;
  residual_callback_args.op_1.is_neox = is_neox != 0;
  residual_callback_args.op_1.normalize_qk = normalize_qk != 0;
  cutlass::gemm::GemmCoord residual_problem_size(N, dense_count, K);
  params.residual_output_op = ResidualCallbacks::to_underlying_arguments(
      residual_problem_size, residual_callback_args, nullptr);

  int feature_tiles =
      (N + Kernel::FullThreadblockShape::kM - 1) /
      Kernel::FullThreadblockShape::kM;
  params.full_row_tiles =
      (full_rows + Kernel::FullThreadblockShape::kN - 1) /
      Kernel::FullThreadblockShape::kN;
  params.residual_row_tiles =
      (dense_count + Kernel::ResidualThreadblockShape::kN - 1) /
      Kernel::ResidualThreadblockShape::kN;
  params.first_problem_tiles = feature_tiles * params.full_row_tiles;
  int residual_tiles = feature_tiles * params.residual_row_tiles;
  params.total_tiles = params.first_problem_tiles + residual_tiles;

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount,
      smem_size);
  if (active_blocks < 1) {
    active_blocks = 1;
  }
  int persistent_blocks = sm_count * active_blocks;
  int grid = params.total_tiles < persistent_blocks ? params.total_tiles
                                                     : persistent_blocks;
  if (worker_blocks > 0 && worker_blocks < grid) {
    grid = worker_blocks;
  }
  if (grid < 2) {
    return -8;
  }

  int residual_workers = requested_residual_workers;
  if (residual_workers == 0) {
    residual_workers =
        (grid * residual_tiles + params.total_tiles - 1) /
        params.total_tiles;
  }
  residual_workers = residual_workers < 1 ? 1 : residual_workers;
  residual_workers = residual_workers > residual_tiles ? residual_tiles
                                                        : residual_workers;
  residual_workers = residual_workers >= grid ? grid - 1 : residual_workers;
  int full_workers = grid - residual_workers;
  if (full_workers < 1) {
    full_workers = 1;
    residual_workers = grid - 1;
  }
  if (full_workers > params.first_problem_tiles) {
    full_workers = params.first_problem_tiles;
    residual_workers = grid - full_workers;
  }
  if (residual_workers < 1) {
    return -9;
  }
  params.full_worker_blocks = full_workers;
  params.residual_worker_blocks = residual_workers;

  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename GateDeviceGemm, typename DownDeviceGemm,
          bool SparseDown = false, bool DynamicDownOwners = false,
          bool GlobalStageBarrier = false>
int sparse24_cutlass_gate_dense_down_pipeline_gemm_run(
    const Element *x, const Element *gate_values,
    DeviceElementE *gate_meta, Element *hidden,
    const Element *down_weight, DeviceElementE *down_meta,
    Element *output, int rows,
    int model_width, int intermediate_size, int *row_counters,
    int worker_blocks, int stage_mode, cudaStream_t stream) {
  using Kernel = Sparse24GateDenseDownPersistentKernel<
      GateDeviceGemm, DownDeviceGemm, SparseDown, DynamicDownOwners,
      GlobalStageBarrier>;
  using GateCallbacks = typename Kernel::GateBaseKernel::FusionCallbacks;
  using DownCallbacks = typename Kernel::DownBaseKernel::FusionCallbacks;
  typename Kernel::Params params;
  params.x = x;
  params.gate_values = gate_values;
  params.gate_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(gate_meta);
  params.hidden = hidden;
  params.down_weight = down_weight;
  params.down_metadata = down_meta;
  params.output = output;
  params.rows = rows;
  params.model_width = model_width;
  params.intermediate_size = intermediate_size;
  params.gate_output_size = 2 * intermediate_size;
  params.stage_mode = stage_mode;
  params.row_counters = row_counters;

  typename GateCallbacks::Arguments gate_callback_args{};
  gate_callback_args.op_1.output = hidden;
  gate_callback_args.op_1.hidden_size = intermediate_size;
  gate_callback_args.op_1.logical_rows = rows;
  cutlass::gemm::GemmCoord gate_problem_size(
      params.gate_output_size, rows, model_width);
  params.gate_output_op = GateCallbacks::to_underlying_arguments(
      gate_problem_size, gate_callback_args, nullptr);

  typename DownCallbacks::Arguments down_callback_args{};
  down_callback_args.op_1.output = output;
  down_callback_args.op_1.output_columns = model_width;
  down_callback_args.op_1.logical_rows = rows;
  cutlass::gemm::GemmCoord down_problem_size(
      model_width, rows, intermediate_size);
  params.down_output_op = DownCallbacks::to_underlying_arguments(
      down_problem_size, down_callback_args, nullptr);

  params.gate_feature_tiles =
      (params.gate_output_size + Kernel::GateShape::kM - 1) /
      Kernel::GateShape::kM;
  params.down_feature_tiles =
      (model_width + Kernel::DownShape::kM - 1) /
      Kernel::DownShape::kM;
  params.gate_row_tiles =
      (rows + Kernel::GateShape::kN - 1) / Kernel::GateShape::kN;
  params.down_row_tiles =
      (rows + Kernel::DownShape::kN - 1) / Kernel::DownShape::kN;
  params.gate_tiles =
      params.gate_feature_tiles * params.gate_row_tiles;
  params.down_tiles =
      params.down_feature_tiles * params.down_row_tiles;
  int total_tiles = params.gate_tiles + params.down_tiles;

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount,
      smem_size);
  if (active_blocks < 1) {
    active_blocks = 1;
  }
  int persistent_blocks = sm_count * active_blocks;
  int grid = total_tiles < persistent_blocks ? total_tiles
                                              : persistent_blocks;
  if (worker_blocks > 0 && worker_blocks < grid) {
    grid = worker_blocks;
  }
  if (grid < 2) {
    return -8;
  }
  params.gate_workers = grid;
  params.down_workers = grid;

  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

// Full W24 and compact R24 tiles write separate outputs without a grid-wide
// barrier. The generic path lets the last CTA for each feature tile add the
// residual. The QKV specialization instead uses reserved full/residual worker
// partitions: residual CTAs add their own dense-row tiles, then full CTAs apply
// QKV post-ops to their own row tiles. This preserves row-level parallelism and
// avoids a second epilogue launch.
template <typename FullDeviceGemm, typename ResidualDeviceGemm,
          bool FinalizeQkvPostop = false>
int sparse24_cutlass_paired_finalize_residual_visitor_gemm_run(
    const Element *x, const Element *full_values,
    DeviceElementE *full_meta, const Element *residual_values,
    DeviceElementE *residual_meta, const int *dense_rows, int dense_count,
    Element *full_output, Element *residual_output, int full_rows, int K,
    int N, int *feature_counters, int worker_blocks, int schedule_mode,
    cudaStream_t stream, const Element *q_weight = nullptr,
    const Element *k_weight = nullptr,
    const Element *cos_sin_cache = nullptr,
    const int64_t *position_ids = nullptr, int q_size = 0, int kv_size = 0,
    int rotary_dim = 0, float epsilon = 0.0f, int is_neox = 0,
    int normalize_qk = 0) {
  using Kernel = Sparse24PairedPersistentVisitorKernel<
      FullDeviceGemm, ResidualDeviceGemm, true, false, false, true, false,
      false, FinalizeQkvPostop>;
  using FullCallbacks = typename Kernel::FullBaseKernel::FusionCallbacks;
  using ResidualCallbacks =
      typename Kernel::ResidualBaseKernel::FusionCallbacks;
  typename Kernel::Params params;
  params.problems[0] = {
      x, full_values, reinterpret_cast<typename Kernel::ElementE *>(full_meta),
      nullptr, full_rows};
  params.problems[1] = {
      x, residual_values,
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta), dense_rows,
      dense_count};
  params.K = K;
  params.N = N;
  params.interleaved_schedule = schedule_mode;
  params.feature_counters = feature_counters;
  if constexpr (FinalizeQkvPostop) {
    params.q_weight = q_weight;
    params.k_weight = k_weight;
    params.cos_sin_cache = cos_sin_cache;
    params.position_ids = position_ids;
    params.q_size = q_size;
    params.kv_size = kv_size;
    params.rotary_dim = rotary_dim;
    params.epsilon = epsilon;
    params.is_neox = is_neox;
    params.normalize_qk = normalize_qk;
  }

  typename FullCallbacks::Arguments full_callback_args{};
  full_callback_args.op_1.output = full_output;
  full_callback_args.op_1.output_columns = N;
  full_callback_args.op_1.logical_rows = full_rows;
  cutlass::gemm::GemmCoord full_problem_size(N, full_rows, K);
  params.full_output_op = FullCallbacks::to_underlying_arguments(
      full_problem_size, full_callback_args, nullptr);

  typename ResidualCallbacks::Arguments residual_callback_args{};
  residual_callback_args.op_1.output = residual_output;
  residual_callback_args.op_1.output_columns = N;
  residual_callback_args.op_1.logical_rows = dense_count;
  cutlass::gemm::GemmCoord residual_problem_size(N, dense_count, K);
  params.residual_output_op = ResidualCallbacks::to_underlying_arguments(
      residual_problem_size, residual_callback_args, nullptr);

  int feature_tiles =
      (N + Kernel::FullThreadblockShape::kM - 1) /
      Kernel::FullThreadblockShape::kM;
  params.full_row_tiles =
      (full_rows + Kernel::FullThreadblockShape::kN - 1) /
      Kernel::FullThreadblockShape::kN;
  params.residual_row_tiles =
      (dense_count + Kernel::ResidualThreadblockShape::kN - 1) /
      Kernel::ResidualThreadblockShape::kN;
  params.first_problem_tiles = feature_tiles * params.full_row_tiles;
  int residual_tiles = feature_tiles * params.residual_row_tiles;
  params.total_tiles = params.first_problem_tiles + residual_tiles;

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount,
      smem_size);
  if (active_blocks < 1) {
    active_blocks = 1;
  }
  int persistent_blocks = sm_count * active_blocks;
  int grid = params.total_tiles < persistent_blocks ? params.total_tiles
                                                     : persistent_blocks;
  if (worker_blocks > 0 && worker_blocks < grid) {
    grid = worker_blocks;
  }
  if (grid < 2) {
    return -8;
  }

  int residual_workers =
      (grid * residual_tiles + params.total_tiles - 1) / params.total_tiles;
  residual_workers = residual_workers < 1 ? 1 : residual_workers;
  residual_workers = residual_workers > residual_tiles ? residual_tiles
                                                        : residual_workers;
  int full_workers = grid - residual_workers;
  if (full_workers < 1) {
    full_workers = 1;
    residual_workers = grid - 1;
  }
  if (full_workers > params.first_problem_tiles) {
    full_workers = params.first_problem_tiles;
    residual_workers = grid - full_workers;
  }
  if (residual_workers < 1) {
    return -9;
  }
  params.full_worker_blocks = full_workers;
  params.residual_worker_blocks = residual_workers;

  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

// The normal path keeps concurrent W24/R24 worker groups. The last-owner
// specialization lets the final W24 tile for each feature run its R24 tile
// directly, so no resident CTA is parked waiting for full-row completion.
template <typename FullDeviceGemm, typename ResidualDeviceGemm,
          bool LastFullTileResidual = false>
int sparse24_cutlass_paired_inplace_residual_visitor_gemm_run(
    const Element *x, const Element *full_values,
    DeviceElementE *full_meta, const Element *residual_values,
    DeviceElementE *residual_meta, const int *dense_rows, int dense_count,
    Element *output, int full_rows, int K, int N, int *feature_counters,
    int worker_blocks, int schedule_mode, cudaStream_t stream) {
  using Kernel = Sparse24PairedPersistentVisitorKernel<
      FullDeviceGemm, ResidualDeviceGemm, true, true, false, false, false,
      LastFullTileResidual>;
  using FullCallbacks = typename Kernel::FullBaseKernel::FusionCallbacks;
  using ResidualCallbacks =
      typename Kernel::ResidualBaseKernel::FusionCallbacks;
  typename Kernel::Params params;
  params.problems[0] = {
      x, full_values, reinterpret_cast<typename Kernel::ElementE *>(full_meta),
      nullptr, full_rows};
  params.problems[1] = {
      x, residual_values,
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta), dense_rows,
      dense_count};
  params.K = K;
  params.N = N;
  params.interleaved_schedule = schedule_mode;
  params.feature_counters = feature_counters;

  typename FullCallbacks::Arguments full_callback_args{};
  full_callback_args.op_1.output = output;
  full_callback_args.op_1.output_columns = N;
  full_callback_args.op_1.logical_rows = full_rows;
  cutlass::gemm::GemmCoord full_problem_size(N, full_rows, K);
  params.full_output_op = FullCallbacks::to_underlying_arguments(
      full_problem_size, full_callback_args, nullptr);

  typename ResidualCallbacks::Arguments residual_callback_args{};
  residual_callback_args.op_1.output = output;
  residual_callback_args.op_1.output_columns = N;
  residual_callback_args.op_1.logical_rows = dense_count;
  residual_callback_args.op_1.output_rows = full_rows;
  residual_callback_args.op_1.row_indices = dense_rows;
  cutlass::gemm::GemmCoord residual_problem_size(N, dense_count, K);
  params.residual_output_op = ResidualCallbacks::to_underlying_arguments(
      residual_problem_size, residual_callback_args, nullptr);

  int feature_tiles =
      (N + Kernel::FullThreadblockShape::kM - 1) /
      Kernel::FullThreadblockShape::kM;
  params.full_row_tiles =
      (full_rows + Kernel::FullThreadblockShape::kN - 1) /
      Kernel::FullThreadblockShape::kN;
  params.residual_row_tiles =
      (dense_count + Kernel::ResidualThreadblockShape::kN - 1) /
      Kernel::ResidualThreadblockShape::kN;
  params.first_problem_tiles = feature_tiles * params.full_row_tiles;
  int residual_tiles = feature_tiles * params.residual_row_tiles;
  params.total_tiles = params.first_problem_tiles + residual_tiles;

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount,
      smem_size);
  if (active_blocks < 1) {
    active_blocks = 1;
  }
  int persistent_blocks = sm_count * active_blocks;
  int launch_tiles = LastFullTileResidual ? params.first_problem_tiles
                                          : params.total_tiles;
  int grid = launch_tiles < persistent_blocks ? launch_tiles
                                               : persistent_blocks;
  if (worker_blocks > 0 && worker_blocks < grid) {
    grid = worker_blocks;
  }
  if (grid < 2) {
    return -8;
  }

  if constexpr (LastFullTileResidual) {
    params.full_worker_blocks = grid;
    params.residual_worker_blocks = 0;
  } else {
    int residual_workers =
        (grid * residual_tiles + params.total_tiles - 1) /
        params.total_tiles;
    residual_workers = residual_workers < 1 ? 1 : residual_workers;
    residual_workers = residual_workers > residual_tiles ? residual_tiles
                                                          : residual_workers;
    int full_workers = grid - residual_workers;
    if (full_workers < 1) {
      full_workers = 1;
      residual_workers = grid - 1;
    }
    if (full_workers > params.first_problem_tiles) {
      full_workers = params.first_problem_tiles;
      residual_workers = grid - full_workers;
    }
    if (residual_workers < 1) {
      return -9;
    }
    params.full_worker_blocks = full_workers;
    params.residual_worker_blocks = residual_workers;
  }

  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename BaseDeviceGemm>
int sparse24_cutlass_gather_gemm_run(
    const Element *x, const Element *values, DeviceElementE *metadata,
    const int *row_indices, Element *output, int rows, int K, int N,
    cudaStream_t stream) {
  using Kernel = Sparse24GatherGemmKernel<BaseDeviceGemm>;
  typename Kernel::Params params;
  params.x = x;
  params.values = values;
  params.metadata =
      reinterpret_cast<typename Kernel::ElementE *>(metadata);
  params.row_indices = row_indices;
  params.output = output;
  params.rows = rows;
  params.output_leading_rows = (rows + 7) / 8 * 8;
  params.K = K;
  params.N = N;

  int feature_tiles =
      (N + Kernel::ThreadblockShape::kM - 1) /
      Kernel::ThreadblockShape::kM;
  int row_tiles =
      (rows + Kernel::ThreadblockShape::kN - 1) /
      Kernel::ThreadblockShape::kN;
  int grid = feature_tiles * row_tiles;
  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename FullDeviceGemm, typename ResidualDeviceGemm,
          bool GatherResidualRows = false>
int sparse24_cutlass_paired_persistent_routed_swiglu_gemm_run(
    const Element *full_x, const Element *full_values,
    DeviceElementE *full_meta, Element *full_output, Element *dense_base,
    const int *dense_slot_by_row, int full_rows, int dense_rows,
    const Element *residual_x, const int *residual_row_indices,
    const Element *residual_values,
    DeviceElementE *residual_meta, Element *residual_output,
    int residual_rows, int K, int N, int interleaved_schedule,
    int requested_grid,
    cudaStream_t stream) {
  using Kernel = Sparse24PairedPersistentVisitorKernel<
      FullDeviceGemm, ResidualDeviceGemm, GatherResidualRows>;
  using FullCallbacks = typename Kernel::FullBaseKernel::FusionCallbacks;
  using ResidualCallbacks =
      typename Kernel::ResidualBaseKernel::FusionCallbacks;
  typename Kernel::Params params;
  params.problems[0] = {
      full_x, full_values,
      reinterpret_cast<typename Kernel::ElementE *>(full_meta), nullptr,
      full_rows};
  params.problems[1] = {
      residual_x, residual_values,
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta),
      GatherResidualRows ? residual_row_indices : nullptr, residual_rows};
  params.K = K;
  params.N = N;
  params.interleaved_schedule = interleaved_schedule;

  typename FullCallbacks::Arguments full_callback_args{};
  full_callback_args.op_1.output = full_output;
  full_callback_args.op_1.dense_base = dense_base;
  full_callback_args.op_1.hidden_size = N / 2;
  full_callback_args.op_1.logical_rows = full_rows;
  full_callback_args.op_1.dense_slot_by_row = dense_slot_by_row;
  full_callback_args.op_1.dense_rows = dense_rows;
  cutlass::gemm::GemmCoord full_problem_size(N, full_rows, K);
  params.full_output_op = FullCallbacks::to_underlying_arguments(
      full_problem_size, full_callback_args, nullptr);

  typename ResidualCallbacks::Arguments residual_callback_args{};
  residual_callback_args.op_1.output = residual_output;
  residual_callback_args.op_1.output_columns = N;
  residual_callback_args.op_1.logical_rows = residual_rows;
  cutlass::gemm::GemmCoord residual_problem_size(N, residual_rows, K);
  params.residual_output_op = ResidualCallbacks::to_underlying_arguments(
      residual_problem_size, residual_callback_args, nullptr);

  int full_feature_tiles =
      (N + Kernel::FullThreadblockShape::kM - 1) /
      Kernel::FullThreadblockShape::kM;
  int full_row_tiles =
      (full_rows + Kernel::FullThreadblockShape::kN - 1) /
      Kernel::FullThreadblockShape::kN;
  int residual_feature_tiles =
      (N + Kernel::ResidualThreadblockShape::kM - 1) /
      Kernel::ResidualThreadblockShape::kM;
  int residual_row_tiles =
      (residual_rows + Kernel::ResidualThreadblockShape::kN - 1) /
      Kernel::ResidualThreadblockShape::kN;
  params.first_problem_tiles = full_feature_tiles * full_row_tiles;
  params.total_tiles = params.first_problem_tiles +
                       residual_feature_tiles * residual_row_tiles;

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount, smem_size);
  if (active_blocks < 1) {
    active_blocks = 1;
  }
  int persistent_blocks = sm_count * active_blocks;
  int grid = params.total_tiles < persistent_blocks ? params.total_tiles
                                                     : persistent_blocks;
  if (requested_grid > 0) {
    grid = requested_grid < params.total_tiles ? requested_grid
                                                : params.total_tiles;
  }
  int residual_tiles = params.total_tiles - params.first_problem_tiles;
  int residual_workers =
      (grid * residual_tiles + params.total_tiles - 1) / params.total_tiles;
  residual_workers = residual_workers < 1 ? 1 : residual_workers;
  residual_workers = residual_workers > residual_tiles ? residual_tiles
                                                        : residual_workers;
  int full_workers = grid - residual_workers;
  if (full_workers < 1) {
    full_workers = 1;
    residual_workers = grid - 1;
  }
  if (full_workers > params.first_problem_tiles) {
    full_workers = params.first_problem_tiles;
    residual_workers = grid - full_workers;
  }
  params.full_worker_blocks = full_workers;
  params.residual_worker_blocks = residual_workers;

  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

// Run the common W24 Gate/Up projection and the complementary R24 projection
// in one persistent grid. The residual mainloop can either gather routed rows
// directly or consume a pre-compacted input. Once every full row tile for a
// feature tile has stored its compact dense base, the residual epilogue applies
// exact SwiGLU and scatters directly into the routed output rows. Both modes
// avoid a raw residual output and correction launch.
template <typename FullDeviceGemm, typename ResidualDeviceGemm,
          bool GatherResidualRows>
int sparse24_cutlass_paired_fused_routed_swiglu_gemm_run(
    const Element *x, const Element *full_values,
    DeviceElementE *full_meta, const Element *residual_x,
    const Element *residual_values,
    DeviceElementE *residual_meta, const int *dense_rows,
    const int *dense_slot_by_row, int dense_count, Element *output,
    Element *dense_base, int full_rows, int residual_rows, int K, int N,
    int *feature_counters, int worker_blocks, int schedule_mode,
    cudaStream_t stream) {
  using Kernel = Sparse24PairedPersistentVisitorKernel<
      FullDeviceGemm, ResidualDeviceGemm, GatherResidualRows, true>;
  using FullCallbacks = typename Kernel::FullBaseKernel::FusionCallbacks;
  using ResidualCallbacks =
      typename Kernel::ResidualBaseKernel::FusionCallbacks;
  typename Kernel::Params params;
  params.problems[0] = {
      x, full_values, reinterpret_cast<typename Kernel::ElementE *>(full_meta),
      nullptr, full_rows};
  params.problems[1] = {
      residual_x, residual_values,
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta),
      GatherResidualRows ? dense_rows : nullptr, residual_rows};
  params.K = K;
  params.N = N;
  params.interleaved_schedule = schedule_mode;
  params.feature_counters = feature_counters;

  typename FullCallbacks::Arguments full_callback_args{};
  full_callback_args.op_1.output = output;
  full_callback_args.op_1.dense_base = dense_base;
  full_callback_args.op_1.hidden_size = N / 2;
  full_callback_args.op_1.logical_rows = full_rows;
  full_callback_args.op_1.dense_slot_by_row = dense_slot_by_row;
  full_callback_args.op_1.dense_rows = dense_count;
  cutlass::gemm::GemmCoord full_problem_size(N, full_rows, K);
  params.full_output_op = FullCallbacks::to_underlying_arguments(
      full_problem_size, full_callback_args, nullptr);

  typename ResidualCallbacks::Arguments residual_callback_args{};
  residual_callback_args.op_1.output = output;
  residual_callback_args.op_1.correction_base = dense_base;
  residual_callback_args.op_1.hidden_size = N / 2;
  residual_callback_args.op_1.logical_rows = dense_count;
  residual_callback_args.op_1.output_rows = full_rows;
  residual_callback_args.op_1.row_indices = dense_rows;
  cutlass::gemm::GemmCoord residual_problem_size(N, residual_rows, K);
  params.residual_output_op = ResidualCallbacks::to_underlying_arguments(
      residual_problem_size, residual_callback_args, nullptr);

  int feature_tiles =
      (N + Kernel::FullThreadblockShape::kM - 1) /
      Kernel::FullThreadblockShape::kM;
  params.full_row_tiles =
      (full_rows + Kernel::FullThreadblockShape::kN - 1) /
      Kernel::FullThreadblockShape::kN;
  params.residual_row_tiles =
      (residual_rows + Kernel::ResidualThreadblockShape::kN - 1) /
      Kernel::ResidualThreadblockShape::kN;
  params.first_problem_tiles = feature_tiles * params.full_row_tiles;
  int residual_tiles = feature_tiles * params.residual_row_tiles;
  params.total_tiles = params.first_problem_tiles + residual_tiles;

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount, smem_size);
  if (active_blocks < 1) {
    active_blocks = 1;
  }
  int persistent_blocks = sm_count * active_blocks;
  int grid = params.total_tiles < persistent_blocks ? params.total_tiles
                                                     : persistent_blocks;
  if (worker_blocks > 0 && worker_blocks < grid) {
    grid = worker_blocks;
  }
  if (grid < 2) {
    return -8;
  }

  int residual_workers =
      (grid * residual_tiles + params.total_tiles - 1) / params.total_tiles;
  residual_workers = residual_workers < 1 ? 1 : residual_workers;
  residual_workers = residual_workers > residual_tiles ? residual_tiles
                                                        : residual_workers;
  int full_workers = grid - residual_workers;
  if (full_workers < 1) {
    full_workers = 1;
    residual_workers = grid - 1;
  }
  if (full_workers > params.first_problem_tiles) {
    full_workers = params.first_problem_tiles;
    residual_workers = grid - full_workers;
  }
  if (residual_workers < 1) {
    return -9;
  }
  params.full_worker_blocks = full_workers;
  params.residual_worker_blocks = residual_workers;

  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename GateFullDeviceGemm, typename GateResidualDeviceGemm,
          typename DownFullDeviceGemm, typename DownResidualDeviceGemm>
int sparse24_cutlass_fused_mixed_mlp_gemm_run(
    const Element *x, const Element *gate_full_values,
    DeviceElementE *gate_full_meta, const Element *gate_residual_values,
    DeviceElementE *gate_residual_meta, const Element *down_full_values,
    DeviceElementE *down_full_meta, const Element *down_residual_values,
    DeviceElementE *down_residual_meta, const int *dense_rows,
    const int *dense_slot_by_row, Element *hidden, Element *gate_dense_base,
    Element *output, int *gate_feature_counters,
    int *down_feature_counters, int *grid_barrier, int rows, int dense_count,
    int model_width, int intermediate_size, int requested_grid,
    cudaStream_t stream) {
  using Kernel = Sparse24FusedMixedMlpKernel<
      GateFullDeviceGemm, GateResidualDeviceGemm, DownFullDeviceGemm,
      DownResidualDeviceGemm>;
  using GateKernel = typename Kernel::GateKernel;
  using DownKernel = typename Kernel::DownKernel;
  using GateFullCallbacks =
      typename GateKernel::FullBaseKernel::FusionCallbacks;
  using GateResidualCallbacks =
      typename GateKernel::ResidualBaseKernel::FusionCallbacks;
  using DownFullCallbacks =
      typename DownKernel::FullBaseKernel::FusionCallbacks;
  using DownResidualCallbacks =
      typename DownKernel::ResidualBaseKernel::FusionCallbacks;

  typename Kernel::Params params{};
  params.grid_barrier = grid_barrier;

  int gate_output_size = 2 * intermediate_size;
  params.gate.problems[0] = {
      x, gate_full_values,
      reinterpret_cast<typename GateKernel::ElementE *>(gate_full_meta),
      nullptr, rows};
  params.gate.problems[1] = {
      x, gate_residual_values,
      reinterpret_cast<typename GateKernel::ElementE *>(gate_residual_meta),
      dense_rows, dense_count};
  params.gate.K = model_width;
  params.gate.N = gate_output_size;
  params.gate.interleaved_schedule = 0;
  params.gate.feature_counters = gate_feature_counters;

  typename GateFullCallbacks::Arguments gate_full_callback_args{};
  gate_full_callback_args.op_1.output = hidden;
  gate_full_callback_args.op_1.dense_base = gate_dense_base;
  gate_full_callback_args.op_1.hidden_size = intermediate_size;
  gate_full_callback_args.op_1.logical_rows = rows;
  gate_full_callback_args.op_1.dense_slot_by_row = dense_slot_by_row;
  gate_full_callback_args.op_1.dense_rows = dense_count;
  cutlass::gemm::GemmCoord gate_full_problem_size(
      gate_output_size, rows, model_width);
  params.gate.full_output_op = GateFullCallbacks::to_underlying_arguments(
      gate_full_problem_size, gate_full_callback_args, nullptr);

  typename GateResidualCallbacks::Arguments gate_residual_callback_args{};
  gate_residual_callback_args.op_1.output = hidden;
  gate_residual_callback_args.op_1.correction_base = gate_dense_base;
  gate_residual_callback_args.op_1.hidden_size = intermediate_size;
  gate_residual_callback_args.op_1.logical_rows = dense_count;
  gate_residual_callback_args.op_1.output_rows = rows;
  gate_residual_callback_args.op_1.row_indices = dense_rows;
  cutlass::gemm::GemmCoord gate_residual_problem_size(
      gate_output_size, dense_count, model_width);
  params.gate.residual_output_op =
      GateResidualCallbacks::to_underlying_arguments(
          gate_residual_problem_size, gate_residual_callback_args, nullptr);

  int gate_feature_tiles =
      (gate_output_size + GateKernel::FullThreadblockShape::kM - 1) /
      GateKernel::FullThreadblockShape::kM;
  params.gate.full_row_tiles =
      (rows + GateKernel::FullThreadblockShape::kN - 1) /
      GateKernel::FullThreadblockShape::kN;
  params.gate.residual_row_tiles =
      (dense_count + GateKernel::ResidualThreadblockShape::kN - 1) /
      GateKernel::ResidualThreadblockShape::kN;
  params.gate.first_problem_tiles =
      gate_feature_tiles * params.gate.full_row_tiles;
  int gate_residual_tiles =
      gate_feature_tiles * params.gate.residual_row_tiles;
  params.gate.total_tiles =
      params.gate.first_problem_tiles + gate_residual_tiles;

  params.down.problems[0] = {
      hidden, down_full_values,
      reinterpret_cast<typename DownKernel::ElementE *>(down_full_meta),
      nullptr, rows};
  params.down.problems[1] = {
      hidden, down_residual_values,
      reinterpret_cast<typename DownKernel::ElementE *>(down_residual_meta),
      dense_rows, dense_count};
  params.down.K = intermediate_size;
  params.down.N = model_width;
  params.down.interleaved_schedule = 0;
  params.down.feature_counters = down_feature_counters;

  typename DownFullCallbacks::Arguments down_full_callback_args{};
  down_full_callback_args.op_1.output = output;
  down_full_callback_args.op_1.output_columns = model_width;
  down_full_callback_args.op_1.logical_rows = rows;
  cutlass::gemm::GemmCoord down_full_problem_size(
      model_width, rows, intermediate_size);
  params.down.full_output_op = DownFullCallbacks::to_underlying_arguments(
      down_full_problem_size, down_full_callback_args, nullptr);

  typename DownResidualCallbacks::Arguments down_residual_callback_args{};
  down_residual_callback_args.op_1.output = output;
  down_residual_callback_args.op_1.output_columns = model_width;
  down_residual_callback_args.op_1.logical_rows = dense_count;
  down_residual_callback_args.op_1.output_rows = rows;
  down_residual_callback_args.op_1.row_indices = dense_rows;
  cutlass::gemm::GemmCoord down_residual_problem_size(
      model_width, dense_count, intermediate_size);
  params.down.residual_output_op =
      DownResidualCallbacks::to_underlying_arguments(
          down_residual_problem_size, down_residual_callback_args, nullptr);

  int down_feature_tiles =
      (model_width + DownKernel::FullThreadblockShape::kM - 1) /
      DownKernel::FullThreadblockShape::kM;
  params.down.full_row_tiles =
      (rows + DownKernel::FullThreadblockShape::kN - 1) /
      DownKernel::FullThreadblockShape::kN;
  params.down.residual_row_tiles =
      (dense_count + DownKernel::ResidualThreadblockShape::kN - 1) /
      DownKernel::ResidualThreadblockShape::kN;
  params.down.first_problem_tiles =
      down_feature_tiles * params.down.full_row_tiles;
  int down_residual_tiles =
      down_feature_tiles * params.down.residual_row_tiles;
  params.down.total_tiles =
      params.down.first_problem_tiles + down_residual_tiles;

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount,
      smem_size);
  if (active_blocks < 1) {
    active_blocks = 1;
  }
  int persistent_blocks = sm_count * active_blocks;
  int launch_tiles = params.gate.total_tiles > params.down.total_tiles
                         ? params.gate.total_tiles
                         : params.down.total_tiles;
  int grid = launch_tiles < persistent_blocks ? launch_tiles
                                               : persistent_blocks;
  if (requested_grid > 0 && requested_grid < grid) {
    grid = requested_grid;
  }
  if (grid < 2) {
    return -8;
  }

  int gate_residual_workers =
      (grid * gate_residual_tiles + params.gate.total_tiles - 1) /
      params.gate.total_tiles;
  gate_residual_workers = gate_residual_workers < 1 ? 1
                                                     : gate_residual_workers;
  gate_residual_workers = gate_residual_workers >= grid
                              ? grid - 1
                              : gate_residual_workers;
  params.gate.residual_worker_blocks = gate_residual_workers;
  params.gate.full_worker_blocks = grid - gate_residual_workers;

  int down_residual_workers =
      (grid * down_residual_tiles + params.down.total_tiles - 1) /
      params.down.total_tiles;
  down_residual_workers = down_residual_workers < 1 ? 1
                                                     : down_residual_workers;
  down_residual_workers = down_residual_workers >= grid
                              ? grid - 1
                              : down_residual_workers;
  params.down.residual_worker_blocks = down_residual_workers;
  params.down.full_worker_blocks = grid - down_residual_workers;

  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename FullDeviceGemm, typename ResidualDeviceGemm>
int sparse24_cutlass_paired_self_contained_routed_swiglu_gemm_run(
    const Element *x, const Element *full_values,
    DeviceElementE *full_meta, const Element *residual_values,
    DeviceElementE *residual_meta, const int *dense_rows,
    const int *dense_slot_by_row, int dense_count, Element *output,
    Element *dense_base, int full_rows, int K, int N, int requested_grid,
    cudaStream_t stream) {
  using Kernel = Sparse24PairedPersistentVisitorKernel<
      FullDeviceGemm, ResidualDeviceGemm, true, false, false, false, true>;
  using FullCallbacks = typename Kernel::FullBaseKernel::FusionCallbacks;
  using ResidualCallbacks =
      typename Kernel::ResidualBaseKernel::FusionCallbacks;
  typename Kernel::Params params;
  params.problems[0] = {
      x, full_values, reinterpret_cast<typename Kernel::ElementE *>(full_meta),
      nullptr, full_rows};
  params.problems[1] = {
      x, residual_values,
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta),
      dense_rows, dense_count};
  params.K = K;
  params.N = N;
  params.interleaved_schedule = 0;

  typename FullCallbacks::Arguments full_callback_args{};
  full_callback_args.op_1.output = output;
  full_callback_args.op_1.dense_base = nullptr;
  full_callback_args.op_1.hidden_size = N / 2;
  full_callback_args.op_1.logical_rows = full_rows;
  full_callback_args.op_1.dense_slot_by_row = dense_slot_by_row;
  full_callback_args.op_1.dense_rows = dense_count;
  cutlass::gemm::GemmCoord full_problem_size(N, full_rows, K);
  params.full_output_op = FullCallbacks::to_underlying_arguments(
      full_problem_size, full_callback_args, nullptr);

  cutlass::gemm::GemmCoord residual_problem_size(N, dense_count, K);
  typename ResidualCallbacks::Arguments residual_base_callback_args{};
  residual_base_callback_args.op_1.output = output;
  residual_base_callback_args.op_1.dense_base = dense_base;
  residual_base_callback_args.op_1.correction_base = nullptr;
  residual_base_callback_args.op_1.hidden_size = N / 2;
  residual_base_callback_args.op_1.logical_rows = dense_count;
  residual_base_callback_args.op_1.output_rows = full_rows;
  residual_base_callback_args.op_1.row_indices = dense_rows;
  residual_base_callback_args.op_1.dense_rows = dense_count;
  params.residual_base_output_op =
      ResidualCallbacks::to_underlying_arguments(
          residual_problem_size, residual_base_callback_args, nullptr);

  typename ResidualCallbacks::Arguments residual_callback_args{};
  residual_callback_args.op_1.output = output;
  residual_callback_args.op_1.dense_base = nullptr;
  residual_callback_args.op_1.correction_base = dense_base;
  residual_callback_args.op_1.hidden_size = N / 2;
  residual_callback_args.op_1.logical_rows = dense_count;
  residual_callback_args.op_1.output_rows = full_rows;
  residual_callback_args.op_1.row_indices = dense_rows;
  residual_callback_args.op_1.dense_rows = dense_count;
  params.residual_output_op = ResidualCallbacks::to_underlying_arguments(
      residual_problem_size, residual_callback_args, nullptr);

  int full_feature_tiles =
      (N + Kernel::FullThreadblockShape::kM - 1) /
      Kernel::FullThreadblockShape::kM;
  int full_row_tiles =
      (full_rows + Kernel::FullThreadblockShape::kN - 1) /
      Kernel::FullThreadblockShape::kN;
  int residual_feature_tiles =
      (N + Kernel::ResidualThreadblockShape::kM - 1) /
      Kernel::ResidualThreadblockShape::kM;
  int residual_row_tiles =
      (dense_count + Kernel::ResidualThreadblockShape::kN - 1) /
      Kernel::ResidualThreadblockShape::kN;
  params.first_problem_tiles = full_feature_tiles * full_row_tiles;
  int residual_tiles = residual_feature_tiles * residual_row_tiles;
  params.total_tiles = params.first_problem_tiles + residual_tiles;

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount, smem_size);
  if (active_blocks < 1) {
    active_blocks = 1;
  }
  int persistent_blocks = sm_count * active_blocks;
  int grid = params.total_tiles < persistent_blocks ? params.total_tiles
                                                     : persistent_blocks;
  if (requested_grid > 0 && requested_grid < grid) {
    grid = requested_grid;
  }
  if (grid < 2) {
    return -8;
  }

  int weighted_residual_tiles = residual_tiles * 2;
  int weighted_total = params.first_problem_tiles + weighted_residual_tiles;
  int residual_workers =
      (grid * weighted_residual_tiles + weighted_total - 1) / weighted_total;
  residual_workers = residual_workers < 1 ? 1 : residual_workers;
  residual_workers = residual_workers > residual_tiles ? residual_tiles
                                                        : residual_workers;
  int full_workers = grid - residual_workers;
  if (full_workers < 1) {
    full_workers = 1;
    residual_workers = grid - 1;
  }
  if (full_workers > params.first_problem_tiles) {
    full_workers = params.first_problem_tiles;
    residual_workers = grid - full_workers;
  }
  if (residual_workers < 1) {
    return -9;
  }
  params.full_worker_blocks = full_workers;
  params.residual_worker_blocks = residual_workers;

  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

// The full W24 route omits confidence-selected dense rows. A residual worker
// owns each compact row tile and runs W24 followed by R24, then its epilogue
// adds both fragments and scatters one exact Down value. This removes the
// feature-counter dependency required by an in-place residual add.
template <typename FullDeviceGemm, typename ResidualDeviceGemm>
int sparse24_cutlass_paired_self_contained_exact_down_gemm_run(
    const Element *x, const Element *full_values,
    DeviceElementE *full_meta, const Element *residual_values,
    DeviceElementE *residual_meta, const int *dense_rows,
    const int *dense_slot_by_row, int dense_count, Element *output,
    Element *dense_base, int full_rows, int K, int N, int requested_grid,
    cudaStream_t stream) {
  using Kernel = Sparse24PairedPersistentVisitorKernel<
      FullDeviceGemm, ResidualDeviceGemm, true, false, false, false, true>;
  using FullCallbacks = typename Kernel::FullBaseKernel::FusionCallbacks;
  using ResidualCallbacks =
      typename Kernel::ResidualBaseKernel::FusionCallbacks;
  typename Kernel::Params params;
  params.problems[0] = {
      x, full_values, reinterpret_cast<typename Kernel::ElementE *>(full_meta),
      nullptr, full_rows};
  params.problems[1] = {
      x, residual_values,
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta),
      dense_rows, dense_count};
  params.K = K;
  params.N = N;
  params.interleaved_schedule = 0;

  typename FullCallbacks::Arguments full_callback_args{};
  full_callback_args.op_1.output = output;
  full_callback_args.op_1.dense_base = nullptr;
  full_callback_args.op_1.output_columns = N;
  full_callback_args.op_1.logical_rows = full_rows;
  full_callback_args.op_1.output_rows = full_rows;
  full_callback_args.op_1.dense_slot_by_row = dense_slot_by_row;
  full_callback_args.op_1.dense_rows = dense_count;
  cutlass::gemm::GemmCoord full_problem_size(N, full_rows, K);
  params.full_output_op = FullCallbacks::to_underlying_arguments(
      full_problem_size, full_callback_args, nullptr);

  cutlass::gemm::GemmCoord residual_problem_size(N, dense_count, K);
  typename ResidualCallbacks::Arguments residual_base_callback_args{};
  residual_base_callback_args.op_1.output = output;
  residual_base_callback_args.op_1.dense_base = dense_base;
  residual_base_callback_args.op_1.routed_residual = nullptr;
  residual_base_callback_args.op_1.output_columns = N;
  residual_base_callback_args.op_1.logical_rows = dense_count;
  residual_base_callback_args.op_1.output_rows = full_rows;
  residual_base_callback_args.op_1.row_indices = dense_rows;
  params.residual_base_output_op =
      ResidualCallbacks::to_underlying_arguments(
          residual_problem_size, residual_base_callback_args, nullptr);

  typename ResidualCallbacks::Arguments residual_callback_args{};
  residual_callback_args.op_1.output = output;
  residual_callback_args.op_1.dense_base = nullptr;
  residual_callback_args.op_1.routed_residual = dense_base;
  residual_callback_args.op_1.output_columns = N;
  residual_callback_args.op_1.logical_rows = dense_count;
  residual_callback_args.op_1.output_rows = full_rows;
  residual_callback_args.op_1.row_indices = dense_rows;
  params.residual_output_op = ResidualCallbacks::to_underlying_arguments(
      residual_problem_size, residual_callback_args, nullptr);

  int full_feature_tiles =
      (N + Kernel::FullThreadblockShape::kM - 1) /
      Kernel::FullThreadblockShape::kM;
  int full_row_tiles =
      (full_rows + Kernel::FullThreadblockShape::kN - 1) /
      Kernel::FullThreadblockShape::kN;
  int residual_feature_tiles =
      (N + Kernel::ResidualThreadblockShape::kM - 1) /
      Kernel::ResidualThreadblockShape::kM;
  int residual_row_tiles =
      (dense_count + Kernel::ResidualThreadblockShape::kN - 1) /
      Kernel::ResidualThreadblockShape::kN;
  params.first_problem_tiles = full_feature_tiles * full_row_tiles;
  int residual_tiles = residual_feature_tiles * residual_row_tiles;
  params.total_tiles = params.first_problem_tiles + residual_tiles;

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount, smem_size);
  if (active_blocks < 1) {
    active_blocks = 1;
  }
  int persistent_blocks = sm_count * active_blocks;
  int grid = params.total_tiles < persistent_blocks ? params.total_tiles
                                                     : persistent_blocks;
  if (requested_grid > 0 && requested_grid < grid) {
    grid = requested_grid;
  }
  if (grid < 2) {
    return -8;
  }

  int weighted_residual_tiles = residual_tiles * 2;
  int weighted_total = params.first_problem_tiles + weighted_residual_tiles;
  int residual_workers =
      (grid * weighted_residual_tiles + weighted_total - 1) / weighted_total;
  residual_workers = residual_workers < 1 ? 1 : residual_workers;
  residual_workers = residual_workers > residual_tiles ? residual_tiles
                                                        : residual_workers;
  int full_workers = grid - residual_workers;
  if (full_workers < 1) {
    full_workers = 1;
    residual_workers = grid - 1;
  }
  if (full_workers > params.first_problem_tiles) {
    full_workers = params.first_problem_tiles;
    residual_workers = grid - full_workers;
  }
  if (residual_workers < 1) {
    return -9;
  }
  params.full_worker_blocks = full_workers;
  params.residual_worker_blocks = residual_workers;

  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename Gemm>
int sparse24_cutlass_inline_transpose_gemm_run(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    Element *output, int M, int K, int N, cudaStream_t stream) {
  int sparse_k = K / Gemm::kSparse;
  int columns_e = K / Gemm::kSparse / Gemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);

  typename Gemm::FusionCallbacks::Arguments callback_args{};
  callback_args.op_1.output = output;
  callback_args.op_1.output_columns = N;
  callback_args.op_1.logical_rows = M;
  typename Gemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, cutlass::layout::ColumnMajor(K)},
      {reinterpret_cast<typename Gemm::ElementE *>(a_meta_e),
       Gemm::LayoutE::packed({N, columns_e})},
      callback_args};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename Gemm, typename LayoutB>
int sparse24_cutlass_inline_indexed_transpose_gemm_run(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    Element *output, const int *row_indices, int M, int logical_rows,
    int output_rows, int K, int N, int ldb, cudaStream_t stream) {
  int sparse_k = K / Gemm::kSparse;
  int columns_e = K / Gemm::kSparse / Gemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);

  typename Gemm::FusionCallbacks::Arguments callback_args{};
  callback_args.op_1.output = output;
  callback_args.op_1.output_columns = N;
  callback_args.op_1.logical_rows = logical_rows;
  callback_args.op_1.output_rows = output_rows;
  callback_args.op_1.row_indices = row_indices;
  typename Gemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, LayoutB(ldb)},
      {reinterpret_cast<typename Gemm::ElementE *>(a_meta_e),
       Gemm::LayoutE::packed({N, columns_e})},
      callback_args};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename Kernel>
int sparse24_routed_exact_launch(
    typename Kernel::Params &params, int sparse_tiles, int dense_tiles,
    cudaStream_t stream, int dense_tile_weight = 2) {
  if (sparse_tiles <= 0 || dense_tiles <= 0) {
    return -7;
  }

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }

  int device = 0;
  int sm_count = 1;
  int active_blocks = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount, smem_size);
  active_blocks = active_blocks < 1 ? 1 : active_blocks;

  int total_tiles = sparse_tiles + dense_tiles;
  int grid = total_tiles < sm_count * active_blocks
                 ? total_tiles
                 : sm_count * active_blocks;
  int weighted_tiles = sparse_tiles + dense_tile_weight * dense_tiles;
  int dense_workers =
      (grid * dense_tile_weight * dense_tiles + weighted_tiles - 1) /
      weighted_tiles;
  dense_workers = dense_workers < 1 ? 1 : dense_workers;
  dense_workers = dense_workers > dense_tiles ? dense_tiles : dense_workers;
  dense_workers = dense_workers >= grid ? grid - 1 : dense_workers;
  int sparse_workers = grid - dense_workers;
  if (sparse_workers > sparse_tiles) {
    int excess = sparse_workers - sparse_tiles;
    sparse_workers -= excess;
    dense_workers += excess;
  }
  if (dense_workers > dense_tiles) {
    int excess = dense_workers - dense_tiles;
    dense_workers -= excess;
    sparse_workers += excess;
  }
  if (sparse_workers <= 0 || dense_workers <= 0) {
    return -8;
  }

  params.sparse_tiles = sparse_tiles;
  params.dense_tiles = dense_tiles;
  params.sparse_workers = sparse_workers;
  params.dense_workers = dense_workers;
  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<sparse_workers + dense_workers, Kernel::kThreadCount, smem_size,
         stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename Kernel>
int sparse24_heterogeneous_direct_launch(
    typename Kernel::Params &params, int sparse_tiles, int dense_tiles,
    cudaStream_t stream) {
  if (sparse_tiles <= 0 || dense_tiles <= 0) {
    return -7;
  }
  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  params.sparse_tiles = sparse_tiles;
  params.dense_tiles = dense_tiles;
  params.sparse_workers = sparse_tiles;
  params.dense_workers = dense_tiles;
  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<sparse_tiles + dense_tiles, Kernel::kThreadCount, smem_size,
         stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename Kernel>
int sparse24_heterogeneous_component_launch(
    typename Kernel::Params &params, int tiles, bool dense_component,
    cudaStream_t stream) {
  if (tiles <= 0) {
    return -7;
  }
  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  params.sparse_tiles = dense_component ? 0 : tiles;
  params.dense_tiles = dense_component ? tiles : 0;
  params.sparse_workers = dense_component ? 0 : tiles;
  params.dense_workers = dense_component ? tiles : 0;
  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<tiles, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename SparseGemm, typename DenseGemm>
int sparse24_cutlass_routed_exact_linear_run(
    const Element *x, const Element *full_values, DeviceElementE *full_meta,
    const Element *residual_values, DeviceElementE *residual_meta,
    const int *dense_rows, int dense_count, const int *sparse_rows,
    int sparse_count, Element *output, int output_rows, int K, int N,
    cudaStream_t stream) {
  using Kernel =
      Sparse24RoutedExactVisitorKernel<SparseGemm, DenseGemm>;
  typename Kernel::Params params;
  params.x = x;
  params.full_values = full_values;
  params.full_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(full_meta);
  params.residual_values = residual_values;
  params.residual_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta);
  params.sparse_problem = {sparse_rows, sparse_count};
  params.dense_problem = {dense_rows, dense_count};
  params.K = K;
  params.N = N;

  typename SparseGemm::FusionCallbacks::Arguments sparse_callback_args{};
  sparse_callback_args.op_1.output = output;
  sparse_callback_args.op_1.output_columns = N;
  sparse_callback_args.op_1.logical_rows = sparse_count;
  sparse_callback_args.op_1.output_rows = output_rows;
  sparse_callback_args.op_1.row_indices = sparse_rows;
  cutlass::gemm::GemmCoord sparse_problem_size(N, sparse_count, K);
  params.sparse_output_op = SparseGemm::FusionCallbacks::to_underlying_arguments(
      sparse_problem_size, sparse_callback_args, nullptr);

  typename DenseGemm::FusionCallbacks::Arguments dense_callback_args{};
  dense_callback_args.op_1.output = output;
  dense_callback_args.op_1.output_columns = N;
  dense_callback_args.op_1.logical_rows = dense_count;
  dense_callback_args.op_1.output_rows = output_rows;
  dense_callback_args.op_1.row_indices = dense_rows;
  cutlass::gemm::GemmCoord dense_problem_size(N, dense_count, K);
  params.dense_output_op = DenseGemm::FusionCallbacks::to_underlying_arguments(
      dense_problem_size, dense_callback_args, nullptr);

  using SparseShape = typename SparseGemm::GemmKernel::Mma::Shape;
  using DenseShape = typename DenseGemm::GemmKernel::Mma::Shape;
  int sparse_tiles =
      ((N + SparseShape::kM - 1) / SparseShape::kM) *
      ((sparse_count + SparseShape::kN - 1) / SparseShape::kN);
  int dense_tiles =
      ((N + DenseShape::kM - 1) / DenseShape::kM) *
      ((dense_count + DenseShape::kN - 1) / DenseShape::kN);
  return sparse24_routed_exact_launch<Kernel>(params, sparse_tiles,
                                               dense_tiles, stream);
}

template <typename SparseGemm, typename DenseGemm>
int sparse24_cutlass_heterogeneous_linear_run(
    const Element *x, const Element *sparse_values,
    DeviceElementE *sparse_meta, const Element *dense_weight,
    const int *dense_rows, int dense_count, const int *sparse_rows,
    int sparse_count, Element *output, int output_rows, int K, int N,
    bool direct_workers, int dense_tile_weight, cudaStream_t stream) {
  using Kernel =
      Sparse24HeterogeneousRoutedVisitorKernel<SparseGemm, DenseGemm>;
  typename Kernel::Params params;
  params.x = x;
  params.sparse_values = sparse_values;
  params.sparse_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(sparse_meta);
  params.dense_weight = dense_weight;
  params.sparse_problem = {sparse_rows, sparse_count};
  params.dense_problem = {dense_rows, dense_count};
  params.K = K;
  params.N = N;

  typename SparseGemm::FusionCallbacks::Arguments sparse_callback_args{};
  sparse_callback_args.op_1.output = output;
  sparse_callback_args.op_1.output_columns = N;
  sparse_callback_args.op_1.logical_rows = sparse_count;
  sparse_callback_args.op_1.output_rows = output_rows;
  sparse_callback_args.op_1.row_indices = sparse_rows;
  cutlass::gemm::GemmCoord sparse_problem_size(N, sparse_count, K);
  params.sparse_output_op = SparseGemm::FusionCallbacks::to_underlying_arguments(
      sparse_problem_size, sparse_callback_args, nullptr);

  typename DenseGemm::FusionCallbacks::Arguments dense_callback_args{};
  dense_callback_args.op_1.output = output;
  dense_callback_args.op_1.output_columns = N;
  dense_callback_args.op_1.logical_rows = dense_count;
  dense_callback_args.op_1.output_rows = output_rows;
  dense_callback_args.op_1.row_indices = dense_rows;
  cutlass::gemm::GemmCoord dense_problem_size(N, dense_count, K);
  params.dense_output_op = DenseGemm::FusionCallbacks::to_underlying_arguments(
      dense_problem_size, dense_callback_args, nullptr);

  using SparseShape = typename SparseGemm::GemmKernel::Mma::Shape;
  using DenseShape = typename DenseGemm::GemmKernel::Mma::Shape;
  int sparse_tiles =
      ((N + SparseShape::kM - 1) / SparseShape::kM) *
      ((sparse_count + SparseShape::kN - 1) / SparseShape::kN);
  int dense_tiles =
      ((N + DenseShape::kM - 1) / DenseShape::kM) *
      ((dense_count + DenseShape::kN - 1) / DenseShape::kN);
  if (direct_workers) {
    return sparse24_heterogeneous_direct_launch<Kernel>(
        params, sparse_tiles, dense_tiles, stream);
  }
  return sparse24_routed_exact_launch<Kernel>(params, sparse_tiles,
                                               dense_tiles, stream,
                                               dense_tile_weight);
}

template <typename SparseGemm, typename DenseGemm>
int sparse24_cutlass_heterogeneous_swiglu_run(
    const Element *x, const Element *sparse_values,
    DeviceElementE *sparse_meta, const Element *dense_weight,
    const int *dense_weight_rows, const int *dense_rows, int dense_count,
    const int *sparse_rows, int sparse_count, Element *output,
    int output_rows, int K, int N, cudaStream_t stream) {
  using Kernel = Sparse24HeterogeneousRoutedVisitorKernel<
      SparseGemm, DenseGemm, true>;
  typename Kernel::Params params;
  params.x = x;
  params.sparse_values = sparse_values;
  params.sparse_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(sparse_meta);
  params.dense_weight = dense_weight;
  params.dense_weight_rows = dense_weight_rows;
  params.sparse_problem = {sparse_rows, sparse_count};
  params.dense_problem = {dense_rows, dense_count};
  params.K = K;
  params.N = N;

  typename SparseGemm::FusionCallbacks::Arguments sparse_callback_args{};
  sparse_callback_args.op_1.output = output;
  sparse_callback_args.op_1.hidden_size = N / 2;
  sparse_callback_args.op_1.logical_rows = sparse_count;
  sparse_callback_args.op_1.output_rows = output_rows;
  sparse_callback_args.op_1.row_indices = sparse_rows;
  cutlass::gemm::GemmCoord sparse_problem_size(N, sparse_count, K);
  params.sparse_output_op =
      SparseGemm::FusionCallbacks::to_underlying_arguments(
          sparse_problem_size, sparse_callback_args, nullptr);

  typename DenseGemm::FusionCallbacks::Arguments dense_callback_args{};
  dense_callback_args.op_1.output = output;
  dense_callback_args.op_1.hidden_size = N / 2;
  dense_callback_args.op_1.logical_rows = dense_count;
  dense_callback_args.op_1.output_rows = output_rows;
  dense_callback_args.op_1.row_indices = dense_rows;
  cutlass::gemm::GemmCoord dense_problem_size(N, dense_count, K);
  params.dense_output_op =
      DenseGemm::FusionCallbacks::to_underlying_arguments(
          dense_problem_size, dense_callback_args, nullptr);

  using SparseShape = typename SparseGemm::GemmKernel::Mma::Shape;
  using DenseShape = typename DenseGemm::GemmKernel::Mma::Shape;
  int sparse_tiles =
      ((N + SparseShape::kM - 1) / SparseShape::kM) *
      ((sparse_count + SparseShape::kN - 1) / SparseShape::kN);
  int dense_tiles =
      ((N + DenseShape::kM - 1) / DenseShape::kM) *
      ((dense_count + DenseShape::kN - 1) / DenseShape::kN);
  return sparse24_routed_exact_launch<Kernel>(params, sparse_tiles,
                                               dense_tiles, stream, 1);
}

// Compute the fast contiguous W24 Gate/Up path for every verifier row while a
// second worker group computes the exact dense Gate/Up result only for routed
// rows. The routed sparse epilogue skips those rows, so both worker groups can
// apply SwiGLU directly into one output without a write race or a correction
// launch.
template <typename SparseGemm, typename DenseGemm>
int sparse24_cutlass_full_sparse_dense_override_swiglu_run(
    const Element *x, const Element *sparse_values,
    DeviceElementE *sparse_meta, const Element *dense_weight,
    const int *dense_weight_rows, const int *dense_rows,
    const int *dense_slot_by_row, int dense_count, Element *output,
    int output_rows, int K, int N, int dense_tile_weight,
    cudaStream_t stream) {
  using Kernel = Sparse24HeterogeneousRoutedVisitorKernel<
      SparseGemm, DenseGemm, true, false>;
  typename Kernel::Params params;
  params.x = x;
  params.sparse_values = sparse_values;
  params.sparse_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(sparse_meta);
  params.dense_weight = dense_weight;
  params.dense_weight_rows = dense_weight_rows;
  params.sparse_problem = {nullptr, output_rows};
  params.dense_problem = {dense_rows, dense_count};
  params.K = K;
  params.N = N;

  typename SparseGemm::FusionCallbacks::Arguments sparse_callback_args{};
  sparse_callback_args.op_1.output = output;
  sparse_callback_args.op_1.dense_base = nullptr;
  sparse_callback_args.op_1.hidden_size = N / 2;
  sparse_callback_args.op_1.logical_rows = output_rows;
  sparse_callback_args.op_1.output_rows = output_rows;
  sparse_callback_args.op_1.dense_slot_by_row = dense_slot_by_row;
  sparse_callback_args.op_1.dense_rows = dense_count;
  cutlass::gemm::GemmCoord sparse_problem_size(N, output_rows, K);
  params.sparse_output_op =
      SparseGemm::FusionCallbacks::to_underlying_arguments(
          sparse_problem_size, sparse_callback_args, nullptr);

  typename DenseGemm::FusionCallbacks::Arguments dense_callback_args{};
  dense_callback_args.op_1.output = output;
  dense_callback_args.op_1.hidden_size = N / 2;
  dense_callback_args.op_1.logical_rows = dense_count;
  dense_callback_args.op_1.output_rows = output_rows;
  dense_callback_args.op_1.row_indices = dense_rows;
  cutlass::gemm::GemmCoord dense_problem_size(N, dense_count, K);
  params.dense_output_op =
      DenseGemm::FusionCallbacks::to_underlying_arguments(
          dense_problem_size, dense_callback_args, nullptr);

  using SparseShape = typename SparseGemm::GemmKernel::Mma::Shape;
  using DenseShape = typename DenseGemm::GemmKernel::Mma::Shape;
  int sparse_tiles =
      ((N + SparseShape::kM - 1) / SparseShape::kM) *
      ((output_rows + SparseShape::kN - 1) / SparseShape::kN);
  int dense_tiles =
      ((N + DenseShape::kM - 1) / DenseShape::kM) *
      ((dense_count + DenseShape::kN - 1) / DenseShape::kN);
  return sparse24_routed_exact_launch<Kernel>(params, sparse_tiles,
                                               dense_tiles, stream,
                                               dense_tile_weight);
}

// Linear counterpart of the fused SwiGLU override above. The contiguous W24
// path owns sparse-row stores, while the gathered dense path owns exact
// confidence-selected rows. This is useful for Down and QKV projections where
// materializing compact sparse activations costs more than the extra W24 work
// on a small dense-row budget.
template <typename SparseGemm, typename DenseGemm>
int sparse24_cutlass_full_sparse_dense_override_linear_run(
    const Element *x, const Element *sparse_values,
    DeviceElementE *sparse_meta, const Element *dense_weight,
    const int *dense_rows, const int *dense_slot_by_row, int dense_count,
    Element *output, int output_rows, int K, int N, int dense_tile_weight,
    cudaStream_t stream) {
  using Kernel = Sparse24HeterogeneousRoutedVisitorKernel<
      SparseGemm, DenseGemm, false, false>;
  typename Kernel::Params params;
  params.x = x;
  params.sparse_values = sparse_values;
  params.sparse_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(sparse_meta);
  params.dense_weight = dense_weight;
  params.sparse_problem = {nullptr, output_rows};
  params.dense_problem = {dense_rows, dense_count};
  params.K = K;
  params.N = N;

  typename SparseGemm::FusionCallbacks::Arguments sparse_callback_args{};
  sparse_callback_args.op_1.output = output;
  sparse_callback_args.op_1.dense_base = nullptr;
  sparse_callback_args.op_1.output_columns = N;
  sparse_callback_args.op_1.logical_rows = output_rows;
  sparse_callback_args.op_1.output_rows = output_rows;
  sparse_callback_args.op_1.dense_slot_by_row = dense_slot_by_row;
  sparse_callback_args.op_1.dense_rows = dense_count;
  cutlass::gemm::GemmCoord sparse_problem_size(N, output_rows, K);
  params.sparse_output_op =
      SparseGemm::FusionCallbacks::to_underlying_arguments(
          sparse_problem_size, sparse_callback_args, nullptr);

  typename DenseGemm::FusionCallbacks::Arguments dense_callback_args{};
  dense_callback_args.op_1.output = output;
  dense_callback_args.op_1.output_columns = N;
  dense_callback_args.op_1.logical_rows = dense_count;
  dense_callback_args.op_1.output_rows = output_rows;
  dense_callback_args.op_1.row_indices = dense_rows;
  cutlass::gemm::GemmCoord dense_problem_size(N, dense_count, K);
  params.dense_output_op =
      DenseGemm::FusionCallbacks::to_underlying_arguments(
          dense_problem_size, dense_callback_args, nullptr);

  using SparseShape = typename SparseGemm::GemmKernel::Mma::Shape;
  using DenseShape = typename DenseGemm::GemmKernel::Mma::Shape;
  int sparse_tiles =
      ((N + SparseShape::kM - 1) / SparseShape::kM) *
      ((output_rows + SparseShape::kN - 1) / SparseShape::kN);
  int dense_tiles =
      ((N + DenseShape::kM - 1) / DenseShape::kM) *
      ((dense_count + DenseShape::kN - 1) / DenseShape::kN);
  return sparse24_routed_exact_launch<Kernel>(params, sparse_tiles,
                                               dense_tiles, stream,
                                               dense_tile_weight);
}

template <typename SparseGemm, typename DenseGemm>
int sparse24_cutlass_heterogeneous_component_run(
    const Element *x, const Element *sparse_values,
    DeviceElementE *sparse_meta, const Element *dense_weight,
    const int *route_rows, int route_count, Element *output, int output_rows,
    int K, int N, bool dense_component, cudaStream_t stream) {
  using Kernel =
      Sparse24HeterogeneousRoutedVisitorKernel<SparseGemm, DenseGemm>;
  typename Kernel::Params params;
  params.x = x;
  params.K = K;
  params.N = N;
  int tiles = 0;
  if (dense_component) {
    params.dense_weight = dense_weight;
    params.dense_problem = {route_rows, route_count};
    typename DenseGemm::FusionCallbacks::Arguments callback_args{};
    callback_args.op_1.output = output;
    callback_args.op_1.output_columns = N;
    callback_args.op_1.logical_rows = route_count;
    callback_args.op_1.output_rows = output_rows;
    callback_args.op_1.row_indices = route_rows;
    cutlass::gemm::GemmCoord problem_size(N, route_count, K);
    params.dense_output_op =
        DenseGemm::FusionCallbacks::to_underlying_arguments(
            problem_size, callback_args, nullptr);
    using Shape = typename DenseGemm::GemmKernel::Mma::Shape;
    tiles = ((N + Shape::kM - 1) / Shape::kM) *
            ((route_count + Shape::kN - 1) / Shape::kN);
  } else {
    params.sparse_values = sparse_values;
    params.sparse_metadata =
        reinterpret_cast<typename Kernel::ElementE *>(sparse_meta);
    params.sparse_problem = {route_rows, route_count};
    typename SparseGemm::FusionCallbacks::Arguments callback_args{};
    callback_args.op_1.output = output;
    callback_args.op_1.output_columns = N;
    callback_args.op_1.logical_rows = route_count;
    callback_args.op_1.output_rows = output_rows;
    callback_args.op_1.row_indices = route_rows;
    cutlass::gemm::GemmCoord problem_size(N, route_count, K);
    params.sparse_output_op =
        SparseGemm::FusionCallbacks::to_underlying_arguments(
            problem_size, callback_args, nullptr);
    using Shape = typename SparseGemm::GemmKernel::Mma::Shape;
    tiles = ((N + Shape::kM - 1) / Shape::kM) *
            ((route_count + Shape::kN - 1) / Shape::kN);
  }
  return sparse24_heterogeneous_component_launch<Kernel>(
      params, tiles, dense_component, stream);
}

template <typename SparseGemm, typename DenseGemm>
int sparse24_cutlass_routed_exact_swiglu_run(
    const Element *x, const Element *full_values, DeviceElementE *full_meta,
    const Element *residual_values, DeviceElementE *residual_meta,
    const int *dense_rows, int dense_count, const int *sparse_rows,
    int sparse_count, Element *output, int output_rows, int K, int N,
    cudaStream_t stream) {
  using Kernel =
      Sparse24RoutedExactVisitorKernel<SparseGemm, DenseGemm>;
  typename Kernel::Params params;
  params.x = x;
  params.full_values = full_values;
  params.full_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(full_meta);
  params.residual_values = residual_values;
  params.residual_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta);
  params.sparse_problem = {sparse_rows, sparse_count};
  params.dense_problem = {dense_rows, dense_count};
  params.K = K;
  params.N = N;

  typename SparseGemm::FusionCallbacks::Arguments sparse_callback_args{};
  sparse_callback_args.op_1.output = output;
  sparse_callback_args.op_1.hidden_size = N / 2;
  sparse_callback_args.op_1.logical_rows = sparse_count;
  sparse_callback_args.op_1.output_rows = output_rows;
  sparse_callback_args.op_1.row_indices = sparse_rows;
  cutlass::gemm::GemmCoord sparse_problem_size(N, sparse_count, K);
  params.sparse_output_op = SparseGemm::FusionCallbacks::to_underlying_arguments(
      sparse_problem_size, sparse_callback_args, nullptr);

  typename DenseGemm::FusionCallbacks::Arguments dense_callback_args{};
  dense_callback_args.op_1.output = output;
  dense_callback_args.op_1.hidden_size = N / 2;
  dense_callback_args.op_1.logical_rows = dense_count;
  dense_callback_args.op_1.output_rows = output_rows;
  dense_callback_args.op_1.row_indices = dense_rows;
  cutlass::gemm::GemmCoord dense_problem_size(N, dense_count, K);
  params.dense_output_op = DenseGemm::FusionCallbacks::to_underlying_arguments(
      dense_problem_size, dense_callback_args, nullptr);

  using SparseShape = typename SparseGemm::GemmKernel::Mma::Shape;
  using DenseShape = typename DenseGemm::GemmKernel::Mma::Shape;
  int sparse_tiles =
      ((N + SparseShape::kM - 1) / SparseShape::kM) *
      ((sparse_count + SparseShape::kN - 1) / SparseShape::kN);
  int dense_tiles =
      ((N + DenseShape::kM - 1) / DenseShape::kM) *
      ((dense_count + DenseShape::kN - 1) / DenseShape::kN);
  return sparse24_routed_exact_launch<Kernel>(params, sparse_tiles,
                                               dense_tiles, stream);
}

template <typename Kernel>
int sparse24_grouped_owner_launch(typename Kernel::Params &params,
                                  cudaStream_t stream) {
  using Shape = typename Kernel::ThreadblockShape;
  int feature_tiles = (params.N + Shape::kM - 1) / Shape::kM;
  int full_row_tiles =
      (params.full_rows + Shape::kN - 1) / Shape::kN;
  params.owner_groups =
      (full_row_tiles + params.group_tiles - 1) / params.group_tiles;
  int grid = feature_tiles * params.owner_groups;
  if (grid <= 0) {
    return -7;
  }

  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel><<<grid, Kernel::kThreadCount, smem_size, stream>>>(
      params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename FullGemm, typename ResidualGemm>
int sparse24_cutlass_grouped_owner_linear_run(
    const Element *x, const Element *full_values, DeviceElementE *full_meta,
    const Element *residual_values, DeviceElementE *residual_meta,
    const int *dense_rows, int dense_count, Element *output, int output_rows,
    int K, int N, int group_tiles, cudaStream_t stream) {
  using Kernel =
      Sparse24GroupedOwnerVisitorKernel<FullGemm, ResidualGemm, false>;
  typename Kernel::Params params;
  params.x = x;
  params.full_values = full_values;
  params.full_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(full_meta);
  params.residual_values = residual_values;
  params.residual_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta);
  params.dense_rows = dense_rows;
  params.dense_count = dense_count;
  params.full_rows = output_rows;
  params.K = K;
  params.N = N;
  params.group_tiles = group_tiles;

  typename FullGemm::FusionCallbacks::Arguments full_callback_args{};
  full_callback_args.op_1.output = output;
  full_callback_args.op_1.output_columns = N;
  full_callback_args.op_1.logical_rows = output_rows;
  cutlass::gemm::GemmCoord full_problem_size(N, output_rows, K);
  params.full_output_op = FullGemm::FusionCallbacks::to_underlying_arguments(
      full_problem_size, full_callback_args, nullptr);

  typename ResidualGemm::FusionCallbacks::Arguments residual_callback_args{};
  residual_callback_args.op_1.output = output;
  residual_callback_args.op_1.output_columns = N;
  residual_callback_args.op_1.logical_rows = dense_count;
  residual_callback_args.op_1.output_rows = output_rows;
  residual_callback_args.op_1.row_indices = dense_rows;
  cutlass::gemm::GemmCoord residual_problem_size(N, dense_count, K);
  params.residual_output_op =
      ResidualGemm::FusionCallbacks::to_underlying_arguments(
          residual_problem_size, residual_callback_args, nullptr);
  return sparse24_grouped_owner_launch<Kernel>(params, stream);
}

template <typename FullGemm, typename ResidualGemm>
int sparse24_cutlass_grouped_owner_swiglu_run(
    const Element *x, const Element *full_values, DeviceElementE *full_meta,
    const Element *residual_values, DeviceElementE *residual_meta,
    const int *dense_rows, const int *dense_slot_by_row, int dense_count,
    Element *dense_base, Element *output, int output_rows, int K, int N,
    int group_tiles, cudaStream_t stream) {
  using Kernel =
      Sparse24GroupedOwnerVisitorKernel<FullGemm, ResidualGemm, true>;
  typename Kernel::Params params;
  params.x = x;
  params.full_values = full_values;
  params.full_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(full_meta);
  params.residual_values = residual_values;
  params.residual_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta);
  params.dense_rows = dense_rows;
  params.dense_count = dense_count;
  params.full_rows = output_rows;
  params.K = K;
  params.N = N;
  params.group_tiles = group_tiles;

  typename FullGemm::FusionCallbacks::Arguments full_callback_args{};
  full_callback_args.op_1.output = output;
  full_callback_args.op_1.dense_base = dense_base;
  full_callback_args.op_1.hidden_size = N / 2;
  full_callback_args.op_1.logical_rows = output_rows;
  full_callback_args.op_1.output_rows = output_rows;
  full_callback_args.op_1.dense_slot_by_row = dense_slot_by_row;
  full_callback_args.op_1.dense_rows = dense_count;
  cutlass::gemm::GemmCoord full_problem_size(N, output_rows, K);
  params.full_output_op = FullGemm::FusionCallbacks::to_underlying_arguments(
      full_problem_size, full_callback_args, nullptr);

  typename ResidualGemm::FusionCallbacks::Arguments residual_callback_args{};
  residual_callback_args.op_1.output = output;
  residual_callback_args.op_1.correction_base = dense_base;
  residual_callback_args.op_1.compact_output = nullptr;
  residual_callback_args.op_1.hidden_size = N / 2;
  residual_callback_args.op_1.logical_rows = dense_count;
  residual_callback_args.op_1.output_rows = output_rows;
  residual_callback_args.op_1.row_indices = dense_rows;
  cutlass::gemm::GemmCoord residual_problem_size(N, dense_count, K);
  params.residual_output_op =
      ResidualGemm::FusionCallbacks::to_underlying_arguments(
          residual_problem_size, residual_callback_args, nullptr);
  return sparse24_grouped_owner_launch<Kernel>(params, stream);
}

template <typename FullGemm, typename ResidualGemm>
int sparse24_cutlass_grouped_owner_qkv_run(
    const Element *x, const Element *full_values, DeviceElementE *full_meta,
    const Element *residual_values, DeviceElementE *residual_meta,
    const int *dense_rows, const int *dense_slot_by_row, int dense_count,
    Element *dense_base, Element *output, int output_rows, int K, int N,
    const Element *q_weight, const Element *k_weight,
    const Element *cos_sin_cache, const int64_t *position_ids,
    int q_size, int kv_size, int rotary_dim, float epsilon, int is_neox,
    int normalize_qk, int group_tiles, cudaStream_t stream) {
  using Kernel =
      Sparse24GroupedOwnerVisitorKernel<FullGemm, ResidualGemm, true>;
  typename Kernel::Params params;
  params.x = x;
  params.full_values = full_values;
  params.full_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(full_meta);
  params.residual_values = residual_values;
  params.residual_metadata =
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta);
  params.dense_rows = dense_rows;
  params.dense_count = dense_count;
  params.full_rows = output_rows;
  params.K = K;
  params.N = N;
  params.group_tiles = group_tiles;

  typename FullGemm::FusionCallbacks::Arguments full_callback_args{};
  full_callback_args.op_1.output = output;
  full_callback_args.op_1.dense_base = dense_base;
  full_callback_args.op_1.q_weight = q_weight;
  full_callback_args.op_1.k_weight = k_weight;
  full_callback_args.op_1.cos_sin_cache = cos_sin_cache;
  full_callback_args.op_1.position_ids = position_ids;
  full_callback_args.op_1.dense_slot_by_row = dense_slot_by_row;
  full_callback_args.op_1.q_size = q_size;
  full_callback_args.op_1.kv_size = kv_size;
  full_callback_args.op_1.logical_rows = output_rows;
  full_callback_args.op_1.output_rows = output_rows;
  full_callback_args.op_1.dense_rows = dense_count;
  full_callback_args.op_1.rotary_dim = rotary_dim;
  full_callback_args.op_1.epsilon = epsilon;
  full_callback_args.op_1.is_neox = is_neox != 0;
  full_callback_args.op_1.normalize_qk = normalize_qk != 0;
  cutlass::gemm::GemmCoord full_problem_size(N, output_rows, K);
  params.full_output_op = FullGemm::FusionCallbacks::to_underlying_arguments(
      full_problem_size, full_callback_args, nullptr);

  typename ResidualGemm::FusionCallbacks::Arguments residual_callback_args{};
  residual_callback_args.op_1.output = output;
  residual_callback_args.op_1.correction_base = dense_base;
  residual_callback_args.op_1.q_weight = q_weight;
  residual_callback_args.op_1.k_weight = k_weight;
  residual_callback_args.op_1.cos_sin_cache = cos_sin_cache;
  residual_callback_args.op_1.position_ids = position_ids;
  residual_callback_args.op_1.row_indices = dense_rows;
  residual_callback_args.op_1.q_size = q_size;
  residual_callback_args.op_1.kv_size = kv_size;
  residual_callback_args.op_1.logical_rows = dense_count;
  residual_callback_args.op_1.output_rows = output_rows;
  residual_callback_args.op_1.dense_rows = dense_count;
  residual_callback_args.op_1.rotary_dim = rotary_dim;
  residual_callback_args.op_1.epsilon = epsilon;
  residual_callback_args.op_1.is_neox = is_neox != 0;
  residual_callback_args.op_1.normalize_qk = normalize_qk != 0;
  cutlass::gemm::GemmCoord residual_problem_size(N, dense_count, K);
  params.residual_output_op =
      ResidualGemm::FusionCallbacks::to_underlying_arguments(
          residual_problem_size, residual_callback_args, nullptr);
  return sparse24_grouped_owner_launch<Kernel>(params, stream);
}

template <typename Gemm>
int sparse24_cutlass_inline_routed_transpose_gemm_run(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    Element *output, Element *dense_base, const int *dense_slot_by_row,
    int M, int dense_rows, int K, int N, cudaStream_t stream) {
  int sparse_k = K / Gemm::kSparse;
  int columns_e = K / Gemm::kSparse / Gemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);

  typename Gemm::FusionCallbacks::Arguments callback_args{};
  callback_args.op_1.output = output;
  callback_args.op_1.dense_base = dense_base;
  callback_args.op_1.output_columns = N;
  callback_args.op_1.logical_rows = M;
  callback_args.op_1.dense_slot_by_row = dense_slot_by_row;
  callback_args.op_1.dense_rows = dense_rows;
  typename Gemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, cutlass::layout::ColumnMajor(K)},
      {reinterpret_cast<typename Gemm::ElementE *>(a_meta_e),
       Gemm::LayoutE::packed({N, columns_e})},
      callback_args};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename Gemm>
int sparse24_cutlass_routed_residual_epilogue_gemm_run(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    const Element *routed_residual, const int *dense_slot_by_row,
    Element *output, int M, int dense_rows, int K, int N,
    cudaStream_t stream) {
  int sparse_k = K / Gemm::kSparse;
  int columns_e = K / Gemm::kSparse / Gemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);

  typename Gemm::FusionCallbacks::Arguments callback_args{};
  callback_args.op_1.output = output;
  callback_args.op_1.routed_residual = routed_residual;
  callback_args.op_1.output_columns = N;
  callback_args.op_1.logical_rows = M;
  callback_args.op_1.dense_slot_by_row = dense_slot_by_row;
  callback_args.op_1.dense_rows = dense_rows;
  typename Gemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, cutlass::layout::ColumnMajor(K)},
      {reinterpret_cast<typename Gemm::ElementE *>(a_meta_e),
       Gemm::LayoutE::packed({N, columns_e})},
      callback_args};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename Gemm>
int sparse24_cutlass_inline_swiglu_gemm_run(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    Element *output, int M, int K, int N, cudaStream_t stream) {
  int sparse_k = K / Gemm::kSparse;
  int columns_e = K / Gemm::kSparse / Gemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);

  typename Gemm::FusionCallbacks::Arguments callback_args{};
  callback_args.op_1.output = output;
  callback_args.op_1.hidden_size = N / 2;
  callback_args.op_1.logical_rows = M;
  typename Gemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, cutlass::layout::ColumnMajor(K)},
      {reinterpret_cast<typename Gemm::ElementE *>(a_meta_e),
       Gemm::LayoutE::packed({N, columns_e})},
      callback_args};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename Gemm>
int sparse24_cutlass_inline_routed_swiglu_gemm_run(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    Element *output, Element *dense_base, const int *dense_slot_by_row,
    int M, int dense_rows, int K, int N, cudaStream_t stream) {
  int sparse_k = K / Gemm::kSparse;
  int columns_e = K / Gemm::kSparse / Gemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);

  typename Gemm::FusionCallbacks::Arguments callback_args{};
  callback_args.op_1.output = output;
  callback_args.op_1.dense_base = dense_base;
  callback_args.op_1.hidden_size = N / 2;
  callback_args.op_1.logical_rows = M;
  callback_args.op_1.dense_slot_by_row = dense_slot_by_row;
  callback_args.op_1.dense_rows = dense_rows;
  typename Gemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, cutlass::layout::ColumnMajor(K)},
      {reinterpret_cast<typename Gemm::ElementE *>(a_meta_e),
       Gemm::LayoutE::packed({N, columns_e})},
      callback_args};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename Gemm>
int sparse24_cutlass_residual_correction_swiglu_gemm_run(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    const Element *dense_base, const int *row_indices, Element *output,
    Element *compact_output, int M, int dense_rows, int output_rows, int K,
    int N, cudaStream_t stream) {
  int sparse_k = K / Gemm::kSparse;
  int columns_e = K / Gemm::kSparse / Gemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);

  typename Gemm::FusionCallbacks::Arguments callback_args{};
  callback_args.op_1.output = output;
  callback_args.op_1.correction_base = dense_base;
  callback_args.op_1.compact_output = compact_output;
  callback_args.op_1.hidden_size = N / 2;
  callback_args.op_1.logical_rows = dense_rows;
  callback_args.op_1.output_rows = output_rows;
  callback_args.op_1.row_indices = row_indices;
  typename Gemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, cutlass::layout::ColumnMajor(K)},
      {reinterpret_cast<typename Gemm::ElementE *>(a_meta_e),
       Gemm::LayoutE::packed({N, columns_e})},
      callback_args};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename Gemm>
int sparse24_cutlass_residual_delta_swiglu_gemm_run(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    const Element *dense_base, Element *dense_delta, int M, int dense_rows,
    int K, int N, cudaStream_t stream) {
  int sparse_k = K / Gemm::kSparse;
  int columns_e = K / Gemm::kSparse / Gemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);

  typename Gemm::FusionCallbacks::Arguments callback_args{};
  callback_args.op_1.output = dense_delta;
  callback_args.op_1.correction_base = dense_base;
  callback_args.op_1.compact_output = dense_delta;
  callback_args.op_1.hidden_size = N / 2;
  callback_args.op_1.logical_rows = M;
  callback_args.op_1.output_rows = dense_rows;
  callback_args.op_1.dense_rows = dense_rows;
  typename Gemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, cutlass::layout::ColumnMajor(K)},
      {reinterpret_cast<typename Gemm::ElementE *>(a_meta_e),
       Gemm::LayoutE::packed({N, columns_e})},
      callback_args};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename Gemm>
int sparse24_cutlass_indexed_swiglu_gemm_run(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    Element *output, const int *row_indices, int M, int logical_rows,
    int output_rows, int K, int N, cudaStream_t stream) {
  int sparse_k = K / Gemm::kSparse;
  int columns_e = K / Gemm::kSparse / Gemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);

  typename Gemm::FusionCallbacks::Arguments callback_args{};
  callback_args.op_1.output = output;
  callback_args.op_1.hidden_size = N / 2;
  callback_args.op_1.logical_rows = logical_rows;
  callback_args.op_1.output_rows = output_rows;
  callback_args.op_1.row_indices = row_indices;
  typename Gemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, cutlass::layout::ColumnMajor(K)},
      {reinterpret_cast<typename Gemm::ElementE *>(a_meta_e),
       Gemm::LayoutE::packed({N, columns_e})},
      callback_args};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename Gemm>
int sparse24_cutlass_dual_swiglu_gemm_run(
    const Element *x, const Element *full_values,
    DeviceElementE *full_meta, const Element *residual_values,
    DeviceElementE *residual_meta, Element *output,
    const int *row_indices, int M, int logical_rows, int output_rows, int K,
    int N, cudaStream_t stream) {
  using Kernel = Sparse24DualSwiGLUKernel<Gemm>;
  using FusionCallbacks = typename Kernel::FusionCallbacks;
  typename Kernel::Params params;
  params.x = x;
  params.values[0] = full_values;
  params.values[1] = residual_values;
  params.metadata[0] =
      reinterpret_cast<typename Kernel::ElementE *>(full_meta);
  params.metadata[1] =
      reinterpret_cast<typename Kernel::ElementE *>(residual_meta);
  params.rows = M;
  params.K = K;
  params.N = N;

  typename FusionCallbacks::Arguments callback_args{};
  callback_args.op_1.output = output;
  callback_args.op_1.hidden_size = N / 2;
  callback_args.op_1.logical_rows = logical_rows;
  callback_args.op_1.output_rows = output_rows;
  callback_args.op_1.row_indices = row_indices;
  cutlass::gemm::GemmCoord problem_size(N, M, K);
  params.output_op = FusionCallbacks::to_underlying_arguments(
      problem_size, callback_args, nullptr);

  int feature_tiles =
      (N + Kernel::ThreadblockShape::kM - 1) /
      Kernel::ThreadblockShape::kM;
  int row_tiles =
      (M + Kernel::ThreadblockShape::kN - 1) /
      Kernel::ThreadblockShape::kN;
  int grid = feature_tiles * row_tiles;
  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  cutlass::arch::synclog_setup();
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename Gemm>
int sparse24_cutlass_inline_pair_add_gemm_run(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    Element *output, const int *row_indices, int M, int logical_rows,
    int output_rows, int K, int N, cudaStream_t stream) {
  int packed_n = 2 * N;
  int sparse_k = K / Gemm::kSparse;
  int columns_e = K / Gemm::kSparse / Gemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(packed_n, M, K);

  typename Gemm::FusionCallbacks::Arguments callback_args{};
  callback_args.op_1.output = output;
  callback_args.op_1.hidden_size = N;
  callback_args.op_1.logical_rows = logical_rows;
  callback_args.op_1.output_rows = output_rows;
  callback_args.op_1.row_indices = row_indices;
  typename Gemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, cutlass::layout::ColumnMajor(K)},
      {reinterpret_cast<typename Gemm::ElementE *>(a_meta_e),
       Gemm::LayoutE::packed({packed_n, columns_e})},
      callback_args};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename Gemm>
int sparse24_cutlass_inline_qkv_postop_gemm_run(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    Element *output, const Element *q_weight, const Element *k_weight,
    const Element *cos_sin_cache, const int64_t *position_ids, int M, int K,
    int q_size, int kv_size, int rotary_dim, float epsilon, bool is_neox,
    bool normalize_qk, cudaStream_t stream) {
  int N = q_size + 2 * kv_size;
  int sparse_k = K / Gemm::kSparse;
  int columns_e = K / Gemm::kSparse / Gemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);

  typename Gemm::FusionCallbacks::Arguments callback_args{};
  callback_args.op_1.output = output;
  callback_args.op_1.q_weight = q_weight;
  callback_args.op_1.k_weight = k_weight;
  callback_args.op_1.cos_sin_cache = cos_sin_cache;
  callback_args.op_1.position_ids = position_ids;
  callback_args.op_1.q_size = q_size;
  callback_args.op_1.kv_size = kv_size;
  callback_args.op_1.logical_rows = M;
  callback_args.op_1.rotary_dim = rotary_dim;
  callback_args.op_1.epsilon = epsilon;
  callback_args.op_1.is_neox = is_neox;
  callback_args.op_1.normalize_qk = normalize_qk;
  typename Gemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, cutlass::layout::ColumnMajor(K)},
      {reinterpret_cast<typename Gemm::ElementE *>(a_meta_e),
       Gemm::LayoutE::packed({N, columns_e})},
      callback_args};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

extern "C" int sparse24_cutlass_device_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *c_tmp_ptr, void *y_ptr, int M, int K, int N, void *stream_ptr);

template <typename LayoutB, typename ThreadblockShape_, typename WarpShape_,
          int Stages_,
          typename ThreadblockSwizzle_ =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
int sparse24_cutlass_device_gemm_run_layout_b(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    Element *c_tmp, int M, int K, int N, int ldb, cudaStream_t stream,
    int lda = 0) {
  using Gemm = DeviceSparseGemmVec8LayoutBVariant<
      LayoutB, ThreadblockShape_, WarpShape_, Stages_, ThreadblockSwizzle_>;
  int sparse_k = K / Gemm::kSparse;
  int values_leading_dim = lda > 0 ? lda : sparse_k;
  int columns_e = K / Gemm::kSparse / Gemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);
  DeviceLayoutC layout_c(M);

  typename Gemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(values_leading_dim)},
      {x, LayoutB(ldb)},
      {c_tmp, layout_c},
      {c_tmp, layout_c},
      {reinterpret_cast<typename Gemm::ElementE *>(a_meta_e),
       Gemm::LayoutE::packed({N, columns_e})},
      typename Gemm::EpilogueOutputOp::Params(),
      1};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename DeviceGemm>
int sparse24_cutlass_device_splitk_gemm_run(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    Element *output, int *tile_counters, int tile_counter_count, int M, int K,
    int N, int split_k_slices, cudaStream_t stream) {
  using BaseKernel = typename DeviceGemm::GemmKernel;
  using Kernel = Sparse24FixedSplitKSparseGemmKernel<BaseKernel>;
  using Mma = typename BaseKernel::Mma;
  using ThreadblockSwizzle = typename BaseKernel::ThreadblockSwizzle;
  int sparse_k = K / DeviceGemm::kSparse;
  int columns_e = sparse_k / DeviceGemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);
  DeviceLayoutC layout_c(M);
  ThreadblockSwizzle threadblock_swizzle;
  cutlass::gemm::GemmCoord grid_shape =
      threadblock_swizzle.get_tiled_shape(
          problem_size,
          {Mma::Shape::kM, Mma::Shape::kN, Mma::Shape::kK},
          split_k_slices);
  int required_counters = grid_shape.m() * grid_shape.n();
  if (tile_counter_count < required_counters) {
    return -6;
  }
  typename DeviceGemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, cutlass::layout::ColumnMajor(K)},
      {output, layout_c},
      {output, layout_c},
      {reinterpret_cast<typename DeviceGemm::ElementE *>(a_meta_e),
       DeviceGemm::LayoutE::packed({N, columns_e})},
      typename DeviceGemm::EpilogueOutputOp::Params(),
      split_k_slices};
  typename BaseKernel::Params params{
      problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.ref_E.non_const_ref(),
      args.epilogue,
      tile_counters};
  cutlass::Status status = BaseKernel::can_implement(
      problem_size, params.ref_A, params.ref_B, params.ref_C, params.ref_D,
      params.ref_E);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  cutlass::arch::synclog_setup();
  dim3 grid = threadblock_swizzle.get_grid_shape(grid_shape);
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename DeviceGemm>
int sparse24_cutlass_device_splitk_indexed_add_gemm_run(
    const Element *x, const Element *a_values, DeviceElementE *a_meta_e,
    Element *residual_output, int *tile_counters, int tile_counter_count,
    Element *full_output, const int *dense_rows, int *ready_state,
    int ready_state_count, int M, int dense_count, int full_rows, int K, int N,
    int split_k_slices, cudaStream_t stream) {
  using BaseKernel = typename DeviceGemm::GemmKernel;
  using Kernel = Sparse24FixedSplitKIndexedAddKernel<BaseKernel>;
  using Mma = typename BaseKernel::Mma;
  using ThreadblockSwizzle = typename BaseKernel::ThreadblockSwizzle;
  int sparse_k = K / DeviceGemm::kSparse;
  int columns_e = sparse_k / DeviceGemm::kElementsPerElementE;
  cutlass::gemm::GemmCoord problem_size(N, M, K);
  DeviceLayoutC layout_c(M);
  ThreadblockSwizzle threadblock_swizzle;
  cutlass::gemm::GemmCoord grid_shape =
      threadblock_swizzle.get_tiled_shape(
          problem_size,
          {Mma::Shape::kM, Mma::Shape::kN, Mma::Shape::kK},
          split_k_slices);
  int required_counters = grid_shape.m() * grid_shape.n();
  if (tile_counter_count < required_counters || ready_state_count < 2) {
    return -6;
  }
  typename DeviceGemm::Arguments args{
      problem_size,
      {a_values, cutlass::layout::RowMajor(sparse_k)},
      {x, cutlass::layout::ColumnMajor(K)},
      {residual_output, layout_c},
      {residual_output, layout_c},
      {reinterpret_cast<typename DeviceGemm::ElementE *>(a_meta_e),
       DeviceGemm::LayoutE::packed({N, columns_e})},
      typename DeviceGemm::EpilogueOutputOp::Params(),
      split_k_slices};
  typename BaseKernel::Params gemm_params{
      problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.ref_E.non_const_ref(),
      args.epilogue,
      tile_counters};
  typename Kernel::Params params{
      gemm_params, full_output, dense_rows, ready_state, dense_count,
      full_rows, N, required_counters};
  cutlass::Status status = BaseKernel::can_implement(
      problem_size, gemm_params.ref_A, gemm_params.ref_B, gemm_params.ref_C,
      gemm_params.ref_D, gemm_params.ref_E);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (attr_status != cudaSuccess) {
      return static_cast<int>(attr_status);
    }
  }
  cutlass::arch::synclog_setup();
  dim3 grid = threadblock_swizzle.get_grid_shape(grid_shape);
  cutlass::Kernel<Kernel>
      <<<grid, Kernel::kThreadCount, smem_size, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

template <typename Gemm>
int dense_cutlass_device_gemm_run(const Element *x, const Element *w,
                                  Element *y, int M, int K, int N,
                                  cudaStream_t stream) {
  using EpilogueCompute = typename Gemm::EpilogueOutputOp::ElementCompute;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmCoord(M, N, K),
      {x, K},
      {w, N},
      {y, N},
      {y, N},
      {EpilogueCompute(1.0f), EpilogueCompute(0.0f)}};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename Gemm>
int dense_cutlass_device_gemm_b_col_run(const Element *x, const Element *w_col,
                                        Element *y, int M, int K, int N,
                                        cudaStream_t stream) {
  using EpilogueCompute = typename Gemm::EpilogueOutputOp::ElementCompute;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmCoord(M, N, K),
      {x, K},
      {w_col, K},
      {y, N},
      {y, N},
      {EpilogueCompute(1.0f), EpilogueCompute(0.0f)}};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

template <typename Gemm>
int dense_cutlass_device_gemm_b_col_add_run(
    const Element *x, const Element *w_col, const Element *residual,
    Element *y, int M, int K, int N, cudaStream_t stream) {
  using EpilogueCompute = typename Gemm::EpilogueOutputOp::ElementCompute;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmCoord(M, N, K),
      {x, K},
      {w_col, K},
      {residual, N},
      {y, N},
      {EpilogueCompute(1.0f), EpilogueCompute(1.0f)}};

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return -1000 - static_cast<int>(status);
  }
  Gemm op;
  status = op(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return -2000 - static_cast<int>(status);
  }
  return 0;
}

const char *dense_cutlass_select_device_config(const char *requested, int M,
                                               int K, int N) {
  if (requested != nullptr && std::strcmp(requested, "auto") != 0) {
    return requested;
  }
  if (M >= 128 && N >= 8192) {
    return "128x128x64_s3";
  }
  if (M >= 128) {
    return "128x64x64_s3";
  }
  if (N >= 8192) {
    return "64x128x64_s3";
  }
  return M <= 64 ? "64x64x64_s4" : "64x64x64_s3";
}

int dense_cutlass_device_dispatch_config(const char *config, const Element *x,
                                         const Element *w, Element *y, int M,
                                         int K, int N, cudaStream_t stream) {
  int status = 0;
#define RUN_DENSE(GemmType)                                                   \
  dense_cutlass_device_gemm_run<GemmType>(x, w, y, M, K, N, stream)
  if (std::strcmp(config, "64x64x64") == 0 ||
      std::strcmp(config, "64x64x64_s3") == 0 ||
      std::strcmp(config, "default") == 0) {
    status = RUN_DENSE(DeviceDenseGemmVec8M64N64K64S3);
  } else if (std::strcmp(config, "64x64x64_s4") == 0) {
    status = RUN_DENSE(DeviceDenseGemmVec8M64N64K64S4);
  } else if (std::strcmp(config, "64x128x64") == 0 ||
             std::strcmp(config, "64x128x64_s3") == 0) {
    status = RUN_DENSE(DeviceDenseGemmVec8M64N128K64S3);
  } else if (std::strcmp(config, "128x64x64") == 0 ||
             std::strcmp(config, "128x64x64_s3") == 0) {
    status = RUN_DENSE(DeviceDenseGemmVec8M128N64K64S3);
  } else if (std::strcmp(config, "128x128x64") == 0 ||
             std::strcmp(config, "128x128x64_s3") == 0) {
    status = RUN_DENSE(DeviceDenseGemmVec8M128N128K64S3);
  } else {
    status = -5;
  }
#undef RUN_DENSE
  return status;
}

int dense_cutlass_device_b_col_dispatch_config(const char *config,
                                               const Element *x,
                                               const Element *w_col,
                                               Element *y, int M, int K, int N,
                                               cudaStream_t stream) {
  int status = 0;
#define RUN_DENSE_BCOL(GemmType)                                             \
  dense_cutlass_device_gemm_b_col_run<GemmType>(x, w_col, y, M, K, N, stream)
  if (std::strcmp(config, "64x64x64") == 0 ||
      std::strcmp(config, "64x64x64_s3") == 0 ||
      std::strcmp(config, "default") == 0) {
    status = RUN_DENSE_BCOL(DeviceDenseGemmBColVec8M64N64K64S3);
  } else if (std::strcmp(config, "64x64x64_s4") == 0) {
    status = RUN_DENSE_BCOL(DeviceDenseGemmBColVec8M64N64K64S4);
  } else if (std::strcmp(config, "64x128x64") == 0 ||
             std::strcmp(config, "64x128x64_s3") == 0) {
    status = RUN_DENSE_BCOL(DeviceDenseGemmBColVec8M64N128K64S3);
  } else if (std::strcmp(config, "128x64x64") == 0 ||
             std::strcmp(config, "128x64x64_s3") == 0) {
    status = RUN_DENSE_BCOL(DeviceDenseGemmBColVec8M128N64K64S3);
  } else if (std::strcmp(config, "128x128x64") == 0 ||
             std::strcmp(config, "128x128x64_s3") == 0) {
    status = RUN_DENSE_BCOL(DeviceDenseGemmBColVec8M128N128K64S3);
  } else {
    status = -5;
  }
#undef RUN_DENSE_BCOL
  return status;
}

int dense_cutlass_device_b_col_add_dispatch_config(
    const char *config, const Element *x, const Element *w_col,
    const Element *residual, Element *y, int M, int K, int N,
    cudaStream_t stream) {
  int status = 0;
#define RUN_DENSE_BCOL_ADD(GemmType)                                         \
  dense_cutlass_device_gemm_b_col_add_run<GemmType>(                         \
      x, w_col, residual, y, M, K, N, stream)
  if (std::strcmp(config, "64x64x64") == 0 ||
      std::strcmp(config, "64x64x64_s3") == 0 ||
      std::strcmp(config, "default") == 0) {
    status = RUN_DENSE_BCOL_ADD(DeviceDenseGemmBColVec8M64N64K64S3);
  } else if (std::strcmp(config, "64x64x64_s4") == 0) {
    status = RUN_DENSE_BCOL_ADD(DeviceDenseGemmBColVec8M64N64K64S4);
  } else if (std::strcmp(config, "64x128x64") == 0 ||
             std::strcmp(config, "64x128x64_s3") == 0) {
    status = RUN_DENSE_BCOL_ADD(DeviceDenseGemmBColVec8M64N128K64S3);
  } else if (std::strcmp(config, "128x64x64") == 0 ||
             std::strcmp(config, "128x64x64_s3") == 0) {
    status = RUN_DENSE_BCOL_ADD(DeviceDenseGemmBColVec8M128N64K64S3);
  } else if (std::strcmp(config, "128x128x64") == 0 ||
             std::strcmp(config, "128x128x64_s3") == 0) {
    status = RUN_DENSE_BCOL_ADD(DeviceDenseGemmBColVec8M128N128K64S3);
  } else {
    status = -5;
  }
#undef RUN_DENSE_BCOL_ADD
  return status;
}

int dense_cutlass_device_f16_accum_dispatch_config(
    const char *config, const Element *x, const Element *w, Element *y, int M,
    int K, int N, cudaStream_t stream) {
  int status = 0;
#define RUN_DENSE_F16(GemmType)                                              \
  dense_cutlass_device_gemm_run<GemmType>(x, w, y, M, K, N, stream)
  if (std::strcmp(config, "64x64x64") == 0 ||
      std::strcmp(config, "64x64x64_s3") == 0 ||
      std::strcmp(config, "default") == 0) {
    status = RUN_DENSE_F16(DeviceDenseGemmVec8F16M64N64K64S3);
  } else if (std::strcmp(config, "64x64x64_s4") == 0) {
    status = RUN_DENSE_F16(DeviceDenseGemmVec8F16M64N64K64S4);
  } else if (std::strcmp(config, "64x128x64") == 0 ||
             std::strcmp(config, "64x128x64_s3") == 0) {
    status = RUN_DENSE_F16(DeviceDenseGemmVec8F16M64N128K64S3);
  } else if (std::strcmp(config, "128x64x64") == 0 ||
             std::strcmp(config, "128x64x64_s3") == 0) {
    status = RUN_DENSE_F16(DeviceDenseGemmVec8F16M128N64K64S3);
  } else if (std::strcmp(config, "128x128x64") == 0 ||
             std::strcmp(config, "128x128x64_s3") == 0) {
    status = RUN_DENSE_F16(DeviceDenseGemmVec8F16M128N128K64S3);
  } else {
    status = -5;
  }
#undef RUN_DENSE_F16
  return status;
}

int dense_cutlass_device_b_col_f16_accum_dispatch_config(
    const char *config, const Element *x, const Element *w_col, Element *y,
    int M, int K, int N, cudaStream_t stream) {
  int status = 0;
#define RUN_DENSE_BCOL_F16(GemmType)                                         \
  dense_cutlass_device_gemm_b_col_run<GemmType>(x, w_col, y, M, K, N, stream)
  if (std::strcmp(config, "64x64x64") == 0 ||
      std::strcmp(config, "64x64x64_s3") == 0 ||
      std::strcmp(config, "default") == 0) {
    status = RUN_DENSE_BCOL_F16(DeviceDenseGemmBColVec8F16M64N64K64S3);
  } else if (std::strcmp(config, "64x64x64_s4") == 0) {
    status = RUN_DENSE_BCOL_F16(DeviceDenseGemmBColVec8F16M64N64K64S4);
  } else if (std::strcmp(config, "64x128x64") == 0 ||
             std::strcmp(config, "64x128x64_s3") == 0) {
    status = RUN_DENSE_BCOL_F16(DeviceDenseGemmBColVec8F16M64N128K64S3);
  } else if (std::strcmp(config, "128x64x64") == 0 ||
             std::strcmp(config, "128x64x64_s3") == 0) {
    status = RUN_DENSE_BCOL_F16(DeviceDenseGemmBColVec8F16M128N64K64S3);
  } else if (std::strcmp(config, "128x128x64") == 0 ||
             std::strcmp(config, "128x128x64_s3") == 0) {
    status = RUN_DENSE_BCOL_F16(DeviceDenseGemmBColVec8F16M128N128K64S3);
  } else {
    status = -5;
  }
#undef RUN_DENSE_BCOL_F16
  return status;
}

int dense_cutlass_device_b_col_add_f16_accum_dispatch_config(
    const char *config, const Element *x, const Element *w_col,
    const Element *residual, Element *y, int M, int K, int N,
    cudaStream_t stream) {
  int status = 0;
#define RUN_DENSE_BCOL_ADD_F16(GemmType)                                     \
  dense_cutlass_device_gemm_b_col_add_run<GemmType>(                         \
      x, w_col, residual, y, M, K, N, stream)
  if (std::strcmp(config, "64x64x64") == 0 ||
      std::strcmp(config, "64x64x64_s3") == 0 ||
      std::strcmp(config, "default") == 0) {
    status = RUN_DENSE_BCOL_ADD_F16(DeviceDenseGemmBColVec8F16M64N64K64S3);
  } else if (std::strcmp(config, "64x64x64_s4") == 0) {
    status = RUN_DENSE_BCOL_ADD_F16(DeviceDenseGemmBColVec8F16M64N64K64S4);
  } else if (std::strcmp(config, "64x128x64") == 0 ||
             std::strcmp(config, "64x128x64_s3") == 0) {
    status = RUN_DENSE_BCOL_ADD_F16(DeviceDenseGemmBColVec8F16M64N128K64S3);
  } else if (std::strcmp(config, "128x64x64") == 0 ||
             std::strcmp(config, "128x64x64_s3") == 0) {
    status = RUN_DENSE_BCOL_ADD_F16(DeviceDenseGemmBColVec8F16M128N64K64S3);
  } else if (std::strcmp(config, "128x128x64") == 0 ||
             std::strcmp(config, "128x128x64_s3") == 0) {
    status =
        RUN_DENSE_BCOL_ADD_F16(DeviceDenseGemmBColVec8F16M128N128K64S3);
  } else {
    status = -5;
  }
#undef RUN_DENSE_BCOL_ADD_F16
  return status;
}

const char *sparse24_cutlass_select_device_config(const char *requested, int M,
                                                  int K, int N) {
  if (requested != nullptr && std::strcmp(requested, "auto") != 0) {
    return requested;
  }

  // The sparse-A GEMM problem is [N, M, K] for public Y[M, N].  The policy is
  // tuned for Qwen3-8B and Llama-3.1-8B serving projection shapes using
  // median-of-5 CUDA event timings against dense cuBLAS.
  if (N == 4096 && K == 12288) {
    if (M >= 336) {
      return "256x64x64_s3";
    }
    if (M >= 168) {
      return "128x64x64_s4";
    }
    if (M >= 72) {
      return "128x32x64_s4_sw4";
    }
    return M >= 32 ? "64x64x64_s6" : "64x32x64_s5";
  }
  if (N == 4096 && K >= 8192) {
    if (K >= 14336) {
      if (M >= 656) {
        return "128x64x64_s5";
      }
      if (M >= 336) {
        return "128x128x64_s3";
      }
      if (M >= 192) {
        return "128x64x64_s4";
      }
      if (M >= 160) {
        return "128x64x64_s3";
      }
      if (M >= 136) {
        return "128x32x64_s4_sw4";
      }
      if (M >= 64) {
        return "64x64x64_s7";
      }
      return M >= 32 ? "64x64x64_s4" : "64x32x64_s5";
    }
    if (M >= 384) {
      return "256x64x64_s3";
    }
    if (M >= 192) {
      return "128x64x64_s4";
    }
    if (M >= 136) {
      return "128x32x64_s4";
    }
    if (M >= 64) {
      return "64x64x64_s6";
    }
    return M >= 32 ? "64x64x64_s4" : "64x32x64_s5";
  }
  if (K == 4096 && N == 4096) {
    if (M >= 384) {
      return "256x64x64_s3";
    }
    if (M >= 136) {
      return "128x64x64_s5";
    }
    if (M >= 128) {
      return "64x64x64_s5";
    }
    return M >= 64 ? "64x64x64_s6" : "64x64x64";
  }
  if (K == 4096 && N == 6144) {
    if (M >= 384) {
      return "256x64x64_s3_sw4";
    }
    if (M >= 256) {
      return "256x64x64_s3";
    }
    if (M >= 192) {
      return "256x64x64_s3_sw4";
    }
    if (M >= 104) {
      return "128x64x64_s5";
    }
    if (M >= 72) {
      return "128x32x64_s4_sw4";
    }
    return M >= 32 ? "64x64x64_s4" : "64x32x64_s4";
  }
  if (K == 4096 && N >= 28672) {
    if (M >= 96) {
      return "256x64x64_s3_sw4";
    }
    if (M >= 72) {
      return "256x32x64_s3_sw4";
    }
    if (M >= 64) {
      return "64x64x64_s5";
    }
    return M >= 32 ? "256x64x64_s3" : "64x32x64_s3";
  }
  if (K == 4096 && N >= 24576) {
    if (M >= 384) {
      return "256x64x64_s3_sw4";
    }
    if (M >= 136) {
      return "256x64x64_s3_sw4";
    }
    if (M >= 104) {
      return "256x64x64_s3";
    }
    if (M >= 72) {
      return "256x32x64_s3_sw4";
    }
    return M >= 32 ? "256x64x64_s3" : "64x32x64_s3";
  }
  if (K == 4096 && N >= 12288) {
    return M >= 128 ? "64x128x64_s4" : "64x32x64_s3";
  }
  return "64x64x64";
}

const char *sparse24_cutlass_select_b_row_device_config(
    const char *requested, int M, int K, int N) {
  if (requested != nullptr && std::strcmp(requested, "auto") != 0) {
    return requested;
  }
  // Qwen3-8B down projection consumes the transposed SwiGLU output. These
  // thresholds are median-of-50 CUDA-event results for bs={16,32,64} and
  // K={6,8,10}; the regular B-column selector is not optimal for this layout.
  if (K == 12288 && N == 4096) {
    if (M >= 640) {
      return "128x64x64_s5";
    }
    if (M >= 384) {
      return "256x64x64_s3";
    }
    if (M >= 192) {
      return "128x64x64_s4";
    }
    if (M >= 136) {
      return "128x32x64_s4_sw4";
    }
    if (M < 96 && M >= 64) {
      return "128x32x64_s4_sw4";
    }
    return M >= 64 ? "64x64x64_s6" : "64x32x64_s5";
  }
  return sparse24_cutlass_select_device_config(requested, M, K, N);
}

int sparse24_cutlass_device_dispatch_config_f16_accum(
    const char *config, const Element *x, const Element *a_values,
    DeviceElementE *a_meta_e, Element *c_tmp, int M, int K, int N,
    cudaStream_t stream) {
#define RUN_F16_ACCUM(GemmType)                                               \
  sparse24_cutlass_device_gemm_run<GemmType>(                                 \
      x, a_values, a_meta_e, c_tmp, M, K, N, stream)
  if (std::strcmp(config, "128x64x64_s4") == 0) {
    return RUN_F16_ACCUM(DeviceSparseGemmF16AccumM128N64K64S4);
  }
  if (std::strcmp(config, "128x64x64_s5") == 0) {
    return RUN_F16_ACCUM(DeviceSparseGemmF16AccumM128N64K64S5);
  }
  if (std::strcmp(config, "64x32x64") == 0 ||
      std::strcmp(config, "64x32x64_s3") == 0) {
    return RUN_F16_ACCUM(DeviceSparseGemmF16AccumM64N32K64S3);
  }
  if (std::strcmp(config, "64x32x64_s4") == 0) {
    return RUN_F16_ACCUM(DeviceSparseGemmF16AccumM64N32K64S4);
  }
  if (std::strcmp(config, "256x32x64_s3_sw4") == 0) {
    return RUN_F16_ACCUM(DeviceSparseGemmF16AccumM256N32K64S3Sw4);
  }
  if (std::strcmp(config, "256x64x64_s3") == 0) {
    return RUN_F16_ACCUM(DeviceSparseGemmF16AccumM256N64K64S3);
  }
  if (std::strcmp(config, "256x64x64_s3_sw4") == 0) {
    return RUN_F16_ACCUM(DeviceSparseGemmF16AccumM256N64K64S3Sw4);
  }
#undef RUN_F16_ACCUM
  return -5;
}

bool sparse24_cutlass_use_f16_accumulator(const char *accumulator, int K,
                                          int N) {
  if (accumulator == nullptr) {
    return false;
  }
  if (std::strcmp(accumulator, "fp16") == 0) {
    return true;
  }
  if (std::strcmp(accumulator, "fp16_gate") == 0) {
    return K == 4096 && N >= 24576;
  }
  if (std::strcmp(accumulator, "fp16_gate_down") == 0) {
    return (K == 4096 && N >= 24576) || (K >= 8192 && N == 4096);
  }
  if (std::strcmp(accumulator, "fp16_qkv_gate") != 0) {
    return false;
  }

  // Qwen3-8B and Llama-3.1-8B both use hidden size 4096. Their fused QKV
  // projection has N=6144, while fused gate_up has N=24576 or N=28672.
  // Keep the output and down projections on FP32 accumulation.
  return K == 4096 && (N == 6144 || N >= 24576);
}

int sparse24_cutlass_device_dispatch_config(const char *config, const Element *x,
                                            const Element *a_values,
                                            DeviceElementE *a_meta_e,
                                            Element *c_tmp, int M, int K, int N,
                                            cudaStream_t stream) {
  int status = 0;
#define RUN_ROW(GemmType)                                                     \
  sparse24_cutlass_device_gemm_run<GemmType>(                                 \
      x, a_values, a_meta_e, c_tmp, M, K, N, stream)
  if (std::strcmp(config, "64x64x64") == 0 ||
      std::strcmp(config, "default") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8);
  } else if (std::strcmp(config, "64x32x64") == 0 ||
             std::strcmp(config, "64x32x64_s3") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M64N32K64S3);
  } else if (std::strcmp(config, "64x32x64_s2") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M64N32K64S2);
  } else if (std::strcmp(config, "64x32x64_s4") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M64N32K64S4);
  } else if (std::strcmp(config, "64x32x64_s5") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M64N32K64S5);
  } else if (std::strcmp(config, "64x128x64_s2") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M64N128K64S2);
  } else if (std::strcmp(config, "64x128x64") == 0 ||
             std::strcmp(config, "64x128x64_s3") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M64N128K64S3);
  } else if (std::strcmp(config, "64x128x64_s4") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M64N128K64S4);
  } else if (std::strcmp(config, "64x64x64_s2") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M64N64K64S2);
  } else if (std::strcmp(config, "64x64x64_s4") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M64N64K64S4);
  } else if (std::strcmp(config, "64x64x64_s5") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M64N64K64S5);
  } else if (std::strcmp(config, "64x64x64_s6") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M64N64K64S6);
  } else if (std::strcmp(config, "64x64x64_s7") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M64N64K64S7);
  } else if (std::strcmp(config, "128x32x64_s4") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M128N32K64S4);
  } else if (std::strcmp(config, "128x32x64_s4_sw2") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M128N32K64S4Sw2);
  } else if (std::strcmp(config, "128x32x64_s4_sw4") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M128N32K64S4Sw4);
  } else if (std::strcmp(config, "128x64x64_s2") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M128N64K64S2);
  } else if (std::strcmp(config, "128x64x64_s3") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M128N64K64S3);
  } else if (std::strcmp(config, "128x64x64_s4") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M128N64K64S4);
  } else if (std::strcmp(config, "128x64x64_s5") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M128N64K64S5);
  } else if (std::strcmp(config, "128x128x64_s2") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M128N128K64S2);
  } else if (std::strcmp(config, "128x128x64_s3") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M128N128K64S3);
  } else if (std::strcmp(config, "128x128x64_s3_sw2") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M128N128K64S3Sw2);
  } else if (std::strcmp(config, "128x128x64_s3_sw4") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M128N128K64S3Sw4);
  } else if (std::strcmp(config, "256x64x64_s2") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M256N64K64S2);
  } else if (std::strcmp(config, "256x64x64_s3") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M256N64K64S3);
  } else if (std::strcmp(config, "256x64x64_s3_sw2") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M256N64K64S3Sw2);
  } else if (std::strcmp(config, "256x64x64_s3_sw4") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M256N64K64S3Sw4);
  } else if (std::strcmp(config, "256x32x64_s3_sw4") == 0) {
    status = RUN_ROW(DeviceSparseGemmVec8M256N32K64S3Sw4);
  } else {
    status = -5;
  }
#undef RUN_ROW
  return status;
}

int sparse24_cutlass_device_dispatch_config_b_row(
    const char *config, const Element *x, const Element *a_values,
    DeviceElementE *a_meta_e, Element *c_tmp, int M, int K, int N, int ldb,
    cudaStream_t stream) {
  int status = 0;
#define RUN_BROW(TB, Warp, Stages)                                            \
  sparse24_cutlass_device_gemm_run_layout_b<cutlass::layout::RowMajor, TB,    \
                                            Warp, Stages>(                    \
      x, a_values, a_meta_e, c_tmp, M, K, N, ldb, stream)
#define RUN_BROW_SW(TB, Warp, Stages, Swizzle)                                \
  sparse24_cutlass_device_gemm_run_layout_b<cutlass::layout::RowMajor, TB,    \
                                            Warp, Stages, Swizzle>(           \
      x, a_values, a_meta_e, c_tmp, M, K, N, ldb, stream)
  if (std::strcmp(config, "64x64x64") == 0 ||
      std::strcmp(config, "default") == 0) {
    status = RUN_BROW(DeviceThreadblockShape, DeviceWarpShape, 3);
  } else if (std::strcmp(config, "64x32x64") == 0 ||
             std::strcmp(config, "64x32x64_s3") == 0) {
    status = RUN_BROW(DeviceThreadblockShape64x32x64, DeviceWarpShape, 3);
  } else if (std::strcmp(config, "64x32x64_s2") == 0) {
    status = RUN_BROW(DeviceThreadblockShape64x32x64, DeviceWarpShape, 2);
  } else if (std::strcmp(config, "64x32x64_s4") == 0) {
    status = RUN_BROW(DeviceThreadblockShape64x32x64, DeviceWarpShape, 4);
  } else if (std::strcmp(config, "64x32x64_s5") == 0) {
    status = RUN_BROW(DeviceThreadblockShape64x32x64, DeviceWarpShape, 5);
  } else if (std::strcmp(config, "64x128x64_s2") == 0) {
    status = RUN_BROW(DeviceThreadblockShape64x128x64,
                      DeviceWarpShape32x64x64, 2);
  } else if (std::strcmp(config, "64x128x64") == 0 ||
             std::strcmp(config, "64x128x64_s3") == 0) {
    status = RUN_BROW(DeviceThreadblockShape64x128x64,
                      DeviceWarpShape32x64x64, 3);
  } else if (std::strcmp(config, "64x128x64_s4") == 0) {
    status = RUN_BROW(DeviceThreadblockShape64x128x64,
                      DeviceWarpShape32x64x64, 4);
  } else if (std::strcmp(config, "64x64x64_s2") == 0) {
    status = RUN_BROW(DeviceThreadblockShape, DeviceWarpShape, 2);
  } else if (std::strcmp(config, "64x64x64_s4") == 0) {
    status = RUN_BROW(DeviceThreadblockShape, DeviceWarpShape, 4);
  } else if (std::strcmp(config, "64x64x64_s5") == 0) {
    status = RUN_BROW(DeviceThreadblockShape, DeviceWarpShape, 5);
  } else if (std::strcmp(config, "64x64x64_s6") == 0) {
    status = RUN_BROW(DeviceThreadblockShape, DeviceWarpShape, 6);
  } else if (std::strcmp(config, "64x64x64_s7") == 0) {
    status = RUN_BROW(DeviceThreadblockShape, DeviceWarpShape, 7);
  } else if (std::strcmp(config, "128x32x64_s4") == 0) {
    status = RUN_BROW(DeviceThreadblockShape128x32x64, DeviceWarpShape, 4);
  } else if (std::strcmp(config, "128x32x64_s4_sw2") == 0) {
    status = RUN_BROW_SW(
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 4,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>);
  } else if (std::strcmp(config, "128x32x64_s4_sw4") == 0) {
    status = RUN_BROW_SW(
        DeviceThreadblockShape128x32x64, DeviceWarpShape, 4,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>);
  } else if (std::strcmp(config, "128x64x64_s2") == 0) {
    status = RUN_BROW(DeviceThreadblockShape128x64x64, DeviceWarpShape, 2);
  } else if (std::strcmp(config, "128x64x64_s3") == 0) {
    status = RUN_BROW(DeviceThreadblockShape128x64x64, DeviceWarpShape, 3);
  } else if (std::strcmp(config, "128x64x64_s4") == 0) {
    status = RUN_BROW(DeviceThreadblockShape128x64x64, DeviceWarpShape, 4);
  } else if (std::strcmp(config, "128x64x64_s5") == 0) {
    status = RUN_BROW(DeviceThreadblockShape128x64x64, DeviceWarpShape, 5);
  } else if (std::strcmp(config, "128x128x64_s2") == 0) {
    status = RUN_BROW(DeviceThreadblockShape128x128x64,
                      DeviceWarpShape32x64x64, 2);
  } else if (std::strcmp(config, "128x128x64_s3") == 0) {
    status = RUN_BROW(DeviceThreadblockShape128x128x64,
                      DeviceWarpShape32x64x64, 3);
  } else if (std::strcmp(config, "128x128x64_s3_sw2") == 0) {
    status = RUN_BROW_SW(
        DeviceThreadblockShape128x128x64, DeviceWarpShape32x64x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>);
  } else if (std::strcmp(config, "128x128x64_s3_sw4") == 0) {
    status = RUN_BROW_SW(
        DeviceThreadblockShape128x128x64, DeviceWarpShape32x64x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>);
  } else if (std::strcmp(config, "256x64x64_s2") == 0) {
    status = RUN_BROW(DeviceThreadblockShape256x64x64,
                      DeviceWarpShape64x32x64, 2);
  } else if (std::strcmp(config, "256x64x64_s3") == 0) {
    status = RUN_BROW(DeviceThreadblockShape256x64x64,
                      DeviceWarpShape64x32x64, 3);
  } else if (std::strcmp(config, "256x64x64_s3_sw2") == 0) {
    status = RUN_BROW_SW(
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>);
  } else if (std::strcmp(config, "256x64x64_s3_sw4") == 0) {
    status = RUN_BROW_SW(
        DeviceThreadblockShape256x64x64, DeviceWarpShape64x32x64, 3,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>);
  } else {
    status = -5;
  }
#undef RUN_BROW
#undef RUN_BROW_SW
  return status;
}

extern "C" int sparse24_cutlass_device_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *c_tmp_ptr, void *y_ptr, int M, int K, int N, void *stream_ptr) {
  if (M <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 32) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *c_tmp = reinterpret_cast<Element *>(c_tmp_ptr);
  Element *y = reinterpret_cast<Element *>(y_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const char *config = sparse24_cutlass_select_device_config(
      std::getenv("SPECLINK_SPARSE24_DEVICE_CONFIG"), M, K, N);
  const char *accumulator = std::getenv("SPECLINK_SPARSE24_ACCUMULATOR");
  bool use_f16_accumulator =
      sparse24_cutlass_use_f16_accumulator(accumulator, K, N);
  int status = -5;
  if (use_f16_accumulator) {
    status = sparse24_cutlass_device_dispatch_config_f16_accum(
        config, x, a_values, a_meta_e, c_tmp, M, K, N, stream);
  }
  if (status == -5) {
    status = sparse24_cutlass_device_dispatch_config(
        config, x, a_values, a_meta_e, c_tmp, M, K, N, stream);
  }
  if (status != 0) {
    return status;
  }
  if (y != nullptr) {
    dim3 block(32, 8);
    dim3 grid((M + 31) / 32, (N + 31) / 32);
    sparse24_cutlass_device_transpose_tiled_kernel<<<grid, block, 0, stream>>>(
        c_tmp, y, M, N);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int sparse24_cutlass_device_strided_input_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *c_tmp_ptr, int M, int K, int N, int ldb, int lda,
    void *stream_ptr) {
  if (x_ptr == nullptr || a_values_ptr == nullptr || a_meta_e_ptr == nullptr ||
      c_tmp_ptr == nullptr || M <= 0 || K <= 0 || N <= 0 || ldb < K ||
      lda < K / 2) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 256) != 0 ||
      (M % 8) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *c_tmp = reinterpret_cast<Element *>(c_tmp_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  return sparse24_cutlass_device_gemm_run_layout_b<
      cutlass::layout::ColumnMajor, DeviceThreadblockShape256x32x64,
      DeviceWarpShape64x32x64, 3,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>(
      x, a_values, a_meta_e, c_tmp, M, K, N, ldb, stream, lda);
}

extern "C" int sparse24_cutlass_device_splitk_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *c_tmp_ptr, void *tile_counters_ptr, int tile_counter_count, int M,
    int K, int N, int split_k_slices, void *stream_ptr) {
  if (x_ptr == nullptr || a_values_ptr == nullptr || a_meta_e_ptr == nullptr ||
      c_tmp_ptr == nullptr || tile_counters_ptr == nullptr ||
      tile_counter_count <= 0 || M <= 0 || K <= 0 || N <= 0 ||
      (split_k_slices != 2 && split_k_slices != 4 && split_k_slices != 8)) {
    return -1;
  }
  if ((K % (DeviceThreadblockShape::kK * split_k_slices)) != 0 ||
      (N % 256) != 0 || (M % 8) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values =
      reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *c_tmp = reinterpret_cast<Element *>(c_tmp_ptr);
  int *tile_counters = reinterpret_cast<int *>(tile_counters_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  using SplitKGemm = DeviceSparseGemmVec8LayoutBVariant<
      cutlass::layout::ColumnMajor, DeviceThreadblockShape256x32x64,
      DeviceWarpShape64x32x64, 3,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;
  return sparse24_cutlass_device_splitk_gemm_run<SplitKGemm>(
      x, a_values, a_meta_e, c_tmp, tile_counters, tile_counter_count, M, K,
      N, split_k_slices, stream);
}

extern "C" int sparse24_cutlass_device_splitk_indexed_add_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *residual_output_ptr, void *tile_counters_ptr,
    int tile_counter_count, void *full_output_ptr, const void *dense_rows_ptr,
    void *ready_state_ptr, int ready_state_count, int M, int dense_count,
    int full_rows, int K, int N, int split_k_slices, void *stream_ptr) {
  if (x_ptr == nullptr || a_values_ptr == nullptr || a_meta_e_ptr == nullptr ||
      residual_output_ptr == nullptr || tile_counters_ptr == nullptr ||
      full_output_ptr == nullptr || dense_rows_ptr == nullptr ||
      ready_state_ptr == nullptr || tile_counter_count <= 0 ||
      ready_state_count < 2 || M <= 0 || dense_count <= 0 ||
      dense_count > M || full_rows <= 0 || K <= 0 || N <= 0 ||
      (split_k_slices != 2 && split_k_slices != 4 && split_k_slices != 8)) {
    return -1;
  }
  if ((K % (DeviceThreadblockShape::kK * split_k_slices)) != 0 ||
      (N % 256) != 0 || (M % 8) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values =
      reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *residual_output =
      reinterpret_cast<Element *>(residual_output_ptr);
  int *tile_counters = reinterpret_cast<int *>(tile_counters_ptr);
  Element *full_output = reinterpret_cast<Element *>(full_output_ptr);
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  int *ready_state = reinterpret_cast<int *>(ready_state_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  using SplitKGemm = DeviceSparseGemmVec8LayoutBVariant<
      cutlass::layout::ColumnMajor, DeviceThreadblockShape256x32x64,
      DeviceWarpShape64x32x64, 3,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>>;
  return sparse24_cutlass_device_splitk_indexed_add_gemm_run<SplitKGemm>(
      x, a_values, a_meta_e, residual_output, tile_counters,
      tile_counter_count, full_output, dense_rows, ready_state,
      ready_state_count, M, dense_count, full_rows, K, N, split_k_slices,
      stream);
}

extern "C" int sparse24_cutlass_signal_ready_f16_stream(
    void *ready_state_ptr, int ready_state_count, void *stream_ptr) {
  if (ready_state_ptr == nullptr || ready_state_count < 2) {
    return -1;
  }
  int *ready_state = reinterpret_cast<int *>(ready_state_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  sparse24_cutlass_signal_ready_kernel<<<1, 1, 0, stream>>>(ready_state);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_gather_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    const void *row_indices_ptr, void *output_ptr, int rows, int K, int N,
    int config_id, void *stream_ptr) {
  if (x_ptr == nullptr || a_values_ptr == nullptr ||
      a_meta_e_ptr == nullptr || row_indices_ptr == nullptr ||
      output_ptr == nullptr || rows <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 32) != 0 ||
      config_id < 0 || config_id > 4) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values =
      reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  const int *row_indices =
      reinterpret_cast<const int *>(row_indices_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = rows <= 32 ? 1 : 2;
  }
  if (config_id == 1) {
    return sparse24_cutlass_gather_gemm_run<
        DeviceSparseGemmVec8M256N32K64S3Sw4>(
        x, a_values, a_meta_e, row_indices, output, rows, K, N, stream);
  }
  if (config_id == 2) {
    return sparse24_cutlass_gather_gemm_run<
        DeviceSparseGemmVec8M256N64K64S3Sw4>(
        x, a_values, a_meta_e, row_indices, output, rows, K, N, stream);
  }
  return sparse24_cutlass_gather_gemm_run<
      DeviceSparseGemmVec8M128N32K64S4Sw4>(
      x, a_values, a_meta_e, row_indices, output, rows, K, N, stream);
}

extern "C" int sparse24_cutlass_paired_persistent_gemm_f16_stream(
    const void *full_x_ptr, const void *full_values_ptr,
    const void *full_meta_ptr, void *full_output_ptr, int full_rows,
    const void *residual_x_ptr, const void *residual_values_ptr,
    const void *residual_meta_ptr, void *residual_output_ptr,
    int residual_rows, int K, int N, int config_id,
    int interleaved_schedule,
    void *stream_ptr) {
  if (full_x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_ptr == nullptr || full_output_ptr == nullptr ||
      residual_x_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_ptr == nullptr || residual_output_ptr == nullptr ||
      full_rows <= 0 || residual_rows <= 0 || residual_rows > full_rows ||
      K <= 0 || N <= 0 || config_id < 0 || config_id > 4 ||
      interleaved_schedule < 0 ||
      interleaved_schedule > 1) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 256) != 0 ||
      (full_rows % 8) != 0 || (residual_rows % 8) != 0) {
    return -2;
  }
  const Element *full_x = reinterpret_cast<const Element *>(full_x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_ptr));
  Element *full_output = reinterpret_cast<Element *>(full_output_ptr);
  const Element *residual_x =
      reinterpret_cast<const Element *>(residual_x_ptr);
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_ptr));
  Element *residual_output =
      reinterpret_cast<Element *>(residual_output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (config_id == 0 && K == 4096 && N == 6144 && residual_rows >= 72) {
    return sparse24_cutlass_paired_persistent_gemm_run<
        DeviceSparseGemmVec8M256N64K64S3,
        DeviceSparseGemmVec8M128N64K64S5>(
        full_x, full_values, full_meta, full_output, full_rows, residual_x,
        residual_values, residual_meta, residual_output, residual_rows, K, N,
        interleaved_schedule, stream);
  }
  if (config_id == 2) {
    return sparse24_cutlass_paired_persistent_gemm_run<
        DeviceSparseGemmVec8M256N128K64S2,
        DeviceSparseGemmVec8M256N64K64S3>(
        full_x, full_values, full_meta, full_output, full_rows, residual_x,
        residual_values, residual_meta, residual_output, residual_rows, K, N,
        interleaved_schedule, stream);
  }
  if (config_id == 3) {
    return sparse24_cutlass_paired_persistent_gemm_run<
        DeviceSparseGemmVec8M256N128K64S2,
        DeviceSparseGemmVec8M256N128K64S2>(
        full_x, full_values, full_meta, full_output, full_rows, residual_x,
        residual_values, residual_meta, residual_output, residual_rows, K, N,
        interleaved_schedule, stream);
  }
  return sparse24_cutlass_paired_persistent_gemm_run<
      DeviceSparseGemmVec8M256N64K64S3,
      DeviceSparseGemmVec8M256N64K64S3>(
      full_x, full_values, full_meta, full_output, full_rows, residual_x,
      residual_values, residual_meta, residual_output, residual_rows, K, N,
      interleaved_schedule, stream);
}

extern "C" int sparse24_cutlass_paired_gather_residual_gemm_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_ptr, void *full_output_ptr, int full_rows,
    const void *residual_values_ptr, const void *residual_meta_ptr,
    void *residual_output_ptr, const void *dense_rows_ptr, int dense_count,
    int K, int N, int interleaved_schedule, int config_id,
    int worker_blocks, void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_ptr == nullptr || full_output_ptr == nullptr ||
      residual_values_ptr == nullptr || residual_meta_ptr == nullptr ||
      residual_output_ptr == nullptr || dense_rows_ptr == nullptr ||
      full_rows <= 0 || dense_count <= 0 || dense_count > full_rows ||
      K <= 0 || N <= 0 || interleaved_schedule < 0 ||
      interleaved_schedule > 1 || config_id < 0 || config_id > 13 ||
      worker_blocks < 0 || worker_blocks == 1) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 64) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_ptr));
  Element *full_output = reinterpret_cast<Element *>(full_output_ptr);
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_ptr));
  Element *residual_output =
      reinterpret_cast<Element *>(residual_output_ptr);
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = 1;
  }
  if (config_id == 1) {
    return sparse24_cutlass_paired_gather_residual_gemm_run<
        DeviceSparseGemmVec8M256N64K64S3,
        DeviceSparseGemmVec8M256N64K64S3>(
        x, full_values, full_meta, full_output, full_rows, residual_values,
        residual_meta, residual_output, dense_rows, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 2) {
    return sparse24_cutlass_paired_gather_residual_gemm_run<
        DeviceSparseGemmVec8M128N64K64S5,
        DeviceSparseGemmVec8M128N64K64S5>(
        x, full_values, full_meta, full_output, full_rows, residual_values,
        residual_meta, residual_output, dense_rows, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 3) {
    return sparse24_cutlass_paired_gather_residual_gemm_run<
        DeviceSparseGemmVec8M256N32K64S3Sw4,
        DeviceSparseGemmVec8M256N32K64S3Sw4>(
        x, full_values, full_meta, full_output, full_rows, residual_values,
        residual_meta, residual_output, dense_rows, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 4) {
    return sparse24_cutlass_paired_gather_residual_gemm_run<
        DeviceSparseGemmVec8M128N32K64S4Sw4,
        DeviceSparseGemmVec8M128N32K64S4Sw4>(
        x, full_values, full_meta, full_output, full_rows, residual_values,
        residual_meta, residual_output, dense_rows, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 5) {
    return sparse24_cutlass_paired_gather_residual_gemm_run<
        DeviceSparseGemmVec8M64N64K64S5,
        DeviceSparseGemmVec8M64N64K64S5>(
        x, full_values, full_meta, full_output, full_rows, residual_values,
        residual_meta, residual_output, dense_rows, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 6) {
    return sparse24_cutlass_paired_gather_residual_gemm_run<
        DeviceSparseGemmVec8M256N64K64S3,
        DeviceSparseGemmVec8M128N64K64S5>(
        x, full_values, full_meta, full_output, full_rows, residual_values,
        residual_meta, residual_output, dense_rows, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 8) {
    return sparse24_cutlass_paired_gather_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM128N64K64S5,
        DeviceSparseGemmInlineVectorTransposeM128N64K64S5>(
        x, full_values, full_meta, full_output, full_rows, residual_values,
        residual_meta, residual_output, dense_rows, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 9) {
    return sparse24_cutlass_paired_gather_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM256N32K64S3Sw4,
        DeviceSparseGemmInlineVectorTransposeM256N32K64S3Sw4>(
        x, full_values, full_meta, full_output, full_rows, residual_values,
        residual_meta, residual_output, dense_rows, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 10) {
    return sparse24_cutlass_paired_gather_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM64N64K64S5,
        DeviceSparseGemmInlineVectorTransposeM64N64K64S5>(
        x, full_values, full_meta, full_output, full_rows, residual_values,
        residual_meta, residual_output, dense_rows, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 11) {
    return sparse24_cutlass_paired_gather_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM256N32K64S3Sw4,
        DeviceSparseGemmInlineVectorTransposeM128N32K64S4Sw4>(
        x, full_values, full_meta, full_output, full_rows, residual_values,
        residual_meta, residual_output, dense_rows, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 12) {
    return sparse24_cutlass_paired_gather_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM256N64K64S3,
        DeviceSparseGemmInlineVectorTransposeM256N64K64S3>(
        x, full_values, full_meta, full_output, full_rows, residual_values,
        residual_meta, residual_output, dense_rows, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 13) {
    return sparse24_cutlass_paired_gather_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM256N64K64S3,
        DeviceSparseGemmInlineVectorTransposeM256N32K64S3W32x32>(
        x, full_values, full_meta, full_output, full_rows, residual_values,
        residual_meta, residual_output, dense_rows, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  return sparse24_cutlass_paired_gather_residual_gemm_run<
      DeviceSparseGemmVec8M128N64K64S5,
      DeviceSparseGemmVec8M256N64K64S3>(
      x, full_values, full_meta, full_output, full_rows, residual_values,
      residual_meta, residual_output, dense_rows, dense_count, K, N,
      interleaved_schedule, worker_blocks, stream);
}

extern "C" int
sparse24_cutlass_paired_gather_residual_qkv_gemm_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_ptr, void *full_output_ptr, int full_rows,
    const void *residual_values_ptr, const void *residual_meta_ptr,
    void *residual_output_ptr, const void *dense_rows_ptr,
    const void *dense_slot_by_row_ptr, int dense_count, int K, int N,
    const void *q_weight_ptr, const void *k_weight_ptr,
    const void *cos_sin_cache_ptr, const void *position_ids_ptr,
    int q_size, int kv_size, int rotary_dim, float epsilon, int is_neox,
    int normalize_qk, void *grid_barrier_ptr, int interleaved_schedule,
    int config_id, int worker_blocks, void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_ptr == nullptr || full_output_ptr == nullptr ||
      residual_values_ptr == nullptr || residual_meta_ptr == nullptr ||
      residual_output_ptr == nullptr || dense_rows_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || cos_sin_cache_ptr == nullptr ||
      position_ids_ptr == nullptr || grid_barrier_ptr == nullptr ||
      full_rows <= 0 || dense_count <= 0 || dense_count > full_rows ||
      K <= 0 || N <= 0 || q_size <= 0 || kv_size <= 0 ||
      N != q_size + 2 * kv_size || rotary_dim <= 0 || rotary_dim > 128 ||
      rotary_dim % 2 != 0 || epsilon < 0.0f ||
      interleaved_schedule < 0 || interleaved_schedule > 1 ||
      config_id != 13 || worker_blocks < 0 || worker_blocks == 1) {
    return -1;
  }
  if (normalize_qk != 0 &&
      (q_weight_ptr == nullptr || k_weight_ptr == nullptr)) {
    return -2;
  }
  if ((K % 64) != 0 || (N % 256) != 0 || (q_size % 256) != 0 ||
      (kv_size % 256) != 0) {
    return -3;
  }

  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_ptr));
  Element *full_output = reinterpret_cast<Element *>(full_output_ptr);
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_ptr));
  Element *residual_output =
      reinterpret_cast<Element *>(residual_output_ptr);
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  const Element *q_weight =
      reinterpret_cast<const Element *>(q_weight_ptr);
  const Element *k_weight =
      reinterpret_cast<const Element *>(k_weight_ptr);
  const Element *cos_sin_cache =
      reinterpret_cast<const Element *>(cos_sin_cache_ptr);
  const int64_t *position_ids =
      reinterpret_cast<const int64_t *>(position_ids_ptr);
  int *grid_barrier = reinterpret_cast<int *>(grid_barrier_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  return sparse24_cutlass_paired_gather_residual_qkv_visitor_gemm_run<
      DeviceSparseGemmInlineVectorTransposeM256N64K64S3,
      DeviceSparseGemmInlineVectorTransposeM256N32K64S3W32x32>(
      x, full_values, full_meta, full_output, full_rows, residual_values,
      residual_meta, residual_output, dense_rows, dense_slot_by_row,
      dense_count, K, N, q_weight, k_weight, cos_sin_cache, position_ids,
      q_size, kv_size, rotary_dim, epsilon, is_neox, normalize_qk,
      grid_barrier, interleaved_schedule, worker_blocks, stream);
}

extern "C" int
sparse24_cutlass_paired_fused_routed_qkv_epilogue_gemm_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_ptr, const void *residual_values_ptr,
    const void *residual_meta_ptr, const void *dense_rows_ptr,
    const void *dense_slot_by_row_ptr, void *output_ptr,
    void *dense_base_ptr, const void *q_weight_ptr,
    const void *k_weight_ptr, const void *cos_sin_cache_ptr,
    const void *position_ids_ptr, void *feature_counters_ptr,
    int full_rows, int dense_count, int K, int N, int q_size, int kv_size,
    int rotary_dim, float epsilon, int is_neox, int normalize_qk,
    int config_id, int worker_blocks, int residual_worker_blocks,
    void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_ptr == nullptr || dense_rows_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || output_ptr == nullptr ||
      dense_base_ptr == nullptr || cos_sin_cache_ptr == nullptr ||
      position_ids_ptr == nullptr || feature_counters_ptr == nullptr ||
      full_rows <= 0 || dense_count <= 0 || dense_count > full_rows ||
      K <= 0 || N <= 0 || q_size <= 0 || kv_size <= 0 ||
      N != q_size + 2 * kv_size || rotary_dim != 128 || is_neox != 1 ||
      epsilon < 0.0f || config_id != 14 || worker_blocks < 0 ||
      worker_blocks == 1 || residual_worker_blocks < 0) {
    return -1;
  }
  if (normalize_qk != 0 &&
      (q_weight_ptr == nullptr || k_weight_ptr == nullptr)) {
    return -2;
  }
  if ((K % 64) != 0 || (N % 256) != 0 || (q_size % 256) != 0 ||
      (kv_size % 256) != 0) {
    return -3;
  }

  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_ptr));
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_ptr));
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  Element *dense_base = reinterpret_cast<Element *>(dense_base_ptr);
  const Element *q_weight =
      reinterpret_cast<const Element *>(q_weight_ptr);
  const Element *k_weight =
      reinterpret_cast<const Element *>(k_weight_ptr);
  const Element *cos_sin_cache =
      reinterpret_cast<const Element *>(cos_sin_cache_ptr);
  const int64_t *position_ids =
      reinterpret_cast<const int64_t *>(position_ids_ptr);
  int *feature_counters = reinterpret_cast<int *>(feature_counters_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  return sparse24_cutlass_paired_fused_routed_qkv_epilogue_gemm_run<
      DeviceSparseGemmRoutedQKVPostOpM256N64K64S3,
      DeviceSparseGemmResidualQKVPostOpM256N32K64S3>(
      x, full_values, full_meta, residual_values, residual_meta, dense_rows,
      dense_slot_by_row, dense_count, output, dense_base, full_rows, K, N,
      q_weight, k_weight, cos_sin_cache, position_ids, q_size, kv_size,
      rotary_dim, epsilon, is_neox, normalize_qk, feature_counters,
      worker_blocks, residual_worker_blocks, stream);
}

extern "C" int
sparse24_cutlass_paired_finalize_residual_gemm_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_ptr, const void *residual_values_ptr,
    const void *residual_meta_ptr, const void *dense_rows_ptr,
    void *full_output_ptr, void *residual_output_ptr,
    void *feature_counters_ptr, int full_rows, int dense_count, int K, int N,
    int config_id, int worker_blocks, int schedule_mode, void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_ptr == nullptr || dense_rows_ptr == nullptr ||
      full_output_ptr == nullptr || residual_output_ptr == nullptr ||
      feature_counters_ptr == nullptr || full_rows <= 0 || dense_count <= 0 ||
      dense_count >= full_rows || K <= 0 || N <= 0 || config_id < 0 ||
      config_id > 1 || worker_blocks < 0 || worker_blocks == 1 ||
      schedule_mode < 0 || schedule_mode > 1) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 256) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_ptr));
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_ptr));
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  Element *full_output = reinterpret_cast<Element *>(full_output_ptr);
  Element *residual_output =
      reinterpret_cast<Element *>(residual_output_ptr);
  int *feature_counters = reinterpret_cast<int *>(feature_counters_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  return sparse24_cutlass_paired_finalize_residual_visitor_gemm_run<
      DeviceSparseGemmInlineVectorTransposeM256N32K64S3Sw4,
      DeviceSparseGemmInlineVectorTransposeM256N32K64S3Sw4>(
      x, full_values, full_meta, residual_values, residual_meta, dense_rows,
      dense_count, full_output, residual_output, full_rows, K, N,
      feature_counters, worker_blocks, schedule_mode, stream);
}

extern "C" int sparse24_cutlass_paired_finalize_qkv_gemm_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_ptr, const void *residual_values_ptr,
    const void *residual_meta_ptr, const void *dense_rows_ptr,
    void *full_output_ptr, void *residual_output_ptr,
    void *feature_counters_ptr, const void *q_weight_ptr,
    const void *k_weight_ptr, const void *cos_sin_cache_ptr,
    const void *position_ids_ptr, int full_rows, int dense_count, int K, int N,
    int q_size, int kv_size, int rotary_dim, float epsilon, int is_neox,
    int normalize_qk, int config_id, int worker_blocks, int schedule_mode,
    void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_ptr == nullptr || dense_rows_ptr == nullptr ||
      full_output_ptr == nullptr || residual_output_ptr == nullptr ||
      feature_counters_ptr == nullptr || cos_sin_cache_ptr == nullptr ||
      position_ids_ptr == nullptr || full_rows <= 0 || dense_count <= 0 ||
      dense_count >= full_rows || K <= 0 || N <= 0 || config_id < 1 ||
      config_id > 3 || worker_blocks < 0 || worker_blocks == 1 ||
      schedule_mode != 0 || q_size <= 0 || kv_size <= 0 ||
      q_size + 2 * kv_size != N || rotary_dim != 128 || is_neox == 0) {
    return -1;
  }
  if (normalize_qk != 0 &&
      (q_weight_ptr == nullptr || k_weight_ptr == nullptr)) {
    return -2;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 256) != 0 ||
      (q_size % 256) != 0 || (kv_size % 256) != 0) {
    return -3;
  }

  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_ptr));
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_ptr));
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  Element *full_output = reinterpret_cast<Element *>(full_output_ptr);
  Element *residual_output =
      reinterpret_cast<Element *>(residual_output_ptr);
  int *feature_counters = reinterpret_cast<int *>(feature_counters_ptr);
  const Element *q_weight =
      reinterpret_cast<const Element *>(q_weight_ptr);
  const Element *k_weight =
      reinterpret_cast<const Element *>(k_weight_ptr);
  const Element *cos_sin_cache =
      reinterpret_cast<const Element *>(cos_sin_cache_ptr);
  const int64_t *position_ids =
      reinterpret_cast<const int64_t *>(position_ids_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

#define SPECLINK_RUN_FINALIZE_QKV(FULL_GEMM, RESIDUAL_GEMM)                  \
  return sparse24_cutlass_paired_finalize_residual_visitor_gemm_run<        \
      FULL_GEMM, RESIDUAL_GEMM, true>(                                      \
      x, full_values, full_meta, residual_values, residual_meta, dense_rows, \
      dense_count, full_output, residual_output, full_rows, K, N,            \
      feature_counters, worker_blocks, schedule_mode, stream, q_weight,      \
      k_weight, cos_sin_cache, position_ids, q_size, kv_size, rotary_dim,     \
      epsilon, is_neox, normalize_qk)

  if (config_id == 1) {
    SPECLINK_RUN_FINALIZE_QKV(
        DeviceSparseGemmInlineVectorTransposeM256N32K64S3Sw4,
        DeviceSparseGemmInlineVectorTransposeM256N32K64S3Sw4);
  }
  if (config_id == 2) {
    SPECLINK_RUN_FINALIZE_QKV(
        DeviceSparseGemmInlineVectorTransposeM256N64K64S3,
        DeviceSparseGemmInlineVectorTransposeM256N64K64S3);
  }
  SPECLINK_RUN_FINALIZE_QKV(
      DeviceSparseGemmInlineVectorTransposeM256N64K64S3,
      DeviceSparseGemmInlineVectorTransposeM256N32K64S3W32x32);

#undef SPECLINK_RUN_FINALIZE_QKV
}

extern "C" int
sparse24_cutlass_paired_inplace_residual_gemm_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_ptr, const void *residual_values_ptr,
    const void *residual_meta_ptr, const void *dense_rows_ptr,
    void *output_ptr, void *feature_counters_ptr, int full_rows,
    int dense_count, int K, int N, int config_id, int worker_blocks,
    int schedule_mode, void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_ptr == nullptr || dense_rows_ptr == nullptr ||
      output_ptr == nullptr || feature_counters_ptr == nullptr ||
      full_rows <= 0 || dense_count <= 0 || dense_count >= full_rows ||
      K <= 0 || N <= 0 || config_id < 0 || config_id > 9 ||
      worker_blocks < 0 || worker_blocks == 1 || schedule_mode < 0 ||
      schedule_mode > 2) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 256) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_ptr));
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_ptr));
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  int *feature_counters = reinterpret_cast<int *>(feature_counters_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = 1;
  }
  if (config_id == 1) {
    return sparse24_cutlass_paired_inplace_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM128N64K64S5,
        DeviceSparseGemmInlineIndexedAddTransposeM128N64K64S5>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, output, full_rows, K, N, feature_counters, worker_blocks,
        schedule_mode, stream);
  }
  if (config_id == 5) {
    return sparse24_cutlass_paired_inplace_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM128N32K64S4Sw4,
        DeviceSparseGemmInlineIndexedAddTransposeM128N32K64S4Sw4>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, output, full_rows, K, N, feature_counters, worker_blocks,
        schedule_mode, stream);
  }
  if (config_id == 6) {
    return sparse24_cutlass_paired_inplace_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM64N64K64S5W32x64,
        DeviceSparseGemmInlineIndexedAddTransposeM64N32K64S3>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, output, full_rows, K, N, feature_counters, worker_blocks,
        schedule_mode, stream);
  }
  if (config_id == 7) {
    return sparse24_cutlass_paired_inplace_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM256N32K64S3Sw4,
        DeviceSparseGemmInlineIndexedAddTransposeM256N32K64S3Sw4, true>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, output, full_rows, K, N, feature_counters, worker_blocks,
        schedule_mode, stream);
  }
  if (config_id == 8) {
    return sparse24_cutlass_paired_inplace_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM256N32K64S3Sw4,
        DeviceSparseGemmInlineIndexedAddTransposeF16M256N32K64S3Sw4>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, output, full_rows, K, N, feature_counters, worker_blocks,
        schedule_mode, stream);
  }
  if (config_id == 9) {
    return sparse24_cutlass_paired_inplace_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeF16M256N32K64S3Sw4,
        DeviceSparseGemmInlineIndexedAddTransposeF16M256N32K64S3Sw4>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, output, full_rows, K, N, feature_counters, worker_blocks,
        schedule_mode, stream);
  }
  if (config_id == 4) {
    return sparse24_cutlass_paired_inplace_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM256N64K64S3W64x64,
        DeviceSparseGemmInlineIndexedAddTransposeM256N32K64S3Sw4>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, output, full_rows, K, N, feature_counters, worker_blocks,
        schedule_mode, stream);
  }
  if (config_id == 3) {
    return sparse24_cutlass_paired_inplace_residual_visitor_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM256N64K64S3,
        DeviceSparseGemmInlineIndexedAddTransposeM256N32K64S3W32x32>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, output, full_rows, K, N, feature_counters, worker_blocks,
        schedule_mode, stream);
  }
  return sparse24_cutlass_paired_inplace_residual_visitor_gemm_run<
      DeviceSparseGemmInlineVectorTransposeM256N32K64S3Sw4,
      DeviceSparseGemmInlineIndexedAddTransposeM256N32K64S3Sw4>(
      x, full_values, full_meta, residual_values, residual_meta, dense_rows,
      dense_count, output, full_rows, K, N, feature_counters, worker_blocks,
      schedule_mode, stream);
}

extern "C" int
sparse24_cutlass_paired_persistent_routed_swiglu_f16_stream(
    const void *full_x_ptr, const void *full_values_ptr,
    const void *full_meta_ptr, void *full_output_ptr, void *dense_base_ptr,
    const void *dense_slot_by_row_ptr, int full_rows, int dense_rows,
    const void *residual_x_ptr, const void *residual_values_ptr,
    const void *residual_meta_ptr, void *residual_output_ptr,
    int residual_rows, int K, int N, int interleaved_schedule, int config_id,
    int worker_blocks, void *stream_ptr) {
  if (full_x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_ptr == nullptr || full_output_ptr == nullptr ||
      dense_base_ptr == nullptr || dense_slot_by_row_ptr == nullptr ||
      residual_x_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_ptr == nullptr || residual_output_ptr == nullptr ||
      full_rows <= 0 || dense_rows <= 0 || dense_rows > full_rows ||
      residual_rows < dense_rows || residual_rows > full_rows || K <= 0 ||
      N <= 0 || interleaved_schedule < 0 || interleaved_schedule > 1 ||
      config_id < 0 || config_id > 5 || worker_blocks < 0 ||
      worker_blocks == 1) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 256) != 0 ||
      (full_rows % 8) != 0 || (residual_rows % 8) != 0) {
    return -2;
  }
  const Element *full_x = reinterpret_cast<const Element *>(full_x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_ptr));
  Element *full_output = reinterpret_cast<Element *>(full_output_ptr);
  Element *dense_base = reinterpret_cast<Element *>(dense_base_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  const Element *residual_x =
      reinterpret_cast<const Element *>(residual_x_ptr);
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_ptr));
  Element *residual_output =
      reinterpret_cast<Element *>(residual_output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 3) {
    return sparse24_cutlass_paired_persistent_routed_swiglu_gemm_run<
        DeviceSparseGemmInlineRoutedSwiGLUF16M256N32K64S3Sw4,
        DeviceSparseGemmInlineVectorTransposeF16M256N32K64S3Sw4>(
        full_x, full_values, full_meta, full_output, dense_base,
        dense_slot_by_row, full_rows, dense_rows, residual_x, nullptr,
        residual_values, residual_meta, residual_output, residual_rows, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 4) {
    return sparse24_cutlass_paired_persistent_routed_swiglu_gemm_run<
        DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S2Sw4,
        DeviceSparseGemmInlineVectorTransposeF16M256N64K64S2Sw4>(
        full_x, full_values, full_meta, full_output, dense_base,
        dense_slot_by_row, full_rows, dense_rows, residual_x, nullptr,
        residual_values, residual_meta, residual_output, residual_rows, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 5) {
    return sparse24_cutlass_paired_persistent_routed_swiglu_gemm_run<
        DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3Sw4,
        DeviceSparseGemmInlineVectorTransposeF16M256N32K64S3W32x32Sw4>(
        full_x, full_values, full_meta, full_output, dense_base,
        dense_slot_by_row, full_rows, dense_rows, residual_x, nullptr,
        residual_values, residual_meta, residual_output, residual_rows, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  using FullGemm = DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3Sw4;
  using ResidualGemm =
      DeviceSparseGemmInlineVectorTransposeF16M256N64K64S3Sw4;
  return sparse24_cutlass_paired_persistent_routed_swiglu_gemm_run<
      FullGemm, ResidualGemm>(
      full_x, full_values, full_meta, full_output, dense_base,
      dense_slot_by_row, full_rows, dense_rows, residual_x, nullptr,
      residual_values, residual_meta, residual_output, residual_rows, K, N,
      interleaved_schedule, worker_blocks, stream);
}

extern "C" int
sparse24_cutlass_paired_gather_routed_swiglu_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_ptr, void *full_output_ptr, void *dense_base_ptr,
    const void *dense_slot_by_row_ptr, const void *dense_rows_ptr,
    const void *residual_values_ptr, const void *residual_meta_ptr,
    void *residual_output_ptr, int full_rows, int dense_count, int K, int N,
    int interleaved_schedule, int config_id, int worker_blocks,
    void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_ptr == nullptr || full_output_ptr == nullptr ||
      dense_base_ptr == nullptr || dense_slot_by_row_ptr == nullptr ||
      dense_rows_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_ptr == nullptr || residual_output_ptr == nullptr ||
      full_rows <= 0 || dense_count <= 0 || dense_count > full_rows ||
      K <= 0 || N <= 0 || interleaved_schedule < 0 ||
      interleaved_schedule > 1 || config_id < 0 || config_id > 5 ||
      worker_blocks < 0 || worker_blocks == 1) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 256) != 0 ||
      (full_rows % 8) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_ptr));
  Element *full_output = reinterpret_cast<Element *>(full_output_ptr);
  Element *dense_base = reinterpret_cast<Element *>(dense_base_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_ptr));
  Element *residual_output =
      reinterpret_cast<Element *>(residual_output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 3) {
    return sparse24_cutlass_paired_persistent_routed_swiglu_gemm_run<
        DeviceSparseGemmInlineRoutedSwiGLUF16M256N32K64S3Sw4,
        DeviceSparseGemmInlineVectorTransposeF16M256N32K64S3Sw4, true>(
        x, full_values, full_meta, full_output, dense_base,
        dense_slot_by_row, full_rows, dense_count, x, dense_rows,
        residual_values, residual_meta, residual_output, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 4) {
    return sparse24_cutlass_paired_persistent_routed_swiglu_gemm_run<
        DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S2Sw4,
        DeviceSparseGemmInlineVectorTransposeF16M256N64K64S2Sw4, true>(
        x, full_values, full_meta, full_output, dense_base,
        dense_slot_by_row, full_rows, dense_count, x, dense_rows,
        residual_values, residual_meta, residual_output, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  if (config_id == 5) {
    return sparse24_cutlass_paired_persistent_routed_swiglu_gemm_run<
        DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3Sw4,
        DeviceSparseGemmInlineVectorTransposeF16M256N32K64S3W32x32Sw4,
        true>(
        x, full_values, full_meta, full_output, dense_base,
        dense_slot_by_row, full_rows, dense_count, x, dense_rows,
        residual_values, residual_meta, residual_output, dense_count, K, N,
        interleaved_schedule, worker_blocks, stream);
  }
  using FullGemm = DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3Sw4;
  using ResidualGemm =
      DeviceSparseGemmInlineVectorTransposeF16M256N64K64S3Sw4;
  return sparse24_cutlass_paired_persistent_routed_swiglu_gemm_run<
      FullGemm, ResidualGemm, true>(
      x, full_values, full_meta, full_output, dense_base,
      dense_slot_by_row, full_rows, dense_count, x, dense_rows,
      residual_values, residual_meta, residual_output, dense_count, K, N,
      interleaved_schedule, worker_blocks, stream);
}

extern "C" int
sparse24_cutlass_gate_dense_down_pipeline_f16_stream(
    const void *x_ptr, const void *gate_values_ptr,
    const void *gate_meta_ptr, void *hidden_ptr,
    const void *down_weight_ptr, void *output_ptr,
    void *row_counters_ptr, int rows, int model_width,
    int intermediate_size, int config_id, int worker_blocks,
    int stage_mode, void *stream_ptr) {
  if (x_ptr == nullptr || gate_values_ptr == nullptr ||
      gate_meta_ptr == nullptr || hidden_ptr == nullptr ||
      down_weight_ptr == nullptr || output_ptr == nullptr ||
      row_counters_ptr == nullptr || rows <= 0 || model_width <= 0 ||
      intermediate_size <= 0 || config_id < 1 || config_id > 4 ||
      worker_blocks < 0 ||
      worker_blocks == 1 || stage_mode < 0 || stage_mode > 2) {
    return -1;
  }
  if ((rows % 8) != 0 || (model_width % 64) != 0 ||
      (intermediate_size % 128) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *gate_values =
      reinterpret_cast<const Element *>(gate_values_ptr);
  DeviceElementE *gate_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(gate_meta_ptr));
  Element *hidden = reinterpret_cast<Element *>(hidden_ptr);
  const Element *down_weight =
      reinterpret_cast<const Element *>(down_weight_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  int *row_counters = reinterpret_cast<int *>(row_counters_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (config_id == 2) {
    return sparse24_cutlass_gate_dense_down_pipeline_gemm_run<
        DeviceSparseGemmInlineSwiGLUF16M256N64K64S3Sw4,
        DeviceDenseGemmInlineVectorTransposeF16M64N128K64S3>(
        x, gate_values, gate_meta, hidden, down_weight, nullptr, output, rows,
        model_width, intermediate_size, row_counters, worker_blocks,
        stage_mode, stream);
  }
  if (config_id == 3) {
    return sparse24_cutlass_gate_dense_down_pipeline_gemm_run<
        DeviceSparseGemmInlineSwiGLUF16M256N64K64S3Sw4,
        DeviceDenseGemmInlineVectorTransposeF16M128N128K64S3>(
        x, gate_values, gate_meta, hidden, down_weight, nullptr, output, rows,
        model_width, intermediate_size, row_counters, worker_blocks,
        stage_mode, stream);
  }
  if (config_id == 4) {
    return sparse24_cutlass_gate_dense_down_pipeline_gemm_run<
        DeviceSparseGemmInlineSwiGLUF16M128N64K64S3Sw4,
        DeviceDenseGemmInlineVectorTransposeF16M64N128K64S3W32x64>(
        x, gate_values, gate_meta, hidden, down_weight, nullptr, output, rows,
        model_width, intermediate_size, row_counters, worker_blocks,
        stage_mode, stream);
  }
  return sparse24_cutlass_gate_dense_down_pipeline_gemm_run<
      DeviceSparseGemmInlineSwiGLUF16M256N64K64S3Sw4,
      DeviceDenseGemmInlineVectorTransposeF16M64N64K64S3W16x32>(
      x, gate_values, gate_meta, hidden, down_weight, nullptr, output, rows,
      model_width, intermediate_size, row_counters, worker_blocks,
      stage_mode, stream);
}

extern "C" int
sparse24_cutlass_gate_sparse_down_pipeline_f16_stream(
    const void *x_ptr, const void *gate_values_ptr,
    const void *gate_meta_ptr, void *hidden_ptr,
    const void *down_values_ptr, const void *down_meta_ptr,
    void *output_ptr, void *row_counters_ptr, int rows,
    int model_width, int intermediate_size, int config_id,
    int worker_blocks, int stage_mode, void *stream_ptr) {
  if (x_ptr == nullptr || gate_values_ptr == nullptr ||
      gate_meta_ptr == nullptr || hidden_ptr == nullptr ||
      down_values_ptr == nullptr || down_meta_ptr == nullptr ||
      output_ptr == nullptr || row_counters_ptr == nullptr || rows <= 0 ||
      model_width <= 0 || intermediate_size <= 0 || config_id < 1 ||
      config_id > 7 || worker_blocks < 0 || worker_blocks == 1 ||
      stage_mode < 0 || stage_mode > 2) {
    return -1;
  }
  if ((rows % 8) != 0 || (model_width % 128) != 0 ||
      (intermediate_size % 128) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *gate_values =
      reinterpret_cast<const Element *>(gate_values_ptr);
  DeviceElementE *gate_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(gate_meta_ptr));
  Element *hidden = reinterpret_cast<Element *>(hidden_ptr);
  const Element *down_values =
      reinterpret_cast<const Element *>(down_values_ptr);
  DeviceElementE *down_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(down_meta_ptr));
  Element *output = reinterpret_cast<Element *>(output_ptr);
  int *row_counters = reinterpret_cast<int *>(row_counters_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (config_id == 7) {
    return sparse24_cutlass_gate_dense_down_pipeline_gemm_run<
        DeviceSparseGemmInlineSwiGLUF16M128N64K64S3Sw4,
        DeviceSparseGemmInlineVectorTransposeF16M128N64K64S3Sw4, true,
        false, true>(
        x, gate_values, gate_meta, hidden, down_values, down_meta, output,
        rows, model_width, intermediate_size, row_counters, worker_blocks,
        stage_mode, stream);
  }
  if (config_id == 6) {
    return sparse24_cutlass_gate_dense_down_pipeline_gemm_run<
        DeviceSparseGemmInlineSwiGLUF16M256N32K64S3Sw4,
        DeviceSparseGemmInlineVectorTransposeF16M256N32K64S3Sw4, true,
        true>(
        x, gate_values, gate_meta, hidden, down_values, down_meta, output,
        rows, model_width, intermediate_size, row_counters, worker_blocks,
        stage_mode, stream);
  }
  if (config_id == 5) {
    return sparse24_cutlass_gate_dense_down_pipeline_gemm_run<
        DeviceSparseGemmInlineSwiGLUF16M128N64K64S3Sw4,
        DeviceSparseGemmInlineVectorTransposeF16M128N64K64S3Sw4, true,
        true>(
        x, gate_values, gate_meta, hidden, down_values, down_meta, output,
        rows, model_width, intermediate_size, row_counters, worker_blocks,
        stage_mode, stream);
  }
  if (config_id == 4) {
    return sparse24_cutlass_gate_dense_down_pipeline_gemm_run<
        DeviceSparseGemmInlineSwiGLUF16M256N64K64S3Sw4,
        DeviceSparseGemmInlineVectorTransposeF16M256N64K64S3Sw4, true,
        true>(
        x, gate_values, gate_meta, hidden, down_values, down_meta, output,
        rows, model_width, intermediate_size, row_counters, worker_blocks,
        stage_mode, stream);
  }
  if (config_id == 3) {
    return sparse24_cutlass_gate_dense_down_pipeline_gemm_run<
        DeviceSparseGemmInlineSwiGLUF16M256N32K64S3Sw4,
        DeviceSparseGemmInlineVectorTransposeF16M256N32K64S3Sw4, true>(
        x, gate_values, gate_meta, hidden, down_values, down_meta, output,
        rows, model_width, intermediate_size, row_counters, worker_blocks,
        stage_mode, stream);
  }
  if (config_id == 2) {
    return sparse24_cutlass_gate_dense_down_pipeline_gemm_run<
        DeviceSparseGemmInlineSwiGLUF16M128N64K64S3Sw4,
        DeviceSparseGemmInlineVectorTransposeF16M128N64K64S3Sw4, true>(
        x, gate_values, gate_meta, hidden, down_values, down_meta, output,
        rows, model_width, intermediate_size, row_counters, worker_blocks,
        stage_mode, stream);
  }
  return sparse24_cutlass_gate_dense_down_pipeline_gemm_run<
      DeviceSparseGemmInlineSwiGLUF16M256N64K64S3Sw4,
      DeviceSparseGemmInlineVectorTransposeF16M256N64K64S3Sw4, true>(
      x, gate_values, gate_meta, hidden, down_values, down_meta, output, rows,
      model_width, intermediate_size, row_counters, worker_blocks,
      stage_mode, stream);
}

extern "C" int
sparse24_cutlass_paired_fused_routed_swiglu_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_ptr, const void *residual_x_ptr,
    const void *residual_values_ptr,
    const void *residual_meta_ptr, const void *dense_rows_ptr,
    const void *dense_slot_by_row_ptr, void *output_ptr,
    void *dense_base_ptr, void *feature_counters_ptr, int full_rows,
    int dense_count, int residual_rows, int K, int N, int config_id,
    int worker_blocks, int gather_residual_rows, int schedule_mode,
    void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_ptr == nullptr || residual_x_ptr == nullptr ||
      residual_values_ptr == nullptr ||
      residual_meta_ptr == nullptr || dense_rows_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || output_ptr == nullptr ||
      dense_base_ptr == nullptr || feature_counters_ptr == nullptr ||
      full_rows <= 0 || dense_count <= 0 || dense_count >= full_rows ||
      residual_rows < dense_count || residual_rows > full_rows || K <= 0 ||
      N <= 0 || config_id < 0 || config_id > 6 || worker_blocks < 0 ||
      worker_blocks == 1 || gather_residual_rows < 0 ||
      gather_residual_rows > 1 ||
      (schedule_mode != 0 && schedule_mode != 2)) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 256) != 0 ||
      (full_rows % 8) != 0 ||
      (!gather_residual_rows && (residual_rows % 8) != 0)) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_ptr));
  const Element *residual_x =
      reinterpret_cast<const Element *>(residual_x_ptr);
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_ptr));
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  Element *dense_base = reinterpret_cast<Element *>(dense_base_ptr);
  int *feature_counters = reinterpret_cast<int *>(feature_counters_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

#define RUN_FUSED_ROUTED_SWIGLU(FULL_GEMM, RESIDUAL_GEMM)                  \
  do {                                                                     \
    if (gather_residual_rows) {                                             \
      return sparse24_cutlass_paired_fused_routed_swiglu_gemm_run<         \
          FULL_GEMM, RESIDUAL_GEMM, true>(                                 \
          x, full_values, full_meta, residual_x, residual_values,           \
          residual_meta, dense_rows, dense_slot_by_row, dense_count,        \
          output, dense_base, full_rows, residual_rows, K, N,               \
          feature_counters, worker_blocks, schedule_mode, stream);          \
    }                                                                      \
    return sparse24_cutlass_paired_fused_routed_swiglu_gemm_run<           \
        FULL_GEMM, RESIDUAL_GEMM, false>(                                  \
        x, full_values, full_meta, residual_x, residual_values,             \
        residual_meta, dense_rows, dense_slot_by_row, dense_count, output,  \
        dense_base, full_rows, residual_rows, K, N, feature_counters,       \
        worker_blocks, schedule_mode, stream);                              \
  } while (false)

  if (config_id == 1) {
    RUN_FUSED_ROUTED_SWIGLU(
        DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3,
        DeviceSparseGemmResidualCorrectionSwiGLUF16M256N64K64S3);
  }
  if (config_id == 3) {
    RUN_FUSED_ROUTED_SWIGLU(
        DeviceSparseGemmInlineRoutedSwiGLUF16M256N32K64S3Sw4,
        DeviceSparseGemmResidualCorrectionSwiGLUF16M256N32K64S3Sw4);
  }
  if (config_id == 4) {
    RUN_FUSED_ROUTED_SWIGLU(
        DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S2Sw4,
        DeviceSparseGemmResidualCorrectionSwiGLUF16M256N64K64S2Sw4);
  }
  if (config_id == 5) {
    RUN_FUSED_ROUTED_SWIGLU(
        DeviceSparseGemmInlineRoutedFastSwiGLUF16M256N64K64S3Sw4,
        DeviceSparseGemmResidualCorrectionFastSwiGLUF16M256N64K64S3Sw4);
  }
  if (config_id == 6) {
    RUN_FUSED_ROUTED_SWIGLU(
        DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3Sw4,
        DeviceSparseGemmResidualCorrectionSwiGLUF16M256N32K64S3W32x32Sw4);
  }
  RUN_FUSED_ROUTED_SWIGLU(
      DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3Sw4,
      DeviceSparseGemmResidualCorrectionSwiGLUF16M256N64K64S3Sw4);
#undef RUN_FUSED_ROUTED_SWIGLU
}

extern "C" int sparse24_cutlass_fused_mixed_mlp_f16_stream(
    const void *x_ptr, const void *gate_full_values_ptr,
    const void *gate_full_meta_ptr, const void *gate_residual_values_ptr,
    const void *gate_residual_meta_ptr, const void *down_full_values_ptr,
    const void *down_full_meta_ptr, const void *down_residual_values_ptr,
    const void *down_residual_meta_ptr, const void *dense_rows_ptr,
    const void *dense_slot_by_row_ptr, void *hidden_ptr,
    void *gate_dense_base_ptr, void *output_ptr,
    void *gate_feature_counters_ptr, void *down_feature_counters_ptr,
    void *grid_barrier_ptr, int rows, int dense_count, int model_width,
    int intermediate_size, int config_id, int worker_blocks,
    void *stream_ptr) {
  if (x_ptr == nullptr || gate_full_values_ptr == nullptr ||
      gate_full_meta_ptr == nullptr || gate_residual_values_ptr == nullptr ||
      gate_residual_meta_ptr == nullptr || down_full_values_ptr == nullptr ||
      down_full_meta_ptr == nullptr || down_residual_values_ptr == nullptr ||
      down_residual_meta_ptr == nullptr || dense_rows_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || hidden_ptr == nullptr ||
      gate_dense_base_ptr == nullptr || output_ptr == nullptr ||
      gate_feature_counters_ptr == nullptr ||
      down_feature_counters_ptr == nullptr || grid_barrier_ptr == nullptr ||
      rows <= 0 || dense_count <= 0 || dense_count >= rows ||
      model_width <= 0 || intermediate_size <= 0 || config_id < 0 ||
      config_id > 1 || worker_blocks < 0 || worker_blocks == 1) {
    return -1;
  }
  if ((rows % 8) != 0 || (model_width % 256) != 0 ||
      ((2 * intermediate_size) % 256) != 0 ||
      (model_width % DeviceThreadblockShape::kK) != 0 ||
      (intermediate_size % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }

  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *gate_full_values =
      reinterpret_cast<const Element *>(gate_full_values_ptr);
  DeviceElementE *gate_full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(gate_full_meta_ptr));
  const Element *gate_residual_values =
      reinterpret_cast<const Element *>(gate_residual_values_ptr);
  DeviceElementE *gate_residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(gate_residual_meta_ptr));
  const Element *down_full_values =
      reinterpret_cast<const Element *>(down_full_values_ptr);
  DeviceElementE *down_full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(down_full_meta_ptr));
  const Element *down_residual_values =
      reinterpret_cast<const Element *>(down_residual_values_ptr);
  DeviceElementE *down_residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(down_residual_meta_ptr));
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *hidden = reinterpret_cast<Element *>(hidden_ptr);
  Element *gate_dense_base =
      reinterpret_cast<Element *>(gate_dense_base_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  int *gate_feature_counters =
      reinterpret_cast<int *>(gate_feature_counters_ptr);
  int *down_feature_counters =
      reinterpret_cast<int *>(down_feature_counters_ptr);
  int *grid_barrier = reinterpret_cast<int *>(grid_barrier_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  return sparse24_cutlass_fused_mixed_mlp_gemm_run<
      DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3Sw4,
      DeviceSparseGemmResidualCorrectionSwiGLUF16M256N64K64S3Sw4,
      DeviceSparseGemmInlineVectorTransposeF16M256N64K64S3Sw4,
      DeviceSparseGemmInlineIndexedAddTransposeF16M256N32K64S3W32x32Sw4>(
      x, gate_full_values, gate_full_meta, gate_residual_values,
      gate_residual_meta, down_full_values, down_full_meta,
      down_residual_values, down_residual_meta, dense_rows,
      dense_slot_by_row, hidden, gate_dense_base, output,
      gate_feature_counters, down_feature_counters, grid_barrier, rows,
      dense_count, model_width, intermediate_size, worker_blocks, stream);
}

extern "C" int
sparse24_cutlass_paired_self_contained_routed_swiglu_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_ptr, const void *residual_values_ptr,
    const void *residual_meta_ptr, const void *dense_rows_ptr,
    const void *dense_slot_by_row_ptr, void *output_ptr,
    void *dense_base_ptr, int full_rows, int dense_count, int K, int N,
    int config_id, int worker_blocks, void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_ptr == nullptr || dense_rows_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || output_ptr == nullptr ||
      dense_base_ptr == nullptr || full_rows <= 0 || dense_count <= 0 ||
      dense_count >= full_rows || K <= 0 || N <= 0 || config_id < 0 ||
      config_id > 1 || worker_blocks < 0 || worker_blocks == 1) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 256) != 0 ||
      (full_rows % 8) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_ptr));
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_ptr));
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  Element *dense_base = reinterpret_cast<Element *>(dense_base_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 1) {
    return sparse24_cutlass_paired_self_contained_routed_swiglu_gemm_run<
        DeviceSparseGemmInlineRoutedSwiGLUF16M256N32K64S3Sw4,
        DeviceSparseGemmResidualCorrectionSwiGLUF16M256N32K64S3Sw4>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_slot_by_row, dense_count, output, dense_base, full_rows, K, N,
        worker_blocks, stream);
  }
  return sparse24_cutlass_paired_self_contained_routed_swiglu_gemm_run<
      DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3Sw4,
      DeviceSparseGemmResidualCorrectionSwiGLUF16M256N64K64S3Sw4>(
      x, full_values, full_meta, residual_values, residual_meta, dense_rows,
      dense_slot_by_row, dense_count, output, dense_base, full_rows, K, N,
      worker_blocks, stream);
}

extern "C" int
sparse24_cutlass_paired_self_contained_exact_down_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_ptr, const void *residual_values_ptr,
    const void *residual_meta_ptr, const void *dense_rows_ptr,
    const void *dense_slot_by_row_ptr, void *output_ptr,
    void *dense_base_ptr, int full_rows, int dense_count, int K, int N,
    int config_id, int worker_blocks, void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_ptr == nullptr || dense_rows_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || output_ptr == nullptr ||
      dense_base_ptr == nullptr || full_rows <= 0 || dense_count <= 0 ||
      dense_count >= full_rows || K <= 0 || N <= 0 || config_id != 0 ||
      worker_blocks < 0 || worker_blocks == 1) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 256) != 0 ||
      (full_rows % 8) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_ptr));
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_ptr));
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  Element *dense_base = reinterpret_cast<Element *>(dense_base_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  return sparse24_cutlass_paired_self_contained_exact_down_gemm_run<
      DeviceSparseGemmInlineRoutedTransposeM256N32K64S3Sw4,
      DeviceSparseGemmInlineIndexedCorrectionTransposeM256N32K64S3Sw4>(
      x, full_values, full_meta, residual_values, residual_meta, dense_rows,
      dense_slot_by_row, dense_count, output, dense_base, full_rows, K, N,
      worker_blocks, stream);
}

extern "C" int sparse24_cutlass_inline_transpose_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *output_ptr, int M, int K, int N, void *stream_ptr) {
  if (M <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 32) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const char *config =
      std::getenv("SPECLINK_SPARSE24_INLINE_EPILOGUE_CONFIG");
  if (config != nullptr && std::strcmp(config, "auto") == 0) {
    config = sparse24_cutlass_select_device_config(nullptr, M, K, N);
  }
  const char *store =
      std::getenv("SPECLINK_SPARSE24_INLINE_EPILOGUE_STORE");
  bool vector_store = store != nullptr && std::strcmp(store, "vector") == 0;
  const char *accumulator =
      std::getenv("SPECLINK_SPARSE24_ACCUMULATOR");
  bool use_f16_accumulator =
      sparse24_cutlass_use_f16_accumulator(accumulator, K, N);
  int status = -5;
  if (vector_store && config != nullptr &&
      std::strcmp(config, "64x32x64_s3") == 0) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_inline_transpose_gemm_run<
          DeviceSparseGemmInlineVectorTransposeF16M64N32K64S3>(
          x, a_values, a_meta_e, output, M, K, N, stream);
    } else {
      status = sparse24_cutlass_inline_transpose_gemm_run<
          DeviceSparseGemmInlineVectorTransposeM64N32K64S3>(
          x, a_values, a_meta_e, output, M, K, N, stream);
    }
  } else if (vector_store && config != nullptr &&
             std::strcmp(config, "64x64x64_s5") == 0) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_inline_transpose_gemm_run<
          DeviceSparseGemmInlineVectorTransposeF16M64N64K64S5>(
          x, a_values, a_meta_e, output, M, K, N, stream);
    } else {
      status = sparse24_cutlass_inline_transpose_gemm_run<
          DeviceSparseGemmInlineVectorTransposeM64N64K64S5>(
          x, a_values, a_meta_e, output, M, K, N, stream);
    }
  } else if (vector_store && config != nullptr &&
      std::strcmp(config, "64x64x64_s6") == 0) {
    status = sparse24_cutlass_inline_transpose_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM64N64K64S6>(
        x, a_values, a_meta_e, output, M, K, N, stream);
  } else if (vector_store && config != nullptr &&
             std::strcmp(config, "128x64x64_s5") == 0) {
    status = sparse24_cutlass_inline_transpose_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM128N64K64S5>(
        x, a_values, a_meta_e, output, M, K, N, stream);
  } else if (vector_store && config != nullptr &&
             std::strcmp(config, "128x32x64_s4_sw4") == 0) {
    status = sparse24_cutlass_inline_transpose_gemm_run<
        DeviceSparseGemmInlineVectorTransposeM128N32K64S4Sw4>(
        x, a_values, a_meta_e, output, M, K, N, stream);
  } else if (vector_store && config != nullptr &&
             std::strcmp(config, "256x32x64_s3_sw4") == 0) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_inline_transpose_gemm_run<
          DeviceSparseGemmInlineVectorTransposeF16M256N32K64S3Sw4>(
          x, a_values, a_meta_e, output, M, K, N, stream);
    } else {
      status = sparse24_cutlass_inline_transpose_gemm_run<
          DeviceSparseGemmInlineVectorTransposeM256N32K64S3Sw4>(
          x, a_values, a_meta_e, output, M, K, N, stream);
    }
  } else if (vector_store && config != nullptr &&
             std::strcmp(config, "256x64x64_s3_sw4") == 0) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_inline_transpose_gemm_run<
          DeviceSparseGemmInlineVectorTransposeF16M256N64K64S3Sw4>(
          x, a_values, a_meta_e, output, M, K, N, stream);
    } else {
      status = sparse24_cutlass_inline_transpose_gemm_run<
          DeviceSparseGemmInlineVectorTransposeM256N64K64S3Sw4>(
          x, a_values, a_meta_e, output, M, K, N, stream);
    }
  } else if (vector_store &&
             (config == nullptr ||
              std::strcmp(config, "256x64x64_s3") == 0)) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_inline_transpose_gemm_run<
          DeviceSparseGemmInlineVectorTransposeF16M256N64K64S3>(
          x, a_values, a_meta_e, output, M, K, N, stream);
    } else {
      status = sparse24_cutlass_inline_transpose_gemm_run<
          DeviceSparseGemmInlineVectorTransposeM256N64K64S3>(
          x, a_values, a_meta_e, output, M, K, N, stream);
    }
  } else if (!vector_store && config != nullptr &&
             std::strcmp(config, "256x64x64_s3_sw4") == 0) {
    status = sparse24_cutlass_inline_transpose_gemm_run<
        DeviceSparseGemmInlineTransposeM256N64K64S3Sw4>(
        x, a_values, a_meta_e, output, M, K, N, stream);
  } else if (!vector_store &&
             (config == nullptr ||
              std::strcmp(config, "256x64x64_s3") == 0)) {
    status = sparse24_cutlass_inline_transpose_gemm_run<
        DeviceSparseGemmInlineTransposeM256N64K64S3>(
        x, a_values, a_meta_e, output, M, K, N, stream);
  } else {
    return -5;
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_inline_routed_transpose_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *output_ptr, void *dense_base_ptr,
    const void *dense_slot_by_row_ptr, int M, int dense_rows, int K, int N,
    void *stream_ptr) {
  if (M <= 0 || dense_rows <= 0 || dense_rows > M || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 32) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *output = reinterpret_cast<Element *>(output_ptr);
  Element *dense_base = reinterpret_cast<Element *>(dense_base_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const char *config =
      std::getenv("SPECLINK_SPARSE24_ROUTED_TRANSPOSE_CONFIG");
  if (config == nullptr || std::strcmp(config, "auto") == 0) {
    config = sparse24_cutlass_select_device_config(nullptr, M, K, N);
    bool supported =
        std::strcmp(config, "64x64x64_s6") == 0 ||
        std::strcmp(config, "128x32x64_s4_sw4") == 0 ||
        std::strcmp(config, "128x64x64_s5") == 0 ||
        std::strcmp(config, "256x64x64_s3") == 0 ||
        std::strcmp(config, "256x64x64_s3_sw4") == 0;
    if (!supported) {
      config = M <= 64 ? "64x64x64_s6"
                       : (M <= 160 ? "128x32x64_s4_sw4"
                                   : "256x64x64_s3_sw4");
    }
  }
  int status = -5;
  if (std::strcmp(config, "64x64x64_s6") == 0) {
    status = sparse24_cutlass_inline_routed_transpose_gemm_run<
        DeviceSparseGemmInlineRoutedTransposeM64N64K64S6>(
        x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
        dense_rows, K, N, stream);
  } else if (std::strcmp(config, "128x32x64_s4_sw4") == 0) {
    status = sparse24_cutlass_inline_routed_transpose_gemm_run<
        DeviceSparseGemmInlineRoutedTransposeM128N32K64S4Sw4>(
        x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
        dense_rows, K, N, stream);
  } else if (std::strcmp(config, "128x64x64_s5") == 0) {
    status = sparse24_cutlass_inline_routed_transpose_gemm_run<
        DeviceSparseGemmInlineRoutedTransposeM128N64K64S5>(
        x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
        dense_rows, K, N, stream);
  } else if (std::strcmp(config, "256x64x64_s3") == 0) {
    status = sparse24_cutlass_inline_routed_transpose_gemm_run<
        DeviceSparseGemmInlineRoutedTransposeM256N64K64S3>(
        x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
        dense_rows, K, N, stream);
  } else if (std::strcmp(config, "256x64x64_s3_sw4") == 0) {
    status = sparse24_cutlass_inline_routed_transpose_gemm_run<
        DeviceSparseGemmInlineRoutedTransposeM256N64K64S3Sw4>(
        x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
        dense_rows, K, N, stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_routed_residual_epilogue_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    const void *routed_residual_ptr, const void *dense_slot_by_row_ptr,
    void *output_ptr, int M, int dense_rows, int K, int N, int config_id,
    void *stream_ptr) {
  if (x_ptr == nullptr || a_values_ptr == nullptr || a_meta_e_ptr == nullptr ||
      routed_residual_ptr == nullptr || dense_slot_by_row_ptr == nullptr ||
      output_ptr == nullptr || M <= 0 || dense_rows <= 0 || dense_rows > M ||
      K <= 0 || N <= 0 || config_id < 0 || config_id > 4) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 32) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values =
      reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  const Element *routed_residual =
      reinterpret_cast<const Element *>(routed_residual_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  int status = -5;
  if (config_id == 0) {
    status = sparse24_cutlass_routed_residual_epilogue_gemm_run<
        DeviceSparseGemmInlineRoutedResidualTransposeM128N32K64S4Sw4>(
        x, a_values, a_meta_e, routed_residual, dense_slot_by_row, output, M,
        dense_rows, K, N, stream);
  } else if (config_id == 1) {
    status = sparse24_cutlass_routed_residual_epilogue_gemm_run<
        DeviceSparseGemmInlineRoutedResidualTransposeM128N64K64S5>(
        x, a_values, a_meta_e, routed_residual, dense_slot_by_row, output, M,
        dense_rows, K, N, stream);
  } else if (config_id == 2) {
    status = sparse24_cutlass_routed_residual_epilogue_gemm_run<
        DeviceSparseGemmInlineRoutedResidualTransposeM256N32K64S3Sw4>(
        x, a_values, a_meta_e, routed_residual, dense_slot_by_row, output, M,
        dense_rows, K, N, stream);
  } else if (config_id == 3) {
    status = sparse24_cutlass_routed_residual_epilogue_gemm_run<
        DeviceSparseGemmInlineRoutedResidualTransposeM256N64K64S3>(
        x, a_values, a_meta_e, routed_residual, dense_slot_by_row, output, M,
        dense_rows, K, N, stream);
  } else if (config_id == 4) {
    status = sparse24_cutlass_routed_residual_epilogue_gemm_run<
        DeviceSparseGemmInlineRoutedResidualTransposeM256N64K64S3Sw4>(
        x, a_values, a_meta_e, routed_residual, dense_slot_by_row, output, M,
        dense_rows, K, N, stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_inline_indexed_transpose_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *output_ptr, const void *row_indices_ptr, int M, int logical_rows,
    int output_rows, int K, int N, int input_transposed, int ldb,
    void *stream_ptr) {
  if (M <= 0 || logical_rows <= 0 || logical_rows > M || output_rows <= 0 ||
      K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 32) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  if (input_transposed != 0 && ldb < M) {
    return -6;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *output = reinterpret_cast<Element *>(output_ptr);
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const char *config =
      std::getenv("SPECLINK_SPARSE24_INLINE_EPILOGUE_CONFIG");
  bool auto_config = config == nullptr || std::strcmp(config, "auto") == 0;
  if (auto_config) {
    if (input_transposed != 0) {
      config = sparse24_cutlass_select_b_row_device_config(
          nullptr, M, K, N);
      bool supported =
          std::strcmp(config, "64x64x64_s6") == 0 ||
          std::strcmp(config, "64x64x64_s7") == 0 ||
          std::strcmp(config, "128x32x64_s4_sw4") == 0 ||
          std::strcmp(config, "128x64x64_s3") == 0 ||
          std::strcmp(config, "128x64x64_s4") == 0 ||
          std::strcmp(config, "128x64x64_s5") == 0 ||
          std::strcmp(config, "128x128x64_s3") == 0 ||
          std::strcmp(config, "256x64x64_s3") == 0;
      if (!supported) {
        config = M <= 64 ? "64x64x64_s6"
                         : (M <= 160 ? "128x32x64_s4_sw4"
                                     : (M < 384 ? "128x64x64_s4"
                                                : "256x64x64_s3"));
      }
    } else {
      config = sparse24_cutlass_select_device_config(nullptr, M, K, N);
      bool supported =
          std::strcmp(config, "64x64x64_s6") == 0 ||
          std::strcmp(config, "128x32x64_s4_sw4") == 0 ||
          std::strcmp(config, "128x64x64_s5") == 0 ||
          std::strcmp(config, "256x64x64_s3") == 0 ||
          std::strcmp(config, "256x64x64_s3_sw4") == 0;
      if (!supported) {
        config = M <= 64 ? "64x64x64_s6"
                         : (M <= 128 ? "128x64x64_s5"
                                     : "256x64x64_s3");
      }
    }
  }
  int status = -5;
  if (input_transposed != 0 &&
      std::strcmp(config, "64x64x64_s6") == 0) {
    status = sparse24_cutlass_inline_indexed_transpose_gemm_run<
        DeviceSparseGemmInlineIndexedTransposeBRowM64N64K64S6,
        cutlass::layout::RowMajor>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, ldb, stream);
  } else if (input_transposed != 0 &&
             std::strcmp(config, "64x64x64_s7") == 0) {
    status = sparse24_cutlass_inline_indexed_transpose_gemm_run<
        DeviceSparseGemmInlineIndexedTransposeBRowM64N64K64S7,
        cutlass::layout::RowMajor>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, ldb, stream);
  } else if (input_transposed != 0 &&
             std::strcmp(config, "128x32x64_s4_sw4") == 0) {
    status = sparse24_cutlass_inline_indexed_transpose_gemm_run<
        DeviceSparseGemmInlineIndexedTransposeBRowM128N32K64S4Sw4,
        cutlass::layout::RowMajor>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, ldb, stream);
  } else if (input_transposed != 0 &&
             std::strcmp(config, "128x64x64_s4") == 0) {
    status = sparse24_cutlass_inline_indexed_transpose_gemm_run<
        DeviceSparseGemmInlineIndexedTransposeBRowM128N64K64S4,
        cutlass::layout::RowMajor>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, ldb, stream);
  } else if (input_transposed != 0 &&
             std::strcmp(config, "128x64x64_s3") == 0) {
    status = sparse24_cutlass_inline_indexed_transpose_gemm_run<
        DeviceSparseGemmInlineIndexedTransposeBRowM128N64K64S3,
        cutlass::layout::RowMajor>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, ldb, stream);
  } else if (input_transposed != 0 &&
             std::strcmp(config, "128x64x64_s5") == 0) {
    status = sparse24_cutlass_inline_indexed_transpose_gemm_run<
        DeviceSparseGemmInlineIndexedTransposeBRowM128N64K64S5,
        cutlass::layout::RowMajor>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, ldb, stream);
  } else if (input_transposed != 0 &&
             std::strcmp(config, "256x64x64_s3") == 0) {
    status = sparse24_cutlass_inline_indexed_transpose_gemm_run<
        DeviceSparseGemmInlineIndexedTransposeBRowM256N64K64S3,
        cutlass::layout::RowMajor>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, ldb, stream);
  } else if (input_transposed != 0 &&
             std::strcmp(config, "128x128x64_s3") == 0) {
    status = sparse24_cutlass_inline_indexed_transpose_gemm_run<
        DeviceSparseGemmInlineIndexedTransposeBRowM128N128K64S3,
        cutlass::layout::RowMajor>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, ldb, stream);
  } else if (input_transposed == 0 &&
             std::strcmp(config, "64x64x64_s6") == 0) {
    status = sparse24_cutlass_inline_indexed_transpose_gemm_run<
        DeviceSparseGemmInlineIndexedTransposeM64N64K64S6,
        cutlass::layout::ColumnMajor>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, K, stream);
  } else if (input_transposed == 0 &&
             std::strcmp(config, "128x32x64_s4_sw4") == 0) {
    status = sparse24_cutlass_inline_indexed_transpose_gemm_run<
        DeviceSparseGemmInlineIndexedTransposeM128N32K64S4Sw4,
        cutlass::layout::ColumnMajor>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, K, stream);
  } else if (input_transposed == 0 &&
             std::strcmp(config, "128x64x64_s5") == 0) {
    status = sparse24_cutlass_inline_indexed_transpose_gemm_run<
        DeviceSparseGemmInlineIndexedTransposeM128N64K64S5,
        cutlass::layout::ColumnMajor>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, K, stream);
  } else if (input_transposed == 0 &&
             std::strcmp(config, "256x64x64_s3_sw4") == 0) {
    status = sparse24_cutlass_inline_indexed_transpose_gemm_run<
        DeviceSparseGemmInlineIndexedTransposeM256N64K64S3Sw4,
        cutlass::layout::ColumnMajor>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, K, stream);
  } else if (input_transposed == 0 &&
             std::strcmp(config, "256x64x64_s3") == 0) {
    status = sparse24_cutlass_inline_indexed_transpose_gemm_run<
        DeviceSparseGemmInlineIndexedTransposeM256N64K64S3,
        cutlass::layout::ColumnMajor>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, K, stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_routed_exact_linear_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_e_ptr, const void *residual_values_ptr,
    const void *residual_meta_e_ptr, const void *dense_rows_ptr,
    const void *sparse_rows_ptr, void *output_ptr, int output_rows,
    int dense_count, int sparse_count, int K, int N, int config_id,
    void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_e_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_e_ptr == nullptr || dense_rows_ptr == nullptr ||
      sparse_rows_ptr == nullptr || output_ptr == nullptr || output_rows <= 0 ||
      dense_count <= 0 || sparse_count <= 0 ||
      dense_count + sparse_count != output_rows || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 128) != 0) {
    return -3;
  }

  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_e_ptr));
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_e_ptr));
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *sparse_rows = reinterpret_cast<const int *>(sparse_rows_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = output_rows <= 160 ? 2 : 3;
  }
  int status = -5;
  if (config_id == 1) {
    status = sparse24_cutlass_routed_exact_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM128N32K64S4Sw4,
        DeviceSparseGemmInlineIndexedTransposeM128N32K64S4Sw4>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, sparse_rows, sparse_count, output, output_rows, K, N,
        stream);
  } else if (config_id == 2) {
    status = sparse24_cutlass_routed_exact_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM128N64K64S5,
        DeviceSparseGemmInlineIndexedTransposeM128N64K64S5>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, sparse_rows, sparse_count, output, output_rows, K, N,
        stream);
  } else if (config_id == 3) {
    status = sparse24_cutlass_routed_exact_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM256N64K64S3,
        DeviceSparseGemmInlineIndexedTransposeM256N64K64S3>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, sparse_rows, sparse_count, output, output_rows, K, N,
        stream);
  } else if (config_id == 4) {
    status = sparse24_cutlass_routed_exact_linear_run<
        DeviceSparseGemmInlineIndexedTransposeF16M256N32K64S3Sw4,
        DeviceSparseGemmInlineIndexedTransposeF16M256N32K64S3Sw4>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, sparse_rows, sparse_count, output, output_rows, K, N,
        stream);
  } else if (config_id == 5) {
    status = sparse24_cutlass_routed_exact_linear_run<
        DeviceSparseGemmInlineIndexedTransposeF16M256N64K64S3,
        DeviceSparseGemmInlineIndexedTransposeF16M256N32K64S3W32x32>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, sparse_rows, sparse_count, output, output_rows, K, N,
        stream);
  } else if (config_id == 6) {
    status = sparse24_cutlass_routed_exact_linear_run<
        DeviceSparseGemmInlineIndexedTransposeF16M256N128K64S2,
        DeviceSparseGemmInlineIndexedTransposeF16M256N32K64S3W32x32>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, sparse_rows, sparse_count, output, output_rows, K, N,
        stream);
  } else if (config_id == 7) {
    status = sparse24_cutlass_routed_exact_linear_run<
        DeviceSparseGemmInlineIndexedTransposeF16M256N64K64S3,
        DeviceSparseGemmInlineIndexedTransposeF16M256N64K64S3>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, sparse_rows, sparse_count, output, output_rows, K, N,
        stream);
  } else if (config_id == 8) {
    status = sparse24_cutlass_routed_exact_linear_run<
        DeviceSparseGemmInlineIndexedTransposeF16M256N128K64S2,
        DeviceSparseGemmInlineIndexedTransposeF16M256N64K64S3>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, sparse_rows, sparse_count, output, output_rows, K, N,
        stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_heterogeneous_linear_f16_stream(
    const void *x_ptr, const void *sparse_values_ptr,
    const void *sparse_meta_e_ptr, const void *dense_weight_ptr,
    const void *dense_rows_ptr, const void *sparse_rows_ptr,
    void *output_ptr, int output_rows, int dense_count, int sparse_count,
    int K, int N, int config_id, void *stream_ptr) {
  if (x_ptr == nullptr || sparse_values_ptr == nullptr ||
      sparse_meta_e_ptr == nullptr || dense_weight_ptr == nullptr ||
      dense_rows_ptr == nullptr || sparse_rows_ptr == nullptr ||
      output_ptr == nullptr || output_rows <= 0 || dense_count <= 0 ||
      sparse_count <= 0 || dense_count + sparse_count != output_rows ||
      K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 128) != 0) {
    return -3;
  }

  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *sparse_values =
      reinterpret_cast<const Element *>(sparse_values_ptr);
  DeviceElementE *sparse_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(sparse_meta_e_ptr));
  const Element *dense_weight =
      reinterpret_cast<const Element *>(dense_weight_ptr);
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *sparse_rows = reinterpret_cast<const int *>(sparse_rows_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = output_rows <= 160 ? 1 : 3;
  }
  int status = -5;
  if (config_id == 1) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM128N32K64S4Sw4,
        DeviceDenseGemmInlineIndexedTransposeM128N32K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, false, 2,
        stream);
  } else if (config_id == 2) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM128N64K64S5,
        DeviceDenseGemmInlineIndexedTransposeM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, false, 2,
        stream);
  } else if (config_id == 3) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM256N64K64S3,
        DeviceDenseGemmInlineIndexedTransposeM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, false, 2,
        stream);
  } else if (config_id == 4) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM128N32K64S4Sw4,
        DeviceDenseGemmInlineIndexedTransposeM128N32K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, true, 2,
        stream);
  } else if (config_id == 5) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM128N64K64S5,
        DeviceDenseGemmInlineIndexedTransposeM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, true, 2,
        stream);
  } else if (config_id == 6) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM256N64K64S3,
        DeviceDenseGemmInlineIndexedTransposeM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, true, 2,
        stream);
  } else if (config_id == 7) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM256N32K64S3Sw4,
        DeviceDenseGemmInlineIndexedTransposeM128N32K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, false, 1,
        stream);
  } else if (config_id == 8) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM256N32K64S3Sw4,
        DeviceDenseGemmInlineIndexedTransposeM128N32K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, true, 1,
        stream);
  } else if (config_id == 9) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM256N64K64S2Sw4,
        DeviceDenseGemmInlineIndexedTransposeM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, false, 1,
        stream);
  } else if (config_id == 10) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM256N64K64S2Sw4,
        DeviceDenseGemmInlineIndexedTransposeM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, true, 1,
        stream);
  } else if (config_id == 11) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeF16M256N32K64S3Sw4,
        DeviceDenseGemmInlineIndexedTransposeM128N32K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, false, 1,
        stream);
  } else if (config_id == 12) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM256N64K64S3,
        DeviceDenseGemmInlineIndexedTransposeM128N32K64S3W16x32>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, false, 1,
        stream);
  } else if (config_id == 13) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeF16M256N64K64S3,
        DeviceDenseGemmInlineIndexedTransposeM128N32K64S3W16x32>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, false, 1,
        stream);
  } else if (config_id == 16) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeF16M256N64K64S3,
        DeviceDenseGemmInlineIndexedTransposeM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, false, 1,
        stream);
  } else if (config_id == 17) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeM256N128K64S2,
        DeviceDenseGemmInlineIndexedTransposeM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, false, 1,
        stream);
  } else if (config_id == 18) {
    status = sparse24_cutlass_heterogeneous_linear_run<
        DeviceSparseGemmInlineIndexedTransposeF16M256N128K64S2,
        DeviceDenseGemmInlineIndexedTransposeM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows, dense_count,
        sparse_rows, sparse_count, output, output_rows, K, N, false, 1,
        stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_heterogeneous_swiglu_f16_stream(
    const void *x_ptr, const void *sparse_values_ptr,
    const void *sparse_meta_e_ptr, const void *dense_weight_ptr,
    const void *dense_weight_rows_ptr, const void *dense_rows_ptr,
    const void *sparse_rows_ptr, void *output_ptr, int output_rows,
    int dense_count, int sparse_count, int K, int N, int config_id,
    void *stream_ptr) {
  if (x_ptr == nullptr || sparse_values_ptr == nullptr ||
      sparse_meta_e_ptr == nullptr || dense_weight_ptr == nullptr ||
      dense_weight_rows_ptr == nullptr || dense_rows_ptr == nullptr ||
      sparse_rows_ptr == nullptr || output_ptr == nullptr ||
      output_rows <= 0 || dense_count <= 0 || sparse_count <= 0 ||
      dense_count + sparse_count != output_rows || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 256) != 0) {
    return -3;
  }

  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *sparse_values =
      reinterpret_cast<const Element *>(sparse_values_ptr);
  DeviceElementE *sparse_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(sparse_meta_e_ptr));
  const Element *dense_weight =
      reinterpret_cast<const Element *>(dense_weight_ptr);
  const int *dense_weight_rows =
      reinterpret_cast<const int *>(dense_weight_rows_ptr);
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *sparse_rows = reinterpret_cast<const int *>(sparse_rows_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = 1;
  }
  int status = -5;
  if (config_id == 1) {
    status = sparse24_cutlass_heterogeneous_swiglu_run<
        DeviceSparseGemmIndexedSwiGLUF16M256N32K64S3Sw4,
        DeviceDenseGemmInlineIndexedSwiGLUM128N32K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_weight_rows,
        dense_rows, dense_count, sparse_rows, sparse_count, output,
        output_rows, K, N, stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_full_sparse_dense_override_swiglu_f16_stream(
    const void *x_ptr, const void *sparse_values_ptr,
    const void *sparse_meta_e_ptr, const void *dense_weight_ptr,
    const void *dense_weight_rows_ptr, const void *dense_rows_ptr,
    const void *dense_slot_by_row_ptr, void *output_ptr, int output_rows,
    int dense_count, int K, int N, int config_id, void *stream_ptr) {
  if (x_ptr == nullptr || sparse_values_ptr == nullptr ||
      sparse_meta_e_ptr == nullptr || dense_weight_ptr == nullptr ||
      dense_weight_rows_ptr == nullptr || dense_rows_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || output_ptr == nullptr ||
      output_rows <= 0 || dense_count <= 0 || dense_count > output_rows ||
      K <= 0 || N <= 0 || config_id < 0 || config_id > 5) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 256) != 0) {
    return -2;
  }

  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *sparse_values =
      reinterpret_cast<const Element *>(sparse_values_ptr);
  DeviceElementE *sparse_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(sparse_meta_e_ptr));
  const Element *dense_weight =
      reinterpret_cast<const Element *>(dense_weight_ptr);
  const int *dense_weight_rows =
      reinterpret_cast<const int *>(dense_weight_rows_ptr);
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = 2;
  }
  int status = -5;
  if (config_id == 1) {
    status = sparse24_cutlass_full_sparse_dense_override_swiglu_run<
        DeviceSparseGemmInlineRoutedSwiGLUF16M256N32K64S3Sw4,
        DeviceDenseGemmInlineIndexedSwiGLUM128N32K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_weight_rows,
        dense_rows, dense_slot_by_row, dense_count, output, output_rows, K, N,
        2, stream);
  } else {
    int dense_tile_weight =
        config_id == 2 ? 2 : (config_id == 3 ? 1 : config_id - 1);
    status = sparse24_cutlass_full_sparse_dense_override_swiglu_run<
        DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3Sw4,
        DeviceDenseGemmInlineIndexedSwiGLUM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_weight_rows,
        dense_rows, dense_slot_by_row, dense_count, output, output_rows, K, N,
        dense_tile_weight, stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_full_sparse_dense_override_linear_f16_stream(
    const void *x_ptr, const void *sparse_values_ptr,
    const void *sparse_meta_e_ptr, const void *dense_weight_ptr,
    const void *dense_rows_ptr, const void *dense_slot_by_row_ptr,
    void *output_ptr, int output_rows, int dense_count, int K, int N,
    int config_id, void *stream_ptr) {
  if (x_ptr == nullptr || sparse_values_ptr == nullptr ||
      sparse_meta_e_ptr == nullptr || dense_weight_ptr == nullptr ||
      dense_rows_ptr == nullptr || dense_slot_by_row_ptr == nullptr ||
      output_ptr == nullptr || output_rows <= 0 || dense_count <= 0 ||
      dense_count > output_rows || K <= 0 || N <= 0 || config_id < 0 ||
      config_id > 2) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 256) != 0) {
    return -2;
  }

  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *sparse_values =
      reinterpret_cast<const Element *>(sparse_values_ptr);
  DeviceElementE *sparse_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(sparse_meta_e_ptr));
  const Element *dense_weight =
      reinterpret_cast<const Element *>(dense_weight_ptr);
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = 2;
  }
  int status = -5;
  if (config_id == 1) {
    status = sparse24_cutlass_full_sparse_dense_override_linear_run<
        DeviceSparseGemmInlineRoutedTransposeF16M256N32K64S3Sw4,
        DeviceDenseGemmInlineIndexedTransposeM128N32K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows,
        dense_slot_by_row, dense_count, output, output_rows, K, N, 2,
        stream);
  } else if (config_id == 2) {
    status = sparse24_cutlass_full_sparse_dense_override_linear_run<
        DeviceSparseGemmInlineRoutedTransposeF16M256N64K64S3Sw4,
        DeviceDenseGemmInlineIndexedTransposeM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, dense_rows,
        dense_slot_by_row, dense_count, output, output_rows, K, N, 2,
        stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_heterogeneous_component_f16_stream(
    const void *x_ptr, const void *sparse_values_ptr,
    const void *sparse_meta_e_ptr, const void *dense_weight_ptr,
    const void *route_rows_ptr, void *output_ptr, int output_rows,
    int route_count, int K, int N, int config_id, int dense_component,
    void *stream_ptr) {
  if (x_ptr == nullptr || route_rows_ptr == nullptr || output_ptr == nullptr ||
      output_rows <= 0 || route_count <= 0 || route_count > output_rows ||
      K <= 0 || N <= 0 || (dense_component != 0 && dense_component != 1)) {
    return -1;
  }
  if (dense_component != 0 && dense_weight_ptr == nullptr) {
    return -1;
  }
  if (dense_component == 0 &&
      (sparse_values_ptr == nullptr || sparse_meta_e_ptr == nullptr)) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 128) != 0) {
    return -3;
  }

  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *sparse_values =
      reinterpret_cast<const Element *>(sparse_values_ptr);
  DeviceElementE *sparse_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(sparse_meta_e_ptr));
  const Element *dense_weight =
      reinterpret_cast<const Element *>(dense_weight_ptr);
  const int *route_rows = reinterpret_cast<const int *>(route_rows_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = output_rows <= 160 ? 1 : 3;
  }
  bool use_dense = dense_component != 0;
  int status = -5;
  if (config_id == 1) {
    status = sparse24_cutlass_heterogeneous_component_run<
        DeviceSparseGemmInlineIndexedTransposeM128N32K64S4Sw4,
        DeviceDenseGemmInlineIndexedTransposeM128N32K64S3>(
        x, sparse_values, sparse_meta, dense_weight, route_rows, route_count,
        output, output_rows, K, N, use_dense, stream);
  } else if (config_id == 2) {
    status = sparse24_cutlass_heterogeneous_component_run<
        DeviceSparseGemmInlineIndexedTransposeM128N64K64S5,
        DeviceDenseGemmInlineIndexedTransposeM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, route_rows, route_count,
        output, output_rows, K, N, use_dense, stream);
  } else if (config_id == 3) {
    status = sparse24_cutlass_heterogeneous_component_run<
        DeviceSparseGemmInlineIndexedTransposeM256N64K64S3,
        DeviceDenseGemmInlineIndexedTransposeM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, route_rows, route_count,
        output, output_rows, K, N, use_dense, stream);
  } else if (config_id == 7) {
    status = sparse24_cutlass_heterogeneous_component_run<
        DeviceSparseGemmInlineIndexedTransposeM256N32K64S3Sw4,
        DeviceDenseGemmInlineIndexedTransposeM128N32K64S3>(
        x, sparse_values, sparse_meta, dense_weight, route_rows, route_count,
        output, output_rows, K, N, use_dense, stream);
  } else if (config_id == 9) {
    status = sparse24_cutlass_heterogeneous_component_run<
        DeviceSparseGemmInlineIndexedTransposeM256N64K64S2Sw4,
        DeviceDenseGemmInlineIndexedTransposeM128N64K64S3>(
        x, sparse_values, sparse_meta, dense_weight, route_rows, route_count,
        output, output_rows, K, N, use_dense, stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_routed_exact_swiglu_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_e_ptr, const void *residual_values_ptr,
    const void *residual_meta_e_ptr, const void *dense_rows_ptr,
    const void *sparse_rows_ptr, void *output_ptr, int output_rows,
    int dense_count, int sparse_count, int K, int N, int config_id,
    void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_e_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_e_ptr == nullptr || dense_rows_ptr == nullptr ||
      sparse_rows_ptr == nullptr || output_ptr == nullptr || output_rows <= 0 ||
      dense_count <= 0 || sparse_count <= 0 ||
      dense_count + sparse_count != output_rows || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 256) != 0) {
    return -3;
  }

  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_e_ptr));
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_e_ptr));
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *sparse_rows = reinterpret_cast<const int *>(sparse_rows_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = output_rows <= 64 ? 1 : 2;
  }
  int status = -5;
  if (config_id == 1) {
    status = sparse24_cutlass_routed_exact_swiglu_run<
        DeviceSparseGemmIndexedSwiGLUM256N32K64S3Sw4,
        DeviceSparseGemmIndexedSwiGLUM256N32K64S3Sw4>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, sparse_rows, sparse_count, output, output_rows, K, N,
        stream);
  } else if (config_id == 2) {
    status = sparse24_cutlass_routed_exact_swiglu_run<
        DeviceSparseGemmIndexedSwiGLUM256N64K64S3Sw4,
        DeviceSparseGemmIndexedSwiGLUM256N64K64S3Sw4>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, sparse_rows, sparse_count, output, output_rows, K, N,
        stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_grouped_owner_linear_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_e_ptr, const void *residual_values_ptr,
    const void *residual_meta_e_ptr, const void *dense_rows_ptr,
    void *output_ptr, int output_rows, int dense_count, int K, int N,
    int group_tiles, int config_id, void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_e_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_e_ptr == nullptr || dense_rows_ptr == nullptr ||
      output_ptr == nullptr || output_rows <= 0 || dense_count <= 0 ||
      dense_count > output_rows || K <= 0 || N <= 0 || group_tiles <= 0 ||
      group_tiles > 4) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 128) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_e_ptr));
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_e_ptr));
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = 1;
  }
  int status = -5;
  if (config_id == 1) {
    status = sparse24_cutlass_grouped_owner_linear_run<
        DeviceSparseGemmInlineVectorTransposeM64N32K64S3,
        DeviceSparseGemmInlineIndexedAddTransposeM64N32K64S3>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, output, output_rows, K, N, group_tiles, stream);
  } else if (config_id == 2) {
    status = sparse24_cutlass_grouped_owner_linear_run<
        DeviceSparseGemmInlineVectorTransposeM128N32K64S4Sw4,
        DeviceSparseGemmInlineIndexedAddTransposeM128N32K64S4Sw4>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_count, output, output_rows, K, N, group_tiles, stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_grouped_owner_swiglu_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_e_ptr, const void *residual_values_ptr,
    const void *residual_meta_e_ptr, const void *dense_rows_ptr,
    const void *dense_slot_by_row_ptr, void *dense_base_ptr,
    void *output_ptr, int output_rows, int dense_count, int K, int N,
    int group_tiles, int config_id, void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_e_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_e_ptr == nullptr || dense_rows_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || dense_base_ptr == nullptr ||
      output_ptr == nullptr || output_rows <= 0 || dense_count <= 0 ||
      dense_count > output_rows || K <= 0 || N <= 0 || group_tiles <= 0 ||
      group_tiles > 4) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (N % 256) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_e_ptr));
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_e_ptr));
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *dense_base = reinterpret_cast<Element *>(dense_base_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = 1;
  }
  int status = -5;
  if (config_id == 1) {
    status = sparse24_cutlass_grouped_owner_swiglu_run<
        DeviceSparseGemmInlineRoutedSwiGLUM256N32K64S3Sw4,
        DeviceSparseGemmResidualCorrectionSwiGLUM256N32K64S3Sw4>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_slot_by_row, dense_count, dense_base, output, output_rows, K, N,
        group_tiles, stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_grouped_owner_qkv_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_e_ptr, const void *residual_values_ptr,
    const void *residual_meta_e_ptr, const void *dense_rows_ptr,
    const void *dense_slot_by_row_ptr, void *dense_base_ptr,
    void *output_ptr, const void *q_weight_ptr, const void *k_weight_ptr,
    const void *cos_sin_cache_ptr, const void *position_ids_ptr,
    int output_rows, int dense_count, int K, int N, int q_size, int kv_size,
    int rotary_dim, float epsilon, int is_neox, int normalize_qk,
    int group_tiles, int config_id, void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_e_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_e_ptr == nullptr || dense_rows_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || dense_base_ptr == nullptr ||
      output_ptr == nullptr || cos_sin_cache_ptr == nullptr ||
      position_ids_ptr == nullptr || output_rows <= 0 || dense_count <= 0 ||
      dense_count > output_rows || K <= 0 || N <= 0 || q_size <= 0 ||
      kv_size <= 0 || N != q_size + 2 * kv_size || rotary_dim != 128 ||
      is_neox != 1 || epsilon < 0.0f || group_tiles <= 0 ||
      group_tiles > 16 || config_id < 1 || config_id > 2) {
    return -1;
  }
  if (normalize_qk != 0 &&
      (q_weight_ptr == nullptr || k_weight_ptr == nullptr)) {
    return -2;
  }
  if ((K % 64) != 0 || (N % 256) != 0 || (q_size % 256) != 0 ||
      (kv_size % 256) != 0) {
    return -3;
  }

  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_e_ptr));
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_e_ptr));
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *dense_base = reinterpret_cast<Element *>(dense_base_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  const Element *q_weight = reinterpret_cast<const Element *>(q_weight_ptr);
  const Element *k_weight = reinterpret_cast<const Element *>(k_weight_ptr);
  const Element *cos_sin_cache =
      reinterpret_cast<const Element *>(cos_sin_cache_ptr);
  const int64_t *position_ids =
      reinterpret_cast<const int64_t *>(position_ids_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  int status = -5;
  if (config_id == 1) {
    status = sparse24_cutlass_grouped_owner_qkv_run<
        DeviceSparseGemmRoutedQKVPostOpM256N32K64S3W64x32,
        DeviceSparseGemmResidualQKVPostOpM256N32K64S3W64x32>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_slot_by_row, dense_count, dense_base, output, output_rows, K, N,
        q_weight, k_weight, cos_sin_cache, position_ids, q_size, kv_size,
        rotary_dim, epsilon, is_neox, normalize_qk, group_tiles, stream);
  } else if (config_id == 2) {
    status = sparse24_cutlass_grouped_owner_qkv_run<
        DeviceSparseGemmRoutedQKVPostOpM256N64K64S3,
        DeviceSparseGemmResidualQKVPostOpM256N64K64S3>(
        x, full_values, full_meta, residual_values, residual_meta, dense_rows,
        dense_slot_by_row, dense_count, dense_base, output, output_rows, K, N,
        q_weight, k_weight, cos_sin_cache, position_ids, q_size, kv_size,
        rotary_dim, epsilon, is_neox, normalize_qk, group_tiles, stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_inline_swiglu_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *output_ptr, int M, int K, int N, void *stream_ptr) {
  if (M <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % DeviceThreadblockShape256x64x64::kM) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const char *config =
      std::getenv("SPECLINK_SPARSE24_SWIGLU_EPILOGUE_CONFIG");
  if (config != nullptr && std::strcmp(config, "auto") == 0) {
    config = sparse24_cutlass_select_device_config(nullptr, M, K, N);
    if (std::strcmp(config, "256x64x64_s3") != 0 &&
        std::strcmp(config, "256x64x64_s3_sw4") != 0) {
      config = "256x64x64_s3";
    }
  }
  int status = -5;
  if (config != nullptr && std::strcmp(config, "256x32x64_s3") == 0) {
    status = sparse24_cutlass_inline_swiglu_gemm_run<
        DeviceSparseGemmInlineSwiGLUM256N32K64S3>(
        x, a_values, a_meta_e, output, M, K, N, stream);
  } else if (config != nullptr &&
             std::strcmp(config, "256x32x64_s3_sw4") == 0) {
    status = sparse24_cutlass_inline_swiglu_gemm_run<
        DeviceSparseGemmInlineSwiGLUM256N32K64S3Sw4>(
        x, a_values, a_meta_e, output, M, K, N, stream);
  } else if (config != nullptr &&
             std::strcmp(config, "256x32x64_s3_sw4_f16") == 0) {
    status = sparse24_cutlass_inline_swiglu_gemm_run<
        DeviceSparseGemmInlineSwiGLUF16M256N32K64S3Sw4>(
        x, a_values, a_meta_e, output, M, K, N, stream);
  } else if (config != nullptr &&
      std::strcmp(config, "256x64x64_s3_sw4") == 0) {
    status = sparse24_cutlass_inline_swiglu_gemm_run<
        DeviceSparseGemmInlineSwiGLUM256N64K64S3Sw4>(
        x, a_values, a_meta_e, output, M, K, N, stream);
  } else if (config != nullptr &&
             std::strcmp(config, "256x64x64_s3_sw4_f16") == 0) {
    status = sparse24_cutlass_inline_swiglu_gemm_run<
        DeviceSparseGemmInlineSwiGLUF16M256N64K64S3Sw4>(
        x, a_values, a_meta_e, output, M, K, N, stream);
  } else if (config == nullptr ||
             std::strcmp(config, "256x64x64_s3") == 0) {
    status = sparse24_cutlass_inline_swiglu_gemm_run<
        DeviceSparseGemmInlineSwiGLUM256N64K64S3>(
        x, a_values, a_meta_e, output, M, K, N, stream);
  } else {
    return -5;
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_inline_routed_swiglu_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *output_ptr, void *dense_base_ptr,
    const void *dense_slot_by_row_ptr, int M, int dense_rows, int K, int N,
    int output_transposed, void *stream_ptr) {
  if (M <= 0 || dense_rows <= 0 || dense_rows > M || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % DeviceThreadblockShape256x64x64::kM) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *output = reinterpret_cast<Element *>(output_ptr);
  Element *dense_base = reinterpret_cast<Element *>(dense_base_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const char *config =
      std::getenv("SPECLINK_SPARSE24_SWIGLU_EPILOGUE_CONFIG");
  if (config != nullptr && std::strcmp(config, "auto") == 0) {
    config = sparse24_cutlass_select_device_config(nullptr, M, K, N);
    if (std::strcmp(config, "256x64x64_s3") != 0 &&
        std::strcmp(config, "256x64x64_s3_sw4") != 0) {
      config = "256x64x64_s3";
    }
  }
  const char *accumulator =
      std::getenv("SPECLINK_SPARSE24_ACCUMULATOR");
  bool use_f16_accumulator =
      output_transposed == 0 &&
      sparse24_cutlass_use_f16_accumulator(accumulator, K, N);
  int status = -5;
  if (output_transposed == 1 && config != nullptr &&
      std::strcmp(config, "256x64x64_s3_sw4") == 0) {
    status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
        DeviceSparseGemmInlineRoutedSwiGLUTransposedM256N64K64S3Sw4>(
        x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
        dense_rows, K, N, stream);
  } else if (output_transposed == 1 &&
             (config == nullptr ||
              std::strcmp(config, "256x64x64_s3") == 0)) {
    status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
        DeviceSparseGemmInlineRoutedSwiGLUTransposedM256N64K64S3>(
        x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
        dense_rows, K, N, stream);
  } else if (output_transposed == 0 && config != nullptr &&
      std::strcmp(config, "256x64x64_s2_sw4") == 0) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
          DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S2Sw4>(
          x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
          dense_rows, K, N, stream);
    } else {
      status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
          DeviceSparseGemmInlineRoutedSwiGLUM256N64K64S2Sw4>(
          x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
          dense_rows, K, N, stream);
    }
  } else if (output_transposed == 0 && config != nullptr &&
      std::strcmp(config, "256x32x64_s3_sw4") == 0) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
          DeviceSparseGemmInlineRoutedSwiGLUF16M256N32K64S3Sw4>(
          x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
          dense_rows, K, N, stream);
    } else {
      status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
          DeviceSparseGemmInlineRoutedSwiGLUM256N32K64S3Sw4>(
          x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
          dense_rows, K, N, stream);
    }
  } else if (output_transposed == 0 && config != nullptr &&
      std::strcmp(config, "256x64x64_s3_sw4") == 0) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
          DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3Sw4>(
          x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
          dense_rows, K, N, stream);
    } else {
      status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
          DeviceSparseGemmInlineRoutedSwiGLUM256N64K64S3Sw4>(
          x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
          dense_rows, K, N, stream);
    }
  } else if (output_transposed == 0 &&
             (config == nullptr ||
              std::strcmp(config, "256x64x64_s3") == 0)) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
          DeviceSparseGemmInlineRoutedSwiGLUF16M256N64K64S3>(
          x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
          dense_rows, K, N, stream);
    } else {
      status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
          DeviceSparseGemmInlineRoutedSwiGLUM256N64K64S3>(
          x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
          dense_rows, K, N, stream);
    }
  } else {
    return -5;
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_inline_routed_approx_swiglu_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *output_ptr, void *dense_base_ptr,
    const void *dense_slot_by_row_ptr, int M, int dense_rows, int K, int N,
    int config_id, void *stream_ptr) {
  if (x_ptr == nullptr || a_values_ptr == nullptr ||
      a_meta_e_ptr == nullptr || output_ptr == nullptr ||
      dense_base_ptr == nullptr || dense_slot_by_row_ptr == nullptr || M <= 0 ||
      dense_rows <= 0 || dense_rows > M || K <= 0 || N <= 0 ||
      (M % 8) != 0 || (K % DeviceThreadblockShape::kK) != 0 ||
      (N % 256) != 0) {
    return -1;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *output = reinterpret_cast<Element *>(output_ptr);
  Element *dense_base = reinterpret_cast<Element *>(dense_base_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (config_id == 0) {
    config_id = M <= 64 ? 1 : 2;
  }
  const char *accumulator = std::getenv("SPECLINK_SPARSE24_ACCUMULATOR");
  bool use_f16_accumulator =
      sparse24_cutlass_use_f16_accumulator(accumulator, K, N);
  int status = -5;
  if (config_id == 1) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
          DeviceSparseGemmInlineRoutedApproxSwiGLUF16M256N32K64S3Sw4>(
          x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
          dense_rows, K, N, stream);
    } else {
      status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
          DeviceSparseGemmInlineRoutedApproxSwiGLUM256N32K64S3Sw4>(
          x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
          dense_rows, K, N, stream);
    }
  } else if (config_id == 2) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
          DeviceSparseGemmInlineRoutedApproxSwiGLUF16M256N64K64S3Sw4>(
          x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
          dense_rows, K, N, stream);
    } else {
      status = sparse24_cutlass_inline_routed_swiglu_gemm_run<
          DeviceSparseGemmInlineRoutedApproxSwiGLUM256N64K64S3Sw4>(
          x, a_values, a_meta_e, output, dense_base, dense_slot_by_row, M,
          dense_rows, K, N, stream);
    }
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_residual_correction_swiglu_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    const void *dense_base_ptr, const void *row_indices_ptr,
    void *output_ptr, void *compact_output_ptr, int M, int dense_rows,
    int output_rows, int K, int N, int config_id, void *stream_ptr) {
  if (x_ptr == nullptr || a_values_ptr == nullptr || a_meta_e_ptr == nullptr ||
      dense_base_ptr == nullptr || row_indices_ptr == nullptr ||
      output_ptr == nullptr || M <= 0 || dense_rows <= 0 || dense_rows > M ||
      output_rows <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 256) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  const Element *dense_base =
      reinterpret_cast<const Element *>(dense_base_ptr);
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  Element *compact_output = reinterpret_cast<Element *>(compact_output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = M <= 64 ? 1 : 2;
  }
  const char *accumulator =
      std::getenv("SPECLINK_SPARSE24_ACCUMULATOR");
  bool use_f16_accumulator =
      sparse24_cutlass_use_f16_accumulator(accumulator, K, N);
  int status = -5;
  if (config_id == 1) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_residual_correction_swiglu_gemm_run<
          DeviceSparseGemmResidualCorrectionSwiGLUF16M256N32K64S3Sw4>(
          x, a_values, a_meta_e, dense_base, row_indices, output,
          compact_output, M, dense_rows, output_rows, K, N, stream);
    } else {
      status = sparse24_cutlass_residual_correction_swiglu_gemm_run<
          DeviceSparseGemmResidualCorrectionSwiGLUM256N32K64S3Sw4>(
          x, a_values, a_meta_e, dense_base, row_indices, output,
          compact_output, M, dense_rows, output_rows, K, N, stream);
    }
  } else if (config_id == 2) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_residual_correction_swiglu_gemm_run<
          DeviceSparseGemmResidualCorrectionSwiGLUF16M256N64K64S3Sw4>(
          x, a_values, a_meta_e, dense_base, row_indices, output,
          compact_output, M, dense_rows, output_rows, K, N, stream);
    } else {
      status = sparse24_cutlass_residual_correction_swiglu_gemm_run<
          DeviceSparseGemmResidualCorrectionSwiGLUM256N64K64S3Sw4>(
          x, a_values, a_meta_e, dense_base, row_indices, output,
          compact_output, M, dense_rows, output_rows, K, N, stream);
    }
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_residual_delta_swiglu_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    const void *dense_base_ptr, void *dense_delta_ptr, int M,
    int dense_rows, int K, int N, int config_id, void *stream_ptr) {
  if (x_ptr == nullptr || a_values_ptr == nullptr || a_meta_e_ptr == nullptr ||
      dense_base_ptr == nullptr || dense_delta_ptr == nullptr || M <= 0 ||
      dense_rows <= 0 || dense_rows > M || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 256) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values =
      reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  const Element *dense_base =
      reinterpret_cast<const Element *>(dense_base_ptr);
  Element *dense_delta = reinterpret_cast<Element *>(dense_delta_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  if (config_id == 0) {
    config_id = M <= 64 ? 1 : 2;
  }
  const char *accumulator = std::getenv("SPECLINK_SPARSE24_ACCUMULATOR");
  bool use_f16_accumulator =
      sparse24_cutlass_use_f16_accumulator(accumulator, K, N);
  int status = -5;
  if (config_id == 1) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_residual_delta_swiglu_gemm_run<
          DeviceSparseGemmResidualDeltaSwiGLUF16M256N32K64S3Sw4>(
          x, a_values, a_meta_e, dense_base, dense_delta, M, dense_rows, K, N,
          stream);
    } else {
      status = sparse24_cutlass_residual_delta_swiglu_gemm_run<
          DeviceSparseGemmResidualDeltaSwiGLUM256N32K64S3Sw4>(
          x, a_values, a_meta_e, dense_base, dense_delta, M, dense_rows, K, N,
          stream);
    }
  } else if (config_id == 2) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_residual_delta_swiglu_gemm_run<
          DeviceSparseGemmResidualDeltaSwiGLUF16M256N64K64S3Sw4>(
          x, a_values, a_meta_e, dense_base, dense_delta, M, dense_rows, K, N,
          stream);
    } else {
      status = sparse24_cutlass_residual_delta_swiglu_gemm_run<
          DeviceSparseGemmResidualDeltaSwiGLUM256N64K64S3Sw4>(
          x, a_values, a_meta_e, dense_base, dense_delta, M, dense_rows, K, N,
          stream);
    }
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_indexed_swiglu_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *output_ptr, const void *row_indices_ptr, int M, int logical_rows,
    int output_rows, int K, int N, int config_id, void *stream_ptr) {
  if (x_ptr == nullptr || a_values_ptr == nullptr || a_meta_e_ptr == nullptr ||
      output_ptr == nullptr || row_indices_ptr == nullptr || M <= 0 ||
      logical_rows <= 0 || logical_rows > M || output_rows <= 0 || K <= 0 ||
      N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 256) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *output = reinterpret_cast<Element *>(output_ptr);
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (config_id == 0) {
    config_id = M <= 64 ? 1 : 2;
  }
  const char *accumulator =
      std::getenv("SPECLINK_SPARSE24_ACCUMULATOR");
  bool use_f16_accumulator =
      sparse24_cutlass_use_f16_accumulator(accumulator, K, N);
  int status = -5;
  if (config_id == 1) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_indexed_swiglu_gemm_run<
          DeviceSparseGemmIndexedSwiGLUF16M256N32K64S3Sw4>(
          x, a_values, a_meta_e, output, row_indices, M, logical_rows,
          output_rows, K, N, stream);
    } else {
      status = sparse24_cutlass_indexed_swiglu_gemm_run<
          DeviceSparseGemmIndexedSwiGLUM256N32K64S3Sw4>(
          x, a_values, a_meta_e, output, row_indices, M, logical_rows,
          output_rows, K, N, stream);
    }
  } else if (config_id == 2) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_indexed_swiglu_gemm_run<
          DeviceSparseGemmIndexedSwiGLUF16M256N64K64S3Sw4>(
          x, a_values, a_meta_e, output, row_indices, M, logical_rows,
          output_rows, K, N, stream);
    } else {
      status = sparse24_cutlass_indexed_swiglu_gemm_run<
          DeviceSparseGemmIndexedSwiGLUM256N64K64S3Sw4>(
          x, a_values, a_meta_e, output, row_indices, M, logical_rows,
          output_rows, K, N, stream);
    }
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_dual_swiglu_gemm_f16_stream(
    const void *x_ptr, const void *full_values_ptr,
    const void *full_meta_ptr, const void *residual_values_ptr,
    const void *residual_meta_ptr, void *output_ptr,
    const void *row_indices_ptr, int M, int logical_rows, int output_rows,
    int K, int N, int config_id, void *stream_ptr) {
  if (x_ptr == nullptr || full_values_ptr == nullptr ||
      full_meta_ptr == nullptr || residual_values_ptr == nullptr ||
      residual_meta_ptr == nullptr || output_ptr == nullptr ||
      row_indices_ptr == nullptr || M <= 0 || logical_rows <= 0 ||
      logical_rows > M || output_rows <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 256) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *full_values =
      reinterpret_cast<const Element *>(full_values_ptr);
  DeviceElementE *full_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(full_meta_ptr));
  const Element *residual_values =
      reinterpret_cast<const Element *>(residual_values_ptr);
  DeviceElementE *residual_meta = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(residual_meta_ptr));
  Element *output = reinterpret_cast<Element *>(output_ptr);
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (config_id == 0) {
    config_id = M <= 64 ? 1 : 2;
  }
  const char *accumulator =
      std::getenv("SPECLINK_SPARSE24_ACCUMULATOR");
  bool use_f16_accumulator =
      sparse24_cutlass_use_f16_accumulator(accumulator, K, N);
  int status = -5;
  if (config_id == 1) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_dual_swiglu_gemm_run<
          DeviceSparseGemmIndexedSwiGLUF16M256N32K64S3Sw4>(
          x, full_values, full_meta, residual_values, residual_meta, output,
          row_indices, M, logical_rows, output_rows, K, N, stream);
    } else {
      status = sparse24_cutlass_dual_swiglu_gemm_run<
          DeviceSparseGemmIndexedSwiGLUM256N32K64S3Sw4>(
          x, full_values, full_meta, residual_values, residual_meta, output,
          row_indices, M, logical_rows, output_rows, K, N, stream);
    }
  } else if (config_id == 2) {
    if (use_f16_accumulator) {
      status = sparse24_cutlass_dual_swiglu_gemm_run<
          DeviceSparseGemmIndexedSwiGLUF16M256N64K64S3Sw4>(
          x, full_values, full_meta, residual_values, residual_meta, output,
          row_indices, M, logical_rows, output_rows, K, N, stream);
    } else {
      status = sparse24_cutlass_dual_swiglu_gemm_run<
          DeviceSparseGemmIndexedSwiGLUM256N64K64S3Sw4>(
          x, full_values, full_meta, residual_values, residual_meta, output,
          row_indices, M, logical_rows, output_rows, K, N, stream);
    }
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_inline_pair_add_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *output_ptr, const void *row_indices_ptr, int M, int logical_rows,
    int output_rows, int K, int N, void *stream_ptr) {
  if (M <= 0 || logical_rows <= 0 || logical_rows > M || output_rows <= 0 ||
      K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 128) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *output = reinterpret_cast<Element *>(output_ptr);
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const char *config =
      std::getenv("SPECLINK_SPARSE24_PAIR_ADD_EPILOGUE_CONFIG");
  if (config == nullptr || std::strcmp(config, "auto") == 0) {
    config = M <= 32 ? "256x32x64_s3_sw4" : "256x64x64_s3";
  }
  int status = -5;
  if (std::strcmp(config, "256x32x64_s3_sw4") == 0) {
    status = sparse24_cutlass_inline_pair_add_gemm_run<
        DeviceSparseGemmInlinePairAddM256N32K64S3Sw4>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, stream);
  } else if (std::strcmp(config, "256x64x64_s3_sw4") == 0) {
    status = sparse24_cutlass_inline_pair_add_gemm_run<
        DeviceSparseGemmInlinePairAddM256N64K64S3Sw4>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, stream);
  } else if (std::strcmp(config, "256x64x64_s3") == 0) {
    status = sparse24_cutlass_inline_pair_add_gemm_run<
        DeviceSparseGemmInlinePairAddM256N64K64S3>(
        x, a_values, a_meta_e, output, row_indices, M, logical_rows,
        output_rows, K, N, stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_inline_swiglu_transposed_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *output_ptr, int M, int K, int N, void *stream_ptr) {
  if (M <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % DeviceThreadblockShape256x64x64::kM) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const char *config =
      std::getenv("SPECLINK_SPARSE24_SWIGLU_EPILOGUE_CONFIG");
  if (config != nullptr && std::strcmp(config, "auto") == 0) {
    config = sparse24_cutlass_select_device_config(nullptr, M, K, N);
    if (std::strcmp(config, "256x64x64_s3") != 0 &&
        std::strcmp(config, "256x64x64_s3_sw4") != 0) {
      config = "256x64x64_s3";
    }
  }
  int status = -5;
  if (config != nullptr &&
      std::strcmp(config, "256x64x64_s3_sw4") == 0) {
    status = sparse24_cutlass_inline_swiglu_gemm_run<
        DeviceSparseGemmInlineSwiGLUTransposedM256N64K64S3Sw4>(
        x, a_values, a_meta_e, output, M, K, N, stream);
  } else if (config == nullptr ||
             std::strcmp(config, "256x64x64_s3") == 0) {
    status = sparse24_cutlass_inline_swiglu_gemm_run<
        DeviceSparseGemmInlineSwiGLUTransposedM256N64K64S3>(
        x, a_values, a_meta_e, output, M, K, N, stream);
  } else {
    return -5;
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_inline_qkv_postop_gemm_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *output_ptr, const void *q_weight_ptr, const void *k_weight_ptr,
    const void *cos_sin_cache_ptr, const void *position_ids_ptr, int M, int K,
    int q_size, int kv_size, int head_dim, int rotary_dim, float epsilon,
    int is_neox, int normalize_qk, void *stream_ptr) {
  int N = q_size + 2 * kv_size;
  if (x_ptr == nullptr || a_values_ptr == nullptr || a_meta_e_ptr == nullptr ||
      output_ptr == nullptr || cos_sin_cache_ptr == nullptr ||
      position_ids_ptr == nullptr || M <= 0 || K <= 0 || q_size <= 0 ||
      kv_size <= 0 || epsilon < 0.0f) {
    return -1;
  }
  if (normalize_qk != 0 &&
      (q_weight_ptr == nullptr || k_weight_ptr == nullptr)) {
    return -2;
  }
  if (head_dim != 128 || q_size % head_dim != 0 ||
      kv_size % head_dim != 0 || N % 128 != 0 || rotary_dim <= 0 ||
      rotary_dim > head_dim || rotary_dim % 2 != 0) {
    return -3;
  }
  if ((K % DeviceThreadblockShape::kK) != 0 || (M % 8) != 0) {
    return -4;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *output = reinterpret_cast<Element *>(output_ptr);
  const Element *q_weight = reinterpret_cast<const Element *>(q_weight_ptr);
  const Element *k_weight = reinterpret_cast<const Element *>(k_weight_ptr);
  const Element *cos_sin_cache =
      reinterpret_cast<const Element *>(cos_sin_cache_ptr);
  const int64_t *position_ids =
      reinterpret_cast<const int64_t *>(position_ids_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const char *config =
      std::getenv("SPECLINK_SPARSE24_QKV_EPILOGUE_CONFIG");
  if (config != nullptr && std::strcmp(config, "auto") == 0) {
    config = sparse24_cutlass_select_device_config(nullptr, M, K, N);
  }
  int status = -5;
  if (config != nullptr &&
      std::strcmp(config, "128x32x64_s4") == 0) {
    status = sparse24_cutlass_inline_qkv_postop_gemm_run<
        DeviceSparseGemmInlineQKVPostOpM128N32K64S4>(
        x, a_values, a_meta_e, output, q_weight, k_weight, cos_sin_cache,
        position_ids, M, K, q_size, kv_size, rotary_dim, epsilon,
        is_neox != 0, normalize_qk != 0, stream);
  } else if (config != nullptr &&
             std::strcmp(config, "128x32x64_s4_sw2") == 0) {
    status = sparse24_cutlass_inline_qkv_postop_gemm_run<
        DeviceSparseGemmInlineQKVPostOpM128N32K64S4Sw2>(
        x, a_values, a_meta_e, output, q_weight, k_weight, cos_sin_cache,
        position_ids, M, K, q_size, kv_size, rotary_dim, epsilon,
        is_neox != 0, normalize_qk != 0, stream);
  } else if (config != nullptr &&
             std::strcmp(config, "128x32x64_s4_sw4") == 0) {
    status = sparse24_cutlass_inline_qkv_postop_gemm_run<
        DeviceSparseGemmInlineQKVPostOpM128N32K64S4Sw4>(
        x, a_values, a_meta_e, output, q_weight, k_weight, cos_sin_cache,
        position_ids, M, K, q_size, kv_size, rotary_dim, epsilon,
        is_neox != 0, normalize_qk != 0, stream);
  } else if (config != nullptr &&
      std::strcmp(config, "128x64x64_s5") == 0) {
    status = sparse24_cutlass_inline_qkv_postop_gemm_run<
        DeviceSparseGemmInlineQKVPostOpM128N64K64S5>(
        x, a_values, a_meta_e, output, q_weight, k_weight, cos_sin_cache,
        position_ids, M, K, q_size, kv_size, rotary_dim, epsilon,
        is_neox != 0, normalize_qk != 0, stream);
  } else if (config != nullptr &&
             std::strcmp(config, "256x32x64_s3_sw4") == 0) {
    status = sparse24_cutlass_inline_qkv_postop_gemm_run<
        DeviceSparseGemmInlineQKVPostOpM256N32K64S3Sw4>(
        x, a_values, a_meta_e, output, q_weight, k_weight, cos_sin_cache,
        position_ids, M, K, q_size, kv_size, rotary_dim, epsilon,
        is_neox != 0, normalize_qk != 0, stream);
  } else if (config != nullptr &&
             std::strcmp(config, "256x64x64_s3_sw4") == 0) {
    status = sparse24_cutlass_inline_qkv_postop_gemm_run<
        DeviceSparseGemmInlineQKVPostOpM256N64K64S3Sw4>(
        x, a_values, a_meta_e, output, q_weight, k_weight, cos_sin_cache,
        position_ids, M, K, q_size, kv_size, rotary_dim, epsilon,
        is_neox != 0, normalize_qk != 0, stream);
  } else if (config == nullptr ||
             std::strcmp(config, "256x64x64_s3") == 0) {
    status = sparse24_cutlass_inline_qkv_postop_gemm_run<
        DeviceSparseGemmInlineQKVPostOpM256N64K64S3>(
        x, a_values, a_meta_e, output, q_weight, k_weight, cos_sin_cache,
        position_ids, M, K, q_size, kv_size, rotary_dim, epsilon,
        is_neox != 0, normalize_qk != 0, stream);
  } else {
    return -5;
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_device_gemm_b_row_f16_stream(
    const void *x_ptr, const void *a_values_ptr, const void *a_meta_e_ptr,
    void *c_tmp_ptr, void *y_ptr, int M, int K, int N, int ldb,
    void *stream_ptr) {
  if (M <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 32) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  if (ldb < M) {
    return -6;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  Element *c_tmp = reinterpret_cast<Element *>(c_tmp_ptr);
  Element *y = reinterpret_cast<Element *>(y_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const char *config = sparse24_cutlass_select_b_row_device_config(
      std::getenv("SPECLINK_SPARSE24_DEVICE_CONFIG"), M, K, N);
  int status = sparse24_cutlass_device_dispatch_config_b_row(
      config, x, a_values, a_meta_e, c_tmp, M, K, N, ldb, stream);
  if (status != 0) {
    return status;
  }
  if (y != nullptr) {
    dim3 block(32, 8);
    dim3 grid((M + 31) / 32, (N + 31) / 32);
    sparse24_cutlass_device_transpose_tiled_kernel<<<grid, block, 0, stream>>>(
        c_tmp, y, M, N);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int dense_cutlass_device_gemm_f16_stream(
    const void *x_ptr, const void *w_ptr, void *y_ptr, int M, int K, int N,
    void *stream_ptr) {
  if (M <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % 64) != 0) {
    return -2;
  }
  if ((N % 8) != 0) {
    return -3;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *w = reinterpret_cast<const Element *>(w_ptr);
  Element *y = reinterpret_cast<Element *>(y_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const char *config = dense_cutlass_select_device_config(
      std::getenv("SPECLINK_DENSE_GEMM_CONFIG"), M, K, N);
  int status = dense_cutlass_device_dispatch_config(config, x, w, y, M, K, N,
                                                    stream);
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int dense_cutlass_device_gemm_f16_accum_f16_stream(
    const void *x_ptr, const void *w_ptr, void *y_ptr, int M, int K, int N,
    void *stream_ptr) {
  if (x_ptr == nullptr || w_ptr == nullptr || y_ptr == nullptr || M <= 0 ||
      K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % 64) != 0 || (N % 8) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *w = reinterpret_cast<const Element *>(w_ptr);
  Element *y = reinterpret_cast<Element *>(y_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const char *config = dense_cutlass_select_device_config(
      std::getenv("SPECLINK_DENSE_GEMM_CONFIG"), M, K, N);
  int status = dense_cutlass_device_f16_accum_dispatch_config(
      config, x, w, y, M, K, N, stream);
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int dense_cutlass_weight_t_gemm_f16_stream(
    const void *x_ptr, const void *weight_t_ptr, void *y_ptr, int M, int K,
    int N, void *stream_ptr) {
  if (x_ptr == nullptr || weight_t_ptr == nullptr || y_ptr == nullptr ||
      M <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % 64) != 0 || (N % 8) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *weight_t = reinterpret_cast<const Element *>(weight_t_ptr);
  Element *y = reinterpret_cast<Element *>(y_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const char *config = dense_cutlass_select_device_config(
      std::getenv("SPECLINK_DENSE_GEMM_CONFIG"), M, K, N);
  int status = dense_cutlass_device_b_col_dispatch_config(
      config, x, weight_t, y, M, K, N, stream);
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int dense_cutlass_weight_t_gemm_f16_accum_f16_stream(
    const void *x_ptr, const void *weight_t_ptr, void *y_ptr, int M, int K,
    int N, void *stream_ptr) {
  if (x_ptr == nullptr || weight_t_ptr == nullptr || y_ptr == nullptr ||
      M <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % 64) != 0 || (N % 8) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *weight_t = reinterpret_cast<const Element *>(weight_t_ptr);
  Element *y = reinterpret_cast<Element *>(y_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const char *config = dense_cutlass_select_device_config(
      std::getenv("SPECLINK_DENSE_GEMM_CONFIG"), M, K, N);
  int status = dense_cutlass_device_b_col_f16_accum_dispatch_config(
      config, x, weight_t, y, M, K, N, stream);
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int dense_cutlass_weight_t_gemm_add_f16_stream(
    const void *x_ptr, const void *weight_t_ptr, const void *residual_ptr,
    void *y_ptr, int M, int K, int N, void *stream_ptr) {
  if (x_ptr == nullptr || weight_t_ptr == nullptr || residual_ptr == nullptr ||
      y_ptr == nullptr || M <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % 64) != 0 || (N % 8) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *weight_t = reinterpret_cast<const Element *>(weight_t_ptr);
  const Element *residual = reinterpret_cast<const Element *>(residual_ptr);
  Element *y = reinterpret_cast<Element *>(y_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const char *config = dense_cutlass_select_device_config(
      std::getenv("SPECLINK_DENSE_GEMM_CONFIG"), M, K, N);
  int status = dense_cutlass_device_b_col_add_dispatch_config(
      config, x, weight_t, residual, y, M, K, N, stream);
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int dense_cutlass_weight_t_gemm_add_f16_accum_f16_stream(
    const void *x_ptr, const void *weight_t_ptr, const void *residual_ptr,
    void *y_ptr, int M, int K, int N, void *stream_ptr) {
  if (x_ptr == nullptr || weight_t_ptr == nullptr || residual_ptr == nullptr ||
      y_ptr == nullptr || M <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % 64) != 0 || (N % 8) != 0) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *weight_t = reinterpret_cast<const Element *>(weight_t_ptr);
  const Element *residual = reinterpret_cast<const Element *>(residual_ptr);
  Element *y = reinterpret_cast<Element *>(y_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const char *config = dense_cutlass_select_device_config(
      std::getenv("SPECLINK_DENSE_GEMM_CONFIG"), M, K, N);
  int status = dense_cutlass_device_b_col_add_f16_accum_dispatch_config(
      config, x, weight_t, residual, y, M, K, N, stream);
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int dense_cutlass_simt_weight_t_gemm_f16_stream(
    const void *weight_t_ptr, const void *x_ptr, void *output_ptr, int M,
    int K, int N, int config_id, void *stream_ptr) {
  if (weight_t_ptr == nullptr || x_ptr == nullptr || output_ptr == nullptr ||
      M <= 0 || K <= 0 || N <= 0) {
    return -1;
  }
  if ((K % 8) != 0 || (M % 8) != 0 || (N % 8) != 0) {
    return -2;
  }
  const Element *weight_t =
      reinterpret_cast<const Element *>(weight_t_ptr);
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int status = -5;
  if (config_id == 0) {
    status = dense_cutlass_device_gemm_b_col_run<
        DeviceDenseGemmSimtBColM64N64K8S2>(weight_t, x, output, N, K, M,
                                           stream);
  } else if (config_id == 1) {
    status = dense_cutlass_device_gemm_b_col_run<
        DeviceDenseGemmSimtBColM128N64K8S2>(weight_t, x, output, N, K, M,
                                            stream);
  }
  if (status != 0) {
    return status;
  }
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_silu_and_mul_transposed_f16_stream(
    const void *input_ptr, void *output_ptr, int M, int hidden_size,
    int leading_dim, void *stream_ptr) {
  if (input_ptr == nullptr || output_ptr == nullptr || M <= 0 ||
      hidden_size <= 0 || leading_dim < M) {
    return -1;
  }
  const Element *input = reinterpret_cast<const Element *>(input_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if ((leading_dim % 2) != 0) {
    return -2;
  }
  int total = ((M + 1) / 2) * hidden_size;
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_silu_and_mul_transposed_kernel<<<grid, block, 0, stream>>>(
      input, output, M, hidden_size, leading_dim);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_silu_and_mul_transposed_to_contiguous_f16_stream(
    const void *input_ptr, void *output_ptr, int M, int hidden_size,
    int leading_dim, void *stream_ptr) {
  if (input_ptr == nullptr || output_ptr == nullptr || M <= 0 ||
      hidden_size <= 0 || leading_dim < M) {
    return -1;
  }
  const Element *input = reinterpret_cast<const Element *>(input_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  dim3 block(32, 8);
  dim3 grid((M + 31) / 32, (hidden_size + 31) / 32);
  sparse24_cutlass_silu_and_mul_transposed_to_contiguous_kernel
      <<<grid, block, 0, stream>>>(input, output, M, hidden_size, leading_dim);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_routed_swiglu_correction_f16_stream(
    const void *dense_base_ptr, const void *dense_residual_ptr,
    const void *dense_rows_ptr, void *output_ptr, int dense_count,
    int output_rows, int hidden_size, void *stream_ptr) {
  if (dense_base_ptr == nullptr || dense_residual_ptr == nullptr ||
      dense_rows_ptr == nullptr || output_ptr == nullptr || dense_count <= 0 ||
      output_rows <= 0 || hidden_size <= 0 || (hidden_size % 2) != 0) {
    return -1;
  }
  const Element *dense_base =
      reinterpret_cast<const Element *>(dense_base_ptr);
  const Element *dense_residual =
      reinterpret_cast<const Element *>(dense_residual_ptr);
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total = dense_count * (hidden_size / 2);
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_routed_swiglu_correction_kernel
      <<<grid, block, 0, stream>>>(dense_base, dense_residual, dense_rows,
                                   output, dense_count, output_rows,
                                   hidden_size);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_routed_swiglu_correction_gather_f16_stream(
    const void *dense_base_ptr, const void *dense_residual_ptr,
    const void *dense_rows_ptr, void *output_ptr, void *dense_hidden_ptr,
    int dense_count, int dense_run, int output_rows, int hidden_size,
    void *stream_ptr) {
  if (dense_base_ptr == nullptr || dense_residual_ptr == nullptr ||
      dense_rows_ptr == nullptr || output_ptr == nullptr ||
      dense_hidden_ptr == nullptr || dense_count <= 0 ||
      dense_run < dense_count || output_rows <= 0 || hidden_size <= 0 ||
      (hidden_size % 2) != 0) {
    return -1;
  }
  const Element *dense_base =
      reinterpret_cast<const Element *>(dense_base_ptr);
  const Element *dense_residual =
      reinterpret_cast<const Element *>(dense_residual_ptr);
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  Element *dense_hidden = reinterpret_cast<Element *>(dense_hidden_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total = dense_run * (hidden_size / 2);
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_routed_swiglu_correction_gather_kernel
      <<<grid, block, 0, stream>>>(
          dense_base, dense_residual, dense_rows, output, dense_hidden,
          dense_count, dense_run, output_rows, hidden_size);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_routed_swiglu_delta_f16_stream(
    const void *dense_base_ptr, const void *dense_residual_ptr,
    void *dense_delta_ptr, int dense_count, int dense_run, int hidden_size,
    void *stream_ptr) {
  if (dense_base_ptr == nullptr || dense_residual_ptr == nullptr ||
      dense_delta_ptr == nullptr || dense_count <= 0 ||
      dense_run < dense_count || hidden_size <= 0 || (hidden_size % 2) != 0) {
    return -1;
  }
  const Element *dense_base =
      reinterpret_cast<const Element *>(dense_base_ptr);
  const Element *dense_residual =
      reinterpret_cast<const Element *>(dense_residual_ptr);
  Element *dense_delta = reinterpret_cast<Element *>(dense_delta_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total = dense_run * (hidden_size / 2);
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_routed_swiglu_delta_kernel<<<grid, block, 0, stream>>>(
      dense_base, dense_residual, dense_delta, dense_count, dense_run,
      hidden_size);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_routed_linear_correction_f16_stream(
    const void *dense_base_ptr, const void *dense_residual_ptr,
    const void *dense_rows_ptr, void *output_ptr, int dense_count,
    int output_rows, int output_columns, void *stream_ptr) {
  if (dense_base_ptr == nullptr || dense_residual_ptr == nullptr ||
      dense_rows_ptr == nullptr || output_ptr == nullptr || dense_count <= 0 ||
      output_rows <= 0 || output_columns <= 0 || (output_columns % 2) != 0) {
    return -1;
  }
  const Element *dense_base =
      reinterpret_cast<const Element *>(dense_base_ptr);
  const Element *dense_residual =
      reinterpret_cast<const Element *>(dense_residual_ptr);
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total = dense_count * (output_columns / 2);
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_routed_linear_correction_kernel
      <<<grid, block, 0, stream>>>(dense_base, dense_residual, dense_rows,
                                   output, dense_count, output_rows,
                                   output_columns);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_routed_swiglu_correction_transposed_f16_stream(
    const void *dense_base_ptr, const void *dense_residual_ptr,
    const void *dense_rows_ptr, void *output_ptr, int dense_count,
    int output_rows, int hidden_size, int base_ld, int residual_ld,
    int output_ld, void *stream_ptr) {
  if (dense_base_ptr == nullptr || dense_residual_ptr == nullptr ||
      dense_rows_ptr == nullptr || output_ptr == nullptr || dense_count <= 0 ||
      output_rows <= 0 || hidden_size <= 0 || base_ld < dense_count ||
      residual_ld < dense_count || output_ld < output_rows) {
    return -1;
  }
  const Element *dense_base =
      reinterpret_cast<const Element *>(dense_base_ptr);
  const Element *dense_residual =
      reinterpret_cast<const Element *>(dense_residual_ptr);
  const int *dense_rows = reinterpret_cast<const int *>(dense_rows_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total = dense_count * hidden_size;
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_routed_swiglu_correction_transposed_kernel
      <<<grid, block, 0, stream>>>(
          dense_base, dense_residual, dense_rows, output, dense_count,
          output_rows, hidden_size, base_ld, residual_ld, output_ld);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_routed_swiglu_correction_transpose_tiled_f16_stream(
    const void *sparse_hidden_ptr, const void *dense_base_ptr,
    const void *dense_residual_ptr, const void *dense_slot_by_row_ptr,
    void *output_ptr, int output_rows, int dense_count, int hidden_size,
    void *stream_ptr) {
  if (sparse_hidden_ptr == nullptr || dense_base_ptr == nullptr ||
      dense_residual_ptr == nullptr || dense_slot_by_row_ptr == nullptr ||
      output_ptr == nullptr || output_rows <= 0 || dense_count <= 0 ||
      dense_count > output_rows || hidden_size <= 0) {
    return -1;
  }
  const Element *sparse_hidden =
      reinterpret_cast<const Element *>(sparse_hidden_ptr);
  const Element *dense_base =
      reinterpret_cast<const Element *>(dense_base_ptr);
  const Element *dense_residual =
      reinterpret_cast<const Element *>(dense_residual_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  dim3 block(32, 8);
  dim3 grid((hidden_size + 31) / 32, (output_rows + 31) / 32);
  sparse24_cutlass_routed_swiglu_correction_transpose_tiled_kernel
      <<<grid, block, 0, stream>>>(
          sparse_hidden, dense_base, dense_residual, dense_slot_by_row, output,
          output_rows, dense_count, hidden_size);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_transpose_output_f16_stream(
    const void *input_ptr, void *output_ptr, int M, int N,
    void *stream_ptr) {
  if (input_ptr == nullptr || output_ptr == nullptr || M <= 0 || N <= 0) {
    return -1;
  }
  const Element *input = reinterpret_cast<const Element *>(input_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  dim3 block(32, 8);
  dim3 grid((M + 31) / 32, (N + 31) / 32);
  sparse24_cutlass_device_transpose_tiled_kernel<<<grid, block, 0, stream>>>(
      input, output, M, N);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_transpose_add_routed_residual_f16_stream(
    const void *full_ptr, const void *residual_ptr,
    const void *dense_slot_by_row_ptr, void *output_ptr, int M, int N,
    int full_leading_dim, int residual_leading_dim, int dense_count,
    void *stream_ptr) {
  if (full_ptr == nullptr || residual_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || output_ptr == nullptr || M <= 0 ||
      N <= 0 || full_leading_dim < M || dense_count <= 0 ||
      residual_leading_dim < dense_count) {
    return -1;
  }
  const Element *full = reinterpret_cast<const Element *>(full_ptr);
  const Element *residual =
      reinterpret_cast<const Element *>(residual_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  dim3 block(32, 8);
  dim3 grid((M + 31) / 32, (N + 31) / 32);
  sparse24_cutlass_transpose_add_routed_residual_kernel
      <<<grid, block, 0, stream>>>(
          full, residual, dense_slot_by_row, output, M, N, full_leading_dim,
          residual_leading_dim, dense_count);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_transpose_add_routed_residual_to_residual_f16_stream(
    const void *full_ptr, const void *routed_residual_ptr,
    const void *dense_slot_by_row_ptr, void *model_residual_ptr, int M, int N,
    int full_leading_dim, int routed_residual_leading_dim, int dense_count,
    void *stream_ptr) {
  if (full_ptr == nullptr || routed_residual_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || model_residual_ptr == nullptr ||
      M <= 0 || N <= 0 || full_leading_dim < M || dense_count <= 0 ||
      routed_residual_leading_dim < dense_count) {
    return -1;
  }
  const Element *full = reinterpret_cast<const Element *>(full_ptr);
  const Element *routed_residual =
      reinterpret_cast<const Element *>(routed_residual_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *model_residual = reinterpret_cast<Element *>(model_residual_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  dim3 block(32, 8);
  dim3 grid((M + 31) / 32, (N + 31) / 32);
  sparse24_cutlass_transpose_add_routed_residual_to_residual_kernel
      <<<grid, block, 0, stream>>>(
          full, routed_residual, dense_slot_by_row, model_residual, M, N,
          full_leading_dim, routed_residual_leading_dim, dense_count);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_transpose_add_routed_residual_rmsnorm_f16_stream(
    const void *full_ptr, const void *routed_residual_ptr,
    const void *dense_slot_by_row_ptr, void *model_residual_ptr,
    void *normalized_ptr, const void *weight_ptr, void *square_partials_ptr,
    int M, int N, int full_leading_dim, int routed_residual_leading_dim,
    int dense_count, float epsilon, void *stream_ptr) {
  if (full_ptr == nullptr || routed_residual_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || model_residual_ptr == nullptr ||
      normalized_ptr == nullptr || weight_ptr == nullptr ||
      square_partials_ptr == nullptr || M <= 0 || N <= 0 || N % 32 != 0 ||
      full_leading_dim < M || dense_count <= 0 ||
      routed_residual_leading_dim < dense_count || epsilon < 0.0f) {
    return -1;
  }
  const Element *full = reinterpret_cast<const Element *>(full_ptr);
  const Element *routed_residual =
      reinterpret_cast<const Element *>(routed_residual_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *model_residual = reinterpret_cast<Element *>(model_residual_ptr);
  Element *normalized = reinterpret_cast<Element *>(normalized_ptr);
  const Element *weight = reinterpret_cast<const Element *>(weight_ptr);
  float *square_partials = reinterpret_cast<float *>(square_partials_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  dim3 block(32, 8);
  dim3 grid((M + 31) / 32, N / 32);
  sparse24_cutlass_transpose_add_routed_residual_partials_kernel
      <<<grid, block, 0, stream>>>(
          full, routed_residual, dense_slot_by_row, model_residual,
          square_partials, M, N, full_leading_dim,
          routed_residual_leading_dim, dense_count);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  sparse24_cutlass_rmsnorm_from_partials_kernel<<<M, 256, 0, stream>>>(
      model_residual, normalized, weight, square_partials, M, N, N / 32,
      epsilon);
  err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_transpose_add_routed_splitk_residual_f16_stream(
    const void *full_ptr, const void *residual_partials_ptr,
    const void *dense_slot_by_row_ptr, void *output_ptr, int M, int N,
    int full_leading_dim, int residual_leading_dim, int dense_count,
    int split_k_slices, void *stream_ptr) {
  if (full_ptr == nullptr || residual_partials_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || output_ptr == nullptr || M <= 0 ||
      N <= 0 || full_leading_dim < M || dense_count <= 0 ||
      residual_leading_dim < dense_count || split_k_slices < 2 ||
      split_k_slices > 8) {
    return -1;
  }
  const Element *full = reinterpret_cast<const Element *>(full_ptr);
  const Element *residual_partials =
      reinterpret_cast<const Element *>(residual_partials_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  dim3 block(32, 8);
  dim3 grid((M + 31) / 32, (N + 31) / 32);
  sparse24_cutlass_transpose_add_routed_splitk_residual_kernel
      <<<grid, block, 0, stream>>>(
          full, residual_partials, dense_slot_by_row, output, M, N,
          full_leading_dim, residual_leading_dim, dense_count,
          split_k_slices);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_transpose_add_residual_f16_stream(
    const void *input_ptr, void *residual_ptr, int M, int N, int leading_dim,
    void *stream_ptr) {
  if (input_ptr == nullptr || residual_ptr == nullptr || M <= 0 || N <= 0 ||
      leading_dim < M) {
    return -1;
  }
  const Element *input = reinterpret_cast<const Element *>(input_ptr);
  Element *residual = reinterpret_cast<Element *>(residual_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  dim3 block(32, 8);
  dim3 grid((M + 31) / 32, (N + 31) / 32);
  sparse24_cutlass_transpose_add_residual_kernel<<<grid, block, 0, stream>>>(
      input, residual, M, N, leading_dim);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_transpose_add_rmsnorm_f16_stream(
    const void *input_ptr, void *residual_ptr, void *normalized_ptr,
    const void *weight_ptr, int M, int N, int leading_dim, float epsilon,
    void *stream_ptr) {
  if (input_ptr == nullptr || residual_ptr == nullptr ||
      normalized_ptr == nullptr || weight_ptr == nullptr || M <= 0 || N <= 0 ||
      leading_dim < M || N % 32 != 0 || epsilon < 0.0f) {
    return -1;
  }
  const Element *input = reinterpret_cast<const Element *>(input_ptr);
  Element *residual = reinterpret_cast<Element *>(residual_ptr);
  Element *normalized = reinterpret_cast<Element *>(normalized_ptr);
  const Element *weight = reinterpret_cast<const Element *>(weight_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const char *config = std::getenv("SPECLINK_MLP_EPILOGUE_CONFIG");
  if (config == nullptr || config[0] == '\0' || std::strcmp(config, "auto") == 0) {
    config = "8";
  }
#define LAUNCH_MLP_EPILOGUE(ROWS)                                          \
  do {                                                                     \
    dim3 block(32, 8);                                                     \
    dim3 grid((M + ROWS - 1) / ROWS);                                     \
    sparse24_cutlass_transpose_add_rmsnorm_kernel<ROWS>                   \
        <<<grid, block, 0, stream>>>(input, residual, normalized, weight,  \
                                     M, N, leading_dim, epsilon);          \
  } while (0)
  if (std::strcmp(config, "2") == 0) {
    LAUNCH_MLP_EPILOGUE(2);
  } else if (std::strcmp(config, "4") == 0) {
    LAUNCH_MLP_EPILOGUE(4);
  } else if (std::strcmp(config, "8") == 0) {
    LAUNCH_MLP_EPILOGUE(8);
  } else if (std::strcmp(config, "16") == 0) {
    LAUNCH_MLP_EPILOGUE(16);
  } else if (std::strcmp(config, "32") == 0) {
    LAUNCH_MLP_EPILOGUE(32);
  } else {
    return -2;
  }
#undef LAUNCH_MLP_EPILOGUE
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_qkv_transpose_rmsnorm_f16_stream(
    const void *input_ptr, void *output_ptr, const void *q_weight_ptr,
    const void *k_weight_ptr, int M, int q_size, int kv_size, int head_dim,
    int leading_dim, float epsilon, void *stream_ptr) {
  if (input_ptr == nullptr || output_ptr == nullptr || q_weight_ptr == nullptr ||
      k_weight_ptr == nullptr || M <= 0 || q_size <= 0 || kv_size <= 0 ||
      leading_dim < M || epsilon < 0.0f) {
    return -1;
  }
  if (head_dim != 128 || q_size % head_dim != 0 || kv_size % head_dim != 0) {
    return -2;
  }
  const Element *input = reinterpret_cast<const Element *>(input_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  const Element *q_weight = reinterpret_cast<const Element *>(q_weight_ptr);
  const Element *k_weight = reinterpret_cast<const Element *>(k_weight_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total_heads = (q_size + 2 * kv_size) / head_dim;
  dim3 block(32, 8);
  dim3 grid((M + 31) / 32, total_heads);
  sparse24_cutlass_qkv_transpose_rmsnorm_kernel<32, 8>
      <<<grid, block, 0, stream>>>(
      input, output, q_weight, k_weight, nullptr, nullptr, M, q_size, kv_size,
      leading_dim, 0, epsilon, true, true, false);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_qkv_transpose_postop_f16_stream(
    const void *input_ptr, void *output_ptr, const void *q_weight_ptr,
    const void *k_weight_ptr, const void *cos_sin_cache_ptr,
    const void *position_ids_ptr, int M, int q_size, int kv_size, int head_dim,
    int leading_dim, int rotary_dim, float epsilon, int is_neox,
    int normalize_qk, void *stream_ptr) {
  if (input_ptr == nullptr || output_ptr == nullptr ||
      cos_sin_cache_ptr == nullptr || position_ids_ptr == nullptr || M <= 0 ||
      q_size <= 0 || kv_size <= 0 || leading_dim < M || epsilon < 0.0f) {
    return -1;
  }
  if (normalize_qk != 0 &&
      (q_weight_ptr == nullptr || k_weight_ptr == nullptr)) {
    return -2;
  }
  if (head_dim != 128 || q_size % head_dim != 0 || kv_size % head_dim != 0 ||
      rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2 != 0) {
    return -3;
  }
  const Element *input = reinterpret_cast<const Element *>(input_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  const Element *q_weight = reinterpret_cast<const Element *>(q_weight_ptr);
  const Element *k_weight = reinterpret_cast<const Element *>(k_weight_ptr);
  const Element *cos_sin_cache =
      reinterpret_cast<const Element *>(cos_sin_cache_ptr);
  const int64_t *position_ids =
      reinterpret_cast<const int64_t *>(position_ids_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total_heads = (q_size + 2 * kv_size) / head_dim;
  const char *config = std::getenv("SPECLINK_QKV_POSTOP_CONFIG");
  if (config == nullptr || config[0] == '\0' || std::strcmp(config, "auto") == 0) {
    config = "32x8";
  }
#define LAUNCH_QKV_POSTOP(ROWS, LANES)                                      \
  do {                                                                      \
    dim3 block(32, LANES);                                                  \
    dim3 grid((M + ROWS - 1) / ROWS, total_heads);                         \
    sparse24_cutlass_qkv_transpose_rmsnorm_kernel<ROWS, LANES>             \
        <<<grid, block, 0, stream>>>(                                       \
            input, output, q_weight, k_weight, cos_sin_cache, position_ids, \
            M, q_size, kv_size, leading_dim, rotary_dim, epsilon,           \
            is_neox != 0, normalize_qk != 0, true);                         \
  } while (0)
  if (std::strcmp(config, "16x4") == 0) {
    LAUNCH_QKV_POSTOP(16, 4);
  } else if (std::strcmp(config, "16x8") == 0) {
    LAUNCH_QKV_POSTOP(16, 8);
  } else if (std::strcmp(config, "32x4") == 0) {
    LAUNCH_QKV_POSTOP(32, 4);
  } else if (std::strcmp(config, "32x8") == 0) {
    LAUNCH_QKV_POSTOP(32, 8);
  } else if (std::strcmp(config, "64x4") == 0) {
    LAUNCH_QKV_POSTOP(64, 4);
  } else if (std::strcmp(config, "64x8") == 0) {
    LAUNCH_QKV_POSTOP(64, 8);
  } else {
    return -4;
  }
#undef LAUNCH_QKV_POSTOP
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_qkv_transpose_add_routed_residual_postop_f16_stream(
    const void *input_ptr, const void *residual_ptr,
    const void *dense_slot_by_row_ptr, void *output_ptr,
    const void *q_weight_ptr, const void *k_weight_ptr,
    const void *cos_sin_cache_ptr, const void *position_ids_ptr, int M,
    int dense_count, int q_size, int kv_size, int head_dim,
    int leading_dim, int residual_leading_dim, int rotary_dim, float epsilon,
    int is_neox, int normalize_qk, int residual_row_major, void *stream_ptr) {
  if (input_ptr == nullptr || residual_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || output_ptr == nullptr ||
      cos_sin_cache_ptr == nullptr || position_ids_ptr == nullptr || M <= 0 ||
      dense_count <= 0 || dense_count > M || q_size <= 0 || kv_size <= 0 ||
      leading_dim < M || residual_leading_dim < dense_count ||
      epsilon < 0.0f) {
    return -1;
  }
  if (normalize_qk != 0 &&
      (q_weight_ptr == nullptr || k_weight_ptr == nullptr)) {
    return -2;
  }
  if (head_dim != 128 || q_size % head_dim != 0 || kv_size % head_dim != 0 ||
      rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2 != 0) {
    return -3;
  }
  const Element *input = reinterpret_cast<const Element *>(input_ptr);
  const Element *residual =
      reinterpret_cast<const Element *>(residual_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  Element *output = reinterpret_cast<Element *>(output_ptr);
  const Element *q_weight = reinterpret_cast<const Element *>(q_weight_ptr);
  const Element *k_weight = reinterpret_cast<const Element *>(k_weight_ptr);
  const Element *cos_sin_cache =
      reinterpret_cast<const Element *>(cos_sin_cache_ptr);
  const int64_t *position_ids =
      reinterpret_cast<const int64_t *>(position_ids_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total_heads = (q_size + 2 * kv_size) / head_dim;
  const char *config = std::getenv("SPECLINK_QKV_POSTOP_CONFIG");
  if (config == nullptr || config[0] == '\0' ||
      std::strcmp(config, "auto") == 0) {
    config = "32x8";
  }
#define LAUNCH_QKV_ROUTED_RESIDUAL_POSTOP(ROWS, LANES)                       \
  do {                                                                       \
    dim3 block(32, LANES);                                                   \
    dim3 grid((M + ROWS - 1) / ROWS, total_heads);                          \
    sparse24_cutlass_qkv_transpose_rmsnorm_kernel<ROWS, LANES, true>         \
        <<<grid, block, 0, stream>>>(                                        \
            input, output, q_weight, k_weight, cos_sin_cache, position_ids,  \
            M, q_size, kv_size, leading_dim, rotary_dim, epsilon,            \
            is_neox != 0, normalize_qk != 0, true, residual,                 \
            dense_slot_by_row, residual_leading_dim, dense_count,            \
            residual_row_major != 0);                                        \
  } while (0)
  if (std::strcmp(config, "16x4") == 0) {
    LAUNCH_QKV_ROUTED_RESIDUAL_POSTOP(16, 4);
  } else if (std::strcmp(config, "16x8") == 0) {
    LAUNCH_QKV_ROUTED_RESIDUAL_POSTOP(16, 8);
  } else if (std::strcmp(config, "32x4") == 0) {
    LAUNCH_QKV_ROUTED_RESIDUAL_POSTOP(32, 4);
  } else if (std::strcmp(config, "32x8") == 0) {
    LAUNCH_QKV_ROUTED_RESIDUAL_POSTOP(32, 8);
  } else if (std::strcmp(config, "64x4") == 0) {
    LAUNCH_QKV_ROUTED_RESIDUAL_POSTOP(64, 4);
  } else if (std::strcmp(config, "64x8") == 0) {
    LAUNCH_QKV_ROUTED_RESIDUAL_POSTOP(64, 8);
  } else {
    return -4;
  }
#undef LAUNCH_QKV_ROUTED_RESIDUAL_POSTOP
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_qkv_add_routed_residual_postop_inplace_f16_stream(
    void *qkv_ptr, const void *residual_ptr,
    const void *dense_slot_by_row_ptr, const void *q_weight_ptr,
    const void *k_weight_ptr, const void *cos_sin_cache_ptr,
    const void *position_ids_ptr, int M, int dense_count, int q_size,
    int kv_size, int head_dim, int rotary_dim, float epsilon, int is_neox,
    int normalize_qk, void *stream_ptr) {
  if (qkv_ptr == nullptr || residual_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || cos_sin_cache_ptr == nullptr ||
      position_ids_ptr == nullptr || M <= 0 || dense_count <= 0 ||
      dense_count > M || q_size <= 0 || kv_size <= 0 || epsilon < 0.0f) {
    return -1;
  }
  if (normalize_qk != 0 &&
      (q_weight_ptr == nullptr || k_weight_ptr == nullptr)) {
    return -2;
  }
  if (head_dim != 128 || q_size % head_dim != 0 || kv_size % head_dim != 0 ||
      rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2 != 0) {
    return -3;
  }
  Element *qkv = reinterpret_cast<Element *>(qkv_ptr);
  const Element *residual = reinterpret_cast<const Element *>(residual_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  const Element *q_weight = reinterpret_cast<const Element *>(q_weight_ptr);
  const Element *k_weight = reinterpret_cast<const Element *>(k_weight_ptr);
  const Element *cos_sin_cache =
      reinterpret_cast<const Element *>(cos_sin_cache_ptr);
  const int64_t *position_ids =
      reinterpret_cast<const int64_t *>(position_ids_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int normalized_heads = (q_size + kv_size) / head_dim;
  const char *config = std::getenv("SPECLINK_QKV_POSTOP_CONFIG");
  if (config == nullptr || config[0] == '\0' ||
      std::strcmp(config, "auto") == 0) {
    config = "32x8";
  }
#define LAUNCH_QKV_ROWMAJOR_WARP_POSTOP(ROWS)                              \
  do {                                                                     \
    dim3 block(256);                                                       \
    dim3 grid((M + ROWS - 1) / ROWS,                                     \
              (q_size + 2 * kv_size + 255) / 256);                        \
    sparse24_cutlass_qkv_rowmajor_routed_postop_kernel<ROWS>              \
        <<<grid, block, 0, stream>>>(                                     \
            qkv, residual, dense_slot_by_row, q_weight, k_weight,         \
            cos_sin_cache, position_ids, M, dense_count, q_size, kv_size, \
            rotary_dim, epsilon, is_neox != 0, normalize_qk != 0);        \
  } while (0)
#define LAUNCH_QKV_ROWMAJOR_VEC4_POSTOP(ROWS)                              \
  do {                                                                     \
    if (is_neox == 0 || rotary_dim != 128) {                               \
      return -5;                                                           \
    }                                                                      \
    dim3 block(256);                                                       \
    dim3 grid((M + ROWS - 1) / ROWS,                                     \
              (q_size + 2 * kv_size + 255) / 256);                        \
    sparse24_cutlass_qkv_rowmajor_routed_postop_vec4_kernel<ROWS>         \
        <<<grid, block, 0, stream>>>(                                     \
            qkv, residual, dense_slot_by_row, q_weight, k_weight,         \
            cos_sin_cache, position_ids, M, dense_count, q_size, kv_size, \
            epsilon, normalize_qk != 0);                                  \
  } while (0)
#define LAUNCH_QKV_ROWMAJOR_RESIDUAL_POSTOP(ROWS, LANES)                    \
  do {                                                                      \
    dim3 block(32, LANES);                                                  \
    dim3 grid((M + ROWS - 1) / ROWS, normalized_heads);                    \
    sparse24_cutlass_qkv_transpose_rmsnorm_kernel<ROWS, LANES, true, true, \
                                                   true>                    \
        <<<grid, block, 0, stream>>>(                                       \
            qkv, qkv, q_weight, k_weight, cos_sin_cache, position_ids, M,  \
            q_size, kv_size, q_size + 2 * kv_size, rotary_dim, epsilon,    \
            is_neox != 0, normalize_qk != 0, true, residual,               \
            dense_slot_by_row, q_size + 2 * kv_size, dense_count, true);   \
  } while (0)
  if (std::strcmp(config, "vec8") == 0) {
    LAUNCH_QKV_ROWMAJOR_VEC4_POSTOP(8);
  } else if (std::strcmp(config, "vec16") == 0) {
    LAUNCH_QKV_ROWMAJOR_VEC4_POSTOP(16);
  } else if (std::strcmp(config, "vec32") == 0) {
    LAUNCH_QKV_ROWMAJOR_VEC4_POSTOP(32);
  } else if (std::strcmp(config, "vec64") == 0) {
    LAUNCH_QKV_ROWMAJOR_VEC4_POSTOP(64);
  } else if (std::strcmp(config, "warp8") == 0) {
    LAUNCH_QKV_ROWMAJOR_WARP_POSTOP(8);
  } else if (std::strcmp(config, "warp16") == 0) {
    LAUNCH_QKV_ROWMAJOR_WARP_POSTOP(16);
  } else if (std::strcmp(config, "warp32") == 0) {
    LAUNCH_QKV_ROWMAJOR_WARP_POSTOP(32);
  } else if (std::strcmp(config, "warp64") == 0) {
    LAUNCH_QKV_ROWMAJOR_WARP_POSTOP(64);
  } else if (std::strcmp(config, "warp128") == 0) {
    LAUNCH_QKV_ROWMAJOR_WARP_POSTOP(128);
  } else if (std::strcmp(config, "warp256") == 0) {
    LAUNCH_QKV_ROWMAJOR_WARP_POSTOP(256);
  } else if (std::strcmp(config, "16x4") == 0) {
    LAUNCH_QKV_ROWMAJOR_RESIDUAL_POSTOP(16, 4);
  } else if (std::strcmp(config, "16x8") == 0) {
    LAUNCH_QKV_ROWMAJOR_RESIDUAL_POSTOP(16, 8);
  } else if (std::strcmp(config, "32x4") == 0) {
    LAUNCH_QKV_ROWMAJOR_RESIDUAL_POSTOP(32, 4);
  } else if (std::strcmp(config, "32x8") == 0) {
    LAUNCH_QKV_ROWMAJOR_RESIDUAL_POSTOP(32, 8);
  } else if (std::strcmp(config, "64x4") == 0) {
    LAUNCH_QKV_ROWMAJOR_RESIDUAL_POSTOP(64, 4);
  } else if (std::strcmp(config, "64x8") == 0) {
    LAUNCH_QKV_ROWMAJOR_RESIDUAL_POSTOP(64, 8);
  } else {
    return -4;
  }
#undef LAUNCH_QKV_ROWMAJOR_RESIDUAL_POSTOP
#undef LAUNCH_QKV_ROWMAJOR_VEC4_POSTOP
#undef LAUNCH_QKV_ROWMAJOR_WARP_POSTOP
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int
sparse24_cutlass_qkv_add_routed_residual_postop_cache_inplace_f16_stream(
    void *qkv_ptr, const void *residual_ptr,
    const void *dense_slot_by_row_ptr, const void *q_weight_ptr,
    const void *k_weight_ptr, const void *cos_sin_cache_ptr,
    const void *position_ids_ptr, const void *slot_mapping_ptr,
    void *key_cache_ptr, void *value_cache_ptr, int M, int dense_count,
    int cache_token_count, int q_size, int kv_size, int head_dim,
    int rotary_dim, int block_size, int64_t cache_block_stride,
    int64_t cache_page_stride, int64_t cache_head_stride, float epsilon,
    int is_neox, int normalize_qk, void *stream_ptr) {
  if (qkv_ptr == nullptr || residual_ptr == nullptr ||
      dense_slot_by_row_ptr == nullptr || cos_sin_cache_ptr == nullptr ||
      position_ids_ptr == nullptr || slot_mapping_ptr == nullptr ||
      key_cache_ptr == nullptr || value_cache_ptr == nullptr || M <= 0 ||
      dense_count <= 0 || dense_count > M || q_size <= 0 || kv_size <= 0 ||
      cache_token_count < 0 || cache_token_count > M ||
      block_size <= 0 || cache_block_stride <= 0 || cache_page_stride <= 0 ||
      cache_head_stride <= 0 || epsilon < 0.0f) {
    return -1;
  }
  if (normalize_qk != 0 &&
      (q_weight_ptr == nullptr || k_weight_ptr == nullptr)) {
    return -2;
  }
  if (head_dim != 128 || rotary_dim != 128 || is_neox == 0 ||
      q_size % head_dim != 0 || kv_size % head_dim != 0) {
    return -3;
  }
  Element *qkv = reinterpret_cast<Element *>(qkv_ptr);
  const Element *residual = reinterpret_cast<const Element *>(residual_ptr);
  const int *dense_slot_by_row =
      reinterpret_cast<const int *>(dense_slot_by_row_ptr);
  const Element *q_weight = reinterpret_cast<const Element *>(q_weight_ptr);
  const Element *k_weight = reinterpret_cast<const Element *>(k_weight_ptr);
  const Element *cos_sin_cache =
      reinterpret_cast<const Element *>(cos_sin_cache_ptr);
  const int64_t *position_ids =
      reinterpret_cast<const int64_t *>(position_ids_ptr);
  const int64_t *slot_mapping =
      reinterpret_cast<const int64_t *>(slot_mapping_ptr);
  Element *key_cache = reinterpret_cast<Element *>(key_cache_ptr);
  Element *value_cache = reinterpret_cast<Element *>(value_cache_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const char *config = std::getenv("SPECLINK_QKV_POSTOP_CONFIG");
  if (config == nullptr || config[0] == '\0' ||
      std::strcmp(config, "auto") == 0) {
    config = "vec8";
  }
#define LAUNCH_QKV_ROWMAJOR_CACHE_VEC4_POSTOP(ROWS)                        \
  do {                                                                     \
    dim3 block(256);                                                       \
    dim3 grid((M + ROWS - 1) / ROWS,                                     \
              (q_size + 2 * kv_size + 255) / 256);                        \
    sparse24_cutlass_qkv_rowmajor_routed_postop_cache_vec4_kernel<ROWS>   \
        <<<grid, block, 0, stream>>>(                                     \
            qkv, residual, dense_slot_by_row, q_weight, k_weight,         \
            cos_sin_cache, position_ids, slot_mapping, key_cache,         \
            value_cache, M, dense_count, cache_token_count, q_size,       \
            kv_size, block_size, cache_block_stride, cache_page_stride,   \
            cache_head_stride, epsilon, normalize_qk != 0);               \
  } while (0)
  if (std::strcmp(config, "vec8") == 0) {
    LAUNCH_QKV_ROWMAJOR_CACHE_VEC4_POSTOP(8);
  } else if (std::strcmp(config, "vec16") == 0) {
    LAUNCH_QKV_ROWMAJOR_CACHE_VEC4_POSTOP(16);
  } else if (std::strcmp(config, "vec32") == 0) {
    LAUNCH_QKV_ROWMAJOR_CACHE_VEC4_POSTOP(32);
  } else if (std::strcmp(config, "vec64") == 0) {
    LAUNCH_QKV_ROWMAJOR_CACHE_VEC4_POSTOP(64);
  } else {
    return -4;
  }
#undef LAUNCH_QKV_ROWMAJOR_CACHE_VEC4_POSTOP
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_qkv_rmsnorm_inplace_f16_stream(
    void *qkv_ptr, const void *q_weight_ptr, const void *k_weight_ptr, int M,
    int q_size, int kv_size, int head_dim, float epsilon, void *stream_ptr) {
  if (qkv_ptr == nullptr || q_weight_ptr == nullptr || k_weight_ptr == nullptr ||
      M <= 0 || q_size <= 0 || kv_size <= 0 || epsilon < 0.0f) {
    return -1;
  }
  if (head_dim != 128 || q_size % head_dim != 0 || kv_size % head_dim != 0) {
    return -2;
  }
  Element *qkv = reinterpret_cast<Element *>(qkv_ptr);
  const Element *q_weight = reinterpret_cast<const Element *>(q_weight_ptr);
  const Element *k_weight = reinterpret_cast<const Element *>(k_weight_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int normalized_heads = (q_size + kv_size) / head_dim;
  dim3 grid(M, normalized_heads);
  sparse24_cutlass_qkv_rmsnorm_inplace_kernel<<<grid, 128, 0, stream>>>(
      qkv, q_weight, k_weight, M, q_size, kv_size, epsilon);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_mixed_dense_override_f16_stream(
    const void *x_ptr, const void *w_col_ptr, const void *a_values_ptr,
    const void *a_meta_e_ptr, const void *row_indices_ptr, void *dense_x_ptr,
    void *dense_y_ptr, void *c_tmp_ptr, void *y_ptr, int M, int K, int N,
    int dense_rows, void *stream_ptr) {
  if (M <= 0 || K <= 0 || N <= 0 || dense_rows <= 0 || dense_rows > M) {
    return -1;
  }
  if ((K % DeviceThreadblockShape::kK) != 0) {
    return -2;
  }
  if ((N % 32) != 0) {
    return -3;
  }
  if ((M % 8) != 0) {
    return -4;
  }
  if (x_ptr == nullptr || w_col_ptr == nullptr || a_values_ptr == nullptr ||
      a_meta_e_ptr == nullptr || row_indices_ptr == nullptr ||
      dense_x_ptr == nullptr || dense_y_ptr == nullptr || c_tmp_ptr == nullptr ||
      y_ptr == nullptr) {
    return -5;
  }

  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  const Element *w_col = reinterpret_cast<const Element *>(w_col_ptr);
  const Element *a_values = reinterpret_cast<const Element *>(a_values_ptr);
  DeviceElementE *a_meta_e = const_cast<DeviceElementE *>(
      reinterpret_cast<const DeviceElementE *>(a_meta_e_ptr));
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  Element *dense_x = reinterpret_cast<Element *>(dense_x_ptr);
  Element *dense_y = reinterpret_cast<Element *>(dense_y_ptr);
  Element *c_tmp = reinterpret_cast<Element *>(c_tmp_ptr);
  Element *y = reinterpret_cast<Element *>(y_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const char *sparse_config = sparse24_cutlass_select_device_config(
      std::getenv("SPECLINK_SPARSE24_DEVICE_CONFIG"), M, K, N);
  const char *accumulator = std::getenv("SPECLINK_SPARSE24_ACCUMULATOR");
  bool use_f16_accumulator =
      sparse24_cutlass_use_f16_accumulator(accumulator, K, N);
  int status = -5;
  if (use_f16_accumulator) {
    status = sparse24_cutlass_device_dispatch_config_f16_accum(
        sparse_config, x, a_values, a_meta_e, c_tmp, M, K, N, stream);
  }
  if (status == -5) {
    status = sparse24_cutlass_device_dispatch_config(
        sparse_config, x, a_values, a_meta_e, c_tmp, M, K, N, stream);
  }
  if (status != 0) {
    return status;
  }

  dim3 transpose_block(32, 8);
  dim3 transpose_grid((M + 31) / 32, (N + 31) / 32);
  sparse24_cutlass_device_transpose_tiled_kernel<<<transpose_grid,
                                                   transpose_block, 0, stream>>>(
      c_tmp, y, M, N);

  if ((K % 8) != 0) {
    return -6;
  }
  int gather_vectors = K / 8;
  int gather_total = dense_rows * gather_vectors;
  int block = 256;
  int gather_grid = (gather_total + block - 1) / block;
  sparse24_cutlass_gather_rows_f16x8_kernel<<<gather_grid, block, 0, stream>>>(
      x_ptr, dense_x_ptr, row_indices, dense_rows, gather_vectors);

  const char *dense_config = dense_cutlass_select_device_config(
      std::getenv("SPECLINK_MIXED_DENSE_GEMM_CONFIG"), dense_rows, K, N);
  status = dense_cutlass_device_b_col_dispatch_config(
      dense_config, dense_x, w_col, dense_y, dense_rows, K, N, stream);
  if (status != 0) {
    return status;
  }

  int copy_total = dense_rows * N;
  int copy_grid = (copy_total + block - 1) / block;
  sparse24_cutlass_copy_indexed_rows_rowmajor_kernel<<<copy_grid, block, 0,
                                                       stream>>>(
      y, dense_y, row_indices, dense_rows, M, N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int sparse24_cutlass_add_prefix_strided_f16_stream(
    void *full_out_ptr, const void *prefix_add_ptr, int dense_rows, int full_m,
    int prefix_m, int N, void *stream_ptr) {
  if (dense_rows <= 0 || full_m <= 0 || prefix_m <= 0 || N <= 0 ||
      dense_rows > full_m || dense_rows > prefix_m) {
    return -1;
  }
  Element *full_out = reinterpret_cast<Element *>(full_out_ptr);
  const Element *prefix_add = reinterpret_cast<const Element *>(prefix_add_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total = dense_rows * N;
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_add_prefix_strided_kernel<<<grid, block, 0, stream>>>(
      full_out, prefix_add, dense_rows, full_m, prefix_m, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int sparse24_cutlass_add_indexed_rows_strided_f16_stream(
    void *full_out_ptr, const void *row_add_ptr, const void *row_indices_ptr,
    int dense_rows, int full_m, int row_m, int N, void *stream_ptr) {
  if (dense_rows <= 0 || full_m <= 0 || row_m <= 0 || N <= 0 ||
      dense_rows > full_m || dense_rows > row_m) {
    return -1;
  }
  if (full_out_ptr == nullptr || row_add_ptr == nullptr ||
      row_indices_ptr == nullptr) {
    return -2;
  }
  Element *full_out = reinterpret_cast<Element *>(full_out_ptr);
  const Element *row_add = reinterpret_cast<const Element *>(row_add_ptr);
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total = dense_rows * N;
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_add_indexed_rows_strided_kernel<<<grid, block, 0, stream>>>(
      full_out, row_add, row_indices, dense_rows, full_m, row_m, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int sparse24_cutlass_add_indexed_rows_contiguous_f16_stream(
    void *full_out_ptr, const void *row_add_ptr, const void *row_indices_ptr,
    int dense_rows, int full_m, int N, void *stream_ptr) {
  if (dense_rows <= 0 || full_m <= 0 || N <= 0 || dense_rows > full_m) {
    return -1;
  }
  if (full_out_ptr == nullptr || row_add_ptr == nullptr ||
      row_indices_ptr == nullptr) {
    return -2;
  }
  Element *full_out = reinterpret_cast<Element *>(full_out_ptr);
  const Element *row_add = reinterpret_cast<const Element *>(row_add_ptr);
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total = dense_rows * N;
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_add_indexed_rows_contiguous_kernel<<<grid, block, 0, stream>>>(
      full_out, row_add, row_indices, dense_rows, full_m, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int
sparse24_cutlass_add_indexed_rows_transposed_to_contiguous_f16_stream(
    void *full_out_ptr, const void *row_add_ptr, const void *row_indices_ptr,
    int dense_rows, int full_m, int row_m, int N, void *stream_ptr) {
  if (dense_rows <= 0 || full_m <= 0 || row_m <= 0 || N <= 0 ||
      dense_rows > full_m || dense_rows > row_m) {
    return -1;
  }
  if (full_out_ptr == nullptr || row_add_ptr == nullptr ||
      row_indices_ptr == nullptr) {
    return -2;
  }
  Element *full_out = reinterpret_cast<Element *>(full_out_ptr);
  const Element *row_add = reinterpret_cast<const Element *>(row_add_ptr);
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  dim3 block(32, 8);
  dim3 grid((dense_rows + 31) / 32, (N + 31) / 32);
  sparse24_cutlass_add_indexed_rows_transposed_to_contiguous_kernel
      <<<grid, block, 0, stream>>>(
          full_out, row_add, row_indices, dense_rows, full_m, row_m, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int sparse24_cutlass_sub_indexed_rows_contiguous_f16_stream(
    void *full_out_ptr, const void *row_sub_ptr, const void *row_indices_ptr,
    int sparse_rows, int full_m, int N, void *stream_ptr) {
  if (sparse_rows <= 0 || full_m <= 0 || N <= 0 || sparse_rows > full_m) {
    return -1;
  }
  if (full_out_ptr == nullptr || row_sub_ptr == nullptr ||
      row_indices_ptr == nullptr) {
    return -2;
  }
  Element *full_out = reinterpret_cast<Element *>(full_out_ptr);
  const Element *row_sub = reinterpret_cast<const Element *>(row_sub_ptr);
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total = sparse_rows * N;
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_sub_indexed_rows_contiguous_kernel<<<grid, block, 0, stream>>>(
      full_out, row_sub, row_indices, sparse_rows, full_m, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int sparse24_cutlass_gather_rows_f16_stream(
    const void *x_ptr, void *out_ptr, const void *row_indices_ptr,
    int dense_rows, int K, void *stream_ptr) {
  if (dense_rows <= 0 || K <= 0) {
    return -1;
  }
  if (x_ptr == nullptr || out_ptr == nullptr || row_indices_ptr == nullptr) {
    return -2;
  }
  if ((K % 8) != 0) {
    return -3;
  }
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int K_vectors = K / 8;
  int total = dense_rows * K_vectors;
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_gather_rows_f16x8_kernel<<<grid, block, 0, stream>>>(
      x_ptr, out_ptr, row_indices, dense_rows, K_vectors);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int sparse24_cutlass_gather_rows_strided_f16_stream(
    const void *x_ptr, void *out_ptr, const void *row_indices_ptr,
    int dense_rows, int full_m, int out_m, int K, void *stream_ptr) {
  if (dense_rows <= 0 || full_m <= 0 || out_m < dense_rows || K <= 0 ||
      dense_rows > full_m) {
    return -1;
  }
  if (x_ptr == nullptr || out_ptr == nullptr || row_indices_ptr == nullptr) {
    return -2;
  }
  const Element *x = reinterpret_cast<const Element *>(x_ptr);
  Element *out = reinterpret_cast<Element *>(out_ptr);
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total = dense_rows * K;
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_gather_rows_strided_kernel<<<grid, block, 0, stream>>>(
      x, out, row_indices, dense_rows, full_m, out_m, K);
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 0 : static_cast<int>(err);
}

extern "C" int sparse24_cutlass_partition_rows_f16_stream(
    const void *x_ptr, void *dense_out_ptr, void *sparse_out_ptr,
    const void *dense_indices_ptr, const void *sparse_indices_ptr,
    int dense_rows, int sparse_rows, int full_m, int K, void *stream_ptr) {
  if (dense_rows <= 0 || sparse_rows <= 0 || full_m <= 0 || K <= 0 ||
      dense_rows + sparse_rows != full_m) {
    return -1;
  }
  if (x_ptr == nullptr || dense_out_ptr == nullptr ||
      sparse_out_ptr == nullptr || dense_indices_ptr == nullptr ||
      sparse_indices_ptr == nullptr) {
    return -2;
  }
  if ((K % 8) != 0) {
    return -3;
  }
  const int *dense_indices =
      reinterpret_cast<const int *>(dense_indices_ptr);
  const int *sparse_indices =
      reinterpret_cast<const int *>(sparse_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int K_vectors = K / 8;
  int total = full_m * K_vectors;
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_partition_rows_f16x8_kernel<<<grid, block, 0, stream>>>(
      x_ptr, dense_out_ptr, sparse_out_ptr, dense_indices, sparse_indices,
      dense_rows, sparse_rows, K_vectors);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int sparse24_cutlass_merge_rows_f16_stream(
    void *out_ptr, const void *dense_values_ptr,
    const void *sparse_values_ptr, const void *dense_indices_ptr,
    const void *sparse_indices_ptr, int dense_rows, int sparse_rows,
    int full_m, int N, void *stream_ptr) {
  if (dense_rows <= 0 || sparse_rows <= 0 || full_m <= 0 || N <= 0 ||
      dense_rows + sparse_rows != full_m) {
    return -1;
  }
  if (out_ptr == nullptr || dense_values_ptr == nullptr ||
      sparse_values_ptr == nullptr || dense_indices_ptr == nullptr ||
      sparse_indices_ptr == nullptr) {
    return -2;
  }
  if ((N % 8) != 0) {
    return -3;
  }
  const int *dense_indices =
      reinterpret_cast<const int *>(dense_indices_ptr);
  const int *sparse_indices =
      reinterpret_cast<const int *>(sparse_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int N_vectors = N / 8;
  int total = full_m * N_vectors;
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_merge_rows_f16x8_kernel<<<grid, block, 0, stream>>>(
      out_ptr, dense_values_ptr, sparse_values_ptr, dense_indices,
      sparse_indices, dense_rows, sparse_rows, N_vectors);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int sparse24_cutlass_copy_indexed_rows_strided_f16_stream(
    void *full_out_ptr, const void *row_values_ptr, const void *row_indices_ptr,
    int dense_rows, int full_m, int row_m, int N, void *stream_ptr) {
  if (dense_rows <= 0 || full_m <= 0 || row_m <= 0 || N <= 0 ||
      dense_rows > full_m || dense_rows > row_m) {
    return -1;
  }
  if (full_out_ptr == nullptr || row_values_ptr == nullptr ||
      row_indices_ptr == nullptr) {
    return -2;
  }
  Element *full_out = reinterpret_cast<Element *>(full_out_ptr);
  const Element *row_values = reinterpret_cast<const Element *>(row_values_ptr);
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total = dense_rows * N;
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_copy_indexed_rows_strided_kernel<<<grid, block, 0, stream>>>(
      full_out, row_values, row_indices, dense_rows, full_m, row_m, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int sparse24_cutlass_copy_indexed_rows_contiguous_f16_stream(
    void *full_out_ptr, const void *row_values_ptr, const void *row_indices_ptr,
    int dense_rows, int full_m, int N, void *stream_ptr) {
  if (dense_rows <= 0 || full_m <= 0 || N <= 0 || dense_rows > full_m) {
    return -1;
  }
  if (full_out_ptr == nullptr || row_values_ptr == nullptr ||
      row_indices_ptr == nullptr) {
    return -2;
  }
  Element *full_out = reinterpret_cast<Element *>(full_out_ptr);
  const Element *row_values = reinterpret_cast<const Element *>(row_values_ptr);
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total = dense_rows * N;
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_copy_indexed_rows_contiguous_kernel<<<grid, block, 0, stream>>>(
      full_out, row_values, row_indices, dense_rows, full_m, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int sparse24_cutlass_copy_indexed_rows_rowmajor_f16_stream(
    void *full_out_ptr, const void *row_values_ptr, const void *row_indices_ptr,
    int dense_rows, int full_m, int N, void *stream_ptr) {
  if (dense_rows <= 0 || full_m <= 0 || N <= 0 || dense_rows > full_m) {
    return -1;
  }
  if (full_out_ptr == nullptr || row_values_ptr == nullptr ||
      row_indices_ptr == nullptr) {
    return -2;
  }
  Element *full_out = reinterpret_cast<Element *>(full_out_ptr);
  const Element *row_values = reinterpret_cast<const Element *>(row_values_ptr);
  const int *row_indices = reinterpret_cast<const int *>(row_indices_ptr);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int total = dense_rows * N;
  int block = 256;
  int grid = (total + block - 1) / block;
  sparse24_cutlass_copy_indexed_rows_rowmajor_kernel<<<grid, block, 0, stream>>>(
      full_out, row_values, row_indices, dense_rows, full_m, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return 0;
}
