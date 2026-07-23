/***************************************************************************
 * Copyright 2025 The SpInfer Authors. All rights reserved.
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
// Extended from CUTLASS and https://github.com/AlibabaResearch/flash-llm/blob/main/csrc/MatMulUtilities.cuh
#ifndef MatMulUtilities_H
#define MatMulUtilities_H
#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "AsyncCopy_PTX.cuh"
#include "MMA_PTX.cuh"
#include "TilingConfig.h"
// New features: Copy size is X * 64, X can be any multiple to 8
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_64(half* __restrict__ SharedPTR,
                                                                const half* GlobalPTR,
                                                                const int   GlobalStride,
                                                                bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 8;
    int row1          = lane_id / 8;
    int row2          = lane_id / 8 + 4;
    int store_column1 = col ^ row1;
    int store_column2 = col ^ row2;
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy / COPY_UNIT_FP16_ROWS;
    const int MaxIteration =
        (TotalNumOfCopyUnit - 1) / (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;
//
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int  COPY_UNIT_I        = (i * (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id);
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  
        const half* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * GlobalStride;
        half* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * TILE_K;
        cp_async<16>(SharedPTR_Unit + store_column1 * HALF_PER_128B + row1 * TILE_K,
                     GlobalPTR_Unit + col * HALF_PER_128B + row1 * GlobalStride,
                     AsyncCopyPredictor);
        cp_async<16>(SharedPTR_Unit + store_column2 * HALF_PER_128B + row2 * TILE_K,
                     GlobalPTR_Unit + col * HALF_PER_128B + row2 * GlobalStride,
                     AsyncCopyPredictor);
    }
}
// New features: Copy size is X * 64 Uint64, X can be  1 // for BitmapV2
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_Bitmap_1_64(uint64_t* __restrict__ SharedPTR,
                                                               const uint64_t* GlobalPTR,
                                                               bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy;  //   
    bool AsyncCopyPredictor = warp_id < TotalNumOfCopyUnit && Pred;  
    const uint64_t* GlobalPTR_Unit        = GlobalPTR;  
    uint64_t* __restrict__ SharedPTR_Unit = SharedPTR; 
    cp_async<16>(SharedPTR_Unit + lane_id * UINT64_PER_128B, 
                     GlobalPTR_Unit + lane_id * UINT64_PER_128B,   
                     AsyncCopyPredictor);
}


template<typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_Sparse(half* __restrict__ SharedPTR,
                                                               const half* GlobalPTR,
                                                               const int   NNZ,
                                                               bool        Pred = true)
{
    if(Pred) {
    int threadPerBlock = blockDim.x;
    int NNZ_8 = (NNZ>>3);
    for(int i = threadIdx.x; i < NNZ_8; i+= threadPerBlock) {
        const half* GlobalPTR_Unit        =  GlobalPTR + i * 8;  
        half* __restrict__ SharedPTR_Unit = SharedPTR + i * 8; 
        cp_async<16>(SharedPTR_Unit, 
                     GlobalPTR_Unit,   
                     Pred);
    }
    }
}

template<typename TilingConfig>
__device__ __forceinline__ void PipelinedCoreComputationsBitmap(float c[][REG_PER_C_TENSOR_16_16],
                                                          uint32_t __restrict__ a[][4],
                                                          uint32_t __restrict__ b[][4],
                                                          half* __restrict__ SharedMemoryPTR,
                                                          int warp_start_row,
                                                          int warp_start_col)
{
    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
    B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
        b, SharedMemoryPTR, warp_start_col, 0);
// Sencond loading & first computation, so on
#pragma unroll
    for (int k = 0; k < BLOCK_K_TENSORS; k++) {
        uint32_t __restrict__(*b_read)[4]  = b;
        uint32_t __restrict__(*b_write)[4] = b;
        b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS;
        b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS;
        // data loading
        if (k + 1 < BLOCK_K_TENSORS) {
            B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
                b_write, SharedMemoryPTR, warp_start_col, (k + 1) * MMA_K);
        }
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
                MMA_FP16_M16N8K16(c_uint32_t[j * WARP_ROW_TENSORS_BITMAP_V1], a[k], b_read[j]);
                if (!TilingConfig::N8)
                    MMA_FP16_M16N8K16(c_uint32_t[j * WARP_ROW_TENSORS_BITMAP_V1] + 4, a[k], b_read[j] + 2);  // c+4; b+2
            }
    }
}
__device__ __forceinline__ half2 maskloadingv2(uint64_t bitmap, const half* __restrict__ startpos, int lane_id) {
    int lid_offset = lane_id << 1;
    uint64_t bit1 = 1ULL << lid_offset;
    uint64_t bit2 = 2ULL << lid_offset;

    // Calculate the number of ones before lane_id * 2
    int num_ones_before = __popcll(bitmap & ((1ULL << lid_offset) - 1));

    // Load A_val1 and adjust the offset for A_val2
    half A_val1 = (bitmap & bit1) ? startpos[num_ones_before++] : __float2half(0.0f);
    half A_val2 = (bitmap & bit2) ? startpos[num_ones_before] : __float2half(0.0f);

    // Combine two half values into a half2
    return __halves2half2(A_val1, A_val2);
}
__device__ __forceinline__ void SpMM_LoadFragAwithBitmapFromShemv2(uint32_t __restrict__ a[][4],
                                                         const half* __restrict__ ShemVal,
                                                         const uint64_t* __restrict__ SharedBitmap,
                                                         int* start_pos,
                                                         int bit_offset)
{
    int lane_id = threadIdx.x % 32;
    const uint64_t* SharedBitmapStart = SharedBitmap + bit_offset;
    // #pragma unroll
    for (int i = 0; i < 4; i++) {
            // #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint64_t bitmap = SharedBitmapStart[i * 4 + j];
                half2 val = maskloadingv2(bitmap, ShemVal+*start_pos, lane_id);
                a[i][j] = *reinterpret_cast<const uint32_t*>(&val);
                *start_pos += __popcll(bitmap);
            }
    }
}
template<typename TilingConfig>
__device__ __forceinline__ void PipelinedCoreComputationsBitmapV2(float c[][REG_PER_C_TENSOR_16_16],
                                                          uint32_t __restrict__ a[][4],
                                                          uint32_t __restrict__ b[][4],
                                                          half* __restrict__ ShemAVal,
                                                          uint64_t* __restrict__ SharedMemoryBitmapPTR,
                                                          half* __restrict__ SharedMemoryPTR,
                                                          int warp_start_row,
                                                          int warp_start_col)
{
    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
    int start_pos = 0;
    SpMM_LoadFragAwithBitmapFromShemv2(a, ShemAVal, SharedMemoryBitmapPTR, &start_pos, 0);
    B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
        b, SharedMemoryPTR, warp_start_col, 0);
// Sencond loading & first computation, so on
#pragma unroll
    for (int k = 0; k < BLOCK_K_TENSORS; k++) {
        uint32_t __restrict__(*a_read)[4]  = a;
        uint32_t __restrict__(*b_read)[4]  = b;
        uint32_t __restrict__(*a_write)[4] = a;
        uint32_t __restrict__(*b_write)[4] = b;
        a_read += ((k) % 2) * WARP_ROW_TENSORS_BITMAP_V2;
        b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS;
        a_write += ((k + 1) % 2) * WARP_ROW_TENSORS_BITMAP_V2;
        b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS;
        // data loading
        if (k + 1 < BLOCK_K_TENSORS) {
            SpMM_LoadFragAwithBitmapFromShemv2(a_write, ShemAVal, SharedMemoryBitmapPTR, &start_pos, (k+1)*16);
            B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
                b_write, SharedMemoryPTR, warp_start_col, (k + 1) * MMA_K);
        }
// computations
#pragma unroll
        for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V2; i++)
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
                MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS_BITMAP_V2], a_read[i], b_read[j]);
                if (!TilingConfig::N8)
                    MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS_BITMAP_V2] + 4, a_read[i], b_read[j] + 2);  // c+4; b+2
            }
    }
}

template<typename TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegister(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C],
                                float c[][REG_PER_C_TENSOR_16_16])
{
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS);
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS);
    //
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*16 Tensor
            int RegSetID        = i + j * WARP_ROW_TENSORS;
            int Tensor_i_offset = Warp_i_offset + i * MMA_M;
            int Tensor_j_offset = Warp_j_offset + j * MMA_N;
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) {
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;
                //
                if (r % 2 > 0)
                    col_offset += 1;
                //
                if (r % 4 >= 2)
                    row_offset += 8;
                if (r >= 4)
                    col_offset += 8;
                //
                (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] = c[RegSetID][r];
            }
        }
    }
}


template<typename TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegisterBitmapV1(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C],
                                float c[][REG_PER_C_TENSOR_16_16])
{
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS_BITMAP_V1);
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS);
    //
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V1; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*16 Tensor
            int RegSetID        = i + j * WARP_ROW_TENSORS_BITMAP_V1;
            int Tensor_i_offset = Warp_i_offset + i * MMA_M;
            int Tensor_j_offset = Warp_j_offset + j * MMA_N;
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) {
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;
                //
                if (r % 2 > 0)
                    col_offset += 1;
                //
                if (r % 4 >= 2)
                    row_offset += 8;
                if (r >= 4)
                    col_offset += 8;
                //
                (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] = c[RegSetID][r];
            }
        }
    }
}

template<typename TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegisterBitmapV2(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C],
                                float c[][REG_PER_C_TENSOR_16_16])
{
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS_BITMAP_V2);
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS);
    //
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V2; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*16 Tensor
            int RegSetID        = i + j * WARP_ROW_TENSORS_BITMAP_V2;
            int Tensor_i_offset = Warp_i_offset + i * MMA_M;
            int Tensor_j_offset = Warp_j_offset + j * MMA_N;
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) {
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;
                //
                if (r % 2 > 0)
                    col_offset += 1;
                //
                if (r % 4 >= 2)
                    row_offset += 8;
                if (r >= 4)
                    col_offset += 8;
                //
                (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] = c[RegSetID][r];
            }
        }
    }
}
template<typename TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegisterBitmapV3(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C],
                                float c[][REG_PER_C_TENSOR_16_16])
{
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS_BITMAP_V3);
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS);
    //
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V3; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*16 Tensor
            int RegSetID        = i + j * WARP_ROW_TENSORS_BITMAP_V3;
            int Tensor_i_offset = Warp_i_offset + i * MMA_M;
            int Tensor_j_offset = Warp_j_offset + j * MMA_N;
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) {
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;
                //
                if (r % 2 > 0)
                    col_offset += 1;
                //
                if (r % 4 >= 2)
                    row_offset += 8;
                if (r >= 4)
                    col_offset += 8;
                //
                (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] = c[RegSetID][r];
            }
        }
    }
}

#endif