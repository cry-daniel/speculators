/***************************************************************************
 * Copyright 2025 The SpInfer Authors. All rights reserved.
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
#include "MatMulUtilities.cuh"
#include <vector>
#include <inttypes.h>
#define __STDC_FORMAT_MACROS
__device__ __forceinline__ half2 maskloadingv1(uint64_t bitmap, const half* __restrict__ startpos, int lane_id) {
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
__device__ __forceinline__ void SpMM_LoadFragAwithBitmapFromShem(uint32_t __restrict__ a[][4],
                                                         const half* __restrict__ ShemVal,
                                                         const uint64_t* __restrict__ SharedBitmap,
                                                         bool        Pred = true)
{
    int lane_id = threadIdx.x % 32;
    int start_pos = 0;
    if (Pred == true) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint64_t bitmap = SharedBitmap[i * 4 + j];
                half2 val = maskloadingv1(bitmap, ShemVal+start_pos, lane_id);
                a[i][j] = *reinterpret_cast<const uint32_t*>(&val);
                start_pos += __popcll(bitmap);  
            }
        }
    }
}


template<typename TilingConfig>
__global__ void SpMM_Kernel_bitmap_v3(const half*  A,
                            const half* Compressed_A,
                            const int*   TileOffsets,
                            const int*   TileOffsets_Median,
                            const uint64_t*   bitmap,
                            const int* ptr_max_nnz_intile,
                            const half*  B,
                            half*        Reduction_Workspace,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int          Split_K)
{
    int max_nnz_intile=*ptr_max_nnz_intile;
    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x;
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    //
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;
    const int* TileOffsets_ThisBlock = nullptr;
    const int BlockOffset = K_Global / TILE_K * y + BatchID * AverageNumKBlock;
    TileOffsets_ThisBlock = TileOffsets + BlockOffset;
    int NNZ_ThisTile = TileOffsets_ThisBlock[1] - TileOffsets_ThisBlock[0];
////////
    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned
    uint64_t* smem_Bitmap = reinterpret_cast<uint64_t*>(&smem[max_nnz_intile+(TILE_K * TilingConfig::TILE_N)*2]);
    half* smem_B = &smem[max_nnz_intile];
// Warp and lane identification.
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_Bitmap = y * TilingConfig::TILE_BITMAP_M_V3;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute a grid of C matrix tiles in each warp.
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS_BITMAP_V3 * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    uint32_t __restrict__ a[WARP_ROW_TENSORS_BITMAP_V3 * BLOCK_K_TENSORS][4];
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4];
    uint64_t* smem_BitmapWarp = smem_Bitmap + Warp_i*16;
    const int* TileOffsets_ThisWarp = nullptr;
    const int WarpOffset = BlockOffset*4 + Warp_i;
    TileOffsets_ThisWarp = TileOffsets_Median + WarpOffset;
// gld addr of copying B tile from GlobalMemory to SharedMemory
    const half* BTileGlobalPTR =
        B + Tile_Start_N * K_Global
        + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
// gld addr of  copying Bitmap tile from GlobalMemory to SharedMemory
    const uint64_t* BitmapTileGlobalPTR =
        bitmap + Tile_Start_Bitmap * K_Global
        + BatchID * AverageNumKBlock * TilingConfig::TILE_BITMAP_K_V3;  // Address for matrix bitmap, taking SplitK into consideration
// Load 1*16 bitmap to after double buffer B shared tile
    CopyTileFromGlobalToShared_Bitmap_1_64<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(smem_Bitmap, BitmapTileGlobalPTR);  // load 1*64的bitmap after double buffer B shared tile
    CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem, Compressed_A + TileOffsets_ThisBlock[0], NNZ_ThisTile);
    cp_async_group_commit();
// Load B to shared mem   
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(smem_B, BTileGlobalPTR, K_Global);  // Load B into the shared memory
    cp_async_group_commit();
    
// Initilazing C Matrix to Zeros.
    float c[WARP_ROW_TENSORS_BITMAP_V3 * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V3 * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
// Waiting for A to complete.
    cp_async_wait_group<1>();  // bitmap loading done
    __syncthreads();
// loading A to reg from shared mem with bitmap.
    SpMM_LoadFragAwithBitmapFromShem(a, smem + TileOffsets_ThisWarp[0], smem_BitmapWarp);
// waiting for B to complete
    cp_async_wait_group<0>(); // B loading done
    __syncthreads();
//Prefetch
    int StartIndex_SparseTiles_Prefetch = TileOffsets_ThisBlock[0 + 1];
    int NNZ_ThisTile_Prefetch           = TileOffsets_ThisBlock[0 + 2] - TileOffsets_ThisBlock[0 + 1];

// Go through the global K dimension by a fixed step at a time.
// write buffer[1] first, read buffer[0] first
#pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
        //
        int StartIndex_SparseTiles = StartIndex_SparseTiles_Prefetch;
        int NNZ_ThisTile          = NNZ_ThisTile_Prefetch;
        StartIndex_SparseTiles_Prefetch = TileOffsets_ThisBlock[tile_id_k + 1 + 1];
        NNZ_ThisTile_Prefetch = TileOffsets_ThisBlock[tile_id_k + 1 + 2] - TileOffsets_ThisBlock[tile_id_k + 1 + 1];
        // copying A&B tile from GlobalMemory to SharedMemory (next tile)
        BTileGlobalPTR = BTileGlobalPTR + TILE_K;
        BitmapTileGlobalPTR = BitmapTileGlobalPTR + TilingConfig::TILE_BITMAP_K_V3; 
        
        // double buffer
        half* __restrict__ smem_write_B_PTR = smem_B;
        half* __restrict__ smem_read_B_PTR  = smem_B;
        smem_write_B_PTR = smem_B + ((tile_id_k + 1) % 2) * (TILE_K * TilingConfig::TILE_N);  // The current write address of B
        smem_read_B_PTR  = smem_B + ((tile_id_k) % 2) * (TILE_K * TilingConfig::TILE_N);  // The current reading address of B

        // COPY indicator
        bool GlobalCopy = (tile_id_k + 1) < NumIter;
        
        // Copying next Bitmap Tile to write shem
        CopyTileFromGlobalToShared_Bitmap_1_64<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(smem_Bitmap, BitmapTileGlobalPTR, GlobalCopy);  // Load the 2*8 bitmap after the double buffer B shared tile
        // Copying next Sparse A Tile to write shem
        CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem, Compressed_A + StartIndex_SparseTiles, NNZ_ThisTile, GlobalCopy);
        cp_async_group_commit();

        // Copying next B Tile to write shem
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_B_PTR, BTileGlobalPTR, K_Global, GlobalCopy);
        cp_async_group_commit();

        PipelinedCoreComputationsBitmap<TilingConfig>(c, a, b, smem_read_B_PTR, warp_start_row, warp_start_col);
       
        cp_async_wait_group<1>();
        __syncthreads();
        // loading next A from shared to reg a. This can be concurrent with loading next B to shem
        SpMM_LoadFragAwithBitmapFromShem(a, smem + TileOffsets_ThisWarp[(tile_id_k+1)*4], smem_BitmapWarp, GlobalCopy);

        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
    }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Store the C fragments to shared memory.
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegisterBitmapV3<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    // Now that shared memory contains all the D tiles, stream them to global memory.
    half* BlockGlobalPTR =
        Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)  // i-th column
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)  // j-th row
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
    
}

