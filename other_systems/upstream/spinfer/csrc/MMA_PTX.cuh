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
// Extended from CUTLASS and https://github.com/AlibabaResearch/flash-llm/blob/main/csrc/MMA_PTX.cuh
#include "TilingConfig.h"

template<int NumOfTensors>
__device__ __forceinline__ void FragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][4],
                                                              half* __restrict__ smem,
                                                              int warp_start_row,
                                                              int k_offset)
{
    //
    int lane_id = threadIdx.x % 32;
    int i       = lane_id % MMA_M;
    int j       = lane_id / MMA_M;
    //
    smem += TILE_K * (warp_start_row + i) + (k_offset + j * HALF_PER_128B);
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    // Row Permutation to eliminating bank-conflict
    uint32_t RowLane_RowPermutation = i % COPY_UNIT_FP16_ROWS;
    uint32_t Mask_RowPermutation    = RowLane_RowPermutation << 4;
    smem_local_ptr                  = smem_local_ptr ^ Mask_RowPermutation;
//
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));

        smem_local_ptr += TILE_K * MMA_M * sizeof(half);
    }
}
template<int NumOfTensors>  // 4
__device__ __forceinline__ void A_FragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][4],
                                                              half* __restrict__ smem,
                                                              int warp_start_row,
                                                              int k_offset)
{
    //
    int lane_id = threadIdx.x % 32;
    int i       = lane_id % MMA_M;
    int j       = lane_id / MMA_M;
    //
    smem += TILE_K_HALF * (warp_start_row + i) + (k_offset + j * HALF_PER_128B);
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    // Row Permutation to eliminating bank-conflict
    uint32_t RowLane_RowPermutation = (i % COPY_UNIT_FP16_ROWS) / 2;
    uint32_t Mask_RowPermutation    = RowLane_RowPermutation << 4;
    smem_local_ptr                  = smem_local_ptr ^ Mask_RowPermutation;
//
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));

        smem_local_ptr += TILE_K_HALF * MMA_M * sizeof(half);
    }
}


template<int NumOfTensors>  // 4
__device__ __forceinline__ void E_FragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[],
                                                              uint32_t* __restrict__ smem_MetaE,
                                                              int warp_start_row)
{
    //
    int lane_id = threadIdx.x % 32;
    //
    smem_MetaE += (TILE_K_HALF * warp_start_row) + (lane_id / 4) * 2 + (((lane_id % 4) / 2) << 4) + (lane_id % 2);
    //
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        Registers[i] = *smem_MetaE;
        smem_MetaE += TILE_K_HALF;
    }
}

template<int NumOfTensors, int N8>
__device__ __forceinline__ void B_FragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][4],
                                                                half* __restrict__ smem,
                                                                int warp_start_row,
                                                                int k_offset)
{
    //
    int      lane_id             = threadIdx.x % 32;
    int      i                   = lane_id % 8;
    uint32_t Mask_RowPermutation = i << 4;

    if (lane_id > 15)
        i += 8;
    int j = (lane_id % 16) >= 8 ? 1 : 0;
    //
    smem += TILE_K * (warp_start_row + i) + (k_offset + j * HALF_PER_128B);
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    // Row Permutation to eliminating bank-conflict

    smem_local_ptr = smem_local_ptr ^ Mask_RowPermutation;
//
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));

        smem_local_ptr += TILE_K * MMA_N * sizeof(half);
    }
}
__device__ __forceinline__ void
MMA_FP16_M16N8K16(uint32_t __restrict__ c[], uint32_t __restrict__* a, uint32_t __restrict__* b)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                 "{ %0, %1, %2, %3},"
                 "{ %4, %5, %6, %7 },"
                 "{ %8, %9 },"
                 "{ %10, %11, %12, %13 };"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]),
                   "r"(a[1]),
                   "r"(a[2]),
                   "r"(a[3]),
                   "r"(b[0]),
                   "r"(b[1]),  /////////////// for column-major B
                   "r"(c[0]),
                   "r"(c[1]),
                   "r"(c[2]),
                   "r"(c[3]));
}
__device__ __forceinline__ void
MMA_SP_FP16_M16N8K32(uint32_t __restrict__ c[], uint32_t __restrict__* a, uint32_t __restrict__* b1, uint32_t __restrict__* b2, uint32_t __restrict__ e)
{
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"
                 "{ %0, %1, %2, %3},"
                 "{ %4, %5, %6, %7 },"
                 "{ %8, %9, %10, %11},"
                 "{ %12, %13, %14, %15}, "
                 "%16, "
                 "0x0; "
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]),
                   "r"(a[1]),
                   "r"(a[2]),
                   "r"(a[3]),
                   "r"(b1[0]),
                   "r"(b1[1]), 
                   "r"(b2[0]),
                   "r"(b2[1]),  /////////////// for column-major B
                   "r"(c[0]),
                   "r"(c[1]),
                   "r"(c[2]),
                   "r"(c[3]),
                   "r"(e));
}
__device__ __forceinline__ void
MMA_SP_FP16_M16N8K32_1(uint32_t __restrict__ c[], uint32_t __restrict__* a, uint32_t __restrict__* b1, uint32_t __restrict__* b2, uint32_t __restrict__ e)
{
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"
                 "{ %0, %1, %2, %3},"
                 "{ %4, %5, %6, %7 },"
                 "{ %8, %9, %10, %11},"
                 "{ %12, %13, %14, %15}, "
                 "%16, "
                 "0x1; "
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]),
                   "r"(a[1]),
                   "r"(a[2]),
                   "r"(a[3]),
                   "r"(b1[0]),
                   "r"(b1[1]), 
                   "r"(b2[0]),
                   "r"(b2[1]),  /////////////// for column-major B
                   "r"(c[0]),
                   "r"(c[1]),
                   "r"(c[2]),
                   "r"(c[3]),
                   "r"(e));
}