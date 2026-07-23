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
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
cudaError_t SpMM_SplitK_API_bitmap_v3(cudaStream_t stream,
                                    const half*  A,
                                    const half*  Compressed_A,
                                    const int*   TileOffsets,
                                    const int* TileOffsets_Median,
                                    const uint64_t* bitmap,
                                    const int* max_nnz_intile,
                                    const half*  B,
                                    half*        C,
                                    const int    M_Global,
                                    const int    N_Global,
                                    const int    K_Global,
                                    half*        Reduction_Workspace,
                                    int          Split_K);
// Our sparsity_llm
__host__ int InitSparseMatrixA_bitmap(
                                        half* A_h,
                                        int M,  // 行数
                                        int K,  // 列数
                                        int tile_M,  // 8
                                        int tile_M_median,  // 16
                                        int tile_M_global,  // 64
                                        int tile_K,  // 8
                                        int tile_K_median,  // 64
                                        int tile_K_global,  // 64
                                        half** Compressed_Val,
                                        int** TileOffsets,
                                        int** TileOffsets_median,
                                        int** TileOffsets_global,
                                        uint64_t** bitmap,
                                        int& max_nnz_count);

// Used by ft-tools
// Our sparsity_llm
extern "C" void Our_GenSparseMatrixBinFile(char* DenseMatrixFileName,
                                            int   M,
                                            int   K,
                                            char* Compressed_ValFileName,
                                            char* bitmap_TileOffsets_globalFileName,
                                            char* bitmap_TileOffsets_medianFileName,
                                            char* bitmapFileName,
                                            char* max_nnz_intileFileName,
                                            char* OutputSizesFileName);