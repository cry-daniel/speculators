"""Focused structured-2:4 GEMM prototypes used by SpecLink experiments."""

from .online_sparse24 import (
    OnlineSparse24Weight,
    SPARSE_RESIDUAL_SMEM,
    pack_sparse24_components,
    prepare_online_sparse24_weight,
)
from .sparse24_gemm import (
    PreparedSparse24Weight,
    TokenRoute,
    assert_24_weight,
    prepare_sparse24_weight,
    route_from_indices,
    route_from_mask,
    select_cusparselt_algorithm,
    sparse24_linear,
)
from .cusparselt_sparse_residual import (
    COMPLEMENT_CTA_VARIANTS,
    COMPLEMENT_SPLITK2_VARIANTS,
    CUSPARSELT_DIRECT_METADATA_LAYOUT,
    CUSPARSELT_SPARSE_RESIDUAL,
    FUSED_BASE_COMPLEMENT_VARIANTS,
    CusparseLtSparseResidualWeight,
    cusparselt_sparse_residual_fused_dense_linear,
    cusparselt_sparse_residual_fused_kernel_attributes,
    cusparselt_sparse_residual_indexed_add_,
    cusparselt_sparse_residual_indexed_copy_,
    cusparselt_sparse_residual_indexed_gather,
    cusparselt_sparse_residual_residual_linear_splitk2,
    cusparselt_sparse_residual_residual_linear_splitk2_indexed,
    cusparselt_sparse_residual_residual_linear_splitk2_chunked,
    cusparselt_sparse_residual_residual_linear_splitk2_persistent,
    cusparselt_sparse_residual_residual_linear_splitk4,
    cusparselt_sparse_residual_splitk2_indexed_add_,
    cusparselt_sparse_residual_splitk4_indexed_add_,
    cusparselt_sparse_residual_kernel_attributes,
    cusparselt_sparse_residual_residual_linear,
    cusparselt_sparse_residual_sparse_linear,
    prepare_cusparselt_sparse_residual_weight,
    tune_cusparselt_sparse_residual_algorithm,
    validate_cusparselt_packed_layout,
)
from .shapes import TP1_FUSED_WEIGHT_SHAPES

from .old_concurrent_cutlass_sparse24 import (
    OLD_CONCURRENT_DENSE_BRANCH,
    OLD_CONCURRENT_SPARSE_BRANCH,
    OldConcurrentWeight,
    old_concurrent_branch_linear_out,
    old_concurrent_kernel_attributes,
    prepare_old_concurrent_weight,
)


__all__ = [name for name in globals() if not name.startswith("_")]
