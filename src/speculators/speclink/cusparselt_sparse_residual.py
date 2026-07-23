"""One-weight cuSPARSELt 2:4 base plus compact residual primitives.

The persistent representation contains exactly two allocations:

* the cuSPARSELt packed allocation (retained BF16 values followed by the
  library's sole metadata payload), and
* the two complement BF16 residual values for every K4 group.

Dense reconstruction and residual-only correction decode the cuSPARSELt 0.8
metadata in place.  They never retain a canonical metadata copy or materialize
a global dense weight.  Preparation validates the physical metadata swizzle
against the existing CUTLASS encoder and then releases all setup-only tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .online_sparse24 import OnlineSparse24Weight
from .sparse24_gemm import (
    PreparedSparse24Weight,
    prepare_sparse24_weight,
    select_cusparselt_algorithm,
    sparse24_linear,
)
from .sparse24_cutlass_metadata import reorder_cutlass_sparse24_metadata


CUSPARSELT_SPARSE_RESIDUAL = "cusparselt_sparse_residual"
CUSPARSELT_DIRECT_METADATA_LAYOUT = (
    "cusparselt_0_8_cutlass_e_256word_macro_swizzle"
)
FUSED_BASE_COMPLEMENT_VARIANTS = {
    "n64_s3": 0,
    "n32_s4": 1,
    "n128_s3": 2,
    "f64_n128_s3": 3,
    "f64_n128_s4": 4,
}
COMPLEMENT_CTA_VARIANTS = {
    "auto": -1,
    "feature128_token64_s4": 0,
    "feature64_token64_s4": 1,
    "feature128_token32_s4": 2,
    "feature64_token32_s4": 3,
    "feature64_token64_s2": 4,
    "feature64_token32_s2": 5,
    "feature128_token64_single_smem": 6,
    "feature256_token64_s3": 7,
    "activation_stationary_token32": 8,
    "b_resident_feature128_token32": 11,
    "b_resident_feature64_token32": 12,
    "b_resident_feature128_token32_a2": 13,
    "b_resident_feature64_token32_a2": 14,
    "b_resident_feature64_token64_b2a1_p40": 15,
    "feature64_token64_s4_p40": 16,
    "b_resident_feature128_token64_b2a1": 17,
    "b_resident_feature128_token64_b2a1_p192": 18,
    "b_resident_feature128_token64_b2a1_p224": 19,
}
COMPLEMENT_SPLITK2_VARIANTS = {
    "b_resident_feature64_token64_b2a1": 0,
    "feature64_token64_s4": 1,
    "feature128_token64_s4": 2,
    "b_resident_feature64_token64_b2a1_p40": 3,
    # gate_up has a long output-feature dimension and enough feature panels
    # to keep the GPU full with a 128-row CTA.  Retaining the low-SMEM
    # B-resident schedule while doubling FeatureM halves duplicated activation
    # and metadata traffic relative to the production F64 kernel.
    "b_resident_feature128_token64_b2a1": 4,
    "b_resident_feature128_token64_b2a1_p192": 5,
    "b_resident_feature128_token64_b2a1_p224": 6,
    # D=128 gate_up candidates: one CTA spans the complete dense-token set,
    # eliminating the duplicate residual/metadata stream of two Token64 CTAs.
    "b_resident_feature128_token128_b2a1": 7,
    "b_resident_feature128_token128_b2a1_p192": 8,
    "b_resident_feature128_token128_b2a1_p224": 9,
    "b_resident_feature256_token128_b2a1": 10,
    "b_resident_feature256_token128_b2a1_p192": 11,
    "b_resident_feature256_token128_b2a1_p224": 12,
    "b_resident_feature128_token128_b2a2": 13,
    "b_resident_feature128_token128_b2a2_p192": 14,
    "b_resident_feature128_token128_b2a2_p224": 15,
    "b_resident_feature128_token128_b3a2": 16,
    "b_resident_feature128_token128_b3a2_p192": 17,
    "b_resident_feature128_token128_b3a2_p224": 18,
}


def _storage_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().nbytes())


def _metadata_block_order(n: int, block_count: int) -> torch.Tensor:
    """Map cuSPARSELt physical 256-word blocks to CUTLASS logical blocks."""

    if not isinstance(n, int) or n <= 0 or n % 128:
        raise ValueError("direct cuSPARSELt metadata decoding requires N % 128 == 0")
    if not isinstance(block_count, int) or block_count <= 0:
        raise ValueError("metadata block count must be positive")
    panels = n // 128
    group_blocks = 2 * panels
    if block_count % group_blocks:
        raise ValueError(
            "cuSPARSELt metadata word count is not a complete macro-swizzle group"
        )
    physical = torch.arange(block_count, dtype=torch.int64)
    group_base = physical.div(group_blocks, rounding_mode="floor") * group_blocks
    within = physical.remainder(group_blocks)
    return (
        group_base
        + within.bitwise_and(1) * panels
        + within.bitwise_right_shift(1)
    ).contiguous()


def validate_cusparselt_packed_layout(
    canonical: OnlineSparse24Weight,
    cusparselt: PreparedSparse24Weight,
) -> dict[str, Any]:
    """Fail closed unless values and metadata match the decoded 0.8 layout.

    The reordered CUTLASS metadata and comparison tensor are setup-only.  They
    are not attached to the returned runtime object.
    """

    if not isinstance(canonical, OnlineSparse24Weight):
        raise TypeError("canonical must be an OnlineSparse24Weight")
    if canonical.storage_family != "sparse_residual":
        raise ValueError("canonical weight must use sparse_residual storage")
    if canonical.packed_values is None or canonical.residual is None:
        raise ValueError("canonical sparse-residual values are incomplete")
    if not isinstance(cusparselt, PreparedSparse24Weight):
        raise TypeError("cusparselt must be a PreparedSparse24Weight")

    n, k = canonical.shape
    if cusparselt.shape != canonical.shape:
        raise ValueError("cuSPARSELt and canonical logical shapes differ")
    if n % 128 or k % 128:
        raise ValueError(
            "direct cuSPARSELt metadata decoding requires N and K multiples of 128"
        )
    packed = cusparselt.packed
    expected_value_numel = n * k // 2
    expected_metadata_words = n * k // 16
    expected_total_numel = expected_value_numel + expected_metadata_words
    if (
        packed.dtype != torch.bfloat16
        or packed.ndim != 2
        or packed.shape[0] != n
        or not packed.is_cuda
        or not packed.is_contiguous()
        or packed.numel() != expected_total_numel
    ):
        raise RuntimeError(
            "cuSPARSELt packed allocation does not match values+metadata contract: "
            f"shape={tuple(packed.shape)}, numel={packed.numel()}, "
            f"expected_numel={expected_total_numel}"
        )

    flat = packed.flatten()
    values = flat[:expected_value_numel].view(n, k // 2)
    if not torch.equal(values, canonical.packed_values):
        mismatch = values.ne(canonical.packed_values).nonzero(as_tuple=False)[0]
        raise RuntimeError(
            "cuSPARSELt retained-value prefix changed at "
            f"row={int(mismatch[0])}, compact_k={int(mismatch[1])}"
        )

    reordered = reorder_cutlass_sparse24_metadata(
        canonical.metadata, canonical.shape
    ).flatten()
    if reordered.numel() != expected_metadata_words:
        raise RuntimeError("CUTLASS setup metadata has an unexpected size")
    if expected_metadata_words % 256:
        raise RuntimeError("metadata payload is not 256-word aligned")
    order = _metadata_block_order(n, expected_metadata_words // 256).to(
        device=packed.device
    )
    expected_physical = reordered.view(-1, 256).index_select(0, order).flatten()
    physical = flat[expected_value_numel:].view(torch.int16)
    mismatch = physical.ne(expected_physical).nonzero(as_tuple=False)
    if mismatch.numel():
        word = int(mismatch[0].item())
        raise RuntimeError(
            "cuSPARSELt metadata layout validation failed at physical word "
            f"{word}: got=0x{int(physical[word].item()) & 0xFFFF:04x}, "
            f"expected=0x{int(expected_physical[word].item()) & 0xFFFF:04x}"
        )

    return {
        "layout": CUSPARSELT_DIRECT_METADATA_LAYOUT,
        "logical_shape": [n, k],
        "value_prefix_exact": True,
        "metadata_exact": True,
        "value_bytes": n * k,
        "metadata_bytes": n * k // 8,
        "metadata_words": expected_metadata_words,
        "metadata_block_words": 256,
        "n_panels_128": n // 128,
    }


@dataclass(slots=True)
class CusparseLtSparseResidualWeight:
    """Runtime state with no dense weight and no duplicate metadata."""

    cusparselt: PreparedSparse24Weight
    residual: torch.Tensor
    layout_validation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.cusparselt, PreparedSparse24Weight):
            raise TypeError("cusparselt must be a PreparedSparse24Weight")
        n, k = self.cusparselt.shape
        if n % 128 or k % 128:
            raise ValueError("N and K must be multiples of 128")
        if (
            not isinstance(self.residual, torch.Tensor)
            or self.residual.dtype != torch.bfloat16
            or self.residual.ndim != 2
            or tuple(self.residual.shape) != (n, k // 2)
            or not self.residual.is_cuda
            or not self.residual.is_contiguous()
            or self.residual.device != self.cusparselt.device
        ):
            raise ValueError(
                "residual must be contiguous CUDA BF16 with shape [N,K/2]"
            )
        packed = self.cusparselt.packed
        if packed.numel() != 9 * n * k // 16:
            raise ValueError("cuSPARSELt packed storage has unexpected padding")
        sparse_tensor = self.cusparselt.sparse_weight
        if getattr(sparse_tensor, "meta", None) is not None:
            raise RuntimeError("cuSPARSELt runtime unexpectedly retained separate metadata")
        if getattr(sparse_tensor, "packed_t", None) is not None:
            raise RuntimeError("cuSPARSELt runtime unexpectedly retained a transpose copy")
        if getattr(sparse_tensor, "meta_t", None) is not None:
            raise RuntimeError(
                "cuSPARSELt runtime unexpectedly retained transpose metadata"
            )
        if getattr(sparse_tensor, "compressed_swizzled_bitmask", None) is not None:
            raise RuntimeError(
                "cuSPARSELt runtime unexpectedly retained a pruning bitmask"
            )
        if self.layout_validation.get("layout") != CUSPARSELT_DIRECT_METADATA_LAYOUT:
            raise RuntimeError("cuSPARSELt physical layout was not validated")

    @property
    def shape(self) -> tuple[int, int]:
        return self.cusparselt.shape

    @property
    def n(self) -> int:
        return self.shape[0]

    @property
    def k(self) -> int:
        return self.shape[1]

    @property
    def device(self) -> torch.device:
        return self.cusparselt.device

    @property
    def packed(self) -> torch.Tensor:
        return self.cusparselt.packed

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "cusparselt_packed_values_and_metadata": self.packed,
            "residual": self.residual,
        }

    def persistent_bytes_by_component(self) -> dict[str, int]:
        return {
            name: _storage_nbytes(tensor)
            for name, tensor in self.persistent_tensors().items()
        }

    def persistent_bytes(self) -> int:
        total = 0
        seen: set[tuple[str, int]] = set()
        for tensor in self.persistent_tensors().values():
            storage = tensor.untyped_storage()
            key = (str(tensor.device), int(storage.data_ptr()))
            if key not in seen:
                total += int(storage.nbytes())
                seen.add(key)
        return total


def prepare_cusparselt_sparse_residual_weight(
    canonical: OnlineSparse24Weight,
    *,
    algorithm_id: int = 0,
    sparse_weight: torch.Tensor | None = None,
) -> CusparseLtSparseResidualWeight:
    """Compress the sparse base and retain only packed cuSPARSELt+residual."""

    if not isinstance(canonical, OnlineSparse24Weight):
        raise TypeError("canonical must be an OnlineSparse24Weight")
    if canonical.storage_family != "sparse_residual":
        raise ValueError("preparation requires sparse_residual canonical storage")
    if canonical.residual is None:
        raise ValueError("canonical residual is missing")
    sparse = (
        canonical.reconstruct_sparse()
        if sparse_weight is None
        else sparse_weight
    )
    if (
        not isinstance(sparse, torch.Tensor)
        or sparse.dtype != torch.bfloat16
        or tuple(sparse.shape) != canonical.shape
        or sparse.device != canonical.device
        or not sparse.is_cuda
        or not sparse.is_contiguous()
    ):
        raise ValueError(
            "sparse_weight must be contiguous CUDA BF16 with the canonical shape"
        )
    cusparselt = prepare_sparse24_weight(sparse, algorithm_id=algorithm_id)
    validation = validate_cusparselt_packed_layout(canonical, cusparselt)
    return CusparseLtSparseResidualWeight(
        cusparselt=cusparselt,
        residual=canonical.residual,
        layout_validation=validation,
    )


def tune_cusparselt_sparse_residual_algorithm(
    prepared: CusparseLtSparseResidualWeight,
    sample_input: torch.Tensor,
) -> int:
    if not isinstance(prepared, CusparseLtSparseResidualWeight):
        raise TypeError("prepared must be a CusparseLtSparseResidualWeight")
    return select_cusparselt_algorithm(prepared.cusparselt, sample_input)


def _validate_input(
    x: torch.Tensor, prepared: CusparseLtSparseResidualWeight
) -> None:
    if not isinstance(prepared, CusparseLtSparseResidualWeight):
        raise TypeError("prepared must be a CusparseLtSparseResidualWeight")
    if (
        not isinstance(x, torch.Tensor)
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.shape[0] <= 0
        or not x.is_cuda
        or not x.is_contiguous()
        or x.device != prepared.device
        or x.shape[1] != prepared.k
    ):
        raise ValueError("x must be contiguous CUDA BF16 [M,K] on the weight device")


def cusparselt_sparse_residual_sparse_linear(
    x: torch.Tensor,
    prepared: CusparseLtSparseResidualWeight,
) -> torch.Tensor:
    _validate_input(x, prepared)
    return sparse24_linear(x, prepared.cusparselt)


def _dense_extension() -> Any:
    # Reuse the existing CUTLASS dense mainloop build.  The extension has a
    # dedicated ABI that receives only the cuSPARSELt packed tensor+residual.
    from .sparse_residual_cutlass_sparse24 import _extension

    return _extension()


def cusparselt_sparse_residual_residual_linear(
    x: torch.Tensor,
    prepared: CusparseLtSparseResidualWeight,
    *,
    variant: str = "auto",
) -> torch.Tensor:
    """Compute the complement residual directly with HMMA.SP.

    The compact residual already contains exactly two values per K4.  The
    kernel reads the sole cuSPARSELt metadata payload, complements its selector
    in registers, and issues sparse tensor-core instructions.  It neither
    reconstructs a dense tile nor stores another metadata tensor.
    """

    _validate_input(x, prepared)
    try:
        variant_id = COMPLEMENT_CTA_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(
            f"unknown complement CTA variant {variant!r}; expected one of "
            f"{tuple(COMPLEMENT_CTA_VARIANTS)}"
        ) from error
    output = _dense_extension().cusparselt_complement_sparse_forward(
        x, prepared.packed, prepared.residual, variant_id
    )
    if (
        output.dtype != torch.bfloat16
        or tuple(output.shape) != (x.shape[0], prepared.n)
        or output.device != x.device
        or not output.is_contiguous()
    ):
        raise RuntimeError("complement-metadata HMMA.SP kernel returned invalid output")
    return output


def _auto_splitk2_variant(n: int, dense_rows: int) -> str:
    """Select the measured F128 schedule only once D supplies four row tiles.

    The F128 kernels win for the paper M=2048, D=256 cases by reducing the
    number of feature panels.  At serving M=1024, D=128 the same schedule is
    slightly slower, so output width alone is not a sufficient dispatch key.
    """

    if dense_rows < 256:
        return (
            "b_resident_feature64_token64_b2a1_p40"
            if n == 5120
            else "b_resident_feature64_token64_b2a1"
        )
    return {
        4096: "b_resident_feature128_token64_b2a1",
        6144: "b_resident_feature128_token64_b2a1",
        5120: "b_resident_feature64_token64_b2a1_p40",
        24576: "b_resident_feature128_token64_b2a1_p192",
        28672: "b_resident_feature128_token64_b2a1_p224",
    }.get(n, "b_resident_feature64_token64_b2a1")


def cusparselt_sparse_residual_residual_linear_splitk2(
    x: torch.Tensor,
    prepared: CusparseLtSparseResidualWeight,
    *,
    variant: str = "auto",
) -> torch.Tensor:
    """Return BF16 K-partials for a selected complement mainloop."""

    _validate_input(x, prepared)
    if variant == "auto":
        variant = _auto_splitk2_variant(prepared.n, int(x.shape[0]))
    try:
        variant_id = COMPLEMENT_SPLITK2_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(
            f"unknown Split-K=2 variant {variant!r}; expected auto or one of "
            f"{tuple(COMPLEMENT_SPLITK2_VARIANTS)}"
        ) from error
    partials = _dense_extension().cusparselt_complement_sparse_splitk2_forward(
        x, prepared.packed, prepared.residual, variant_id
    )
    if (
        partials.dtype != torch.bfloat16
        or tuple(partials.shape) != (2, x.shape[0], prepared.n)
        or partials.device != x.device
        or not partials.is_contiguous()
    ):
        raise RuntimeError("Split-K=2 complement returned invalid partials")
    return partials


def cusparselt_sparse_residual_residual_linear_splitk4(
    x: torch.Tensor,
    prepared: CusparseLtSparseResidualWeight,
    *,
    variant: str,
) -> torch.Tensor:
    """Return four BF16 K-partials for a Token128 gate_up complement."""

    _validate_input(x, prepared)
    try:
        variant_id = COMPLEMENT_SPLITK2_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(f"unknown Split-K=4 variant {variant!r}") from error
    if variant_id not in (7, 8, 9, 10, 11, 12):
        raise ValueError("Split-K=4 requires a Token128 gate_up variant")
    partials = _dense_extension().cusparselt_complement_sparse_splitk4_forward(
        x, prepared.packed, prepared.residual, variant_id
    )
    expected = (4, x.shape[0], prepared.n)
    if (
        partials.dtype != torch.bfloat16
        or tuple(partials.shape) != expected
        or partials.device != x.device
        or not partials.is_contiguous()
    ):
        raise RuntimeError("Split-K=4 complement returned invalid partials")
    return partials


def cusparselt_sparse_residual_residual_linear_splitk2_persistent(
    x: torch.Tensor,
    prepared: CusparseLtSparseResidualWeight,
    *,
    variant: str,
    persistent_m_blocks: int,
) -> torch.Tensor:
    """Run Split-K2 with a quota of persistent feature-panel CTAs."""

    _validate_input(x, prepared)
    try:
        variant_id = COMPLEMENT_SPLITK2_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(f"unknown persistent Split-K2 variant {variant!r}") from error
    if variant_id not in (7, 8, 9):
        raise ValueError("persistent Split-K2 requires Feature128/Token128")
    if not isinstance(persistent_m_blocks, int) or persistent_m_blocks <= 0:
        raise ValueError("persistent_m_blocks must be positive")
    partials = (
        _dense_extension()
        .cusparselt_complement_sparse_splitk2_persistent_forward(
            x,
            prepared.packed,
            prepared.residual,
            variant_id,
            persistent_m_blocks,
        )
    )
    expected = (2, x.shape[0], prepared.n)
    if (
        partials.dtype != torch.bfloat16
        or tuple(partials.shape) != expected
        or partials.device != x.device
        or not partials.is_contiguous()
    ):
        raise RuntimeError("persistent Split-K2 returned invalid partials")
    return partials


def cusparselt_sparse_residual_residual_linear_splitk2_chunked(
    x: torch.Tensor,
    prepared: CusparseLtSparseResidualWeight,
    *,
    variant: str,
    chunk_m_blocks: int,
) -> torch.Tensor:
    """Run Split-K2 as short feature-panel grids to bound SM admission."""

    _validate_input(x, prepared)
    try:
        variant_id = COMPLEMENT_SPLITK2_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(f"unknown chunked Split-K2 variant {variant!r}") from error
    if variant_id not in (7, 8, 9):
        raise ValueError("chunked Split-K2 requires Feature128/Token128")
    if not isinstance(chunk_m_blocks, int) or chunk_m_blocks <= 0:
        raise ValueError("chunk_m_blocks must be positive")
    partials = (
        _dense_extension()
        .cusparselt_complement_sparse_splitk2_chunked_forward(
            x,
            prepared.packed,
            prepared.residual,
            variant_id,
            chunk_m_blocks,
        )
    )
    expected = (2, x.shape[0], prepared.n)
    if (
        partials.dtype != torch.bfloat16
        or tuple(partials.shape) != expected
        or partials.device != x.device
        or not partials.is_contiguous()
    ):
        raise RuntimeError("chunked Split-K2 returned invalid partials")
    return partials


def cusparselt_sparse_residual_residual_linear_splitk2_indexed(
    x: torch.Tensor,
    indices: torch.Tensor,
    prepared: CusparseLtSparseResidualWeight,
    *,
    variant: str = "auto",
) -> torch.Tensor:
    """Return complement partials while gathering B inside the mainloop."""

    _validate_input(x, prepared)
    if indices.dtype != torch.int32 or indices.ndim != 1:
        raise ValueError("indexed Split-K=2 requires contiguous int32 indices")
    if not indices.is_cuda or indices.device != x.device or not indices.is_contiguous():
        raise ValueError("indices must be contiguous on the input CUDA device")
    if indices.numel() == 0:
        raise ValueError("indices must be non-empty")
    if variant == "auto":
        variant = _auto_splitk2_variant(prepared.n, int(indices.numel()))
    try:
        variant_id = COMPLEMENT_SPLITK2_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(
            f"unknown Split-K=2 variant {variant!r}; expected auto or one of "
            f"{tuple(COMPLEMENT_SPLITK2_VARIANTS)}"
        ) from error
    if variant_id not in (0, 3, 4, 5, 6):
        raise ValueError(
            "indexed Split-K=2 currently supports the production B-resident "
            "variants only"
        )
    partials = (
        _dense_extension()
        .cusparselt_complement_sparse_splitk2_indexed_forward(
            x, indices, prepared.packed, prepared.residual, variant_id
        )
    )
    expected = (2, indices.numel(), prepared.n)
    if (
        partials.dtype != torch.bfloat16
        or tuple(partials.shape) != expected
        or partials.device != x.device
        or not partials.is_contiguous()
    ):
        raise RuntimeError("indexed Split-K=2 complement returned invalid partials")
    return partials


def cusparselt_sparse_residual_fused_dense_linear(
    x: torch.Tensor,
    prepared: CusparseLtSparseResidualWeight,
    *,
    variant: str = "n64_s3",
) -> torch.Tensor:
    """Compute exact dense rows as base-2:4 + complement-2:4 in one kernel.

    The two compact A operands have separate shared-memory stage buffers.  The
    activation B tile and sole cuSPARSELt metadata E tile are loaded once per
    stage and reused by the normal and register-complemented HMMA.SP calls.
    Both products update one FP32 accumulator and use one epilogue/output write.
    """

    _validate_input(x, prepared)
    try:
        variant_id = FUSED_BASE_COMPLEMENT_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(
            f"unknown fused variant {variant!r}; expected one of "
            f"{tuple(FUSED_BASE_COMPLEMENT_VARIANTS)}"
        ) from error
    output = _dense_extension().cusparselt_fused_base_complement_forward(
        x, prepared.packed, prepared.residual, variant_id
    )
    if (
        output.dtype != torch.bfloat16
        or tuple(output.shape) != (x.shape[0], prepared.n)
        or output.device != x.device
        or not output.is_contiguous()
    ):
        raise RuntimeError("fused base+complement HMMA.SP kernel returned invalid output")
    return output


def cusparselt_sparse_residual_indexed_add_(
    base: torch.Tensor,
    correction: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Add compact BF16 correction rows into ``base`` with one CUDA kernel."""

    if (
        not isinstance(base, torch.Tensor)
        or base.dtype != torch.bfloat16
        or base.ndim != 2
        or not base.is_cuda
        or not base.is_contiguous()
        or base.shape[1] % 8
    ):
        raise ValueError(
            "base must be contiguous CUDA BF16 [M,N] with N divisible by 8"
        )
    if (
        not isinstance(correction, torch.Tensor)
        or correction.dtype != torch.bfloat16
        or correction.ndim != 2
        or not correction.is_cuda
        or not correction.is_contiguous()
        or correction.device != base.device
    ):
        raise ValueError("correction must be contiguous CUDA BF16 [D,N]")
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.ndim != 1
        or not indices.is_cuda
        or not indices.is_contiguous()
        or indices.device != base.device
    ):
        raise ValueError("indices must be contiguous CUDA int32/int64 [D]")
    if correction.shape != (indices.numel(), base.shape[1]):
        raise ValueError("correction must have shape [indices.numel(), N]")
    return _dense_extension().cusparselt_indexed_add_inplace(
        base, correction, indices
    )


def cusparselt_sparse_residual_indexed_copy_(
    destination: torch.Tensor,
    source: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Copy compact BF16 rows into indexed rows with 128-bit transactions."""

    if (
        not isinstance(destination, torch.Tensor)
        or destination.dtype != torch.bfloat16
        or destination.ndim != 2
        or not destination.is_cuda
        or not destination.is_contiguous()
        or destination.shape[1] % 8
    ):
        raise ValueError("destination must be contiguous CUDA BF16 [M,N]")
    if (
        not isinstance(source, torch.Tensor)
        or source.dtype != torch.bfloat16
        or source.ndim != 2
        or not source.is_cuda
        or not source.is_contiguous()
        or source.device != destination.device
    ):
        raise ValueError("source must be contiguous CUDA BF16 [D,N]")
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.ndim != 1
        or not indices.is_cuda
        or not indices.is_contiguous()
        or indices.device != destination.device
        or indices.numel() == 0
    ):
        raise ValueError("indices must be non-empty contiguous CUDA int32/int64")
    if source.shape != (indices.numel(), destination.shape[1]):
        raise ValueError("source must have shape [indices.numel(), N]")
    return _dense_extension().cusparselt_indexed_copy_inplace(
        destination, source, indices
    )


def cusparselt_sparse_residual_splitk2_indexed_add_(
    base: torch.Tensor,
    partials: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Reduce two compact K-partials and add them into indexed base rows."""

    if (
        not isinstance(base, torch.Tensor)
        or base.dtype != torch.bfloat16
        or base.ndim != 2
        or not base.is_cuda
        or not base.is_contiguous()
        or base.shape[1] % 8
    ):
        raise ValueError("base must be contiguous CUDA BF16 [M,N]")
    if (
        not isinstance(partials, torch.Tensor)
        or partials.dtype != torch.bfloat16
        or partials.ndim != 3
        or partials.shape[0] != 2
        or not partials.is_cuda
        or not partials.is_contiguous()
        or partials.device != base.device
    ):
        raise ValueError("partials must be contiguous CUDA BF16 [2,D,N]")
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.ndim != 1
        or not indices.is_cuda
        or not indices.is_contiguous()
        or indices.device != base.device
        or indices.numel() == 0
    ):
        raise ValueError(
            "indices must be non-empty contiguous CUDA int32/int64 [D]"
        )
    if partials.shape[1:] != (indices.numel(), base.shape[1]):
        raise ValueError("partials must have shape [2, indices.numel(), N]")
    return _dense_extension().cusparselt_splitk2_indexed_add_inplace(
        base, partials, indices
    )


def cusparselt_sparse_residual_splitk4_indexed_add_(
    base: torch.Tensor,
    partials: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Reduce four compact K-partials and add them into indexed base rows."""

    if (
        not isinstance(base, torch.Tensor)
        or base.dtype != torch.bfloat16
        or base.ndim != 2
        or not base.is_cuda
        or not base.is_contiguous()
        or base.shape[1] % 8
    ):
        raise ValueError("base must be contiguous CUDA BF16 [M,N]")
    if (
        not isinstance(partials, torch.Tensor)
        or partials.dtype != torch.bfloat16
        or partials.ndim != 3
        or partials.shape[0] != 4
        or not partials.is_cuda
        or not partials.is_contiguous()
        or partials.device != base.device
    ):
        raise ValueError("partials must be contiguous CUDA BF16 [4,D,N]")
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.ndim != 1
        or not indices.is_cuda
        or not indices.is_contiguous()
        or indices.device != base.device
        or indices.numel() == 0
    ):
        raise ValueError(
            "indices must be non-empty contiguous CUDA int32/int64 [D]"
        )
    if partials.shape[1:] != (indices.numel(), base.shape[1]):
        raise ValueError("partials must have shape [4, indices.numel(), N]")
    return _dense_extension().cusparselt_splitk4_indexed_add_inplace(
        base, partials, indices
    )


def cusparselt_sparse_residual_indexed_gather(
    source: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather BF16 activation rows with a shape-specialized CUDA kernel."""

    if (
        not isinstance(source, torch.Tensor)
        or source.dtype != torch.bfloat16
        or source.ndim != 2
        or not source.is_cuda
        or not source.is_contiguous()
        or source.shape[1] % 8
    ):
        raise ValueError(
            "source must be contiguous CUDA BF16 [M,K] with K divisible by 8"
        )
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.ndim != 1
        or not indices.is_cuda
        or not indices.is_contiguous()
        or indices.device != source.device
        or indices.numel() == 0
    ):
        raise ValueError(
            "indices must be non-empty contiguous CUDA int32/int64 [D]"
        )
    return _dense_extension().cusparselt_indexed_gather(source, indices)


def cusparselt_sparse_residual_fused_kernel_attributes(
    token_rows: int,
    output_features: int,
    *,
    variant: str = "n64_s3",
) -> dict[str, int]:
    if not isinstance(token_rows, int) or token_rows <= 0:
        raise ValueError("token_rows must be positive")
    if token_rows % 32:
        raise ValueError("fused token_rows must be a multiple of 32")
    if not isinstance(output_features, int) or output_features <= 0:
        raise ValueError("output_features must be positive")
    try:
        variant_id = FUSED_BASE_COMPLEMENT_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(
            f"unknown fused variant {variant!r}; expected one of "
            f"{tuple(FUSED_BASE_COMPLEMENT_VARIANTS)}"
        ) from error
    raw = _dense_extension().cusparselt_fused_base_complement_kernel_attributes(
        token_rows, output_features, variant_id
    )
    names = (
        "num_regs",
        "dynamic_smem_bytes",
        "local_bytes",
        "max_threads_per_block",
        "active_blocks_per_sm",
        "theoretical_occupancy_pct",
        "actual_threads_per_block",
    )
    if not isinstance(raw, (list, tuple)) or len(raw) != len(names):
        raise RuntimeError("fused base+complement attribute query failed")
    return {name: int(value) for name, value in zip(names, raw, strict=True)}


def cusparselt_sparse_residual_kernel_attributes(
    token_rows: int,
    output_features: int,
    *,
    residual_only: bool = False,
    variant: str = "auto",
) -> dict[str, int]:
    if not isinstance(token_rows, int) or token_rows <= 0:
        raise ValueError("token_rows must be positive")
    if not isinstance(output_features, int) or output_features <= 0:
        raise ValueError("output_features must be positive")
    if not residual_only:
        raise ValueError(
            "only the complement HMMA.SP kernel remains; pass residual_only=True"
        )
    try:
        variant_id = COMPLEMENT_CTA_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(
            f"unknown complement CTA variant {variant!r}; expected one of "
            f"{tuple(COMPLEMENT_CTA_VARIANTS)}"
        ) from error
    raw = _dense_extension().cusparselt_complement_sparse_kernel_attributes(
        token_rows, output_features, variant_id
    )
    names = (
        "num_regs",
        "dynamic_smem_bytes",
        "local_bytes",
        "max_threads_per_block",
        "active_blocks_per_sm",
        "theoretical_occupancy_pct",
        "actual_threads_per_block",
    )
    if not isinstance(raw, (list, tuple)) or len(raw) != len(names):
        raise RuntimeError("cuSPARSELt direct-metadata attribute query failed")
    return {name: int(value) for name, value in zip(names, raw, strict=True)}


__all__ = [
    "CUSPARSELT_DIRECT_METADATA_LAYOUT",
    "CUSPARSELT_SPARSE_RESIDUAL",
    "COMPLEMENT_CTA_VARIANTS",
    "FUSED_BASE_COMPLEMENT_VARIANTS",
    "CusparseLtSparseResidualWeight",
    "cusparselt_sparse_residual_fused_dense_linear",
    "cusparselt_sparse_residual_fused_kernel_attributes",
    "cusparselt_sparse_residual_indexed_add_",
    "cusparselt_sparse_residual_indexed_copy_",
    "cusparselt_sparse_residual_indexed_gather",
    "cusparselt_sparse_residual_kernel_attributes",
    "cusparselt_sparse_residual_residual_linear",
    "cusparselt_sparse_residual_residual_linear_splitk2",
    "cusparselt_sparse_residual_residual_linear_splitk2_indexed",
    "cusparselt_sparse_residual_residual_linear_splitk2_chunked",
    "cusparselt_sparse_residual_residual_linear_splitk2_persistent",
    "cusparselt_sparse_residual_residual_linear_splitk4",
    "cusparselt_sparse_residual_splitk2_indexed_add_",
    "cusparselt_sparse_residual_splitk4_indexed_add_",
    "cusparselt_sparse_residual_sparse_linear",
    "prepare_cusparselt_sparse_residual_weight",
    "tune_cusparselt_sparse_residual_algorithm",
    "validate_cusparselt_packed_layout",
]
