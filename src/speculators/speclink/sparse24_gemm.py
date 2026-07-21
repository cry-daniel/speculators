"""Reusable BF16 2:4 GEMM primitives for isolated SpecLink experiments.

The public weight convention matches ``torch.nn.functional.linear`` and model
checkpoints: a weight has shape ``[N, K]`` and an input has shape ``[M, K]``.
Every sparse operation in this module uses PyTorch's real cuSPARSELt-backed
``SparseSemiStructuredTensorCUSPARSELT``.  Unsupported devices, dtypes, shapes,
or backends fail explicitly; there is no dense fallback.

Route construction and validation intentionally live outside the timed GEMM
functions.  Benchmarks can therefore report both a route-ready kernel path and
an end-to-end path that includes route construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.sparse.semi_structured import (
    SparseSemiStructuredTensorCUSPARSELT,
)


_BACKEND = "cusparselt"
_WEIGHT_ALIGNMENT = 16


@dataclass(slots=True)
class PreparedSparse24Weight:
    """A compressed cuSPARSELt weight prepared outside the timed region.

    ``shape`` is the original checkpoint-style ``(N, K)`` shape.  The sparse
    tensor uses cuSPARSELt's fused-result-transpose mode so that
    :func:`sparse24_linear` produces a contiguous ``[M, N]`` tensor.
    """

    sparse_weight: torch.Tensor
    shape: tuple[int, int]
    algorithm_id: int
    backend: str = _BACKEND

    def __post_init__(self) -> None:
        if self.backend != _BACKEND:
            raise RuntimeError(
                f"unsupported sparse backend {self.backend!r}; expected {_BACKEND!r}"
            )
        if not isinstance(
            self.sparse_weight, SparseSemiStructuredTensorCUSPARSELT
        ):
            raise RuntimeError(
                "prepared weight must be a "
                "SparseSemiStructuredTensorCUSPARSELT; dense fallback is forbidden"
            )
        if tuple(self.sparse_weight.shape) != self.shape:
            raise ValueError(
                "prepared shape does not match sparse tensor: "
                f"{self.shape} != {tuple(self.sparse_weight.shape)}"
            )
        if self.sparse_weight.BACKEND != _BACKEND:
            raise RuntimeError(
                f"PyTorch selected backend {self.sparse_weight.BACKEND!r}, "
                f"expected {_BACKEND!r}"
            )
        if not self.sparse_weight.fuse_transpose_cusparselt:
            raise RuntimeError("cuSPARSELt fused result transpose must be enabled")
        if not isinstance(self.algorithm_id, int) or self.algorithm_id < 0:
            raise ValueError("algorithm_id must be a non-negative integer")
        if self.sparse_weight.alg_id_cusparselt != self.algorithm_id:
            raise ValueError(
                "algorithm_id does not match the compressed sparse tensor"
            )

    @property
    def n(self) -> int:
        """Output-feature dimension."""

        return self.shape[0]

    @property
    def k(self) -> int:
        """Input-feature/reduction dimension."""

        return self.shape[1]

    @property
    def dtype(self) -> torch.dtype:
        return self.sparse_weight.dtype

    @property
    def device(self) -> torch.device:
        return self.sparse_weight.device

    @property
    def packed(self) -> torch.Tensor:
        """The opaque cuSPARSELt compressed buffer."""

        packed = self.sparse_weight.packed
        if packed is None:
            raise RuntimeError("cuSPARSELt compressed buffer is missing")
        return packed


@dataclass(slots=True)
class TokenRoute:
    """A complete, sorted partition of token rows on one CUDA device."""

    rows: int
    dense_indices: torch.Tensor
    sparse_indices: torch.Tensor
    dense_mask: torch.Tensor | None = None
    # True only when a constructor either proved the partition or created both
    # branches as a mask and its exact complement.  Native indexed kernels use
    # this to avoid synchronizing validation in their timed hot path.
    validated_complete_partition: bool = False

    @property
    def dense_count(self) -> int:
        return self.dense_indices.numel()

    @property
    def sparse_count(self) -> int:
        return self.sparse_indices.numel()

    @property
    def device(self) -> torch.device:
        return self.dense_indices.device


def _require_cuda_bf16_matrix(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got {tensor.device}")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must use torch.bfloat16, got {tensor.dtype}")


def assert_24_weight(weight24: torch.Tensor) -> None:
    """Validate an exact K-axis 2:4 BF16 CUDA weight in ``[N, K]`` layout.

    This validation synchronizes the device and must be called outside timed
    regions.  Every consecutive group of four values on the K axis must contain
    exactly two nonzero values.
    """

    _require_cuda_bf16_matrix(weight24, "weight24")
    n, k = weight24.shape
    if n < _WEIGHT_ALIGNMENT or k < _WEIGHT_ALIGNMENT:
        raise ValueError(
            f"weight24 shape {(n, k)} is too small for BF16 cuSPARSELt"
        )
    if n % _WEIGHT_ALIGNMENT != 0 or k % _WEIGHT_ALIGNMENT != 0:
        raise ValueError(
            "BF16 cuSPARSELt requires N and K to be multiples of "
            f"{_WEIGHT_ALIGNMENT}, got {(n, k)}"
        )

    nonzero_per_group = weight24.reshape(n, k // 4, 4).ne(0).sum(dim=-1)
    valid = nonzero_per_group.eq(2)
    if not bool(valid.all().item()):
        first_bad = (~valid).nonzero(as_tuple=False)[0]
        row = int(first_bad[0].item())
        group = int(first_bad[1].item())
        count = int(nonzero_per_group[row, group].item())
        raise ValueError(
            "weight24 is not exact 2:4 on the K axis: "
            f"row {row}, K-group {group} contains {count} nonzeros"
        )


def prepare_sparse24_weight(
    weight24: torch.Tensor, algorithm_id: int = 0
) -> PreparedSparse24Weight:
    """Compress an exact ``[N, K]`` weight with the BF16 cuSPARSELt backend."""

    if not isinstance(algorithm_id, int) or algorithm_id < 0:
        raise ValueError("algorithm_id must be a non-negative integer")
    assert_24_weight(weight24)

    # Compression and this contiguous conversion are setup costs, not GEMM
    # costs.  Avoid the global PyTorch backend/fused-transpose switches so this
    # experimental helper cannot affect unrelated sparse tensors.
    # Instantiate the requested backend directly.  The generic
    # ``to_sparse_semi_structured`` helper is controlled by process-global
    # flags and could otherwise select CUTLASS (which does not run on CC 12.0
    # in the current PyTorch build).
    sparse_weight = SparseSemiStructuredTensorCUSPARSELT.from_dense(
        weight24.contiguous()
    )
    if not isinstance(sparse_weight, SparseSemiStructuredTensorCUSPARSELT):
        raise RuntimeError(
            "PyTorch did not create a cuSPARSELt sparse tensor; "
            "dense or CUTLASS fallback is forbidden"
        )
    sparse_weight.fuse_transpose_cusparselt = True
    sparse_weight.alg_id_cusparselt = algorithm_id

    return PreparedSparse24Weight(
        sparse_weight=sparse_weight,
        shape=(weight24.shape[0], weight24.shape[1]),
        algorithm_id=algorithm_id,
    )


def select_cusparselt_algorithm(
    prepared: PreparedSparse24Weight, sample_input: torch.Tensor
) -> int:
    """Search and install the cuSPARSELt algorithm for ``sample_input``.

    The search runs kernels and synchronizes CUDA, so callers must use a sample
    reserved for tuning and invoke this function outside formal measurements.
    It returns the chosen algorithm ID and updates ``prepared`` in place.
    """

    _validate_prepared(prepared)
    _validate_input(sample_input, prepared, "sample_input")

    cusparselt = getattr(torch._C, "_cusparselt", None)
    mm_search = getattr(cusparselt, "mm_search", None)
    if mm_search is None:
        raise RuntimeError(
            "this PyTorch build does not expose torch._C._cusparselt.mm_search"
        )

    # Match F.linear's cuSPARSELt operand orientation.  The private padding
    # helper is the same helper used by PyTorch's semi-structured dispatcher.
    dense_b = prepared.sparse_weight._pad_dense_input(sample_input.t())
    search_result = mm_search(
        prepared.packed,
        dense_b,
        None,
        None,
        None,
        True,
    )
    algorithm_id = int(
        search_result[0]
        if isinstance(search_result, (tuple, list))
        else search_result
    )
    if algorithm_id < 0:
        raise RuntimeError(
            f"cuSPARSELt returned invalid algorithm ID {algorithm_id}"
        )

    prepared.sparse_weight.alg_id_cusparselt = algorithm_id
    prepared.algorithm_id = algorithm_id
    return algorithm_id


def _require_route_index(index: torch.Tensor, name: str) -> None:
    if not isinstance(index, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if index.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(index.shape)}")
    if not index.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got {index.device}")
    if index.dtype != torch.int64:
        raise ValueError(f"{name} must use torch.int64, got {index.dtype}")


def _validate_route_metadata(route: TokenRoute) -> None:
    """Validate route invariants that do not inspect CUDA tensor values."""

    if not isinstance(route.rows, int) or route.rows < 0:
        raise ValueError("rows must be a non-negative integer")
    _require_route_index(route.dense_indices, "dense_indices")
    _require_route_index(route.sparse_indices, "sparse_indices")
    if route.dense_indices.device != route.sparse_indices.device:
        raise ValueError("dense_indices and sparse_indices must share a device")
    if route.dense_count + route.sparse_count != route.rows:
        raise ValueError(
            "dense and sparse index counts must sum to rows: "
            f"{route.dense_count} + {route.sparse_count} != {route.rows}"
        )
    if route.dense_mask is not None:
        mask = route.dense_mask
        if not isinstance(mask, torch.Tensor):
            raise TypeError("dense_mask must be a torch.Tensor")
        if not mask.is_cuda or mask.device != route.device:
            raise ValueError("dense_mask must be on the route CUDA device")
        if mask.dtype != torch.bool or mask.ndim != 1 or mask.numel() != route.rows:
            raise ValueError(
                "dense_mask must be a 1D bool tensor with exactly rows entries"
            )


def _validate_route(route: TokenRoute) -> None:
    _validate_route_metadata(route)

    for name, index in (
        ("dense_indices", route.dense_indices),
        ("sparse_indices", route.sparse_indices),
    ):
        if index.numel() == 0:
            continue
        if int(index[0].item()) < 0 or int(index[-1].item()) >= route.rows:
            raise ValueError(f"{name} contains an index outside [0, {route.rows})")
        if index.numel() > 1 and not bool((index[1:] > index[:-1]).all().item()):
            raise ValueError(f"{name} must be strictly increasing and unique")

    expected_dense_mask = torch.zeros(
        route.rows, dtype=torch.bool, device=route.device
    )
    expected_dense_mask[route.dense_indices] = True
    if route.sparse_count and bool(
        expected_dense_mask[route.sparse_indices].any().item()
    ):
        raise ValueError("dense_indices and sparse_indices overlap")
    if route.sparse_count and not bool(
        (~expected_dense_mask)[route.sparse_indices].all().item()
    ):
        raise ValueError("dense_indices and sparse_indices are not complementary")

    if route.dense_mask is not None:
        mask = route.dense_mask
        if not bool(mask.eq(expected_dense_mask).all().item()):
            raise ValueError("dense_mask does not match dense_indices")


def route_from_mask(mask: torch.Tensor, validate: bool = True) -> TokenRoute:
    """Build sorted dense/sparse row indices from a GPU boolean mask."""

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1D, got shape {tuple(mask.shape)}")
    if not mask.is_cuda:
        raise ValueError(f"mask must be a CUDA tensor, got {mask.device}")
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must use torch.bool, got {mask.dtype}")

    mask = mask.contiguous()
    dense_indices = mask.nonzero(as_tuple=False).flatten().to(dtype=torch.int64)
    sparse_indices = (~mask).nonzero(as_tuple=False).flatten().to(dtype=torch.int64)
    route = TokenRoute(
        rows=mask.numel(),
        dense_indices=dense_indices.contiguous(),
        sparse_indices=sparse_indices.contiguous(),
        dense_mask=mask,
        validated_complete_partition=True,
    )
    if validate:
        _validate_route(route)
    return route


def route_from_indices(
    rows: int,
    dense_indices: torch.Tensor,
    sparse_indices: torch.Tensor,
    dense_mask: torch.Tensor | None = None,
    validate: bool = True,
) -> TokenRoute:
    """Wrap an already prepared, sorted GPU row partition.

    Expensive range, ordering, partition, and mask checks are controlled by
    ``validate``.  Device, rank, and dtype invariants are always checked so an
    invalid route cannot silently select a different execution path.
    """

    if not isinstance(rows, int) or rows < 0:
        raise ValueError("rows must be a non-negative integer")
    _require_route_index(dense_indices, "dense_indices")
    _require_route_index(sparse_indices, "sparse_indices")
    if dense_indices.device != sparse_indices.device:
        raise ValueError("dense_indices and sparse_indices must share a device")

    normalized_mask = None
    if dense_mask is not None:
        if not isinstance(dense_mask, torch.Tensor):
            raise TypeError("dense_mask must be a torch.Tensor")
        normalized_mask = dense_mask.contiguous()

    route = TokenRoute(
        rows=rows,
        dense_indices=dense_indices.contiguous(),
        sparse_indices=sparse_indices.contiguous(),
        dense_mask=normalized_mask,
        validated_complete_partition=validate,
    )
    _validate_route_metadata(route)
    if validate:
        _validate_route(route)
    return route


def _validate_prepared(prepared: PreparedSparse24Weight) -> None:
    if not isinstance(prepared, PreparedSparse24Weight):
        raise TypeError("prepared must be a PreparedSparse24Weight")
    prepared.__post_init__()


def _validate_input(
    x: torch.Tensor, prepared: PreparedSparse24Weight, name: str = "x"
) -> None:
    _require_cuda_bf16_matrix(x, name)
    if x.device != prepared.device:
        raise ValueError(
            f"{name} and sparse weight must share a device: "
            f"{x.device} != {prepared.device}"
        )
    if x.shape[1] != prepared.k:
        raise ValueError(
            f"{name} K={x.shape[1]} does not match weight K={prepared.k}"
        )


def sparse24_linear(
    x: torch.Tensor, prepared: PreparedSparse24Weight
) -> torch.Tensor:
    """Compute BF16 ``X @ W24.T`` through cuSPARSELt.

    The returned tensor is always contiguous with shape ``[M, N]``.
    """

    _validate_prepared(prepared)
    _validate_input(x, prepared)
    output = F.linear(x, prepared.sparse_weight)
    if output.shape != (x.shape[0], prepared.n):
        raise RuntimeError(
            "cuSPARSELt returned an unexpected output shape: "
            f"{tuple(output.shape)} != {(x.shape[0], prepared.n)}"
        )
    return output.contiguous()


def hybrid_split_linear(
    x: torch.Tensor,
    dense_weight: torch.Tensor,
    prepared: PreparedSparse24Weight,
    route: TokenRoute,
) -> torch.Tensor:
    """Compute an exact serial dense/2:4 row-routed linear operation.

    Dense rows use ``dense_weight`` and sparse rows use ``prepared``.  Both row
    subsets are gathered first, their GEMMs execute serially on the current CUDA
    stream, and the results are merged with ``index_copy_``.  Empty and full
    dense routes avoid launching empty GEMMs.
    """

    _validate_prepared(prepared)
    _validate_input(x, prepared)
    _require_cuda_bf16_matrix(dense_weight, "dense_weight")
    if dense_weight.device != x.device:
        raise ValueError("dense_weight and x must share a CUDA device")
    if tuple(dense_weight.shape) != prepared.shape:
        raise ValueError(
            "dense and sparse weight shapes must match: "
            f"{tuple(dense_weight.shape)} != {prepared.shape}"
        )
    if not isinstance(route, TokenRoute):
        raise TypeError("route must be a TokenRoute")
    _validate_route_metadata(route)
    if route.rows != x.shape[0]:
        raise ValueError(f"route rows={route.rows} does not match M={x.shape[0]}")
    if route.device != x.device:
        raise ValueError("route indices and x must share a CUDA device")

    if route.dense_count == 0:
        return sparse24_linear(x, prepared)
    if route.sparse_count == 0:
        return F.linear(x, dense_weight).contiguous()

    dense_x = x.index_select(0, route.dense_indices)
    sparse_x = x.index_select(0, route.sparse_indices)
    dense_output = F.linear(dense_x, dense_weight)
    sparse_output = sparse24_linear(sparse_x, prepared)

    output = torch.empty(
        (x.shape[0], prepared.n), dtype=torch.bfloat16, device=x.device
    )
    output.index_copy_(0, route.dense_indices, dense_output)
    output.index_copy_(0, route.sparse_indices, sparse_output)
    return output
