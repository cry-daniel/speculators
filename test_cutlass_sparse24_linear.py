import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

from vllm import speclink_linear, speclink_mlp
from vllm import _custom_ops as vllm_ops

from vllm.speclink_kernel import (
    apply_random_24_mask,
    assert_24_weight,
    dense_cutlass_routed_rows_weight_t,
    dense_cutlass_weight_t_gemm,
    dense_cutlass_weight_t_gemm_add,
    decompress_24,
    pack_24,
    prepare_cutlass_sparse24_device_gemm,
    prepare_cutlass_sparse24_gate_up_swiglu,
    prepare_cutlass_sparse24_pair_add,
    sparse24_cutlass_device_gemm_prepacked,
    sparse24_cutlass_dual_swiglu_prepacked,
    sparse24_cutlass_gather_gemm_prepacked,
    sparse24_cutlass_gate_up_swiglu_prepacked,
    sparse24_cutlass_heterogeneous_linear_prepacked,
    sparse24_cutlass_heterogeneous_swiglu_prepacked,
    sparse24_cutlass_indexed_output_gemm_prepacked,
    sparse24_cutlass_indexed_swiglu_prepacked,
    sparse24_cutlass_inline_transpose_gemm_prepacked,
    sparse24_cutlass_pair_add_indexed_prepacked,
    sparse24_cutlass_paired_persistent_routed_swiglu_prepacked,
    sparse24_cutlass_qkv_postop_prepacked,
    sparse24_cutlass_residual_correction_swiglu_prepacked,
    sparse24_cutlass_routed_output_gemm_prepacked,
    sparse24_cutlass_routed_sparse_rows_prepacked,
    sparse24_cutlass_routed_swiglu_prepacked,
    sparse24_gather_rows_strided_,
    sparse24_merge_rows_,
    sparse24_mixed_dense_override_prepacked,
    sparse24_partition_rows_,
    sparse24_qkv_rmsnorm_inplace_,
    sparse24_qkv_transpose_postop,
    sparse24_qkv_transpose_rmsnorm,
    sparse24_routed_linear_correction_,
    sparse24_routed_swiglu_correction_,
    sparse24_routed_swiglu_correction_gather_,
    sparse24_silu_and_mul_transposed,
    sparse24_silu_and_mul_transposed_to_contiguous,
    sparse24_transpose_add_residual_,
    sparse24_transpose_add_routed_residual,
    sparse24_transpose_add_rmsnorm,
    sparse24_transpose_input_to_strided,
    sparse24_transpose_output_contiguous,
)


REPO_ROOT = Path(__file__).resolve().parent


def _qkv_rmsnorm_reference(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
    head_dim: int,
    epsilon: float,
) -> torch.Tensor:
    q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)

    def normalize(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        by_head = x.float().reshape(-1, shape[-1] // head_dim, head_dim)
        inverse_rms = torch.rsqrt(by_head.square().mean(dim=-1, keepdim=True) + epsilon)
        return (by_head * inverse_rms * weight.float()).to(torch.float16).reshape(shape)

    return torch.cat((normalize(q, q_weight), normalize(k, k_weight), v), dim=-1)


def _load_bench_module():
    path = REPO_ROOT / "scripts" / "bench_cutlass_sparse24_linear.py"
    spec = importlib.util.spec_from_file_location("bench_cutlass_sparse24_linear", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CutlassSparse24LinearTests(unittest.TestCase):
    def test_pack_roundtrip_cpu(self) -> None:
        weight = torch.tensor(
            [[1.0, 0.0], [2.0, 3.0], [0.0, 4.0], [0.0, 0.0]],
            dtype=torch.float16,
        )
        packed = pack_24(weight, layout="n_major")
        actual = decompress_24(packed.values, packed.meta, layout=packed.layout)
        self.assertTrue(torch.equal(actual, weight))

    def test_serving_preset_has_expected_projection_shapes(self) -> None:
        bench = _load_bench_module()
        cases = bench.preset_shape_cases("serving", [8, 64], mlp_projections="fused")
        labels = {case.label for case in cases}
        self.assertIn("qwen3_8b:gate_up_proj:M8", labels)
        self.assertIn("llama3_1_8b:down_proj:M64", labels)
        self.assertEqual(len(cases), 16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_selective_benchmark_supports_padded_full_output_cuda(self) -> None:
        bench = _load_bench_module()
        rows = bench.run_case(
            bench.ShapeCase("padded_selective", (16, 64, 32)),
            seed=17,
            warmup=1,
            repeat=1,
            measure_trials=1,
            device_config_values=["auto"],
            pad_m_multiple_values=[32],
            selective_dense_fractions=[0.25],
            reuse_output=True,
            row_selection="random",
            random_gather_backend="cutlass",
            selective_dense_strategies=["full_sparse_residual"],
            rtol=2e-2,
            atol=8e-2,
        )
        selective = [
            row for row in rows if row["backend"].startswith("selective_dense_")
        ]
        self.assertTrue(selective)
        self.assertTrue(all(row["M_padded"] == 32 for row in selective))
        self.assertTrue(all(row["pass"] for row in selective))
        self.assertTrue(all(not row["error"] for row in selective))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cutlass_prepacked_gemm_matches_dense_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(7)
        x = torch.randn((8, 64), device="cuda", dtype=torch.float16, generator=generator)
        weight = torch.randn((64, 32), device="cuda", dtype=torch.float16, generator=generator)
        weight24, _meta = apply_random_24_mask(weight, generator=generator)
        assert_24_weight(weight24)
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=64,
        )

        actual = sparse24_cutlass_device_gemm_prepacked(x, values, meta)
        expected = x @ weight24
        torch.cuda.synchronize()

        self.assertFalse(actual.is_contiguous())
        self.assertTrue(torch.allclose(actual, expected, rtol=2e-2, atol=8e-2))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dense_weight_t_fp16_and_residual_epilogue_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(11)
        x = torch.randn(
            (8, 64), device="cuda", dtype=torch.float16, generator=generator
        ).mul_(0.1)
        weight_t = torch.randn(
            (32, 64), device="cuda", dtype=torch.float16, generator=generator
        ).mul_(0.1)
        residual = torch.randn(
            (8, 32), device="cuda", dtype=torch.float16, generator=generator
        ).mul_(0.1)
        expected = torch.mm(x, weight_t.t())

        actual_fp32 = dense_cutlass_weight_t_gemm(
            x, weight_t, accumulator="fp32"
        )
        actual_fp16 = dense_cutlass_weight_t_gemm(
            x, weight_t, accumulator="fp16"
        )
        residual_inplace = residual.clone()
        actual_add = dense_cutlass_weight_t_gemm_add(
            x, weight_t, residual_inplace, out=residual_inplace
        )
        actual_add_fp32 = dense_cutlass_weight_t_gemm_add(
            x,
            weight_t,
            residual,
            accumulator="fp32",
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.allclose(actual_fp32, expected, rtol=2e-2, atol=2e-2)
        )
        self.assertTrue(
            torch.allclose(actual_fp16, expected, rtol=3e-2, atol=3e-2)
        )
        self.assertIs(actual_add, residual_inplace)
        self.assertTrue(
            torch.allclose(
                actual_add,
                expected + residual,
                rtol=3e-2,
                atol=3e-2,
            )
        )
        self.assertTrue(
            torch.allclose(
                actual_add_fp32,
                expected + residual,
                rtol=2e-2,
                atol=2e-2,
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_heterogeneous_routed_linear_matches_exact_rows_and_graph_cuda(
        self,
    ) -> None:
        generator = torch.Generator(device="cuda").manual_seed(20260715)
        rows, inner, output = 48, 64, 128
        x = torch.randn(
            (rows, inner),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.05)
        weight = torch.randn(
            (inner, output),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.05)
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=inner,
        )
        dense_rows = torch.arange(
            1, rows, 4, device="cuda", dtype=torch.int32
        )
        row_is_dense = torch.zeros(rows, device="cuda", dtype=torch.bool)
        row_is_dense[dense_rows.long()] = True
        sparse_rows = torch.arange(
            rows, device="cuda", dtype=torch.int32
        )[~row_is_dense]
        dense_weight = weight.t().contiguous()
        out = torch.empty(
            (rows, output), device="cuda", dtype=torch.float16
        )

        actual = sparse24_cutlass_heterogeneous_linear_prepacked(
            x,
            values,
            meta,
            dense_weight,
            dense_rows,
            sparse_rows,
            out=out,
            config="128x32x64_s4_sw4",
        )
        expected = x @ weight24
        expected[dense_rows.long()] = x[dense_rows.long()] @ weight
        torch.cuda.synchronize()
        self.assertEqual(actual.data_ptr(), out.data_ptr())
        self.assertTrue(
            torch.allclose(actual, expected, rtol=3e-2, atol=3e-2)
        )
        f16_accum_out = torch.empty_like(out)
        sparse24_cutlass_heterogeneous_linear_prepacked(
            x,
            values,
            meta,
            dense_weight,
            dense_rows,
            sparse_rows,
            out=f16_accum_out,
            config="256x32x64_s3_sw4_f16",
        )
        torch.cuda.synchronize()
        self.assertTrue(
            torch.allclose(
                f16_accum_out, expected, rtol=5e-2, atol=5e-2
            )
        )
        for config in (
            "256x64_sparse_128x32_dense_s3_f16",
            "256x64_sparse_128x64_dense_s3_f16",
            "256x128_sparse_128x64_dense_s2_f16",
        ):
            with self.subTest(config=config):
                hybrid_out = torch.empty_like(out)
                sparse24_cutlass_heterogeneous_linear_prepacked(
                    x,
                    values,
                    meta,
                    dense_weight,
                    dense_rows,
                    sparse_rows,
                    out=hybrid_out,
                    config=config,
                )
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.allclose(
                        hybrid_out, expected, rtol=5e-2, atol=5e-2
                    ),
                    (
                        f"config={config} max_abs_diff="
                        f"{(hybrid_out.float() - expected.float()).abs().max().item()}"
                    ),
                )

        gate_output = 256
        gate_weight = torch.randn(
            (inner, gate_output),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.05)
        gate_weight24, _ = apply_random_24_mask(
            gate_weight, generator=generator
        )
        gate_packed = pack_24(gate_weight24, layout="n_major")
        gate_values, gate_meta = prepare_cutlass_sparse24_gate_up_swiglu(
            gate_packed.values,
            gate_packed.meta,
            layout=gate_packed.layout,
            K=inner,
        )
        hidden_size = gate_output // 2
        dense_pair_channels = 64
        gate_indices = torch.arange(
            hidden_size, device="cuda", dtype=torch.int32
        ).reshape(-1, dense_pair_channels)
        dense_weight_rows = torch.cat(
            (gate_indices, gate_indices + hidden_size), dim=1
        ).flatten().contiguous()
        dense_gate_weight = gate_weight.t().contiguous()
        expected_gate_up = x @ gate_weight24
        expected_gate_up[dense_rows.long()] = (
            x[dense_rows.long()] @ gate_weight
        )
        expected_gate, expected_up = expected_gate_up.chunk(2, dim=-1)
        expected_hidden = (
            torch.nn.functional.silu(expected_gate.float()).half()
            * expected_up
        ).half()
        for config in ("256x32x64_s3_sw4_f16",):
            with self.subTest(heterogeneous_swiglu_config=config):
                hidden_out = torch.empty_like(expected_hidden)
                sparse24_cutlass_heterogeneous_swiglu_prepacked(
                    x,
                    gate_values,
                    gate_meta,
                    dense_gate_weight,
                    dense_weight_rows,
                    dense_rows,
                    sparse_rows,
                    out=hidden_out,
                    config=config,
                )
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.allclose(
                        hidden_out,
                        expected_hidden,
                        rtol=5e-2,
                        atol=8e-2,
                    ),
                    (
                        f"config={config} max_abs_diff="
                        f"{(hidden_out.float() - expected_hidden.float()).abs().max().item()}"
                    ),
                )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sparse24_cutlass_heterogeneous_linear_prepacked(
                x,
                values,
                meta,
                dense_weight,
                dense_rows,
                sparse_rows,
                out=out,
                config="128x32x64_s4_sw4",
            )
        replay_x = torch.randn_like(x).mul_(0.05)
        x.copy_(replay_x)
        graph.replay()
        torch.cuda.synchronize()
        replay_expected = replay_x @ weight24
        replay_expected[dense_rows.long()] = (
            replay_x[dense_rows.long()] @ weight
        )
        self.assertTrue(
            torch.allclose(out, replay_expected, rtol=3e-2, atol=3e-2)
        )

        component_out = torch.empty_like(out)
        sparse_stream = torch.cuda.Stream()
        dense_stream = torch.cuda.Stream()

        def dual_stream_components() -> torch.Tensor:
            current = torch.cuda.current_stream()
            sparse_stream.wait_stream(current)
            dense_stream.wait_stream(current)
            with torch.cuda.stream(sparse_stream):
                sparse24_cutlass_routed_sparse_rows_prepacked(
                    x,
                    values,
                    meta,
                    sparse_rows,
                    out=component_out,
                    config="128x32x64_s4_sw4",
                )
            with torch.cuda.stream(dense_stream):
                dense_cutlass_routed_rows_weight_t(
                    x,
                    dense_weight,
                    dense_rows,
                    out=component_out,
                    config="128x32x64_s4_sw4",
                )
            current.wait_stream(sparse_stream)
            current.wait_stream(dense_stream)
            return component_out

        dual_stream_components()
        torch.cuda.synchronize()
        self.assertTrue(
            torch.allclose(
                component_out, replay_expected, rtol=3e-2, atol=3e-2
            )
        )
        component_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(component_graph):
            dual_stream_components()
        second_replay_x = torch.randn_like(x).mul_(0.05)
        x.copy_(second_replay_x)
        component_graph.replay()
        torch.cuda.synchronize()
        second_expected = second_replay_x @ weight24
        second_expected[dense_rows.long()] = (
            second_replay_x[dense_rows.long()] @ weight
        )
        self.assertTrue(
            torch.allclose(
                component_out, second_expected, rtol=3e-2, atol=3e-2
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cutlass_prepacked_gemm_reuses_output_and_workspace_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(13)
        x = torch.randn((8, 64), device="cuda", dtype=torch.float16, generator=generator)
        weight = torch.randn((64, 32), device="cuda", dtype=torch.float16, generator=generator)
        weight24, _meta = apply_random_24_mask(weight, generator=generator)
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=64,
        )
        out = torch.empty((8, 32), device="cuda", dtype=torch.float16)
        workspace = torch.empty((32, 8), device="cuda", dtype=torch.float16)

        actual = sparse24_cutlass_device_gemm_prepacked(
            x,
            values,
            meta,
            contiguous_output=True,
            out=out,
            workspace=workspace,
        )
        expected = x @ weight24
        torch.cuda.synchronize()

        self.assertEqual(actual.data_ptr(), out.data_ptr())
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.allclose(actual, expected, rtol=2e-2, atol=8e-2))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cutlass_gather_gemm_matches_explicit_gather_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(17)
        x = torch.randn(
            (19, 64), device="cuda", dtype=torch.float16, generator=generator
        )
        weight = torch.randn(
            (64, 256),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_device_gemm(
            packed.values, packed.meta, layout=packed.layout, K=64
        )
        rows = torch.tensor(
            [18, 1, 7, 4, 15, 0, 11], device="cuda", dtype=torch.int32
        )
        expected = x.index_select(0, rows.long()) @ weight24

        for config in (
            "256x32x64_s3_sw4",
            "256x64x64_s3_sw4",
            "128x32x64_s4_sw4",
        ):
            with self.subTest(config=config):
                actual = sparse24_cutlass_gather_gemm_prepacked(
                    x, values, meta, rows, config=config
                )
                torch.cuda.synchronize()
                self.assertEqual(tuple(actual.stride()), (1, 8))
                self.assertTrue(
                    torch.allclose(actual, expected, rtol=2e-2, atol=8e-2)
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_inline_transpose_sparse_epilogue_matches_dense_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(19)
        shapes = (
            (16, 64, 256),
            (128, 64, 256),
            (16, 64, 512),
            (128, 64, 512),
            (112, 4096, 4096),
            (144, 4096, 4096),
            (448, 4096, 4096),
        )
        for rows, inner, columns in shapes:
            x = torch.randn(
                (rows, inner),
                device="cuda",
                dtype=torch.float16,
                generator=generator,
            )
            weight = torch.randn(
                (inner, columns),
                device="cuda",
                dtype=torch.float16,
                generator=generator,
            )
            weight.add_(torch.where(weight >= 0, 0.25, -0.25))
            weight24, _ = apply_random_24_mask(weight, generator=generator)
            packed = pack_24(weight24, layout="n_major")
            values, meta = prepare_cutlass_sparse24_device_gemm(
                packed.values, packed.meta, layout=packed.layout, K=inner
            )
            out = torch.empty(
                (rows, columns), device="cuda", dtype=torch.float16
            )
            expected = x @ weight24
            for store_mode in ("scalar", "vector"):
                with self.subTest(
                    rows=rows,
                    inner=inner,
                    columns=columns,
                    store_mode=store_mode,
                ):
                    out.fill_(float("nan"))
                    actual = sparse24_cutlass_inline_transpose_gemm_prepacked(
                        x, values, meta, out=out, store_mode=store_mode
                    )
                    torch.cuda.synchronize()

                    self.assertEqual(actual.data_ptr(), out.data_ptr())
                    self.assertTrue(actual.is_contiguous())
                    close = torch.allclose(
                        actual, expected, rtol=2e-2, atol=8e-2
                    )
                    if not close:
                        finite = torch.isfinite(actual)
                        row_counts = torch.unique(finite.sum(dim=1)).cpu().tolist()
                        column_counts = torch.unique(finite.sum(dim=0)).cpu().tolist()
                        self.fail(
                            f"inline epilogue mismatch: finite="
                            f"{int(finite.sum().item())}/{actual.numel()}, "
                            f"row_counts={row_counts}, "
                            f"column_counts={column_counts}"
                        )

            if inner == 4096 and columns == 4096:
                out.fill_(float("nan"))
                actual = sparse24_cutlass_inline_transpose_gemm_prepacked(
                    x,
                    values,
                    meta,
                    out=out,
                    config="auto",
                    store_mode="vector",
                )
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.allclose(actual, expected, rtol=2e-2, atol=8e-2)
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_routed_sparse_output_epilogue_matches_reference_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(89)
        rows = 16
        inner = 64
        columns = 256
        x = torch.randn(
            (rows, inner),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight = torch.randn(
            (inner, columns),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight.add_(torch.where(weight >= 0, 0.25, -0.25))
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        residual24 = weight - weight24
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=inner,
        )
        dense_rows = torch.tensor(
            [1, 4, 7, 11, 14], device="cuda", dtype=torch.int32
        )
        dense_slots = torch.full(
            (rows,), -1, device="cuda", dtype=torch.int32
        )
        dense_slots[dense_rows.long()] = torch.arange(
            dense_rows.numel(), device="cuda", dtype=torch.int32
        )
        sparse_mask = dense_slots < 0
        sparse_expected = x @ weight24
        exact_expected = sparse_expected.clone()
        dense_residual = x[dense_rows.long()] @ residual24
        exact_expected[dense_rows.long()] += dense_residual

        for config in (
            "auto",
            "64x64x64_s6",
            "128x32x64_s4_sw4",
            "128x64x64_s5",
            "256x64x64_s3",
            "256x64x64_s3_sw4",
        ):
            with self.subTest(config=config):
                out = torch.full(
                    (rows, columns),
                    float("nan"),
                    device="cuda",
                    dtype=torch.float16,
                )
                dense_base = torch.empty(
                    (dense_rows.numel(), columns),
                    device="cuda",
                    dtype=torch.float16,
                )
                actual, actual_base = (
                    sparse24_cutlass_routed_output_gemm_prepacked(
                        x,
                        values,
                        meta,
                        dense_slots,
                        dense_count=dense_rows.numel(),
                        out=out,
                        dense_base=dense_base,
                        config=config,
                    )
                )
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.allclose(
                        actual[sparse_mask],
                        sparse_expected[sparse_mask],
                        rtol=2e-2,
                        atol=8e-2,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        actual_base,
                        sparse_expected[dense_rows.long()],
                        rtol=2e-2,
                        atol=8e-2,
                    )
                )
                self.assertFalse(torch.isfinite(actual[~sparse_mask]).any())
                corrected = sparse24_routed_linear_correction_(
                    actual_base,
                    dense_residual,
                    dense_rows,
                    actual,
                )
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.allclose(
                        corrected, exact_expected, rtol=2e-2, atol=8e-2
                    )
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_transpose_add_routed_residual_matches_reference_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(97)
        rows = 112
        columns = 6144
        dense_count = 16
        dense_run = 16
        dense_rows = torch.randperm(
            rows, device="cuda", generator=generator
        )[:dense_count].sort().values.to(torch.int32)
        dense_slots = torch.full(
            (rows,), -1, device="cuda", dtype=torch.int32
        )
        dense_slots[dense_rows.long()] = torch.arange(
            dense_count, device="cuda", dtype=torch.int32
        )
        full = torch.randn(
            (rows, columns),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        residual = torch.randn(
            (dense_count, columns),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        full_transposed = torch.empty_strided(
            full.shape, (1, rows), device="cuda", dtype=torch.float16
        )
        full_transposed.copy_(full)
        residual_transposed = torch.zeros(
            (dense_run, columns), device="cuda", dtype=torch.float16
        ).as_strided((dense_run, columns), (1, dense_run))
        residual_transposed[:dense_count].copy_(residual)
        expected = full.clone()
        expected[dense_rows.long()] += residual

        actual = sparse24_transpose_add_routed_residual(
            full_transposed,
            residual_transposed,
            dense_slots,
            dense_count=dense_count,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_indexed_sparse_epilogue_scatter_matches_dense_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(23)
        inner = 64
        columns = 256
        logical_rows = 13
        padded_rows = 16
        output_rows = 27
        x = torch.randn(
            (padded_rows, inner),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        x[logical_rows:].zero_()
        weight = torch.randn(
            (inner, columns),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight.add_(torch.where(weight >= 0, 0.25, -0.25))
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=inner,
        )
        row_indices = torch.tensor(
            [0, 2, 3, 6, 8, 11, 13, 14, 17, 20, 22, 24, 26],
            device="cuda",
            dtype=torch.int32,
        )
        expected = x[:logical_rows] @ weight24

        for config in (
            "auto",
            "64x64x64_s6",
            "128x32x64_s4_sw4",
            "128x64x64_s5",
            "256x64x64_s3",
            "256x64x64_s3_sw4",
        ):
            with self.subTest(config=config):
                out = torch.full(
                    (output_rows, columns),
                    float("nan"),
                    device="cuda",
                    dtype=torch.float16,
                )
                actual = sparse24_cutlass_indexed_output_gemm_prepacked(
                    x,
                    values,
                    meta,
                    row_indices,
                    output_rows=output_rows,
                    out=out,
                    config=config,
                )
                torch.cuda.synchronize()

                self.assertEqual(actual.data_ptr(), out.data_ptr())
                self.assertTrue(
                    torch.allclose(
                        actual[row_indices.long()],
                        expected,
                        rtol=2e-2,
                        atol=8e-2,
                    )
                )
                untouched = torch.ones(
                    output_rows, device="cuda", dtype=torch.bool
                )
                untouched[row_indices.long()] = False
                self.assertFalse(torch.isfinite(actual[untouched]).any())

        x_transposed = torch.empty_strided(
            (padded_rows, inner),
            (1, padded_rows),
            device="cuda",
            dtype=torch.float16,
        )
        x_transposed.copy_(x)
        for config in (
            "auto",
            "64x64x64_s6",
            "64x64x64_s7",
            "128x32x64_s4_sw4",
            "128x64x64_s3",
            "128x64x64_s4",
            "128x64x64_s5",
            "128x128x64_s3",
            "256x64x64_s3",
        ):
            with self.subTest(config=config, input_transposed=True):
                out = torch.full(
                    (output_rows, columns),
                    float("nan"),
                    device="cuda",
                    dtype=torch.float16,
                )
                actual = sparse24_cutlass_indexed_output_gemm_prepacked(
                    x_transposed,
                    values,
                    meta,
                    row_indices,
                    output_rows=output_rows,
                    out=out,
                    config=config,
                    input_transposed=True,
                )
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.allclose(
                        actual[row_indices.long()],
                        expected,
                        rtol=2e-2,
                        atol=8e-2,
                    )
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_pair_add_indexed_sparse_epilogue_matches_reference_cuda(
        self,
    ) -> None:
        generator = torch.Generator(device="cuda").manual_seed(67)
        inner = 64
        columns = 256
        logical_rows = 13
        padded_rows = 16
        output_rows = 29
        x = torch.randn(
            (padded_rows, inner),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        x[logical_rows:].zero_()
        weight = torch.randn(
            (inner, columns),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight.add_(torch.where(weight >= 0, 0.25, -0.25))
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        residual24 = weight - weight24
        assert_24_weight(weight24)
        assert_24_weight(residual24)

        paired = torch.cat((weight24, residual24), dim=1)
        packed = pack_24(paired, layout="n_major")
        values, meta = prepare_cutlass_sparse24_pair_add(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=inner,
        )
        row_indices = torch.tensor(
            [0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 25, 28],
            device="cuda",
            dtype=torch.int32,
        )
        expected = x[:logical_rows] @ weight24
        expected.add_(x[:logical_rows] @ residual24)

        for config in (
            "auto",
            "256x32x64_s3_sw4",
            "256x64x64_s3",
            "256x64x64_s3_sw4",
        ):
            with self.subTest(config=config):
                out = torch.full(
                    (output_rows, columns),
                    float("nan"),
                    device="cuda",
                    dtype=torch.float16,
                )
                actual = sparse24_cutlass_pair_add_indexed_prepacked(
                    x,
                    values,
                    meta,
                    row_indices,
                    output_rows=output_rows,
                    out=out,
                    config=config,
                )
                torch.cuda.synchronize()

                self.assertEqual(actual.data_ptr(), out.data_ptr())
                self.assertTrue(
                    torch.allclose(
                        actual[row_indices.long()],
                        expected,
                        rtol=2e-2,
                        atol=8e-2,
                    )
                )
                untouched = torch.ones(
                    output_rows, device="cuda", dtype=torch.bool
                )
                untouched[row_indices.long()] = False
                self.assertFalse(torch.isfinite(actual[untouched]).any())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_inline_sparse_swiglu_matches_reference_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(71)
        inner = 64
        output_size = 512
        weight = torch.randn(
            (inner, output_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight.add_(torch.where(weight >= 0, 0.25, -0.25))
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_gate_up_swiglu(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=inner,
        )
        for rows in (16, 128):
            x = torch.randn(
                (rows, inner),
                device="cuda",
                dtype=torch.float16,
                generator=generator,
            )
            gate_up = x @ weight24
            expected = torch.empty(
                (rows, output_size // 2),
                device="cuda",
                dtype=torch.float16,
            )
            torch.ops._C.silu_and_mul(expected, gate_up)
            for config in (
                "256x64x64_s3",
                "256x64x64_s3_sw4",
                "256x64x64_s3_sw4_f16",
            ):
                output_modes = (
                    (False,)
                    if config.endswith("_f16")
                    else (False, True)
                )
                for output_transposed in output_modes:
                    with self.subTest(
                        rows=rows,
                        config=config,
                        output_transposed=output_transposed,
                    ):
                        if output_transposed:
                            out = torch.empty_strided(
                                expected.shape,
                                (1, rows),
                                device="cuda",
                                dtype=torch.float16,
                            )
                            out.fill_(float("nan"))
                        else:
                            out = torch.full_like(expected, float("nan"))
                        actual = sparse24_cutlass_gate_up_swiglu_prepacked(
                            x,
                            values,
                            meta,
                            out=out,
                            config=config,
                            output_transposed=output_transposed,
                        )
                        torch.cuda.synchronize()
                        self.assertEqual(actual.data_ptr(), out.data_ptr())
                        self.assertEqual(
                            tuple(actual.stride()),
                            (1, rows)
                            if output_transposed
                            else (output_size // 2, 1),
                        )
                        self.assertTrue(torch.isfinite(actual).all())
                        self.assertTrue(
                            torch.allclose(
                                actual, expected, rtol=3e-2, atol=1e-1
                            )
                        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_routed_sparse_swiglu_epilogue_matches_reference_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(79)
        rows = 16
        inner = 64
        hidden_size = 256
        output_size = 2 * hidden_size
        x = torch.randn(
            (rows, inner),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight = torch.randn(
            (inner, output_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight.add_(torch.where(weight >= 0, 0.25, -0.25))
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        residual24 = weight - weight24
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_gate_up_swiglu(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=inner,
        )
        dense_rows = torch.tensor(
            [1, 4, 7, 11, 14], device="cuda", dtype=torch.int32
        )
        dense_slots = torch.full(
            (rows,), -1, device="cuda", dtype=torch.int32
        )
        dense_slots[dense_rows.long()] = torch.arange(
            dense_rows.numel(), device="cuda", dtype=torch.int32
        )
        sparse_mask = dense_slots < 0

        full_gate_up = x @ weight24
        dense_residual = x[dense_rows.long()] @ residual24
        sparse_expected = torch.empty(
            (rows, hidden_size), device="cuda", dtype=torch.float16
        )
        torch.ops._C.silu_and_mul(sparse_expected, full_gate_up)
        exact_gate_up = full_gate_up.clone()
        exact_gate_up[dense_rows.long()] = (
            full_gate_up[dense_rows.long()] + dense_residual
        )
        exact_expected = torch.empty_like(sparse_expected)
        torch.ops._C.silu_and_mul(exact_expected, exact_gate_up)

        for config in ("256x64x64_s3", "256x64x64_s3_sw4"):
            for output_transposed in (False, True):
                with self.subTest(
                    config=config, output_transposed=output_transposed
                ):
                    if output_transposed:
                        out = torch.empty_strided(
                            sparse_expected.shape,
                            (1, rows),
                            device="cuda",
                            dtype=torch.float16,
                        )
                        dense_base = torch.empty_strided(
                            (dense_rows.numel(), output_size),
                            (1, dense_rows.numel()),
                            device="cuda",
                            dtype=torch.float16,
                        )
                        correction_residual = torch.empty_strided(
                            dense_residual.shape,
                            (1, dense_rows.numel()),
                            device="cuda",
                            dtype=torch.float16,
                        )
                        correction_residual.copy_(dense_residual)
                    else:
                        out = torch.empty_like(sparse_expected)
                        dense_base = torch.empty(
                            (dense_rows.numel(), output_size),
                            device="cuda",
                            dtype=torch.float16,
                        )
                        correction_residual = dense_residual
                    out.fill_(float("nan"))
                    dense_base.fill_(float("nan"))
                    actual, actual_base = (
                        sparse24_cutlass_routed_swiglu_prepacked(
                            x,
                            values,
                            meta,
                            dense_slots,
                            dense_count=dense_rows.numel(),
                            out=out,
                            dense_base=dense_base,
                            config=config,
                            output_transposed=output_transposed,
                        )
                    )
                    torch.cuda.synchronize()
                    self.assertTrue(
                        torch.allclose(
                            actual[sparse_mask],
                            sparse_expected[sparse_mask],
                            rtol=3e-2,
                            atol=1e-1,
                        )
                    )
                    self.assertTrue(
                        torch.allclose(
                            actual_base,
                            full_gate_up[dense_rows.long()],
                            rtol=2e-2,
                            atol=8e-2,
                        )
                    )
                    self.assertFalse(
                        torch.isfinite(actual[~sparse_mask]).any()
                    )

                    corrected = sparse24_routed_swiglu_correction_(
                        actual_base,
                        correction_residual,
                        dense_rows,
                        actual,
                    )
                    torch.cuda.synchronize()
                    self.assertEqual(corrected.data_ptr(), out.data_ptr())
                    self.assertTrue(
                        torch.allclose(
                            corrected, exact_expected, rtol=3e-2, atol=1e-1
                        )
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_indexed_and_dual_sparse_swiglu_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(8500)
        logical_rows = 7
        run_rows = 8
        output_rows = 19
        inner = 64
        hidden_size = 256
        output_size = 2 * hidden_size
        x = torch.zeros(
            (run_rows, inner), device="cuda", dtype=torch.float16
        )
        x[:logical_rows].normal_(generator=generator)
        weight = torch.randn(
            (inner, output_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        residual24 = weight - weight24
        assert_24_weight(weight24)
        assert_24_weight(residual24)

        def prepare(weight24: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            packed = pack_24(weight24, layout="n_major")
            return prepare_cutlass_sparse24_gate_up_swiglu(
                packed.values,
                packed.meta,
                layout=packed.layout,
                K=inner,
            )

        full_values, full_meta = prepare(weight24)
        residual_values, residual_meta = prepare(residual24)
        row_indices = torch.tensor(
            [18, 1, 7, 4, 15, 0, 11], device="cuda", dtype=torch.int32
        )

        def activation(gate_up: torch.Tensor) -> torch.Tensor:
            gate, up = gate_up.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate.float()).mul(up.float()).half()

        sparse_expected = activation(x[:logical_rows] @ weight24)
        dense_expected = activation(x[:logical_rows] @ weight)
        for config in (
            "256x32x64_s3_sw4",
            "256x64x64_s3_sw4",
        ):
            with self.subTest(config=config):
                indexed_out = torch.full(
                    (output_rows, hidden_size),
                    -123.0,
                    device="cuda",
                    dtype=torch.float16,
                )
                dual_out = torch.full_like(indexed_out, -123.0)
                sparse24_cutlass_indexed_swiglu_prepacked(
                    x,
                    full_values,
                    full_meta,
                    row_indices,
                    indexed_out,
                    config=config,
                )
                sparse24_cutlass_dual_swiglu_prepacked(
                    x,
                    full_values,
                    full_meta,
                    residual_values,
                    residual_meta,
                    row_indices,
                    dual_out,
                    config=config,
                )
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.allclose(
                        indexed_out[row_indices.long()],
                        sparse_expected,
                        rtol=3e-2,
                        atol=1e-1,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        dual_out[row_indices.long()],
                        dense_expected,
                        rtol=5e-2,
                        atol=2e-1,
                    ),
                    (
                        f"config={config} max_abs_diff="
                        f"{(dual_out[row_indices.long()].float() - dense_expected.float()).abs().max().item()}"
                    ),
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_residual_correction_swiglu_epilogue_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(8501)
        dense_count = 7
        dense_run = 8
        output_rows = 19
        inner = 64
        hidden_size = 256
        output_size = 2 * hidden_size
        dense_x = torch.zeros(
            (dense_run, inner), device="cuda", dtype=torch.float16
        )
        dense_x[:dense_count].normal_(generator=generator)
        residual_weight = torch.randn(
            (inner, output_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        residual_weight, _ = apply_random_24_mask(
            residual_weight, generator=generator
        )
        packed = pack_24(residual_weight, layout="n_major")
        residual_values, residual_meta = (
            prepare_cutlass_sparse24_gate_up_swiglu(
                packed.values,
                packed.meta,
                layout=packed.layout,
                K=inner,
            )
        )
        dense_base = torch.randn(
            (dense_count, output_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        dense_rows = torch.tensor(
            [18, 1, 7, 4, 15, 0, 11], device="cuda", dtype=torch.int32
        )
        corrected_gate_up = dense_base + (
            dense_x[:dense_count] @ residual_weight
        )
        gate, up = corrected_gate_up.chunk(2, dim=-1)
        corrected_hidden = torch.nn.functional.silu(gate.float()).mul(
            up.float()
        ).half()

        for config in (
            "256x32x64_s3_sw4",
            "256x64x64_s3_sw4",
        ):
            with self.subTest(config=config):
                out = torch.full(
                    (output_rows, hidden_size),
                    -123.0,
                    device="cuda",
                    dtype=torch.float16,
                )
                dense_hidden = torch.full(
                    (dense_run, hidden_size),
                    -321.0,
                    device="cuda",
                    dtype=torch.float16,
                )
                sparse24_cutlass_residual_correction_swiglu_prepacked(
                    dense_x,
                    residual_values,
                    residual_meta,
                    dense_base,
                    dense_rows,
                    out,
                    dense_hidden=dense_hidden,
                    config=config,
                )
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.allclose(
                        out[dense_rows.long()],
                        corrected_hidden,
                        rtol=3e-2,
                        atol=1e-1,
                    ),
                    (
                        f"config={config} max_abs_diff="
                        f"{(out[dense_rows.long()].float() - corrected_hidden.float()).abs().max().item()}"
                    ),
                )
                sparse_mask = torch.ones(
                    output_rows, device="cuda", dtype=torch.bool
                )
                sparse_mask[dense_rows.long()] = False
                self.assertTrue(
                    torch.equal(
                        out[sparse_mask],
                        torch.full_like(out[sparse_mask], -123.0),
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        dense_hidden[:dense_count],
                        corrected_hidden,
                        rtol=3e-2,
                        atol=1e-1,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        dense_hidden[dense_count:],
                        torch.full_like(
                            dense_hidden[dense_count:],
                            -321.0,
                        ),
                    )
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_routed_sparse_swiglu_supports_all_dense_rows_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(83)
        rows = 16
        inner = 64
        hidden_size = 256
        output_size = 2 * hidden_size
        x = torch.randn(
            (rows, inner),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight = torch.randn(
            (inner, output_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight.add_(torch.where(weight >= 0, 0.25, -0.25))
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        residual24 = weight - weight24
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_gate_up_swiglu(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=inner,
        )
        dense_rows = torch.arange(rows, device="cuda", dtype=torch.int32)
        dense_residual = x @ residual24
        out = torch.full(
            (rows, hidden_size),
            float("nan"),
            device="cuda",
            dtype=torch.float16,
        )
        dense_base = torch.empty(
            (rows, output_size), device="cuda", dtype=torch.float16
        )

        sparse24_cutlass_routed_swiglu_prepacked(
            x,
            values,
            meta,
            dense_rows,
            dense_count=rows,
            out=out,
            dense_base=dense_base,
            config="256x64x64_s3_sw4",
        )
        sparse24_routed_swiglu_correction_(
            dense_base,
            dense_residual,
            dense_rows,
            out,
        )
        expected_gate_up = x @ weight
        expected = torch.empty_like(out)
        torch.ops._C.silu_and_mul(expected, expected_gate_up)
        torch.cuda.synchronize()

        self.assertTrue(
            torch.allclose(out, expected, rtol=3e-2, atol=1e-1)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_paired_persistent_routed_swiglu_matches_reference_cuda(
        self,
    ) -> None:
        generator = torch.Generator(device="cuda").manual_seed(87)
        rows = 16
        dense_count = 5
        dense_run = 8
        inner = 64
        hidden_size = 256
        output_size = 2 * hidden_size
        x = torch.randn(
            (rows, inner),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight = torch.randn(
            (inner, output_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight.add_(torch.where(weight >= 0, 0.25, -0.25))
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        residual24 = weight - weight24
        full_packed = pack_24(weight24, layout="n_major")
        full_values, full_meta = prepare_cutlass_sparse24_gate_up_swiglu(
            full_packed.values,
            full_packed.meta,
            layout=full_packed.layout,
            K=inner,
        )
        residual_packed = pack_24(residual24, layout="n_major")
        residual_values, residual_meta = prepare_cutlass_sparse24_device_gemm(
            residual_packed.values,
            residual_packed.meta,
            layout=residual_packed.layout,
            K=inner,
        )
        dense_rows = torch.tensor(
            [1, 4, 7, 11, 14], device="cuda", dtype=torch.int32
        )
        dense_slots = torch.full(
            (rows,), -1, device="cuda", dtype=torch.int32
        )
        dense_slots[dense_rows.long()] = torch.arange(
            dense_count, device="cuda", dtype=torch.int32
        )
        residual_x = torch.zeros(
            (dense_run, inner), device="cuda", dtype=torch.float16
        )
        residual_x[:dense_count].copy_(x[dense_rows.long()])
        out = torch.full(
            (rows, hidden_size),
            float("nan"),
            device="cuda",
            dtype=torch.float16,
        )
        dense_base = torch.empty(
            (dense_count, output_size), device="cuda", dtype=torch.float16
        )
        residual_out = torch.empty(
            (dense_run, output_size), device="cuda", dtype=torch.float16
        )

        actual_by_schedule: dict[str, torch.Tensor] = {}
        for schedule in ("partitioned", "interleaved"):
            out.fill_(float("nan"))
            sparse24_cutlass_paired_persistent_routed_swiglu_prepacked(
                x,
                full_values,
                full_meta,
                dense_slots,
                residual_x,
                residual_values,
                residual_meta,
                dense_count=dense_count,
                full_out=out,
                dense_base=dense_base,
                residual_out=residual_out,
                schedule=schedule,
            )
            sparse24_routed_swiglu_correction_(
                dense_base,
                residual_out[:dense_count],
                dense_rows,
                out,
            )
            actual_by_schedule[schedule] = out.clone()
        expected_gate_up = x @ weight24
        expected_gate_up[dense_rows.long()] = x[dense_rows.long()] @ weight
        expected = torch.empty_like(out)
        torch.ops._C.silu_and_mul(expected, expected_gate_up)
        torch.cuda.synchronize()

        sparse_rows = dense_slots < 0
        dense_rows_long = dense_rows.long()
        for schedule, actual in actual_by_schedule.items():
            max_diff = float(
                (actual.float() - expected.float()).abs().max().item()
            )
            sparse_diff = float(
                (
                    actual[sparse_rows].float()
                    - expected[sparse_rows].float()
                )
                .abs()
                .max()
                .item()
            )
            dense_diff = float(
                (
                    actual[dense_rows_long].float()
                    - expected[dense_rows_long].float()
                )
                .abs()
                .max()
                .item()
            )
            self.assertTrue(
                torch.allclose(actual, expected, rtol=3e-2, atol=1e-1),
                f"{schedule}: max={max_diff}, sparse={sparse_diff}, "
                f"dense={dense_diff}",
            )

        for config in (
            "256x32x64_s3_sw4",
            "256x64x64_s2_sw4",
        ):
            out.fill_(float("nan"))
            sparse24_cutlass_paired_persistent_routed_swiglu_prepacked(
                x,
                full_values,
                full_meta,
                dense_slots,
                residual_x,
                residual_values,
                residual_meta,
                dense_count=dense_count,
                full_out=out,
                dense_base=dense_base,
                residual_out=residual_out,
                schedule="interleaved",
                config=config,
                worker_blocks=2,
            )
            sparse24_routed_swiglu_correction_(
                dense_base,
                residual_out[:dense_count],
                dense_rows,
                out,
            )
            torch.cuda.synchronize()
            self.assertTrue(
                torch.allclose(out, expected, rtol=3e-2, atol=1e-1),
                config,
            )

        with self.assertRaisesRegex(ValueError, r"0 \(auto\) or at least 2"):
            sparse24_cutlass_paired_persistent_routed_swiglu_prepacked(
                x,
                full_values,
                full_meta,
                dense_slots,
                residual_x,
                residual_values,
                residual_meta,
                dense_count=dense_count,
                worker_blocks=1,
            )

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            sparse24_cutlass_paired_persistent_routed_swiglu_prepacked(
                x,
                full_values,
                full_meta,
                dense_slots,
                residual_x,
                residual_values,
                residual_meta,
                dense_count=dense_count,
                full_out=out,
                dense_base=dense_base,
                residual_out=residual_out,
                schedule="interleaved",
            )
            sparse24_routed_swiglu_correction_(
                dense_base,
                residual_out[:dense_count],
                dense_rows,
                out,
            )

        updated_x = torch.randn(
            x.shape,
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        x.copy_(updated_x)
        residual_x[:dense_count].copy_(updated_x[dense_rows_long])
        graph.replay()
        expected_gate_up = updated_x @ weight24
        expected_gate_up[dense_rows_long] = (
            updated_x[dense_rows_long] @ weight
        )
        torch.ops._C.silu_and_mul(expected, expected_gate_up)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.allclose(out, expected, rtol=3e-2, atol=1e-1)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_routed_swiglu_correction_gathers_and_zeros_padding_cuda(
        self,
    ) -> None:
        generator = torch.Generator(device="cuda").manual_seed(89)
        output_rows = 16
        dense_count = 5
        dense_run = 8
        hidden_size = 256
        dense_rows = torch.tensor(
            [1, 4, 7, 11, 14], device="cuda", dtype=torch.int32
        )
        dense_base = torch.randn(
            (dense_count, 2 * hidden_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        dense_residual = torch.randn(
            dense_base.shape,
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        expected = torch.empty(
            (dense_count, hidden_size),
            device="cuda",
            dtype=torch.float16,
        )
        torch.ops._C.silu_and_mul(expected, dense_base + dense_residual)
        out = torch.full(
            (output_rows, hidden_size),
            -7.0,
            device="cuda",
            dtype=torch.float16,
        )
        dense_hidden = torch.full(
            (dense_run, hidden_size),
            float("nan"),
            device="cuda",
            dtype=torch.float16,
        )

        actual_out, actual_compact = (
            sparse24_routed_swiglu_correction_gather_(
                dense_base,
                dense_residual,
                dense_rows,
                out,
                dense_hidden,
            )
        )
        torch.cuda.synchronize()

        self.assertEqual(actual_out.data_ptr(), out.data_ptr())
        self.assertEqual(actual_compact.data_ptr(), dense_hidden.data_ptr())
        self.assertTrue(
            torch.allclose(
                actual_out[dense_rows.long()], expected, rtol=1e-3, atol=1e-3
            )
        )
        self.assertTrue(
            torch.allclose(
                actual_compact[:dense_count], expected, rtol=1e-3, atol=1e-3
            )
        )
        self.assertTrue(
            torch.equal(
                actual_compact[dense_count:],
                torch.zeros_like(actual_compact[dense_count:]),
            )
        )
        sparse_rows = torch.tensor(
            [0, 2, 3, 5, 6, 8, 9, 10, 12, 13, 15],
            device="cuda",
        )
        self.assertTrue(
            torch.equal(
                actual_out[sparse_rows],
                torch.full_like(actual_out[sparse_rows], -7.0),
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_inline_sparse_qkv_postop_matches_vllm_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(73)
        inner = 64
        q_size = 256
        kv_size = 128
        head_dim = 128
        output_size = q_size + 2 * kv_size
        weight = torch.randn(
            (inner, output_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight.add_(torch.where(weight >= 0, 0.25, -0.25))
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_device_gemm(
            packed.values, packed.meta, layout=packed.layout, K=inner
        )
        q_weight = torch.randn(
            (head_dim,), device="cuda", dtype=torch.float16, generator=generator
        )
        k_weight = torch.randn(
            (head_dim,), device="cuda", dtype=torch.float16, generator=generator
        )
        angles = torch.randn(
            (256, head_dim // 2),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).half()
        for rows in (16, 128):
            x = torch.randn(
                (rows, inner),
                device="cuda",
                dtype=torch.float16,
                generator=generator,
            )
            position_ids = torch.arange(rows, device="cuda", dtype=torch.int64)
            expected = x @ weight24
            vllm_ops.fused_qk_norm_rope(
                expected,
                q_size // head_dim,
                kv_size // head_dim,
                kv_size // head_dim,
                head_dim,
                1e-6,
                q_weight,
                k_weight,
                cos_sin_cache,
                True,
                position_ids,
                1,
            )
            for config in (
                "128x32x64_s4",
                "128x32x64_s4_sw2",
                "128x32x64_s4_sw4",
                "128x64x64_s5",
                "256x64x64_s3",
                "256x64x64_s3_sw4",
            ):
                with self.subTest(rows=rows, config=config):
                    out = torch.full_like(expected, float("nan"))
                    actual = sparse24_cutlass_qkv_postop_prepacked(
                        x,
                        values,
                        meta,
                        q_weight,
                        k_weight,
                        cos_sin_cache,
                        position_ids,
                        q_size=q_size,
                        kv_size=kv_size,
                        out=out,
                        config=config,
                    )
                    torch.cuda.synchronize()
                    self.assertEqual(actual.data_ptr(), out.data_ptr())
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertTrue(
                        torch.allclose(actual, expected, rtol=2e-2, atol=8e-2)
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_silu_and_mul_preserves_transposed_layout_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(23)
        logical = torch.randn(
            (13, 64), device="cuda", dtype=torch.float16, generator=generator
        )
        gate_up = torch.empty_strided(
            (13, 64), (1, 16), device="cuda", dtype=torch.float16
        )
        gate_up.copy_(logical)

        actual = sparse24_silu_and_mul_transposed(gate_up)
        expected = torch.empty((13, 32), device="cuda", dtype=torch.float16)
        torch.ops._C.silu_and_mul(expected, logical)
        torch.cuda.synchronize()

        self.assertEqual(tuple(actual.shape), (13, 32))
        self.assertEqual(tuple(actual.stride()), (1, 16))
        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_silu_and_mul_materializes_contiguous_layout_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(29)
        logical = torch.randn(
            (13, 64), device="cuda", dtype=torch.float16, generator=generator
        )
        gate_up = torch.empty_strided(
            (13, 64), (1, 16), device="cuda", dtype=torch.float16
        )
        gate_up.copy_(logical)

        actual = sparse24_silu_and_mul_transposed_to_contiguous(gate_up)
        expected = torch.empty((13, 32), device="cuda", dtype=torch.float16)
        torch.ops._C.silu_and_mul(expected, logical)
        torch.cuda.synchronize()

        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_gather_and_materialize_transposed_rows_cuda(self) -> None:
        logical = torch.arange(
            16 * 64,
            device="cuda",
            dtype=torch.float16,
        ).reshape(16, 64)
        transposed = torch.empty_strided(
            (16, 64),
            (1, 16),
            device="cuda",
            dtype=torch.float16,
        )
        transposed.copy_(logical)
        rows = torch.tensor(
            [0, 2, 4, 6, 9, 11, 13, 15],
            device="cuda",
            dtype=torch.int32,
        )
        gathered = torch.empty_strided(
            (8, 64),
            (1, 8),
            device="cuda",
            dtype=torch.float16,
        )

        sparse24_gather_rows_strided_(transposed, rows, gathered)
        materialized = sparse24_transpose_output_contiguous(transposed)
        roundtrip = sparse24_transpose_input_to_strided(logical)
        torch.cuda.synchronize()

        self.assertEqual(tuple(gathered.stride()), (1, 8))
        self.assertTrue(torch.equal(gathered, logical[rows.long()]))
        self.assertTrue(materialized.is_contiguous())
        self.assertTrue(torch.equal(materialized, logical))
        self.assertEqual(tuple(roundtrip.stride()), (1, 16))
        self.assertTrue(torch.equal(roundtrip, logical))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_transpose_add_residual_matches_reference_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(53)
        logical = torch.randn(
            (13, 128),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        transposed = torch.empty_strided(
            logical.shape,
            (1, 16),
            device="cuda",
            dtype=torch.float16,
        )
        transposed.copy_(logical)
        residual = torch.randn(
            logical.shape,
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        expected = residual + logical

        result = sparse24_transpose_add_residual_(transposed, residual)
        torch.cuda.synchronize()

        self.assertIsNone(result)
        self.assertTrue(torch.equal(residual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_transpose_add_rmsnorm_configs_match_reference_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(59)
        logical = torch.randn(
            (13, 128),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        transposed = torch.empty_strided(
            logical.shape,
            (1, 16),
            device="cuda",
            dtype=torch.float16,
        )
        transposed.copy_(logical)
        residual_initial = torch.randn(
            logical.shape,
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight = torch.randn(
            (128,), device="cuda", dtype=torch.float16, generator=generator
        )
        expected = logical.clone()
        expected_residual = residual_initial.clone()
        vllm_ops.fused_add_rms_norm(
            expected,
            expected_residual,
            weight,
            1e-6,
        )

        for config in ("2", "4", "8", "16", "32"):
            with self.subTest(config=config):
                residual = residual_initial.clone()
                actual = sparse24_transpose_add_rmsnorm(
                    transposed,
                    residual,
                    weight,
                    epsilon=1e-6,
                    epilogue_config=config,
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(residual, expected_residual))
                self.assertTrue(
                    torch.allclose(actual, expected, rtol=2e-3, atol=2e-3)
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_qkv_transpose_rmsnorm_matches_reference_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(31)
        rows = 13
        q_size = 256
        kv_size = 128
        head_dim = 128
        logical = torch.randn(
            (rows, q_size + 2 * kv_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        q_weight = torch.randn(
            (head_dim,), device="cuda", dtype=torch.float16, generator=generator
        )
        k_weight = torch.randn(
            (head_dim,), device="cuda", dtype=torch.float16, generator=generator
        )
        transposed = torch.empty_strided(
            logical.shape,
            (1, 16),
            device="cuda",
            dtype=torch.float16,
        )
        transposed.copy_(logical)

        actual = sparse24_qkv_transpose_rmsnorm(
            transposed,
            q_weight,
            k_weight,
            q_size=q_size,
            kv_size=kv_size,
            head_dim=head_dim,
            epsilon=1e-6,
        )
        expected = _qkv_rmsnorm_reference(
            logical,
            q_weight,
            k_weight,
            q_size=q_size,
            kv_size=kv_size,
            head_dim=head_dim,
            epsilon=1e-6,
        )
        torch.cuda.synchronize()

        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.allclose(actual, expected, rtol=2e-3, atol=2e-3))
        self.assertTrue(torch.equal(actual[:, q_size + kv_size :], logical[:, q_size + kv_size :]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_qkv_transpose_postop_matches_vllm_fused_norm_rope_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(43)
        rows = 13
        q_size = 256
        kv_size = 128
        head_dim = 128
        logical = torch.randn(
            (rows, q_size + 2 * kv_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        q_weight = torch.randn(
            (head_dim,), device="cuda", dtype=torch.float16, generator=generator
        )
        k_weight = torch.randn(
            (head_dim,), device="cuda", dtype=torch.float16, generator=generator
        )
        angles = torch.randn(
            (64, head_dim // 2),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).half()
        position_ids = torch.arange(rows, device="cuda", dtype=torch.int64)
        transposed = torch.empty_strided(
            logical.shape,
            (1, 16),
            device="cuda",
            dtype=torch.float16,
        )
        transposed.copy_(logical)

        expected = logical.clone()
        vllm_ops.fused_qk_norm_rope(
            expected,
            q_size // head_dim,
            kv_size // head_dim,
            kv_size // head_dim,
            head_dim,
            1e-6,
            q_weight,
            k_weight,
            cos_sin_cache,
            True,
            position_ids,
            1,
        )
        actual = sparse24_qkv_transpose_postop(
            transposed,
            cos_sin_cache,
            position_ids,
            q_size=q_size,
            kv_size=kv_size,
            head_dim=head_dim,
            epsilon=1e-6,
            is_neox=True,
            q_weight=q_weight,
            k_weight=k_weight,
        )
        torch.cuda.synchronize()

        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.allclose(actual, expected, rtol=3e-3, atol=3e-3))
        self.assertTrue(
            torch.equal(
                actual[:, q_size + kv_size :],
                logical[:, q_size + kv_size :],
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_qkv_transpose_postop_matches_vllm_rope_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(47)
        rows = 17
        q_size = 256
        kv_size = 128
        head_dim = 128
        logical = torch.randn(
            (rows, q_size + 2 * kv_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        angles = torch.randn(
            (64, head_dim // 2),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).half()
        position_ids = torch.arange(rows, device="cuda", dtype=torch.int64)
        transposed = torch.empty_strided(
            logical.shape,
            (1, 24),
            device="cuda",
            dtype=torch.float16,
        )
        transposed.copy_(logical)

        expected = logical.clone()
        q, k, _v = expected.split((q_size, kv_size, kv_size), dim=-1)
        vllm_ops.rotary_embedding(
            position_ids,
            q,
            k,
            head_dim,
            cos_sin_cache,
            True,
        )
        actual = sparse24_qkv_transpose_postop(
            transposed,
            cos_sin_cache,
            position_ids,
            q_size=q_size,
            kv_size=kv_size,
            head_dim=head_dim,
            epsilon=0.0,
            is_neox=True,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.allclose(actual, expected, rtol=3e-3, atol=3e-3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_qkv_rmsnorm_inplace_matches_reference_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(37)
        rows = 17
        q_size = 256
        kv_size = 128
        head_dim = 128
        qkv = torch.randn(
            (rows, q_size + 2 * kv_size),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        original = qkv.clone()
        q_weight = torch.randn(
            (head_dim,), device="cuda", dtype=torch.float16, generator=generator
        )
        k_weight = torch.randn(
            (head_dim,), device="cuda", dtype=torch.float16, generator=generator
        )
        expected = _qkv_rmsnorm_reference(
            original,
            q_weight,
            k_weight,
            q_size=q_size,
            kv_size=kv_size,
            head_dim=head_dim,
            epsilon=1e-6,
        )

        result = sparse24_qkv_rmsnorm_inplace_(
            qkv,
            q_weight,
            k_weight,
            q_size=q_size,
            kv_size=kv_size,
            head_dim=head_dim,
            epsilon=1e-6,
        )
        torch.cuda.synchronize()

        self.assertIsNone(result)
        self.assertTrue(torch.allclose(qkv, expected, rtol=2e-3, atol=2e-3))
        self.assertTrue(torch.equal(qkv[:, q_size + kv_size :], original[:, q_size + kv_size :]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cutlass_prepacked_gemm_accepts_transposed_sparse_input_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(19)
        x = torch.randn((8, 64), device="cuda", dtype=torch.float16, generator=generator)
        weight1 = torch.randn((64, 64), device="cuda", dtype=torch.float16, generator=generator)
        weight2 = torch.randn((64, 32), device="cuda", dtype=torch.float16, generator=generator)
        weight1_24, _meta1 = apply_random_24_mask(weight1, generator=generator)
        weight2_24, _meta2 = apply_random_24_mask(weight2, generator=generator)

        packed1 = pack_24(weight1_24, layout="n_major")
        values1, meta1 = prepare_cutlass_sparse24_device_gemm(
            packed1.values,
            packed1.meta,
            layout=packed1.layout,
            K=64,
        )
        packed2 = pack_24(weight2_24, layout="n_major")
        values2, meta2 = prepare_cutlass_sparse24_device_gemm(
            packed2.values,
            packed2.meta,
            layout=packed2.layout,
            K=64,
        )

        hidden = sparse24_cutlass_device_gemm_prepacked(x, values1, meta1)
        actual = sparse24_cutlass_device_gemm_prepacked(
            hidden,
            values2,
            meta2,
            input_transposed=True,
            contiguous_output=True,
        )
        expected = (x @ weight1_24) @ weight2_24
        torch.cuda.synchronize()

        self.assertFalse(hidden.is_contiguous())
        self.assertEqual(tuple(hidden.stride()), (1, 8))
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.allclose(actual, expected, rtol=3e-2, atol=1.5e-1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cutlass_mixed_dense_override_matches_reference_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(11)
        x = torch.randn((8, 64), device="cuda", dtype=torch.float16, generator=generator)
        weight = torch.randn((64, 32), device="cuda", dtype=torch.float16, generator=generator)
        weight24, _meta = apply_random_24_mask(weight, generator=generator)
        assert_24_weight(weight24)
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=64,
        )
        dense_rows = torch.tensor([1, 4], device="cuda", dtype=torch.int32)

        actual = sparse24_mixed_dense_override_prepacked(
            x,
            weight.t().contiguous(),
            values,
            meta,
            dense_rows,
        )
        expected = x @ weight24
        expected[dense_rows.to(dtype=torch.long)] = (
            x.index_select(0, dense_rows.to(dtype=torch.long)) @ weight
        )
        torch.cuda.synchronize()

        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.allclose(actual, expected, rtol=2e-2, atol=8e-2))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cutlass_mixed_dense_override_reuses_workspace_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(17)
        x = torch.randn((8, 64), device="cuda", dtype=torch.float16, generator=generator)
        weight = torch.randn((64, 32), device="cuda", dtype=torch.float16, generator=generator)
        weight24, _meta = apply_random_24_mask(weight, generator=generator)
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=64,
        )
        dense_rows = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
        out = torch.empty((8, 32), device="cuda", dtype=torch.float16)
        dense_x = torch.empty((3, 64), device="cuda", dtype=torch.float16)
        dense_y = torch.empty((3, 32), device="cuda", dtype=torch.float16)
        workspace = torch.empty((32, 8), device="cuda", dtype=torch.float16)

        actual = sparse24_mixed_dense_override_prepacked(
            x,
            weight.t().contiguous(),
            values,
            meta,
            dense_rows,
            out=out,
            dense_x=dense_x,
            dense_y=dense_y,
            workspace=workspace,
        )
        expected = x @ weight24
        expected[dense_rows.to(dtype=torch.long)] = (
            x.index_select(0, dense_rows.to(dtype=torch.long)) @ weight
        )
        torch.cuda.synchronize()

        self.assertEqual(actual.data_ptr(), out.data_ptr())
        self.assertTrue(torch.allclose(actual, expected, rtol=2e-2, atol=8e-2))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_parallel_mixed_override_graph_replay_reads_updated_rows_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(19)
        x = torch.randn(
            (16, 64), device="cuda", dtype=torch.float16, generator=generator
        )
        weight = torch.randn(
            (64, 32), device="cuda", dtype=torch.float16, generator=generator
        )
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=64,
        )
        dense_rows = torch.tensor(
            [0, 3, 7, 15], device="cuda", dtype=torch.int32
        )
        sparse_rows = torch.tensor(
            [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14],
            device="cuda",
            dtype=torch.int32,
        )
        summary = (
            torch.zeros(16, device="cuda", dtype=torch.bool),
            4,
            dense_rows,
            sparse_rows,
        )
        self.assertTrue(
            speclink_linear.prepare_mixed_linear_streams(torch.device("cuda"))
        )
        graph = torch.cuda.CUDAGraph()
        with (
            mock.patch.object(
                speclink_linear,
                "current_verify_dense_row_summary",
                return_value=summary,
            ),
            mock.patch.object(
                speclink_linear,
                "current_verify_contiguous_dense_prefix",
                return_value=-1,
            ),
            mock.patch.object(
                speclink_linear,
                "_PARALLEL_MIXED_OVERRIDE_ENABLED",
                True,
            ),
        ):
            for _ in range(2):
                speclink_linear._mixed_dense_override_linear_impl(
                    x, weight.t().contiguous(), values, meta
                )
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                actual = speclink_linear._mixed_dense_override_linear_impl(
                    x, weight.t().contiguous(), values, meta
                )

        new_dense_rows = torch.tensor(
            [1, 4, 8, 12], device="cuda", dtype=torch.int32
        )
        dense_rows.copy_(new_dense_rows)
        graph.replay()
        torch.cuda.synchronize()

        expected = x @ weight24
        expected[new_dense_rows.long()] = x[new_dense_rows.long()] @ weight
        self.assertTrue(
            torch.allclose(actual, expected, rtol=2e-2, atol=8e-2)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_parallel_split_linear_matches_reference_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(23)
        x = torch.randn(
            (13, 64),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight = torch.randn(
            (64, 32),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=64,
        )
        dense_rows = torch.tensor(
            [0, 3, 7],
            device="cuda",
            dtype=torch.int32,
        )
        sparse_rows = torch.tensor(
            [1, 2, 4, 5, 6, 8, 9, 10, 11, 12],
            device="cuda",
            dtype=torch.int32,
        )
        summary = (
            torch.zeros(13, device="cuda", dtype=torch.bool),
            3,
            dense_rows,
            sparse_rows,
        )
        self.assertTrue(
            speclink_linear.prepare_mixed_linear_streams(
                torch.device("cuda")
            )
        )

        expected = x @ weight24
        dense_rows_long = dense_rows.long()
        expected[dense_rows_long] = x[dense_rows_long] @ weight
        for indexed_output in (False, True):
            with (
                mock.patch.object(
                    speclink_linear,
                    "current_verify_dense_row_summary",
                    return_value=summary,
                ),
                mock.patch.object(
                    speclink_linear,
                    "_INDEXED_OUTPUT_EPILOGUE_ENABLED",
                    indexed_output,
                ),
            ):
                actual = speclink_linear._split_dense_sparse_linear_impl(
                    x,
                    weight.t().contiguous(),
                    values,
                    meta,
                )
            torch.cuda.synchronize()

            with self.subTest(indexed_output=indexed_output):
                self.assertTrue(actual.is_contiguous())
                self.assertTrue(
                    torch.allclose(actual, expected, rtol=2e-2, atol=8e-2)
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_parallel_split_graph_replay_reads_updated_rows_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(41)
        x = torch.randn(
            (16, 64),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight = torch.randn(
            (64, 32),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        weight24, _ = apply_random_24_mask(weight, generator=generator)
        packed = pack_24(weight24, layout="n_major")
        values, meta = prepare_cutlass_sparse24_device_gemm(
            packed.values,
            packed.meta,
            layout=packed.layout,
            K=64,
        )
        dense_rows = torch.tensor(
            [0, 3, 7, 15],
            device="cuda",
            dtype=torch.int32,
        )
        sparse_rows = torch.tensor(
            [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14],
            device="cuda",
            dtype=torch.int32,
        )
        summary = (
            torch.zeros(16, device="cuda", dtype=torch.bool),
            4,
            dense_rows,
            sparse_rows,
        )
        self.assertTrue(
            speclink_linear.prepare_mixed_linear_streams(
                torch.device("cuda")
            )
        )
        graph = torch.cuda.CUDAGraph()
        with (
            mock.patch.object(
                speclink_linear,
                "current_verify_dense_row_summary",
                return_value=summary,
            ),
            mock.patch.object(
                speclink_linear,
                "_INDEXED_OUTPUT_EPILOGUE_ENABLED",
                True,
            ),
        ):
            for _ in range(2):
                speclink_linear._split_dense_sparse_linear_impl(
                    x,
                    weight.t().contiguous(),
                    values,
                    meta,
                )
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                actual = speclink_linear._split_dense_sparse_linear_impl(
                    x,
                    weight.t().contiguous(),
                    values,
                    meta,
                )

        new_dense_rows = torch.tensor(
            [1, 4, 8, 12],
            device="cuda",
            dtype=torch.int32,
        )
        new_sparse_rows = torch.tensor(
            [0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15],
            device="cuda",
            dtype=torch.int32,
        )
        dense_rows.copy_(new_dense_rows)
        sparse_rows.copy_(new_sparse_rows)
        graph.replay()
        torch.cuda.synchronize()

        expected = x @ weight24
        dense_rows_long = new_dense_rows.long()
        expected[dense_rows_long] = x[dense_rows_long] @ weight
        self.assertTrue(
            torch.allclose(actual, expected, rtol=2e-2, atol=8e-2)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_mixed_row_mlp_matches_dense_sparse_reference_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(29)
        x = torch.randn(
            (8, 64), device="cuda", dtype=torch.float16, generator=generator
        )
        gate_weight = torch.randn(
            (64, 128), device="cuda", dtype=torch.float16, generator=generator
        )
        down_weight = torch.randn(
            (64, 64), device="cuda", dtype=torch.float16, generator=generator
        )
        gate_weight24, _ = apply_random_24_mask(
            gate_weight, generator=generator
        )
        down_weight24, _ = apply_random_24_mask(
            down_weight, generator=generator
        )

        gate_packed = pack_24(gate_weight24, layout="n_major")
        gate_values, gate_meta = prepare_cutlass_sparse24_device_gemm(
            gate_packed.values,
            gate_packed.meta,
            layout=gate_packed.layout,
            K=64,
        )
        down_packed = pack_24(down_weight24, layout="n_major")
        down_values, down_meta = prepare_cutlass_sparse24_device_gemm(
            down_packed.values,
            down_packed.meta,
            layout=down_packed.layout,
            K=64,
        )
        dense_rows = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
        sparse_rows = torch.tensor(
            [1, 2, 4, 5, 6], device="cuda", dtype=torch.int32
        )
        summary = (
            torch.tensor(
                [True, False, False, True, False, False, False, True],
                device="cuda",
            ),
            3,
            dense_rows,
            sparse_rows,
        )
        self.assertTrue(
            speclink_mlp.prepare_mixed_mlp_streams(torch.device("cuda"))
        )
        expected_gate = x @ gate_weight24
        dense_rows_long = dense_rows.to(dtype=torch.long)
        expected_gate[dense_rows_long] = x[dense_rows_long] @ gate_weight
        expected_hidden = torch.empty(
            (8, 64), device="cuda", dtype=torch.float16
        )
        torch.ops._C.silu_and_mul(expected_hidden, expected_gate)
        expected = expected_hidden @ down_weight24
        expected[dense_rows_long] = (
            expected_hidden[dense_rows_long] @ down_weight
        )

        for indexed_down in (False, True):
            with (
                mock.patch.object(
                    speclink_mlp,
                    "current_verify_dense_row_summary",
                    return_value=summary,
                ),
                mock.patch.object(
                    speclink_mlp,
                    "_INDEXED_DOWN_EPILOGUE_ENABLED",
                    indexed_down,
                ),
            ):
                actual = speclink_mlp._batch_routed_mlp_impl(
                    x,
                    gate_weight.t().contiguous(),
                    down_weight.t().contiguous(),
                    gate_values,
                    gate_meta,
                    down_values,
                    down_meta,
                )
            torch.cuda.synchronize()

            with self.subTest(indexed_down=indexed_down):
                self.assertTrue(actual.is_contiguous())
                self.assertTrue(
                    torch.allclose(
                        actual,
                        expected,
                        rtol=3e-2,
                        atol=2e-1,
                    )
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_inline_swiglu_batch_routed_mlp_matches_reference_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(79)
        gate_weight = torch.randn(
            (64, 256), device="cuda", dtype=torch.float16, generator=generator
        )
        down_weight = torch.randn(
            (128, 64), device="cuda", dtype=torch.float16, generator=generator
        )
        gate_weight.mul_(0.02)
        down_weight.mul_(0.02)
        gate_weight.add_(torch.where(gate_weight >= 0, 0.005, -0.005))
        down_weight.add_(torch.where(down_weight >= 0, 0.005, -0.005))
        gate_weight24, _ = apply_random_24_mask(
            gate_weight, generator=generator
        )
        down_weight24, _ = apply_random_24_mask(
            down_weight, generator=generator
        )
        gate_packed = pack_24(gate_weight24, layout="n_major")
        gate_values, gate_meta = prepare_cutlass_sparse24_gate_up_swiglu(
            gate_packed.values,
            gate_packed.meta,
            layout=gate_packed.layout,
            K=64,
        )
        down_packed = pack_24(down_weight24, layout="n_major")
        down_values, down_meta = prepare_cutlass_sparse24_device_gemm(
            down_packed.values,
            down_packed.meta,
            layout=down_packed.layout,
            K=128,
        )

        for rows, dense_indices in ((8, (0, 3, 7)), (13, ())):
            x = torch.randn(
                (rows, 64),
                device="cuda",
                dtype=torch.float16,
                generator=generator,
            )
            dense_rows = torch.tensor(
                dense_indices, device="cuda", dtype=torch.int32
            )
            dense_set = set(dense_indices)
            sparse_indices = [i for i in range(rows) if i not in dense_set]
            sparse_rows = torch.tensor(
                sparse_indices, device="cuda", dtype=torch.int32
            )
            summary = (
                torch.zeros(rows, device="cuda", dtype=torch.bool),
                len(dense_indices),
                dense_rows,
                sparse_rows,
            )
            with (
                mock.patch.object(
                    speclink_mlp, "_INLINE_SWIGLU_MLP_ENABLED", True
                ),
                mock.patch.object(
                    speclink_mlp,
                    "current_verify_dense_row_summary",
                    return_value=summary,
                ),
                mock.patch.object(
                    speclink_mlp, "_mixed_mlp_streams", return_value=None
                ),
            ):
                actual = speclink_mlp._batch_routed_mlp_impl(
                    x,
                    gate_weight.t().contiguous(),
                    down_weight.t().contiguous(),
                    gate_values,
                    gate_meta,
                    down_values,
                    down_meta,
                )

            expected_gate = x @ gate_weight24
            if dense_indices:
                dense_rows_long = dense_rows.long()
                expected_gate[dense_rows_long] = (
                    x[dense_rows_long] @ gate_weight
                )
            expected_hidden = torch.empty(
                (rows, 128), device="cuda", dtype=torch.float16
            )
            torch.ops._C.silu_and_mul(expected_hidden, expected_gate)
            expected = expected_hidden @ down_weight24
            if dense_indices:
                expected[dense_rows_long] = (
                    expected_hidden[dense_rows_long] @ down_weight
                )
            torch.cuda.synchronize()

            with self.subTest(rows=rows, dense_rows=len(dense_indices)):
                self.assertTrue(actual.is_contiguous())
                close = torch.allclose(
                    actual, expected, rtol=3e-2, atol=2e-1
                )
                if not close:
                    difference = (actual.float() - expected.float()).abs()
                    self.fail(
                        f"inline batch MLP mismatch: max="
                        f"{float(difference.max().item())}, mean="
                        f"{float(difference.mean().item())}, "
                        f"actual_abs_max={float(actual.float().abs().max().item())}, "
                        f"expected_abs_max="
                        f"{float(expected.float().abs().max().item())}"
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_partition_and_merge_rows_roundtrip_cuda(self) -> None:
        x = torch.arange(8 * 64, device="cuda", dtype=torch.float16).reshape(8, 64)
        dense_rows = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
        sparse_rows = torch.tensor(
            [1, 2, 4, 5, 6], device="cuda", dtype=torch.int32
        )
        dense_x = torch.empty((3, 64), device="cuda", dtype=torch.float16)
        sparse_x = torch.empty((5, 64), device="cuda", dtype=torch.float16)

        sparse24_partition_rows_(
            x,
            dense_rows,
            sparse_rows,
            dense_x,
            sparse_x,
        )
        actual = torch.empty_like(x)
        sparse24_merge_rows_(
            actual,
            dense_x,
            sparse_x,
            dense_rows,
            sparse_rows,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(dense_x, x[dense_rows.long()]))
        self.assertTrue(torch.equal(sparse_x, x[sparse_rows.long()]))
        self.assertTrue(torch.equal(actual, x))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_residual_mlp_matches_dense_sparse_reference_cuda(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(31)
        x = torch.randn(
            (16, 64), device="cuda", dtype=torch.float16, generator=generator
        )
        gate_weight = torch.randn(
            (64, 128), device="cuda", dtype=torch.float16, generator=generator
        ) * 0.1
        down_weight = torch.randn(
            (64, 64), device="cuda", dtype=torch.float16, generator=generator
        ) * 0.1
        gate_weight24, _ = apply_random_24_mask(
            gate_weight, generator=generator
        )
        down_weight24, _ = apply_random_24_mask(
            down_weight, generator=generator
        )
        gate_residual = gate_weight - gate_weight24
        down_residual = down_weight - down_weight24
        assert_24_weight(gate_residual)
        assert_24_weight(down_residual)

        def prepack(
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            packed = pack_24(weight, layout="n_major")
            return prepare_cutlass_sparse24_device_gemm(
                packed.values,
                packed.meta,
                layout=packed.layout,
                K=weight.shape[0],
            )

        gate_values, gate_meta = prepack(gate_weight24)
        gate_residual_values, gate_residual_meta = prepack(gate_residual)
        down_values, down_meta = prepack(down_weight24)
        down_residual_values, down_residual_meta = prepack(down_residual)
        dense_rows = torch.tensor(
            [0, 2, 4, 6, 9, 11, 13, 15],
            device="cuda",
            dtype=torch.int32,
        )
        sparse_rows = torch.tensor(
            [1, 3, 5, 7, 8, 10, 12, 14],
            device="cuda",
            dtype=torch.int32,
        )
        summary = (
            torch.tensor(
                [
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                ],
                device="cuda",
            ),
            8,
            dense_rows,
            sparse_rows,
        )
        self.assertTrue(
            speclink_mlp.prepare_mixed_mlp_streams(torch.device("cuda"))
        )

        with mock.patch.object(
            speclink_mlp,
            "current_verify_dense_row_summary",
            return_value=summary,
        ):
            actual = speclink_mlp._batch_routed_residual_mlp_impl(
                x,
                gate_values,
                gate_meta,
                gate_residual_values,
                gate_residual_meta,
                down_values,
                down_meta,
                down_residual_values,
                down_residual_meta,
            )

        expected_gate = x @ gate_weight24
        dense_rows_long = dense_rows.to(dtype=torch.long)
        expected_gate[dense_rows_long] = x[dense_rows_long] @ gate_weight
        expected_hidden = torch.empty(
            (16, 64), device="cuda", dtype=torch.float16
        )
        torch.ops._C.silu_and_mul(expected_hidden, expected_gate)
        expected = expected_hidden @ down_weight24
        expected[dense_rows_long] = (
            expected_hidden[dense_rows_long] @ down_weight
        )
        torch.cuda.synchronize()

        self.assertTrue(actual.is_contiguous())
        dense_error = (actual[dense_rows_long] - expected[dense_rows_long]).abs()
        sparse_error = (actual[sparse_rows.long()] - expected[sparse_rows.long()]).abs()
        self.assertTrue(
            torch.allclose(actual, expected, rtol=3e-2, atol=2e-1),
            msg=(
                f"dense_max={dense_error.max().item():.6f}, "
                f"dense_mean={dense_error.mean().item():.6f}, "
                f"sparse_max={sparse_error.max().item():.6f}, "
                f"sparse_mean={sparse_error.mean().item():.6f}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
